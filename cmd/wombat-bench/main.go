// Command wombat-bench runs the benchmark suite: n samples of each task,
// scored, and reported as pass@k.
//
//	wombat-bench -n 8 -c 4 -tasks todo-app,fix-bug -out runs/ -temp 1.0
//
// It is the thin part. Everything interesting lives in rl (rollout, scoring,
// pass@k) and benchmarks (the tasks); this command's whole job is to turn
// flags into an [rl.AgentFunc], drive the rollouts, and put the numbers where
// a person and a CI job can both read them.
//
// # Output
//
//   - stdout gets a line per episode as it finishes, a line per task group,
//     and the full table at the end. A benchmark that shows nothing for ten
//     minutes gets killed by whoever started it, so nothing waits for the end.
//   - -out/report.txt is the same table, rewritten after every group.
//   - -out/episodes.jsonl is one line per episode — the whole trajectory,
//     reward included — appended as each group finishes. Both files are
//     written incrementally so a run interrupted at minute nine still has
//     eight minutes of data.
//
// # Exit status
//
// 0 when every rollout completed, 1 when one failed to run at all, 2 on bad
// usage. NOT a function of the scores: a suite that scores zero has run
// perfectly and told you something, and a green/red exit code on reward would
// make the numbers unreadable from CI.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/benchmarks"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/internal/fspath"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/anthropic"
	"github.com/hoobean1996/wombat-go/llm/openai"
	"github.com/hoobean1996/wombat-go/rl"
	"github.com/hoobean1996/wombat-go/tool"
	"github.com/hoobean1996/wombat-go/tool/builtin"
)

const (
	exitOK    = 0
	exitFail  = 1
	exitUsage = 2
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

type options struct {
	n         int
	c         int
	tasks     string
	out       string
	root      string
	temp      float64
	provider  string
	model     string
	maxIters  int
	maxTokens int
	budgetUSD float64
	wall      time.Duration
	threshold float64
	penalize  bool
	keep      bool
	list      bool
	logLevel  string
	confine   bool
}

// run is the whole command, factored out of main so that deferred cleanup runs
// before exit and so that a test can drive it against a fake provider.
func run(args []string, stdout, stderr io.Writer) int {
	var o options
	fs := flag.NewFlagSet("wombat-bench", flag.ContinueOnError)
	fs.SetOutput(stderr)

	fs.IntVar(&o.n, "n", 8, "samples per task. pass@k is only as good as n: one rollout of a\n"+
		"sampled agent is one draw from a distribution nobody has looked at")
	fs.IntVar(&o.c, "c", rl.DefaultConcurrency, "episodes to run at once, across all tasks")
	fs.StringVar(&o.tasks, "tasks", "", "comma-separated task ids, or a tier (default: the whole suite).\n"+
		"Tiers: "+strings.Join(benchmarks.TierNames(), ", ")+". The easy tier is the floor —\n"+
		"it is SUPPOSED to score 1.000, and when it does not the harness is broken, not the\n"+
		"agent. The hard tier is the one that discriminates.\n"+
		"Tasks: "+strings.Join(benchmarks.IDs(), ", "))
	fs.StringVar(&o.out, "out", "runs", "directory for report.txt and episodes.jsonl")
	fs.StringVar(&o.root, "root", "", "workspace root (default: a sibling of -out, <out>-workspaces).\n"+
		"It must not sit inside -out. An agent has a shell, a shell has '..', and a\n"+
		"workspace nested under the results directory puts episodes.jsonl — every other\n"+
		"episode's full transcript and score — two levels above the agent's own files.")

	fs.Float64Var(&o.temp, "temp", 1.0, "sampling temperature.\n"+
		"Default 1.0 and NOT 0, which is the one value that breaks this whole command:\n"+
		"at temperature 0 the n samples are the same trajectory, every episode of a group\n"+
		"succeeds or fails together, and pass@n collapses to pass@1 while looking like it\n"+
		"measured something. Use 0 only to diff two agents on a fixed trajectory, with -n 1.")

	fs.StringVar(&o.provider, "provider", envOr("WOMBAT_PROVIDER", "anthropic"), "anthropic | openai")
	fs.StringVar(&o.model, "model", "", "model id (empty uses the provider default)")
	fs.IntVar(&o.maxIters, "max-iters", 24, "ReAct iteration cap per episode")
	fs.IntVar(&o.maxTokens, "max-tokens", wombat.DefaultMaxTokens, "reply token cap")
	fs.Float64Var(&o.budgetUSD, "budget", 0.50, "per-EPISODE cost cap in USD (0 = uncapped)")
	fs.DurationVar(&o.wall, "wall", 10*time.Minute, "per-EPISODE wall-clock cap (0 = uncapped)")

	fs.BoolVar(&o.confine, "confine", false,
		"run every command an episode's tools spawn inside the platform sandbox, with the\n"+
			"network closed (loopback stays open for tests). The suite is hermetic, so nothing a\n"+
			"task needs is lost, and an agent that writes outside its workspace fails instead of\n"+
			"quietly reaching a sibling episode's files.")
	fs.Float64Var(&o.threshold, "threshold", rl.DefaultSuccessThreshold,
		"reward at or above which an episode counts as a success.\n"+
			"Every task's verifier weights sum to 1.0, so the default means \"everything verified\".")
	fs.BoolVar(&o.penalize, "penalize", false,
		"add turn, tool-error and cost penalties to every task, and drop -threshold to\n"+
			"accommodate them. Off by default: penalties are for RANKING two agents that both\n"+
			"work, and they make the success column depend on how expensive the run was.")
	fs.BoolVar(&o.keep, "keep", false, "keep every workspace, not just the failed ones")
	fs.BoolVar(&o.list, "list", false, "list the suite and exit")
	fs.StringVar(&o.logLevel, "log", "warn", "stderr log level: debug | info | warn | error")

	if err := fs.Parse(args); err != nil {
		return exitUsage
	}

	if o.list {
		listTasks(stdout)
		return exitOK
	}
	if o.n <= 0 {
		fmt.Fprintln(stderr, "wombat-bench: -n must be positive")
		return exitUsage
	}

	slog.SetDefault(slog.New(slog.NewTextHandler(stderr, &slog.HandlerOptions{Level: parseLevel(o.logLevel)})))
	logger := slog.Default()

	// Applied to this PROCESS, before anything is built, because the agent's
	// bash tool inherits os.Environ() and there is no seam to add to it later.
	// A suite that is hermetic for its verifiers and not for its agent is not
	// hermetic; the agent is the one that would run `go get`.
	if err := benchmarks.ApplyGoEnv(); err != nil {
		fmt.Fprintln(stderr, "wombat-bench:", err)
		return exitFail
	}

	tasks, err := benchmarks.Select(splitList(o.tasks))
	if err != nil {
		fmt.Fprintln(stderr, "wombat-bench:", err)
		return exitUsage
	}

	client, err := buildClient(o, logger)
	if err != nil {
		fmt.Fprintln(stderr, "wombat-bench:", err)
		return exitFail
	}

	out, err := filepath.Abs(o.out)
	if err != nil {
		fmt.Fprintf(stderr, "wombat-bench: resolving -out %q: %v\n", o.out, err)
		return exitFail
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		fmt.Fprintln(stderr, "wombat-bench:", err)
		return exitFail
	}
	// A SIBLING of -out, not a child. The results directory holds
	// episodes.jsonl, which is every episode's full transcript and reward, and
	// a workspace underneath it is reachable from the agent's shell with two
	// "..". Two of the four samples in the run that prompted this went looking
	// outside their workspace, and one of them read episodes.jsonl.
	root := o.root
	if root == "" {
		root = out + "-workspaces"
	}
	root, err = filepath.Abs(root)
	if err != nil {
		fmt.Fprintf(stderr, "wombat-bench: resolving -root %q: %v\n", root, err)
		return exitFail
	}
	if contains(out, root) || contains(root, out) {
		fmt.Fprintf(stderr, "wombat-bench: -root %s and -out %s must not contain one another;\n"+
			"the agent can reach anything above its workspace, and the results are the answers\n", root, out)
		return exitFail
	}

	sink, err := newSink(out)
	if err != nil {
		fmt.Fprintln(stderr, "wombat-bench:", err)
		return exitFail
	}
	defer sink.close()

	// Ctrl-C cancels the rollout context, which unwinds the in-flight model
	// call and any running verifier on its own. Whatever finished is already
	// on disk.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	p := &printer{w: stdout}
	p.printf("wombat-bench: %d task(s) x %d samples, concurrency %d, temp %.2f\n\n",
		len(tasks), o.n, o.c, o.temp)

	failed := false
	for _, t := range tasks {
		if o.penalize {
			t = t.With(benchmarks.Penalties()...)
		}
		g, err := rl.Rollout(ctx, agentFunc(o, client, logger, t), t.Env(root), o.n, rolloutOpts(o, p)...)
		if err != nil {
			fmt.Fprintf(stderr, "wombat-bench: %s: %v\n", t.ID, err)
			failed = true
			if ctx.Err() != nil {
				break
			}
			continue
		}
		p.printf("  = %-18s pass@1 %.3f  pass@%d %.3f  mean %.3f ± %.3f\n\n",
			g.TaskID, g.PassAt(1), o.n, g.PassAt(o.n), g.Mean(), g.Std())

		if err := sink.add(g); err != nil {
			fmt.Fprintln(stderr, "wombat-bench:", err)
			failed = true
		}
	}

	if err := sink.report.WriteText(stdout); err != nil {
		fmt.Fprintln(stderr, "wombat-bench:", err)
		failed = true
	}
	fmt.Fprintf(stdout, "\n%s\n%s\n", filepath.Join(out, "report.txt"), filepath.Join(out, "episodes.jsonl"))

	if failed {
		return exitFail
	}
	return exitOK
}

// rolloutOpts turns the flags into rollout options for one task.
func rolloutOpts(o options, p *printer) []rl.Option {
	opts := []rl.Option{
		rl.WithConcurrency(o.c),

		// Per EPISODE, which is what rl.WithBudget already means. A budget
		// shared across the group would let one runaway sample starve the
		// seven that had not started, and those seven would be recorded as
		// failures for a reason that had nothing to do with them — exactly the
		// correlation that makes a pass@k estimate meaningless.
		rl.WithBudget(governor.Limits{
			CostUSD: o.budgetUSD,
			Wall:    o.wall,
			Steps:   o.maxIters + 1,
		}),
		rl.WithProgress(p.episode),

		// The same table buildClient hands to wombat.TrackCost, which is the
		// thing that actually fills in Episode.Spend. Passing it here is what
		// lets the report name the model that has no rate, instead of
		// printing $0.0000 for sixteen episodes and being believed.
		rl.WithPricing(benchPricing),
	}
	if o.keep {
		opts = append(opts, rl.WithKeepWorkspaces())
	}

	threshold := o.threshold
	if o.penalize {
		// Penalties are a negative term on a reward whose positive terms sum
		// to 1.0, so leaving the threshold at 1.0 would report every episode
		// as VerifierFailed — including the ones that did the task perfectly
		// and merely took a few turns doing it.
		threshold = min(threshold, 0.90)
	}
	return append(opts, rl.WithSuccessThreshold(threshold))
}

// agentFunc builds the agent for one episode of task.
//
// A factory and not a shared agent, and that is forced: every file and shell
// tool captures its FS and its Runner when it is constructed, so the sample's
// workspace has to reach tool construction. Two samples sharing one agent would
// share a working directory and overwrite each other's work.
//
// It takes the benchmarks.Task and not just the rl.Task because the tool set
// is part of a task's definition — see [benchmarks.Task.Tools].
func agentFunc(o options, client llm.Client, logger *slog.Logger, task benchmarks.Task) rl.AgentFunc {
	return func(t rl.Task) (*wombat.Agent, error) {
		runner := builtin.OSRunner(t.Workspace)
		if o.confine {
			r, err := builtin.SandboxRunner(t.Workspace, builtin.Sandbox{})
			if err != nil {
				return nil, fmt.Errorf("-confine: %w", err)
			}
			runner = r
		}
		tools := builtin.Default(builtin.Deps{
			FS:   builtin.OSFS(t.Workspace),
			Exec: runner,
			Now:  time.Now,
		})

		// Two tools are removed, and both removals are about making the
		// measurement mean something rather than about safety.
		//
		// http_get, because the suite claims to be hermetic. A task solved by
		// fetching an answer off the internet has measured the internet.
		//
		// The builtin ask_user, because a CapPause tool ENDS the run — outcome
		// Paused — so one call to it costs the whole episode. A task that
		// WANTS the agent to be able to ask supplies its own non-pausing
		// stand-in through Task.Tools; see benchmarks.AskUserTool.
		tools = tool.Filter(tools, tool.Provided(tool.NeedFSRead|tool.NeedFSWrite|tool.NeedExec))
		tools = tool.Filter(tools, func(d tool.Def) bool { return !d.Has(tool.CapPause) })
		tools = append(tools, task.Tools...)

		return wombat.New(
			wombat.WithName("bench"),
			wombat.WithClient(client),
			wombat.WithModel(o.model),
			wombat.WithMaxIters(o.maxIters),
			wombat.WithMaxTokens(o.maxTokens),
			wombat.WithTools(tools...),
			wombat.WithLogger(logger),
			wombat.WithSystemPrompt(systemPromptFor(canAsk(task))),

			// Say what the directory is FOR, not just what it is. A bare path
			// does not connect to the bash tool's exec_dir parameter in the
			// model's head; naming the parameter does.
			wombat.WithEnvBlock("working_directory", t.Workspace+
				"\n\nThis is the only directory you may touch. Pass it as exec_dir to the bash "+
				"tool and use it as the base for every path — the file tools take absolute "+
				"paths and will refuse anything outside this directory."),

			// A benchmark episode is capped, and an episode guillotined
			// mid-thought scores zero however close it was. Warning the model
			// at 80% lets it stop and write down what it has.
			wombat.WithTurnNotice(governor.NoticeAt(0.8)),
		)
	}
}

// canAsk reports whether the task supplies an ask_user tool.
//
// By NAME and not by "has any extra tools", because the sentence the system
// prompt is about to make — somebody will answer you — is a promise, and
// making it to an agent that has no way to ask would be the worst kind of
// prompt bug: invisible, and it would look like the model was too timid.
func canAsk(task benchmarks.Task) bool {
	for _, d := range task.Tools {
		if d.Name == benchmarks.AskUserName {
			return true
		}
	}
	return false
}

// systemPromptFor replaces the harness default, which assumes a conversation.
//
// Three things it has to say that the default does not: whether there is
// anybody to ask, that the network is off (so `go get` is a dead end rather
// than a retry), and that finishing is the agent's own decision (an episode
// that never stops is an episode that scores zero at the iteration cap).
//
// The first of those varies, and it has to. Telling an agent "nobody can
// answer a question" and then handing it ask_user would make ambiguous-spec
// measure whether the model disobeys its system prompt; telling every agent
// "you may ask" when only one task can answer would invite a pause that costs
// the other seven tasks a turn each. So the sentence follows the tool set.
func systemPromptFor(canAsk bool) string {
	return systemPromptHead(canAsk) + systemPromptTail
}

func systemPromptHead(canAsk bool) string {
	if canAsk {
		return `You are a software engineer working in a single directory. Nobody is watching
you work, but the person who set the task is reachable: you have an ask_user
tool and they will answer it.

Use it when a decision genuinely is not yours to make — when the task is silent
about something that changes the result, and picking one option and moving on
would hide the gap rather than close it. Do not use it for anything you can
settle by reading the code.
`
	}
	return `You are a software engineer working alone, without supervision, in a single
directory. Nobody is watching and nobody can answer a question: work it out
from what is in front of you.
`
}

const systemPromptTail = `
There is no network. Do not try to install anything, fetch anything, or add a
third-party dependency — the Go module proxy is switched off and the attempt
will only cost you time. The standard library is enough for every task you will
be given.

Verify your own work before you stop. If the task mentions a command, run it.
When it passes, say briefly what you did and stop; do not keep polishing.`

// benchPricing is the one table this command prices with: TrackCost charges
// the budget with it and the rollout judges its own cost column against it.
//
// One variable rather than two mentions of [llm.DefaultPricing], because those
// two have to be the same table or the report will vouch for a number that was
// computed some other way. Nothing is added to it for an internal or gateway
// model: an invented rate looks exactly like a measurement, and a COST column
// reading n/a is the honest answer until someone puts a real rate in.
var benchPricing llm.Pricing = llm.DefaultPricing

// buildClient assembles the provider client and its middleware.
func buildClient(o options, logger *slog.Logger) (llm.Client, error) {
	var (
		client llm.Client
		err    error
	)
	switch o.provider {
	case "anthropic", "claude":
		cfg := anthropic.ConfigFromEnv()
		if o.model != "" {
			cfg.Model = o.model
		}
		client, err = anthropic.New(cfg)
	case "openai":
		cfg := openai.ConfigFromEnv()
		if o.model != "" {
			cfg.Model = o.model
		}
		client, err = openai.New(cfg)
	default:
		return nil, fmt.Errorf("unknown provider %q (want anthropic or openai)", o.provider)
	}
	if err != nil {
		return nil, err
	}

	// Innermost first. withTemperature is closest to the wire on purpose: it is
	// the last thing to touch the request, so no layer above can rebuild a
	// request without it — and a benchmark that silently sampled at the
	// provider's default temperature would report a pass@k for an experiment
	// nobody ran.
	return llm.Chain(client,
		withTemperature(o.temp),
		llm.WithValidation,
		wombat.TrackCost(benchPricing),
		llm.WithRetry(llm.DefaultRetryPolicy),
		wombat.WithOverflowRecovery(),
		llm.WithLogging(logger),
	), nil
}

// withTemperature pins the sampling temperature on every request.
//
// Middleware rather than an [wombat.Option], because the agent materializes its
// own [llm.Request] and exposes no seam for the sampling parameters — which is
// right, since they belong to the model and not to the loop. This is the seam.
//
// A caller that set Temperature itself wins, so a future agent that wants a
// deterministic sub-call can still have one.
func withTemperature(t float64) llm.Middleware {
	return func(next llm.Client) llm.Client {
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			if req.Temperature == nil {
				// A fresh copy per call: the pointer escapes into the request
				// and a shared one would be a value several goroutines hold.
				v := t
				req.Temperature = &v
			}
			return next.Complete(ctx, req)
		})
	}
}

// printer serializes the live progress lines.
//
// A mutex and not "just write from one goroutine", because [rl.WithProgress]
// calls back from the finishing EPISODE's goroutine — its doc comment says so —
// and up to -c of those finish at once. Unsynchronised, two Fprintf calls to
// one writer interleave and the operator reads a corrupted line at exactly the
// moment they are watching for a failure.
type printer struct {
	mu sync.Mutex
	w  io.Writer
}

// episode prints one finished episode. Safe for concurrent use.
func (p *printer) episode(_ rl.Task, ep *rl.Episode) {
	p.mu.Lock()
	defer p.mu.Unlock()
	// Tokens on the live line too, and not only cost: they are the number that
	// is still there when the model has no rate.
	fmt.Fprintf(p.w, "  %-21s %-14s reward %6.3f  turns %2d  tools %2d/%-2d  tok %7d/%-6d  %s  %s\n",
		ep.Label(), ep.Failure, ep.Reward, ep.Turns(),
		ep.ToolErrors(), ep.ToolCalls(),
		ep.PromptTokens(), ep.OutputTokens(),
		dollars(ep), ep.Wall.Round(100*time.Millisecond))
}

// printf writes a line outside the concurrent phase, taking the same lock so
// the group summaries cannot land inside a progress line.
func (p *printer) printf(format string, args ...any) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fmt.Fprintf(p.w, format, args...)
}

// ===== output =====

// sink writes the two artifacts incrementally.
//
// Incrementally because a benchmark run is long and gets interrupted: a report
// written only at the end is a report that does not exist for the nine minutes
// it took to earn. episodes.jsonl is appended per group, report.txt is rewritten
// whole (it is a page of text, and rewriting is how it stays consistent with
// the groups above it).
type sink struct {
	dir    string
	jsonl  *os.File
	report rl.Report
}

func newSink(dir string) (*sink, error) {
	f, err := os.Create(filepath.Join(dir, "episodes.jsonl"))
	if err != nil {
		return nil, fmt.Errorf("creating episodes.jsonl: %w", err)
	}
	return &sink{dir: dir, jsonl: f}, nil
}

// add records one finished group and flushes both artifacts.
func (s *sink) add(g *rl.Group) error {
	s.report.Add(g)

	// One group at a time into the JSONL, so a line is written once and the
	// file is append-only — rewriting it whole would mean re-encoding every
	// transcript in the run on every group.
	one := rl.Report{Groups: []*rl.Group{g}}
	if err := one.WriteJSONL(s.jsonl); err != nil {
		return err
	}
	if err := s.jsonl.Sync(); err != nil {
		return fmt.Errorf("flushing episodes.jsonl: %w", err)
	}

	f, err := os.Create(filepath.Join(s.dir, "report.txt"))
	if err != nil {
		return fmt.Errorf("creating report.txt: %w", err)
	}
	defer f.Close()
	return s.report.WriteText(f)
}

func (s *sink) close() {
	if err := s.jsonl.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
		slog.Error("wombat-bench: closing episodes.jsonl", "err", err)
	}
}

// ===== small helpers =====

// listTasks prints the suite for -list, grouped by tier.
//
// By tier and not as one flat list, because the two tiers answer different
// questions and a reader who does not know that will run all eight, wait, and
// conclude the suite is broken when four tasks come back 1.000.
func listTasks(w io.Writer) {
	width := 0
	for _, t := range benchmarks.All() {
		width = max(width, len(t.ID))
	}
	for _, tier := range []struct {
		name  string
		note  string
		tasks []benchmarks.Task
	}{
		{"easy", "the floor: these are supposed to pass, and tell you the harness works", benchmarks.Easy()},
		{"hard", "the ones that discriminate", benchmarks.Hard()},
	} {
		fmt.Fprintf(w, "%s — %s\n", tier.name, tier.note)
		for _, t := range tier.tasks {
			fmt.Fprintf(w, "  %-*s  %s\n", width, t.ID, t.Summary)
		}
		fmt.Fprintln(w)
	}
}

// dollars renders an episode's cost for the live line, or "n/a" when that cost
// is the absence of a rate rather than a price.
//
// The same rule the report table follows, for the same reason: this line is
// what an operator watches for ten minutes, and "$0" scrolling past sixteen
// times is how a missing table entry gets mistaken for a free run.
func dollars(ep *rl.Episode) string {
	if !ep.Priced {
		return "    n/a"
	}
	if ep.Spend.CostUSD == 0 {
		return "     $0"
	}
	return fmt.Sprintf("%7s", fmt.Sprintf("$%.4f", ep.Spend.CostUSD))
}

func splitList(s string) []string {
	var out []string
	for f := range strings.SplitSeq(s, ",") {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func parseLevel(s string) slog.Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "error":
		return slog.LevelError
	default:
		return slog.LevelWarn
	}
}

// contains reports whether parent is dir or an ancestor of it. Both must
// already be absolute and cleaned.
//
// A plain strings.HasPrefix would call /tmp/runs an ancestor of
// /tmp/runs-workspaces, which is exactly the pair the default layout produces.
// The separator is what makes the test a path test rather than a string test.
func contains(parent, dir string) bool {
	return fspath.Within(parent, dir)
}
