// Command wombat-jsonl runs one agent turn and writes a JSON-Lines event
// stream on stdout.
//
// It is the sidecar a UI front end spawns. The interesting thing about it is
// how little there is: the harness already exposes a stream of events, so the
// adapter is a loop and an encoder.
//
//	for run.Next() { enc.Encode(run.Event()) }
//
// In the OCaml original the harness returned a value and emitted four separate
// observer callbacks, and 473 lines of sidecar existed to sew them back into
// one stream. Making the stream the primary API deletes that job rather than
// reimplementing it.
//
// Contract:
//   - stdout is JSONL and nothing else, one object per line, each with a
//     "type" discriminator. A front end can read it with a line reader and no
//     state machine.
//   - stderr carries diagnostics (slog).
//   - exit 0 on a completed turn (including a pause), 1 on failure,
//     2 on bad usage.
//
// That contract is also why -permission behaves the way it does here. A policy
// can allow and it can deny, but it cannot ask: there is no console to prompt
// on, and stdout belongs to the protocol. An Ask is therefore refused, with an
// explanation the model can read. See the flag's help.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/accept"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/anthropic"
	"github.com/hoobean1996/wombat-go/llm/openai"
	"github.com/hoobean1996/wombat-go/mcp"
	"github.com/hoobean1996/wombat-go/metric"
	"github.com/hoobean1996/wombat-go/permission"
	"github.com/hoobean1996/wombat-go/review"
	"github.com/hoobean1996/wombat-go/skill"
	"github.com/hoobean1996/wombat-go/tape"
	"github.com/hoobean1996/wombat-go/tool"
	"github.com/hoobean1996/wombat-go/tool/builtin"
	"github.com/hoobean1996/wombat-go/trace"
)

const (
	exitOK    = 0
	exitFail  = 1
	exitUsage = 2
)

func main() {
	// All work happens in run so that deferred cleanup executes before exit.
	os.Exit(run())
}

type options struct {
	provider      string
	stream        string
	model         string
	system        string
	workdir       string
	fsRoot        string
	sandbox       string // deprecated alias for fsRoot
	permission    string
	resumePath    string
	sessionPath   string
	skillsDir     string
	tapePath      string
	tapeMode      string
	tracePath     string
	traceHTML     string
	metricsFile   string
	delegate      bool
	review        bool
	regressions   string
	planOnly      bool
	accept        bool
	acceptRetries int
	reviewSweeps  int
	reviewDry     int
	reviewOnly    bool
	reviewIters   int
	reviewFix     int
	maxIters      int
	maxTokens     int
	budgetUSD     float64
	wall          time.Duration
	parallel      int
	readOnly      bool
	confine       bool
	confineNet    string
	media         bool
	window        int
	compactAt     float64
	check         string
	checkAttempts int
	mcpConfig     string
	verbose       bool
	logLevel      string

	// mcpConns is not a flag: run connects to the -mcp servers once, and
	// every agent build makes — the reviewer and the fixer too — shares them.
	mcpConns *mcp.Connections
}

func run() int {
	var o options
	flag.StringVar(&o.provider, "provider", envOr("WOMBAT_PROVIDER", "anthropic"), "anthropic | openai")
	flag.StringVar(&o.stream, "stream", "", "auto | always | never (default: the provider's own env setting)")
	flag.StringVar(&o.model, "model", "", "model id (empty uses the provider default)")
	flag.StringVar(&o.system, "system", "", "system prompt override")
	flag.StringVar(&o.workdir, "working-dir", ".", "working directory for shell and git tools")
	flag.StringVar(&o.fsRoot, "fs-root", "",
		"confine the FILE tools (view_file, write_file, str_replace, save_tool_result) to this prefix.\n"+
			"It does NOT cover bash or grep_search, which shell out and can read and write anywhere\n"+
			"this process can; nor is it a security boundary even for the file tools (symlinks are\n"+
			"not resolved). Use -permission for a rule that also has an opinion about the shell.")
	flag.StringVar(&o.sandbox, "sandbox", "", "deprecated alias for -fs-root; the name oversold what it does")
	flag.StringVar(&o.permission, "permission", "off",
		"off | readonly | workspace | ask — the policy every tool call is checked against.\n"+
			"off: no gate, today's behaviour. readonly: only read-only tools run. workspace: reads\n"+
			"are free, writes inside -working-dir are free, exec and anything outside it must be\n"+
			"approved. ask: nothing runs without approval.\n"+
			"This front end CANNOT collect an approval — stdout is a JSONL protocol and there is no\n"+
			"console to prompt on — so an Ask is denied and the model is told why. -permission ask\n"+
			"here therefore means \"deny anything not explicitly allowed\", which is a useful mode and\n"+
			"not a broken one. Use wombat-serve when you want to answer.")
	flag.StringVar(&o.resumePath, "resume", "", "JSON file holding a prior transcript to continue (read only)")
	flag.StringVar(&o.sessionPath, "session", "",
		"JSON file holding the conversation: read at start if it exists, rewritten when the\n"+
			"turn ends. This is what makes -resume usable — nothing else in this binary ever\n"+
			"wrote a transcript, so the flag that read one could only be fed by a front end\n"+
			"that reconstructed the conversation itself. With -session, multi-turn is:\n"+
			"  wombat-jsonl -session s.json 'start the task'\n"+
			"  wombat-jsonl -session s.json 'now do the next part'\n"+
			"and if the first turn stopped on ask_user, the second turn's text is the ANSWER\n"+
			"to that question rather than a new instruction.")
	flag.StringVar(&o.skillsDir, "skills-dir", "", "directory of <name>/SKILL.md to offer as loadable skills")
	flag.StringVar(&o.tapePath, "tape", "", "record/replay JSONL tape")
	flag.StringVar(&o.tapeMode, "tape-mode", "auto", "auto | record | replay")
	flag.StringVar(&o.tracePath, "trace", "", "write an NDJSON trace here")
	flag.StringVar(&o.traceHTML, "trace-html", "", "also render the trace as a self-contained HTML report")
	flag.StringVar(&o.metricsFile, "metrics-file", "",
		"write one Prometheus exposition dump to this file as the process exits.\n"+
			"A file rather than an endpoint because this command is one-shot: it lives for the\n"+
			"length of a single turn and there is nothing for a scraper to poll — by the time a\n"+
			"scrape interval came round the process would be gone. Point node_exporter's textfile\n"+
			"collector at the directory, or have CI read the file, and one turn's token spend and\n"+
			"tool failures land in the same dashboard as the long-lived services.\n"+
			"The dump is written even when the turn fails, since that is when it matters most.")
	flag.BoolVar(&o.delegate, "delegate", false, "offer a read-only sub-agent through the delegate tool")
	flag.BoolVar(&o.review, "review", false,
		"after the turn answers, hand the workspace to a fresh sub-agent for adversarial review.\n"+
			"It reports only defects it can demonstrate with a Go test that fails against the code,\n"+
			"so a finding is established by running it rather than by reading what the reviewer said.\n"+
			"The builder has just spent its context concluding the code is right; the reviewer has not.")
	flag.StringVar(&o.regressions, "regressions", "",
		"directory holding one test file per defect ever found on this task. They are\n"+
			"copied into -working-dir before the build and must pass at the gate, and a\n"+
			"repaired review finding is added to it. Without this each run starts from\n"+
			"nothing: a defect repaired on one run came back on the next, and six review\n"+
			"sweeps missed it.")
	flag.BoolVar(&o.planOnly, "plan-only", false,
		"produce PLAN.md, TODO.md and Verifier.md for the task and stop, writing no code.\n"+
			"Run this with -model set to your strongest model, then build with a cheaper one:\n"+
			"knowing WHAT to verify needs breadth, and costs a few thousand tokens once,\n"+
			"while the build costs a hundred iterations.")
	flag.BoolVar(&o.accept, "accept", false,
		"gate the answer on deterministic checks: go build, go vet, go test, that the\n"+
			"tree has any tests at all, and that TODO.md has no unticked items. A failing\n"+
			"gate is handed back as work and the run continues.")
	flag.IntVar(&o.acceptRetries, "accept-retries", 2,
		"how many times a rejected answer may be handed back before the run fails.")
	flag.IntVar(&o.reviewSweeps, "review-sweeps", 1,
		"how many independent review passes to run. Each gets a FRESH reviewer and a\n"+
			"different lens (error paths, shared state, entry points, coverage gaps).\n"+
			"One pass is not a clean bill of health: a measured sweep reported one defect\n"+
			"and stopped, and a hand review of the same code then found five more.\n"+
			"Converging needs at least 6: every lens must run before a dry sweep counts,\n"+
			"or the streak describes the lenses that ran rather than the code.")
	flag.IntVar(&o.reviewDry, "review-dry", 0,
		"stop early after this many consecutive sweeps find nothing new (0 = default 2).\n"+
			"This, not the sweep count, is the real stop condition: running out of sweeps\n"+
			"is not evidence that anything is clean.")
	flag.BoolVar(&o.reviewOnly, "review-only", false,
		"skip the build turn and review the code already in -working-dir.\n"+
			"The query is used as the task description, not as work to do.")
	flag.IntVar(&o.reviewIters, "review-iters", 0,
		"ReAct iteration cap for the reviewer (0 uses the built-in default).\n"+
			"Scale it with the size of the code under review: a reviewer that runs out\n"+
			"mid-read reports only what it reached first, and that reads like a thorough\n"+
			"review that found little.")
	flag.IntVar(&o.reviewFix, "review-fix", 0,
		"repair rounds after a review. Each round hands the failing tests back and re-runs them;\n"+
			"a defect counts as fixed only when its own test still exists and passes.")
	flag.IntVar(&o.maxIters, "max-iters", wombat.DefaultMaxIters, "ReAct iteration cap")
	flag.IntVar(&o.maxTokens, "max-tokens", wombat.DefaultMaxTokens, "reply token cap")
	flag.Float64Var(&o.budgetUSD, "budget", 0, "cost cap in USD (0 = uncapped)")
	flag.DurationVar(&o.wall, "wall", 0, "wall-clock cap (0 = uncapped)")
	flag.IntVar(&o.parallel, "tool-parallelism", wombat.DefaultToolParallelism,
		"read-only tool calls of one batch run at once; anything that writes or executes always runs alone, in order")
	flag.BoolVar(&o.readOnly, "read-only", false, "hide mutating and exec tools")
	flag.BoolVar(&o.confine, "confine", false,
		"run every command the tools spawn inside the platform sandbox (sandbox-exec on macOS,\n"+
			"bwrap on Linux): writes only under -working-dir and the temp directory, credential\n"+
			"directories unreadable, network per -confine-network. Fails at start if the sandbox\n"+
			"cannot be applied rather than running unconfined.")
	flag.StringVar(&o.confineNet, "confine-network", "loopback",
		"loopback | none | full — what a -confine'd command may reach")
	flag.IntVar(&o.window, "context-window", 0,
		"the model's context window in tokens. With it set, the transcript is kept under 70% of it\n"+
			"by eliding old tool results (wombat.TokenWindow), instead of being sent whole until the\n"+
			"provider refuses it. 0 leaves the transcript alone.")
	flag.Float64Var(&o.compactAt, "compact-at", 0,
		"with -context-window: summarize the oldest part of the conversation into the transcript once\n"+
			"the whole conversation reaches this fraction of the window (e.g. 0.85), keeping the newest\n"+
			"quarter of a window verbatim.\n"+
			"Costs one model call per compaction. 0 disables.")
	flag.StringVar(&o.check, "check", "",
		"a shell command that says whether the work is done — e.g. 'go test ./...'. It runs in\n"+
			"-working-dir, through the same runner as the agent's own commands (so -confine applies),\n"+
			"whenever the model ends the run; a non-zero exit sends the model back with its output.")
	flag.IntVar(&o.checkAttempts, "check-attempts", 3, "how many times -check may send the model back")
	flag.BoolVar(&o.media, "media", false,
		"offer view_media, which shows the model an image or a PDF from disk. Off by default:\n"+
			"a model that cannot read images answers a request carrying one with a 400.")
	flag.StringVar(&o.mcpConfig, "mcp", "",
		"JSON file of MCP servers to offer the tools of, in the usual {\"mcpServers\": {...}} format.\n"+
			"Each server's tools are named <server>__<tool>; a stdio server gets a filtered environment\n"+
			"plus the env its entry names (\"${VAR}\" passes one of this process's variables through).")
	flag.BoolVar(&o.verbose, "v", false, "debug logging on stderr (same as -log debug)")
	flag.StringVar(&o.logLevel, "log", "info", "stderr log level: debug | info | warn | error")
	flag.Parse()

	query := strings.TrimSpace(strings.Join(flag.Args(), " "))
	if query == "" && o.resumePath == "" && o.sessionPath == "" {
		fmt.Fprintln(os.Stderr, "usage: wombat-jsonl [flags] <query>")
		flag.PrintDefaults()
		return exitUsage
	}
	// Both would mean "read from here, write to there", and the silent version
	// of that discards whatever -session already held without saying so.
	// resolveFSRoot took the same view of -sandbox and -fs-root.
	if o.resumePath != "" && o.sessionPath != "" {
		fmt.Fprintln(os.Stderr,
			"wombat-jsonl: -resume and -session both read a transcript; give one.\n"+
				"  -resume  reads and never writes (fork a conversation)\n"+
				"  -session reads and rewrites (carry one on)")
		return exitUsage
	}
	if err := checkSessionWritable(o.sessionPath); err != nil {
		fmt.Fprintln(os.Stderr, "wombat-jsonl:", err)
		return exitUsage
	}

	// Diagnostics are for whoever is operating the process, and an interactive
	// reader is not that person: at Info this prints a line per model call
	// straight through the transcript they are trying to read. -log warn is
	// what the Makefile's `ask` target uses for exactly that reason.
	level := slog.LevelInfo
	switch strings.ToLower(o.logLevel) {
	case "debug":
		level = slog.LevelDebug
	case "warn", "warning":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	}
	if o.verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	resolveFSRoot(&o, logger)

	out := newEmitter(os.Stdout)

	// The tape and the trace sink both outlive the run and both report real
	// failures only on Close, so they are opened here and not inside build.
	tp, err := openTape(o)
	if err != nil {
		out.emit(failed{Type: "failed", Reason: err.Error()})
		return exitFail
	}
	if tp != nil {
		defer func() {
			if err := tp.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "wombat-jsonl: tape: %v\n", err)
			}
		}()
	}

	tracer, traceDone, err := openTrace(o, logger)
	if err != nil {
		out.emit(failed{Type: "failed", Reason: err.Error()})
		return exitFail
	}
	defer traceDone()

	// The registry outlives the agent by exactly one deferred call. Registered
	// here rather than inside build for the same reason the tape and the trace
	// are: build assembles a chain, and something has to hold the object after
	// the chain is gone in order to write it out.
	var m *metric.Metrics
	if o.metricsFile != "" {
		reg := metric.NewRegistry()
		m = metric.New(reg)
		defer dumpMetrics(reg, o.metricsFile)
	}

	if o.mcpConfig != "" {
		conns, err := connectMCP(o.mcpConfig, logger)
		if err != nil {
			out.emit(failed{Type: "failed", Reason: err.Error()})
			return exitFail
		}
		defer conns.Close()
		o.mcpConns = conns
	}

	a, err := build(o, logger, tp, tracer, m)
	if err != nil {
		out.emit(failed{Type: "failed", Reason: err.Error()})
		return exitFail
	}

	// Ctrl-C and SIGTERM cancel the context, which unwinds the in-flight HTTP
	// request and any running subprocess on its own.
	ctx, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	ctx, cancelBudget := governor.WithBudget(ctx, governor.Limits{
		CostUSD: o.budgetUSD,
		Wall:    o.wall,
	})
	defer cancelBudget()

	// Review an existing tree. The build turn is what makes a run expensive and
	// non-repeatable; separating them lets a review be re-run against the same
	// code after the reviewer itself changes, which is the only way to tell a
	// reviewer improvement from a different build.
	if o.reviewOnly {
		emitReview(ctx, o, logger, tp, tracer, m, query, out)
		return exitOK
	}

	if o.planOnly {
		query = planOnlyPrompt(query)
	}

	// Counted before the build so the gate can tell a finished list from a
	// shortened one. A deleted item is not an unticked item, and shortening the
	// list has been the shortcut taken on three runs across two tasks.
	if o.accept && !o.planOnly {
		if err := accept.RecordCensus(o.workdir, review.DefaultTodoFile, accept.DefaultVerifierFile); err != nil {
			logger.Warn("could not record the plan census", "err", err)
		}
	}

	// Installed before the build, not checked after it: a defect the ledger
	// knows about should be a failing test the builder sees from its first
	// `go test`, rather than something a review has to find again.
	if o.regressions != "" && !o.planOnly {
		led, lerr := accept.LoadLedger(o.regressions)
		if lerr != nil {
			out.emit(failed{Type: "failed", Reason: "regressions: " + lerr.Error()})
			return exitFail
		}
		if lerr := led.Install(o.workdir); lerr != nil {
			out.emit(failed{Type: "failed", Reason: lerr.Error()})
			return exitFail
		}
		if n := len(led.Files); n > 0 {
			logger.Info("installed regression tests", "count", n, "from", o.regressions)
		}
	}

	in, err := input(o, query, tool.NewSet(a.Tools(ctx)...))
	if err != nil {
		out.emit(failed{Type: "failed", Reason: err.Error()})
		return exitFail
	}

	out.emit(sessionStarted{
		Type:      "session_started",
		Model:     o.model,
		Provider:  o.provider,
		Query:     query,
		MaxIters:  o.maxIters,
		BudgetUSD: o.budgetUSD,
	})

	var r *wombat.Run
	var msgs []llm.Message

	// Deferred, and registered before the drain, so the transcript is written
	// however this turn ends — answered, paused, out of budget, interrupted.
	// The paused and out-of-budget cases are the ones worth resuming.
	defer func() { saveSession(o.sessionPath, msgs) }()

	// An answer is a claim, not a delivery. Each attempt runs the agent, then
	// the deterministic gate; a rejected answer is handed back as work and the
	// conversation continues from where it was.
	for attempt := 1; ; attempt++ {
		if r != nil {
			r.Close()
		}
		r = a.Start(ctx, in)

		for r.Next() {
			out.emit(r.Event())
		}
		msgs = r.Messages()

		if err := r.Err(); err != nil {
			r.Close()
			out.emit(failed{Type: "failed", Reason: err.Error(), Class: classOf(err)})
			return exitFail
		}

		res, answered := r.Outcome().(wombat.Answer)
		if !answered || !o.accept || o.planOnly {
			break
		}
		rep := accept.Run(ctx, accept.Config{
			Workspace:     o.workdir,
			RequireTests:  true,
			TodoFile:      review.DefaultTodoFile,
			VerifierFile:  accept.DefaultVerifierFile,
			RegressionDir: o.regressions,
			Log:           func(line string) { logger.Info(line) },
		})
		ev := accepted{Type: "accepted", Attempt: attempt, OK: rep.OK(), Summary: rep.Summary()}
		for _, c := range rep.Failed() {
			ev.Failed = append(ev.Failed, c.Name)
		}
		ev.Skipped = rep.Skipped
		out.emit(ev)

		if rep.OK() || attempt > o.acceptRetries {
			if !rep.OK() {
				// Say so rather than emitting the answer as if it had passed.
				out.emit(failed{Type: "failed",
					Reason: fmt.Sprintf("answer rejected by the acceptance gate after %d attempt(s): %s",
						attempt, rep.Summary()),
					Class: "unaccepted"})
			}
			break
		}
		_ = res
		in = wombat.Then(msgs, accept.Feedback(rep))
	}
	defer r.Close()

	switch res := r.Outcome().(type) {
	case wombat.Answer:
		out.emit(done{Type: "done", Answer: res.Text})
		if o.review {
			emitReview(ctx, o, logger, tp, tracer, m, query, out)
		}
	case wombat.Paused:
		out.emit(waiting{
			Type:      "agent_waiting",
			ToolUseID: string(res.ToolUseID),
			Question:  res.Schema.Question,
			Schema:    res.Schema.Schema,
		})
	case wombat.Submitted:
		out.emit(submitted{Type: "submitted", Tool: res.Tool, Payload: res.Payload})
	default:
		out.emit(failed{Type: "failed", Reason: "run produced no outcome"})
		return exitFail
	}
	return exitOK
}

// emitReview runs the post-turn review and reports it on the stream.
//
// A fresh agent rather than another turn of the builder. The builder has just
// argued itself into believing the code is correct, and asking it to find its
// own defects asks it to overturn the conclusion it reached one message ago.
// Measured against three defects a human review had confirmed, a fresh
// reviewer found two and filed no false positives, for about five cents.
func emitReview(
	ctx context.Context,
	o options,
	logger *slog.Logger,
	tp *tape.Tape,
	tracer *trace.Tracer,
	m *metric.Metrics,
	task string,
	out *emitter,
) {
	// The reviewer gets its own iteration budget and no delegate tool: it is
	// the last agent in the chain and nothing below it needs a sub-agent.
	ro := o
	ro.delegate = false
	if o.reviewIters > 0 {
		ro.maxIters = o.reviewIters
	} else if ro.maxIters > reviewMaxIters {
		ro.maxIters = reviewMaxIters
	}

	// A factory, not one agent: every sweep needs a reader that has not already
	// concluded the code is fine.
	newReviewer := func() (*wombat.Agent, error) { return build(ro, logger, tp, tracer, m) }
	if _, err := newReviewer(); err != nil {
		out.emit(reviewed{Type: "reviewed", Summary: "review could not start", Error: err.Error()})
		return
	}
	cfg := review.Config{
		NewReviewer: newReviewer,
		Sweeps:      o.reviewSweeps,
		DryStreak:   o.reviewDry,
		Workspace:   o.workdir,
		Task:        task,
		Rounds:      o.reviewFix,
		Log:         func(line string) { logger.Info(line) },
	}
	if o.reviewFix > 0 {
		fixer, ferr := build(ro, logger, tp, tracer, m)
		if ferr != nil {
			logger.Warn("review: no fixer", "err", ferr)
		} else {
			cfg.Fixer = fixer
		}
	}

	rep := review.Run(ctx, cfg)

	// A repaired finding becomes a permanent guard. The test that demonstrated
	// the defect is exactly the thing that stops it returning, and promoting it
	// is what makes the next run start from what this one learned instead of
	// from nothing.
	if o.regressions != "" {
		seen := map[string]bool{}
		for _, f := range rep.Findings {
			if !f.Fixed || f.File == "" || seen[f.File] {
				continue
			}
			seen[f.File] = true
			if err := accept.Promote(o.regressions, o.workdir, f.File); err != nil {
				logger.Warn("could not promote regression", "file", f.File, "err", err)
				continue
			}
			logger.Info("promoted to the regression ledger", "file", f.File)
		}
	}
	ev := reviewed{
		Type: "reviewed", Ran: rep.Ran, Summary: rep.Summary(), Rounds: rep.Rounds,
		Unfinished: rep.Unfinished, Discarded: len(rep.Discarded),
		PlanEdited: rep.PlanEdited,
		Sweeps:     rep.Sweeps, Converged: rep.Converged,
	}
	if rep.Err != nil {
		ev.Error = rep.Err.Error()
	}
	for _, f := range rep.Findings {
		ev.Findings = append(ev.Findings, reviewFinding{
			Test: f.Test, File: f.File, Message: f.Message, Fixed: f.Fixed,
		})
	}
	out.emit(ev)
}

// reviewMaxIters is the default cap on the reviewer, overridable with
// -review-iters. It is a cost bound, not a correctness one: a reviewer that
// runs out mid-read reports the few defects it happened to reach first, which
// reads exactly like a thorough review that found little. That is why running
// out sets Err — a truncated review must never be reported as a clean one.
//
// The default is high on purpose: this is meant for long tasks, and the failure
// it guards against is asymmetric. Running long costs tokens; running out
// produces a review that looks thorough and is not — the measured case stopped
// after one shallow finding on a 20KB tree and reported nothing else.
//
// reviewMaxIters caps the reviewer. It writes tests and runs them; it does not
// build anything, and a reviewer given the builder's budget spends it widening
// its search long after the useful findings are in.
const reviewMaxIters = 200

// resolveFSRoot folds the deprecated -sandbox into -fs-root.
//
// The old name was renamed rather than kept because it was measurably
// misleading: a run with -sandbox set refused view_file on /etc/hosts and then
// read the same file with bash, in the same turn. A flag that names a boundary
// it does not enforce is worse than no flag. The alias keeps existing callers
// working; the warning tells them the name changed and why.
func resolveFSRoot(o *options, logger *slog.Logger) {
	if o.sandbox == "" {
		return
	}
	if o.fsRoot != "" {
		logger.Warn("-sandbox is a deprecated alias for -fs-root; both were given, using -fs-root",
			"fs_root", o.fsRoot, "sandbox", o.sandbox)
		return
	}
	logger.Warn("-sandbox is deprecated, use -fs-root: it confines the file tools and NOT bash",
		"fs_root", o.sandbox)
	o.fsRoot = o.sandbox
}

// dumpMetrics writes the registry out, once, on the way to exit.
//
// Deferred rather than written after the run, so a bad -permission, an
// unreadable transcript or a panic-free early return still produces whatever
// was measured before things went wrong. Written to a temporary file and
// renamed, because the consumer is a textfile collector that reads the
// directory on its own schedule and would otherwise eventually read a
// half-written exposition and treat the missing half as series that no longer
// exist.
//
// Failures go to stderr and change nothing else. stdout is a JSONL protocol
// and the run's exit status is about the turn, not about telemetry: a full
// disk must not turn a completed turn into a failed one.
func dumpMetrics(reg *metric.Registry, path string) {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".*.tmp")
	if err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: metrics: %v\n", err)
		return
	}
	defer os.Remove(tmp.Name()) // a no-op once the rename below has succeeded

	if err := reg.WriteText(tmp); err != nil {
		tmp.Close()
		fmt.Fprintf(os.Stderr, "wombat-jsonl: metrics: %v\n", err)
		return
	}
	// CreateTemp makes the file 0600, and the reader is a collector daemon
	// running as somebody else. A dump nobody can open is the same as no dump,
	// and it fails silently on the far side where nobody is looking.
	if err := tmp.Chmod(0o644); err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: metrics: %v\n", err)
	}
	if err := tmp.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: metrics: %v\n", err)
		return
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: metrics: %v\n", err)
	}
}

// outcomeClass is the vocabulary of the "outcome" label, and the reason
// [metric.WithErrorClass] exists.
//
// The metric package cannot write this function: saying "denied" means knowing
// about permission.ErrDenied, and metric importing permission would make it
// un-importable FROM permission. So the host supplies the words.
//
// The split that earns its keep is denied versus error. Under -permission
// anything but off this front end refuses every Ask by design (see
// [errNoConsole]), so a headless run produces refusals as a matter of routine
// — counting them as failures would make every dashboard panel red and teach
// whoever reads it to ignore the colour. Cancelled is the same argument for a
// SIGTERM from CI.
//
// Timeout is checked BEFORE cancellation deliberately: a tool that outran
// tool.WithTimeout is a cancelled context underneath and would otherwise match
// the broader arm, and it is the one of the two that means something is wrong.
//
// Five fixed words, so this adds at most five series per tool — see the
// cardinality section of the metric package doc for why that promise matters.
func outcomeClass(err error) string {
	switch {
	case err == nil:
		return "ok"
	case errors.Is(err, permission.ErrDenied):
		return "denied"
	case errors.Is(err, tool.ErrTimeout):
		return "timeout"
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return "cancelled"
	default:
		return "error"
	}
}

// build assembles the agent. Every dependency is supplied here, at
// construction: the client, the tools, and the tools' own dependencies.
//
// m is nil unless -metrics-file was given, and every use of it is guarded
// rather than replaced with a no-op middleware: a layer that measures nothing
// still costs a closure per call and still appears in a trace as something
// that ran.
func build(
	o options,
	logger *slog.Logger,
	tp *tape.Tape,
	tracer *trace.Tracer,
	m *metric.Metrics,
) (*wombat.Agent, error) {
	var client llm.Client
	var err error

	// ConfigFromEnv supplies the deployment's defaults; the flags override
	// them. The library itself never reads the environment behind the
	// caller's back, so this is the one place configuration is implicit — and
	// it is a main package, which is where implicit configuration belongs.
	switch o.provider {
	case "anthropic", "claude":
		cfg := anthropic.ConfigFromEnv()
		if o.model != "" {
			cfg.Model = o.model
		}
		if o.stream != "" {
			if cfg.Stream, err = parseStream(o.stream); err != nil {
				return nil, err
			}
		}
		client, err = anthropic.New(cfg)
	case "openai":
		cfg := openai.ConfigFromEnv()
		if o.model != "" {
			cfg.Model = o.model
		}
		if o.stream != "" {
			if cfg.Stream, err = parseStream(o.stream); err != nil {
				return nil, err
			}
		}
		client, err = openai.New(cfg)
	default:
		return nil, fmt.Errorf("unknown provider %q (want anthropic or openai)", o.provider)
	}
	if err != nil {
		return nil, err
	}

	// Order is semantic. Cost is charged per attempt, so it sits inside retry.
	// Overflow recovery sits outside retry, so a transient blip is retried at
	// the current transcript size instead of escalating. The tape is
	// outermost of the transport concerns, so a replayed call skips all of it.
	client = llm.Chain(client,
		llm.WithValidation,
		wombat.TrackCost(llm.DefaultPricing),
		llm.WithRetry(llm.DefaultRetryPolicy),
		wombat.WithOverflowRecovery(),
		llm.WithLogging(logger),
	)
	if tp != nil {
		client = llm.Chain(client, tp.LLM())
	}
	if m != nil {
		// Outermost, in a chain of its own so it stays outermost however the
		// list above grows: a call retried three times is one call, which is
		// the boundary a latency histogram has to draw to mean anything.
		// Outside the tape too, so a replayed turn still reports the tokens it
		// would have spent — which is the number a CI budget check wants.
	}

	// Resolve the working directory before anything sees it. The file and
	// shell tools all demand absolute paths, and a model handed a relative
	// one has no way to expand it — observed behavior was a run that spent
	// its entire iteration budget shelling around the filesystem looking for
	// the project it had been pointed at as ".".
	workdir, err := filepath.Abs(o.workdir)
	if err != nil {
		return nil, fmt.Errorf("resolving working directory %q: %w", o.workdir, err)
	}

	// Everything below assembles the tool surface, and the ORDER is the whole
	// point. Three earlier attempts got it wrong in the same way: each layer
	// was appended as it was thought of, and wombat.WithTools resets the tool
	// SET, so registering the delegate tool last silently discarded the
	// skill-gated set built before it. A run then looked correct and had no
	// gating at all.
	//
	// So: decide the tool list completely, THEN gate it, THEN install it once.

	var shared []wombat.Option
	if tp != nil {
		shared = append(shared, wombat.WithToolMiddleware(tp.Tools()))
	}
	if tracer != nil {
		shared = append(shared, wombat.WithTracing(tracer))
	}

	// The gate goes in `shared`, so the sub-agent is governed by the same
	// policy as its parent. A child that could run what its parent had to ask
	// about would make the whole thing decorative — the same argument that
	// puts the skill registry on both. permission.Gate also follows the
	// delegate call into the child by itself, so this is belt and braces; the
	// copy the child carries and the copy it inherits are one gate, and decide
	// each call once.
	//
	// Position in the chain is load-bearing twice over, and both are satisfied
	// by WithToolMiddleware rather than by anything written here: every
	// caller-supplied middleware sits OUTSIDE tool.WithTimeout, so a blocked
	// approval is not guillotined by the per-call deadline, and outside
	// tool.WithRetry, so an approved call is not re-asked on the second
	// attempt.
	gate, err := permissionGate(o.permission, workdir)
	if err != nil {
		return nil, err
	}
	if gate != nil {
		shared = append(shared, wombat.WithToolMiddleware(gate))
	}

	// measure is applied LAST at both call sites below, after gateOptions, and
	// that ordering is load-bearing. WithToolMiddleware appends and a later
	// entry wraps an earlier one, so installed any sooner this would sit
	// INSIDE the permission gate and inside the skill gate and would never see
	// a call either of them refused. The denied/error split that outcomeClass
	// exists for would read zero denials for ever, which looks exactly like a
	// healthy run.
	//
	// Being last also puts it outside retry, the circuit breaker and the
	// per-call timeout, so one dispatch is one observation with the attempts
	// already collapsed. Both agents share it on purpose: a sub-agent's tool
	// calls cost the same money and break in the same ways as its parent's.
	var measure []wombat.Option
	// One switch rather than three hand-wired pieces: WithMetrics installs the
	// LLM middleware, the tool middleware and the run counters together, and
	// applies them in New AFTER every other option — so the tool middleware
	// lands outside the permission gate and a refused call is actually counted.
	if m != nil {
		measure = append(measure, wombat.WithMetrics(m, metric.WithErrorClass(outcomeClass)))
	}

	runner, err := newRunner(o, workdir)
	if err != nil {
		return nil, err
	}
	fsys := builtin.OSFS(o.fsRoot)
	tools := builtin.Default(builtin.Deps{
		FS:   fsys,
		Exec: runner,
		HTTP: &http.Client{Timeout: 30 * time.Second},
		Now:  time.Now,
	})
	if o.media {
		tools = append(tools, builtin.ViewMedia(fsys))
	}
	// Before the read-only filter and the delegate's, so a server's tools are
	// held to the same capability rules as the built-in ones.
	tools = append(tools, o.mcpConns.Tools()...)
	if o.readOnly {
		tools = tool.Filter(tools, tool.OnlyCaps(tool.CapReadOnly|tool.CapNetwork|tool.CapPause))
	}

	// 1. Skills are discovered once, here. The catalogue goes into the system
	// prompt and is therefore part of the prompt-cache prefix, so it has to be
	// fixed before the first call; loading and unloading are what happen
	// during a run.
	var reg *skill.Registry
	if o.skillsDir != "" {
		skills, err := skill.LoadDir(o.skillsDir, func(err error) {
			logger.Warn("skipping skill", "err", err)
		})
		if err != nil {
			return nil, fmt.Errorf("loading skills: %w", err)
		}
		if len(skills) > 0 {
			reg = skill.New(skills...)
			// Demonstration gate: history archaeology is a distinct mode, and
			// its two tools are noise in the surface until the model says it
			// is doing that. Real deployments register gates from a manifest.
			for _, t := range []string{"git_log", "git_show"} {
				if _, ok := reg.GateFor(t); !ok {
					reg.Gate("git-history", t)
				}
			}
			logger.Info("skills loaded", "count", len(skills), "dir", o.skillsDir)
		}
	}

	// 2. The sub-agent, if asked for. Same client and middleware, read-only
	// tools, a short leash, and no delegate tool of its own so recursion stops
	// here. It is gated by the SAME registry: a child that could reach tools
	// its parent has to unlock would make the gate decorative.
	if o.delegate {
		childTools := tool.Filter(tools, tool.OnlyCaps(tool.CapReadOnly|tool.CapNetwork))
		childOpts := append([]wombat.Option{
			wombat.WithName("researcher"),
			wombat.WithClient(client),
			wombat.WithModel(o.model),
			wombat.WithPurpose(llm.PurposeSubagent),
			wombat.WithMaxIters(12),
			wombat.WithLogger(logger),
		}, shared...)
		childOpts = append(childOpts, gateOptions(reg, childTools)...)
		childOpts = append(childOpts, measure...)

		child, err := wombat.New(childOpts...)
		if err != nil {
			return nil, fmt.Errorf("building the sub-agent: %w", err)
		}
		tools = append(tools, wombat.DelegateTool(child))
	}

	// 3. Now the list is final, install it — gated if there is a registry.
	opts := append([]wombat.Option{
		wombat.WithName("wombat"),
		wombat.WithClient(client),
		wombat.WithModel(o.model),
		wombat.WithMaxIters(o.maxIters),
		wombat.WithMaxTokens(o.maxTokens),
		wombat.WithToolParallelism(o.parallel),
		wombat.WithTurnNotice(iterationNotice(o.maxIters)),
		wombat.WithLogger(logger),
	}, shared...)
	opts = append(opts, gateOptions(reg, tools)...)
	opts = append(opts, measure...)
	if o.check != "" {
		opts = append(opts, wombat.WithCompletionCheck(o.checkAttempts, wombat.CommandCheck(runner, o.check, 0)))
	}
	if o.window > 0 {
		// Elide below the compaction threshold, so a compaction is the rare
		// event and elision the everyday one.
		opts = append(opts, wombat.WithContextWindow(o.window), wombat.WithStrategy(wombat.TokenWindow(0.7, 0.5)))
		if o.compactAt > 0 && o.compactAt < 1 {
			opts = append(opts, wombat.WithCompaction(int(o.compactAt*float64(o.window)), o.window/4,
				wombat.Summarize(client, o.model)))
		}
	}

	// Grants are installed on the PARENT only. A sub-agent runs on a context
	// derived from its parent's, so it inherits this set and an approval given
	// once is not asked again on the other side of a delegation; giving the
	// child its own decorator would replace the inherited set with an empty
	// one and undo exactly that.
	if gate != nil {
		opts = append(opts, wombat.WithRunContext(func(ctx context.Context) context.Context {
			return permission.WithGrants(ctx, permission.NewGrants())
		}))
	}

	if o.delegate {
		// A delegated task runs a whole child agent; 120 seconds of real work
		// is nothing. The child's own iteration cap and the run budget bound
		// it instead.
		opts = append(opts, wombat.WithToolTimeoutFallback(0))
	}
	if o.system != "" {
		opts = append(opts, wombat.WithSystemPrompt(o.system))
	}

	if notes := o.mcpConns.Instructions(); notes != "" {
		opts = append(opts, wombat.WithEnvBlock("mcp_servers",
			"What the MCP servers say about their tools, which are named <server>__<tool>:\n\n"+notes))
	}

	// Say what the directory is FOR, not just what it is. A bare path in the
	// system prompt does not connect to the bash tool's exec_dir parameter in
	// the model's head; naming the parameter does.
	opts = append(opts, wombat.WithEnvBlock("working_directory",
		workdir+"\n\nThis is the project you are working on. Pass it as exec_dir "+
			"to the bash tool and use it as the base for any relative path, "+
			"unless the user names a different location."))

	return wombat.New(opts...)
}

// gateOptions installs a tool list, gated by reg when there is one. Returning
// the options rather than applying them keeps the one rule that matters
// visible at both call sites: the tool set is installed exactly once, after
// the list is final.
func gateOptions(reg *skill.Registry, tools []tool.Def) []wombat.Option {
	if reg == nil {
		return []wombat.Option{wombat.WithTools(tools...)}
	}
	// Install is the set, the middleware that enforces it, the two system
	// blocks — installed skills first, as method before reference — and the
	// per-run activation state, as one option, so none of them can be left
	// out.
	return []wombat.Option{reg.Bind(tools).Install()}
}

// permissionGate turns -permission into tool middleware, or nil for "off".
//
// Nil rather than an always-allow gate: "off" has to be exactly today's
// behaviour, and an installed middleware that decides nothing still costs a
// context lookup and still shows up in a trace as a layer that ran.
func permissionGate(mode, root string) (tool.Middleware, error) {
	var p permission.Policy
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "off", "":
		return nil, nil
	case "readonly", "read-only":
		p = permission.ReadOnly()
	case "workspace":
		p = permission.Workspace(root)
	case "ask":
		p = permission.AskEverything()
	default:
		return nil, fmt.Errorf("bad -permission %q (want off, readonly, workspace or ask)", mode)
	}
	return permission.Gate(p, permission.ApproverFunc(refuseToAsk)), nil
}

// errNoConsole is what an Ask becomes in a headless front end.
//
// It is an error rather than a bare false so the model reads a sentence
// instead of a verdict: it can then choose a different approach — read the
// file with view_file instead of cat, say — rather than retrying the same
// call and burning the iteration budget on a question nobody is going to
// answer.
var errNoConsole = errors.New(
	"this operation needs a human approval, and wombat-jsonl cannot collect one: " +
		"its stdout is a JSONL protocol and it has no console to prompt on. " +
		"Shell commands run without approval only when the WHOLE command is built " +
		"from these, with no substitution, no redirect to a file, no ~ and no ..: " +
		strings.Join(permission.DefaultSafeCommands, ", ") + ". " +
		"A pipeline of them is fine and so is 2>&1, but `; && || $() >file` are not — " +
		"reach for view_file and grep_search instead of cat and grep where you can. " +
		"Otherwise try an approach the policy allows outright, or tell the user to re-run " +
		"with a front end that can ask")

// refuseToAsk is the headless approver: it denies, with an explanation.
//
// The signature still takes a context it never waits on, because that is the
// [permission.Approver] contract — an approver that CAN ask blocks on it.
func refuseToAsk(context.Context, permission.Request) (bool, error) {
	return false, errNoConsole
}

func openTape(o options) (*tape.Tape, error) {
	if o.tapePath == "" {
		return nil, nil
	}
	var mode tape.Mode
	switch strings.ToLower(o.tapeMode) {
	case "auto", "":
		mode = tape.Auto
	case "record":
		mode = tape.Record
	case "replay":
		mode = tape.Replay
	default:
		return nil, fmt.Errorf("bad -tape-mode %q (want auto, record or replay)", o.tapeMode)
	}
	return tape.Open(o.tapePath, mode)
}

// openTrace returns a tracer and a function that flushes it, rendering the
// HTML report if one was asked for.
func openTrace(o options, logger *slog.Logger) (*trace.Tracer, func(), error) {
	if o.tracePath == "" {
		return nil, func() {}, nil
	}
	sink, closer, err := trace.FileSink(o.tracePath)
	if err != nil {
		return nil, func() {}, fmt.Errorf("opening the trace: %w", err)
	}
	return trace.New(sink), func() {
		_ = closer.Close()
		if o.traceHTML == "" {
			return
		}
		if err := renderTrace(o.tracePath, o.traceHTML); err != nil {
			logger.Error("rendering the trace report", "err", err)
		}
	}, nil
}

func renderTrace(in, out string) error {
	spans, err := trace.ReadFile(in)
	if err != nil {
		return err
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()
	return trace.WriteHTML(f, spans)
}

// newRunner is the Runner every exec-flavoured tool spawns through: confined
// when -confine asks for it, and otherwise the plain one, which still filters
// the environment so the provider key this process holds stays here.
// connectMCP starts or reaches every server in the -mcp file. The handshake
// and the tool listing get a minute between them; a server that needs longer
// than that to start is not one to wait on silently.
func connectMCP(path string, logger *slog.Logger) (*mcp.Connections, error) {
	servers, err := mcp.LoadConfig(path)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	conns, err := mcp.ConnectAll(ctx, servers)
	if err != nil {
		return nil, err
	}
	logger.Info("mcp servers connected", "servers", len(servers), "tools", len(conns.Tools()))
	return conns, nil
}

func newRunner(o options, workdir string) (builtin.Runner, error) {
	if !o.confine {
		return builtin.OSRunner(workdir), nil
	}
	network, err := builtin.ParseNetwork(o.confineNet)
	if err != nil {
		return nil, err
	}
	r, err := builtin.SandboxRunner(workdir, builtin.Sandbox{Network: network})
	if err != nil {
		return nil, fmt.Errorf("-confine: %w", err)
	}
	return r, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func parseStream(s string) (llm.StreamMode, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "auto", "":
		return llm.StreamAuto, nil
	case "always", "on", "true":
		return llm.StreamAlways, nil
	case "never", "off", "false":
		return llm.StreamNever, nil
	default:
		return 0, fmt.Errorf("bad -stream %q (want auto, always or never)", s)
	}
}

// input decides what this invocation is continuing from.
//
// -resume reads and never writes; -session reads if the file is there and is
// rewritten when the turn ends. A -session file that does not exist yet is a
// fresh conversation, not an error, because that is what the first turn of a
// new session looks like and making the user create an empty file first would
// be a worse interface than the one this replaces.
func input(o options, query string, set tool.Set) (wombat.Input, error) {
	path := o.resumePath
	if path == "" {
		path = o.sessionPath
	}
	if path == "" {
		return wombat.Ask(query), nil
	}

	b, err := os.ReadFile(path)
	switch {
	case errors.Is(err, fs.ErrNotExist) && path == o.sessionPath:
		return fresh(query)
	case err != nil:
		return wombat.Input{}, fmt.Errorf("reading transcript: %w", err)
	}

	var prior []llm.Message
	if err := json.Unmarshal(b, &prior); err != nil {
		return wombat.Input{}, fmt.Errorf("parsing transcript: %w", err)
	}
	if len(prior) == 0 {
		return fresh(query)
	}
	// wombat.Resume, not wombat.Then: a transcript suspended on ask_user must
	// receive the query as the ANSWER to the question the model asked, not as a
	// new instruction preceded by "(cancelled)".
	return wombat.Resume(prior, query, set), nil
}

// fresh starts a new conversation, refusing an empty one.
//
// The usage check accepts -session with no query, because resuming a saved
// conversation without adding anything is a real thing to want. It is not a
// real thing to want when there is nothing to resume: wombat.Ask("") builds a
// user turn holding an empty text block, Convo.Validate is happy with it (the
// slice is non-empty and starts with a user turn), and the provider answers
// "text content blocks must be non-empty" after the request has already gone
// out. Saying so here costs nothing and names the actual problem.
func fresh(query string) (wombat.Input, error) {
	if query == "" {
		return wombat.Input{}, errors.New(
			"nothing to do: there is no saved conversation to continue and no query was given")
	}
	return wombat.Ask(query), nil
}

// checkSessionWritable fails before the model is called, rather than after.
//
// saveSession runs on the way out and can only report to stderr — stdout is a
// JSONL protocol and the exit status is about the turn, so a save that fails
// there is invisible to a front end reading stdout, which sees a clean `done`
// and reasonably concludes the conversation persisted. Every subsequent
// invocation then starts from zero. The tokens are already spent by that point,
// which is what makes it worth one stat call up front.
func checkSessionWritable(path string) error {
	if path == "" {
		return nil
	}
	if st, err := os.Stat(path); err == nil {
		if st.IsDir() {
			return fmt.Errorf("-session %s is a directory", path)
		}
		return nil
	}
	// Absent is fine — that is a new session — but its directory has to exist
	// and take a file, which is what saveSession will need at the end.
	//
	// The directory is created rather than demanded. builtin's WriteFile takes
	// the same view for the same reason: the caller asked for a path, not for a
	// lecture about mkdir -p, and refusing costs them a round trip to learn
	// something the harness could just do. A typo then creates one empty
	// directory, which is a much smaller problem than the alternative.
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("-session %s: %w", path, err)
	}
	probe, err := os.CreateTemp(dir, "."+filepath.Base(path)+".probe*")
	if err != nil {
		return fmt.Errorf("-session %s is not writable: %w", path, err)
	}
	name := probe.Name()
	probe.Close()
	return os.Remove(name)
}

// saveSession writes the transcript so the next invocation can pick it up.
//
// Written on every exit path, including failure, and that is the case it exists
// for: a run that died on a budget cap or stopped to ask a question is exactly
// the one worth resuming, and a save that only happened on success would drop
// it. The caller defers this.
//
// Written to a temporary file and renamed, so a reader — the next invocation,
// or a front end watching the file — never sees a half-written transcript, and
// a crash mid-write leaves the previous turn's file intact rather than a
// truncated one. A failure is reported on stderr and changes nothing else:
// stdout is a JSONL protocol and the exit status is about the turn.
func saveSession(path string, msgs []llm.Message) {
	if path == "" || len(msgs) == 0 {
		return
	}
	b, err := json.MarshalIndent(msgs, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: encoding the session: %v\n", err)
		return
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: writing the session: %v\n", err)
		return
	}
	name := tmp.Name()
	defer os.Remove(name) // no-op once the rename has succeeded

	if _, err := tmp.Write(append(b, '\n')); err != nil {
		tmp.Close()
		fmt.Fprintf(os.Stderr, "wombat-jsonl: writing the session: %v\n", err)
		return
	}
	// Sync before the rename, or the rename can be durable while the data
	// blocks are not — a power loss then leaves a zero-length session file and
	// the PREVIOUS turn's transcript is gone too, which is worse than the
	// truncated file the temp-and-rename dance was protecting against.
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		fmt.Fprintf(os.Stderr, "wombat-jsonl: writing the session: %v\n", err)
		return
	}
	if err := tmp.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: writing the session: %v\n", err)
		return
	}
	// os.CreateTemp makes the file 0600. Renaming it over the session would
	// silently strip whatever the user or a front end running as another
	// account had set, on every single turn.
	mode := os.FileMode(0o600)
	if st, err := os.Stat(path); err == nil {
		mode = st.Mode().Perm()
	}
	if err := os.Chmod(name, mode); err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: writing the session: %v\n", err)
		return
	}
	if err := os.Rename(name, path); err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: writing the session: %v\n", err)
	}
}

// classOf gives the front end a stable machine-readable failure class, so it
// can render "out of budget" differently from "the model refused" without
// pattern-matching on prose.
func classOf(err error) string {
	switch {
	case errors.Is(err, governor.ErrBudgetExhausted):
		return "budget_exhausted"
	case errors.Is(err, governor.ErrWallClock):
		return "wall_clock"
	case errors.Is(err, governor.ErrStepLimit), errors.Is(err, wombat.ErrMaxIterations):
		return "max_iterations"
	case errors.Is(err, wombat.ErrMaxTokens):
		return "max_tokens"
	case errors.Is(err, wombat.ErrRefused):
		return "refused"
	case errors.Is(err, llm.ErrContextWindow):
		return "context_window"
	case errors.Is(err, llm.ErrAuth):
		return "auth"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	default:
		return "error"
	}
}

// ===== stdout =====

// emitter serializes writes.
//
// A mutex rather than "just write from one goroutine", because tool
// observations can be emitted from a parallel batch. One interleaved write
// corrupts a line, and the reader on the other side is a line reader.
//
// There is no flush and no fsync, and neither is missing. Writes to an
// *os.File are not buffered in this process, so each Encode is one write(2)
// and the line is in the reader's pipe the moment it returns. fsync is about
// surviving a power cut, not about being seen; it used to follow every line,
// and with stdout redirected to a file that was a disk flush per streamed
// delta — hundreds a turn. What has to survive a crash is the transcript, and
// -session writes that with an fsync of its own.
type emitter struct {
	mu  sync.Mutex
	enc *json.Encoder
}

func newEmitter(f *os.File) *emitter {
	enc := json.NewEncoder(f)
	// Model output routinely contains <, > and &. HTML escaping would corrupt
	// it for every consumer that is not a browser.
	enc.SetEscapeHTML(false)
	return &emitter{enc: enc}
}

func (e *emitter) emit(v any) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.enc.Encode(v); err != nil {
		fmt.Fprintf(os.Stderr, "wombat-jsonl: encode: %v\n", err)
	}
}

type sessionStarted struct {
	Type      string  `json:"type"`
	Model     string  `json:"model,omitempty"`
	Provider  string  `json:"provider"`
	Query     string  `json:"query,omitempty"`
	MaxIters  int     `json:"max_iters"`
	BudgetUSD float64 `json:"budget_usd,omitempty"`
}

type done struct {
	Type   string `json:"type"`
	Answer string `json:"answer"`
}

// iterationNotice tells the model how much of its iteration budget is left.
//
// It had no way to know. A measured run spent all 80 of its iterations writing
// correct code and stopped mid-work, with the tests the task explicitly asked
// for never written — not a misjudgement, since nothing in the loop said how
// much room was left or that anything was still outstanding.
//
// It starts at two thirds rather than at the end. A warning delivered with two
// iterations left can only be obeyed by stopping; one delivered with a third of
// the budget left can be obeyed by finishing.
//
// The note goes into the materialized request only, never the transcript, so it
// cannot accumulate.
func iterationNotice(maxIters int) func(context.Context, int) string {
	if maxIters <= 0 {
		return nil
	}
	const startAt = 2.0 / 3.0
	return func(_ context.Context, iter int) string {
		if float64(iter) < startAt*float64(maxIters) {
			return ""
		}
		left := maxIters - iter
		if left < 0 {
			left = 0
		}
		return fmt.Sprintf(
			"<budget_status>Iteration %d of %d; %d left, and the run ends when they "+
				"are gone.\n\nBefore doing anything else, check what the task asked for "+
				"that does not exist yet — re-read your checklist file if you wrote one, "+
				"and the task's own list of requirements. Finish what is missing before "+
				"improving what is already there. Do not shorten a checklist to make it "+
				"look complete.</budget_status>",
			iter, maxIters, left)
	}
}

// accepted reports one pass of the deterministic gate.
type accepted struct {
	Type    string   `json:"type"`
	Attempt int      `json:"attempt"`
	OK      bool     `json:"ok"`
	Summary string   `json:"summary"`
	Failed  []string `json:"failed,omitempty"`
	Skipped []string `json:"skipped,omitempty"`
}

// reviewed reports what the post-turn review found.
//
// Emitted after done, so a consumer that only renders the answer is unchanged
// and one that cares reads the next line. Ran is separate from a zero Findings
// count on purpose: a review that never started and one that looked and found
// nothing are the same number and opposite facts.
type reviewed struct {
	Type     string          `json:"type"`
	Ran      bool            `json:"ran"`
	Error    string          `json:"error,omitempty"`
	Summary  string          `json:"summary"`
	Rounds   int             `json:"rounds,omitempty"`
	Findings []reviewFinding `json:"findings,omitempty"`

	// Unfinished is what the author's own TODO file still has unticked.
	Unfinished []string `json:"unfinished,omitempty"`

	// Discarded counts test sites that throw away a call's last return value.
	Discarded int `json:"discarded_error_sites,omitempty"`

	// PlanEdited names planning artifacts an agent rewrote during the run.
	PlanEdited []string `json:"plan_edited,omitempty"`

	// Sweeps is how many review passes ran; Converged says whether the run
	// ended because they stopped finding things, rather than running out.
	Sweeps    int  `json:"sweeps,omitempty"`
	Converged bool `json:"converged"`
}

type reviewFinding struct {
	Test    string `json:"test"`
	File    string `json:"file"`
	Message string `json:"message"`
	Fixed   bool   `json:"fixed"`
}

type waiting struct {
	Type      string          `json:"type"`
	ToolUseID string          `json:"tool_use_id"`
	Question  string          `json:"question,omitempty"`
	Schema    json.RawMessage `json:"schema,omitempty"`
}

type submitted struct {
	Type    string          `json:"type"`
	Tool    string          `json:"tool"`
	Payload json.RawMessage `json:"payload"`
}

type failed struct {
	Type   string `json:"type"`
	Reason string `json:"reason"`
	Class  string `json:"class,omitempty"`
}

// planOnlyPrompt turns a build task into a planning task.
//
// The split exists because the two halves need different things. Knowing what
// must be verified is a question of breadth — how many ways this kind of system
// has failed before — and a measured run showed the cost of missing one: five of
// nine defects in one build came from a single unconsidered question, whether a
// task handle could be confused with an integer. Knowing how to write the code
// needs the code in front of you, which planning does not have.
//
// So the plan is deliberately not asked for test code. Before any of the API
// exists, a test file is a guess at names and signatures, and a wrong guess does
// not compile and blocks the build. Prose that names its own test is checkable
// without being written blind.
func planOnlyPrompt(task string) string {
	return `Plan this task. Write no implementation code and no test code.

Produce exactly three files in the working directory, then answer with a one
paragraph summary.

PLAN.md — the invariants, stated as facts that hold at every moment, each with
the paths that establish it and the paths that could break it. Architecture and
file layout.

TODO.md — an ordered markdown checklist, "- [ ] item", covering everything the
task requires, including a test for each invariant. Nothing in it may depend on
something below it.

Verifier.md — what must be TRUE of the finished work, one item per heading:

## V1. <the single thing that must hold>
Given: <the situation that exercises it>
Expect: <what must happen, specifically>
Status: [ ]
Test:

An item that a COMMAND decides — the build succeeds, vet is clean, a suite
passes, the CLI exits zero on some input — says so instead, and needs no Given
or Expect:

## V2. the three acceptance commands succeed
Command: go test -race ./...
Status: [ ]

Use Command for anything the task states AS a command. Forcing those through a
test function adds a layer that can fail on its own: asked to prove "go test
-race ./... passes" with a test, one build wrote a test that ran the whole
suite as a subprocess, and the suite then took longer than the requirement
allowed — defeated by the code written to prove it.

Whoever builds this will fill in Status and Test. An item is settled only by a
test that exists and passes, so write each one so a test could decide it: not
"errors are handled well" but "dividing by zero returns an *Error whose Line is
the line of the division".

Spend your effort on the items nobody would think to write down. Requirements
get implemented; what gets shipped broken is the consequence of a requirement
that nobody stated. For each value the system represents, ask whether two kinds
of it can be confused. For each thing acquired, ask what releases it. For each
"after X, Y is true", ask what must stop being allowed once Y is true. For each
operation that reports success, ask what is true when its last step fails.

Cover the task's stated requirements too, but those are the cheap half.

THE TASK

` + strings.TrimSpace(task)
}
