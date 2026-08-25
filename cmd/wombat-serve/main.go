// Command wombat-serve is an agent you can hold a conversation with over HTTP.
//
//	go run ./cmd/wombat-serve
//	open http://localhost:8080
//
// Everything about sessions, streaming and approvals lives in
// [github.com/hoobean1996/wombat-go/httpapi], which is an [http.Handler] and
// is tested without a socket. What is left here is the part that genuinely
// belongs to a command: reading flags, building one provider client, and
// deciding what an agent is allowed to do — per session, because the directory
// the file tools are rooted at and the policy that gates them are exactly the
// choices a client makes when it starts a conversation.
//
// The previous version of this command was a one-shot-turn demo: a session was
// a single run, the prompt rode in a query string, and a page that reloaded
// had lost everything. See the httpapi package for the resource model that
// replaced it.
package main

import (
	"context"
	"embed"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/httpapi"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/anthropic"
	"github.com/hoobean1996/wombat-go/llm/openai"
	"github.com/hoobean1996/wombat-go/metric"
	"github.com/hoobean1996/wombat-go/permission"
	"github.com/hoobean1996/wombat-go/tool"
	"github.com/hoobean1996/wombat-go/tool/builtin"
)

//go:embed index.html
var assets embed.FS

// shutdownGrace is how long an in-flight request has to finish once a signal
// arrives. Short, because the only long-lived response is the event stream and
// nothing is lost by dropping one: the session is being cancelled anyway, and
// a client that reconnects will be told there is no such session.
const shutdownGrace = 5 * time.Second

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	provider := flag.String("provider", envOr("WOMBAT_PROVIDER", "openai"), "anthropic | openai")
	model := flag.String("model", "", "default model id; a session may override it")
	workdir := flag.String("working-dir", ".", "root for the shell and file tools.\n"+
		"A session may name a workspace inside this directory and no session can escape it.")
	budget := flag.Float64("budget", 1.0, "cost cap in USD per TURN, not per conversation.\n"+
		"Per turn because a long conversation should be bounded per exchange; a cumulative\n"+
		"cap kills a session mid-sentence on its fourth question for reasons nobody can see.")
	maxIters := flag.Int("max-iters", 12, "iteration cap per turn, and the ceiling a session may ask for")
	perm := flag.String("permission", "workspace",
		"off | readonly | workspace | ask — the default policy every tool call is checked\n"+
			"against, and the strictest a session may choose. workspace gives the agent its full\n"+
			"tool set but parks exec and anything outside -working-dir on an Allow/Deny card in the\n"+
			"browser. off offers read-only tools and no gate at all; a browser-facing agent with an\n"+
			"ungated shell is not a reasonable default.")
	metrics := flag.Bool("metrics", true,
		"serve Prometheus metrics on GET /metrics. On by default: this process is already\n"+
			"listening, the endpoint costs one handler and an in-memory registry, and a server\n"+
			"whose token spend and tool failures are invisible is one nobody can operate.")
	cors := flag.String("cors", "",
		"comma-separated browser origins allowed to call the API, or * for any.\n"+
			"Empty (the default) sends no CORS headers, which is correct when the bundled UI is\n"+
			"served from this same origin. There is no authentication here, so * lets any page the\n"+
			"operator visits drive this agent with the operator's network access.")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	// permission.Gate audits every verdict through slog.Default, so the
	// deployment's handler has to be installed there too or the audit trail
	// goes to a different stream from everything else.
	slog.SetDefault(logger)

	if err := run(*addr, *provider, *model, *workdir, *perm, *cors,
		*budget, *maxIters, *metrics, logger); err != nil {
		logger.Error("wombat-serve", "err", err)
		os.Exit(1)
	}
}

func run(
	addr, provider, model, workdir, perm, cors string,
	budget float64,
	maxIters int,
	metrics bool,
	logger *slog.Logger,
) error {
	root, err := filepath.Abs(workdir)
	if err != nil {
		return fmt.Errorf("resolving -working-dir: %w", err)
	}
	if _, err := defaultPolicy(perm, root); err != nil {
		return err // fail on a bad -permission now, not on the first session
	}

	client, err := newClient(provider, model, logger)
	if err != nil {
		return err
	}

	// One registry per process. Nil when the flag is off, which is what keeps
	// the instruments out of the middleware chains entirely rather than
	// installing one that measures into nowhere.
	var reg *metric.Registry
	var m *metric.Metrics
	if metrics {
		reg = metric.NewRegistry()
		m = metric.New(reg)
	}

	factory := &factory{
		client:   client,
		root:     root,
		model:    model,
		perm:     perm,
		maxIters: maxIters,
		logger:   logger,
		metrics:  m,
	}

	// Built once at startup for two reasons: it fails a misconfiguration here
	// rather than on a client's first request, and GET /api/config describes
	// the tool surface from an agent that was actually built instead of from a
	// list typed a second time.
	probe, err := factory.build(httpapi.SessionOptions{})
	if err != nil {
		return fmt.Errorf("building the reference agent: %w", err)
	}

	mgr, err := httpapi.NewManager(httpapi.ManagerConfig{
		Build: factory.build,
		Limits: governor.Limits{
			CostUSD: budget,
			// One more than the iteration cap, so a turn that legitimately uses
			// every iteration is stopped by wombat.ErrMaxIterations — which says
			// what happened — rather than by the governor, which says the budget
			// ran out.
			Steps:             maxIters + 2,
			RepeatedToolCalls: 5,
		},
		Logger: logger,
	})
	if err != nil {
		return err
	}
	defer mgr.Close()

	opts := []httpapi.Option{
		httpapi.WithVersion(version()),
		httpapi.WithUI(assets),
		httpapi.WithCapabilities(httpapi.Capabilities{
			DefaultModel: model,
			Approvals:    perm != "off",
			Tools:        httpapi.ToolsOf(probe),
		}),
	}
	if reg != nil {
		opts = append(opts, httpapi.WithMetrics(reg.Handler()))
	}
	if origins := splitList(cors); len(origins) > 0 {
		opts = append(opts, httpapi.WithCORS(origins...))
	}

	srv := &http.Server{
		Addr:    addr,
		Handler: httpapi.New(mgr, opts...),
		// No WriteTimeout: it bounds a whole response, and the event stream is
		// meant to stay open for the length of a conversation. What ends a turn
		// is the per-turn budget; what ends a session is its TTL.
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errs := make(chan error, 1)
	go func() {
		logger.Info("listening", "addr", addr, "provider", provider,
			"permission", perm, "working_dir", root, "metrics", metrics)
		errs <- srv.ListenAndServe()
	}()

	select {
	case err := <-errs:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("serving: %w", err)
		}
		return nil
	case <-ctx.Done():
		logger.Info("shutting down")
		shutdown, cancel := context.WithTimeout(context.Background(), shutdownGrace)
		defer cancel()
		// Shutdown first, then the deferred mgr.Close: draining requests before
		// cancelling the sessions they are reading means a client that is mid-
		// poll gets an answer rather than a reset connection.
		return srv.Shutdown(shutdown)
	}
}

// ===== the per-session agent =====

// factory turns [httpapi.SessionOptions] into an agent.
//
// The client, the metric registry and the tracer are shared — they are process
// resources, and one provider connection pool serving every conversation is
// the point of having one. Everything a session can influence is decided here,
// per session, and CLAMPED: a field a browser can set is a field a hostile
// browser can set, and httpapi deliberately passes the options through without
// looking at them because only the host knows what its deployment allows.
type factory struct {
	client   llm.Client
	root     string
	model    string
	perm     string
	maxIters int
	logger   *slog.Logger
	metrics  *metric.Metrics
}

func (f *factory) build(opts httpapi.SessionOptions) (*wombat.Agent, error) {
	workspace, err := f.workspace(opts.Workspace)
	if err != nil {
		return nil, err
	}
	gate, err := defaultPolicy(f.mode(opts.Permission), workspace)
	if err != nil {
		return nil, err
	}
	model := f.model
	if opts.Model != "" {
		model = opts.Model
	}
	iters := f.maxIters
	if opts.MaxIters > 0 && opts.MaxIters < iters {
		// Only downward. A session may ask for a shorter leash and never a
		// longer one, because the flag is the operator's budget and the option
		// is the client's preference.
		iters = opts.MaxIters
	}

	tools := builtin.Default(builtin.Deps{
		FS:   builtin.OSFS(""),
		Exec: builtin.OSRunner(workspace),
		HTTP: &http.Client{Timeout: 30 * time.Second},
		Now:  time.Now,
	})

	// ask_user suspends the run, and this front end has somewhere to put the
	// answer — a paused turn leaves the session Waiting and the next prompt is
	// routed back through the pause with wombat.AnswerPause. So, unlike the
	// old one-shot server, the pause tools stay.

	// With no gate installed, the only thing keeping a browser-reachable agent
	// away from the shell is the tool list, so the tool list is what does it.
	// With a gate, the full surface is on offer and the policy decides — which
	// is the point of having one.
	if gate == nil {
		tools = tool.Filter(tools, tool.OnlyCaps(tool.CapReadOnly|tool.CapNetwork))
	}

	agentOpts := []wombat.Option{
		wombat.WithName("web"),
		wombat.WithClient(f.client),
		wombat.WithModel(model),
		wombat.WithTools(tools...),
		wombat.WithMaxIters(iters),
		wombat.WithLogger(f.logger),
		wombat.WithEnvBlock("working_directory", workspace+
			"\n\nPass this as exec_dir to shell tools and as the base for relative paths."),
	}
	if gate != nil {
		agentOpts = append(agentOpts, wombat.WithToolMiddleware(gate))
	}
	// AFTER the gate, and that is the whole reason this is down here rather
	// than next to the LLM middleware: WithToolMiddleware appends, and a later
	// entry wraps an earlier one. Installed before the gate this would sit
	// INSIDE it, a refused call would never reach it, and the denied/error
	// split that outcomeClass exists for would count zero denials for ever and
	// look like a healthy dashboard.
	//
	// The position also puts it outside retry, the breaker and the per-call
	// timeout, so one dispatch is one observation with the attempts already
	// collapsed.
	if f.metrics != nil {
		agentOpts = append(agentOpts, wombat.WithMetrics(f.metrics, metric.WithErrorClass(outcomeClass)))
	}
	return wombat.New(agentOpts...)
}

// mode clamps a session's requested policy against the operator's.
//
// Only in the restrictive direction, and "off" is never reachable from a
// request: a client asking for no gate at all is a client asking for an
// ungated shell, and the answer is the operator's flag, not theirs.
func (f *factory) mode(requested string) string {
	want := strings.ToLower(strings.TrimSpace(requested))
	if want == "" {
		return f.perm
	}
	if strictness(want) > strictness(f.perm) {
		return want
	}
	return f.perm
}

// strictness orders the policies so mode can pick the tighter of two.
func strictness(mode string) int {
	switch mode {
	case "off":
		return 0
	case "workspace":
		return 1
	case "ask":
		return 2
	case "readonly", "read-only":
		return 3
	default:
		return -1 // unknown: never wins, and defaultPolicy reports it
	}
}

// workspace resolves a session's requested directory INSIDE the process root.
//
// The check is on the resolved absolute path, not on the string, so "../.."
// and a symlink that points out both fail. Without it, -working-dir would be a
// suggestion: SessionOptions.Workspace arrives from a JSON body, and a browser
// that asked for "/" would get an agent rooted at the filesystem.
func (f *factory) workspace(requested string) (string, error) {
	if strings.TrimSpace(requested) == "" {
		return f.root, nil
	}
	abs := requested
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(f.root, abs)
	}
	abs, err := filepath.Abs(abs)
	if err != nil {
		return "", fmt.Errorf("resolving the requested workspace: %w", err)
	}
	rel, err := filepath.Rel(f.root, abs)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		// httpapi.ErrBadRequest, so this reaches the client as a 400 naming
		// what it asked for. Without it a refusal the CLIENT caused arrives as
		// a 500, which reads as "the server is broken" and sends whoever is
		// debugging it to the wrong logs.
		return "", fmt.Errorf("%w: workspace %q is outside %s",
			httpapi.ErrBadRequest, requested, f.root)
	}
	return abs, nil
}

// defaultPolicy turns a permission mode into tool middleware, or nil for
// "off".
//
// The approver is [httpapi.Approver], which BLOCKS until a person answers over
// the API. That is safe exactly because wombat.WithToolMiddleware installs
// this OUTSIDE tool.WithTimeout and tool.WithRetry: a call waiting on a person
// is not racing a per-call deadline, and one that is approved is not asked
// about again by a retry.
func defaultPolicy(mode, root string) (tool.Middleware, error) {
	var p permission.Policy
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "off":
		return nil, nil
	case "readonly", "read-only":
		p = permission.ReadOnly()
	case "workspace", "":
		p = permission.Workspace(root)
	case "ask":
		p = permission.AskEverything()
	default:
		return nil, fmt.Errorf("bad permission %q (want off, readonly, workspace or ask)", mode)
	}
	return permission.Gate(p, httpapi.Approver()), nil
}

// ===== the provider =====

func newClient(provider, model string, logger *slog.Logger) (llm.Client, error) {
	var client llm.Client
	var err error

	switch provider {
	case "anthropic", "claude":
		cfg := anthropic.ConfigFromEnv()
		if model != "" {
			cfg.Model = model
		}
		client, err = anthropic.New(cfg)
	case "openai":
		cfg := openai.ConfigFromEnv()
		if model != "" {
			cfg.Model = model
		}
		client, err = openai.New(cfg)
	default:
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
	if err != nil {
		return nil, fmt.Errorf("building the %s client: %w", provider, err)
	}

	return llm.Chain(client,
		llm.WithValidation,
		wombat.TrackCost(llm.DefaultPricing),
		llm.WithRetry(llm.DefaultRetryPolicy),
		wombat.WithOverflowRecovery(),
		llm.WithLogging(logger),
	), nil
}

// outcomeClass is the vocabulary of the "outcome" label on this server's tool
// counters, and the reason [metric.WithErrorClass] exists.
//
// The metric package cannot write this function: saying "denied" requires
// knowing about permission.ErrDenied, and metric importing permission would
// make it un-importable FROM permission. So the host supplies the words, and
// the words are chosen for the dashboard rather than for the error hierarchy.
//
// The distinction that earns its keep is denied versus error. A refused tool
// call is the sandbox working — a session under -permission workspace produces
// them all day — and a panel that counts it as a failure pages somebody at 3am
// for a policy doing its job. Likewise cancelled: a browser that closed the
// tab is not an incident.
//
// Timeout is checked BEFORE cancellation on purpose. A tool that outran
// tool.WithTimeout is a cancellation at the context layer and would match the
// broader arm, and "timeout" is both more specific and the only one of the two
// that means something is actually wrong.
//
// Every arm returns one of five fixed words, so this can add at most five
// series per tool — see the cardinality section of the metric package doc for
// what happens to a classifier that does not promise that.
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

// ===== odds and ends =====

// version is what GET /api/health and GET /api/config report. Read from the
// build info rather than a -ldflags variable, so `go install` produces a
// binary that knows what it is.
func version() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "dev"
	}
	for _, s := range info.Settings {
		if s.Key == "vcs.revision" && len(s.Value) >= 12 {
			return s.Value[:12]
		}
	}
	if info.Main.Version != "" {
		return info.Main.Version
	}
	return "dev"
}

func splitList(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}
