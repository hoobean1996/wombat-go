// Package core is a ReAct agent harness: a loop that calls a model, runs
// the tools it asks for, and feeds the results back until the model is done.
//
// An agent is built once and used many times:
//
//	client, err := anthropic.New(anthropic.Config{APIKey: key})
//	if err != nil { return err }
//
//	a, err := core.New(
//	    core.WithClient(client),
//	    core.WithModel("claude-opus-5"),
//	    core.WithTools(builtin.Default(builtin.Deps{})...),
//	)
//
//	ctx, cancel := governor.WithBudget(ctx, governor.Limits{CostUSD: 1.00})
//	defer cancel()
//
//	run := a.Start(ctx, core.Ask("what does this repo do?"))
//	for run.Next() {
//	    switch ev := run.Event().(type) {
//	    case core.TextDelta: os.Stdout.WriteString(ev.Text)
//	    case core.ToolStart: log.Println("→", ev.Name)
//	    }
//	}
//	if err := run.Err(); err != nil { return err }
//
// Everything the agent depends on — the model client, the tools, the tools'
// own dependencies — is supplied at construction. There is no ambient runtime
// and nothing to install: substituting a fake client for a test is
// core.WithClient(fake), and a tool is an ordinary function that can be
// called without a harness at all.
package core

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/metric"
	"wombat/pkg/core/tool"
	"wombat/pkg/core/trace"
)

// Defaults applied by [New] when the corresponding option is not given.
const (
	DefaultMaxIters           = 30
	DefaultMaxTokens          = 8192
	DefaultForceTerminalLastN = 2
	// DefaultToolParallelism is how many read-only calls of one batch run at
	// once. A model routinely asks for several reads in one turn — four greps
	// and a view_file — and running them one after another is latency the
	// user waits through for nothing.
	DefaultToolParallelism   = 4
	DefaultToolRetryAttempts = 3
	DefaultBreakerThreshold  = 5
	DefaultBreakerCooldown   = 60 * time.Second
	DefaultDedupThreshold    = 3
	DefaultToolOutputLimit   = 64 * 1024
	DefaultSystemPrompt      = "You are a helpful assistant. Use the available tools when they help; answer directly when they do not."

	// DefaultToolTimeoutFallback caps a tool that declares no Timeout.
	DefaultToolTimeoutFallback = 120 * time.Second
)

type block struct{ Name, Body string }

type config struct {
	name string

	client  llm.Client
	pricing llm.Pricing

	tools               []tool.Def
	set                 tool.Set
	dispatcher          tool.Dispatcher
	toolMW              []tool.Middleware
	parallel            int
	toolFallbackTimeout time.Duration
	toolFallbackSet     bool
	callEstimate        time.Duration
	untruncated         []string

	systemPrompt string
	systemBlocks []block
	envBlocks    []block

	strategy   Strategy
	window     int
	compaction *compaction
	check      *completionCheck
	maxIters   int
	model      string
	maxTokens  int
	purpose    llm.Purpose
	terminal   string
	forceLastN int

	llmMW       []llm.Middleware
	runCtx      []func(context.Context) context.Context
	turnNotice  func(context.Context, int) string
	tracer      *trace.Tracer
	metrics     *metric.Metrics
	metricOpts  []metric.Option
	eventBuffer int

	logger *slog.Logger
}

// Option configures an [Agent].
//
// Options rather than an exported config struct, because several defaults are
// non-zero (30 iterations, 8192 max tokens, force-terminal in the last 2) and
// a struct literal cannot distinguish "leave it alone" from "set it to zero".
// With options, WithForceTerminalLastN(0) genuinely means off.
type Option func(*config)

// Options combines several options into one, applied in order. It is how a
// feature that needs more than one piece of wiring hands them over as a unit —
// skill.Gated.Install, tape.Tape.Install — so a caller cannot take the tool
// set and forget the middleware that enforces it.
func Options(opts ...Option) Option {
	return func(c *config) {
		for _, o := range opts {
			if o != nil {
				o(c)
			}
		}
	}
}

// WithLLMMiddleware wraps the agent's client in mw when the agent is built,
// outside whatever the client already carries — retry, cost tracking — and
// inside the metrics and tracing layers [New] adds, so a call they observe is
// one logical call whatever happened inside. Later entries are outermost, as
// with llm.Chain.
func WithLLMMiddleware(mw ...llm.Middleware) Option {
	return func(c *config) { c.llmMW = append(c.llmMW, mw...) }
}

// WithName tags the agent in logs, traces and events.
func WithName(s string) Option { return func(c *config) { c.name = s } }

// WithClient sets the model client. Required.
func WithClient(cl llm.Client) Option { return func(c *config) { c.client = cl } }

// WithPricing sets the cost table used for budget accounting.
func WithPricing(p llm.Pricing) Option { return func(c *config) { c.pricing = p } }

// WithTools sets the tool surface. Filter before passing it in — the agent's
// tool list is the single authority for both what the model sees and what can
// execute:
//
//	core.WithTools(tool.Filter(all, tool.OnlyCaps(tool.CapReadOnly))...)
func WithTools(defs ...tool.Def) Option {
	return func(c *config) { c.tools = defs; c.set = nil }
}

// WithToolSet supplies a dynamic surface, for cases where visibility changes
// mid-run (skill gating). Overrides WithTools.
func WithToolSet(s tool.Set) Option { return func(c *config) { c.set = s } }

// WithTruncationExempt names tools whose output the dispatch chain never
// clips to [DefaultToolOutputLimit]. skill.Gated.Install passes load_skill: a
// skill whose middle was cut is worse than none, because the model follows
// instructions it only half saw, and loading it again returns only a reminder
// that it is already loaded.
func WithTruncationExempt(names ...string) Option {
	return func(c *config) { c.untruncated = append(c.untruncated, names...) }
}

// WithToolMiddleware appends to the dispatch chain, outside the built-in
// middleware but inside observation.
func WithToolMiddleware(mw ...tool.Middleware) Option {
	return func(c *config) { c.toolMW = append(c.toolMW, mw...) }
}

// WithDispatcher replaces the whole dispatch pipeline.
//
// Everything the built-in chain does becomes yours to reproduce: panic
// recovery, validation, per-call timeout, retry, the circuit breaker, dedup
// annotation, output truncation, budget accounting, the [Journal]'s record of
// each call, logging and observation.
// Two omissions bite hardest and neither announces itself — without
// [tool.WithObserver] no tool events reach the stream, and without
// [tool.WithRecovery] a panicking tool escapes the innermost guard.
//
// A Dispatcher built by [tool.NewDispatcher] still has the per-call panic
// backstop, since that lives in the dispatcher itself. A hand-written
// Dispatcher implementation has neither layer, and a panic on one of its
// goroutines will take the process down.
//
// Prefer [WithToolMiddleware] unless you genuinely need to change how a batch
// is scheduled.
func WithDispatcher(d tool.Dispatcher) Option { return func(c *config) { c.dispatcher = d } }

// WithToolParallelism runs up to n of one batch's read-only calls concurrently.
// Default [DefaultToolParallelism]; 1 runs every call in order.
//
// Only calls to [tool.CapReadOnly] tools ever overlap: anything that writes,
// executes or declares nothing runs alone, in the order the model wrote it.
// That is what makes a default above one safe — see [tool.WithParallelism].
func WithToolParallelism(n int) Option { return func(c *config) { c.parallel = n } }

// WithToolTimeoutFallback caps any tool that declares no Timeout of its own.
// Default [DefaultToolTimeoutFallback]; 0 removes the cap.
//
// The default exists so a tool that forgets to bound itself cannot hang a run
// forever. It is wrong for exactly one class of tool: [DelegateTool] runs a
// whole child agent, whose natural bound is its own iteration cap and the run
// budget, and 120 seconds of a real task is nothing. A parent that delegates
// should pass 0 here and rely on governor.Limits instead.
func WithToolTimeoutFallback(d time.Duration) Option {
	return func(c *config) { c.toolFallbackTimeout, c.toolFallbackSet = d, true }
}

// WithSystemPrompt sets the base system prompt — the most stable part, and the
// prompt-cache prefix.
func WithSystemPrompt(s string) Option { return func(c *config) { c.systemPrompt = s } }

// WithSystemBlock appends a contributed fragment, rendered as <name>body</name>
// after the base prompt and before the env blocks.
func WithSystemBlock(name, body string) Option {
	return func(c *config) { c.systemBlocks = append(c.systemBlocks, block{name, body}) }
}

// WithEnvBlock appends ambient context — workspace brief, current time,
// project state — rendered as <tag>body</tag> at the end of the system prompt.
func WithEnvBlock(tag, body string) Option {
	return func(c *config) { c.envBlocks = append(c.envBlocks, block{tag, body}) }
}

// WithStrategy sets how the transcript is materialized per call. Default [Flat].
func WithStrategy(s Strategy) Option { return func(c *config) { c.strategy = s } }

// WithContextWindow tells the agent how many input tokens its model accepts.
//
// Nothing can ask a provider for this, and the number differs by model and by
// deployment, so it is configuration. Strategies read it as [View.Window]:
// [TokenWindow] is a fraction of it, and does nothing without it. 0, the
// default, means unknown.
func WithContextWindow(tokens int) Option { return func(c *config) { c.window = tokens } }

// WithMaxIters caps ReAct iterations. Default [DefaultMaxIters].
func WithMaxIters(n int) Option { return func(c *config) { c.maxIters = n } }

// WithModel overrides the client's default model for this agent, so a
// planner on one model and an executor on another can share a client and all
// of its middleware.
func WithModel(m string) Option { return func(c *config) { c.model = m } }

// WithMaxTokens caps the reply length. Default [DefaultMaxTokens].
func WithMaxTokens(n int) Option { return func(c *config) { c.maxTokens = n } }

// WithPurpose tags every call this agent makes, so middleware can branch on
// the semantic role rather than sniffing the prompt.
func WithPurpose(p llm.Purpose) Option { return func(c *config) { c.purpose = p } }

// WithTerminalTool ends the run when the model calls name. The tool's handler
// is never invoked; its arguments become [Submitted].Payload.
//
// The call is still dispatched through the tool middleware for a verdict, so a
// permission policy or a skill gate can refuse it. A refused call is answered
// with the refusal, like any other refused call, and the run continues.
//
// The named tool must be in the set and carry [tool.CapTerminal]; [New]
// rejects the agent otherwise.
func WithTerminalTool(name string) Option { return func(c *config) { c.terminal = name } }

// WithForceTerminalLastN pins tool_choice to the terminal tool once the loop
// is within n iterations of its cap, pushing the model to submit rather than
// run out of iterations with nothing to show. 0 disables. Default
// [DefaultForceTerminalLastN].
func WithForceTerminalLastN(n int) Option { return func(c *config) { c.forceLastN = n } }

// WithRunContext installs a decorator applied to the context once per run,
// inside [Agent.Start].
//
// This is where per-run mutable state is born. An Agent is immutable and
// shared across goroutines, so anything that changes during a run — a skill
// activation set, a scratch workspace, a request-scoped cache — cannot live on
// the Agent without leaking between concurrent runs. Creating it here gives
// each run its own, and a sub-agent that inherits the context inherits the
// state, which is usually what you want.
//
//	core.WithRunContext(func(ctx context.Context) context.Context {
//	    return skill.WithState(ctx, skill.NewState())
//	})
//
// Decorators run in registration order.
func WithRunContext(f func(context.Context) context.Context) Option {
	return func(c *config) { c.runCtx = append(c.runCtx, f) }
}

// WithTurnNotice appends a short note to the last user turn of every request.
//
// It exists because there is otherwise no way to tell a running model
// something. The system prompt is rendered once and frozen — deliberately, it
// is the prompt-cache prefix — and the transcript is the model's own history,
// which the harness should not rewrite. The last turn is the one place left,
// and it is also the cheapest: it sits after every cache breakpoint, so a
// notice that changes each turn disturbs nothing that was cached.
//
// The motivating case is a budget about to run out. Without a warning a
// governed run spends to the cap and is then guillotined mid-thought; with
// one it can decide to summarise instead:
//
//	core.WithTurnNotice(governor.NoticeAt(0.8))
//
// Return "" to say nothing this turn. The note is added to the materialized
// request only — it never enters the stored transcript, so it cannot
// accumulate. Skipped when the last turn is not a user turn.
func WithTurnNotice(f func(ctx context.Context, iter int) string) Option {
	return func(c *config) { c.turnNotice = f }
}

// WithMetrics records this agent's work into m, in one switch.
//
// It wires all four places metrics come from: the LLM middleware around the
// client, the tool middleware in the dispatch chain, and the run and iteration
// counters in the loop. Wiring three of the four and forgetting the last
// produces a dashboard that looks healthy because the thing that went wrong is
// the thing that is not being counted.
//
//	reg := metric.NewRegistry()
//	http.Handle("/metrics", reg.Handler())
//	a, err := core.New(..., core.WithMetrics(metric.New(reg)))
//
// Pass [metric.WithErrorClass] to give an outcome label better words than "ok"
// and "error" — mapping a permission refusal to "denied", say. The metric
// package cannot know those vocabularies without importing the packages that
// define them, which is why the seam is here.
//
// Applied in [New] after every other option, so the tool middleware ends up
// OUTSIDE anything installed with [WithToolMiddleware] — including a
// permission gate, whose refusals would otherwise never be counted.
func WithMetrics(m *metric.Metrics, opts ...metric.Option) Option {
	return func(c *config) { c.metrics, c.metricOpts = m, opts }
}

// WithEventBuffer gives the event channel n slots of slack. Default 0.
//
// A run is consumer-paced: [Run.Next] blocks the agent until the caller reads,
// which is the right default because the alternative is an unbounded queue
// growing behind a consumer that cannot keep up. The cost is that a consumer
// which pauses — a browser tab suspended mid-answer — stops the read from the
// provider, and the client's own idle timeout eventually kills the call.
//
// A small buffer absorbs that without giving up the property: the agent still
// blocks once the slack is used. A reasoning model emits hundreds of deltas per
// turn, so a few hundred slots covers a stall of seconds.
//
// To decouple entirely — a run that keeps going with no consumer at all, which
// is what a resumable HTTP session needs — use [Buffer] instead.
func WithEventBuffer(n int) Option {
	return func(c *config) {
		if n > 0 {
			c.eventBuffer = n
		}
	}
}

// WithTracing installs t as this agent's tracer, in one switch.
//
// It wires three things that are otherwise the caller's to remember: the LLM
// span middleware around the client, the tool span middleware in the dispatch
// chain, and the tracer itself on each run's context so the loop's own spans
// find it. Wiring one and forgetting another produces a report that looks
// complete and is not, which is worse than no report.
//
//	sink, closer, err := trace.FileSink("run.ndjson")
//	defer closer.Close()
//	a, err := core.New(..., core.WithTracing(trace.New(sink)))
//
// Applied in [New] after every other option, so the order relative to
// WithClient does not matter.
func WithTracing(t *trace.Tracer) Option { return func(c *config) { c.tracer = t } }

// WithLogger sets the diagnostic logger. Default [slog.Default].
//
// Diagnostics only. Semantic events go through the [Event] stream; conflating
// the two is what forces a UI to parse log strings.
func WithLogger(l *slog.Logger) Option { return func(c *config) { c.logger = l } }

// Agent is an immutable, concurrency-safe agent definition. Construct with
// [New], derive with [Agent.With], run with [Agent.Start] or [Agent.Run].
type Agent struct {
	cfg config

	// base is the configuration exactly as the options left it, before New
	// layered the metrics and tracing middleware onto the client, the tool
	// chain and the run context. [Agent.With] derives from this and never from
	// cfg: cfg already carries those layers, and handing it back to New would
	// install them a second time — every model call, token and tool call then
	// counted twice, and every span recorded twice, on every derived agent.
	base config

	// system is rendered once, here, and reused by every call. That is what
	// makes the prompt-cache prefix byte-stable across a run: it cannot drift,
	// because there is no code path that rebuilds it.
	system string

	set  tool.Set
	disp tool.Dispatcher
	log  *slog.Logger
}

// New builds an agent.
func New(opts ...Option) (*Agent, error) {
	c := config{
		systemPrompt: DefaultSystemPrompt,
		strategy:     Flat,
		maxIters:     DefaultMaxIters,
		maxTokens:    DefaultMaxTokens,
		purpose:      llm.PurposeOther,
		forceLastN:   DefaultForceTerminalLastN,
		parallel:     DefaultToolParallelism,
		name:         "agent",
		pricing:      llm.DefaultPricing,
	}
	for _, o := range opts {
		o(&c)
	}

	if c.client == nil {
		return nil, errors.New("wombat: no llm client (use WithClient)")
	}
	if c.strategy == nil {
		c.strategy = Flat
	}
	if c.logger == nil {
		c.logger = slog.Default()
	}

	// Kept before anything below is derived from it; see Agent.base. Every
	// slice is clipped first, so an append — the ones below, or a derived
	// agent's options in With — always copies rather than writing into a
	// backing array another agent still shares. Two agents derived from one
	// parent used to share llmMW's spare capacity, and deriving the second
	// overwrote the middleware the first had been given.
	c.clipSlices()
	base := c

	if len(c.llmMW) > 0 {
		c.client = llm.Chain(c.client, c.llmMW...)
	}

	// Applied last so it wraps whatever client and middleware the options
	// ended up with, whatever order they were given in.
	if c.metrics != nil {
		c.client = llm.Chain(c.client, c.metrics.LLMMiddleware(c.pricing, c.metricOpts...))
		c.toolMW = append(c.toolMW, c.metrics.ToolMiddleware(c.metricOpts...))
	}

	if c.tracer != nil {
		c.client = llm.Chain(c.client, trace.LLMMiddleware())
		c.toolMW = append(c.toolMW, trace.ToolMiddleware())
		tracer := c.tracer
		c.runCtx = append(c.runCtx, func(ctx context.Context) context.Context {
			return trace.WithTracer(ctx, tracer)
		})
	}

	set := c.set
	if set == nil {
		set = tool.NewSet(c.tools...)
	}

	if c.terminal != "" {
		d, ok := set.Find(c.terminal)
		if !ok {
			return nil, fmt.Errorf("wombat: terminal tool %q is not in the tool set", c.terminal)
		}
		if !d.Has(tool.CapTerminal) {
			return nil, fmt.Errorf("wombat: tool %q is named as terminal but does not declare tool.CapTerminal", c.terminal)
		}
	}

	a := &Agent{cfg: c, base: base, system: renderSystem(c), set: set, log: c.logger}
	a.checkWiring()

	a.disp = c.dispatcher
	if a.disp == nil {
		a.disp = tool.NewDispatcher(a.defaultToolChain(), tool.WithParallelism(c.parallel))
	}
	return a, nil
}

// checkWiring asks a tool set that needs per-run state whether a run of this
// agent will have it, by building the context a run would get — the agent's
// own WithRunContext decorators over an empty one — and handing it to the set's
// tool.RunChecker.
//
// It warns rather than fails. An agent used only as a sub-agent may rightly
// rely on inheriting the state from its parent's run, and New cannot tell such
// an agent from one that will be run on its own and break. The warning is for
// the second; the first can ignore it, and nothing that mattered is lost —
// the set refuses loudly at run time either way.
func (a *Agent) checkWiring() {
	rc, ok := a.set.(tool.RunChecker)
	if !ok {
		return
	}
	ctx := context.Background()
	for _, dec := range a.cfg.runCtx {
		ctx = dec(ctx)
	}
	if err := rc.CheckRun(ctx); err != nil {
		a.log.Warn("wombat: the tool set needs per-run state the agent does not install; "+
			"fine for a sub-agent that inherits its parent's, a mistake for an agent run on its own",
			slog.String("agent", a.cfg.name), slog.Any("err", err))
	}
}

// toolFallback resolves the per-call timeout applied to tools with no Timeout
// of their own. A caller that explicitly asked for 0 gets 0.
func (a *Agent) toolFallback() time.Duration {
	if a.cfg.toolFallbackSet {
		return a.cfg.toolFallbackTimeout
	}
	return DefaultToolTimeoutFallback
}

// defaultToolChain assembles dispatch middleware.
//
// Order is semantic, and each layer's position is a decision:
//   - recovery sits innermost, immediately around tool.Direct, so that retry,
//     the breaker and dedup all see a panic as the failure it is;
//   - retry sits inside the breaker so the breaker counts logical failures,
//     not attempts;
//   - the ToolCalls count sits inside every gate — inherited ones included —
//     so only a call that is about to run is counted;
//   - the journal's write of a call as running sits inside that count, the
//     last thing before the call takes effect, and outside retry, the
//     timeout and truncation, so what it records is one logical call and
//     the result the model will read (see [Journal]);
//   - inherited middleware — a parent run's permission gate, carried into this
//     run by the delegate call it allowed — sits just inside the caller's own
//     middleware, so this agent's own gate rules first (see
//     [tool.WithInherited]);
//   - dedup and the budget's loop detection sit outside the caller's
//     middleware, so they see every call the model asked for, refused ones
//     included: asking again for a call a policy just refused is being stuck
//     too, and the model is the one who has to be told;
//   - observation is outermost so one logical call produces exactly one
//     Start/Done pair, with retries already collapsed.
//
// Unlike a stack of effect handlers, position here cannot cause an inner layer
// to be shadowed — a decorator that forgets to call next simply does not, and
// that is visible in its own body.
func (a *Agent) defaultToolChain() tool.Handler {
	mw := []tool.Middleware{
		tool.WithRecovery,
		tool.WithValidation,
		tool.WithTimeout(a.toolFallback()),
		tool.WithRetry(tool.RetryPolicy{MaxAttempts: DefaultToolRetryAttempts}),
		tool.WithCircuitBreaker(DefaultBreakerThreshold, DefaultBreakerCooldown),
		tool.WithTruncation(DefaultToolOutputLimit, a.cfg.untruncated...),
		journalCalls,
		budgetToolCalls,
		tool.WithInherited,
	}
	mw = append(mw, a.cfg.toolMW...)
	mw = append(mw,
		tool.WithDedupRepeats(DefaultDedupThreshold),
		budgetAttempts,
		tool.WithLogging(a.log),
		tool.WithObserver(observeTool),
	)
	return tool.Chain(tool.Direct, mw...)
}

// With derives a new agent with opts applied on top of this one's
// configuration. The receiver is unchanged.
//
//	fast, err := a.With(core.WithModel("claude-haiku-4-5"), core.WithMaxIters(8))
//
// The derived agent is built from the options this one was given, not from
// what New made of them, so the observation that [WithMetrics] and
// [WithTracing] install is installed once, and an opts entry that replaces the
// client or the tracer is wrapped exactly like the original was.
func (a *Agent) With(opts ...Option) (*Agent, error) {
	base := a.base // every slice in it is clipped (see New), so opts append to copies

	all := make([]Option, 0, len(opts)+1)
	all = append(all, func(c *config) { *c = base })
	all = append(all, opts...)
	return New(all...)
}

// Name reports the agent's telemetry tag.
func (a *Agent) Name() string { return a.cfg.name }

// System returns the rendered system prompt.
func (a *Agent) System() string { return a.system }

// Tools returns the surface visible in ctx. Pass the context of a live run to
// see its gating; context.Background gives the ungated set.
func (a *Agent) Tools(ctx context.Context) []tool.Def { return a.set.Visible(ctx) }

// String renders a one-line summary for logs.
func (a *Agent) String() string {
	term := "free-text"
	if a.cfg.terminal != "" {
		term = "tool:" + a.cfg.terminal
	}
	return fmt.Sprintf("agent(%s model=%s tools=%d iters=%d strategy=%s terminal=%s)",
		a.cfg.name, a.cfg.model, len(a.set.Visible(context.Background())), a.cfg.maxIters, a.cfg.strategy, term)
}

// renderSystem concatenates the base prompt, contributed blocks and env blocks
// in registration order. Order is fixed so the rendered prefix is stable
// across calls, which is what makes prompt caching work.
func renderSystem(c config) string {
	var b strings.Builder
	b.WriteString(c.systemPrompt)
	write := func(blocks []block) {
		for _, bl := range blocks {
			if bl.Body == "" {
				continue
			}
			if b.Len() > 0 {
				b.WriteString("\n\n")
			}
			fmt.Fprintf(&b, "<%s>\n%s\n</%s>", bl.Name, bl.Body, bl.Name)
		}
	}
	write(c.systemBlocks)
	write(c.envBlocks)
	return b.String()
}

// observeTool turns dispatch observations into stream events.
//
// It reads the sink from ctx, so this single function — built once, when the
// agent is constructed — serves every run.
func observeTool(ctx context.Context, o tool.Observation) {
	switch o.Phase {
	case tool.PhaseStart:
		Emit(ctx, ToolStart{
			UseID:    o.Use.ID,
			Name:     o.Def.Name,
			Category: o.Def.Category,
			Input:    o.Use.Input,
		})
	case tool.PhaseDone:
		ev := ToolDone{
			UseID:  o.Use.ID,
			Name:   o.Def.Name,
			OK:     o.Err == nil,
			Output: o.Output,
			Millis: millis(o.Dur),
		}
		if o.Err != nil {
			ev.Error, ev.Err = o.Err.Error(), o.Err
		}
		Emit(ctx, ev)
	}
}

// clipSlices clips every slice in c, so that appending to any of them copies.
// TestClipSlicesCoversEveryField fails when config gains a slice field this
// does not clip: an unclipped one is shared by every agent derived from it.
func (c *config) clipSlices() {
	c.tools = slices.Clip(c.tools)
	c.toolMW = slices.Clip(c.toolMW)
	c.systemBlocks = slices.Clip(c.systemBlocks)
	c.envBlocks = slices.Clip(c.envBlocks)
	c.llmMW = slices.Clip(c.llmMW)
	c.runCtx = slices.Clip(c.runCtx)
	c.metricOpts = slices.Clip(c.metricOpts)
	c.untruncated = slices.Clip(c.untruncated)
}
