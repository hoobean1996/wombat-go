package wombat

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/metric"
	"github.com/hoobean1996/wombat-go/tool"
	"github.com/hoobean1996/wombat-go/trace"
)

func TestNewValidation(t *testing.T) {
	cl := scripted()

	t.Run("a client is required", func(t *testing.T) {
		_, err := New(WithModel("m"))
		if err == nil {
			t.Fatal("New with no client: got nil error, want one")
		}
		if !strings.Contains(err.Error(), "WithClient") {
			t.Errorf("error = %q, want it to point at WithClient", err)
		}
	})

	t.Run("terminal tool must be in the set", func(t *testing.T) {
		_, err := New(WithClient(cl), WithLogger(quietLogger), WithTerminalTool("nope"))
		if err == nil {
			t.Fatal("got nil error, want one")
		}
		if !strings.Contains(err.Error(), "not in the tool set") {
			t.Errorf("error = %q, want it to say the tool is missing", err)
		}
	})

	t.Run("terminal tool must declare CapTerminal", func(t *testing.T) {
		bad := terminalTool("submit")
		bad.Caps = tool.CapReadOnly
		_, err := New(WithClient(cl), WithLogger(quietLogger), WithTools(bad), WithTerminalTool("submit"))
		if err == nil {
			t.Fatal("got nil error, want one")
		}
		if !strings.Contains(err.Error(), "CapTerminal") {
			t.Errorf("error = %q, want it to name tool.CapTerminal", err)
		}
	})

	t.Run("a well-formed terminal tool is accepted", func(t *testing.T) {
		_, err := New(WithClient(cl), WithLogger(quietLogger),
			WithTools(terminalTool("submit")), WithTerminalTool("submit"))
		if err != nil {
			t.Fatalf("got error %v, want nil", err)
		}
	})

	t.Run("a nil strategy falls back to Flat", func(t *testing.T) {
		a, err := New(WithClient(cl), WithLogger(quietLogger), WithStrategy(nil))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := a.cfg.strategy.String(); got != "flat" {
			t.Errorf("strategy = %q, want %q", got, "flat")
		}
	})
}

func TestAgentDefaults(t *testing.T) {
	a := newAgent(t, scripted(), nil)

	if got, want := a.cfg.maxIters, DefaultMaxIters; got != want {
		t.Errorf("maxIters = %d, want %d", got, want)
	}
	if got, want := a.cfg.maxTokens, DefaultMaxTokens; got != want {
		t.Errorf("maxTokens = %d, want %d", got, want)
	}
	if got, want := a.cfg.forceLastN, DefaultForceTerminalLastN; got != want {
		t.Errorf("forceLastN = %d, want %d", got, want)
	}
	if got, want := a.toolFallback(), DefaultToolTimeoutFallback; got != want {
		t.Errorf("toolFallback = %v, want %v", got, want)
	}
	if got, want := a.Name(), "agent"; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
}

// WithToolTimeoutFallback(0) must genuinely mean "no cap", which is the whole
// reason the options carry a "was it set" flag instead of relying on the zero
// value.
func TestToolTimeoutFallbackZeroMeansOff(t *testing.T) {
	a := newAgent(t, scripted(), nil, WithToolTimeoutFallback(0))
	if got := a.toolFallback(); got != 0 {
		t.Errorf("toolFallback = %v, want 0 (explicitly disabled)", got)
	}

	b := newAgent(t, scripted(), nil, WithToolTimeoutFallback(5*time.Second))
	if got, want := b.toolFallback(), 5*time.Second; got != want {
		t.Errorf("toolFallback = %v, want %v", got, want)
	}
}

func TestRenderSystemOrderAndStability(t *testing.T) {
	a := newAgent(t, scripted(), nil,
		WithSystemPrompt("BASE"),
		WithSystemBlock("first", "one"),
		WithSystemBlock("second", "two"),
		WithEnvBlock("env_a", "A"),
		WithEnvBlock("env_b", "B"),
	)

	want := "BASE\n\n" +
		"<first>\none\n</first>\n\n" +
		"<second>\ntwo\n</second>\n\n" +
		"<env_a>\nA\n</env_a>\n\n" +
		"<env_b>\nB\n</env_b>"
	if got := a.System(); got != want {
		t.Errorf("System():\ngot  %q\nwant %q", got, want)
	}

	// Rendered once and reused: the prompt-cache prefix is byte-stable only
	// because there is no code path that rebuilds it.
	if a.System() != a.System() {
		t.Error("System() is not stable across calls")
	}
	if a.System() != renderSystem(a.cfg) {
		t.Error("System() drifted from renderSystem(cfg)")
	}

	t.Run("empty blocks are skipped", func(t *testing.T) {
		b := newAgent(t, scripted(), nil,
			WithSystemPrompt("BASE"),
			WithSystemBlock("nothing", ""),
			WithSystemBlock("something", "x"),
		)
		if strings.Contains(b.System(), "nothing") {
			t.Errorf("System() = %q, want the empty block omitted", b.System())
		}
		if !strings.Contains(b.System(), "<something>") {
			t.Errorf("System() = %q, want the non-empty block present", b.System())
		}
	})

	t.Run("env blocks always come after system blocks", func(t *testing.T) {
		b := newAgent(t, scripted(), nil,
			WithEnvBlock("env", "E"),
			WithSystemBlock("sys", "S"),
		)
		si := strings.Index(b.System(), "<sys>")
		ei := strings.Index(b.System(), "<env>")
		if si < 0 || ei < 0 {
			t.Fatalf("System() = %q, want both blocks", b.System())
		}
		if si > ei {
			t.Errorf("System() = %q, want <sys> before <env> regardless of option order", b.System())
		}
	})
}

func TestAgentWithDerivesWithoutMutating(t *testing.T) {
	base := newAgent(t, scripted(), []tool.Def{echoTool("a", "A")},
		WithName("base"),
		WithModel("model-1"),
		WithMaxIters(10),
		WithSystemBlock("shared", "S"),
	)

	derived, err := base.With(
		WithName("derived"),
		WithModel("model-2"),
		WithMaxIters(3),
		WithSystemBlock("extra", "E"),
	)
	if err != nil {
		t.Fatalf("With: %v", err)
	}

	if got, want := derived.Name(), "derived"; got != want {
		t.Errorf("derived name = %q, want %q", got, want)
	}
	if got, want := derived.cfg.model, "model-2"; got != want {
		t.Errorf("derived model = %q, want %q", got, want)
	}
	if got, want := derived.cfg.maxIters, 3; got != want {
		t.Errorf("derived maxIters = %d, want %d", got, want)
	}

	// The receiver is unchanged in every dimension.
	if got, want := base.Name(), "base"; got != want {
		t.Errorf("base name = %q, want %q — With mutated the receiver", got, want)
	}
	if got, want := base.cfg.model, "model-1"; got != want {
		t.Errorf("base model = %q, want %q", got, want)
	}
	if got, want := base.cfg.maxIters, 10; got != want {
		t.Errorf("base maxIters = %d, want %d", got, want)
	}
	if strings.Contains(base.System(), "extra") {
		t.Errorf("base System() = %q, want the derived block absent", base.System())
	}

	// Inherited state carries over, and the derived agent's own additions go
	// after it.
	if !strings.Contains(derived.System(), "<shared>") {
		t.Errorf("derived System() = %q, want the inherited block", derived.System())
	}
	if !strings.Contains(derived.System(), "<extra>") {
		t.Errorf("derived System() = %q, want the added block", derived.System())
	}
	if got := toolDefNames(derived.Tools(context.Background())); !contains(got, "a") {
		t.Errorf("derived tools = %v, want the inherited tool", got)
	}

	// Appending to a derived agent's block list must not reach back into the
	// parent's slice, which is what the explicit copies in With are for.
	d2, err := base.With(WithSystemBlock("second-branch", "X"))
	if err != nil {
		t.Fatalf("With: %v", err)
	}
	if strings.Contains(derived.System(), "second-branch") {
		t.Errorf("sibling branches share a block slice: derived = %q", derived.System())
	}
	if strings.Contains(d2.System(), "extra") {
		t.Errorf("sibling branches share a block slice: d2 = %q", d2.System())
	}
}

// TestAgentWithInstallsObservationOnce is a regression test.
//
// New layers the metrics and tracing middleware onto the client, the tool
// chain and the run context. With used to hand New the configuration New had
// already layered, so a derived agent got every layer twice: one model call
// was counted as two in wombat_llm_calls_total, its tokens twice, every tool
// call twice, and every span recorded twice. Nothing failed — the dashboard
// simply showed double the traffic for every agent built with With.
func TestAgentWithInstallsObservationOnce(t *testing.T) {
	type counts struct{ leaf, llmCalls, tokens, toolCalls, iterations, llmSpans, toolSpans int }

	run := func(t *testing.T, derive func(a *Agent, leaf llm.Client) (*Agent, error)) counts {
		t.Helper()
		var leafCalls atomic.Int64
		sc := scripted(
			llm.Response{Content: toolTurn("t1", "echo", `{}`).Content, StopReason: llm.StopToolUse,
				Usage: llm.Usage{InputTokens: 10, OutputTokens: 5}},
			llm.Response{Content: textTurn("done").Content, StopReason: llm.StopEndTurn,
				Usage: llm.Usage{InputTokens: 10, OutputTokens: 5}},
		)
		leaf := llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			leafCalls.Add(1)
			return sc.Complete(ctx, req)
		})

		reg := metric.NewRegistry()
		sink := &kindCounter{}
		a := newAgent(t, leaf, []tool.Def{echoTool("echo", "ok")},
			WithMetrics(metric.New(reg)), WithTracing(trace.New(sink)))
		d, err := derive(a, leaf)
		if err != nil {
			t.Fatalf("With: %v", err)
		}
		if _, err := d.Run(context.Background(), Ask("go")); err != nil {
			t.Fatalf("Run: %v", err)
		}
		return counts{
			leaf:       int(leafCalls.Load()),
			llmCalls:   int(metricTotal(reg, metric.NameLLMCalls)),
			tokens:     int(metricTotal(reg, metric.NameLLMTokens)),
			toolCalls:  int(metricTotal(reg, metric.NameToolCalls)),
			iterations: int(metricTotal(reg, metric.NameIterations)),
			llmSpans:   sink.count(trace.KindLLM),
			toolSpans:  sink.count(trace.KindTool),
		}
	}

	// Two model calls of 15 tokens each, one tool call, two iterations.
	want := counts{leaf: 2, llmCalls: 2, tokens: 30, toolCalls: 1, iterations: 2, llmSpans: 2, toolSpans: 1}

	tests := []struct {
		name   string
		derive func(a *Agent, leaf llm.Client) (*Agent, error)
	}{
		{"the agent itself", func(a *Agent, _ llm.Client) (*Agent, error) { return a, nil }},
		{"derived once", func(a *Agent, _ llm.Client) (*Agent, error) { return a.With(WithMaxIters(5)) }},
		{"derived twice", func(a *Agent, _ llm.Client) (*Agent, error) {
			d, err := a.With(WithMaxIters(5))
			if err != nil {
				return nil, err
			}
			return d.With(WithName("grandchild"))
		}},
		// A replacement client must be wrapped exactly like the original was,
		// not left bare and not wrapped around the old client's layers.
		{"derived with a new client", func(a *Agent, leaf llm.Client) (*Agent, error) {
			return a.With(WithClient(leaf))
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := run(t, tc.derive); got != want {
				t.Errorf("observed %+v, want %+v — the leaf is the ground truth, and anything above it "+
					"means a layer was installed more than once", got, want)
			}
		})
	}
}

// kindCounter is a trace sink that counts spans by kind.
type kindCounter struct {
	mu sync.Mutex
	n  map[trace.Kind]int
}

func (k *kindCounter) Emit(sp trace.Span) {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.n == nil {
		k.n = map[trace.Kind]int{}
	}
	k.n[sp.Kind]++
}

func (k *kindCounter) count(kind trace.Kind) int {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.n[kind]
}

// metricTotal sums every series of one metric: counter values, or observation
// counts for a histogram.
func metricTotal(reg *metric.Registry, name string) float64 {
	var sum float64
	for _, s := range reg.Snapshot() {
		if s.Name != name {
			continue
		}
		if s.Type == "histogram" {
			sum += float64(s.Count)
		} else {
			sum += s.Value
		}
	}
	return sum
}

func TestToolChoiceEndgameForcing(t *testing.T) {
	tests := []struct {
		name       string
		terminal   string
		maxIters   int
		forceLastN int
		iter       int
		wantForced string
	}{
		{name: "no terminal tool never forces", terminal: "", maxIters: 6, forceLastN: 2, iter: 6},
		{name: "forceLastN 0 disables", terminal: "submit", maxIters: 6, forceLastN: 0, iter: 6},
		{name: "below the threshold", terminal: "submit", maxIters: 6, forceLastN: 2, iter: 3},
		{name: "at the threshold", terminal: "submit", maxIters: 6, forceLastN: 2, iter: 4, wantForced: "submit"},
		{name: "past the threshold", terminal: "submit", maxIters: 6, forceLastN: 2, iter: 6, wantForced: "submit"},
		// The threshold floors at 1, so a one-iteration run still forces on its
		// only iteration rather than computing a threshold of -1 or 0.
		{name: "threshold floors at 1", terminal: "submit", maxIters: 1, forceLastN: 5, iter: 1, wantForced: "submit"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := []Option{WithMaxIters(tc.maxIters), WithForceTerminalLastN(tc.forceLastN)}
			tools := []tool.Def{terminalTool("submit")}
			if tc.terminal != "" {
				opts = append(opts, WithTerminalTool(tc.terminal))
			}
			a := newAgent(t, scripted(), tools, opts...)

			choice, forced := a.toolChoice(tc.iter)
			if forced != tc.wantForced {
				t.Errorf("toolChoice(%d) forced = %q, want %q", tc.iter, forced, tc.wantForced)
			}
			if tc.wantForced == "" {
				if choice.Mode != llm.ChoiceAuto {
					t.Errorf("choice.Mode = %q, want auto", choice.Mode)
				}
				return
			}
			if choice.Mode != llm.ChoiceTool || choice.Name != tc.wantForced {
				t.Errorf("choice = %+v, want ChoiceTool(%q)", choice, tc.wantForced)
			}
		})
	}
}

// The forcing must reach the wire, not just the helper.
func TestToolChoiceForcingReachesTheRequest(t *testing.T) {
	cl := scripted(
		toolTurn("t1", "echo", `{}`),
		llm.Response{
			Content: []llm.ContentBlock{
				llm.ToolUse{ID: "t2", Name: "submit", Input: json.RawMessage(`{"summary":"done"}`)},
			},
			StopReason: llm.StopToolUse,
		},
	)
	a := newAgent(t, cl, []tool.Def{echoTool("echo", "ok"), terminalTool("submit")},
		WithTerminalTool("submit"),
		WithMaxIters(3),
		WithForceTerminalLastN(2))

	out, err := a.Run(context.Background(), Ask("go"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if _, ok := out.(Submitted); !ok {
		t.Fatalf("outcome = %T, want Submitted", out)
	}

	// maxIters=3, forceLastN=2 -> threshold 1, so even iteration 1 is forced.
	first := cl.request(t, 0)
	if first.Choice.Mode != llm.ChoiceTool || first.Choice.Name != "submit" {
		t.Errorf("iteration 1 choice = %+v, want ChoiceTool(submit)", first.Choice)
	}
	if got := cl.request(t, 1).Choice.Name; got != "submit" {
		t.Errorf("iteration 2 choice name = %q, want %q", got, "submit")
	}
}

func TestAgentToolsAndString(t *testing.T) {
	a := newAgent(t, scripted(), []tool.Def{echoTool("a", "A"), echoTool("b", "B")},
		WithName("tester"), WithModel("m"), WithMaxIters(7))

	got := toolDefNames(a.Tools(context.Background()))
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("Tools() = %v, want [a b] in declaration order", got)
	}

	s := a.String()
	for _, want := range []string{"tester", "model=m", "tools=2", "iters=7", "terminal=free-text"} {
		if !strings.Contains(s, want) {
			t.Errorf("String() = %q, want it to contain %q", s, want)
		}
	}

	b := newAgent(t, scripted(), []tool.Def{terminalTool("submit")}, WithTerminalTool("submit"))
	if !strings.Contains(b.String(), "terminal=tool:submit") {
		t.Errorf("String() = %q, want terminal=tool:submit", b.String())
	}
}

// ===== the WithTools / WithToolSet ordering hazard =====

// gatedSet is a minimal dynamic tool.Set: "secret" is hidden until the run
// activates it. It stands in for skill gating without pulling the skill
// package into a root-package test.
type gatedSet struct {
	all    []tool.Def
	hidden string
	active bool
}

func (g *gatedSet) Visible(context.Context) []tool.Def {
	out := make([]tool.Def, 0, len(g.all))
	for _, d := range g.all {
		if d.Name == g.hidden && !g.active {
			continue
		}
		out = append(out, d)
	}
	return out
}

func (g *gatedSet) Find(name string) (tool.Def, bool) {
	for _, d := range g.all {
		if d.Name == name {
			return d, true
		}
	}
	return tool.Def{}, false
}

// TestWithToolsAfterWithToolSetDiscardsGating is a regression test.
//
// WithTools resets the tool SET — it assigns c.tools and clears c.set — so a
// gated Set installed by an earlier WithToolSet is silently discarded and the
// gated tool becomes visible from turn one. Nothing errors; the agent just
// quietly offers a tool it was configured to hide.
//
// This pins the hazard rather than the fix, because the fix is an ordering
// rule the caller has to follow: finalize the tool list FIRST, gate it, then
// install the gated set once. A test that asserted only the good path would
// let the hazard stop being visible.
func TestWithToolsAfterWithToolSetDiscardsGating(t *testing.T) {
	secret := echoTool("secret", "classified")
	newGated := func() tool.Set {
		return &gatedSet{all: []tool.Def{echoTool("loader", "loads"), secret}, hidden: "secret"}
	}

	t.Run("WithToolSet alone hides the gated tool", func(t *testing.T) {
		a, err := New(WithClient(scripted()), WithLogger(quietLogger), WithToolSet(newGated()))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		got := toolDefNames(a.Tools(context.Background()))
		if contains(got, "secret") {
			t.Errorf("Tools() = %v, want the gated tool hidden", got)
		}
		if !contains(got, "loader") {
			t.Errorf("Tools() = %v, want the ungated tool visible", got)
		}
	})

	t.Run("WithTools after WithToolSet throws the gating away", func(t *testing.T) {
		a, err := New(
			WithClient(scripted()), WithLogger(quietLogger),
			WithToolSet(newGated()),
			WithTools(secret), // <- clobbers the line above
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		got := toolDefNames(a.Tools(context.Background()))
		if !contains(got, "secret") {
			t.Fatalf("Tools() = %v — the hazard has changed shape; "+
				"this test exists to keep the ordering rule visible, so update the rule's "+
				"documentation before relaxing it", got)
		}
		if contains(got, "loader") {
			t.Errorf("Tools() = %v, want the gated set's other tools gone too "+
				"(WithTools replaces the whole set, it does not merge)", got)
		}
	})

	// WithToolSet after WithTools is the winning order, because WithToolSet
	// only sets c.set and New prefers it.
	t.Run("WithToolSet after WithTools keeps the gating", func(t *testing.T) {
		a, err := New(
			WithClient(scripted()), WithLogger(quietLogger),
			WithTools(secret),
			WithToolSet(newGated()),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := toolDefNames(a.Tools(context.Background())); contains(got, "secret") {
			t.Errorf("Tools() = %v, want the gated tool hidden", got)
		}
	})
}

func TestWithEventBufferIgnoresNonPositive(t *testing.T) {
	for _, n := range []int{-1, 0} {
		a := newAgent(t, scripted(), nil, WithEventBuffer(n))
		if a.cfg.eventBuffer != 0 {
			t.Errorf("WithEventBuffer(%d): buffer = %d, want 0", n, a.cfg.eventBuffer)
		}
	}
	a := newAgent(t, scripted(), nil, WithEventBuffer(64))
	if a.cfg.eventBuffer != 64 {
		t.Errorf("WithEventBuffer(64): buffer = %d, want 64", a.cfg.eventBuffer)
	}
}

// ===== option plumbing =====

func TestOptionsReachTheRequestAndTheConfig(t *testing.T) {
	cl := scripted(textTurn("hi"))
	a := newAgent(t, cl, nil,
		WithMaxTokens(123),
		WithPurpose(llm.PurposePlanner),
		WithPricing(llm.DefaultPricing),
		WithToolParallelism(4),
	)

	if _, err := a.Run(context.Background(), Ask("x")); err != nil {
		t.Fatalf("Run: %v", err)
	}
	req := cl.request(t, 0)
	if req.MaxTokens != 123 {
		t.Errorf("MaxTokens = %d, want 123", req.MaxTokens)
	}
	if req.Purpose != llm.PurposePlanner {
		t.Errorf("Purpose = %q, want %q", req.Purpose, llm.PurposePlanner)
	}
	if a.cfg.parallel != 4 {
		t.Errorf("parallel = %d, want 4", a.cfg.parallel)
	}
	if a.cfg.pricing == nil {
		t.Error("pricing = nil, want the table that was set")
	}
}

func TestWithToolMiddlewareWrapsDispatch(t *testing.T) {
	cl := scripted(toolTurn("u1", "echo", `{}`), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{echoTool("echo", "raw")},
		WithToolMiddleware(func(next tool.Handler) tool.Handler {
			return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
				out, err := next(ctx, d, use)
				return "wrapped(" + out + ")", err
			}
		}))

	if _, err := a.Run(context.Background(), Ask("go")); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := allText(cl.request(t, 1).Messages); !strings.Contains(got, "wrapped(raw)") {
		t.Errorf("tool result = %q, want the middleware's rewrite", got)
	}
}

// WithDispatcher hands the whole pipeline over — including the observation
// that produces tool events, which is the omission the doc comment warns about.
func TestWithDispatcherReplacesThePipeline(t *testing.T) {
	cl := scripted(toolTurn("u1", "echo", `{}`), textTurn("done"))
	called := 0
	custom := tool.DispatcherFunc(func(_ context.Context, _ tool.Set, uses []llm.ToolUse) []tool.Result {
		called++
		out := make([]tool.Result, len(uses))
		for i, u := range uses {
			out[i] = tool.Result{UseID: u.ID, Name: u.Name, Output: "from the custom dispatcher"}
		}
		return out
	})
	a := newAgent(t, cl, []tool.Def{echoTool("echo", "never runs")}, WithDispatcher(custom))

	run := a.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })
	kinds, _ := drain(t, run)

	if called != 1 {
		t.Errorf("custom dispatcher called %d times, want 1", called)
	}
	if got := allText(cl.request(t, 1).Messages); !strings.Contains(got, "from the custom dispatcher") {
		t.Errorf("tool result = %q, want the custom dispatcher's output", got)
	}
	if contains(kinds, "tool_start") || contains(kinds, "tool_done") {
		t.Errorf("event kinds = %v, want no tool events: a hand-written Dispatcher has no observer", kinds)
	}
}

// A tool can reach back into the run's transcript for a previous observation.
// The resolver is per-run and travels on the context.
func TestToolLookupResolvesAPriorObservation(t *testing.T) {
	recall := tool.Def{
		Name: "recall", Description: "reads a prior result", InputSchema: json.RawMessage(objSchema),
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			l := tool.LookupFrom(ctx)
			if l == nil {
				return "", errors.New("no resolver on ctx")
			}
			return l("u1")
		},
	}
	cl := scripted(
		toolTurn("u1", "echo", `{}`),
		toolTurn("u2", "recall", `{}`),
		textTurn("done"),
	)
	a := newAgent(t, cl, []tool.Def{echoTool("echo", "the earlier answer"), recall})

	if _, err := a.Run(context.Background(), Ask("go")); err != nil {
		t.Fatalf("Run: %v", err)
	}
	got := allText(cl.request(t, 2).Messages)
	if strings.Count(got, "the earlier answer") != 2 {
		t.Errorf("transcript = %q, want the recalled value alongside the original", got)
	}
}
