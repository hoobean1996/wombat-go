package metric_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/metric"
	"wombat/pkg/core/tool"
)

// ===== fakes =====

// fakeClient answers with a canned response, optionally after a delay.
type fakeClient struct {
	resp  llm.Response
	err   error
	delay time.Duration
}

func (c fakeClient) Complete(context.Context, llm.Request) (llm.Response, error) {
	if c.delay > 0 {
		time.Sleep(c.delay)
	}
	return c.resp, c.err
}

// errDenied stands in for permission.ErrDenied. It lives here rather than
// being imported because the whole point of WithErrorClass is that metric does
// not know about the permission package.
var errDenied = errors.New("test: denied")

func toolHandler(out string, err error) tool.Handler {
	return func(context.Context, tool.Def, llm.ToolUse) (string, error) { return out, err }
}

// ===== LLM middleware =====

func TestLLMMiddlewareSuccess(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	m := metric.New(r)

	pricing := llm.Table{"test-model": {In: 3, Out: 15, CacheRead: 0.3, CacheWrite: 3.75}}
	client := llm.Chain(fakeClient{
		resp: llm.Response{
			Model: "test-model",
			Usage: llm.Usage{InputTokens: 1000, OutputTokens: 500, CacheReadTokens: 200},
		},
	}, m.LLMMiddleware(pricing))

	_, err := client.Complete(t.Context(), llm.Request{Model: "test-model", Purpose: llm.PurposePlanner})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	model := metric.Label{Key: "model", Value: "test-model"}
	purpose := metric.Label{Key: "purpose", Value: "planner"}

	if got := find(t, r, "wombat_llm_calls_total", model, purpose, metric.Label{Key: "outcome", Value: "ok"}); got.Value != 1 {
		t.Errorf("calls = %v, want 1", got.Value)
	}
	if got := find(t, r, "wombat_llm_duration_seconds", model, purpose); got.Count != 1 {
		t.Errorf("duration count = %d, want 1", got.Count)
	}

	// All four kinds, including the zeros: a series that appears mid-day the
	// first time a cache is written breaks every rate() over it.
	wantTokens := map[string]float64{"input": 1000, "output": 500, "cache_read": 200, "cache_write": 0}
	for kind, want := range wantTokens {
		got := find(t, r, "wombat_llm_tokens_total", model, metric.Label{Key: "kind", Value: kind})
		if got.Value != want {
			t.Errorf("tokens{kind=%s} = %v, want %v", kind, got.Value, want)
		}
	}

	// 1000*3/1e6 + 500*15/1e6 + 200*0.3/1e6
	wantCost := 0.003 + 0.0075 + 0.00006
	if got := find(t, r, "wombat_llm_cost_usd_total", model); !closeTo(got.Value, wantCost) {
		t.Errorf("cost = %v, want %v", got.Value, wantCost)
	}
}

func TestLLMMiddlewareModelLabel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		reqModel  string
		respModel string
		err       error
		want      string
	}{
		{
			// The bill follows what answered, not what was asked for — a
			// gateway or llm.WithModelRouting can substitute freely.
			name:     "resolved model wins",
			reqModel: "claude-sonnet-5", respModel: "claude-sonnet-5-20250101",
			want: "claude-sonnet-5-20250101",
		},
		{"falls back to the request", "claude-sonnet-5", "", nil, "claude-sonnet-5"},
		{"empty request means the client default", "", "", nil, "default"},
		{"on error the request model is used", "claude-opus-5", "", errors.New("boom"), "claude-opus-5"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			m := metric.New(r)
			client := llm.Chain(
				fakeClient{resp: llm.Response{Model: tc.respModel}, err: tc.err},
				m.LLMMiddleware(nil),
			)
			_, _ = client.Complete(t.Context(), llm.Request{Model: tc.reqModel, Purpose: llm.PurposeExecutor})

			found := false
			for _, s := range r.Snapshot() {
				if s.Name != "wombat_llm_duration_seconds" {
					continue
				}
				for _, l := range s.Labels {
					if l.Key == "model" && l.Value == tc.want {
						found = true
					}
				}
			}
			if !found {
				t.Errorf("no duration series with model=%q:\n%s", tc.want, dump(r))
			}
		})
	}
}

func TestLLMMiddlewarePurposeDefaults(t *testing.T) {
	t.Parallel()

	// An empty label value is indistinguishable from an absent label in
	// Prometheus, so an untagged call is reported as llm.PurposeOther.
	r := metric.NewRegistry()
	m := metric.New(r)
	client := llm.Chain(fakeClient{resp: llm.Response{Model: "m"}}, m.LLMMiddleware(nil))
	_, _ = client.Complete(t.Context(), llm.Request{Model: "m"})

	got := find(t, r, "wombat_llm_duration_seconds",
		metric.Label{Key: "model", Value: "m"},
		metric.Label{Key: "purpose", Value: "other"})
	if got.Count != 1 {
		t.Errorf("count = %d, want 1", got.Count)
	}
}

func TestLLMMiddlewareError(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	m := metric.New(r)
	client := llm.Chain(
		fakeClient{err: llm.ErrRateLimit},
		m.LLMMiddleware(llm.DefaultPricing),
	)

	if _, err := client.Complete(t.Context(), llm.Request{Model: "m", Purpose: llm.PurposeExecutor}); !errors.Is(err, llm.ErrRateLimit) {
		t.Fatalf("err = %v, want ErrRateLimit passed through", err)
	}

	model := metric.Label{Key: "model", Value: "m"}
	if got := find(t, r, "wombat_llm_calls_total", model,
		metric.Label{Key: "purpose", Value: "executor"},
		metric.Label{Key: "outcome", Value: "error"}); got.Value != 1 {
		t.Errorf("calls = %v, want 1", got.Value)
	}
	// A failed call still took time; that latency is exactly what a timeout
	// investigation needs.
	if got := find(t, r, "wombat_llm_duration_seconds", model,
		metric.Label{Key: "purpose", Value: "executor"}); got.Count != 1 {
		t.Errorf("duration count = %d, want 1", got.Count)
	}
	// No usage and no cost on a failure.
	missing(t, r, "wombat_llm_tokens_total")
	missing(t, r, "wombat_llm_cost_usd_total")
}

func TestLLMMiddlewareNilPricingCostsZero(t *testing.T) {
	t.Parallel()

	// The series must exist and read 0 rather than being absent, so a
	// dashboard panel does not go blank against a self-hosted model.
	r := metric.NewRegistry()
	m := metric.New(r)
	client := llm.Chain(
		fakeClient{resp: llm.Response{Model: "local-llama", Usage: llm.Usage{InputTokens: 10}}},
		m.LLMMiddleware(nil),
	)
	_, _ = client.Complete(t.Context(), llm.Request{Model: "local-llama"})

	if got := find(t, r, "wombat_llm_cost_usd_total", metric.Label{Key: "model", Value: "local-llama"}); got.Value != 0 {
		t.Errorf("cost = %v, want 0", got.Value)
	}
}

// TestLLMMiddlewareUnpricedModelIsCounted is the dashboard half of the bug
// this counter exists for: a gateway model with no rate spends real tokens and
// adds 0.00 to the cost counter, which reads as "cheap" and not as "unknown".
func TestLLMMiddlewareUnpricedModelIsCounted(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	m := metric.New(r)
	pricing := llm.Table{"claude-sonnet-5": {In: 3, Out: 15}}
	client := llm.Chain(fakeClient{
		resp: llm.Response{
			Model: "some-gateway-model",
			Usage: llm.Usage{InputTokens: 100_000, OutputTokens: 20_000},
		},
	}, m.LLMMiddleware(pricing))

	if _, err := client.Complete(t.Context(), llm.Request{Model: "some-gateway-model"}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	model := metric.Label{Key: "model", Value: "some-gateway-model"}
	if got := find(t, r, "wombat_llm_unpriced_calls_total", model); got.Value != 1 {
		t.Errorf("unpriced = %v, want 1", got.Value)
	}
	// The cost series still exists and still reads zero. That zero is not a
	// lie once the series above says it is not a measurement.
	if got := find(t, r, "wombat_llm_cost_usd_total", model); got.Value != 0 {
		t.Errorf("cost = %v, want 0", got.Value)
	}
	// The tokens are real whatever the rate table knows, which is what makes
	// them the fallback metric.
	if got := find(t, r, "wombat_llm_tokens_total", model,
		metric.Label{Key: "kind", Value: "input"}); got.Value != 100_000 {
		t.Errorf("input tokens = %v, want 100000", got.Value)
	}
}

// TestLLMMiddlewarePricedModelMaterialisesZeroUnpriced pins the other side: a
// priced model's series exists and reads 0, so an alert on the counter can be
// written before anyone points the harness at a model nobody has a rate for.
func TestLLMMiddlewarePricedModelMaterialisesZeroUnpriced(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	m := metric.New(r)
	pricing := llm.Table{"test-model": {In: 3, Out: 15}}
	client := llm.Chain(fakeClient{
		resp: llm.Response{Model: "test-model", Usage: llm.Usage{InputTokens: 1000, OutputTokens: 100}},
	}, m.LLMMiddleware(pricing))

	if _, err := client.Complete(t.Context(), llm.Request{Model: "test-model"}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	model := metric.Label{Key: "model", Value: "test-model"}
	if got := find(t, r, "wombat_llm_unpriced_calls_total", model); got.Value != 0 {
		t.Errorf("unpriced = %v, want 0", got.Value)
	}
	if got := find(t, r, "wombat_llm_cost_usd_total", model); !closeTo(got.Value, 0.0045) {
		t.Errorf("cost = %v, want 0.0045", got.Value)
	}
}

// TestLLMMiddlewareNilPricingCountsEveryCallUnpriced: no table, no rate, and
// saying so is the whole point. A nil pricing is not a claim that the run was
// free.
func TestLLMMiddlewareNilPricingCountsEveryCallUnpriced(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	m := metric.New(r)
	client := llm.Chain(
		fakeClient{resp: llm.Response{Model: "local-llama", Usage: llm.Usage{InputTokens: 10}}},
		m.LLMMiddleware(nil),
	)
	for range 3 {
		_, _ = client.Complete(t.Context(), llm.Request{Model: "local-llama"})
	}

	if got := find(t, r, "wombat_llm_unpriced_calls_total",
		metric.Label{Key: "model", Value: "local-llama"}); got.Value != 3 {
		t.Errorf("unpriced = %v, want 3", got.Value)
	}
}

// TestLLMMiddlewareFailedCallIsNotCountedUnpriced: a call that never returned
// a usage has no cost to be missing, and counting it would put every rate
// limit in a series that is supposed to mean "the pricing table is stale".
func TestLLMMiddlewareFailedCallIsNotCountedUnpriced(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	m := metric.New(r)
	client := llm.Chain(fakeClient{err: llm.ErrRateLimit}, m.LLMMiddleware(llm.DefaultPricing))
	_, _ = client.Complete(t.Context(), llm.Request{Model: "some-gateway-model"})

	missing(t, r, "wombat_llm_unpriced_calls_total")
}

func TestLLMMiddlewareDurationLandsInABucket(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	m := metric.New(r)
	client := llm.Chain(
		fakeClient{resp: llm.Response{Model: "m"}, delay: 2 * time.Millisecond},
		m.LLMMiddleware(nil),
	)
	_, _ = client.Complete(t.Context(), llm.Request{Model: "m"})

	got := find(t, r, "wombat_llm_duration_seconds",
		metric.Label{Key: "model", Value: "m"},
		metric.Label{Key: "purpose", Value: "other"})

	// Seconds, not milliseconds: a 2ms call must be a tiny number, well under
	// the smallest LLM bucket, and certainly not 2.
	if got.Sum <= 0 || got.Sum > 1 {
		t.Errorf("sum = %v seconds, want a small positive number of SECONDS", got.Sum)
	}
	if got.Buckets[0].LE != metric.DefaultLLMBuckets[0] {
		t.Errorf("first bucket = %v, want %v", got.Buckets[0].LE, metric.DefaultLLMBuckets[0])
	}
	if got.Buckets[0].Count != 1 {
		t.Errorf("a 2ms call did not land in the first bucket: %+v", got.Buckets)
	}
}

// ===== tool middleware =====

func TestToolMiddleware(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		def         tool.Def
		err         error
		opts        []metric.Option
		wantTool    string
		wantOutcome string
	}{
		{
			name: "success", def: tool.Def{Name: "Read"},
			wantTool: "Read", wantOutcome: "ok",
		},
		{
			name: "failure", def: tool.Def{Name: "Bash"}, err: errors.New("exit 1"),
			wantTool: "Bash", wantOutcome: "error",
		},
		{
			name: "unnamed tool", def: tool.Def{},
			wantTool: "unknown", wantOutcome: "ok",
		},
		{
			// The reason WithErrorClass exists: a policy refusal is not a tool
			// failure, and metric cannot know that without importing
			// permission.
			name: "denied via WithErrorClass", def: tool.Def{Name: "Bash"}, err: fmt.Errorf("gate: %w", errDenied),
			opts: []metric.Option{metric.WithErrorClass(func(err error) string {
				switch {
				case err == nil:
					return "ok"
				case errors.Is(err, errDenied):
					return "denied"
				default:
					return "error"
				}
			})},
			wantTool: "Bash", wantOutcome: "denied",
		},
		{
			name: "a classifier returning empty falls back to the default",
			def:  tool.Def{Name: "Bash"}, err: errors.New("boom"),
			opts:     []metric.Option{metric.WithErrorClass(func(error) string { return "" })},
			wantTool: "Bash", wantOutcome: "error",
		},
		{
			name:     "a nil classifier is ignored",
			def:      tool.Def{Name: "Bash"},
			opts:     []metric.Option{metric.WithErrorClass(nil)},
			wantTool: "Bash", wantOutcome: "ok",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			m := metric.New(r)

			h := tool.Chain(toolHandler("out", tc.err), m.ToolMiddleware(tc.opts...))
			out, err := h(t.Context(), tc.def, llm.ToolUse{Name: tc.def.Name})

			if out != "out" {
				t.Errorf("out = %q, want the handler's output passed through", out)
			}
			if !errors.Is(err, tc.err) {
				t.Errorf("err = %v, want %v", err, tc.err)
			}

			toolLabel := metric.Label{Key: "tool", Value: tc.wantTool}
			if got := find(t, r, "wombat_tool_calls_total", toolLabel,
				metric.Label{Key: "outcome", Value: tc.wantOutcome}); got.Value != 1 {
				t.Errorf("calls = %v, want 1", got.Value)
			}
			if got := find(t, r, "wombat_tool_duration_seconds", toolLabel); got.Count != 1 {
				t.Errorf("duration count = %d, want 1", got.Count)
			}
		})
	}
}

func TestWithErrorClassOnLLMMiddleware(t *testing.T) {
	t.Parallel()

	// The same seam on the other middleware: a rate limit is not the same
	// operational event as a bad request, and only the host decides that.
	r := metric.NewRegistry()
	m := metric.New(r)
	client := llm.Chain(fakeClient{err: llm.ErrRateLimit}, m.LLMMiddleware(nil,
		metric.WithErrorClass(func(err error) string {
			if errors.Is(err, llm.ErrRateLimit) {
				return "rate_limited"
			}
			return "error"
		})))

	_, _ = client.Complete(t.Context(), llm.Request{Model: "m", Purpose: llm.PurposeExecutor})

	if got := find(t, r, "wombat_llm_calls_total",
		metric.Label{Key: "model", Value: "m"},
		metric.Label{Key: "purpose", Value: "executor"},
		metric.Label{Key: "outcome", Value: "rate_limited"}); got.Value != 1 {
		t.Errorf("calls = %v, want 1", got.Value)
	}
}

func TestToolMiddlewareBucketsAreFinerThanLLM(t *testing.T) {
	t.Parallel()

	// A tool that returns instantly must be distinguishable from one that
	// takes a second; the LLM buckets would put both in the same place.
	if metric.DefaultToolBuckets[0] >= metric.DefaultLLMBuckets[0] {
		t.Errorf("tool buckets start at %v, LLM at %v — tool must be finer",
			metric.DefaultToolBuckets[0], metric.DefaultLLMBuckets[0])
	}
	last := func(f []float64) float64 { return f[len(f)-1] }
	if last(metric.DefaultLLMBuckets) <= last(metric.DefaultToolBuckets) {
		t.Errorf("LLM buckets end at %v, tool at %v — a model call can run for minutes",
			last(metric.DefaultLLMBuckets), last(metric.DefaultToolBuckets))
	}
	// Both are in seconds, so no boundary may look like a millisecond count.
	for _, b := range append(append([]float64{}, metric.DefaultLLMBuckets...), metric.DefaultToolBuckets...) {
		if b > 10000 {
			t.Errorf("bucket %v is too large to be seconds", b)
		}
	}
}

// ===== recorders =====

func TestRunRecorders(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	m := metric.New(r)

	// The zero is materialized at construction: "no runs" must not look like
	// "exporter is down".
	if got := find(t, r, "wombat_runs_active"); got.Value != 0 {
		t.Errorf("runs_active = %v, want 0 before anything happens", got.Value)
	}

	m.RunStarted()
	m.RunStarted()
	if got := find(t, r, "wombat_runs_active"); got.Value != 2 {
		t.Errorf("runs_active = %v, want 2", got.Value)
	}

	m.RunFinished("success", 90*time.Second)
	m.RunFinished("failed", 2*time.Second)

	if got := find(t, r, "wombat_runs_active"); got.Value != 0 {
		t.Errorf("runs_active = %v, want 0", got.Value)
	}
	for _, outcome := range []string{"success", "failed"} {
		if got := find(t, r, "wombat_runs_total", metric.Label{Key: "outcome", Value: outcome}); got.Value != 1 {
			t.Errorf("runs_total{outcome=%s} = %v, want 1", outcome, got.Value)
		}
	}
	if got := find(t, r, "wombat_run_duration_seconds", metric.Label{Key: "outcome", Value: "success"}); got.Sum != 90 {
		t.Errorf("run duration sum = %v, want 90 seconds", got.Sum)
	}
}

func TestRunFinishedNormalizesOutcome(t *testing.T) {
	t.Parallel()

	// outcome="" is treated by Prometheus as the label being absent, which
	// silently merges into a different series.
	r := metric.NewRegistry()
	metric.New(r).RunFinished("", time.Second)

	if got := find(t, r, "wombat_runs_total", metric.Label{Key: "outcome", Value: "unknown"}); got.Value != 1 {
		t.Errorf("runs_total{outcome=unknown} = %v, want 1", got.Value)
	}
}

func TestIteration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		agent string
		want  string
	}{
		{"named agent", "reviewer", "reviewer"},
		{"unnamed agent is root", "", "root"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			m := metric.New(r)
			m.Iteration(tc.agent)
			m.Iteration(tc.agent)

			if got := find(t, r, "wombat_iterations_total", metric.Label{Key: "agent", Value: tc.want}); got.Value != 2 {
				t.Errorf("iterations = %v, want 2", got.Value)
			}
		})
	}
}

// ===== the whole set =====

func TestNewRegistersTheRequiredInstruments(t *testing.T) {
	t.Parallel()

	// Every instrument, exercised once, then checked for the Prometheus
	// naming rules — a counter ends in _total, a latency in _seconds.
	r := metric.NewRegistry()
	m := metric.New(r)

	client := llm.Chain(fakeClient{
		resp: llm.Response{Model: "m", Usage: llm.Usage{InputTokens: 1, OutputTokens: 1, CacheReadTokens: 1, CacheWriteTokens: 1}},
	}, m.LLMMiddleware(llm.DefaultPricing))
	_, _ = client.Complete(t.Context(), llm.Request{Model: "m", Purpose: llm.PurposePlanner})

	h := tool.Chain(toolHandler("ok", nil), m.ToolMiddleware())
	_, _ = h(t.Context(), tool.Def{Name: "Read"}, llm.ToolUse{Name: "Read"})

	m.RunStarted()
	m.Iteration("root")
	m.RunFinished("success", time.Second)

	want := map[string]string{
		"wombat_llm_calls_total":          "counter",
		"wombat_llm_duration_seconds":     "histogram",
		"wombat_llm_tokens_total":         "counter",
		"wombat_llm_cost_usd_total":       "counter",
		"wombat_llm_unpriced_calls_total": "counter",
		"wombat_tool_calls_total":         "counter",
		"wombat_tool_duration_seconds":    "histogram",
		"wombat_runs_total":               "counter",
		"wombat_run_duration_seconds":     "histogram",
		"wombat_runs_active":              "gauge",
		"wombat_iterations_total":         "counter",
	}

	seen := map[string]string{}
	for _, s := range r.Snapshot() {
		seen[s.Name] = s.Type
		if s.Help == "" {
			t.Errorf("%s has no HELP text", s.Name)
		}
	}
	for name, typ := range want {
		got, ok := seen[name]
		if !ok {
			t.Errorf("instrument %s is missing:\n%s", name, dump(r))
			continue
		}
		if got != typ {
			t.Errorf("%s is a %s, want a %s", name, got, typ)
		}
		switch typ {
		case "counter":
			if !strings.HasSuffix(name, "_total") {
				t.Errorf("counter %s does not end in _total", name)
			}
		case "histogram":
			if !strings.HasSuffix(name, "_seconds") {
				t.Errorf("histogram %s does not end in _seconds", name)
			}
		}
	}
	for name := range seen {
		if _, ok := want[name]; !ok {
			t.Errorf("unexpected instrument %s", name)
		}
	}
}

func TestNewTwiceSharesInstruments(t *testing.T) {
	t.Parallel()

	// Building a middleware twice — two agents, an LLM chain and a tool chain
	// wired separately — is ordinary and must not double-register or panic.
	r := metric.NewRegistry()
	a := metric.New(r)
	b := metric.New(r)

	a.Iteration("root")
	b.Iteration("root")

	if got := find(t, r, "wombat_iterations_total", metric.Label{Key: "agent", Value: "root"}); got.Value != 2 {
		t.Errorf("iterations = %v, want 2 — both Metrics must share one series", got.Value)
	}
}

func TestMiddlewaresUnderConcurrency(t *testing.T) {
	t.Parallel()

	// A fan-out of sub-agents writing through the middlewares while a scrape
	// reads. Meaningful under -race.
	const (
		workers = 12
		each    = 60
	)

	r := metric.NewRegistry()
	m := metric.New(r)
	client := llm.Chain(
		fakeClient{resp: llm.Response{Model: "m", Usage: llm.Usage{InputTokens: 2, OutputTokens: 3}}},
		m.LLMMiddleware(llm.DefaultPricing),
	)
	th := tool.Chain(toolHandler("ok", nil), m.ToolMiddleware())

	stop := make(chan struct{})
	var scrape sync.WaitGroup
	scrape.Add(1)
	go func() {
		defer scrape.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = dump(r)
			}
		}
	}()

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx := context.Background()
			for i := 0; i < each; i++ {
				m.RunStarted()
				m.Iteration("root")
				_, _ = client.Complete(ctx, llm.Request{Model: "m", Purpose: llm.PurposeExecutor})
				_, _ = th(ctx, tool.Def{Name: "Read"}, llm.ToolUse{Name: "Read"})
				m.RunFinished("success", time.Millisecond)
			}
		}()
	}
	wg.Wait()
	close(stop)
	scrape.Wait()

	total := float64(workers * each)
	checks := []struct {
		name   string
		labels []metric.Label
		want   float64
	}{
		{"wombat_llm_calls_total", []metric.Label{
			{Key: "model", Value: "m"}, {Key: "purpose", Value: "executor"}, {Key: "outcome", Value: "ok"},
		}, total},
		{"wombat_tool_calls_total", []metric.Label{
			{Key: "tool", Value: "Read"}, {Key: "outcome", Value: "ok"},
		}, total},
		{"wombat_runs_total", []metric.Label{{Key: "outcome", Value: "success"}}, total},
		{"wombat_iterations_total", []metric.Label{{Key: "agent", Value: "root"}}, total},
		{"wombat_runs_active", nil, 0},
		{"wombat_llm_tokens_total", []metric.Label{
			{Key: "model", Value: "m"}, {Key: "kind", Value: "input"},
		}, total * 2},
	}
	for _, c := range checks {
		if got := find(t, r, c.name, c.labels...); !closeTo(got.Value, c.want) {
			t.Errorf("%s = %v, want %v", c.name, got.Value, c.want)
		}
	}

	for _, s := range r.Snapshot() {
		if s.Type == "histogram" && s.Count != uint64(total) {
			t.Errorf("%s count = %d, want %v", s.Name, s.Count, total)
		}
		if math.IsNaN(s.Value) {
			t.Errorf("%s is NaN", s.Name)
		}
	}
}
