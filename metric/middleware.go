package metric

import (
	"context"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// Instrument names. Prometheus naming: a counter ends in _total, a duration is
// in _seconds, and the unit is the base SI one — never milliseconds, because
// every dashboard, alert expression and recording rule in the ecosystem
// assumes seconds and nobody writes the divide.
const (
	NameLLMCalls    = "wombat_llm_calls_total"
	NameLLMDuration = "wombat_llm_duration_seconds"
	NameLLMTokens   = "wombat_llm_tokens_total"
	NameLLMCost     = "wombat_llm_cost_usd_total"

	// NameLLMUnpriced counts calls whose cost could not be computed, by model.
	//
	// It exists because [NameLLMCost] cannot report ignorance: a model with no
	// entry in the pricing table adds 0, which on a dashboard is
	// indistinguishable from a cheap model and reads as good news. This is the
	// series that says the cost number is missing rather than small — a number
	// you cannot see is missing is worse than no number at all.
	NameLLMUnpriced = "wombat_llm_unpriced_calls_total"

	NameToolCalls    = "wombat_tool_calls_total"
	NameToolDuration = "wombat_tool_duration_seconds"

	NameRuns        = "wombat_runs_total"
	NameRunDuration = "wombat_run_duration_seconds"
	NameRunsActive  = "wombat_runs_active"
	NameIterations  = "wombat_iterations_total"
)

// Token kinds, the values of the "kind" label on [NameLLMTokens].
const (
	KindInput      = "input"
	KindOutput     = "output"
	KindCacheRead  = "cache_read"
	KindCacheWrite = "cache_write"
)

// DefaultLLMBuckets are latency buckets for a model call, in SECONDS.
//
// The unit is the whole point. A model call is not an HTTP request: the fast
// path is a sub-second haiku classification, the ordinary path is a few
// seconds, and a long completion with extended thinking and a large context
// runs for minutes. Millisecond buckets — the reflex from web instrumentation
// — would put every single call in the top bucket and tell you nothing.
//
// The boundaries are roughly geometric to 8s, where most of the distribution
// lives and resolution is worth paying for, then coarse. 300s is there because
// five minutes is the conventional client timeout, so "how many calls are
// hitting the timeout" is one query against one bucket rather than an
// interpolation; 600s catches the pathological tail that a timeout did not
// stop. Thirteen boundaries plus +Inf is fourteen series per label set, which
// against {model,purpose} stays comfortably inside [MaxSeries].
var DefaultLLMBuckets = []float64{
	0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300, 600,
}

// DefaultToolBuckets are latency buckets for a tool dispatch, in SECONDS.
//
// Much finer at the bottom than [DefaultLLMBuckets] because the distribution
// is genuinely different: a file read or a calculator is tens of microseconds
// to a millisecond, a grep across a repository is tens of milliseconds, and a
// shell command or a network fetch is seconds. One histogram has to cover four
// orders of magnitude, so the boundaries are geometric across the whole range.
//
// It starts at 1ms rather than lower because below a millisecond the number is
// not actionable — nobody optimises a tool that already returns instantly —
// and it ends at 30s because that is the wall clock a tool.Def.Timeout is
// usually set to, making the last finite bucket the "about to be killed" one.
var DefaultToolBuckets = []float64{
	0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30,
}

// runBuckets cover a whole agent run: seconds for a one-shot answer, minutes
// for real work, an hour for a long autonomous session. Unexported because a
// caller who wants different run buckets can register the histogram themselves
// against the same registry.
var runBuckets = []float64{1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600, 7200}

// Metrics is the harness's concrete instrument set: the middlewares that fill
// it in and the recorders the root package calls.
//
// Safe for concurrent use — everything it owns is an instrument, and
// instruments are.
type Metrics struct {
	llmCalls    *Counter
	llmDur      *Histogram
	llmTokens   *Counter
	llmCost     *Counter
	llmUnpriced *Counter

	toolCalls *Counter
	toolDur   *Histogram

	runs       *Counter
	runDur     *Histogram
	runsActive *Gauge
	iterations *Counter
}

// New registers the harness instrument set on r and returns handles to it.
//
// Calling it twice with the same registry is fine and returns an equivalent
// set pointing at the same instruments — see [Registry.Counter] for why that
// has to be true.
//
// Panics on a nil registry: there is no default registry to fall back to, and
// a Metrics whose writes go nowhere is a bug that would otherwise only surface
// as an empty dashboard weeks later.
func New(r *Registry) *Metrics {
	if r == nil {
		panic("metric: New requires a non-nil Registry")
	}
	m := &Metrics{
		llmCalls: r.Counter(NameLLMCalls,
			"Model calls completed, by model, purpose and outcome."),
		llmDur: r.Histogram(NameLLMDuration,
			"Model call latency in seconds, by model and purpose.", DefaultLLMBuckets),
		llmTokens: r.Counter(NameLLMTokens,
			"Tokens consumed, by model and kind (input, output, cache_read, cache_write)."),
		llmCost: r.Counter(NameLLMCost,
			"Estimated model spend in USD, by model."),
		llmUnpriced: r.Counter(NameLLMUnpriced,
			"Model calls whose cost could not be computed because the pricing has no rate for the model, by model."),

		toolCalls: r.Counter(NameToolCalls,
			"Tool dispatches completed, by tool and outcome."),
		toolDur: r.Histogram(NameToolDuration,
			"Tool dispatch latency in seconds, by tool.", DefaultToolBuckets),

		runs: r.Counter(NameRuns,
			"Agent runs finished, by outcome."),
		runDur: r.Histogram(NameRunDuration,
			"Agent run duration in seconds, by outcome.", runBuckets),
		runsActive: r.Gauge(NameRunsActive,
			"Agent runs currently in flight."),
		iterations: r.Counter(NameIterations,
			"Agent loop iterations, by agent."),
	}

	// Materialise the zero. A gauge that only appears once the first run
	// starts makes "no runs" indistinguishable from "exporter is down", and
	// this is the one series in the set whose label-free zero is meaningful
	// before anything has happened.
	m.runsActive.Set(0)
	return m
}

// ===== options =====

type config struct {
	class func(error) string
}

// Option configures a middleware.
type Option func(*config)

// WithErrorClass replaces the function that turns an error into the value of
// the "outcome" label.
//
// This is the seam through which a host adds vocabulary this package must not
// have. See the package doc: classifying a
// [github.com/hoobean1996/wombat-go/permission.ErrDenied] as "denied" rather
// than "error" requires knowing about the permission package, and metric
// cannot import it without becoming un-importable from there.
//
// f must be safe for concurrent use and should return a small, bounded set of
// words. Returning something derived from the error's text — err.Error(), a
// file path, an id — is the exact failure [MaxSeries] exists to contain, and
// containing it costs you the breakdown you were trying to get.
//
// A nil f, or an f that returns "", falls back to the default: "ok" for a nil
// error and "error" for anything else.
func WithErrorClass(f func(error) string) Option {
	return func(c *config) {
		if f != nil {
			c.class = f
		}
	}
}

// defaultErrorClass knows two words, because those are the only two it can
// justify without importing something.
func defaultErrorClass(err error) string {
	if err == nil {
		return "ok"
	}
	return "error"
}

func newConfig(opts []Option) config {
	c := config{class: defaultErrorClass}
	for _, o := range opts {
		if o != nil {
			o(&c)
		}
	}
	// Wrap rather than trust: a caller's classifier returning "" would emit
	// outcome="", which Prometheus treats as the label being absent and which
	// therefore silently merges into a series that has no outcome at all.
	inner := c.class
	c.class = func(err error) string {
		if s := inner(err); s != "" {
			return s
		}
		return defaultErrorClass(err)
	}
	return c
}

// ===== middlewares =====

// LLMMiddleware records one call, its latency, its tokens and its cost.
//
// Install it outermost, so a call that was retried three times counts once —
// the same "one logical call" boundary the trace package draws:
//
//	llm.Chain(client, llm.WithRetry(p), m.LLMMiddleware(llm.DefaultPricing))
//
// The model label is the RESOLVED model, [llm.Response.Model], falling back to
// the requested one when the response does not say and to "default" when the
// request did not either. That is deliberate: a gateway or llm.WithModelRouting
// can answer with something other than what was asked for, and the bill —
// which is what this histogram sits next to — follows what answered.
//
// A nil p prices everything at zero, so wombat_llm_cost_usd_total still exists
// and reads 0 rather than being absent. All four token kinds are recorded on
// every successful call, including the zeros, because a series that appears
// mid-day the first time a cache is written breaks every rate() over it.
//
// That zero is not free, though, and the difference is what
// [NameLLMUnpriced] is for: every call p cannot price — see [llm.Priced] —
// increments wombat_llm_unpriced_calls_total{model}, so a cost of 0 can be
// read as "cheap" or "unknown" rather than being ambiguous. A nil p, or
// [llm.FreePricing], knows no model at all, so under either every call counts
// as unpriced. That is the truth: nobody supplied a rate.
func (m *Metrics) LLMMiddleware(p llm.Pricing, opts ...Option) llm.Middleware {
	if p == nil {
		p = llm.FreePricing
	}
	cfg := newConfig(opts)

	return func(next llm.Client) llm.Client {
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			purpose := string(req.Purpose)
			if purpose == "" {
				purpose = string(llm.PurposeOther)
			}

			start := time.Now()
			resp, err := next.Complete(ctx, req)
			elapsed := time.Since(start).Seconds()

			model := resp.Model
			if model == "" {
				model = req.Model
			}
			if model == "" {
				// llm.Request documents "" as "inherit the client default", and
				// an empty label value is indistinguishable from an absent one.
				model = "default"
			}
			modelLabel := Label{Key: "model", Value: model}
			purposeLabel := Label{Key: "purpose", Value: purpose}

			m.llmCalls.Inc(modelLabel, purposeLabel, Label{Key: "outcome", Value: cfg.class(err)})
			m.llmDur.Observe(elapsed, modelLabel, purposeLabel)
			if err != nil {
				// No usage to account for, and no cost: a failed call that
				// nonetheless burned tokens is the provider's problem to report
				// and not something this response carries.
				return resp, err
			}

			u := resp.Usage
			m.llmTokens.Add(float64(u.InputTokens), modelLabel, Label{Key: "kind", Value: KindInput})
			m.llmTokens.Add(float64(u.OutputTokens), modelLabel, Label{Key: "kind", Value: KindOutput})
			m.llmTokens.Add(float64(u.CacheReadTokens), modelLabel, Label{Key: "kind", Value: KindCacheRead})
			m.llmTokens.Add(float64(u.CacheWriteTokens), modelLabel, Label{Key: "kind", Value: KindCacheWrite})
			m.llmCost.Add(p.CostUSD(model, u), modelLabel)

			// The zero is recorded for a priced model too, so that
			// wombat_llm_unpriced_calls_total exists for every model in the
			// scrape from the first call. A series that springs into existence
			// the day someone points the harness at a new gateway is a series
			// no alert was written against.
			unpriced := 0.0
			if !llm.Priced(p, model) {
				unpriced = 1
			}
			m.llmUnpriced.Add(unpriced, modelLabel)
			return resp, nil
		})
	}
}

// ToolMiddleware records one dispatch and its latency.
//
//	wombat.WithToolMiddleware(m.ToolMiddleware(metric.WithErrorClass(classify)))
//
// That position puts it outside retry, the circuit breaker and dedup, so a
// call attempted three times is one observation.
//
// The tool name is the label because it is bounded by the tool set. Nothing
// derived from the arguments or the output goes anywhere near a label — see
// the package doc on cardinality.
func (m *Metrics) ToolMiddleware(opts ...Option) tool.Middleware {
	cfg := newConfig(opts)

	return func(next tool.Handler) tool.Handler {
		return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
			name := d.Name
			if name == "" {
				name = "unknown"
			}
			toolLabel := Label{Key: "tool", Value: name}

			start := time.Now()
			out, err := next(ctx, d, use)
			m.toolDur.Observe(time.Since(start).Seconds(), toolLabel)
			m.toolCalls.Inc(toolLabel, Label{Key: "outcome", Value: cfg.class(err)})
			return out, err
		}
	}
}

// ===== recorders =====

// RunStarted marks a run as in flight. Pair it with [Metrics.RunFinished] from
// a defer, or wombat_runs_active leaks upward for the life of the process and
// the only fix is a restart.
func (m *Metrics) RunStarted() {
	m.runsActive.Add(1)
}

// RunFinished marks a run as done, records its outcome and its wall clock.
//
// outcome is the host's word — "success", "failed", "budget_exhausted",
// "canceled" — and must come from a bounded set for the same reason every
// other label must. An empty outcome is recorded as "unknown" rather than as
// an empty label, which Prometheus would treat as no label at all.
func (m *Metrics) RunFinished(outcome string, d time.Duration) {
	if outcome == "" {
		outcome = "unknown"
	}
	l := Label{Key: "outcome", Value: outcome}
	m.runsActive.Add(-1)
	m.runs.Inc(l)
	m.runDur.Observe(d.Seconds(), l)
}

// Iteration counts one turn of an agent's loop.
//
// agent is the agent's name, which is bounded by the configured agent set; a
// sub-agent reports under its own name, so the counter shows where the loop
// iterations actually went. An empty name is recorded as "root", the agent
// that has no name because it was never given one.
func (m *Metrics) Iteration(agent string) {
	if agent == "" {
		agent = "root"
	}
	m.iterations.Inc(Label{Key: "agent", Value: agent})
}
