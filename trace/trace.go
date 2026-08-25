// Package trace records what a run actually did — one span per model call,
// tool dispatch, iteration and sub-agent — as NDJSON, and renders it as a
// self-contained HTML report.
//
// # Why this is not OpenTelemetry
//
// OpenTelemetry is the right answer to distributed tracing and it is a
// third-party dependency, which this library does not have. So the shape here
// is deliberately OTel's — a [Span] with an id, a parent id, a trace id, a
// start, a duration, key/value attributes and an error status; a [Tracer] that
// hangs off a context; attribute keys taken from the OTel gen-ai semantic
// conventions ("gen_ai.request.model", "gen_ai.usage.input_tokens") — so that
// bridging to a real TracerProvider is something a user writes in their own
// module, where the dependency belongs.
//
// The bridge is a [Sink]. Spans arrive already finished, so it replays each one
// into an OTel tracer with explicit timestamps:
//
//	type otelSink struct{ tr oteltrace.Tracer }
//
//	func (s otelSink) Emit(sp trace.Span) {
//	    tid, _ := oteltrace.TraceIDFromHex(fmt.Sprintf("%032s", sp.TraceID))
//	    pid, _ := oteltrace.SpanIDFromHex(fmt.Sprintf("%016s", sp.ParentID))
//	    ctx := oteltrace.ContextWithSpanContext(context.Background(),
//	        oteltrace.NewSpanContext(oteltrace.SpanContextConfig{
//	            TraceID: tid, SpanID: pid, Remote: true,
//	            TraceFlags: oteltrace.FlagsSampled,
//	        }))
//	    _, span := s.tr.Start(ctx, sp.Name,
//	        oteltrace.WithTimestamp(sp.Start),
//	        oteltrace.WithSpanKind(oteltrace.SpanKindClient))
//	    for _, a := range sp.Attrs {
//	        span.SetAttributes(attribute.String(a.Key, fmt.Sprint(a.Value)))
//	    }
//	    if sp.Error != "" {
//	        span.SetStatus(codes.Error, sp.Error)
//	    }
//	    span.End(oteltrace.WithTimestamp(sp.Start.Add(sp.Duration)))
//	}
//
//	t := trace.New(otelSink{tr: provider.Tracer("wombat")})
//
// Ids are hex for exactly that reason: OTel wants a 16-hex span id and a
// 32-hex trace id, and the padding above is the whole conversion. A bridge
// that would rather let the SDK mint its own ids can ignore them.
//
// # Parenting
//
// A span's parent is whatever span is on the context, and nothing else. The
// OCaml original had to keep an explicit stack in domain-local storage plus a
// Trace.fork to hand a parent id to a new Domain; a
// goroutine that inherits a context inherits its parent span for free, so
// there is no fork here and no stack to race on.
//
// # What is never recorded
//
// No message content, no tool arguments, no tool output. Traces get shipped to
// dashboards, ticket attachments and vendor backends; transcripts must not. The
// middleware records names, counts, sizes, durations and outcomes — enough to
// answer "what was slow, what failed, what did it cost" and nothing that
// answers "what did the user say". A caller who wants more can [Active.Set] it
// deliberately, at its own risk.
//
// # Use
//
//	sink, closer, err := trace.FileSink("trace.ndjson")
//	if err != nil { return err }
//	defer closer.Close()
//	t := trace.New(sink)
//
//	a, err := wombat.New(
//	    wombat.WithClient(llm.Chain(client, trace.LLMMiddleware())),
//	    wombat.WithToolMiddleware(trace.ToolMiddleware()),
//	    wombat.WithRunContext(func(ctx context.Context) context.Context {
//	        return trace.WithTracer(ctx, t)
//	    }),
//	)
package trace

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Kind is the coarse category of a span. Kept as a small closed-ish set
// because it drives colour and filtering in the report; anything finer belongs
// in the name or an attribute.
type Kind string

// Span kinds.
const (
	KindRun       Kind = "run"
	KindIteration Kind = "iteration"
	KindLLM       Kind = "llm"
	KindTool      Kind = "tool"
	KindSubagent  Kind = "subagent"
)

// Attr is one key/value on a span. Value is any, matching OTel's any-typed
// attribute; a value JSON cannot encode is written as its fmt.Sprint form
// rather than failing the span.
type Attr struct {
	Key   string `json:"key"`
	Value any    `json:"value"`
}

// MarshalJSON implements json.Marshaler.
//
// The fallback matters more than it looks: one attribute holding a channel, a
// func or a cyclic struct would otherwise take the entire span line down with
// it, and a trace is exactly the artifact you cannot go back and re-collect.
func (a Attr) MarshalJSON() ([]byte, error) {
	v, err := json.Marshal(a.Value)
	if err != nil {
		v, err = json.Marshal(fmt.Sprint(a.Value))
		if err != nil {
			return nil, fmt.Errorf("trace: attr %q: %w", a.Key, err)
		}
	}
	k, err := json.Marshal(a.Key)
	if err != nil {
		return nil, fmt.Errorf("trace: attr key: %w", err)
	}
	out := make([]byte, 0, len(k)+len(v)+18)
	out = append(out, `{"key":`...)
	out = append(out, k...)
	out = append(out, `,"value":`...)
	out = append(out, v...)
	return append(out, '}'), nil
}

// Span is one completed unit of work.
type Span struct {
	ID, ParentID, TraceID string

	Kind     Kind
	Name     string
	Start    time.Time
	Duration time.Duration

	// Attrs is a slice and not a map because this is written to a file that
	// people diff. Go marshals map keys in sorted order and reorders nothing
	// else, so a map would silently interleave "gen_ai.*" and "wombat.*" keys
	// by name; a slice preserves the order the recorder chose, which is stable
	// across runs and therefore diffable.
	Attrs []Attr

	// Error is the failure message, empty on success. A string rather than an
	// error because a span outlives the error value — it is written to a file
	// and read back in another process.
	Error string
}

// spanJSON is the wire shape. Separate from [Span] so the duration is
// milliseconds — the unit every reader of the file and the report thinks in —
// instead of the bare nanosecond integer a time.Duration marshals to.
type spanJSON struct {
	ID       string    `json:"id"`
	ParentID string    `json:"parent_id,omitempty"`
	TraceID  string    `json:"trace_id,omitempty"`
	Kind     Kind      `json:"kind"`
	Name     string    `json:"name"`
	Start    time.Time `json:"start"`
	DurMS    float64   `json:"dur_ms"`
	Attrs    []Attr    `json:"attrs,omitempty"`
	Error    string    `json:"error,omitempty"`
}

// MarshalJSON implements json.Marshaler.
func (s Span) MarshalJSON() ([]byte, error) {
	return json.Marshal(spanJSON{
		ID:       s.ID,
		ParentID: s.ParentID,
		TraceID:  s.TraceID,
		Kind:     s.Kind,
		Name:     s.Name,
		Start:    s.Start,
		DurMS:    float64(s.Duration) / float64(time.Millisecond),
		Attrs:    s.Attrs,
		Error:    s.Error,
	})
}

// UnmarshalJSON implements json.Unmarshaler.
//
// Attribute values come back as whatever encoding/json chooses — every number
// is a float64 — because [Attr.Value] is any and the file carries no type tag.
// Readers that need the original Go type should assert on float64.
func (s *Span) UnmarshalJSON(b []byte) error {
	var raw spanJSON
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	*s = Span{
		ID:       raw.ID,
		ParentID: raw.ParentID,
		TraceID:  raw.TraceID,
		Kind:     raw.Kind,
		Name:     raw.Name,
		Start:    raw.Start,
		Duration: time.Duration(raw.DurMS * float64(time.Millisecond)),
		Attrs:    raw.Attrs,
		Error:    raw.Error,
	}
	return nil
}

// ===== Tracer =====

// Tracer mints spans and hands the finished ones to a [Sink].
//
// Immutable once built, which is what makes [Tracer.Start] safe to call
// concurrently on one tracer with no lock: the only mutable state a span needs
// is its parent, and that lives on the context.
type Tracer struct {
	sink Sink

	// ids nil means "no-op tracer", the single flag both Start and Active
	// check. See [FromContext] for why a no-op tracer exists at all.
	ids func() string
}

// noop is the tracer [FromContext] hands back when the context carries none.
// Shared and stateless — it allocates nothing and emits nothing.
var noop = &Tracer{}

// New builds a tracer writing to sink. A nil sink yields the no-op tracer, so
// "tracing off" is New(nil) rather than a branch at every call site.
func New(sink Sink) *Tracer {
	if sink == nil || sink == Discard {
		return noop
	}
	return &Tracer{sink: sink, ids: randomID}
}

// WithIDs returns a copy of t using gen for span and trace ids.
//
// It exists because random ids are the single largest source of noise when
// diffing two trace files: every line differs, and the real change hides in
// the churn. Pin them with a counter and the diff shows only what actually
// moved. gen must be safe for concurrent use — sub-agents mint ids from
// several goroutines at once.
//
// A nil gen restores the default. The default is 16 hex characters from
// crypto/rand: hex so it can be widened into an OTel id by zero-padding, 64
// bits because collisions within one trace are the only ones that matter.
func (t *Tracer) WithIDs(gen func() string) *Tracer {
	if t.ids == nil {
		return t // a no-op tracer has nothing to identify
	}
	if gen == nil {
		gen = randomID
	}
	return &Tracer{sink: t.sink, ids: gen}
}

// randomID is the default generator.
func randomID() string {
	var b [8]byte
	// Since Go 1.24 crypto/rand.Read cannot fail: it panics inside the
	// runtime if the OS entropy source is broken, which is not a condition a
	// tracer can do anything sensible about anyway.
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

type tracerKey struct{}

// WithTracer attaches t to ctx. Install it once per run — from
// wombat.WithRunContext — and every span raised under that context, on any
// goroutine, finds it.
func WithTracer(ctx context.Context, t *Tracer) context.Context {
	if t == nil {
		t = noop
	}
	return context.WithValue(ctx, tracerKey{}, t)
}

// FromContext returns the tracer on ctx, never nil.
//
// The no-op fallback is the reason instrumented code needs no nil checks and
// no "is tracing on" branch: a tool, a middleware or a test that never
// installed a tracer still calls Start and End, and nothing happens.
func FromContext(ctx context.Context) *Tracer {
	if t, ok := ctx.Value(tracerKey{}).(*Tracer); ok && t != nil {
		return t
	}
	return noop
}

// activeKey carries the enclosing span's identity, not the [Active] itself:
// the child only needs two strings, and copying them keeps a finished parent
// from being reachable — and mutable — through its children's contexts.
type activeKey struct{}

type activeSpan struct{ id, traceID string }

// Active is a span in flight. Safe for concurrent use: a span that brackets a
// fan-out is written to by every goroutine in it.
type Active struct {
	t *Tracer

	mu    sync.Mutex
	span  Span
	ended bool
}

// Start opens a span and returns a context whose children parent to it.
//
// Safe to call concurrently on one tracer. The returned context is what makes
// parenting work — pass it down, and a goroutine launched from it nests
// correctly with no further ceremony.
//
//	ctx, span := trace.FromContext(ctx).Start(ctx, trace.KindTool, def.Name)
//	defer func() { span.End(err) }()
func (t *Tracer) Start(ctx context.Context, kind Kind, name string) (context.Context, *Active) {
	if t == nil || t.ids == nil {
		// No-op: no ids minted, no context value added, no allocation beyond
		// the handle the caller is going to call End on.
		return ctx, &Active{}
	}

	id := t.ids()
	parent, _ := ctx.Value(activeKey{}).(activeSpan)
	traceID := parent.traceID
	if traceID == "" {
		traceID = t.ids()
	}

	a := &Active{
		t: t,
		span: Span{
			ID:       id,
			ParentID: parent.id,
			TraceID:  traceID,
			Kind:     kind,
			Name:     name,
			Start:    time.Now(),
		},
	}
	return context.WithValue(ctx, activeKey{}, activeSpan{id: id, traceID: traceID}), a
}

// Set records an attribute. Later Sets with the same key append rather than
// replace, so the file shows what the code actually did instead of only its
// last word.
//
// Do not put message content or tool arguments here. See the package doc.
func (a *Active) Set(key string, v any) {
	if a == nil || a.t == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.ended {
		return // an attribute set after End would never reach the sink
	}
	a.span.Attrs = append(a.span.Attrs, Attr{Key: key, Value: v})
}

// End closes the span and emits it. A non-nil err becomes [Span.Error].
//
// Call it from a defer: a span whose End is skipped by a panic or an early
// return is simply absent from the file, and a missing parent is the one thing
// that makes a trace hard to read. Idempotent — a second End is ignored, so a
// deferred End and an explicit one cannot double-count.
func (a *Active) End(err error) {
	if a == nil || a.t == nil {
		return
	}
	a.mu.Lock()
	if a.ended {
		a.mu.Unlock()
		return
	}
	a.ended = true
	a.span.Duration = time.Since(a.span.Start)
	if err != nil {
		a.span.Error = err.Error()
	}
	done := a.span
	a.mu.Unlock()

	// Emitted outside the lock: a sink may block on a file write or a network
	// hop, and holding a's lock across that would stall every Set on a sibling
	// goroutine that happens to share it.
	a.t.sink.Emit(done)
}

// Span returns a snapshot of the span so far. Useful to a caller that wants to
// log the id it just created.
func (a *Active) Span() Span {
	if a == nil || a.t == nil {
		return Span{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	s := a.span
	s.Attrs = append([]Attr(nil), a.span.Attrs...)
	return s
}
