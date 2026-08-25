// External test package (trace_test): everything asserted here is part of the
// contract a user of the package relies on — parenting through context, the
// no-op tracer, the file format, the two middlewares, the HTML report — so the
// tests are written against the exported surface only. Nothing here needs an
// unexported field, and keeping it out means the tests cannot accidentally
// depend on internals the package is free to change.
package trace_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
	"github.com/hoobean1996/wombat-go/trace"
)

// ===== helpers =====

// memSink collects finished spans. Safe for concurrent use, because sub-agents
// on separate goroutines share one sink.
type memSink struct {
	mu    sync.Mutex
	spans []trace.Span
}

func (m *memSink) Emit(sp trace.Span) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.spans = append(m.spans, sp)
}

func (m *memSink) all() []trace.Span {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]trace.Span(nil), m.spans...)
}

func (m *memSink) len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.spans)
}

// counterIDs is a deterministic id generator. Locked because Start is called
// from several goroutines in these tests, exactly as WithIDs documents.
func counterIDs() func() string {
	var mu sync.Mutex
	n := 0
	return func() string {
		mu.Lock()
		defer mu.Unlock()
		n++
		return fmt.Sprintf("%016x", n)
	}
}

func byID(spans []trace.Span) map[string]trace.Span {
	m := make(map[string]trace.Span, len(spans))
	for _, sp := range spans {
		m[sp.ID] = sp
	}
	return m
}

func find(t *testing.T, spans []trace.Span, name string) trace.Span {
	t.Helper()
	for _, sp := range spans {
		if sp.Name == name {
			return sp
		}
	}
	var got []string
	for _, sp := range spans {
		got = append(got, sp.Name)
	}
	t.Fatalf("no span named %q; got %v", name, got)
	return trace.Span{}
}

func attrs(sp trace.Span) map[string]any {
	m := make(map[string]any, len(sp.Attrs))
	for _, a := range sp.Attrs {
		m[a.Key] = a.Value
	}
	return m
}

func attrKeys(sp trace.Span) []string {
	out := make([]string, 0, len(sp.Attrs))
	for _, a := range sp.Attrs {
		out = append(out, a.Key)
	}
	return out
}

func wantAttr(t *testing.T, sp trace.Span, key string, want any) {
	t.Helper()
	got, ok := attrs(sp)[key]
	if !ok {
		t.Errorf("span %q has no attribute %q; got keys %v", sp.Name, key, attrKeys(sp))
		return
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("span %q attribute %q: got %v, want %v", sp.Name, key, got, want)
	}
}

// ===== parenting =====

// TestParentingIsPurelyContextual: a span's parent is whatever span is on the
// context and nothing else, which is what lets a goroutine that merely
// inherits a context nest correctly with no fork call and no shared stack.
func TestParentingIsPurelyContextual(t *testing.T) {
	sink := &memSink{}
	tr := trace.New(sink)

	rootCtx, root := tr.Start(context.Background(), trace.KindRun, "root")
	iterCtx, iter := tr.Start(rootCtx, trace.KindIteration, "iter")

	// A child started on another goroutine, from the iteration's context.
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, llmSpan := tr.Start(iterCtx, trace.KindLLM, "llm-on-goroutine")
		llmSpan.End(nil)
	}()
	<-done

	// A sibling started from the ROOT context after the iteration exists must
	// parent to the root, not to the iteration: there is no ambient stack.
	_, sibling := tr.Start(rootCtx, trace.KindTool, "sibling")
	sibling.End(nil)

	iter.End(nil)
	root.End(nil)

	spans := sink.all()
	if len(spans) != 4 {
		t.Fatalf("spans: got %d, want 4", len(spans))
	}
	index := byID(spans)

	rootSpan := find(t, spans, "root")
	iterSpan := find(t, spans, "iter")
	llmSpan := find(t, spans, "llm-on-goroutine")
	sibSpan := find(t, spans, "sibling")

	if rootSpan.ParentID != "" {
		t.Errorf("root ParentID: got %q, want empty", rootSpan.ParentID)
	}
	if iterSpan.ParentID != rootSpan.ID {
		t.Errorf("iter ParentID: got %q, want the root's id %q", iterSpan.ParentID, rootSpan.ID)
	}
	if llmSpan.ParentID != iterSpan.ID {
		t.Errorf("goroutine child ParentID: got %q, want the iteration's id %q", llmSpan.ParentID, iterSpan.ID)
	}
	if sibSpan.ParentID != rootSpan.ID {
		t.Errorf("sibling ParentID: got %q, want the root's id %q", sibSpan.ParentID, rootSpan.ID)
	}

	// One trace id for the whole tree, and no orphans.
	for _, sp := range spans {
		if sp.TraceID != rootSpan.TraceID {
			t.Errorf("span %q TraceID: got %q, want %q", sp.Name, sp.TraceID, rootSpan.TraceID)
		}
		if sp.ParentID == "" {
			continue
		}
		if _, ok := index[sp.ParentID]; !ok {
			t.Errorf("span %q is orphaned: parent %q is not in the trace", sp.Name, sp.ParentID)
		}
	}
	if rootSpan.TraceID == "" {
		t.Error("root TraceID: got empty, want a minted id")
	}
	if rootSpan.Kind != trace.KindRun || llmSpan.Kind != trace.KindLLM {
		t.Errorf("kinds: got %q and %q, want %q and %q", rootSpan.Kind, llmSpan.Kind, trace.KindRun, trace.KindLLM)
	}

	// The parent must be emitted with a duration that covers its children.
	if rootSpan.Duration <= 0 {
		t.Errorf("root Duration: got %v, want a positive duration", rootSpan.Duration)
	}
}

// TestConcurrentChildrenOfOneSpan runs the fan-out case under -race: one
// Active written to from many goroutines while its children are minted from
// one tracer.
func TestConcurrentChildrenOfOneSpan(t *testing.T) {
	sink := &memSink{}
	tr := trace.New(sink)
	ctx, parent := tr.Start(context.Background(), trace.KindRun, "fanout")

	const n = 16
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			parent.Set(fmt.Sprintf("child.%d", i), i)
			_, child := tr.Start(ctx, trace.KindTool, fmt.Sprintf("tool-%d", i))
			child.Set("i", i)
			_ = parent.Span() // snapshot while others are writing
			child.End(nil)
		}(i)
	}
	wg.Wait()
	parent.End(nil)

	spans := sink.all()
	if len(spans) != n+1 {
		t.Fatalf("spans: got %d, want %d", len(spans), n+1)
	}
	parentSpan := find(t, spans, "fanout")
	if len(parentSpan.Attrs) != n {
		t.Errorf("parent attrs: got %d, want %d", len(parentSpan.Attrs), n)
	}
	for _, sp := range spans {
		if sp.Name == "fanout" {
			continue
		}
		if sp.ParentID != parentSpan.ID {
			t.Errorf("span %q ParentID: got %q, want %q", sp.Name, sp.ParentID, parentSpan.ID)
		}
	}
}

// ===== the no-op tracer =====

// TestNoTracerIsASafeNoOp is the reason instrumented code carries no "is
// tracing on" branch: a context that never had a tracer installed still
// supports the whole Start/Set/End/Span dance.
func TestNoTracerIsASafeNoOp(t *testing.T) {
	ctx := context.Background()

	tr := trace.FromContext(ctx)
	if tr == nil {
		t.Fatal("FromContext on a bare context: got nil, want the no-op tracer")
	}

	got, span := tr.Start(ctx, trace.KindLLM, "nothing")
	if got != ctx {
		t.Error("no-op Start returned a different context; it must add no value")
	}
	span.Set("k", "v")
	span.Set("k2", func() {}) // unmarshalable, and must still be harmless
	if s := span.Span(); s.ID != "" || s.Name != "" || len(s.Attrs) != 0 {
		t.Errorf("no-op Span(): got %+v, want the zero span", s)
	}
	span.End(errors.New("boom"))
	span.End(nil)

	// A nil *Active is the shape a caller gets from an early return; it must
	// not panic either.
	var nilActive *trace.Active
	nilActive.Set("k", "v")
	nilActive.End(nil)
	if s := nilActive.Span(); s.ID != "" || s.Name != "" || len(s.Attrs) != 0 {
		t.Errorf("nil Active Span(): got %+v, want the zero span", s)
	}

	// A nil tracer and a Discard sink take the same path.
	if _, sp := trace.New(nil).Start(ctx, trace.KindRun, "x"); sp == nil {
		t.Error("New(nil).Start: got a nil Active, want a usable handle")
	}
	if _, sp := trace.New(trace.Discard).Start(ctx, trace.KindRun, "x"); sp == nil {
		t.Error("New(Discard).Start: got a nil Active, want a usable handle")
	}
	trace.Discard.Emit(trace.Span{Name: "dropped"})

	// WithTracer(nil) must install the no-op rather than a nil pointer.
	nilCtx := trace.WithTracer(ctx, nil)
	_, sp := trace.FromContext(nilCtx).Start(nilCtx, trace.KindRun, "x")
	sp.End(nil)
}

// TestDoubleEndDoesNotDoubleCount: End is documented as idempotent so that a
// deferred End and an explicit one cannot emit the same span twice — which
// would show up in a report as a phantom duplicate call and in a cost rollup
// as double spend.
func TestDoubleEndDoesNotDoubleCount(t *testing.T) {
	sink := &memSink{}
	tr := trace.New(sink)

	_, span := tr.Start(context.Background(), trace.KindLLM, "once")
	span.End(nil)
	first := sink.all()[0]

	span.End(errors.New("late failure"))
	span.Set("late", true)
	span.End(nil)

	if got := sink.len(); got != 1 {
		t.Fatalf("emitted spans after three Ends: got %d, want 1", got)
	}
	after := sink.all()[0]
	if after.Error != first.Error {
		t.Errorf("a second End rewrote the error: got %q, want %q", after.Error, first.Error)
	}
	if len(after.Attrs) != 0 {
		t.Errorf("an attribute set after End reached the sink: got %v, want none", attrKeys(after))
	}

	// Concurrent Ends must also produce exactly one span.
	_, span2 := tr.Start(context.Background(), trace.KindLLM, "racy")
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); span2.End(nil) }()
	}
	wg.Wait()
	if got := sink.len(); got != 2 {
		t.Errorf("spans after 8 concurrent Ends on one Active: got %d, want 2", got)
	}
}

func TestEndRecordsTheError(t *testing.T) {
	sink := &memSink{}
	_, span := trace.New(sink).Start(context.Background(), trace.KindTool, "boom")
	span.End(errors.New("disk on fire"))
	if got := sink.all()[0].Error; got != "disk on fire" {
		t.Errorf("Span.Error: got %q, want %q", got, "disk on fire")
	}
}

// ===== deterministic ids =====

// TestWithIDsMakesTwoRunsByteIdentical is the whole reason WithIDs exists:
// random ids make every line of a trace differ between two runs, and the real
// change hides in the churn.
func TestWithIDsMakesTwoRunsByteIdentical(t *testing.T) {
	run := func() []byte {
		sink := &memSink{}
		tr := trace.New(sink).WithIDs(counterIDs())

		ctx, root := tr.Start(context.Background(), trace.KindRun, "root")
		for i := 0; i < 3; i++ {
			ictx, iter := tr.Start(ctx, trace.KindIteration, fmt.Sprintf("iteration %d", i+1))
			_, call := tr.Start(ictx, trace.KindLLM, "executor")
			call.Set(trace.AttrInputTokens, 100+i)
			call.End(nil)
			iter.End(nil)
		}
		root.End(nil)

		// Start and Duration are wall-clock and are not what WithIDs pins.
		spans := sink.all()
		for i := range spans {
			spans[i].Start = time.Unix(0, 0).UTC()
			spans[i].Duration = 0
		}
		b, err := json.Marshal(spans)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		return b
	}

	a, b := run(), run()
	if string(a) != string(b) {
		t.Errorf("two identical runs with pinned ids differ:\n a: %s\n b: %s", a, b)
	}
	if !strings.Contains(string(a), `"id":"0000000000000001"`) {
		t.Errorf("ids do not come from the supplied generator: %s", a)
	}

	// A nil generator restores the random default rather than minting "".
	tr := trace.New(&memSink{}).WithIDs(nil)
	_, sp := tr.Start(context.Background(), trace.KindRun, "x")
	if got := sp.Span().ID; got == "" {
		t.Error("WithIDs(nil) minted an empty id, want the random default")
	}
	sp.End(nil)

	// WithIDs on the no-op tracer is still the no-op tracer.
	noop := trace.New(nil).WithIDs(counterIDs())
	_, nsp := noop.Start(context.Background(), trace.KindRun, "x")
	if s := nsp.Span(); s.ID != "" {
		t.Errorf("WithIDs on a no-op tracer produced a span: got %+v, want the zero span", s)
	}
}

func TestDefaultIDsAreDistinct(t *testing.T) {
	sink := &memSink{}
	tr := trace.New(sink)
	seen := map[string]bool{}
	for i := 0; i < 64; i++ {
		_, sp := tr.Start(context.Background(), trace.KindRun, "x")
		id := sp.Span().ID
		if len(id) != 16 {
			t.Fatalf("default id %q: got length %d, want 16 hex characters", id, len(id))
		}
		if seen[id] {
			t.Fatalf("duplicate id %q", id)
		}
		seen[id] = true
		sp.End(nil)
	}
}

// ===== attributes on the wire =====

// TestAttrsStayAnOrderedSliceOnTheWire: a map would let encoding/json sort the
// keys, interleaving "gen_ai.*" and "wombat.*" by name and making two traces
// of the same run diff differently for no reason.
func TestAttrsStayAnOrderedSliceOnTheWire(t *testing.T) {
	sink := &memSink{}
	_, span := trace.New(sink).Start(context.Background(), trace.KindLLM, "ordered")
	// Deliberately anti-alphabetical.
	span.Set("wombat.purpose", "executor")
	span.Set("gen_ai.request.model", "m")
	span.Set("aaa", 1)
	span.Set("wombat.purpose", "second value") // repeat: appends, never replaces
	span.End(nil)

	sp := sink.all()[0]
	wantKeys := []string{"wombat.purpose", "gen_ai.request.model", "aaa", "wombat.purpose"}
	if got := attrKeys(sp); fmt.Sprint(got) != fmt.Sprint(wantKeys) {
		t.Errorf("attr order in memory: got %v, want %v", got, wantKeys)
	}

	b, err := json.Marshal(sp)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	want := `"attrs":[{"key":"wombat.purpose","value":"executor"},` +
		`{"key":"gen_ai.request.model","value":"m"},` +
		`{"key":"aaa","value":1},` +
		`{"key":"wombat.purpose","value":"second value"}]`
	if !strings.Contains(string(b), want) {
		t.Errorf("attrs on the wire:\n got %s\nwant a line containing %s", b, want)
	}

	var back trace.Span
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got := attrKeys(back); fmt.Sprint(got) != fmt.Sprint(wantKeys) {
		t.Errorf("attr order after a round trip: got %v, want %v", got, wantKeys)
	}
}

// One attribute holding a channel must not take the whole span down with it —
// a trace is the one artifact you cannot go back and re-collect.
func TestAttrWithUnmarshalableValueDegradesInsteadOfFailing(t *testing.T) {
	sink := &memSink{}
	_, span := trace.New(sink).Start(context.Background(), trace.KindTool, "weird")
	span.Set("ok", 1)
	span.Set("chan", make(chan int))
	span.End(nil)

	b, err := json.Marshal(sink.all()[0])
	if err != nil {
		t.Fatalf("Marshal: got error %v, want the span to survive one bad attribute", err)
	}
	if !strings.Contains(string(b), `"key":"ok","value":1`) {
		t.Errorf("the good attribute was lost: %s", b)
	}
	if !strings.Contains(string(b), `"key":"chan"`) {
		t.Errorf("the bad attribute was dropped rather than stringified: %s", b)
	}
}

func TestSpanRoundTripsDuration(t *testing.T) {
	in := trace.Span{
		ID: "a", ParentID: "b", TraceID: "c",
		Kind: trace.KindLLM, Name: "n",
		Start:    time.Date(2026, 8, 23, 1, 2, 3, 0, time.UTC),
		Duration: 1500 * time.Millisecond,
		Error:    "e",
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(b), `"dur_ms":1500`) {
		t.Errorf("duration is not written in milliseconds: %s", b)
	}
	var out trace.Span
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if out.Duration != in.Duration {
		t.Errorf("Duration after a round trip: got %v, want %v", out.Duration, in.Duration)
	}
	if !out.Start.Equal(in.Start) {
		t.Errorf("Start after a round trip: got %v, want %v", out.Start, in.Start)
	}
}

// ===== FileSink =====

func TestFileSinkCreatesParentDirectories(t *testing.T) {
	// The common path is a per-run directory a UI has just invented.
	path := filepath.Join(t.TempDir(), "run-3", "nested", "trace.ndjson")
	sink, closer, err := trace.FileSink(path)
	if err != nil {
		t.Fatalf("FileSink(%q): got error %v, want nil", path, err)
	}
	sink.Emit(trace.Span{ID: "a", Kind: trace.KindRun, Name: "root"})
	if err := closer.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	spans, err := trace.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: got error %v, want nil", err)
	}
	if len(spans) != 1 || spans[0].ID != "a" {
		t.Errorf("read back: got %+v, want one span with id \"a\"", spans)
	}
}

func TestFileSinkSerialisesConcurrentWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.ndjson")
	sink, closer, err := trace.FileSink(path)
	if err != nil {
		t.Fatalf("FileSink: got error %v, want nil", err)
	}

	const n = 64
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sink.Emit(trace.Span{
				ID:   fmt.Sprintf("%04d", i),
				Kind: trace.KindTool,
				// A long name makes an interleaved write obvious.
				Name:  strings.Repeat(fmt.Sprintf("s%02d-", i), 40),
				Attrs: []trace.Attr{{Key: "i", Value: i}},
			})
		}(i)
	}
	wg.Wait()
	if err := closer.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	spans, err := trace.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: got error %v, want nil (a torn line means writes interleaved)", err)
	}
	if len(spans) != n {
		t.Fatalf("spans: got %d, want %d", len(spans), n)
	}
	seen := map[string]bool{}
	for _, sp := range spans {
		if seen[sp.ID] {
			t.Errorf("duplicate span id %q", sp.ID)
		}
		seen[sp.ID] = true
		if want := strings.Repeat("s"+sp.ID[2:]+"-", 40); sp.Name != want {
			t.Errorf("span %q name is corrupt: got %q", sp.ID, sp.Name)
		}
	}
}

// A stray late span from a goroutine that outlived its run is not worth
// crashing a process over, so Emit after Close drops the span instead of
// writing to a closed file.
func TestFileSinkDropsEmitAfterClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.ndjson")
	sink, closer, err := trace.FileSink(path)
	if err != nil {
		t.Fatalf("FileSink: got error %v, want nil", err)
	}
	sink.Emit(trace.Span{ID: "before", Kind: trace.KindRun, Name: "before"})
	if err := closer.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}
	if err := closer.Close(); err != nil {
		t.Errorf("second Close: got error %v, want nil (Close is idempotent)", err)
	}

	sink.Emit(trace.Span{ID: "after", Kind: trace.KindRun, Name: "after"}) // must not panic

	spans, err := trace.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: got error %v, want nil", err)
	}
	if len(spans) != 1 || spans[0].ID != "before" {
		var ids []string
		for _, sp := range spans {
			ids = append(ids, sp.ID)
		}
		t.Errorf("spans: got %v, want [before]", ids)
	}
}

func TestFileSinkAppendsToAnExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.ndjson")
	for _, id := range []string{"one", "two"} {
		sink, closer, err := trace.FileSink(path)
		if err != nil {
			t.Fatalf("FileSink: got error %v, want nil", err)
		}
		sink.Emit(trace.Span{ID: id, Kind: trace.KindRun, Name: id})
		if err := closer.Close(); err != nil {
			t.Fatalf("Close: got error %v, want nil", err)
		}
	}
	spans, err := trace.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: got error %v, want nil", err)
	}
	if len(spans) != 2 {
		t.Errorf("spans after two sessions: got %d, want 2 (truncated?)", len(spans))
	}
}

func TestMultiSink(t *testing.T) {
	a, b := &memSink{}, &memSink{}
	trace.MultiSink(a, nil, trace.Discard, b).Emit(trace.Span{ID: "x"})
	if a.len() != 1 || b.len() != 1 {
		t.Errorf("fan-out: got %d and %d spans, want 1 and 1", a.len(), b.len())
	}
	if got := trace.MultiSink(); got != trace.Discard {
		t.Errorf("MultiSink(): got %T, want Discard", got)
	}
	if got := trace.MultiSink(nil, trace.Discard); got != trace.Discard {
		t.Errorf("MultiSink with nothing useful: got %T, want Discard", got)
	}
	if got := trace.MultiSink(a); got != trace.Sink(a) {
		t.Error("MultiSink with one sink should return it unwrapped")
	}
}

// ===== ReadFile =====

func TestReadFileTornAndCorrupt(t *testing.T) {
	valid := `{"id":"a","kind":"run","name":"root","start":"2026-08-23T00:00:00Z","dur_ms":1}`

	t.Run("truncated final line is tolerated", func(t *testing.T) {
		// A process killed mid-write is precisely the run whose trace you most
		// want to look at, so the complete lines before it must still load.
		path := filepath.Join(t.TempDir(), "t.ndjson")
		body := valid + "\n" + valid + "\n" + `{"id":"c","kind":"ru`
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		spans, err := trace.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: got error %v, want nil", err)
		}
		if len(spans) != 2 {
			t.Errorf("spans: got %d, want 2", len(spans))
		}
	})

	t.Run("a complete final line without a newline is kept", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "t.ndjson")
		if err := os.WriteFile(path, []byte(valid+"\n"+valid), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		spans, err := trace.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: got error %v, want nil", err)
		}
		if len(spans) != 2 {
			t.Errorf("spans: got %d, want 2 (a valid unterminated line is not torn)", len(spans))
		}
	})

	t.Run("interior corruption errors with path and line", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "t.ndjson")
		body := valid + "\n" + "{not json\n" + valid + "\n"
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, err := trace.ReadFile(path)
		if err == nil {
			t.Fatal("got nil error, want a corruption error")
		}
		if want := path + ":2"; !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not contain %q", err, want)
		}
	})

	t.Run("blank lines are skipped", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "t.ndjson")
		if err := os.WriteFile(path, []byte("\n"+valid+"\n\n"+valid+"\n"), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		spans, err := trace.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: got error %v, want nil", err)
		}
		if len(spans) != 2 {
			t.Errorf("spans: got %d, want 2", len(spans))
		}
	})

	t.Run("missing file", func(t *testing.T) {
		if _, err := trace.ReadFile(filepath.Join(t.TempDir(), "absent")); err == nil {
			t.Error("got nil error, want one")
		}
	})

	t.Run("empty file", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "empty.ndjson")
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		spans, err := trace.ReadFile(path)
		if err != nil || len(spans) != 0 {
			t.Errorf("got (%v, %v), want (0 spans, nil)", spans, err)
		}
	})
}

// ===== middleware =====

const (
	secretPrompt   = "SYSTEM-PROMPT-SECRET-4f2a"
	secretMessage  = "USER-TRANSCRIPT-SECRET-9c1b"
	secretReply    = "ASSISTANT-TRANSCRIPT-SECRET-7d3e"
	secretToolArgs = "TOOL-ARGUMENTS-SECRET-2b8f"
	secretToolOut  = "TOOL-OUTPUT-SECRET-6e4d"
)

func TestLLMMiddlewareRecordsTheDocumentedAttributes(t *testing.T) {
	sink := &memSink{}
	tr := trace.New(sink)

	client := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		return llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: secretReply}},
			StopReason: llm.StopEndTurn,
			Model:      "resolved-model",
			Usage:      llm.Usage{InputTokens: 120, OutputTokens: 15, CacheReadTokens: 7, CacheWriteTokens: 9},
		}, nil
	})

	ctx := trace.WithTracer(context.Background(), tr)
	req := llm.Request{
		System:   secretPrompt,
		Messages: []llm.Message{llm.UserText(secretMessage), llm.UserText("second")},
		Tools:    []llm.ToolSpec{{Name: "calculator"}, {Name: "bash"}},
		Model:    "requested-model",
		Purpose:  llm.PurposePlanner,
		Choice:   llm.ForceTool("calculator"),
	}
	if _, err := llm.Chain(client, trace.LLMMiddleware()).Complete(ctx, req); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}

	spans := sink.all()
	if len(spans) != 1 {
		t.Fatalf("spans: got %d, want 1", len(spans))
	}
	sp := spans[0]
	if sp.Kind != trace.KindLLM {
		t.Errorf("Kind: got %q, want %q", sp.Kind, trace.KindLLM)
	}
	// The span is named for the purpose, because that is what you scan for.
	if sp.Name != string(llm.PurposePlanner) {
		t.Errorf("Name: got %q, want %q", sp.Name, llm.PurposePlanner)
	}
	wantAttr(t, sp, trace.AttrPurpose, string(llm.PurposePlanner))
	wantAttr(t, sp, trace.AttrRequestModel, "requested-model")
	wantAttr(t, sp, trace.AttrResponseModel, "resolved-model")
	wantAttr(t, sp, trace.AttrFinishReason, string(llm.StopEndTurn))
	wantAttr(t, sp, trace.AttrMessageCount, 2)
	wantAttr(t, sp, trace.AttrToolCount, 2)
	wantAttr(t, sp, trace.AttrForcedTool, "calculator")
	wantAttr(t, sp, trace.AttrInputTokens, 120)
	wantAttr(t, sp, trace.AttrOutputTokens, 15)
	wantAttr(t, sp, trace.AttrCacheRead, 7)
	wantAttr(t, sp, trace.AttrCacheWrite, 9)
	if sp.Error != "" {
		t.Errorf("Error: got %q, want empty", sp.Error)
	}
}

func TestLLMMiddlewareDefaultsAndErrors(t *testing.T) {
	t.Run("no purpose is named llm", func(t *testing.T) {
		sink := &memSink{}
		ctx := trace.WithTracer(context.Background(), trace.New(sink))
		client := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
			return llm.Response{StopReason: llm.StopEndTurn}, nil
		})
		if _, err := llm.Chain(client, trace.LLMMiddleware()).Complete(ctx, llm.Request{Model: "m"}); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		sp := sink.all()[0]
		if sp.Name != "llm" {
			t.Errorf("Name: got %q, want %q", sp.Name, "llm")
		}
		// The resolved model falls back to the requested one.
		wantAttr(t, sp, trace.AttrResponseModel, "m")
		if _, ok := attrs(sp)[trace.AttrCacheRead]; ok {
			t.Error("cache attributes are set for a zero cache read, want them omitted")
		}
		if _, ok := attrs(sp)[trace.AttrForcedTool]; ok {
			t.Error("forced-tool attribute is set with ChoiceAuto, want it omitted")
		}
	})

	t.Run("a failed call still emits a span", func(t *testing.T) {
		sink := &memSink{}
		ctx := trace.WithTracer(context.Background(), trace.New(sink))
		client := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
			return llm.Response{}, errors.New("upstream died")
		})
		if _, err := llm.Chain(client, trace.LLMMiddleware()).Complete(ctx, llm.Request{Model: "m"}); err == nil {
			t.Fatal("Complete: got nil error, want one")
		}
		spans := sink.all()
		if len(spans) != 1 {
			t.Fatalf("spans: got %d, want 1", len(spans))
		}
		if spans[0].Error != "upstream died" {
			t.Errorf("Span.Error: got %q, want %q", spans[0].Error, "upstream died")
		}
		if _, ok := attrs(spans[0])[trace.AttrFinishReason]; ok {
			t.Error("a failed call recorded a finish reason, want none")
		}
	})

	t.Run("with no tracer it is inert", func(t *testing.T) {
		client := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
			return llm.Response{StopReason: llm.StopEndTurn}, nil
		})
		if _, err := llm.Chain(client, trace.LLMMiddleware()).Complete(context.Background(), llm.Request{}); err != nil {
			t.Fatalf("Complete without a tracer: got error %v, want nil", err)
		}
	})
}

func TestToolMiddlewareRecordsTheDocumentedAttributes(t *testing.T) {
	sink := &memSink{}
	ctx := trace.WithTracer(context.Background(), trace.New(sink))

	def := tool.Def{
		Name:        "write_file",
		Description: "writes",
		Category:    "fs",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn:          func(context.Context, json.RawMessage) (string, error) { return secretToolOut, nil },
	}
	h := trace.ToolMiddleware()(tool.Direct)
	out, err := h(ctx, def, llm.ToolUse{
		ID:    "tu_42",
		Name:  "write_file",
		Input: json.RawMessage(`{"path":"` + secretToolArgs + `"}`),
	})
	if err != nil || out != secretToolOut {
		t.Fatalf("handler: got (%q, %v), want the tool's output and nil", out, err)
	}

	spans := sink.all()
	if len(spans) != 1 {
		t.Fatalf("spans: got %d, want 1", len(spans))
	}
	sp := spans[0]
	if sp.Kind != trace.KindTool || sp.Name != "write_file" {
		t.Errorf("span: got kind %q name %q, want %q and %q", sp.Kind, sp.Name, trace.KindTool, "write_file")
	}
	wantAttr(t, sp, trace.AttrToolName, "write_file")
	wantAttr(t, sp, trace.AttrToolCallID, "tu_42")
	wantAttr(t, sp, trace.AttrToolCategory, "fs")
	wantAttr(t, sp, trace.AttrToolOK, true)
	wantAttr(t, sp, trace.AttrOutputBytes, len(secretToolOut))

	t.Run("failure", func(t *testing.T) {
		sink := &memSink{}
		ctx := trace.WithTracer(context.Background(), trace.New(sink))
		bad := def
		bad.Category = ""
		bad.Fn = func(context.Context, json.RawMessage) (string, error) { return "", errors.New("no such path") }
		if _, err := trace.ToolMiddleware()(tool.Direct)(ctx, bad, llm.ToolUse{Name: "write_file"}); err == nil {
			t.Fatal("got nil error, want one")
		}
		sp := sink.all()[0]
		if sp.Error != "no such path" {
			t.Errorf("Span.Error: got %q, want %q", sp.Error, "no such path")
		}
		wantAttr(t, sp, trace.AttrToolOK, false)
		wantAttr(t, sp, trace.AttrOutputBytes, 0)
		if _, ok := attrs(sp)[trace.AttrToolCallID]; ok {
			t.Error("an empty tool_use id was recorded, want it omitted")
		}
		if _, ok := attrs(sp)[trace.AttrToolCategory]; ok {
			t.Error("an empty category was recorded, want it omitted")
		}
	})
}

// TestMiddlewaresRecordNoContent is the package's central privacy claim:
// "traces get shipped to dashboards, ticket attachments and vendor backends;
// transcripts must not". It asserts on the SERIALISED span, because that is
// what actually leaves the process.
func TestMiddlewaresRecordNoContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.ndjson")
	sink, closer, err := trace.FileSink(path)
	if err != nil {
		t.Fatalf("FileSink: got error %v, want nil", err)
	}
	tr := trace.New(sink)
	ctx := trace.WithTracer(context.Background(), tr)

	client := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		return llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: secretReply}},
			StopReason: llm.StopEndTurn,
			Usage:      llm.Usage{InputTokens: 10, OutputTokens: 2},
		}, nil
	})
	if _, err := llm.Chain(client, trace.LLMMiddleware()).Complete(ctx, llm.Request{
		System:   secretPrompt,
		Messages: []llm.Message{llm.UserText(secretMessage)},
		Model:    "m",
	}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}

	def := tool.Def{
		Name:        "bash",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn:          func(context.Context, json.RawMessage) (string, error) { return secretToolOut, nil },
	}
	if _, err := trace.ToolMiddleware()(tool.Direct)(ctx, def, llm.ToolUse{
		ID: "tu_1", Name: "bash", Input: json.RawMessage(`{"cmd":"` + secretToolArgs + `"}`),
	}); err != nil {
		t.Fatalf("tool: got error %v, want nil", err)
	}
	if err := closer.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	for _, secret := range []string{secretPrompt, secretMessage, secretReply, secretToolArgs, secretToolOut} {
		if strings.Contains(string(raw), secret) {
			t.Errorf("the trace leaked %q:\n%s", secret, raw)
		}
	}
	// The trace is still useful: the tool NAME and the counts are there.
	if !strings.Contains(string(raw), `"bash"`) {
		t.Errorf("the trace does not name the tool:\n%s", raw)
	}
}

// ===== HTML report =====

func TestWriteHTMLIsSelfContained(t *testing.T) {
	spans := []trace.Span{
		{ID: "1", Kind: trace.KindRun, Name: "root", Start: time.Now(), Duration: time.Second},
		{ID: "2", ParentID: "1", Kind: trace.KindLLM, Name: "planner", Start: time.Now(), Duration: 50 * time.Millisecond,
			Attrs: []trace.Attr{{Key: trace.AttrInputTokens, Value: 100}}},
		{ID: "3", ParentID: "1", Kind: trace.KindTool, Name: "bash", Error: "exit 1"},
	}

	var sb strings.Builder
	if err := trace.WriteHTML(&sb, spans); err != nil {
		t.Fatalf("WriteHTML: got error %v, want nil", err)
	}
	html := sb.String()

	if len(html) < 5000 {
		t.Errorf("report length: got %d bytes, want a non-trivial document", len(html))
	}
	// The artifact promised is one file that opens from a ticket attachment on
	// a machine with no network.
	for _, forbidden := range []string{"http://", "https://", "<link", "<img", "src=", "@import"} {
		if strings.Contains(html, forbidden) {
			t.Errorf("report contains %q, so it is not self-contained", forbidden)
		}
	}
	if strings.Contains(html, "__WOMBAT_TRACE_DATA__") {
		t.Error("the data marker survived into the report; the spans were not spliced in")
	}
	for _, want := range []string{`"planner"`, `"bash"`, `"exit 1"`} {
		if !strings.Contains(html, want) {
			t.Errorf("report does not contain %s", want)
		}
	}

	t.Run("nil spans", func(t *testing.T) {
		var sb strings.Builder
		if err := trace.WriteHTML(&sb, nil); err != nil {
			t.Fatalf("WriteHTML(nil): got error %v, want nil", err)
		}
		// "null" would make the viewer's empty-file path harder, so nil is
		// normalised to an empty array before it is spliced in.
		if !strings.Contains(sb.String(), ">[]<") {
			t.Errorf("nil spans were not rendered as an empty array; got a report without >[]<")
		}
	})

	t.Run("write failure is reported", func(t *testing.T) {
		err := trace.WriteHTML(failWriter{}, spans)
		if err == nil {
			t.Fatal("got nil error, want the write failure")
		}
	})
}

// TestWriteHTMLEscapesAScriptEndTag: the span array is spliced into a <script>
// element, so a span named "</script>" would otherwise close the element and
// turn a trace of a hostile tool call into markup the browser runs. json.Marshal
// (not an encoder with SetEscapeHTML(false)) is what prevents it, which is the
// opposite default from the event stream — an easy thing to "fix" by accident.
func TestWriteHTMLEscapesAScriptEndTag(t *testing.T) {
	spans := []trace.Span{{
		ID:   "1",
		Kind: trace.KindTool,
		Name: `</script><script>alert('xss')</script>`,
		Attrs: []trace.Attr{
			{Key: "evil", Value: `</script>`},
			{Key: `</script>`, Value: 1},
		},
		Error: `</script>`,
	}}

	var sb strings.Builder
	if err := trace.WriteHTML(&sb, spans); err != nil {
		t.Fatalf("WriteHTML: got error %v, want nil", err)
	}
	html := sb.String()

	// One <script> open and one </script> close per script element in the
	// template: the injected data must add neither.
	base := func() string {
		var b strings.Builder
		if err := trace.WriteHTML(&b, nil); err != nil {
			t.Fatalf("WriteHTML(nil): %v", err)
		}
		return b.String()
	}()
	if got, want := strings.Count(html, "</script>"), strings.Count(base, "</script>"); got != want {
		t.Errorf("</script> occurrences: got %d, want %d — the span data broke out of the script element", got, want)
	}
	if !strings.Contains(html, `\u003c/script\u003e`) {
		t.Errorf("the span name was not escaped; report data:\n%s", html)
	}

	// And the escaped form still decodes back to the original.
	i := strings.Index(html, `[{"id":"1"`)
	if i < 0 {
		t.Fatalf("could not find the span array in the report")
	}
	j := strings.Index(html[i:], "\n")
	if j < 0 {
		j = len(html) - i
	}
	var back []trace.Span
	if err := json.Unmarshal([]byte(strings.TrimSuffix(html[i:i+j], "</script>")), &back); err != nil {
		t.Fatalf("the injected data is not valid JSON: %v", err)
	}
	if len(back) != 1 || back[0].Name != spans[0].Name {
		t.Errorf("round trip: got %+v, want the original name %q", back, spans[0].Name)
	}
}

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }
