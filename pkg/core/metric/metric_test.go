// External test package: everything asserted here is part of the contract a
// user relies on — the instruments, the label handling, the overflow cap, the
// exposition bytes, the two middlewares. Nothing needs an unexported field,
// and keeping the tests outside means they cannot quietly depend on internals
// the package is free to change.
package metric_test

import (
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"wombat/pkg/core/metric"
)

// ===== helpers =====

// find returns the one series with this name and exactly these labels. Label
// order does not matter: the package canonicalizes, so the test does too.
func find(t *testing.T, r *metric.Registry, name string, labels ...metric.Label) metric.Series {
	t.Helper()
	want := slices.Clone(labels)
	slices.SortFunc(want, func(a, b metric.Label) int { return strings.Compare(a.Key, b.Key) })

	for _, s := range r.Snapshot() {
		if s.Name != name || !slices.Equal(s.Labels, want) {
			continue
		}
		return s
	}
	t.Fatalf("no series %s%v in snapshot:\n%s", name, want, dump(r))
	return metric.Series{}
}

// missing asserts that no series with this name exists at all.
func missing(t *testing.T, r *metric.Registry, name string) {
	t.Helper()
	for _, s := range r.Snapshot() {
		if s.Name == name {
			t.Fatalf("series %s exists, want it absent:\n%s", name, dump(r))
		}
	}
}

func dump(r *metric.Registry) string {
	var b strings.Builder
	if err := r.WriteText(&b); err != nil {
		return "WriteText: " + err.Error()
	}
	return b.String()
}

func text(t *testing.T, r *metric.Registry) string {
	t.Helper()
	var b strings.Builder
	if err := r.WriteText(&b); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	return b.String()
}

// mustPanic runs fn and fails unless it panicked.
func mustPanic(t *testing.T, what string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: no panic, want one", what)
		}
	}()
	fn()
}

const eps = 1e-9

func closeTo(got, want float64) bool { return math.Abs(got-want) < eps }

// ===== counters =====

func TestCounter(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		do     func(*metric.Counter)
		labels []metric.Label
		want   float64
	}{
		{
			name: "inc once",
			do:   func(c *metric.Counter) { c.Inc() },
			want: 1,
		},
		{
			name: "inc repeatedly",
			do:   func(c *metric.Counter) { c.Inc(); c.Inc(); c.Inc() },
			want: 3,
		},
		{
			name: "add fractional",
			do:   func(c *metric.Counter) { c.Add(0.25); c.Add(0.5) },
			want: 0.75,
		},
		{
			name: "add zero materializes the series",
			do:   func(c *metric.Counter) { c.Add(0) },
			want: 0,
		},
		{
			// A counter that goes backwards makes rate() produce a nonsense
			// spike across the whole lookback window; one dropped sample is
			// the smaller lie.
			name: "negative add is dropped",
			do:   func(c *metric.Counter) { c.Inc(); c.Add(-5) },
			want: 1,
		},
		{
			name: "NaN add is dropped",
			do:   func(c *metric.Counter) { c.Inc(); c.Add(math.NaN()) },
			want: 1,
		},
		{
			name:   "labelled",
			do:     func(c *metric.Counter) { c.Add(2, metric.Label{Key: "a", Value: "x"}) },
			labels: []metric.Label{{Key: "a", Value: "x"}},
			want:   2,
		},
		{
			name: "distinct label sets are distinct series",
			do: func(c *metric.Counter) {
				c.Add(2, metric.Label{Key: "a", Value: "x"})
				c.Add(7, metric.Label{Key: "a", Value: "y"})
			},
			labels: []metric.Label{{Key: "a", Value: "y"}},
			want:   7,
		},
		{
			// The same pairs in a different order must land on one series, or
			// a caller who builds labels from a map gets two.
			name: "label order does not create a second series",
			do: func(c *metric.Counter) {
				c.Inc(metric.Label{Key: "a", Value: "1"}, metric.Label{Key: "b", Value: "2"})
				c.Inc(metric.Label{Key: "b", Value: "2"}, metric.Label{Key: "a", Value: "1"})
			},
			labels: []metric.Label{{Key: "a", Value: "1"}, {Key: "b", Value: "2"}},
			want:   2,
		},
		{
			name: "repeated label name keeps the last value",
			do: func(c *metric.Counter) {
				c.Inc(metric.Label{Key: "a", Value: "first"}, metric.Label{Key: "a", Value: "last"})
			},
			labels: []metric.Label{{Key: "a", Value: "last"}},
			want:   1,
		},
		{
			name:   "empty label name is dropped",
			do:     func(c *metric.Counter) { c.Inc(metric.Label{Key: "", Value: "orphan"}) },
			labels: nil,
			want:   1,
		},
		{
			name:   "invalid label name is sanitized",
			do:     func(c *metric.Counter) { c.Inc(metric.Label{Key: "tool.name-x", Value: "v"}) },
			labels: []metric.Label{{Key: "tool_name_x", Value: "v"}},
			want:   1,
		},
		{
			name:   "leading digit in a label name is sanitized",
			do:     func(c *metric.Counter) { c.Inc(metric.Label{Key: "9lives", Value: "v"}) },
			labels: []metric.Label{{Key: "_lives", Value: "v"}},
			want:   1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			c := r.Counter("wombat_test_total", "help")
			tc.do(c)

			got := find(t, r, "wombat_test_total", tc.labels...)
			if !closeTo(got.Value, tc.want) {
				t.Errorf("value = %v, want %v", got.Value, tc.want)
			}
			if got.Type != "counter" {
				t.Errorf("type = %q, want counter", got.Type)
			}
		})
	}
}

func TestCounterUntouchedIsAbsent(t *testing.T) {
	t.Parallel()

	// Registering an instrument declares it; it does not create a series. That
	// keeps Snapshot and WriteText describing exactly the same set.
	r := metric.NewRegistry()
	r.Counter("wombat_never_total", "help")
	missing(t, r, "wombat_never_total")

	if got := text(t, r); got != "" {
		t.Errorf("WriteText = %q, want empty", got)
	}
}

// ===== gauges =====

func TestGauge(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		do   func(*metric.Gauge)
		want float64
	}{
		{"set", func(g *metric.Gauge) { g.Set(4) }, 4},
		{"set overwrites", func(g *metric.Gauge) { g.Set(4); g.Set(1) }, 1},
		{"set negative", func(g *metric.Gauge) { g.Set(-2.5) }, -2.5},
		{"add", func(g *metric.Gauge) { g.Add(3); g.Add(2) }, 5},
		{"add negative", func(g *metric.Gauge) { g.Add(3); g.Add(-4) }, -1},
		{"set then add", func(g *metric.Gauge) { g.Set(10); g.Add(-1) }, 9},
		{"NaN set is dropped", func(g *metric.Gauge) { g.Set(2); g.Set(math.NaN()) }, 2},
		{"NaN add is dropped", func(g *metric.Gauge) { g.Set(2); g.Add(math.NaN()) }, 2},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			tc.do(r.Gauge("wombat_test_gauge", "help"))

			got := find(t, r, "wombat_test_gauge")
			if !closeTo(got.Value, tc.want) {
				t.Errorf("value = %v, want %v", got.Value, tc.want)
			}
			if got.Type != "gauge" {
				t.Errorf("type = %q, want gauge", got.Type)
			}
		})
	}
}

// ===== histograms =====

func TestHistogramBuckets(t *testing.T) {
	t.Parallel()

	buckets := []float64{1, 2, 5}

	tests := []struct {
		name    string
		observe []float64
		wantCum []uint64 // cumulative counts for le=1, 2, 5, +Inf
		wantSum float64
		wantN   uint64
	}{
		{
			name:    "below the first bound",
			observe: []float64{0.5},
			wantCum: []uint64{1, 1, 1, 1},
			wantSum: 0.5,
			wantN:   1,
		},
		{
			// le is "less than or EQUAL": a value exactly on a boundary counts
			// in that bucket, not the next one.
			name:    "exactly on a boundary",
			observe: []float64{1, 2, 5},
			wantCum: []uint64{1, 2, 3, 3},
			wantSum: 8,
			wantN:   3,
		},
		{
			name:    "spread across buckets",
			observe: []float64{0.5, 1.5, 3, 9},
			wantCum: []uint64{1, 2, 3, 4},
			wantSum: 14,
			wantN:   4,
		},
		{
			name:    "everything above the last bound lands only in +Inf",
			observe: []float64{100, 200},
			wantCum: []uint64{0, 0, 0, 2},
			wantSum: 300,
			wantN:   2,
		},
		{
			name:    "zero and negative",
			observe: []float64{0, -3},
			wantCum: []uint64{2, 2, 2, 2},
			wantSum: -3,
			wantN:   2,
		},
		{
			name:    "NaN is dropped",
			observe: []float64{1, math.NaN()},
			wantCum: []uint64{1, 1, 1, 1},
			wantSum: 1,
			wantN:   1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			h := r.Histogram("wombat_test_seconds", "help", buckets)
			for _, v := range tc.observe {
				h.Observe(v)
			}

			got := find(t, r, "wombat_test_seconds")
			if got.Type != "histogram" {
				t.Errorf("type = %q, want histogram", got.Type)
			}
			if got.Count != tc.wantN {
				t.Errorf("count = %d, want %d", got.Count, tc.wantN)
			}
			if !closeTo(got.Sum, tc.wantSum) {
				t.Errorf("sum = %v, want %v", got.Sum, tc.wantSum)
			}

			wantLE := []float64{1, 2, 5, math.Inf(1)}
			if len(got.Buckets) != len(wantLE) {
				t.Fatalf("got %d buckets, want %d: %+v", len(got.Buckets), len(wantLE), got.Buckets)
			}
			for i, b := range got.Buckets {
				if b.LE != wantLE[i] {
					t.Errorf("bucket %d LE = %v, want %v", i, b.LE, wantLE[i])
				}
				if b.Count != tc.wantCum[i] {
					t.Errorf("bucket le=%v count = %d, want %d", b.LE, b.Count, tc.wantCum[i])
				}
			}
			// The last bucket is +Inf and must equal _count, so a consumer
			// never has to reconstruct it.
			if last := got.Buckets[len(got.Buckets)-1]; last.Count != got.Count {
				t.Errorf("+Inf bucket = %d, count = %d, want equal", last.Count, got.Count)
			}
		})
	}
}

func TestHistogramBucketNormalization(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		in     []float64
		wantLE []float64
	}{
		{"already sorted", []float64{1, 2, 3}, []float64{1, 2, 3, math.Inf(1)}},
		{"unsorted is sorted", []float64{3, 1, 2}, []float64{1, 2, 3, math.Inf(1)}},
		{"duplicates collapse", []float64{1, 1, 2}, []float64{1, 2, math.Inf(1)}},
		{"explicit +Inf is dropped", []float64{1, math.Inf(1)}, []float64{1, math.Inf(1)}},
		{"NaN is dropped", []float64{1, math.NaN()}, []float64{1, math.Inf(1)}},
		{"empty yields +Inf only", nil, []float64{math.Inf(1)}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			r.Histogram("wombat_test_seconds", "help", tc.in).Observe(0.5)

			got := find(t, r, "wombat_test_seconds")
			les := make([]float64, len(got.Buckets))
			for i, b := range got.Buckets {
				les[i] = b.LE
			}
			if !slices.Equal(les, tc.wantLE) {
				t.Errorf("buckets = %v, want %v", les, tc.wantLE)
			}
		})
	}
}

func TestHistogramInfinityIsObserved(t *testing.T) {
	t.Parallel()

	// +Inf belongs in the +Inf bucket and, deliberately, still lands in _sum:
	// an infinite _sum is how an infinite duration upstream announces itself.
	r := metric.NewRegistry()
	r.Histogram("wombat_test_seconds", "help", []float64{1}).Observe(math.Inf(1))

	got := find(t, r, "wombat_test_seconds")
	if got.Count != 1 {
		t.Fatalf("count = %d, want 1", got.Count)
	}
	if !math.IsInf(got.Sum, 1) {
		t.Errorf("sum = %v, want +Inf", got.Sum)
	}
	if got.Buckets[0].Count != 0 || got.Buckets[1].Count != 1 {
		t.Errorf("buckets = %+v, want le=1:0 +Inf:1", got.Buckets)
	}
}

// ===== registration =====

func TestDuplicateRegistrationReturnsSameInstrument(t *testing.T) {
	t.Parallel()

	// A middleware built twice is ordinary; it must not be a crash, and both
	// handles must feed one series.
	r := metric.NewRegistry()

	c1 := r.Counter("wombat_dup_total", "first")
	c2 := r.Counter("wombat_dup_total", "second")
	if c1 != c2 {
		t.Errorf("Counter returned two distinct instruments")
	}
	c1.Inc()
	c2.Inc()
	if got := find(t, r, "wombat_dup_total"); got.Value != 2 {
		t.Errorf("value = %v, want 2 — both handles must share a series", got.Value)
	}
	// The first help text wins; a later registration does not rewrite it.
	if got := find(t, r, "wombat_dup_total"); got.Help != "first" {
		t.Errorf("help = %q, want %q", got.Help, "first")
	}

	g1 := r.Gauge("wombat_dup_gauge", "help")
	if g2 := r.Gauge("wombat_dup_gauge", "help"); g1 != g2 {
		t.Errorf("Gauge returned two distinct instruments")
	}

	h1 := r.Histogram("wombat_dup_seconds", "help", []float64{1, 2})
	// The second call's buckets are ignored: rewriting the boundaries of a
	// histogram that already has observations would corrupt every one of them.
	h2 := r.Histogram("wombat_dup_seconds", "help", []float64{99})
	if h1 != h2 {
		t.Fatalf("Histogram returned two distinct instruments")
	}
	h2.Observe(1.5)
	if got := find(t, r, "wombat_dup_seconds"); len(got.Buckets) != 3 {
		t.Errorf("buckets = %+v, want the original 1,2,+Inf", got.Buckets)
	}
}

func TestRegistrationPanics(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		fn   func(*metric.Registry)
	}{
		{"empty name", func(r *metric.Registry) { r.Counter("", "help") }},
		{"dash in name", func(r *metric.Registry) { r.Counter("wombat-calls", "help") }},
		{"leading digit", func(r *metric.Registry) { r.Counter("1wombat", "help") }},
		{"dot in name", func(r *metric.Registry) { r.Gauge("wombat.calls", "help") }},
		{
			// The two call sites disagree about what the metric means, and
			// whichever lost would write samples the TYPE line forbids.
			name: "counter then gauge",
			fn: func(r *metric.Registry) {
				r.Counter("wombat_clash", "help")
				r.Gauge("wombat_clash", "help")
			},
		},
		{
			name: "counter then histogram",
			fn: func(r *metric.Registry) {
				r.Counter("wombat_clash", "help")
				r.Histogram("wombat_clash", "help", nil)
			},
		},
		{"nil registry", func(*metric.Registry) { metric.New(nil) }},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			mustPanic(t, tc.name, func() { tc.fn(metric.NewRegistry()) })
		})
	}
}

func TestValidNamesDoNotPanic(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	for _, name := range []string{"a", "_x", ":ns:total", "wombat_llm_calls_total", "A9_b"} {
		r.Counter(name, "help")
	}
}

// ===== cardinality =====

func TestOverflowCap(t *testing.T) {
	t.Parallel()

	const extra = 50
	r := metric.NewRegistry()
	c := r.Counter("wombat_wide_total", "help")

	for i := 0; i < metric.MaxSeries+extra; i++ {
		c.Inc(metric.Label{Key: "id", Value: strconv.Itoa(i)})
	}

	snap := r.Snapshot()
	// MaxSeries distinct sets, plus the one shared overflow series.
	if len(snap) != metric.MaxSeries+1 {
		t.Fatalf("got %d series, want %d", len(snap), metric.MaxSeries+1)
	}

	over := find(t, r, "wombat_wide_total", metric.Label{Key: "overflow", Value: "true"})
	if over.Value != extra {
		t.Errorf("overflow value = %v, want %d — the total must survive even though the breakdown does not", over.Value, extra)
	}

	// The label sets admitted before the cap are still individually tracked.
	if got := find(t, r, "wombat_wide_total", metric.Label{Key: "id", Value: "0"}); got.Value != 1 {
		t.Errorf("id=0 value = %v, want 1", got.Value)
	}

	// Folding is sticky: a set already folded stays folded.
	c.Inc(metric.Label{Key: "id", Value: strconv.Itoa(metric.MaxSeries)})
	if got := find(t, r, "wombat_wide_total", metric.Label{Key: "overflow", Value: "true"}); got.Value != extra+1 {
		t.Errorf("overflow value = %v, want %d", got.Value, extra+1)
	}
	if len(r.Snapshot()) != metric.MaxSeries+1 {
		t.Errorf("series count grew past the cap")
	}
}

func TestOverflowSharesAPreexistingOverflowSeries(t *testing.T) {
	t.Parallel()

	// A caller who legitimately uses overflow="true" before the cap must not
	// end up with two series carrying identical labels — that is an invalid
	// exposition, not just an ugly one.
	r := metric.NewRegistry()
	c := r.Counter("wombat_wide_total", "help")
	c.Inc(metric.Label{Key: "overflow", Value: "true"})

	for i := 0; i < metric.MaxSeries; i++ {
		c.Inc(metric.Label{Key: "id", Value: strconv.Itoa(i)})
	}

	snap := r.Snapshot()
	if len(snap) != metric.MaxSeries {
		t.Fatalf("got %d series, want %d", len(snap), metric.MaxSeries)
	}
	// One legitimate hit, plus the one that folded once the cap was reached.
	if got := find(t, r, "wombat_wide_total", metric.Label{Key: "overflow", Value: "true"}); got.Value != 2 {
		t.Errorf("overflow value = %v, want 2", got.Value)
	}
	if strings.Count(text(t, r), `{overflow="true"}`) != 1 {
		t.Errorf("overflow appears more than once:\n%s", text(t, r))
	}
}

func TestOverflowIsPerInstrument(t *testing.T) {
	t.Parallel()

	// The cap is per instrument, so one runaway metric does not silence the
	// rest of the registry.
	r := metric.NewRegistry()
	wide := r.Counter("wombat_wide_total", "help")
	narrow := r.Counter("wombat_narrow_total", "help")

	for i := 0; i < metric.MaxSeries+10; i++ {
		wide.Inc(metric.Label{Key: "id", Value: strconv.Itoa(i)})
	}
	narrow.Inc(metric.Label{Key: "id", Value: "only"})

	if got := find(t, r, "wombat_narrow_total", metric.Label{Key: "id", Value: "only"}); got.Value != 1 {
		t.Errorf("narrow value = %v, want 1", got.Value)
	}
	missing(t, r, "wombat_narrow_overflow")
}

func TestLabelValueTruncation(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	c := r.Counter("wombat_test_total", "help")

	long := strings.Repeat("x", metric.MaxLabelValueBytes+40)
	c.Inc(metric.Label{Key: "v", Value: long})

	snap := r.Snapshot()
	if len(snap) != 1 {
		t.Fatalf("got %d series, want 1", len(snap))
	}
	if got := len(snap[0].Labels[0].Value); got != metric.MaxLabelValueBytes {
		t.Errorf("value length = %d, want %d", got, metric.MaxLabelValueBytes)
	}

	// Truncation must land on a rune boundary or the exposition carries
	// invalid UTF-8.
	multi := strings.Repeat("é", metric.MaxLabelValueBytes)
	c.Inc(metric.Label{Key: "u", Value: multi})
	for _, s := range r.Snapshot() {
		for _, l := range s.Labels {
			if !utf8.ValidString(l.Value) {
				t.Errorf("label %q value is not valid UTF-8 after truncation", l.Key)
			}
		}
	}
}

// ===== snapshot =====

func TestSnapshotIsACopy(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	r.Counter("wombat_test_total", "help").Inc(metric.Label{Key: "a", Value: "x"})
	r.Histogram("wombat_test_seconds", "help", []float64{1}).Observe(0.5)

	snap := r.Snapshot()
	for i := range snap {
		for j := range snap[i].Labels {
			snap[i].Labels[j].Value = "clobbered"
		}
		for j := range snap[i].Buckets {
			snap[i].Buckets[j].Count = 9999
		}
	}

	if got := find(t, r, "wombat_test_total", metric.Label{Key: "a", Value: "x"}); got.Value != 1 {
		t.Errorf("mutating a snapshot disturbed the registry")
	}
	if got := find(t, r, "wombat_test_seconds"); got.Buckets[0].Count != 1 {
		t.Errorf("mutating a snapshot disturbed a histogram: %+v", got.Buckets)
	}
}

func TestSnapshotIsSorted(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	c := r.Counter("wombat_b_total", "help")
	r.Counter("wombat_a_total", "help").Inc()
	c.Inc(metric.Label{Key: "z", Value: "2"})
	c.Inc(metric.Label{Key: "z", Value: "1"})
	c.Inc(metric.Label{Key: "a", Value: "9"})

	snap := r.Snapshot()
	var got []string
	for _, s := range snap {
		got = append(got, s.Name+labelString(s.Labels))
	}
	want := []string{
		"wombat_a_total",
		`wombat_b_total{a="9"}`,
		`wombat_b_total{z="1"}`,
		`wombat_b_total{z="2"}`,
	}
	if !slices.Equal(got, want) {
		t.Errorf("order =\n%v\nwant\n%v", got, want)
	}
}

func labelString(labels []metric.Label) string {
	if len(labels) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteByte('{')
	for i, l := range labels {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(l.Key)
		b.WriteString(`="`)
		b.WriteString(l.Value)
		b.WriteByte('"')
	}
	b.WriteByte('}')
	return b.String()
}

// ===== concurrency =====

func TestConcurrentWritersAndScrapes(t *testing.T) {
	t.Parallel()

	// The shape this package exists to survive: a fan-out of sub-agents
	// writing while a scrape reads. Meaningful under -race.
	const (
		writers = 16
		each    = 200
		sets    = 4
	)

	r := metric.NewRegistry()
	c := r.Counter("wombat_test_total", "help")
	h := r.Histogram("wombat_test_seconds", "help", []float64{0.5, 1, 2})
	g := r.Gauge("wombat_test_gauge", "help")

	stop := make(chan struct{})
	var scrapes sync.WaitGroup
	scrapes.Add(2)
	go func() {
		defer scrapes.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = dump(r)
			}
		}
	}()
	go func() {
		defer scrapes.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = r.Snapshot()
			}
		}
	}()

	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				l := metric.Label{Key: "set", Value: strconv.Itoa(i % sets)}
				c.Inc(l)
				h.Observe(float64(i%3)*0.6, l)
				g.Add(1, l)
				g.Add(-1, l)
				// Re-registering from several goroutines at once is what a
				// middleware built per-request would do.
				r.Counter("wombat_test_total", "help").Inc(metric.Label{Key: "w", Value: strconv.Itoa(w)})
			}
		}(w)
	}
	wg.Wait()
	close(stop)
	scrapes.Wait()

	var total float64
	for _, s := range r.Snapshot() {
		if s.Name == "wombat_test_total" {
			total += s.Value
		}
	}
	if want := float64(writers * each * 2); total != want {
		t.Errorf("counter total = %v, want %v", total, want)
	}

	var observed uint64
	for _, s := range r.Snapshot() {
		if s.Name == "wombat_test_seconds" {
			observed += s.Count
			// The triple must be internally consistent even though it was
			// written concurrently: +Inf equals _count.
			if last := s.Buckets[len(s.Buckets)-1]; last.Count != s.Count {
				t.Errorf("+Inf = %d, count = %d, want equal", last.Count, s.Count)
			}
		}
	}
	if want := uint64(writers * each); observed != want {
		t.Errorf("histogram count = %d, want %d", observed, want)
	}

	for _, s := range r.Snapshot() {
		if s.Name == "wombat_test_gauge" && !closeTo(s.Value, 0) {
			t.Errorf("gauge = %v, want 0 after balanced adds", s.Value)
		}
	}
}

func TestConcurrentOverflow(t *testing.T) {
	t.Parallel()

	// Racing goroutines onto an unbounded label space: the cap must hold
	// exactly, with no torn double-creation of the overflow series.
	const (
		writers = 8
		each    = 200
	)

	r := metric.NewRegistry()
	c := r.Counter("wombat_wide_total", "help")

	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				c.Inc(metric.Label{Key: "id", Value: strconv.Itoa(w*each + i)})
			}
		}(w)
	}
	wg.Wait()

	snap := r.Snapshot()
	if len(snap) > metric.MaxSeries+1 {
		t.Fatalf("got %d series, want at most %d", len(snap), metric.MaxSeries+1)
	}

	var total float64
	seen := map[string]bool{}
	for _, s := range snap {
		total += s.Value
		key := labelString(s.Labels)
		if seen[key] {
			t.Errorf("duplicate label set %s", key)
		}
		seen[key] = true
	}
	if want := float64(writers * each); total != want {
		t.Errorf("total = %v, want %v — folding must not lose counts", total, want)
	}
}
