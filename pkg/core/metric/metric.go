// Package metric aggregates what a run did into counters, histograms and
// gauges, and exposes them in the Prometheus text exposition format.
//
// The harness already has structured logs and a trace file. Neither
// aggregates: [wombat/pkg/core/governor.Progress] dies with
// its run, a Spend event reaches exactly one consumer, and trace attributes
// sit in a file nobody sums. This package is the third leg — the one a scraper
// reads.
//
// # Why there is no Prometheus dependency
//
// client_golang and the OpenTelemetry metrics SDK are both the obvious answer,
// and both are third-party modules. This library has none and that is the
// point, so the exposition format is written here instead. It is a few dozen
// lines of text — a HELP line, a TYPE line, one sample per series — and every
// scraper in existence reads it. [Registry.Handler] serves it; point a scrape
// config at it and you are done.
//
// The API is shaped so that a bridge to a real SDK is short. [Registry.Snapshot]
// hands back the whole registry as flat [Series] values, already sorted, with
// histogram buckets cumulative and a +Inf bucket present — which is exactly the
// shape an OTel asynchronous instrument wants:
//
//	meter := provider.Meter("wombat")
//	_, _ = meter.Float64ObservableCounter("placeholder",
//	    metricapi.WithFloat64Callback(func(ctx context.Context, o metricapi.Float64Observer) error {
//	        for _, s := range reg.Snapshot() {
//	            attrs := make([]attribute.KeyValue, 0, len(s.Labels))
//	            for _, l := range s.Labels {
//	                attrs = append(attrs, attribute.String(l.Key, l.Value))
//	            }
//	            switch s.Type {
//	            case "counter", "gauge":
//	                o.Observe(s.Value, metricapi.WithAttributes(attrs...))
//	            case "histogram":
//	                // OTel has no way to push a pre-bucketed histogram through
//	                // the API; emit _sum and _count as counters, or use the
//	                // SDK's metricdata.Histogram directly from an exporter.
//	                o.Observe(s.Sum, metricapi.WithAttributes(attrs...))
//	            }
//	        }
//	        return nil
//	    }))
//
// That bridge lives in the user's module, where the dependency belongs.
//
// # Why WithErrorClass exists
//
// An outcome label is the most useful label on the whole surface and its
// vocabulary is not this package's to decide. A permission refusal is not a
// tool failure — it is a policy decision, and a dashboard that lumps the two
// together will page someone at 3am for a working sandbox. But teaching this
// package about
// [wombat/pkg/core/permission.ErrDenied] would make metric
// import permission, and metric must stay at the bottom of the graph so that
// permission, governor and the root package can all use it.
//
// So the host supplies the vocabulary:
//
//	m.ToolMiddleware(metric.WithErrorClass(func(err error) string {
//	    switch {
//	    case err == nil:                       return "ok"
//	    case errors.Is(err, permission.ErrDenied): return "denied"
//	    case errors.Is(err, context.Canceled):     return "canceled"
//	    default:                                   return "error"
//	    }
//	}))
//
// The default knows two words, "ok" and "error", because those are the only
// two it can justify without importing something.
//
// # Cardinality
//
// A label value taken from a tool name is bounded by the tool set. One taken
// from an error message is bounded by nothing, and a metrics backend that
// receives a new series per unique error string falls over — first the
// exporter, then the TSDB. [WithErrorClass] makes that mistake easy to write,
// so this package refuses to let it run away.
//
// Each instrument admits at most [MaxSeries] distinct label sets. Past that,
// every further label set folds into one shared series labelled
// {overflow="true"}: the breakdown is lost, the total is not, and the series
// count stops growing. Label values are also truncated to
// [MaxLabelValueBytes], because a 40KB stack trace in a label is a problem
// even if there is only one of it.
//
// The cap is 256 because the widest legitimate instrument here is
// wombat_tool_calls_total{tool,outcome} — a large harness runs on the order of
// fifty tools crossed with a handful of outcomes, so roughly 150 to 200 series
// — and 256 leaves headroom while keeping a whole scrape in the tens of
// kilobytes. A run that trips it has a bug, and {overflow="true"} appearing in
// a dashboard is how you find out.
//
// # Concurrency
//
// A scrape reads the registry while a fan-out of sub-agents writes to it, so
// there is no single lock. Three levels, each as narrow as it can be:
//
//   - The registry holds an RWMutex over its name index only. It is taken
//     during [Registry.Counter] and friends — construction time — and read-held
//     for the instant it takes a scrape to copy the instrument list.
//   - Each instrument holds its own RWMutex over its label-set map. The hot
//     path — a label set that already exists — takes it read-only. The write
//     lock is taken once per distinct label set, ever.
//   - Counter and gauge values are atomic.Uint64 holding float64 bits, so two
//     goroutines incrementing the same series never block each other.
//
// Histograms are the exception: each series has its own small mutex, because a
// bucket increment, a sum and a count must move together or a scrape can
// report a count of five with the sum of four observations. The critical
// section is a handful of instructions and it is per label set, so contention
// costs less than the inconsistency would.
//
// # Use
//
//	reg := metric.NewRegistry()
//	m := metric.New(reg)
//
//	a, err := core.New(
//	    core.WithClient(llm.Chain(client, m.LLMMiddleware(llm.DefaultPricing))),
//	    core.WithToolMiddleware(m.ToolMiddleware()),
//	)
//	http.Handle("/metrics", reg.Handler())
package metric

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"
)

// MaxSeries is the number of distinct label sets one instrument will track
// before folding the rest into {overflow="true"}. See the package doc for why
// it is this number.
const MaxSeries = 256

// MaxLabelValueBytes is the length a label value is truncated to. Truncation
// happens on a UTF-8 rune boundary, so the result is always valid UTF-8 and
// therefore always encodable in the exposition format.
//
// 128 bytes fits every model id, tool name and outcome word this harness
// produces, and cuts an error message down to something a dashboard can render
// in a legend.
const MaxLabelValueBytes = 128

// Metric type names, as they appear in a TYPE line and in [Series.Type].
const (
	typeCounter   = "counter"
	typeGauge     = "gauge"
	typeHistogram = "histogram"
)

// Label is one key/value pair on a series.
//
// A slice of these and not a map, for the same reason [Series] is sorted: the
// exposition format is a text artifact that people diff and scrapers hash, and
// a stable order is worth more than the convenience of a map.
type Label struct{ Key, Value string }

// Bucket is one cumulative histogram bucket: the number of observations less
// than or equal to LE.
type Bucket struct {
	LE    float64
	Count uint64
}

// Series is one instrument at one label set, flattened.
//
// Type is "counter", "gauge" or "histogram". For a counter or a gauge only
// Value is meaningful. For a histogram only Buckets, Sum and Count are: the
// buckets are cumulative and the last one is always LE=+Inf with Count equal
// to Count, so a consumer never has to reconstruct it.
type Series struct {
	Name, Help, Type string
	Labels           []Label
	Value            float64
	Buckets          []Bucket
	Sum              float64
	Count            uint64
}

// ===== registry =====

// Registry is a set of instruments that are scraped together.
//
// Safe for concurrent use, including a scrape running against a live fan-out.
// See the package doc for what is locked and what is atomic.
type Registry struct {
	mu   sync.RWMutex
	vecs map[string]*vec
}

// NewRegistry returns an empty registry.
//
// There is no package-level default registry on purpose. A global would make
// two agents in one process — a test suite, a server handling two tenants —
// silently share counters, and the resulting numbers are worse than no numbers.
func NewRegistry() *Registry {
	return &Registry{vecs: make(map[string]*vec)}
}

// Counter returns the counter called name, creating it if this is the first
// time. A counter only goes up; see [Counter.Add].
func (r *Registry) Counter(name, help string) *Counter {
	return r.register(name, help, typeCounter, nil).wrap.(*Counter)
}

// Gauge returns the gauge called name, creating it if this is the first time.
func (r *Registry) Gauge(name, help string) *Gauge {
	return r.register(name, help, typeGauge, nil).wrap.(*Gauge)
}

// Histogram returns the histogram called name, creating it if this is the
// first time.
//
// buckets are upper bounds in the instrument's own unit — seconds, for
// everything this package defines. They are sorted and de-duplicated on the
// way in, so a caller need not; NaN and +Inf entries are dropped, the latter
// because the +Inf bucket is implicit and always emitted. An empty slice is
// legal and yields a histogram that reports only _sum, _count and +Inf, which
// is a perfectly good way to track a mean.
//
// On a repeat call the buckets argument is ignored and the original histogram
// comes back unchanged — changing the boundaries of a histogram that already
// has observations in it would silently corrupt every one of them.
func (r *Registry) Histogram(name, help string, buckets []float64) *Histogram {
	return r.register(name, help, typeHistogram, normalizeBuckets(buckets)).wrap.(*Histogram)
}

// register is the one place an instrument comes into existence.
//
// Re-registering the same name and type returns the identical instrument
// rather than panicking, because building a middleware twice — a test that
// constructs two agents, a caller that wires both LLMMiddleware and
// ToolMiddleware from separate [New] calls — is ordinary and must not be a
// crash.
//
// Re-registering the same name with a DIFFERENT type does panic. That is a
// programmer error at construction time with no sensible recovery: the two
// call sites disagree about what the metric means, and whichever loses would
// write samples that the winner's TYPE line declares impossible.
func (r *Registry) register(name, help, typ string, buckets []float64) *vec {
	if !validMetricName(name) {
		panic(fmt.Sprintf("metric: invalid metric name %q: must match [a-zA-Z_:][a-zA-Z0-9_:]*", name))
	}

	r.mu.RLock()
	v := r.vecs[name]
	r.mu.RUnlock()
	if v == nil {
		r.mu.Lock()
		if v = r.vecs[name]; v == nil {
			v = &vec{
				name:     name,
				help:     help,
				typ:      typ,
				buckets:  buckets,
				children: make(map[string]*series),
			}
			switch typ {
			case typeCounter:
				v.wrap = &Counter{v: v}
			case typeGauge:
				v.wrap = &Gauge{v: v}
			case typeHistogram:
				v.wrap = &Histogram{v: v}
			}
			r.vecs[name] = v
		}
		r.mu.Unlock()
	}

	if v.typ != typ {
		panic(fmt.Sprintf("metric: %q is already registered as a %s, cannot re-register as a %s", name, v.typ, typ))
	}
	return v
}

// Snapshot returns every series in the registry, sorted by name and then by
// label set.
//
// This is the escape hatch for a consumer that is not a Prometheus scraper —
// an OTel bridge, a test, a JSON status endpoint. The returned slice and every
// slice inside it are copies; mutating them cannot disturb the registry.
//
// The snapshot is not a transaction. Counters keep moving while it is being
// taken, so two series in one snapshot may be microseconds apart. That is true
// of every scrape of every metrics system and no consumer of a counter cares,
// because the thing they compute is a rate.
func (r *Registry) Snapshot() []Series {
	r.mu.RLock()
	vecs := make([]*vec, 0, len(r.vecs))
	for _, v := range r.vecs {
		vecs = append(vecs, v)
	}
	r.mu.RUnlock()

	var out []Series
	for _, v := range vecs {
		out = v.appendSnapshot(out)
	}
	slices.SortFunc(out, compareSeries)
	return out
}

// compareSeries orders by name, then by label set, which is what makes two
// scrapes of an unchanged registry byte-identical.
func compareSeries(a, b Series) int {
	if c := strings.Compare(a.Name, b.Name); c != 0 {
		return c
	}
	return compareLabels(a.Labels, b.Labels)
}

func compareLabels(a, b []Label) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if c := strings.Compare(a[i].Key, b[i].Key); c != 0 {
			return c
		}
		if c := strings.Compare(a[i].Value, b[i].Value); c != 0 {
			return c
		}
	}
	return cmp.Compare(len(a), len(b))
}

// ===== vec: one instrument, many label sets =====

type vec struct {
	name, help string
	typ        string
	buckets    []float64 // histogram only; sorted, finite, no duplicates
	wrap       any       // the *Counter, *Gauge or *Histogram handed to callers

	// mu guards children only. Values inside a series are atomic (counter,
	// gauge) or guarded by the series' own mutex (histogram), so the hot path
	// holds this read-only and concurrent writers to different label sets —
	// or the same one — never serialize on it.
	mu       sync.RWMutex
	children map[string]*series
}

// series is one label set's worth of state.
//
// One struct for all three instrument types: bits is the counter/gauge value
// and the histogram fields are unused, or the reverse. The waste is a few dozen
// bytes on at most MaxSeries entries per instrument, which is cheaper than the
// interface dispatch the alternative would put on the hot path.
type series struct {
	labels []Label

	// bits is a float64's bit pattern, for counters and gauges. Atomic so that
	// the overwhelmingly common operation in this package — increment a
	// counter that already exists — takes no lock at all.
	bits atomic.Uint64

	// mu guards the histogram triple. They must move together: a scrape that
	// sees the bucket increment but not the sum reports a mean that never
	// happened.
	mu      sync.Mutex
	counts  []uint64 // len(vec.buckets)+1; the last entry is the +Inf-only tail
	sum     float64
	count   uint64
	isHisto bool
}

// overflowLabels is the label set every series past the cap folds into.
var overflowLabels = []Label{{Key: "overflow", Value: "true"}}

// overflowKey is its canonical key, precomputed so the cap check is a map
// lookup. Reserving the key rather than minting a second series means a caller
// who legitimately passes overflow="true" shares the same series instead of
// producing a duplicate line in the exposition, which would make the scrape
// invalid.
var overflowKey = labelKey(overflowLabels)

// get returns the series for labels, creating it or folding it into the
// overflow series.
func (v *vec) get(labels []Label) *series {
	canon, key := canonicalize(labels)

	v.mu.RLock()
	s := v.children[key]
	v.mu.RUnlock()
	if s != nil {
		return s
	}

	v.mu.Lock()
	defer v.mu.Unlock()
	if s = v.children[key]; s != nil {
		return s
	}
	if len(v.children) >= MaxSeries {
		if s = v.children[overflowKey]; s != nil {
			return s
		}
		// Admitted past the cap on purpose: it is the one series that must
		// exist for the total to stay correct, and there is exactly one of it.
		s = v.newSeries(overflowLabels)
		v.children[overflowKey] = s
		return s
	}
	s = v.newSeries(canon)
	v.children[key] = s
	return s
}

func (v *vec) newSeries(labels []Label) *series {
	s := &series{labels: labels}
	if v.typ == typeHistogram {
		s.isHisto = true
		s.counts = make([]uint64, len(v.buckets)+1)
	}
	return s
}

// appendSnapshot flattens the instrument onto out.
func (v *vec) appendSnapshot(out []Series) []Series {
	v.mu.RLock()
	kids := make([]*series, 0, len(v.children))
	for _, s := range v.children {
		kids = append(kids, s)
	}
	v.mu.RUnlock()

	for _, s := range kids {
		out = append(out, v.snapshotOne(s))
	}
	return out
}

func (v *vec) snapshotOne(s *series) Series {
	out := Series{
		Name:   v.name,
		Help:   v.help,
		Type:   v.typ,
		Labels: slices.Clone(s.labels),
	}
	if v.typ != typeHistogram {
		out.Value = math.Float64frombits(s.bits.Load())
		return out
	}

	s.mu.Lock()
	counts := slices.Clone(s.counts)
	out.Sum, out.Count = s.sum, s.count
	s.mu.Unlock()

	// Buckets are stored per-bucket and made cumulative here, because the
	// cumulative form is what both the exposition format and every consumer
	// want, and doing it once per scrape beats doing it once per observation.
	out.Buckets = make([]Bucket, 0, len(v.buckets)+1)
	var running uint64
	for i, le := range v.buckets {
		running += counts[i]
		out.Buckets = append(out.Buckets, Bucket{LE: le, Count: running})
	}
	out.Buckets = append(out.Buckets, Bucket{LE: math.Inf(1), Count: out.Count})
	return out
}

// ===== instruments =====

// Counter is a monotonically increasing total, one per label set.
type Counter struct{ v *vec }

// Inc adds one.
func (c *Counter) Inc(labels ...Label) { c.Add(1, labels...) }

// Add adds v, which must not be negative.
//
// A negative or NaN v is dropped rather than panicking: a counter that goes
// backwards makes every rate() over it produce a nonsense spike for the whole
// lookback window, and dropping one sample is a far smaller lie. Panicking on
// the hot path of an observability library, meanwhile, would take down the run
// the metrics exist to explain.
func (c *Counter) Add(v float64, labels ...Label) {
	if v < 0 || math.IsNaN(v) {
		return
	}
	addFloat(&c.v.get(labels).bits, v)
}

// Gauge is a value that goes up and down, one per label set.
type Gauge struct{ v *vec }

// Set replaces the value.
func (g *Gauge) Set(v float64, labels ...Label) {
	if math.IsNaN(v) {
		return
	}
	g.v.get(labels).bits.Store(math.Float64bits(v))
}

// Add adds v, which may be negative.
func (g *Gauge) Add(v float64, labels ...Label) {
	if math.IsNaN(v) {
		return
	}
	addFloat(&g.v.get(labels).bits, v)
}

// Histogram counts observations into buckets, one set per label set.
type Histogram struct{ v *vec }

// Observe records one value.
//
// A NaN is dropped — it belongs in no bucket and would poison _sum for the
// lifetime of the process. An infinity is counted in +Inf and, deliberately,
// still added to _sum, which is where an infinite _sum correctly advertises
// that something upstream produced an infinite duration.
func (h *Histogram) Observe(v float64, labels ...Label) {
	if math.IsNaN(v) {
		return
	}
	s := h.v.get(labels)
	// sort.SearchFloat64s gives the smallest i with buckets[i] >= v, which is
	// exactly the "less than or equal" bucket. len(buckets) means +Inf only.
	i := sort.SearchFloat64s(h.v.buckets, v)

	s.mu.Lock()
	s.counts[i]++
	s.sum += v
	s.count++
	s.mu.Unlock()
}

// addFloat is a compare-and-swap loop over a float64 stored as bits. Go has no
// atomic.Float64, and this is the standard substitute.
func addFloat(bits *atomic.Uint64, delta float64) {
	for {
		old := bits.Load()
		next := math.Float64bits(math.Float64frombits(old) + delta)
		if bits.CompareAndSwap(old, next) {
			return
		}
	}
}

// ===== labels =====

// canonicalize normalizes a label set and returns it with its map key.
//
// Normalization is what makes {a,b} and {b,a} the same series: names are
// sanitized, values truncated, empty names dropped, the whole thing sorted by
// name, and a repeated name reduced to its last occurrence — the exposition
// format forbids duplicates, and "last wins" matches how a caller appending an
// override to a slice expects it to behave.
func canonicalize(labels []Label) ([]Label, string) {
	if len(labels) == 0 {
		return nil, ""
	}

	out := make([]Label, 0, len(labels))
	for _, l := range labels {
		name := sanitizeLabelName(l.Key)
		if name == "" {
			continue // nothing left to call it by
		}
		out = append(out, Label{Key: name, Value: truncate(l.Value, MaxLabelValueBytes)})
	}
	if len(out) == 0 {
		return nil, ""
	}

	// Stable, so that among equal names the input order — and therefore "the
	// last one wins" — is preserved.
	slices.SortStableFunc(out, func(a, b Label) int { return strings.Compare(a.Key, b.Key) })
	kept := out[:0]
	for i, l := range out {
		if i+1 < len(out) && out[i+1].Key == l.Key {
			continue
		}
		kept = append(kept, l)
	}
	return kept, labelKey(kept)
}

// labelKey builds an injective map key. Length-prefixed rather than delimited,
// because any delimiter can also appear inside a label value taken from an
// error message, and a collision there would merge two unrelated series.
func labelKey(labels []Label) string {
	if len(labels) == 0 {
		return ""
	}
	var b strings.Builder
	for _, l := range labels {
		fmt.Fprintf(&b, "%d:%s%d:%s", len(l.Key), l.Key, len(l.Value), l.Value)
	}
	return b.String()
}

// sanitizeLabelName coerces s into [a-zA-Z_][a-zA-Z0-9_]*, returning "" if
// nothing is left.
//
// Coercion rather than a panic because label names can be assembled at runtime
// — from a tool's category, from a caller's own map — and emitting an
// unparseable exposition that breaks the whole scrape is a worse failure than
// showing a name with an underscore in it.
func sanitizeLabelName(s string) string {
	if s == "" {
		return ""
	}
	var b []byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c == '_':
		case c >= '0' && c <= '9' && i > 0:
		default:
			if b == nil {
				b = []byte(s)
			}
			b[i] = '_'
		}
	}
	if b == nil {
		return s
	}
	return string(b)
}

// validMetricName reports whether name is a legal Prometheus metric name.
func validMetricName(name string) bool {
	if name == "" {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c == '_', c == ':':
		case c >= '0' && c <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// truncate cuts s to at most n bytes on a rune boundary.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// normalizeBuckets sorts, de-duplicates and drops the boundaries a histogram
// cannot use. +Inf is dropped because it is always emitted anyway; keeping it
// would produce two +Inf lines and an invalid scrape.
func normalizeBuckets(bs []float64) []float64 {
	out := make([]float64, 0, len(bs))
	for _, b := range bs {
		if math.IsNaN(b) || math.IsInf(b, 1) {
			continue
		}
		out = append(out, b)
	}
	slices.Sort(out)
	return slices.Compact(out)
}
