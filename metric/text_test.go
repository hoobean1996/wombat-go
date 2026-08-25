package metric_test

import (
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/metric"
)

func TestWriteTextGolden(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()

	c := r.Counter("wombat_test_total", "A test counter.")
	c.Add(2, metric.Label{Key: "b", Value: "2"}, metric.Label{Key: "a", Value: "1"})
	c.Inc(metric.Label{Key: "a", Value: "1"}, metric.Label{Key: "b", Value: "1"})

	h := r.Histogram("wombat_lat_seconds", "Latency.", []float64{1, 2})
	h.Observe(0.5)
	h.Observe(1.5)
	h.Observe(5)

	r.Gauge("wombat_g", "A gauge.").Set(-1.5)

	// Note what the shape asserts: names sorted, label sets sorted within a
	// name, labels sorted within a set, one HELP/TYPE pair per name, buckets
	// cumulative and ascending with +Inf last, _sum and _count after them.
	want := `# HELP wombat_g A gauge.
# TYPE wombat_g gauge
wombat_g -1.5
# HELP wombat_lat_seconds Latency.
# TYPE wombat_lat_seconds histogram
wombat_lat_seconds_bucket{le="1"} 1
wombat_lat_seconds_bucket{le="2"} 2
wombat_lat_seconds_bucket{le="+Inf"} 3
wombat_lat_seconds_sum 7
wombat_lat_seconds_count 3
# HELP wombat_test_total A test counter.
# TYPE wombat_test_total counter
wombat_test_total{a="1",b="1"} 1
wombat_test_total{a="1",b="2"} 2
`

	if got := text(t, r); got != want {
		t.Errorf("WriteText =\n%s\nwant\n%s", got, want)
	}
}

func TestWriteTextHistogramWithLabels(t *testing.T) {
	t.Parallel()

	// le rides last on the bucket lines, after the series' own labels, which
	// is the convention every existing exposition follows.
	r := metric.NewRegistry()
	h := r.Histogram("wombat_tool_duration_seconds", "Tool latency.", []float64{0.5})
	h.Observe(0.1, metric.Label{Key: "tool", Value: "Bash"})

	want := `# HELP wombat_tool_duration_seconds Tool latency.
# TYPE wombat_tool_duration_seconds histogram
wombat_tool_duration_seconds_bucket{tool="Bash",le="0.5"} 1
wombat_tool_duration_seconds_bucket{tool="Bash",le="+Inf"} 1
wombat_tool_duration_seconds_sum{tool="Bash"} 0.1
wombat_tool_duration_seconds_count{tool="Bash"} 1
`
	if got := text(t, r); got != want {
		t.Errorf("WriteText =\n%s\nwant\n%s", got, want)
	}
}

func TestWriteTextEscaping(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		value string
		want  string
	}{
		{"plain", `simple`, `simple`},
		{"backslash", `a\b`, `a\\b`},
		{"quote", `say "hi"`, `say \"hi\"`},
		{"newline", "line1\nline2", `line1\nline2`},
		{"all three", "a\\b\"c\nd", `a\\b\"c\nd`},
		{"carriage return is left alone", "a\rb", "a\rb"},
		{"tab is left alone", "a\tb", "a\tb"},
		{"utf-8 passes through", "héllo→", "héllo→"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			r.Counter("wombat_test_total", "help").Inc(metric.Label{Key: "v", Value: tc.value})

			want := "# HELP wombat_test_total help\n" +
				"# TYPE wombat_test_total counter\n" +
				`wombat_test_total{v="` + tc.want + `"} 1` + "\n"
			if got := text(t, r); got != want {
				t.Errorf("WriteText =\n%q\nwant\n%q", got, want)
			}
		})
	}
}

func TestWriteTextHelpEscaping(t *testing.T) {
	t.Parallel()

	// A HELP line is not quoted, so a quote inside it is harmless and stays
	// literal; a backslash or a newline would end or corrupt the line.
	r := metric.NewRegistry()
	r.Counter("wombat_test_total", "path C:\\x, said \"hi\"\nand more").Inc()

	want := `# HELP wombat_test_total path C:\\x, said "hi"\nand more
# TYPE wombat_test_total counter
wombat_test_total 1
`
	if got := text(t, r); got != want {
		t.Errorf("WriteText =\n%q\nwant\n%q", got, want)
	}
}

func TestWriteTextOmitsEmptyHelp(t *testing.T) {
	t.Parallel()

	// A blank HELP line is legal but pure noise; the TYPE line is what a
	// scraper actually needs.
	r := metric.NewRegistry()
	r.Counter("wombat_test_total", "").Inc()

	want := "# TYPE wombat_test_total counter\nwombat_test_total 1\n"
	if got := text(t, r); got != want {
		t.Errorf("WriteText =\n%q\nwant\n%q", got, want)
	}
}

func TestWriteTextIsDeterministic(t *testing.T) {
	t.Parallel()

	// Two scrapes of an unchanged registry must be byte-identical, or a
	// caller cannot checksum, cache or diff one. Map iteration order is
	// randomized per range in Go, so this catches an unsorted path quickly.
	r := metric.NewRegistry()
	c := r.Counter("wombat_test_total", "help")
	h := r.Histogram("wombat_lat_seconds", "help", metric.DefaultToolBuckets)
	g := r.Gauge("wombat_g", "help")

	for i := 0; i < 30; i++ {
		l := metric.Label{Key: "k", Value: string(rune('a' + i%26))}
		c.Inc(l, metric.Label{Key: "j", Value: string(rune('z' - i%26))})
		h.Observe(float64(i)/10, l)
		g.Set(float64(i), l)
	}

	first := text(t, r)
	for i := 0; i < 8; i++ {
		if got := text(t, r); got != first {
			t.Fatalf("scrape %d differs:\n%s\nvs\n%s", i, got, first)
		}
	}
	if first == "" {
		t.Fatal("empty exposition")
	}
}

func TestWriteTextNumberFormat(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		set  float64
		want string
	}{
		{"integer stays integral", 3, "3"},
		{"zero", 0, "0"},
		{"fraction", 0.005, "0.005"},
		{"negative", -1.25, "-1.25"},
		{"large uses exponent", 1e21, "1e+21"},
		{"tiny uses exponent", 1e-9, "1e-09"},
		{"positive infinity", math.Inf(1), "+Inf"},
		{"negative infinity", math.Inf(-1), "-Inf"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r := metric.NewRegistry()
			r.Gauge("wombat_g", "").Set(tc.set)

			want := "# TYPE wombat_g gauge\nwombat_g " + tc.want + "\n"
			if got := text(t, r); got != want {
				t.Errorf("WriteText = %q, want %q", got, want)
			}
		})
	}
}

func TestWriteTextEmptyRegistry(t *testing.T) {
	t.Parallel()

	if got := text(t, metric.NewRegistry()); got != "" {
		t.Errorf("WriteText = %q, want empty", got)
	}
}

// errWriter fails every write, standing in for a client that hung up
// mid-scrape.
type errWriter struct{ err error }

func (w errWriter) Write([]byte) (int, error) { return 0, w.err }

func TestWriteTextWrapsWriteError(t *testing.T) {
	t.Parallel()

	boom := errors.New("boom")
	r := metric.NewRegistry()
	r.Counter("wombat_test_total", "help").Inc()

	err := r.WriteText(errWriter{err: boom})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want it to wrap boom", err)
	}
	if !strings.HasPrefix(err.Error(), "metric: ") {
		t.Errorf("err = %q, want a metric: prefix", err)
	}
}

func TestHandler(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	r.Counter("wombat_test_total", "A test counter.").Inc(metric.Label{Key: "a", Value: "x"})

	srv := httptest.NewServer(r.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close() //nolint:errcheck // test

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	// The version token is what a Prometheus-compatible scraper negotiates on.
	if got := resp.Header.Get("Content-Type"); got != metric.ContentType {
		t.Errorf("Content-Type = %q, want %q", got, metric.ContentType)
	}
	if got := resp.Header.Get("Content-Type"); !strings.Contains(got, "version=0.0.4") {
		t.Errorf("Content-Type = %q, want it to name the exposition version", got)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if got, want := string(body), text(t, r); got != want {
		t.Errorf("body =\n%s\nwant\n%s", got, want)
	}
	if got, want := resp.ContentLength, int64(len(body)); got != want {
		t.Errorf("Content-Length = %d, want %d", got, want)
	}
}

func TestHandlerServesADeterministicBody(t *testing.T) {
	t.Parallel()

	r := metric.NewRegistry()
	c := r.Counter("wombat_test_total", "help")
	for i := 0; i < 10; i++ {
		c.Inc(metric.Label{Key: "k", Value: string(rune('a' + i))})
	}

	h := r.Handler()
	var first string
	for i := 0; i < 5; i++ {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
		body := rec.Body.String()
		if i == 0 {
			first = body
			continue
		}
		if body != first {
			t.Fatalf("scrape %d differs:\n%s\nvs\n%s", i, body, first)
		}
	}
}
