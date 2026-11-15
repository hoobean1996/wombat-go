package metric

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
)

// ContentType is the media type of the exposition [Registry.WriteText]
// produces, and what [Registry.Handler] sets.
//
// version=0.0.4 names the text exposition format, not this library. Every
// Prometheus-compatible scraper — Prometheus, VictoriaMetrics, the OTel
// collector's prometheus receiver, vmagent, telegraf — negotiates on it.
const ContentType = "text/plain; version=0.0.4; charset=utf-8"

// WriteText writes the whole registry in the Prometheus text exposition
// format.
//
// The output is deterministic: series are sorted by name and then by label
// set, labels within a series by name, buckets ascending. Two calls against an
// unchanged registry produce byte-identical output, which is what lets a test
// diff them and a caller cache or checksum a scrape.
//
// An instrument that has never been touched contributes nothing — not even a
// HELP line — so that this and [Registry.Snapshot] describe exactly the same
// set of series.
func (r *Registry) WriteText(w io.Writer) error {
	all := r.Snapshot()

	// Rendered into a buffer and written once. A partially written exposition
	// is not a truncated document, it is an invalid one, and building it in
	// memory also means the per-line error checking that would otherwise
	// clutter every branch below collapses into a single check here.
	var buf bytes.Buffer
	var lastName string
	for _, s := range all {
		if s.Name != lastName {
			if s.Help != "" {
				buf.WriteString("# HELP ")
				buf.WriteString(s.Name)
				buf.WriteByte(' ')
				buf.WriteString(escapeHelp(s.Help))
				buf.WriteByte('\n')
			}
			buf.WriteString("# TYPE ")
			buf.WriteString(s.Name)
			buf.WriteByte(' ')
			buf.WriteString(s.Type)
			buf.WriteByte('\n')
			lastName = s.Name
		}
		writeSeries(&buf, s)
	}

	if _, err := w.Write(buf.Bytes()); err != nil {
		return fmt.Errorf("metric: write exposition: %w", err)
	}
	return nil
}

func writeSeries(buf *bytes.Buffer, s Series) {
	if s.Type != typeHistogram {
		buf.WriteString(s.Name)
		writeLabels(buf, s.Labels, Label{})
		buf.WriteByte(' ')
		buf.WriteString(formatFloat(s.Value))
		buf.WriteByte('\n')
		return
	}

	// A histogram is three metric names sharing one series' labels, and the
	// bucket lines carry an extra "le" label. le is appended last rather than
	// merged and re-sorted because the format's own convention puts it last,
	// and every existing exposition a reader has seen looks that way.
	for _, b := range s.Buckets {
		buf.WriteString(s.Name)
		buf.WriteString("_bucket")
		writeLabels(buf, s.Labels, Label{Key: "le", Value: formatFloat(b.LE)})
		buf.WriteByte(' ')
		buf.WriteString(strconv.FormatUint(b.Count, 10))
		buf.WriteByte('\n')
	}
	buf.WriteString(s.Name)
	buf.WriteString("_sum")
	writeLabels(buf, s.Labels, Label{})
	buf.WriteByte(' ')
	buf.WriteString(formatFloat(s.Sum))
	buf.WriteByte('\n')

	buf.WriteString(s.Name)
	buf.WriteString("_count")
	writeLabels(buf, s.Labels, Label{})
	buf.WriteByte(' ')
	buf.WriteString(strconv.FormatUint(s.Count, 10))
	buf.WriteByte('\n')
}

// writeLabels renders {k="v",...}, with extra appended when its key is set.
func writeLabels(buf *bytes.Buffer, labels []Label, extra Label) {
	if len(labels) == 0 && extra.Key == "" {
		return
	}
	buf.WriteByte('{')
	for i, l := range labels {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(l.Key)
		buf.WriteString(`="`)
		buf.WriteString(escapeLabelValue(l.Value))
		buf.WriteByte('"')
	}
	if extra.Key != "" {
		if len(labels) > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(extra.Key)
		buf.WriteString(`="`)
		buf.WriteString(escapeLabelValue(extra.Value))
		buf.WriteByte('"')
	}
	buf.WriteByte('}')
}

// labelValueEscaper handles the three characters the exposition format
// escapes inside a quoted label value. Nothing else is touched — a UTF-8
// label value goes out as UTF-8, which is what the format specifies.
var labelValueEscaper = strings.NewReplacer(
	`\`, `\\`,
	`"`, `\"`,
	"\n", `\n`,
)

func escapeLabelValue(s string) string { return labelValueEscaper.Replace(s) }

// helpEscaper handles what a HELP line escapes. A quote is NOT escaped here:
// help text is not quoted, so a backslash and a newline are the only two
// characters that could end the line early or be misread.
var helpEscaper = strings.NewReplacer(
	`\`, `\\`,
	"\n", `\n`,
)

func escapeHelp(s string) string { return helpEscaper.Replace(s) }

// formatFloat renders a sample value.
//
// 'g' with -1 precision gives the shortest representation that round-trips,
// so 1 stays "1" instead of becoming "1.000000" and a 0.005 bucket bound
// prints as "0.005". The three non-finite values have literal spellings in the
// format and no numeric one.
func formatFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "+Inf"
	case math.IsInf(f, -1):
		return "-Inf"
	case math.IsNaN(f):
		return "NaN"
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// Handler serves [Registry.WriteText] over HTTP. Mount it at /metrics.
//
//	http.Handle("/metrics", reg.Handler())
//
// The body is rendered before any header is written, so a failure becomes a
// clean 500 rather than a 200 carrying half an exposition — which a scraper
// would happily ingest as "these series no longer exist".
func (r *Registry) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		var buf bytes.Buffer
		if err := r.WriteText(&buf); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		h := w.Header()
		h.Set("Content-Type", ContentType)
		h.Set("Content-Length", strconv.Itoa(buf.Len()))
		_, _ = w.Write(buf.Bytes())
	})
}
