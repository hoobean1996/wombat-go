package trace

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// viewerHTML is the report template. Embedded rather than fetched, generated
// or templated at runtime, because the artifact this package promises is one
// file that opens from a ticket attachment on a machine with no network and no
// wombat binary.
//
//go:embed viewer.html
var viewerHTML string

// dataMarker is where the span array is spliced in.
const dataMarker = "__WOMBAT_TRACE_DATA__"

// viewerHead and viewerTail split the template once, at init. A template
// missing its marker is a corrupt binary rather than a runtime condition, so
// it panics here — at program start, with a message naming the cause — instead
// of failing inside a report that somebody is trying to file a bug with.
var viewerHead, viewerTail = func() (string, string) {
	i := strings.Index(viewerHTML, dataMarker)
	if i < 0 {
		panic("trace: embedded viewer.html has no " + dataMarker + " marker")
	}
	return viewerHTML[:i], viewerHTML[i+len(dataMarker):]
}()

// WriteHTML renders spans as one self-contained HTML report: no CDN, no
// sidecar JSON, no server. The whole file is the template plus the spans as a
// JSON array.
//
// Order is preserved as given. Spans arrive in completion order from a
// [Sink] — a parent lands after its children — and the report re-nests them by
// parent id, so no sorting is required of the caller.
//
//	spans, err := trace.ReadFile("trace.ndjson")
//	if err != nil { return err }
//	f, err := os.Create("trace.html")
//	if err != nil { return err }
//	defer f.Close()
//	return trace.WriteHTML(f, spans)
func WriteHTML(w io.Writer, spans []Span) error {
	if spans == nil {
		spans = []Span{} // "null" would make the viewer's empty-file path harder
	}

	// json.Marshal, not an encoder with SetEscapeHTML(false): HTML escaping is
	// exactly what is wanted here. It turns any "</script>" inside a name or
	// an attribute into an inert escape, which is the whole safety argument for
	// injecting into a script element. This package therefore wants the
	// opposite default from the event stream, which turns escaping off so
	// non-browser consumers read model output verbatim.
	data, err := json.Marshal(spans)
	if err != nil {
		return fmt.Errorf("trace: encode spans for report: %w", err)
	}

	for _, part := range []string{viewerHead, string(data), viewerTail} {
		if _, err := io.WriteString(w, part); err != nil {
			return fmt.Errorf("trace: write report: %w", err)
		}
	}
	return nil
}
