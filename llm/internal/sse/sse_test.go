// White-box (package sse): the package is internal, the surface under test is
// its whole exported API, and maxLine has to be reachable to check that a line
// under the cap still parses.
package sse

import (
	"bufio"
	"errors"
	"io"
	"strings"
	"testing"
)

// readAll drains a reader into a comparable slice of events.
func readAll(t *testing.T, body string) ([]Event, error) {
	t.Helper()
	return readAllFrom(t, strings.NewReader(body))
}

func readAllFrom(t *testing.T, r io.Reader) ([]Event, error) {
	t.Helper()
	rd := NewReader(r)
	var out []Event
	for rd.Next() {
		ev := rd.Event()
		// Copy: Event.Data aliases a slice the reader rebuilds each call, and a
		// test that compares events after the fact must not depend on that.
		data := append([]byte(nil), ev.Data...)
		out = append(out, Event{Type: ev.Type, Data: data})
	}
	return out, rd.Err()
}

// wantEvents compares by (type, string(data)) so failures read as text.
func wantEvents(t *testing.T, got []Event, want []Event) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("event count: got %d, want %d\ngot  %s\nwant %s", len(got), len(want), format(got), format(want))
	}
	for i := range got {
		if got[i].Type != want[i].Type || string(got[i].Data) != string(want[i].Data) {
			t.Errorf("event %d:\ngot  {Type:%q Data:%q}\nwant {Type:%q Data:%q}",
				i, got[i].Type, got[i].Data, want[i].Type, want[i].Data)
		}
	}
}

func format(evs []Event) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, e := range evs {
		if i > 0 {
			b.WriteString(" ")
		}
		b.WriteString("{" + e.Type + " " + string(e.Data) + "}")
	}
	b.WriteByte(']')
	return b.String()
}

func TestReader(t *testing.T) {
	tests := []struct {
		name string
		body string
		want []Event
	}{
		{
			name: "empty stream yields nothing",
			body: "",
		},
		{
			name: "only blank lines yield nothing",
			body: "\n\n\n",
		},
		{
			name: "a single data event",
			body: "data: hello\n\n",
			want: []Event{{Data: []byte("hello")}},
		},
		{
			name: "the event type is carried",
			body: "event: content_block_delta\ndata: {\"i\":0}\n\n",
			want: []Event{{Type: "content_block_delta", Data: []byte(`{"i":0}`)}},
		},
		{
			name: "two events",
			body: "data: one\n\ndata: two\n\n",
			want: []Event{{Data: []byte("one")}, {Data: []byte("two")}},
		},
		{
			// This is the framing rule that matters: consecutive data fields are
			// one payload joined with \n, not two events.
			name: "multi-line data is joined with a newline",
			body: "data: line one\ndata: line two\ndata: line three\n\n",
			want: []Event{{Data: []byte("line one\nline two\nline three")}},
		},
		{
			name: "an interior empty data line becomes a blank line in the payload",
			body: "data: a\ndata: \ndata: b\n\n",
			want: []Event{{Data: []byte("a\n\nb")}},
		},
		{
			name: "multi-line data with a type",
			body: "event: ping\ndata: a\ndata: b\n\n",
			want: []Event{{Type: "ping", Data: []byte("a\nb")}},
		},
		{
			name: "comments are skipped",
			body: ": this is a comment\ndata: hello\n\n",
			want: []Event{{Data: []byte("hello")}},
		},
		{
			// Providers send bare ":" or ": keep-alive" to hold the connection
			// open. They must not dispatch an empty event.
			name: "keep-alives alone dispatch nothing",
			body: ":\n\n: keep-alive\n\n:ping\n\n",
		},
		{
			name: "a keep-alive between events does not split them",
			body: "data: a\n: keep-alive\ndata: b\n\n",
			want: []Event{{Data: []byte("a\nb")}},
		},
		{
			name: "CRLF line endings",
			body: "event: message\r\ndata: hello\r\n\r\n",
			want: []Event{{Type: "message", Data: []byte("hello")}},
		},
		{
			name: "CRLF with multi-line data",
			body: "data: a\r\ndata: b\r\n\r\ndata: c\r\n\r\n",
			want: []Event{{Data: []byte("a\nb")}, {Data: []byte("c")}},
		},
		{
			name: "a CRLF comment is still a comment",
			body: ": keep-alive\r\ndata: x\r\n\r\n",
			want: []Event{{Data: []byte("x")}},
		},
		{
			// Per the spec a line with no colon is a field with an empty value.
			name: "a field with no colon is not mistaken for data",
			body: "event\ndata: payload\n\n",
			want: []Event{{Type: "", Data: []byte("payload")}},
		},
		{
			name: "an unknown field with no colon is ignored",
			body: "retry\ndata: payload\n\n",
			want: []Event{{Data: []byte("payload")}},
		},
		{
			// An abrupt close is normal at the end of a provider stream; the
			// accumulated event must not be dropped.
			name: "an event with no trailing blank line at EOF is still dispatched",
			body: "event: message_stop\ndata: {\"type\":\"message_stop\"}",
			want: []Event{{Type: "message_stop", Data: []byte(`{"type":"message_stop"}`)}},
		},
		{
			name: "a final event with only a newline and no blank line",
			body: "data: a\n\ndata: last\n",
			want: []Event{{Data: []byte("a")}, {Data: []byte("last")}},
		},
		{
			name: "id and retry fields are ignored",
			body: "id: 42\nretry: 3000\ndata: hello\n\n",
			want: []Event{{Data: []byte("hello")}},
		},
		{
			name: "an unknown field alone dispatches nothing",
			body: "id: 42\nretry: 3000\n\n",
		},
		{
			name: "no space after the colon",
			body: "data:hello\n\n",
			want: []Event{{Data: []byte("hello")}},
		},
		{
			name: "only ONE leading space is stripped",
			body: "data:  indented\n\n",
			want: []Event{{Data: []byte(" indented")}},
		},
		{
			name: "a colon inside the value is preserved",
			body: "data: {\"a\": \"b:c\"}\n\n",
			want: []Event{{Data: []byte(`{"a": "b:c"}`)}},
		},
		{
			name: "the [DONE] sentinel is just data for the caller to notice",
			body: "data: {\"x\":1}\n\ndata: [DONE]\n\n",
			want: []Event{{Data: []byte(`{"x":1}`)}, {Data: []byte("[DONE]")}},
		},
		{
			name: "extra blank lines between events are harmless",
			body: "data: a\n\n\n\ndata: b\n\n",
			want: []Event{{Data: []byte("a")}, {Data: []byte("b")}},
		},
		{
			// The last "event:" field wins, which is what a spec-compliant
			// producer would never send but a gateway might.
			name: "a repeated event field takes the last value",
			body: "event: first\nevent: second\ndata: x\n\n",
			want: []Event{{Type: "second", Data: []byte("x")}},
		},
		{
			name: "an event field with no data still dispatches",
			body: "event: ping\n\n",
			want: []Event{{Type: "ping", Data: nil}},
		},
		{
			name: "state does not leak from one event to the next",
			body: "event: a\ndata: 1\n\ndata: 2\n\n",
			want: []Event{{Type: "a", Data: []byte("1")}, {Type: "", Data: []byte("2")}},
		},
		{
			name: "a full anthropic-shaped exchange",
			body: "event: message_start\ndata: {\"type\":\"message_start\"}\n\n" +
				": ping\n\n" +
				"event: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n" +
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
			want: []Event{
				{Type: "message_start", Data: []byte(`{"type":"message_start"}`)},
				{Type: "content_block_delta", Data: []byte(`{"delta":{"text":"hi"}}`)},
				{Type: "message_stop", Data: []byte(`{"type":"message_stop"}`)},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := readAll(t, tt.body)
			if err != nil {
				t.Fatalf("Err(): got %v, want nil", err)
			}
			wantEvents(t, got, tt.want)
		})
	}
}

func TestReaderLineLongerThanTheInitialBufferButUnderTheCap(t *testing.T) {
	// NewReader starts with a 64 KiB buffer and allows growth to maxLine. A
	// provider payload carrying a whole content block routinely exceeds 64 KiB,
	// and if the Scanner were not given room to grow this would surface as a
	// truncated stream rather than an error.
	const size = 512 << 10 // 512 KiB: 8x the initial buffer, well under maxLine
	if size <= 64<<10 {
		t.Fatalf("test payload %d is not larger than the initial buffer", size)
	}
	if size >= maxLine {
		t.Fatalf("test payload %d is not under the %d cap", size, maxLine)
	}
	payload := strings.Repeat("x", size)

	got, err := readAll(t, "event: big\ndata: "+payload+"\n\n")
	if err != nil {
		t.Fatalf("Err(): got %v, want nil", err)
	}
	if len(got) != 1 {
		t.Fatalf("event count: got %d, want 1", len(got))
	}
	if got[0].Type != "big" {
		t.Errorf("Type: got %q, want %q", got[0].Type, "big")
	}
	if len(got[0].Data) != size {
		t.Errorf("Data length: got %d, want %d", len(got[0].Data), size)
	}
	if string(got[0].Data) != payload {
		t.Error("Data: the payload was corrupted")
	}
}

func TestReaderRejectsALineOverTheCap(t *testing.T) {
	// The cap is what keeps a runaway or hostile stream from being unbounded.
	oversized := strings.Repeat("x", maxLine+1024)
	_, err := readAll(t, "data: "+oversized+"\n\n")
	if !errors.Is(err, bufio.ErrTooLong) {
		t.Errorf("Err(): got %v, want bufio.ErrTooLong", err)
	}
}

// errReader yields some bytes and then fails, the way a dropped connection does.
type errReader struct {
	data []byte
	err  error
	off  int
}

func (r *errReader) Read(p []byte) (int, error) {
	if r.off >= len(r.data) {
		return 0, r.err
	}
	n := copy(p, r.data[r.off:])
	r.off += n
	return n, nil
}

func TestReaderSurfacesReadFailures(t *testing.T) {
	boom := errors.New("connection reset by peer")
	rd := NewReader(&errReader{data: []byte("data: one\n\ndata: partial"), err: boom})

	if !rd.Next() {
		t.Fatal("Next(): got false on the first complete event, want true")
	}
	if got := string(rd.Event().Data); got != "one" {
		t.Errorf("first event: got %q, want %q", got, "one")
	}
	// The partial trailing event must be lost to the error, not silently
	// dispatched as if the stream had ended cleanly.
	if rd.Next() {
		t.Errorf("Next(): got true after a read failure, want false (event %q)", rd.Event().Data)
	}
	if !errors.Is(rd.Err(), boom) {
		t.Errorf("Err(): got %v, want %v", rd.Err(), boom)
	}
}

func TestReaderErrIsNilAtCleanEOF(t *testing.T) {
	rd := NewReader(strings.NewReader("data: x\n\n"))
	for rd.Next() {
	}
	if err := rd.Err(); err != nil {
		t.Errorf("Err(): got %v, want nil at a clean end of stream", err)
	}
}

func TestReaderNextIsIdempotentAtEOF(t *testing.T) {
	// A caller that keeps polling after the stream ends must not get a repeat
	// of the last event.
	rd := NewReader(strings.NewReader("data: only"))
	if !rd.Next() {
		t.Fatal("Next(): got false, want true for the trailing event")
	}
	for i := range 3 {
		if rd.Next() {
			t.Fatalf("Next() call %d after EOF: got true, want false", i+2)
		}
		if rd.Err() != nil {
			t.Fatalf("Err(): got %v, want nil", rd.Err())
		}
	}
}

func TestReaderEventIsStableUntilTheNextAdvance(t *testing.T) {
	rd := NewReader(strings.NewReader("data: a\n\ndata: b\n\n"))
	if !rd.Next() {
		t.Fatal("Next(): got false, want true")
	}
	first := rd.Event()
	if string(first.Data) != "a" {
		t.Fatalf("first: got %q, want %q", first.Data, "a")
	}
	// Calling Event() twice must not consume anything.
	if string(rd.Event().Data) != "a" {
		t.Errorf("Event() is not idempotent: got %q, want %q", rd.Event().Data, "a")
	}
	if !rd.Next() {
		t.Fatal("Next(): got false, want true for the second event")
	}
	if string(rd.Event().Data) != "b" {
		t.Errorf("second: got %q, want %q", rd.Event().Data, "b")
	}
}

// TestReaderLeadingEmptyDataLine documents a spec deviation that needs a
// production change to fix, so it is skipped rather than asserted away.
//
// Next() decides whether to insert the joining "\n" by testing `data != nil`,
// but appending a zero-length value to a nil slice leaves it nil. So a payload
// whose FIRST data field is empty loses its leading newline:
//
//	data:
//	data: b
//
// should accumulate to "\nb" and instead yields "b". Interior empty data lines
// are fine (see the table above) because the buffer is non-nil by then.
//
// Impact today is nil — neither provider client emits a leading empty data
// field — but the fix is one line (track "seen a data field" in its own bool)
// and the current behaviour is silently wrong rather than loud.
func TestReaderLeadingEmptyDataLine(t *testing.T) {

	tests := []struct {
		name string
		body string
		want string
	}{
		{"leading empty data field", "data:\ndata: b\n\n", "\nb"},
		{"leading empty data field with a space", "data: \ndata: b\n\n", "\nb"},
		{"a data field with no colon at all", "data\ndata: b\n\n", "\nb"},
		{"a lone empty data field", "data:\n\n", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := readAll(t, tt.body)
			if err != nil {
				t.Fatalf("Err(): got %v, want nil", err)
			}
			if len(got) != 1 {
				t.Fatalf("event count: got %d, want 1", len(got))
			}
			if string(got[0].Data) != tt.want {
				t.Errorf("Data: got %q, want %q", got[0].Data, tt.want)
			}
		})
	}
}
