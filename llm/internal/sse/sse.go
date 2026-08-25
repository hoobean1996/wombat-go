// Package sse reads a text/event-stream body.
//
// Both provider clients need the same framing, and neither needs anything the
// standard library does not already give: a bufio.Scanner over the response
// body, with the caller's context handling cancellation and idle timeouts
// handled by the http.Client. There is no hand-rolled select loop here
// because there is nothing left for one to do.
package sse

import (
	"bufio"
	"bytes"
	"io"
)

// Event is one dispatched server-sent event.
type Event struct {
	// Type is the "event:" field, empty when the stream omits it.
	Type string
	// Data is the accumulated "data:" payload, newline-joined.
	Data []byte
}

// Reader iterates events. Use it like bufio.Scanner:
//
//	r := sse.NewReader(resp.Body)
//	for r.Next() { handle(r.Event()) }
//	if err := r.Err(); err != nil { ... }
type Reader struct {
	sc  *bufio.Scanner
	cur Event
	err error
}

// maxLine bounds a single SSE line. Provider payloads carrying a whole content
// block can be large; 8 MiB is generous without being unbounded.
const maxLine = 8 << 20

// NewReader wraps an event-stream body.
func NewReader(r io.Reader) *Reader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), maxLine)
	return &Reader{sc: sc}
}

// Next advances to the next complete event.
func (r *Reader) Next() bool {
	var (
		typ     string
		data    []byte
		sawData bool
		any     bool
	)

	for r.sc.Scan() {
		line := bytes.TrimSuffix(r.sc.Bytes(), []byte("\r"))

		// A blank line dispatches whatever has accumulated.
		if len(line) == 0 {
			if any {
				r.cur = Event{Type: typ, Data: data}
				return true
			}
			continue
		}

		// Comments and keep-alives.
		if line[0] == ':' {
			continue
		}

		field, value, found := bytes.Cut(line, []byte(":"))
		if !found {
			field, value = line, nil
		}
		value = bytes.TrimPrefix(value, []byte(" "))

		switch string(field) {
		case "event":
			typ, any = string(value), true
		case "data":
			// sawData, not data != nil: appending an empty value to a nil
			// slice leaves it nil, so a stream whose FIRST data field is empty
			// would silently lose the newline that separates it from the next.
			if sawData {
				data = append(data, '\n')
			}
			data = append(data, value...)
			sawData, any = true, true
		default:
			// id, retry and unknown fields are not used here.
		}
	}

	if err := r.sc.Err(); err != nil {
		r.err = err
		return false
	}
	// A stream that ends without a trailing blank line still has a final event.
	if any {
		r.cur = Event{Type: typ, Data: data}
		return true
	}
	return false
}

// Event returns the most recent event.
func (r *Reader) Event() Event { return r.cur }

// Err reports a read failure, or nil at clean end of stream.
func (r *Reader) Err() error { return r.err }
