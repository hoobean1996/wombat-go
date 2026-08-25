package trace

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Sink receives finished spans.
//
// One method, no error return, no context: a tracer that can fail is a tracer
// callers have to handle failures from, in the middle of code whose actual job
// is something else. A sink that cannot write drops the span. Implementations
// must be safe for concurrent use — sub-agents on separate goroutines share
// one sink.
type Sink interface {
	Emit(Span)
}

type discardSink struct{}

func (discardSink) Emit(Span) {}

// Discard is the sink that throws spans away.
var Discard Sink = discardSink{}

// FileSink appends spans to path as NDJSON — one span per line, written when
// the span ends.
//
// NDJSON and not a JSON array because a run that is killed, or that is still
// going, must still leave a readable file: every complete line is a complete
// span, and [ReadFile] tolerates a torn last one. Parent directories are
// created, since the common paths ("/tmp/wombat/run-3/trace.ndjson", a
// per-run directory a UI just invented) do not exist yet.
//
// The returned io.Closer must be closed; spans emitted afterwards are dropped
// rather than panicking on a closed file, because a stray late span from a
// goroutine that outlived its run is not worth crashing a process over.
func FileSink(path string) (Sink, io.Closer, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, nil, fmt.Errorf("trace: create trace directory: %w", err)
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, nil, fmt.Errorf("trace: open trace file: %w", err)
	}
	s := &fileSink{f: f}
	return s, s, nil
}

type fileSink struct {
	mu     sync.Mutex
	f      *os.File
	closed bool
}

// Emit implements Sink. The line is built first and written with one call so
// that a span is never interleaved with another one, and O_APPEND then makes
// concurrent writers to the same path safe at the OS level too.
func (s *fileSink) Emit(sp Span) {
	line, err := json.Marshal(sp)
	if err != nil {
		return // Attr.MarshalJSON already degrades gracefully; nothing left to do
	}
	line = append(line, '\n')

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	// The error is deliberately unexamined: a full disk or a closed pipe must
	// not take down the run whose failure this file exists to explain.
	_, _ = s.f.Write(line)
}

// Close implements io.Closer.
func (s *fileSink) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if err := s.f.Close(); err != nil {
		return fmt.Errorf("trace: close trace file: %w", err)
	}
	return nil
}

// MultiSink fans each span out to every sink, in order.
//
// The motivating pair is a file for the report plus a live sink for a UI — or,
// during a migration, this package's file and an OTel bridge side by side.
func MultiSink(sinks ...Sink) Sink {
	kept := make([]Sink, 0, len(sinks))
	for _, s := range sinks {
		if s != nil && s != Discard {
			kept = append(kept, s)
		}
	}
	switch len(kept) {
	case 0:
		return Discard
	case 1:
		return kept[0]
	}
	return multiSink(kept)
}

type multiSink []Sink

// Emit implements Sink.
func (m multiSink) Emit(sp Span) {
	for _, s := range m {
		s.Emit(sp)
	}
}

// ReadFile loads an NDJSON trace written by [FileSink].
//
// A malformed final line is dropped rather than reported: it is the signature
// of a process killed mid-write, which is precisely the run whose trace you
// most want to look at. A malformed line anywhere else is a real corruption
// and is returned as an error naming the line number.
func ReadFile(path string) ([]Span, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("trace: open trace file: %w", err)
	}
	defer f.Close() //nolint:errcheck // read-only

	var out []Span
	br := bufio.NewReader(f)
	for n := 1; ; n++ {
		line, readErr := br.ReadString('\n')
		// No trailing newline means the writer stopped here.
		torn := errors.Is(readErr, io.EOF) && line != ""

		if text := strings.TrimSpace(line); text != "" {
			var sp Span
			if err := json.Unmarshal([]byte(text), &sp); err != nil {
				if torn {
					break
				}
				return nil, fmt.Errorf("trace: %s:%d: %w", path, n, err)
			}
			out = append(out, sp)
		}

		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return nil, fmt.Errorf("trace: read %s: %w", path, readErr)
		}
	}
	return out, nil
}
