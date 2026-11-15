// Package httpcall is the HTTP plumbing every provider client shares: the
// layered call deadlines, the stream idle watchdog, the decision whether a
// failed call was a stall or a cancellation, bounded body reads and the
// classification of a non-2xx reply.
//
// It exists because each client used to carry its own copy, and the copies
// drifted. The Anthropic client grew a total cap and an idle watchdog after a
// production wedge; the OpenAI client, written against the same failure modes,
// had neither, and a stream that stalled behind a gateway hung a run for as
// long as the caller's context allowed — forever, for a caller without a
// deadline. The two also disagreed on details nobody chose: one parsed a
// fractional Retry-After and the other read it as zero, one bounded an error
// body's message and the other quoted 64 KiB of HTML into it, one reset its
// watchdog on bytes and the other only on whole events. A fix made once here
// lands in every client.
package httpcall

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

// The layered call budget.
//
// The transport's ResponseHeaderTimeout bounds the wait for the first byte and
// nothing after it. These cover the rest:
//
//   - StreamIdleTimeout cancels a stream that delivers no bytes at all for that
//     long. Every byte resets it, keep-alive comments included, so a long
//     reasoning pause the server keeps alive is not mistaken for a dead socket.
//   - RequestTimeout is the total cap, applied ONLY when the caller's context
//     carries no deadline of its own. A caller that set one — governor's wall
//     clock included — always wins.
const (
	StreamIdleTimeout = 2 * time.Minute
	RequestTimeout    = 10 * time.Minute
)

// Body bounds.
//
// MaxErrorBody is how much of a failure body is read and MaxErrorMessage how
// much of it reaches the error string: an HTML error page from a gateway is
// megabytes of nothing, and the provider's own JSON error puts the useful part
// first, which is also where llm.ClassifyStatus looks for its markers.
// MaxResponseBody bounds a successful non-streamed reply; a real one is
// kilobytes, one with the maximum output and a long reasoning trace a few
// megabytes, and anything past this is a malfunction whose bytes would
// otherwise be the process's memory.
const (
	MaxErrorBody    = 64 << 10
	MaxErrorMessage = 2 << 10
	MaxResponseBody = 32 << 20
)

// ErrStreamIdle is the cancellation cause when a stream goes silent. Callers
// match llm.ErrTransport rather than this: a stall is what retry is for.
var ErrStreamIdle = errors.New("llm: stream went silent")

// Bound returns ctx with total applied as a deadline when ctx has none of its
// own, and ctx unchanged otherwise. The cancel func is always safe to call.
//
// Middleware and the governor express their intent as a context deadline;
// shortening one here would make those settings a lie.
func Bound(ctx context.Context, total time.Duration) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok || total <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, total)
}

// Watchdog cancels a streamed call whose body goes silent.
type Watchdog struct {
	cancel context.CancelCauseFunc
	idle   time.Duration
	timer  *time.Timer
}

// Watch derives the context a streamed call runs under, with a cause-cancel
// the watchdog fires when the body goes quiet for idle. The timer starts when
// [Watchdog.Body] wraps the response body; [Watchdog.Stop] stops it and
// releases the context, and has to be deferred.
//
// The context is derived before the request is built — the request carries
// it — even though silence only starts counting once the headers are in.
func Watch(ctx context.Context, idle time.Duration) (context.Context, *Watchdog) {
	ctx, cancel := context.WithCancelCause(ctx)
	return ctx, &Watchdog{cancel: cancel, idle: idle}
}

// Body wraps a response body so that every byte read from it resets the
// watchdog, and starts the timer.
//
// Bytes, not events. A server keeps a long pause alive with SSE comments,
// which the SSE reader discards before any event is produced; a watchdog reset
// only by events would kill exactly the streams that are working hardest.
func (w *Watchdog) Body(r io.Reader) io.Reader {
	if w.idle <= 0 {
		return r
	}
	w.timer = time.AfterFunc(w.idle, func() { w.cancel(ErrStreamIdle) })
	return &idleReader{r: r, w: w}
}

// Stop stops the timer and releases the derived context.
func (w *Watchdog) Stop() {
	if w.timer != nil {
		w.timer.Stop()
	}
	w.cancel(nil)
}

type idleReader struct {
	r io.Reader
	w *Watchdog
}

func (r *idleReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	if n > 0 {
		r.w.timer.Reset(r.w.idle)
	}
	return n, err
}

// Abort decides what a failed call was, from the context it ran under.
//
// The distinction matters to the retry middleware. A dead socket, a silent
// stream or the total cap expiring is a stall, worth another attempt, and is
// classified as transport. A context cancelled by the caller or the governor
// is the run being stopped on purpose: its cause is surfaced intact, so
// errors.Is(err, governor.ErrBudgetExhausted) — or context.Canceled — still
// answers, and nothing retries it.
//
// prefix names the provider in the error.
func Abort(ctx context.Context, prefix string, err error) error {
	cause := context.Cause(ctx)
	switch {
	case ctx.Err() == nil || cause == nil:
		return llm.ClassifyTransport(err)
	case errors.Is(cause, context.DeadlineExceeded), errors.Is(cause, ErrStreamIdle):
		return llm.ClassifyTransport(fmt.Errorf("%w: %w", cause, err))
	default:
		return fmt.Errorf("%s: call aborted: %w", prefix, cause)
	}
}

// Settle is the last word on a call whose body has been read. A failure while
// ctx is done happened BECAUSE ctx is done — the read was cut off under the
// decoder — and only Abort can say which cancellation it was; any other
// failure, and every success, is returned as it is.
func Settle(ctx context.Context, prefix string, err error) error {
	if err != nil && ctx.Err() != nil {
		return Abort(ctx, prefix, err)
	}
	return err
}

// ReadBody reads a successful non-streamed body of at most limit bytes.
//
// One byte past the limit is read, so a body of exactly the limit is not
// mistaken for an oversized one. An oversized body is a server error rather
// than a transport one: it is not the connection that failed, and retrying it
// fetches the same megabytes again.
func ReadBody(r io.Reader, prefix string, limit int) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(r, int64(limit)+1))
	if err != nil {
		return nil, llm.ClassifyTransport(fmt.Errorf("%s: read response: %w", prefix, err))
	}
	if len(raw) > limit {
		return nil, &llm.APIError{Class: llm.ErrServer, Status: http.StatusOK,
			Message: fmt.Sprintf("%s: response body exceeds %d bytes", prefix, limit)}
	}
	return raw, nil
}

// Cap bounds a streamed body at limit bytes, as [ReadBody] bounds a whole
// one: reading past it fails with the same server error. A stream has no
// length to check up front, and without a bound a gateway that never stops
// sending is memory that grows for as long as the total deadline allows.
func Cap(r io.Reader, prefix string, limit int) io.Reader {
	if limit <= 0 {
		return r
	}
	return &capped{r: r, left: int64(limit), prefix: prefix, limit: limit}
}

type capped struct {
	r      io.Reader
	left   int64
	prefix string
	limit  int
}

func (c *capped) Read(p []byte) (int, error) {
	if c.left <= 0 {
		// At the limit: only a byte past it makes the stream too long.
		var one [1]byte
		if n, err := c.r.Read(one[:]); n == 0 {
			return 0, err
		}
		return 0, &llm.APIError{Class: llm.ErrServer, Status: http.StatusOK,
			Message: fmt.Sprintf("%s: streamed response exceeds %d bytes", c.prefix, c.limit)}
	}
	if int64(len(p)) > c.left {
		p = p[:c.left]
	}
	n, err := c.r.Read(p)
	c.left -= int64(n)
	return n, err
}

// StatusError classifies a non-2xx reply, reading at most MaxErrorBody of it
// and quoting at most MaxErrorMessage, scrubbed to valid UTF-8.
func StatusError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, MaxErrorBody))
	msg := Truncate(Scrub(strings.TrimSpace(string(body))), MaxErrorMessage)
	return llm.ClassifyStatus(resp.StatusCode, msg, RetryAfter(resp.Header))
}

// RetryAfter reads the server's backoff hint: seconds, fractional ones
// included, or an HTTP date. Gateways disagree on the form, so both are read.
// Anything else, a past date, or a non-positive value is no hint at all.
func RetryAfter(h http.Header) time.Duration {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := strconv.ParseFloat(v, 64); err == nil {
		if !(secs > 0) { // also rejects NaN
			return 0
		}
		return time.Duration(secs * float64(time.Second))
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

// Truncate cuts s to at most n bytes, on a rune boundary, and says so.
func Truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && s[n]&0xC0 == 0x80 {
		n--
	}
	return s[:n] + "… (truncated)"
}

// Scrub replaces invalid UTF-8 with U+FFFD.
func Scrub(s string) string { return strings.ToValidUTF8(s, "�") }
