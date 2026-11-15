// Package providertest is the behaviour every provider client owes the
// harness at the HTTP boundary, as one suite each client runs against itself.
//
// It exists for the same reason llm/internal/httpcall does. The clients are
// written separately, against different wire formats, and every failure mode
// below was at some point handled by one client and not the other — the
// OpenAI client hung forever on a stalled stream that the Anthropic client
// cancelled after two minutes. A shared implementation removes one way for
// them to drift; a shared suite catches the rest, because a client that
// passes it handles stalls, cancellations, error bodies and oversized replies
// the way the retry middleware and the governor rely on, whatever its wire
// format.
package providertest

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/internal/httpcall"
)

// Limits are the client's call budget, shortened so a test of a timeout does
// not take two minutes.
type Limits struct {
	Idle    time.Duration
	Total   time.Duration
	MaxBody int
}

// Wire describes one provider's protocol well enough to fake it.
type Wire struct {
	// New builds a client against baseURL with l applied. The client must
	// stream when the request has an OnDelta sink and not otherwise.
	New func(t *testing.T, baseURL string, l Limits) llm.Client

	// StreamStart is the beginning of a valid streamed reply whose text so
	// far is "hel", and StreamFinish the rest of it, after which the reply's
	// text is "hello" and the stream is complete.
	StreamStart  string
	StreamFinish string

	// Reply is a complete non-streamed reply, and Partial a prefix of one that
	// is not yet valid.
	Reply   string
	Partial string
}

// defaults is a budget long enough never to fire in a test that is not about
// it.
var defaults = Limits{Idle: time.Minute, Total: time.Minute, MaxBody: httpcall.MaxResponseBody}

// Run runs the suite.
func Run(t *testing.T, w Wire) {
	t.Run("a stream that stalls is cancelled by the idle watchdog", func(t *testing.T) {
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, release <-chan struct{}) {
			startStream(rw, w.StreamStart)
			stall(r, release)
		})
		l := defaults
		l.Idle = 50 * time.Millisecond
		_, err := complete(t, w.New(t, srv.URL, l), context.Background(), streamed())
		if !errors.Is(err, llm.ErrTransport) {
			t.Fatalf("err = %v, want llm.ErrTransport so the retry middleware gets another attempt", err)
		}
		if !errors.Is(err, httpcall.ErrStreamIdle) {
			t.Errorf("err = %v, want it to say the stream went silent", err)
		}
	})

	t.Run("keep-alive comments hold the watchdog off", func(t *testing.T) {
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, _ <-chan struct{}) {
			startStream(rw, w.StreamStart)
			f := rw.(http.Flusher)
			for range 8 { // 8 x 20ms of comments, far past the 60ms watchdog
				time.Sleep(20 * time.Millisecond)
				fmt.Fprint(rw, ": keep-alive\n\n")
				f.Flush()
			}
			fmt.Fprint(rw, w.StreamFinish)
			f.Flush()
		})
		l := defaults
		l.Idle = 60 * time.Millisecond
		resp, err := complete(t, w.New(t, srv.URL, l), context.Background(), streamed())
		if err != nil {
			t.Fatalf("Complete: %v — keep-alive comments did not reset the watchdog", err)
		}
		if got := llm.TextOf(resp.Content); got != "hello" {
			t.Errorf("text = %q, want hello", got)
		}
	})

	t.Run("the total cap applies when the caller set no deadline", func(t *testing.T) {
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, release <-chan struct{}) {
			rw.Header().Set("Content-Type", "application/json")
			fmt.Fprint(rw, w.Partial)
			rw.(http.Flusher).Flush()
			stall(r, release)
		})
		l := defaults
		l.Total = 100 * time.Millisecond
		_, err := complete(t, w.New(t, srv.URL, l), context.Background(), plain())
		if !errors.Is(err, llm.ErrTransport) || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("err = %v, want a transport error carrying the deadline", err)
		}
	})

	t.Run("the caller's deadline wins over the total cap", func(t *testing.T) {
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, _ <-chan struct{}) {
			time.Sleep(150 * time.Millisecond)
			rw.Header().Set("Content-Type", "application/json")
			fmt.Fprint(rw, w.Reply)
		})
		l := defaults
		l.Total = 20 * time.Millisecond
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := complete(t, w.New(t, srv.URL, l), ctx, plain()); err != nil {
			t.Fatalf("Complete: %v — the client's cap overrode the caller's longer deadline", err)
		}
	})

	t.Run("a deliberate cancellation mid-stream keeps its cause", func(t *testing.T) {
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, release <-chan struct{}) {
			startStream(rw, w.StreamStart)
			stall(r, release)
		})
		stop := errors.New("budget exhausted")
		ctx, cancel := context.WithCancelCause(context.Background())
		req := streamed()
		req.OnDelta = func(llm.Delta) { cancel(stop) }
		_, err := complete(t, w.New(t, srv.URL, defaults), ctx, req)
		if !errors.Is(err, stop) {
			t.Errorf("err = %v, want it to carry the cancellation cause", err)
		}
		if llm.Retryable(err) {
			t.Errorf("err = %v is retryable; a deliberate stop must not be", err)
		}
	})

	t.Run("a context cancelled before the call is not a transport failure", func(t *testing.T) {
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, _ <-chan struct{}) {
			fmt.Fprint(rw, w.Reply)
		})
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := complete(t, w.New(t, srv.URL, defaults), ctx, plain())
		if !errors.Is(err, context.Canceled) || llm.Retryable(err) {
			t.Errorf("err = %v, want context.Canceled and not retryable", err)
		}
	})

	t.Run("a non-2xx reply is classified and carries the backoff hint", func(t *testing.T) {
		for _, tc := range []struct {
			status int
			after  string
			class  error
			want   time.Duration
		}{
			{429, "1.5", llm.ErrRateLimit, 1500 * time.Millisecond},
			{503, "2", llm.ErrServer, 2 * time.Second},
			{401, "", llm.ErrAuth, 0},
		} {
			srv := server(t, func(rw http.ResponseWriter, r *http.Request, _ <-chan struct{}) {
				if tc.after != "" {
					rw.Header().Set("Retry-After", tc.after)
				}
				rw.WriteHeader(tc.status)
				fmt.Fprint(rw, `{"error":{"message":"no"}}`)
			})
			_, err := complete(t, w.New(t, srv.URL, defaults), context.Background(), plain())
			var ae *llm.APIError
			if !errors.Is(err, tc.class) || !errors.As(err, &ae) {
				t.Errorf("%d: err = %v, want %v as an *llm.APIError", tc.status, err, tc.class)
				continue
			}
			if ae.RetryAfter != tc.want {
				t.Errorf("%d: RetryAfter = %v, want %v from %q", tc.status, ae.RetryAfter, tc.want, tc.after)
			}
		}
	})

	t.Run("an error body is bounded and scrubbed before it is quoted", func(t *testing.T) {
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, _ <-chan struct{}) {
			rw.WriteHeader(http.StatusBadGateway)
			fmt.Fprint(rw, "<html>\xff"+strings.Repeat("gateway noise ", 10_000))
		})
		_, err := complete(t, w.New(t, srv.URL, defaults), context.Background(), plain())
		var ae *llm.APIError
		if !errors.As(err, &ae) {
			t.Fatalf("err = %v, want an *llm.APIError", err)
		}
		if len(ae.Message) > httpcall.MaxErrorMessage+len("… (truncated)") {
			t.Errorf("Message is %d bytes, want at most about %d", len(ae.Message), httpcall.MaxErrorMessage)
		}
		if !utf8.ValidString(ae.Message) {
			t.Error("Message carries invalid UTF-8 from the body")
		}
	})

	t.Run("an oversized non-streamed body is refused, not read whole", func(t *testing.T) {
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, _ <-chan struct{}) {
			rw.Header().Set("Content-Type", "application/json")
			fmt.Fprint(rw, w.Reply)
		})
		l := defaults
		l.MaxBody = len(w.Reply) - 1
		if _, err := complete(t, w.New(t, srv.URL, l), context.Background(), plain()); err == nil || !strings.Contains(err.Error(), "exceeds") {
			t.Errorf("err = %v, want the size bound to refuse the body", err)
		}
		l.MaxBody = len(w.Reply)
		if _, err := complete(t, w.New(t, srv.URL, l), context.Background(), plain()); err != nil {
			t.Errorf("a body of exactly the limit: %v, want it accepted", err)
		}
	})

	t.Run("an oversized stream is refused, not accumulated without end", func(t *testing.T) {
		whole := w.StreamStart + w.StreamFinish
		srv := server(t, func(rw http.ResponseWriter, r *http.Request, _ <-chan struct{}) {
			startStream(rw, whole)
		})
		l := defaults
		l.MaxBody = len(whole) - 1
		_, err := complete(t, w.New(t, srv.URL, l), context.Background(), streamed())
		if err == nil || !strings.Contains(err.Error(), "exceeds") {
			t.Errorf("err = %v, want the size bound to end the stream", err)
		}
		if errors.Is(err, llm.ErrTransport) {
			t.Errorf("err = %v is a transport failure, which is retried: the same stream would be read again", err)
		}
		l.MaxBody = len(whole)
		if resp, err := complete(t, w.New(t, srv.URL, l), context.Background(), streamed()); err != nil || llm.TextOf(resp.Content) != "hello" {
			t.Errorf("a stream of exactly the limit: %q, %v; want it accepted", llm.TextOf(resp.Content), err)
		}
	})
}

func streamed() llm.Request {
	return llm.Request{Messages: []llm.Message{llm.UserText("hi")}, OnDelta: func(llm.Delta) {}}
}

func plain() llm.Request {
	return llm.Request{Messages: []llm.Message{llm.UserText("hi")}}
}

func startStream(rw http.ResponseWriter, start string) {
	rw.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprint(rw, start)
	rw.(http.Flusher).Flush()
}

// server starts h, handing it a channel closed when the test ends. A handler
// that stalls on purpose waits on it too: httptest's Close blocks until every
// request has finished, so a stall outliving a failed assertion would hang the
// test binary instead of failing the test.
func server(t *testing.T, h func(http.ResponseWriter, *http.Request, <-chan struct{})) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) { h(rw, r, release) }))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) }) // cleanups run LIFO: this one first
	return srv
}

// stall holds a request open until the client goes away or the test ends.
func stall(r *http.Request, release <-chan struct{}) {
	select {
	case <-r.Context().Done():
	case <-release:
	}
}

// complete fails the test if the call has not returned within ten seconds:
// the bugs this suite pins are hangs, and a hung test should say so.
func complete(t *testing.T, c llm.Client, ctx context.Context, req llm.Request) (llm.Response, error) {
	t.Helper()
	type result struct {
		resp llm.Response
		err  error
	}
	done := make(chan result, 1)
	go func() {
		resp, err := c.Complete(ctx, req)
		done <- result{resp, err}
	}()
	select {
	case r := <-done:
		return r.resp, r.err
	case <-time.After(10 * time.Second):
		t.Fatal("Complete did not return within 10s: the call hung")
		return llm.Response{}, nil
	}
}
