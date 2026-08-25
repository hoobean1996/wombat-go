package builtin

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/tool"
)

func TestHTTPGet(t *testing.T) {
	var lastReq *http.Request
	mux := http.NewServeMux()
	mux.HandleFunc("/ok", func(w http.ResponseWriter, r *http.Request) {
		lastReq = r
		fmt.Fprint(w, "the body")
	})
	mux.HandleFunc("/empty", func(w http.ResponseWriter, r *http.Request) {})
	mux.HandleFunc("/notfound", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, "no such thing")
	})
	mux.HandleFunc("/huge", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, strings.Repeat("x", maxToolOutput*3))
	})
	mux.HandleFunc("/binary", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte{'o', 'k', 0xff, 0xfe})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	d := HTTPGet(srv.Client())

	t.Run("fetches a body", func(t *testing.T) {
		out, err := call(t, d, fmt.Sprintf(`{"url":%q}`, srv.URL+"/ok"))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if out != "the body" {
			t.Errorf("out = %q, want %q", out, "the body")
		}
		if got := lastReq.Header.Get("User-Agent"); got != userAgent {
			t.Errorf("User-Agent = %q, want %q", got, userAgent)
		}
		// Accept-Encoding is deliberately not set, so the transport negotiates
		// and decodes gzip transparently.
		if got := lastReq.Header.Get("Accept-Encoding"); !strings.Contains(got, "gzip") {
			t.Errorf("Accept-Encoding = %q, want the transport's own gzip negotiation", got)
		}
	})

	t.Run("an empty body is reported rather than returned blank", func(t *testing.T) {
		out, err := call(t, d, fmt.Sprintf(`{"url":%q}`, srv.URL+"/empty"))
		if err != nil || out != "(empty response body)" {
			t.Errorf("(%q, %v), want ((empty response body), nil)", out, err)
		}
	})

	// An API's error body is usually the thing that tells the model what it
	// did wrong, so it is carried into the error.
	t.Run("a non-2xx carries the status and the body", func(t *testing.T) {
		_, err := call(t, d, fmt.Sprintf(`{"url":%q}`, srv.URL+"/notfound"))
		if err == nil {
			t.Fatal("err = nil, want an HTTP status error")
		}
		var se *httpStatusError
		if !errors.As(err, &se) {
			t.Fatalf("err is %T, want *httpStatusError", err)
		}
		if se.code != http.StatusNotFound {
			t.Errorf("code = %d, want 404", se.code)
		}
		if !strings.Contains(err.Error(), "no such thing") {
			t.Errorf("err = %q, want the response body carried through", err)
		}
	})

	t.Run("a huge body is truncated", func(t *testing.T) {
		out, err := call(t, d, fmt.Sprintf(`{"url":%q}`, srv.URL+"/huge"))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if len(out) > maxToolOutput+clipMarkerMax {
			t.Errorf("len(out) = %d, want it capped near %d", len(out), maxToolOutput)
		}
		if !strings.Contains(out, "omitted") {
			t.Error("out does not mention truncation, want the marker")
		}
	})

	t.Run("a non-UTF-8 body is scrubbed", func(t *testing.T) {
		out, err := call(t, d, fmt.Sprintf(`{"url":%q}`, srv.URL+"/binary"))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if !strings.Contains(out, "�") {
			t.Errorf("out = %q, want the invalid bytes replaced", out)
		}
	})

	// Exotic schemes are refused by this tool rather than by whatever the
	// client happens to have registered, so the model gets an actionable
	// message.
	t.Run("rejects non-http(s) and host-less URLs", func(t *testing.T) {
		for _, u := range []string{
			"file:///etc/passwd",
			"ftp://example.com/x",
			"/just/a/path",
			"",
			"http://",
		} {
			out, err := call(t, d, fmt.Sprintf(`{"url":%q}`, u))
			if err == nil {
				t.Errorf("url %q = %q, want a refusal", u, out)
				continue
			}
			if !strings.Contains(err.Error(), "must be an absolute http(s) URL") {
				t.Errorf("url %q: err = %q, want the actionable message", u, err)
			}
		}
	})

	t.Run("a malformed URL is reported", func(t *testing.T) {
		_, err := call(t, d, `{"url":"http://[::1"}`)
		if err == nil || !strings.Contains(err.Error(), "not a valid URL") {
			t.Errorf("err = %v, want the parse failure", err)
		}
	})

	t.Run("a cancelled context stops the fetch", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := d.Fn(ctx, []byte(fmt.Sprintf(`{"url":%q}`, srv.URL+"/ok")))
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})

	t.Run("metadata", func(t *testing.T) {
		if !d.Has(tool.CapReadOnly | tool.CapNetwork) {
			t.Errorf("Caps = %b, want CapReadOnly|CapNetwork", d.Caps)
		}
		if d.Needs != tool.NeedNetwork {
			t.Errorf("Needs = %b, want NeedNetwork", d.Needs)
		}
		if !d.Idempotent || d.Retryable == nil {
			t.Error("want Idempotent with retryHTTP")
		}
		if d.Timeout != httpGetTimeout {
			t.Errorf("Timeout = %v, want %v to match the description", d.Timeout, httpGetTimeout)
		}
	})

	t.Run("a nil client panics at construction", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("HTTPGet(nil) did not panic, want a construction-time panic")
			}
		}()
		HTTPGet(nil)
	})
}

// retryHTTP is the one classifier that is not about errnos.
func TestRetryHTTP(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		// WithRetry sits OUTSIDE WithTimeout, so each attempt gets a fresh
		// deadline — which is why an overrun counts as transient. Both
		// spellings are matched.
		{"tool.ErrTimeout", tool.ErrTimeout, true},
		{"context.DeadlineExceeded", context.DeadlineExceeded, true},
		{"a wrapped deadline", fmt.Errorf("GET x: %w", context.DeadlineExceeded), true},

		// The caller has gone; there is nobody to retry for.
		{"context.Canceled", context.Canceled, false},
		{"a wrapped cancellation", fmt.Errorf("GET x: %w", context.Canceled), false},

		{"408", &httpStatusError{code: 408}, true},
		{"429", &httpStatusError{code: 429}, true},
		{"502", &httpStatusError{code: 502}, true},
		{"503", &httpStatusError{code: 503}, true},
		{"504", &httpStatusError{code: 504}, true},
		{"529 overloaded", &httpStatusError{code: 529}, true},

		{"400", &httpStatusError{code: 400}, false},
		{"401", &httpStatusError{code: 401}, false},
		{"404", &httpStatusError{code: 404}, false},
		{"500", &httpStatusError{code: 500}, false},
		{"a wrapped 404", fmt.Errorf("GET x: %w", &httpStatusError{code: 404}), false},

		// Anything the client itself reports arrives wrapped in *url.Error and
		// is transient by nature. A malformed URL never reaches c.Do.
		{"a transport failure", &url.Error{Op: "Get", URL: "http://x", Err: errors.New("connection refused")}, true},
		{"a wrapped transport failure", fmt.Errorf("GET x: %w", &url.Error{Err: errors.New("dns")}), true},

		{"an unrelated error", errors.New("something else"), false},
		{"nil", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := retryHTTP(tt.err); got != tt.want {
				t.Errorf("retryHTTP(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestHTTPStatusErrorMessage(t *testing.T) {
	if got := (&httpStatusError{code: 404}).Error(); got != "HTTP 404 Not Found" {
		t.Errorf("Error() = %q, want %q", got, "HTTP 404 Not Found")
	}
	if got := (&httpStatusError{code: 500, body: "oops"}).Error(); got != "HTTP 500 Internal Server Error:\noops" {
		t.Errorf("Error() = %q, want the body appended", got)
	}
}

// A url.Error from a real transport must classify as transient; the
// httptest server is closed first so nothing leaves the machine.
func TestRetryHTTPAgainstARealTransportFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	addr := srv.URL
	srv.Close()

	c := &http.Client{Timeout: time.Second}
	_, err := call(t, HTTPGet(c), fmt.Sprintf(`{"url":%q}`, addr))
	if err == nil {
		t.Fatal("err = nil, want a connection failure against a closed server")
	}
	if !retryHTTP(err) {
		t.Errorf("retryHTTP(%v) = false, want true for a transport failure", err)
	}
}
