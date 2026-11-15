package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"wombat/pkg/core/tool"
)

const (
	// httpGetTimeout matches the "~15s" the description promises.
	httpGetTimeout = 15 * time.Second

	// maxFetchBytes bounds what is read off the wire before truncation. It is
	// well above maxToolOutput because the truncation note should report a
	// meaningful "how much did I miss", but finite because an endpoint that
	// streams forever must not be able to grow the heap without bound.
	maxFetchBytes = 1 << 20

	userAgent = "wombat/0.1"
)

// httpStatusError is a non-2xx response. It keeps the code separate from the
// message so [retryHTTP] can classify without matching on substrings, and it
// carries the body because an API's error body is usually the thing that
// tells the model what it did wrong.
type httpStatusError struct {
	code int
	body string
}

func (e *httpStatusError) Error() string {
	if e.body == "" {
		return fmt.Sprintf("HTTP %d %s", e.code, http.StatusText(e.code))
	}
	return fmt.Sprintf("HTTP %d %s:\n%s", e.code, http.StatusText(e.code), e.body)
}

type httpGetIn struct {
	URL string `json:"url"`
}

// HTTPGet fetches a URL and returns the body as text.
//
// net/http rather than a curl subprocess: the context cancels the request for
// free, redirects and gzip are handled by the transport, and there is no
// shell string to quote a model-supplied URL into. The OCaml had to pass
// --compressed explicitly to stop servers returning binary that broke the
// next LLM call; Go's transport negotiates and decodes gzip transparently
// whenever the caller does not set Accept-Encoding, so we do not set it.
func HTTPGet(c *http.Client) tool.Def {
	mustNotBeNil(c != nil, "HTTPGet requires a non-nil *http.Client")

	return tool.Typed(tool.Def{
		Name: "http_get",
		Description: "Fetch a URL via HTTP GET and return the response body as text. " +
			"Truncated to ~8000 chars. Useful for reading web pages or API endpoints.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "url": {
      "type": "string",
      "description": "Absolute URL to fetch (https://...)"
    }
  },
  "required": ["url"]
}`),
		Caps:       tool.CapReadOnly | tool.CapNetwork,
		Needs:      tool.NeedNetwork,
		Idempotent: true,
		Timeout:    httpGetTimeout,
		Category:   "network",
		Retryable:  retryHTTP,
	}, func(ctx context.Context, in httpGetIn) (string, error) {
		u, err := url.Parse(strings.TrimSpace(in.URL))
		if err != nil {
			return "", tool.CallerError(fmt.Errorf("field 'url' is not a valid URL: %w", err))
		}
		// Scheme and host are checked here rather than left to the transport
		// so the model gets an actionable message, and so that file:// and
		// other exotic schemes are refused by this tool rather than by
		// whatever the client happens to have registered.
		if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return "", tool.CallerError(fmt.Errorf("field 'url' must be an absolute http(s) URL, got: %q", in.URL))
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
		if err != nil {
			return "", fmt.Errorf("build request for %s: %w", u.Redacted(), err)
		}
		req.Header.Set("User-Agent", userAgent)

		resp, err := c.Do(req)
		if err != nil {
			return "", fmt.Errorf("GET %s: %w", u.Redacted(), err)
		}
		defer resp.Body.Close()

		raw, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
		if err != nil {
			return "", fmt.Errorf("reading %s: %w", u.Redacted(), err)
		}

		// Scrub before truncating: a page that is not UTF-8 (or a body cut at
		// the byte limit mid-rune) would otherwise produce a tool_result the
		// provider rejects on the following call.
		body := truncate(strings.ToValidUTF8(string(raw), "�"), maxToolOutput)

		if resp.StatusCode < 200 || resp.StatusCode > 299 {
			return "", &httpStatusError{code: resp.StatusCode, body: body}
		}
		if body == "" {
			return "(empty response body)", nil
		}
		return body, nil
	})
}

// retryHTTP decides whether a failed fetch is worth another attempt.
//
// WithRetry sits OUTSIDE WithTimeout, so each attempt gets a fresh deadline —
// which is why an overrun counts as transient. Both spellings are matched:
// tool.ErrTimeout is what the middleware normalises a per-call deadline into,
// context.DeadlineExceeded is what the caller sees without the middleware. A
// cancelled run is never retried, whichever way it arrives: the caller has
// gone, and a governor abort surfaces as its own cause rather than as either
// of these.
func retryHTTP(err error) bool {
	if errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, tool.ErrTimeout) {
		return true
	}

	var se *httpStatusError
	if errors.As(err, &se) {
		switch se.code {
		case http.StatusRequestTimeout, // 408
			http.StatusTooManyRequests,    // 429
			http.StatusBadGateway,         // 502
			http.StatusServiceUnavailable, // 503
			http.StatusGatewayTimeout,     // 504
			529:                           // Anthropic-style "overloaded"
			return true
		}
		// Everything else — 404, 401, 400 — will fail identically next time.
		return false
	}

	// Anything the client itself reports (DNS failure, connection refused or
	// reset, TLS handshake) arrives wrapped in *url.Error and is transient by
	// nature. A malformed URL never reaches c.Do, so this cannot mask a
	// permanent input error.
	var ue *url.Error
	return errors.As(err, &ue)
}
