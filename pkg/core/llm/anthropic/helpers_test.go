package anthropic

// These tests are white-box (package anthropic, not anthropic_test) on
// purpose. Everything this package is judged on lives in unexported code —
// encode's cache_control placement, normalizeProxy, retryAfter, the stream
// accumulator — and the whole point of the suite is to assert on the REQUEST
// BYTES, which no exported API hands back.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"wombat/pkg/core/llm"
)

// capture records everything the client put on the wire. Guarded by a mutex
// because the handler runs on the server's goroutine and the assertions run on
// the test's.
type capture struct {
	mu      sync.Mutex
	bodies  [][]byte
	headers []http.Header
	urls    []string
}

func (c *capture) record(r *http.Request) int {
	body := make([]byte, 0, 1024)
	buf := make([]byte, 4096)
	for {
		n, err := r.Body.Read(buf)
		body = append(body, buf[:n]...)
		if err != nil {
			break
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.bodies = append(c.bodies, body)
	c.headers = append(c.headers, r.Header.Clone())
	c.urls = append(c.urls, r.URL.String())
	return len(c.bodies) - 1
}

func (c *capture) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.bodies)
}

func (c *capture) body(t *testing.T, i int) []byte {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.bodies) {
		t.Fatalf("no request %d was made; got %d request(s)", i, len(c.bodies))
	}
	return c.bodies[i]
}

func (c *capture) header(t *testing.T, i int) http.Header {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.headers) {
		t.Fatalf("no request %d was made; got %d request(s)", i, len(c.headers))
	}
	return c.headers[i]
}

func (c *capture) url(t *testing.T, i int) string {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.urls) {
		t.Fatalf("no request %d was made; got %d request(s)", i, len(c.urls))
	}
	return c.urls[i]
}

// newServer starts a recording httptest server. reply writes the n-th answer.
func newServer(t *testing.T, reply func(w http.ResponseWriter, n int)) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := cap.record(r)
		reply(w, n)
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

// okJSON replies with one non-streaming message body.
func okJSON(body string) func(http.ResponseWriter, int) {
	return func(w http.ResponseWriter, _ int) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}
}

// minimalMessage is the smallest legal non-streaming reply.
const minimalMessage = `{"model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

// sse writes framed events. Each entry is "type\x00data"; a bare data payload
// is written without an event: line so the "gateway dropped the event line"
// path is reachable too.
func sseReply(events ...string) func(http.ResponseWriter, int) {
	return func(w http.ResponseWriter, _ int) {
		w.Header().Set("Content-Type", "text/event-stream")
		f, _ := w.(http.Flusher)
		for _, e := range events {
			typ, data, tagged := strings.Cut(e, "\x00")
			if tagged {
				fmt.Fprintf(w, "event: %s\n", typ)
			} else {
				data = typ
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			if f != nil {
				f.Flush()
			}
		}
	}
}

func ev(typ, data string) string { return typ + "\x00" + data }

// clearEnv makes New deterministic regardless of the developer's shell.
func clearEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		envAPIKey, envBaseURL, envProxy, envModel, envCustomHeaders, envBeta, envStream,
		envTemperature, envTopP, envBaseURLAlt, envProxyAlt, envModelAlt,
	} {
		t.Setenv(k, "")
	}
}

// newTestClient builds a client pointed at srv with the env neutralised.
func newTestClient(t *testing.T, srv *httptest.Server, mutate func(*Config)) *Client {
	t.Helper()
	cfg := Config{APIKey: "k", BaseURL: srv.URL, Model: "test-model"}
	if mutate != nil {
		mutate(&cfg)
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	return c
}

// userTurn is a one-block user message.
func userTurn(s string) llm.Message { return llm.UserText(s) }

// decodeBody parses a recorded request body into v, failing the test on error.
func decodeBody(t *testing.T, raw []byte, v any) {
	t.Helper()
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("request body is not valid JSON: %v\nbody: %s", err, raw)
	}
}

// cachePaths lists, in wire order, every place the body carries a
// cache_control breakpoint. Positions are the assertion: a breakpoint that
// moves silently doubles the bill.
func cachePaths(t *testing.T, raw []byte) []string {
	t.Helper()
	var doc struct {
		System   []json.RawMessage `json:"system"`
		Tools    []json.RawMessage `json:"tools"`
		Messages []struct {
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	decodeBody(t, raw, &doc)

	has := func(b json.RawMessage) bool {
		var probe struct {
			CacheControl *struct {
				Type string `json:"type"`
			} `json:"cache_control"`
		}
		decodeBody(t, b, &probe)
		if probe.CacheControl == nil {
			return false
		}
		if probe.CacheControl.Type != "ephemeral" {
			t.Errorf("cache_control type: got %q, want %q", probe.CacheControl.Type, "ephemeral")
		}
		return true
	}

	var out []string
	for i, b := range doc.System {
		if has(b) {
			out = append(out, fmt.Sprintf("system[%d]", i))
		}
	}
	for i, b := range doc.Tools {
		if has(b) {
			out = append(out, fmt.Sprintf("tools[%d]", i))
		}
	}
	for i, m := range doc.Messages {
		for j, b := range m.Content {
			if has(b) {
				out = append(out, fmt.Sprintf("messages[%d].content[%d]", i, j))
			}
		}
	}
	return out
}
