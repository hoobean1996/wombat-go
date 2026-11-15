package openai

// White-box (package openai, not openai_test) on purpose: the whole point of
// this suite is the REQUEST BYTES that encodeRequest produces and the decoders
// that consume a canned wire body — none of which the exported surface hands
// back — plus unexported helpers like parseProxy and retryAfter.

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

// capture records what the client put on the wire.
type capture struct {
	mu      sync.Mutex
	bodies  [][]byte
	headers []http.Header
	paths   []string
}

func (c *capture) record(r *http.Request) {
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
	c.paths = append(c.paths, r.URL.Path)
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

func (c *capture) path(t *testing.T, i int) string {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.paths) {
		t.Fatalf("no request %d was made; got %d request(s)", i, len(c.paths))
	}
	return c.paths[i]
}

func newServer(t *testing.T, reply func(w http.ResponseWriter)) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.record(r)
		reply(w)
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func okJSON(body string) func(http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}
}

// minimalReply is the smallest complete non-streaming answer.
const minimalReply = `{"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`

// sseReply writes each entry as one `data:` frame.
func sseReply(frames ...string) func(http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		f, _ := w.(http.Flusher)
		for _, fr := range frames {
			typ, data, tagged := strings.Cut(fr, "\x00")
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

// clearEnv neutralises every variable New or ConfigFromEnv reads.
func clearEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_PROXY", "OPENAI_MODEL",
		"WOMBAT_MODEL", "OPENAI_EXTRA_HEADERS", "OPENAI_CUSTOM_HEADERS", "OPENAI_STREAM",
		"OPENAI_TEMPERATURE", "OPENAI_TOP_P",
	} {
		t.Setenv(k, "")
	}
}

func newTestClient(t *testing.T, srv *httptest.Server, mutate func(*Config)) *Client {
	t.Helper()
	clearEnv(t)
	cfg := Config{APIKey: "k", BaseURL: srv.URL + "/v1", Model: "test-model"}
	if mutate != nil {
		mutate(&cfg)
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	return c
}

func decodeBody(t *testing.T, raw []byte, v any) {
	t.Helper()
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("body is not valid JSON: %v\nbody: %s", err, raw)
	}
}

// wireMessage is the shape assertions care about, decoded loosely so a missing
// key and a null are distinguishable.
type wireMessage struct {
	Role       string          `json:"role"`
	Content    *string         `json:"content"`
	ToolCallID string          `json:"tool_call_id"`
	ToolCalls  []wireToolCall  `json:"tool_calls"`
	Raw        json.RawMessage `json:"-"`
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// encodeMessagesFor renders one request and hands back the wire messages.
func wireMessages(t *testing.T, system string, msgs []llm.Message) []wireMessage {
	t.Helper()
	raw, err := encodeRequest(llm.Request{System: system, Messages: msgs}, "m", 100, false)
	if err != nil {
		t.Fatalf("encodeRequest: got error %v, want nil", err)
	}
	var doc struct {
		Messages []wireMessage `json:"messages"`
	}
	decodeBody(t, raw, &doc)
	return doc.Messages
}

func roles(msgs []wireMessage) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = m.Role
	}
	return out
}

func contentOf(t *testing.T, m wireMessage) string {
	t.Helper()
	if m.Content == nil {
		t.Fatalf("message %s: content is null, want a string", m.Role)
	}
	return *m.Content
}
