// Package wire is a scriptable OpenAI-compatible fault server.
//
// Every fault test in this repository substitutes an [llm.Client] — a Go
// interface — which is a fine way to test the agent loop and a useless way to
// test the adapter, because it bypasses the whole HTTP + SSE + JSON decode
// stack. The conditions the harness's most interesting code claims to handle
// are WIRE conditions, and none of them are reachable that way:
//
//   - a turn that finishes cleanly with no content and NO usage chunk at all
//   - text deltas followed by a socket reset
//   - a stream that ends without its [DONE] boundary
//   - a provider error body that has to be recognised as context overflow
//
// This server produces them against the real client, so the thing under test is
// the code that ships.
//
// # The script
//
// Behaviors are consumed FIFO, one per ACCEPTED request. A request with the
// wrong method, path, bearer token or an unparseable body does NOT consume one.
//
// That rule is load-bearing. If a malformed request ate a script entry, one
// adapter bug would shift every later behavior by one and the suite would
// report a cascade of unrelated failures with the real cause three tests back.
// Rejecting without consuming keeps a bug local to the assertion that found it.
package wire

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
)

// Server is a running fault server.
type Server struct {
	// BaseURL is what an openai.Config should point at; the client appends
	// "/chat/completions".
	BaseURL string

	APIKey string

	srv *httptest.Server

	mu        sync.Mutex
	script    []Behavior
	next      int
	requests  []Captured
	rejected  int
	repeatEnd bool
}

// Captured is one accepted request, kept for inspection.
type Captured struct {
	Behavior string
	Body     map[string]any
	Raw      []byte
}

// Options configures a server.
type Options struct {
	APIKey string

	// RepeatLast reuses the final behavior once the script is exhausted.
	// Without it, exhaustion is an explicit 500 that names the overrun rather
	// than a mystery hang.
	RepeatLast bool
}

// Start brings up a server running script.
func Start(script []Behavior, opts Options) *Server {
	s := &Server{script: script, APIKey: opts.APIKey, repeatEnd: opts.RepeatLast}
	if s.APIKey == "" {
		s.APIKey = "wire-test-key"
	}
	s.srv = httptest.NewServer(http.HandlerFunc(s.handle))
	s.BaseURL = s.srv.URL + "/v1"
	return s
}

// Close shuts the server down.
func (s *Server) Close() { s.srv.CloseClientConnections(); s.srv.Close() }

// Requests returns the accepted requests, in order.
func (s *Server) Requests() []Captured {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Captured(nil), s.requests...)
}

// Rejected counts requests that were refused without consuming a behavior.
//
// A test should assert this is zero unless it is deliberately sending a bad
// request: a non-zero count means the adapter and this server disagree about
// the protocol, which would otherwise surface as a confusing script offset.
func (s *Server) Rejected() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.rejected
}

// Consumed reports how many behaviors have been used.
func (s *Server) Consumed() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.next
}

func (s *Server) reject(w http.ResponseWriter, status int, msg string) {
	s.mu.Lock()
	s.rejected++
	s.mu.Unlock()
	writeError(w, status, msg, "invalid_request_error")
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.reject(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		s.reject(w, http.StatusNotFound, "no such endpoint: "+r.URL.Path)
		return
	}
	if got := r.Header.Get("Authorization"); got != "Bearer "+s.APIKey {
		s.reject(w, http.StatusUnauthorized, "bad bearer token")
		return
	}

	var body map[string]any
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(&body); err != nil {
		s.reject(w, http.StatusBadRequest, "unparseable request body: "+err.Error())
		return
	}

	s.mu.Lock()
	var b Behavior
	switch {
	case s.next < len(s.script):
		b = s.script[s.next]
		s.next++
	case s.repeatEnd && len(s.script) > 0:
		b = s.script[len(s.script)-1]
	default:
		n := s.next
		s.mu.Unlock()
		writeError(w, http.StatusInternalServerError,
			fmt.Sprintf("wire: script exhausted after %d behaviors; the client made another call", n),
			"script_exhausted")
		return
	}
	raw, _ := json.Marshal(body)
	s.requests = append(s.requests, Captured{Behavior: b.Name, Body: body, Raw: raw})
	s.mu.Unlock()

	b.serve(w, r)
}

// ===== wire helpers =====

type chunkChoice struct {
	Delta        chunkDelta `json:"delta"`
	FinishReason string     `json:"finish_reason,omitempty"`
	Index        int        `json:"index"`
}

type chunkDelta struct {
	Content          string      `json:"content,omitempty"`
	ReasoningContent string      `json:"reasoning_content,omitempty"`
	ToolCalls        []chunkCall `json:"tool_calls,omitempty"`
}

type chunkCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function"`
}

type chunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
	Usage   *chunkUsage   `json:"usage,omitempty"`
}

type chunkUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

const testModel = "wire-model"

func sseHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flush(w)
}

func flush(w http.ResponseWriter) {
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func send(w http.ResponseWriter, c chunk) {
	c.Object, c.Model = "chat.completion.chunk", testModel
	if c.ID == "" {
		c.ID = "wire-1"
	}
	b, err := json.Marshal(c)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", b)
	flush(w)
}

func sendRaw(w http.ResponseWriter, payload string) {
	fmt.Fprintf(w, "data: %s\n\n", payload)
	flush(w)
}

func done(w http.ResponseWriter) { sendRaw(w, "[DONE]") }

// reset tears the connection down without a FIN, which is what a client sees
// as a mid-stream failure rather than as a clean end.
func reset(w http.ResponseWriter) {
	h, ok := w.(http.Hijacker)
	if !ok {
		return
	}
	conn, _, err := h.Hijack()
	if err != nil {
		return
	}
	if tcp, ok := conn.(*net.TCPConn); ok {
		// Linger 0 makes Close send RST instead of FIN.
		_ = tcp.SetLinger(0)
	}
	_ = conn.Close()
}

// closeClean ends the connection with a normal FIN, mid-stream.
func closeClean(w http.ResponseWriter) {
	h, ok := w.(http.Hijacker)
	if !ok {
		return
	}
	conn, buf, err := h.Hijack()
	if err != nil {
		return
	}
	_ = buf.Flush()
	_ = conn.Close()
}

func writeError(w http.ResponseWriter, status int, msg, kind string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": kind},
	})
}

func chunkText(s string, size int) []string {
	if size <= 0 {
		size = 4
	}
	var out []string
	for i := 0; i < len(s); i += size {
		out = append(out, s[i:min(i+size, len(s))])
	}
	return out
}
