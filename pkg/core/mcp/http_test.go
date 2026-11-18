package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// httpFake is a Streamable HTTP server that holds its clients to the parts of
// the specification this client relies on: the session id on every request
// after initialize, the negotiated protocol version, and an Accept header that
// takes both kinds of answer.
type httpFake struct {
	t *testing.T

	mu       sync.Mutex
	sessions map[string]bool
	inits    int
	deleted  []string
	waiting  map[string]chan message
}

func newHTTPFake(t *testing.T) (*httpFake, *httptest.Server) {
	f := &httpFake{t: t, sessions: map[string]bool{}, waiting: map[string]chan message{}}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	return f, srv
}

// expire forgets every session, as a server that restarted would.
func (f *httpFake) expire() {
	f.mu.Lock()
	clear(f.sessions)
	f.mu.Unlock()
}

func (f *httpFake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer sekret" {
		http.Error(w, "who are you", http.StatusUnauthorized)
		return
	}
	if r.Method == http.MethodDelete {
		f.mu.Lock()
		id := r.Header.Get("Mcp-Session-Id")
		delete(f.sessions, id)
		f.deleted = append(f.deleted, id)
		f.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	if a := r.Header.Get("Accept"); !strings.Contains(a, "application/json") || !strings.Contains(a, "text/event-stream") {
		http.Error(w, "Accept must take both", http.StatusNotAcceptable)
		return
	}
	var m message
	if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if m.Method == "initialize" {
		f.mu.Lock()
		f.inits++
		id := fmt.Sprintf("session-%d", f.inits)
		f.sessions[id] = true
		f.mu.Unlock()
		w.Header().Set("Mcp-Session-Id", id)
		writeJSON(w, message{JSONRPC: "2.0", ID: m.ID, Result: json.RawMessage(
			`{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"httpfake","version":"0"}}`)})
		return
	}
	id := r.Header.Get("Mcp-Session-Id")
	f.mu.Lock()
	live := f.sessions[id]
	f.mu.Unlock()
	switch {
	case id == "":
		http.Error(w, "no session", http.StatusBadRequest)
		return
	case !live:
		http.Error(w, "no such session", http.StatusNotFound)
		return
	case r.Header.Get("MCP-Protocol-Version") != ProtocolVersion:
		http.Error(w, "wrong protocol version "+r.Header.Get("MCP-Protocol-Version"), http.StatusBadRequest)
		return
	}

	switch {
	case m.isResponse():
		f.mu.Lock()
		ch := f.waiting[string(m.ID)]
		f.mu.Unlock()
		if ch != nil {
			ch <- m
		}
		w.WriteHeader(http.StatusAccepted)
		return
	case !m.isRequest():
		w.WriteHeader(http.StatusAccepted)
		return
	}

	switch m.Method {
	case "tools/list":
		writeJSON(w, message{JSONRPC: "2.0", ID: m.ID, Result: json.RawMessage(
			`{"tools":[{"name":"echo","inputSchema":{"type":"object"},"annotations":{"readOnlyHint":true}}]}`)})
	case "tools/call":
		f.streamCall(w, m)
	default:
		writeJSON(w, message{JSONRPC: "2.0", ID: m.ID, Error: &rpcError{Code: errMethodNotFound, Message: "no"}})
	}
}

func writeJSON(w http.ResponseWriter, m message) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(m)
}

// streamCall answers a tool call with an event stream: a log notification, a
// ping the client must answer, then the result.
func (f *httpFake) streamCall(w http.ResponseWriter, m message) {
	var p struct {
		Arguments struct {
			Text string `json:"text"`
		} `json:"arguments"`
	}
	_ = json.Unmarshal(m.Params, &p)
	w.Header().Set("Content-Type", "text/event-stream")
	flush := w.(http.Flusher)
	event := func(v message) {
		v.JSONRPC = "2.0"
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "event: message\ndata: %s\n\n", b)
		flush.Flush()
	}
	event(message{Method: "notifications/message", Params: json.RawMessage(`{"level":"info","data":"working"}`)})

	pingID := fmt.Sprintf(`"ping-%s"`, m.ID)
	ch := make(chan message, 1)
	f.mu.Lock()
	f.waiting[pingID] = ch
	f.mu.Unlock()
	event(message{ID: json.RawMessage(pingID), Method: "ping"})
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		event(message{ID: m.ID, Result: json.RawMessage(`{"content":[{"type":"text","text":"the client never answered the ping"}],"isError":true}`)})
		return
	}
	res, _ := json.Marshal(textResult("echo: " + p.Arguments.Text))
	event(message{ID: m.ID, Result: res})
}

func httpServer(url string) Server {
	return Server{Name: "remote", URL: url, Headers: map[string]string{"Authorization": "Bearer sekret"}}
}

func TestHTTPSessionsEventStreamsAndServerRequests(t *testing.T) {
	f, srv := newHTTPFake(t)
	c := connect(t, httpServer(srv.URL))
	defs, err := c.Tools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(defs) != 1 || defs[0].Name != "remote__echo" {
		t.Fatalf("defs = %+v", defs)
	}
	out, err := defs[0].Fn(context.Background(), json.RawMessage(`{"text":"over http"}`))
	if err != nil || out != "echo: over http" {
		t.Fatalf("echo = %q, %v", out, err)
	}

	// The server forgets the session: the next call starts a new one and
	// still succeeds, once, however many calls hit the expiry together.
	f.expire()
	var wg sync.WaitGroup
	for i := range 4 {
		wg.Go(func() {
			text := fmt.Sprint(i)
			out, err := c.CallTool(context.Background(), "echo", json.RawMessage(`{"text":"`+text+`"}`))
			if err != nil || out != "echo: "+text {
				t.Errorf("after expiry: echo = %q, %v", out, err)
			}
		})
	}
	wg.Wait()
	f.mu.Lock()
	inits := f.inits
	f.mu.Unlock()
	if inits != 2 {
		t.Errorf("%d handshakes, want 2: one at the start and one after the session expired", inits)
	}

	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	deleted := f.deleted
	f.mu.Unlock()
	if len(deleted) != 1 || deleted[0] != "session-2" {
		t.Errorf("deleted sessions = %v, want the live one", deleted)
	}
	if _, err := c.CallTool(context.Background(), "echo", nil); err == nil {
		t.Error("a call after Close succeeded")
	}
}

func TestHTTPRefusalNamesTheProblem(t *testing.T) {
	_, srv := newHTTPFake(t)
	s := httpServer(srv.URL)
	s.Headers = nil
	_, err := Connect(context.Background(), s)
	if err == nil || !strings.Contains(err.Error(), "credentials") {
		t.Errorf("err = %v, want one pointing at the credentials", err)
	}
	if _, err := Connect(context.Background(), Server{Name: "x", URL: "ftp://example.com"}); err == nil {
		t.Error("Connect accepted an ftp URL")
	}
}

// The calling session reaches a server configured to be told it, in each tool
// call's _meta, and no other server; and nothing is sent outside a session.
func TestSessionReachesOnlyTheServersThatAskForIt(t *testing.T) {
	var mu sync.Mutex
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		var m message
		if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		switch m.Method {
		case "initialize":
			writeJSON(w, message{JSONRPC: "2.0", ID: m.ID, Result: json.RawMessage(
				`{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"meta","version":"0"}}`)})
		case "tools/list":
			writeJSON(w, message{JSONRPC: "2.0", ID: m.ID, Result: json.RawMessage(
				`{"tools":[{"name":"whoami","inputSchema":{"type":"object"}}]}`)})
		case "tools/call":
			var p struct {
				Meta json.RawMessage `json:"_meta"`
			}
			_ = json.Unmarshal(m.Params, &p)
			mu.Lock()
			got = append(got, string(p.Meta))
			mu.Unlock()
			res, _ := json.Marshal(textResult("ok"))
			writeJSON(w, message{JSONRPC: "2.0", ID: m.ID, Result: res})
		default: // notifications
			w.WriteHeader(http.StatusAccepted)
		}
	}))
	t.Cleanup(srv.Close)

	ctx := context.Background()
	for _, tc := range []struct {
		name    string
		session bool
		ctx     context.Context
		want    string
	}{
		{"a server that asks, in a session", true, WithSession(ctx, "sess-1"), `{"wombat/session":"sess-1"}`},
		{"a server that asks, outside a session", true, ctx, ""},
		{"a server that does not ask", false, WithSession(ctx, "sess-1"), ""},
		// A call the harness dispatched names itself to every server, as a
		// digest that says nothing about the session.
		{"a dispatched call, to a server that does not ask", false,
			tool.WithInfo(WithSession(ctx, "sess-1"), tool.Info{UseID: "u1"}),
			`{"wombat/call":"` + callDigest("u1") + `"}`},
		{"a dispatched call, to a server that asks", true,
			tool.WithCallKeys(tool.WithInfo(WithSession(ctx, "sess-1"), tool.Info{UseID: "u1"}),
				func(llm.ToolUseID) string { return "sess-1/4.0" }),
			`{"wombat/call":"` + callDigest("sess-1/4.0") + `","wombat/session":"sess-1"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := Connect(ctx, Server{Name: "meta", URL: srv.URL, Session: tc.session})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if _, err := c.CallTool(tc.ctx, "whoami", nil); err != nil {
				t.Fatal(err)
			}
			mu.Lock()
			last := got[len(got)-1]
			mu.Unlock()
			if last != tc.want {
				t.Errorf("_meta = %q, want %q", last, tc.want)
			}
		})
	}

	path := filepath.Join(t.TempDir(), "mcp.json")
	if err := os.WriteFile(path, []byte(`{"mcpServers":{"oda":{"type":"http","url":"https://example.com/mcp","session":true}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	servers, err := LoadConfig(path)
	if err != nil || len(servers) != 1 || !servers[0].Session {
		t.Errorf("LoadConfig with \"session\": true = %+v, %v; want one server with Session set", servers, err)
	}
}
