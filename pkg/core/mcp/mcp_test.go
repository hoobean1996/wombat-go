package mcp

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// The stdio tests run this test binary as the server: TestMain sees the
// variable and serves instead of testing.
func TestMain(m *testing.M) {
	if os.Getenv("WOMBAT_MCP_FAKE") == "stdio" {
		serveFake(os.Stdin, os.Stdout, os.Getenv("WOMBAT_MCP_STUBBORN"))
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func fakeServer(env map[string]string) Server {
	// GORACE: the race runtime sleeps a second on exit by default, which
	// every test that closes a server would wait out.
	e := map[string]string{"WOMBAT_MCP_FAKE": "stdio", "GORACE": "atexit_sleep_ms=0"}
	for k, v := range env {
		e[k] = v
	}
	return Server{Name: "fake", Command: os.Args[0], Env: e}
}

func connect(t *testing.T, s Server) *Client {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	c, err := Connect(ctx, s)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

// ===== the fake server =====

var fakePNG = []byte("\x89PNG\r\n\x1a\nnot really a picture")

const fakePage1 = `[
 {"name":"echo","description":"Echo the text back.","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]},
  "annotations":{"readOnlyHint":true,"openWorldHint":false}},
 {"name":"write.file","inputSchema":{"type":"object"},"annotations":{"title":"Write a file","destructiveHint":true,"openWorldHint":false}},
 {"name":"lookup","description":"Look something up.","inputSchema":{"type":"object"},"annotations":{"readOnlyHint":true}}
]`

const fakePage2 = `[
 {"name":"picture","description":"Draw a picture.","inputSchema":{"type":"object"}},
 {"name":"fail","description":"Fail.","inputSchema":{"type":"object"}},
 {"name":"slow","description":"Take a long time.","inputSchema":{"type":"object"}},
 {"name":"cancelled","description":"Say which requests were cancelled.","inputSchema":{"type":"object"}},
 {"name":"env","description":"Show the environment.","inputSchema":{"type":"object"}},
 {"name":"ping_client","description":"Ping the client.","inputSchema":{"type":"object"}},
 {"name":"exit","description":"Die.","inputSchema":{"type":"object"}}
]`

type fake struct {
	out       io.Writer
	wmu       sync.Mutex
	mu        sync.Mutex
	waiting   map[string]chan message
	cancelled []int64
	seq       int
	eager     string // how the ping sent before the handshake was answered
}

func serveFake(in io.Reader, out io.Writer, stubborn string) {
	if stubborn == "term" {
		signal.Ignore(syscall.SIGTERM)
	}
	f := &fake{out: out, waiting: map[string]chan message{}}
	if os.Getenv("WOMBAT_MCP_EAGER_PING") == "1" {
		// The specification lets a server ping before initialization.
		go func() {
			r := f.request("ping")
			f.mu.Lock()
			defer f.mu.Unlock()
			if r.Error != nil {
				f.eager = "error: " + r.Error.Message
			} else {
				f.eager = "answered " + string(r.Result)
			}
		}()
	}
	sc := bufio.NewScanner(in)
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		var m message
		if json.Unmarshal(sc.Bytes(), &m) != nil {
			continue
		}
		go f.handle(m)
	}
	if stubborn != "" {
		time.Sleep(time.Hour) // a server that does not take the hint
	}
}

func (f *fake) write(m message) {
	m.JSONRPC = "2.0"
	b, _ := json.Marshal(m)
	f.wmu.Lock()
	defer f.wmu.Unlock()
	_, _ = f.out.Write(append(b, '\n'))
}

func (f *fake) reply(req message, result any) {
	b, _ := json.Marshal(result)
	f.write(message{ID: req.ID, Result: b})
}

func textResult(s string) map[string]any {
	return map[string]any{"content": []any{map[string]any{"type": "text", "text": s}}}
}

// request asks the client something and waits for its answer.
func (f *fake) request(method string) message {
	f.mu.Lock()
	f.seq++
	id := fmt.Sprintf(`"s%d"`, f.seq)
	ch := make(chan message, 1)
	f.waiting[id] = ch
	f.mu.Unlock()
	f.write(message{ID: json.RawMessage(id), Method: method})
	select {
	case r := <-ch:
		return r
	case <-time.After(5 * time.Second):
		return message{Error: &rpcError{Message: "no answer"}}
	}
}

func (f *fake) handle(m message) {
	if m.isResponse() {
		f.mu.Lock()
		ch := f.waiting[string(m.ID)]
		f.mu.Unlock()
		if ch != nil {
			ch <- m
		}
		return
	}
	switch m.Method {
	case "initialize":
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		_ = json.Unmarshal(m.Params, &p)
		f.reply(m, map[string]any{
			"protocolVersion": p.ProtocolVersion,
			"capabilities":    map[string]any{"tools": map[string]any{"listChanged": true}},
			"serverInfo":      map[string]any{"name": "fake", "version": "0"},
			"instructions":    "Prefer echo.",
		})
	case "notifications/cancelled":
		var p struct {
			RequestID int64 `json:"requestId"`
		}
		_ = json.Unmarshal(m.Params, &p)
		f.mu.Lock()
		f.cancelled = append(f.cancelled, p.RequestID)
		f.mu.Unlock()
	case "tools/list":
		var p struct {
			Cursor string `json:"cursor"`
		}
		_ = json.Unmarshal(m.Params, &p)
		if p.Cursor == "" {
			f.reply(m, map[string]any{"tools": json.RawMessage(fakePage1), "nextCursor": "page-2"})
		} else {
			f.reply(m, map[string]any{"tools": json.RawMessage(fakePage2)})
		}
	case "tools/call":
		f.call(m)
	default:
		if m.isRequest() {
			f.write(message{ID: m.ID, Error: &rpcError{Code: errMethodNotFound, Message: "no " + m.Method}})
		}
	}
}

func (f *fake) call(m message) {
	var p struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	_ = json.Unmarshal(m.Params, &p)
	switch p.Name {
	case "echo":
		var a struct {
			Text string `json:"text"`
		}
		_ = json.Unmarshal(p.Arguments, &a)
		f.reply(m, textResult("echo: "+a.Text))
	case "picture":
		f.write(message{Method: "notifications/tools/list_changed"})
		f.reply(m, map[string]any{"content": []any{
			map[string]any{"type": "text", "text": "a red dot"},
			map[string]any{"type": "image", "mimeType": "image/png", "data": base64.StdEncoding.EncodeToString(fakePNG)},
			map[string]any{"type": "audio", "mimeType": "audio/wav", "data": "AAAA"},
			map[string]any{"type": "resource_link", "uri": "file:///tmp/x.txt", "name": "x.txt"},
			map[string]any{"type": "resource", "resource": map[string]any{"uri": "file:///tmp/notes.md", "text": "# notes"}},
		}})
	case "fail":
		f.reply(m, map[string]any{"content": []any{map[string]any{"type": "text", "text": "the widget is broken"}}, "isError": true})
	case "slow":
		time.Sleep(30 * time.Second)
		f.reply(m, textResult("finally"))
	case "cancelled":
		f.mu.Lock()
		b, _ := json.Marshal(f.cancelled)
		f.mu.Unlock()
		f.reply(m, textResult(string(b)))
	case "env":
		f.reply(m, textResult(os.Getenv("WOMBAT_MCP_TEST_SECRET")+"|"+os.Getenv("WOMBAT_MCP_GRANTED")))
	case "eager":
		f.mu.Lock()
		got := f.eager
		f.mu.Unlock()
		f.reply(m, textResult(got))
	case "ping_client":
		r := f.request("ping")
		if r.Error != nil {
			f.reply(m, textResult("ping failed: "+r.Error.Message))
			return
		}
		f.reply(m, textResult("pong "+string(r.Result)))
	case "exit":
		fmt.Fprintln(os.Stderr, "fatal: the database is on fire")
		os.Exit(3)
	default:
		f.write(message{ID: m.ID, Error: &rpcError{Code: errInvalidParams, Message: "unknown tool " + p.Name}})
	}
}

// ===== tests =====

func TestStdioListsEveryPageAndCallsTools(t *testing.T) {
	c := connect(t, fakeServer(nil))
	if got := c.Instructions(); got != "Prefer echo." {
		t.Errorf("Instructions() = %q", got)
	}
	defs, err := c.Tools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, d := range defs {
		names = append(names, d.Name)
	}
	want := []string{"fake__echo", "fake__write_file", "fake__lookup", "fake__picture", "fake__fail",
		"fake__slow", "fake__cancelled", "fake__env", "fake__ping_client", "fake__exit"}
	if !slices.Equal(names, want) {
		t.Fatalf("names = %v\nwant    %v", names, want)
	}

	echo := defs[0]
	out, err := echo.Fn(context.Background(), json.RawMessage(`{"text":"hi"}`))
	if err != nil || out != "echo: hi" {
		t.Errorf("echo = %q, %v; want %q", out, err, "echo: hi")
	}
	if string(echo.InputSchema) == "" || !strings.Contains(string(echo.InputSchema), `"required":["text"]`) {
		t.Errorf("InputSchema = %s, want the server's own schema", echo.InputSchema)
	}
	if echo.Category != "mcp:fake" {
		t.Errorf("Category = %q", echo.Category)
	}
	if defs[1].Description != "Write a file" {
		t.Errorf("write_file description = %q, want the annotation title when there is nothing else", defs[1].Description)
	}

	// Concurrent calls are multiplexed over the one connection.
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			msg := fmt.Sprint(i)
			out, err := echo.Fn(context.Background(), json.RawMessage(`{"text":"`+msg+`"}`))
			if err != nil || out != "echo: "+msg {
				t.Errorf("concurrent echo %d = %q, %v", i, out, err)
			}
		})
	}
	wg.Wait()
}

func TestAnnotationsBecomeCapabilities(t *testing.T) {
	yes, no := true, false
	tests := []struct {
		name       string
		a          *Annotations
		ignore     bool
		want       tool.Cap
		idempotent bool
	}{
		{"no annotations: the spec's defaults", nil, false, tool.CapMutating | tool.CapNetwork, false},
		{"read-only, closed world", &Annotations{ReadOnlyHint: &yes, OpenWorldHint: &no}, false, tool.CapReadOnly, true},
		{"read-only, open world by default", &Annotations{ReadOnlyHint: &yes}, false, tool.CapReadOnly | tool.CapNetwork, true},
		{"writes, closed world, idempotent", &Annotations{ReadOnlyHint: &no, OpenWorldHint: &no, IdempotentHint: &yes}, false, tool.CapMutating, true},
		{"hints ignored", &Annotations{ReadOnlyHint: &yes, OpenWorldHint: &no}, true, tool.CapMutating | tool.CapNetwork, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Client{server: Server{Name: "s", IgnoreHints: tt.ignore}}
			d := c.Def(Tool{Name: "t", Annotations: tt.a})
			if d.Caps != tt.want || d.Idempotent != tt.idempotent {
				t.Errorf("caps = %b, idempotent = %v; want %b, %v", d.Caps, d.Idempotent, tt.want, tt.idempotent)
			}
		})
	}
}

func TestMediaIsAttachedAndTheRestIsDescribed(t *testing.T) {
	c := connect(t, fakeServer(nil))
	defs, err := c.Tools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := tool.NewDispatcher(tool.Direct).Dispatch(context.Background(), tool.NewSet(defs...),
		[]llm.ToolUse{{ID: "u1", Name: "fake__picture", Input: json.RawMessage(`{}`)}})
	r := got[0]
	if r.Err != nil {
		t.Fatal(r.Err)
	}
	if len(r.Media) != 1 || r.Media[0].MediaType != "image/png" || string(r.Media[0].Data) != string(fakePNG) {
		t.Fatalf("Media = %+v, want the one PNG", r.Media)
	}
	for _, want := range []string{
		"a red dot",
		"[attached image 2: image/png",
		"[audio 3 (audio/wav) omitted",
		"[resource link: file:///tmp/x.txt (x.txt)]",
		"[resource file:///tmp/notes.md]\n# notes",
	} {
		if !strings.Contains(r.Output, want) {
			t.Errorf("output lacks %q:\n%s", want, r.Output)
		}
	}
	if !c.ToolsChanged() {
		t.Error("ToolsChanged() = false after the server said its list changed")
	}
}

func TestToolErrorsReachTheModelAsCallerFaults(t *testing.T) {
	c := connect(t, fakeServer(nil))
	_, err := c.CallTool(context.Background(), "fail", nil)
	if err == nil || err.Error() != "the widget is broken" {
		t.Fatalf("err = %v, want the tool's own message", err)
	}
	if !tool.IsCallerFault(err) {
		t.Error("a tool that answered with isError is not a sick server; want a caller fault")
	}

	_, err = c.CallTool(context.Background(), "no_such_tool", nil)
	if !errors.Is(err, tool.ErrInvalidInput) {
		t.Errorf("unknown tool: err = %v, want it to match tool.ErrInvalidInput", err)
	}
}

func TestStdioServerSeesOnlyWhatItIsGiven(t *testing.T) {
	t.Setenv("WOMBAT_MCP_TEST_SECRET", "hunter2")
	c := connect(t, fakeServer(map[string]string{"WOMBAT_MCP_GRANTED": "yes"}))
	out, err := c.CallTool(context.Background(), "env", nil)
	if err != nil {
		t.Fatal(err)
	}
	if out != "|yes" {
		t.Errorf("the server saw %q, want only the granted variable (\"|yes\")", out)
	}
}

func TestCancelledCallIsAbandonedAndTheServerTold(t *testing.T) {
	c := connect(t, fakeServer(nil))
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := c.CallTool(ctx, "slow", nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want the deadline", err)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("the call took %v to give up", time.Since(start))
	}
	for deadline := time.Now().Add(5 * time.Second); ; {
		out, err := c.CallTool(context.Background(), "cancelled", nil)
		if err != nil {
			t.Fatal(err)
		}
		if out != "null" && out != "[]" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the server was never told the call was cancelled")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestServerRequestsAreAnswered(t *testing.T) {
	c := connect(t, fakeServer(nil))
	out, err := c.CallTool(context.Background(), "ping_client", nil)
	if err != nil || out != "pong {}" {
		t.Errorf("ping_client = %q, %v; want the client's empty ping result", out, err)
	}
}

// TestPingBeforeTheHandshakeIsAnswered: the reader used to start before the
// client had its transport, so a server that pinged at once raced the
// assignment — and could find no transport at all to answer with.
func TestPingBeforeTheHandshakeIsAnswered(t *testing.T) {
	c := connect(t, fakeServer(map[string]string{"WOMBAT_MCP_EAGER_PING": "1"}))
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		out, err := c.CallTool(context.Background(), "eager", nil)
		if err != nil {
			t.Fatal(err)
		}
		if out == "answered {}" {
			return
		}
		if out != "" || time.Now().After(deadline) {
			t.Fatalf("the early ping: %q, want it answered", out)
		}
	}
}

func TestServerThatDiesFailsItsCallsWithItsLastWords(t *testing.T) {
	c := connect(t, fakeServer(nil))
	_, err := c.CallTool(context.Background(), "exit", nil)
	if err == nil {
		t.Fatal("a call whose server died succeeded")
	}
	for _, want := range []string{"exited", "exit status 3", "the database is on fire"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("err = %v, want it to mention %q", err, want)
		}
	}
	if _, err := c.CallTool(context.Background(), "echo", json.RawMessage(`{"text":"x"}`)); err == nil {
		t.Error("a call after the server died succeeded")
	}
}

func TestCloseStopsAServerThatIgnoresItsInput(t *testing.T) {
	old := stopGrace
	stopGrace = 200 * time.Millisecond
	t.Cleanup(func() { stopGrace = old })
	for _, mode := range []string{"stdin", "term"} {
		t.Run(mode, func(t *testing.T) {
			c := connect(t, fakeServer(map[string]string{"WOMBAT_MCP_STUBBORN": mode}))
			start := time.Now()
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
			select {
			case <-c.t.(*stdio).exited:
			default:
				t.Fatal("Close returned with the server still running")
			}
			if d := time.Since(start); d > 5*time.Second {
				t.Errorf("Close took %v", d)
			}
			if _, err := c.CallTool(context.Background(), "echo", nil); err == nil {
				t.Error("a call after Close succeeded")
			}
		})
	}
}

func TestConnectAllPrefixesAndCloses(t *testing.T) {
	a, b := fakeServer(nil), fakeServer(nil)
	a.Name, b.Name = "one", "two.v2"
	cs, err := ConnectAll(context.Background(), []Server{a, b})
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()
	set := tool.NewSet(cs.Tools()...) // panics on a duplicate name
	for _, name := range []string{"one__echo", "two_v2__echo"} {
		if _, ok := set.Find(name); !ok {
			t.Errorf("no tool %s", name)
		}
	}
	if got := cs.Instructions(); !strings.Contains(got, "## one\nPrefer echo.") || !strings.Contains(got, "## two.v2\nPrefer echo.") {
		t.Errorf("Instructions() = %q", got)
	}

	bad := Server{Name: "broken", Command: "/nonexistent/wombat-mcp-server"}
	if _, err := ConnectAll(context.Background(), []Server{a, bad}); err == nil || !strings.Contains(err.Error(), `"broken"`) {
		t.Errorf("err = %v, want one naming the server that failed", err)
	}
}

func TestToolNamesFitTheProviders(t *testing.T) {
	long := strings.Repeat("x", 80)
	tests := []struct{ server, tool string }{
		{"github", "create_issue"},
		{"my server", "read.file"},
		{"s", long},
		{long, "t"},
		{"ünïcode", "名前"},
	}
	seen := map[string]bool{}
	for _, tt := range tests {
		n := uniqueName(seen, tt.server, tt.tool)
		if len(n) == 0 || len(n) > maxName || sanitize(n) != n {
			t.Errorf("name(%q, %q) = %q, which a provider would refuse", tt.server, tt.tool, n)
		}
	}
	if got := toolName("my server", "read.file"); got != "my_server__read_file" {
		t.Errorf("toolName = %q", got)
	}
	// Names that sanitize alike still come out distinct.
	taken := map[string]bool{}
	x, y := uniqueName(taken, "s", "a.b"), uniqueName(taken, "s", "a_b")
	if x == y {
		t.Errorf("a.b and a_b both became %q", x)
	}
	if uniqueName(taken, "s", "a_b") == y {
		t.Error("a third tool reused a name already taken")
	}
}

func TestParseConfig(t *testing.T) {
	t.Setenv("WOMBAT_MCP_TOKEN", "t0ken")
	raw := []byte(`{"mcpServers": {
		"zeta":  {"command": "server", "args": ["--token=${WOMBAT_MCP_TOKEN}"], "env": {"TOKEN": "${WOMBAT_MCP_TOKEN}", "MODE": "${WOMBAT_MCP_UNSET:-safe}"}, "timeout": "90s"},
		"alpha": {"type": "http", "url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${WOMBAT_MCP_TOKEN}"}, "ignoreHints": true},
		"off":   {"command": "whatever", "disabled": true}
	}}`)
	servers, err := ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(servers) != 2 || servers[0].Name != "alpha" || servers[1].Name != "zeta" {
		t.Fatalf("servers = %+v, want alpha and zeta in that order, and no disabled one", servers)
	}
	if h := servers[0].Headers["Authorization"]; h != "Bearer t0ken" || !servers[0].IgnoreHints {
		t.Errorf("alpha = %+v", servers[0])
	}
	z := servers[1]
	if z.Args[0] != "--token=t0ken" || z.Env["TOKEN"] != "t0ken" || z.Env["MODE"] != "safe" || z.Timeout != 90*time.Second {
		t.Errorf("zeta = %+v", z)
	}

	for _, bad := range []string{
		`{"mcpServers": {"x": {"command": "s", "env": {"T": "${WOMBAT_MCP_UNSET}"}}}}`,
		`{"mcpServers": {"x": {"type": "sse", "url": "https://example.com/sse"}}}`,
		`{"mcpServers": {"x": {"command": "s", "url": "https://example.com"}}}`,
		`{"mcpServers": {"x": {}}}`,
		`{"mcpServers": {"x": {"type": "carrier-pigeon", "command": "s"}}}`,
		`{"mcpServers": {"x": {"command": "s", "timeout": "soon"}}}`,
	} {
		if _, err := ParseConfig([]byte(bad)); err == nil {
			t.Errorf("ParseConfig(%s) succeeded", bad)
		}
	}
}
