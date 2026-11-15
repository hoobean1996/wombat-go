package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// These tests run the real command: the test binary re-executes itself, and
// TestMain hands the child to run() — or, for -mcp, to a small MCP server —
// instead of to the tests. The provider is a fake Anthropic endpoint on
// httptest, so everything from flag parsing to the JSONL on stdout is the
// code that ships, wired the way main wires it.
func TestMain(m *testing.M) {
	switch os.Getenv("WOMBAT_JSONL_AS") {
	case "cli":
		os.Exit(run())
	case "mcp":
		serveMCP(os.Stdin, os.Stdout)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// ===== a fake provider =====

type wireRequest struct {
	System   json.RawMessage `json:"system"`
	Messages []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	Tools []struct {
		Name string `json:"name"`
	} `json:"tools"`
}

func (r wireRequest) toolNames() []string {
	var out []string
	for _, t := range r.Tools {
		out = append(out, t.Name)
	}
	return out
}

// lastText is every text and tool_result in the request's last message.
func (r wireRequest) lastText() string {
	var blocks []struct {
		Type    string          `json:"type"`
		Text    string          `json:"text"`
		Content json.RawMessage `json:"content"`
	}
	raw := r.Messages[len(r.Messages)-1].Content
	if json.Unmarshal(raw, &blocks) != nil {
		var s string
		_ = json.Unmarshal(raw, &s)
		return s
	}
	var b strings.Builder
	for _, blk := range blocks {
		b.WriteString(blk.Text)
		if len(blk.Content) > 0 {
			var s string
			if json.Unmarshal(blk.Content, &s) == nil {
				b.WriteString(s)
			} else {
				b.Write(blk.Content)
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}

type provider struct {
	t      *testing.T
	script func(n int, req wireRequest) any
	mu     sync.Mutex
	seen   []wireRequest
}

func newProvider(t *testing.T, script func(n int, req wireRequest) any) (*provider, string) {
	p := &provider{t: t, script: script}
	srv := httptest.NewServer(p)
	t.Cleanup(srv.Close)
	return p, srv.URL
}

func (p *provider) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !strings.HasSuffix(r.URL.Path, "/messages") {
		http.NotFound(w, r)
		return
	}
	var req wireRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	p.mu.Lock()
	n := len(p.seen)
	p.seen = append(p.seen, req)
	p.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(p.script(n, req))
}

func (p *provider) requests() []wireRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.seen)
}

func message(stop string, content ...any) map[string]any {
	return map[string]any{
		"id": "msg", "type": "message", "role": "assistant", "model": "fake-model",
		"content": content, "stop_reason": stop,
		"usage": map[string]int{"input_tokens": 10, "output_tokens": 5},
	}
}

func say(text string) any { return message("end_turn", map[string]any{"type": "text", "text": text}) }

func call(id, name string, input any) any {
	return message("tool_use", map[string]any{"type": "tool_use", "id": id, "name": name, "input": input})
}

// ===== running the command =====

type result struct {
	code   int
	events []map[string]any
	stderr string
}

func (r result) last() map[string]any { return r.events[len(r.events)-1] }

func (r result) ofType(typ string) []map[string]any {
	var out []map[string]any
	for _, e := range r.events {
		if e["type"] == typ {
			out = append(out, e)
		}
	}
	return out
}

// cli runs the command against the provider at url. The child's environment
// is this one's without any provider or wombat settings, so a developer's own
// configuration cannot change what the test sees.
func cli(t *testing.T, url string, args ...string) result {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], append([]string{"-provider", "anthropic", "-model", "fake-model", "-stream", "never", "-log", "error"}, args...)...)
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "ANTHROPIC_") && !strings.HasPrefix(kv, "OPENAI_") && !strings.HasPrefix(kv, "WOMBAT_") {
			cmd.Env = append(cmd.Env, kv)
		}
	}
	cmd.Env = append(cmd.Env, "WOMBAT_JSONL_AS=cli", "ANTHROPIC_API_KEY=sk-test-not-a-key", "ANTHROPIC_BASE_URL="+url,
		"GORACE=atexit_sleep_ms=0")
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	res := result{stderr: stderr.String()}
	if ee, ok := err.(*exec.ExitError); ok {
		res.code = ee.ExitCode()
	} else if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(&stdout)
	sc.Buffer(nil, 4<<20)
	for sc.Scan() {
		var ev map[string]any
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			t.Fatalf("stdout is not JSONL: %q: %v", sc.Text(), err)
		}
		res.events = append(res.events, ev)
	}
	if len(res.events) == 0 {
		t.Fatalf("no events on stdout; exit %d, stderr:\n%s", res.code, res.stderr)
	}
	return res
}

// ===== tests =====

func TestCLIAnswersAsJSONL(t *testing.T) {
	_, url := newProvider(t, func(int, wireRequest) any { return say("hello there") })
	r := cli(t, url, "-working-dir", t.TempDir(), "say hello")
	if r.code != exitOK {
		t.Fatalf("exit %d; stderr:\n%s", r.code, r.stderr)
	}
	if r.events[0]["type"] != "session_started" || r.events[0]["query"] != "say hello" {
		t.Errorf("first event = %v", r.events[0])
	}
	if last := r.last(); last["type"] != "done" || last["answer"] != "hello there" {
		t.Errorf("last event = %v", last)
	}
}

func TestCLIRunsToolsWhereItIsPointedWithoutItsCredentials(t *testing.T) {
	work := t.TempDir()
	_, url := newProvider(t, func(n int, req wireRequest) any {
		if n == 0 {
			return call("t1", "bash", map[string]string{"command": "echo built > out.txt; env > env.txt", "exec_dir": work})
		}
		return say("built it")
	})
	r := cli(t, url, "-working-dir", work, "build it")
	if r.code != exitOK || r.last()["answer"] != "built it" {
		t.Fatalf("exit %d, last %v; stderr:\n%s", r.code, r.last(), r.stderr)
	}
	out, err := os.ReadFile(filepath.Join(work, "out.txt"))
	if err != nil || strings.TrimSpace(string(out)) != "built" {
		t.Fatalf("out.txt = %q, %v", out, err)
	}
	env, _ := os.ReadFile(filepath.Join(work, "env.txt"))
	if strings.Contains(string(env), "ANTHROPIC_API_KEY") || strings.Contains(string(env), "sk-test-not-a-key") {
		t.Error("the model's command could read the provider key")
	}
	if !strings.Contains(string(env), "PATH=") {
		t.Error("the model's command lost PATH along with the credentials")
	}
}

func TestCLISessionCarriesTheConversationOn(t *testing.T) {
	work := t.TempDir()
	session := filepath.Join(t.TempDir(), "session.json")
	p, url := newProvider(t, func(n int, req wireRequest) any { return say(fmt.Sprintf("answer %d", n+1)) })
	if r := cli(t, url, "-working-dir", work, "-session", session, "first question"); r.code != exitOK {
		t.Fatalf("first: exit %d; stderr:\n%s", r.code, r.stderr)
	}
	if _, err := os.Stat(session); err != nil {
		t.Fatalf("the session was not written: %v", err)
	}
	r := cli(t, url, "-working-dir", work, "-session", session, "second question")
	if r.code != exitOK || r.last()["answer"] != "answer 2" {
		t.Fatalf("second: exit %d, last %v; stderr:\n%s", r.code, r.last(), r.stderr)
	}
	second := p.requests()[1]
	if len(second.Messages) != 3 || !strings.Contains(second.lastText(), "second question") {
		t.Errorf("the second run sent %d messages ending %q, want the first exchange and the new question",
			len(second.Messages), second.lastText())
	}
}

func TestCLIReadOnlyOffersNothingThatWrites(t *testing.T) {
	p, url := newProvider(t, func(int, wireRequest) any { return say("looked") })
	if r := cli(t, url, "-working-dir", t.TempDir(), "-read-only", "look"); r.code != exitOK {
		t.Fatalf("exit %d; stderr:\n%s", r.code, r.stderr)
	}
	names := p.requests()[0].toolNames()
	for _, banned := range []string{"bash", "write_file", "str_replace"} {
		if slices.Contains(names, banned) {
			t.Errorf("-read-only offered %s: %v", banned, names)
		}
	}
	if !slices.Contains(names, "view_file") {
		t.Errorf("-read-only offered no view_file: %v", names)
	}
}

func TestCLICheckSendsTheModelBackUntilTheWorkIsDone(t *testing.T) {
	work := t.TempDir()
	p, url := newProvider(t, func(n int, req wireRequest) any {
		switch n {
		case 0:
			return say("finished") // it is not: done.txt does not exist
		case 1:
			return call("t1", "bash", map[string]string{"command": "touch done.txt", "exec_dir": work})
		default:
			return say("finished now")
		}
	})
	r := cli(t, url, "-working-dir", work, "-check", "test -f done.txt", "make done.txt")
	if r.code != exitOK || r.last()["answer"] != "finished now" {
		t.Fatalf("exit %d, last %v; stderr:\n%s", r.code, r.last(), r.stderr)
	}
	feedback := p.requests()[1].lastText()
	if !strings.Contains(feedback, "not finished") || !strings.Contains(feedback, "test -f done.txt") {
		t.Errorf("the model was sent back with %q, want the check's verdict", feedback)
	}
	checks := r.ofType("completion_checked")
	if len(checks) != 2 || checks[0]["accepted"] == true || checks[1]["accepted"] != true {
		t.Errorf("completion_checked events = %v, want a rejection and then an acceptance", checks)
	}
}

func TestCLIOffersAnMCPServersTools(t *testing.T) {
	config := filepath.Join(t.TempDir(), "mcp.json")
	body := fmt.Sprintf(`{"mcpServers": {"greeter": {"command": %q, "env": {"WOMBAT_JSONL_AS": "mcp", "GORACE": "atexit_sleep_ms=0"}}}}`, os.Args[0])
	if err := os.WriteFile(config, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	p, url := newProvider(t, func(n int, req wireRequest) any {
		if n == 0 {
			return call("m1", "greeter__greet", map[string]string{"name": "wombat"})
		}
		return say("the server said: " + strings.TrimSpace(req.lastText()))
	})
	r := cli(t, url, "-working-dir", t.TempDir(), "-mcp", config, "greet")
	if r.code != exitOK {
		t.Fatalf("exit %d; stderr:\n%s", r.code, r.stderr)
	}
	first := p.requests()[0]
	if !slices.Contains(first.toolNames(), "greeter__greet") {
		t.Errorf("tools = %v, want the server's", first.toolNames())
	}
	if !strings.Contains(string(first.System), "Greet people by name.") {
		t.Errorf("the server's instructions are not in the system prompt: %s", first.System)
	}
	if got := r.last()["answer"]; got != "the server said: hello, wombat" {
		t.Errorf("answer = %v", got)
	}

	bad := cli(t, url, "-working-dir", t.TempDir(), "-mcp", filepath.Join(t.TempDir(), "missing.json"), "greet")
	if bad.code != exitFail || bad.last()["type"] != "failed" {
		t.Errorf("a missing -mcp file: exit %d, last %v", bad.code, bad.last())
	}
}

// serveMCP is the smallest stdio MCP server the -mcp test needs: one tool,
// and instructions for the system prompt.
func serveMCP(in io.Reader, out io.Writer) {
	sc := bufio.NewScanner(in)
	enc := json.NewEncoder(out)
	for sc.Scan() {
		var m struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params struct {
				Arguments struct {
					Name string `json:"name"`
				} `json:"arguments"`
			} `json:"params"`
		}
		if json.Unmarshal(sc.Bytes(), &m) != nil || len(m.ID) == 0 {
			continue // notifications
		}
		var result any
		switch m.Method {
		case "initialize":
			result = map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{"tools": map[string]any{}},
				"serverInfo": map[string]any{"name": "greeter", "version": "0"}, "instructions": "Greet people by name."}
		case "tools/list":
			result = map[string]any{"tools": []any{map[string]any{
				"name": "greet", "description": "Greet someone.",
				"inputSchema": map[string]any{"type": "object", "properties": map[string]any{"name": map[string]string{"type": "string"}}},
				"annotations": map[string]any{"readOnlyHint": true, "openWorldHint": false},
			}}}
		case "tools/call":
			result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "hello, " + m.Params.Arguments.Name}}}
		default:
			_ = enc.Encode(map[string]any{"jsonrpc": "2.0", "id": m.ID, "error": map[string]any{"code": -32601, "message": "no"}})
			continue
		}
		_ = enc.Encode(map[string]any{"jsonrpc": "2.0", "id": m.ID, "result": result})
	}
}
