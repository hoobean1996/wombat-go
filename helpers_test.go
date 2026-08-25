package wombat

// White-box tests (package wombat, not wombat_test).
//
// The loop's interesting behaviour lives in unexported code — renderSystem,
// toolChoice, withNotice, observationsIn, the overflow state — and several of
// the regressions being pinned here are about exactly those. Testing from
// outside would mean asserting on them indirectly through the public surface,
// which is how a regression test stops describing the bug it defends.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// ===== the one scripted client =====

// scriptedClient replays canned responses in order and records what it was
// asked. Every test in this package uses it; there is no second fake.
//
// Safe for concurrent use because sub-agent fan-out shares one client across
// goroutines, and these tests run under -race.
type scriptedClient struct {
	mu    sync.Mutex
	turns []llm.Response
	i     int
	seen  []llm.Request
}

func scripted(turns ...llm.Response) *scriptedClient {
	return &scriptedClient{turns: turns}
}

// Complete implements llm.Client.
func (s *scriptedClient) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	s.mu.Lock()
	s.seen = append(s.seen, req)
	if s.i >= len(s.turns) {
		n := s.i
		s.mu.Unlock()
		return llm.Response{}, fmt.Errorf("scriptedClient: out of turns after %d calls", n)
	}
	r := s.turns[s.i]
	s.i++
	s.mu.Unlock()

	// Stream the reply's text the way a real client does, so the loop's delta
	// plumbing is exercised rather than bypassed.
	if req.OnDelta != nil {
		if t := llm.TextOf(r.Content); t != "" {
			req.OnDelta(llm.Delta{Text: t})
		}
	}
	return r, nil
}

func (s *scriptedClient) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.i
}

// requests returns a snapshot of every request the client has seen.
func (s *scriptedClient) requests() []llm.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.Request(nil), s.seen...)
}

// request returns the n-th request (0-based), failing the test if there is no
// such call.
func (s *scriptedClient) request(t *testing.T, n int) llm.Request {
	t.Helper()
	got := s.requests()
	if n >= len(got) {
		t.Fatalf("wanted request %d, client only saw %d", n, len(got))
	}
	return got[n]
}

// ===== response builders =====

func textTurn(s string) llm.Response {
	return llm.Response{
		Content:    []llm.ContentBlock{llm.Text{Text: s}},
		StopReason: llm.StopEndTurn,
	}
}

func toolTurn(id, name, input string) llm.Response {
	return llm.Response{
		Content: []llm.ContentBlock{
			llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(input)},
		},
		StopReason: llm.StopToolUse,
	}
}

// ===== agent construction =====

var quietLogger = slog.New(slog.NewTextHandler(io.Discard, nil))

// TestMain silences the package default logger. Not every diagnostic goes
// through the agent's own logger — the overflow middleware logs its escalation
// with slog.WarnContext — and a passing test should not print anything.
func TestMain(m *testing.M) {
	slog.SetDefault(quietLogger)
	os.Exit(m.Run())
}

// newAgent builds an agent with a discarding logger so a failing test's output
// is the assertion and not a wall of slog lines.
func newAgent(t *testing.T, cl llm.Client, tools []tool.Def, opts ...Option) *Agent {
	t.Helper()
	base := []Option{
		WithClient(cl),
		WithModel("test-model"),
		WithLogger(quietLogger),
		WithTools(tools...),
	}
	a, err := New(append(base, opts...)...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

// ===== tools =====

const objSchema = `{"type":"object"}`

// echoTool returns a fixed string, and records that it ran.
func echoTool(name, out string) tool.Def {
	return tool.Def{
		Name: name, Description: "echoes", InputSchema: json.RawMessage(objSchema),
		Fn: func(context.Context, json.RawMessage) (string, error) { return out, nil },
	}
}

func panicTool(name, msg string) tool.Def {
	return tool.Def{
		Name: name, Description: "panics", InputSchema: json.RawMessage(objSchema),
		Fn: func(context.Context, json.RawMessage) (string, error) { panic(msg) },
	}
}

// slowTool blocks until its own timeout fires, then reports partial output the
// way a killed subprocess does.
func slowTool(name string, timeout time.Duration) tool.Def {
	return tool.Def{
		Name: name, Description: "sleeps", InputSchema: json.RawMessage(objSchema),
		Timeout: timeout,
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			select {
			case <-time.After(30 * time.Second):
				return "finished", nil
			case <-ctx.Done():
				return "", errors.New("aborted with partial output: abc")
			}
		},
	}
}

func terminalTool(name string) tool.Def {
	return tool.Def{
		Name: name, Description: "submits", InputSchema: json.RawMessage(objSchema),
		Caps: tool.CapTerminal,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			return "", errors.New("terminal tool handler must never run")
		},
	}
}

func pauseTool(name string) tool.Def {
	return tool.Def{
		Name: name, Description: "asks the user", InputSchema: json.RawMessage(objSchema),
		Caps: tool.CapPause,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			return "", errors.New("pause tool handler must never run")
		},
	}
}

// ===== draining a run =====

// drain iterates a run to completion, returning the event kinds in order and
// the events themselves.
func drain(t *testing.T, r *Run) ([]string, []Event) {
	t.Helper()
	var kinds []string
	var evs []Event
	for r.Next() {
		kinds = append(kinds, r.Event().Kind())
		evs = append(evs, r.Event())
	}
	return kinds, evs
}

// ===== transcript inspection =====

// allText flattens every Text and ToolResult body in a transcript, which is
// what "did the model see X" reduces to.
func allText(msgs []llm.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		for _, blk := range m.Content {
			switch v := blk.(type) {
			case llm.Text:
				b.WriteString(v.Text)
				b.WriteString("\n")
			case llm.ToolResult:
				b.WriteString(v.Content)
				b.WriteString("\n")
			}
		}
	}
	return b.String()
}

func specNames(ts []llm.ToolSpec) []string {
	out := make([]string, 0, len(ts))
	for _, t := range ts {
		out = append(out, t.Name)
	}
	return out
}

func toolDefNames(defs []tool.Def) []string {
	out := make([]string, 0, len(defs))
	for _, d := range defs {
		out = append(out, d.Name)
	}
	return out
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

// ===== transcript builders =====

func userMsg(s string) llm.Message { return llm.UserText(s) }

func assistantText(s string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.Text{Text: s}}}
}

func assistantUse(id, name string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(objSchema)},
	}}
}

func userResult(id, body string) llm.Message {
	return llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{
		llm.ToolResult{ToolUseID: llm.ToolUseID(id), Content: body},
	}}
}

// longTranscript builds n alternating plain turns, starting with a user turn.
func longTranscript(n int) []llm.Message {
	msgs := make([]llm.Message, 0, n)
	for i := range n {
		if i%2 == 0 {
			msgs = append(msgs, userMsg(fmt.Sprintf("question %d", i)))
		} else {
			msgs = append(msgs, assistantText(fmt.Sprintf("answer %d", i)))
		}
	}
	return msgs
}
