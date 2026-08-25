package rl

// White-box tests (package rl). The interesting behaviour is in unexported
// code — the event folder and the failure classifier — and asserting on them
// through Rollout alone would mean every case needs a whole scripted run to
// reach it.

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"sync"
	"testing"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

var quiet = slog.New(slog.NewTextHandler(io.Discard, nil))

// TestMain silences the default logger: the verifiers and the cleanup path log
// on purpose, and a passing test should print nothing.
func TestMain(m *testing.M) {
	slog.SetDefault(quiet)
	os.Exit(m.Run())
}

// ===== the fake client =====
//
// A function of the request rather than a queue of canned turns, because the
// isolation test runs eight agents against ONE client concurrently and a
// shared cursor would hand sample 3's reply to sample 5.

// turnClient answers based on how far into the conversation the request is,
// which is a property of the request itself and therefore per-run.
func turnClient(f func(turn int, req llm.Request) llm.Response) llm.Client {
	return llm.ClientFunc(func(_ context.Context, req llm.Request) (llm.Response, error) {
		return f(assistantTurns(req.Messages), req), nil
	})
}

// failingClient always fails with err.
func failingClient(err error) llm.Client {
	return llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		return llm.Response{}, err
	})
}

// assistantTurns counts the assistant messages already in the transcript,
// which is the 0-based index of the reply about to be produced.
func assistantTurns(msgs []llm.Message) int {
	n := 0
	for _, m := range msgs {
		if m.Role == llm.RoleAssistant {
			n++
		}
	}
	return n
}

func textTurn(s string) llm.Response {
	return llm.Response{
		Content:    []llm.ContentBlock{llm.Text{Text: s}},
		StopReason: llm.StopEndTurn,
		Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
	}
}

func toolTurn(id, name, input string) llm.Response {
	return llm.Response{
		Content: []llm.ContentBlock{
			llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(input)},
		},
		StopReason: llm.StopToolUse,
		Usage:      llm.Usage{InputTokens: 20, OutputTokens: 7},
	}
}

const objSchema = `{"type":"object"}`

func fnTool(name string, fn func(context.Context, json.RawMessage) (string, error)) tool.Def {
	return tool.Def{
		Name: name, Description: name, InputSchema: json.RawMessage(objSchema), Fn: fn,
	}
}

func newAgent(t *testing.T, cl llm.Client, tools []tool.Def, opts ...wombat.Option) *wombat.Agent {
	t.Helper()
	base := []wombat.Option{
		wombat.WithClient(cl),
		wombat.WithModel("test-model"),
		wombat.WithLogger(quiet),
		wombat.WithTools(tools...),
	}
	a, err := wombat.New(append(base, opts...)...)
	if err != nil {
		t.Fatalf("wombat.New: %v", err)
	}
	return a
}

// ===== a trivial in-memory Env =====

// memEnv scores whatever it is told to and records what it was asked to do,
// so a test can assert on Reset/Cleanup without touching a disk.
type memEnv struct {
	name    string
	prompt  string
	reset   func(sample int) (Task, error)
	scoreFn func(ep *Episode) (float64, map[string]float64, error)

	mu       sync.Mutex
	cleaned  []int
	keptFlag map[int]bool
}

func newMemEnv(name, prompt string) *memEnv {
	return &memEnv{name: name, prompt: prompt, keptFlag: map[int]bool{}}
}

func (e *memEnv) Name() string { return e.name }

func (e *memEnv) Reset(_ context.Context, sample int) (Task, error) {
	if e.reset != nil {
		return e.reset(sample)
	}
	return Task{ID: e.name, Sample: sample, Prompt: e.prompt}, nil
}

func (e *memEnv) Score(_ context.Context, ep *Episode) (float64, map[string]float64, error) {
	if e.scoreFn != nil {
		return e.scoreFn(ep)
	}
	return 1, map[string]float64{"ok": 1}, nil
}

func (e *memEnv) Cleanup(ctx context.Context, t Task) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cleaned = append(e.cleaned, t.Sample)
	e.keptFlag[t.Sample] = Keep(ctx)
	return nil
}

func (e *memEnv) kept(sample int) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.keptFlag[sample]
}

func (e *memEnv) cleanups() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.cleaned)
}
