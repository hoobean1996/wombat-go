package eval

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/permission"
	"github.com/hoobean1996/wombat-go/tool"
)

var quiet = slog.New(slog.NewTextHandler(io.Discard, nil))

func TestMain(m *testing.M) {
	slog.SetDefault(quiet)
	os.Exit(m.Run())
}

// replay answers from the transcript rather than from a fixed script, because
// samples of different arms run at once on the one client: a fresh prompt is
// answered with a call to note, and a tool result with the final answer.
func replay(_ context.Context, req llm.Request) (llm.Response, error) {
	if hasResult(req.Messages[len(req.Messages)-1]) {
		return llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: "done"}},
			StopReason: llm.StopEndTurn,
			Usage:      llm.Usage{InputTokens: 120, OutputTokens: 5},
		}, nil
	}
	return llm.Response{
		Content:    []llm.ContentBlock{llm.ToolUse{ID: "u1", Name: "note", Input: json.RawMessage(`{"text":"hello"}`)}},
		StopReason: llm.StopToolUse,
		Usage:      llm.Usage{InputTokens: 100, OutputTokens: 10},
	}, nil
}

func hasResult(m llm.Message) bool {
	for _, b := range m.Content {
		if _, ok := b.(llm.ToolResult); ok {
			return true
		}
	}
	return false
}

// note writes into the sample's workspace, so a test can tell the workspace
// it was given is the one it ran in.
func note(dir string) tool.Def {
	return tool.Def{
		Name:        "note",
		Description: "Write a note.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
		Caps:        tool.CapMutating,
		Fn: func(_ context.Context, in json.RawMessage) (string, error) {
			return "noted", os.WriteFile(filepath.Join(dir, "note.json"), in, 0o644)
		},
	}
}

var spare = tool.Def{
	Name:        "spare",
	Description: "Never called.",
	InputSchema: json.RawMessage(`{"type":"object"}`),
	Caps:        tool.CapReadOnly,
	Fn:          func(context.Context, json.RawMessage) (string, error) { return "unused", nil },
}

func arm(name string, opts []wombat.Option, extra ...tool.Def) Arm {
	return Arm{Name: name, Build: func(d Deps) (*wombat.Agent, error) {
		base := []wombat.Option{
			wombat.WithClient(d.Client),
			wombat.WithModel(d.Model),
			wombat.WithLogger(quiet),
			wombat.WithTools(append([]tool.Def{note(d.Workspace)}, extra...)...),
		}
		return wombat.New(append(base, opts...)...)
	}}
}

func entries(t *testing.T, dir string) []string {
	t.Helper()
	es, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range es {
		names = append(names, e.Name())
	}
	return names
}

func TestRunFoldsEverySampleOfEveryArm(t *testing.T) {
	dir := t.TempDir()
	var progress atomic.Int32
	var ranIn atomic.Int32
	p := &Probe{
		ID:     "contrast",
		Prompt: "write a note",
		Setup: func(dir string) error {
			return os.WriteFile(filepath.Join(dir, "seed.txt"), []byte("x"), 0o644)
		},
		Arms: []Arm{arm("wide", nil, spare), arm("narrow", nil)},
		Expect: []Expect{
			{Name: "note is the first call", Kind: Invariant, Check: func(r Runs) Verdict {
				for _, name := range []string{"wide", "narrow"} {
					for _, o := range r.Arm(name) {
						if o.FirstIndex("note") != 0 {
							return Verdict{Detail: name + " did not call note first"}
						}
					}
				}
				return Verdict{Pass: true, Detail: "6/6"}
			}},
			{Name: "wide offers more", Kind: Contrast, Check: func(r Runs) Verdict {
				w, n := r.Arm("wide")[0].Offered[0], r.Arm("narrow")[0].Offered[0]
				return Verdict{Pass: w == n+1, Detail: "offered"}
			}},
		},
	}
	cfg := Config{
		Client: llm.ClientFunc(replay), Model: "test-model", Samples: 3, Workers: 4, Dir: dir,
		Progress: func(probe, arm string, sample int, o Obs) {
			progress.Add(1)
			if o.Calls[0].OK {
				ranIn.Add(1)
			}
		},
	}
	results := Run(context.Background(), cfg, []*Probe{p})
	if len(results) != 1 {
		t.Fatalf("%d results, want 1", len(results))
	}
	r := results[0]
	if r.Err != nil {
		t.Fatal(r.Err)
	}
	if !r.Pass() || len(r.Verdicts) != 2 {
		t.Fatalf("verdicts = %+v, want both to pass", r.Verdicts)
	}
	if r.Verdicts[1].Kind != Contrast || r.Verdicts[1].Kind.String() != "contrast" {
		t.Errorf("verdict kinds = %v, %v", r.Verdicts[0].Kind, r.Verdicts[1].Kind)
	}
	if got := progress.Load(); got != 6 {
		t.Errorf("Progress called %d times, want 6", got)
	}
	if got := ranIn.Load(); got != 6 {
		t.Errorf("note succeeded in %d samples, want 6 — the workspace it was given must exist", got)
	}

	for _, name := range []string{"wide", "narrow"} {
		obs := r.Runs.Arm(name)
		if len(obs) != 3 {
			t.Fatalf("%s: %d samples, want 3", name, len(obs))
		}
		for i, o := range obs {
			if o.Sample != i || o.Arm != name {
				t.Errorf("%s[%d] is sample %d of %q: want them in sample order", name, i, o.Sample, o.Arm)
			}
			if o.LLMCalls != 2 || len(o.Offered) != 2 || o.Iters != 2 {
				t.Errorf("%s[%d]: %d calls, offered %v, %d iterations; want 2, two counts, 2", name, i, o.LLMCalls, o.Offered, o.Iters)
			}
			if o.Usage != (Usage{In: 220, Out: 15}) {
				t.Errorf("%s[%d]: usage = %+v, want the two calls summed", name, i, o.Usage)
			}
			if o.Outcome != "answer" || o.Answer != "done" || o.Err != "" {
				t.Errorf("%s[%d]: outcome %q, answer %q, err %q", name, i, o.Outcome, o.Answer, o.Err)
			}
			if len(o.Calls) != 1 {
				t.Fatalf("%s[%d]: calls = %+v", name, i, o.Calls)
			}
			c := o.Calls[0]
			if c.Name != "note" || !c.OK || c.Out != "noted" || c.Turn != 1 || c.ArgString("text") != "hello" {
				t.Errorf("%s[%d]: call = %+v", name, i, c)
			}
			if len(o.Messages) != 4 || o.Wall <= 0 {
				t.Errorf("%s[%d]: %d messages, wall %v", name, i, len(o.Messages), o.Wall)
			}
		}
	}
	if left := entries(t, dir); len(left) != 0 {
		t.Errorf("workspaces left behind after a passing run: %v", left)
	}
}

func TestAFailedSampleIsDataAndKeepsItsWorkspace(t *testing.T) {
	dir := t.TempDir()
	down := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		return llm.Response{}, errors.New("provider down")
	})
	p := &Probe{ID: "broken", Prompt: "anything", Arms: []Arm{arm("down", []wombat.Option{wombat.WithClient(down)})}}
	r := Run(context.Background(), Config{Client: llm.ClientFunc(replay), Samples: 1, Dir: dir}, []*Probe{p})[0]
	if r.Err != nil {
		t.Fatalf("Err = %v: a sample that failed is an answer, not a broken probe", r.Err)
	}
	o := r.Runs.Arm("down")[0]
	if !strings.Contains(o.Err, "provider down") || o.Outcome != "none" {
		t.Errorf("obs = %q / %q, want the failure recorded", o.Err, o.Outcome)
	}
	if left := entries(t, dir); len(left) != 1 {
		t.Errorf("workspaces = %v, want the failed sample's kept", left)
	}
}

func TestProbesThatCannotBeEvaluated(t *testing.T) {
	cfg := func(dir string) Config { return Config{Client: llm.ClientFunc(replay), Samples: 1, Dir: dir} }

	t.Run("no arms", func(t *testing.T) {
		r := Run(context.Background(), cfg(t.TempDir()), []*Probe{{ID: "empty"}})[0]
		if r.Err == nil || r.Pass() {
			t.Errorf("Err = %v, Pass = %v", r.Err, r.Pass())
		}
	})
	t.Run("setup fails", func(t *testing.T) {
		dir := t.TempDir()
		p := &Probe{ID: "setup", Arms: []Arm{arm("a", nil)},
			Setup: func(string) error { return errors.New("no disk") }}
		r := Run(context.Background(), cfg(dir), []*Probe{p})[0]
		if r.Err == nil || !strings.Contains(r.Err.Error(), "setup: no disk") {
			t.Errorf("Err = %v", r.Err)
		}
		if len(entries(t, dir)) != 1 {
			t.Error("the workspace a setup failed in was removed")
		}
	})
	t.Run("build fails", func(t *testing.T) {
		p := &Probe{ID: "build", Arms: []Arm{{Name: "a", Build: func(Deps) (*wombat.Agent, error) {
			return nil, errors.New("no agent")
		}}}}
		r := Run(context.Background(), cfg(t.TempDir()), []*Probe{p})[0]
		if r.Err == nil || !strings.Contains(r.Err.Error(), "building the agent: no agent") {
			t.Errorf("Err = %v", r.Err)
		}
	})
	t.Run("expensive probes run only when asked", func(t *testing.T) {
		p := &Probe{ID: "slow", Prompt: "x", Expensive: true, Arms: []Arm{arm("a", nil)}}
		if got := Run(context.Background(), cfg(t.TempDir()), []*Probe{p}); len(got) != 0 {
			t.Errorf("ran an expensive probe nobody asked for: %d results", len(got))
		}
		c := cfg(t.TempDir())
		c.Expensive = true
		if got := Run(context.Background(), c, []*Probe{p}); len(got) != 1 || got[0].Err != nil {
			t.Errorf("Expensive: %d results", len(got))
		}
	})
}

// TestProbeMaxItersCapsEveryArm pins the field doing something: it used to be
// declared, documented, and read by nothing, so a probe that set it ran on
// each agent's default cap.
func TestProbeMaxItersCapsEveryArm(t *testing.T) {
	forever := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		return llm.Response{
			Content:    []llm.ContentBlock{llm.ToolUse{ID: "u", Name: "note", Input: json.RawMessage(`{"text":"again"}`)}},
			StopReason: llm.StopToolUse,
		}, nil
	})
	p := &Probe{ID: "capped", Prompt: "loop", MaxIters: 3, Arms: []Arm{arm("a", nil), arm("b", nil)}}
	r := Run(context.Background(), Config{Client: forever, Samples: 1, Dir: t.TempDir()}, []*Probe{p})[0]
	if r.Err != nil {
		t.Fatal(r.Err)
	}
	for _, name := range []string{"a", "b"} {
		o := r.Runs.Arm(name)[0]
		if o.Iters != 3 || o.LLMCalls != 3 || o.Err == "" {
			t.Errorf("%s: %d iterations, %d calls, err %q; want the run stopped at 3", name, o.Iters, o.LLMCalls, o.Err)
		}
	}
}

func TestObserveRecordsPermissionVerdictsAndRefusedCalls(t *testing.T) {
	gate := wombat.WithToolMiddleware(permission.Gate(permission.ReadOnly(), nil))
	p := &Probe{ID: "perm", Prompt: "write a note", Arms: []Arm{arm("gated", []wombat.Option{gate})}}
	r := Run(context.Background(), Config{Client: llm.ClientFunc(replay), Samples: 1, Dir: t.TempDir()}, []*Probe{p})[0]
	if r.Err != nil {
		t.Fatal(r.Err)
	}
	o := r.Runs.Arm("gated")[0]
	if len(o.Perms) != 1 || o.Perms[0].Tool != "note" || o.Perms[0].Allowed || o.Perms[0].Source == "" || o.Perms[0].Turn != 1 {
		t.Errorf("perms = %+v, want one refusal of note", o.Perms)
	}
	if len(o.Calls) != 1 || o.Calls[0].OK || o.Calls[0].Err == "" {
		t.Errorf("calls = %+v, want the refused call recorded as failed", o.Calls)
	}
}

func TestObserveCountsWhatADelegateStreams(t *testing.T) {
	client := llm.ClientFunc(func(_ context.Context, req llm.Request) (llm.Response, error) {
		last := req.Messages[len(req.Messages)-1]
		switch {
		case hasResult(last):
			return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: "done"}}, StopReason: llm.StopEndTurn}, nil
		case llm.TextOf(req.Messages[0].Content) == "look around":
			return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: "found it"}}, StopReason: llm.StopEndTurn}, nil
		}
		return llm.Response{
			Content:    []llm.ContentBlock{llm.ToolUse{ID: "d1", Name: wombat.DefaultDelegateName, Input: json.RawMessage(`{"task":"look around"}`)}},
			StopReason: llm.StopToolUse,
		}, nil
	})
	parent := Arm{Name: "delegating", Build: func(d Deps) (*wombat.Agent, error) {
		child, err := wombat.New(wombat.WithName("child"), wombat.WithClient(d.Client), wombat.WithLogger(quiet))
		if err != nil {
			return nil, err
		}
		return wombat.New(wombat.WithClient(d.Client), wombat.WithLogger(quiet), wombat.WithTools(wombat.DelegateTool(child)))
	}}
	p := &Probe{ID: "delegate", Prompt: "find something", Arms: []Arm{parent}}
	r := Run(context.Background(), Config{Client: client, Samples: 1, Dir: t.TempDir()}, []*Probe{p})[0]
	if r.Err != nil {
		t.Fatal(r.Err)
	}
	o := r.Runs.Arm("delegating")[0]
	if len(o.Subs) != 1 || !o.Subs[0].OK || o.Subs[0].Depth != 1 || o.Subs[0].Events == 0 {
		t.Errorf("subs = %+v, want one finished child whose events reached the parent", o.Subs)
	}
	if o.Answer != "done" {
		t.Errorf("answer = %q", o.Answer)
	}
}

func TestObsHelpers(t *testing.T) {
	o := Obs{Calls: []Call{
		{Name: "a", Input: json.RawMessage(`{"s":"x","n":1}`)},
		{Name: "b"},
		{Name: "a", Input: json.RawMessage(`not json`)},
	}}
	if o.FirstIndex("b") != 1 || o.FirstIndex("z") != -1 {
		t.Errorf("FirstIndex = %d, %d", o.FirstIndex("b"), o.FirstIndex("z"))
	}
	if got := o.CallsTo("a", "b"); len(got) != 3 {
		t.Errorf("CallsTo(a, b) = %d calls, want 3", len(got))
	}
	c := o.Calls[0]
	if c.ArgString("s") != "x" || c.ArgString("n") != "" || c.ArgString("missing") != "" {
		t.Errorf("ArgString: %q %q %q", c.ArgString("s"), c.ArgString("n"), c.ArgString("missing"))
	}
	if o.Calls[1].ArgString("s") != "" || o.Calls[2].ArgString("s") != "" {
		t.Error("ArgString read something out of no input, or out of input that is not JSON")
	}
}
