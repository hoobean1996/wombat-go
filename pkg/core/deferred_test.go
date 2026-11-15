package core

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

func toolsTurn(uses ...llm.ToolUse) llm.Response {
	blocks := make([]llm.ContentBlock, len(uses))
	for i, u := range uses {
		blocks[i] = u
	}
	return llm.Response{Content: blocks, StopReason: llm.StopToolUse}
}

// deferWrites is the smallest gate there is: a write runs only once a person
// has decided it may, and is deferred until then.
func deferWrites(next tool.Handler) tool.Handler {
	return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
		if d.Name != "write" {
			return next(ctx, d, use)
		}
		if allow, decided := tool.Decision(ctx, use.ID); decided {
			if !allow {
				return "", errors.New("refused by the person")
			}
			return next(ctx, d, use)
		}
		return "", tool.Defer("a write needs a person")
	}
}

type writes struct{ n atomic.Int64 }

func (w *writes) tool() tool.Def {
	return tool.Def{
		Name: "write", Description: "writes", InputSchema: json.RawMessage(objSchema),
		Fn: func(context.Context, json.RawMessage) (string, error) {
			w.n.Add(1)
			return "written", nil
		},
	}
}

var (
	readCall  = llm.ToolUse{ID: "r1", Name: "read", Input: json.RawMessage(`{"path":"a"}`)}
	writeCall = llm.ToolUse{ID: "w1", Name: "write", Input: json.RawMessage(`{"path":"b"}`)}
)

// deferredRun runs a turn that reads and writes at once, and returns where it
// stopped.
func deferredRun(t *testing.T, cl llm.Client, w *writes) (*Agent, []llm.Message, Deferred) {
	t.Helper()
	a := newAgent(t, cl, []tool.Def{echoTool("read", "the contents of a"), w.tool()},
		WithToolMiddleware(deferWrites))
	r := a.Start(context.Background(), Ask("copy a to b"))
	drain(t, r)
	d, ok := r.Outcome().(Deferred)
	if !ok {
		t.Fatalf("outcome = %#v (err %v), want Deferred", r.Outcome(), r.Err())
	}
	return a, r.Messages(), d
}

func TestADeferredCallEndsTheRunAndKeepsWhatRan(t *testing.T) {
	w := &writes{}
	_, msgs, d := deferredRun(t, scripted(toolsTurn(readCall, writeCall)), w)

	want := Deferred{
		Calls: []DeferredCall{{UseID: "w1", Tool: "write", Input: writeCall.Input, Reason: "a write needs a person"}},
		Done:  []llm.ToolResult{{ToolUseID: "r1", Content: "the contents of a"}},
	}
	if !reflect.DeepEqual(d, want) {
		t.Errorf("got  %#v\nwant %#v", d, want)
	}
	if n := w.n.Load(); n != 0 {
		t.Errorf("the write ran %d times before anyone decided", n)
	}
	if last := msgs[len(msgs)-1]; last.Role != llm.RoleAssistant || len(llm.ToolUses(last.Content)) != 2 {
		t.Errorf("the transcript does not end on the deferred turn: %+v", last)
	}

	// And it keeps across a process boundary.
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	var back Deferred
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, d) {
		t.Errorf("round trip: %s\ngot  %#v\nwant %#v", b, back, d)
	}
}

// Approved, the call runs for real — its output, not the approval, is what
// the model reads — and both calls of the turn are answered in one message,
// in the order the model made them.
func TestDecideRunsAnApprovedCall(t *testing.T) {
	w := &writes{}
	cl := scripted(toolsTurn(readCall, writeCall), textTurn("copied"))
	a, msgs, d := deferredRun(t, cl, w)

	r := a.Start(context.Background(), Decide(msgs, d, map[llm.ToolUseID]bool{"w1": true}))
	drain(t, r)
	if ans, ok := r.Outcome().(Answer); !ok || ans.Text != "copied" {
		t.Fatalf("outcome = %#v (err %v), want the answer", r.Outcome(), r.Err())
	}
	if n := w.n.Load(); n != 1 {
		t.Errorf("the write ran %d times, want once", n)
	}
	last := cl.request(t, 1).Messages
	results := last[len(last)-1].Content
	if len(results) != 2 {
		t.Fatalf("the model was sent %d results, want 2: %+v", len(results), results)
	}
	if r0 := results[0].(llm.ToolResult); r0.ToolUseID != "r1" || r0.Content != "the contents of a" {
		t.Errorf("first result = %+v, want the read's, kept from before the stop", r0)
	}
	if r1 := results[1].(llm.ToolResult); r1.ToolUseID != "w1" || r1.Content != "written" || r1.IsError {
		t.Errorf("second result = %+v, want the write's real output", r1)
	}
}

func TestDecideAnswersARefusedCallWithTheRefusal(t *testing.T) {
	w := &writes{}
	cl := scripted(toolsTurn(readCall, writeCall), textTurn("I could not copy it"))
	a, msgs, d := deferredRun(t, cl, w)

	r := a.Start(context.Background(), Decide(msgs, d, map[llm.ToolUseID]bool{"w1": false}))
	drain(t, r)
	if _, ok := r.Outcome().(Answer); !ok {
		t.Fatalf("outcome = %#v (err %v), want an answer", r.Outcome(), r.Err())
	}
	if n := w.n.Load(); n != 0 {
		t.Errorf("a refused write ran %d times", n)
	}
	last := cl.request(t, 1).Messages
	if r1 := last[len(last)-1].Content[1].(llm.ToolResult); !r1.IsError || !strings.Contains(r1.Content, "refused") {
		t.Errorf("the refused call's result = %+v, want the refusal", r1)
	}
}

// A call nobody has decided is asked about again, and defers again; what ran
// before the first stop is still kept.
func TestDecideWithoutAVerdictDefersAgain(t *testing.T) {
	w := &writes{}
	a, msgs, d := deferredRun(t, scripted(toolsTurn(readCall, writeCall)), w)

	r := a.Start(context.Background(), Decide(msgs, d, nil))
	drain(t, r)
	again, ok := r.Outcome().(Deferred)
	if !ok {
		t.Fatalf("outcome = %#v (err %v), want Deferred again", r.Outcome(), r.Err())
	}
	if !reflect.DeepEqual(again, d) {
		t.Errorf("got  %#v\nwant the same deferral %#v", again, d)
	}
}

// A sub-agent cannot wait for a person: its deferral reaches the parent as a
// result that says what the child was about to do.
func TestASubagentsDeferralReachesTheParent(t *testing.T) {
	w := &writes{}
	child := newAgent(t, scripted(toolsTurn(writeCall)), []tool.Def{w.tool()},
		WithName("child"), WithToolMiddleware(deferWrites))
	parentClient := scripted(
		toolTurn("d1", "delegate", `{"task":"write b"}`),
		textTurn("the child could not"),
	)
	parent := newAgent(t, parentClient, []tool.Def{DelegateTool(child)})
	if _, err := parent.Run(context.Background(), Ask("go")); err != nil {
		t.Fatal(err)
	}
	last := parentClient.request(t, 1).Messages
	res := last[len(last)-1].Content[0].(llm.ToolResult)
	if !res.IsError || !strings.Contains(res.Content, "need a person's approval") || !strings.Contains(res.Content, "write") {
		t.Errorf("the delegate call's result = %+v, want the child's deferral explained", res)
	}
	if n := w.n.Load(); n != 0 {
		t.Errorf("the child's write ran %d times", n)
	}
}
