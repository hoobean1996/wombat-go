package wombat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// echoTaskClient answers every call with the text of the last user turn,
// prefixed. It lets a fan-out be asserted without depending on the order the
// branches happen to reach the client in.
func echoTaskClient() llm.Client {
	return llm.ClientFunc(func(_ context.Context, req llm.Request) (llm.Response, error) {
		task := ""
		for _, m := range req.Messages {
			if m.Role == llm.RoleUser {
				task = llm.TextOf(m.Content)
			}
		}
		if strings.Contains(task, "explode") {
			return llm.Response{}, errors.New("branch blew up")
		}
		return textTurn("answer to " + task), nil
	})
}

func TestDelegateToolRunsTheChildAndForwardsItsEvents(t *testing.T) {
	childCl := scripted(textTurn("the readme says it is a harness"))
	child := newAgent(t, childCl, nil, WithName("researcher"))

	parentCl := scripted(
		toolTurn("d1", "delegate", `{"task":"read the readme"}`),
		textTurn("done"),
	)
	parent := newAgent(t, parentCl, []tool.Def{DelegateTool(child)}, WithName("parent"))

	run := parent.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })
	kinds, evs := drain(t, run)

	if err := run.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
	if childCl.calls() != 1 {
		t.Errorf("child llm calls = %d, want 1", childCl.calls())
	}

	var (
		nested  int
		start   SubagentStart
		end     SubagentEnd
		hasEnd  bool
		hasBeg  bool
		innerOK bool
	)
	for _, ev := range evs {
		switch e := ev.(type) {
		case SubagentStart:
			start, hasBeg = e, true
		case SubagentEnd:
			end, hasEnd = e, true
		case SubagentEvent:
			nested++
			if e.Inner != nil && e.Inner.Kind() == "text_delta" {
				innerOK = true
			}
		}
	}
	if !hasBeg || !hasEnd {
		t.Fatalf("event kinds = %v, want subagent_start and subagent_end", kinds)
	}
	if nested == 0 {
		t.Error("no subagent_event: the child's events were not forwarded")
	}
	if !innerOK {
		t.Error("no wrapped text_delta from the child")
	}
	if start.Task != "read the readme" {
		t.Errorf("SubagentStart.Task = %q, want %q", start.Task, "read the readme")
	}
	if start.Name != "delegate" {
		t.Errorf("SubagentStart.Name = %q, want the tool's name %q", start.Name, "delegate")
	}
	if !end.OK || end.Error != "" {
		t.Errorf("SubagentEnd = %+v, want OK", end)
	}

	// The bracketing must actually bracket.
	first, last := -1, -1
	for i, k := range kinds {
		if k == "subagent_start" {
			first = i
		}
		if k == "subagent_end" {
			last = i
		}
	}
	for i, k := range kinds {
		if k == "subagent_event" && (i < first || i > last) {
			t.Errorf("subagent_event at %d falls outside subagent_start(%d)..subagent_end(%d): %v", i, first, last, kinds)
		}
	}

	// The child's answer reaches the parent's model, but the child's own
	// transcript does not pollute the parent's.
	if got := allText(parentCl.request(t, 1).Messages); !strings.Contains(got, "the readme says") {
		t.Errorf("parent's second request = %q, want the child's answer", got)
	}
	if got := allText(run.Messages()); strings.Contains(got, "read the readme\n") {
		t.Errorf("parent transcript = %q, want the child's own turns excluded", got)
	}
}

func TestDelegateToolRejectsAnEmptyTask(t *testing.T) {
	child := newAgent(t, scripted(textTurn("never")), nil)
	parentCl := scripted(
		toolTurn("d1", "delegate", `{"task":"   "}`),
		textTurn("ok, I'll do it myself"),
	)
	parent := newAgent(t, parentCl, []tool.Def{DelegateTool(child)})

	run := parent.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	var toolErr string
	_, evs := drain(t, run)
	for _, ev := range evs {
		if td, ok := ev.(ToolDone); ok && !td.OK {
			toolErr = td.Error
		}
	}
	if run.Err() != nil {
		t.Fatalf("Err() = %v, want nil", run.Err())
	}
	if !strings.Contains(toolErr, ErrEmptyTask.Error()) {
		t.Errorf("tool error = %q, want it to wrap ErrEmptyTask", toolErr)
	}
}

func TestDelegateToolReportsAChildThatPaused(t *testing.T) {
	childCl := scripted(toolTurn("p", "ask_user", `{"question":"which branch?"}`))
	child := newAgent(t, childCl, []tool.Def{pauseTool("ask_user")}, WithName("kid"))

	parentCl := scripted(
		toolTurn("d1", "delegate", `{"task":"deploy"}`),
		textTurn("I'll ask the user myself"),
	)
	parent := newAgent(t, parentCl, []tool.Def{DelegateTool(child)})

	run := parent.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	var toolErr string
	_, evs := drain(t, run)
	for _, ev := range evs {
		if td, ok := ev.(ToolDone); ok && !td.OK {
			toolErr = td.Error
		}
	}
	if run.Err() != nil {
		t.Fatalf("Err() = %v, want nil — a paused child must not kill the parent", run.Err())
	}
	if !strings.Contains(toolErr, ErrSubagentPaused.Error()) {
		t.Errorf("tool error = %q, want it to wrap ErrSubagentPaused", toolErr)
	}
	if !strings.Contains(toolErr, "which branch?") {
		t.Errorf("tool error = %q, want the child's question so the parent can answer it", toolErr)
	}
}

func TestDelegateToolPassesThroughAStructuredSubmission(t *testing.T) {
	childCl := scripted(toolTurn("s", "submit", `{"summary":"structured"}`))
	child := newAgent(t, childCl, []tool.Def{terminalTool("submit")},
		WithTerminalTool("submit"), WithName("kid"))

	parentCl := scripted(
		toolTurn("d1", "delegate", `{"task":"do it"}`),
		textTurn("done"),
	)
	parent := newAgent(t, parentCl, []tool.Def{DelegateTool(child)})

	if _, err := parent.Run(context.Background(), Ask("go")); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := allText(parentCl.request(t, 1).Messages); !strings.Contains(got, `{"summary":"structured"}`) {
		t.Errorf("parent saw %q, want the child's payload verbatim", got)
	}
}

func TestDelegateToolWrapsAChildFailure(t *testing.T) {
	// Three, because a truncated reply is now retried twice before it becomes
	// an error: the child has to actually exhaust that budget for this test to
	// be about how the PARENT reports a child failure.
	child := newAgent(t, scripted(
		llm.Response{StopReason: llm.StopMaxTokens},
		llm.Response{StopReason: llm.StopMaxTokens},
		llm.Response{StopReason: llm.StopMaxTokens},
	), nil, WithName("kid"))
	parentCl := scripted(
		toolTurn("d1", "delegate", `{"task":"do it"}`),
		textTurn("I'll retry differently"),
	)
	parent := newAgent(t, parentCl, []tool.Def{DelegateTool(child)})

	run := parent.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	var toolErr string
	_, evs := drain(t, run)
	for _, ev := range evs {
		if td, ok := ev.(ToolDone); ok && !td.OK {
			toolErr = td.Error
		}
	}
	if run.Err() != nil {
		t.Fatalf("Err() = %v, want nil", run.Err())
	}
	// The parent's model reads this string and decides what to do, so it has
	// to name which child failed and how.
	if !strings.Contains(toolErr, `sub-agent "delegate" failed`) {
		t.Errorf("tool error = %q, want it to name the child", toolErr)
	}
	if !strings.Contains(toolErr, ErrMaxTokens.Error()) {
		t.Errorf("tool error = %q, want it to carry the child's own error", toolErr)
	}
}

// The depth cap must REFUSE the delegation without aborting the run: nesting
// depth is a shape, not a resource, and a parent told "not deeper" can still
// do the job itself.
func TestSubagentDepthCapRefusesWithoutKillingTheRun(t *testing.T) {
	grandchild := newAgent(t, scripted(textTurn("too deep to ever run")), nil, WithName("grandchild"))

	midCl := scripted(
		toolTurn("dd", "delegate", `{"task":"deeper"}`),
		textTurn("could not delegate, did it myself"),
	)
	mid := newAgent(t, midCl, []tool.Def{DelegateTool(grandchild)}, WithName("mid"))

	topCl := scripted(
		toolTurn("t1", "delegate", `{"task":"go deep"}`),
		textTurn("top done"),
	)
	top := newAgent(t, topCl, []tool.Def{DelegateTool(mid)}, WithName("top"))

	ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{SubagentDepth: 1})
	defer cancel()

	out, err := top.Run(ctx, Ask("go"))
	if err != nil {
		t.Fatalf("Run: %v — the depth cap killed the parent run", err)
	}
	if _, ok := out.(Answer); !ok {
		t.Fatalf("Outcome = %T, want Answer", out)
	}
	if errors.Is(context.Cause(ctx), governor.ErrDepthLimit) {
		t.Errorf("context cause = %v, want the depth cap NOT to abort the run", context.Cause(ctx))
	}
	if got := allText(topCl.request(t, 1).Messages); !strings.Contains(got, "did it myself") {
		t.Errorf("top saw %q, want the mid agent's fallback answer", got)
	}
}

func TestTooDeepIsPhrasedAsAnInstruction(t *testing.T) {
	err := tooDeep()
	if !errors.Is(err, ErrDelegateTooDeep) {
		t.Fatalf("error = %v, want ErrDelegateTooDeep", err)
	}
	for _, want := range []string{"maximum nesting level", "Do it yourself"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %q, want it to contain %q", err, want)
		}
	}
}

// A delegated task's natural bound is the child's own iteration cap and the
// run budget; the dispatcher's 120s fallback is wrong for it. Both halves of
// that trade-off are pinned here.
func TestToolTimeoutFallbackBoundsAChild(t *testing.T) {
	slowChild := func() *Agent {
		return newAgent(t, llm.ClientFunc(func(ctx context.Context, _ llm.Request) (llm.Response, error) {
			select {
			case <-time.After(2 * time.Second):
				return textTurn("slow but done"), nil
			case <-ctx.Done():
				return llm.Response{}, ctx.Err()
			}
		}), nil)
	}

	t.Run("a fallback timeout does bound a child", func(t *testing.T) {
		parent := newAgent(t, scripted(
			toolTurn("s1", "delegate", `{"task":"slow"}`),
			textTurn("ok"),
		), []tool.Def{DelegateTool(slowChild())}, WithToolTimeoutFallback(30*time.Millisecond))

		run := parent.Start(context.Background(), Ask("go"))
		t.Cleanup(func() { _ = run.Close() })

		var toolErr string
		_, evs := drain(t, run)
		for _, ev := range evs {
			if td, ok := ev.(ToolDone); ok && !td.OK {
				toolErr = td.Error
			}
		}
		if !strings.Contains(toolErr, "timeout") {
			t.Errorf("tool error = %q, want a timeout", toolErr)
		}
	})

	t.Run("WithToolTimeoutFallback(0) lets it finish", func(t *testing.T) {
		fast := newAgent(t, llm.ClientFunc(func(_ context.Context, _ llm.Request) (llm.Response, error) {
			time.Sleep(20 * time.Millisecond)
			return textTurn("slow but done"), nil
		}), nil)
		parent := newAgent(t, scripted(
			toolTurn("s2", "delegate", `{"task":"slow"}`),
			textTurn("ok"),
		), []tool.Def{DelegateTool(fast)}, WithToolTimeoutFallback(0))

		run := parent.Start(context.Background(), Ask("go"))
		t.Cleanup(func() { _ = run.Close() })

		var out string
		_, evs := drain(t, run)
		for _, ev := range evs {
			if td, ok := ev.(ToolDone); ok && td.OK {
				out = td.Output
			}
		}
		if !strings.Contains(out, "slow but done") {
			t.Errorf("tool output = %q, want the child's answer", out)
		}
	})
}

// ===== construction =====

func TestDelegateToolConstruction(t *testing.T) {
	child := newAgent(t, scripted(), nil)

	t.Run("defaults", func(t *testing.T) {
		d := DelegateTool(child)
		if d.Name != DefaultDelegateName {
			t.Errorf("Name = %q, want %q", d.Name, DefaultDelegateName)
		}
		if d.Caps != tool.CapMeta {
			t.Errorf("Caps = %v, want tool.CapMeta", d.Caps)
		}
		if d.Idempotent {
			t.Error("Idempotent = true, want false: a delegated task can write files and spend money")
		}
		if d.Timeout != 0 {
			t.Errorf("Timeout = %v, want 0", d.Timeout)
		}
	})

	t.Run("options", func(t *testing.T) {
		d := DelegateTool(child,
			WithDelegateName("delegate_to_reviewer"),
			WithDelegateDescription("ask the reviewer"))
		if d.Name != "delegate_to_reviewer" {
			t.Errorf("Name = %q, want %q", d.Name, "delegate_to_reviewer")
		}
		if d.Description != "ask the reviewer" {
			t.Errorf("Description = %q, want the override", d.Description)
		}
	})

	t.Run("a nil child is a wiring bug reported at construction", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("DelegateTool(nil) did not panic")
			}
		}()
		DelegateTool(nil)
	})

	t.Run("ParallelDelegateTool defaults", func(t *testing.T) {
		d := ParallelDelegateTool(child)
		if d.Name != DefaultParallelDelegateName {
			t.Errorf("Name = %q, want %q", d.Name, DefaultParallelDelegateName)
		}
	})

	t.Run("ParallelDelegateTool(nil) panics", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("ParallelDelegateTool(nil) did not panic")
			}
		}()
		ParallelDelegateTool(nil)
	})

	t.Run("WithMaxBranches below 1 panics at construction", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("WithMaxBranches(0) did not panic")
			}
		}()
		WithMaxBranches(0)
	})
}

// ===== parallel fan-out =====

func TestParallelDelegateFansOut(t *testing.T) {
	child := newAgent(t, echoTaskClient(), nil, WithName("worker"))
	parentCl := scripted(
		toolTurn("p1", "parallel_delegate", `{"tasks":["alpha","beta","gamma"]}`),
		textTurn("collated"),
	)
	parent := newAgent(t, parentCl, []tool.Def{ParallelDelegateTool(child, WithMaxBranches(2))})

	run := parent.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	var out string
	names := map[string]bool{}
	_, evs := drain(t, run)
	for _, ev := range evs {
		switch e := ev.(type) {
		case ToolDone:
			if e.OK {
				out = e.Output
			}
		case SubagentStart:
			names[e.Name] = true
		}
	}
	if run.Err() != nil {
		t.Fatalf("Err() = %v, want nil", run.Err())
	}

	// Results come back in the order the tasks were listed, whatever order the
	// branches finished in.
	for i, task := range []string{"alpha", "beta", "gamma"} {
		header := fmt.Sprintf("=== task %d of 3: %s ===", i+1, task)
		if !strings.Contains(out, header) {
			t.Errorf("output is missing %q:\n%s", header, out)
		}
		if !strings.Contains(out, "answer to "+task) {
			t.Errorf("output is missing the answer for %q:\n%s", task, out)
		}
	}
	ai := strings.Index(out, "alpha")
	bi := strings.Index(out, "beta")
	gi := strings.Index(out, "gamma")
	if !(ai < bi && bi < gi) {
		t.Errorf("results are out of order (alpha@%d beta@%d gamma@%d):\n%s", ai, bi, gi, out)
	}

	// Branch events are labelled so a consumer can demultiplex the interleave.
	for i := range 3 {
		want := fmt.Sprintf("parallel_delegate[%d]", i)
		if !names[want] {
			t.Errorf("no subagent_start named %q; got %v", want, names)
		}
	}
}

// A failing branch reports its error in place; three good answers and one
// diagnosis is a result the parent model can work with, four cancellations is
// not.
func TestParallelDelegateKeepsGoodBranchesWhenOneFails(t *testing.T) {
	child := newAgent(t, echoTaskClient(), nil)
	parentCl := scripted(
		toolTurn("p1", "parallel_delegate", `{"tasks":["alpha","explode","gamma"]}`),
		textTurn("collated"),
	)
	parent := newAgent(t, parentCl, []tool.Def{ParallelDelegateTool(child)})

	run := parent.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	var done ToolDone
	_, evs := drain(t, run)
	for _, ev := range evs {
		if td, ok := ev.(ToolDone); ok {
			done = td
		}
	}
	if run.Err() != nil {
		t.Fatalf("Err() = %v, want nil", run.Err())
	}
	if !done.OK {
		t.Fatalf("tool_done OK = false (%q); a partial result set must come back as output, not as an error", done.Error)
	}
	if !strings.Contains(done.Output, "answer to alpha") || !strings.Contains(done.Output, "answer to gamma") {
		t.Errorf("output lost the successful branches:\n%s", done.Output)
	}
	if !strings.Contains(done.Output, "[failed]") {
		t.Errorf("output does not report the failed branch:\n%s", done.Output)
	}
}

func TestParallelDelegateRejectsEmptyTasks(t *testing.T) {
	child := newAgent(t, echoTaskClient(), nil)

	tests := []struct {
		name  string
		input string
	}{
		{name: "a blank entry", input: `{"tasks":["ok","  "]}`},
		{name: "no tasks at all", input: `{"tasks":[]}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			parentCl := scripted(
				llm.Response{
					Content: []llm.ContentBlock{
						llm.ToolUse{ID: "p1", Name: "parallel_delegate", Input: json.RawMessage(tc.input)},
					},
					StopReason: llm.StopToolUse,
				},
				textTurn("fine, I'll do it myself"),
			)
			parent := newAgent(t, parentCl, []tool.Def{ParallelDelegateTool(child)})

			run := parent.Start(context.Background(), Ask("go"))
			t.Cleanup(func() { _ = run.Close() })

			var toolErr string
			_, evs := drain(t, run)
			for _, ev := range evs {
				if td, ok := ev.(ToolDone); ok && !td.OK {
					toolErr = td.Error
				}
			}
			if !strings.Contains(toolErr, ErrEmptyTask.Error()) {
				t.Errorf("tool error = %q, want it to wrap ErrEmptyTask", toolErr)
			}
		})
	}
}

// The cap is RECURSION depth, not sibling count: N children at the same
// nesting level are depth+1, not depth+N, so a wide fan-out must not look like
// a runaway recursion.
func TestParallelDelegateChargesOneDepthLevelForTheWholeFanOut(t *testing.T) {
	child := newAgent(t, echoTaskClient(), nil)
	parentCl := scripted(
		toolTurn("p1", "parallel_delegate", `{"tasks":["a","b","c","d"]}`),
		textTurn("collated"),
	)
	parent := newAgent(t, parentCl, []tool.Def{ParallelDelegateTool(child)})

	ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{SubagentDepth: 1})
	defer cancel()

	run := parent.Start(ctx, Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	var done ToolDone
	_, evs := drain(t, run)
	for _, ev := range evs {
		if td, ok := ev.(ToolDone); ok {
			done = td
		}
	}
	if run.Err() != nil {
		t.Fatalf("Err() = %v, want nil", run.Err())
	}
	if !done.OK {
		t.Fatalf("tool_done error = %q, want a four-branch fan-out to fit in depth 1", done.Error)
	}
	for _, task := range []string{"a", "b", "c", "d"} {
		if !strings.Contains(done.Output, "answer to "+task) {
			t.Errorf("branch %q did not run:\n%s", task, done.Output)
		}
	}
}

// Concurrent branches share one emitter; -race is what this is for.
func TestParallelDelegateEventsAreRaceFree(t *testing.T) {
	child := newAgent(t, echoTaskClient(), nil)
	parentCl := scripted(
		toolTurn("p1", "parallel_delegate", `{"tasks":["a","b","c","d","e","f"]}`),
		textTurn("collated"),
	)
	parent := newAgent(t, parentCl,
		[]tool.Def{ParallelDelegateTool(child, WithMaxBranches(4))})

	run := parent.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	count := 0
	for run.Next() {
		count++
	}
	if run.Err() != nil {
		t.Fatalf("Err() = %v, want nil", run.Err())
	}
	if count == 0 {
		t.Error("no events")
	}
}

// ===== preview =====

func TestPreview(t *testing.T) {
	tests := []struct {
		name  string
		in    string
		limit int
		want  string
	}{
		{name: "short strings pass through", in: "hello", limit: 10, want: "hello"},
		{name: "whitespace is collapsed", in: "a\n\n  b\tc", limit: 10, want: "a b c"},
		{name: "long strings are cut", in: "abcdefghij", limit: 5, want: "abcde…"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := preview(tc.in, tc.limit); got != tc.want {
				t.Errorf("preview(%q, %d) = %q, want %q", tc.in, tc.limit, got, tc.want)
			}
		})
	}

	// The cut must land on a rune boundary or the transcript carries invalid
	// UTF-8.
	t.Run("cuts on a rune boundary", func(t *testing.T) {
		in := strings.Repeat("中", 10) // 3 bytes each
		got := preview(in, 8)         // 8 is mid-rune
		body := strings.TrimSuffix(got, "…")
		if !utf8.ValidString(body) {
			t.Errorf("preview produced invalid UTF-8: %q", got)
		}
		if len(body)%3 != 0 {
			t.Errorf("preview cut mid-rune: %d bytes of a 3-byte-rune string", len(body))
		}
	})
}

func TestParallelDelegateRefusesTooManyTasks(t *testing.T) {
	childCl := scripted(textTurn("never"))
	child := newAgent(t, childCl, nil)
	d := ParallelDelegateTool(child, WithMaxTasks(2))
	_, err := d.Fn(context.Background(), json.RawMessage(`{"tasks":["a","b","c"]}`))
	if !errors.Is(err, ErrTooManyTasks) || !tool.IsCallerFault(err) {
		t.Errorf("err = %v, want ErrTooManyTasks as a caller fault", err)
	}
	if childCl.calls() != 0 {
		t.Errorf("the child ran %d times for a refused call", childCl.calls())
	}
}

// TestParallelDelegateStartsNoQueuedBranchAfterCancel: a branch waiting for a
// slot used to wait it out and then start a child run on the cancelled
// context, one queued task after another.
func TestParallelDelegateStartsNoQueuedBranchAfterCancel(t *testing.T) {
	blocked := llm.ClientFunc(func(ctx context.Context, _ llm.Request) (llm.Response, error) {
		<-ctx.Done()
		return llm.Response{}, context.Cause(ctx)
	})
	child := newAgent(t, blocked, nil)
	d := ParallelDelegateTool(child, WithMaxBranches(1))

	var mu sync.Mutex
	started := 0
	first := make(chan struct{})
	ctx, cancel := context.WithCancel(WithEmitter(context.Background(), func(e Event) {
		if _, ok := e.(SubagentStart); ok {
			mu.Lock()
			if started++; started == 1 {
				close(first)
			}
			mu.Unlock()
		}
	}))
	go func() { <-first; cancel() }()
	if _, err := d.Fn(ctx, json.RawMessage(`{"tasks":["a","b","c"]}`)); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if started != 1 {
		t.Errorf("%d branches started, want only the one running when the run was cancelled", started)
	}
}
