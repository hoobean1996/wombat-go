package wombat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// A cap of N lets N happen. Both caps used to trip with >= on a count taken
// before the unit ran, so Steps: 1 made no model call at all and ToolCalls: 1
// sent the model's request and then refused the only call it asked for.
func TestLimitsAllowExactlyWhatTheySay(t *testing.T) {
	t.Run("steps", func(t *testing.T) {
		for _, steps := range []int{1, 2, 3} {
			// A run that needs two iterations: a tool call, then the answer.
			cl := scripted(toolTurn("u1", "echo", `{}`), textTurn("done"))
			a := newAgent(t, cl, []tool.Def{echoTool("echo", "ok")})
			ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{Steps: steps})
			_, err := a.Run(ctx, Ask("q"))
			cancel()
			wantCalls, wantErr := min(steps, 2), steps < 2
			if cl.calls() != wantCalls || (err != nil) != wantErr {
				t.Errorf("Steps=%d: %d model calls, err %v; want %d calls and error=%v", steps, cl.calls(), err, wantCalls, wantErr)
			}
			if wantErr && !errors.Is(err, governor.ErrStepLimit) {
				t.Errorf("Steps=%d: err = %v, want ErrStepLimit", steps, err)
			}
		}
	})
	t.Run("tool calls", func(t *testing.T) {
		var ran atomic.Int32
		counting := tool.Def{Name: "echo", Description: "x", InputSchema: json.RawMessage(objSchema),
			Fn: func(context.Context, json.RawMessage) (string, error) { ran.Add(1); return "ok", nil }}
		cl := scripted(toolTurn("u1", "echo", `{}`), textTurn("done"))
		a := newAgent(t, cl, []tool.Def{counting})
		ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{ToolCalls: 1})
		defer cancel()
		if _, err := a.Run(ctx, Ask("q")); err != nil || ran.Load() != 1 {
			t.Errorf("ToolCalls=1 with one call: ran %d, err %v; want it run and the run to finish", ran.Load(), err)
		}
	})
}

// refuse stands in for a permission gate: caller middleware that turns the
// call down without running it.
func refuse(next tool.Handler) tool.Handler {
	return func(context.Context, tool.Def, llm.ToolUse) (string, error) {
		return "", errors.New("denied by policy")
	}
}

// A refused call did not use the ToolCalls budget, but asking for it again
// and again is still a loop: it counts for loop detection, and the model is
// told it is repeating itself.
func TestRefusedCallsAreAttemptsNotCalls(t *testing.T) {
	cl := scripted(toolTurn("u1", "x", `{}`), toolTurn("u2", "x", `{}`), toolTurn("u3", "x", `{}`), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{echoTool("x", "ran")}, WithToolMiddleware(refuse))
	ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{ToolCalls: 1})
	defer cancel()
	r := a.Start(ctx, Ask("q"))
	for r.Next() {
	}
	defer r.Close()
	if err := r.Err(); err != nil {
		t.Fatalf("err = %v: three refused calls used up a ToolCalls budget of 1", err)
	}
	pr := governor.FromContext(ctx).Progress()
	if pr.ToolCalls != 0 || pr.Repeats != 3 {
		t.Errorf("ToolCalls = %d, Repeats = %d; want 0 calls counted and a streak of 3", pr.ToolCalls, pr.Repeats)
	}
	var last string
	for _, m := range r.Messages() {
		for _, b := range m.Content {
			if tr, ok := b.(llm.ToolResult); ok {
				last = tr.Content
			}
		}
	}
	if !strings.Contains(last, "[repeat]") {
		t.Errorf("the third identical refusal reads %q, want the dedup note telling the model it is stuck", last)
	}
}

func TestRefusedCallsStillTripTheLoopDetector(t *testing.T) {
	cl := scripted(toolTurn("u1", "x", `{}`), toolTurn("u2", "x", `{}`), toolTurn("u3", "x", `{}`), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{echoTool("x", "ran")}, WithToolMiddleware(refuse))
	ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{RepeatedToolCalls: 3})
	defer cancel()
	if _, err := a.Run(ctx, Ask("q")); !errors.Is(err, governor.ErrToolLoop) {
		t.Errorf("err = %v, want ErrToolLoop: asking for the same refused call three times is a loop", err)
	}
}

func marker(log *[]string, mu *sync.Mutex, name string) llm.Middleware {
	return func(next llm.Client) llm.Client {
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			mu.Lock()
			*log = append(*log, name)
			mu.Unlock()
			return next.Complete(ctx, req)
		})
	}
}

// TestWithSiblingsDoNotShareMiddleware: two agents derived from one parent
// shared the spare capacity of its llmMW slice, so deriving the second
// overwrote what the first had been given — visible as soon as the first was
// derived from again.
func TestWithSiblingsDoNotShareMiddleware(t *testing.T) {
	var log []string
	var mu sync.Mutex
	leaf := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) { return textTurn("ok"), nil })
	a := newAgent(t, leaf, nil,
		WithLLMMiddleware(marker(&log, &mu, "m1")),
		WithLLMMiddleware(marker(&log, &mu, "m2")),
		WithLLMMiddleware(marker(&log, &mu, "m3")))
	first, err := a.With(WithLLMMiddleware(marker(&log, &mu, "first")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.With(WithLLMMiddleware(marker(&log, &mu, "second"))); err != nil {
		t.Fatal(err)
	}
	again, err := first.With()
	if err != nil {
		t.Fatal(err)
	}
	log = nil
	if _, err := again.Run(context.Background(), Ask("q")); err != nil {
		t.Fatal(err)
	}
	if !contains(log, "first") || contains(log, "second") {
		t.Errorf("first.With() ran %v, want first's own middleware and not its sibling's", log)
	}

	// And concurrently, which -race reports when the array is shared.
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() { _, _ = a.With(WithLLMMiddleware(marker(&log, &mu, "x")), WithToolMiddleware(refuse)) })
	}
	wg.Wait()
}

// TestClipSlicesCoversEveryField keeps clipSlices honest as config grows: a
// slice field it does not clip is one every derived agent shares.
func TestClipSlicesCoversEveryField(t *testing.T) {
	var c config
	v := reflect.ValueOf(&c).Elem()
	for i := range v.NumField() {
		f := v.Field(i)
		if f.Kind() != reflect.Slice {
			continue
		}
		// Settable through unsafe-free reflection only for exported fields,
		// so build the value with spare capacity via reflect.MakeSlice and
		// write it with the unexported field's address.
		s := reflect.MakeSlice(f.Type(), 1, 4)
		reflect.NewAt(f.Type(), f.Addr().UnsafePointer()).Elem().Set(s)
	}
	c.clipSlices()
	for i := range v.NumField() {
		if f := v.Field(i); f.Kind() == reflect.Slice && f.Cap() != f.Len() {
			t.Errorf("config.%s is not clipped by clipSlices", v.Type().Field(i).Name)
		}
	}
}

// TestTrackCostAddsUpWithoutARun: a client used on its own has no run budget
// on its context, and every call used to be charged to a fresh throwaway, so
// each Spend event reported exactly one call.
func TestTrackCostAddsUpWithoutARun(t *testing.T) {
	leaf := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		r := textTurn("ok")
		r.Usage = llm.Usage{InputTokens: 1000, OutputTokens: 100}
		return r, nil
	})
	c := llm.Chain(leaf, TrackCost(llm.DefaultPricing))
	var calls []int
	ctx := WithEmitter(context.Background(), func(e Event) {
		if s, ok := e.(Spend); ok {
			calls = append(calls, s.Calls)
		}
	})
	for range 3 {
		if _, err := c.Complete(ctx, llm.Request{Model: "claude-sonnet-5", Messages: []llm.Message{userMsg("q")}}); err != nil {
			t.Fatal(err)
		}
	}
	if fmt.Sprint(calls) != "[1 2 3]" {
		t.Errorf("Spend.Calls = %v, want a running total", calls)
	}

	// A context with its own budget is charged there, not to the client's.
	scoped, cancel := governor.WithBudget(ctx, governor.Limits{})
	defer cancel()
	calls = nil
	_, _ = c.Complete(scoped, llm.Request{Model: "claude-sonnet-5", Messages: []llm.Message{userMsg("q")}})
	if fmt.Sprint(calls) != "[1]" {
		t.Errorf("with a budget on the context: Spend.Calls = %v, want that budget's own count", calls)
	}
}
