package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

// ===== Chain =====

// TestChainOrdering pins the documented direction: later entries end up
// further OUT, so Chain(Direct, WithRetry, WithLogging) lets logging see the
// post-retry verdict. Getting this backwards silently inverts every
// position-sensitive comment in middleware.go.
func TestChainOrdering(t *testing.T) {
	var order []string
	mark := func(name string) Middleware {
		return func(next Handler) Handler {
			return func(ctx context.Context, d Def, u llm.ToolUse) (string, error) {
				order = append(order, "enter "+name)
				out, err := next(ctx, d, u)
				order = append(order, "exit "+name)
				return out, err
			}
		}
	}

	h := Chain(func(context.Context, Def, llm.ToolUse) (string, error) {
		order = append(order, "leaf")
		return "done", nil
	}, mark("inner"), mark("middle"), mark("outer"))

	out, err := h(context.Background(), Def{Name: "x"}, use("1", "x", `{}`))
	if err != nil || out != "done" {
		t.Fatalf("handler = (%q, %v), want (done, nil)", out, err)
	}

	want := []string{
		"enter outer", "enter middle", "enter inner",
		"leaf",
		"exit inner", "exit middle", "exit outer",
	}
	if !equalStrings(order, want) {
		t.Errorf("call order = %v, want %v", order, want)
	}
}

func TestChainWithNoMiddlewareIsTheHandler(t *testing.T) {
	called := false
	h := Chain(func(context.Context, Def, llm.ToolUse) (string, error) {
		called = true
		return "bare", nil
	})
	out, _ := h(context.Background(), Def{Name: "x"}, use("1", "x", `{}`))
	if !called || out != "bare" {
		t.Errorf("Chain(h) = %q (called=%v), want (bare, true)", out, called)
	}
}

// ===== Direct =====

func TestDirect(t *testing.T) {
	t.Run("invokes Fn with the raw input", func(t *testing.T) {
		var seen json.RawMessage
		d := Def{Name: "echo", Fn: func(_ context.Context, in json.RawMessage) (string, error) {
			seen = in
			return "ran", nil
		}}
		out, err := Direct(context.Background(), d, use("1", "echo", `{"a":1}`))
		if err != nil || out != "ran" {
			t.Fatalf("Direct = (%q, %v), want (ran, nil)", out, err)
		}
		if string(seen) != `{"a":1}` {
			t.Errorf("Fn saw input %s, want %s", seen, `{"a":1}`)
		}
	})

	// A Def with no Fn is a configuration mistake — a CapTerminal tool that
	// escaped interception, or a hand-built Def. It must be an ordinary error
	// the model can read, not a nil-func panic that kills the goroutine.
	t.Run("a nil Fn is an error naming the tool", func(t *testing.T) {
		out, err := Direct(context.Background(), Def{Name: "submit_task_result"}, use("1", "submit_task_result", `{}`))
		if err == nil {
			t.Fatal("Direct with a nil Fn returned nil error, want an error")
		}
		if out != "" {
			t.Errorf("out = %q, want %q", out, "")
		}
		if !strings.Contains(err.Error(), "submit_task_result") || !strings.Contains(err.Error(), "no handler") {
			t.Errorf("error = %q, want it to name the tool and say it has no handler", err)
		}
	})
}

// ===== Result =====

func TestResultBlock(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		b := Result{UseID: "u1", Output: "42"}.Block()
		tr, isResult := b.(llm.ToolResult)
		if !isResult {
			t.Fatalf("Block() = %T, want llm.ToolResult", b)
		}
		if tr.ToolUseID != "u1" || tr.Content != "42" || tr.IsError {
			t.Errorf("Block() = %+v, want {u1 42 false}", tr)
		}
	})

	// An error must reach the model as text with is_error, never be dropped:
	// the model has to see what went wrong to pick another approach.
	t.Run("failure carries the message with is_error", func(t *testing.T) {
		b := Result{UseID: "u2", Output: "partial", Err: errors.New("no such file")}.Block()
		tr := b.(llm.ToolResult)
		if !tr.IsError {
			t.Error("IsError = false, want true")
		}
		if tr.Content != "no such file" {
			t.Errorf("Content = %q, want %q", tr.Content, "no such file")
		}
	})

	t.Run("Blocks preserves order and length", func(t *testing.T) {
		got := Blocks([]Result{{UseID: "a", Output: "1"}, {UseID: "b", Err: errors.New("x")}})
		if len(got) != 2 {
			t.Fatalf("len = %d, want 2", len(got))
		}
		if got[0].(llm.ToolResult).ToolUseID != "a" || got[1].(llm.ToolResult).ToolUseID != "b" {
			t.Errorf("order = %v, want [a b]", got)
		}
	})
}

// ===== Dispatcher =====

// TestDispatchOneResultPerUseInInputOrder is the invariant every caller of
// Dispatch relies on: a missing or reordered result leaves a dangling
// tool_use, which the provider rejects on the NEXT turn — far from the cause.
func TestDispatchOneResultPerUseInInputOrder(t *testing.T) {
	slowFirst := def(t, "slow", func(context.Context, json.RawMessage) (string, error) {
		time.Sleep(5 * time.Millisecond)
		return "slow output", nil
	})
	fast := def(t, "fast", ok("fast output"))
	broken := def(t, "broken", fails(errors.New("deliberate")))

	set := NewSet(slowFirst, fast, broken)
	uses := []llm.ToolUse{
		use("u1", "slow", `{}`),
		use("u2", "ghost", `{}`), // not in the set
		use("u3", "fast", `{}`),
		use("u4", "broken", `{}`),
		use("u5", "fast", `{}`),
	}

	for _, parallel := range []int{1, 4} {
		t.Run(fmt.Sprintf("parallel=%d", parallel), func(t *testing.T) {
			d := NewDispatcher(Direct, WithParallelism(parallel))
			got := d.Dispatch(context.Background(), set, uses)

			if len(got) != len(uses) {
				t.Fatalf("len(results) = %d, want %d (one per tool_use)", len(got), len(uses))
			}
			for i, r := range got {
				if r.UseID != uses[i].ID {
					t.Errorf("result[%d].UseID = %q, want %q (input order)", i, r.UseID, uses[i].ID)
				}
				if r.Name != uses[i].Name {
					t.Errorf("result[%d].Name = %q, want %q", i, r.Name, uses[i].Name)
				}
			}

			// An unknown tool still gets a result, and a named one.
			if !errors.Is(got[1].Err, ErrUnknownTool) {
				t.Errorf("result for an unknown tool: Err = %v, want ErrUnknownTool", got[1].Err)
			}
			if got[1].Output != "" {
				t.Errorf("unknown tool Output = %q, want %q", got[1].Output, "")
			}
			if got[0].Output != "slow output" || got[2].Output != "fast output" {
				t.Errorf("outputs = %q,%q, want slow output,fast output", got[0].Output, got[2].Output)
			}
			if got[3].Err == nil || got[3].Err.Error() != "deliberate" {
				t.Errorf("result[3].Err = %v, want deliberate", got[3].Err)
			}
			if got[0].Dur <= 0 {
				t.Errorf("result[0].Dur = %v, want a positive duration", got[0].Dur)
			}
		})
	}
}

func TestDispatchEmptyBatch(t *testing.T) {
	got := NewDispatcher(Direct).Dispatch(context.Background(), NewSet(), nil)
	if len(got) != 0 {
		t.Errorf("Dispatch(nil) = %d results, want 0", len(got))
	}
}

// TestDispatchParallelismActuallyOverlaps: with WithParallelism(2) two calls
// must be in flight at once. If dispatch were still sequential the second
// handler would never start and this deadlocks into the timeout below.
func TestDispatchParallelismActuallyOverlaps(t *testing.T) {
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	block := def(t, "block", func(ctx context.Context, _ json.RawMessage) (string, error) {
		started <- struct{}{}
		select {
		case <-release:
			return "released", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})

	d := NewDispatcher(Direct, WithParallelism(2))
	done := make(chan []Result, 1)
	go func() {
		done <- d.Dispatch(context.Background(), NewSet(block),
			[]llm.ToolUse{use("a", "block", `{}`), use("b", "block", `{}`)})
	}()

	deadline := time.After(2 * time.Second)
	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-deadline:
			t.Fatalf("only %d of 2 calls started concurrently under WithParallelism(2), want both", i)
		}
	}
	close(release)

	select {
	case got := <-done:
		if len(got) != 2 || got[0].UseID != "a" || got[1].UseID != "b" {
			t.Errorf("results = %+v, want a then b", got)
		}
	case <-deadline:
		t.Fatal("Dispatch did not return after the calls were released")
	}
}

func TestDispatchParallelismIsBounded(t *testing.T) {
	var inflight, maxInflight atomic.Int64
	d := def(t, "count", func(context.Context, json.RawMessage) (string, error) {
		n := inflight.Add(1)
		for {
			old := maxInflight.Load()
			if n <= old || maxInflight.CompareAndSwap(old, n) {
				break
			}
		}
		time.Sleep(2 * time.Millisecond)
		inflight.Add(-1)
		return "ok", nil
	})

	uses := make([]llm.ToolUse, 12)
	for i := range uses {
		uses[i] = use(fmt.Sprintf("u%d", i), "count", `{}`)
	}

	for _, limit := range []int{1, 3} {
		maxInflight.Store(0)
		NewDispatcher(Direct, WithParallelism(limit)).Dispatch(context.Background(), NewSet(d), uses)
		if got := maxInflight.Load(); got > int64(limit) {
			t.Errorf("WithParallelism(%d): peak concurrency = %d, want <= %d", limit, got, limit)
		}
	}
}

func TestWithParallelismIgnoresNonPositive(t *testing.T) {
	for _, n := range []int{0, -1} {
		d := &dispatcher{parallel: 1}
		WithParallelism(n)(d)
		if d.parallel != 1 {
			t.Errorf("WithParallelism(%d) set parallel=%d, want it left at the default 1", n, d.parallel)
		}
	}
}

// TestDispatchParallelRace fans a batch out over shared state so `go test
// -race` has something to complain about if the annotation buffer, the results
// slice or the semaphore is misused.
func TestDispatchParallelRace(t *testing.T) {
	var calls atomic.Int64
	tagger := def(t, "tagger", func(ctx context.Context, in json.RawMessage) (string, error) {
		calls.Add(1)
		Annotate(ctx, "tag:"+string(InfoFrom(ctx).UseID))
		// Two annotations from two goroutines of the same call would also race.
		var wg sync.WaitGroup
		for i := 0; i < 4; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				Annotate(ctx, fmt.Sprintf("extra%d", i))
			}(i)
		}
		wg.Wait()
		return string(in), nil
	})

	uses := make([]llm.ToolUse, 32)
	for i := range uses {
		uses[i] = use(fmt.Sprintf("u%d", i), "tagger", fmt.Sprintf(`{"i":%d}`, i))
	}

	// A stateful chain shared by every worker: the breaker map and the dedup
	// counters are the interesting things to race on.
	h := Chain(Direct,
		WithRecovery,
		WithTimeout(time.Second),
		WithRetry(RetryPolicy{MaxAttempts: 2, Base: time.Millisecond, Jitter: -1}),
		WithDedupRepeats(2),
		WithCircuitBreaker(3, 10*time.Millisecond),
		WithTruncation(1<<16),
	)
	ctx := WithCallStats(context.Background(), NewCallStats())
	got := NewDispatcher(h, WithParallelism(8)).Dispatch(ctx, NewSet(tagger), uses)

	if calls.Load() != int64(len(uses)) {
		t.Errorf("calls = %d, want %d", calls.Load(), len(uses))
	}
	for i, r := range got {
		if r.UseID != uses[i].ID {
			t.Fatalf("result[%d].UseID = %q, want %q", i, r.UseID, uses[i].ID)
		}
		if r.Err != nil {
			t.Fatalf("result[%d].Err = %v, want nil", i, r.Err)
		}
		if r.Output != string(uses[i].Input) {
			t.Errorf("result[%d].Output = %q, want %q", i, r.Output, uses[i].Input)
		}
		if len(r.Tags) != 5 {
			t.Errorf("result[%d] has %d tags, want 5", i, len(r.Tags))
		}
	}
}

// TestDispatchBackstopCatchesPanickingMiddleware is the reason the recover
// lives on the worker goroutine and not only in WithRecovery. WithRecovery
// wraps only what is INSIDE it; a middleware that panics is outside, and under
// WithParallelism that panic unwinds a goroutine nobody can recover from — the
// process is simply gone.
func TestDispatchBackstopCatchesPanickingMiddleware(t *testing.T) {
	panicking := func(next Handler) Handler {
		return func(context.Context, Def, llm.ToolUse) (string, error) {
			panic("middleware exploded")
		}
	}
	// WithRecovery is INSIDE the panicking layer, so it cannot help here.
	h := Chain(Direct, WithRecovery, panicking)

	uses := []llm.ToolUse{use("u1", "t", `{}`), use("u2", "t", `{}`), use("u3", "t", `{}`)}
	set := NewSet(def(t, "t", ok("never reached")))

	for _, parallel := range []int{1, 4} {
		t.Run(fmt.Sprintf("parallel=%d", parallel), func(t *testing.T) {
			got := NewDispatcher(h, WithParallelism(parallel)).Dispatch(context.Background(), set, uses)

			if len(got) != len(uses) {
				t.Fatalf("len(results) = %d, want %d: a dropped result dangles a tool_use", len(got), len(uses))
			}
			for i, r := range got {
				if r.UseID != uses[i].ID {
					t.Errorf("result[%d].UseID = %q, want %q", i, r.UseID, uses[i].ID)
				}
				if !errors.Is(r.Err, ErrPanic) {
					t.Errorf("result[%d].Err = %v, want it to match ErrPanic", i, r.Err)
				}
				if r.Output != "" {
					t.Errorf("result[%d].Output = %q, want %q", i, r.Output, "")
				}
				var pe *PanicError
				if !errors.As(r.Err, &pe) {
					t.Fatalf("result[%d].Err is %T, want *PanicError", i, r.Err)
				}
				if pe.Tool != "t" {
					t.Errorf("PanicError.Tool = %q, want %q", pe.Tool, "t")
				}
				if !strings.Contains(pe.Value, "middleware exploded") {
					t.Errorf("PanicError.Value = %q, want it to contain %q", pe.Value, "middleware exploded")
				}
				if r.Dur <= 0 {
					t.Errorf("result[%d].Dur = %v, want a positive duration", i, r.Dur)
				}
			}
		})
	}
}

// A panic inside Set.Find is outside every middleware too.
func TestDispatchBackstopCatchesPanickingSet(t *testing.T) {
	got := NewDispatcher(Direct).Dispatch(context.Background(), panicSet{}, []llm.ToolUse{use("u1", "x", `{}`)})
	if len(got) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(got))
	}
	if !errors.Is(got[0].Err, ErrPanic) {
		t.Errorf("Err = %v, want ErrPanic", got[0].Err)
	}
	if got[0].UseID != "u1" {
		t.Errorf("UseID = %q, want u1", got[0].UseID)
	}
}

type panicSet struct{}

func (panicSet) Visible(context.Context) []Def { return nil }
func (panicSet) Find(string) (Def, bool)       { panic("set exploded") }

func TestDispatcherFunc(t *testing.T) {
	var f Dispatcher = DispatcherFunc(func(_ context.Context, _ Set, uses []llm.ToolUse) []Result {
		out := make([]Result, len(uses))
		for i, u := range uses {
			out[i] = Result{UseID: u.ID, Output: "stub"}
		}
		return out
	})
	got := f.Dispatch(context.Background(), NewSet(), []llm.ToolUse{use("a", "x", `{}`)})
	if len(got) != 1 || got[0].Output != "stub" {
		t.Errorf("DispatcherFunc.Dispatch = %+v, want one stub result", got)
	}
}
