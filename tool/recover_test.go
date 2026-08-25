package tool

import (
	"context"
	"encoding/json"
	"errors"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestWithRecoveryTurnsAPanicIntoAnError. A panicking tool is a broken tool,
// not a broken agent: the model must be able to read the failure as an
// is_error tool_result and route around it.
func TestWithRecoveryTurnsAPanicIntoAnError(t *testing.T) {
	tests := []struct {
		name      string
		fn        Fn
		wantValue string
	}{
		{
			name:      "panic with a string",
			fn:        func(context.Context, json.RawMessage) (string, error) { panic("kaboom") },
			wantValue: "kaboom",
		},
		{
			name:      "panic with an error",
			fn:        func(context.Context, json.RawMessage) (string, error) { panic(errors.New("wrapped up")) },
			wantValue: "wrapped up",
		},
		{
			name: "runtime error: index out of range",
			fn: func(context.Context, json.RawMessage) (string, error) {
				var xs []int
				_ = xs[3]
				return "", nil
			},
			wantValue: "index out of range",
		},
		{
			name: "runtime error: nil map write",
			fn: func(context.Context, json.RawMessage) (string, error) {
				var m map[string]int
				m["k"] = 1
				return "", nil
			},
			wantValue: "nil map",
		},
		{
			// Since Go 1.21 panic(nil) becomes a *runtime.PanicNilError, so
			// recover() sees a non-nil value and this is caught like any other.
			// On an older toolchain (or GODEBUG=panicnil=1) it would sail
			// straight through the `v != nil` guard and kill the process.
			name:      "panic(nil)",
			fn:        func(context.Context, json.RawMessage) (string, error) { panic(nil) },
			wantValue: "panic called with nil argument",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := Def{Name: "boom", Fn: tt.fn}
			out, err := WithRecovery(Direct)(context.Background(), d, use("1", "boom", `{}`))

			if err == nil {
				t.Fatal("err = nil, want a recovered panic")
			}
			if out != "" {
				t.Errorf("out = %q, want %q: a handler that panicked mid-write observed nothing", out, "")
			}
			if !errors.Is(err, ErrPanic) {
				t.Errorf("errors.Is(err, ErrPanic) = false for %v, want true", err)
			}

			var pe *PanicError
			if !errors.As(err, &pe) {
				t.Fatalf("err is %T, want *PanicError", err)
			}
			if pe.Tool != "boom" {
				t.Errorf("PanicError.Tool = %q, want %q", pe.Tool, "boom")
			}
			if !strings.Contains(pe.Value, tt.wantValue) {
				t.Errorf("PanicError.Value = %q, want it to contain %q", pe.Value, tt.wantValue)
			}

			// The model reads Error(): one line, no Go stack, no token bill.
			msg := err.Error()
			if strings.Contains(msg, "goroutine") || strings.Contains(msg, ".go:") {
				t.Errorf("Error() = %q, want no stack in the model-facing form", msg)
			}
			if len(msg) > 300 {
				t.Errorf("len(Error()) = %d, want a short model-facing message", len(msg))
			}
			if !strings.Contains(msg, "boom crashed") {
				t.Errorf("Error() = %q, want it to say which tool crashed", msg)
			}

			// The operator reads Details(): the stack is captured at recover
			// time, while it still exists.
			if pe.Stack == "" {
				t.Error("PanicError.Stack is empty, want the stack captured at recover time")
			}
			if !strings.Contains(pe.Stack, "goroutine") {
				t.Errorf("PanicError.Stack = %q..., want a real goroutine dump", preview(pe.Stack, 80))
			}
			if !strings.HasPrefix(pe.Details(), msg) {
				t.Errorf("Details() = %q..., want it to start with the short form", preview(pe.Details(), 80))
			}
			if len(pe.Details()) <= len(msg) {
				t.Error("Details() is no longer than Error(), want it to add the stack")
			}
		})
	}
}

func TestWithRecoveryPassesThroughOrdinaryResults(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		d := Def{Name: "t", Fn: ok("fine")}
		out, err := WithRecovery(Direct)(context.Background(), d, use("1", "t", `{}`))
		if err != nil || out != "fine" {
			t.Errorf("(%q, %v), want (fine, nil)", out, err)
		}
	})

	t.Run("an ordinary error is not turned into a panic", func(t *testing.T) {
		boom := errors.New("no such file")
		d := Def{Name: "t", Fn: fails(boom)}
		_, err := WithRecovery(Direct)(context.Background(), d, use("1", "t", `{}`))
		if err != boom {
			t.Errorf("err = %v, want the tool's own error unchanged", err)
		}
		if errors.Is(err, ErrPanic) {
			t.Error("errors.Is(err, ErrPanic) = true, want false for an ordinary failure")
		}
	})
}

func TestPanicErrorDetailsWithoutAStack(t *testing.T) {
	e := &PanicError{Tool: "t", Value: "v"}
	if got := e.Details(); got != e.Error() {
		t.Errorf("Details() = %q, want it to fall back to Error() = %q when the stack is empty", got, e.Error())
	}
	if e.Unwrap() != ErrPanic {
		t.Errorf("Unwrap() = %v, want ErrPanic", e.Unwrap())
	}
}

func TestPanicValueIsClipped(t *testing.T) {
	huge := strings.Repeat("s", 100_000)
	d := Def{Name: "t", Fn: func(context.Context, json.RawMessage) (string, error) { panic(huge) }}
	_, err := WithRecovery(Direct)(context.Background(), d, use("1", "t", `{}`))

	var pe *PanicError
	if !errors.As(err, &pe) {
		t.Fatalf("err is %T, want *PanicError", err)
	}
	// A tool that panics with a megabyte of state must not push the
	// conversation out of its window.
	if len(pe.Value) > maxPanicValue+3 {
		t.Errorf("len(Value) = %d, want <= %d", len(pe.Value), maxPanicValue+3)
	}
	if len(pe.Stack) > maxPanicStack+3 {
		t.Errorf("len(Stack) = %d, want <= %d", len(pe.Stack), maxPanicStack+3)
	}
}

// TestRecoveredPanicTripsTheBreakerLikeAnyOtherFailure is why WithRecovery
// goes INNERMOST: every layer above it — retry, the breaker, dedup, logging —
// then sees a panic as the failure it is. A tool that panics every time must
// trip a breaker exactly like a tool that returns an error every time.
func TestRecoveredPanicTripsTheBreaker(t *testing.T) {
	calls := 0
	d := Def{Name: "boom", Fn: func(context.Context, json.RawMessage) (string, error) {
		calls++
		panic("kaboom")
	}}

	// Later entries are further out, so the breaker sees the recovered error.
	h := Chain(Direct, WithRecovery, WithCircuitBreaker(3, time.Minute))

	for i := 1; i <= 3; i++ {
		_, err := h(context.Background(), d, use("1", "boom", `{}`))
		if !errors.Is(err, ErrPanic) {
			t.Fatalf("call %d: err = %v, want a recovered panic", i, err)
		}
	}
	_, err := h(context.Background(), d, use("1", "boom", `{}`))
	if !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("call 4: err = %v, want ErrCircuitOpen — panics must count as failures", err)
	}
	if calls != 3 {
		t.Errorf("tool invocations = %d, want 3: the 4th call must be rejected without being attempted", calls)
	}
}

// Dedup treats a recovered panic like any other repeated wall, and %w keeps
// errors.Is(err, ErrPanic) working through the enrichment.
func TestRecoveredPanicIsDeduped(t *testing.T) {
	d := Def{Name: "boom", Fn: func(context.Context, json.RawMessage) (string, error) { panic("kaboom") }}
	h := Chain(Direct, WithRecovery, WithDedupRepeats(2))
	ctx := WithCallStats(context.Background(), NewCallStats())

	var last error
	for i := 0; i < 2; i++ {
		_, last = h(ctx, d, use("1", "boom", `{}`))
	}
	if !strings.Contains(last.Error(), "[repeat]") {
		t.Errorf("err = %q, want the repeat notice after two identical panics", last)
	}
	if !errors.Is(last, ErrPanic) {
		t.Error("errors.Is(enriched, ErrPanic) = false, want the identity preserved")
	}
}

// A retried tool that panics on every attempt still ends as one error.
func TestRecoveredPanicIsRetriedWhenTheDefSaysSo(t *testing.T) {
	calls := 0
	d := Def{
		Name:       "boom",
		Idempotent: true,
		Retryable:  func(err error) bool { return errors.Is(err, ErrPanic) },
		Fn: func(context.Context, json.RawMessage) (string, error) {
			calls++
			panic("kaboom")
		},
	}
	h := Chain(Direct, WithRecovery, WithRetry(RetryPolicy{MaxAttempts: 3, Base: time.Millisecond, Jitter: -1}))
	_, err := h(context.Background(), d, use("1", "boom", `{}`))
	if !errors.Is(err, ErrPanic) {
		t.Fatalf("err = %v, want ErrPanic", err)
	}
	if calls != 3 {
		t.Errorf("attempts = %d, want 3", calls)
	}
}

// A panic on a goroutine the tool spawned itself is documented as out of
// reach: nothing in the chain can catch it. Pin the documentation rather than
// the crash — running it would kill the test binary.
func TestPanicOnASpawnedGoroutineIsOutOfReach(t *testing.T) {
	t.Skip("documented limitation: a panic on a goroutine the tool spawned cannot be recovered by any caller, and demonstrating it would kill the test process")
}

// Sanity check that this toolchain really does convert panic(nil), which the
// table above depends on.
func TestPanicNilIsARuntimeError(t *testing.T) {
	defer func() {
		v := recover()
		if v == nil {
			t.Fatalf("recover() after panic(nil) = nil on %s; WithRecovery's `v != nil` guard would let it through", runtime.Version())
		}
		if _, isNilErr := v.(*runtime.PanicNilError); !isNilErr {
			t.Errorf("recover() = %T, want *runtime.PanicNilError", v)
		}
	}()
	panic(nil)
}
