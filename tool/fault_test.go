package tool

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

func TestCallerError(t *testing.T) {
	t.Run("nil in, nil out", func(t *testing.T) {
		if err := CallerError(nil); err != nil {
			t.Errorf("CallerError(nil) = %v, want nil", err)
		}
	})

	t.Run("leaves the message alone", func(t *testing.T) {
		const msg = "no such file or directory: /tmp/nope"
		if got := CallerError(errors.New(msg)).Error(); got != msg {
			t.Errorf("Error() = %q, want %q — the model must read the same bytes", got, msg)
		}
	})

	t.Run("errors.Is survives the marker and further wrapping", func(t *testing.T) {
		// This is the property that lets a package-level sentinel be marked
		// in place. builtin.ErrOutsideRoot is CallerError(errors.New(…)) and
		// is then wrapped with %w at the call site; if either hop lost the
		// identity, every errors.Is check against it would quietly go false.
		sentinel := CallerError(errors.New("builtin: outside root"))
		wrapped := fmt.Errorf("%w: %q is outside %q", sentinel, "/etc/passwd", "/proj")

		if !errors.Is(wrapped, sentinel) {
			t.Error("errors.Is(wrapped, sentinel) = false, want true")
		}
		if !IsCallerFault(wrapped) {
			t.Error("IsCallerFault(wrapped) = false, want true")
		}
	})

	t.Run("an unmarked error defaults to the tool's fault", func(t *testing.T) {
		// The default has to run this way: a breaker that under-counts keeps
		// calling a dependency that is down, one that over-counts takes away
		// a tool that works, and only the first is recoverable.
		for _, err := range []error{
			nil,
			errors.New("connection refused"),
			fmt.Errorf("wrapped: %w", errors.New("i/o timeout")),
			ErrTimeout,
			ErrPanic,
		} {
			if IsCallerFault(err) {
				t.Errorf("IsCallerFault(%v) = true, want false", err)
			}
		}
	})

	t.Run("the middleware sentinels that are the caller's fault say so", func(t *testing.T) {
		for _, err := range []error{ErrInvalidInput, ErrUnknownTool} {
			if !IsCallerFault(err) {
				t.Errorf("IsCallerFault(%v) = false, want true", err)
			}
		}
	})
}

// TestCircuitBreakerIgnoresCallerFaults is the regression this whole file
// exists for.
//
// A benchmark run against a live model lost an entire episode to it: the agent
// guessed five paths outside its filesystem root, the breaker read five
// refusals as five failures, and view_file went dark for a minute — which in a
// thirteen-turn budget is the rest of the run. Nine of the remaining calls came
// back "circuit breaker open" and the agent never wrote a line of code.
func TestCircuitBreakerIgnoresCallerFaults(t *testing.T) {
	badPath := CallerError(errors.New("builtin: path escapes the filesystem root"))

	t.Run("a wall of caller faults never opens it", func(t *testing.T) {
		inner := &countingHandler{err: badPath}
		h := WithCircuitBreaker(3, time.Minute)(inner.handle)
		d := Def{Name: "view_file"}

		for i := 1; i <= 20; i++ {
			_, err := h(context.Background(), d, use("1", "view_file", `{}`))
			if errors.Is(err, ErrCircuitOpen) {
				t.Fatalf("call %d: breaker opened on the caller's mistakes", i)
			}
		}
		if inner.n() != 20 {
			t.Errorf("attempts = %d, want all 20 to reach the tool", inner.n())
		}
	})

	t.Run("a caller fault neither counts nor forgives", func(t *testing.T) {
		// Not resetting is as load-bearing as not counting: a dependency that
		// is genuinely down, poked at with an occasional bad path in between,
		// must still be diagnosed.
		down := errors.New("dependency down")
		inner := &countingHandler{perCall: []error{down, down, badPath, down}}
		h := WithCircuitBreaker(3, time.Minute)(inner.handle)
		d := Def{Name: "search"}

		for i := 1; i <= 4; i++ {
			_, err := h(context.Background(), d, use("1", "search", `{}`))
			if errors.Is(err, ErrCircuitOpen) {
				t.Fatalf("call %d tripped early", i)
			}
		}
		if _, err := h(context.Background(), d, use("1", "search", `{}`)); !errors.Is(err, ErrCircuitOpen) {
			t.Fatalf("err = %v, want ErrCircuitOpen — three real failures happened", err)
		}
	})
}
