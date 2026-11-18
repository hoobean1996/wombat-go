package tool

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"wombat/pkg/core/llm"
)

func TestRecoveryOfDerivesFromWhatTheToolDeclares(t *testing.T) {
	for _, tc := range []struct {
		name string
		d    Def
		want Recovery
	}{
		{"idempotent", Def{Idempotent: true, Caps: CapMutating}, RecoverRerun},
		{"read-only", Def{Caps: CapReadOnly}, RecoverRerun},
		{"read-only and exec", Def{Caps: CapReadOnly | CapExec}, RecoverReport},
		{"a write", Def{Caps: CapMutating}, RecoverReport},
		{"declares nothing", Def{}, RecoverReport},
		{"says ask", Def{Caps: CapReadOnly, Recovery: RecoverAsk}, RecoverAsk},
		{"says report", Def{Idempotent: true, Recovery: RecoverReport}, RecoverReport},
	} {
		if got := RecoveryOf(tc.d); got != tc.want {
			t.Errorf("%s: RecoveryOf = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestStatesThatAreFinal(t *testing.T) {
	final := map[State]bool{StateSucceeded: true, StateFailed: true, StateDenied: true, StateCancelled: true}
	for _, s := range States() {
		if s.Final() != final[s] {
			t.Errorf("%s.Final() = %v", s, s.Final())
		}
		if !s.Valid() {
			t.Errorf("%s is not Valid", s)
		}
	}
	if State("lost").Valid() {
		t.Error("an unknown state is Valid")
	}
}

func TestCallKeyPrefersTheHostsName(t *testing.T) {
	if got := CallKey(context.Background()); got != "" {
		t.Errorf("outside a call: %q", got)
	}
	ctx := WithInfo(context.Background(), Info{UseID: "u1"})
	if got := CallKey(ctx); got != "u1" {
		t.Errorf("without names: %q, want the id", got)
	}
	named := WithCallKeys(ctx, func(id llm.ToolUseID) string {
		if id == "u1" {
			return "s/3.0"
		}
		return ""
	})
	if got := CallKey(named); got != "s/3.0" {
		t.Errorf("named: %q", got)
	}
	if got := CallKey(WithInfo(named, Info{UseID: "u2"})); got != "u2" {
		t.Errorf("a call the names do not know: %q, want its id", got)
	}
}

func TestAwaitSaysWhatTheCallWaitsFor(t *testing.T) {
	err := fmt.Errorf("wrapped: %w", Await("job 7", "the build"))
	h, note, ok := Awaiting(err)
	if !ok || h != "job 7" || note != "the build" || !errors.Is(err, ErrAwait) {
		t.Errorf("Awaiting = %q %q %v", h, note, ok)
	}
	if _, _, ok := Awaiting(errors.New("no")); ok {
		t.Error("a plain error awaits")
	}
}

// A call waiting for a decision or a result has not failed: it is not tried
// again, and it neither trips nor resets a breaker.
func TestAWaitingCallIsNotAFailure(t *testing.T) {
	for _, wait := range []error{Await("h", "n"), Defer("ask")} {
		calls := 0
		h := Chain(func(context.Context, Def, llm.ToolUse) (string, error) {
			calls++
			return "", wait
		}, WithRetry(RetryPolicy{MaxAttempts: 3, Base: time.Microsecond}), WithCircuitBreaker(1, time.Hour))
		d := Def{Name: "x", Idempotent: true, Retryable: func(error) bool { return true }}
		for range 3 {
			if _, err := h(context.Background(), d, llm.ToolUse{}); !errors.Is(err, wait) {
				t.Fatalf("err = %v, want the wait back", err)
			}
		}
		if calls != 3 {
			t.Errorf("%v: the tool ran %d times for 3 calls: retried, or the breaker opened", wait, calls)
		}
	}
}

func TestTaggedReturnsTheCallsLabels(t *testing.T) {
	ann := &annotations{}
	ctx := context.WithValue(context.Background(), annotateKey{}, ann)
	Annotate(ctx, "a", "b")
	if got := Tagged(ctx); len(got) != 2 || got[1] != "b" {
		t.Errorf("Tagged = %v", got)
	}
	if Tagged(context.Background()) != nil {
		t.Error("outside a call it has tags")
	}
}
