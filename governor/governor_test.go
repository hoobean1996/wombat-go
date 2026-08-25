// Tests are white-box (package governor) on purpose: the package's whole job
// is enforcement, and a couple of the invariants worth pinning — that
// FromContext falls back to the shared `unlimited` singleton, that a
// non-*Budget value under budgetKey{} is ignored — are only observable from
// inside. Everything else here goes through the exported surface anyway.
package governor

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

// waitDone blocks until ctx is cancelled, failing rather than hanging.
func waitDone(t *testing.T, ctx context.Context) {
	t.Helper()
	select {
	case <-ctx.Done():
	case <-time.After(2 * time.Second):
		t.Fatalf("context still live after 2s, want cancelled")
	}
}

// ===== caps =====

// TestCapsTripWithCause walks every resource cap up to its edge, checks the
// run is still alive one unit short, and then checks the last unit cancels the
// context with the right context.Cause. The cause is the whole API: callers
// recover it with errors.Is(context.Cause(ctx), governor.ErrX), so a cap that
// cancelled with the wrong cause would look like an ordinary cancellation.
func TestCapsTripWithCause(t *testing.T) {
	tests := []struct {
		name   string
		limits Limits
		upTo   func(*Budget) // spend right up to, but not over, the cap
		last   func(*Budget) // the unit that trips it
		want   error
	}{
		{
			name:   "cost",
			limits: Limits{CostUSD: 1.00},
			upTo:   func(b *Budget) { b.AddCall(0.50, Tokens{In: 10, Out: 5}) },
			last:   func(b *Budget) { b.AddCall(0.50, Tokens{In: 10, Out: 5}) },
			want:   ErrBudgetExhausted,
		},
		{
			name:   "steps",
			limits: Limits{Steps: 3},
			upTo:   func(b *Budget) { b.Step(); b.Step() },
			last:   func(b *Budget) { b.Step() },
			want:   ErrStepLimit,
		},
		{
			name:   "tool calls",
			limits: Limits{ToolCalls: 3},
			upTo:   func(b *Budget) { b.AddToolCall(""); b.AddToolCall("") },
			last:   func(b *Budget) { b.AddToolCall("") },
			want:   ErrToolCallLimit,
		},
		{
			name:   "repeated identical tool calls",
			limits: Limits{RepeatedToolCalls: 3},
			upTo:   func(b *Budget) { b.AddToolCall("calculator:1+1"); b.AddToolCall("calculator:1+1") },
			last:   func(b *Budget) { b.AddToolCall("calculator:1+1") },
			want:   ErrToolLoop,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := WithBudget(context.Background(), tt.limits)
			defer cancel()
			b := FromContext(ctx)

			tt.upTo(b)
			if err := ctx.Err(); err != nil {
				t.Fatalf("one unit short of the cap: ctx.Err() = %v (cause %v), want nil", err, context.Cause(ctx))
			}

			tt.last(b)
			waitDone(t, ctx)
			if got := context.Cause(ctx); !errors.Is(got, tt.want) {
				t.Errorf("context.Cause = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestWallClockCap(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{Wall: 20 * time.Millisecond})
	defer cancel()

	if err := ctx.Err(); err != nil {
		t.Fatalf("immediately after WithBudget: ctx.Err() = %v, want nil", err)
	}

	waitDone(t, ctx)
	if got := context.Cause(ctx); !errors.Is(got, ErrWallClock) {
		t.Errorf("context.Cause = %v, want %v", got, ErrWallClock)
	}
	if !errors.Is(ctx.Err(), context.DeadlineExceeded) {
		t.Errorf("ctx.Err() = %v, want %v", ctx.Err(), context.DeadlineExceeded)
	}
}

func TestWallClockNotYetElapsed(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{Wall: time.Hour})
	defer cancel()

	select {
	case <-ctx.Done():
		t.Fatalf("context cancelled with cause %v, want it live for the next hour", context.Cause(ctx))
	case <-time.After(10 * time.Millisecond):
	}
}

// TestZeroLimitsAreUnlimited pins the zero-field convention: a Limits field
// left at zero is not "a cap of zero", it is "no cap on this dimension".
func TestZeroLimitsAreUnlimited(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{})
	defer cancel()
	b := FromContext(ctx)

	for i := 0; i < 100; i++ {
		b.AddCall(1000, Tokens{In: 1_000_000})
		b.Step()
		b.AddToolCall("same-key-every-time")
	}
	if err := ctx.Err(); err != nil {
		t.Fatalf("ctx.Err() = %v (cause %v), want nil: a zero Limits caps nothing", err, context.Cause(ctx))
	}
	if p := b.Progress(); p.Fraction() != 0 {
		t.Errorf("Fraction() = %v, want 0 when nothing is capped", p.Fraction())
	}
}

// TestFirstCauseWins: once a run is cancelled the cause is frozen. The
// operator must see what actually ended the run, not whichever cap was checked
// last on the way out.
func TestFirstCauseWins(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{Steps: 1, CostUSD: 0.01})
	defer cancel()
	b := FromContext(ctx)

	b.Step() // trips ErrStepLimit
	b.AddCall(99, Tokens{})
	b.AddToolCall("")

	if got := context.Cause(ctx); !errors.Is(got, ErrStepLimit) {
		t.Errorf("context.Cause = %v, want %v (the cap that tripped first)", got, ErrStepLimit)
	}
}

// TestRepeatRunResets covers the two ways a run of identical calls is broken:
// a different key, and the "" key that opts out of loop detection entirely.
func TestRepeatRunResets(t *testing.T) {
	t.Run("a different key restarts the run", func(t *testing.T) {
		ctx, cancel := WithBudget(context.Background(), Limits{RepeatedToolCalls: 3})
		defer cancel()
		b := FromContext(ctx)

		b.AddToolCall("a")
		b.AddToolCall("a")
		b.AddToolCall("b") // breaks the streak
		b.AddToolCall("a")
		b.AddToolCall("a")
		if err := ctx.Err(); err != nil {
			t.Fatalf("ctx.Err() = %v (cause %v), want nil: the streak was broken by \"b\"",
				err, context.Cause(ctx))
		}
		if got := b.Progress().Repeats; got != 2 {
			t.Errorf("Progress().Repeats = %d, want 2", got)
		}

		b.AddToolCall("a")
		if got := context.Cause(ctx); !errors.Is(got, ErrToolLoop) {
			t.Errorf("context.Cause = %v, want %v", got, ErrToolLoop)
		}
	})

	t.Run("the empty key opts out", func(t *testing.T) {
		ctx, cancel := WithBudget(context.Background(), Limits{RepeatedToolCalls: 2})
		defer cancel()
		b := FromContext(ctx)

		for i := 0; i < 10; i++ {
			b.AddToolCall("")
		}
		if err := ctx.Err(); err != nil {
			t.Fatalf("ctx.Err() = %v (cause %v), want nil: \"\" means do not loop-detect",
				err, context.Cause(ctx))
		}
		if got := b.Progress().ToolCalls; got != 10 {
			t.Errorf("Progress().ToolCalls = %d, want 10: unkeyed calls still count", got)
		}
	})
}

// ===== FromContext =====

func TestFromContextNeverNil(t *testing.T) {
	tests := []struct {
		name string
		ctx  func() context.Context
	}{
		{"background", func() context.Context { return context.Background() }},
		{"todo", func() context.Context { return context.TODO() }},
		{"unrelated value", func() context.Context {
			return context.WithValue(context.Background(), struct{ k int }{}, "x")
		}},
		{"wrong type under the key", func() context.Context {
			return context.WithValue(context.Background(), budgetKey{}, "not a budget")
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := FromContext(tt.ctx())
			if b == nil {
				t.Fatal("FromContext returned nil, want the unlimited budget")
			}
			if b != unlimited {
				t.Errorf("FromContext returned %p, want the shared unlimited budget %p", b, unlimited)
			}
			if got := b.Progress().Limits; got != (Limits{}) {
				t.Errorf("Progress().Limits = %+v, want the zero Limits", got)
			}
		})
	}
}

// TestUngovernedSpendIsInert: code that runs outside a budget — a unit test, a
// one-off script — must behave as if there were no caps rather than crash on a
// nil budget or a nil cancel func.
func TestUngovernedSpendIsInert(t *testing.T) {
	b := FromContext(context.Background())

	// NOTE: this is the process-wide `unlimited` singleton, so its counters are
	// shared with every other ungoverned caller. Assert only that nothing
	// panics and nothing is cancelled; the tallies are not this budget's to
	// own. See the report for why that sharing is worth a second look.
	b.AddCall(12.34, Tokens{In: 1, Out: 2, CacheRead: 3, CacheWrite: 4})
	b.Step()
	b.AddToolCall("k")
	if !b.EnterSubagent() {
		t.Error("EnterSubagent() = false, want true when uncapped")
	}
	b.ExitSubagent()

	if got := b.Progress().Limits; got != (Limits{}) {
		t.Errorf("Progress().Limits = %+v, want the zero Limits", got)
	}
}

func TestFromContextReturnsTheSameBudget(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{Steps: 5})
	defer cancel()

	if FromContext(ctx) != FromContext(ctx) {
		t.Error("FromContext returned two different budgets for one context")
	}
	// A derived context still finds it: sub-agents inherit the parent's budget,
	// which is what stops a fan-out from spending past the cap.
	child, childCancel := context.WithCancel(ctx)
	defer childCancel()
	if FromContext(child) != FromContext(ctx) {
		t.Error("a derived context found a different budget, want the parent's")
	}
}

func TestCancelFuncCancelsWithoutACapCause(t *testing.T) {
	for _, wall := range []time.Duration{0, time.Hour} {
		t.Run(fmt.Sprintf("wall=%v", wall), func(t *testing.T) {
			ctx, cancel := WithBudget(context.Background(), Limits{Wall: wall})
			cancel()
			waitDone(t, ctx)
			if !errors.Is(ctx.Err(), context.Canceled) {
				t.Errorf("ctx.Err() = %v, want %v", ctx.Err(), context.Canceled)
			}
			if got := context.Cause(ctx); !errors.Is(got, context.Canceled) {
				t.Errorf("context.Cause = %v, want %v (an ordinary cancel is not a cap)", got, context.Canceled)
			}
			cancel() // idempotent
		})
	}
}

// ===== sub-agent depth: the one cap that refuses instead of aborting =====

// TestEnterSubagentRefusesWithoutAborting is the regression test for the
// defect this package's depth cap was reworked to fix.
//
// Every other cap here is a resource — money, iterations, wall clock — and
// hitting one means there is nothing left to continue with, so it cancels the
// context. Nesting depth is a shape, not a resource. A parent told "not
// deeper" can still do the work itself, but only if it is still running.
// Aborting on a depth refusal killed a parent that had a perfectly good
// fallback and took the transcript explaining why with it.
//
// So: EnterSubagent returns false, and context.Cause(ctx) is STILL NIL.
func TestEnterSubagentRefusesWithoutAborting(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{SubagentDepth: 1})
	defer cancel()
	b := FromContext(ctx)

	if !b.EnterSubagent() {
		t.Fatal("first EnterSubagent() = false, want true (depth 0 of 1)")
	}
	if b.EnterSubagent() {
		t.Fatal("second EnterSubagent() = true, want false (depth 1 of 1 is the cap)")
	}

	// The assertion the fix exists for.
	if err := ctx.Err(); err != nil {
		t.Errorf("ctx.Err() = %v, want nil: a depth refusal must not end the run", err)
	}
	if cause := context.Cause(ctx); cause != nil {
		t.Errorf("context.Cause = %v, want nil: a depth refusal must not end the run", cause)
	}
	// And the parent is genuinely still able to act: it can spend, and it can
	// delegate again once the level it is holding is released.
	b.Step()
	b.AddToolCall("do-it-myself")
	if err := ctx.Err(); err != nil {
		t.Errorf("after the refusal, ctx.Err() = %v, want nil: the parent must be able to do the work itself", err)
	}
	if got := b.Progress().Depth; got != 1 {
		t.Errorf("Progress().Depth = %d, want 1: a refused entry must not deepen the level", got)
	}

	b.ExitSubagent()
	if !b.EnterSubagent() {
		t.Error("EnterSubagent() = false after ExitSubagent, want true: the level was released")
	}
	b.ExitSubagent()
}

func TestSubagentDepth(t *testing.T) {
	t.Run("zero means unlimited", func(t *testing.T) {
		ctx, cancel := WithBudget(context.Background(), Limits{})
		defer cancel()
		b := FromContext(ctx)
		for i := 0; i < 50; i++ {
			if !b.EnterSubagent() {
				t.Fatalf("EnterSubagent() = false at depth %d, want true when uncapped", i)
			}
		}
		if got := b.Progress().Depth; got != 50 {
			t.Errorf("Progress().Depth = %d, want 50", got)
		}
		for i := 0; i < 50; i++ {
			b.ExitSubagent()
		}
		if got := b.Progress().Depth; got != 0 {
			t.Errorf("Progress().Depth = %d, want 0 after unwinding", got)
		}
	})

	t.Run("nests up to the cap", func(t *testing.T) {
		ctx, cancel := WithBudget(context.Background(), Limits{SubagentDepth: 3})
		defer cancel()
		b := FromContext(ctx)
		for i := 0; i < 3; i++ {
			if !b.EnterSubagent() {
				t.Fatalf("EnterSubagent() = false at depth %d, want true (cap is 3)", i)
			}
		}
		if b.EnterSubagent() {
			t.Error("EnterSubagent() = true at depth 3, want false")
		}
	})

	t.Run("ExitSubagent does not go negative", func(t *testing.T) {
		ctx, cancel := WithBudget(context.Background(), Limits{SubagentDepth: 2})
		defer cancel()
		b := FromContext(ctx)

		b.ExitSubagent()
		b.ExitSubagent()
		if got := b.Progress().Depth; got != 0 {
			t.Fatalf("Progress().Depth = %d, want 0: unmatched exits must not underflow", got)
		}
		// An underflowed counter would have made this refuse.
		if !b.EnterSubagent() {
			t.Error("EnterSubagent() = false, want true: depth must still be 0")
		}
	})
}

// ===== Progress =====

func TestProgressAccounting(t *testing.T) {
	limits := Limits{CostUSD: 10, Steps: 20, ToolCalls: 30, Wall: time.Hour, SubagentDepth: 2, RepeatedToolCalls: 5}
	ctx, cancel := WithBudget(context.Background(), limits)
	defer cancel()
	b := FromContext(ctx)

	b.AddCall(0.25, Tokens{In: 100, Out: 10, CacheWrite: 5, CacheRead: 1})
	b.AddCall(0.75, Tokens{In: 200, Out: 20, CacheWrite: 6, CacheRead: 2})
	b.Step()
	b.Step()
	b.Step()
	b.AddToolCall("grep:x")
	b.AddToolCall("grep:x")
	b.EnterSubagent()

	p := b.Progress()

	if p.Limits != limits {
		t.Errorf("Progress().Limits = %+v, want %+v", p.Limits, limits)
	}
	if p.CostUSD != 1.00 {
		t.Errorf("CostUSD = %v, want 1", p.CostUSD)
	}
	if p.Calls != 2 {
		t.Errorf("Calls = %d, want 2", p.Calls)
	}
	if p.Steps != 3 {
		t.Errorf("Steps = %d, want 3", p.Steps)
	}
	if p.ToolCalls != 2 {
		t.Errorf("ToolCalls = %d, want 2", p.ToolCalls)
	}
	if p.Depth != 1 {
		t.Errorf("Depth = %d, want 1", p.Depth)
	}
	if p.Repeats != 2 {
		t.Errorf("Repeats = %d, want 2", p.Repeats)
	}
	want := Tokens{In: 300, Out: 30, CacheWrite: 11, CacheRead: 3}
	if p.Tokens != want {
		t.Errorf("Tokens = %+v, want %+v", p.Tokens, want)
	}
	if p.Elapsed <= 0 {
		t.Errorf("Elapsed = %v, want > 0", p.Elapsed)
	}

	// A snapshot is a value: further spend must not mutate it.
	b.Step()
	if p.Steps != 3 {
		t.Errorf("the snapshot changed under us: Steps = %d, want 3", p.Steps)
	}
}

// TestFraction works on literal Progress values rather than a live Budget, so
// the wall-clock dimension is exercised without depending on real elapsed time.
func TestFraction(t *testing.T) {
	tests := []struct {
		name string
		p    Progress
		want float64
	}{
		{"nothing capped", Progress{CostUSD: 99, Steps: 99}, 0},
		{"cost only", Progress{Limits: Limits{CostUSD: 2}, CostUSD: 0.5}, 0.25},
		{"steps only", Progress{Limits: Limits{Steps: 10}, Steps: 9}, 0.9},
		{"tool calls only", Progress{Limits: Limits{ToolCalls: 4}, ToolCalls: 1}, 0.25},
		{"wall only", Progress{Limits: Limits{Wall: time.Second}, Elapsed: 500 * time.Millisecond}, 0.5},
		{
			name: "the nearest cap across dimensions wins",
			p: Progress{
				Limits:  Limits{CostUSD: 10, Steps: 10, ToolCalls: 100, Wall: 100 * time.Second},
				CostUSD: 1, Steps: 9, ToolCalls: 5, Elapsed: 10 * time.Second,
			},
			want: 0.9,
		},
		{
			name: "wall can be the nearest",
			p: Progress{
				Limits:  Limits{CostUSD: 10, Steps: 10, Wall: 100 * time.Second},
				CostUSD: 1, Steps: 1, Elapsed: 95 * time.Second,
			},
			want: 0.95,
		},
		{"past the cap exceeds 1", Progress{Limits: Limits{Steps: 4}, Steps: 5}, 1.25},
		{
			name: "an uncapped dimension is ignored however large",
			p:    Progress{Limits: Limits{Steps: 100}, Steps: 1, CostUSD: 1e9, ToolCalls: 1e6},
			want: 0.01,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.p.Fraction(); math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("Fraction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCostFraction(t *testing.T) {
	tests := []struct {
		name string
		p    Progress
		want float64
	}{
		{"uncapped", Progress{CostUSD: 5}, 0},
		{"negative cap is uncapped", Progress{Limits: Limits{CostUSD: -1}, CostUSD: 5}, 0},
		{"half spent", Progress{Limits: Limits{CostUSD: 4}, CostUSD: 2}, 0.5},
		{"overspent", Progress{Limits: Limits{CostUSD: 2}, CostUSD: 3}, 1.5},
		{
			name: "ignores the other dimensions",
			p:    Progress{Limits: Limits{CostUSD: 10, Steps: 2}, CostUSD: 1, Steps: 2},
			want: 0.1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.p.CostFraction(); math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("CostFraction() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestProgressString pins the exact rendering, because this string is shown to
// the model: it is the run's only view of how much room it has left.
func TestProgressString(t *testing.T) {
	tests := []struct {
		name string
		p    Progress
		want string
	}{
		{"no limits", Progress{CostUSD: 3, Steps: 7}, "no limits configured"},
		{
			name: "cost only",
			p:    Progress{Limits: Limits{CostUSD: 1}, CostUSD: 0.12345},
			want: "cost $0.1235 of $1.00",
		},
		{"steps only", Progress{Limits: Limits{Steps: 30}, Steps: 4}, "step 4 of 30"},
		{"tool calls only", Progress{Limits: Limits{ToolCalls: 9}, ToolCalls: 2}, "2 of 9 tool calls"},
		{
			name: "wall only, rounded to the second",
			p:    Progress{Limits: Limits{Wall: 30 * time.Second}, Elapsed: 12_400 * time.Millisecond},
			want: "12s of 30s elapsed",
		},
		{
			name: "every dimension, in a fixed order",
			p: Progress{
				Limits:  Limits{CostUSD: 2, Steps: 10, ToolCalls: 5, Wall: 30 * time.Second},
				CostUSD: 0.5, Steps: 3, ToolCalls: 2, Elapsed: 12 * time.Second,
			},
			want: "cost $0.5000 of $2.00, step 3 of 10, 2 of 5 tool calls, 12s of 30s elapsed",
		},
		{
			name: "dimensions with no cap are omitted",
			p:    Progress{Limits: Limits{Steps: 10, SubagentDepth: 3, RepeatedToolCalls: 4}, Steps: 3, Depth: 2},
			want: "step 3 of 10",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.p.String(); got != tt.want {
				t.Errorf("String() =\n  %q\nwant\n  %q", got, tt.want)
			}
		})
	}
}

// ===== NoticeAt =====

func TestNoticeAtStaysSilentBelowTheThreshold(t *testing.T) {
	notice := NoticeAt(0.8)

	t.Run("outside any budget", func(t *testing.T) {
		if got := notice(context.Background(), 1); got != "" {
			t.Errorf("notice = %q, want \"\" outside a budget", got)
		}
	})

	t.Run("far from the cap", func(t *testing.T) {
		ctx, cancel := WithBudget(context.Background(), Limits{Steps: 100})
		defer cancel()
		FromContext(ctx).Step()
		if got := notice(ctx, 1); got != "" {
			t.Errorf("notice = %q, want \"\" at 1 of 100 steps", got)
		}
	})

	t.Run("just below the threshold", func(t *testing.T) {
		ctx, cancel := WithBudget(context.Background(), Limits{Steps: 10})
		defer cancel()
		b := FromContext(ctx)
		for i := 0; i < 7; i++ {
			b.Step()
		}
		if got := notice(ctx, 1); got != "" {
			t.Errorf("notice = %q, want \"\" at 0.7 of a 0.8 threshold", got)
		}
	})
}

// TestNoticeAtSpeaksWithRoomLeft is the point of the whole mechanism: a
// governed run has no way to wind down, so it spends up to the cap and is
// guillotined mid-thought. The notice has to arrive while the run can still
// act on it — warning a run that is already dead is just a log line.
func TestNoticeAtSpeaksWithRoomLeft(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{Steps: 10})
	defer cancel()
	b := FromContext(ctx)
	for i := 0; i < 8; i++ {
		b.Step()
	}

	got := NoticeAt(0.8)(ctx, 1)
	if got == "" {
		t.Fatal("notice = \"\" at 8 of 10 steps, want a warning at the 0.8 threshold")
	}
	for _, want := range []string{"<budget_status>", "</budget_status>", "step 8 of 10", "finish and answer now"} {
		if !strings.Contains(got, want) {
			t.Errorf("notice = %q, want it to contain %q", got, want)
		}
	}

	// Room still left to act on the warning.
	if err := ctx.Err(); err != nil {
		t.Fatalf("ctx.Err() = %v (cause %v), want nil: the warning must arrive before the guillotine",
			err, context.Cause(ctx))
	}
	b.Step() // step 9 of 10: still alive
	if err := ctx.Err(); err != nil {
		t.Fatalf("after the 9th of 10 steps: ctx.Err() = %v, want nil", err)
	}
	b.Step() // step 10 of 10: now the cap
	if got := context.Cause(ctx); !errors.Is(got, ErrStepLimit) {
		t.Errorf("context.Cause = %v, want %v", got, ErrStepLimit)
	}
}

func TestNoticeAtThresholds(t *testing.T) {
	// 8 of 10 steps == a fraction of exactly 0.8.
	tests := []struct {
		threshold float64
		wantSpeak bool
	}{
		{0.5, true},
		{0.8, true}, // the comparison is `< threshold`, so exactly at it speaks
		{0.81, false},
		{0.95, false},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("threshold=%v", tt.threshold), func(t *testing.T) {
			ctx, cancel := WithBudget(context.Background(), Limits{Steps: 10})
			defer cancel()
			b := FromContext(ctx)
			for i := 0; i < 8; i++ {
				b.Step()
			}
			got := NoticeAt(tt.threshold)(ctx, 1)
			if speaks := got != ""; speaks != tt.wantSpeak {
				t.Errorf("notice = %q (speaks=%v), want speaks=%v", got, speaks, tt.wantSpeak)
			}
		})
	}
}

// ===== concurrency =====

// TestConcurrentAccounting: sub-agents run on separate goroutines and share one
// budget, so a fan-out must not be able to spend past the cap by racing. Run
// with -race.
func TestConcurrentAccounting(t *testing.T) {
	const goroutines, per = 8, 250

	ctx, cancel := WithBudget(context.Background(), Limits{
		CostUSD: 1e6, Steps: 1e6, ToolCalls: 1e6, SubagentDepth: 4,
	})
	defer cancel()
	b := FromContext(ctx)

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < per; i++ {
				b.AddCall(0.001, Tokens{In: 1, Out: 2, CacheWrite: 3, CacheRead: 4})
				b.Step()
				b.AddToolCall(fmt.Sprintf("tool:%d:%d", g, i))
				if b.EnterSubagent() {
					b.ExitSubagent()
				}
				_ = b.Progress().Fraction()
				_ = b.Progress().String()
			}
		}(g)
	}
	wg.Wait()

	total := goroutines * per
	p := b.Progress()
	if p.Calls != total {
		t.Errorf("Calls = %d, want %d", p.Calls, total)
	}
	if p.Steps != total {
		t.Errorf("Steps = %d, want %d", p.Steps, total)
	}
	if p.ToolCalls != total {
		t.Errorf("ToolCalls = %d, want %d", p.ToolCalls, total)
	}
	if p.Depth != 0 {
		t.Errorf("Depth = %d, want 0: every Enter was paired with an Exit", p.Depth)
	}
	wantTokens := Tokens{In: total, Out: 2 * total, CacheWrite: 3 * total, CacheRead: 4 * total}
	if p.Tokens != wantTokens {
		t.Errorf("Tokens = %+v, want %+v", p.Tokens, wantTokens)
	}
	if err := ctx.Err(); err != nil {
		t.Errorf("ctx.Err() = %v (cause %v), want nil: the caps were far away", err, context.Cause(ctx))
	}
}

// TestConcurrentCapTripsOnce: many goroutines blowing through one cap at the
// same time must still produce exactly one, correct cause.
func TestConcurrentCapTripsOnce(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{CostUSD: 1.00})
	defer cancel()
	b := FromContext(ctx)

	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				b.AddCall(0.01, Tokens{In: 1})
			}
		}()
	}
	wg.Wait()

	waitDone(t, ctx)
	if got := context.Cause(ctx); !errors.Is(got, ErrBudgetExhausted) {
		t.Errorf("context.Cause = %v, want %v", got, ErrBudgetExhausted)
	}
	if got := b.Progress().Calls; got != 800 {
		t.Errorf("Calls = %d, want 800: tripping a cap must not stop the accounting", got)
	}
}

// TestConcurrentSubagentDepthCap: the depth cap must hold under a fan-out, and
// must still not cancel the run when it refuses.
func TestConcurrentSubagentDepthCap(t *testing.T) {
	ctx, cancel := WithBudget(context.Background(), Limits{SubagentDepth: 3})
	defer cancel()
	b := FromContext(ctx)

	const contenders = 20
	results := make(chan bool, contenders)
	release := make(chan struct{})

	var wg sync.WaitGroup
	for g := 0; g < contenders; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ok := b.EnterSubagent()
			results <- ok
			if !ok {
				return
			}
			// Hold the level until everyone has had their turn, so the count
			// below cannot be inflated by a level being recycled.
			<-release
			b.ExitSubagent()
		}()
	}

	granted := 0
	for i := 0; i < contenders; i++ {
		select {
		case ok := <-results:
			if ok {
				granted++
			}
		case <-time.After(2 * time.Second):
			close(release)
			t.Fatalf("only %d of %d goroutines reported, want all", i, contenders)
		}
	}
	if got := b.Progress().Depth; got != 3 {
		t.Errorf("Progress().Depth = %d while all levels are held, want 3", got)
	}
	close(release)
	wg.Wait()

	if granted != 3 {
		t.Errorf("granted = %d levels, want exactly 3", granted)
	}
	if got := b.Progress().Depth; got != 0 {
		t.Errorf("Progress().Depth = %d, want 0 after unwinding", got)
	}
	if cause := context.Cause(ctx); cause != nil {
		t.Errorf("context.Cause = %v, want nil: refusals must never cancel", cause)
	}
}
