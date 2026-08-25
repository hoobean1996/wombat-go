// Package governor enforces run-level budgets — cost, steps, wall clock,
// tool calls, sub-agent depth.
//
// Enforcement is a context cancellation, which is the whole design. An
// effect-based runtime has to invent a way to abort from arbitrary depth and
// then be careful that nobody swallows it; here, exceeding a limit cancels the
// context, and every blocking operation in the process — HTTP requests,
// exec.CommandContext, errgroup members, channel selects — already unwinds on
// its own. No call site needs a new return value.
//
//	ctx, cancel := governor.WithBudget(ctx, governor.Limits{CostUSD: 1.00})
//	defer cancel()
//	...
//	if errors.Is(context.Cause(ctx), governor.ErrBudgetExhausted) { ... }
package governor

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Reasons a run was stopped. Recover them with context.Cause(ctx).
var (
	ErrBudgetExhausted = errors.New("governor: cost budget exhausted")
	ErrStepLimit       = errors.New("governor: step limit reached")
	ErrWallClock       = errors.New("governor: wall-clock limit reached")
	ErrToolCallLimit   = errors.New("governor: tool-call limit reached")
	ErrDepthLimit      = errors.New("governor: sub-agent depth limit reached")
	ErrToolLoop        = errors.New("governor: repeated identical tool call limit reached")
)

// Limits caps a run. A zero field means that dimension is unlimited.
type Limits struct {
	CostUSD       float64
	Steps         int
	Wall          time.Duration
	ToolCalls     int
	SubagentDepth int

	// RepeatedToolCalls stops a run that calls the same tool with the same
	// arguments this many times in a row.
	//
	// Distinct from the dedup middleware, which annotates the error so the
	// model can notice it is stuck and pivot. That is advisory and usually
	// works. This is the backstop for when it does not: an agent looping on
	// an identical call is not making progress, and every iteration of the
	// loop costs money.
	RepeatedToolCalls int
}

// Tokens is per-call token accounting, kept as plain ints so this package
// depends on nothing but the standard library.
type Tokens struct {
	In, Out, CacheWrite, CacheRead int
}

// Budget is the live tally for one run. Safe for concurrent use: sub-agents
// running on separate goroutines share one budget, so a fan-out cannot spend
// past the cap by racing.
type Budget struct {
	limits Limits
	start  time.Time
	cancel context.CancelCauseFunc

	mu        sync.Mutex
	usd       float64
	calls     int
	steps     int
	toolCalls int
	depth     int
	tokens    Tokens

	lastCall  string
	repeatRun int
}

type budgetKey struct{}

// WithBudget attaches a Budget to ctx and returns a context that cancels when
// any limit is breached. The returned CancelFunc must be called to release
// resources, as with context.WithCancel.
func WithBudget(parent context.Context, l Limits) (context.Context, context.CancelFunc) {
	ctx := parent
	var deadlineCancel context.CancelFunc
	if l.Wall > 0 {
		ctx, deadlineCancel = context.WithDeadlineCause(ctx, time.Now().Add(l.Wall), ErrWallClock)
	}

	ctx, cancelCause := context.WithCancelCause(ctx)
	b := &Budget{limits: l, start: time.Now(), cancel: cancelCause}
	ctx = context.WithValue(ctx, budgetKey{}, b)

	return ctx, func() {
		cancelCause(nil)
		if deadlineCancel != nil {
			deadlineCancel()
		}
	}
}

// FromContext returns the budget attached to ctx, or an unlimited one.
//
// It never returns nil. Callers therefore never nil-check, and code that runs
// outside a governed context — a unit test, a one-off script — behaves as if
// there were no caps rather than crashing.
func FromContext(ctx context.Context) *Budget {
	if b, ok := ctx.Value(budgetKey{}).(*Budget); ok {
		return b
	}
	return unlimited
}

var unlimited = &Budget{start: time.Now()}

// abort cancels the run. Called with b.mu held.
func (b *Budget) abort(reason error) {
	if b.cancel != nil {
		b.cancel(reason)
	}
}

// AddCall records one completed model call and trips the cost cap if the spend
// now exceeds it. Called by the cost-tracking middleware, which is the only
// place that sees every response.
func (b *Budget) AddCall(usd float64, t Tokens) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.usd += usd
	b.calls++
	b.tokens.In += t.In
	b.tokens.Out += t.Out
	b.tokens.CacheWrite += t.CacheWrite
	b.tokens.CacheRead += t.CacheRead

	if b.limits.CostUSD > 0 && b.usd >= b.limits.CostUSD {
		b.abort(ErrBudgetExhausted)
	}
}

// Step records one agent iteration.
func (b *Budget) Step() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.steps++
	if b.limits.Steps > 0 && b.steps >= b.limits.Steps {
		b.abort(ErrStepLimit)
	}
}

// AddToolCall records one dispatched tool call.
//
// key identifies the call for loop detection — conventionally the tool name
// plus its arguments. Pass "" to count the call without loop detection.
func (b *Budget) AddToolCall(key string) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.toolCalls++
	if b.limits.ToolCalls > 0 && b.toolCalls >= b.limits.ToolCalls {
		b.abort(ErrToolCallLimit)
	}

	if key == "" {
		b.lastCall, b.repeatRun = "", 0
		return
	}
	if key == b.lastCall {
		b.repeatRun++
	} else {
		b.lastCall, b.repeatRun = key, 1
	}
	if b.limits.RepeatedToolCalls > 0 && b.repeatRun >= b.limits.RepeatedToolCalls {
		b.abort(ErrToolLoop)
	}
}

// EnterSubagent deepens the nesting level and reports whether it is allowed.
// The caller must pair a true return with ExitSubagent.
//
// A refusal does NOT abort the run, and that is the difference between this
// cap and every other one in this package. The others are resources the run is
// consuming — money, time, iterations — and hitting one means there is nothing
// left to continue with. Nesting depth is a shape, not a resource: refusing to
// go deeper leaves the caller perfectly able to do the work itself, and it
// only finds that out if it is still running. Aborting here would kill a
// parent that had a legitimate fallback, and take the transcript explaining
// why with it.
func (b *Budget) EnterSubagent() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.limits.SubagentDepth > 0 && b.depth >= b.limits.SubagentDepth {
		return false
	}
	b.depth++
	return true
}

// ExitSubagent unwinds one nesting level.
func (b *Budget) ExitSubagent() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.depth > 0 {
		b.depth--
	}
}

// Progress is a snapshot of spend against the caps.
type Progress struct {
	Limits    Limits
	CostUSD   float64
	Calls     int
	Steps     int
	ToolCalls int
	Depth     int
	Tokens    Tokens
	Elapsed   time.Duration

	// Repeats is the current run of identical consecutive tool calls.
	Repeats int
}

// CostFraction reports spend as a fraction of the cap, or 0 when uncapped.
func (p Progress) CostFraction() float64 {
	if p.Limits.CostUSD <= 0 {
		return 0
	}
	return p.CostUSD / p.Limits.CostUSD
}

// Fraction reports how close the run is to its nearest cap, across every
// dimension that has one. 0 when nothing is capped.
func (p Progress) Fraction() float64 {
	worst := 0.0
	consider := func(used, cap float64) {
		if cap > 0 {
			worst = max(worst, used/cap)
		}
	}
	consider(p.CostUSD, p.Limits.CostUSD)
	consider(float64(p.Steps), float64(p.Limits.Steps))
	consider(float64(p.ToolCalls), float64(p.Limits.ToolCalls))
	consider(p.Elapsed.Seconds(), p.Limits.Wall.Seconds())
	return worst
}

// String renders a one-line summary suitable for showing to a model.
func (p Progress) String() string {
	parts := make([]string, 0, 4)
	if p.Limits.CostUSD > 0 {
		parts = append(parts, fmt.Sprintf("cost $%.4f of $%.2f", p.CostUSD, p.Limits.CostUSD))
	}
	if p.Limits.Steps > 0 {
		parts = append(parts, fmt.Sprintf("step %d of %d", p.Steps, p.Limits.Steps))
	}
	if p.Limits.ToolCalls > 0 {
		parts = append(parts, fmt.Sprintf("%d of %d tool calls", p.ToolCalls, p.Limits.ToolCalls))
	}
	if p.Limits.Wall > 0 {
		parts = append(parts, fmt.Sprintf("%s of %s elapsed", p.Elapsed.Round(time.Second), p.Limits.Wall))
	}
	if len(parts) == 0 {
		return "no limits configured"
	}
	return strings.Join(parts, ", ")
}

// NoticeAt builds a per-turn notice that warns a model once a run passes
// threshold (0..1) of its nearest cap.
//
// The point is that a governed run currently has no way to wind down: it
// spends right up to the cap and is then guillotined mid-thought, discarding
// whatever it was about to produce. Telling the model how much room is left
// lets it decide to summarise now instead. Wire it with wombat.WithTurnNotice.
//
// Returns "" below the threshold, so the notice costs nothing until it
// matters — and, being appended to the last turn rather than the system
// prompt, it never disturbs the cached prefix.
func NoticeAt(threshold float64) func(context.Context, int) string {
	return func(ctx context.Context, _ int) string {
		p := FromContext(ctx).Progress()
		if p.Fraction() < threshold {
			return ""
		}
		return fmt.Sprintf(
			"<budget_status>%s. You are near a limit: finish and answer now rather than starting new work.</budget_status>",
			p)
	}
}

// Progress snapshots the current tally.
func (b *Budget) Progress() Progress {
	b.mu.Lock()
	defer b.mu.Unlock()
	return Progress{
		Limits:    b.limits,
		CostUSD:   b.usd,
		Calls:     b.calls,
		Steps:     b.steps,
		ToolCalls: b.toolCalls,
		Depth:     b.depth,
		Tokens:    b.tokens,
		Elapsed:   time.Since(b.start),
		Repeats:   b.repeatRun,
	}
}
