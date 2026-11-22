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
//
// Steps and ToolCalls are checked BEFORE the unit they count — an iteration
// about to start, a call about to run — so a limit of N lets exactly N happen
// and refuses the next. CostUSD can only be checked after the fact, because a
// call's price is known once it has been paid: the call that reaches the cap
// completes, and the run stops there.
type Limits struct {
	CostUSD float64
	// Steps is how many iterations the run may take, sub-agents' included.
	Steps int
	Wall  time.Duration
	// ToolCalls is how many tool calls the run may make. Only calls that
	// reach execution count: one a permission policy refused never ran.
	ToolCalls     int
	SubagentDepth int

	// RepeatedToolCalls stops a run that calls the same tool with the same
	// arguments this many times in a row: the Nth identical call is refused
	// rather than run. Refused calls count here, unlike for ToolCalls —
	// asking again for a call that was just refused is the same loop.
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
	return WithBudgetFrom(parent, l, Used{})
}

// Used is what a run had used of its budget, for a run that carries on from
// it; see [WithBudgetFrom]. It survives a process boundary.
type Used struct {
	CostUSD      float64       `json:"cost_usd,omitempty"`
	Calls        int           `json:"calls,omitempty"`
	Steps        int           `json:"steps,omitempty"`
	ToolCalls    int           `json:"tool_calls,omitempty"`
	InputTokens  int           `json:"input_tokens,omitempty"`
	OutputTokens int           `json:"output_tokens,omitempty"`
	CacheWrite   int           `json:"cache_write_tokens,omitempty"`
	CacheRead    int           `json:"cache_read_tokens,omitempty"`
	Elapsed      time.Duration `json:"elapsed_ns,omitempty"`

	// LastCall and RepeatRun are where loop detection had got to (see
	// Limits.RepeatedToolCalls): a run carried on in slices is still one
	// run, and a loop that spans a slice is still a loop.
	LastCall  string `json:"last_call,omitempty"`
	RepeatRun int    `json:"repeat_run,omitempty"`
}

// Used is what p has used, to carry on from.
func (p Progress) Used() Used {
	return Used{
		CostUSD: p.CostUSD, Calls: p.Calls, Steps: p.Steps, ToolCalls: p.ToolCalls,
		InputTokens: p.Tokens.In, OutputTokens: p.Tokens.Out, CacheWrite: p.Tokens.CacheWrite, CacheRead: p.Tokens.CacheRead,
		Elapsed: p.Elapsed,
	}
}

// Used is what b has used, loop detection included, to carry on from with
// [WithBudgetFrom].
func (b *Budget) Used() Used {
	u := b.Progress().Used()
	b.mu.Lock()
	u.LastCall, u.RepeatRun = b.lastCall, b.repeatRun
	b.mu.Unlock()
	return u
}

// WithBudgetFrom is [WithBudget] for a run that carries on from an earlier
// one — the same turn, resumed after its process died — and starts out
// charged with what that one used, so a limit caps the two together rather
// than starting again at every resume. The wall clock counts the earlier
// run's elapsed time, not the time between the two. A cost cap the earlier
// run had reached cancels the context at once; a count is refused at the
// next unit, as it always is.
func WithBudgetFrom(parent context.Context, l Limits, u Used) (context.Context, context.CancelFunc) {
	ctx := parent
	start := time.Now().Add(-max(u.Elapsed, 0))
	var deadlineCancel context.CancelFunc
	if l.Wall > 0 {
		ctx, deadlineCancel = context.WithDeadlineCause(ctx, start.Add(l.Wall), ErrWallClock)
	}

	ctx, cancelCause := context.WithCancelCause(ctx)
	b := &Budget{
		limits: l, start: start, cancel: cancelCause,
		usd: u.CostUSD, calls: u.Calls, steps: u.Steps, toolCalls: u.ToolCalls,
		tokens:   Tokens{In: u.InputTokens, Out: u.OutputTokens, CacheWrite: u.CacheWrite, CacheRead: u.CacheRead},
		lastCall: u.LastCall, repeatRun: u.RepeatRun,
	}
	if l.CostUSD > 0 && b.usd >= l.CostUSD {
		cancelCause(ErrBudgetExhausted)
	}
	ctx = context.WithValue(ctx, budgetKey{}, b)

	return ctx, func() {
		cancelCause(nil)
		if deadlineCancel != nil {
			deadlineCancel()
		}
	}
}

// FromContext returns the budget attached to ctx, or a fresh unlimited one.
//
// It never returns nil. Callers therefore never nil-check, and code that runs
// outside a governed context — a unit test, a one-off script — behaves as if
// there were no caps rather than crashing.
//
// The fallback is a throwaway, not a shared default: whatever is charged to it
// is recorded nowhere else. It used to be one process-wide budget that every
// ungoverned caller charged, so a run's Progress included the calls of every
// other ungoverned run in the process, and the sub-agent depth one run saw was
// whatever concurrent runs had left it at. A run that wants a tally without any
// caps gets its own from [Ensure], which the agent loop calls for every run.
func FromContext(ctx context.Context) *Budget {
	if b, ok := ctx.Value(budgetKey{}).(*Budget); ok && b != nil {
		return b
	}
	return &Budget{start: time.Now()}
}

// Lookup returns the budget ctx carries, and whether it carries one at all —
// for a caller that has somewhere better than a throwaway to put a tally.
func Lookup(ctx context.Context) (*Budget, bool) {
	b, ok := ctx.Value(budgetKey{}).(*Budget)
	return b, ok && b != nil
}

// Ensure returns ctx unchanged if it already carries a budget, and otherwise
// ctx with a fresh, unlimited one attached.
//
// It is how a run that was given no limits still has a tally of its own: the
// cost middleware, the loop's step counter and the delegate tools' depth
// accounting all charge whatever [FromContext] finds, and without this they
// would each charge a different throwaway. Everything that inherits the
// returned context — sub-agents in particular — shares the one budget, exactly
// as it would share one installed by [WithBudget].
//
// No cancel func, because an unlimited budget never cancels anything.
func Ensure(ctx context.Context) context.Context {
	if b, ok := ctx.Value(budgetKey{}).(*Budget); ok && b != nil {
		return ctx
	}
	return context.WithValue(ctx, budgetKey{}, &Budget{start: time.Now()})
}

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
	if b.limits.Steps > 0 && b.steps > b.limits.Steps {
		b.abort(ErrStepLimit)
	}
}

// AddToolCall records one tool call that is about to run: it counts toward
// ToolCalls, and key feeds loop detection exactly as [Budget.Attempt] does.
// It is the two halves together, for a caller with one place to put them.
func (b *Budget) AddToolCall(key string) {
	b.CountToolCall()
	b.Attempt(key)
}

// CountToolCall counts one tool call that is about to run against ToolCalls,
// and refuses it — by cancelling the run — when the run has already made
// every call it may.
func (b *Budget) CountToolCall() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.toolCalls++
	if b.limits.ToolCalls > 0 && b.toolCalls > b.limits.ToolCalls {
		b.abort(ErrToolCallLimit)
	}
}

// Attempt feeds loop detection with one tool call the model asked for,
// whether it goes on to run or is refused: asking again for a call that was
// just refused is as much a loop as repeating one that ran.
//
// key identifies the call — conventionally the tool name plus its arguments.
// "" is a call that is not part of any loop, and breaks the current streak.
func (b *Budget) Attempt(key string) {
	b.mu.Lock()
	defer b.mu.Unlock()
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
// lets it decide to summarise now instead. Wire it with core.WithTurnNotice.
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
