package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"time"

	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// ===== Slices =====
//
// A run need not be run in one go. Given a time to stop by ([Input.Until]), it
// stops on its own at the first step boundary where going on could run past
// it — before a model call, or before a batch of tool calls — and ends in
// [Suspended], with a [Checkpoint] that [Recover] carries it on from, in this
// process or another. A run carried on in slices ends exactly as one run in
// one go would: the checkpoint holds the transcript, the calls of the step it
// stopped in, and everything else the run keeps while it goes (see
// [Carried]). That is what lets a host that cannot keep a process alive for
// as long as a run takes — a serverless function, a job with a time limit —
// run agents anyway: each invocation runs one slice.

// DefaultCallEstimate is how long a model call is expected to take until a
// run has timed one; see [WithCallEstimate].
const DefaultCallEstimate = 60 * time.Second

// WithCallEstimate is how long a model call is expected to take, for a run
// with a time to stop by: one is not started with less than this left. A run
// that has timed a slower call expects that instead. Default
// [DefaultCallEstimate].
func WithCallEstimate(d time.Duration) Option {
	return func(c *config) { c.callEstimate = d }
}

// Suspended is a run that stopped on purpose, at a step boundary, because its
// slice was up ([Input.Until]). Nothing failed. Checkpoint is where it got to:
// [Recover] carries it on. A run whose journal is a [Suspender] has written it
// there too.
type Suspended struct {
	Checkpoint Checkpoint
}

func (Suspended) isOutcome() {}

// Suspender is a [Journal] that writes down where a run stopped when it stops
// at the end of its slice. A run with a journal that is not one does not stop
// for its slice when it is a sub-agent — nothing would carry it on — and
// leaves the checkpoint to its host when it is not.
//
// Like the journal's other writes, it is made with the run's context, which
// may be done by then, and must be durable when it returns.
type Suspender interface {
	Suspend(ctx context.Context, cp Checkpoint) error
}

// Until makes the run stop at the first step boundary where going on could
// run past t, and end in [Suspended]. A slice always does one unit of work
// first — a model call, or a batch of tool calls — so a run carried on in
// slices always gets somewhere, however short they are.
//
// Before a model call, the time left must cover the slowest call the run has
// timed, or [WithCallEstimate]; before a batch, the longest timeout of its
// calls — one with no timeout is only started at the start of a slice. A tool
// call is never stopped part-way: one that cannot finish in a slice should
// end its call with [tool.Await].
//
// The sub-agents a delegate call runs stop with the run, when its journal
// journals them ([Nester]): the call carries on in the next slice.
func (in Input) Until(t time.Time) Input {
	in.until = t
	return in
}

// Steer queues messages for the run to take before its next model call, as
// [Run.Steer] does once it is going: for a run carried on that was told
// something while it was stopped.
func (in Input) Steer(msgs ...string) Input {
	in.steer = append(slices.Clone(in.steer), msgs...)
	return in
}

// SteerFrom gives the run a source of messages from outside its process,
// asked at every point the run can take one — the next model call — for
// whatever has arrived. What it returns is delivered as [Run.Steer]'s is.
func (in Input) SteerFrom(src func(context.Context) []string) Input {
	in.steerFrom = src
	return in
}

// suspendedReason is the reason a call that carries on in the next slice is
// written down with: running still, and to be run again, which carries it on.
const suspendedReason = "carries on in the next slice"

type sliceKey struct{}

// withSlice puts the time the run stops by on its context, for the
// sub-agents it starts.
func withSlice(ctx context.Context, t time.Time) context.Context {
	return context.WithValue(ctx, sliceKey{}, t)
}

func sliceFrom(ctx context.Context) time.Time {
	t, _ := ctx.Value(sliceKey{}).(time.Time)
	return t
}

// mayStop reports whether the run should stop here rather than start
// something that needs need — negative for no bound — with the time it has
// left. Never before the slice's first unit of work.
func (r *Run) mayStop(need time.Duration) bool {
	if r.until.IsZero() || r.units == 0 {
		return false
	}
	if need < 0 {
		return true
	}
	return time.Until(r.until) < need
}

// modelNeed is the time a model call is expected to take.
func (r *Run) modelNeed() time.Duration {
	est := r.a.cfg.callEstimate
	if est <= 0 {
		est = DefaultCallEstimate
	}
	return max(est, r.modelMax)
}

// batchNeed is the longest a batch of calls may take: the longest timeout of
// its calls, or -1 when one has none.
func (r *Run) batchNeed(uses []llm.ToolUse) time.Duration {
	var need time.Duration
	for _, u := range uses {
		d, ok := r.a.set.Find(u.Name)
		if !ok {
			continue
		}
		t := d.Timeout
		if t == 0 {
			t = r.a.toolFallback()
		}
		if t <= 0 {
			return -1
		}
		need = max(need, t)
	}
	return need
}

// suspend ends the run in Suspended where it stands: the transcript convo,
// the calls of the step it ends on, if it ends on one, and iterations made.
func (r *Run) suspend(ctx context.Context, convo Convo, calls []Call, iterations int) (Convo, bool) {
	cp := Checkpoint{
		Messages:   convo,
		Calls:      calls,
		Iterations: iterations,
		State:      r.state(),
		Used:       governor.FromContext(ctx).Used(),
	}
	if r.journal != nil {
		if s, ok := r.journal.j.(Suspender); ok {
			if err := s.Suspend(ctx, cp); err != nil {
				r.finish(nil, journalErr("writing down where the run stopped", err))
				return convo, true
			}
		}
	}
	r.finish(Suspended{Checkpoint: cp}, nil)
	return convo, true
}

// suspendedIn reports whether a call of the batch carries on in the next
// slice.
func suspendedIn(results []tool.Result) bool {
	return slices.ContainsFunc(results, func(res tool.Result) bool { return errors.Is(res.Err, tool.ErrSuspended) })
}

// ===== What a run carries from slice to slice =====

// Carried is state a run keeps while it goes, on its context, that a run
// carried on in slices must have back when it carries on: what a person
// allowed for the turn, which skills are loaded, how often a call has failed
// the same way. Registered with [Carry], it is written into every checkpoint
// the run makes and put back by a run [Recover] resumes.
type Carried interface {
	MarshalState() (json.RawMessage, error)
	UnmarshalState(json.RawMessage) error
}

type carriedKey struct{}

// carried is one Carried registered on a context, and the ones registered
// before it.
type carried struct {
	name string
	c    Carried
	up   *carried
}

// Carry registers c under name for the run started on ctx — or, from a
// [WithRunContext] decorator, for the run being started — to carry from slice
// to slice. A name registered again later shadows the earlier one.
//
// A sub-agent's run carries what is registered for it alone: what it shares
// with its parent, on the context it inherits, is its parent's to carry.
func Carry(ctx context.Context, name string, c Carried) context.Context {
	return context.WithValue(ctx, carriedKey{}, &carried{name: name, c: c, up: carriedFrom(ctx)})
}

func carriedFrom(ctx context.Context) *carried {
	c, _ := ctx.Value(carriedKey{}).(*carried)
	return c
}

// carriedParts is what the run carries, by name, innermost first: from what
// is registered on its context up to, and not including, base.
func (r *Run) carriedParts() map[string]Carried {
	out := map[string]Carried{}
	for c := r.carry; c != nil && c != r.carryBase; c = c.up {
		if _, shadowed := out[c.name]; !shadowed {
			out[c.name] = c.c
		}
	}
	return out
}

// runState is what a checkpoint holds of a run besides its transcript: the
// loop's own counters, and what is registered with Carry.
type runState struct {
	EmptyTurns     int                          `json:"empty_turns,omitempty"`
	Truncated      int                          `json:"truncated,omitempty"`
	TruncNotice    string                       `json:"trunc_notice,omitempty"`
	CompactRetryAt int                          `json:"compact_retry_at,omitempty"`
	Trimmed        [2]int                       `json:"trimmed,omitzero"`
	Rejections     int                          `json:"rejections,omitempty"`
	Results        map[llm.ToolUseID]ResultInfo `json:"results,omitempty"`
	PerByte        float64                      `json:"per_byte,omitempty"`
	Calibrated     bool                         `json:"calibrated,omitempty"`
	ModelMillis    int64                        `json:"model_ms,omitempty"`
	Parts          map[string]json.RawMessage   `json:"parts,omitempty"`
}

// state is the run's state for a checkpoint. Loop-owned.
func (r *Run) state() json.RawMessage {
	st := runState{
		EmptyTurns: r.emptyTurns, Truncated: r.truncated, TruncNotice: r.truncNotice,
		CompactRetryAt: r.compactRetryAt, Trimmed: r.trimmed, Rejections: r.rejections,
		Results: r.results, ModelMillis: r.modelMax.Milliseconds(),
	}
	if r.tokens != nil {
		st.PerByte, st.Calibrated = r.tokens.perByte, r.tokens.calibrated
	}
	for name, c := range r.carriedParts() {
		b, err := c.MarshalState()
		if err != nil {
			r.a.log.Warn("wombat: run state not carried", slog.String("part", name), slog.Any("err", err))
			continue
		}
		if st.Parts == nil {
			st.Parts = map[string]json.RawMessage{}
		}
		st.Parts[name] = b
	}
	b, err := json.Marshal(st)
	if err != nil {
		r.a.log.Warn("wombat: run state not carried", slog.Any("err", err))
		return nil
	}
	return b
}

// restore puts back a checkpoint's state. Advisory: a part that will not go
// back is said and skipped, and the run carries on without it.
func (r *Run) restore(b json.RawMessage) error {
	if len(b) == 0 {
		return nil
	}
	var st runState
	if err := json.Unmarshal(b, &st); err != nil {
		return fmt.Errorf("the run's state: %w", err)
	}
	r.emptyTurns, r.truncated, r.truncNotice = st.EmptyTurns, st.Truncated, st.TruncNotice
	r.compactRetryAt, r.trimmed, r.rejections = st.CompactRetryAt, st.Trimmed, st.Rejections
	for id, info := range st.Results {
		r.results[id] = info
	}
	r.modelMax = time.Duration(st.ModelMillis) * time.Millisecond
	if r.tokens != nil && st.Calibrated && st.PerByte > 0 {
		r.tokens.perByte, r.tokens.calibrated = st.PerByte, true
	}
	var errs []error
	for name, c := range r.carriedParts() {
		part, ok := st.Parts[name]
		if !ok {
			continue
		}
		if err := c.UnmarshalState(part); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", name, err))
		}
	}
	return errors.Join(errs...)
}
