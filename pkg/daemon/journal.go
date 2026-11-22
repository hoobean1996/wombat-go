package daemon

import (
	"context"
	"slices"
	"sync"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// journal is a turn's [core.Journal]: it writes the turn down in the ledger as
// it goes — every reply of the model before anything the reply asks for
// happens, and every call that is not safe to run again as it starts and as it
// ends — and, with each reply, where the turn has got to, beside the session:
// what a daemon that takes the session over carries the turn on from. See
// [Store.OpenStep].
//
// It is the turn's [core.Keyer] too, naming each call as [CallKey] does, and
// its [core.Nester]: the sub-agent a delegate call starts is written down the
// same way, under the call, and carried on from where it got to when the call
// runs again.
//
// Every write fences on the lease the turn began under, and is made whatever
// the run's context says: a turn being stopped is when writing down that a
// call was interrupted matters most.
type journal struct {
	m       *Manager
	session string
	owner   string
	epoch   int64
	turn    int

	// parent and run are, for a sub-agent's journal, the call that started
	// it and which of that call's sub-agents it is; empty for the session's
	// own.
	parent string
	run    int

	// frame is what each of the turn's steps says of it besides how far its
	// transcript has got — see [Inflight] — for the session's own journal:
	// Base is how much of the session's transcript the turn carries on from,
	// and Folds and Counted are what runs of the turn before this one left.
	frame Inflight

	// opened tells the session where its turn has got to, so the record it
	// writes at the turn's end carries it.
	opened func(Inflight)

	// ready holds the first write back until the record the turn began with
	// is written: that write would otherwise land on top of it.
	ready <-chan struct{}

	mu sync.Mutex
	// step is the step open — the one the transcript ends on — and index
	// its calls, by tool_use id.
	step  int
	index map[llm.ToolUseID]int
	// tracked are the calls of the open step this run wrote down as running.
	tracked map[int]bool
	// decided are decisions on calls of the open step, written with the
	// step's Close.
	decided map[llm.ToolUseID]*Decision
	// folds are the compactions the run had made as of the open step.
	folds []int
	// taken are the events the turn took from the inbox as it went: what
	// steered it, a cancellation; offers what it is offered to take next.
	taken, offers []Event
}

// take records events the turn took from the inbox as it went: they are the
// turn's from now on, and settled with it.
func (j *journal) take(evs []Event) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.taken = append(j.taken, evs...)
	j.frame.Events = slices.Concat(j.frame.Events, eventIDs(evs))
}

// offer gives the turn events to take at its next model call.
func (j *journal) offer(evs []Event) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.offers = append(j.offers, evs...)
}

// offered removes and returns what was offered and not taken.
func (j *journal) offered() []Event {
	j.mu.Lock()
	defer j.mu.Unlock()
	out := j.offers
	j.offers = nil
	return out
}

func (j *journal) takenEvents() []Event {
	j.mu.Lock()
	defer j.mu.Unlock()
	return slices.Clone(j.taken)
}

var (
	_ core.Journal   = (*journal)(nil)
	_ core.Keyer     = (*journal)(nil)
	_ core.Nester    = (*journal)(nil)
	_ core.Suspender = (*journal)(nil)
)

// reopen makes the step the transcript msgs ends on the open one, without
// writing anything: the step a resumed run starts on, which the ledger has.
func (j *journal) reopen(step int, msgs []llm.Message) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.step, j.index, j.tracked = step, map[llm.ToolUseID]int{}, map[int]bool{}
	if n := len(msgs); n > 0 && msgs[n-1].Role == llm.RoleAssistant {
		for i, u := range llm.ToolUses(msgs[n-1].Content) {
			if _, dup := j.index[u.ID]; !dup {
				j.index[u.ID] = i
			}
		}
	}
}

// at is the step the journal has open, 0 for none.
func (j *journal) at() int {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.step
}

// write is a context for one write: the run's values, none of its
// cancellation, and the daemon's own bound.
func (j *journal) write(ctx context.Context) (context.Context, context.CancelFunc) {
	if j.ready != nil {
		<-j.ready
	}
	return context.WithTimeout(context.WithoutCancel(ctx), storeTimeout)
}

func (j *journal) Open(ctx context.Context, st core.Step) error {
	o := StepOpen{Session: j.session, Owner: j.owner, Epoch: j.epoch, Turn: j.turn, Parent: j.parent, Run: j.run, At: time.Now()}
	for _, c := range st.Calls {
		o.Calls = append(o.Calls, ToolCall{Index: c.Index, UseID: c.Use.ID, Tool: c.Use.Name, Input: c.Use.Input})
	}
	var f Inflight
	if j.parent == "" {
		f = j.inflightAt(ctx, st)
		o.Inflight = &f
	} else {
		o.Sub = &SubRun{Messages: st.Messages, Iterations: st.Iteration, State: st.State}
	}
	wctx, cancel := j.write(ctx)
	defer cancel()
	step, err := j.m.store.OpenStep(wctx, o)
	if err != nil {
		return err
	}
	j.mu.Lock()
	j.step, j.index, j.tracked, j.decided = step, make(map[llm.ToolUseID]int, len(st.Calls)), map[int]bool{}, nil
	j.folds = st.Folds
	for _, c := range st.Calls {
		if _, dup := j.index[c.Use.ID]; !dup {
			j.index[c.Use.ID] = c.Index
		}
	}
	j.mu.Unlock()
	if j.parent == "" && j.opened != nil {
		f.Step = step
		j.opened(f)
	}
	return nil
}

// inflightAt is where the turn has got to at st: what the session's
// transcript carries on with — all of it, once the turn has compacted what
// came before — and what the turn has used of its budget.
func (j *journal) inflightAt(ctx context.Context, st core.Step) Inflight {
	j.mu.Lock()
	f := j.frame
	f.Events = slices.Clone(f.Events)
	j.mu.Unlock()
	f.Iterations, f.State, f.Suspended = st.Iteration, st.State, false
	f.Folds = slices.Concat(j.frame.Folds, st.Folds)
	if len(f.Folds) > 0 || f.Base < 0 || f.Base > len(st.Messages) {
		f.Base, f.Messages = -1, st.Messages
	} else {
		f.Messages = st.Messages[f.Base:]
	}
	f.Used = governor.FromContext(ctx).Used()
	return f
}

// Suspend writes down where the run stopped at the end of its slice: the
// whole transcript — a compaction since the last step would make a tail of
// it wrong — on the step it stopped in, which stays open.
func (j *journal) Suspend(ctx context.Context, cp core.Checkpoint) error {
	j.mu.Lock()
	step, folds, f := j.step, j.folds, j.frame
	f.Events = slices.Clone(f.Events)
	j.mu.Unlock()
	p := Progress{Session: j.session, Owner: j.owner, Epoch: j.epoch, At: time.Now()}
	if j.parent == "" {
		f.Step, f.Base, f.Messages, f.Iterations = step, -1, cp.Messages, cp.Iterations
		f.State, f.Suspended, f.Used = cp.State, true, cp.Used
		f.Folds = slices.Concat(j.frame.Folds, folds)
		p.Inflight = &f
	} else {
		p.Parent, p.Run = j.parent, j.run
		p.Sub = &SubRun{Step: step, Messages: cp.Messages, Iterations: cp.Iterations, State: cp.State}
	}
	wctx, cancel := j.write(ctx)
	defer cancel()
	if err := j.m.store.SaveProgress(wctx, p); err != nil {
		return err
	}
	if j.parent == "" && j.opened != nil {
		j.opened(f)
	}
	return nil
}

func (j *journal) Track(ctx context.Context, c core.Call) error {
	m := moveOf(c)
	j.mu.Lock()
	step := j.step
	if c.State == tool.StateRunning {
		m.Attempt = true
		j.tracked[c.Index] = true
	}
	j.mu.Unlock()
	return j.move(ctx, step, []CallMove{m})
}

func (j *journal) Close(ctx context.Context, calls []core.Call) error {
	j.mu.Lock()
	step := j.step
	moves := make([]CallMove, 0, len(calls))
	for _, c := range calls {
		m := moveOf(c)
		// A call that ran without being written down as running — one that
		// is simply run again should the turn be resumed — counts here.
		m.Attempt = c.Ran && !j.tracked[c.Index]
		m.Decision = j.decided[c.Use.ID]
		moves = append(moves, m)
	}
	j.mu.Unlock()
	return j.move(ctx, step, moves)
}

func (j *journal) move(ctx context.Context, step int, moves []CallMove) error {
	if step == 0 || len(moves) == 0 {
		// A step this journal never opened: one written before the ledger
		// was, which there is nothing in it to move.
		return nil
	}
	wctx, cancel := j.write(ctx)
	defer cancel()
	return j.m.store.MoveCalls(wctx, CallMoves{Session: j.session, Owner: j.owner, Epoch: j.epoch, Step: step, Moves: moves, At: time.Now()})
}

func (j *journal) Key(id llm.ToolUseID) string {
	j.mu.Lock()
	defer j.mu.Unlock()
	i, ok := j.index[id]
	if !ok || j.step == 0 {
		return ""
	}
	return CallKey(j.session, j.step, i)
}

func (j *journal) Nest(ctx context.Context, key string, n int) (core.Journal, *core.Checkpoint, error) {
	sub := &journal{m: j.m, session: j.session, owner: j.owner, epoch: j.epoch, turn: j.turn, parent: key, run: n, ready: j.ready}
	sub.reopen(0, nil)
	rctx, cancel := j.write(ctx)
	defer cancel()
	sr, ok, err := j.m.store.SubRun(rctx, j.session, key, n)
	if err != nil || !ok {
		return sub, nil, err
	}
	rows, err := j.m.store.ToolCalls(rctx, CallFilter{Session: j.session, Step: sr.Step})
	if err != nil {
		return nil, nil, err
	}
	sub.reopen(sr.Step, sr.Messages)
	return sub, &core.Checkpoint{Messages: sr.Messages, Calls: callsOf(rows), Iterations: sr.Iterations, State: sr.State}, nil
}
