package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// ErrJournal means the run could not write down what it was about to do, or
// what it had done, and stopped rather than carry on unrecorded. Nothing it
// had not written down first was run; a host keeping the journal may resume
// the run from what was written.
var ErrJournal = errors.New("wombat: the journal could not be written")

// Journal is where a run writes down how far it has got, as it goes, so that
// one resumed after its process died — see [Recover] — carries on from there
// rather than starting over, and knows which of its tool calls ran.
//
// A run started on a context carrying one ([WithJournal]) calls it at three
// points, and each write must be durable when it returns:
//
//   - Open, once the model has replied and before anything the reply asks for
//     happens: the transcript, ending with the reply, and the reply's calls,
//     every one pending. It is the write-ahead of everything the step will do.
//     A reply that calls nothing — the model's answer — is opened too, so a
//     run resumed after it does not ask for the answer again.
//   - Track, as a call is about to take effect and again once it has. Only
//     for a call whose tool is not safe to run again (see [tool.RecoveryOf]):
//     any other is simply run again if the run is resumed, and needs no write
//     before it runs.
//   - Close, once every call of the step has its answer, or the run has
//     stopped at one: each call's state and result, in one write.
//
// An error from any of them stops the run with [ErrJournal]: a call that
// cannot be written down first must not run, and a run that cannot record
// what it did must not carry on as if it had. A journal is called with the
// run's context, which is already cancelled when the run is being stopped —
// and that is when writing down that a call was interrupted matters most — so
// it should write regardless.
//
// Calls to one journal do not overlap, except Track for the calls of one
// batch that run side by side.
type Journal interface {
	Open(ctx context.Context, s Step) error
	Track(ctx context.Context, c Call) error
	Close(ctx context.Context, calls []Call) error
}

// Keyer is a [Journal] that names the calls it records. A run installs the
// names for its tools to read, as [tool.CallKey]: a name that stays the same
// when a resumed run runs the call again is what lets a tool make its effect
// idempotent.
type Keyer interface {
	// Key names the call of the open step with this id, or returns "" for
	// one it does not know.
	Key(id llm.ToolUseID) string
}

// Nester is a [Journal] that also journals the runs a tool call starts — the
// sub-agent a delegate call runs — so that a call run again after its parent
// was resumed carries its sub-agent on from where it got to instead of
// starting it over.
type Nester interface {
	// Nest returns the journal for the n-th run the call named key starts
	// (see [tool.CallKey]), and where that run had got to when the call last
	// ran, or nil if it had not got anywhere.
	Nest(ctx context.Context, key string, n int) (Journal, *Checkpoint, error)
}

// Step is one reply of the model, written down before anything it asks for
// happens; see [Journal].
type Step struct {
	// Messages is the transcript, ending with the reply.
	Messages []llm.Message

	// Calls are the reply's tool calls, in order, each pending. None for a
	// reply that is the model's answer.
	Calls []Call

	// Iteration is the iteration of the run the reply was made in, counting
	// those of the run it was resumed from.
	Iteration int

	// Folds are the compactions the run has made so far, oldest first: each
	// folded messages 1 through n of the transcript into the first, as
	// [TranscriptCompacted] reports. A journal that keeps only what a turn
	// added to a transcript it already has must keep the whole of it once a
	// compaction rewrote what it had.
	Folds []int

	// State is what the run keeps besides its transcript, as a checkpoint
	// holds it (see [Checkpoint.State]): opaque, for a journal to keep and
	// give back.
	State json.RawMessage
}

// Call is one tool call of a step, as a journal records it.
type Call struct {
	// Index is its place in the reply, which is what identifies it within
	// the step: a tool_use id is the provider's, and some gateways reuse them.
	Index int

	Use   llm.ToolUse
	State tool.State

	// Result is what the model is shown for it, once there is something to
	// show: for a call that finished, was refused, cancelled or interrupted,
	// and for a waiting one once its result is posted.
	Result *llm.ToolResult

	// Tags are the labels its tool put on the result ([tool.Annotate]).
	Tags []string

	// Reason says why it waits — for a decision, for a result — or why it
	// did not run.
	Reason string

	// Handle names what a waiting call waits for ([tool.Await]).
	Handle string

	// Ran reports, of a call as it ended, whether this run got it to its
	// tool — rather than answering it from the record, refusing it, or not
	// running it at all — so that a journal can count the times it ran,
	// those it did not Track included.
	Ran bool
}

// Checkpoint is where a journaled run had got to, as its journal last wrote
// it down: what a run resumed from it starts with. See [Recover].
type Checkpoint struct {
	// Messages is the transcript as the last step left it.
	Messages []llm.Message

	// Calls are what the journal recorded of the calls of the step the
	// transcript ends on.
	Calls []Call

	// Iterations is how many iterations the run had made.
	Iterations int

	// State is everything else the run keeps while it goes — its loop's
	// counters, what is registered with [Carry] — opaque, to be given back
	// as it was. A checkpoint without one starts those afresh.
	State json.RawMessage

	// Used is what the run's budget had used, for a host that carries the
	// budget on itself (governor.WithBudgetFrom). Set by [Suspended].
	Used governor.Used
}

type journalKey struct{}

// WithJournal makes the run started on ctx write itself down in j. Only that
// run: a run started from inside it — a sub-agent's, a completion check's —
// does not inherit it. A delegate call gives its sub-agent a journal of its
// own when j is a [Nester].
func WithJournal(ctx context.Context, j Journal) context.Context {
	return context.WithValue(ctx, journalKey{}, j)
}

func journalFrom(ctx context.Context) Journal {
	j, _ := ctx.Value(journalKey{}).(Journal)
	return j
}

// runJournal is one run's side of its journal: the step it has open, which of
// that step's calls have begun, and the writes that failed. Loop-owned, except
// for begin and finish, which dispatch calls from its goroutines.
type runJournal struct {
	j Journal

	mu    sync.Mutex
	uses  []llm.ToolUse
	index map[llm.ToolUseID]int
	began map[int]bool
	// beginErr is a call the journal could not mark as running, which did
	// not run and stops the run; finishErr is one whose outcome could not be
	// written, which the step's Close writes again.
	beginErr, finishErr error
}

type runJournalKey struct{}

// withRunJournal puts the run's journal state on its context. Installed for
// every run, as nil when it has no journal, so a sub-agent's run never
// mistakes its parent's for its own.
func withRunJournal(ctx context.Context, rj *runJournal) context.Context {
	return context.WithValue(ctx, runJournalKey{}, rj)
}

func runJournalFrom(ctx context.Context) *runJournal {
	rj, _ := ctx.Value(runJournalKey{}).(*runJournal)
	return rj
}

func journalErr(what string, err error) error {
	return fmt.Errorf("%w: %s: %w", ErrJournal, what, err)
}

// key names a call of the open step for its tool; see [Keyer].
func (rj *runJournal) key(id llm.ToolUseID) string {
	k, ok := rj.j.(Keyer)
	if !ok {
		return ""
	}
	return k.Key(id)
}

// reopen makes uses the open step without writing anything: the step a
// resumed run starts on, which its journal already has.
func (rj *runJournal) reopen(uses []llm.ToolUse) {
	if rj == nil {
		return
	}
	rj.mu.Lock()
	defer rj.mu.Unlock()
	rj.uses = uses
	rj.index = make(map[llm.ToolUseID]int, len(uses))
	for i, u := range uses {
		if _, dup := rj.index[u.ID]; !dup {
			rj.index[u.ID] = i
		}
	}
	rj.began = map[int]bool{}
	rj.beginErr, rj.finishErr = nil, nil
}

// open writes a step down: a reply, and its calls as pending.
func (rj *runJournal) open(ctx context.Context, s Step) error {
	if rj == nil {
		return nil
	}
	var uses []llm.ToolUse
	for _, c := range s.Calls {
		uses = append(uses, c.Use)
	}
	rj.reopen(uses)
	if err := rj.j.Open(ctx, s); err != nil {
		return journalErr("writing the model's reply down before acting on it", err)
	}
	return nil
}

// begin marks a call of the open step as having begun, and reports its place
// in the step.
func (rj *runJournal) begin(id llm.ToolUseID) (int, bool) {
	rj.mu.Lock()
	defer rj.mu.Unlock()
	i, ok := rj.index[id]
	if ok {
		rj.began[i] = true
	}
	return i, ok
}

func (rj *runJournal) hasBegun(i int) bool {
	rj.mu.Lock()
	defer rj.mu.Unlock()
	return rj.began[i]
}

// unbegin takes back begin, for a call whose running state could not be
// written: it did not run.
func (rj *runJournal) unbegin(i int, err error) {
	rj.mu.Lock()
	defer rj.mu.Unlock()
	delete(rj.began, i)
	if rj.beginErr == nil {
		rj.beginErr = err
	}
}

func (rj *runJournal) failedFinish(err error) {
	rj.mu.Lock()
	defer rj.mu.Unlock()
	if rj.finishErr == nil {
		rj.finishErr = err
	}
}

// close writes the step's calls as they ended. It reports a call that could
// not be marked running — it did not run, and the run stops — and a failure
// of the write itself; one of a call's outcome that this write records again
// is forgiven.
func (rj *runJournal) close(ctx context.Context, calls []Call) error {
	if rj == nil {
		return nil
	}
	err := rj.j.Close(ctx, calls)
	rj.mu.Lock()
	beginErr := rj.beginErr
	rj.beginErr, rj.finishErr = nil, nil
	rj.mu.Unlock()
	switch {
	case beginErr != nil:
		return beginErr
	case err != nil:
		return journalErr("writing down how the model's calls ended", err)
	}
	return nil
}

// journalCalls writes a call down as running just before it takes effect,
// and with its outcome once it has, for a run with a journal and a tool that
// is not safe to run again. It sits where budgetToolCalls does — inside every
// gate, so only a call that is about to run is written as running; outside
// retry, the timeout and truncation, so what is written is the one logical
// call, and the result the model will read.
//
// A pause or terminal call passes straight through: its handler never runs,
// so there is nothing to protect.
func journalCalls(next tool.Handler) tool.Handler {
	return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
		rj := runJournalFrom(ctx)
		if rj == nil || d.Caps&(tool.CapPause|tool.CapTerminal) != 0 {
			return next(ctx, d, use)
		}
		i, ok := rj.begin(use.ID)
		if !ok || tool.RecoveryOf(d) == tool.RecoverRerun {
			// Not a call of the open step, or one that is simply run again
			// should the run be resumed: nothing to write before it runs.
			return next(ctx, d, use)
		}
		if err := rj.j.Track(ctx, Call{Index: i, Use: use, State: tool.StateRunning}); err != nil {
			err = journalErr("writing the call down before running it", err)
			rj.unbegin(i, err)
			return "", fmt.Errorf("%w; it did not run", err)
		}
		out, err := next(ctx, d, use)
		if errors.Is(err, tool.ErrSuspended) {
			// Not over: still running, as the step's Close writes it, and run
			// again when the run carries on.
			return out, err
		}
		c := outcome(ctx, i, use, d.Name, out, err, tool.Tagged(ctx), tool.Attached(ctx), true)
		c.Ran = true
		if werr := rj.j.Track(ctx, c); werr != nil {
			rj.failedFinish(werr)
		}
		return out, err
	}
}

// outcome is how a call that was dispatched ended, as a journal records it.
// began says whether it got past every gate to its tool: an error from one
// that did not is a refusal.
func outcome(ctx context.Context, i int, use llm.ToolUse, name, out string, err error, tags []string, media []llm.Media, began bool) Call {
	c := Call{Index: i, Use: use, Tags: tags}
	res := tool.Result{UseID: use.ID, Name: name, Output: out, Err: err, Tags: tags}
	if err == nil {
		res.Media = media
	}
	blk, _ := res.Block().(llm.ToolResult)
	if errors.Is(err, tool.ErrSuspended) {
		c.State, c.Reason = tool.StateRunning, suspendedReason
		return c
	}
	if handle, note, ok := tool.Awaiting(err); ok {
		c.State, c.Handle, c.Reason = tool.StateWaiting, handle, note
		return c
	}
	if reason, ok := tool.DeferralReason(err); ok {
		c.State, c.Reason = tool.StateDeferred, reason
		return c
	}
	switch {
	case err == nil:
		c.State = tool.StateSucceeded
	case !began:
		c.State = tool.StateDenied
	case ctx.Err() != nil:
		// The run is being stopped, and the call with it: it may have done
		// part of what it does.
		c.State = tool.StateInterrupted
	default:
		c.State = tool.StateFailed
	}
	c.Result = &blk
	return c
}

// settled is what became of one call of a step, for the step's Close; ran
// is whether this run got it to its tool.
type settled struct {
	state  tool.State
	res    *tool.Result
	reason string
	handle string
	ran    bool
}

func settledResult(state tool.State, res tool.Result) settled {
	return settled{state: state, res: &res}
}

// calls renders a step's calls as they ended.
func stepCalls(uses []llm.ToolUse, st []settled) []Call {
	out := make([]Call, len(uses))
	for i, u := range uses {
		s := st[i]
		c := Call{Index: i, Use: u, State: s.state, Reason: s.reason, Handle: s.handle, Ran: s.ran}
		if s.res != nil {
			blk, _ := s.res.Block().(llm.ToolResult)
			c.Result, c.Tags = &blk, s.res.Tags
		}
		out[i] = c
	}
	return out
}

// dispatched is what became of a dispatched call, judged by its result:
// whether it got as far as its tool decides between a refusal and a failure.
func (r *Run) dispatched(ctx context.Context, i int, res tool.Result) settled {
	began := r.journal != nil && r.journal.hasBegun(i)
	c := outcome(ctx, i, llm.ToolUse{ID: res.UseID}, res.Name, res.Output, res.Err, res.Tags, res.Media, began)
	s := settled{state: c.State, reason: c.Reason, handle: c.Handle, ran: began}
	if c.Result != nil {
		res := res
		s.res = &res
	}
	return s
}

// closeStep writes down how the step's calls ended: each one's result, or,
// for a run that stopped at one of them, what it stopped for and what never
// ran. It returns the calls as it wrote them.
func (r *Run) closeStep(ctx context.Context, uses []llm.ToolUse, results []tool.Result, stop Outcome) ([]Call, error) {
	st := make([]settled, len(uses))
	for i, u := range uses {
		if i < len(results) && results[i].UseID != "" {
			// A pause or terminal call the run stopped at is here too, with
			// the note its accepting handler returned.
			st[i] = r.dispatched(ctx, i, results[i])
			continue
		}
		st[i] = settledResult(tool.StateCancelled, notRun(u, notRunText(stop)))
	}
	calls := stepCalls(uses, st)
	return calls, r.journal.close(ctx, calls)
}

// openStep writes a step down, with the run's state as it stands.
func (r *Run) openStep(ctx context.Context, s Step) error {
	if r.journal == nil {
		return nil
	}
	s.State = r.state()
	return r.journal.open(ctx, s)
}

// notRunText is what a call the run never dispatched is answered with — the
// same words [AnswerPause] closes it with, when the run stopped to ask.
func notRunText(stop Outcome) string {
	if _, ok := stop.(Paused); ok {
		return "not run: the agent stopped to ask the user before this call was dispatched"
	}
	return "not run: the run ended before this call was dispatched"
}

func notRun(u llm.ToolUse, why string) tool.Result {
	return tool.Result{UseID: u.ID, Name: u.Name, Err: errors.New(why)}
}
