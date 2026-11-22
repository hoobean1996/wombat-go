package core

import (
	"context"
	"errors"
	"fmt"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// Recover resumes a run that stopped part-way — its process died, it was shut
// down, or it failed in a way worth another try — from where its [Journal]
// last wrote it down.
//
// A transcript that ends on the model's calls has them answered before the
// model is called again, each according to what the journal recorded of it:
//
//   - one that finished is answered with what it returned, and is not run
//     again;
//   - one that never started runs now;
//   - one that was running when the run stopped, and so may or may not have
//     taken effect, is asked about with its tool's [tool.Def.Reconcile] if it
//     has one, and otherwise handled as its [tool.Recovery] says: run again,
//     reported to the model as interrupted, or deferred for a person to
//     decide whether it runs again;
//   - one waiting for a result stays waiting, and the run ends in [Deferred]
//     again.
//
// A transcript that ends on the model's answer is not asked for again: the
// run checks it, as it would have, and ends. Then the run carries on as if it
// had never stopped, counting the iterations it had made.
func Recover(cp Checkpoint) Input {
	return Input{Messages: cp.Messages, resume: true, recorded: cp.Calls, iterations: cp.Iterations, state: cp.State}
}

// Recorded tells a run resumed with [Decide] what a journal recorded of the
// calls its transcript ends on, so that one that finished after the decision
// — and before a crash cut the resumed run short — is answered from the
// record instead of being run a second time. See [Recover] for how each call
// is answered.
func (in Input) Recorded(calls []Call) Input {
	in.recorded = calls
	return in
}

// Completed gives a run that ended in [Deferred] on calls waiting for a
// result ([tool.Await]) the results that have come in, by tool_use id: each
// becomes its call's tool_result as if the tool had returned it. A call
// still without one keeps waiting, and the run ends in Deferred again. It
// combines with decisions:
//
//	core.Decide(msgs, d, verdicts).Completed(results)
func (in Input) Completed(results map[llm.ToolUseID]llm.ToolResult) Input {
	in.completed = results
	return in
}

// What a resumed run tells the model about a call that was running when the
// run it belongs to stopped.
const (
	interruptedText = "interrupted: this call was running when the agent stopped — the process restarted, or the " +
		"turn was cut short — so it may or may not have taken effect. Check whether it did before you run it again."
	rerunQuestion = "it was running when the agent stopped, and may or may not have taken effect: run it again?"
	notRerunText  = "not run again: this call was running when the agent stopped, so it may or may not have taken " +
		"effect, and the person supervising this run decided it should not run a second time. Check what it did " +
		"before you go on."
	undecidedText = "not run: the run stopped for a decision before this call was dispatched"
)

// settle answers the calls of the step the transcript ends on, before the
// model is called, for a run started with [Decide] or [Recover]: from what
// was decided, posted or recorded about each, by running the ones that are
// to run — through the same middleware as any call, so a permission gate sees
// the decisions — and in one message, in the order the model made them. A
// call that still waits — for a decision, or for a result — ends the run in a
// new [Deferred] that keeps what has been answered so far.
//
// It reports whether the run is over.
func (r *Run) settle(ctx context.Context, convo Convo) (Convo, bool) {
	d := r.decisions
	if len(convo) == 0 || convo[len(convo)-1].Role != llm.RoleAssistant {
		if d != nil {
			r.finish(nil, errors.New("core: Decide was given a transcript that does not end on the deferred turn"))
			return convo, true
		}
		return convo, false // the model is next
	}
	last := convo[len(convo)-1]
	uses := llm.ToolUses(last.Content)
	if len(uses) == 0 {
		if d != nil {
			r.finish(nil, errors.New("core: Decide was given a transcript that does not end on the deferred turn"))
			return convo, true
		}
		// The model's answer, written down before the run stopped: checked
		// as it would have been, and not asked for again.
		next, o := r.answered(ctx, convo, last.Content)
		if o != nil {
			r.finish(o, nil)
			return next, true
		}
		return next, false
	}
	r.journal.reopen(uses)

	var waiting map[llm.ToolUseID]DeferredCall
	var done map[llm.ToolUseID]llm.ToolResult
	var verdicts map[llm.ToolUseID]bool
	if d != nil {
		waiting = make(map[llm.ToolUseID]DeferredCall, len(d.deferred.Calls))
		for _, c := range d.deferred.Calls {
			waiting[c.UseID] = c
		}
		done = make(map[llm.ToolUseID]llm.ToolResult, len(d.deferred.Done))
		for _, res := range d.deferred.Done {
			done[res.ToolUseID] = res
		}
		verdicts = d.verdicts
	}
	recorded := make(map[int]Call, len(r.recorded))
	for _, c := range r.recorded {
		if c.Index >= 0 && c.Index < len(uses) && c.Use.ID == uses[c.Index].ID {
			recorded[c.Index] = c
		}
	}

	st := make([]settled, len(uses))
	answered := make([]bool, len(uses))
	answer := func(i int, s settled) { st[i], answered[i] = s, true }
	var again []DeferredCall // calls that wait once more
	var run []int            // calls to dispatch now
	for i, u := range uses {
		c, has := recorded[i]
		w, isWaiting := waiting[u.ID]
		switch {
		case done != nil && hasResult(done, u.ID):
			// It ran before the run stopped for decisions: answered as it was.
			blk := done[u.ID]
			state := tool.StateSucceeded
			if blk.IsError {
				state = tool.StateFailed
			}
			if has && c.State.Final() {
				state = c.State
			}
			answer(i, settledResult(state, resultOf(u, blk, c.Tags)))
		case has && c.State.Final() && c.Result != nil:
			// It finished, and the run stopped before the transcript said so:
			// what it returned is the answer, and it does not run again.
			answer(i, settledResult(c.State, resultOf(u, *c.Result, c.Tags)))
			r.emit(ToolRecovered{UseID: u.ID, Name: u.Name, State: c.State, Action: "replayed"})
		case has && c.State == tool.StateRunning && c.Reason == suspendedReason:
			// It stopped with the slice, where its journal wrote it down:
			// run again, it carries on from there.
			run = append(run, i)
		case (has && c.State == tool.StateWaiting) || (isWaiting && w.Awaiting):
			if blk, ok := r.completed[u.ID]; ok {
				blk.ToolUseID = u.ID
				state := tool.StateSucceeded
				if blk.IsError {
					state = tool.StateFailed
				}
				answer(i, settledResult(state, resultOf(u, blk, nil)))
				r.emit(ToolRecovered{UseID: u.ID, Name: u.Name, State: tool.StateWaiting, Action: "completed"})
				continue
			}
			call := DeferredCall{UseID: u.ID, Tool: u.Name, Input: u.Input, Reason: c.Reason, Awaiting: true, Handle: c.Handle}
			if isWaiting {
				call.Reason, call.Handle = w.Reason, w.Handle
			}
			again = append(again, call)
			st[i] = settled{state: tool.StateWaiting, reason: call.Reason, handle: call.Handle}
		case has && (c.State == tool.StateRunning || c.State == tool.StateInterrupted):
			switch s, act := r.recoverCall(ctx, u, verdicts); act {
			case "rerun":
				run = append(run, i)
			case "asked":
				again = append(again, DeferredCall{UseID: u.ID, Tool: u.Name, Input: u.Input, Reason: rerunQuestion})
				st[i] = settled{state: tool.StateInterrupted, reason: rerunQuestion}
			default:
				answer(i, s)
			}
		case d != nil && !isWaiting && !has:
			answer(i, settledResult(tool.StateCancelled, notRun(u, undecidedText)))
		default:
			// Pending, deferred to a decision that is now in, or not known
			// at all: it has not run, so it runs.
			run = append(run, i)
		}
	}

	if len(run) > 0 {
		pending := make([]llm.ToolUse, len(run))
		for j, i := range run {
			pending[j] = uses[i]
		}
		dctx := ctx
		if verdicts != nil {
			dctx = tool.WithDecisions(ctx, verdicts)
		}
		results, stop := r.dispatch(dctx, convo, pending)
		r.units++
		for j, i := range run {
			if j < len(results) && results[j].UseID != "" {
				st[i] = r.dispatched(ctx, i, results[j])
			} else {
				st[i] = settledResult(tool.StateCancelled, notRun(uses[i], notRunText(stop)))
			}
		}
		if suspendedIn(results) {
			// A sub-agent stopped with the slice again: the step stays open,
			// with what has been answered so far, until the next.
			calls := stepCalls(uses, st)
			if err := r.journal.close(ctx, calls); err != nil && ctx.Err() == nil {
				r.finish(nil, err)
				return convo, true
			}
			return r.suspend(ctx, convo, calls, r.iterations)
		}
		switch o := stop.(type) {
		case Paused, Submitted:
			// A pause or a submission the decisions let through ends the run
			// here, as it would have the first time.
			return r.stopAt(ctx, convo, uses, st, o)
		case Deferred:
			again = append(again, o.Calls...)
		}
	}

	if len(again) > 0 {
		// Something still waits: the run stops again, keeping, in the model's
		// order, everything answered so far.
		out := Deferred{}
		waitingNow := make(map[llm.ToolUseID]bool, len(again))
		for _, c := range again {
			waitingNow[c.UseID] = true
		}
		for _, u := range uses {
			for _, c := range again {
				if c.UseID == u.ID {
					out.Calls = append(out.Calls, c)
					break
				}
			}
		}
		for i, u := range uses {
			if waitingNow[u.ID] || st[i].res == nil {
				continue
			}
			if blk, ok := st[i].res.Block().(llm.ToolResult); ok {
				out.Done = append(out.Done, blk)
			}
		}
		return r.stopAt(ctx, convo, uses, st, out)
	}

	blocks := make([]llm.ContentBlock, len(uses))
	for i, u := range uses {
		res := tool.Result{UseID: u.ID, Name: u.Name, Err: errors.New(undecidedText)}
		if st[i].res != nil {
			res = *st[i].res
		}
		blocks[i] = res.Block()
		r.results[u.ID] = ResultInfo{Tool: u.Name, Tags: res.Tags, Bytes: res.Size()}
	}
	convo = convo.PushToolResults(blocks)
	r.setConvo(convo)
	if err := r.journal.close(ctx, stepCalls(uses, st)); err != nil && ctx.Err() == nil {
		r.finish(nil, err)
		return convo, true
	}
	return convo, false
}

// stopAt writes the step down as st has it and ends the run in o.
func (r *Run) stopAt(ctx context.Context, convo Convo, uses []llm.ToolUse, st []settled, o Outcome) (Convo, bool) {
	if err := r.journal.close(ctx, stepCalls(uses, st)); err != nil && ctx.Err() == nil {
		r.finish(nil, err)
		return convo, true
	}
	r.finish(o, nil)
	return convo, true
}

// recoverCall decides what becomes of a call that was running when the run it
// belongs to stopped: "rerun", "asked" — deferred for a person — or anything
// else, with the call answered by s.
func (r *Run) recoverCall(ctx context.Context, u llm.ToolUse, verdicts map[llm.ToolUseID]bool) (s settled, action string) {
	d, known := r.a.set.Find(u.Name)
	defer func() {
		r.emit(ToolRecovered{UseID: u.ID, Name: u.Name, State: tool.StateInterrupted, Action: action})
	}()
	if known && d.Reconcile != nil {
		cctx := tool.WithInfo(ctx, tool.Info{UseID: u.ID})
		out, landed, err := d.Reconcile(cctx, u.Input)
		switch {
		case err == nil && landed:
			return settledResult(tool.StateSucceeded, tool.Result{UseID: u.ID, Name: u.Name, Output: out}), "reconciled"
		case err == nil:
			return settled{}, "rerun"
		}
		// It cannot tell: its Recovery decides.
	}
	policy := tool.RecoverReport
	if known {
		policy = tool.RecoveryOf(d)
	}
	switch policy {
	case tool.RecoverRerun:
		return settled{}, "rerun"
	case tool.RecoverAsk:
		allow, decided := verdicts[u.ID]
		switch {
		case !decided:
			return settled{}, "asked"
		case allow:
			return settled{}, "rerun"
		default:
			return settledResult(tool.StateFailed, tool.Result{UseID: u.ID, Name: u.Name, Err: errors.New(notRerunText)}), "refused"
		}
	default:
		return settledResult(tool.StateFailed, tool.Result{UseID: u.ID, Name: u.Name, Err: errors.New(interruptedText)}), "reported"
	}
}

// answered ends a turn on the model's answer, unless the completion check
// sends the model back, in which case it returns the transcript to carry on
// from and no outcome.
func (r *Run) answered(ctx context.Context, convo Convo, content []llm.ContentBlock) (Convo, Outcome) {
	answer := llm.TextOf(content)
	if feedback := r.checkCompletion(ctx, Completion{Answer: answer, Messages: convo}); feedback != "" {
		convo = convo.PushUserText(notDoneYet(feedback))
		r.setConvo(convo)
		return convo, nil
	}
	return convo, Answer{Text: answer}
}

// resultOf is a recorded result as a dispatcher would have returned it.
func resultOf(u llm.ToolUse, blk llm.ToolResult, tags []string) tool.Result {
	res := tool.Result{UseID: u.ID, Name: u.Name, Tags: tags}
	if blk.IsError {
		res.Err = recordedError(blk.Content)
	} else {
		res.Output, res.Media = blk.Content, blk.Media
	}
	return res
}

// recordedError is a tool's error as a journal kept it: its message.
type recordedError string

func (e recordedError) Error() string { return string(e) }

func hasResult(m map[llm.ToolUseID]llm.ToolResult, id llm.ToolUseID) bool {
	_, ok := m[id]
	return ok
}

// ToolRecovered reports a call of the step a resumed run starts on that the
// run did not simply dispatch: one answered from what its journal recorded or
// from a result posted while it waited, or one that was running when the run
// stopped, and what was done with it.
type ToolRecovered struct {
	UseID llm.ToolUseID `json:"use_id"`
	Name  string        `json:"name"`

	// State is what the journal had recorded.
	State tool.State `json:"state"`

	// Action is what this run did with the call: replayed, with the result
	// recorded; completed, with the result posted while it waited;
	// reconciled, its tool having found that it had taken effect; rerun;
	// reported to the model as interrupted; asked, deferred for a person to
	// decide whether it runs again; or refused, a person having decided it
	// does not.
	Action string `json:"action"`
}

// Kind implements Event.
func (ToolRecovered) Kind() string { return "tool_recovered" }

// MarshalJSON implements json.Marshaler.
func (e ToolRecovered) MarshalJSON() ([]byte, error) {
	type r ToolRecovered
	return eventJSON(e.Kind(), r(e))
}

// String renders the event for a log line.
func (e ToolRecovered) String() string {
	return fmt.Sprintf("%s %s (%s): %s", e.Name, e.UseID, e.State, e.Action)
}
