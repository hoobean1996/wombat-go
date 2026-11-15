package wombat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
	"github.com/hoobean1996/wombat-go/trace"
)

// Input is the transcript an agent starts from.
type Input struct {
	Messages []llm.Message
}

// Ask starts a fresh conversation with one user turn.
func Ask(q string) Input {
	return Input{Messages: []llm.Message{llm.UserText(q)}}
}

// Continue resumes an existing transcript.
//
// The transcript must be complete: if the last assistant turn has an
// unanswered tool_use, the provider will reject the request. Close it with
// [Convo.CloseDangling] or answer it with [AnswerPause].
func Continue(prior []llm.Message) Input {
	return Input{Messages: prior}
}

// AnswerPause resumes a run that ended in [Paused], answering the tool call
// the model used to ask — and closing every other call the model made in the
// same turn.
//
// The siblings are not a corner case. A model that asks a question routinely
// asks it alongside other work: one assistant turn carrying
// [tool_use grep_search, tool_use ask_user] is an ordinary thing for a model to
// emit. The loop returns Paused as soon as the pause call is allowed through,
// WITHOUT dispatching any of its siblings, so those siblings never ran and
// never will.
//
// Answering only the pause leaves them dangling, and the consequence is not a
// warning: [Convo.Validate] rejects the resumed transcript, the turn fails
// before reaching the provider, and — for a caller persisting the conversation,
// which is the whole reason a pause has to survive a process boundary — the
// broken transcript is what gets saved. Every subsequent turn then fails the
// same way. One dropped sibling bricks the session.
//
// They are closed as errors rather than as empty output because the model has
// to be able to tell "I ran it and it returned nothing" from "it never ran";
// only the second is worth re-issuing.
func AnswerPause(prior []llm.Message, id llm.ToolUseID, answer string) Input {
	c := Convo(prior)

	pending := c.Dangling()
	blocks := make([]llm.ContentBlock, 0, len(pending)+1)
	answered := false
	for _, p := range pending {
		if p == id {
			blocks = append(blocks, llm.ToolResult{ToolUseID: p, Content: answer})
			answered = true
			continue
		}
		blocks = append(blocks, llm.ToolResult{
			ToolUseID: p,
			Content:   "not run: the agent stopped to ask the user before this call was dispatched",
			IsError:   true,
		})
	}
	// An id that is not actually pending is a caller bug. Adding the block
	// anyway keeps the mistake visible — Validate reports it as an orphan —
	// rather than silently discarding the answer.
	if !answered {
		blocks = append(blocks, llm.ToolResult{ToolUseID: id, Content: answer})
	}

	return Input{Messages: c.PushToolResults(blocks)}
}

// Then appends a new user instruction to a finished transcript, closing any
// dangling tool_use with an acknowledgement first.
//
// "(cancelled)" is the right answer for a tool call the run abandoned, and the
// WRONG one for a pause the user is about to answer — the model asked a
// question and would be told the question was withdrawn, then handed the answer
// as a fresh instruction with no memory of what it was for. Check
// [PendingPause] before reaching for this; [Resume] does that for you.
func Then(prior []llm.Message, instruction string) Input {
	c := Convo(prior)
	if len(c.Dangling()) > 0 {
		return Input{Messages: c.CloseDangling("(cancelled)", llm.Text{Text: instruction})}
	}
	return Input{Messages: c.PushUserText(instruction)}
}

// PendingPause reports the tool_use a transcript is suspended on, when it ends
// on a call to a [tool.CapPause] tool.
//
// A dangling tool_use has two very different meanings and the id alone does not
// distinguish them: the run may have been killed mid-tool-call, or it may have
// stopped deliberately to ask the user something. Only the tool's capability
// says which, which is why this takes a Set rather than matching on a name —
// "ask_user" is a convention, CapPause is the contract.
func PendingPause(prior []llm.Message, set tool.Set) (llm.ToolUseID, bool) {
	c := Convo(prior)
	if len(c) == 0 || len(c.Dangling()) == 0 {
		return "", false
	}
	last := c[len(c)-1]
	if last.Role != llm.RoleAssistant {
		return "", false
	}
	tu, ok := tool.FindPause(set, llm.ToolUses(last.Content))
	return tu.ID, ok
}

// Resume continues a saved transcript, doing the right thing with whatever the
// previous run left behind.
//
//   - No instruction: carry straight on from where it stopped.
//   - An instruction, and the transcript is suspended on a pause tool: the
//     instruction IS the answer to that question.
//   - An instruction otherwise: a new user turn, closing any abandoned tool
//     call first.
//
// The middle case is the one worth having a function for. Getting it wrong is
// silent — [Then] produces a perfectly valid transcript in which the model was
// told its question was cancelled and then handed the answer with no idea what
// it answers — and it is the entire point of a pause tool surviving a process
// boundary.
func Resume(prior []llm.Message, instruction string, set tool.Set) Input {
	if instruction == "" {
		return Continue(prior)
	}
	if id, ok := PendingPause(prior, set); ok {
		return AnswerPause(prior, id, instruction)
	}
	return Then(prior, instruction)
}

// Run is a single execution in progress. Iterate it like a [bufio.Scanner]:
//
//	run := a.Start(ctx, wombat.Ask(q))
//	defer run.Close()
//	for run.Next() { handle(run.Event()) }
//	if err := run.Err(); err != nil { return err }
//	answer := run.Outcome()
//
// [Run.Err] and [Run.Outcome] are meaningful once [Run.Next] has returned
// false. A caller that abandons the iteration early must call [Run.Close], or
// the producing goroutine stays parked on its next send.
type Run struct {
	a      *Agent
	ctx    context.Context
	cancel context.CancelFunc

	events chan Event
	cur    Event

	mu      sync.Mutex
	convo   Convo
	outcome Outcome
	err     error

	// results is what the harness learned while dispatching this run's tool
	// calls, handed to the strategy as View.Results. Loop-owned: written and
	// read only from loop, never from a consumer goroutine.
	results map[llm.ToolUseID]ResultInfo

	// emptyTurns counts CONSECUTIVE turns that said nothing and called
	// nothing; see the retry in iterate. Reset by any turn with content, so
	// two empty turns an hour apart never add up to a failure. Loop-owned.
	emptyTurns int

	// truncated counts CONSECUTIVE replies cut off at max_tokens, and
	// truncNotice is what the next request tells the model about it.
	//
	// Both loop-owned. The notice is held here rather than pushed into the
	// transcript for the reason withNotice gives: a notice that leaked into
	// the stored conversation would accumulate one copy per iteration.
	truncated   int
	truncNotice string

	// tokens is this run's calibrated token estimator, handed to strategies
	// as View.Tokens. compactRetryAt holds off a compaction that failed until
	// the transcript has grown enough to be worth another attempt, and
	// trimmed is what the last ContextTrimmed event reported, so the event
	// marks a change instead of repeating every iteration.
	tokens         *tokenEstimator
	compactRetryAt int
	trimmed        [2]int

	// steering holds what Steer queued and the loop has not delivered yet;
	// ended is set once the run can take no more, and what was still queued
	// then moves to undelivered. Guarded by steerMu, because Steer is called
	// from the consumer's side.
	// rejections counts the times the completion check sent the model back.
	rejections int

	steerMu     sync.Mutex
	steering    []string
	ended       bool
	undelivered []string

	span *trace.Active
}

// Start begins a run and returns immediately; the loop proceeds in the
// background, paced by the consumer's calls to [Run.Next].
func (a *Agent) Start(ctx context.Context, in Input) *Run {
	ctx, cancel := context.WithCancel(ctx)

	r := &Run{
		a:       a,
		cancel:  cancel,
		events:  make(chan Event, a.cfg.eventBuffer),
		convo:   Convo(in.Messages),
		results: make(map[llm.ToolUseID]ResultInfo),
		tokens:  newTokenEstimator(),
	}

	// Every run has a budget, even one given no limits, so that what it spends
	// is tallied as ITS spend: Progress, the Spend events and the sub-agent
	// depth all read it. A context that already carries one — a caller's
	// WithBudget, or a parent run's when this is a sub-agent — keeps it, which
	// is what makes a delegated task charge its parent's tally.
	ctx = governor.Ensure(ctx)

	// The event sink and the transcript resolver are per-run, so they travel
	// on the context; the middleware chains that consume them were built once,
	// when the agent was constructed.
	ctx = tool.WithLookup(WithEmitter(ctx, r.emit), r.lookup)

	// Dedup counters are per run: being stuck in a loop is a property of one
	// conversation, and a second run of the same agent must not inherit the
	// first one's frustration. Installed before the caller's decorators so a
	// caller can still substitute its own.
	ctx = tool.WithCallStats(ctx, tool.NewCallStats())

	// The overflow ladder is per run for the same reason, and the result
	// metadata is published so an llm.Middleware — which sees a Request, not a
	// Run — can still make a semantic decision about what to drop.
	ctx = withOverflowState(ctx, &overflowState{})
	ctx = withResults(ctx, r.results)
	ctx = withEstimator(ctx, r.tokens)
	ctx = withWindow(ctx, a.cfg.window)

	// A retry inside the LLM chain is otherwise a call that is merely slow.
	ctx = llm.WithRetryObserver(ctx, func(rt llm.Retry) {
		ev := LLMRetry{Attempt: rt.Attempt, Of: rt.Of, WaitMillis: millis(rt.Wait)}
		if rt.Err != nil {
			ev.Error = rt.Err.Error()
		}
		r.emit(ev)
	})

	// Per-run state is created here, not on the Agent, so two concurrent runs
	// of the same agent never share it.
	for _, dec := range a.cfg.runCtx {
		ctx = dec(ctx)
	}

	// After the decorators, so a tracer installed by WithRunContext is the one
	// this span is opened on.
	ctx, r.span = trace.FromContext(ctx).Start(ctx, trace.KindRun, a.cfg.name)
	r.ctx = ctx

	// The estimator's fixed part before the first measurement: the system
	// prompt and whatever tools the run starts with.
	r.tokens.fixedBytes = len(a.system) + specBytes(tool.Specs(a.set.Visible(ctx)))

	go r.loop()
	return r
}

// Run executes to completion, discarding events.
func (a *Agent) Run(ctx context.Context, in Input) (Outcome, error) {
	r := a.Start(ctx, in)
	defer r.Close()
	for r.Next() { //nolint:revive // draining is the point
	}
	return r.Outcome(), r.Err()
}

// Next advances to the next event, reporting false when the run is over.
func (r *Run) Next() bool {
	ev, ok := <-r.events
	if !ok {
		return false
	}
	r.cur = ev
	return true
}

// Event returns the event [Run.Next] most recently produced.
func (r *Run) Event() Event { return r.cur }

// Err reports why the run failed, or nil. Valid once Next returns false.
func (r *Run) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.err
}

// Outcome reports how the run ended, or nil if it failed. Valid once Next
// returns false.
func (r *Run) Outcome() Outcome {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.outcome
}

// Messages returns the transcript so far. Safe to call at any time; the result
// is a snapshot.
func (r *Run) Messages() []llm.Message {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]llm.Message(nil), r.convo...)
}

// Close cancels the run and releases its goroutine. Idempotent, and safe to
// call after a full drain.
//
// It releases the goroutine by consuming whatever events are still in flight
// and discarding them, so it belongs to the consumer: call it from the
// goroutine that calls [Run.Next], once that goroutine is done reading. Called
// while another goroutine is still iterating, it competes with that reader for
// the remaining events and drops the ones it wins.
func (r *Run) Close() error {
	r.cancel()
	for range r.events { //nolint:revive // drain so the producer can exit
	}
	return nil
}

func (r *Run) emit(ev Event) {
	select {
	case r.events <- ev:
	case <-r.ctx.Done():
	}
}

func (r *Run) lookup(id llm.ToolUseID) (string, error) {
	r.mu.Lock()
	c := r.convo
	r.mu.Unlock()
	return c.Lookup(id)
}

func (r *Run) setConvo(c Convo) {
	r.mu.Lock()
	r.convo = c
	r.mu.Unlock()
}

func (r *Run) finish(o Outcome, err error) {
	r.mu.Lock()
	r.outcome, r.err = o, err
	r.mu.Unlock()

	r.steerMu.Lock()
	r.ended = true
	r.undelivered = append(r.undelivered, r.steering...)
	r.steering = nil
	r.steerMu.Unlock()
}

// Steer adds a message from the user to a run that is still going — "use the
// v2 API instead", typed while the agent is three tool calls into the wrong
// one. It reports whether the run took it.
//
// The message is delivered at the next point the conversation can take one:
// with the results of the tool calls in flight, before the next model call. A
// message that arrives while the model is writing what would have been its
// final answer does not get lost behind it either: the answer stays in the
// transcript, the message follows as a new user turn, and the run carries on
// to answer that too.
//
// A run that ends before it can deliver a message — paused, submitted, failed,
// cancelled — keeps it in [Run.Undelivered], for the caller to send as the next
// turn instead. A run that has already ended takes nothing and returns false.
func (r *Run) Steer(text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	r.steerMu.Lock()
	defer r.steerMu.Unlock()
	if r.ended {
		return false
	}
	r.steering = append(r.steering, text)
	return true
}

// Undelivered returns what [Run.Steer] accepted and the run ended without
// delivering. Meaningful once [Run.Next] has returned false.
func (r *Run) Undelivered() []string {
	r.steerMu.Lock()
	defer r.steerMu.Unlock()
	return append([]string(nil), r.undelivered...)
}

// takeSteering removes and returns everything queued.
func (r *Run) takeSteering() []string {
	r.steerMu.Lock()
	defer r.steerMu.Unlock()
	out := r.steering
	r.steering = nil
	return out
}

// steeringBlocks renders queued messages for the transcript. Each is marked
// as coming from the user mid-task: delivered next to tool results — and, on
// the OpenAI wire, folded into the last of them — an unmarked sentence would
// read as something the tool printed.
func steeringBlocks(msgs []string) []llm.ContentBlock {
	out := make([]llm.ContentBlock, len(msgs))
	for i, m := range msgs {
		out[i] = llm.Text{Text: "[a message from the user, sent while you were working]\n" + m}
	}
	return out
}

// deliverSteering puts queued messages into convo: onto the last turn when it
// is the user's — tool results or the prompt — or as a new user turn after an
// assistant one. Convo.Append does exactly that.
func (r *Run) deliverSteering(convo Convo, msgs []string) Convo {
	r.emit(Steered{Messages: len(msgs)})
	return convo.Append(llm.Message{Role: llm.RoleUser, Content: steeringBlocks(msgs)})
}

// causeOr prefers the reason a governed context was cancelled over whatever
// error the interrupted call happened to report. A budget-exhausted run should
// say so, not surface "context canceled" from the HTTP layer.
func (r *Run) causeOr(err error) error {
	if c := context.Cause(r.ctx); c != nil && !errors.Is(c, context.Canceled) {
		return c
	}
	return err
}

// loop is the ReAct cycle.
//
// Everything the OCaml original expressed as an effect performed from inside
// the loop — pausing for the user, ending on a terminal tool — is a return
// value here, so the loop is one function with one exit path per outcome and
// no handler installed above it.
func (r *Run) loop() {
	// Three deferred steps that must happen in this order, so they are
	// registered in reverse (defers run LIFO):
	//
	//	1. recover      — turn a panic into r.err
	//	2. span.End     — record the run span, now that r.err is final
	//	3. close(events)— tell the consumer the run is over
	//
	// Step 2 before step 3 is not cosmetic. Run.Next reports false as soon as
	// the channel closes, and a delegate tool takes that as its cue to return;
	// if the span were still unemitted at that moment, a caller closing its
	// trace sink right after the run could lose a whole sub-agent's run span
	// and orphan every span beneath it. Observed exactly that, under -race,
	// with the previous ordering.
	defer close(r.events)

	defer func() { r.span.End(r.Err()) }()

	// Last line of defence: a panic anywhere in the loop — in a strategy, a
	// client, a dispatcher's own bookkeeping — becomes an error the caller can
	// read, instead of taking down a process that may be serving other runs.
	defer func() {
		if p := recover(); p != nil {
			r.finish(nil, fmt.Errorf("%w: %v\n%s", ErrPanic, p, truncateStack(debug.Stack())))
		}
	}()

	a := r.a
	ctx := r.ctx
	budget := governor.FromContext(ctx)

	if m := a.cfg.metrics; m != nil {
		started := time.Now()
		m.RunStarted()
		// Registered here rather than at each exit: the loop has ten ways to
		// end and a counter that misses one is worse than no counter, because
		// the gap is invisible on a dashboard.
		defer func() { m.RunFinished(outcomeLabel(r.Outcome(), r.Err()), time.Since(started)) }()
	}

	convo := r.snapshot()
	if err := convo.Validate(); err != nil {
		r.finish(nil, fmt.Errorf("wombat: invalid input transcript: %w", err))
		return
	}

	for iter := 1; ; iter++ {
		if ctx.Err() != nil {
			r.finish(nil, r.causeOr(ctx.Err()))
			return
		}
		if iter > a.cfg.maxIters {
			r.finish(nil, fmt.Errorf("%w (%d)", ErrMaxIterations, a.cfg.maxIters))
			return
		}

		budget.Step()
		if ctx.Err() != nil {
			r.finish(nil, r.causeOr(ctx.Err()))
			return
		}

		r.emit(IterStart{N: iter, Max: a.cfg.maxIters})
		// The fourth of the four places WithMetrics promises to wire. It went
		// uncounted for a while, and nothing said so: an iterations series that
		// is never written is simply absent, which reads as "no data yet".
		if m := a.cfg.metrics; m != nil {
			m.Iteration(a.cfg.name)
		}

		next, stop := r.iterate(ctx, iter, convo)
		if stop {
			return
		}
		convo = next
	}
}

// iterate runs one ReAct cycle: materialize, call, act on the reply.
//
// It is a separate function purely so the iteration's trace span has ONE exit
// to close on. The body has eight ways to end a run, and ending the span at
// each of them is the kind of bookkeeping that survives exactly until someone
// adds a ninth — after which the span never closes and everything nested under
// it is silently orphaned from the report.
//
// Returns the transcript to carry forward and whether the run is over. When it
// is over the outcome or error has already been recorded with finish.
func (r *Run) iterate(parent context.Context, iter int, convo Convo) (Convo, bool) {
	a := r.a

	ctx, span := trace.FromContext(parent).Start(parent, trace.KindIteration,
		fmt.Sprintf("%s iteration %d", a.cfg.name, iter))
	var iterErr error
	defer func() { span.End(iterErr) }()
	span.Set("wombat.iteration", iter)

	fail := func(err error) (Convo, bool) {
		iterErr = err
		r.finish(nil, err)
		return convo, true
	}
	done := func(o Outcome) (Convo, bool) {
		r.finish(o, nil)
		return convo, true
	}

	// Materialize, reconcile, then read the surface — in that order.
	//
	// The strategy may have just evicted the tool_result that carried a
	// skill's body; a Set that gates tools on that skill has to learn so
	// before it reports what is visible, or this turn offers the model
	// tools whose knowledge left the context one line ago.
	// What the user said while the last step ran goes in before anything
	// looks at the transcript, so the model reads it on this very call.
	if steer := r.takeSteering(); len(steer) > 0 {
		convo = r.deliverSteering(convo, steer)
		r.setConvo(convo)
	}

	// Compaction first, because it rewrites the STORED transcript: what it
	// folds away is gone for every later call, not just this one.
	if folded, ok := r.compact(ctx, convo); ok {
		convo = folded
		r.setConvo(convo)
	}

	msgs := a.cfg.strategy.Apply(View{Messages: convo, Results: r.results, Tokens: r.tokens.estimate, Window: a.cfg.window})
	// The Strategy contract is that its output still validates, and the stored
	// transcript is only checked once, before the first iteration. A strategy
	// that breaks the contract — orphans a tool_result, leaves a tool_use
	// unanswered, empties the transcript — would otherwise surface as the
	// provider's 400, naming none of that, after a paid round trip.
	//
	// Only a strategy that broke a transcript that was fine going in is
	// blamed. The stored transcript can itself fail Validate for reasons that
	// are the model's, not the strategy's — a gateway that reuses tool_use ids
	// across turns does — and the loop has always left those to the provider
	// to accept or refuse. Validating the stored copy costs nothing on the
	// common path: it only runs once the strategy's output has failed.
	if err := Convo(msgs).Validate(); err != nil && convo.Validate() == nil {
		return fail(fmt.Errorf("wombat: strategy %s produced a transcript that cannot be sent: %w", a.cfg.strategy, err))
	}
	if rec, ok := a.set.(tool.Reconciler); ok {
		rec.Reconcile(ctx, observationsIn(msgs))
	}
	r.reportTrim(convo, msgs)

	if a.cfg.turnNotice != nil {
		if note := a.cfg.turnNotice(ctx, iter); note != "" {
			msgs = withNotice(msgs, note)
		}
	}
	// A notice left by a truncated reply, applied once and cleared. It is what
	// makes the retry a DIFFERENT request; without it the model would compose
	// the same over-long answer and be cut off in the same place.
	if r.truncNotice != "" {
		msgs = withNotice(msgs, r.truncNotice)
		r.truncNotice = ""
	}

	visible := a.set.Visible(ctx)
	choice, forced := a.toolChoice(iter)
	span.Set(trace.AttrToolCount, len(visible))
	span.Set(trace.AttrMessageCount, len(msgs))
	if forced != "" {
		span.Set(trace.AttrForcedTool, forced)
	}

	r.emit(LLMStart{
		Model:   a.cfg.model,
		Purpose: a.cfg.purpose,
		Tools:   len(visible),
		Forced:  forced,
	})

	// Everything the caller has already SEEN is accumulated here, so that a
	// call which fails part-way still leaves it in the transcript. Guarded
	// because a client is free to stream from its own goroutine.
	var (
		seenMu sync.Mutex
		seen   strings.Builder
		// How much scratchpad the model streamed. Not kept, only counted: a
		// turn that ends with nothing to show for it is a different animal
		// depending on whether the model was silent or was cut off mid-thought,
		// and this is the only evidence available when the response carries no
		// usage at all. See emptyTurn's handling below.
		reasoned int
	)

	req := llm.Request{
		System:    a.system,
		Messages:  msgs,
		Tools:     tool.Specs(visible),
		Choice:    choice,
		Model:     a.cfg.model,
		MaxTokens: a.cfg.maxTokens,
		Purpose:   a.cfg.purpose,
		OnDelta: func(d llm.Delta) {
			if d.Text != "" {
				seenMu.Lock()
				seen.WriteString(d.Text)
				seenMu.Unlock()
				r.emit(TextDelta{Text: d.Text})
			}
			if d.Reasoning != "" {
				seenMu.Lock()
				reasoned += len(d.Reasoning)
				seenMu.Unlock()
				r.emit(ReasoningDelta{Text: d.Reasoning})
			}
			if ta := d.ToolArgs; ta != nil {
				r.emit(ToolArgsDelta{Index: ta.Index, UseID: ta.ID, Name: ta.Name, Text: ta.JSON})
			}
		},
	}

	started := time.Now()
	sent := &sentRecorder{}
	resp, err := a.cfg.client.Complete(withSent(ctx, sent), req)
	if err == nil {
		// Calibrate on what was actually sent, which is not msgs when the
		// overflow ladder shrank the request on its way out.
		r.tokens.observe(sent.or(msgs), len(req.System)+specBytes(req.Tools), resp.Usage.PromptTokens())
	}
	if err != nil {
		// Keep the half-answer. The user watched it arrive; a transcript that
		// pretends it never happened cannot be resumed from, and "stop, then
		// continue" is the most ordinary thing a chat UI does. Trailing
		// whitespace is trimmed because Anthropic rejects a final assistant
		// turn that ends in it — and a final assistant turn is exactly what
		// this becomes, which conveniently makes the next call a prefill that
		// carries straight on.
		//
		// Reasoning is deliberately NOT kept: a thinking block without its
		// signature is rejected on the next turn, and a partial one has none.
		seenMu.Lock()
		partial := strings.TrimRight(seen.String(), " \t\r\n")
		seenMu.Unlock()
		if partial != "" {
			convo = convo.PushAssistant([]llm.ContentBlock{llm.Text{Text: partial}})
			r.setConvo(convo)
		}
		return fail(r.causeOr(err))
	}
	r.emit(LLMDone{
		Model:      resp.Model,
		StopReason: resp.StopReason,
		Usage:      resp.Usage,
		Millis:     millis(time.Since(started)),
	})

	// A turn that ended cleanly having said nothing and called nothing is not
	// an answer, and the loop retries it rather than believing it.
	//
	// What this is defending against, observed live: a reasoning model streams
	// its scratchpad for forty-five seconds, the stream ends with
	// finish_reason "stop", and the response carries no text, no tool call and
	// no usage — not zero tokens, no usage chunk at all. Nothing in it is an
	// answer, but StopEndTurn said the turn was over, so the run reported
	// Answer{Text: ""} and success. Every episode of a benchmark task died
	// that way and the failure was attributed to the verifiers.
	//
	// Nothing is pushed to the transcript before retrying, and that is the
	// point of doing this HERE rather than in the switch below: the discarded
	// content is a lone thinking block, which is worse than useless on the
	// next request — Anthropic rejects a thinking block whose signature is
	// missing, and every provider would read it as a turn the model already
	// took. Retrying from the unchanged transcript re-asks the identical
	// question, which is exactly what gets a real answer.
	//
	// The retry costs an iteration, deliberately: it shows up as an IterStart,
	// it is bounded by the run's own iteration cap as well as by the counter,
	// and a model that has genuinely gone mute cannot spin here for free.
	if emptyTurn(resp) {
		seenMu.Lock()
		thought := reasoned
		seenMu.Unlock()

		r.emptyTurns++
		r.emit(EmptyTurn{Attempt: r.emptyTurns, StopReason: resp.StopReason, ReasoningBytes: thought})
		e := &EmptyTurnError{
			Attempts:       r.emptyTurns,
			StopReason:     resp.StopReason,
			ReasoningBytes: thought,
			MaxTokens:      a.cfg.maxTokens,
		}
		// Retrying is for a provider having a bad minute. A turn that streamed a
		// budget's worth of reasoning and then stopped is not that: the request
		// is unchanged, so the next attempt spends the same budget the same way
		// and stops in the same place. Two more identical attempts at over two
		// minutes each is a bad trade for a conclusion already in hand — this is
		// the same reason ErrMaxTokens is not retryable.
		if e.budgetExhausted() || r.emptyTurns > maxEmptyTurns {
			return fail(e)
		}
		return convo, false
	}
	r.emptyTurns = 0

	// A reply cut off at max_tokens is handled BEFORE the push, for the same
	// reason an empty turn is: the content is half a thought, and pushing it
	// makes the provider read it as a turn the model already took. Retrying
	// from the unchanged transcript re-asks the identical question, which with
	// the notice attached is what gets a shorter answer.
	if resp.StopReason == llm.StopMaxTokens {
		if r.truncated < maxTruncatedTurns {
			r.truncated++
			r.truncNotice = truncationNotice(a, r)
			r.emit(TurnTruncated{Attempt: r.truncated, MaxTokens: a.cfg.maxTokens})
			return convo, false
		}
		return fail(fmt.Errorf("%w after %d attempt(s) to shorten it", ErrMaxTokens, r.truncated))
	}
	// Reset only on a turn that was not truncated. Resetting unconditionally
	// zeroed the counter immediately before incrementing it, so the retry
	// budget never ran out — caught by the bounded-retry test.
	r.truncated = 0

	convo = convo.PushAssistant(resp.Content)
	r.setConvo(convo)

	switch resp.StopReason {
	case llm.StopEndTurn, llm.StopStopSequence:
		// A message the user sent while this answer was being written is
		// still owed an answer: carry on rather than end over it.
		if steer := r.takeSteering(); len(steer) > 0 {
			convo = r.deliverSteering(convo, steer)
			r.setConvo(convo)
			return convo, false
		}
		answer := llm.TextOf(resp.Content)
		if feedback := r.checkCompletion(ctx, Completion{Answer: answer, Messages: convo}); feedback != "" {
			convo = convo.PushUserText(notDoneYet(feedback))
			r.setConvo(convo)
			return convo, false
		}
		return done(Answer{Text: answer})

	case llm.StopRefusal:
		return fail(&RefusalError{Reason: llm.TextOf(resp.Content)})

	case llm.StopToolUse:
		uses := llm.ToolUses(resp.Content)
		if len(uses) == 0 {
			return fail(errors.New("wombat: provider reported stop_reason=tool_use with no tool_use block"))
		}

		span.Set("wombat.tool_calls", len(uses))
		results, stop := r.dispatch(ctx, convo, uses)
		if stop != nil {
			return done(stop)
		}
		for _, res := range results {
			r.results[res.UseID] = ResultInfo{
				Tool:  res.Name,
				Tags:  res.Tags,
				Bytes: res.Size(),
			}
		}
		convo = convo.PushToolResults(tool.Blocks(results))
		r.setConvo(convo)
		return convo, false

	default:
		return fail(&UnexpectedStopError{StopReason: resp.StopReason})
	}
}

// dispatch runs one turn's tool calls, and returns an Outcome instead when one
// of them is allowed to end or suspend the run.
//
// Terminal and pause are properties of the tool, not of a hard-coded name: a
// set containing one can end or suspend, a set without one never can. But a
// call to one is still a CALL, and it is decided by the same chain as every
// other. It used to short-circuit before dispatch, which put it beyond every
// middleware: a permission policy that refused it, or a skill gate that had
// not unlocked it and was not even offering it to the model, was never asked —
// the call ended or suspended the run regardless, and produced no
// ToolStart/ToolDone, no metric and no budget charge on the way.
//
// Visibility alone still decides nothing, exactly as for any other tool: a
// [tool.Set] that hides a tool from Visible without middleware that refuses it
// lets the call through. Enforcement is the middleware's job.
//
// So each candidate is dispatched on its own through the agent's dispatcher,
// with its handler swapped for one that accepts without doing anything (see
// controlSet): the handler of such a tool must never run, and everything around
// it — gates, validation, the tape, metrics, the budget, the observer — runs as
// it would for any call. If the chain lets it through, the run ends or pauses
// exactly as before. If something refuses it, the refusal becomes that call's
// tool_result and the turn carries on as an ordinary batch, so the model learns
// why and can do something else.
//
// Terminal calls are considered before pause calls, each in the order the model
// made them, so ending still beats suspending when both are allowed. Siblings
// are never dispatched when the run ends or pauses; [AnswerPause] and [Then]
// close them when the conversation resumes.
//
// A terminal call the chain lets through still has to satisfy the agent's
// [CompletionCheck], when it has one. One the check rejects is answered with
// the check's feedback and handled from there exactly like a refused call.
func (r *Run) dispatch(ctx context.Context, convo Convo, uses []llm.ToolUse) ([]tool.Result, Outcome) {
	a := r.a
	order := controlOrder(a.set, uses)
	if len(order) == 0 {
		return a.disp.Dispatch(ctx, a.set, uses), nil
	}

	results := make([]tool.Result, len(uses))
	decided := make([]bool, len(uses))
	verdicts := controlSet{a.set}
	for _, i := range order {
		u := uses[i]
		res := dispatchOne(ctx, a.disp, verdicts, u)
		if res.Err == nil {
			if d, _ := a.set.Find(u.Name); d.Has(tool.CapTerminal) {
				sub := Submitted{Tool: u.Name, Payload: u.Input}
				feedback := r.checkCompletion(ctx, Completion{Submitted: &sub, Messages: convo})
				if feedback == "" {
					return nil, sub
				}
				// Not done, says the check: the submission is answered with
				// why, and the turn carries on like one the gate refused.
				res = tool.Result{UseID: u.ID, Name: u.Name, Err: errNotAccepted(feedback)}
			} else {
				return nil, Paused{ToolUseID: u.ID, Schema: ParsePauseSchema(u.Input)}
			}
		}
		results[i], decided[i] = res, true
	}

	// Nothing was allowed to stop the run, so the rest of the turn is ordinary
	// work, dispatched as one batch exactly as it would have been.
	var rest []llm.ToolUse
	var at []int
	for i, u := range uses {
		if !decided[i] {
			rest = append(rest, u)
			at = append(at, i)
		}
	}
	if len(rest) > 0 {
		got := a.disp.Dispatch(ctx, a.set, rest)
		for j, u := range rest {
			if j < len(got) {
				results[at[j]] = got[j]
			} else {
				results[at[j]] = missingResult(u)
			}
		}
	}
	return results, nil
}

// controlOrder lists the indices of the calls that would end or suspend the
// run: terminal calls first, then pause calls, each in the order made.
func controlOrder(set tool.Set, uses []llm.ToolUse) []int {
	var terminal, pause []int
	for i, u := range uses {
		d, ok := set.Find(u.Name)
		switch {
		case !ok:
		case d.Has(tool.CapTerminal):
			terminal = append(terminal, i)
		case d.Has(tool.CapPause):
			pause = append(pause, i)
		}
	}
	return append(terminal, pause...)
}

// dispatchOne dispatches a single call, holding a dispatcher to the
// one-result-per-call contract rather than indexing past what it returned.
func dispatchOne(ctx context.Context, d tool.Dispatcher, set tool.Set, u llm.ToolUse) tool.Result {
	if rs := d.Dispatch(ctx, set, []llm.ToolUse{u}); len(rs) == 1 {
		return rs[0]
	}
	return missingResult(u)
}

func missingResult(u llm.ToolUse) tool.Result {
	return tool.Result{UseID: u.ID, Name: u.Name,
		Err: fmt.Errorf("wombat: the dispatcher returned no result for %s", u.Name)}
}

// controlSet is a tool set whose pause and terminal tools resolve to a handler
// that accepts the call without running anything.
//
// It is how such a call gets a verdict from the dispatch chain without its real
// handler ever being reached — which [tool.CapTerminal] promises, and which the
// builtin ask_user enforces by failing loudly. Swapping the handler in the set,
// rather than teaching the chain's leaf about control tools, is what lets a
// dispatcher supplied with [WithDispatcher] give the same verdict: every
// dispatcher resolves a call through the set it is handed.
//
// The accepting handler's output is what ToolDone reports. It never reaches the
// transcript: an accepted call ends or suspends the run before any
// tool_result is written.
type controlSet struct{ tool.Set }

func (s controlSet) Find(name string) (tool.Def, bool) {
	d, ok := s.Set.Find(name)
	if !ok {
		return d, false
	}
	switch {
	case d.Has(tool.CapTerminal):
		d.Fn = accept("(submitted: the run ends here)")
	case d.Has(tool.CapPause):
		d.Fn = accept("(paused: waiting for the user to answer)")
	}
	return d, true
}

func accept(note string) tool.Fn {
	return func(context.Context, json.RawMessage) (string, error) { return note, nil }
}

// Progress reports the run's spend against its budget. A run started with no
// budget on its context still has a tally of its own, with no limits; see
// [governor.Ensure].
func (r *Run) Progress() governor.Progress {
	return governor.FromContext(r.ctx).Progress()
}

// withNotice appends a note to the last user turn of a materialized request.
//
// Copies rather than mutating: msgs may alias the stored transcript, and a
// notice that leaked into it would accumulate one copy per iteration.
func withNotice(msgs []llm.Message, note string) []llm.Message {
	n := len(msgs)
	if n == 0 || msgs[n-1].Role != llm.RoleUser {
		return msgs
	}
	out := make([]llm.Message, n)
	copy(out, msgs)

	last := out[n-1]
	content := make([]llm.ContentBlock, len(last.Content), len(last.Content)+1)
	copy(content, last.Content)
	out[n-1] = llm.Message{Role: last.Role, Content: append(content, llm.Text{Text: note})}
	return out
}

// truncateStack keeps a panic's stack useful without dumping a screenful into
// an error string.
func truncateStack(b []byte) string {
	const limit = 4 << 10
	if len(b) <= limit {
		return string(b)
	}
	return string(b[:limit]) + "\n… stack truncated"
}

// outcomeLabel names how a run ended, for a metric label.
//
// A bounded vocabulary on purpose: the error's own text would be unbounded
// cardinality, which is how a metrics backend falls over.
func outcomeLabel(o Outcome, err error) string {
	switch {
	case err != nil:
		return "failed"
	case o == nil:
		return "none"
	}
	switch o.(type) {
	case Answer:
		return "answered"
	case Paused:
		return "paused"
	case Submitted:
		return "submitted"
	default:
		return "other"
	}
}

// observationsIn reports which tool calls still have their observation in the
// materialized transcript.
//
// Keyed on the tool_result and not the tool_use: the request block is a few
// bytes of arguments, the result is the payload, and the result is what a
// gated skill's knowledge actually lives in.
func observationsIn(msgs []llm.Message) map[llm.ToolUseID]bool {
	present := make(map[llm.ToolUseID]bool)
	for _, m := range msgs {
		for _, b := range m.Content {
			// An elided result is an evicted one: its id is still there, its
			// body is not, and a skill whose body went with it is retired.
			if tr, ok := b.(llm.ToolResult); ok && !strings.HasPrefix(tr.Content, elidedPrefix) {
				present[tr.ToolUseID] = true
			}
		}
	}
	return present
}

// reportTrim emits ContextTrimmed when what the strategy sends differs from
// the stored transcript in a way it did not on the previous iteration — a new
// cut, more results elided. A trimmed transcript stays trimmed for the rest of
// the run, and an event every iteration would say nothing the first did not.
func (r *Run) reportTrim(stored Convo, sent []llm.Message) {
	elided := 0
	for _, m := range sent {
		for _, b := range m.Content {
			if tr, ok := b.(llm.ToolResult); ok && strings.HasPrefix(tr.Content, elidedPrefix) {
				elided++
			}
		}
	}
	dropped := len(stored) - len(sent)
	if dropped <= 0 && elided == 0 {
		r.trimmed = [2]int{}
		return
	}
	if r.trimmed == [2]int{dropped, elided} {
		return
	}
	r.trimmed = [2]int{dropped, elided}
	r.emit(ContextTrimmed{
		Strategy: r.a.cfg.strategy.String(),
		Dropped:  dropped,
		Elided:   elided,
		Tokens:   r.tokens.estimate(sent),
	})
}

func (r *Run) snapshot() Convo {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.convo
}

// toolChoice decides whether to force the terminal tool this iteration.
//
// Without this, a run that is one iteration from its cap tends to spend it on
// another tool call and then fail with nothing submitted. Forcing the choice
// converts "ran out of iterations" into "submitted what it had".
func (a *Agent) toolChoice(iter int) (llm.ToolChoice, string) {
	if a.cfg.terminal == "" || a.cfg.forceLastN <= 0 {
		return llm.ToolChoice{}, ""
	}
	threshold := max(a.cfg.maxIters-a.cfg.forceLastN, 1)
	if iter >= threshold {
		return llm.ForceTool(a.cfg.terminal), a.cfg.terminal
	}
	return llm.ToolChoice{}, ""
}

// maxEmptyTurns is how many consecutive content-free turns the loop retries
// before it gives up with [ErrEmptyTurn].
//
// Two, so three calls are made in all. One retry is not enough to ride out a
// provider having a bad minute, and a large number is not a retry policy —
// each of these costs a full model call, and a model that has answered nothing
// three times running is not about to start.
const maxEmptyTurns = 2

// maxTruncatedTurns is how many times a reply cut off at max_tokens is asked
// for again, shorter.
//
// Two. One retry is not enough — a model that has just been told to be brief
// often still overshoots on the first correction — and a large number is not a
// policy: each attempt costs a full generation at the cap, which is the most
// expensive call the loop can make.
const maxTruncatedTurns = 2

// emptyTurn reports whether resp claims to have finished while containing
// nothing to act on.
//
// Restricted to the stop reasons that MEAN finished. The others already have
// precise handling that must not be swallowed by a retry: max_tokens has to
// tell the caller to shorten the request, a refusal has to surface as a
// refusal, an unknown stop_reason is a diagnostic, and a tool_use with no
// tool_use block is a provider bug worth naming exactly.
//
// Text is trimmed before the test because a turn of pure whitespace is a turn
// that said nothing, and providers do emit one.
func emptyTurn(resp llm.Response) bool {
	switch resp.StopReason {
	case llm.StopEndTurn, llm.StopStopSequence:
		return strings.TrimSpace(llm.TextOf(resp.Content)) == "" && len(llm.ToolUses(resp.Content)) == 0
	default:
		return false
	}
}

// truncationNotice is what the retry tells the model, and it escalates.
//
// The second notice is not the first one repeated. A model that was cut off
// once and told to be shorter has been observed to resend the same oversized
// reply verbatim, three times running, until the run failed with nothing
// written. Restating the advice does not change the reply; removing the option
// to attempt the whole thing is what changes it.
func truncationNotice(a *Agent, r *Run) string {
	if r.truncated <= 1 {
		return fmt.Sprintf(
			"[system] Your previous reply was cut off after %d tokens, so none of it was "+
				"used. Reply again, shorter. If you were writing a long file, write it in "+
				"several smaller calls instead of one; if you were writing a report, give "+
				"the conclusion first and keep it to a few lines.", a.cfg.maxTokens)
	}
	return fmt.Sprintf(
		"[system] Cut off at %d tokens again, at the same place, and again none of it was "+
			"used. Do not send that reply a third time. Send a FRACTION of it now — the "+
			"first part of the file, the first of the changes, one section — small enough "+
			"to finish well inside the limit, and continue in the calls after this one. A "+
			"part that arrives is worth more than a whole that never does.", a.cfg.maxTokens)
}
