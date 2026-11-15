package httpapi

import (
	"context"
	"errors"
	"iter"
	"slices"
	"sync"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/permission"
	"github.com/hoobean1996/wombat-go/tool"
)

// Session is one conversation: an agent, a transcript that grows across turns,
// and a single event log whose sequence numbers span all of them.
//
// One log and not one per run is the load-bearing decision. A UI reconnects
// with the last sequence it rendered; if numbering restarted at every turn,
// a reconnect that lands after a turn boundary would either replay the whole
// new turn or skip it, and there is no way for the client to tell which.
//
// Every method is safe to call from any goroutine.
type Session struct {
	id      string
	m       *Manager
	agent   *wombat.Agent
	opts    SessionOptions
	created time.Time

	// bufCap is how many frames the log retains; see [Session.Follow] for how
	// a follower detects that eviction passed it.
	bufCap int

	// wg tracks the pump goroutine, so [Manager.Close] can promise that no
	// run is still executing when it returns.
	wg sync.WaitGroup

	// approvals outlives any one turn: a session's questions are the session's,
	// and closing the queue is what makes it unusable.
	approvals *approvals

	mu      sync.Mutex
	state   State
	turns   int
	updated time.Time
	closed  bool

	// frames[i] carries sequence base+i. base rises only on eviction.
	frames []Frame
	base   int
	// changed is closed — and replaced — on every append, and once more when
	// the session is dropped. A per-wait channel rather than a sync.Cond
	// because a follower has to be able to wake on its own ctx too, and
	// Cond.Wait cannot be selected on.
	changed chan struct{}

	// run is the turn in progress, for Steer; nil between turns.
	run *wombat.Buffered

	// msgs is the conversation so far, replaced wholesale at the end of every
	// turn with the run's own final transcript.
	msgs  []llm.Message
	spend Spend

	outcome   string
	answer    string
	errText   string
	errKind   string
	pauseID   llm.ToolUseID
	turnStart time.Time

	// cancel ends the turn in flight. nil between turns.
	cancel context.CancelFunc

	// dropped marks a session deleted or expired, which the store forgets;
	// saveSeq numbers snapshots under mu, and savedSeq — under saveMu — is
	// the newest one written, so a slow save of an older snapshot never
	// overwrites a newer one.
	dropped  bool
	saveSeq  uint64
	saveMu   sync.Mutex
	savedSeq uint64
}

// recordLocked snapshots the session for the store, with msgs standing in for
// the transcript when it is not nil. Callers hold s.mu.
func (s *Session) recordLocked(msgs []llm.Message) (SessionRecord, uint64) {
	if msgs == nil {
		msgs = s.msgs
	}
	s.saveSeq++
	return SessionRecord{
		Version: RecordVersion, ID: s.id, Options: s.opts, Created: s.created, Updated: s.updated,
		Turns: s.turns, State: s.state, Messages: msgs, Spend: s.spend,
		Outcome: s.outcome, Answer: s.answer, Error: s.errText, ErrorKind: s.errKind,
		PauseID: s.pauseID, Events: s.base + len(s.frames),
	}, s.saveSeq
}

// persist writes a snapshot unless a newer one has been written already, or
// the session has been dropped. A failed save is logged: the conversation
// goes on, it is only its survival across a restart that is at risk.
func (s *Session) persist(rec SessionRecord, seq uint64) {
	st := s.m.cfg.Store
	if st == nil {
		return
	}
	s.saveMu.Lock()
	defer s.saveMu.Unlock()
	s.mu.Lock()
	dropped := s.dropped
	s.mu.Unlock()
	if dropped || seq <= s.savedSeq {
		return
	}
	if err := st.Save(rec); err != nil {
		s.m.logger().Warn("session store: save failed", "session", s.id, "err", err)
		return
	}
	s.savedSeq = seq
}

// drop closes the session and has the store forget it. The store is told
// after the pump has finished, so its last save cannot bring the session back.
func (s *Session) drop() {
	s.mu.Lock()
	s.dropped = true
	s.mu.Unlock()
	s.close()
	if st := s.m.cfg.Store; st != nil {
		s.saveMu.Lock()
		defer s.saveMu.Unlock()
		if err := st.Delete(s.id); err != nil {
			s.m.logger().Warn("session store: delete failed", "session", s.id, "err", err)
		}
	}
}

// ID returns the session's opaque identifier.
func (s *Session) ID() string { return s.id }

// Info snapshots everything about the session that does not require reading
// its log.
func (s *Session) Info() SessionInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.infoLocked()
}

func (s *Session) infoLocked() SessionInfo {
	state := s.state
	// Approving is derived rather than stored: the gate parks a call on a
	// goroutine this type never hears from, and a stored flag would have to be
	// cleared from the same three places a verdict can arrive from. Asking the
	// queue cannot go stale.
	if state == Running && s.approvals.count() > 0 {
		state = Approving
	}
	return SessionInfo{
		ID:        s.id,
		Title:     s.opts.Title,
		State:     state,
		Turns:     s.turns,
		Events:    s.base + len(s.frames),
		Created:   s.created,
		Updated:   s.updated,
		Options:   s.opts,
		Spend:     s.spend,
		Outcome:   s.outcome,
		Answer:    s.answer,
		Error:     s.errText,
		ErrorKind: s.errKind,
	}
}

// Send starts the next turn and returns its number, counting from one.
//
// It returns [ErrBusy] if a turn is already running, [ErrDone] if the model
// has ended the conversation with a terminal tool, and [ErrNoSuchSession] if
// the session has been dropped. It does NOT wait for the turn: the reply
// arrives on the event log, which is the only place it can arrive if the
// client that asked is allowed to disconnect.
//
// How the turn is seeded depends on where the last one left off:
//
//   - the first turn is [wombat.Ask];
//   - a turn after one that ended in [wombat.Paused] is [wombat.AnswerPause],
//     so the prompt answers the model's question through the tool call it
//     asked with, rather than arriving as an unrelated new instruction;
//   - anything else is [wombat.Then], which closes a dangling tool_use before
//     appending — a cancelled turn leaves one, and a provider rejects a
//     transcript that still has it open.
func (s *Session) Send(prompt string) (int, error) {
	// Registered before the unlock below, so it runs after it: the snapshot
	// is taken under the lock and written outside it.
	var rec SessionRecord
	var seq uint64
	defer func() {
		if seq > 0 {
			s.persist(rec, seq)
		}
	}()
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return 0, ErrNoSuchSession
	}
	if s.cancel != nil {
		return 0, ErrBusy
	}
	if s.state == Done {
		return 0, ErrDone
	}

	// The same three-way decision wombat.Resume makes, kept here rather than
	// delegated because this side has better information: pauseID came from the
	// Paused outcome itself, so there is nothing to infer from the transcript.
	// wombat.Resume exists for the caller that only has the messages — the CLI
	// reading a -session file. Both route a pause to AnswerPause, which is what
	// closes the sibling tool calls the pause left undispatched; change one and
	// read the other.
	turn := s.turns + 1
	var in wombat.Input
	switch {
	case turn == 1:
		in = wombat.Ask(prompt)
	case s.pauseID != "":
		in = wombat.AnswerPause(s.msgs, s.pauseID, prompt)
	default:
		in = wombat.Then(s.msgs, prompt)
	}

	// Background, not a request context: the request is over in a millisecond
	// and the turn must not be. The budget is per turn, so a long conversation
	// is capped per exchange rather than dying halfway through its fourth.
	ctx, cancel := governor.WithBudget(context.Background(), s.m.cfg.Limits)
	ctx = withApprovals(ctx, s.approvals)

	// One grant set per TURN, not per session: an approval is a statement
	// about the work in front of the user now, not a standing licence for
	// every later exchange. Within the turn it stops the same question being
	// asked at iteration 2 and again at iteration 9.
	ctx = permission.WithGrants(ctx, permission.NewGrants())

	// Under the lock, and deliberately: Start and Buffer only spawn
	// goroutines, and doing them here is what closes the window in which
	// Cancel could run between the busy check and s.cancel being set.
	buf := wombat.Buffer(s.agent.Start(ctx, in))

	s.turns = turn
	s.state = Running
	s.cancel = cancel
	s.run = buf
	s.turnStart = time.Now()
	s.pauseID = ""
	s.outcome, s.answer, s.errText, s.errKind = "", "", "", ""
	s.emitLocked(turn, TurnStarted{Turn: turn, Prompt: prompt})
	// Saved as running with the turn's own input, prompt included: if the
	// process dies before the turn ends, the restored session still knows
	// what it was asked.
	rec, seq = s.recordLocked(in.Messages)

	s.wg.Add(1)
	go s.pump(ctx, turn, buf, cancel)
	return turn, nil
}

// pump copies one run's events into the session log and records how it ended.
// Steer passes prompt to the turn in progress; see wombat.Run.Steer. It is
// echoed into the session's log, like a prompt, so a second client watching
// the session sees what was said.
func (s *Session) Steer(prompt string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrNoSuchSession
	}
	if s.run == nil || !s.run.Steer(prompt) {
		return ErrNotRunning
	}
	s.emitLocked(s.turns, SteerSent{Turn: s.turns, Prompt: prompt})
	return nil
}

func (s *Session) pump(ctx context.Context, turn int, buf *wombat.Buffered, cancel context.CancelFunc) {
	defer s.wg.Done()
	defer cancel()

	// context.Background and not ctx: Follow must drain the buffer to the very
	// end even when the run was cancelled, or the events explaining WHY it
	// stopped never reach the log. Follow returns on its own once the run is
	// over, which cancelling guarantees.
	for _, ev := range buf.Follow(context.Background(), 0) {
		s.append(turn, ev)
	}

	end := TurnEnded{Turn: turn, Spend: turnSpend(ctx)}
	switch o := buf.Outcome().(type) {
	case wombat.Answer:
		// Idle, not Done. A chat is not over because the model stopped
		// talking: an answer is the ordinary end of a TURN and the ordinary
		// invitation to the next one, and the follow-up is why this package
		// keeps a transcript at all. Ending the session here would make every
		// conversation one exchange long and would leave POST .../messages
		// reachable only after a pause — which is to say, reachable by
		// accident.
		end.State, end.Outcome, end.Answer = Idle, "answer", o.Text
	case wombat.Paused:
		end.State, end.Outcome, end.Question = Waiting, "paused", o.Schema.Question
	case wombat.Submitted:
		// Done, and this is the only thing that produces it. A terminal tool
		// is the model saying the TASK is finished rather than the turn, so
		// there is nothing to continue from: Send refuses with [ErrDone] and
		// every follower is released.
		end.State, end.Outcome, end.Tool = Done, "submitted", o.Tool
		// The payload IS the answer, in structured form. Without this the one
		// thing the run was for would appear nowhere outside the transcript.
		end.Answer = string(o.Payload)
	default:
		err := buf.Err()
		end.State, end.Error, end.ErrorKind = Failed, errText(err), ErrorKind(err)
	}

	msgs := buf.Messages()
	pauseID := llm.ToolUseID("")
	if p, ok := buf.Outcome().(wombat.Paused); ok {
		pauseID = p.ToolUseID
	}

	s.mu.Lock()
	// Keep the transcript even on failure. The user watched half an answer
	// arrive; a conversation that pretends it never happened cannot be
	// continued from, and "it broke, carry on" is the most ordinary thing a
	// chat UI does.
	s.msgs = msgs
	s.spend.add(end.Spend)
	s.state = end.State
	s.outcome, s.answer = end.Outcome, end.Answer
	if end.Question != "" {
		s.answer = end.Question
	}
	s.errText, s.errKind = end.Error, end.ErrorKind
	s.pauseID = pauseID
	s.cancel = nil
	s.run = nil
	// What was sent with Steer and never reached the model — the turn paused,
	// failed or was cancelled first — is handed back, so the UI can offer to
	// send it as the next turn instead of losing it.
	end.Undelivered = buf.Undelivered()
	if s.closed && !s.dropped && end.State == Failed {
		// Not deleted, so the manager is shutting down: the turn was cut
		// short by the restart, not by anything about the turn.
		s.errText, s.errKind = "the server shut down while this turn was running", KindInterrupted
	}
	s.emitLocked(turn, end)
	rec, seq := s.recordLocked(nil)
	s.mu.Unlock()
	s.persist(rec, seq)

	s.m.logger().Info("turn ended", "session", s.id, "turn", turn,
		"state", string(end.State), "cost_usd", end.Spend.CostUSD)
}

// Follow yields the session's log from sequence from, blocking for frames that
// do not exist yet.
//
// It returns on exactly three things, and NOT at the end of a turn:
//
//   - ctx is done — the client went away;
//   - the session was dropped by [Manager.Delete] or [Manager.Close];
//   - the session reached [Done], the one terminal state.
//
// Not returning between turns is the point: one EventSource spans the whole
// conversation, and a client does not have to notice a turn boundary and
// reconnect through it. Returning on [Done] is the other half of the same
// deal — a stream that never ends when the conversation has is a connection
// held open for nothing, and a client waiting for an end frame that cannot
// come.
//
// Sequence numbers start at 0, are dense, and never change. from is clamped:
// below the oldest retained frame it starts at the oldest retained one, which
// is the only way eviction is observable — a consumer that cares compares the
// first sequence it is yielded against the one it asked for.
func (s *Session) Follow(ctx context.Context, from int) iter.Seq2[int, Frame] {
	return func(yield func(int, Frame) bool) {
		seq := max(from, 0)
		for {
			s.mu.Lock()
			// Clamped inside the loop, not once up front: eviction can pass a
			// slow follower at any point, not just before its first read.
			seq = max(seq, s.base)
			var batch []Frame
			if i := seq - s.base; i < len(s.frames) {
				batch = slices.Clone(s.frames[i:])
			}
			// Both under the same lock as the snapshot. Reading the end
			// condition any later would race with the final append, and a
			// follower that concluded "over, nothing pending" from a stale
			// pair would drop the last frames of the session — including the
			// TurnEnded that says how it finished — on the floor.
			wait, over := s.changed, s.closed || s.state == Done
			s.mu.Unlock()

			for _, f := range batch {
				if !yield(f.Seq, f) {
					return
				}
				seq = f.Seq + 1
			}
			if len(batch) > 0 {
				continue
			}
			if over {
				return
			}

			select {
			case <-wait:
			case <-ctx.Done():
				return
			}
		}
	}
}

// Messages returns a snapshot of the conversation as it stood at the end of
// the last completed turn.
//
// Deliberately not the live transcript of a running turn: the harness's own
// copy is mid-mutation, and a UI that rendered it would show a tool_use with
// no result and then have to un-show it.
func (s *Session) Messages() []llm.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.msgs)
}

// Pending lists the tool calls parked on a human, oldest first.
func (s *Session) Pending() []Approval { return s.approvals.pending() }

// Resolve answers one parked call, letting an allowed one go on to actually
// execute. It reports [ErrNoSuchApproval] or [ErrAlreadyAnswered].
func (s *Session) Resolve(useID string, allow bool) error {
	if err := s.approvals.resolve(llm.ToolUseID(useID), allow); err != nil {
		return err
	}
	s.touch()
	return nil
}

// Cancel stops the turn in flight — the model call, and any subprocess a tool
// started — and leaves the session usable for another prompt.
//
// A no-op between turns. Parked approvals are denied first: cancelling would
// unblock them anyway, but denying explicitly settles the race between the
// cancel and a verdict arriving in the same instant.
func (s *Session) Cancel() {
	s.approvals.deny()

	s.mu.Lock()
	cancel := s.cancel
	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}

// close drops the session for good: the turn in flight is cancelled, the
// approval queue is shut, and every follower is released.
//
// It waits for the pump so that a caller dropping the last reference knows no
// goroutine of this session is still running. Called only by [Manager].
func (s *Session) close() {
	s.approvals.close()

	s.mu.Lock()
	cancel := s.cancel
	s.closed = true
	s.wake()
	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	s.wg.Wait()
}

// append records one event under the session's global sequence.
func (s *Session) append(turn int, ev wombat.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.emitLocked(turn, ev)
}

// emitLocked appends a frame. Callers hold s.mu.
func (s *Session) emitLocked(turn int, ev wombat.Event) {
	if s.closed {
		return
	}
	s.frames = append(s.frames, Frame{Seq: s.base + len(s.frames), Turn: turn, Event: ev})

	// Overflow drops the OLDEST frames, never the newest. Dropping the newest
	// would stall a live follower behind a permanent gap; dropping the oldest
	// costs a resuming client only the part of the transcript it is most
	// likely to have already rendered.
	if over := len(s.frames) - s.bufCap; over > 0 {
		copy(s.frames, s.frames[over:])
		// Clear the vacated tail so the evicted frames — a tool_done can carry
		// a large output — are actually collectable.
		clear(s.frames[s.bufCap:])
		s.frames = s.frames[:s.bufCap]
		s.base += over
	}
	s.updated = time.Now()
	s.wake()
}

// touch marks activity, which is what the TTL is measured from.
func (s *Session) touch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.updated = time.Now()
}

// wake releases every follower parked on the current generation. Callers hold
// s.mu.
func (s *Session) wake() {
	close(s.changed)
	s.changed = make(chan struct{})
}

// idleSince reports when the session was last touched, and whether it is
// currently between turns. Only an idle session is reapable.
func (s *Session) idleSince() (time.Time, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.updated, s.cancel == nil
}

// turnSpend reads one turn's tally off its governed context.
func turnSpend(ctx context.Context) Spend {
	p := governor.FromContext(ctx).Progress()
	return Spend{
		CostUSD:      p.CostUSD,
		InputTokens:  p.Tokens.In,
		OutputTokens: p.Tokens.Out,
		CacheRead:    p.Tokens.CacheRead,
		Calls:        p.Calls,
		ElapsedSec:   p.Elapsed.Seconds(),
	}
}

func errText(err error) string {
	if err == nil {
		return "the turn produced no outcome"
	}
	return err.Error()
}

// ErrorKind names why a turn failed, in a bounded vocabulary a UI can branch
// on: budget_exhausted, tool_loop, max_iterations, cancelled, context_window,
// denied, timeout or error.
//
// Bounded is the requirement. The error's own text is unbounded and changes
// with every reword, so a front end that matched on it would break silently;
// these eight tokens are a contract, and a new one is a deliberate change to
// this function rather than an accident of some error message elsewhere.
func ErrorKind(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, governor.ErrBudgetExhausted):
		return "budget_exhausted"
	case errors.Is(err, governor.ErrToolLoop):
		return "tool_loop"
	case errors.Is(err, wombat.ErrMaxIterations), errors.Is(err, governor.ErrStepLimit):
		return "max_iterations"
	case errors.Is(err, permission.ErrDenied):
		return "denied"
	case errors.Is(err, tool.ErrTimeout), errors.Is(err, context.DeadlineExceeded),
		errors.Is(err, governor.ErrWallClock):
		return "timeout"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, llm.ErrContextWindow):
		return "context_window"
	default:
		return "error"
	}
}
