package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/mcp"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/tool"
)

// KindInterrupted is the ErrorKind of a turn the daemon stopped in the
// middle of — shut down, or died. Its events run again.
const KindInterrupted = "interrupted"

// Session is one conversation in memory: its agent, its record as it stands,
// and a single event log whose sequence numbers span every turn.
//
// One log and not one per run is the load-bearing decision. A UI reconnects
// with the last sequence it rendered; if numbering restarted at every turn,
// a reconnect that lands after a turn boundary would either replay the whole
// new turn or skip it, and there is no way for the client to tell which.
//
// Every method is safe to call from any goroutine.
type Session struct {
	id      string
	thread  string
	m       *Manager
	agent   *core.Agent
	opts    SessionOptions
	created time.Time

	// bufCap is how many frames the log retains; see [Session.Follow] for how
	// a follower detects that eviction passed it.
	bufCap int

	// wg tracks the turn in progress, so close can promise that nothing of
	// the session is still running when it returns.
	wg sync.WaitGroup

	// writing orders the record's writes from outside a turn — a permission
	// changed while the session sits between turns — against the writes of
	// the turn that may begin meanwhile, so an older snapshot never lands on
	// top of a newer one. Taken without mu held: a writer that holds it takes
	// mu only to snapshot. See [Session.SetPermission].
	writing sync.Mutex

	mu      sync.Mutex
	state   State
	turns   int
	updated time.Time
	// closed is a session no longer live in memory: deleted, unloaded or
	// shut down. dropped is the first of those, which the store forgets too.
	closed    bool
	dropped   bool
	followers int

	// frames[i] carries sequence base+i. base rises only on eviction.
	frames []Frame
	base   int
	// changed is closed — and replaced — on every append, and once more when
	// the session is closed. A per-wait channel rather than a sync.Cond
	// because a follower has to be able to wake on its own ctx too, and
	// Cond.Wait cannot be selected on.
	changed chan struct{}

	// run and cancel are the turn in progress; nil between turns.
	run    *core.Buffered
	cancel context.CancelFunc

	// msgs is the conversation so far, replaced at the end of every turn with
	// the run's own final transcript; origins say where its messages came
	// from, and are replaced with it.
	msgs    []llm.Message
	origins map[int][]Origin
	spend   Spend

	outcome string
	answer  string
	errText string
	errKind string
	pauseID llm.ToolUseID

	deferred   *core.Deferred
	verdicts   map[llm.ToolUseID]bool
	deferredAt time.Time
	decidedBy  []Origin
	replyTo    []int64
}

func (m *Manager) newSession(rec SessionRecord, agent *core.Agent) *Session {
	return &Session{
		id: rec.ID, thread: rec.Thread, m: m, agent: agent, opts: rec.Options, created: rec.Created,
		bufCap: m.cfg.BufferSize, changed: make(chan struct{}),
		state: rec.State, turns: rec.Turns, updated: rec.Updated, base: rec.Events,
		msgs: rec.Messages, origins: rec.Origins, spend: rec.Spend,
		outcome: rec.Outcome, answer: rec.Answer, errText: rec.Error, errKind: rec.ErrorKind, pauseID: rec.PauseID,
		deferred: rec.Deferred, verdicts: rec.Verdicts, deferredAt: rec.DeferredAt, decidedBy: rec.DecidedBy,
		replyTo: rec.ReplyTo,
	}
}

// record snapshots the session for the store.
func (s *Session) record() SessionRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recordLocked()
}

func (s *Session) recordLocked() SessionRecord {
	return SessionRecord{
		// The owner, so the store writes the record only while this daemon
		// still holds the session.
		Lease:   Lease{Owner: s.m.cfg.Name},
		Version: RecordVersion, ID: s.id, Thread: s.thread, Options: s.opts,
		Created: s.created, Updated: s.updated,
		Turns: s.turns, State: s.state, Messages: s.msgs, Origins: maps.Clone(s.origins), Spend: s.spend,
		Outcome: s.outcome, Answer: s.answer, Error: s.errText, ErrorKind: s.errKind, PauseID: s.pauseID,
		Deferred: s.deferred, Verdicts: maps.Clone(s.verdicts), DeferredAt: s.deferredAt,
		DecidedBy: slices.Clone(s.decidedBy), ReplyTo: slices.Clone(s.replyTo),
		Events: s.base + len(s.frames),
	}
}

// ID returns the session's opaque identifier.
func (s *Session) ID() string { return s.id }

// Info snapshots everything about the session that does not require reading
// its log.
func (s *Session) Info() SessionInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	info := infoOf(s.recordLocked())
	return info
}

func infoOf(rec SessionRecord) SessionInfo {
	return SessionInfo{
		ID: rec.ID, Title: rec.Options.Title, Thread: rec.Thread, State: rec.State,
		Turns: rec.Turns, Events: rec.Events, Created: rec.Created, Updated: rec.Updated,
		Options: rec.Options, Spend: rec.Spend,
		Outcome: rec.Outcome, Answer: rec.Answer, Error: rec.Error, ErrorKind: rec.ErrorKind,
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

// Transcript is [Session.Messages] with where its messages came from: for
// each message inbox events made, keyed by its index, the events' origins.
// One call, so the two cannot straddle the end of a turn.
func (s *Session) Transcript() ([]llm.Message, map[int][]Origin) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.msgs), maps.Clone(s.origins)
}

// ===== What comes in =====

// MessagePayload is the payload of a [KindMessage] event.
type MessagePayload struct {
	Text string `json:"text"`
}

// Send puts a person's message in the session's inbox, for its next turn —
// which starts at once if the session is at rest, and after the one running
// if it is not. It returns the event; the reply arrives on the event log, and
// in the outbox.
func (s *Session) Send(ctx context.Context, prompt string) (Event, error) {
	return s.SendAs(ctx, "", prompt)
}

// SendAs is [Session.Send] from someone named, for a session more than one
// person talks to: the model is told who said it, and the transcript's
// origins keep it. An empty from is Send.
func (s *Session) SendAs(ctx context.Context, from, prompt string) (Event, error) {
	if err := CheckFrom(from); err != nil {
		return Event{}, err
	}
	if err := s.open(); err != nil {
		return Event{}, err
	}
	payload, _ := json.Marshal(MessagePayload{Text: prompt})
	e, _, err := s.m.enqueue(ctx, Event{Session: s.id, Kind: KindMessage, Source: SourceAPI, From: from, Payload: payload})
	if err == nil {
		s.touch()
	}
	return e, err
}

// ApprovalPayload is the payload of a [KindApproval] event.
type ApprovalPayload struct {
	UseID llm.ToolUseID `json:"use_id"`
	Allow bool          `json:"allow"`
}

// Resolve decides a call the session is waiting on: allow runs it, and a
// refusal is what the model is told instead. The decision is an event like
// any other; once every waiting call has one, the turn resumes.
//
// It reports [ErrNoSuchApproval] for a call that is not waiting, and
// [ErrAlreadyAnswered] for one that has been decided.
func (s *Session) Resolve(ctx context.Context, useID string, allow bool) (Event, error) {
	return s.ResolveAs(ctx, "", useID, allow)
}

// ResolveAs is [Session.Resolve] by someone named: who decided is kept with
// the call's result in the transcript's origins. An empty from is Resolve.
func (s *Session) ResolveAs(ctx context.Context, from, useID string, allow bool) (Event, error) {
	if err := CheckFrom(from); err != nil {
		return Event{}, err
	}
	if err := s.open(); err != nil {
		return Event{}, err
	}
	s.mu.Lock()
	waiting := s.waitingOnLocked(llm.ToolUseID(useID))
	_, decided := s.verdicts[llm.ToolUseID(useID)]
	s.mu.Unlock()
	switch {
	case !waiting:
		return Event{}, fmt.Errorf("%w: %s", ErrNoSuchApproval, useID)
	case decided:
		return Event{}, fmt.Errorf("%w: %s", ErrAlreadyAnswered, useID)
	}
	payload, _ := json.Marshal(ApprovalPayload{UseID: llm.ToolUseID(useID), Allow: allow})
	e, created, err := s.m.enqueue(ctx, Event{
		Session: s.id, Kind: KindApproval, Source: SourceAPI, From: from, Payload: payload,
		// One decision per call, however many times it is posted.
		DedupKey: "approval:" + s.id + ":" + useID,
	})
	if err != nil {
		return Event{}, err
	}
	if !created {
		return Event{}, fmt.Errorf("%w: %s", ErrAlreadyAnswered, useID)
	}
	s.touch()
	return e, nil
}

// Pending lists the calls waiting for a decision that has not been made.
func (s *Session) Pending() []Approval {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.deferred == nil {
		return nil
	}
	var out []Approval
	for _, c := range s.deferred.Calls {
		if _, decided := s.verdicts[c.UseID]; !decided {
			out = append(out, approvalOf(c, s.deferredAt))
		}
	}
	return out
}

// SetPermission changes the policy the session's tool calls are checked
// against, for every turn that starts from now on: the agent is built again
// with it — the server may refuse the mode, or clamp it — and kept with the
// session. A turn already running carries on under the policy it began with.
//
// [PermissionFull] asks nobody, so switching to it also allows every call the
// session is waiting on, and those the running turn stops on, if it does.
// The decisions go through the inbox like anyone's, from nobody.
func (s *Session) SetPermission(ctx context.Context, mode string) error {
	if err := s.open(); err != nil {
		return err
	}
	s.mu.Lock()
	opts := s.opts
	s.mu.Unlock()
	opts.Permission = mode
	agent, err := s.m.cfg.Build(opts)
	if err != nil {
		return fmt.Errorf("daemon: building the agent for this session: %w", err)
	}

	s.writing.Lock()
	s.mu.Lock()
	if s.dropped {
		s.mu.Unlock()
		s.writing.Unlock()
		return fmt.Errorf("%w: %s", ErrNoSuchSession, s.id)
	}
	s.opts, s.agent, s.updated = opts, agent, time.Now()
	// A running turn writes the record when it ends, and end writes it once
	// more if the options changed after its snapshot: a write from here could
	// land between the two and undo the first.
	running := s.run != nil
	rec := s.recordLocked()
	s.mu.Unlock()
	if !running {
		s.m.save(rec)
	}
	s.writing.Unlock()
	s.m.logger().Info("session permission changed", "session", s.id, "permission", mode)

	if mode == PermissionFull {
		s.allowPending(ctx)
	}
	return nil
}

// allowPending allows every call the session is waiting on that has no
// decision yet. One decided meanwhile, or a wait that ended, is not an error:
// the call is not waiting any more, which is all this wants.
func (s *Session) allowPending(ctx context.Context) {
	for _, a := range s.Pending() {
		if _, err := s.Resolve(ctx, a.UseID, true); err != nil &&
			!errors.Is(err, ErrAlreadyAnswered) && !errors.Is(err, ErrNoSuchApproval) {
			s.m.logger().Warn("daemon: allowing a call under full access failed", "session", s.id, "use_id", a.UseID, "err", err)
		}
	}
}

func approvalOf(c core.DeferredCall, since time.Time) Approval {
	return Approval{UseID: string(c.UseID), Tool: c.Tool, Reason: c.Reason, Input: jsonOrNil(c.Input), Since: since}
}

func (s *Session) waitingOnLocked(id llm.ToolUseID) bool {
	return s.deferred != nil && slices.ContainsFunc(s.deferred.Calls, func(c core.DeferredCall) bool { return c.UseID == id })
}

// Steer passes prompt to the turn in progress; see core.Run.Steer. It is
// echoed into the session's log, like a prompt, so a second client watching
// the session sees what was said.
func (s *Session) Steer(prompt string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dropped {
		return ErrNoSuchSession
	}
	if s.run == nil || !s.run.Steer(prompt) {
		return ErrNotRunning
	}
	s.emitLocked(s.turns, SteerSent{Turn: s.turns, Prompt: prompt})
	return nil
}

// Cancel stops the turn in flight — the model call, and any subprocess a tool
// started. A no-op between turns. The turn fails as cancelled, and is not
// tried again: its events are settled, and the session takes the next.
func (s *Session) Cancel() {
	s.mu.Lock()
	cancel := s.cancel
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// open reports whether the session can take an event. One that was only
// unloaded can: the event goes to the store, and the session is loaded again
// to run it.
func (s *Session) open() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case s.dropped:
		return fmt.Errorf("%w: %s", ErrNoSuchSession, s.id)
	case s.state == Done:
		return ErrDone
	}
	return nil
}

// ===== A turn =====

// handle runs one claimed batch of the session's events.
func (s *Session) handle(batch []Event) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		s.m.settle(Settlement{Retry: eventIDs(batch), Error: "the session left memory before its turn began"})
		return
	}
	s.wg.Add(1)
	defer s.wg.Done()

	var decisions, rest []Event
	for _, e := range batch {
		if e.Kind == KindApproval {
			decisions = append(decisions, e)
		} else {
			rest = append(rest, e)
		}
	}

	if s.state == Done {
		s.mu.Unlock()
		s.m.settle(Settlement{Failed: eventIDs(batch), Error: ErrDone.Error()})
		return
	}

	var in core.Input
	var prompt string
	var replyTo []int64
	prov := provenance{at: -1}
	handled := batch
	if s.deferred != nil {
		for _, e := range decisions {
			var v ApprovalPayload
			if json.Unmarshal(e.Payload, &v) == nil && s.waitingOnLocked(v.UseID) {
				if s.verdicts == nil {
					s.verdicts = map[llm.ToolUseID]bool{}
				}
				s.verdicts[v.UseID] = v.Allow
				// Once, though a resumed turn that is tried again sees its
				// decisions again.
				if !slices.ContainsFunc(s.decidedBy, func(o Origin) bool { return o.Event == e.ID }) {
					s.decidedBy = append(s.decidedBy, OriginOf(e))
				}
			}
		}
		if len(rest) > 0 {
			// A store hands a session waiting on a decision nothing else; if
			// one did, what it handed over waits for the turn after this.
			handled = decisions
			defer s.m.settle(Settlement{Retry: eventIDs(rest), Error: "held back: the session was waiting on a decision"})
		}
		if !s.decidedLocked() {
			s.updated = time.Now()
			s.mu.Unlock()
			// Snapshotted once the write is ours, so a permission changed in
			// between is in it rather than written over.
			s.writing.Lock()
			s.settleOrForget(Settlement{Session: s.record(), Done: eventIDs(decisions)})
			s.writing.Unlock()
			return
		}
		in = core.Decide(s.msgs, *s.deferred, maps.Clone(s.verdicts))
		replyTo = s.replyTo
		// The resumed run answers every call of the turn it stopped on in one
		// message, put right after the transcript it stopped with.
		prov = provenance{at: len(s.msgs), origins: slices.Clone(s.decidedBy)}
	} else {
		if len(rest) == 0 {
			// Decisions on calls nobody is waiting on any more: a race with
			// the turn that settled them, and nothing to do.
			s.mu.Unlock()
			s.m.settle(Settlement{Done: eventIDs(decisions)})
			return
		}
		prompt = render(rest, s.pauseID != "")
		switch {
		case len(s.msgs) == 0:
			in = core.Ask(prompt)
		case s.pauseID != "":
			// The model asked something, and this is what came back: its
			// tool call gets it, and the siblings it made alongside are closed.
			in = core.AnswerPause(s.msgs, s.pauseID, prompt)
		default:
			// Then closes a dangling tool_use first — a cancelled turn leaves
			// one, and a provider rejects a transcript that still has it.
			in = core.Then(s.msgs, prompt)
		}
		// Each of the three puts what came in at the end of the transcript.
		prov = provenance{at: len(in.Messages) - 1, origins: originsOf(rest)}
		for _, e := range rest {
			if e.Kind == KindMessage {
				replyTo = append(replyTo, e.ID)
			}
		}
	}

	// Background, not a request's context: the turn is no client's. The
	// budget is per turn, so a long conversation is capped per exchange.
	ctx, cancel := governor.WithBudget(context.Background(), s.m.cfg.Limits)
	// One grant set per turn: an approval is about the work in front of the
	// person now, not a standing licence for every later exchange.
	ctx = permission.WithGrants(ctx, permission.NewGrants())
	ctx = withScope(ctx, &scope{m: s.m, session: s.id, thread: s.thread})
	// For the MCP servers that act for a session's owner (Server.Session).
	ctx = mcp.WithSession(ctx, s.id)

	// Under the lock, and deliberately: Start and Buffer only spawn
	// goroutines, and doing them here is what keeps Cancel and Steer from
	// seeing a turn half begun.
	buf := core.Buffer(s.agent.Start(ctx, in))
	s.turns++
	turn := s.turns
	s.state, s.run, s.cancel = Running, buf, cancel
	s.outcome, s.answer, s.errText, s.errKind = "", "", "", ""
	ids := eventIDs(handled)
	s.emitLocked(turn, TurnStarted{Turn: turn, Prompt: prompt, Events: ids, Origins: prov.origins})
	// Saved as running, so a daemon that dies mid-turn is found out when the
	// session is next loaded; the transcript is the one the turn started
	// from, and the events that make it are in the inbox.
	rec := s.recordLocked()
	s.mu.Unlock()
	// After any write from outside already under way: it snapshotted before
	// this turn began, so it is the older of the two.
	s.writing.Lock()
	s.m.save(rec)
	s.writing.Unlock()

	// context.Background, not ctx: the log has to get every event, the ones
	// saying why a cancelled turn stopped included. Follow returns on its own
	// once the run is over, which cancelling guarantees.
	for _, ev := range buf.Follow(context.Background(), 0) {
		if c, ok := ev.(core.TranscriptCompacted); ok {
			prov.folds = append(prov.folds, c.Folded)
		}
		s.append(turn, ev)
	}
	s.end(ctx, cancel, turn, buf, handled, replyTo, prov)
}

// end records how a turn ended, says so in the log and the outbox, and
// settles its events.
func (s *Session) end(ctx context.Context, cancel context.CancelFunc, turn int, buf *core.Buffered, events []Event, replyTo []int64, prov provenance) {
	defer cancel()
	now := time.Now()
	end := TurnEnded{Turn: turn, Events: eventIDs(events), Spend: turnSpend(ctx)}
	var post []Message
	reply := func(kind, text string) {
		post = append(post, Message{Session: s.id, Thread: s.thread, Kind: kind, Text: text, InReplyTo: replyTo, Created: now})
	}
	var pauseID llm.ToolUseID
	var deferred *core.Deferred
	switch o := buf.Outcome().(type) {
	case core.Answer:
		// Idle, not Done. A conversation is not over because the model
		// stopped talking: an answer is the ordinary end of a TURN.
		end.State, end.Outcome, end.Answer = Idle, "answer", o.Text
		if len(replyTo) > 0 {
			reply(MessageReply, o.Text)
		}
	case core.Paused:
		end.State, end.Outcome, end.Question = Waiting, "paused", o.Schema.Question
		pauseID = o.ToolUseID
		reply(MessageQuestion, o.Schema.Question)
	case core.Submitted:
		// Done, and this is the only thing that produces it: a terminal tool
		// is the model saying the TASK is finished, not the turn.
		end.State, end.Outcome, end.Tool, end.Answer = Done, "submitted", o.Tool, string(o.Payload)
		if len(replyTo) > 0 {
			reply(MessageReply, end.Answer)
		}
	case core.Deferred:
		end.State, end.Outcome = Approving, "deferred"
		deferred = &o
		for _, c := range o.Calls {
			end.Approvals = append(end.Approvals, approvalOf(c, now))
		}
		data, _ := json.Marshal(struct {
			Approvals []Approval `json:"approvals"`
		}{end.Approvals})
		// InReplyTo, like a reply: the calls are part of answering those
		// messages, and a client showing the conversation already shows them.
		post = append(post, Message{Session: s.id, Thread: s.thread, Kind: MessageApproval,
			Text: approvalText(end.Approvals), Data: data, InReplyTo: replyTo, Created: now})
	default:
		err := buf.Err()
		end.State, end.Error, end.ErrorKind = Failed, errText(err), ErrorKind(err)
	}

	s.mu.Lock()
	if end.State == Failed && s.closed && !s.dropped {
		// Not deleted, so the daemon is stopping: the turn was cut short by
		// that, not by anything about the turn.
		end.Error, end.ErrorKind = "the daemon stopped while this turn was running", KindInterrupted
	}
	var retry, failed []int64
	if end.State == Failed {
		retry, failed = s.m.retryable(end.ErrorKind == KindInterrupted || llm.Retryable(buf.Err()), events)
		switch {
		case len(retry) > 0 && end.ErrorKind == KindInterrupted:
			// Nothing about the turn failed: the next daemon takes it up at
			// once.
			end.RetryAt = now
		case len(retry) > 0:
			end.RetryAt = now.Add(s.m.backoff(maxAttempts(events)))
		case len(replyTo) > 0:
			reply(MessageError, end.Error)
		}
	}
	// The record the turn leaves, built before anything about the session
	// changes in memory: it is settled first, and only then does the session
	// say the turn has ended — so a client that reads the inbox when it sees
	// turn_ended finds the events settled. Its Events counts the turn_ended
	// frame, which takes the next number.
	next := s.recordLocked()
	if len(retry) == 0 {
		// Keep the transcript even on failure: the user watched half an
		// answer arrive, and "it broke, carry on" is the most ordinary thing
		// a conversation does. A turn that will run again starts over from
		// the transcript it started from, so it is not there twice.
		next.Messages = buf.Messages()
		next.Origins = prov.apply(s.origins, next.Messages)
		next.PauseID, next.Deferred, next.Verdicts, next.DeferredAt, next.ReplyTo = pauseID, deferred, nil, time.Time{}, nil
		next.DecidedBy = nil
		if deferred != nil {
			next.Verdicts, next.DeferredAt, next.ReplyTo = map[llm.ToolUseID]bool{}, now, replyTo
		}
	}
	next.Spend.add(end.Spend)
	next.State = end.State
	next.Outcome, next.Answer = end.Outcome, end.Answer
	if end.Question != "" {
		next.Answer = end.Question
	}
	next.Error, next.ErrorKind = end.Error, end.ErrorKind
	next.Updated = now
	next.Events++
	s.mu.Unlock()

	set := Settlement{Session: next, Outbox: post}
	switch {
	case len(retry) > 0:
		set.Retry, set.RetryAt, set.Failed, set.Error = retry, end.RetryAt, failed, end.Error
	case end.State == Failed:
		set.Failed, set.Error = eventIDs(events), end.Error
	default:
		set.Done = eventIDs(events)
	}
	s.settleOrForget(set)

	s.mu.Lock()
	s.msgs, s.origins, s.spend, s.state = next.Messages, next.Origins, next.Spend, next.State
	s.outcome, s.answer, s.errText, s.errKind = next.Outcome, next.Answer, next.Error, next.ErrorKind
	s.pauseID, s.deferred, s.verdicts, s.deferredAt, s.replyTo = next.PauseID, next.Deferred, next.Verdicts, next.DeferredAt, next.ReplyTo
	s.decidedBy = next.DecidedBy
	s.run, s.cancel = nil, nil
	// What was sent with Steer and never reached the model is handed back,
	// so a client can send it as a message instead of losing it.
	end.Undelivered = buf.Undelivered()
	s.emitLocked(turn, end)
	// The permission may have changed while the turn ran, after its record
	// was snapshotted; and a session switched to full access meanwhile asks
	// nobody, the calls this turn stopped on included.
	changed := s.opts != next.Options
	allow := deferred != nil && s.opts.Permission == PermissionFull
	s.mu.Unlock()
	s.m.logger().Info("turn ended", "session", s.id, "turn", turn, "events", len(events),
		"state", string(end.State), "cost_usd", end.Spend.CostUSD, "retry", len(retry) > 0)
	if changed {
		s.writing.Lock()
		s.m.save(s.record())
		s.writing.Unlock()
	}
	if allow {
		s.allowPending(context.Background())
	}
}

// settleOrForget settles, and drops the session from memory if the store
// says it is gone: deleted while its turn ran.
func (s *Session) settleOrForget(set Settlement) {
	// Deleted, or taken over by another daemon after this one's lease ran
	// out: either way it is not this daemon's any more.
	if err := s.m.settle(set); errors.Is(err, ErrNoSuchSession) || errors.Is(err, ErrOwned) {
		s.m.forget(s)
		// Not waited for: close waits for the turn, and this is the turn.
		go s.close()
	}
}

// retryable splits a failed turn's events into those to try again and those
// to give up on. Only a failure another try might not repeat is worth one:
// the daemon stopping, or the provider failing in a way it says is passing —
// rate-limited, overloaded, a 5xx, the network. The provider's client has
// retried those already, so this is the longer wait, for an outage. A turn
// out of budget, out of iterations, cancelled or refused by the provider
// for what it asked would only fail again, and cost again doing it.
func (m *Manager) retryable(worth bool, events []Event) (retry, failed []int64) {
	if !worth {
		return nil, eventIDs(events)
	}
	for _, e := range events {
		if e.Attempts < m.cfg.Attempts {
			retry = append(retry, e.ID)
		} else {
			failed = append(failed, e.ID)
		}
	}
	return retry, failed
}

func maxAttempts(events []Event) int {
	n := 0
	for _, e := range events {
		n = max(n, e.Attempts)
	}
	return n
}

func (s *Session) decidedLocked() bool {
	for _, c := range s.deferred.Calls {
		if _, ok := s.verdicts[c.UseID]; !ok {
			return false
		}
	}
	return true
}

// approvalText is an approval message for a person to read.
func approvalText(approvals []Approval) string {
	var b strings.Builder
	if len(approvals) == 1 {
		b.WriteString("Waiting for your decision on a tool call:\n")
	} else {
		fmt.Fprintf(&b, "Waiting for your decision on %d tool calls:\n", len(approvals))
	}
	for _, a := range approvals {
		fmt.Fprintf(&b, "\n- %s %s (use_id %s)", a.Tool, a.Input, a.UseID)
		if a.Reason != "" {
			fmt.Fprintf(&b, ": %s", a.Reason)
		}
	}
	return b.String()
}

// ===== The log =====

// Follow yields the session's log from sequence from, blocking for frames that
// do not exist yet.
//
// It returns on exactly three things, and NOT at the end of a turn:
//
//   - ctx is done — the client went away;
//   - the session left memory: deleted, or the daemon is stopping;
//   - the session reached [Done], the one terminal state.
//
// Not returning between turns is the point: one EventSource spans the whole
// conversation, timers and all, and a client does not have to notice a turn
// boundary and reconnect through it. A session being followed is never
// dropped from memory for being idle.
//
// Sequence numbers start at 0 and never change. from is clamped: below the
// oldest retained frame it starts at the oldest retained one, which is the
// only way eviction is observable — a consumer that cares compares the first
// sequence it is yielded against the one it asked for.
func (s *Session) Follow(ctx context.Context, from int) iter.Seq2[int, Frame] {
	return func(yield func(int, Frame) bool) {
		s.mu.Lock()
		s.followers++
		s.mu.Unlock()
		defer func() {
			s.mu.Lock()
			s.followers--
			s.updated = time.Now()
			s.mu.Unlock()
		}()

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
			// pair would drop the TurnEnded that says how it finished.
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

// append records one event under the session's global sequence.
func (s *Session) append(turn int, ev core.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.emitLocked(turn, ev)
}

// emitLocked appends a frame. Callers hold s.mu.
func (s *Session) emitLocked(turn int, ev core.Event) {
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
		clear(s.frames[s.bufCap:])
		s.frames = s.frames[:s.bufCap]
		s.base += over
	}
	s.updated = time.Now()
	s.wakeLocked()
}

// wake releases every follower parked on the current generation.
func (s *Session) wake() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.wakeLocked()
}

func (s *Session) wakeLocked() {
	close(s.changed)
	s.changed = make(chan struct{})
}

// touch marks activity, which Idle is measured from.
func (s *Session) touch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.updated = time.Now()
}

// idle reports whether the session has had nothing to do for d: no turn, no
// follower, no activity.
func (s *Session) idle(now time.Time, d time.Duration) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.run == nil && s.followers == 0 && now.Sub(s.updated) >= d
}

// close takes the session out of memory: the turn in flight is cancelled,
// and every follower is released. It waits for the turn to settle, so a
// caller knows nothing of the session is still running when it returns.
func (s *Session) close() {
	s.mu.Lock()
	cancel := s.cancel
	if !s.closed {
		s.closed = true
		s.wakeLocked()
	}
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	s.wg.Wait()
}

// drop closes a session that is being deleted.
func (s *Session) drop() {
	s.mu.Lock()
	s.dropped = true
	s.mu.Unlock()
	s.close()
}

// ===== Plumbing =====

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

func eventIDs(events []Event) []int64 {
	out := make([]int64, len(events))
	for i, e := range events {
		out[i] = e.ID
	}
	return out
}

func errText(err error) string {
	if err == nil {
		return "the turn produced no outcome"
	}
	return err.Error()
}

// jsonOrNil keeps an empty argument list out of a payload: an empty non-nil
// json.RawMessage is not valid JSON and would fail the encode for the whole
// response.
func jsonOrNil(in json.RawMessage) json.RawMessage {
	if len(in) == 0 {
		return nil
	}
	return in
}

// ErrorKind names why a turn failed, in a bounded vocabulary a UI can branch
// on: budget_exhausted, tool_loop, max_iterations, cancelled, context_window,
// denied, timeout or error — and interrupted, for a turn the daemon stopped.
//
// Bounded is the requirement. The error's own text is unbounded and changes
// with every reword, so a front end that matched on it would break silently;
// these tokens are a contract, and a new one is a deliberate change to this
// function rather than an accident of some error message elsewhere.
func ErrorKind(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, governor.ErrBudgetExhausted):
		return "budget_exhausted"
	case errors.Is(err, governor.ErrToolLoop):
		return "tool_loop"
	case errors.Is(err, core.ErrMaxIterations), errors.Is(err, governor.ErrStepLimit):
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
