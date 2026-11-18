package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
)

// Store is everything a daemon keeps: its sessions, and the three tables
// that keep them going with nobody watching — the inbox of events waiting for
// a turn, the timers that will put more there, and the outbox of what the
// agent has said.
//
// Two implementations ship: [MemoryStore], and the Postgres one in
// wombat/pkg/store/postgres. Both pass wombat/pkg/daemon/storetest, which is
// the specification: what a test of a [Manager] proves against the one holds
// for the other.
//
// Any number of Managers may share a store — daemons on as many machines as
// one Postgres serves. What must happen once happens once all the same: each
// session is run by one daemon at a time, the one holding its lease (see
// [Store.Acquire]); a timer fires once; a message is delivered by one
// dispatcher (see [Store.Lead]). Its methods are safe for concurrent use.
// Times are kept to the microsecond, in UTC.
type Store interface {
	// CreateSession records a new session, held by rec.Lease when it names an
	// owner, in its first epoch: 1. A Thread another session already has is
	// [ErrThreadTaken].
	CreateSession(ctx context.Context, rec SessionRecord) error

	// SaveSession replaces a session's record, its Inflight with it. One that
	// is not there — deleted while its turn ran — is [ErrNoSuchSession], and
	// stays deleted. With rec.Lease.Owner set, it is written only while that
	// daemon holds the session — under rec.Lease.Epoch, when that is set — and
	// is an *[OwnedError] otherwise: a daemon whose lease ran out does not
	// write over the one that took the session over.
	SaveSession(ctx context.Context, rec SessionRecord) error

	// LoadSession returns a session's record, its Lease and Inflight
	// included, or [ErrNoSuchSession].
	LoadSession(ctx context.Context, id string) (SessionRecord, error)

	// SessionByThread returns the id of the session tied to an external
	// thread, or [ErrNoSuchSession].
	SessionByThread(ctx context.Context, thread string) (string, error)

	// ListSessions returns every session, oldest first, without its Messages
	// but with its Lease.
	ListSessions(ctx context.Context) ([]SessionRecord, error)

	// DeleteSession forgets a session, and its events, timers, messages and
	// ledger with it. One that is not there is [ErrNoSuchSession].
	DeleteSession(ctx context.Context, id string) error

	// Enqueue adds an event to its session's inbox, as pending. An event with
	// the Source and DedupKey of one already there is not added again: that
	// one is returned instead, with false. The session must exist.
	Enqueue(ctx context.Context, e Event) (Event, bool, error)

	// Ready lists up to limit sessions with events to claim at now that owner
	// holds or could take — with no owner, or one whose lease ran out — the
	// one that has waited longest first, leaving out those in busy. An event
	// is there to claim when it is pending and not held back past now; a
	// session waiting on a decision takes nothing but approvals. A session
	// whose owner's lease ran out with events still running counts too:
	// taking it puts them back (see [Store.Acquire]).
	Ready(ctx context.Context, now time.Time, owner string, busy []string, limit int) ([]string, error)

	// Claim takes up to limit of the events Ready counts for the session, in
	// the order they arrived, if owner holds it — none otherwise: each becomes
	// running, and has one more attempt.
	Claim(ctx context.Context, session, owner string, now time.Time, limit int) ([]Event, error)

	// Settle records what a turn did, all at once or not at all. A Session
	// with Lease.Owner set is written as SaveSession writes it, and a refusal
	// leaves everything as it was.
	Settle(ctx context.Context, s Settlement) error

	// ===== Who runs a session =====
	//
	// A session's turns are run by one daemon at a time: the one holding its
	// lease, its owner — the only one that claims its events, writes its
	// record and keeps it in memory. The owner renews the lease while it has
	// the session; a lease that runs out with nobody renewing it is a daemon
	// that died, and any other may take the session over.

	// Acquire makes l.Owner the owner of a session until l.Until, if nobody
	// else holds a live lease on it at now: it has no owner, l.Owner holds it
	// already, or its owner's lease ran out. Taking it from another daemon puts
	// the events that daemon left running back in the inbox — failing those
	// tried attempts times already — and reports how many it put back. A
	// session another daemon holds is an *[OwnedError] naming it.
	Acquire(ctx context.Context, session string, l Lease, now time.Time, attempts int) (int, error)

	// Renew extends owner's lease on those of sessions it still holds to
	// until, and returns them: one missing from the answer was deleted, or
	// taken over after the lease ran out.
	Renew(ctx context.Context, owner string, sessions []string, until time.Time) ([]string, error)

	// Release gives up owner's leases on sessions.
	Release(ctx context.Context, owner string, sessions []string) error

	// Recover is for a daemon starting again under the name it had: the turns
	// its sessions were running died with it. It puts their events back in the
	// inbox — failing those tried attempts times already — gives the sessions
	// up, and reports how many events it put back. No other daemon's sessions
	// are touched.
	Recover(ctx context.Context, owner string, attempts int) (int, error)

	// Lead reports whether owner leads role: "deliver", the outbox's
	// dispatcher, which has to be one daemon so that a message is sent once and
	// in order. The first to ask leads, and asking again keeps it leading, at
	// least until until; then — or once its store is closed, or its process
	// dies — the next to ask does.
	Lead(ctx context.Context, role, owner string, until time.Time) (bool, error)

	// Events lists inbox events, newest first.
	Events(ctx context.Context, f EventFilter) ([]Event, error)

	// AddTimer adds an active timer to its session, which must exist. A
	// DedupKey one already there has is not added again: that one is
	// returned instead.
	AddTimer(ctx context.Context, t Timer) (Timer, error)

	// CancelTimer cancels an active timer — of that session, when session is
	// not empty. Anything else is [ErrNoSuchTimer].
	CancelTimer(ctx context.Context, id int64, session string) error

	// Timers lists timers, the soonest due first.
	Timers(ctx context.Context, f TimerFilter) ([]Timer, error)

	// Fire enqueues an event for every active timer due by now and moves the
	// timer on — a one-shot one has fired, a repeating one is due again at its
	// first period after now — and returns the events. All or nothing.
	Fire(ctx context.Context, now time.Time) ([]Event, error)

	// Post adds a message to the outbox, as pending. A DedupKey one already
	// there has is not added again: that one is returned instead, with false.
	Post(ctx context.Context, m Message) (Message, bool, error)

	// Outbox lists messages, oldest first.
	Outbox(ctx context.Context, f OutboxFilter) ([]Message, error)

	// Delivered marks a message delivered at at. Doing it twice is not an
	// error; a message that is not there is [ErrNoSuchMessage].
	Delivered(ctx context.Context, id int64, at time.Time) error

	// Undelivered records a failed delivery: the message is tried again no
	// sooner than retry, or — when retry is zero — never, and is failed.
	Undelivered(ctx context.Context, id int64, reason string, retry time.Time) error

	// ===== The ledger =====
	//
	// Every tool call a session's turns make, from the moment the model asks
	// for it until it has its result: see [ToolCall]. A turn writes it as it
	// goes, so a daemon that takes the turn over carries it on from there.

	// OpenStep numbers a model reply — the session's next step — and writes
	// it down before anything it asks for happens: its calls, pending, and
	// where the run had got to, all at once, and only while o.Owner holds the
	// session under o.Epoch (an *[OwnedError] otherwise). It returns the
	// step's number.
	OpenStep(ctx context.Context, o StepOpen) (int, error)

	// MoveCalls records calls of one step changing state, all at once, and
	// only while m.Owner holds the session under m.Epoch. A call that has
	// ended moves only to another ending — the more exact one its step
	// closes with — and a call it names that is not there is an error.
	MoveCalls(ctx context.Context, m CallMoves) error

	// ToolCalls lists the ledger, the newest call first.
	ToolCalls(ctx context.Context, f CallFilter) ([]ToolCall, error)

	// SubRun returns where the n-th sub-agent a call started had got to, by
	// the call's key, or false when it had not got anywhere.
	SubRun(ctx context.Context, session, parent string, n int) (SubRun, bool, error)

	// PruneCalls forgets the calls of sessions with no turn in flight and
	// nothing waiting that ended before before — or, never having ended,
	// were made before it: nothing will carry them on — and reports how many
	// it forgot. Of those sessions it forgets as well where the sub-agents of
	// calls that are gone or have ended had got to.
	PruneCalls(ctx context.Context, before time.Time) (int, error)

	// Close releases what the store holds. The Manager does not call it: the
	// store belongs to whoever opened it.
	Close() error
}

// Settlement is what one turn changed: its session, the events it was given,
// and what it said. [Store.Settle] writes it in one transaction, so a turn
// whose session was saved has had its events settled too, and a crash between
// the two cannot run them again on a transcript that already holds them.
type Settlement struct {
	// Session is saved when its ID is set.
	Session SessionRecord

	// Done are the events the turn handled.
	Done []int64

	// Retry are events to try again, no sooner than RetryAt.
	Retry   []int64
	RetryAt time.Time

	// Failed are events given up on.
	Failed []int64

	// Error says why the Retry and Failed events did not get done.
	Error string

	// Outbox is what the turn has to say.
	Outbox []Message
}

// Store errors.
var (
	// ErrThreadTaken reports a new session for a thread that has one.
	ErrThreadTaken = errors.New("daemon: that thread has a session already")

	// ErrNoSuchTimer reports a timer id that names no active timer.
	ErrNoSuchTimer = errors.New("daemon: no such timer")

	// ErrNoSuchMessage reports an outbox id that names nothing.
	ErrNoSuchMessage = errors.New("daemon: no such message")

	// ErrOwned reports a session another daemon holds; see [OwnedError].
	ErrOwned = errors.New("daemon: another daemon runs that session")
)

// Lease is a daemon's hold on a session; see [Store.Acquire].
type Lease struct {
	// Owner names the daemon: unique among those sharing a store.
	Owner string `json:"owner,omitempty"`

	// Addr is where the daemon answers HTTP, for another to forward the
	// session's requests to. Optional.
	Addr string `json:"addr,omitempty"`

	Until time.Time `json:"until,omitzero"`

	// Epoch counts the times the session has changed hands: [Store.Acquire]
	// moves it on whenever a daemon takes the session from another or from
	// nobody. A write that names one is made only under it, so a daemon that
	// let a session go and took it back cannot have a write from before land
	// on what it does now. Zero names none.
	Epoch int64 `json:"epoch,omitempty"`
}

// OwnedError is [ErrOwned]: the session, and the lease another daemon holds
// on it — whose Addr a server can forward the request to.
type OwnedError struct {
	Session string
	Lease   Lease
}

func (e *OwnedError) Error() string {
	return fmt.Sprintf("daemon: session %s runs on daemon %q", e.Session, e.Lease.Owner)
}

func (e *OwnedError) Unwrap() error { return ErrOwned }

// RecordVersion is the newest layout of [SessionRecord] this package writes,
// and the newest it reads.
//
// A change that a reader of the old layout would misread — a field renamed,
// retyped or given a new meaning — bumps it. A record from a newer layout is
// then left alone rather than restored wrong: a server rolled back past a
// format change keeps the sessions the newer one saved, and gets them back
// when it is rolled forward again. Adding a field a reader may ignore is not
// such a change.
//
// Version 2 is a session waiting on a call that awaits a result: a reader of
// version 1 would take the call for one waiting on a decision, and a yes
// would run it again. Any other record is still written as version 1, which
// is what it is: an older daemon, which knows nothing of the ledger, reads it
// right, and only starts a turn in flight over rather than carrying it on.
const RecordVersion = 2

// layoutOf is the version a record is written in: the oldest that holds it.
func layoutOf(rec SessionRecord) int {
	if rec.Deferred != nil && len(rec.Deferred.Awaited()) > 0 {
		return 2
	}
	return 1
}

// Posted is a result posted for a call that awaits one.
type Posted struct {
	Output  string `json:"output"`
	IsError bool   `json:"is_error,omitempty"`
}

// ErrNewerRecord reports a saved session written in a newer layout than
// [RecordVersion].
var ErrNewerRecord = errors.New("daemon: session saved by a newer version")

// SessionRecord is what a [Store] keeps of one session: the conversation, and
// where it stands.
//
// What is not in it is what only exists while a process runs: the event log a
// stream resumes from. A turn in flight is kept beside it, as its journal
// last wrote it down (Inflight): a turn a crash cut short is carried on from
// there, and its events — still in the inbox — run it.
type SessionRecord struct {
	// Version is the layout the record was written in; see [RecordVersion].
	Version int `json:"version"`

	ID string `json:"id"`

	// Thread ties the session to a conversation somewhere else — an issue, a
	// chat thread — so the events about it find it. Optional, and unique.
	Thread string `json:"thread,omitempty"`

	Options SessionOptions `json:"options"`

	// Lease is who runs the session, as the store has it, beside the record
	// rather than in it: LoadSession and ListSessions fill it in,
	// CreateSession sets it, and SaveSession and Settle write a record only
	// while its Lease.Owner, when set, holds the session.
	Lease Lease `json:"-"`

	Created  time.Time     `json:"created"`
	Updated  time.Time     `json:"updated"`
	Turns    int           `json:"turns"`
	State    State         `json:"state"`
	Messages []llm.Message `json:"messages"`
	Spend    Spend         `json:"spend"`

	// Origins are where messages of the transcript came from, keyed by their
	// index in Messages: for each message a turn's inbox events made — the
	// one a turn starts with, or the one a decision resumes it with — the
	// events. Other messages have none: the model's, tools' results, and what
	// was steered into a running turn, which is not an event. A compaction
	// renumbers them with the transcript: the messages it folds away take
	// their origins with them, and the head it folds them into keeps the
	// first message's.
	Origins map[int][]Origin `json:"origins,omitempty"`

	Outcome   string        `json:"outcome,omitempty"`
	Answer    string        `json:"answer,omitempty"`
	Error     string        `json:"error,omitempty"`
	ErrorKind string        `json:"error_kind,omitempty"`
	PauseID   llm.ToolUseID `json:"pause_id,omitempty"`

	// Deferred is the turn the session stopped on to wait for a person, and
	// Verdicts what has been decided about its calls so far. A session with
	// Deferred set takes nothing but decisions until they are all in.
	Deferred   *core.Deferred         `json:"deferred,omitempty"`
	Verdicts   map[llm.ToolUseID]bool `json:"verdicts,omitempty"`
	DeferredAt time.Time              `json:"deferred_at,omitzero"`

	// DecidedBy are the decisions behind Verdicts, as origins: they arrive
	// one at a time, and become the origins of the message the resumed turn
	// answers the calls in. The results posted for calls that await one are
	// here too.
	DecidedBy []Origin `json:"decided_by,omitempty"`

	// DeferredStep is the step of the ledger Deferred is on: the model reply
	// the transcript ends with, whose calls the ledger has.
	DeferredStep int `json:"deferred_step,omitempty"`

	// Results are what has been posted so far for the calls of Deferred that
	// await a result ([tool.Await]), by tool_use id.
	Results map[llm.ToolUseID]Posted `json:"results,omitempty"`

	// Inflight is the turn in flight, as its journal last wrote it down, for
	// a daemon that takes the session over to carry on: nil between turns.
	// Like Lease, it is kept beside the record rather than in it; SaveSession
	// and Settle write it with the record, and OpenStep on its own.
	Inflight *Inflight `json:"-"`

	// ReplyTo are the messages a stopped turn has not answered yet: when the
	// turn resumes and answers, that answer is their reply.
	ReplyTo []int64 `json:"reply_to,omitempty"`

	// Events is the next sequence number the session's log would have
	// issued. A restored log starts there, so a client resuming with a
	// Last-Event-ID from before the restart is told it missed events rather
	// than being handed new ones under old numbers.
	Events int `json:"events"`
}

// ===== The inbox =====

// Event is one thing that happened, waiting in a session's inbox for the turn
// that will see it.
type Event struct {
	ID      int64  `json:"id"`
	Session string `json:"session_id"`

	// Kind is what happened: [KindMessage], [KindApproval], [KindTimer], or
	// anything a client names.
	Kind string `json:"kind"`

	// Source is who said so: [SourceAPI], [SourceTimer], or a client's name.
	Source string `json:"source"`

	// From is who, at Source, sent it, when the sender says: a person's name
	// or handle in a chat, the account that decided a call. Optional. The
	// model is told it, and so is whoever reads the transcript; see [Origin].
	From string `json:"from,omitempty"`

	// DedupKey makes posting the same event twice — a webhook delivered
	// again — post it once. Unique per Source; optional.
	DedupKey string `json:"dedup_key,omitempty"`

	Payload json.RawMessage `json:"payload,omitempty"`

	Status   EventStatus `json:"status"`
	Attempts int         `json:"attempts"`
	Error    string      `json:"error,omitempty"`

	Created time.Time `json:"created"`

	// NotBefore holds a pending event back until then: an event to retry
	// waits out its backoff.
	NotBefore time.Time `json:"not_before"`
}

// EventStatus is where an event is.
type EventStatus string

// The statuses an event goes through: pending until a turn claims it,
// running while the turn does, then done or failed.
const (
	EventPending EventStatus = "pending"
	EventRunning EventStatus = "running"
	EventDone    EventStatus = "done"
	EventFailed  EventStatus = "failed"
)

// The kinds of event this package makes, and gives a meaning to.
const (
	// KindMessage is a person talking to the session. Its payload is
	// {"text": "..."}, and a turn that sees one replies to it in the outbox.
	KindMessage = "message"

	// KindApproval is a decision on a call the session is waiting on:
	// {"use_id": "...", "allow": true}.
	KindApproval = "approval"

	// KindTimer is a timer that fired: {"timer": 7, "note": "...", ...}.
	KindTimer = "timer"

	// KindResult is the result of a call the session is waiting on, which
	// started something that finishes outside it ([tool.Await]):
	// {"use_id": "...", "output": "...", "is_error": false}.
	KindResult = "result"
)

// The sources this package names.
const (
	SourceAPI   = "api"
	SourceTimer = "timer"
)

// Origin is an inbox event a message of the transcript was made from: what it
// was, where it came from, who sent it, and when.
//
// The transcript is exactly what the model was sent, and says where its
// messages came from only as prose the model reads. A session keeps origins
// beside it — see [SessionRecord.Origins] — so a program reading the
// conversation can tell a person's message from a webhook, a timer or a
// decision, and one person from another.
type Origin struct {
	Event  int64     `json:"event"`
	Kind   string    `json:"kind"`
	Source string    `json:"source"`
	From   string    `json:"from,omitempty"`
	At     time.Time `json:"at"`
}

// OriginOf is where e comes from, as a message made from it records it.
func OriginOf(e Event) Origin {
	return Origin{Event: e.ID, Kind: e.Kind, Source: e.Source, From: e.From, At: e.Created}
}

// EventFilter selects inbox events. A zero field selects everything.
type EventFilter struct {
	Session string
	Status  EventStatus
	Limit   int
}

// ===== Timers =====

// Timer wakes a session up later: once, at Due, or at Due and every Every
// after it.
type Timer struct {
	ID      int64  `json:"id"`
	Session string `json:"session_id"`

	// Note is what the session is told when the timer fires: the reason it
	// was set, in the words of whoever set it.
	Note string `json:"note"`

	Due   time.Time     `json:"due"`
	Every time.Duration `json:"every,omitempty"`

	Status TimerStatus `json:"status"`

	// CreatedBy is who set it: "agent", or "api".
	CreatedBy string    `json:"created_by"`
	Created   time.Time `json:"created"`

	// DedupKey makes setting the same timer twice set it once: the agent's
	// set_timer names it after its call, so a call run again sets nothing
	// new. Optional.
	DedupKey string `json:"-"`

	// Fired is how many times it has.
	Fired int `json:"fired"`
}

// MarshalJSON writes Every as a duration a person can read: "24h0m0s".
func (t Timer) MarshalJSON() ([]byte, error) {
	type plain Timer
	var every string
	if t.Every > 0 {
		every = t.Every.String()
	}
	return json.Marshal(struct {
		plain
		Every string `json:"every,omitempty"`
	}{plain(t), every})
}

// UnmarshalJSON reads what MarshalJSON writes.
func (t *Timer) UnmarshalJSON(b []byte) error {
	type plain Timer
	var v struct {
		*plain
		Every string `json:"every,omitempty"`
	}
	v.plain = (*plain)(t)
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	t.Every = 0
	if v.Every != "" {
		d, err := time.ParseDuration(v.Every)
		if err != nil {
			return err
		}
		t.Every = d
	}
	return nil
}

// TimerStatus is where a timer is.
type TimerStatus string

// The statuses a timer goes through.
const (
	TimerActive    TimerStatus = "active"
	TimerFired     TimerStatus = "fired"
	TimerCancelled TimerStatus = "cancelled"
)

// TimerFilter selects timers. A zero field selects everything.
type TimerFilter struct {
	Session string
	Status  TimerStatus
}

// ===== The outbox =====

// Message is something the agent has to say to the outside: a reply, a
// question, a call waiting for a decision, or what it chose to send.
type Message struct {
	ID      int64  `json:"id"`
	Session string `json:"session_id"`

	// Thread is the session's, so a bridge that posts messages to where the
	// conversation is can find it without a lookup.
	Thread string `json:"thread,omitempty"`

	// Kind is what it is: [MessageReply], [MessageQuestion],
	// [MessageApproval], [MessageError] or [MessageSent].
	Kind string `json:"kind"`
	Text string `json:"text"`

	// Data is the structured part, where there is one: the calls a
	// [MessageApproval] is about.
	Data json.RawMessage `json:"data,omitempty"`

	// InReplyTo are the inbox events a reply answers.
	InReplyTo []int64 `json:"in_reply_to,omitempty"`

	// DedupKey makes posting it twice post it once. Optional.
	DedupKey string `json:"-"`

	Status    MessageStatus `json:"status"`
	Attempts  int           `json:"attempts"`
	Error     string        `json:"error,omitempty"`
	Created   time.Time     `json:"created"`
	Delivered time.Time     `json:"delivered,omitzero"`

	// NotBefore holds a failed delivery back until then.
	NotBefore time.Time `json:"not_before,omitzero"`
}

// MessageStatus is where a message is.
type MessageStatus string

// The statuses a message goes through: pending until delivered, or failed
// once the dispatcher gives up on it.
const (
	MessagePending   MessageStatus = "pending"
	MessageDelivered MessageStatus = "delivered"
	MessageFailed    MessageStatus = "failed"
)

// The kinds of message the daemon posts.
const (
	// MessageReply answers the messages a turn was given.
	MessageReply = "reply"

	// MessageQuestion is the model asking the person something; the answer
	// is the next message to the session.
	MessageQuestion = "question"

	// MessageApproval lists calls waiting for a decision; see
	// POST /api/sessions/{id}/approvals/{use_id}.
	MessageApproval = "approval"

	// MessageAwaiting lists calls waiting for a result from outside the
	// session; see POST /api/sessions/{id}/results/{use_id}.
	MessageAwaiting = "awaiting"

	// MessageError is a turn that failed on messages a person sent, and will
	// not be tried again.
	MessageError = "error"

	// MessageSent is what the agent sent with the send_message tool.
	MessageSent = "message"
)

// OutboxFilter selects messages. A zero field selects everything.
type OutboxFilter struct {
	Session string
	Status  MessageStatus

	// After selects messages with a larger id: a reader passes the last one
	// it has.
	After int64
	Limit int
}
