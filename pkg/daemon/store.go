package daemon

import (
	"context"
	"encoding/json"
	"errors"
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
// A Store serves one Manager at a time — [Store.Recover] assumes that nobody
// else is running the events it puts back — and its methods are safe for
// concurrent use. Times are kept to the microsecond, in UTC.
type Store interface {
	// CreateSession records a new session. A Thread another session already
	// has is [ErrThreadTaken].
	CreateSession(ctx context.Context, rec SessionRecord) error

	// SaveSession replaces a session's record. One that is not there —
	// deleted while its turn ran — is [ErrNoSuchSession], and stays deleted.
	SaveSession(ctx context.Context, rec SessionRecord) error

	// LoadSession returns a session's record, or [ErrNoSuchSession].
	LoadSession(ctx context.Context, id string) (SessionRecord, error)

	// SessionByThread returns the id of the session tied to an external
	// thread, or [ErrNoSuchSession].
	SessionByThread(ctx context.Context, thread string) (string, error)

	// ListSessions returns every session, oldest first, without its Messages.
	ListSessions(ctx context.Context) ([]SessionRecord, error)

	// DeleteSession forgets a session, and its events, timers and messages
	// with it. One that is not there is [ErrNoSuchSession].
	DeleteSession(ctx context.Context, id string) error

	// Enqueue adds an event to its session's inbox, as pending. An event with
	// the Source and DedupKey of one already there is not added again: that
	// one is returned instead, with false. The session must exist.
	Enqueue(ctx context.Context, e Event) (Event, bool, error)

	// Ready lists up to limit sessions with events to claim at now, the one
	// that has waited longest first, leaving out those in busy. An event is
	// there to claim when it is pending and not held back past now; a session
	// waiting on a decision takes nothing but approvals.
	Ready(ctx context.Context, now time.Time, busy []string, limit int) ([]string, error)

	// Claim takes up to limit of the events Ready counts for the session, in
	// the order they arrived: each becomes running, and has one more attempt.
	Claim(ctx context.Context, session string, now time.Time, limit int) ([]Event, error)

	// Settle records what a turn did, all at once or not at all.
	Settle(ctx context.Context, s Settlement) error

	// Recover puts every running event back to pending, for a manager
	// starting after one that did not stop cleanly: their turns died with it.
	// One tried attempts times already is failed instead. It reports how many
	// events it put back.
	Recover(ctx context.Context, attempts int) (int, error)

	// Events lists inbox events, newest first.
	Events(ctx context.Context, f EventFilter) ([]Event, error)

	// AddTimer adds an active timer to its session, which must exist.
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
)

// RecordVersion is the layout of [SessionRecord] this package writes, and
// the newest it reads.
//
// A change that a reader of the old layout would misread — a field renamed,
// retyped or given a new meaning — bumps it. A record from a newer layout is
// then left alone rather than restored wrong: a server rolled back past a
// format change keeps the sessions the newer one saved, and gets them back
// when it is rolled forward again. Adding a field a reader may ignore is not
// such a change.
const RecordVersion = 1

// ErrNewerRecord reports a saved session written in a newer layout than
// [RecordVersion].
var ErrNewerRecord = errors.New("daemon: session saved by a newer version")

// SessionRecord is what a [Store] keeps of one session: the conversation, and
// where it stands.
//
// What is not in it is what only exists while a process runs: the event log a
// stream resumes from, and a turn in flight. A turn a crash cut short is not
// lost, though — its events are still in the inbox, and are run again.
type SessionRecord struct {
	// Version is the layout the record was written in; see [RecordVersion].
	Version int `json:"version"`

	ID string `json:"id"`

	// Thread ties the session to a conversation somewhere else — an issue, a
	// chat thread — so the events about it find it. Optional, and unique.
	Thread string `json:"thread,omitempty"`

	Options  SessionOptions `json:"options"`
	Created  time.Time      `json:"created"`
	Updated  time.Time      `json:"updated"`
	Turns    int            `json:"turns"`
	State    State          `json:"state"`
	Messages []llm.Message  `json:"messages"`
	Spend    Spend          `json:"spend"`

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
)

// The sources this package names.
const (
	SourceAPI   = "api"
	SourceTimer = "timer"
)

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
