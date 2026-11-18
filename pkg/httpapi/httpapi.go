// Package httpapi serves an always-on agent over HTTP: its sessions, and the
// inbox, timers and outbox they run on.
//
// The unit of the API is a session, and a session owns a conversation rather
// than a single run. Everything about how one runs — events in, one turn at a
// time, timers, decisions that come back later, what goes out — is
// wombat/pkg/daemon, which is HTTP-free so all of it is tested without a
// socket; [New] is a thin projection of a [daemon.Manager] onto routes:
//
//	GET    /api/health                            {status, version, uptime_sec}
//	GET    /api/config                            what this server offers
//	POST   /api/sessions                          create, with its first message: {prompt, from, ...}
//	GET    /api/sessions                          list, newest first
//	GET    /api/sessions/{id}                     one SessionInfo
//	PATCH  /api/sessions/{id}                     {"permission": "full"|""|...}: from the next turn on
//	DELETE /api/sessions/{id}                     cancel and forget, timers and all
//	GET    /api/sessions/{id}/events              SSE, Last-Event-ID resume
//	GET    /api/sessions/{id}/messages            the transcript, and where its messages came from
//	GET    /api/sessions/{id}/file?path=          a file in its workspace, and a /files/ URL to read it
//	POST   /api/sessions/{id}/messages            a message, for the next turn: {prompt, from}
//	POST   /api/sessions/{id}/steer               a message for the turn in progress
//	GET    /api/sessions/{id}/approvals           what is waiting for a decision, and for a result
//	POST   /api/sessions/{id}/approvals/{use_id}  {"allow": true|false, "from": "..."}
//	POST   /api/sessions/{id}/results/{use_id}    {"output": "...", "is_error": false, "from": "..."}
//	GET    /api/tool-calls                        the ledger, newest first; ?session= ?state= ?turn= ?step= ?parent=
//	POST   /api/events                            an event from outside: a webhook, an alert, a relayed chat
//	GET    /api/inbox                             events, newest first; ?session= ?status=
//	GET    /api/outbox                            messages, oldest first; ?after= ?wait=
//	POST   /api/outbox/{id}/ack                   a message handled
//	GET    /api/timers                            ?session= ?status=
//	POST   /api/timers                            {session_id, note, in|at, every}
//	DELETE /api/timers/{id}                       cancel one
//	GET    /metrics                               with [WithMetrics]
//
// Everything that makes a session run is an event, and is answered with 202:
// a message, a decision, an event from outside. What the agent says back
// arrives on the session's event stream as it happens, and in the outbox,
// which a client long-polls — GET /api/outbox?after=<last id>&wait=30s — and
// acks.
//
// # Several daemons
//
// Daemons sharing one store each run the sessions they hold, and a request
// about a session reaching one that does not hold it is forwarded to the one
// that does, at the address its lease names (daemon.ManagerConfig.Addr). It
// is answered 503 "elsewhere" when it cannot be: no address, or the session
// moved while the request was on its way.
//
// # Who said what
//
// Every event in the inbox says what it is, where it came from and, when the
// sender knows, who sent it: kind, source and from. A message is kind
// "message", with the payload {"text": "..."}. Posted to
// /api/sessions/{id}/messages it is source "api"; a bridge relaying a chat
// posts it to /api/events with a source of its own ("slack") and from naming
// the person. A decision is kind "approval", and its from is who decided.
//
// The model is told all three. A direct message — source "api", no from — on
// its own reads as plain text, the conversation it is. Anything else arrives
// in a tag that names its kind, source and from, so a webhook's payload does
// not pass for an instruction, nor one speaker for another.
//
// The transcript is kept exactly as the model was sent it, so it says where a
// message came from only as prose for the model. GET
// /api/sessions/{id}/messages returns, beside it, origins: for each message
// inbox events made, keyed by its index in messages, those events —
//
//	{"messages": [...],
//	 "origins": {"0": [{"event": 1, "kind": "message", "source": "api", "at": "…"}],
//	             "4": [{"event": 9, "kind": "message", "source": "slack", "from": "alice", "at": "…"}]}}
//
// A message no event made — the model's, a tool's result, a steer — has
// none. A turn's turn_started frame carries the origins of the message it
// starts with, for a client rendering the conversation live.
//
// Sequence numbers on a session's stream are global to the SESSION, not to
// the turn. A client that reconnects across a turn boundary therefore resumes
// into the same numbering it left, instead of silently losing the earlier
// turn to a stream that restarted at zero.
//
// Every error has the same shape, so a client can branch on a bounded token
// instead of parsing prose:
//
//	{"error": {"kind": "no_such_session", "message": "…"}}
//
// A request body must be declared application/json, or it is refused with 415
// unsupported_media_type — a security check, not pedantry; see [New]. Who may
// call at all is [WithBearerToken] and [WithAllowedHosts].
package httpapi

import (
	"encoding/json"

	"wombat/pkg/daemon"
)

// ===== Wire bodies =====

// CreateRequest is the body of POST /api/sessions.
//
// The prompt rides in a JSON body rather than a query string because a prompt
// is prose: it is routinely longer than a URL may safely be, it is logged by
// every proxy in between when it is in the URL, and it is the one field a user
// typed.
//
// [daemon.SessionOptions] is embedded rather than nested so the wire stays flat —
// {"prompt":"…","model":"…"} — which is what a form posts and what a curl by
// hand looks like.
type CreateRequest struct {
	Prompt string `json:"prompt"`

	// From names who is talking, for a session more than one person talks
	// to; see [SendRequest].
	From string `json:"from,omitempty"`

	daemon.SessionOptions
}

// SendRequest is the body of POST /api/sessions/{id}/messages: a message for
// the session's next turn.
type SendRequest struct {
	Prompt string `json:"prompt"`

	// From names who said it, when more than one person talks to the
	// session: the model is told, and the transcript's origins keep it.
	// Optional; a short line of text.
	From string `json:"from,omitempty"`
}

// PatchRequest is the body of PATCH /api/sessions/{id}: what to change about
// a session. Permission is a pointer because "" means something — the
// server's default — and absent means leave it.
type PatchRequest struct {
	Permission *string `json:"permission,omitempty"`
}

// SteerRequest is the body of POST /api/sessions/{id}/steer: a message for
// the turn in progress. It has no From: it goes straight into the running
// turn rather than through the inbox, so there is no event to say it on.
type SteerRequest struct {
	Prompt string `json:"prompt"`
}

// ResolveRequest is the body of POST /api/sessions/{id}/approvals/{use_id}:
// a decision on a call the session is waiting on.
//
// Allow is a pointer so that a body which omits it is a 400 rather than a
// silent denial. Deny-by-default would be the safe reading, but a UI whose
// field name drifted would then refuse every call while looking like it
// worked. It is a body and not a query parameter so a verdict cannot be
// delivered by a link somebody was tricked into following.
type ResolveRequest struct {
	Allow *bool `json:"allow"`

	// From names who decided, kept with the call's result in the
	// transcript's origins. Optional.
	From string `json:"from,omitempty"`
}

// Accepted is the body of a 202: the event the inbox took, and the session
// it is for. Duplicate is set — with a 200 — for an event whose dedup key says
// it was received before: Event is that one.
type Accepted struct {
	Event     daemon.Event       `json:"event"`
	Session   daemon.SessionInfo `json:"session"`
	Duplicate bool               `json:"duplicate,omitempty"`
}

// EventRequest is the body of POST /api/events.
//
// It goes to the session named, or to the session tied to Thread — an issue,
// a chat thread; made, with Options, the first time — or, with neither, to a
// new session. A sender that may deliver twice sets DedupKey: the same key
// from the same Source is received once.
//
// A bridge relaying a chat posts each message as Kind "message", with the
// payload {"text": "..."}, its own Source ("slack"), and From naming the
// person; see "Who said what" in the package documentation.
type EventRequest struct {
	SessionID string          `json:"session_id,omitempty"`
	Thread    string          `json:"thread,omitempty"`
	Kind      string          `json:"kind"`
	Source    string          `json:"source,omitempty"`
	From      string          `json:"from,omitempty"`
	DedupKey  string          `json:"dedup_key,omitempty"`
	Payload   json.RawMessage `json:"payload,omitempty"`

	// Options are for the session the event makes, when it makes one.
	Options *daemon.SessionOptions `json:"options,omitempty"`
}

// TimerRequest is the body of POST /api/timers: a timer due In from now or At
// a time — exactly one — and, with Every, again at that interval after.
type TimerRequest struct {
	SessionID string `json:"session_id"`
	Note      string `json:"note"`
	In        string `json:"in,omitempty"`
	At        string `json:"at,omitempty"`
	Every     string `json:"every,omitempty"`
}

// EventList is the body of GET /api/inbox.
type EventList struct {
	Events []daemon.Event `json:"events"`
}

// MessageList is the body of GET /api/outbox.
type MessageList struct {
	Messages []daemon.Message `json:"messages"`
}

// TimerList is the body of GET /api/timers.
type TimerList struct {
	Timers []daemon.Timer `json:"timers"`
}

// ErrorBody is the envelope every failure is reported in.
type ErrorBody struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the envelope's payload. Kind is a bounded machine-readable
// token; Message is for a human and may change without notice.
type ErrorDetail struct {
	Kind    string `json:"kind"`
	Message string `json:"message"`
}

// Health is the body of GET /api/health.
type Health struct {
	Status    string  `json:"status"`
	Version   string  `json:"version,omitempty"`
	UptimeSec float64 `json:"uptime_sec"`
}

// SessionList is the body of GET /api/sessions.
//
// An object with one field rather than a bare array: a top-level array cannot
// grow a sibling — a cursor, a total — without breaking every client, and this
// collection will want one.
type SessionList struct {
	Sessions []daemon.SessionInfo `json:"sessions"`
}

// ApprovalList is the body of GET /api/sessions/{id}/approvals, for the same
// reason [SessionList] is an object: the calls waiting for a decision, and
// those waiting for a result (POST /api/sessions/{id}/results/{use_id}).
type ApprovalList struct {
	Approvals []daemon.Approval `json:"approvals"`
	Awaiting  []daemon.Approval `json:"awaiting"`
}

// ResultRequest is the body of POST /api/sessions/{id}/results/{use_id}: the
// result of a call the session is waiting on, which started something that
// finishes outside it — a build, a job. The call returns Output, as if its
// tool had, or fails with it when IsError.
//
// Output is a pointer for the reason [ResolveRequest.Allow] is: a body that
// omits it is a 400, not an empty result.
type ResultRequest struct {
	Output  *string `json:"output"`
	IsError bool    `json:"is_error,omitempty"`

	// From names who, or what, posted it; kept with the call's result in the
	// transcript's origins. Optional.
	From string `json:"from,omitempty"`
}

// ToolCallList is the body of GET /api/tool-calls: calls of the ledger, the
// newest first. See [daemon.ToolCall].
type ToolCallList struct {
	Calls []daemon.ToolCall `json:"calls"`
}
