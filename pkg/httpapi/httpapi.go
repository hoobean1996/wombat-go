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
//	POST   /api/sessions                          create, with its first message: {prompt, ...}
//	GET    /api/sessions                          list, newest first
//	GET    /api/sessions/{id}                     one SessionInfo
//	DELETE /api/sessions/{id}                     cancel and forget, timers and all
//	GET    /api/sessions/{id}/events              SSE, Last-Event-ID resume
//	GET    /api/sessions/{id}/messages            the transcript
//	POST   /api/sessions/{id}/messages            a message, for the next turn
//	POST   /api/sessions/{id}/steer               a message for the turn in progress
//	GET    /api/sessions/{id}/approvals           what is waiting for a decision
//	POST   /api/sessions/{id}/approvals/{use_id}  {"allow": true|false}
//	POST   /api/events                            an event from outside: a webhook, an alert
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
	daemon.SessionOptions
}

// SendRequest is the body of POST /api/sessions/{id}/messages — a message for
// the session's next turn — and of POST /api/sessions/{id}/steer.
type SendRequest struct {
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
type EventRequest struct {
	SessionID string          `json:"session_id,omitempty"`
	Thread    string          `json:"thread,omitempty"`
	Kind      string          `json:"kind"`
	Source    string          `json:"source,omitempty"`
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
// reason [SessionList] is an object.
type ApprovalList struct {
	Approvals []daemon.Approval `json:"approvals"`
}
