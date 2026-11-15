// Package httpapi serves a multi-turn agent CONVERSATION over HTTP.
//
// The unit of the API is a session, and a session owns a conversation rather
// than a single run. That is the whole reason this package exists as an
// [http.Handler] instead of as another command: a run is over in a minute, a
// conversation is what a person actually has, and every hard problem here —
// resuming a stream across a turn boundary, answering a permission prompt
// that is parked in a dispatcher, reaping what nobody came back for — is a
// lifecycle problem. A lifecycle you can only test through a socket is one
// nobody tests, so [Manager] is HTTP-free and [New] is a thin projection of it
// onto a resource model:
//
//	GET    /api/health                            {status, version, uptime_sec}
//	GET    /api/config                            what this server offers
//	POST   /api/sessions                          create; {prompt, ...} in the body
//	GET    /api/sessions                          list
//	GET    /api/sessions/{id}                     one SessionInfo
//	DELETE /api/sessions/{id}                     cancel and drop
//	GET    /api/sessions/{id}/events              SSE, Last-Event-ID resume
//	GET    /api/sessions/{id}/messages            the transcript
//	POST   /api/sessions/{id}/messages            continue the conversation
//	POST   /api/sessions/{id}/steer               a message for the turn in progress
//	GET    /api/sessions/{id}/approvals           what is waiting on a human
//	POST   /api/sessions/{id}/approvals/{use_id}  {"allow": true|false}
//	GET    /metrics                               with [WithMetrics]
//
// Sequence numbers are global to the SESSION, not to the run. A UI that
// reconnects across a turn boundary therefore resumes into the same numbering
// it left, instead of silently losing the earlier turn to a stream that
// restarted at zero.
//
// Every error has the same shape, so a UI can branch on a bounded token
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
	"errors"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
)

// Errors this package reports. Match them with [errors.Is]; [New] maps each to
// a status code and to a stable "kind" token in the error envelope.
var (
	// ErrBusy reports that a turn is already running on this session. A
	// conversation is sequential — the model needs the previous turn's
	// transcript to answer the next one — so a second prompt is refused rather
	// than queued behind something whose outcome may change what the user
	// wanted to say.
	ErrBusy = errors.New("httpapi: a turn is already running")

	// ErrNotRunning reports a steering message sent to a session with no turn
	// in progress — or one that ended while the message was on its way. It is
	// the converse of ErrBusy: a message for a session at rest is a new turn,
	// and goes to POST .../messages.
	ErrNotRunning = errors.New("httpapi: no turn is running")

	// ErrNoSuchSession reports that the id names nothing. Indistinguishable
	// from "expired", on purpose: a reaped session and one that never existed
	// are the same fact to a client, and the TTL is not a secret worth leaking
	// through two different messages.
	ErrNoSuchSession = errors.New("httpapi: no such session")

	// ErrNoSuchApproval reports that no call is parked on that tool_use id.
	ErrNoSuchApproval = errors.New("httpapi: no such approval")

	// ErrAlreadyAnswered reports a second verdict on one question. A double
	// click on a slow network is ordinary, so this is a status code and never
	// a second answer delivered to the waiting dispatcher.
	ErrAlreadyAnswered = errors.New("httpapi: approval already answered")

	// ErrTooManySessions reports that the manager is at capacity. Sessions
	// spend real money with no connection attached, so an unbounded registry
	// is an unbounded bill as much as it is unbounded memory.
	ErrTooManySessions = errors.New("httpapi: too many sessions")

	// ErrClosed reports use of a [Manager] after [Manager.Close].
	ErrClosed = errors.New("httpapi: manager is closed")
)

// SessionOptions is what a client may choose per session.
//
// Deliberately small. Everything here is either a policy the server can still
// veto (Workspace, Permission, MaxIters) or a label (Title): a field a browser
// can set is a field a hostile browser can set, so the server's [ManagerConfig.Build]
// clamps rather than trusts.
type SessionOptions struct {
	// Model overrides the server's default model id, when the server allows it.
	Model string `json:"model,omitempty"`

	// Permission is the policy every tool call is checked against:
	// off, readonly, workspace or ask.
	Permission string `json:"permission,omitempty"`

	// Workspace roots the file and shell tools. A server is expected to
	// confine this to a directory it chose; see cmd/wombat-serve.
	Workspace string `json:"workspace,omitempty"`

	// MaxIters caps the ReAct loop for one turn.
	MaxIters int `json:"max_iters,omitempty"`

	// Title is for the human reading a session list. It has no effect on the
	// run.
	Title string `json:"title,omitempty"`
}

// State is where a session is in its lifecycle.
type State string

// The states a session can be in.
//
// Idle and Waiting are both "ready for another prompt" and are still
// distinguished, because they mean opposite things to the person at the
// keyboard: Idle is the agent's turn finished and yours may begin, Waiting is
// the agent asked YOU something and is holding the conversation open for the
// answer.
const (
	// Idle is between turns: the last turn finished and another may begin.
	Idle State = "idle"

	// Running is a turn in progress.
	Running State = "running"

	// Waiting is a turn that ended in [wombat.Paused] — the model asked the
	// user a question, and the next prompt answers it.
	Waiting State = "waiting"

	// Approving is a turn in progress with at least one tool call parked on a
	// human. It is a projection of Running, not a separate resting place: the
	// run has not stopped, it is blocked inside the permission gate.
	Approving State = "approving"

	// Done is a finished conversation with nothing left to continue from.
	Done State = "done"

	// Failed is a turn that ended in an error. The conversation is still
	// readable and, unless the transcript itself is broken, still continuable.
	Failed State = "failed"
)

// States returns every [State] in lifecycle order.
//
// It exists for the same reason [wombat.EventTypes] does: Go cannot check a
// switch over a string type for exhaustiveness, and a UI or a code generator
// needs one list to walk rather than a hand-copied parallel one. cmd/wombat-tsgen
// turns it into a TypeScript union.
func States() []State {
	return []State{Idle, Running, Waiting, Approving, Done, Failed}
}

// SessionInfo is everything about a session that is worth showing without
// reading its event log.
type SessionInfo struct {
	ID      string         `json:"id"`
	Title   string         `json:"title,omitempty"`
	State   State          `json:"state"`
	Turns   int            `json:"turns"`
	Events  int            `json:"events"`
	Created time.Time      `json:"created"`
	Updated time.Time      `json:"updated"`
	Options SessionOptions `json:"options"`
	Spend   Spend          `json:"spend"`

	// Outcome is how the LAST turn ended: answer, paused or submitted. Empty
	// while a turn is running and after a failure.
	Outcome string `json:"outcome,omitempty"`

	// Answer is the last turn's reply, or the question it is waiting on.
	Answer string `json:"answer,omitempty"`

	// Error and ErrorKind describe the last failure. ErrorKind is a bounded
	// token — see [ErrorKind] — so a UI can distinguish "you ran out of
	// budget" from "the provider fell over" without matching on prose.
	Error     string `json:"error,omitempty"`
	ErrorKind string `json:"error_kind,omitempty"`
}

// Spend is money and tokens, summed over every turn of a session.
//
// Distinct from [wombat.Spend], which is one run's live budget snapshot: this
// one survives the run that produced it, because the bill does.
type Spend struct {
	CostUSD      float64 `json:"cost_usd"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	CacheRead    int     `json:"cache_read_tokens"`
	Calls        int     `json:"calls"`
	ElapsedSec   float64 `json:"elapsed_sec"`
}

// add accumulates one turn's spend.
func (s *Spend) add(o Spend) {
	s.CostUSD += o.CostUSD
	s.InputTokens += o.InputTokens
	s.OutputTokens += o.OutputTokens
	s.CacheRead += o.CacheRead
	s.Calls += o.Calls
	s.ElapsedSec += o.ElapsedSec
}

// Frame is one entry in a session's log: an event, the turn it belongs to, and
// the session-global sequence number a client resumes from.
//
// The turn number is carried on the frame rather than inferred from the last
// [TurnStarted] a client happened to see, because a client that resumes from
// the middle of a turn never saw one.
type Frame struct {
	Seq   int          `json:"seq"`
	Turn  int          `json:"turn"`
	Event wombat.Event `json:"event"`
}

// Approval is one tool call parked on a human.
//
// Input is included because "approve bash?" is not a question anybody can
// answer, and "approve `rm -rf build/`?" is.
type Approval struct {
	UseID  string          `json:"use_id"`
	Tool   string          `json:"tool"`
	Reason string          `json:"reason"`
	Input  json.RawMessage `json:"input,omitempty"`
	Since  time.Time       `json:"since"`
}

// ===== Wire bodies =====

// CreateRequest is the body of POST /api/sessions.
//
// The prompt rides in a JSON body rather than a query string because a prompt
// is prose: it is routinely longer than a URL may safely be, it is logged by
// every proxy in between when it is in the URL, and it is the one field a user
// typed.
//
// [SessionOptions] is embedded rather than nested so the wire stays flat —
// {"prompt":"…","model":"…"} — which is what a form posts and what a curl by
// hand looks like.
type CreateRequest struct {
	Prompt string `json:"prompt"`
	SessionOptions
}

// SendRequest is the body of POST /api/sessions/{id}/messages: the next turn
// of the conversation.
type SendRequest struct {
	Prompt string `json:"prompt"`
}

// ResolveRequest is the body of POST /api/sessions/{id}/approvals/{use_id}.
//
// Allow is a pointer so that a body which omits it is a 400 rather than a
// silent denial. Deny-by-default would be the safe reading, but a UI whose
// field name drifted would then refuse every call while looking like it
// worked. It is a body and not a query parameter so a verdict cannot be
// delivered by a link somebody was tricked into following.
type ResolveRequest struct {
	Allow *bool `json:"allow"`
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
	Sessions []SessionInfo `json:"sessions"`
}

// ApprovalList is the body of GET /api/sessions/{id}/approvals, for the same
// reason [SessionList] is an object.
type ApprovalList struct {
	Approvals []Approval `json:"approvals"`
}

// APITypes returns a zero value of every type this package puts on the wire,
// for a code generator to reflect over.
//
// Same contract as [wombat.EventTypes], and it exists for the same reason: a
// front end that hand-writes these will get them wrong, and it will get them
// wrong silently — a field renamed here is a field that is simply undefined
// over there, with tsc vouching for it. The list is maintained beside the
// types it names, which is the only place the rot is visible.
//
// Two things are deliberately absent. The event types are [wombat.Event]
// implementations and belong to the stream's own union, so they are in
// [EventTypes]. And the transcript is []llm.Message in the harness's own
// persistence shape, which this package does not own and must not fork a
// second declaration of.
func APITypes() []any {
	return []any{
		Health{},
		Capabilities{},
		ToolInfo{},
		SessionOptions{},
		SessionInfo{},
		Spend{},
		Frame{},
		Approval{},
		SessionList{},
		ApprovalList{},
		CreateRequest{},
		SendRequest{},
		ResolveRequest{},
		ErrorBody{},
		ErrorDetail{},
	}
}
