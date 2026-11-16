package daemon

import (
	"encoding/json"
	"errors"
	"time"

	"wombat/pkg/core"
)

// Errors this package reports. Match them with [errors.Is]; wombat/pkg/httpapi
// maps each to a status code and a stable "kind" token in its error envelope.
var (
	// ErrNotRunning reports a steering message sent to a session with no turn
	// in progress — or one that ended while the message was on its way. A
	// message for a session at rest is an event for its next turn, and goes
	// to [Session.Send].
	ErrNotRunning = errors.New("daemon: no turn is running")

	// ErrNoSuchSession reports that the id names nothing.
	ErrNoSuchSession = errors.New("daemon: no such session")

	// ErrNoSuchApproval reports that no call is waiting on that tool_use id.
	ErrNoSuchApproval = errors.New("daemon: no such approval")

	// ErrAlreadyAnswered reports a second verdict on one call. A double click
	// on a slow network is ordinary, so this is a status code and never a
	// second decision.
	ErrAlreadyAnswered = errors.New("daemon: approval already answered")

	// ErrDone reports an event for a conversation that has ended.
	//
	// Only [Done] is final, and only a terminal tool produces it. A FAILED
	// turn does not end the conversation: a rate limit, a blown budget or a
	// cancelled turn must not cost the user the transcript, so the session
	// takes the next event and carries on where it stopped.
	ErrDone = errors.New("daemon: the session has finished")

	// ErrBadRequest marks anything the client got wrong: malformed input, a
	// missing prompt, a timer due at no time — and whatever a
	// [ManagerConfig.Build] judges a client may not ask for. Build is the one
	// place a HOST judges client input — a workspace outside the directory
	// the operator allowed, a model this deployment does not offer — and
	// without a sentinel to say so with, every such refusal reaches the
	// client as a 500 that reads like the server broke. Wrap it:
	//
	//	return nil, fmt.Errorf("%w: workspace %q is outside %s",
	//	    daemon.ErrBadRequest, opts.Workspace, root)
	ErrBadRequest = errors.New("daemon: malformed request")

	// ErrClosed reports use of a [Manager] after [Manager.Close].
	ErrClosed = errors.New("daemon: manager is closed")
)

// PermissionFull is full access: the whole tool surface, and nobody asked
// about any of it. A server offers it only when its operator says so, and a
// session switched to it has the calls it is waiting on allowed; see
// [Session.SetPermission].
const PermissionFull = "full"

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
	// off, readonly, workspace, ask or [PermissionFull]. A session can change
	// it later; see [Session.SetPermission].
	Permission string `json:"permission,omitempty"`

	// Workspace roots the file and shell tools. A server is expected to
	// confine this to a directory it chose; see cmd/wombatd.
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
// Idle, Waiting and Approving are all at rest, and mean different things to
// the person the agent works for: Idle is the agent done and ready for more,
// Waiting is the agent asking YOU something, and Approving is the agent
// stopped on a call it may not make until you say so.
const (
	// Idle is between turns: the last turn finished and another may begin.
	Idle State = "idle"

	// Running is a turn in progress.
	Running State = "running"

	// Waiting is a turn that ended in [core.Paused] — the model asked the
	// user a question, and the next message answers it.
	Waiting State = "waiting"

	// Approving is a turn that ended in [core.Deferred]: a call is waiting
	// for a person's decision, and the session takes nothing but decisions
	// until they are all in. Whatever else arrives meanwhile waits in the
	// inbox, for the turn after.
	Approving State = "approving"

	// Done is a finished conversation with nothing left to continue from.
	Done State = "done"

	// Failed is a turn that ended in an error. The conversation is still
	// readable and, unless the transcript itself is broken, still continuable.
	Failed State = "failed"
)

// SessionInfo is everything about a session that is worth showing without
// reading its event log.
type SessionInfo struct {
	ID      string         `json:"id"`
	Title   string         `json:"title,omitempty"`
	Thread  string         `json:"thread,omitempty"`
	State   State          `json:"state"`
	Turns   int            `json:"turns"`
	Events  int            `json:"events"`
	Created time.Time      `json:"created"`
	Updated time.Time      `json:"updated"`
	Options SessionOptions `json:"options"`
	Spend   Spend          `json:"spend"`

	// Outcome is how the LAST turn ended: answer, paused, submitted or
	// deferred. Empty while a turn is running and after a failure.
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
// Distinct from [core.Spend], which is one run's live budget snapshot: this
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
	Seq   int        `json:"seq"`
	Turn  int        `json:"turn"`
	Event core.Event `json:"event"`
}

// Approval is one tool call waiting for a person's decision.
//
// Input is included because "approve bash?" is not a question anybody can
// answer, and "approve `rm -rf build/`?" is.
type Approval struct {
	UseID  string          `json:"use_id"`
	Tool   string          `json:"tool"`
	Reason string          `json:"reason"`
	Input  json.RawMessage `json:"input,omitempty"`

	// Since is when the turn stopped for it.
	Since time.Time `json:"since"`
}
