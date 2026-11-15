package daemon

import (
	"bytes"
	"encoding/json"
	"time"

	"wombat/pkg/core"
)

// TurnStarted opens one turn of a conversation.
//
// The harness has no such event because a [core.Run] IS one turn: there is
// nothing to delimit. A session is many runs sharing one event log and one
// sequence space, so the boundary has to be in the log or a client cannot draw
// it — and it cannot be inferred from a gap either, because there is no gap: a
// second turn's first event follows the first turn's last one immediately.
type TurnStarted struct {
	Turn int `json:"turn"`

	// Prompt is what the model was given: a person's message, as they wrote
	// it, or the events the turn is for, rendered. Echoed into the stream so
	// that a client which reconnects, or a second one watching, renders the
	// question as well as the answer. Empty for a turn resuming after a
	// decision, which the permission events in the turn show.
	Prompt string `json:"prompt,omitempty"`

	// Events are the inbox events the turn is for.
	Events []int64 `json:"events"`
}

// TurnEnded closes one turn and says how.
//
// It carries what a UI would otherwise have to reconstruct from the outcome of
// a run it cannot see: the answer, or the question the model is now waiting
// on, or the calls waiting for a decision, or the failure — plus the turn's
// own spend, which is the only place the cost of THIS turn is ever reported.
// [SessionInfo.Spend] is the running total.
type TurnEnded struct {
	Turn  int   `json:"turn"`
	State State `json:"state"`

	// Events are the inbox events the turn was for.
	Events []int64 `json:"events"`

	// Outcome is answer, paused, submitted or deferred, and empty when the
	// turn failed.
	Outcome string `json:"outcome,omitempty"`

	// Answer is the model's reply, when the turn ended in one.
	Answer string `json:"answer,omitempty"`

	// Question is what the model asked, when the turn ended paused. The next
	// message to this session answers it.
	Question string `json:"question,omitempty"`

	// Tool is the terminal tool the model called, when the turn ended
	// submitted.
	Tool string `json:"tool,omitempty"`

	// Approvals are the calls waiting for a decision, when the turn ended
	// deferred.
	Approvals []Approval `json:"approvals,omitempty"`

	// Error and ErrorKind describe a failed turn. ErrorKind is bounded; see
	// [ErrorKind].
	Error     string `json:"error,omitempty"`
	ErrorKind string `json:"error_kind,omitempty"`

	// RetryAt is when a failed turn's events run again, if they do.
	RetryAt time.Time `json:"retry_at,omitzero"`

	// Spend is what THIS turn cost.
	Spend Spend `json:"spend"`

	// Undelivered is what was sent with POST .../steer during this turn and
	// never reached the model, because the turn ended first.
	Undelivered []string `json:"undelivered,omitempty"`
}

// SteerSent echoes a message sent to the turn in progress, the way
// TurnStarted echoes a prompt. The run reports when it delivers it, with a
// core.Steered event.
type SteerSent struct {
	Turn   int    `json:"turn"`
	Prompt string `json:"prompt"`
}

// Kind implements [core.Event].
func (TurnStarted) Kind() string { return "turn_started" }

// Kind implements [core.Event].
func (TurnEnded) Kind() string { return "turn_ended" }

// Kind implements [core.Event].
func (SteerSent) Kind() string { return "steer_sent" }

// MarshalJSON implements [json.Marshaler].
func (e SteerSent) MarshalJSON() ([]byte, error) {
	type r SteerSent
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements [json.Marshaler].
func (e TurnStarted) MarshalJSON() ([]byte, error) {
	type r TurnStarted
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements [json.Marshaler].
func (e TurnEnded) MarshalJSON() ([]byte, error) {
	type r TurnEnded
	return eventJSON(e.Kind(), r(e))
}

// EventTypes returns a zero value of every [core.Event] this package defines.
//
// Same contract as [core.EventTypes] and [permission.EventTypes]: the Event
// interface is open, Go cannot check a switch over it for exhaustiveness, and
// a renderer needs one list to walk: this one, alongside the others, or these
// frames reach it as kinds it has never heard of.
func EventTypes() []core.Event {
	return []core.Event{TurnStarted{}, TurnEnded{}, SteerSent{}}
}

// eventJSON marshals v as an object with "type" spliced in as the first key.
//
// A third copy of the helper in the root package's event.go, for the same
// reason the permission package holds the second: it is unexported there, and
// exporting a marshalling detail so a sibling can share nine lines is the
// worse trade. The wire format it implements — discriminator first, no HTML
// escaping — is written down in event.go and all three must stay identical,
// because a front end reads every stream through the same decoder.
func eventJSON(kind string, v any) ([]byte, error) {
	body, err := marshalNoEscape(v)
	if err != nil {
		return nil, err
	}
	k, err := marshalNoEscape(kind)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, len(body)+len(k)+9)
	out = append(out, `{"type":`...)
	out = append(out, k...)
	if len(body) > 2 {
		out = append(out, ',')
		out = append(out, body[1:len(body)-1]...)
	}
	return append(out, '}'), nil
}

// marshalNoEscape encodes v without HTML escaping. See the root package for
// the full reasoning: model output is full of <, > and &, and escaping is
// decided twice — here, and again by the outer encoder's compact pass — so the
// bytes survive only if both halves agree.
func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
