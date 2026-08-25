package httpapi

import (
	"bytes"
	"encoding/json"

	wombat "github.com/hoobean1996/wombat-go"
)

// TurnStarted opens one turn of a conversation.
//
// The harness has no such event because a [wombat.Run] IS one turn: there is
// nothing to delimit. A session is many runs sharing one event log and one
// sequence space, so the boundary has to be in the log or a client cannot draw
// it — and it cannot be inferred from a gap either, because there is no gap: a
// second turn's first event follows the first turn's last one immediately.
type TurnStarted struct {
	Turn int `json:"turn"`

	// Prompt is what the user sent. Echoed into the stream rather than left
	// implicit so that a client which reconnects, or a second client watching
	// the same session, renders the question as well as the answer.
	Prompt string `json:"prompt"`
}

// TurnEnded closes one turn and says how.
//
// It carries what a UI would otherwise have to reconstruct from the outcome of
// a run it cannot see: the answer, or the question the model is now waiting
// on, or the failure — plus the turn's own spend, which is the only place the
// cost of THIS turn is ever reported. [SessionInfo.Spend] is the running
// total.
type TurnEnded struct {
	Turn  int   `json:"turn"`
	State State `json:"state"`

	// Outcome is answer, paused or submitted, and empty when the turn failed.
	Outcome string `json:"outcome,omitempty"`

	// Answer is the model's reply, when the turn ended in one.
	Answer string `json:"answer,omitempty"`

	// Question is what the model asked, when the turn ended paused. The next
	// prompt on this session answers it.
	Question string `json:"question,omitempty"`

	// Tool is the terminal tool the model called, when the turn ended
	// submitted.
	Tool string `json:"tool,omitempty"`

	// Error and ErrorKind describe a failed turn. ErrorKind is bounded; see
	// [ErrorKind].
	Error     string `json:"error,omitempty"`
	ErrorKind string `json:"error_kind,omitempty"`

	// Spend is what THIS turn cost.
	Spend Spend `json:"spend"`
}

// Kind implements [wombat.Event].
func (TurnStarted) Kind() string { return "turn_started" }

// Kind implements [wombat.Event].
func (TurnEnded) Kind() string { return "turn_ended" }

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

// EventTypes returns a zero value of every [wombat.Event] this package defines.
//
// Same contract as [wombat.EventTypes] and [permission.EventTypes]: the Event
// interface is open, Go cannot check a switch over it for exhaustiveness, and
// a renderer or generator needs one list to walk. cmd/wombat-tsgen names this
// registry alongside the others, or these two frames reach the front end as
// types tsc has never heard of.
func EventTypes() []wombat.Event {
	return []wombat.Event{TurnStarted{}, TurnEnded{}}
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
