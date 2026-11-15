package permission

import (
	"bytes"
	"encoding/json"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
)

// Requested announces that a call is waiting for a human.
//
// Emitted by [Gate] immediately before it blocks on an [Approver], and only
// then: an allow, a deny and a remembered grant all reach a verdict without
// anyone being asked, and a UI that drew a prompt for those would be asking a
// question that has already been answered. The Input is included because
// "approve bash?" is not a question anybody can answer — "approve
// `rm -rf build/`?" is.
type Requested struct {
	UseID  llm.ToolUseID   `json:"use_id"`
	Tool   string          `json:"tool"`
	Reason string          `json:"reason,omitempty"`
	Input  json.RawMessage `json:"input,omitempty"`
}

// Decided reports the verdict on a call. Emitted for EVERY call that passes
// through [Gate], including the allowed ones — a permission log that records
// only refusals cannot answer "what did this run actually do", which is the
// question asked after the fact.
//
// Source names what settled it, and is one of:
//
//	"policy"      the rules reached a verdict on their own
//	"grant"       a person had already approved this exact call this run
//	"approver"    a person answered just now
//	"no-approver" the policy asked and there was nobody to ask
type Decided struct {
	UseID   llm.ToolUseID `json:"use_id"`
	Tool    string        `json:"tool"`
	Allowed bool          `json:"allowed"`
	Reason  string        `json:"reason,omitempty"`
	Source  string        `json:"source,omitempty"`
}

// Decision sources, as they appear in [Decided.Source].
const (
	sourcePolicy     = "policy"
	sourceGrant      = "grant"
	sourceApprover   = "approver"
	sourceNoApprover = "no-approver"
)

// Kind implements core.Event.
func (Requested) Kind() string { return "permission_requested" }

// Kind implements core.Event.
func (Decided) Kind() string { return "permission_decided" }

// MarshalJSON implements json.Marshaler.
func (e Requested) MarshalJSON() ([]byte, error) { type r Requested; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e Decided) MarshalJSON() ([]byte, error) { type r Decided; return eventJSON(e.Kind(), r(e)) }

// EventTypes returns a zero value of every event type this package defines.
//
// Same contract as core.EventTypes and for the same reason: [core.Event]
// is an open interface, Go cannot check a type switch over it for
// exhaustiveness, and a renderer or code generator needs one list to walk. A
// front end that wants to render permission prompts appends this to the
// harness's own list.
func EventTypes() []core.Event {
	return []core.Event{Requested{}, Decided{}}
}

// eventJSON marshals v as an object with "type" spliced in as the first key.
//
// This is a copy of the helper in the root package's event.go, which is
// unexported there and cannot be reached from here. Saying so plainly rather
// than hiding it: the alternative is exporting a marshalling detail from the
// root package purely so a sibling can share nine lines, and the wire format
// it implements — discriminator first, no HTML escaping — is a contract
// already written down in event.go. The two must stay identical; a front end
// reads both streams through the same decoder.
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
// the full reasoning: a tool's arguments are full of <, > and &, escaping is
// decided twice (here and again by the outer encoder's compact pass), and the
// bytes survive unescaped only if both halves agree.
func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
