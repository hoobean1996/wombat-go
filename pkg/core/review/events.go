package review

import (
	"bytes"
	"encoding/json"

	"wombat/pkg/core"
)

// Reviewed reports a finished review on an event stream. A host that runs a
// review after a turn — wombatd with review enabled — emits it, so whoever is
// watching the stream reads what the review found next to the answer it was
// about.
//
// Ran is separate from a zero Findings count on purpose: a review that never
// started and one that looked and found nothing are the same number and
// opposite facts.
type Reviewed struct {
	Ran      bool              `json:"ran"`
	Error    string            `json:"error,omitempty"`
	Summary  string            `json:"summary"`
	Rounds   int               `json:"rounds,omitempty"`
	Findings []ReviewedFinding `json:"findings,omitempty"`

	// Unfinished is what the author's own TODO file still has unticked.
	Unfinished []string `json:"unfinished,omitempty"`

	// Discarded counts test sites that throw away a call's last return value.
	Discarded int `json:"discarded_error_sites,omitempty"`

	// PlanEdited names planning artifacts an agent rewrote during the run.
	PlanEdited []string `json:"plan_edited,omitempty"`

	// Sweeps is how many review passes ran; Converged says whether the run
	// ended because they stopped finding things, rather than running out.
	Sweeps    int  `json:"sweeps,omitempty"`
	Converged bool `json:"converged"`
}

// ReviewedFinding is one defect in a [Reviewed] event: the test that
// demonstrates it, and whether a repair round made that test pass.
type ReviewedFinding struct {
	Test    string `json:"test"`
	File    string `json:"file"`
	Message string `json:"message"`
	Fixed   bool   `json:"fixed"`
}

// Kind implements core.Event.
func (Reviewed) Kind() string { return "reviewed" }

// MarshalJSON implements json.Marshaler.
func (e Reviewed) MarshalJSON() ([]byte, error) { type r Reviewed; return eventJSON(e.Kind(), r(e)) }

// Event is the report as a [Reviewed] event.
func (r *Report) Event() Reviewed {
	ev := Reviewed{
		Ran: r.Ran, Summary: r.Summary(), Rounds: r.Rounds,
		Unfinished: r.Unfinished, Discarded: len(r.Discarded), PlanEdited: r.PlanEdited,
		Sweeps: r.Sweeps, Converged: r.Converged,
	}
	if r.Err != nil {
		ev.Error = r.Err.Error()
	}
	for _, f := range r.Findings {
		ev.Findings = append(ev.Findings, ReviewedFinding{Test: f.Test, File: f.File, Message: f.Message, Fixed: f.Fixed})
	}
	return ev
}

// EventTypes returns a zero value of every event type this package defines,
// for the same reason core.EventTypes does.
func EventTypes() []core.Event { return []core.Event{Reviewed{}} }

// eventJSON marshals v as an object with "type" spliced in as the first key,
// without HTML escaping — review findings quote code, which is full of <, >
// and &.
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

func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
