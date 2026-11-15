package core

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
)

func TestPauseSchemaSummary(t *testing.T) {
	tests := []struct {
		name string
		p    PauseSchema
		want string
	}{
		{name: "question only", p: PauseSchema{Question: "which branch?"}, want: "which branch?"},
		{name: "question and schema", p: PauseSchema{Question: "pick", Schema: json.RawMessage(`{}`)}, want: "pick (structured)"},
		{name: "schema only", p: PauseSchema{Schema: json.RawMessage(`{}`)}, want: "structured input"},
		{name: "empty", p: PauseSchema{}, want: "input"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.p.Summary(); got != tc.want {
				t.Errorf("Summary() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestParsePauseSchema(t *testing.T) {
	tests := []struct {
		name       string
		in         string
		wantQ      string
		wantSchema bool
	}{
		{name: "empty input", in: "", wantQ: ""},
		{name: "bare question", in: `{"question":"which branch?"}`, wantQ: "which branch?"},
		{name: "question plus schema", in: `{"question":"pick","schema":{"type":"string"}}`, wantQ: "pick", wantSchema: true},
		{name: "schema only", in: `{"schema":{"type":"string"}}`, wantSchema: true},
		// Tolerance: a model that used a different key still yields a usable
		// prompt rather than failing the run.
		{name: "prompt key", in: `{"prompt":"go on?"}`, wantQ: "go on?"},
		{name: "message key", in: `{"message":"go on?"}`, wantQ: "go on?"},
		// Unrecognised shapes fall back to the raw payload, which is at least
		// something a front end can render.
		{name: "unknown keys", in: `{"other":"x"}`, wantQ: `{"other":"x"}`},
		{name: "not an object", in: `"just a string"`, wantQ: `"just a string"`},
		{name: "malformed", in: `{not json`, wantQ: `{not json`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var raw json.RawMessage
			if tc.in != "" {
				raw = json.RawMessage(tc.in)
			}
			got := ParsePauseSchema(raw)
			if got.Question != tc.wantQ {
				t.Errorf("Question = %q, want %q", got.Question, tc.wantQ)
			}
			if hasSchema := len(got.Schema) > 0; hasSchema != tc.wantSchema {
				t.Errorf("has schema = %v, want %v (schema=%s)", hasSchema, tc.wantSchema, got.Schema)
			}
		})
	}
}

func TestExpectAnswer(t *testing.T) {
	tests := []struct {
		name    string
		out     Outcome
		want    string
		wantErr string
	}{
		{name: "answer", out: Answer{Text: "42"}, want: "42"},
		{name: "submitted", out: Submitted{Tool: "submit"}, wantErr: `terminal tool "submit"`},
		{name: "paused", out: Paused{Schema: PauseSchema{Question: "which?"}}, wantErr: "asked the user (which?)"},
		{name: "nil", out: nil, wantErr: "got <nil>"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ExpectAnswer(tc.out, "planner")
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("got (%q, nil), want an error", got)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("error = %q, want it to contain %q", err, tc.wantErr)
				}
				if !strings.Contains(err.Error(), "planner") {
					t.Errorf("error = %q, want it to name the caller %q", err, "planner")
				}
				return
			}
			if err != nil {
				t.Fatalf("got error %v, want nil", err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestExpectSubmitted(t *testing.T) {
	payload := json.RawMessage(`{"summary":"done"}`)

	t.Run("matching tool", func(t *testing.T) {
		got, err := ExpectSubmitted(Submitted{Tool: "submit", Payload: payload}, "submit")
		if err != nil {
			t.Fatalf("got error %v, want nil", err)
		}
		if string(got) != string(payload) {
			t.Errorf("payload = %s, want %s", got, payload)
		}
	})

	tests := []struct {
		name    string
		out     Outcome
		wantErr string
	}{
		{name: "wrong tool", out: Submitted{Tool: "other"}, wantErr: `got "other"`},
		{name: "answer", out: Answer{Text: "hi"}, wantErr: "answered in text"},
		{name: "paused", out: Paused{Schema: PauseSchema{Question: "which?"}}, wantErr: "asked the user"},
		{name: "nil", out: nil, wantErr: "got <nil>"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ExpectSubmitted(tc.out, "submit")
			if err == nil {
				t.Fatalf("got (%s, nil), want an error", got)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err, tc.wantErr)
			}
			if got != nil {
				t.Errorf("payload = %s, want nil on error", got)
			}
		})
	}
}

func TestSentinelErrors(t *testing.T) {
	t.Run("RefusalError keeps the reason and matches ErrRefused", func(t *testing.T) {
		err := error(&RefusalError{Reason: "I can't help with that."})
		if !errors.Is(err, ErrRefused) {
			t.Errorf("errors.Is(err, ErrRefused) = false, want true")
		}
		if !strings.Contains(err.Error(), "can't help") {
			t.Errorf("Error() = %q, want it to carry the reason", err)
		}
	})

	t.Run("RefusalError with no reason", func(t *testing.T) {
		err := error(&RefusalError{})
		if got, want := err.Error(), ErrRefused.Error(); got != want {
			t.Errorf("Error() = %q, want %q", got, want)
		}
		if !errors.Is(err, ErrRefused) {
			t.Errorf("errors.Is(err, ErrRefused) = false, want true")
		}
	})

	t.Run("UnexpectedStopError names the stop reason", func(t *testing.T) {
		err := error(&UnexpectedStopError{StopReason: llm.StopReason("content_filter")})
		if !errors.Is(err, ErrUnexpectedStop) {
			t.Errorf("errors.Is(err, ErrUnexpectedStop) = false, want true")
		}
		if !strings.Contains(err.Error(), `"content_filter"`) {
			t.Errorf("Error() = %q, want it to quote the stop_reason", err)
		}
	})

	// The sentinels must be distinct values, or errors.Is collapses them and a
	// caller branching on max_tokens catches a refusal.
	t.Run("sentinels are distinct", func(t *testing.T) {
		all := []error{ErrMaxIterations, ErrMaxTokens, ErrRefused, ErrUnexpectedStop, ErrPanic}
		for i, a := range all {
			for j, b := range all {
				if i != j && errors.Is(a, b) {
					t.Errorf("errors.Is(%v, %v) = true, want false", a, b)
				}
			}
		}
	})
}

// The Outcome set is closed by an unexported method; this pins that the three
// documented shapes are the ones that satisfy it.
func TestOutcomeImplementations(t *testing.T) {
	var outs = []Outcome{Answer{}, Paused{}, Submitted{}}
	if len(outs) != 3 {
		t.Fatalf("got %d outcomes, want 3", len(outs))
	}
}
