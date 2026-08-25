package wombat

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

func TestEventKinds(t *testing.T) {
	tests := []struct {
		ev   Event
		want string
	}{
		{IterStart{}, "iter_start"},
		{LLMStart{}, "llm_start"},
		{TextDelta{}, "text_delta"},
		{ReasoningDelta{}, "reasoning_delta"},
		{ToolArgsDelta{}, "tool_args_delta"},
		{LLMDone{}, "llm_done"},
		{ToolStart{}, "tool_start"},
		{ToolDone{}, "tool_done"},
		{Spend{}, "spend"},
		{SubagentStart{}, "subagent_start"},
		{SubagentEvent{}, "subagent_event"},
		{SubagentEnd{}, "subagent_end"},
	}
	for _, tc := range tests {
		if got := tc.ev.Kind(); got != tc.want {
			t.Errorf("%T.Kind() = %q, want %q", tc.ev, got, tc.want)
		}
	}
}

func TestEventJSONPutsTypeFirst(t *testing.T) {
	tests := []struct {
		name string
		ev   Event
		want string
	}{
		{
			name: "iter_start",
			ev:   IterStart{N: 1, Max: 30},
			want: `{"type":"iter_start","n":1,"max":30}`,
		},
		{
			name: "tool_start passes the input through unreordered",
			ev:   ToolStart{UseID: "u1", Name: "bash", Category: "exec", Input: json.RawMessage(`{"cmd":"ls","z":1,"a":2}`)},
			want: `{"type":"tool_start","use_id":"u1","name":"bash","category":"exec","input":{"cmd":"ls","z":1,"a":2}}`,
		},
		{
			name: "omitempty fields stay out",
			ev:   TextDelta{Text: "hi"},
			want: `{"type":"text_delta","text":"hi"}`,
		},
		{
			name: "subagent_event nests the inner event with its own type",
			ev:   SubagentEvent{Name: "child", Depth: 1, Inner: TextDelta{Text: "hi"}},
			want: `{"type":"subagent_event","name":"child","depth":1,"inner":{"type":"text_delta","text":"hi"}}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.ev)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			if string(b) != tc.want {
				t.Errorf("got  %s\nwant %s", b, tc.want)
			}
		})
	}
}

// TestEventTypesAreExhaustivelyDiscriminated is the exhaustiveness guard the
// Event interface cannot give.
//
// Go cannot check that a type switch over an open interface covers every
// variant, and it cannot check that a new variant remembered to implement
// MarshalJSON. Walking EventTypes() catches the second half: a variant added
// to the list but given no MarshalJSON marshals without a "type" key at all,
// and a front end silently never learns to render it.
func TestEventTypesAreExhaustivelyDiscriminated(t *testing.T) {
	types := EventTypes()
	if len(types) == 0 {
		t.Fatal("EventTypes() is empty")
	}

	seen := make(map[string]bool, len(types))
	for _, ev := range types {
		t.Run(ev.Kind(), func(t *testing.T) {
			if ev.Kind() == "" {
				t.Fatalf("%T has an empty Kind()", ev)
			}
			if seen[ev.Kind()] {
				t.Fatalf("kind %q is claimed by more than one type", ev.Kind())
			}
			seen[ev.Kind()] = true

			b, err := json.Marshal(ev)
			if err != nil {
				t.Fatalf("Marshal(%T): %v", ev, err)
			}
			var probe map[string]json.RawMessage
			if err := json.Unmarshal(b, &probe); err != nil {
				t.Fatalf("Unmarshal(%T -> %s): %v", ev, b, err)
			}
			raw, ok := probe["type"]
			if !ok {
				t.Fatalf("%T marshalled without a \"type\" key: %s "+
					"(it is in EventTypes() but has no MarshalJSON)", ev, b)
			}
			var kind string
			if err := json.Unmarshal(raw, &kind); err != nil {
				t.Fatalf("%T: \"type\" is not a string: %s", ev, raw)
			}
			if kind != ev.Kind() {
				t.Errorf("%T: json type = %q, want Kind() = %q", ev, kind, ev.Kind())
			}
			if !strings.HasPrefix(string(b), `{"type":`) {
				t.Errorf("%T: type is not the first key: %s", ev, b)
			}
		})
	}

	// Every type the package defines must be reachable from the list. This is
	// the half a test can only approximate; the list is checked against the
	// kinds the loop actually emits in TestRunToolCallThenAnswerEventOrder.
	for _, want := range []string{
		"iter_start", "llm_start", "text_delta", "reasoning_delta", "tool_args_delta",
		"llm_done", "tool_start", "tool_done", "spend",
		"subagent_start", "subagent_event", "subagent_end",
	} {
		if !seen[want] {
			t.Errorf("EventTypes() omits %q", want)
		}
	}
}

// TestEventJSONIsNotHTMLEscaped is a regression test.
//
// Model output is full of <, > and &. encoding/json escapes them by default,
// and the escaping is decided TWICE: the outer encoder re-runs a compact pass
// over whatever a custom MarshalJSON returned. An event therefore survives
// unescaped only if both halves agree. This asserts on the encoder path a
// consumer actually uses — an encoder with SetEscapeHTML(false), which is what
// cmd/wombat-jsonl does — because that is the path where the bug was visible
// and json.Marshal alone would hide it behind its own re-escaping.
func TestEventJSONIsNotHTMLEscaped(t *testing.T) {
	const text = "a<b>c & d 中文"

	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(TextDelta{Text: text}); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	line := strings.TrimSpace(buf.String())

	want := `{"type":"text_delta","text":"` + text + `"}`
	if line != want {
		t.Fatalf("got  %s\nwant %s", line, want)
	}
	for _, escaped := range []string{`\u003c`, `\u003e`, `\u0026`} {
		if strings.Contains(line, escaped) {
			t.Errorf("output contains the HTML escape %s: %s", escaped, line)
		}
	}

	// And it still decodes back to exactly the text the model produced.
	var back struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal([]byte(line), &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if back.Text != text {
		t.Errorf("round trip: got %q, want %q", back.Text, text)
	}

	// The nested case: an inner event's text must survive the wrapper too.
	buf.Reset()
	enc = json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(SubagentEvent{Name: "child", Inner: TextDelta{Text: text}}); err != nil {
		t.Fatalf("Encode nested: %v", err)
	}
	if got := strings.TrimSpace(buf.String()); !strings.Contains(got, text) {
		t.Errorf("nested event escaped its inner text: %s", got)
	}
}

// A caller that reaches for plain json.Marshal still gets HTML escaping,
// because the compact pass re-escapes. That is documented behaviour, not a
// bug, and pinning it keeps the doc comment honest.
func TestPlainMarshalStillEscapesHTML(t *testing.T) {
	b, err := json.Marshal(TextDelta{Text: "a<b>"})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(b), `\u003c`) {
		t.Errorf("json.Marshal output: got %s, want it to carry the escaped form", b)
	}
}

func TestEmitter(t *testing.T) {
	t.Run("Emit with no sink is a no-op", func(t *testing.T) {
		Emit(context.Background(), TextDelta{Text: "nobody is listening"})
	})

	t.Run("Emit with a nil sink is a no-op", func(t *testing.T) {
		ctx := WithEmitter(context.Background(), nil)
		Emit(ctx, TextDelta{Text: "still nobody"})
	})

	t.Run("Emit reaches the sink on ctx", func(t *testing.T) {
		var got []Event
		ctx := WithEmitter(context.Background(), func(e Event) { got = append(got, e) })
		Emit(ctx, IterStart{N: 1})
		Emit(ctx, TextDelta{Text: "hi"})
		if len(got) != 2 {
			t.Fatalf("got %d events, want 2", len(got))
		}
		if got[0].Kind() != "iter_start" || got[1].Kind() != "text_delta" {
			t.Errorf("got kinds %q,%q, want iter_start,text_delta", got[0].Kind(), got[1].Kind())
		}
	})

	t.Run("the innermost emitter wins", func(t *testing.T) {
		var outer, inner int
		ctx := WithEmitter(context.Background(), func(Event) { outer++ })
		ctx = WithEmitter(ctx, func(Event) { inner++ })
		Emit(ctx, TextDelta{})
		if outer != 0 || inner != 1 {
			t.Errorf("got outer=%d inner=%d, want 0 and 1", outer, inner)
		}
	})
}

func TestMillisHelper(t *testing.T) {
	if got := millis(1500 * time.Millisecond); got != 1500 {
		t.Errorf("millis(1.5s) = %d, want 1500", got)
	}
}

// LLMStart carries an llm.Purpose, which must not leak into the wire shape
// when it is the zero value.
func TestLLMStartOmitsEmptyFields(t *testing.T) {
	b, err := json.Marshal(LLMStart{Tools: 3})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if want := `{"type":"llm_start","tools":3}`; string(b) != want {
		t.Errorf("got %s, want %s", b, want)
	}

	b, err = json.Marshal(LLMStart{Model: "m", Purpose: llm.PurposePlanner, Tools: 1, Forced: "submit"})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	want := `{"type":"llm_start","model":"m","purpose":"planner","tools":1,"forced_tool":"submit"}`
	if string(b) != want {
		t.Errorf("got  %s\nwant %s", b, want)
	}
}
