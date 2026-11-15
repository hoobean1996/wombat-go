package openai

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
)

// collector records everything a sink saw, in order.
type collector struct {
	text      []string
	reasoning []string
	args      []llm.ToolArgsDelta
	all       []llm.Delta
}

func (c *collector) sink() func(llm.Delta) {
	return func(d llm.Delta) {
		c.all = append(c.all, d)
		if d.Text != "" {
			c.text = append(c.text, d.Text)
		}
		if d.Reasoning != "" {
			c.reasoning = append(c.reasoning, d.Reasoning)
		}
		if d.ToolArgs != nil {
			c.args = append(c.args, *d.ToolArgs)
		}
	}
}

// streamBody frames chunks the way sseReply does, for the direct decodeStream
// tests (no HTTP needed to exercise the accumulator).
func streamBody(frames ...string) string {
	var b strings.Builder
	for _, f := range frames {
		typ, data, tagged := strings.Cut(f, "\x00")
		if tagged {
			b.WriteString("event: " + typ + "\n")
		} else {
			data = typ
		}
		b.WriteString("data: " + data + "\n\n")
	}
	return b.String()
}

func decodeFrames(t *testing.T, onDelta func(llm.Delta), frames ...string) (llm.Response, error) {
	t.Helper()
	return decodeStream(strings.NewReader(streamBody(frames...)), onDelta, "requested-model")
}

func TestStreamTextAndUsage(t *testing.T) {
	var got collector
	resp, err := decodeFrames(t, got.sink(),
		`{"id":"1","model":"served","choices":[{"index":0,"delta":{"role":"assistant","content":"It is "}}]}`,
		`{"id":"1","choices":[{"index":0,"delta":{"content":"42."}}]}`,
		`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		// Real OpenAI puts usage on a trailing chunk with no choices.
		`{"id":"1","choices":[],"usage":{"prompt_tokens":200,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":150}}}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}
	if want := "It is 42."; llm.TextOf(resp.Content) != want {
		t.Errorf("text: got %q, want %q", llm.TextOf(resp.Content), want)
	}
	if want := []string{"It is ", "42."}; !reflect.DeepEqual(got.text, want) {
		t.Errorf("text deltas: got %v, want %v", got.text, want)
	}
	if resp.StopReason != llm.StopEndTurn {
		t.Errorf("stop reason: got %q, want %q", resp.StopReason, llm.StopEndTurn)
	}
	if resp.Model != "served" {
		t.Errorf("model: got %q, want %q", resp.Model, "served")
	}
	want := llm.Usage{InputTokens: 50, OutputTokens: 5, CacheReadTokens: 150}
	if resp.Usage != want {
		t.Errorf("usage: got %+v, want %+v", resp.Usage, want)
	}
}

// Some servers attach usage to the finish_reason chunk instead of a trailing
// one, and empty usage objects appear on intermediate chunks. Only a non-empty
// one may overwrite.
func TestStreamUsageOnTheFinishChunk(t *testing.T) {
	resp, err := decodeFrames(t, nil,
		`{"choices":[{"index":0,"delta":{"content":"x"},"usage":{"prompt_tokens":0,"completion_tokens":0}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":3}}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}
	if want := (llm.Usage{InputTokens: 11, OutputTokens: 3}); resp.Usage != want {
		t.Errorf("usage: got %+v, want %+v", resp.Usage, want)
	}
}

// TestStreamToolCallFragmentsAreKeyedByIndex: id and name arrive once, in the
// first fragment; arguments arrive afterwards as bare slices with neither, and
// two parallel calls interleave. Index — not id — is the stable key.
func TestStreamToolCallFragmentsAreKeyedByIndex(t *testing.T) {
	var got collector
	resp, err := decodeFrames(t, got.sink(),
		`{"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"calculator","arguments":"{\"expr"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"weather","arguments":"{\"ci"}}]}}]}`,
		// interleaved, out of index order
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"ty\":\"SF\"}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ession\":\"6*7\"}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}

	wantDeltas := []llm.ToolArgsDelta{
		{Index: 0, ID: "call_a", Name: "calculator", JSON: `{"expr`},
		{Index: 1, ID: "call_b", Name: "weather", JSON: `{"ci`},
		{Index: 1, JSON: `ty":"SF"}`},
		{Index: 0, JSON: `ession":"6*7"}`},
	}
	if !reflect.DeepEqual(got.args, wantDeltas) {
		t.Errorf("tool args deltas:\n got %+v\nwant %+v", got.args, wantDeltas)
	}

	// Blocks come out in INDEX order regardless of arrival order, so two
	// replays of one stream produce the same transcript.
	uses := llm.ToolUses(resp.Content)
	if len(uses) != 2 {
		t.Fatalf("tool uses: got %d, want 2", len(uses))
	}
	if uses[0].ID != "call_a" || uses[1].ID != "call_b" {
		t.Errorf("tool use order: got %q,%q, want call_a,call_b", uses[0].ID, uses[1].ID)
	}
	if got, want := string(uses[0].Input), `{"expression":"6*7"}`; got != want {
		t.Errorf("call_a input: got %s, want %s", got, want)
	}
	if got, want := string(uses[1].Input), `{"city":"SF"}`; got != want {
		t.Errorf("call_b input: got %s, want %s", got, want)
	}
	if resp.StopReason != llm.StopToolUse {
		t.Errorf("stop reason: got %q, want %q", resp.StopReason, llm.StopToolUse)
	}
	// Argument fragments are never merged into Text — they are not the answer.
	if len(got.text) != 0 {
		t.Errorf("text deltas: got %v, want none", got.text)
	}
}

// TestRepeatedIDGatewayAnnouncesExactlyOnce: some gateways repeat id and name
// on EVERY fragment. The shared contract says a consumer sees them exactly
// once, so "ID non-empty" can mean "a new call has started".
func TestRepeatedIDGatewayAnnouncesExactlyOnce(t *testing.T) {
	var got collector
	resp, err := decodeFrames(t, got.sink(),
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"1,"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"\"b\":2}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}

	want := []llm.ToolArgsDelta{
		{Index: 0, ID: "call_1", Name: "f", JSON: `{"a":`},
		{Index: 0, JSON: `1,`},
		{Index: 0, JSON: `"b":2}`},
	}
	if !reflect.DeepEqual(got.args, want) {
		t.Errorf("tool args deltas:\n got %+v\nwant %+v", got.args, want)
	}

	ids, names := 0, 0
	for _, d := range got.args {
		if d.ID != "" {
			ids++
		}
		if d.Name != "" {
			names++
		}
	}
	if ids != 1 {
		t.Errorf("ID announcements: got %d, want exactly 1", ids)
	}
	if names != 1 {
		t.Errorf("Name announcements: got %d, want exactly 1", names)
	}

	uses := llm.ToolUses(resp.Content)
	if len(uses) != 1 {
		t.Fatalf("tool uses: got %d, want 1 — a repeated id must not fan out into several calls", len(uses))
	}
	if got, want := string(uses[0].Input), `{"a":1,"b":2}`; got != want {
		t.Errorf("input: got %s, want %s", got, want)
	}
}

// A gateway that announces the call with no arguments must still emit the
// announcement, and a bare fragment with nothing in it must emit nothing.
func TestStreamAnnouncementWithoutArguments(t *testing.T) {
	var got collector
	_, err := decodeFrames(t, got.sink(),
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}
	want := []llm.ToolArgsDelta{
		{Index: 0, ID: "call_1", Name: "f"},
		{Index: 0, JSON: `{}`},
	}
	if !reflect.DeepEqual(got.args, want) {
		t.Errorf("tool args deltas:\n got %+v\nwant %+v (an empty fragment says nothing and must not be emitted)", got.args, want)
	}
}

// A fragment with no "index" field defaults to 0 rather than being dropped.
func TestStreamToolCallWithoutAnIndex(t *testing.T) {
	resp, err := decodeFrames(t, nil,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","function":{"name":"f","arguments":"{\"a\":1}"}}]}}]}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}
	uses := llm.ToolUses(resp.Content)
	if len(uses) != 1 || uses[0].ID != "call_1" {
		t.Fatalf("tool uses: got %v, want one call_1", uses)
	}
	if got, want := string(uses[0].Input), `{"a":1}`; got != want {
		t.Errorf("input: got %s, want %s", got, want)
	}
}

func TestStreamReasoningIsNeverMergedIntoText(t *testing.T) {
	var got collector
	resp, err := decodeFrames(t, got.sink(),
		`{"choices":[{"index":0,"delta":{"reasoning_content":"let me "}}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"the answer"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}
	if want := []string{"let me ", "think"}; !reflect.DeepEqual(got.reasoning, want) {
		t.Errorf("reasoning deltas: got %v, want %v", got.reasoning, want)
	}
	if want := []string{"the answer"}; !reflect.DeepEqual(got.text, want) {
		t.Errorf("text deltas: got %v, want %v", got.text, want)
	}
	if len(resp.Content) != 2 {
		t.Fatalf("content: got %d blocks (%v), want 2", len(resp.Content), resp.Content)
	}
	// Reasoning leads, matching decodeResponse and the model's own order.
	th, ok := resp.Content[0].(llm.Thinking)
	if !ok {
		t.Fatalf("block 0: got %T, want llm.Thinking", resp.Content[0])
	}
	if th.Text != "let me think" {
		t.Errorf("reasoning: got %q, want %q", th.Text, "let me think")
	}
	if got, want := llm.TextOf(resp.Content), "the answer"; got != want {
		t.Errorf("TextOf: got %q, want %q — the scratchpad must not be quoted back at the user", got, want)
	}
}

// TestStreamResponseIsIdenticalWithoutASink is the property the task names:
// deltas are display-only, so dropping every one of them must change nothing
// about what the harness dispatches.
func TestStreamResponseIsIdenticalWithoutASink(t *testing.T) {
	frames := []string{
		`{"model":"served","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"pre"}}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"amble"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"he"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"llo"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\"a\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}`,
		doneSentinel,
	}

	var seen collector
	withSink, err1 := decodeFrames(t, seen.sink(), frames...)
	if err1 != nil {
		t.Fatalf("decodeStream with sink: got error %v, want nil", err1)
	}
	withoutSink, err2 := decodeFrames(t, nil, frames...)
	if err2 != nil {
		t.Fatalf("decodeStream without sink: got error %v, want nil", err2)
	}
	if !reflect.DeepEqual(withSink, withoutSink) {
		t.Errorf("Response differs by whether OnDelta was set:\n with sink: %+v\nwithout:   %+v", withSink, withoutSink)
	}
	a, _ := json.Marshal(llm.Message{Role: llm.RoleAssistant, Content: withSink.Content})
	b, _ := json.Marshal(llm.Message{Role: llm.RoleAssistant, Content: withoutSink.Content})
	if string(a) != string(b) {
		t.Errorf("encoded content differs:\n with sink: %s\nwithout:   %s", a, b)
	}
	if len(seen.all) == 0 {
		t.Error("the sink saw nothing; the comparison above would be vacuous")
	}
}

func TestStreamErrors(t *testing.T) {
	t.Run("error envelope mid-stream after a 200", func(t *testing.T) {
		_, err := decodeFrames(t, nil,
			`{"choices":[{"index":0,"delta":{"content":"partial"}}]}`,
			`{"error":{"type":"server_error","message":"model died"}}`,
			doneSentinel,
		)
		if err == nil {
			t.Fatal("decodeStream: got nil error, want one — a truncated prefix must not be returned as the answer")
		}
		if !errors.Is(err, llm.ErrServer) {
			t.Errorf("errors.Is(err, llm.ErrServer): got false, want true (err=%v)", err)
		}
		if !strings.Contains(err.Error(), "model died") {
			t.Errorf("error text: got %q, want it to carry the provider detail", err)
		}
	})

	t.Run("an SSE event named error", func(t *testing.T) {
		_, err := decodeFrames(t, nil, ev("error", `{"message":"upstream gone"}`))
		if !errors.Is(err, llm.ErrServer) {
			t.Errorf("errors.Is(err, llm.ErrServer): got false, want true (err=%v)", err)
		}
	})

	t.Run("a 200 followed by silence is transport", func(t *testing.T) {
		_, err := decodeStream(strings.NewReader(""), nil, "m")
		if !errors.Is(err, llm.ErrTransport) {
			t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
		}
		if !llm.Retryable(err) {
			t.Errorf("llm.Retryable: got false, want true — a wedged gateway is worth another attempt")
		}
	})

	t.Run("undecodable chunks are skipped, not fatal", func(t *testing.T) {
		resp, err := decodeFrames(t, nil,
			`ping`,
			`{"choices":[{"index":0,"delta":{"content":"ok"}}]}`,
			`{ not json`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			doneSentinel,
		)
		if err != nil {
			t.Fatalf("decodeStream: got error %v, want nil", err)
		}
		if got, want := llm.TextOf(resp.Content), "ok"; got != want {
			t.Errorf("text: got %q, want %q", got, want)
		}
	})
}

// TestStreamWithOnlyDONE documents current behaviour rather than blessing it:
// a gateway that sends [DONE] and nothing else yields an EMPTY successful
// response — no content, no stop reason — because the "closed before any
// event" guard is skipped once the sentinel was seen. The agent loop then sees
// an empty assistant turn instead of a retryable transport failure.
func TestStreamWithOnlyDONE(t *testing.T) {
	resp, err := decodeFrames(t, nil, doneSentinel)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil (current behaviour)", err)
	}
	if len(resp.Content) != 0 {
		t.Errorf("content: got %v, want none", resp.Content)
	}
	if resp.StopReason != "" {
		t.Errorf("stop reason: got %q, want empty", resp.StopReason)
	}
}

func TestStreamTruncatedArgumentsArePreserved(t *testing.T) {
	resp, err := decodeFrames(t, nil,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write","arguments":"{\"body\":\"half a fi"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}
	uses := llm.ToolUses(resp.Content)
	if len(uses) != 1 {
		t.Fatalf("tool uses: got %d, want 1", len(uses))
	}
	if !json.Valid(uses[0].Input) {
		t.Errorf("input is not valid JSON: %s", uses[0].Input)
	}
	var s string
	if err := json.Unmarshal(uses[0].Input, &s); err != nil {
		t.Fatalf("input should be a JSON string carrying the fragment; got %s", uses[0].Input)
	}
	if !strings.Contains(s, "half a fi") {
		t.Errorf("preserved fragment: got %q, want it to contain the truncated arguments", s)
	}
	if resp.StopReason != llm.StopMaxTokens {
		t.Errorf("stop reason: got %q, want %q", resp.StopReason, llm.StopMaxTokens)
	}
}

// A gateway that echoes content as typed parts must still stream and assemble.
func TestStreamContentAsTypedParts(t *testing.T) {
	var got collector
	resp, err := decodeFrames(t, got.sink(),
		`{"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"part one"}]}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		doneSentinel,
	)
	if err != nil {
		t.Fatalf("decodeStream: got error %v, want nil", err)
	}
	if want := "part one"; llm.TextOf(resp.Content) != want {
		t.Errorf("text: got %q, want %q", llm.TextOf(resp.Content), want)
	}
	if want := []string{"part one"}; !reflect.DeepEqual(got.text, want) {
		t.Errorf("text deltas: got %v, want %v", got.text, want)
	}
}
