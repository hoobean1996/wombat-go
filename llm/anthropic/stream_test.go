package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
)

// streamOnce runs one streamed Complete against a canned event list.
func streamOnce(t *testing.T, events []string, onDelta func(llm.Delta)) (llm.Response, error) {
	t.Helper()
	srv, _ := newServer(t, sseReply(events...))
	c := newTestClient(t, srv, func(cfg *Config) { cfg.Stream = llm.StreamAlways })
	return c.Complete(context.Background(), llm.Request{
		Messages: []llm.Message{userTurn("hi")},
		OnDelta:  onDelta,
	})
}

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

const (
	evStart      = `{"type":"message_start","message":{"model":"claude-served","usage":{"input_tokens":1200,"cache_read_input_tokens":800,"cache_creation_input_tokens":40}}}`
	evStopEnd    = `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":37}}`
	evStopTool   = `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}`
	evMessageEnd = `{"type":"message_stop"}`
)

func TestStreamTextAndUsageMerge(t *testing.T) {
	var got collector
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"It is "}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"42."}}`),
		ev("content_block_stop", `{"type":"content_block_stop","index":0}`),
		ev("message_delta", evStopEnd),
		ev("message_stop", evMessageEnd),
	}, got.sink())
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
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
	if resp.Model != "claude-served" {
		t.Errorf("model: got %q, want the served id %q", resp.Model, "claude-served")
	}

	// THE merge: message_start carries the input side (including the cache
	// split that says whether the breakpoints in request.go are earning their
	// keep), message_delta carries only the output count. A blind copy on
	// message_delta would erase all three input numbers.
	want := llm.Usage{InputTokens: 1200, OutputTokens: 37, CacheWriteTokens: 40, CacheReadTokens: 800}
	if resp.Usage != want {
		t.Errorf("usage: got %+v, want %+v", resp.Usage, want)
	}
}

// TestStreamUsageDeltaDoesNotEraseInput is the narrow regression: a
// message_delta whose usage object repeats input_tokens:0 must leave the
// message_start numbers alone.
func TestStreamUsageDeltaDoesNotEraseInput(t *testing.T) {
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"x"}}`),
		ev("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":0,"output_tokens":5,"cache_read_input_tokens":0}}`),
	}, nil)
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	want := llm.Usage{InputTokens: 1200, OutputTokens: 5, CacheWriteTokens: 40, CacheReadTokens: 800}
	if resp.Usage != want {
		t.Errorf("usage: got %+v, want %+v — a zero in message_delta must not clobber message_start", resp.Usage, want)
	}
}

func TestStreamThinkingAndSignature(t *testing.T) {
	var got collector
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step one, "}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step two"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}`),
		ev("content_block_stop", `{"type":"content_block_stop","index":0}`),
		ev("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"the answer"}}`),
		ev("message_delta", evStopEnd),
	}, got.sink())
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}

	if len(resp.Content) != 2 {
		t.Fatalf("content blocks: got %d (%v), want 2", len(resp.Content), resp.Content)
	}
	th, ok := resp.Content[0].(llm.Thinking)
	if !ok {
		t.Fatalf("block 0: got %T, want llm.Thinking", resp.Content[0])
	}
	if want := "step one, step two"; th.Text != want {
		t.Errorf("thinking text: got %q, want %q", th.Text, want)
	}
	// The signature must be reassembled from its own deltas: the API rejects
	// the next turn if a thinking block comes back without the signature it
	// was issued with.
	if want := "sig-abc"; th.Signature != want {
		t.Errorf("thinking signature: got %q, want %q", th.Signature, want)
	}

	// Reasoning rides its own Delta field and is NEVER merged into Text: it is
	// the model's scratchpad, not the answer.
	if want := []string{"step one, ", "step two"}; !reflect.DeepEqual(got.reasoning, want) {
		t.Errorf("reasoning deltas: got %v, want %v", got.reasoning, want)
	}
	if want := []string{"the answer"}; !reflect.DeepEqual(got.text, want) {
		t.Errorf("text deltas: got %v, want %v (thinking must not leak into Text)", got.text, want)
	}
	// And a signature delta says nothing to a UI, so it emits nothing.
	for _, d := range got.all {
		if d.Text == "" && d.Reasoning == "" && d.ToolArgs == nil {
			t.Errorf("an empty delta was emitted: %+v", d)
		}
	}
	if llm.TextOf(resp.Content) != "the answer" {
		t.Errorf("TextOf: got %q, want %q", llm.TextOf(resp.Content), "the answer")
	}
}

// TestStreamToolArgsContract pins the shared rule that llm/openai implements
// too: Index on every fragment, ID and Name exactly once, JSON never merged
// into Text.
func TestStreamToolArgsContract(t *testing.T) {
	var got collector
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"let me compute"}}`),
		ev("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"calculator","input":{}}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"expre"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"ssion\":\"6*7\"}"}}`),
		ev("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu_2","name":"weather","input":{}}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"SF\"}"}}`),
		ev("message_delta", evStopTool),
	}, got.sink())
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}

	// Fragments per index, in order, with the announcement first and carrying
	// no JSON (Anthropic's content_block_start always has input:{}).
	want := []llm.ToolArgsDelta{
		{Index: 1, ID: "tu_1", Name: "calculator"},
		{Index: 1, JSON: `{"expre`},
		{Index: 1, JSON: `ssion":"6*7"}`},
		{Index: 2, ID: "tu_2", Name: "weather"},
		{Index: 2, JSON: `{"city":"SF"}`},
	}
	if !reflect.DeepEqual(got.args, want) {
		t.Errorf("tool args deltas:\n got %+v\nwant %+v", got.args, want)
	}

	// ID and Name exactly once per index.
	ids := map[int]int{}
	names := map[int]int{}
	for _, d := range got.args {
		if d.ID != "" {
			ids[d.Index]++
		}
		if d.Name != "" {
			names[d.Index]++
		}
	}
	for _, idx := range []int{1, 2} {
		if ids[idx] != 1 {
			t.Errorf("index %d announced its ID %d times, want exactly 1", idx, ids[idx])
		}
		if names[idx] != 1 {
			t.Errorf("index %d announced its Name %d times, want exactly 1", idx, names[idx])
		}
	}

	// Argument fragments must never surface as answer text.
	if joined := strings.Join(got.text, ""); joined != "let me compute" {
		t.Errorf("text deltas: got %q, want %q — tool args must never be merged into Text", joined, "let me compute")
	}

	// And the assembled response carries the reassembled calls, in index order.
	uses := llm.ToolUses(resp.Content)
	if len(uses) != 2 {
		t.Fatalf("tool uses: got %d, want 2", len(uses))
	}
	if got, want := string(uses[0].Input), `{"expression":"6*7"}`; got != want {
		t.Errorf("call 0 input: got %s, want %s", got, want)
	}
	if got, want := string(uses[1].Input), `{"city":"SF"}`; got != want {
		t.Errorf("call 1 input: got %s, want %s", got, want)
	}
	if uses[0].ID != "tu_1" || uses[1].ID != "tu_2" {
		t.Errorf("tool use ids: got %q,%q, want tu_1,tu_2", uses[0].ID, uses[1].ID)
	}
}

// TestStreamOrphanArgsFragmentIsNotEmitted: input_json_delta also carries the
// arguments of SERVER-side tools, whose content_block_start this package does
// not model. Forwarding an index the UI never saw opened would be an orphan.
func TestStreamOrphanArgsFragmentIsNotEmitted(t *testing.T) {
	var got collector
	_, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_search"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}`),
		ev("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}`),
		ev("message_delta", evStopEnd),
	}, got.sink())
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if len(got.args) != 0 {
		t.Errorf("tool args deltas: got %+v, want none for an unannounced (server-side) block", got.args)
	}
}

// TestStreamUnmodelledBlockIsDropped: an unknown content_block type leaves
// kind empty and must not appear in the assembled Response.
func TestStreamUnmodelledBlockIsDropped(t *testing.T) {
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}`),
		ev("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":"visible"}}`),
		ev("message_delta", evStopEnd),
	}, nil)
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("content: got %d blocks (%v), want 1", len(resp.Content), resp.Content)
	}
	if got, want := llm.TextOf(resp.Content), "visible"; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
}

// TestStreamEmptyTextBlockIsDropped: an empty text block is legal to receive
// and illegal to send back, and this reply goes straight into the transcript
// that becomes the next request.
func TestStreamEmptyTextBlockIsDropped(t *testing.T) {
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		ev("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"t","input":{}}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}`),
		ev("message_delta", evStopTool),
	}, nil)
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	for _, b := range resp.Content {
		if txt, ok := b.(llm.Text); ok {
			t.Errorf("an empty text block survived into the response: %q", txt.Text)
		}
	}
	if len(resp.Content) != 1 {
		t.Errorf("content: got %d blocks (%v), want just the tool use", len(resp.Content), resp.Content)
	}
}

// TestStreamResponseIsIdenticalWithoutASink is the property the task names:
// deltas are display-only, so the assembled Response must not depend on
// whether anybody was listening.
func TestStreamResponseIsIdenticalWithoutASink(t *testing.T) {
	events := []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"pre"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"amble"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`),
		ev("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":"he"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"llo"}}`),
		ev("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu_1","name":"calc","input":{}}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"1}"}}`),
		ev("message_delta", evStopTool),
		ev("message_stop", evMessageEnd),
	}

	var seen collector
	withSink, err1 := streamOnce(t, events, seen.sink())
	if err1 != nil {
		t.Fatalf("Complete with sink: got error %v, want nil", err1)
	}
	withoutSink, err2 := streamOnce(t, events, nil)
	if err2 != nil {
		t.Fatalf("Complete without sink: got error %v, want nil", err2)
	}

	if !reflect.DeepEqual(withSink, withoutSink) {
		t.Errorf("Response differs by whether OnDelta was set:\n with sink: %+v\nwithout:   %+v", withSink, withoutSink)
	}
	// Byte-identical through the domain JSON encoding too.
	a, _ := json.Marshal(llm.Message{Role: llm.RoleAssistant, Content: withSink.Content})
	b, _ := json.Marshal(llm.Message{Role: llm.RoleAssistant, Content: withoutSink.Content})
	if string(a) != string(b) {
		t.Errorf("encoded content differs:\n with sink: %s\nwithout:   %s", a, b)
	}
	if len(seen.all) == 0 {
		t.Error("the sink saw nothing; the comparison above would be vacuous")
	}
}

// TestStreamTruncatedToolArgsArePreserved: a generation cut off mid-argument
// leaves unparseable JSON. Dropping it produces a dead turn; keeping it as a
// JSON string round-trips and lets the tool's schema validation tell the model
// what went wrong.
func TestStreamTruncatedToolArgsArePreserved(t *testing.T) {
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"write","input":{}}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"/tmp/x\",\"body\":\"half a fi"}}`),
		ev("message_delta", `{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":8192}}`),
	}, nil)
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	uses := llm.ToolUses(resp.Content)
	if len(uses) != 1 {
		t.Fatalf("tool uses: got %d, want 1", len(uses))
	}
	if !json.Valid(uses[0].Input) {
		t.Errorf("input is not valid JSON: %s — raw invalid JSON in a RawMessage fails the whole next request", uses[0].Input)
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

func TestStreamToolInputDefaultsToEmptyObject(t *testing.T) {
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"noargs","input":{}}}`),
		ev("message_delta", evStopTool),
	}, nil)
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	uses := llm.ToolUses(resp.Content)
	if len(uses) != 1 {
		t.Fatalf("tool uses: got %d, want 1", len(uses))
	}
	if got, want := string(uses[0].Input), "{}"; got != want {
		t.Errorf("input: got %s, want %s", got, want)
	}
}

func TestStreamErrorEventAborts(t *testing.T) {
	_, err := streamOnce(t, []string{
		ev("message_start", evStart),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"partial"}}`),
		ev("error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`),
	}, nil)
	if err == nil {
		t.Fatal("Complete: got nil error, want the in-band stream error")
	}
	if !errors.Is(err, llm.ErrOverloaded) {
		t.Errorf("errors.Is(err, llm.ErrOverloaded): got false, want true (err=%v)", err)
	}
	if !llm.Retryable(err) {
		t.Errorf("llm.Retryable: got false, want true — an overloaded_error mid-stream is the most common way a long generation dies")
	}
}

// TestStreamEventLineOnlyStillWorks: gateways have been known to drop the
// "event:" line, and the JSON "type" is what actually decides.
func TestStreamTypeFromJSONWins(t *testing.T) {
	resp, err := streamOnce(t, []string{
		// no event: line at all
		evStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"bare"}}`,
		evStopEnd,
	}, nil)
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if got, want := llm.TextOf(resp.Content), "bare"; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
}

// TestStreamFallsBackToTheEventLine: and the reverse — a payload with no
// "type" field falls back to the SSE event name.
func TestStreamFallsBackToTheEventLine(t *testing.T) {
	resp, err := streamOnce(t, []string{
		ev("message_start", `{"message":{"model":"m","usage":{"input_tokens":7}}}`),
		ev("content_block_start", `{"index":0,"content_block":{"type":"text","text":"x"}}`),
		ev("message_delta", `{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`),
	}, nil)
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if got, want := llm.TextOf(resp.Content), "x"; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
	if got, want := (llm.Usage{InputTokens: 7, OutputTokens: 2}), resp.Usage; got != want {
		t.Errorf("usage: got %+v, want %+v", want, got)
	}
}

func TestStreamGarbageAndUnknownEventsAreIgnored(t *testing.T) {
	resp, err := streamOnce(t, []string{
		ev("ping", `{"type":"ping"}`),
		`not json at all`,
		ev("message_start", evStart),
		ev("some_future_event", `{"type":"some_future_event","payload":1}`),
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"ok"}}`),
		ev("message_delta", evStopEnd),
	}, nil)
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if got, want := llm.TextOf(resp.Content), "ok"; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
}

// TestStreamEmptyBodyIsTransport: a 200 with no events is a proxy artifact,
// never the model, so it must be retryable rather than an empty answer.
func TestStreamEmptyBodyIsTransport(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter, _ int) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
	})
	c := newTestClient(t, srv, func(cfg *Config) { cfg.Stream = llm.StreamAlways })
	_, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
	if !errors.Is(err, llm.ErrTransport) {
		t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
	}
	if !llm.Retryable(err) {
		t.Errorf("llm.Retryable: got false, want true")
	}
}

// ===== the non-streaming decode path =====

func TestDecodeMessage(t *testing.T) {
	srv, _ := newServer(t, okJSON(`{
		"model":"claude-served",
		"content":[
			{"type":"thinking","thinking":"hmm","signature":"sig"},
			{"type":"text","text":"the answer"},
			{"type":"tool_use","id":"tu_1","name":"calc","input":{"expression":"6*7"}},
			{"type":"web_search_result","url":"https://example.com"}
		],
		"stop_reason":"tool_use",
		"usage":{"input_tokens":10,"output_tokens":3,"cache_read_input_tokens":5}
	}`))
	c := newTestClient(t, srv, nil)
	resp, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if len(resp.Content) != 3 {
		t.Fatalf("content: got %d blocks (%v), want 3 (the unknown block must be dropped, not fatal)", len(resp.Content), resp.Content)
	}
	if got, want := resp.Content[0].(llm.Thinking).Signature, "sig"; got != want {
		t.Errorf("signature: got %q, want %q", got, want)
	}
	// Tool input bytes must survive untouched: they get replayed into the next
	// request's cache prefix, and reordering them would break the match.
	if got, want := string(resp.Content[2].(llm.ToolUse).Input), `{"expression":"6*7"}`; got != want {
		t.Errorf("tool input: got %s, want %s", got, want)
	}
	if resp.StopReason != llm.StopToolUse {
		t.Errorf("stop reason: got %q, want %q", resp.StopReason, llm.StopToolUse)
	}
	if want := (llm.Usage{InputTokens: 10, OutputTokens: 3, CacheReadTokens: 5}); resp.Usage != want {
		t.Errorf("usage: got %+v, want %+v", resp.Usage, want)
	}
	if resp.Model != "claude-served" {
		t.Errorf("model: got %q, want %q", resp.Model, "claude-served")
	}
}

func TestDecodeMessageFallsBackToRequestedModel(t *testing.T) {
	srv, _ := newServer(t, okJSON(`{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn"}`))
	c := newTestClient(t, srv, nil)
	resp, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if got, want := resp.Model, "test-model"; got != want {
		t.Errorf("model: got %q, want the requested id %q", got, want)
	}
}

func TestDecodeMessageBadJSONIsTransport(t *testing.T) {
	srv, _ := newServer(t, okJSON(`<html>502 Bad Gateway</html>`))
	c := newTestClient(t, srv, nil)
	_, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
	if !errors.Is(err, llm.ErrTransport) {
		t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v) — a 200 that is not our JSON is worth a retry", err)
	}
}

// TestScrubRawKeepsToolInputUsable is the poisoned-RawMessage case: invalid
// UTF-8 inside a tool_use input bypasses json.Marshal on the way back out
// (RawMessage is copied verbatim) and 400s every subsequent request forever.
func TestScrubRawKeepsToolInputUsable(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter, _ int) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("{\"content\":[{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"n\",\"input\":{\"path\":\"a\xffb\"}}],\"stop_reason\":\"tool_use\"}"))
	})
	c := newTestClient(t, srv, nil)
	resp, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	uses := llm.ToolUses(resp.Content)
	if len(uses) != 1 {
		t.Fatalf("tool uses: got %d, want 1", len(uses))
	}
	in := string(uses[0].Input)
	if strings.ContainsRune(in, 0xff) {
		t.Errorf("tool input still carries the invalid byte: %q", in)
	}
	if !strings.Contains(in, "�") {
		t.Errorf("tool input: got %q, want the invalid byte replaced with U+FFFD", in)
	}
	// And the scrubbed bytes must still round-trip into the next request.
	body, err := c.encode(llm.Request{Messages: []llm.Message{
		llm.UserText("q"),
		{Role: llm.RoleAssistant, Content: resp.Content},
	}}, "m", 100, false)
	if err != nil {
		t.Fatalf("re-encode: got error %v, want nil", err)
	}
	if !json.Valid(body) {
		t.Errorf("re-encoded request is not valid JSON: %s", body)
	}
}

func TestScrubRawEmptyInput(t *testing.T) {
	if got, want := string(scrubRaw(nil)), "{}"; got != want {
		t.Errorf("scrubRaw(nil): got %s, want %s", got, want)
	}
	raw := json.RawMessage(`{"a":1}`)
	if got := scrubRaw(raw); string(got) != string(raw) {
		t.Errorf("scrubRaw: got %s, want it unchanged %s", got, raw)
	}
}

// TestStreamHostileIndices: a buggy gateway can send a negative or
// non-contiguous content-block index. Neither may panic, and a gap must not
// produce a nil block in the assembled response.
func TestStreamHostileIndices(t *testing.T) {
	var got collector
	resp, err := streamOnce(t, []string{
		ev("message_start", evStart),
		// negative index: ignored, never indexed into
		ev("content_block_start", `{"type":"content_block_start","index":-1,"content_block":{"type":"text","text":"nope"}}`),
		ev("content_block_delta", `{"type":"content_block_delta","index":-1,"delta":{"type":"text_delta","text":"nope"}}`),
		// a tool_use announced with neither id nor name says nothing to a UI
		ev("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","input":{}}}`),
		// index 3 with no start at all: indices 0 and 2 stay nil
		ev("content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"orphan"}}`),
		ev("content_block_start", `{"type":"content_block_start","index":4,"content_block":{"type":"text","text":"real"}}`),
		// a content_block_delta with no delta object at all
		ev("content_block_delta", `{"type":"content_block_delta","index":4}`),
		// a message_start with no message
		ev("message_start", `{"type":"message_start"}`),
		ev("message_delta", evStopEnd),
	}, got.sink())
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	for _, d := range got.args {
		if d.Index == 1 {
			t.Errorf("an anonymous tool_use was announced: %+v, want no delta at all", d)
		}
	}
	// Index 3 never had a content_block_start, so it has no kind and is
	// dropped; the nil gaps left at 0 and 2 by growing the slice must be
	// skipped rather than emitted as nil blocks.
	if got, want := llm.TextOf(resp.Content), "real"; got != want {
		t.Errorf("text: got %q, want %q (blocks: %v)", got, want, resp.Content)
	}
	// Two blocks survive: the anonymous tool_use at index 1 and the text at
	// index 4. Note the asymmetry, pinned deliberately — the tool-args DELTA
	// for an id-less call is suppressed (it says nothing to a UI) but the block
	// itself still materialises, as an llm.ToolUse with an empty ID and Name.
	// A dispatcher would fail that call and be unable to pair the result. Real
	// Anthropic always sends id and name, so this only bites behind a broken
	// gateway.
	if len(resp.Content) != 2 {
		t.Fatalf("content: got %d blocks (%v), want 2", len(resp.Content), resp.Content)
	}
	anon, ok := resp.Content[0].(llm.ToolUse)
	if !ok {
		t.Fatalf("block 0: got %T, want llm.ToolUse", resp.Content[0])
	}
	if anon.ID != "" || anon.Name != "" {
		t.Errorf("anonymous tool use: got id=%q name=%q, want both empty", anon.ID, anon.Name)
	}
}

// TestStreamTruncatedConnectionIsTransport: a stream that dies mid-body must
// be reported as retryable transport, not returned as a short answer.
func TestStreamTruncatedConnectionIsTransport(t *testing.T) {
	srv, _ := newServer(t, func(w http.ResponseWriter, _ int) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		fmt.Fprintf(w, "data: %s\n\n", evStart)
		f.Flush()
		// Abort the connection without a clean end of stream.
		panic(http.ErrAbortHandler)
	})
	c := newTestClient(t, srv, func(cfg *Config) { cfg.Stream = llm.StreamAlways })
	_, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
	if err == nil {
		t.Fatal("Complete: got nil error, want a truncated-stream failure")
	}
	if !errors.Is(err, llm.ErrTransport) {
		t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
	}
}

// TestCompleteTransportFailure: a dead endpoint is a transport failure the
// retry middleware can act on, not a bad request.
func TestCompleteTransportFailure(t *testing.T) {
	clearEnv(t)
	srv, _ := newServer(t, okJSON(minimalMessage))
	c := newTestClient(t, srv, nil)
	srv.Close() // nothing is listening any more

	_, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
	if !errors.Is(err, llm.ErrTransport) {
		t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
	}
	if !llm.Retryable(err) {
		t.Errorf("llm.Retryable: got false, want true (err=%v)", err)
	}
}
