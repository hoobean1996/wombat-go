package openai

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
)

func decodeJSON(t *testing.T, body string) (llm.Response, error) {
	t.Helper()
	return decodeResponse(strings.NewReader(body), "requested-model")
}

func TestFinishReasonMapping(t *testing.T) {
	tests := []struct {
		finish string
		want   llm.StopReason
	}{
		{"stop", llm.StopEndTurn},
		{"tool_calls", llm.StopToolUse},
		{"length", llm.StopMaxTokens},
		{"content_filter", llm.StopRefusal},
		// Gateways invent reasons; flattening them into one bucket would
		// destroy the only diagnostic the caller has, so they pass through.
		{"eos", llm.StopReason("eos")},
		{"function_call", llm.StopReason("function_call")},
		{"", llm.StopReason("")},
	}
	for _, tt := range tests {
		t.Run(tt.finish, func(t *testing.T) {
			if got := stopReason(tt.finish); got != tt.want {
				t.Errorf("stopReason(%q): got %q, want %q", tt.finish, got, tt.want)
			}
			resp, err := decodeJSON(t, `{"choices":[{"message":{"content":"x"},"finish_reason":"`+tt.finish+`"}]}`)
			if err != nil {
				t.Fatalf("decodeResponse: got error %v, want nil", err)
			}
			if resp.StopReason != tt.want {
				t.Errorf("decoded stop reason: got %q, want %q", resp.StopReason, tt.want)
			}
		})
	}
}

// TestCachedTokensAreSubtractedFromPromptTokens is the accounting bug this
// function exists to prevent. OpenAI-compatible servers count cached prompt
// tokens INSIDE prompt_tokens; llm.Usage is the Anthropic shape, where
// InputTokens and CacheReadTokens are disjoint and billed at DIFFERENT rates.
// Passing prompt_tokens through unchanged bills the cached prefix twice.
func TestCachedTokensAreSubtractedFromPromptTokens(t *testing.T) {
	tests := []struct {
		name string
		body *usageBody
		want llm.Usage
	}{
		{name: "nil usage", body: nil, want: llm.Usage{}},
		{
			name: "no cache reporting",
			body: &usageBody{PromptTokens: 1000, CompletionTokens: 50},
			want: llm.Usage{InputTokens: 1000, OutputTokens: 50},
		},
		{
			name: "OpenAI nested spelling",
			body: &usageBody{
				PromptTokens:     1000,
				CompletionTokens: 50,
				PromptTokensDetails: &struct {
					CachedTokens int `json:"cached_tokens"`
				}{CachedTokens: 800},
			},
			want: llm.Usage{InputTokens: 200, OutputTokens: 50, CacheReadTokens: 800},
		},
		{
			name: "DeepSeek top-level spelling",
			body: &usageBody{PromptTokens: 1000, CompletionTokens: 50, PromptCacheHitTokens: 640},
			want: llm.Usage{InputTokens: 360, OutputTokens: 50, CacheReadTokens: 640},
		},
		{
			name: "the top-level spelling wins when both are present",
			body: &usageBody{
				PromptTokens:         1000,
				PromptCacheHitTokens: 640,
				PromptTokensDetails: &struct {
					CachedTokens int `json:"cached_tokens"`
				}{CachedTokens: 100},
			},
			want: llm.Usage{InputTokens: 360, CacheReadTokens: 640},
		},
		{
			name: "a gateway reporting more hits than prompt tokens is clamped",
			body: &usageBody{PromptTokens: 100, PromptCacheHitTokens: 900},
			want: llm.Usage{InputTokens: 0, CacheReadTokens: 900},
		},
		{
			// Reasoning tokens are counted INSIDE completion_tokens by the
			// servers that report them, so folding them in would double-bill.
			name: "reasoning tokens are not added to the output count",
			body: &usageBody{
				PromptTokens:     10,
				CompletionTokens: 108,
				CompletionTokensDetails: &struct {
					ReasoningTokens int `json:"reasoning_tokens"`
				}{ReasoningTokens: 34},
			},
			want: llm.Usage{InputTokens: 10, OutputTokens: 108},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.body.toUsage(); got != tt.want {
				t.Errorf("toUsage: got %+v, want %+v", got, tt.want)
			}
		})
	}

	t.Run("end to end", func(t *testing.T) {
		resp, err := decodeJSON(t, `{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1000,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":800}}}`)
		if err != nil {
			t.Fatalf("decodeResponse: got error %v, want nil", err)
		}
		want := llm.Usage{InputTokens: 200, OutputTokens: 50, CacheReadTokens: 800}
		if resp.Usage != want {
			t.Errorf("usage: got %+v, want %+v", resp.Usage, want)
		}
	})
}

func TestDecodeResponseContent(t *testing.T) {
	resp, err := decodeJSON(t, `{
		"model":"served-model",
		"choices":[{"index":0,"message":{
			"role":"assistant",
			"reasoning_content":"first I will",
			"content":"the answer",
			"tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"calc","arguments":"{\"expression\":\"6*7\"}"}},
				{"id":"call_2","type":"function","function":{"name":"raw","arguments":{"already":"an object"}}}
			]},
			"finish_reason":"tool_calls"}]}`)
	if err != nil {
		t.Fatalf("decodeResponse: got error %v, want nil", err)
	}
	if len(resp.Content) != 4 {
		t.Fatalf("content: got %d blocks (%v), want 4", len(resp.Content), resp.Content)
	}
	// Reasoning leads: that is the order the model produced it in, and a
	// transcript showing the scratchpad after the answer reads backwards.
	th, ok := resp.Content[0].(llm.Thinking)
	if !ok {
		t.Fatalf("block 0: got %T, want llm.Thinking", resp.Content[0])
	}
	if th.Text != "first I will" {
		t.Errorf("reasoning: got %q, want %q", th.Text, "first I will")
	}
	if th.Signature != "" {
		t.Errorf("signature: got %q, want empty — the compat API has no signed-reasoning channel", th.Signature)
	}
	if got, want := llm.TextOf(resp.Content), "the answer"; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
	uses := llm.ToolUses(resp.Content)
	if len(uses) != 2 {
		t.Fatalf("tool uses: got %d, want 2", len(uses))
	}
	if got, want := string(uses[0].Input), `{"expression":"6*7"}`; got != want {
		t.Errorf("call 0 input: got %s, want %s", got, want)
	}
	// A gateway that sends the arguments object instead of the documented
	// JSON-encoded string decodes to the same thing.
	if got, want := string(uses[1].Input), `{"already":"an object"}`; got != want {
		t.Errorf("call 1 input: got %s, want %s", got, want)
	}
	if resp.Model != "served-model" {
		t.Errorf("model: got %q, want %q", resp.Model, "served-model")
	}
}

func TestDecodeResponseFallsBackToTheRequestedModel(t *testing.T) {
	resp, err := decodeJSON(t, `{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}]}`)
	if err != nil {
		t.Fatalf("decodeResponse: got error %v, want nil", err)
	}
	if got, want := resp.Model, "requested-model"; got != want {
		t.Errorf("model: got %q, want %q", got, want)
	}
}

func TestContentTextAcceptsEveryShapeInTheWild(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{name: "string", raw: `"hello"`, want: "hello"},
		{name: "null on a tool-call-only turn", raw: `null`, want: ""},
		{name: "typed parts array, as some gateways echo back", raw: `[{"type":"text","text":"a"},{"type":"text","text":"b"}]`, want: "a\nb"},
		{name: "parts array with empties", raw: `[{"type":"image"},{"type":"text","text":"only"}]`, want: "only"},
		{name: "empty array", raw: `[]`, want: ""},
		{name: "a number is unsalvageable but must not fail the call", raw: `42`, want: ""},
		{name: "a bool likewise", raw: `true`, want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got contentText
			if err := json.Unmarshal([]byte(tt.raw), &got); err != nil {
				t.Fatalf("UnmarshalJSON(%s): got error %v, want nil", tt.raw, err)
			}
			if string(got) != tt.want {
				t.Errorf("contentText(%s): got %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
	t.Run("a malformed string is an error", func(t *testing.T) {
		var got contentText
		if err := json.Unmarshal([]byte(`"unterminated`), &got); err == nil {
			t.Error("UnmarshalJSON: got nil error, want one")
		}
	})
}

func TestArgStringAcceptsBothShapes(t *testing.T) {
	tests := []struct {
		raw  string
		want string
	}{
		{`"{\"a\":1}"`, `{"a":1}`},
		{`{"a":1}`, `{"a":1}`},
		{`null`, ``},
		{`""`, ``},
	}
	for _, tt := range tests {
		var got argString
		if err := json.Unmarshal([]byte(tt.raw), &got); err != nil {
			t.Fatalf("UnmarshalJSON(%s): got error %v, want nil", tt.raw, err)
		}
		if string(got) != tt.want {
			t.Errorf("argString(%s): got %q, want %q", tt.raw, got, tt.want)
		}
	}
}

func TestToolInput(t *testing.T) {
	tests := []struct {
		name string
		args string
		want string
	}{
		{name: "empty becomes an empty object", args: "", want: `{}`},
		{name: "whitespace becomes an empty object", args: "  \n", want: `{}`},
		{name: "valid JSON passes through byte for byte", args: `{"z":1,"a":2}`, want: `{"z":1,"a":2}`},
		// A model truncated under a token cap leaves a fragment. Handing it to
		// the tool layer produces a schema-validation message the model can act
		// on; failing here produces a dead turn.
		{name: "a truncated fragment is preserved as a JSON string", args: `{"path":"/tmp/x","body":"half`, want: `"{\"path\":\"/tmp/x\",\"body\":\"half"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toolInput(tt.args)
			if string(got) != tt.want {
				t.Errorf("toolInput(%q): got %s, want %s", tt.args, got, tt.want)
			}
			if !json.Valid(got) {
				t.Errorf("toolInput(%q): got invalid JSON %s — a RawMessage that is not valid JSON fails the whole next request", tt.args, got)
			}
		})
	}
}

// TestErrorEnvelopeWithA200: gateways return {"error":{...}} with a 200.
// llm.ClassifyStatus panics on a 2xx, so the class has to be chosen here.
func TestErrorEnvelopeWithA200(t *testing.T) {
	_, err := decodeJSON(t, `{"error":{"type":"invalid_request_error","message":"model not found"}}`)
	if err == nil {
		t.Fatal("decodeResponse: got nil error, want one")
	}
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Errorf("errors.Is(err, llm.ErrBadRequest): got false, want true (err=%v)", err)
	}
	var ae *llm.APIError
	if !errors.As(err, &ae) {
		t.Fatalf("errors.As(*llm.APIError): got false, want true")
	}
	if ae.Status != 200 {
		t.Errorf("Status: got %d, want 200", ae.Status)
	}
	if want := "invalid_request_error: model not found"; ae.Message != want {
		t.Errorf("Message: got %q, want %q", ae.Message, want)
	}
}

func TestDecodeResponseFailures(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "not JSON at all", body: `<html>502</html>`},
		{name: "no choices", body: `{"model":"m","choices":[]}`},
		{name: "choices missing entirely", body: `{"model":"m"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeJSON(t, tt.body)
			if !errors.Is(err, llm.ErrTransport) {
				t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v) — a broken 200 is worth a retry", err)
			}
			if !llm.Retryable(err) {
				t.Errorf("llm.Retryable: got false, want true (err=%v)", err)
			}
		})
	}
}

func TestErrorBodyString(t *testing.T) {
	var nilErr *errorBody
	if got := nilErr.String(); got != "" {
		t.Errorf("(*errorBody)(nil).String(): got %q, want empty", got)
	}
	if got, want := (&errorBody{Message: "m"}).String(), "m"; got != want {
		t.Errorf("String: got %q, want %q", got, want)
	}
	if got, want := (&errorBody{Type: "t", Message: "m"}).String(), "t: m"; got != want {
		t.Errorf("String: got %q, want %q", got, want)
	}
}

// failingReader yields some bytes and then a read error, the way a connection
// dropped mid-body does.
type failingReader struct {
	head string
	n    int
}

func (r *failingReader) Read(p []byte) (int, error) {
	if r.n < len(r.head) {
		n := copy(p, r.head[r.n:])
		r.n += n
		return n, nil
	}
	return 0, errors.New("connection reset by peer")
}

func TestDecodeResponseReadErrorIsTransport(t *testing.T) {
	_, err := decodeResponse(&failingReader{head: `{"choices":[`}, "m")
	if !errors.Is(err, llm.ErrTransport) {
		t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
	}
}

func TestDecodeStreamReadErrorIsTransport(t *testing.T) {
	_, err := decodeStream(&failingReader{head: "data: {\"choices\":[]}\n\n"}, nil, "m")
	if !errors.Is(err, llm.ErrTransport) {
		t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
	}
}

func TestPermissiveDecodersOnEmptyInput(t *testing.T) {
	var c contentText
	if err := c.UnmarshalJSON(nil); err != nil || c != "" {
		t.Errorf("contentText.UnmarshalJSON(nil): got (%q, %v), want (\"\", nil)", c, err)
	}
	var a argString
	if err := a.UnmarshalJSON(nil); err != nil || a != "" {
		t.Errorf("argString.UnmarshalJSON(nil): got (%q, %v), want (\"\", nil)", a, err)
	}
	if err := a.UnmarshalJSON([]byte(`"unterminated`)); err == nil {
		t.Error("argString.UnmarshalJSON on a malformed string: got nil error, want one")
	}
	var bad contentText
	if err := bad.UnmarshalJSON([]byte(`[{"text":1}]`)); err == nil {
		t.Error("contentText.UnmarshalJSON on a malformed parts array: got nil error, want one")
	}
}
