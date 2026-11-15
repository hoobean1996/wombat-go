package openai

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/llm/internal/httpcall"
)

// chatResponse is the non-streaming reply. Only the first choice is read: the
// domain has no representation for n>1, and nothing in the harness sets n.
type chatResponse struct {
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *usageBody   `json:"usage"`
	Error   *errorBody   `json:"error"`
}

type chatChoice struct {
	Message struct {
		Content contentText `json:"content"`

		// ReasoningContent is the model's scratchpad. It is not in the OpenAI
		// spec — it is a DeepSeek-lineage extension that the reasoning models
		// behind several corporate gateways also emit, verified against a live
		// endpoint — but decoding an absent field costs nothing, and on those
		// models reasoning is the majority of the generated tokens, so dropping
		// it throws away most of the reply.
		ReasoningContent contentText `json:"reasoning_content"`

		ToolCalls []respToolCall `json:"tool_calls"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

type respToolCall struct {
	ID       string `json:"id"`
	Function struct {
		Name      string    `json:"name"`
		Arguments argString `json:"arguments"`
	} `json:"function"`
}

// errorBody is the {"error":{...}} envelope. Gateways sometimes return it with
// a 200, so it is checked even on success.
type errorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

func (e *errorBody) String() string {
	if e == nil {
		return ""
	}
	var b strings.Builder
	if e.Type != "" {
		b.WriteString(e.Type)
		b.WriteString(": ")
	}
	b.WriteString(e.Message)
	return b.String()
}

// contentText decodes a content field permissively.
//
// OpenAI sends a string, or null on a tool-call-only turn. Several gateways
// send an array of typed parts instead (that is the multimodal request shape
// echoed back). Accepting all three keeps one non-conformant server from
// failing an otherwise perfectly good reply.
type contentText string

// UnmarshalJSON implements json.Unmarshaler.
func (t *contentText) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		*t = ""
		return nil
	}
	switch b[0] {
	case '"':
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		*t = contentText(s)
	case '[':
		var parts []struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(b, &parts); err != nil {
			return err
		}
		var sb strings.Builder
		for _, p := range parts {
			if p.Text == "" {
				continue
			}
			if sb.Len() > 0 {
				sb.WriteByte('\n')
			}
			sb.WriteString(p.Text)
		}
		*t = contentText(sb.String())
	default:
		// A number or bool here means the server is inventing its own shape;
		// there is no text to salvage, and failing the call would be worse.
		*t = ""
	}
	return nil
}

// argString is a tool call's arguments. The API says JSON-encoded string; a
// few gateways send the object itself. Both decode to the same thing here,
// since the next step re-parses it anyway.
type argString string

// UnmarshalJSON implements json.Unmarshaler.
func (a *argString) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		*a = ""
		return nil
	}
	if b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		*a = argString(s)
		return nil
	}
	*a = argString(b)
	return nil
}

// usageBody is the token accounting, in OpenAI's spelling.
type usageBody struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`

	// PromptCacheHitTokens is DeepSeek's top-level spelling.
	PromptCacheHitTokens int `json:"prompt_cache_hit_tokens"`

	// PromptTokensDetails is OpenAI's nested spelling.
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`

	// CompletionTokensDetails breaks the output count down. ReasoningTokens
	// is decoded but deliberately NOT folded into llm.Usage: the servers that
	// report it count reasoning inside completion_tokens (verified: 108
	// completion of which 34 reasoning), so adding it would bill those tokens
	// twice, and llm.Usage has no separate line for them. It is kept because
	// it is the only way to see how much of a bill is scratchpad.
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

// toUsage converts to the domain shape, subtracting cache hits out of the
// input count.
//
// The subtraction is the whole point of this function. OpenAI-compatible
// servers count cached prompt tokens INSIDE prompt_tokens; Anthropic keeps
// input_tokens and cache_read_input_tokens disjoint. llm.Usage is the
// Anthropic shape, and llm.Pricing bills InputTokens and CacheReadTokens at
// different rates, so passing prompt_tokens through unchanged would bill the
// cached prefix twice — once at full price and once at the cache-read price.
func (u *usageBody) toUsage() llm.Usage {
	if u == nil {
		return llm.Usage{}
	}
	cached := u.PromptCacheHitTokens
	if cached == 0 && u.PromptTokensDetails != nil {
		cached = u.PromptTokensDetails.CachedTokens
	}
	in := u.PromptTokens - cached
	if in < 0 {
		// A gateway that reports more cache hits than prompt tokens is lying;
		// clamp rather than emit a negative that would credit the budget.
		in = 0
	}
	return llm.Usage{
		InputTokens:  in,
		OutputTokens: u.CompletionTokens,
		// No CacheWriteTokens: the compat API has no cache-write concept —
		// caching is automatic and unbilled as a separate line item.
		CacheReadTokens: cached,
	}
}

// stopReason maps finish_reason onto llm.StopReason. Unknown values pass
// through verbatim, which is why llm.StopReason is an open string: gateways
// invent reasons ("eos", "tool_use", "error") and flattening them all into one
// bucket would destroy the only diagnostic the caller has.
func stopReason(finish string) llm.StopReason {
	switch finish {
	case "stop":
		return llm.StopEndTurn
	case "tool_calls":
		return llm.StopToolUse
	case "length":
		return llm.StopMaxTokens
	case "content_filter":
		return llm.StopRefusal
	default:
		return llm.StopReason(finish)
	}
}

// toolInput turns a tool call's arguments into a JSON value.
//
// Invalid JSON is preserved as a JSON string rather than dropped or errored
// on: models truncate arguments under a token cap, and handing the fragment to
// the tool layer produces a schema-validation message the model can act on,
// where failing the call here produces a dead turn.
func toolInput(args string) json.RawMessage {
	if strings.TrimSpace(args) == "" {
		return json.RawMessage(`{}`)
	}
	if json.Valid([]byte(args)) {
		return json.RawMessage(args)
	}
	quoted, err := json.Marshal(args)
	if err != nil {
		return json.RawMessage(`{}`)
	}
	return json.RawMessage(quoted)
}

// decodeResponse parses a non-streaming reply. model is the requested id, used
// when the server does not echo one back.
//
// The body is read whole, up to limit bytes; see httpcall.ReadBody.
func decodeResponse(r io.Reader, model string, limit int) (llm.Response, error) {
	raw, err := httpcall.ReadBody(r, "openai", limit)
	if err != nil {
		return llm.Response{}, err
	}
	var body chatResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		// Classified as transport, not bad-request: a body that is not JSON
		// after a 200 means something between us and the model cut it short,
		// and that is worth a retry.
		return llm.Response{}, llm.ClassifyTransport(fmt.Errorf("openai: parse response: %w: %s", err, truncate(string(raw))))
	}
	if body.Error != nil && body.Error.Message != "" {
		// A 200 carrying an error envelope. ClassifyStatus panics on 2xx, so
		// the class is chosen here.
		return llm.Response{}, &llm.APIError{Class: llm.ErrBadRequest, Status: 200, Message: body.Error.String()}
	}
	if len(body.Choices) == 0 {
		return llm.Response{}, llm.ClassifyTransport(fmt.Errorf("openai: response has no choices: %s", truncate(string(raw))))
	}

	choice := body.Choices[0]
	var content []llm.ContentBlock
	// Reasoning leads, because that is the order the model produced it in and
	// a transcript that shows the scratchpad after the answer reads backwards.
	//
	// Round-trip safety: encodeAssistant in request.go ignores llm.Thinking
	// blocks entirely (verified — its switch handles only Text and ToolUse),
	// so this block is display-only and never goes back upstream. There is no
	// Signature to carry either: the compat API has no signed-reasoning
	// channel to validate one against.
	if r := string(choice.Message.ReasoningContent); r != "" {
		content = append(content, llm.Thinking{Text: r})
	}
	if choice.Message.Content != "" {
		content = append(content, llm.Text{Text: string(choice.Message.Content)})
	}
	for _, tc := range choice.Message.ToolCalls {
		content = append(content, llm.ToolUse{
			ID:    llm.ToolUseID(tc.ID),
			Name:  tc.Function.Name,
			Input: toolInput(string(tc.Function.Arguments)),
		})
	}

	if body.Model != "" {
		model = body.Model
	}
	return llm.Response{
		Content:    content,
		StopReason: stopReason(choice.FinishReason),
		Usage:      body.Usage.toUsage(),
		Model:      model,
	}, nil
}

// truncate bounds a body quoted into an error message, on a rune boundary.
func truncate(s string) string { return httpcall.Truncate(s, httpcall.MaxErrorMessage) }
