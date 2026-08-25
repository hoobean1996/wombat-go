package openai

import (
	"encoding/json"
	"fmt"

	"github.com/hoobean1996/wombat-go/llm"
)

// The request body is built from structs, never from map[string]any.
//
// This is not a style preference: encoding/json emits map keys in sorted
// order, so a tool's input schema round-tripped through a map would reach the
// model with its properties reordered. That changes what the model sees and
// invalidates any prompt-cache prefix. Field order in a struct is the
// declaration order, which is stable and reviewable.
type chatRequest struct {
	Model     string        `json:"model"`
	Messages  []chatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens,omitempty"`

	// Temperature and TopP are pointers so that omitempty can distinguish "not
	// set" from "set to zero". A float64 with omitempty would silently drop
	// temperature:0 — the one value that pins a run — and a float64 without
	// omitempty would send temperature:0 on every request that never mentioned
	// sampling, overriding a default the provider chose with one we did not.
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`

	Tools         []toolDef      `json:"tools,omitempty"`
	ToolChoice    any            `json:"tool_choice,omitempty"`
	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

// streamOptions asks for a final usage chunk. OpenAI omits usage from streamed
// responses unless this is set, and a client that silently reports zero tokens
// would defeat the governor's cost budget. Gateways that do not know the field
// ignore it; the ones that already send usage unconditionally send it twice at
// worst, and the last chunk wins.
//
// It must never be sent on a non-streamed request: at least one gateway 400s
// with "`stream_options` requires `stream` to be true" (verified). Hence the
// pointer plus omitempty, and the single assignment guarded by `if stream`.
//
// Sending it is still right even against a gateway that ignores it and returns
// no usage at all — that is a server bug the caller works around with
// llm.StreamNever, not a reason to stop asking correctly.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// chatMessage is one wire message. Content is a pointer because an assistant
// turn that is nothing but tool calls must send `"content": null` — omitting
// the key entirely makes some gateways reject the turn.
type chatMessage struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
}

type toolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function toolCallFunc `json:"function"`
}

// toolCallFunc carries Arguments as a JSON-encoded *string*, not an object.
// That double encoding is the API's, not ours.
type toolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolDef struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// toolChoiceFunction is the ChoiceTool encoding. Anthropic writes
// {"type":"tool","name":x}; OpenAI nests the name one level deeper.
type toolChoiceFunction struct {
	Type     string             `json:"type"`
	Function toolChoiceFuncName `json:"function"`
}

type toolChoiceFuncName struct {
	Name string `json:"name"`
}

// emptySchema is what a tool with no declared input sends. OpenAI requires
// `parameters` to be a JSON object; null or an absent key is a 400.
var emptySchema = json.RawMessage(`{"type":"object","properties":{}}`)

// encodeRequest renders one llm.Request as a chat-completions body.
func encodeRequest(req llm.Request, model string, maxTokens int, stream bool) ([]byte, error) {
	msgs := encodeMessages(req.System, req.Messages)
	if len(msgs) == 0 {
		// Caught here rather than at the provider so the failure is cheap and
		// says what is actually wrong. Classified as a bad request because
		// that is exactly what the provider would call it.
		return nil, &llm.APIError{Class: llm.ErrBadRequest, Message: "openai: request has no messages"}
	}

	body := chatRequest{
		Model:     model,
		Messages:  msgs,
		MaxTokens: maxTokens,
		// Already merged with the client defaults by Client.sampling; nil here
		// means nobody had an opinion, so nothing is sent.
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      stream,
	}
	if stream {
		body.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	if len(req.Tools) > 0 {
		body.Tools = make([]toolDef, 0, len(req.Tools))
		for _, t := range req.Tools {
			schema := t.InputSchema
			if len(schema) == 0 {
				schema = emptySchema
			}
			body.Tools = append(body.Tools, toolDef{
				Type: "function",
				Function: toolFunction{
					Name:        t.Name,
					Description: t.Description,
					// Passed through byte for byte, per llm.ToolSpec.
					Parameters: schema,
				},
			})
		}
	}

	choice, err := encodeToolChoice(req.Choice)
	if err != nil {
		return nil, err
	}
	body.ToolChoice = choice

	out, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("openai: encode request: %w", err)
	}
	// No UTF-8 scrubbing pass, unlike the OCaml original: encoding/json already
	// replaces invalid bytes with U+FFFD when it writes a string, so malformed
	// tool output cannot produce a body the provider rejects.
	return out, nil
}

// sampling resolves the sampling controls for one call: whatever the request
// asked for, field by field, falling back to the [Config] defaults.
//
// Field by field, and not as a pair, because this API takes temperature and
// top_p together — a configured top_p plus a per-request temperature is a
// legal, meaningful body, and both reach the wire. (llm/anthropic resolves the
// same two fields as a pair, because the Messages API refuses to accept both;
// the difference is the providers', not a disagreement between the clients.)
func (c *Client) sampling(req llm.Request) (temperature, topP *float64) {
	temperature, topP = req.Temperature, req.TopP
	if temperature == nil {
		temperature = c.temperature
	}
	if topP == nil {
		topP = c.topP
	}
	return temperature, topP
}

// encodeToolChoice maps llm.ToolChoice onto OpenAI's spelling, which differs
// from Anthropic's in every case: "required" rather than {"type":"any"}, and a
// nested function object rather than a flat name.
func encodeToolChoice(tc llm.ToolChoice) (any, error) {
	switch tc.Mode {
	case llm.ChoiceAuto:
		// The API default. Omitted so the body stays byte-identical to one
		// built without a choice at all, which keeps the cache prefix stable.
		return nil, nil
	case llm.ChoiceAny:
		return "required", nil
	case llm.ChoiceNone:
		return "none", nil
	case llm.ChoiceTool:
		if tc.Name == "" {
			return nil, &llm.APIError{Class: llm.ErrBadRequest, Message: "openai: tool choice mode \"tool\" needs a name"}
		}
		return toolChoiceFunction{Type: "function", Function: toolChoiceFuncName{Name: tc.Name}}, nil
	default:
		return nil, &llm.APIError{Class: llm.ErrBadRequest, Message: fmt.Sprintf("openai: unknown tool choice mode %q", tc.Mode)}
	}
}

// encodeMessages flattens the conversation. The system prompt becomes the
// leading message (OpenAI has no out-of-band system field), and each domain
// message expands to one or more wire messages.
func encodeMessages(system string, msgs []llm.Message) []chatMessage {
	out := make([]chatMessage, 0, len(msgs)+1)
	if system != "" {
		out = append(out, chatMessage{Role: "system", Content: strptr(system)})
	}
	for _, m := range msgs {
		switch m.Role {
		case llm.RoleAssistant:
			out = append(out, encodeAssistant(m.Content))
		default:
			out = append(out, encodeUser(m.Content)...)
		}
	}
	return out
}

// encodeAssistant folds one assistant turn into exactly one wire message: text
// blocks join into content, ToolUse blocks become the tool_calls array.
//
// Thinking blocks are dropped. There is nowhere to put them — the compat API
// has no signed-reasoning channel — and inventing a text block for them would
// feed the model its own scratchpad as if it were an answer.
//
// That drop is what makes the decoders' reasoning_content handling safe: both
// decodeResponse and decodeStream turn the gateway's reasoning channel into an
// llm.Thinking block, and this switch is the reason such a block can never
// travel back upstream. Do not add a `case llm.Thinking` here.
func encodeAssistant(blocks []llm.ContentBlock) chatMessage {
	msg := chatMessage{Role: "assistant"}
	var text string
	first := true
	for _, b := range blocks {
		switch v := b.(type) {
		case llm.Text:
			if !first {
				text += "\n"
			}
			text += v.Text
			first = false
		case llm.ToolUse:
			args := string(v.Input)
			if args == "" {
				args = "{}"
			}
			msg.ToolCalls = append(msg.ToolCalls, toolCall{
				ID:       v.ID.String(),
				Type:     "function",
				Function: toolCallFunc{Name: v.Name, Arguments: args},
			})
		}
	}
	if !first {
		msg.Content = strptr(text)
	}
	// Content stays nil (encoded as null) for a tool-call-only turn.
	return msg
}

// encodeUser expands one user turn into the several messages OpenAI wants.
//
// Rules, ported from the OCaml encoder:
//   - Text seen before any ToolResult accumulates into a single leading
//     {role:"user"} message.
//   - Each ToolResult becomes its own {role:"tool", tool_call_id, content}
//     message. The API pairs results to calls by id, and rejects a turn that
//     answers three tool calls with one message.
//   - Text seen AFTER a ToolResult is appended to the most recent tool
//     message instead of becoming a user message of its own. That looks odd
//     until you try the alternative: the API requires every assistant
//     tool_calls turn to be followed immediately by its tool messages, so a
//     user message wedged in between (which is what the harness produces when
//     it staples a nudge onto a results turn) is a 400. Folding the text into
//     the last result keeps the words and keeps the turn legal.
//
// ToolUse blocks in a user turn are ignored, and forwarding them would be a
// 400. They are a caller bug caught upstream by wombat.Convo.Validate's
// invariant 4 — which is worth stating precisely, because this comment
// previously claimed that protection existed at a time when it did not.
// Validate checked role alternation and orphaned results and nothing about
// which role a block was allowed to appear in, so an externally supplied
// transcript with a tool_use in a user turn reached both encoders. Dropping it
// here is now belt and braces rather than the only line of defence.
func encodeUser(blocks []llm.ContentBlock) []chatMessage {
	var (
		leading  string
		haveText bool
		tools    []chatMessage
	)
	for _, b := range blocks {
		switch v := b.(type) {
		case llm.Text:
			if n := len(tools); n > 0 {
				last := &tools[n-1]
				appended := *last.Content + "\n" + v.Text
				last.Content = &appended
				continue
			}
			if haveText {
				leading += "\n"
			}
			leading += v.Text
			haveText = true
		case llm.ToolResult:
			// IsError has no wire representation here: OpenAI has no
			// is_error flag on a tool message, so the tool layer's error
			// text is the only signal the model gets.
			tools = append(tools, chatMessage{
				Role:       "tool",
				ToolCallID: v.ToolUseID.String(),
				Content:    strptr(v.Content),
			})
		}
	}

	out := make([]chatMessage, 0, len(tools)+1)
	if haveText {
		out = append(out, chatMessage{Role: "user", Content: strptr(leading)})
	}
	return append(out, tools...)
}

func strptr(s string) *string { return &s }
