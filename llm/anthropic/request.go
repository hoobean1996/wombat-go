package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/hoobean1996/wombat-go/llm"
)

// The request body is built out of structs, never map[string]any.
//
// This is not a style preference. Go marshals map keys in sorted order, so a
// tool schema round-tripped through a map reaches the model with its fields
// rearranged — different bytes for the same declaration. That changes model
// behavior and, because the prompt cache is a byte-prefix match, it also
// invalidates the cached prefix on every call. Struct field order is fixed at
// compile time, and llm.ToolSpec.InputSchema is passed through as
// json.RawMessage without ever being decoded.

type requestBody struct {
	Model     string        `json:"model"`
	MaxTokens int           `json:"max_tokens"`
	Messages  []messageJSON `json:"messages"`

	System            []systemBlock      `json:"system,omitempty"`
	Tools             []toolJSON         `json:"tools,omitempty"`
	ContextManagement *contextManagement `json:"context_management,omitempty"`
	ToolChoice        *toolChoiceJSON    `json:"tool_choice,omitempty"`

	// Temperature and TopP are pointers so that omitempty distinguishes "not
	// set" from "set to zero". A float64 with omitempty would silently drop
	// temperature:0 — the one setting that makes a run reproducible — and a
	// float64 without omitempty would send temperature:0 on every request that
	// never asked for it, overriding the provider's own default.
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`

	Stream bool `json:"stream,omitempty"`
}

type messageJSON struct {
	Role string `json:"role"`
	// Content holds *textBlock, *toolUseBlock, *toolResultBlock,
	// *thinkingBlock, *imageBlock or *documentBlock. Pointers, so the cache
	// breakpoint can be stamped on the last one after the slice is built.
	Content []any `json:"content"`
}

type cacheControl struct {
	Type string `json:"type"`
}

// ephemeral is shared by every breakpoint in every request. It is immutable
// and only ever read by the encoder, so one value is enough.
var ephemeral = &cacheControl{Type: "ephemeral"}

type systemBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

type textBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

type toolUseBlock struct {
	Type         string          `json:"type"`
	ID           string          `json:"id"`
	Name         string          `json:"name"`
	Input        json.RawMessage `json:"input"`
	CacheControl *cacheControl   `json:"cache_control,omitempty"`
}

// toolResultBlock's Content is the result's text as a plain string, or — when
// the tool attached media — a list of text, image and document blocks, which
// is the other form the API accepts. The string form is kept whenever it
// suffices so that a text-only transcript encodes byte for byte as it always
// did, and its cache entries still match.
type toolResultBlock struct {
	Type         string        `json:"type"`
	ToolUseID    string        `json:"tool_use_id"`
	Content      any           `json:"content"`
	IsError      bool          `json:"is_error,omitempty"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

// mediaSource is the inline, base64 form of an image or a document.
type mediaSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type imageBlock struct {
	Type         string        `json:"type"`
	Source       mediaSource   `json:"source"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

type documentBlock struct {
	Type         string        `json:"type"`
	Source       mediaSource   `json:"source"`
	Title        string        `json:"title,omitempty"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

// thinkingBlock carries the signature back verbatim. The API rejects the next
// turn if a thinking block returns without the signature it was issued with.
type thinkingBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking"`
	Signature string `json:"signature,omitempty"`
}

type toolJSON struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// InputSchema is the author's bytes, untouched. See the note at the top of
	// this file.
	InputSchema  json.RawMessage `json:"input_schema"`
	CacheControl *cacheControl   `json:"cache_control,omitempty"`
}

type toolChoiceJSON struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

// emptySchema stands in for a tool that declares no parameters. A nil
// RawMessage would marshal as null, which the API rejects.
var emptySchema = json.RawMessage(`{"type":"object","properties":{}}`)

// emptyInput stands in for a tool_use whose input was never populated, for the
// same reason.
var emptyInput = json.RawMessage(`{}`)

// encode builds the wire body for one call.
func (c *Client) encode(req llm.Request, model string, maxTokens int, stream bool) ([]byte, error) {
	msgs, err := encodeMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	temperature, topP, err := c.sampling(req)
	if err != nil {
		return nil, err
	}

	body := requestBody{
		Model:       model,
		MaxTokens:   maxTokens,
		Messages:    msgs,
		Temperature: temperature,
		TopP:        topP,
		Stream:      stream,
	}

	// Breakpoint 1: the system prompt.
	//
	// It is the longest-lived bytes in the whole request — the agent renders it
	// once at construction and never rebuilds it — so it is the prefix every
	// later turn hits. Caching it costs one write on the first call of a run
	// and is read for free on every call after.
	if req.System != "" {
		body.System = []systemBlock{{
			Type:         "text",
			Text:         req.System,
			CacheControl: ephemeral,
		}}
	}

	if len(req.Tools) > 0 {
		// Breakpoint 2: the END of the tool list.
		//
		// The cache is a prefix match over tools → system → messages, so a
		// breakpoint on the last tool covers every tool before it plus the
		// system prompt. Marking any earlier tool would leave the rest of the
		// list outside the cached prefix and re-bill it every turn; marking
		// each tool would burn the four-breakpoint budget on one section.
		body.Tools = encodeTools(req.Tools)

		if c.ctxMgmt {
			// Only meaningful alongside tools — the edit prunes tool_use
			// blocks — and only legal with the beta header, hence the gate on
			// both.
			body.ContextManagement = defaultContextManagement(req.Tools)
		}
	}

	body.ToolChoice = encodeChoice(req.Choice)

	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("anthropic: encode request: %w", err)
	}
	return b, nil
}

// sampling resolves the sampling controls for one call: what the request asked
// for, falling back to the client defaults from [Config].
//
// # Why the two are resolved as a pair
//
// The Messages API rejects a body that carries BOTH temperature and top_p
// ("only one of temperature and top_p may be specified"), so this client must
// never send both. Resolving field by field — request temperature else config
// temperature, request top_p else config top_p — would manufacture exactly that
// illegal body out of two individually legal settings: an operator who exports
// $ANTHROPIC_TOP_P and a caller who sets Request.Temperature (which is what
// sampling n trajectories of one task needs) would 400 on every single call,
// with no way to unset the default from the request side, since nil there means
// "no opinion" rather than "off".
//
// So the pair is the unit: if the request names EITHER control, it supplies
// both, and the config defaults for both are ignored. A request can therefore
// always override a configured default, whichever of the two it is.
//
// A single request that names both is the one remaining conflict, and that one
// is a caller error rather than a layering accident, so it is refused here.
// [New] refuses the same mistake in a Config, where it can be named earlier.
func (c *Client) sampling(req llm.Request) (temperature, topP *float64, err error) {
	temperature, topP = req.Temperature, req.TopP
	if temperature == nil && topP == nil {
		temperature, topP = c.temperature, c.topP
	}
	if temperature != nil && topP != nil {
		return nil, nil, fmt.Errorf("anthropic: a request may set Temperature or TopP, not both (got %v and %v): %w",
			*temperature, *topP, llm.ErrBadRequest)
	}
	return temperature, topP, nil
}

func encodeMessages(msgs []llm.Message) ([]messageJSON, error) {
	out := make([]messageJSON, len(msgs))
	for i, m := range msgs {
		blocks := make([]any, 0, len(m.Content))
		for _, b := range m.Content {
			enc, err := encodeBlock(b)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, enc)
		}
		out[i] = messageJSON{Role: string(m.Role), Content: blocks}
	}
	markLastCacheable(out)
	return out, nil
}

func encodeBlock(b llm.ContentBlock) (any, error) {
	switch v := b.(type) {
	case llm.Text:
		return &textBlock{Type: "text", Text: v.Text}, nil
	case llm.ToolUse:
		in := v.Input
		if len(in) == 0 {
			in = emptyInput
		}
		return &toolUseBlock{Type: "tool_use", ID: string(v.ID), Name: v.Name, Input: in}, nil
	case llm.ToolResult:
		var content any = v.Content
		if len(v.Media) > 0 {
			parts := make([]any, 0, len(v.Media)+1)
			if v.Content != "" {
				parts = append(parts, &textBlock{Type: "text", Text: v.Content})
			}
			for _, m := range v.Media {
				enc, err := encodeMedia(m)
				if err != nil {
					return nil, err
				}
				parts = append(parts, enc)
			}
			content = parts
		}
		return &toolResultBlock{
			Type:      "tool_result",
			ToolUseID: string(v.ToolUseID),
			Content:   content,
			IsError:   v.IsError,
		}, nil
	case llm.Thinking:
		return &thinkingBlock{Type: "thinking", Thinking: v.Text, Signature: v.Signature}, nil
	case llm.Media:
		return encodeMedia(v)
	default:
		// llm.ContentBlock is a closed set, so this is unreachable today; it
		// stays an error rather than a panic so that adding a variant upstream
		// degrades into a failed call instead of a crashed agent.
		return nil, fmt.Errorf("anthropic: cannot encode content block %T: %w", b, llm.ErrBadRequest)
	}
}

// encodeMedia renders an image as an image block and a PDF as a document
// block, both inline as base64. Anything else is refused here, where the error
// can name the type, rather than by the provider.
func encodeMedia(m llm.Media) (any, error) {
	src := mediaSource{Type: "base64", MediaType: m.MediaType, Data: base64.StdEncoding.EncodeToString(m.Data)}
	switch {
	case m.IsImage():
		return &imageBlock{Type: "image", Source: src}, nil
	case m.MediaType == "application/pdf":
		return &documentBlock{Type: "document", Source: src, Title: m.Name}, nil
	default:
		return nil, fmt.Errorf("anthropic: cannot send media of type %q (want an image or application/pdf): %w",
			m.MediaType, llm.ErrBadRequest)
	}
}

// markLastCacheable stamps breakpoint 3 on the final content block of the
// final message.
//
// This is the moving breakpoint, and it is what turns two static breakpoints
// into a sliding three-tier cache: turn N writes a cache entry covering
// everything up to and including its own last block, and turn N+1 — whose
// history is that same prefix plus new turns — reads it, then writes a new
// entry at its own tail. Without it, only the system prompt and tools would be
// cached and the whole transcript would be re-billed at full price every
// iteration, which on a thirty-step run is most of the cost of the run.
//
// A thinking block is skipped: the API does not accept cache_control there, and
// losing one breakpoint is cheaper than a 400. In practice the last message is
// a user turn ending in text or a tool_result, so the case does not arise.
func markLastCacheable(msgs []messageJSON) {
	if len(msgs) == 0 {
		return
	}
	blocks := msgs[len(msgs)-1].Content
	if len(blocks) == 0 {
		return
	}
	switch v := blocks[len(blocks)-1].(type) {
	case *textBlock:
		v.CacheControl = ephemeral
	case *toolUseBlock:
		v.CacheControl = ephemeral
	case *toolResultBlock:
		v.CacheControl = ephemeral
	case *imageBlock:
		v.CacheControl = ephemeral
	case *documentBlock:
		v.CacheControl = ephemeral
	}
}

func encodeTools(specs []llm.ToolSpec) []toolJSON {
	out := make([]toolJSON, len(specs))
	for i, s := range specs {
		schema := s.InputSchema
		if len(schema) == 0 {
			schema = emptySchema
		}
		out[i] = toolJSON{Name: s.Name, Description: s.Description, InputSchema: schema}
	}
	if len(out) > 0 {
		out[len(out)-1].CacheControl = ephemeral
	}
	return out
}

// encodeChoice omits tool_choice for the auto mode. Auto is the API default,
// and an omitted field keeps the body — and the request the model sees —
// identical to what it was before anyone thought about tool choice.
func encodeChoice(tc llm.ToolChoice) *toolChoiceJSON {
	switch tc.Mode {
	case llm.ChoiceAuto:
		return nil
	case llm.ChoiceTool:
		return &toolChoiceJSON{Type: "tool", Name: tc.Name}
	default:
		// ChoiceAny and ChoiceNone are already the wire spellings.
		return &toolChoiceJSON{Type: string(tc.Mode)}
	}
}

// ===== Server-side context management =====

type contextManagement struct {
	Edits []contextEdit `json:"edits"`
}

type contextEdit struct {
	Type         string       `json:"type"`
	Trigger      *cmThreshold `json:"trigger,omitempty"`
	Keep         *cmThreshold `json:"keep,omitempty"`
	ClearAtLeast *cmThreshold `json:"clear_at_least,omitempty"`
	ExcludeTools []string     `json:"exclude_tools,omitempty"`
}

type cmThreshold struct {
	Type  string `json:"type"`
	Value int    `json:"value"`
}

// clear_tool_uses parameters. Every one of these numbers is a production
// default carried over from the OCaml harness, and each is a tradeoff:
//
//   - trigger at 20k input tokens, not lower: below that the transcript is
//     cheap and pruning it only costs a cache invalidation;
//   - keep the last 4 tool_uses: enough for the model to see what it just did
//     and why, which is the context it actually reasons over;
//   - clear at least 3k tokens: pruning less than that churns the cached prefix
//     for no meaningful saving.
const (
	clearToolUsesType     = "clear_tool_uses_20250919"
	clearTriggerTokens    = 20_000
	clearKeepToolUses     = 4
	clearAtLeastTokens    = 3_000
	inputTokensThreshold  = "input_tokens"
	toolUsesThresholdKind = "tool_uses"
)

// neverCleared names the tools whose results the server must leave alone.
//
// load_skill, because a skill's activation lives on the client: the harness
// retires a skill whose body leaves the transcript, and it can only see what
// it removed itself. A body the server cleared would leave the skill's tools
// on offer with the instructions for them gone, and nothing on this side
// would know.
var neverCleared = []string{"load_skill"}

func defaultContextManagement(tools []llm.ToolSpec) *contextManagement {
	var exclude []string
	for _, t := range tools {
		if slices.Contains(neverCleared, t.Name) {
			exclude = append(exclude, t.Name)
		}
	}
	return &contextManagement{
		Edits: []contextEdit{{
			Type:         clearToolUsesType,
			Trigger:      &cmThreshold{Type: inputTokensThreshold, Value: clearTriggerTokens},
			Keep:         &cmThreshold{Type: toolUsesThresholdKind, Value: clearKeepToolUses},
			ClearAtLeast: &cmThreshold{Type: inputTokensThreshold, Value: clearAtLeastTokens},
			ExcludeTools: exclude,
		}},
	}
}
