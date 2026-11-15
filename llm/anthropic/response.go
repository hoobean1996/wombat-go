package anthropic

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/internal/httpcall"
)

// apiMessage is the non-streaming response body, and also the payload of the
// message_start stream event.
//
// Usage decodes straight into llm.Usage because that type's JSON tags are
// already the Anthropic field names — the domain type has that lineage. A
// missing or null field simply leaves the zero.
type apiMessage struct {
	Model      string     `json:"model"`
	Content    []apiBlock `json:"content"`
	StopReason string     `json:"stop_reason"`
	Usage      llm.Usage  `json:"usage"`
}

// apiBlock is one response content block. Every known variant's fields are
// flattened into one struct: the set is small, closed by the API version, and
// a flat struct decodes in one pass without a discriminator dance.
type apiBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

// decodeMessage reads the non-streaming response, refusing a body larger than
// limit bytes. Unbounded, a json.Decoder over a misbehaving gateway's body
// reads until the process runs out of memory, one value at a time.
func decodeMessage(r io.Reader, requestedModel string, limit int) (llm.Response, error) {
	raw, err := httpcall.ReadBody(r, "anthropic", limit)
	if err != nil {
		return llm.Response{}, err
	}
	var msg apiMessage
	if err := json.Unmarshal(raw, &msg); err != nil {
		// A 2xx whose body is not the JSON we asked for means something
		// between us and the model mangled it — a truncated body, a gateway
		// error page served with the wrong status. Transport class, so the
		// retry middleware gets a chance rather than the agent giving up on a
		// well-formed request.
		return llm.Response{}, llm.ClassifyTransport(fmt.Errorf("anthropic: decode response: %w", err))
	}
	return llm.Response{
		Content:    decodeBlocks(msg.Content),
		StopReason: llm.StopReason(msg.StopReason),
		Usage:      msg.Usage,
		Model:      pick(msg.Model, requestedModel),
	}, nil
}

func decodeBlocks(raw []apiBlock) []llm.ContentBlock {
	out := make([]llm.ContentBlock, 0, len(raw))
	for _, b := range raw {
		if blk, ok := b.toBlock(); ok {
			out = append(out, blk)
		}
	}
	return out
}

// toBlock converts one wire block, reporting false for anything we do not
// model.
//
// Unknown types — redacted_thinking, server_tool_use, web search results — are
// dropped rather than erroring: the API adds block types without a version
// bump, and a new one appearing must not fail a run that never asked for it.
func (b apiBlock) toBlock() (llm.ContentBlock, bool) {
	switch b.Type {
	case "text":
		return llm.Text{Text: scrub(b.Text)}, true
	case "thinking":
		return llm.Thinking{Text: scrub(b.Thinking), Signature: b.Signature}, true
	case "tool_use":
		return llm.ToolUse{
			ID:    llm.ToolUseID(b.ID),
			Name:  b.Name,
			Input: scrubRaw(b.Input),
		}, true
	default:
		return nil, false
	}
}

// scrubRaw scrubs a JSON document without decoding it.
//
// Invalid bytes can only occur inside string literals (anywhere else the JSON
// was already broken), and replacing them with U+FFFD keeps the document
// valid. Decoding to any and re-encoding would fix the bytes too — and reorder
// every object key on the way out, which is exactly what must not happen to a
// tool input that gets replayed into the next request's cache prefix.
func scrubRaw(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return emptyInput
	}
	s := string(raw)
	if clean := scrub(s); clean != s {
		return json.RawMessage(clean)
	}
	return raw
}

func pick(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
