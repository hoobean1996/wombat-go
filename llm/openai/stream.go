package openai

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/hoobean1996/wombat-go/internal/sse"
	"github.com/hoobean1996/wombat-go/llm"
)

// streamChunk is one `data:` payload of a streamed completion.
//
// The protocol is nothing like Anthropic's typed block start/delta/stop
// events: every chunk is a slice of the same completion object, and the client
// is expected to merge them itself.
type streamChunk struct {
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
	Usage   *usageBody     `json:"usage"`
	Error   *errorBody     `json:"error"`
}

type streamChoice struct {
	Delta struct {
		Content contentText `json:"content"`

		// ReasoningContent is the scratchpad channel these gateways emit
		// alongside content; see chatChoice in response.go. It arrives in its
		// own chunks, typically all of them before the first content chunk.
		ReasoningContent contentText `json:"reasoning_content"`

		ToolCalls []streamToolCall `json:"tool_calls"`
	} `json:"delta"`
	FinishReason string `json:"finish_reason"`
}

// streamToolCall is a fragment of a tool call. Index — not id — is the key
// that is stable across chunks: id and name arrive once, in the first fragment,
// and arguments arrive afterwards as bare string slices with neither.
type streamToolCall struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string    `json:"name"`
		Arguments argString `json:"arguments"`
	} `json:"function"`
}

// callBuilder accumulates one tool call across chunks.
type callBuilder struct {
	id   string
	name string
	args strings.Builder

	// announced records that the opening delta for this index has gone out.
	// Needed because some gateways repeat id and name on every fragment, and
	// the shared rule is that a consumer sees them exactly once.
	announced bool
}

// emitToolArgs forwards one fragment of a tool call's arguments to the sink.
//
// SHARED RULE — llm/anthropic implements the identical contract, because a
// front end switches on the event, not on the provider:
//
//   - Index keys the call within one response and is on every fragment. Here it
//     is the tool_calls[] index; in llm/anthropic it is the content-block
//     index. Both are dense per response and both match the order of the
//     ToolUse blocks in the assembled Response.
//   - ID and Name are sent exactly once, on the first delta for an Index, and
//     are empty on every fragment after it. "ID non-empty" therefore means "a
//     new call has started", whichever provider is behind it.
//   - JSON is a raw slice of the arguments text. It is not valid JSON alone;
//     only the concatenation of every fragment for one Index parses. Nothing
//     downstream may treat it as a document, and it must never be merged into
//     Delta.Text — it is not the answer.
//   - A delta with no ID, no Name and no JSON is not emitted: it says nothing.
//
// These deltas are display-only. The client still accumulates independently and
// reports only the finished call in the Response, so dropping every one of them
// changes nothing about what the harness dispatches.
func emitToolArgs(onDelta func(llm.Delta), d llm.ToolArgsDelta) {
	if onDelta == nil {
		return
	}
	if d.ID == "" && d.Name == "" && d.JSON == "" {
		return
	}
	onDelta(llm.Delta{ToolArgs: &d})
}

// doneSentinel ends the stream. It is not JSON, so it must be matched before
// any decode attempt.
const doneSentinel = "[DONE]"

// decodeStream consumes an SSE body and assembles the same llm.Response the
// non-streaming path would have produced. onDelta may be nil.
//
// One caveat it cannot paper over: usage only lands here if the server sends a
// usage chunk. Some gateways never do, even with stream_options set — see
// Config.Stream — and the assembled response then reports zero tokens.
func decodeStream(r io.Reader, onDelta func(llm.Delta), model string) (llm.Response, error) {
	var (
		text      strings.Builder
		reasoning strings.Builder
		calls     = map[int]*callBuilder{}
		finish    string
		usage     llm.Usage
		chunks    int
		reader    = sse.NewReader(r)
		didDone   bool
	)

	for reader.Next() {
		ev := reader.Event()
		data := strings.TrimSpace(string(ev.Data))
		if data == "" {
			continue
		}
		if data == doneSentinel {
			didDone = true
			break
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Keep-alives, provider-specific "ping" payloads and comment-ish
			// noise all land here. Dropping an undecodable chunk is safer
			// than aborting a stream that is otherwise producing text.
			continue
		}
		chunks++

		// An error can arrive mid-stream after a 200: the status was sent
		// before the model failed. Surface it instead of returning the
		// truncated prefix as if it were the answer.
		if chunk.Error != nil && chunk.Error.Message != "" {
			return llm.Response{}, &llm.APIError{Class: llm.ErrServer, Status: 200, Message: chunk.Error.String()}
		}
		if strings.EqualFold(ev.Type, "error") {
			return llm.Response{}, &llm.APIError{Class: llm.ErrServer, Status: 200, Message: truncate(data)}
		}

		if chunk.Model != "" {
			model = chunk.Model
		}
		// Usage rides on the final chunk (and only when stream_options asked
		// for it). Zero-valued usage objects appear on intermediate chunks
		// from some gateways, so only a non-empty one overwrites.
		if u := chunk.Usage.toUsage(); u.InputTokens > 0 || u.OutputTokens > 0 {
			usage = u
		}

		for _, ch := range chunk.Choices {
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
			// Reasoning first, so a chunk carrying both keeps the model's own
			// order, and always in its own Delta: merging it into Text would
			// print the scratchpad to the user as if it were the answer.
			if s := string(ch.Delta.ReasoningContent); s != "" {
				if onDelta != nil {
					onDelta(llm.Delta{Reasoning: s})
				}
				reasoning.WriteString(s)
			}
			if s := string(ch.Delta.Content); s != "" {
				if onDelta != nil {
					onDelta(llm.Delta{Text: s})
				}
				text.WriteString(s)
			}
			for _, tc := range ch.Delta.ToolCalls {
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}
				b := calls[idx]
				if b == nil {
					b = &callBuilder{}
					calls[idx] = b
				}
				if tc.ID != "" {
					b.id = tc.ID
				}
				if tc.Function.Name != "" {
					b.name = tc.Function.Name
				}
				// Arguments are concatenated verbatim: each fragment is a raw
				// slice of the JSON text, and only the join is valid JSON.
				frag := string(tc.Function.Arguments)
				b.args.WriteString(frag)

				// The opening fragment for an index usually carries id, name
				// and the first slice of arguments in one go, so all three ride
				// out on a single delta rather than an empty announcement
				// followed by the same bytes.
				d := llm.ToolArgsDelta{Index: idx, JSON: frag}
				if !b.announced && (b.id != "" || b.name != "") {
					d.ID, d.Name = llm.ToolUseID(b.id), b.name
					b.announced = true
				}
				emitToolArgs(onDelta, d)
			}
		}
	}

	if err := reader.Err(); err != nil {
		return llm.Response{}, llm.ClassifyTransport(fmt.Errorf("openai: read stream: %w", err))
	}
	if chunks == 0 && !didDone {
		// A 200 followed by silence is the classic wedged-gateway signature.
		// Reporting it as transport makes it retryable, which is right.
		return llm.Response{}, llm.ClassifyTransport(fmt.Errorf("openai: stream closed before any event"))
	}

	var content []llm.ContentBlock
	// Same ordering and the same round-trip argument as decodeResponse: the
	// scratchpad goes before the text, and request.go's encodeAssistant drops
	// llm.Thinking, so it never travels back upstream.
	if reasoning.Len() > 0 {
		content = append(content, llm.Thinking{Text: reasoning.String()})
	}
	if text.Len() > 0 {
		content = append(content, llm.Text{Text: text.String()})
	}
	// Emitted in index order so two replays of the same stream produce the
	// same block order regardless of map iteration.
	idxs := make([]int, 0, len(calls))
	for i := range calls {
		idxs = append(idxs, i)
	}
	sort.Ints(idxs)
	for _, i := range idxs {
		b := calls[i]
		content = append(content, llm.ToolUse{
			ID:    llm.ToolUseID(b.id),
			Name:  b.name,
			Input: toolInput(b.args.String()),
		})
	}

	return llm.Response{
		Content:    content,
		StopReason: stopReason(finish),
		Usage:      usage,
		Model:      model,
	}, nil
}
