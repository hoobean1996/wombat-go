package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/internal/sse"
)

// streamEvent is the union of every Messages-API stream event. One struct
// rather than a decode-twice discriminator: the variants share most of their
// fields, and the pointers make "absent" unambiguous.
type streamEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`

	Message      *apiMessage  `json:"message"`       // message_start
	ContentBlock *apiBlock    `json:"content_block"` // content_block_start
	Delta        *streamDelta `json:"delta"`         // content_block_delta, message_delta
	Usage        *llm.Usage   `json:"usage"`         // message_delta
	Error        *apiError    `json:"error"`         // error
}

type streamDelta struct {
	Type string `json:"type"`

	Text        string `json:"text"`         // text_delta
	PartialJSON string `json:"partial_json"` // input_json_delta
	Thinking    string `json:"thinking"`     // thinking_delta
	Signature   string `json:"signature"`    // signature_delta

	StopReason string `json:"stop_reason"` // message_delta
}

type apiError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// blockAcc accumulates one content block as its deltas arrive.
type blockAcc struct {
	kind      string // text | tool_use | thinking; empty for a type we ignore
	text      strings.Builder
	thinking  strings.Builder
	signature strings.Builder
	input     strings.Builder
	id        string
	name      string
}

// streamState is the whole reply under construction. Owned by one goroutine —
// the one calling Complete — so no locking is needed or wanted.
type streamState struct {
	blocks []*blockAcc
	stop   llm.StopReason
	usage  llm.Usage
	model  string
}

// at returns the accumulator for an SSE block index, growing the slice as
// needed. Indices are contiguous in practice; the slice preserves emission
// order without a sort, and a gap left by a skipped index stays nil.
func (s *streamState) at(i int) *blockAcc {
	if i < 0 {
		return nil
	}
	for len(s.blocks) <= i {
		s.blocks = append(s.blocks, nil)
	}
	if s.blocks[i] == nil {
		s.blocks[i] = &blockAcc{}
	}
	return s.blocks[i]
}

// readStream consumes the event stream and assembles the same llm.Response the
// non-streaming path would have produced.
//
// cancel is the request context's cause-cancel: it powers the idle watchdog,
// which is the one timeout net/http cannot express. ResponseHeaderTimeout is
// already spent by the time we get here — the headers arrived — so a stream
// that wedges mid-body has nothing else watching it.
func (c *Client) readStream(
	ctx context.Context,
	body io.Reader,
	onDelta func(llm.Delta),
	model string,
	cancel context.CancelCauseFunc,
) (llm.Response, error) {
	idle := time.AfterFunc(streamIdleTimeout, func() { cancel(errStreamIdle) })
	defer idle.Stop()

	st := streamState{model: model}
	r := sse.NewReader(body)
	saw := false

	for r.Next() {
		idle.Reset(streamIdleTimeout)

		ev := r.Event()
		if len(ev.Data) == 0 {
			continue
		}
		saw = true
		if err := st.handle(ev, onDelta); err != nil {
			return llm.Response{}, err
		}
	}

	if err := r.Err(); err != nil {
		return llm.Response{}, abortError(ctx, fmt.Errorf("anthropic: read stream: %w", err))
	}
	if !saw {
		// A 200 with an empty body is a proxy or gateway artifact, never the
		// model. Classified as transport so it is retried rather than reported
		// as an empty answer.
		return llm.Response{}, llm.ClassifyTransport(errors.New("anthropic: empty event stream"))
	}
	return st.response(), nil
}

// handle folds one event into the state.
func (s *streamState) handle(ev sse.Event, onDelta func(llm.Delta)) error {
	var e streamEvent
	if err := json.Unmarshal(ev.Data, &e); err != nil {
		// Ignore, as the OCaml original did. A payload we cannot parse is
		// either a keep-alive comment the framing let through or a field we do
		// not model; killing a half-finished reply over it loses real work.
		return nil
	}

	// The SSE "event:" line and the JSON "type" always agree; the JSON wins
	// because gateways have been known to drop the event line.
	typ := e.Type
	if typ == "" {
		typ = ev.Type
	}

	switch typ {
	case "message_start":
		if e.Message == nil {
			return nil
		}
		if e.Message.Model != "" {
			s.model = e.Message.Model
		}
		// Usage arrives split in two: message_start carries the input side
		// (including the cache read/write split, which is the number that says
		// whether the breakpoints in request.go are working), message_delta
		// carries the final output count. Merge, do not overwrite.
		s.usage = e.Message.Usage

	case "content_block_start":
		b := s.at(e.Index)
		if b == nil || e.ContentBlock == nil {
			return nil
		}
		switch e.ContentBlock.Type {
		case "text":
			b.kind = "text"
			b.text.WriteString(e.ContentBlock.Text)
		case "thinking":
			b.kind = "thinking"
			b.thinking.WriteString(e.ContentBlock.Thinking)
			b.signature.WriteString(e.ContentBlock.Signature)
		case "tool_use":
			b.kind = "tool_use"
			b.id, b.name = e.ContentBlock.ID, e.ContentBlock.Name
			// The start event's "input" is always {}; the real arguments come
			// as input_json_delta fragments. So the opening delta announces the
			// call — id and name — and carries no JSON.
			emitToolArgs(onDelta, llm.ToolArgsDelta{
				Index: e.Index,
				ID:    llm.ToolUseID(b.id),
				Name:  b.name,
			})
		default:
			// Leaves kind empty, so the block is dropped at finalize.
		}

	case "content_block_delta":
		b := s.at(e.Index)
		if b == nil || e.Delta == nil {
			return nil
		}
		switch e.Delta.Type {
		case "text_delta":
			b.text.WriteString(e.Delta.Text)
			if onDelta != nil {
				// Forwarded raw. Scrubbing here would be wrong as well as
				// wasteful: a multi-byte character can straddle two deltas, so
				// a per-delta scrub would replace the two halves of a valid
				// codepoint with two U+FFFDs. The accumulated block is
				// scrubbed once, at finalize.
				onDelta(llm.Delta{Text: e.Delta.Text})
			}
		case "input_json_delta":
			b.input.WriteString(e.Delta.PartialJSON)
			// Only for a block we announced. The same delta type also carries
			// the arguments of server-side tools (web_search and friends),
			// whose content_block_start we deliberately do not model; an index
			// the UI never saw opened would arrive as an orphan fragment.
			//
			// Forwarded unscrubbed for the same reason text deltas are: a
			// multi-byte character can straddle two fragments. The joined
			// document is scrubbed once, in toolInput.
			if b.kind == "tool_use" {
				emitToolArgs(onDelta, llm.ToolArgsDelta{Index: e.Index, JSON: e.Delta.PartialJSON})
			}
		case "thinking_delta":
			b.thinking.WriteString(e.Delta.Thinking)
			if onDelta != nil {
				// Kept on its own field, never merged into Text: extended
				// thinking is often most of the generated tokens, so a UI that
				// cannot show it looks stalled — but it is display-only, and a
				// caller that concatenated it into the answer would be quoting
				// the model's scratchpad back at the user.
				onDelta(llm.Delta{Reasoning: e.Delta.Thinking})
			}
		case "signature_delta":
			b.signature.WriteString(e.Delta.Signature)
		}

	case "message_delta":
		if e.Delta != nil && e.Delta.StopReason != "" {
			s.stop = llm.StopReason(e.Delta.StopReason)
		}
		if e.Usage != nil {
			// Only the non-zero fields: this event reports output_tokens and
			// usually repeats nothing else, and a blind copy would erase the
			// input and cache counts from message_start.
			if e.Usage.OutputTokens > 0 {
				s.usage.OutputTokens = e.Usage.OutputTokens
			}
			if e.Usage.InputTokens > 0 {
				s.usage.InputTokens = e.Usage.InputTokens
			}
			if e.Usage.CacheWriteTokens > 0 {
				s.usage.CacheWriteTokens = e.Usage.CacheWriteTokens
			}
			if e.Usage.CacheReadTokens > 0 {
				s.usage.CacheReadTokens = e.Usage.CacheReadTokens
			}
		}

	case "error":
		return streamError(e.Error)

	case "content_block_stop", "message_stop", "ping":
		// Nothing to do: blocks are finalized in one pass at the end, and the
		// stream's own EOF is what ends the loop.

	default:
		// Unknown event types are ignored on purpose — the API introduces them
		// without a version bump.
	}
	return nil
}

// emitToolArgs forwards one fragment of a tool call's arguments to the sink.
//
// SHARED RULE — llm/openai implements the identical contract, because a front
// end switches on the event, not on the provider:
//
//   - Index keys the call within one response and is on every fragment. Here it
//     is the content-block index; in llm/openai it is the tool_calls[] index.
//     Both are dense per response and both match the order of the ToolUse
//     blocks in the assembled Response.
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

// streamError converts an in-band error event.
//
// llm.ClassifyStatus is unusable here: the HTTP status was 200 and the failure
// was announced inside the body, so there is no status to classify. The
// mapping is done by the provider's own error type string instead, which keeps
// an overloaded_error mid-stream retryable — the single most common way a long
// generation dies.
func streamError(e *apiError) error {
	if e == nil {
		return &llm.APIError{Class: llm.ErrServer, Message: "anthropic: stream error event with no detail"}
	}
	class := llm.ErrServer
	switch e.Type {
	case "overloaded_error":
		class = llm.ErrOverloaded
	case "rate_limit_error":
		class = llm.ErrRateLimit
	case "invalid_request_error":
		class = llm.ErrBadRequest
	case "authentication_error", "permission_error":
		class = llm.ErrAuth
	case "not_found_error":
		class = llm.ErrNotFound
	}
	return &llm.APIError{
		Class:   class,
		Message: truncate(scrub(e.Type+": "+e.Message), maxErrorMessage),
	}
}

// response flattens the accumulators in index order.
func (s *streamState) response() llm.Response {
	content := make([]llm.ContentBlock, 0, len(s.blocks))
	for _, b := range s.blocks {
		if b == nil {
			continue
		}
		switch b.kind {
		case "text":
			t := scrub(b.text.String())
			if t == "" {
				// An empty text block is legal to receive and illegal to send
				// back, and this reply goes straight into the transcript that
				// becomes the next request.
				continue
			}
			content = append(content, llm.Text{Text: t})
		case "thinking":
			content = append(content, llm.Thinking{
				Text:      scrub(b.thinking.String()),
				Signature: b.signature.String(),
			})
		case "tool_use":
			content = append(content, llm.ToolUse{
				ID:    llm.ToolUseID(b.id),
				Name:  b.name,
				Input: toolInput(b.input.String()),
			})
		}
	}
	return llm.Response{
		Content:    content,
		StopReason: s.stop,
		Usage:      s.usage,
		Model:      s.model,
	}
}

// toolInput turns accumulated partial_json fragments into an input document.
//
// A generation cut off by max_tokens mid-argument leaves an unparseable
// fragment. It is kept, encoded as a JSON string rather than discarded: that
// round-trips through json.Marshal on the next request (raw invalid JSON in a
// RawMessage would not — it fails the whole call), and the tool's own schema
// validation rejects it with a message the model can read and correct.
func toolInput(raw string) json.RawMessage {
	if raw == "" {
		return emptyInput
	}
	clean := scrub(raw)
	if json.Valid([]byte(clean)) {
		return json.RawMessage(clean)
	}
	b, err := json.Marshal(clean)
	if err != nil {
		return emptyInput
	}
	return json.RawMessage(b)
}
