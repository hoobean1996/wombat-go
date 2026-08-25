// Package llm is the provider-neutral vocabulary for talking to a chat model,
// plus the middleware chain that wraps a provider client.
//
// It deliberately imports nothing outside the standard library: everything
// above it (tool, governor, wombat) depends on these types, so they have to
// sit at the bottom of the graph.
//
// Provider wire encoding does NOT live here. [Message] and [ContentBlock] are
// the domain shapes; llm/anthropic and llm/openai each own their own request
// encoders, because the two APIs disagree about almost everything (tool
// results are content blocks in one and whole messages in the other).
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
)

// ToolUseID identifies one tool invocation. It is minted by the provider and
// pairs a ToolUse block with its ToolResult.
//
// The named type is not a security boundary — ToolUseID(someString) compiles —
// but it does stop the accidental case, which is the one that actually
// happened: passing a tool *name* where a use-id was expected.
type ToolUseID string

func (id ToolUseID) String() string { return string(id) }

// Role is who produced a message. There is no System role: both providers take
// the system prompt out of band.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// StopReason is why the model stopped generating.
//
// Modeled as an open string rather than a closed enum: providers add reasons
// (and gateways invent them), and an unknown value should flow through to the
// caller intact rather than being flattened into a catch-all.
type StopReason string

const (
	StopEndTurn      StopReason = "end_turn"
	StopToolUse      StopReason = "tool_use"
	StopMaxTokens    StopReason = "max_tokens"
	StopStopSequence StopReason = "stop_sequence"
	StopRefusal      StopReason = "refusal"
)

// Purpose tags what a call is FOR, so middleware can branch on the semantic
// role of the call instead of guessing from the prompt or the model name.
type Purpose string

const (
	PurposeOther      Purpose = "other"
	PurposePlanner    Purpose = "planner"
	PurposeExecutor   Purpose = "executor"
	PurposeRecovery   Purpose = "recovery"
	PurposeSummarizer Purpose = "summarizer"
	PurposeSubagent   Purpose = "subagent"
	PurposeVerifier   Purpose = "verifier"
)

// Usage is the token accounting for one call.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheWriteTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadTokens  int `json:"cache_read_input_tokens,omitempty"`
}

// Add accumulates u2 into u.
func (u *Usage) Add(u2 Usage) {
	u.InputTokens += u2.InputTokens
	u.OutputTokens += u2.OutputTokens
	u.CacheWriteTokens += u2.CacheWriteTokens
	u.CacheReadTokens += u2.CacheReadTokens
}

// ===== Content blocks =====

// ContentBlock is one piece of a message. The set is closed: the unexported
// method means only this package can add variants.
//
//	Text | ToolUse | ToolResult | Thinking
type ContentBlock interface {
	blockKind() string
}

// Text is a plain text span.
type Text struct {
	Text string
}

// ToolUse is the model asking to run a tool.
type ToolUse struct {
	ID    ToolUseID
	Name  string
	Input json.RawMessage
}

// ToolResult answers a ToolUse. It always rides in a RoleUser message.
type ToolResult struct {
	ToolUseID ToolUseID
	Content   string
	IsError   bool
}

// Thinking is an extended-thinking block. Carried through verbatim: the
// signature must survive a round trip or the provider rejects the next turn.
type Thinking struct {
	Text      string
	Signature string
}

func (Text) blockKind() string       { return "text" }
func (ToolUse) blockKind() string    { return "tool_use" }
func (ToolResult) blockKind() string { return "tool_result" }
func (Thinking) blockKind() string   { return "thinking" }

// Message is one conversational turn.
type Message struct {
	Role    Role
	Content []ContentBlock
}

// TextOf concatenates every Text block, newline-separated. Thinking blocks are
// excluded — they are not the answer.
func TextOf(blocks []ContentBlock) string {
	var out []byte
	for _, b := range blocks {
		t, ok := b.(Text)
		if !ok {
			continue
		}
		if len(out) > 0 {
			out = append(out, '\n')
		}
		out = append(out, t.Text...)
	}
	return string(out)
}

// ToolUses returns every ToolUse block in order.
func ToolUses(blocks []ContentBlock) []ToolUse {
	var out []ToolUse
	for _, b := range blocks {
		if tu, ok := b.(ToolUse); ok {
			out = append(out, tu)
		}
	}
	return out
}

// UserText builds a single-block user turn.
func UserText(s string) Message {
	return Message{Role: RoleUser, Content: []ContentBlock{Text{Text: s}}}
}

// ===== Domain JSON =====
//
// This is our own persistence / JSONL shape, not a provider's. It happens to
// look like Anthropic's because that is the lineage, but providers encode
// their own requests; nothing here is on the wire to a model.

type blockEnvelope struct {
	Type string `json:"type"`

	Text      string          `json:"text,omitempty"`
	ID        ToolUseID       `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID ToolUseID       `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	Signature string          `json:"signature,omitempty"`
}

// MarshalJSON implements json.Marshaler for Message.
func (m Message) MarshalJSON() ([]byte, error) {
	envs := make([]blockEnvelope, 0, len(m.Content))
	for _, b := range m.Content {
		e := blockEnvelope{Type: b.blockKind()}
		switch v := b.(type) {
		case Text:
			e.Text = v.Text
		case ToolUse:
			e.ID, e.Name, e.Input = v.ID, v.Name, v.Input
		case ToolResult:
			e.ToolUseID, e.Content, e.IsError = v.ToolUseID, v.Content, v.IsError
		case Thinking:
			e.Text, e.Signature = v.Text, v.Signature
		default:
			return nil, fmt.Errorf("llm: unknown content block %T", b)
		}
		envs = append(envs, e)
	}
	return marshalNoEscape(struct {
		Role    Role            `json:"role"`
		Content []blockEnvelope `json:"content"`
	}{m.Role, envs})
}

// marshalNoEscape encodes without turning <, > and & into \u003c and friends.
//
// A transcript is model traffic: angle brackets and ampersands are everywhere
// in it, and a persisted conversation, a recorded tape or a diff of either is
// read by people. Escaping is also decided twice — encoding/json re-compacts a
// Marshaler's output with the OUTER encoder's setting — so a consumer that
// carefully disables escaping still gets escaped messages unless it is
// disabled here too, at the point the bytes are made.
func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// UnmarshalJSON implements json.Unmarshaler for Message.
func (m *Message) UnmarshalJSON(b []byte) error {
	var raw struct {
		Role    Role            `json:"role"`
		Content []blockEnvelope `json:"content"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	m.Content = make([]ContentBlock, 0, len(raw.Content))
	for _, e := range raw.Content {
		switch e.Type {
		case "text":
			m.Content = append(m.Content, Text{Text: e.Text})
		case "tool_use":
			m.Content = append(m.Content, ToolUse{ID: e.ID, Name: e.Name, Input: e.Input})
		case "tool_result":
			m.Content = append(m.Content, ToolResult{ToolUseID: e.ToolUseID, Content: e.Content, IsError: e.IsError})
		case "thinking":
			m.Content = append(m.Content, Thinking{Text: e.Text, Signature: e.Signature})
		default:
			return fmt.Errorf("llm: unknown content block type %q", e.Type)
		}
	}
	return nil
}

// ===== Request / Response =====

// ToolSpec is what the model is told a tool looks like. It is the wire subset
// of a tool definition — no handler, no policy metadata.
type ToolSpec struct {
	Name        string
	Description string

	// InputSchema is passed through byte for byte.
	//
	// It is json.RawMessage and not map[string]any on purpose: Go marshals
	// map keys in sorted order, so round-tripping a schema through a map
	// reorders it. Schemas supplied by MCP servers and extensions would then
	// reach the model as different bytes than their author declared, which
	// both changes model behavior and breaks the prompt-cache prefix.
	InputSchema json.RawMessage
}

// ToolChoiceMode constrains whether and which tool the model must call.
type ToolChoiceMode string

const (
	// ChoiceAuto is the API default: the model decides. Omitted from the body.
	ChoiceAuto ToolChoiceMode = ""
	// ChoiceAny forces some tool call.
	ChoiceAny ToolChoiceMode = "any"
	// ChoiceTool forces one named tool.
	ChoiceTool ToolChoiceMode = "tool"
	// ChoiceNone offers tools but requires a text answer.
	ChoiceNone ToolChoiceMode = "none"
)

// ToolChoice is the tool_choice field. The zero value means ChoiceAuto.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string // only for ChoiceTool
}

// StreamMode controls whether a client uses server-sent events.
//
// A plain bool cannot express this: the useful default is "stream when
// somebody is listening", and Go's zero value would have to mean either
// always or never. StreamNever is not hypothetical — some gateways drop the
// usage record from a streamed reply, and a caller who needs token accounting
// more than it needs deltas has to be able to say so.
type StreamMode int

// Stream modes. The zero value is StreamAuto.
const (
	// StreamAuto streams when [Request.OnDelta] is set.
	StreamAuto StreamMode = iota
	// StreamAlways streams even with no delta sink, which keeps a long
	// generation from tripping an idle proxy timeout.
	StreamAlways
	// StreamNever always uses a single buffered response.
	StreamNever
)

// Enabled reports whether a request with the given sink should stream.
func (m StreamMode) Enabled(hasSink bool) bool {
	switch m {
	case StreamAlways:
		return true
	case StreamNever:
		return false
	default:
		return hasSink
	}
}

// Delta is a streamed fragment of a reply.
//
// One sink rather than one callback per channel: providers keep inventing
// channels (text, reasoning, citations, refusals), and a struct absorbs a new
// field where a second function field would be a breaking change to every
// caller.
type Delta struct {
	// Text is visible answer text.
	Text string

	// Reasoning is the model's scratchpad, where the provider exposes one:
	// Anthropic streams it as thinking deltas, OpenAI-compatible reasoning
	// models as a reasoning_content field. It is display-only — it is never
	// sent back upstream, and on some models it is the majority of the
	// generated tokens, so a UI that ignores it looks stalled.
	Reasoning string

	// ToolArgs is a fragment of a tool call's arguments, as the model writes
	// them. Nil on a text or reasoning delta.
	//
	// A client must still accumulate these itself and only report the finished
	// call in the Response — a half-written argument object is not something
	// the harness can dispatch. They are surfaced anyway because a model
	// composing a long tool call is otherwise several silent seconds, and a UI
	// that cannot say "it is writing a call to write_file" just looks hung.
	ToolArgs *ToolArgsDelta
}

// ToolArgsDelta is one fragment of a tool call being written.
type ToolArgsDelta struct {
	// Index distinguishes concurrent tool calls within one response. It is
	// the only field guaranteed present: both providers stream the id and the
	// name once, at the start, and the fragments that follow carry neither.
	Index int

	ID   ToolUseID // empty after the opening fragment
	Name string    // empty after the opening fragment

	// JSON is the raw fragment. Not valid JSON on its own — concatenating
	// every fragment for one Index is what yields the arguments object.
	JSON string
}

// ForceTool returns a ToolChoice pinning the model to one tool.
func ForceTool(name string) ToolChoice { return ToolChoice{Mode: ChoiceTool, Name: name} }

// Request is one model call.
type Request struct {
	System    string
	Messages  []Message
	Tools     []ToolSpec
	Choice    ToolChoice
	Model     string // "" inherits the client default
	MaxTokens int    // 0 inherits the client default
	Purpose   Purpose

	// Temperature and TopP control sampling. Nil means "leave it to the
	// provider", which is not the same as zero — 0 is a valid temperature and
	// the one that makes a run reproducible, so the two must be
	// distinguishable and a plain float64 cannot do it.
	//
	// They exist for two jobs the harness cannot do without them. Sampling n
	// trajectories for the same task needs temperature above zero or every
	// sample is the same trajectory; and pinning a run for a differential
	// diff needs it at zero.
	Temperature *float64
	TopP        *float64

	// OnDelta receives fragments as they arrive. Nil disables streaming
	// unless the client is configured with [StreamAlways].
	//
	// A function field on a request struct is a callback, not data: it is an
	// observability sink in the same spirit as httptrace.ClientTrace, and it
	// must not influence the bytes sent upstream. Middleware forwards it
	// unchanged.
	OnDelta func(Delta)
}

// Response is one model reply.
type Response struct {
	Content    []ContentBlock
	StopReason StopReason
	Usage      Usage
	Model      string
}

// ===== Client and middleware =====

// Client talks to one upstream model API.
//
// Implementations must be safe for concurrent use: the harness fans out
// sub-agents across goroutines that share a client.
type Client interface {
	Complete(ctx context.Context, req Request) (Response, error)
}

// ClientFunc adapts a function to Client, like http.HandlerFunc.
type ClientFunc func(ctx context.Context, req Request) (Response, error)

// Complete implements Client.
func (f ClientFunc) Complete(ctx context.Context, req Request) (Response, error) {
	return f(ctx, req)
}

// Middleware wraps a Client with one extra behavior.
type Middleware func(Client) Client

// Chain wraps base in mws. Later entries end up further OUT, so the call
// order reads top to bottom:
//
//	Chain(leaf, WithRetry(p), WithLogging(l))  // logging sees the post-retry verdict
//
// Ordering here is about semantics, never about dispatch: unlike a stack of
// effect handlers, a decorator cannot be shadowed by an outer one.
func Chain(base Client, mws ...Middleware) Client {
	for _, mw := range mws {
		base = mw(base)
	}
	return base
}
