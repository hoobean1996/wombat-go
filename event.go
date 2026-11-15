package wombat

import (
	"bytes"
	"context"
	"encoding/json"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

// Event is one observation from a running agent.
//
// The interface is open on purpose: an orchestration layer built on top of
// this package (a plan-act driver, a verifier loop) defines its own event
// types and they flow through the same stream. That is the extensibility an
// effect-based runtime buys with an open effect type, recovered here for free
// because a stream of interface values is already open.
//
// Diagnostics do NOT go through here. Log lines belong to log/slog; this
// stream carries semantic events a front end renders.
type Event interface {
	Kind() string
}

// IterStart opens one ReAct iteration.
type IterStart struct {
	N   int `json:"n"`
	Max int `json:"max"`
}

// LLMStart precedes a model call.
type LLMStart struct {
	Model   string      `json:"model,omitempty"`
	Purpose llm.Purpose `json:"purpose,omitempty"`
	Tools   int         `json:"tools"`
	Forced  string      `json:"forced_tool,omitempty"`
}

// TextDelta is streamed assistant text.
type TextDelta struct {
	Text string `json:"text"`
}

// ReasoningDelta is streamed scratchpad text from a reasoning model.
//
// A separate event rather than more TextDelta, because a front end must be
// able to render it differently: it is not the answer, and on some models it
// is most of the generated tokens. Folding it into TextDelta would put the
// model's private deliberation in the user's reply.
type ReasoningDelta struct {
	Text string `json:"text"`
}

// ToolArgsDelta is streamed tool-call arguments, as the model writes them.
//
// Separate from [ToolStart], which fires only once the call is complete and
// about to run. Between the two there can be several seconds of the model
// composing a long argument — a file's whole new contents, say — and without
// this the UI has nothing to show for it.
type ToolArgsDelta struct {
	Index int           `json:"index"`
	UseID llm.ToolUseID `json:"use_id,omitempty"`
	Name  string        `json:"name,omitempty"`
	Text  string        `json:"text"`
}

// LLMDone reports a completed model call.
type LLMDone struct {
	Model      string         `json:"model,omitempty"`
	StopReason llm.StopReason `json:"stop_reason,omitempty"`
	Usage      llm.Usage      `json:"usage"`
	Millis     int64          `json:"ms"`
}

// ToolStart opens one logical tool call, after retry and dedup have collapsed
// any internal attempts.
type ToolStart struct {
	UseID    llm.ToolUseID   `json:"use_id"`
	Name     string          `json:"name"`
	Category string          `json:"category,omitempty"`
	Input    json.RawMessage `json:"input,omitempty"`
}

// ToolDone closes one logical tool call.
type ToolDone struct {
	UseID  llm.ToolUseID `json:"use_id"`
	Name   string        `json:"name"`
	OK     bool          `json:"ok"`
	Output string        `json:"output,omitempty"`
	Error  string        `json:"error,omitempty"`
	Millis int64         `json:"ms"`

	// Err is the failure itself, for an in-process consumer that has to
	// classify it. Not serialized, and Error carries the same text.
	//
	// Both, because an error's identity does not survive being stringified.
	// "Was this call refused by the permission gate?" is answered by
	// errors.Is(ev.Err, permission.ErrDenied) and cannot be answered by
	// searching Error for a prefix — a substring match on an error message is
	// a contract nobody agreed to, and it breaks the first time the gate
	// rewords itself. A consumer reading events back from JSON necessarily
	// sees a nil Err and must fall back to structured signals such as
	// permission.Decided.
	Err error `json:"-"`
}

// TurnTruncated reports a reply cut off at max_tokens that is being asked for
// again, shorter.
//
// Emitted rather than kept quiet because the retry costs a full generation at
// the cap and shows up nowhere else: a front end that only renders text would
// otherwise show a long pause and then an unrelated answer.
type TurnTruncated struct {
	Attempt   int `json:"attempt"`
	MaxTokens int `json:"max_tokens"`
}

// EmptyTurn reports a reply that said nothing and called nothing, which the
// loop asks for again rather than reporting as an answer. See EmptyTurnError
// for what it gives up on.
type EmptyTurn struct {
	Attempt        int            `json:"attempt"`
	StopReason     llm.StopReason `json:"stop_reason,omitempty"`
	ReasoningBytes int            `json:"reasoning_bytes,omitempty"`
}

// LLMRetry reports that the retry middleware is about to try a failed model
// call again, after Wait. Without it a UI shows a call that is merely slow —
// half a minute of nothing while a rate limit backs off.
type LLMRetry struct {
	Attempt    int    `json:"attempt"`
	Of         int    `json:"of"`
	WaitMillis int64  `json:"wait_ms"`
	Error      string `json:"error,omitempty"`
}

// ContextTrimmed reports that the transcript strategy is now sending less than
// the stored transcript: Dropped messages left out, Elided tool results sent
// without their bodies. Emitted when that changes, not on every call.
type ContextTrimmed struct {
	Strategy string `json:"strategy"`
	Dropped  int    `json:"dropped,omitempty"`
	Elided   int    `json:"elided,omitempty"`
	// Tokens is the estimated size of what is sent.
	Tokens int `json:"tokens"`
}

// ContextOverflow reports that the provider refused a request as too long and
// [WithOverflowRecovery] is retrying it shorter, at Level of its ladder.
type ContextOverflow struct {
	Level          int    `json:"level"`
	Strategy       string `json:"strategy"`
	MessagesBefore int    `json:"messages_before"`
	MessagesAfter  int    `json:"messages_after"`
}

// TranscriptCompacted reports that [WithCompaction] folded the oldest Folded
// messages of the stored transcript into a summary.
type TranscriptCompacted struct {
	Folded       int `json:"folded"`
	TokensBefore int `json:"tokens_before"`
	TokensAfter  int `json:"tokens_after"`
	SummaryBytes int `json:"summary_bytes"`
}

// Steered reports that messages sent with [Run.Steer] have been put into the
// conversation, ahead of the next model call.
type Steered struct {
	Messages int `json:"messages"`
}

// Spend is a budget snapshot, emitted once per model call.
type Spend struct {
	CostUSD      float64 `json:"cost_usd"`
	Calls        int     `json:"calls"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	CacheWrite   int     `json:"cache_write_tokens,omitempty"`
	CacheRead    int     `json:"cache_read_tokens,omitempty"`
	ElapsedSec   float64 `json:"elapsed_sec"`
}

// Kind implements Event.
func (IterStart) Kind() string { return "iter_start" }

// Kind implements Event.
func (TurnTruncated) Kind() string { return "turn_truncated" }

// Kind implements Event.
func (LLMStart) Kind() string { return "llm_start" }

// Kind implements Event.
func (TextDelta) Kind() string { return "text_delta" }

// Kind implements Event.
func (ReasoningDelta) Kind() string { return "reasoning_delta" }

// Kind implements Event.
func (ToolArgsDelta) Kind() string { return "tool_args_delta" }

// Kind implements Event.
func (LLMDone) Kind() string { return "llm_done" }

// Kind implements Event.
func (ToolStart) Kind() string { return "tool_start" }

// Kind implements Event.
func (ToolDone) Kind() string { return "tool_done" }

// Kind implements Event.
func (Spend) Kind() string { return "spend" }

// Kind implements Event.
func (Steered) Kind() string { return "steered" }

// MarshalJSON implements json.Marshaler.
func (e Steered) MarshalJSON() ([]byte, error) { type r Steered; return eventJSON(e.Kind(), r(e)) }

// Kind implements Event.
func (EmptyTurn) Kind() string { return "empty_turn" }

// Kind implements Event.
func (LLMRetry) Kind() string { return "llm_retry" }

// Kind implements Event.
func (ContextTrimmed) Kind() string { return "context_trimmed" }

// Kind implements Event.
func (ContextOverflow) Kind() string { return "context_overflow" }

// Kind implements Event.
func (TranscriptCompacted) Kind() string { return "transcript_compacted" }

// MarshalJSON implements json.Marshaler.
func (e EmptyTurn) MarshalJSON() ([]byte, error) { type r EmptyTurn; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e LLMRetry) MarshalJSON() ([]byte, error) { type r LLMRetry; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e ContextTrimmed) MarshalJSON() ([]byte, error) {
	type r ContextTrimmed
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements json.Marshaler.
func (e ContextOverflow) MarshalJSON() ([]byte, error) {
	type r ContextOverflow
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements json.Marshaler.
func (e TranscriptCompacted) MarshalJSON() ([]byte, error) {
	type r TranscriptCompacted
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements json.Marshaler.
func (e IterStart) MarshalJSON() ([]byte, error) { type r IterStart; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e LLMStart) MarshalJSON() ([]byte, error) { type r LLMStart; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e TextDelta) MarshalJSON() ([]byte, error) { type r TextDelta; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e ReasoningDelta) MarshalJSON() ([]byte, error) {
	type r ReasoningDelta
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements json.Marshaler.
func (e ToolArgsDelta) MarshalJSON() ([]byte, error) {
	type r ToolArgsDelta
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements json.Marshaler.
func (e LLMDone) MarshalJSON() ([]byte, error) { type r LLMDone; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e ToolStart) MarshalJSON() ([]byte, error) { type r ToolStart; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e ToolDone) MarshalJSON() ([]byte, error) { type r ToolDone; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e Spend) MarshalJSON() ([]byte, error) { type r Spend; return eventJSON(e.Kind(), r(e)) }

// MarshalJSON implements json.Marshaler.
func (e TurnTruncated) MarshalJSON() ([]byte, error) {
	type r TurnTruncated
	return eventJSON(e.Kind(), r(e))
}

// EventTypes returns a zero value of every event type this package defines,
// in the order a reader most likely wants them.
//
// It exists because Go cannot check that a type switch over an interface is
// exhaustive, and [Event] is deliberately open so orchestration layers can add
// their own. A renderer, a code generator or a test can walk this list instead
// of hand-maintaining a parallel one somewhere else, where the rot is silent.
//
// The list itself still has to be updated when a variant is added — but it is
// one list, in the same file as the types, and the generator that consumes it
// fails loudly rather than quietly omitting an event a front end then never
// learns to render.
func EventTypes() []Event {
	return []Event{
		IterStart{},
		TurnTruncated{},
		EmptyTurn{},
		Steered{},
		CompletionChecked{},
		ContextTrimmed{},
		TranscriptCompacted{},
		LLMStart{},
		LLMRetry{},
		ContextOverflow{},
		ReasoningDelta{},
		ToolArgsDelta{},
		TextDelta{},
		LLMDone{},
		ToolStart{},
		ToolDone{},
		Spend{},
		SubagentStart{},
		SubagentEvent{},
		SubagentEnd{},
	}
}

// eventJSON marshals v as an object with "type" spliced in as the first key.
//
// Field order is part of the wire contract for any consumer that golden-tests
// the stream, so the events are structs (ordered) rather than maps (sorted),
// and the discriminator is prepended rather than merged.
func eventJSON(kind string, v any) ([]byte, error) {
	body, err := marshalNoEscape(v)
	if err != nil {
		return nil, err
	}
	k, err := marshalNoEscape(kind)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, len(body)+len(k)+9)
	out = append(out, `{"type":`...)
	out = append(out, k...)
	if len(body) > 2 {
		out = append(out, ',')
		out = append(out, body[1:len(body)-1]...)
	}
	return append(out, '}'), nil
}

// marshalNoEscape encodes v without HTML escaping.
//
// Model output is full of <, > and &. Escaping them to < is valid JSON,
// but every consumer that is not a browser then has to read it that way.
//
// The subtlety is that escaping is decided twice: encoding/json re-runs its
// compact pass over whatever a custom MarshalJSON returns, using the OUTER
// encoder's setting. An event therefore survives unescaped only if both halves
// agree — the consumer calls SetEscapeHTML(false) (see cmd/wombat-jsonl), and
// this function does the same here. Plain json.Marshal inside would escape the
// bytes before the outer encoder ever saw them, and nothing downstream could
// undo it. Conversely a caller using json.Marshal directly still gets escaped
// output, because its compact pass re-escapes; that is the caller's choice.
func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// Millis is a small helper for the *Done events.
func millis(d time.Duration) int64 { return d.Milliseconds() }

// ===== Emitter =====

// Emitter receives events from a run.
type Emitter func(Event)

type emitterKey struct{}

// WithEmitter attaches an event sink to ctx.
//
// The sink is per-run while the middleware chain that feeds it is built once
// per agent, so it travels on the context rather than being captured in a
// closure. Same rule as tool.Info: dependencies are injected at construction,
// request-scoped data rides on ctx.
func WithEmitter(ctx context.Context, e Emitter) context.Context {
	return context.WithValue(ctx, emitterKey{}, e)
}

// Emit sends an event to the sink on ctx. A no-op when there is none, so code
// that emits does not need to know whether anyone is listening.
func Emit(ctx context.Context, ev Event) {
	if e, ok := ctx.Value(emitterKey{}).(Emitter); ok && e != nil {
		e(ev)
	}
}
