package tape

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

	"github.com/hoobean1996/wombat-go/llm"
)

// entry is one line of the tape.
//
// Field order is the wire contract — this file gets diffed, and a reordered
// key set turns a one-line change into a whole-file change. Structs everywhere
// for the same reason: encoding/json sorts map keys, so a map would silently
// impose its own order on anything nested inside.
type entry struct {
	Kind     string          `json:"kind"`
	Key      string          `json:"key"`
	Seq      int             `json:"seq"`
	Request  json.RawMessage `json:"request"`
	Response json.RawMessage `json:"response"`
}

// canonical encodes v the one way this package ever encodes anything.
//
// Using it for BOTH the hash input and the bytes written to the file buys a
// property worth having: the "key" field on a line is the SHA-256 of the
// "request" field exactly as it appears on that line, verifiable with sha256
// and jq. That only holds if the two encodings agree about HTML escaping, so
// there is one function and no caller reaches for json.Marshal directly.
//
// Escaping is off because model traffic is full of <, > and &, and < in a
// file a human is expected to read a diff of helps nobody. json.RawMessage
// nested inside is passed through byte for byte when escaping is off, which is
// what keeps the request field identical to the bytes that were hashed.
func canonical(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// digest is the key: hex SHA-256 of the canonical request bytes.
func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// ===== LLM =====

// llmRequest is the model-facing projection of an llm.Request.
//
// What is here is what changes the bytes the provider sees. What is NOT here
// is as much of a decision:
//
//   - OnDelta is a func. It cannot be hashed, it cannot be recorded, and by
//     its own contract it must not influence the request — it is an
//     observability sink in the spirit of httptrace.ClientTrace.
//
//   - Purpose is a harness tag for routing middleware, never sent upstream. A
//     planner and an executor that build the identical request share a tape
//     entry, which is correct: the model cannot tell them apart either.
//
// Tool choice is a pointer so that a zero llm.ToolChoice and an explicit
// ChoiceAuto hash the same — the API omits the field in both cases.
//
// Temperature and TopP are pointers for the reason they are pointers on
// llm.Request: unset and zero are different requests, and 0 is the temperature
// that pins a run. They were missing at first, so two requests that differed
// only in sampling shared an entry — a recording at temperature 1 replayed for
// the run that asked for 0. They sit at the tail and are omitted when nil, so
// a request that sets neither encodes byte for byte as it always did and every
// key on an existing tape still matches.
//
// Only what the REQUEST says is keyed. A default a client fills in from its
// own config (anthropic.Config.Temperature, say) is applied below this
// middleware and is invisible to it; two tapes recorded through clients with
// different defaults are not told apart.
type llmRequest struct {
	Model       string        `json:"model,omitempty"`
	System      string        `json:"system,omitempty"`
	Messages    []llm.Message `json:"messages"`
	Tools       []toolSpec    `json:"tools,omitempty"`
	Choice      *toolChoice   `json:"tool_choice,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
}

type toolSpec struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`

	// InputSchema is hashed verbatim, matching llm.ToolSpec's own rule: a
	// schema round-tripped through map[string]any comes out with sorted keys,
	// which is different bytes to the model and a different prompt-cache
	// prefix. If the bytes the model saw differ, the entry must not match.
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type toolChoice struct {
	Mode llm.ToolChoiceMode `json:"mode"`
	Name string             `json:"name,omitempty"`
}

// llmResponse is a recorded reply.
//
// Content rides inside an llm.Message rather than being encoded here, because
// llm.ContentBlock is a closed interface whose only codec lives in that
// package. Duplicating the block envelope would work today and rot the moment
// a fifth block type is added: llm's decoder rejects an unknown type, a
// private copy would drop it. The cost is a redundant "role":"assistant" on
// every line, which is cheap.
type llmResponse struct {
	Content    llm.Message    `json:"content"`
	StopReason llm.StopReason `json:"stop_reason,omitempty"`
	Usage      llm.Usage      `json:"usage"`
	Model      string         `json:"model,omitempty"`

	// Streamed records whether the LIVE call actually delivered deltas, so a
	// replay can reproduce the caller's delta stream only when there was one.
	//
	// Without this the tape makes the diff worse instead of better in exactly
	// the setup where it is easiest to reach for: a buffered client, or
	// llm.StreamNever, emits no deltas at all, and a replay that synthesised
	// them would invent wombat.TextDelta events the recording never had.
	//
	// Last field on the line on purpose — an additive tail is the cheapest
	// place for a format to grow in a file that gets diffed.
	Streamed bool `json:"streamed,omitempty"`
}

// canonLLM projects and encodes a request, returning the bytes and their key.
func canonLLM(req llm.Request) ([]byte, string, error) {
	// Nil and empty slices marshal as null and [], so two logically identical
	// requests would hash differently depending on how the caller built them.
	msgs := req.Messages
	if msgs == nil {
		msgs = []llm.Message{}
	}

	var specs []toolSpec
	if len(req.Tools) > 0 {
		specs = make([]toolSpec, len(req.Tools))
		for i, s := range req.Tools {
			specs[i] = toolSpec{Name: s.Name, Description: s.Description, InputSchema: s.InputSchema}
		}
	}

	var choice *toolChoice
	if req.Choice.Mode != llm.ChoiceAuto || req.Choice.Name != "" {
		choice = &toolChoice{Mode: req.Choice.Mode, Name: req.Choice.Name}
	}

	b, err := canonical(llmRequest{
		Model:       req.Model,
		System:      req.System,
		Messages:    msgs,
		Tools:       specs,
		Choice:      choice,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	})
	if err != nil {
		return nil, "", err
	}
	return b, digest(b), nil
}

func encodeLLMResponse(resp llm.Response, streamed bool) ([]byte, error) {
	return canonical(llmResponse{
		Content:    llm.Message{Role: llm.RoleAssistant, Content: resp.Content},
		StopReason: resp.StopReason,
		Usage:      resp.Usage,
		Model:      resp.Model,
		Streamed:   streamed,
	})
}

func decodeLLMResponse(b []byte) (llm.Response, bool, error) {
	var r llmResponse
	if err := json.Unmarshal(b, &r); err != nil {
		return llm.Response{}, false, err
	}
	return llm.Response{
		Content:    r.Content.Content,
		StopReason: r.StopReason,
		Usage:      r.Usage,
		Model:      r.Model,
	}, r.Streamed, nil
}

// ===== Tools =====

// toolRequest keys a dispatch on the pair the model actually chose.
//
// Input is hashed verbatim rather than re-marshalled. Re-marshalling would let
// {"a":1,"b":2} and {"b":2,"a":1} share an entry, which sounds like a feature
// until a tool's behavior depends on argument order or on a duplicated key.
// The model emits the same bytes for the same decision, so verbatim is both
// stricter and, in practice, just as hit-prone.
type toolRequest struct {
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input,omitempty"`
}

// toolResponse records the Ok/Err distinction explicitly.
//
// OK is a separate field rather than being inferred from a non-empty Error,
// because a tool may legitimately fail with an empty message and a replay that
// turned that into success would be wrong in the worst possible direction:
// silently.
// toolResponse is a recorded tool call. Media is what the tool attached with
// tool.Attach; omitted when there was none, so every entry recorded before it
// existed decodes unchanged.
type toolResponse struct {
	OK     bool        `json:"ok"`
	Output string      `json:"output,omitempty"`
	Error  string      `json:"error,omitempty"`
	Media  []llm.Media `json:"media,omitempty"`
}

func canonTool(name string, input json.RawMessage) ([]byte, string, error) {
	b, err := canonical(toolRequest{Name: name, Input: input})
	if err != nil {
		return nil, "", err
	}
	return b, digest(b), nil
}

func encodeToolResponse(out string, err error, media []llm.Media) ([]byte, error) {
	tr := toolResponse{OK: err == nil, Output: out}
	if err != nil {
		tr.Error = err.Error()
	} else {
		tr.Media = media
	}
	return canonical(tr)
}
