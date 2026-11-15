package wire

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"wombat/pkg/core/llm"
)

// Behavior is one scripted wire outcome.
type Behavior struct {
	Name  string
	serve func(w http.ResponseWriter, r *http.Request)
}

// Success streams text and finishes properly, with a usage record.
func Success(text string) Behavior {
	return Behavior{Name: "success", serve: func(w http.ResponseWriter, _ *http.Request) {
		sseHeaders(w)
		for _, part := range chunkText(text, 5) {
			send(w, chunk{Choices: []chunkChoice{{Delta: chunkDelta{Content: part}}}})
		}
		send(w, chunk{Choices: []chunkChoice{{FinishReason: "stop"}}})
		send(w, chunk{Choices: []chunkChoice{}, Usage: &chunkUsage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120}})
		done(w)
	}}
}

// ToolCall streams one tool call, the way a gateway fragments it: id and name
// in the opening delta, arguments as bare slices afterwards.
func ToolCall(id, name, args string) Behavior {
	return Behavior{Name: "tool_call", serve: func(w http.ResponseWriter, _ *http.Request) {
		sseHeaders(w)
		open := chunkCall{Index: 0, ID: id, Type: "function"}
		open.Function.Name = name
		send(w, chunk{Choices: []chunkChoice{{Delta: chunkDelta{ToolCalls: []chunkCall{open}}}}})
		for _, part := range chunkText(args, 6) {
			frag := chunkCall{Index: 0}
			frag.Function.Arguments = part
			send(w, chunk{Choices: []chunkChoice{{Delta: chunkDelta{ToolCalls: []chunkCall{frag}}}}})
		}
		send(w, chunk{Choices: []chunkChoice{{FinishReason: "tool_calls"}}})
		send(w, chunk{Choices: []chunkChoice{}, Usage: &chunkUsage{PromptTokens: 120, CompletionTokens: 30, TotalTokens: 150}})
		done(w)
	}}
}

// EmptyNoUsage finishes cleanly having said nothing, and sends NO usage chunk
// at all.
//
// This is the exact shape run.go's empty-turn handling was written for, quoted
// from its own comment: "the stream ends with finish_reason 'stop', and the
// response carries no text, no tool call and no usage — not zero tokens, no
// usage chunk at all." Nothing in the repository has ever produced it on the
// wire; every existing test hands the loop a synthetic llm.Response instead.
func EmptyNoUsage() Behavior {
	return Behavior{Name: "empty_no_usage", serve: func(w http.ResponseWriter, _ *http.Request) {
		sseHeaders(w)
		send(w, chunk{Choices: []chunkChoice{{FinishReason: "stop"}}})
		done(w)
	}}
}

// EmptyAfterReasoning streams bytes of scratchpad, then finishes empty with no
// usage.
//
// The other observed cause, and the one that must NOT be retried: a model that
// spent its whole output budget thinking will spend it the same way next time.
// The loop tells the two apart by how much reasoning it saw, which is only
// possible because the deltas arrived — so this behavior is the only way to
// exercise that discrimination end to end.
func EmptyAfterReasoning(bytes int) Behavior {
	return Behavior{Name: "empty_after_reasoning", serve: func(w http.ResponseWriter, _ *http.Request) {
		sseHeaders(w)
		for _, part := range chunkText(strings.Repeat("think ", bytes/6+1), 60) {
			send(w, chunk{Choices: []chunkChoice{{Delta: chunkDelta{ReasoningContent: part}}}})
		}
		send(w, chunk{Choices: []chunkChoice{{FinishReason: "stop"}}})
		done(w)
	}}
}

// PartialThenReset streams text and then resets the socket mid-stream.
func PartialThenReset(text string) Behavior {
	return Behavior{Name: "partial_then_reset", serve: func(w http.ResponseWriter, _ *http.Request) {
		sseHeaders(w)
		for _, part := range chunkText(text, 5) {
			send(w, chunk{Choices: []chunkChoice{{Delta: chunkDelta{Content: part}}}})
		}
		reset(w)
	}}
}

// NoDoneSentinel streams a complete reply and then ends the connection without
// the [DONE] boundary.
func NoDoneSentinel(text string) Behavior {
	return Behavior{Name: "no_done_sentinel", serve: func(w http.ResponseWriter, _ *http.Request) {
		sseHeaders(w)
		for _, part := range chunkText(text, 5) {
			send(w, chunk{Choices: []chunkChoice{{Delta: chunkDelta{Content: part}}}})
		}
		send(w, chunk{Choices: []chunkChoice{{FinishReason: "stop"}}})
		closeClean(w)
	}}
}

// MalformedEvent sends a data line that is not valid JSON.
func MalformedEvent() Behavior {
	return Behavior{Name: "malformed_event", serve: func(w http.ResponseWriter, _ *http.Request) {
		sseHeaders(w)
		sendRaw(w, `{"choices":[{"delta":{"content":`)
		done(w)
	}}
}

// Status returns a non-2xx JSON error.
func Status(code int, message string) Behavior {
	return Behavior{
		Name: fmt.Sprintf("status_%d", code),
		serve: func(w http.ResponseWriter, _ *http.Request) {
			writeError(w, code, message, "api_error")
		},
	}
}

// RateLimit is a 429 with a Retry-After hint.
func RateLimit(after time.Duration) Behavior {
	return Behavior{Name: "rate_limit", serve: func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Retry-After", fmt.Sprintf("%d", int(after.Seconds())))
		writeError(w, http.StatusTooManyRequests, "slow down", "rate_limit_error")
	}}
}

// ContextOverflow is the 400 an OpenAI-compatible gateway returns when the
// prompt does not fit.
//
// The body matters more than the status: providers give overflow no distinct
// code, so llm.ClassifyStatus has to recognise it from the prose. Get that
// wrong and the error is a plain bad request, which is not retryable, so
// core.WithOverflowRecovery never engages and a run that could have
// compacted simply dies. That failure is invisible from the outside, which is
// why it is worth a wire test rather than a unit test over the classifier.
func ContextOverflow() Behavior {
	const body = "This model's maximum context length is 8192 tokens. " +
		"However, your messages resulted in 9001 tokens. Please reduce the length of the messages."
	return Behavior{Name: "context_overflow", serve: func(w http.ResponseWriter, _ *http.Request) {
		writeError(w, http.StatusBadRequest, body, "invalid_request_error")
	}}
}

// Stall sends SSE headers and then does nothing until the client gives up.
func Stall() Behavior {
	return Behavior{Name: "stall", serve: func(w http.ResponseWriter, r *http.Request) {
		sseHeaders(w)
		<-r.Context().Done()
	}}
}

// Assert that the package's public error vocabulary is what the behaviors are
// written against; a rename in llm should break here, loudly, rather than
// quietly making a fault test assert nothing.
var _ = []error{llm.ErrContextWindow, llm.ErrRateLimit, llm.ErrServer, llm.ErrTransport}
