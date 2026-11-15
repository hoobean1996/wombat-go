package wombat

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/hoobean1996/wombat-go/llm"
)

// Outcome is how a run ended successfully. Failure is not an Outcome — it is
// an error, reachable through [Run.Err], because Go has one failure channel
// and splitting semantic failure from infrastructure failure into two return
// values only forces every caller to check both.
//
//	Answer | Paused | Submitted
type Outcome interface {
	isOutcome()
}

// Answer is a free-text reply: the model ended its turn.
type Answer struct {
	Text string
}

// Paused means the model called a tool with [tool.CapPause] — it wants input
// from the user — and the tool middleware let the call through. The tool's
// handler was never invoked.
//
// Resume by answering the tool_use and calling the agent again:
//
//	in := wombat.AnswerPause(run.Messages(), p.ToolUseID, reply)
//	out, err := a.Run(ctx, in)
type Paused struct {
	ToolUseID llm.ToolUseID
	Schema    PauseSchema
}

// Submitted means the model called a tool with [tool.CapTerminal] and the tool
// middleware let the call through. The tool's handler was never invoked; its
// arguments are the structured return value.
//
// The transcript ends with that call unanswered, together with any other call
// the model made in the same turn, none of which was dispatched. That is the
// right shape for a finished task, and the wrong one to send again: to carry on
// from a Submitted run anyway, use [Then] or [Resume], which close every
// dangling call before the new instruction. [Continue] would be rejected by
// [Convo.Validate].
type Submitted struct {
	Tool    string
	Payload json.RawMessage
}

func (Answer) isOutcome()    {}
func (Paused) isOutcome()    {}
func (Submitted) isOutcome() {}

// PauseSchema describes the form a front end should render to collect the
// user's answer.
type PauseSchema struct {
	// Question is the plain-text prompt, present in every shape.
	Question string `json:"question,omitempty"`

	// Schema is the JSON Schema for a structured answer, passed through
	// verbatim. Empty means a free-text reply.
	Schema json.RawMessage `json:"schema,omitempty"`
}

// Summary renders a one-line description for logs.
func (p PauseSchema) Summary() string {
	switch {
	case p.Question != "" && len(p.Schema) > 0:
		return p.Question + " (structured)"
	case p.Question != "":
		return p.Question
	case len(p.Schema) > 0:
		return "structured input"
	default:
		return "input"
	}
}

// ParsePauseSchema reads a pause tool's arguments.
//
// Tolerant by design: a model that emits a bare {"question": "..."} and a
// model that emits a full schema both have to work, and a malformed payload
// should still surface a usable prompt rather than failing the run.
func ParsePauseSchema(in json.RawMessage) PauseSchema {
	var p PauseSchema
	if len(in) == 0 {
		return p
	}
	if err := json.Unmarshal(in, &p); err == nil && (p.Question != "" || len(p.Schema) > 0) {
		return p
	}
	var loose map[string]json.RawMessage
	if err := json.Unmarshal(in, &loose); err != nil {
		return PauseSchema{Question: string(in)}
	}
	for _, key := range []string{"question", "prompt", "message"} {
		if raw, ok := loose[key]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil {
				p.Question = s
				break
			}
		}
	}
	if p.Question == "" {
		p.Question = string(in)
	}
	return p
}

// Reasons a run failed. Match with errors.Is.
var (
	// ErrMaxIterations means the loop hit its iteration cap without the model
	// producing a final answer.
	ErrMaxIterations = errors.New("wombat: max iterations reached")

	// ErrMaxTokens means the reply was truncated. Retrying identically will
	// truncate identically; the caller has to shorten the request or raise the
	// cap.
	ErrMaxTokens = errors.New("wombat: reply truncated at max_tokens")

	// ErrRefused means the model declined.
	ErrRefused = errors.New("wombat: model refused")

	// ErrUnexpectedStop means the provider returned a stop_reason this harness
	// does not know how to continue from.
	ErrUnexpectedStop = errors.New("wombat: unexpected stop_reason")

	// ErrPanic means something in the run panicked and was contained. It is
	// always a bug in the panicking code, never a runtime condition, and the
	// error carries a truncated stack to find it with.
	ErrPanic = errors.New("wombat: agent loop panicked")

	// ErrEmptyTurn means the model ended its turn having produced no text and
	// called no tool, repeatedly enough that the loop stopped waiting.
	//
	// The loop retries an empty turn before giving up, because one observed
	// cause is transient: a reasoning model streams its scratchpad, the stream
	// ends with finish_reason "stop" and no content and no usage at all, and
	// forty-five seconds have gone by. Retrying gets a real answer.
	//
	// The other observed cause is not transient, and is not retried. A hard task
	// made a reasoning model want 17,577 output tokens against a cap of 8,192;
	// the gateway ran out mid-thought and reported finish_reason "stop" with no
	// content and no usage — the identical wire shape. Three attempts, two
	// minutes each, all exhausting the same budget in the same place.
	// [EmptyTurnError] carries how much reasoning streamed first, which is the
	// only evidence available for telling the two apart, and says which one it
	// thinks happened.
	//
	// It is an error and not an empty [Answer] because the alternative was
	// tried and it lies. Answer{Text: ""} is indistinguishable from a model
	// that finished and had nothing to add, so a run that produced nothing
	// reported success — and downstream, a benchmark blamed its verifiers for
	// four episodes in which the agent had simply never spoken.
	ErrEmptyTurn = errors.New("wombat: model ended its turn with no text and no tool call")
)

// RefusalError carries the model's stated reason, when it gave one.
type RefusalError struct{ Reason string }

// Error implements error.
func (e *RefusalError) Error() string {
	if e.Reason == "" {
		return ErrRefused.Error()
	}
	return ErrRefused.Error() + ": " + e.Reason
}

// Unwrap implements errors.Is against ErrRefused.
func (e *RefusalError) Unwrap() error { return ErrRefused }

// UnexpectedStopError carries the stop_reason that could not be handled.
type UnexpectedStopError struct{ StopReason llm.StopReason }

// Error implements error.
func (e *UnexpectedStopError) Error() string {
	return fmt.Sprintf("%s: %q", ErrUnexpectedStop.Error(), e.StopReason)
}

// Unwrap implements errors.Is against ErrUnexpectedStop.
func (e *UnexpectedStopError) Unwrap() error { return ErrUnexpectedStop }

// EmptyTurnError says what was seen before the silence, because "the model had
// nothing to say" and "the model was cut off mid-thought" are the same event on
// the wire and completely different bugs to chase.
type EmptyTurnError struct {
	// Attempts is how many consecutive empty turns had been seen when the loop
	// gave up.
	Attempts int

	// StopReason is what the provider called it. Observed value on a truncated
	// reasoning block: "end_turn".
	StopReason llm.StopReason

	// ReasoningBytes is how much scratchpad streamed before the response ended.
	// The only evidence there is when the response carries no usage — and a
	// truncated turn carries none, because the usage chunk comes last.
	ReasoningBytes int

	// MaxTokens is the reply cap the request was made with, quoted so the
	// message can name the number to change.
	MaxTokens int
}

// budgetExhausted reports whether the evidence points at the reply cap rather
// than at a mute model.
//
// The ratio is a heuristic and deliberately a conservative one. English prose
// runs about four characters per token, so a turn that streamed two characters
// of reasoning per token of budget has, on any plausible tokenizer, spent most
// of that budget — while a provider that hiccups and returns nothing streams
// close to zero. There is no exact test available: the response that would have
// carried the token count is the one that did not arrive.
func (e *EmptyTurnError) budgetExhausted() bool {
	return e.MaxTokens > 0 && e.ReasoningBytes >= 2*e.MaxTokens
}

// Error implements error.
func (e *EmptyTurnError) Error() string {
	var b strings.Builder
	b.WriteString(ErrEmptyTurn.Error())
	fmt.Fprintf(&b, " (%d attempt(s), stop_reason %q", e.Attempts, e.StopReason)
	if e.ReasoningBytes > 0 {
		fmt.Fprintf(&b, ", %d chars of reasoning streamed first", e.ReasoningBytes)
	}
	b.WriteString(")")

	if e.budgetExhausted() {
		fmt.Fprintf(&b, ": the reply cap of %d tokens was almost certainly consumed by reasoning. "+
			"A gateway that runs out mid-thought may report finish_reason \"stop\" with no content "+
			"and no usage rather than \"length\", which is indistinguishable from a mute model except "+
			"by how much came out first. Raise MaxTokens (wombat.WithMaxTokens, or -max-tokens). "+
			"Not retried: the request is unchanged, so it would spend the same budget the same way",
			e.MaxTokens)
	}
	return b.String()
}

// Unwrap implements errors.Is against ErrEmptyTurn.
func (e *EmptyTurnError) Unwrap() error { return ErrEmptyTurn }

// ===== Outcome projections =====
//
// Most orchestration call sites want one shape and treat anything else as a
// structural error. These consolidate the branching that would otherwise be
// copy-pasted at each of them.

// ExpectAnswer requires a free-text answer. who names the caller's role so the
// diagnostic is readable from a log line alone.
func ExpectAnswer(out Outcome, who string) (string, error) {
	switch o := out.(type) {
	case Answer:
		return o.Text, nil
	case Submitted:
		return "", fmt.Errorf("%s: expected a text answer but the model called terminal tool %q", who, o.Tool)
	case Paused:
		return "", fmt.Errorf("%s: expected a text answer but the model asked the user (%s)", who, o.Schema.Summary())
	default:
		return "", fmt.Errorf("%s: expected a text answer, got %T", who, out)
	}
}

// ExpectSubmitted requires the named terminal tool and returns its arguments.
func ExpectSubmitted(out Outcome, toolName string) (json.RawMessage, error) {
	switch o := out.(type) {
	case Submitted:
		if o.Tool != toolName {
			return nil, fmt.Errorf("expected terminal tool %q, got %q", toolName, o.Tool)
		}
		return o.Payload, nil
	case Answer:
		return nil, fmt.Errorf("expected terminal tool %q but the model answered in text", toolName)
	case Paused:
		return nil, fmt.Errorf("expected terminal tool %q but the model asked the user (%s)", toolName, o.Schema.Summary())
	default:
		return nil, fmt.Errorf("expected terminal tool %q, got %T", toolName, out)
	}
}
