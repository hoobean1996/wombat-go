package tool

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"

	"github.com/hoobean1996/wombat-go/llm"
)

// ErrPanic is a tool call that panicked instead of returning. Match with
// errors.Is; the concrete error is a [*PanicError], which also carries the
// recovered value and the stack.
//
// A panicking tool is a broken tool, not a broken agent: the model can read
// the failure as an is_error tool_result and route around it, exactly as it
// would for "no such file".
var ErrPanic = errors.New("tool: panicked")

// Stack and value budgets for a recovered panic.
//
// The stack is for a human reading a log, and the frames that matter are the
// innermost ones, which come first — so a prefix is the right cut. The value is
// clipped harder because it is the part the MODEL sees, and a tool that panics
// with a megabyte of state should not push the conversation out of its window.
const (
	maxPanicStack = 4096
	maxPanicValue = 200
)

// PanicError reports a tool call that panicked.
//
// It serves two audiences with one value, which is why it is a struct and not
// a formatted string:
//
//   - [PanicError.Error] is what the MODEL reads, via Result.Block. It says
//     which tool crashed and with what, in one line. The model needs to decide
//     whether to try different arguments, a different tool, or give up; a Go
//     stack helps it with none of that and costs real tokens.
//   - Value and Stack are what an OPERATOR reads. A panic with no stack is
//     nearly unactionable — "index out of range" without the frame that indexed
//     is a needle in the whole tool set — so the stack is captured at recover
//     time, while it still exists, and kept on the error. [PanicError.Details]
//     renders both for a log or a bug report.
//
// The recovery layers also emit the stack once to [slog.Default] at Error
// level, because nothing else in the pipeline would: WithLogging previews
// err.Error(), which is deliberately the short form.
type PanicError struct {
	// Tool is the tool that was in flight. Set even when the panic came from a
	// middleware rather than from the tool body.
	Tool string

	// Value is the argument to panic, rendered with %v and clipped.
	Value string

	// Stack is the goroutine stack at the moment of recovery, clipped to
	// [maxPanicStack] bytes. Empty is possible only if debug.Stack failed.
	Stack string
}

// Error renders the short, model-facing form.
func (e *PanicError) Error() string {
	return fmt.Sprintf("%s: %s crashed: %s", ErrPanic.Error(), e.Tool, e.Value)
}

// Unwrap reports [ErrPanic], so errors.Is matches regardless of which layer
// recovered and of any %w wrapping applied above it.
func (e *PanicError) Unwrap() error { return ErrPanic }

// Details renders the operator-facing form: the short message plus the stack.
func (e *PanicError) Details() string {
	if e.Stack == "" {
		return e.Error()
	}
	return e.Error() + "\n" + e.Stack
}

// recovered converts a value from recover into a [*PanicError], capturing the
// stack of the goroutine that is unwinding and logging it once.
//
// Call it directly from the deferred function, not from a helper the deferred
// function calls with the already-recovered value: debug.Stack must run before
// the frames of interest are gone.
func recovered(ctx context.Context, name string, v any) error {
	e := &PanicError{
		Tool:  name,
		Value: preview(fmt.Sprintf("%v", v), maxPanicValue),
		Stack: preview(string(debug.Stack()), maxPanicStack),
	}
	slog.Default().ErrorContext(ctx, "tool panicked",
		slog.String("tool", e.Tool),
		slog.String("panic", e.Value),
		slog.String("stack", e.Stack),
	)
	return e
}

// WithRecovery converts a panicking tool into an ordinary error.
//
// It goes INNERMOST in the chain, immediately around [Direct], and the position
// is the whole point: every layer above it — retry, the circuit breaker, dedup,
// logging, the observer — then sees a panic as the failure it is. A tool that
// panics every time trips the breaker exactly like a tool that returns an error
// every time, and dedup tells the model it has hit the same wall three times
// instead of the process dying on the first.
//
// It recovers from everything, and re-panics on nothing. A runtime.Error such
// as an out-of-range index or a nil map write is still just a broken tool, and
// there is no class of panic for which killing the whole agent — and every
// other run sharing the process — is the better answer. Two things are
// genuinely outside its reach and neither is a panic: runtime.Goexit, and a
// panic on a goroutine the tool spawned itself, which no caller can catch.
//
// It is a Handler decorator rather than a Middleware factory because it takes
// no configuration; it is still usable anywhere a [Middleware] is, since the
// signatures are identical.
//
// This layer alone is NOT enough — see the recover in (*dispatcher).Dispatch,
// which backstops everything outside it.
func WithRecovery(next Handler) Handler {
	return func(ctx context.Context, d Def, use llm.ToolUse) (out string, err error) {
		defer func() {
			if v := recover(); v != nil {
				// Drop any partial output: a handler that panicked mid-write
				// has no claim to have observed anything.
				out, err = "", recovered(ctx, d.Name, v)
			}
		}()
		return next(ctx, d, use)
	}
}
