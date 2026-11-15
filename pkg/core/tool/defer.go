package tool

import (
	"context"
	"errors"

	"wombat/pkg/core/llm"
)

// ErrDeferred marks a call nobody could decide during the run — a person has
// to approve it, and nobody is waiting to be asked — so the run ends instead
// of waiting, and resumes once the decision is in.
//
// A middleware defers a call by returning [Defer]. The agent loop turns a
// batch that holds one into a core.Deferred outcome; core.Decide resumes it.
var ErrDeferred = errors.New("tool: deferred")

// Defer returns the error a middleware defers a call with. reason is what the
// person deciding it will read.
func Defer(reason string) error { return &deferredError{reason: reason} }

type deferredError struct{ reason string }

func (e *deferredError) Error() string { return "deferred for a decision: " + e.reason }
func (e *deferredError) Unwrap() error { return ErrDeferred }

// DeferralReason reports why err defers its call, if it does.
func DeferralReason(err error) (string, bool) {
	var d *deferredError
	if errors.As(err, &d) {
		return d.reason, true
	}
	if errors.Is(err, ErrDeferred) {
		return "", true
	}
	return "", false
}

type decisionsKey struct{}

// WithDecisions carries what people decided about deferred calls, by
// tool_use id: true to run the call, false to refuse it. A run resumed with
// core.Decide installs them, and a gate that would have asked consults them
// first.
func WithDecisions(ctx context.Context, verdicts map[llm.ToolUseID]bool) context.Context {
	return context.WithValue(ctx, decisionsKey{}, verdicts)
}

// Decision reports what a person decided about the call with this id, if
// anyone has.
func Decision(ctx context.Context, id llm.ToolUseID) (allow, decided bool) {
	verdicts, _ := ctx.Value(decisionsKey{}).(map[llm.ToolUseID]bool)
	allow, decided = verdicts[id]
	return allow, decided
}
