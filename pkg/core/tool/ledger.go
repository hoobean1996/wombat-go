package tool

import (
	"context"
	"encoding/json"
	"errors"

	"wombat/pkg/core/llm"
)

// ===== Durable execution =====
//
// A run that writes itself down as it goes (core.Journal) records where each
// of its tool calls is, so that a run resumed after its process died knows
// which calls ran: one that finished is answered from the record rather than
// run again, and one that was running when the process died — and so may or
// may not have taken effect — is handled as its tool says. What this package
// contributes is the vocabulary: [State], [Recovery], [CallKey], [Await] and
// [Def.Reconcile].

// State is where one tool call is, as a journal records it.
//
//	pending ──▶ running ──▶ succeeded | failed
//	   │           ├──▶ waiting ──(its result is posted)──▶ succeeded | failed
//	   │           └──(the run stopped)──▶ interrupted ──(Recovery)──▶ running | failed | deferred
//	   ├──▶ denied
//	   ├──▶ deferred ──(allowed)──▶ running
//	   │       └──(refused)──▶ denied
//	   └──▶ cancelled
type State string

// The states of a call.
const (
	// StatePending is a call written down with the model's reply, before it
	// was dispatched.
	StatePending State = "pending"

	// StateRunning is a call about to take effect, or taking it. A run
	// resumed after its process died that finds one running cannot tell
	// whether it took effect; see [Recovery].
	StateRunning State = "running"

	// StateSucceeded is a call that returned its output.
	StateSucceeded State = "succeeded"

	// StateFailed is a call that returned an error, which the model reads —
	// including one that was interrupted and reported to the model as such.
	StateFailed State = "failed"

	// StateDenied is a call refused before it ran: by a permission gate, a
	// skill gate, a person, or for naming no tool at all.
	StateDenied State = "denied"

	// StateDeferred is a call waiting for a person's decision.
	StateDeferred State = "deferred"

	// StateWaiting is a call that started something which finishes outside
	// the run, waiting for its result to be posted; see [Await].
	StateWaiting State = "waiting"

	// StateInterrupted is a call whose run stopped while it ran — the process
	// shutting down, the turn cancelled. It may or may not have taken effect.
	StateInterrupted State = "interrupted"

	// StateCancelled is a call that never ran and never will: its run ended
	// before dispatching it, to ask the user something or to submit.
	StateCancelled State = "cancelled"
)

// States lists every state, in the order a call goes through them.
func States() []State {
	return []State{StatePending, StateRunning, StateSucceeded, StateFailed, StateDenied,
		StateDeferred, StateWaiting, StateInterrupted, StateCancelled}
}

// Final reports whether a call in this state is over: it has its result, and
// nothing more will happen to it.
func (s State) Final() bool {
	switch s {
	case StateSucceeded, StateFailed, StateDenied, StateCancelled:
		return true
	}
	return false
}

// Valid reports whether s is one of the states.
func (s State) Valid() bool {
	for _, v := range States() {
		if s == v {
			return true
		}
	}
	return false
}

// Recovery says what a resumed run does with a call it finds running or
// interrupted: one that may or may not have taken effect before the run
// stopped. A tool that can find out says so first, with [Def.Reconcile].
type Recovery int

// The ways to recover a call.
const (
	// RecoverAuto derives it from the tool: a call that cannot have had an
	// effect, or is safe to repeat — Idempotent, or read-only — is run again,
	// and any other is reported.
	RecoverAuto Recovery = iota

	// RecoverRerun runs the call again. Right for a tool that is safe to
	// repeat: a read, a write of a whole file, anything made idempotent by
	// its [CallKey].
	RecoverRerun

	// RecoverReport runs nothing, and answers the call with an error saying
	// it was interrupted and may have taken effect, so the model checks
	// before it tries again.
	RecoverReport

	// RecoverAsk defers the call, for a person to decide whether it runs
	// again.
	RecoverAsk
)

// String names the recovery, for logs and events.
func (r Recovery) String() string {
	switch r {
	case RecoverRerun:
		return "rerun"
	case RecoverReport:
		return "report"
	case RecoverAsk:
		return "ask"
	default:
		return "auto"
	}
}

// RecoveryOf is what a resumed run does with an interrupted call of d:
// d.Recovery, with [RecoverAuto] resolved.
func RecoveryOf(d Def) Recovery {
	switch {
	case d.Recovery != RecoverAuto:
		return d.Recovery
	case d.Idempotent, d.Has(CapReadOnly) && d.Caps&(CapMutating|CapExec) == 0:
		return RecoverRerun
	default:
		return RecoverReport
	}
}

// ReconcileFunc is the type of [Def.Reconcile]: given an interrupted call's
// arguments, it reports whether the call's effect landed — done, with the
// output the call would have returned — or did not, so the call can simply
// run again. An error means it cannot tell, and leaves the call to its
// [Recovery].
type ReconcileFunc func(ctx context.Context, in json.RawMessage) (out string, done bool, err error)

// ===== Call keys =====

type callKeysKey struct{}

// WithCallKeys makes keys name the calls dispatched on ctx: for each, by its
// tool_use id, the [CallKey] its tool sees. An empty name falls back to the
// id. A host that journals its runs installs it, with the names its journal
// gave the calls.
func WithCallKeys(ctx context.Context, keys func(llm.ToolUseID) string) context.Context {
	return context.WithValue(ctx, callKeysKey{}, keys)
}

// CallKey names the call ctx belongs to, for a tool that makes its effect
// idempotent: an outbox message posted once, a timer set once, a request sent
// with an idempotency key. The same call has the same key whenever it runs —
// retried, or run again by a run resumed after its process died — and no
// other call of the session has it.
//
// A host that journals its runs supplies the names; see [WithCallKeys].
// Without one it is the call's tool_use id, which is only as unique as the
// provider makes it, and some gateways reuse ids from one reply to the next.
// Empty outside a dispatched call.
func CallKey(ctx context.Context) string {
	id := InfoFrom(ctx).UseID
	if id == "" {
		return ""
	}
	if keys, ok := ctx.Value(callKeysKey{}).(func(llm.ToolUseID) string); ok && keys != nil {
		if k := keys(id); k != "" {
			return k
		}
	}
	return string(id)
}

// ===== Waiting for a result =====

// ErrAwait marks a call that has started something which finishes outside the
// run — a job on another machine, a CI run, a person doing a step by hand — so
// the run ends instead of holding the call open, and resumes when the call's
// result is posted. A tool returns it with [Await].
var ErrAwait = errors.New("tool: awaiting a result")

// Await returns the error a tool ends a call with when its result will come
// later, from outside the run. note says what the call is waiting for, for
// whoever will post the result; handle names it where they are — a job id, a
// URL — and may be empty.
//
// The run ends at the call, in core.Deferred, the call marked as awaiting a
// result. It resumes when the result is in, as if the tool had returned it;
// see core.Input.Completed.
func Await(handle, note string) error { return &awaitError{handle: handle, note: note} }

type awaitError struct{ handle, note string }

func (e *awaitError) Error() string {
	s := "awaiting a result"
	if e.note != "" {
		s += ": " + e.note
	}
	if e.handle != "" {
		s += " (" + e.handle + ")"
	}
	return s
}

func (e *awaitError) Unwrap() error { return ErrAwait }

// Awaiting reports what err says its call is waiting for, if it does.
func Awaiting(err error) (handle, note string, ok bool) {
	var a *awaitError
	if errors.As(err, &a) {
		return a.handle, a.note, true
	}
	if errors.Is(err, ErrAwait) {
		return "", "", true
	}
	return "", "", false
}

// ErrSuspended marks a call that did not finish in its run's slice and
// carries on in the next: a delegate call whose sub-agent stopped, at the end
// of the slice, where its journal wrote it down. The call is not over, and has
// not failed; the run it belongs to stops too, and runs it again — which
// carries the sub-agent on — when it is resumed. See core.Input.Until.
var ErrSuspended = errors.New("tool: carries on in the next slice")

// pending reports an error that is not a failure: the call is waiting — for a
// decision, or for a result — or carries on in the next slice, and no
// middleware should count it against the tool, nor try it again.
func pending(err error) bool {
	return errors.Is(err, ErrDeferred) || errors.Is(err, ErrAwait) || errors.Is(err, ErrSuspended)
}

// Tagged returns the labels the call ctx belongs to has put on its result
// with [Annotate] so far. Like [Attached], it is for middleware that records
// a call and has to give back what it recorded.
func Tagged(ctx context.Context) []string {
	if a, ok := ctx.Value(annotateKey{}).(*annotations); ok && a != nil {
		a.mu.Lock()
		defer a.mu.Unlock()
		return append([]string(nil), a.tags...)
	}
	return nil
}
