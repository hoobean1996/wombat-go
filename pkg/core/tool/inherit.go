package tool

import (
	"context"

	"wombat/pkg/core/llm"
)

// ===== Inherited middleware =====

// Inherit attaches mw to ctx so that every dispatch chain honouring inheritance
// applies it to the calls made under ctx — including the calls of any agent
// started on ctx, which is the point.
//
// It is how a decision follows the work it allowed. A permission gate that lets
// a delegate call through has allowed a whole sub-agent, and that sub-agent's
// tool calls must not escape the policy merely because the child was built
// without the gate. So the gate inherits itself into the context of the call it
// allowed, and the child's chain applies it.
//
// key identifies mw and must be comparable. Attaching a key that is already
// attached is a no-op, so middleware inherited down a chain of sub-agents is
// applied once per call rather than once per generation.
//
// The agent's default dispatch chain applies inherited middleware with
// [WithInherited]; a chain assembled by hand must place that layer itself.
func Inherit(ctx context.Context, key any, mw Middleware) context.Context {
	cur, _ := ctx.Value(inheritKey{}).([]inherited)
	for _, in := range cur {
		if in.key == key {
			return ctx
		}
	}
	next := make([]inherited, len(cur), len(cur)+1)
	copy(next, cur)
	return context.WithValue(ctx, inheritKey{}, append(next, inherited{key: key, mw: mw}))
}

type inheritKey struct{}

type inherited struct {
	key any
	mw  Middleware
}

// WithInherited applies the middleware that [Inherit] attached to the context
// of the call in flight, the earliest-attached outermost.
//
// Where it goes in a chain is a decision. Inside the caller's own middleware,
// so that a sub-agent's own gate, usually the narrower one, rules first: a call
// it refuses never reaches the parent's policy, and never puts a question to a
// person that the child would have refused anyway. Outside retry, the per-call
// timeout and the circuit breaker, for the reasons a permission gate belongs
// there: a verdict is not a transient failure, and a person's thinking time is
// not the tool's running time.
func WithInherited(next Handler) Handler {
	return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
		list, _ := ctx.Value(inheritKey{}).([]inherited)
		h := next
		for i := len(list) - 1; i >= 0; i-- {
			h = list[i].mw(h)
		}
		return h(ctx, d, use)
	}
}

// Claim reports whether key is being claimed for the call in flight for the
// first time, and claims it.
//
// It lets a decision that can meet the same call more than once happen once: a
// gate an agent was built with and the same gate inherited from its parent sit
// in one chain, and the first copy to see a call decides while the other
// passes it through. The claim belongs to one dispatched call — a call made
// beneath it, by a sub-agent, is a new call with claims of its own — so a
// verdict on a delegate call is never mistaken for a verdict on the calls the
// delegated agent goes on to make. Outside a dispatcher there is no call to
// scope a claim to, and Claim always reports true.
func Claim(ctx context.Context, key any) bool {
	a, ok := ctx.Value(annotateKey{}).(*annotations)
	if !ok || a == nil {
		return true
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, dup := a.claimed[key]; dup {
		return false
	}
	if a.claimed == nil {
		a.claimed = make(map[any]struct{}, 1)
	}
	a.claimed[key] = struct{}{}
	return true
}
