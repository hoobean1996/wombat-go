package permission

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"sync"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// Grants remembers per-run approvals so the same question is asked once.
//
// Per-run, not per-agent, for the reason skill.State is: a core.Agent is
// immutable and shared across goroutines, so approvals stored on it would leak
// between concurrent runs — one user saying yes would answer for everybody.
// Grants travel on the context instead, which also means a sub-agent that
// inherits the context inherits its parent's answers, which is what a person
// who approved the parent's task expects.
//
// # What counts as "the same question"
//
// A grant is keyed on the tool name AND the exact arguments (see [Gate]).
// Approving `rm -rf /tmp/build` must not approve `rm -rf /`, and no textual
// similarity measure is trustworthy enough to decide the two are the same
// question. So the memory is narrow on purpose: it stops the second identical
// call — a retry, a loop the model has fallen into, the same file written
// twice — from interrupting a person again, and nothing more.
//
// A broader standing permission is a policy decision, not an approval one:
// express "bash is fine today" as [AllowTools] in the [Policy], where it is
// written down and reviewable, rather than as a grant that widens invisibly.
type Grants struct {
	// A plain Mutex: the write side runs inside tool handlers, which the
	// dispatcher may execute concurrently, and the read side runs in the same
	// place. There is no read-mostly phase to optimise for.
	mu   sync.Mutex
	seen map[string]struct{}
}

// NewGrants builds an empty approval memory. One per run; install it with
// [WithGrants], typically from core.WithRunContext.
func NewGrants() *Grants {
	return &Grants{seen: make(map[string]struct{}, 4)}
}

type grantsKey struct{}

// WithGrants attaches g to ctx.
func WithGrants(ctx context.Context, g *Grants) context.Context {
	return context.WithValue(ctx, grantsKey{}, g)
}

// GrantsFrom retrieves the approval memory. It NEVER returns nil.
//
// Outside a run — a unit test calling a gated handler directly, a REPL — there
// is nothing on the context, and every caller would otherwise nil-check. A
// throwaway is returned instead. The consequence is deliberate and is the safe
// one: a throwaway remembers nothing, so a call outside a run asks every time
// rather than silently inheriting an approval from somewhere global.
func GrantsFrom(ctx context.Context) *Grants {
	if g, ok := ctx.Value(grantsKey{}).(*Grants); ok && g != nil {
		return g
	}
	return NewGrants()
}

// Remember records that key has been approved for the rest of the run.
func (g *Grants) Remember(key string) {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.seen == nil {
		g.seen = make(map[string]struct{}, 4)
	}
	g.seen[key] = struct{}{}
}

// Has reports whether key has already been approved.
func (g *Grants) Has(key string) bool {
	if g == nil {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	_, ok := g.seen[key]
	return ok
}

// grantKey identifies one question: the tool name, a NUL, then the call's
// arguments with insignificant whitespace removed.
//
// Compacting matters because the bytes come off the wire from a model and the
// same call can arrive formatted two ways; without it the second copy of an
// identical question would be asked again. Unexported, but the format is
// documented so a host with a reason to pre-approve something can build the
// key itself and hand it to [Grants.Remember].
func grantKey(d tool.Def, use llm.ToolUse) string {
	var buf bytes.Buffer
	buf.WriteString(d.Name)
	buf.WriteByte(0)

	var compact bytes.Buffer
	if err := json.Compact(&compact, use.Input); err != nil {
		// Not JSON (or empty). Key on the raw bytes: an unreadable argument
		// still identifies itself, and being wrong here means asking twice,
		// which is the harmless direction.
		buf.Write(use.Input)
		return buf.String()
	}
	buf.Write(compact.Bytes())
	return buf.String()
}

// MarshalState is what has been granted, as a run suspended at the end of its
// slice carries it to the next (see core.Carried): what a person allowed for
// the turn stays allowed when the turn carries on.
func (g *Grants) MarshalState() (json.RawMessage, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	keys := make([]string, 0, len(g.seen))
	for k := range g.seen {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return json.Marshal(keys)
}

// UnmarshalState puts back what MarshalState wrote.
func (g *Grants) UnmarshalState(b json.RawMessage) error {
	var keys []string
	if err := json.Unmarshal(b, &keys); err != nil {
		return err
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.seen = make(map[string]struct{}, len(keys))
	for _, k := range keys {
		g.seen[k] = struct{}{}
	}
	return nil
}
