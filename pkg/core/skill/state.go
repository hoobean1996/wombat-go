package skill

import (
	"context"
	"slices"
	"sync"

	"wombat/pkg/core/llm"
)

// State is one run's activation set: which skills are loaded, and which
// tool_result carried each one's body.
//
// Per-run, not per-agent. A core.Agent is immutable and shared across
// goroutines, so activations stored on it would leak between concurrent runs —
// one run loading pdf-forms would reveal fill_pdf to every other. The state
// travels on the context instead, which also means a sub-agent that inherits
// the context inherits the parent's loaded skills for free.
//
// The OCaml original held this in a module-global behind an algebraic effect,
// with a shared default for callers outside a handler. A context value is the
// same idea with less machinery and no fallback that silently couples tests.
type State struct {
	// Guarded because the read side runs on the agent loop (Visible and
	// Reconcile, once per turn) while the write side runs inside tool
	// handlers, which the dispatcher may execute concurrently.
	mu     sync.RWMutex
	active map[string]llm.ToolUseID
}

// NewState builds an empty activation set. One per run; see
// core.WithRunContext.
func NewState() *State {
	return &State{active: make(map[string]llm.ToolUseID, 4)}
}

type stateKey struct{}

// WithState attaches s to ctx.
func WithState(ctx context.Context, s *State) context.Context {
	return context.WithValue(ctx, stateKey{}, s)
}

// StateFrom retrieves the activation set. It NEVER returns nil.
//
// Outside a run — a unit test calling the tool directly, a REPL — there is no
// state on the context, and a caller would otherwise have to nil-check at
// every use. A throwaway state is returned instead, so the load_skill handler
// is an ordinary function that runs anywhere. The consequence is deliberate:
// activations made against a throwaway are not shared with anything, so a
// gated tool stays hidden. Nothing is silently global.
//
// The one place that refuses the throwaway is load_skill inside a dispatch —
// see [ErrNoState] — because there a missing State is a wiring mistake that
// would otherwise only show up as gated tools that never appear.
func StateFrom(ctx context.Context) *State {
	if s, ok := stateOn(ctx); ok {
		return s
	}
	return NewState()
}

// stateOn returns the State attached to ctx, reporting false when there is none.
func stateOn(ctx context.Context) (*State, bool) {
	s, ok := ctx.Value(stateKey{}).(*State)
	return s, ok && s != nil
}

// Activate records name as loaded, with body naming the tool_use whose result
// carries its text.
//
// The id is not bookkeeping: [Gated.Set]'s Reconcile matches on it to notice
// when the body has been evicted from the transcript. Re-activating an already
// active skill overwrites the id, which is what a genuine second load should
// do — but the load_skill handler stubs that case instead, so in practice the
// first body's id is the one that sticks.
func (s *State) Activate(name string, body llm.ToolUseID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active == nil {
		s.active = make(map[string]llm.ToolUseID, 4)
	}
	s.active[name] = body
}

// Deactivate retires name. Idempotent.
func (s *State) Deactivate(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.active, name)
}

// IsActive reports whether name is loaded. This is the gate predicate, read
// once per tool per turn.
func (s *State) IsActive(name string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.active[name]
	return ok
}

// Active lists the loaded skills, sorted.
//
// Sorted because this string is shown to the model in load/unload
// observations, and an observation that reorders itself between otherwise
// identical turns is noise the model has to reconcile.
func (s *State) Active() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.active))
	for n := range s.active {
		out = append(out, n)
	}
	slices.Sort(out)
	return out
}

// BodyOf returns the tool_use id whose result carried name's body.
func (s *State) BodyOf(name string) (llm.ToolUseID, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	id, ok := s.active[name]
	return id, ok
}
