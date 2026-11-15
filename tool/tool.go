// Package tool defines what a tool is, how one is dispatched, and the
// middleware that hardens dispatch.
//
// The central decision is the handler signature:
//
//	type Fn func(ctx context.Context, in json.RawMessage) (string, error)
//
// A tool receives a context and its arguments. Nothing else. Whatever it
// needs — a filesystem, a clock, an HTTP client, a shell — is captured by the
// closure when the tool is constructed:
//
//	func ViewFile(fsys FS) tool.Def { ... }
//
// That is why there is no ambient runtime threaded through dispatch. It also
// means a tool is an ordinary function: testing one requires no harness.
package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

// Cap is what a tool DOES. Used to build a tool set for a restricted agent:
// a planner is handed tool.Filter(all, tool.OnlyCaps(tool.CapReadOnly|tool.CapNetwork)).
//
// Filtering happens once, when the set is constructed, rather than being
// re-derived per dispatch from a mode tag. The agent's tool list stays the
// single authority for both what the model sees and what can execute.
type Cap uint32

const (
	CapReadOnly Cap = 1 << iota // no filesystem, network or subprocess effect
	CapMutating                 // writes to the filesystem or external state
	CapExec                     // spawns a subprocess
	CapNetwork                  // performs network I/O
	CapMeta                     // orchestration: spawns sub-agents, loads skills
	CapPause                    // interrupts the agent loop to ask the user
	CapTerminal                 // ends the run; Fn is never invoked
)

// Need is what a tool requires FROM THE HOST. Distinct from Cap: view_file is
// CapReadOnly but NeedFSRead, and a sandbox with no filesystem must hide it.
type Need uint32

const (
	NeedFSRead Need = 1 << iota
	NeedFSWrite
	NeedExec
	NeedNetwork
)

// Fn executes a tool. The string return is the observation handed back to the
// model; a non-nil error becomes an is_error tool_result, which the model sees
// and can react to. Returning an error is normal, not exceptional.
type Fn func(ctx context.Context, in json.RawMessage) (string, error)

// Def is an executable tool: its wire description plus the policy metadata
// that dispatch middleware reads.
type Def struct {
	Name        string
	Description string

	// InputSchema is handed to the model byte for byte. See llm.ToolSpec for
	// why this is json.RawMessage and never map[string]any.
	InputSchema json.RawMessage

	Fn Fn

	Caps  Cap
	Needs Need

	// Idempotent gates retry. Default false: a write or exec tool must not be
	// replayed just because the first attempt looked transient.
	Idempotent bool

	// Timeout is the per-call wall clock. Zero means no cap. Enforced by
	// WithTimeout, which cancels the context — so a tool that honours ctx is
	// actually stopped, not merely abandoned.
	Timeout time.Duration

	// Category is a coarse label for logs and traces. Gating is done by Caps
	// and Needs; this is only a human-readable hint.
	Category string

	// Retryable classifies an error as worth retrying. Nil means never, which
	// is the conservative default: most tool errors are deterministic.
	Retryable func(error) bool
}

// Spec projects the wire subset handed to the model.
func (d Def) Spec() llm.ToolSpec {
	return llm.ToolSpec{Name: d.Name, Description: d.Description, InputSchema: d.InputSchema}
}

// Specs projects a slice of definitions.
func Specs(defs []Def) []llm.ToolSpec {
	out := make([]llm.ToolSpec, len(defs))
	for i, d := range defs {
		out[i] = d.Spec()
	}
	return out
}

// Has reports whether every capability in c is present.
func (d Def) Has(c Cap) bool { return d.Caps&c == c }

// Typed wraps a decoder around a strongly typed handler, so the body works on
// a Go value instead of raw JSON.
//
//	tool.Typed(tool.Def{Name: "view_file", ...},
//	    func(ctx context.Context, in viewFileIn) (string, error) { ... })
//
// Decode failures are reported to the model as "invalid input: ..." so it can
// correct the call itself.
func Typed[I any](d Def, fn func(context.Context, I) (string, error)) Def {
	d.Fn = func(ctx context.Context, raw json.RawMessage) (string, error) {
		var in I
		if len(raw) > 0 {
			if err := json.Unmarshal(raw, &in); err != nil {
				// Wrapping ErrInvalidInput rather than only the decode error
				// puts this on the same footing as WithValidation's rejection:
				// one errors.Is target for "the arguments were wrong", and a
				// [CallerFault] so a model fumbling a schema does not get its
				// own tool taken away.
				return "", fmt.Errorf("%w: %s: %w", ErrInvalidInput, d.Name, err)
			}
		}
		return fn(ctx, in)
	}
	return d
}

// ===== Sets =====

// Set is the tool surface an agent presents.
//
// An interface rather than a slice because the surface changes mid-run: a
// skill-gated set hides tools until the corresponding skill is loaded, and
// Visible is re-evaluated before every model call.
//
// Visible takes a context and Find does not, and the split is load-bearing:
//
//   - Visibility is per-run state. An Agent is immutable and shared across
//     goroutines, so a Set that kept activations in a field would leak them
//     between concurrent runs. Reading them off the context keeps each run's
//     surface its own, and a forked sub-agent inherits its parent's simply by
//     inheriting the context.
//
//   - Find is the full registry, deliberately unfiltered. When the model calls
//     a hidden tool, the dispatcher needs to answer "load the X skill first"
//     rather than "unknown tool" — which it can only do if it can still see
//     the tool it is refusing to run.
type Set interface {
	Visible(ctx context.Context) []Def
	Find(name string) (Def, bool)
}

// Reconciler is an optional interface a [Set] may implement to learn which
// tool_use ids survived the transcript strategy.
//
// It closes a gap that only shows up once trimming and gating coexist. A
// skill's body enters the conversation as a tool_result; the strategy — or a
// server-side context edit — can drop that message while the skill stays
// activated, leaving the model holding tools whose knowledge has been
// evicted. A Set that tracks where its state came from can undo the
// activation instead.
//
// The agent loop calls this after materializing each request, with the ids
// still present.
type Reconciler interface {
	Reconcile(ctx context.Context, present map[llm.ToolUseID]bool)
}

// RunChecker is an optional interface a [Set] may implement to say whether a
// run's context carries what the set needs — the skill set's activation
// state, say. The agent asks once, when it is built, with the context its runs
// would get, and warns on an error: wiring that would otherwise only show up
// as tools that never appear.
type RunChecker interface {
	CheckRun(ctx context.Context) error
}

type staticSet struct {
	defs  []Def
	index map[string]int
}

// NewSet builds an immutable set. Duplicate names are a programming error and
// panic: silently shadowing a tool produces a model that calls one thing and
// runs another.
func NewSet(defs ...Def) Set {
	s := &staticSet{defs: defs, index: make(map[string]int, len(defs))}
	for i, d := range defs {
		if _, dup := s.index[d.Name]; dup {
			panic("tool: duplicate tool name " + d.Name)
		}
		s.index[d.Name] = i
	}
	return s
}

func (s *staticSet) Visible(context.Context) []Def { return s.defs }

func (s *staticSet) Find(name string) (Def, bool) {
	i, ok := s.index[name]
	if !ok {
		return Def{}, false
	}
	return s.defs[i], true
}

// Filter keeps the definitions satisfying pred, preserving order.
func Filter(defs []Def, pred func(Def) bool) []Def {
	out := make([]Def, 0, len(defs))
	for _, d := range defs {
		if pred(d) {
			out = append(out, d)
		}
	}
	return out
}

// OnlyCaps keeps tools whose capabilities are a subset of allowed.
func OnlyCaps(allowed Cap) func(Def) bool {
	return func(d Def) bool { return d.Caps&^allowed == 0 }
}

// Provided keeps tools whose needs are met by the host.
func Provided(have Need) func(Def) bool {
	return func(d Def) bool { return d.Needs&^have == 0 }
}

// ===== Annotation =====

// Annotate labels the tool_result this call is about to produce.
//
// A tool's return type is (string, error) and nothing else, which is what
// keeps a tool an ordinary function. But the harness sometimes needs to know
// something about an observation that the observation itself cannot carry —
// that this 9 KB of text is a skill body rather than grep output, say, so a
// transcript strategy can evict it on purpose instead of by position.
//
// Labels ride on the context for the same reason [Info] does: they are
// per-call, and threading them through the signature would tax every tool for
// a facility two of them use. Tags are opaque here; conventions live with
// whoever reads them ("skill:pdf-forms").
//
// A no-op outside a dispatcher, so a tool that annotates is still callable
// directly in a test.
func Annotate(ctx context.Context, tags ...string) {
	if a, ok := ctx.Value(annotateKey{}).(*annotations); ok && a != nil {
		a.mu.Lock()
		a.tags = append(a.tags, tags...)
		a.mu.Unlock()
	}
}

type annotateKey struct{}

// Attach adds media to the result of the call ctx belongs to: a screenshot, a
// rendered page, a PDF. The model receives it together with the call's text
// output, which should still say what was attached — a model that cannot see
// it, and a person reading a text log, have only the words.
//
// A tool.Fn returns a string, and that stays the whole signature; media rides
// on the call's context the way [Annotate]'s tags do, so a tool that never
// attaches anything is unchanged, and so is every middleware that only looks at
// the string. Media attached by a call that then fails is dropped with its
// output: an error result is text.
//
// A no-op outside a dispatcher.
func Attach(ctx context.Context, media ...llm.Media) {
	if a, ok := ctx.Value(annotateKey{}).(*annotations); ok && a != nil {
		a.mu.Lock()
		a.media = append(a.media, media...)
		a.mu.Unlock()
	}
}

// Attached returns the media attached to the call ctx belongs to so far. It is
// for middleware that records a call and has to give the media back when it
// replays one — the tape.
func Attached(ctx context.Context) []llm.Media {
	if a, ok := ctx.Value(annotateKey{}).(*annotations); ok && a != nil {
		a.mu.Lock()
		defer a.mu.Unlock()
		return append([]llm.Media(nil), a.media...)
	}
	return nil
}

// annotations is the per-call scratch the dispatcher hangs on each call's
// context: the tags of [Annotate], the media of [Attach], and the keys of
// [Claim]. One per dispatched call, which is what scopes all three to exactly
// that call.
type annotations struct {
	mu      sync.Mutex
	tags    []string
	media   []llm.Media
	claimed map[any]struct{}
}

// FindPause returns the first tool_use naming a CapPause tool.
//
// Pause is a property of the tool, not of a hard-coded name: any set that
// includes a CapPause tool can suspend, and a set without one never can.
//
// It finds a candidate and decides nothing. [Set.Find] is unfiltered, so the
// tool may be hidden this turn or refused by policy; the agent loop dispatches
// such a call through the middleware chain for a verdict before it suspends.
func FindPause(set Set, uses []llm.ToolUse) (llm.ToolUse, bool) {
	for _, u := range uses {
		if d, ok := set.Find(u.Name); ok && d.Has(CapPause) {
			return u, true
		}
	}
	return llm.ToolUse{}, false
}

// FindTerminal returns the first tool_use naming a CapTerminal tool. Like
// [FindPause], it finds a candidate and decides nothing.
func FindTerminal(set Set, uses []llm.ToolUse) (llm.ToolUse, bool) {
	for _, u := range uses {
		if d, ok := set.Find(u.Name); ok && d.Has(CapTerminal) {
			return u, true
		}
	}
	return llm.ToolUse{}, false
}

// ===== Per-call context =====

// Info is per-call data about the tool invocation in flight.
//
// This and Lookup are the only things that travel in the context rather than
// in a closure, and the distinction is deliberate: dependencies are injected
// at construction, request-scoped data rides on ctx.
type Info struct {
	UseID llm.ToolUseID
}

type infoKey struct{}

// WithInfo attaches per-call information to ctx. The dispatcher does this.
func WithInfo(ctx context.Context, i Info) context.Context {
	return context.WithValue(ctx, infoKey{}, i)
}

// InfoFrom retrieves per-call information. The zero value is usable.
func InfoFrom(ctx context.Context) Info {
	i, _ := ctx.Value(infoKey{}).(Info)
	return i
}

// Lookup resolves a previous tool_result by its tool_use_id.
//
// It lets a tool act on an earlier observation without the model re-emitting
// it — and paying output tokens for the privilege. Per-run rather than
// per-call, because it reads the transcript.
type Lookup func(llm.ToolUseID) (string, error)

type lookupKey struct{}

// WithLookup attaches a transcript resolver to ctx. The agent loop does this.
func WithLookup(ctx context.Context, l Lookup) context.Context {
	return context.WithValue(ctx, lookupKey{}, l)
}

// LookupFrom retrieves the transcript resolver, or nil when unavailable.
// Tools that use it must handle nil — they can run outside an agent loop.
func LookupFrom(ctx context.Context) Lookup {
	l, _ := ctx.Value(lookupKey{}).(Lookup)
	return l
}

// ===== Observation =====

// Phase distinguishes the two halves of an observed call.
type Phase int

// Observation phases.
const (
	PhaseStart Phase = iota
	PhaseDone
)

// Observation is one bracket of a logical tool call, reported by WithObserver.
//
// "Logical" means post-retry: the middleware sits outermost, so retries,
// circuit-breaker trips and dedup collapse into a single Start/Done pair
// rather than leaking every internal attempt to the UI.
type Observation struct {
	Phase  Phase
	Def    Def
	Use    llm.ToolUse
	Output string        // PhaseDone only
	Err    error         // PhaseDone only
	Dur    time.Duration // PhaseDone only
}

// WithObserver brackets each logical call with Start/Done callbacks.
//
// The callback receives the context so it can route to a per-run sink without
// the middleware itself being per-run — the chain is built once, when the
// agent is constructed, and every run reuses it.
func WithObserver(obs func(context.Context, Observation)) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
			obs(ctx, Observation{Phase: PhaseStart, Def: d, Use: use})
			start := time.Now()
			out, err := next(ctx, d, use)
			obs(ctx, Observation{
				Phase: PhaseDone, Def: d, Use: use,
				Output: out, Err: err, Dur: time.Since(start),
			})
			return out, err
		}
	}
}
