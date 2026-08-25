package skill

import (
	"context"
	"fmt"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// Gated is what an agent needs to run skill-gated tools: a dynamic tool
// surface, the middleware that enforces it, and the system-prompt index.
//
// Three values rather than one option, because they plug into three different
// seams of the harness and one of them (Index) is a string that has to reach
// the prompt before the first call. See the package doc for the wiring.
type Gated struct {
	// Set is the per-turn tool surface. It also implements [tool.Reconciler],
	// which the agent loop discovers by type assertion — pass it through
	// wombat.WithToolSet and reconciliation comes along.
	Set tool.Set

	// Middleware refuses gated tools whose skill is not loaded.
	Middleware tool.Middleware

	// Index is the body of the <available_skills> system block, or "" when
	// there are no loadable skills. See [Registry.Index] for why it carries
	// no tags.
	Index string

	// Installed is the concatenated body of every skill declaring
	// install: true, for a system block of its own. Empty when there are none.
	//
	// It goes in the prompt rather than through load_skill because a method
	// skill has to be present before the first decision, and asking a model to
	// route to it is asking too late. See [Skill.Install].
	Installed string
}

// Bind produces the surface, the middleware and the index over base.
//
// base is the agent's full tool list, gated and ungated alike. The two meta
// tools are appended by Bind, so callers must not add them; that also means
// the pair is always present, and a run can always load its way out of an
// empty surface.
func (r *Registry) Bind(base []tool.Def) Gated {
	s := &gatedSet{
		reg:    r,
		base:   append([]tool.Def(nil), base...),
		load:   r.loadTool(),
		unload: r.unloadTool(),
	}

	s.all = make(map[string]tool.Def, len(s.base)+2)
	for _, d := range s.base {
		if d.Name == LoadToolName || d.Name == UnloadToolName {
			panic("skill: base tool set already contains " + d.Name + "; Bind adds it")
		}
		if _, dup := s.all[d.Name]; dup {
			panic("skill: duplicate tool name " + d.Name)
		}
		s.all[d.Name] = d
	}
	s.all[s.load.Name] = s.load
	s.all[s.unload.Name] = s.unload

	return Gated{
		Set:        s,
		Middleware: r.middleware(),
		Index:      r.Index(),
		Installed:  r.InstalledBody(),
	}
}

// gatedSet is a [tool.Set] whose visibility depends on the run's [State].
type gatedSet struct {
	reg          *Registry
	base         []tool.Def
	load, unload tool.Def
	all          map[string]tool.Def
}

// Visible reports the tools offered to the model this turn: every ungated base
// tool, every gated tool whose skill is loaded in ctx, then load_skill and
// unload_skill.
//
// Order is stable — base order, then the two meta tools — for the same reason
// the index is sorted: the tool list is serialized into every request, and a
// list that permutes itself between turns invalidates the cached prefix.
func (s *gatedSet) Visible(ctx context.Context) []tool.Def {
	st := StateFrom(ctx)

	out := make([]tool.Def, 0, len(s.base)+2)
	for _, d := range s.base {
		gate, gated := s.reg.GateFor(d.Name)
		if gated && !s.reg.gateOpen(gate, st) {
			continue
		}
		out = append(out, d)
	}
	// Always last, always present: without load_skill the model has no way to
	// reach anything hidden, and a surface it cannot escape from is worse than
	// no gating at all.
	return append(out, s.load, s.unload)
}

// Find resolves any tool in the registry, INCLUDING hidden ones.
//
// Deliberately unfiltered, per the [tool.Set] contract. The dispatcher looks a
// tool up before it runs it, and a model can call a name it saw in an earlier
// turn long after the skill was unloaded. If Find hid it, that call would come
// back "unknown tool" — a dead end the model cannot act on — instead of
// reaching the middleware, which answers "call load_skill first". You can only
// refuse a tool by name if you can still see it.
func (s *gatedSet) Find(name string) (tool.Def, bool) {
	d, ok := s.all[name]
	return d, ok
}

// Reconcile retires skills whose body has left the transcript.
//
// This closes a real defect that only appears once trimming and gating coexist.
// A skill's knowledge lives in a tool_result; wombat.DropTagged — or a
// server-side context edit, or a resumed transcript — can drop that message
// while the activation, which lives in [State], survives. The model is then
// holding tools whose instructions are gone: it will keep calling fill_pdf
// with no idea how, and the failures look like model error rather than
// harness error.
//
// The agent loop calls this after materializing each request and before
// reading Visible, so the surface reported this turn already reflects the
// eviction.
func (s *gatedSet) Reconcile(ctx context.Context, present map[llm.ToolUseID]bool) {
	st := StateFrom(ctx)
	for _, name := range st.Active() {
		id, ok := st.BodyOf(name)
		if !ok {
			continue
		}
		if id == "" {
			// Activated outside a dispatch (a direct handler call in a test),
			// so there is no tool_result to look for. Absence of evidence is
			// not eviction; leave it alone.
			continue
		}
		if !present[id] {
			st.Deactivate(name)
		}
	}
}

// gateOpen reports whether a gating skill's knowledge is in the conversation.
//
// One helper for both the visibility and the enforcement side, because those
// two answering differently is precisely the bug the gate exists to prevent:
// a tool offered but refused, or hidden but runnable.
//
// An installed skill is permanently open. Its body is in the system prompt, so
// the condition the gate protects — that the model has the knowledge before it
// gets the tool — is satisfied before the run begins.
func (r *Registry) gateOpen(gate string, st *State) bool {
	return r.IsInstalled(gate) || st.IsActive(gate)
}

// middleware refuses a gated tool whose skill is not loaded.
//
// Visibility and enforcement are two different jobs, and only doing the first
// is the classic mistake. [gatedSet.Visible] controls what the model is OFFERED
// this turn; it does not control what the model CALLS. A tool name that
// appeared in turn 3 is still sitting in the transcript at turn 30, models
// routinely call from memory, and a provider will happily forward a call for a
// tool absent from the current list. Without this layer, gating is a
// suggestion.
//
// The refusal names the fix, because an error the model can act on costs one
// iteration and an error it cannot costs the run.
func (r *Registry) middleware() tool.Middleware {
	return func(next tool.Handler) tool.Handler {
		return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
			// The escape hatch is never gated. Refusing load_skill because its
			// skill is not loaded would be a deadlock, and gating unload_skill
			// would strand resources the model was trying to release.
			if d.Name == LoadToolName || d.Name == UnloadToolName {
				return next(ctx, d, use)
			}
			if gate, gated := r.GateFor(d.Name); gated && !r.gateOpen(gate, StateFrom(ctx)) {
				return "", fmt.Errorf("tool %q needs the %q skill: call %s(%q) first",
					d.Name, gate, LoadToolName, gate)
			}
			return next(ctx, d, use)
		}
	}
}
