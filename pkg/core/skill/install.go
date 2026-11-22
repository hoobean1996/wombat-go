package skill

import (
	"context"

	"wombat/pkg/core"
)

// Install is every piece of wiring a gated tool surface needs, as one option:
//
//	g := reg.Bind(builtin.Default(deps))
//	a, _ := core.New(core.WithClient(client), g.Install())
//
// It used to be five options a caller had to know to pass together — the set,
// the middleware that enforces it, the two system blocks and a per-run State
// — and leaving any one out failed quietly: without the middleware a hidden
// tool still ran when the model named it, and without the State every
// load_skill reported success while its tools never appeared.
//
// Install replaces the agent's tool surface, so it goes AFTER any
// core.WithTools: WithTools resets the set.
func (g Gated) Install() core.Option {
	opts := []core.Option{
		core.WithToolSet(g.Set),
		core.WithToolMiddleware(g.Middleware),
		// A skill body is never clipped. Cutting its middle leaves the model
		// following instructions it only half saw, and loading the skill again
		// returns a reminder rather than the body, so the cut part could never
		// be recovered.
		core.WithTruncationExempt(LoadToolName),
	}
	// Installed skills first, then the loadable catalogue: method before
	// reference, in an order that never changes, because both blocks are part
	// of the prompt-cache prefix.
	if g.Installed != "" {
		opts = append(opts, core.WithSystemBlock("installed_skills", g.Installed))
	}
	if g.Index != "" {
		opts = append(opts, core.WithSystemBlock("available_skills", g.Index))
	}
	opts = append(opts, core.WithRunContext(func(ctx context.Context) context.Context {
		// Carried, so a run suspended at the end of its slice still has the
		// skills it loaded when it carries on.
		st := NewState()
		return core.Carry(WithState(ctx, st), "skill.active", st)
	}))
	return core.Options(opts...)
}

// CheckRun implements tool.RunChecker: a run of an agent offering this set
// must carry a State, or every load_skill fails with ErrNoState.
func (s *gatedSet) CheckRun(ctx context.Context) error {
	if _, ok := stateOn(ctx); !ok {
		return ErrNoState
	}
	return nil
}
