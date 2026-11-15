package skill

import (
	"context"

	wombat "github.com/hoobean1996/wombat-go"
)

// Install is every piece of wiring a gated tool surface needs, as one option:
//
//	g := reg.Bind(builtin.Default(deps))
//	a, _ := wombat.New(wombat.WithClient(client), g.Install())
//
// It used to be five options a caller had to know to pass together — the set,
// the middleware that enforces it, the two system blocks and a per-run State
// — and leaving any one out failed quietly: without the middleware a hidden
// tool still ran when the model named it, and without the State every
// load_skill reported success while its tools never appeared.
//
// Install replaces the agent's tool surface, so it goes AFTER any
// wombat.WithTools: WithTools resets the set.
func (g Gated) Install() wombat.Option {
	opts := []wombat.Option{
		wombat.WithToolSet(g.Set),
		wombat.WithToolMiddleware(g.Middleware),
	}
	// Installed skills first, then the loadable catalogue: method before
	// reference, in an order that never changes, because both blocks are part
	// of the prompt-cache prefix.
	if g.Installed != "" {
		opts = append(opts, wombat.WithSystemBlock("installed_skills", g.Installed))
	}
	if g.Index != "" {
		opts = append(opts, wombat.WithSystemBlock("available_skills", g.Index))
	}
	opts = append(opts, wombat.WithRunContext(func(ctx context.Context) context.Context {
		return WithState(ctx, NewState())
	}))
	return wombat.Options(opts...)
}

// CheckRun implements tool.RunChecker: a run of an agent offering this set
// must carry a State, or every load_skill fails with ErrNoState.
func (s *gatedSet) CheckRun(ctx context.Context) error {
	if _, ok := stateOn(ctx); !ok {
		return ErrNoState
	}
	return nil
}
