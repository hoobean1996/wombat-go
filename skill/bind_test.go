package skill

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// gatedFixture is the wiring used by most of these tests: one skill, one tool
// gated behind it, one ungated tool.
func gatedFixture(t *testing.T) (*Registry, Gated, tool.Handler) {
	t.Helper()
	r := New(demoSkill())
	r.Gate("demo", "secret")
	g := r.Bind([]tool.Def{echoTool("plain"), secretTool()})
	return r, g, tool.Chain(tool.Direct, g.Middleware)
}

func TestBindPanics(t *testing.T) {
	t.Run("base already contains load_skill", func(t *testing.T) {
		mustPanic(t, "base tool set already contains "+LoadToolName, func() {
			New(demoSkill()).Bind([]tool.Def{echoTool(LoadToolName)})
		})
	})
	t.Run("base already contains unload_skill", func(t *testing.T) {
		mustPanic(t, "base tool set already contains "+UnloadToolName, func() {
			New(demoSkill()).Bind([]tool.Def{echoTool(UnloadToolName)})
		})
	})
	t.Run("duplicate base tool", func(t *testing.T) {
		mustPanic(t, "duplicate tool name dup", func() {
			New(demoSkill()).Bind([]tool.Def{echoTool("dup"), echoTool("dup")})
		})
	})
}

// TestBindCopiesBase: Bind must not alias the caller's slice, or a later
// append by the caller could reshape the agent's tool surface underneath it.
func TestBindCopiesBase(t *testing.T) {
	base := []tool.Def{echoTool("plain"), secretTool()}
	g := New(demoSkill()).Bind(base)

	base[0] = echoTool("swapped")
	ctx := WithState(t.Context(), NewState())
	if got := orderedNames(g.Set.Visible(ctx)); got[0] != "plain" {
		t.Errorf("Visible()[0] = %q after mutating the caller's slice, want plain", got[0])
	}
}

func TestVisibleHidesAndReveals(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	before := orderedNames(g.Set.Visible(ctx))
	want := []string{"plain", LoadToolName, UnloadToolName}
	if strings.Join(before, ",") != strings.Join(want, ",") {
		t.Fatalf("Visible before load = %v, want %v", before, want)
	}

	res := dispatch1(t, ctx, h, g.Set, loadUse("u1", "demo"))
	if res.Err != nil {
		t.Fatalf("load_skill error = %v, want nil", res.Err)
	}

	after := orderedNames(g.Set.Visible(ctx))
	want = []string{"plain", "secret", LoadToolName, UnloadToolName}
	if strings.Join(after, ",") != strings.Join(want, ",") {
		t.Errorf("Visible after load = %v, want %v", after, want)
	}

	// Unloading hides it again.
	if res := dispatch1(t, ctx, h, g.Set, unloadUse("u2", "demo")); res.Err != nil {
		t.Fatalf("unload_skill error = %v, want nil", res.Err)
	}
	if got := orderedNames(g.Set.Visible(ctx)); strings.Join(got, ",") != strings.Join(before, ",") {
		t.Errorf("Visible after unload = %v, want %v", got, before)
	}
}

// TestVisibleAlwaysOffersTheMetaTools: a surface the model cannot escape from
// is worse than no gating at all.
func TestVisibleAlwaysOffersTheMetaTools(t *testing.T) {
	// Everything gated, nothing loaded.
	r := New(demoSkill())
	r.Gate("demo", "a")
	r.Gate("demo", "b")
	g := r.Bind([]tool.Def{echoTool("a"), echoTool("b")})

	got := orderedNames(g.Set.Visible(WithState(t.Context(), NewState())))
	want := []string{LoadToolName, UnloadToolName}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("Visible = %v, want exactly %v", got, want)
	}
}

// TestVisibleOrderIsStable: the tool list is serialized into every request, and
// a list that permutes itself between turns invalidates the cached prefix.
func TestVisibleOrderIsStable(t *testing.T) {
	r := New(demoSkill())
	r.Gate("demo", "secret")
	g := r.Bind([]tool.Def{echoTool("zulu"), secretTool(), echoTool("alpha")})

	ctx := WithState(t.Context(), NewState())
	first := orderedNames(g.Set.Visible(ctx))
	for i := 0; i < 20; i++ {
		if got := orderedNames(g.Set.Visible(ctx)); strings.Join(got, ",") != strings.Join(first, ",") {
			t.Fatalf("Visible call %d = %v, want the same order as %v", i, got, first)
		}
	}
	// Base order preserved (not sorted), meta tools last.
	want := []string{"zulu", "alpha", LoadToolName, UnloadToolName}
	if strings.Join(first, ",") != strings.Join(want, ",") {
		t.Errorf("Visible = %v, want %v (base order, then the meta tools)", first, want)
	}
}

func TestVisibleWithNoStateOnTheContext(t *testing.T) {
	// StateFrom hands back a throwaway, so a gated tool stays hidden. Nothing
	// is silently global.
	_, g, _ := gatedFixture(t)
	if got := names(g.Set.Visible(context.Background())); contains(got, "secret") {
		t.Errorf("Visible = %v with no State on the context, want the gated tool hidden", got)
	}
}

// TestFindIsUnfiltered: the dispatcher looks a tool up before it runs it, and
// a model can call a name it saw in an earlier turn. If Find hid it, that call
// would come back "unknown tool" — a dead end — instead of reaching the
// middleware, which answers "call load_skill first".
func TestFindIsUnfiltered(t *testing.T) {
	_, g, _ := gatedFixture(t)

	for _, name := range []string{"plain", "secret", LoadToolName, UnloadToolName} {
		if d, ok := g.Set.Find(name); !ok || d.Name != name {
			t.Errorf("Find(%q) = %+v, %v; want the def, true", name, d, ok)
		}
	}
	if d, ok := g.Set.Find("nope"); ok {
		t.Errorf("Find(nope) = %+v, true; want false", d)
	}
}

// ===== middleware =====

// TestMiddlewareRefusesAHiddenTool. Visibility controls what the model is
// OFFERED; it does not control what the model CALLS. A tool name that appeared
// in turn 3 is still sitting in the transcript at turn 30, models routinely
// call from memory, and a provider will happily forward a call for a tool
// absent from the current list. Without this layer, gating is a suggestion.
func TestMiddlewareRefusesAHiddenTool(t *testing.T) {
	_, g, _ := gatedFixture(t)

	ran := false
	next := func(context.Context, tool.Def, llm.ToolUse) (string, error) {
		ran = true
		return "should not happen", nil
	}
	h := g.Middleware(next)

	def, _ := g.Set.Find("secret")
	out, err := h(WithState(t.Context(), NewState()), def, llm.ToolUse{ID: "u1", Name: "secret"})

	if err == nil {
		t.Fatalf("error = nil (output %q), want a refusal", out)
	}
	if ran {
		t.Error("the underlying tool ran, want it refused before execution")
	}
	if out != "" {
		t.Errorf("output = %q, want \"\" on a refusal", out)
	}
	// The refusal names the fix, because an error the model can act on costs
	// one iteration and an error it cannot costs the run.
	for _, want := range []string{`"secret"`, `"demo"`, LoadToolName} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal = %q, want it to mention %q", err.Error(), want)
		}
	}
}

func TestMiddlewareAllows(t *testing.T) {
	r, g, _ := gatedFixture(t)
	// Even a gated meta tool must pass: refusing load_skill because its skill
	// is not loaded would be a deadlock, and gating unload_skill would strand
	// resources the model was trying to release.
	r.Gate("demo", LoadToolName)
	r.Gate("demo", UnloadToolName)

	st := NewState()
	ctx := WithState(t.Context(), st)

	tests := []struct {
		name     string
		toolName string
		activate bool
	}{
		{"an ungated tool", "plain", false},
		{"load_skill even when gated", LoadToolName, false},
		{"unload_skill even when gated", UnloadToolName, false},
		{"a gated tool once its skill is loaded", "secret", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.activate {
				st.Activate("demo", "u1")
				t.Cleanup(func() { st.Deactivate("demo") })
			}
			ran := false
			h := g.Middleware(func(context.Context, tool.Def, llm.ToolUse) (string, error) {
				ran = true
				return "ok", nil
			})
			def, _ := g.Set.Find(tt.toolName)
			out, err := h(ctx, def, llm.ToolUse{ID: "u", Name: tt.toolName})
			if err != nil {
				t.Fatalf("error = %v, want nil", err)
			}
			if !ran || out != "ok" {
				t.Errorf("ran = %v, output = %q; want true, \"ok\"", ran, out)
			}
		})
	}
}

// TestGatingEndToEnd walks a run through the dispatcher the way the agent loop
// does: hidden, refused, loaded, allowed.
func TestGatingEndToEnd(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	refused := dispatch1(t, ctx, h, g.Set, llm.ToolUse{ID: "u1", Name: "secret", Input: json.RawMessage(`{}`)})
	if refused.Err == nil {
		t.Fatalf("output = %q, want a refusal before the skill is loaded", refused.Output)
	}
	if !strings.Contains(refused.Err.Error(), LoadToolName) {
		t.Errorf("refusal = %q, want it to name %s", refused.Err, LoadToolName)
	}

	if res := dispatch1(t, ctx, h, g.Set, loadUse("u2", "demo")); res.Err != nil {
		t.Fatalf("load_skill error = %v, want nil", res.Err)
	}

	allowed := dispatch1(t, ctx, h, g.Set, llm.ToolUse{ID: "u3", Name: "secret", Input: json.RawMessage(`{}`)})
	if allowed.Err != nil {
		t.Fatalf("secret error = %v, want nil once demo is loaded", allowed.Err)
	}
	if allowed.Output != "secret-ran" {
		t.Errorf("secret output = %q, want %q", allowed.Output, "secret-ran")
	}
}

// TestPerRunIsolation: an Agent is immutable and shared across goroutines, so
// one run loading pdf-forms must not reveal fill_pdf to every other run.
func TestPerRunIsolation(t *testing.T) {
	_, g, h := gatedFixture(t)

	runA := WithState(t.Context(), NewState())
	runB := WithState(t.Context(), NewState())

	if res := dispatch1(t, runA, h, g.Set, loadUse("a1", "demo")); res.Err != nil {
		t.Fatalf("load_skill in run A: %v", res.Err)
	}

	if got := names(g.Set.Visible(runA)); !contains(got, "secret") {
		t.Errorf("run A Visible = %v, want it to include secret", got)
	}
	if got := names(g.Set.Visible(runB)); contains(got, "secret") {
		t.Errorf("run B Visible = %v, want secret still hidden", got)
	}
	res := dispatch1(t, runB, h, g.Set, llm.ToolUse{ID: "b1", Name: "secret", Input: json.RawMessage(`{}`)})
	if res.Err == nil {
		t.Errorf("run B called secret and got %q, want a refusal", res.Output)
	}
}

// TestConcurrentRunsShareOneGatedSet exercises the read/write split under
// -race: Visible runs on the agent loop while load_skill runs inside a tool
// handler, and both touch the run's State.
func TestConcurrentRunsShareOneGatedSet(t *testing.T) {
	_, g, h := gatedFixture(t)

	const runs = 16
	var wg sync.WaitGroup
	errs := make(chan string, runs*4)

	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := WithState(context.Background(), NewState())
			loader := i%2 == 0

			if loader {
				if res := dispatch1t(ctx, h, g.Set, loadUse("u", "demo")); res.Err != nil {
					errs <- "load_skill: " + res.Err.Error()
					return
				}
			}
			for j := 0; j < 20; j++ {
				visible := contains(names(g.Set.Visible(ctx)), "secret")
				if visible != loader {
					errs <- "visibility leaked across runs"
					return
				}
				res := dispatch1t(ctx, h, g.Set, llm.ToolUse{ID: "s", Name: "secret", Input: json.RawMessage(`{}`)})
				if loader && res.Err != nil {
					errs <- "a loaded run was refused: " + res.Err.Error()
					return
				}
				if !loader && res.Err == nil {
					errs <- "an unloaded run was allowed to call secret"
					return
				}
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for msg := range errs {
		t.Error(msg)
	}
}

// dispatch1t is dispatch1 without *testing.T, for use off the test goroutine.
func dispatch1t(ctx context.Context, h tool.Handler, set tool.Set, use llm.ToolUse) tool.Result {
	rs := tool.NewDispatcher(h).Dispatch(ctx, set, []llm.ToolUse{use})
	return rs[0]
}

// ===== Reconcile =====

// TestReconcileRetiresAnEvictedSkill covers a defect that only appears once
// trimming and gating coexist. A skill's knowledge lives in a tool_result;
// DropTagged — or a server-side context edit, or a resumed transcript — can
// drop that message while the activation, which lives in State, survives. The
// model is then holding tools whose instructions are gone: it keeps calling
// them with no idea how, and the failures look like model error rather than
// harness error.
func TestReconcileRetiresAnEvictedSkill(t *testing.T) {
	_, g, h := gatedFixture(t)
	rec, ok := g.Set.(tool.Reconciler)
	if !ok {
		t.Fatalf("Gated.Set is %T, want it to implement tool.Reconciler", g.Set)
	}

	ctx := WithState(t.Context(), NewState())
	if res := dispatch1(t, ctx, h, g.Set, loadUse("body-1", "demo")); res.Err != nil {
		t.Fatalf("load_skill error = %v, want nil", res.Err)
	}

	// The body is still in the transcript: the skill survives.
	rec.Reconcile(ctx, map[llm.ToolUseID]bool{"body-1": true, "other": true})
	if got := names(g.Set.Visible(ctx)); !contains(got, "secret") {
		t.Errorf("Visible = %v while the body is present, want secret still offered", got)
	}

	// The body has been trimmed out: the activation goes with it.
	rec.Reconcile(ctx, map[llm.ToolUseID]bool{"other": true})
	if got := names(g.Set.Visible(ctx)); contains(got, "secret") {
		t.Errorf("Visible = %v after the body was evicted, want secret withdrawn", got)
	}
	if StateFrom(ctx).IsActive("demo") {
		t.Error("demo is still active after its body was evicted, want it retired")
	}

	// And the middleware refuses it again, rather than running a tool whose
	// instructions are gone.
	res := dispatch1(t, ctx, h, g.Set, llm.ToolUse{ID: "u9", Name: "secret", Input: json.RawMessage(`{}`)})
	if res.Err == nil {
		t.Errorf("secret ran with output %q, want a refusal after eviction", res.Output)
	}
}

func TestReconcileEdgeCases(t *testing.T) {
	_, g, _ := gatedFixture(t)
	rec := g.Set.(tool.Reconciler)

	t.Run("an activation with no body id is left alone", func(t *testing.T) {
		// Activated outside a dispatch (a direct handler call in a test), so
		// there is no tool_result to look for. Absence of evidence is not
		// eviction.
		st := NewState()
		st.Activate("demo", "")
		ctx := WithState(t.Context(), st)

		rec.Reconcile(ctx, map[llm.ToolUseID]bool{})
		if !st.IsActive("demo") {
			t.Error("demo was retired despite having no body id, want it left alone")
		}
	})

	t.Run("an empty present set retires everything with a body", func(t *testing.T) {
		st := NewState()
		st.Activate("demo", "b1")
		ctx := WithState(t.Context(), st)

		rec.Reconcile(ctx, nil)
		if st.IsActive("demo") {
			t.Error("demo survived an empty present set, want it retired")
		}
	})

	t.Run("nothing active is a no-op", func(t *testing.T) {
		ctx := WithState(t.Context(), NewState())
		rec.Reconcile(ctx, map[llm.ToolUseID]bool{"x": true})
		if got := StateFrom(ctx).Active(); len(got) != 0 {
			t.Errorf("Active() = %v, want none", got)
		}
	})

	t.Run("no state on the context is a no-op", func(t *testing.T) {
		rec.Reconcile(context.Background(), map[llm.ToolUseID]bool{})
	})

	t.Run("only the evicted skill is retired", func(t *testing.T) {
		r := New(Skill{Name: "a", Description: "A."}, Skill{Name: "b", Description: "B."})
		g := r.Bind(nil)
		rec := g.Set.(tool.Reconciler)

		st := NewState()
		st.Activate("a", "body-a")
		st.Activate("b", "body-b")
		ctx := WithState(t.Context(), st)

		rec.Reconcile(ctx, map[llm.ToolUseID]bool{"body-b": true})
		if st.IsActive("a") {
			t.Error("a is still active, want it retired")
		}
		if !st.IsActive("b") {
			t.Error("b was retired, want it kept: its body is still present")
		}
	})
}
