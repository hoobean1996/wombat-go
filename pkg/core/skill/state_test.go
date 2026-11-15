package skill

import (
	"context"
	"strings"
	"sync"
	"testing"

	"wombat/pkg/core/llm"
)

func TestStateActivateDeactivate(t *testing.T) {
	s := NewState()

	if s.IsActive("demo") {
		t.Error("IsActive(demo) = true on a fresh state, want false")
	}
	if got := s.Active(); len(got) != 0 {
		t.Errorf("Active() = %v on a fresh state, want none", got)
	}
	if id, ok := s.BodyOf("demo"); ok {
		t.Errorf("BodyOf(demo) = %q, true; want \"\", false", id)
	}

	s.Activate("demo", "u1")
	if !s.IsActive("demo") {
		t.Error("IsActive(demo) = false after Activate, want true")
	}
	if id, ok := s.BodyOf("demo"); !ok || id != "u1" {
		t.Errorf("BodyOf(demo) = %q, %v; want u1, true", id, ok)
	}

	// Re-activating overwrites the id, which is what a genuine second load
	// should do.
	s.Activate("demo", "u2")
	if id, _ := s.BodyOf("demo"); id != "u2" {
		t.Errorf("BodyOf(demo) = %q after re-activation, want u2", id)
	}

	s.Deactivate("demo")
	if s.IsActive("demo") {
		t.Error("IsActive(demo) = true after Deactivate, want false")
	}
	s.Deactivate("demo") // idempotent
	s.Deactivate("never-activated")
}

// TestStateActiveIsSorted: this list is shown to the model in load/unload
// observations, and an observation that reorders itself between otherwise
// identical turns is noise the model has to reconcile.
func TestStateActiveIsSorted(t *testing.T) {
	s := NewState()
	for _, n := range []string{"zebra", "alpha", "mike", "bravo"} {
		s.Activate(n, llm.ToolUseID("id-"+n))
	}
	want := "alpha, bravo, mike, zebra"
	for i := 0; i < 20; i++ {
		if got := strings.Join(s.Active(), ", "); got != want {
			t.Fatalf("Active() call %d = %q, want %q", i, got, want)
		}
	}
}

func TestStateFrom(t *testing.T) {
	t.Run("round trips", func(t *testing.T) {
		s := NewState()
		ctx := WithState(context.Background(), s)
		if got := StateFrom(ctx); got != s {
			t.Errorf("StateFrom returned %p, want the state we attached %p", got, s)
		}
		// A derived context still finds it: a sub-agent inherits the parent's
		// loaded skills for free.
		child, cancel := context.WithCancel(ctx)
		defer cancel()
		if got := StateFrom(child); got != s {
			t.Error("a derived context found a different state, want the parent's")
		}
	})

	t.Run("never nil", func(t *testing.T) {
		for _, ctx := range []context.Context{
			context.Background(),
			context.WithValue(context.Background(), struct{ k int }{}, "x"),
			WithState(context.Background(), nil), // a nil *State under the key
		} {
			if got := StateFrom(ctx); got == nil {
				t.Error("StateFrom returned nil, want a throwaway state")
			}
		}
	})

	t.Run("the fallback is a throwaway, not a global", func(t *testing.T) {
		// The consequence is deliberate: activations made against a throwaway
		// are not shared with anything, so a gated tool stays hidden. Nothing
		// is silently global.
		ctx := context.Background()
		StateFrom(ctx).Activate("demo", "u1")
		if StateFrom(ctx).IsActive("demo") {
			t.Error("an activation against the throwaway state persisted, want it discarded")
		}
		if a, b := StateFrom(ctx), StateFrom(ctx); a == b {
			t.Error("two StateFrom calls returned the same state, want a fresh throwaway each time")
		}
	})
}

// TestStateZeroValueActivates: Activate lazily builds the map, so a State
// obtained any other way than NewState still works rather than panicking on a
// nil map write.
func TestStateZeroValueActivates(t *testing.T) {
	var s State
	s.Activate("demo", "u1")
	if !s.IsActive("demo") {
		t.Error("IsActive(demo) = false on a zero-value State, want true")
	}
}

// TestStateIsRaceFree: the read side runs on the agent loop (Visible and
// Reconcile, once per turn) while the write side runs inside tool handlers,
// which the dispatcher may execute concurrently. Run with -race.
func TestStateIsRaceFree(t *testing.T) {
	s := NewState()
	const writers, readers, iters = 8, 8, 300

	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			name := string(rune('a' + w))
			for i := 0; i < iters; i++ {
				s.Activate(name, llm.ToolUseID(name))
				s.Deactivate(name)
			}
		}(w)
	}
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				_ = s.Active()
				_ = s.IsActive("a")
				_, _ = s.BodyOf("b")
			}
		}()
	}
	wg.Wait()

	if got := s.Active(); len(got) != 0 {
		t.Errorf("Active() = %v, want none: every Activate was paired with a Deactivate", got)
	}
}

// TestStatesAreIndependent: two runs of one agent must not share activation.
func TestStatesAreIndependent(t *testing.T) {
	a, b := NewState(), NewState()
	a.Activate("demo", "u1")

	if b.IsActive("demo") {
		t.Error("run B sees run A's activation, want per-run isolation")
	}
	b.Activate("other", "u2")
	if a.IsActive("other") {
		t.Error("run A sees run B's activation, want per-run isolation")
	}
	if got := strings.Join(a.Active(), ","); got != "demo" {
		t.Errorf("run A Active() = %q, want demo", got)
	}
	if got := strings.Join(b.Active(), ","); got != "other" {
		t.Errorf("run B Active() = %q, want other", got)
	}
}
