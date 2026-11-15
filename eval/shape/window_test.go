package shape_test

import (
	"context"
	"testing"

	"wombat/eval/shape"
	"wombat/pkg/core"
)

// Inside ONE run a position-based window cannot trim a tool loop at all, and
// that is a property of SafeCut rather than of the windows.
//
// SafeCut will only cut immediately before a user turn carrying no
// tool_result, because any other user turn is answering a tool_use that would
// be orphaned. In a ReAct loop every user turn after the first is exactly
// that, and the opening instruction at index 0 is excluded by the search
// itself — so there is no legal cut point and every window returns the whole
// transcript.
//
// This is not a bug: an over-long request is recoverable and an orphaned
// tool_result is a hard provider error, so refusing to cut is the right
// failure. It IS worth pinning, because it decides where windowing can help at
// all — see TestSessionWindowCachePrefix, which measures the multi-turn case
// where it does.
func TestSingleRunToolLoopHasNoLegalCutPoint(t *testing.T) {
	const turns = 20

	for _, tc := range []struct {
		name     string
		strategy core.Strategy
	}{
		{"flat", core.Flat},
		{"sliding_window", core.SlidingWindow(6)},
		{"sliding_window_at", core.SlidingWindowAt(14, 6)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			shots, err := shape.Run(context.Background(), shape.Config{
				Turns:   turns,
				Payload: 128,
				Options: []core.Option{core.WithStrategy(tc.strategy)},
			})
			if err != nil {
				t.Fatal(err)
			}
			st := shape.PrefixStability(shots)
			t.Logf("%s", st)
			if st.Reanchored != 0 {
				t.Errorf("re-anchored %d times inside one run; a tool loop has no legal cut "+
					"point, so every window should degrade to flat here", st.Reanchored)
			}
			// The transcript only ever grows, so each turn's message count
			// must be the previous one plus the pair that was just added.
			for i := 1; i < len(shots); i++ {
				if len(shots[i].Messages) <= len(shots[i-1].Messages) {
					t.Errorf("turn %d materialized %d messages, turn %d had %d — something trimmed",
						shots[i].Turn, len(shots[i].Messages), shots[i-1].Turn, len(shots[i-1].Messages))
				}
			}
		})
	}
}

// The system prompt is the head of the cache prefix. New renders it once and
// freezes the string, so nothing in a run may move it — including a turn
// notice, which appends to the last USER turn precisely so it does not.
func TestSystemPromptNeverDriftsWithinARun(t *testing.T) {
	shots, err := shape.Run(context.Background(), shape.Config{
		Turns:   8,
		Payload: 64,
		Options: []core.Option{
			core.WithSystemBlock("policy", "be brief"),
			core.WithStrategy(core.SlidingWindowAt(10, 4)),
			core.WithTurnNotice(func(_ context.Context, iter int) string {
				if iter == 4 {
					return "(reminder: you are halfway through the budget)"
				}
				return ""
			}),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if drift := shape.SystemDrift(shots); len(drift) != 0 {
		t.Errorf("system prompt changed at turns %v; the cache prefix cannot survive that", drift)
	}
	if shots[0].SystemLen == 0 {
		t.Error("system prompt is empty, so this test proved nothing")
	}
}

// The tool list is serialized into every request and is part of the cache
// prefix, so a set that permutes itself between turns is as expensive as one
// that changes. A static set must never move.
func TestStaticToolListNeverPermutes(t *testing.T) {
	shots, err := shape.Run(context.Background(), shape.Config{Turns: 6, Payload: 32})
	if err != nil {
		t.Fatal(err)
	}
	if drift := shape.ToolDrift(shots); len(drift) != 0 {
		t.Errorf("tool list moved during a run with a static set: %v", drift)
	}
}
