package shape_test

import (
	"context"
	"testing"

	"github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/eval/shape"
)

// A long multi-turn session is where a position-based window actually bites,
// and where SlidingWindowAt's doc says a plain window misses the cache every
// turn: "Nothing reports this. It is not an error, the answers are correct,
// and the only symptom is the bill."
//
// This is the report. No model, no key, no provider usage field — the cache
// boundary is computed from the requests the harness assembled.
func TestSessionWindowCachePrefix(t *testing.T) {
	const (
		turns   = 24
		calls   = 1
		keep    = 8
		trigger = 24 // delta = 16
	)

	run := func(name string, s wombat.Strategy) shape.Stability {
		t.Helper()
		shots, err := shape.RunSession(context.Background(),
			shape.Config{Payload: 200, Options: []wombat.Option{wombat.WithStrategy(s)}},
			turns, calls)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		st := shape.PrefixStability(shots)
		t.Logf("%-22s %s  (sent %d bytes)", name, st, shape.MaterializedBytes(shots))
		return st
	}

	flat := run("flat", wombat.Flat)
	sliding := run("sliding_window", wombat.SlidingWindow(keep))
	anchored := run("sliding_window_at", wombat.SlidingWindowAt(trigger, keep))

	if flat.Reanchored != 0 {
		t.Errorf("flat re-anchored %d times; a strategy that never trims cannot move the prefix",
			flat.Reanchored)
	}

	// Guard first: without this the whole comparison can pass by measuring a
	// session too short to trim, which is how a mechanism test quietly stops
	// testing its mechanism.
	if sliding.Reanchored < 5 {
		t.Fatalf("sliding re-anchored only %d times over %d session turns; too little trimming "+
			"happened for this comparison to mean anything", sliding.Reanchored, turns)
	}

	if anchored.Reanchored >= sliding.Reanchored {
		t.Errorf("re-anchors: anchored %d, sliding %d — quantizing the cut bought nothing",
			anchored.Reanchored, sliding.Reanchored)
	}
	t.Logf("quantizing cut re-anchors from %d to %d over %d turns",
		sliding.Reanchored, anchored.Reanchored, turns)
}
