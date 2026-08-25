package wombat

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// overflowClient rejects any request carrying more than limit messages, the
// way a provider rejects an oversized context. It answers with one tool call
// and then a text answer, so the run spans more than one iteration and the
// stickiness of the escalation is observable.
type overflowClient struct {
	limit int

	mu       sync.Mutex
	sizes    []int
	rejects  int
	toolOnce bool
}

func (c *overflowClient) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	c.mu.Lock()
	c.sizes = append(c.sizes, len(req.Messages))
	if len(req.Messages) > c.limit {
		c.rejects++
		c.mu.Unlock()
		return llm.Response{}, &llm.APIError{
			Class:   llm.ErrContextWindow,
			Status:  400,
			Message: "prompt is too long: 300000 tokens > 200000 maximum",
		}
	}
	first := !c.toolOnce
	c.toolOnce = true
	c.mu.Unlock()

	if first {
		return toolTurn("o1", "echo", `{}`), nil
	}
	return textTurn("ok"), nil
}

func (c *overflowClient) stats() ([]int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]int(nil), c.sizes...), c.rejects
}

func TestOverflowIsFatalWithoutRecovery(t *testing.T) {
	bare := &overflowClient{limit: 12}
	a := newAgent(t, bare, nil)

	_, err := a.Run(context.Background(), Continue(longTranscript(30)))
	if !errors.Is(err, llm.ErrContextWindow) {
		t.Fatalf("error = %v, want llm.ErrContextWindow", err)
	}

	// ErrContextWindow is deliberately not retryable: the identical request
	// fails identically, so retrying is money spent on the same 400.
	_, rejects := bare.stats()
	if rejects != 1 {
		t.Errorf("rejects = %d, want 1 — an oversized request must not be retried blindly", rejects)
	}
}

func TestOverflowRecoveryClimbsTheDefaultLadder(t *testing.T) {
	cl := &overflowClient{limit: 12}
	a := newAgent(t, llm.Chain(cl, WithOverflowRecovery()), []tool.Def{echoTool("echo", "ok")})

	out, err := a.Run(context.Background(), Continue(longTranscript(30)))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if _, ok := out.(Answer); !ok {
		t.Fatalf("Outcome = %T, want Answer", out)
	}

	sizes, rejects := cl.stats()
	if len(sizes) < 3 || sizes[0] != 30 || sizes[1] != 24 || sizes[2] != 10 {
		t.Fatalf("request sizes = %v, want the ladder 30 -> 24 -> 10", sizes)
	}

	// Stickiness is the whole point: iteration 2 must start already shrunk
	// rather than paying for another rejected round trip to rediscover it.
	if rejects != 2 {
		t.Errorf("rejects = %d, want 2 — the escalation was rediscovered rather than remembered (sizes %v)", rejects, sizes)
	}
	if len(sizes) < 4 || sizes[3] > 12 {
		t.Errorf("request sizes = %v, want the fourth call to start at the escalated level", sizes)
	}
}

func TestOverflowRecoveryGivesUpWhenTheLadderRunsOut(t *testing.T) {
	tiny := &overflowClient{limit: 1}
	a := newAgent(t, llm.Chain(tiny, WithOverflowRecovery(SlidingWindow(24))), nil)

	_, err := a.Run(context.Background(), Continue(longTranscript(30)))
	if !errors.Is(err, llm.ErrContextWindow) {
		t.Fatalf("error = %v, want llm.ErrContextWindow", err)
	}
	// The provider's own error is reported rather than a synthetic one,
	// because it names the actual limit an operator needs.
	if !strings.Contains(err.Error(), "200000 maximum") {
		t.Errorf("error = %q, want the provider's own message", err)
	}
	sizes, rejects := tiny.stats()
	if rejects > 3 {
		t.Errorf("rejects = %d (sizes %v), want the ladder to stop rather than spin", rejects, sizes)
	}
}

// TestOverflowLadderSkipsANoOpRung is a regression test.
//
// A semantic first rung — "drop the loaded skill bodies" — removes nothing
// from a transcript that never loaded one. Treating that as the end of the
// ladder strands every windowing rung behind it: exactly the rungs that would
// have worked. The ladder must SKIP a rung that shortens nothing and try the
// next, and only give up when none of them do.
//
// The fixture: rung 1 is DropTagged over a tag no result carries, so it is a
// guaranteed no-op. Rung 2 is a window that does fit. A ladder that stopped at
// rung 1 would return the provider's 400 instead of an answer.
func TestOverflowLadderSkipsANoOpRung(t *testing.T) {
	cl := &overflowClient{limit: 12}
	a := newAgent(t, llm.Chain(cl, WithOverflowRecovery(
		DropTagged(0, 0, "bulk"), // removes nothing: no result carries "bulk"
		SlidingWindow(8),
	)), nil)

	out, err := a.Run(context.Background(), Continue(longTranscript(30)))
	if err != nil {
		t.Fatalf("Run: %v — the ladder gave up on the no-op rung instead of skipping it", err)
	}
	if _, ok := out.(Answer); !ok {
		t.Fatalf("Outcome = %T, want Answer", out)
	}

	sizes, rejects := cl.stats()
	if len(sizes) < 2 {
		t.Fatalf("request sizes = %v, want at least two calls", sizes)
	}
	if sizes[0] != 30 {
		t.Errorf("first request = %d messages, want 30", sizes[0])
	}
	// The no-op rung must not have produced a request of its own: it changed
	// nothing, so re-sending 30 messages would just buy another rejection.
	if sizes[1] != 8 {
		t.Errorf("second request = %d messages, want 8 (the no-op rung skipped straight to the window); "+
			"a second request of 30 means the no-op rung was sent anyway", sizes[1])
	}
	if rejects != 1 {
		t.Errorf("rejects = %d, want 1", rejects)
	}
}

func TestNextRung(t *testing.T) {
	ctx := context.Background()
	original := longTranscript(30)
	noop := DropTagged(0, 0, "bulk")

	t.Run("skips past rungs that remove nothing", func(t *testing.T) {
		ladder := []Strategy{noop, noop, SlidingWindow(8)}
		level, shrunk, ok := nextRung(ctx, ladder, 0, original, original)
		if !ok {
			t.Fatal("ok = false, want the ladder to reach the window")
		}
		if level != 3 {
			t.Errorf("level = %d, want 3", level)
		}
		if len(shrunk) >= len(original) {
			t.Errorf("shrunk to %d messages, want fewer than %d", len(shrunk), len(original))
		}
	})

	t.Run("reports exhaustion when nothing shortens the transcript", func(t *testing.T) {
		ladder := []Strategy{noop, noop}
		if _, _, ok := nextRung(ctx, ladder, 0, original, original); ok {
			t.Error("ok = true, want false when no rung removes anything")
		}
	})

	t.Run("starts above the current level", func(t *testing.T) {
		ladder := []Strategy{SlidingWindow(24), SlidingWindow(10)}
		current := applyLevel(ctx, ladder, 1, original)
		level, shrunk, ok := nextRung(ctx, ladder, 1, current, original)
		if !ok {
			t.Fatal("ok = false, want the second rung")
		}
		if level != 2 {
			t.Errorf("level = %d, want 2", level)
		}
		if len(shrunk) >= len(current) {
			t.Errorf("shrunk to %d, want fewer than %d", len(shrunk), len(current))
		}
	})

	t.Run("past the end of the ladder", func(t *testing.T) {
		ladder := []Strategy{SlidingWindow(24)}
		if _, _, ok := nextRung(ctx, ladder, 1, original, original); ok {
			t.Error("ok = true, want false past the last rung")
		}
	})
}

func TestApplyLevel(t *testing.T) {
	ctx := context.Background()
	msgs := longTranscript(30)
	ladder := []Strategy{SlidingWindow(24), SlidingWindow(10)}

	tests := []struct {
		name  string
		level int
		want  int
	}{
		{name: "level 0 is the transcript as given", level: 0, want: 30},
		{name: "negative level", level: -1, want: 30},
		{name: "level 1", level: 1, want: 24},
		{name: "level 2", level: 2, want: 10},
		{name: "past the ladder", level: 3, want: 30},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := len(applyLevel(ctx, ladder, tc.level, msgs)); got != tc.want {
				t.Errorf("got %d messages, want %d", got, tc.want)
			}
		})
	}
}

func TestOverflowState(t *testing.T) {
	t.Run("nil is safe and reads as level 0", func(t *testing.T) {
		var s *overflowState
		if got := s.level(); got != 0 {
			t.Errorf("level() = %d, want 0", got)
		}
		s.raise(3) // must not panic
	})

	t.Run("the level only ever rises", func(t *testing.T) {
		s := &overflowState{}
		s.raise(2)
		s.raise(1)
		if got := s.level(); got != 2 {
			t.Errorf("level() = %d, want 2 — a transcript that outgrew the window does not shrink back", got)
		}
		s.raise(3)
		if got := s.level(); got != 3 {
			t.Errorf("level() = %d, want 3", got)
		}
	})

	t.Run("a context with no state degrades to a throwaway", func(t *testing.T) {
		s := overflowFrom(context.Background())
		if s == nil {
			t.Fatal("overflowFrom returned nil")
		}
		s.raise(1)
		if got := s.level(); got != 1 {
			t.Errorf("level() = %d, want 1 — a throwaway still recovers within one call", got)
		}
		// ...but it cannot remember: a fresh lookup starts over.
		if got := overflowFrom(context.Background()).level(); got != 0 {
			t.Errorf("a second throwaway = %d, want 0", got)
		}
	})

	t.Run("the state on ctx is the one used", func(t *testing.T) {
		st := &overflowState{}
		ctx := withOverflowState(context.Background(), st)
		overflowFrom(ctx).raise(4)
		if got := st.level(); got != 4 {
			t.Errorf("level() = %d, want 4", got)
		}
	})
}

// Two runs of one agent must not share an escalation level: a fresh
// conversation starts at the top of the ladder, because the state lives on the
// run's context and not on the Agent.
func TestOverflowEscalationIsPerRun(t *testing.T) {
	cl := &overflowClient{limit: 12}
	a := newAgent(t, llm.Chain(cl, WithOverflowRecovery(SlidingWindow(10))), []tool.Def{echoTool("echo", "ok")})

	if _, err := a.Run(context.Background(), Continue(longTranscript(30))); err != nil {
		t.Fatalf("first run: %v", err)
	}
	afterFirst, _ := cl.stats()
	if len(afterFirst) < 2 || afterFirst[0] != 30 {
		t.Fatalf("first run request sizes = %v, want it to start at 30", afterFirst)
	}

	// Same agent, same client, a second run from the same oversized transcript.
	if _, err := a.Run(context.Background(), Continue(longTranscript(30))); err != nil {
		t.Fatalf("second run: %v", err)
	}
	all, _ := cl.stats()

	if len(all) <= len(afterFirst) {
		t.Fatalf("the second run made no calls: sizes %v", all)
	}
	if got := all[len(afterFirst)]; got != 30 {
		t.Errorf("the second run's first request = %d messages, want 30 — "+
			"it inherited the first run's escalation level (sizes %v)", got, all)
	}
}

func TestResultsOnContext(t *testing.T) {
	if got := resultsFrom(context.Background()); got != nil {
		t.Errorf("resultsFrom(background) = %v, want nil", got)
	}
	m := map[llm.ToolUseID]ResultInfo{"a": {Tool: "t", Tags: []string{"skill"}}}
	ctx := withResults(context.Background(), m)
	got := resultsFrom(ctx)
	if len(got) != 1 || got["a"].Tool != "t" {
		t.Errorf("resultsFrom = %v, want the published map", got)
	}
}

// The default ladder is documented as three progressively tighter windows;
// pinning it keeps the doc comment and the code from drifting apart.
func TestDefaultOverflowEscalation(t *testing.T) {
	want := []string{"sliding_window(keep=24)", "sliding_window(keep=10)", "sliding_window(keep=4)"}
	if len(DefaultOverflowEscalation) != len(want) {
		t.Fatalf("ladder has %d rungs, want %d", len(DefaultOverflowEscalation), len(want))
	}
	for i, s := range DefaultOverflowEscalation {
		if s.String() != want[i] {
			t.Errorf("rung %d = %q, want %q", i, s.String(), want[i])
		}
	}
}
