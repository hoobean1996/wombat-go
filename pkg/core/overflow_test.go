package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
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
		level, shrunk, ok := nextRung(ctx, nil, ladder, 0, original, original)
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
		if _, _, ok := nextRung(ctx, nil, ladder, 0, original, original); ok {
			t.Error("ok = true, want false when no rung removes anything")
		}
	})

	t.Run("passes over a level that changes nothing again", func(t *testing.T) {
		ladder := []Strategy{SlidingWindow(24), SlidingWindow(10)}
		current := applyLevel(ctx, nil, ladder, 1, original)
		level, shrunk, ok := nextRung(ctx, nil, ladder, 1, current, original)
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
		current := applyLevel(ctx, nil, ladder, 1, original)
		if _, _, ok := nextRung(ctx, nil, ladder, 1, current, original); ok {
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
			if got := len(applyLevel(ctx, nil, ladder, tc.level, msgs)); got != tc.want {
				t.Errorf("got %d messages, want %d", got, tc.want)
			}
		})
	}
}

// A ladder's rungs compose: level 2 is rung 2 applied to rung 1's output, not
// rung 2 applied to the original.
func TestApplyLevelComposesTheRungs(t *testing.T) {
	ctx := withResults(context.Background(), map[llm.ToolUseID]ResultInfo{"s1": {Tool: "loader", Tags: []string{"skill"}}})
	msgs := append(longTranscript(13),
		assistantUse("s1", "loader"),
		userResult("s1", "BODY"),
		assistantText("noted"),
		userMsg("go on"),
	)
	ladder := []Strategy{DropTagged(0, 0, "skill"), SlidingWindow(6)}

	got := applyLevel(ctx, nil, ladder, 2, msgs)
	if strings.Contains(allText(got), "BODY") {
		t.Errorf("level 2 still carries the skill body rung 1 dropped: %d messages", len(got))
	}
	if want := ladder[1].Apply(View{Messages: ladder[0].Apply(View{Messages: msgs, Results: resultsFrom(ctx)})}); len(got) != len(want) {
		t.Errorf("level 2 = %d messages, want %d (rung 2 applied to rung 1's output)", len(got), len(want))
	}
	if err := Convo(got).Validate(); err != nil {
		t.Errorf("level 2 is not a valid transcript: %v", err)
	}
}

// bodyAverseClient rejects a request carrying more than limit messages, or any
// tool_result that still holds the skill body. It first asks for the skill to be
// loaded, then answers.
type bodyAverseClient struct {
	limit int

	mu       sync.Mutex
	accepted []llm.Request
}

func (c *bodyAverseClient) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(req.Messages) > c.limit || strings.Contains(allText(req.Messages), "SKILL-BODY") {
		return llm.Response{}, &llm.APIError{Class: llm.ErrContextWindow, Status: 400, Message: "prompt is too long"}
	}
	c.accepted = append(c.accepted, req)
	if len(c.accepted) == 1 {
		return toolTurn("s1", "loader", `{}`), nil
	}
	return textTurn("done"), nil
}

// TestOverflowLadderKeepsASemanticRungWhenItWindows is a regression test.
//
// The ladder is the one the WithOverflowRecovery doc recommends: drop skill
// bodies first, then window. The first call already overflows, and with no
// skill loaded yet the first rung is a no-op, so the level goes straight to the
// window and stays there. Then the model loads a skill. Its body sits inside
// the window, so the window alone no longer fits, and dropping the body alone
// would not fit either — only both together do.
//
// When each level applied only its own rung, level 2 was the bare window and
// brought the body straight back, the ladder ran out, and the run died with the
// provider's 400.
func TestOverflowLadderKeepsASemanticRungWhenItWindows(t *testing.T) {
	cl := &bodyAverseClient{limit: 10}
	loader := tool.Def{
		Name: "loader", Description: "loads a skill", InputSchema: json.RawMessage(objSchema),
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			tool.Annotate(ctx, "skill")
			return "SKILL-BODY " + strings.Repeat("x", 64), nil
		},
	}
	a := newAgent(t, llm.Chain(cl, WithOverflowRecovery(DropTagged(0, 0, "skill"), SlidingWindow(8))),
		[]tool.Def{loader})

	out, err := a.Run(context.Background(), Continue(longTranscript(13)))
	if err != nil {
		t.Fatalf("Run: %v — the window rung brought back the skill body the first rung had dropped", err)
	}
	if ans, ok := out.(Answer); !ok || ans.Text != "done" {
		t.Fatalf("Outcome = %#v, want the answer", out)
	}
	cl.mu.Lock()
	defer cl.mu.Unlock()
	if len(cl.accepted) != 2 {
		t.Fatalf("accepted %d requests, want 2", len(cl.accepted))
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

// The default ladder is documented as a token rung and then three
// progressively tighter windows; pinning it keeps the doc comment and the code
// from drifting apart.
func TestDefaultOverflowEscalation(t *testing.T) {
	want := []string{"token_window(trigger=0.85, keep=0.6)", "sliding_window(keep=24)", "sliding_window(keep=10)", "sliding_window(keep=4)"}
	if len(DefaultOverflowEscalation) != len(want) {
		t.Fatalf("ladder has %d rungs, want %d", len(DefaultOverflowEscalation), len(want))
	}
	for i, s := range DefaultOverflowEscalation {
		if s.String() != want[i] {
			t.Errorf("rung %d = %q, want %q", i, s.String(), want[i])
		}
	}
}

// refuseOver is a provider whose window is limit estimated tokens. It records
// the estimated size of every request it is sent.
type refuseOver struct {
	limit int
	sizes []int
}

func (r *refuseOver) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	n := estimateAt(ctx, req.Messages)
	r.sizes = append(r.sizes, n)
	if n > r.limit {
		return llm.Response{}, &llm.APIError{Class: llm.ErrContextWindow, Status: 400, Message: "prompt is too long"}
	}
	if err := Convo(req.Messages).Validate(); err != nil {
		return llm.Response{}, fmt.Errorf("the ladder sent an invalid transcript: %w", err)
	}
	return textTurn("ok"), nil
}

// TestOverflowLadderShrinksAnAgentLoop: the transcript that overflows most —
// one instruction and a long run of tool calls — has nowhere a window may cut,
// and a rung that thins it without shortening it was passed over for changing
// no message count. Every ladder below used to send the refused request once
// and give up.
func TestOverflowLadderShrinksAnAgentLoop(t *testing.T) {
	msgs := agentLoop(40, 4000) // about 40k tokens
	for name, ladder := range map[string][]Strategy{
		"the default ladder":       nil,
		"FitTokens alone":          {FitTokens(20_000, 10_000)},
		"FitTokens, then a window": {FitTokens(20_000, 10_000), SlidingWindow(4)},
		"a window alone":           {SlidingWindow(4)},
	} {
		t.Run(name, func(t *testing.T) {
			p := &refuseOver{limit: 25_000}
			_, err := llm.Chain(p, WithOverflowRecovery(ladder...)).Complete(context.Background(), llm.Request{Messages: msgs})
			if name == "a window alone" {
				// Still nothing it can do; it must say so after one request.
				if !errors.Is(err, llm.ErrContextWindow) || len(p.sizes) != 1 {
					t.Errorf("err = %v after %v, want the provider's refusal after one request", err, p.sizes)
				}
				return
			}
			if err != nil {
				t.Fatalf("err = %v after requests of %v tokens", err, p.sizes)
			}
			if last := p.sizes[len(p.sizes)-1]; len(p.sizes) < 2 || last > p.limit {
				t.Errorf("requests of %v tokens, want a refused one and then one under %d", p.sizes, p.limit)
			}
		})
	}
}

// A refusal teaches the ladder an upper bound on the window, so a token rung
// that did not shrink a request far enough is applied again, smaller — and
// the attempts are bounded, because each refusal is paid for.
func TestOverflowTokenRungConvergesAndIsBounded(t *testing.T) {
	msgs := agentLoop(40, 4000)
	ladder := []Strategy{TokenWindow(0.85, 0.6)}

	p := &refuseOver{limit: 22_000}
	if _, err := llm.Chain(p, WithOverflowRecovery(ladder...)).Complete(context.Background(), llm.Request{Messages: msgs}); err != nil {
		t.Fatalf("err = %v after %v: one rung should converge on a window it is not told", err, p.sizes)
	}
	for i := 1; i < len(p.sizes); i++ {
		if p.sizes[i] >= p.sizes[i-1] {
			t.Errorf("request %d (%d tokens) is no smaller than the one refused before it: %v", i, p.sizes[i], p.sizes)
		}
	}

	hopeless := &refuseOver{limit: 50}
	_, err := llm.Chain(hopeless, WithOverflowRecovery(ladder...)).Complete(context.Background(), llm.Request{Messages: msgs})
	if !errors.Is(err, llm.ErrContextWindow) || len(hopeless.sizes) > maxOverflowAttempts {
		t.Errorf("err = %v after %d requests, want the refusal within %d", err, len(hopeless.sizes), maxOverflowAttempts)
	}
}

// elideAll is a rung that elides every tool result, for observing whether a
// request went out with the ladder applied.
type elideAll struct{}

func (elideAll) Apply(v View) []llm.Message { return elideResults(v, v.Messages, len(v.Messages)) }
func (elideAll) String() string             { return "elide_all" }

// TestCompactionStartsTheLadderAgain: the overflow level is sticky, and it
// used to stay up after WithCompaction had replaced the transcript that drove
// it there, so every later request was still trimmed for a problem that was
// gone.
func TestCompactionStartsTheLadderAgain(t *testing.T) {
	var turns []llm.Response
	for i := range 8 {
		turns = append(turns, toolTurn(fmt.Sprintf("c%d", i), "big", `{}`))
	}
	turns = append(turns, textTurn("done"))
	cl := scripted(turns...)
	// Refuse the second call once: by then there is a tool result to elide.
	calls := 0
	refuseSecond := llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
		if calls++; calls == 2 {
			return llm.Response{}, &llm.APIError{Class: llm.ErrContextWindow, Status: 400, Message: "prompt is too long"}
		}
		return cl.Complete(ctx, req)
	})
	client := llm.Chain(refuseSecond, WithOverflowRecovery(elideAll{}))
	compactor := func(context.Context, []llm.Message) (string, error) { return "summary", nil }
	a := newAgent(t, client, []tool.Def{bigTool("big", 8000)},
		WithCompaction(6000, 2500, compactor), WithForceTerminalLastN(0))

	run := a.Start(context.Background(), Ask("do the big thing"))
	_, evs := drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	compacted := false
	for _, ev := range evs {
		if _, ok := ev.(TranscriptCompacted); ok {
			compacted = true
		}
	}
	if !compacted {
		t.Fatal("premise: the run never compacted")
	}
	elided := func(req llm.Request) bool {
		return strings.Contains(allText(req.Messages)+resultText(req.Messages), elidedPrefix)
	}
	if !elided(cl.request(t, 1)) {
		t.Fatal("premise: the request after the overflow went out with the ladder applied")
	}
	if last := cl.request(t, cl.calls()-1); elided(last) {
		t.Error("the last request was still trimmed by the ladder after compaction replaced the transcript")
	}
}

func resultText(msgs []llm.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		for _, blk := range m.Content {
			if tr, ok := blk.(llm.ToolResult); ok {
				b.WriteString(tr.Content)
			}
		}
	}
	return b.String()
}
