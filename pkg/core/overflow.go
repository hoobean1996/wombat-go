package core

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"sync"

	"wombat/pkg/core/llm"
)

// DefaultOverflowEscalation is the ladder [WithOverflowRecovery] climbs when
// no other is given: keep progressively less of the transcript.
//
// It starts in tokens. The run that overflows most is one instruction followed
// by a long run of tool calls, and a window has exactly one place it may cut
// such a run — before the instruction, which is no cut at all — so a ladder of
// windows alone sent the refused request back unchanged and gave up. Eliding
// old tool results works on any transcript. With no [WithContextWindow], the
// window it works to is the size of the request the provider refused: the real
// limit is below that, which is all the rung needs to know.
//
// A caller who knows their transcript can do better — dropping loaded skill
// bodies first costs far less than dropping history:
//
//	core.WithOverflowRecovery(
//	    core.DropTagged(0, 4, skill.Tag),
//	    core.TokenWindow(0.85, 0.6),
//	    core.SlidingWindow(8),
//	)
var DefaultOverflowEscalation = []Strategy{
	TokenWindow(0.85, 0.6),
	SlidingWindow(24),
	SlidingWindow(10),
	SlidingWindow(4),
}

// maxOverflowAttempts bounds the round trips one call may spend shrinking:
// each refusal is paid for and thrown away.
const maxOverflowAttempts = 8

// WithOverflowRecovery retries a request the provider rejected for exceeding
// the context window, shrinking the transcript a step further each attempt.
//
// Without it, overflow is terminal: llm.ErrContextWindow is deliberately not
// retryable — the identical request fails identically — so a long-running
// agent simply dies the first time its history outgrows the window. That is
// the wrong failure for the one error the harness is actually equipped to fix.
//
// The rungs are CUMULATIVE: level N is rungs 1 through N applied in order, so
// a semantic first rung such as [DropTagged] stays in force when the ladder
// moves on to windowing.
//
// The escalation is STICKY for the rest of the run. Once level N has been
// needed, later calls start there instead of re-discovering the overflow, and
// each rediscovery is a full round trip that is paid for and thrown away. The
// level only rises, with one exception: [WithCompaction] folding the stored
// transcript starts it again from the bottom, because the transcript that
// overflowed is gone.
//
// A rung counts as a step when it makes the request smaller in estimated
// tokens, not in messages: eliding tool results keeps every message and can
// still remove nearly all of the bytes. Every refusal also teaches the ladder
// an upper bound on the window — the real limit is below whatever was
// refused — and the token rungs work to the smaller of that and
// [WithContextWindow], so a rung that did not shrink a request far enough can
// be applied again and shrink it further.
//
// Install it OUTSIDE the ordinary retry middleware, so a transient failure is
// retried at the current level rather than escalating:
//
//	llm.Chain(leaf, llm.WithRetry(p), core.WithOverflowRecovery())
//
// It rewrites only the request in flight; the stored transcript is untouched.
func WithOverflowRecovery(escalation ...Strategy) llm.Middleware {
	if len(escalation) == 0 {
		escalation = DefaultOverflowEscalation
	}

	return func(next llm.Client) llm.Client {
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			st := overflowFrom(ctx)
			original := req.Messages

			for attempt := 1; ; attempt++ {
				level := st.level()
				req.Messages = applyLevel(ctx, st, escalation, level, original)
				if level > 0 {
					recordSent(ctx, req.Messages)
				}

				resp, err := next.Complete(ctx, req)
				if err == nil || !errors.Is(err, llm.ErrContextWindow) {
					return resp, err
				}

				sent := estimateAt(ctx, req.Messages)
				st.refuse(sent)
				if attempt >= maxOverflowAttempts {
					return resp, err
				}
				next, shrunk, ok := nextRung(ctx, st, escalation, level, req.Messages, original)
				if !ok {
					// Out of ladder, or nothing left on it removes anything.
					// Report the provider's own error rather than a synthetic
					// one: it names the actual limit, which is what an
					// operator needs.
					return resp, err
				}

				st.raise(next)
				after := estimateAt(ctx, shrunk)
				slog.WarnContext(ctx, "context window exceeded, shrinking transcript",
					"level", next,
					"strategy", escalation[next-1].String(),
					"tokens_before", sent,
					"tokens_after", after,
					"messages_before", len(req.Messages),
					"messages_after", len(shrunk))
				Emit(ctx, ContextOverflow{
					Level:          next,
					Strategy:       escalation[next-1].String(),
					MessagesBefore: len(req.Messages),
					MessagesAfter:  len(shrunk),
					TokensBefore:   sent,
					TokensAfter:    after,
				})
			}
		})
	}
}

// nextRung returns the first level, from the current one up, that makes the
// request smaller than the one just refused.
//
// Smaller in estimated tokens. Comparing message counts — which this once did
// — skipped every rung that thins a transcript without shortening it, and
// eliding tool results is exactly that: a [FitTokens] that would have cut 200
// KB to a hundred bytes was passed over as changing nothing.
//
// The current level is a candidate again because the refusal just taught the
// ladder a smaller window, which a token rung at the same level now shrinks
// to. Rungs that change nothing are skipped rather than ending the ladder: a
// no-op is the normal case for a semantic step — "drop the loaded skill
// bodies" removes nothing from a transcript that never loaded one — and
// stopping there would strand the rungs behind it.
func nextRung(ctx context.Context, st *overflowState, escalation []Strategy, level int, current, original []llm.Message) (int, []llm.Message, bool) {
	size := estimateAt(ctx, current)
	for l := max(level, 1); l <= len(escalation); l++ {
		shrunk := applyLevel(ctx, st, escalation, l, original)
		if len(shrunk) < len(current) || estimateAt(ctx, shrunk) < size {
			return l, shrunk, true
		}
	}
	return 0, nil, false
}

// estimateAt is the run's own estimate of what msgs cost — calibrated, when
// the run has seen a response — and the uncalibrated one outside a run.
func estimateAt(ctx context.Context, msgs []llm.Message) int {
	return viewFrom(ctx, msgs).EstimateTokens(msgs)
}

// applyLevel materializes the transcript at an escalation level. Level 0 is
// the transcript as the agent's own strategy already produced it; level N
// applies rungs 1 through N in order, each to the previous one's output.
//
// Cumulative, because a ladder is read as one: "drop the skill bodies, then
// keep 20, then keep 8" means every rung goes a step further than the one
// before it. Applying only rung N to the original — which is what this used to
// do — made the rungs alternatives instead of steps. Once the level reached a
// window, the skill bodies the first rung had dropped came back inside it, so a
// transcript that only fit with BOTH dropped could never be made to fit; and
// because the level is sticky, every later call of the run paid for the bodies
// again. For a ladder of windows alone the two readings agree, since a window
// of a window is just the narrower window.
//
// Each rung sees the window the ladder has learned (see [overflowState.window]).
func applyLevel(ctx context.Context, st *overflowState, escalation []Strategy, level int, msgs []llm.Message) []llm.Message {
	if level <= 0 || level > len(escalation) {
		return msgs
	}
	for _, rung := range escalation[:level] {
		v := viewFrom(ctx, msgs)
		v.Window = st.window(v.Window)
		msgs = rung.Apply(v)
	}
	return msgs
}

// ===== per-run overflow level =====

type overflowState struct {
	mu sync.Mutex
	n  int
	// refused is the smallest estimate of a request the provider refused as
	// too long: an upper bound on the window, whatever it really is. 0 until
	// the first refusal.
	refused int
}

// MarshalState is the ladder as a run suspended at the end of its slice
// carries it to the next; see Carried.
func (s *overflowState) MarshalState() (json.RawMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return json.Marshal(struct {
		N       int `json:"n,omitempty"`
		Refused int `json:"refused,omitempty"`
	}{s.n, s.refused})
}

// UnmarshalState puts back what MarshalState wrote.
func (s *overflowState) UnmarshalState(b json.RawMessage) error {
	var st struct {
		N       int `json:"n"`
		Refused int `json:"refused"`
	}
	if err := json.Unmarshal(b, &st); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.n, s.refused = st.N, st.Refused
	return nil
}

// refuse records a refused request's estimated size.
func (s *overflowState) refuse(tokens int) {
	if s == nil || tokens <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.refused == 0 || tokens < s.refused {
		s.refused = tokens
	}
}

// window is what the ladder's rungs are told the window is: the configured
// one, or what the refusals have shown, whichever is smaller. A configured
// window the provider refuses requests under is simply wrong, and the
// refusals are the better evidence.
func (s *overflowState) window(configured int) int {
	if s == nil {
		return configured
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.refused > 0 && (configured <= 0 || s.refused < configured) {
		return s.refused
	}
	return configured
}

// reset starts the ladder from the bottom again; what the refusals taught
// about the window is kept.
func (s *overflowState) reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.n = 0
}

func (s *overflowState) level() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.n
}

func (s *overflowState) raise(to int) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if to > s.n {
		s.n = to
	}
}

type overflowKey struct{}

func withOverflowState(ctx context.Context, s *overflowState) context.Context {
	return context.WithValue(ctx, overflowKey{}, s)
}

// overflowFrom returns the run's escalation level, or a throwaway when there
// is none. A throwaway still recovers within a single call; it just cannot
// remember, which is the right degradation for a client used outside a run.
func overflowFrom(ctx context.Context) *overflowState {
	if s, ok := ctx.Value(overflowKey{}).(*overflowState); ok {
		return s
	}
	return &overflowState{}
}

// ===== result metadata on the context =====
//
// The overflow middleware is an llm.Middleware: it sees a Request, not a Run,
// so it has no way to reach View.Results on its own. Publishing them on the
// context is what lets an escalation ladder start with a semantic step —
// DropTagged evicting skill bodies — instead of being limited to position.

type resultsKey struct{}

func withResults(ctx context.Context, m map[llm.ToolUseID]ResultInfo) context.Context {
	return context.WithValue(ctx, resultsKey{}, m)
}

func resultsFrom(ctx context.Context) map[llm.ToolUseID]ResultInfo {
	m, _ := ctx.Value(resultsKey{}).(map[llm.ToolUseID]ResultInfo)
	return m
}
