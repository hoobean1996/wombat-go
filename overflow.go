package wombat

import (
	"context"
	"errors"
	"log/slog"
	"sync"

	"github.com/hoobean1996/wombat-go/llm"
)

// DefaultOverflowEscalation is the ladder [WithOverflowRecovery] climbs when
// no other is given: keep progressively less of the transcript.
//
// Windowing rather than anything cleverer, because at this point the request
// has already been rejected and the only certainty is that it must get
// smaller. A caller who knows their transcript can do better — dropping loaded
// skill bodies first costs far less than dropping history:
//
//	wombat.WithOverflowRecovery(
//	    wombat.DropTagged(0, 4, skill.Tag),
//	    wombat.SlidingWindow(20),
//	    wombat.SlidingWindow(8),
//	)
var DefaultOverflowEscalation = []Strategy{
	SlidingWindow(24),
	SlidingWindow(10),
	SlidingWindow(4),
}

// WithOverflowRecovery retries a request the provider rejected for exceeding
// the context window, shrinking the transcript a step further each attempt.
//
// Without it, overflow is terminal: llm.ErrContextWindow is deliberately not
// retryable — the identical request fails identically — so a long-running
// agent simply dies the first time its history outgrows the window. That is
// the wrong failure for the one error the harness is actually equipped to fix.
//
// The escalation is STICKY for the rest of the run. Once level N has been
// needed, later calls start there instead of re-discovering the overflow, and
// each rediscovery is a full round trip that is paid for and thrown away. The
// level only ever rises: a transcript that outgrew the window does not shrink
// on its own.
//
// Install it OUTSIDE the ordinary retry middleware, so a transient failure is
// retried at the current level rather than escalating:
//
//	llm.Chain(leaf, llm.WithRetry(p), wombat.WithOverflowRecovery())
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

			for {
				level := st.level()
				req.Messages = applyLevel(ctx, escalation, level, original)

				resp, err := next.Complete(ctx, req)
				if err == nil || !errors.Is(err, llm.ErrContextWindow) {
					return resp, err
				}

				next, shrunk, ok := nextRung(ctx, escalation, level, req.Messages, original)
				if !ok {
					// Out of ladder, or nothing left on it removes anything.
					// Report the provider's own error rather than a synthetic
					// one: it names the actual limit, which is what an
					// operator needs.
					return resp, err
				}

				st.raise(next)
				slog.WarnContext(ctx, "context window exceeded, shrinking transcript",
					"level", next,
					"strategy", escalation[next-1].String(),
					"messages_before", len(req.Messages),
					"messages_after", len(shrunk))
			}
		})
	}
}

// nextRung climbs past rungs that would change nothing, returning the first
// level that actually shortens the transcript.
//
// Skipping rather than stopping, because a no-op rung is the normal case for a
// semantic step: "drop the loaded skill bodies" removes nothing from a
// transcript that never loaded one, and treating that as the end of the ladder
// would strand the windowing rungs behind it — exactly the rungs that would
// have worked.
func nextRung(ctx context.Context, escalation []Strategy, level int, current, original []llm.Message) (int, []llm.Message, bool) {
	for l := level + 1; l <= len(escalation); l++ {
		shrunk := applyLevel(ctx, escalation, l, original)
		if len(shrunk) < len(current) {
			return l, shrunk, true
		}
	}
	return 0, nil, false
}

// applyLevel materializes the transcript at an escalation level. Level 0 is
// the transcript as the agent's own strategy already produced it.
func applyLevel(ctx context.Context, escalation []Strategy, level int, msgs []llm.Message) []llm.Message {
	if level <= 0 || level > len(escalation) {
		return msgs
	}
	return escalation[level-1].Apply(View{Messages: msgs, Results: resultsFrom(ctx)})
}

// ===== per-run overflow level =====

type overflowState struct {
	mu sync.Mutex
	n  int
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
