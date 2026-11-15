package wombat

import (
	"fmt"
	"strings"

	"github.com/hoobean1996/wombat-go/llm"
)

// FitTokens keeps a request under limit input tokens, as estimated by
// [View.EstimateTokens], by thinning out the OLD part of the transcript until
// what is left is back near target.
//
// It is the strategy for the run the message-count windows cannot help: one
// instruction followed by a hundred tool calls. A window cuts only before a
// fresh user instruction — anywhere else would orphan a tool_result — and such
// a run has exactly one, at the very start, so [SlidingWindow] has nowhere to
// cut and sends everything. Counting messages was the wrong unit anyway: one
// tool result can be a line of grep output or sixty kilobytes of a log, and a
// window of twenty messages is either of those twenty times.
//
// So the cut is made in tokens, and made inside the exchanges rather than
// around them:
//
//  1. The old tool results are elided: each keeps its place, its tool_use and
//     a note saying what was there and how to get it back, and gives up its
//     body. The model keeps the record of what it did and loses only the
//     bytes it already read — almost always the bulk of an agent's
//     transcript.
//  2. If that is still not enough, the old exchanges — an assistant turn of
//     tool calls and the turn of results answering it — are dropped whole.
//
// The first message, which is the task, is never touched; neither is anything
// recent. Recent means the newest part of the transcript worth about target
// tokens, and old is everything before it.
//
// # The boundary moves in steps
//
// Both providers cache on the request's prefix, and a boundary recomputed
// every turn would edit the prefix every turn: every call a cache miss. So it
// is quantized the way [SlidingWindowAt]'s anchor is, but on token mass: the
// old part grows in steps of limit-target tokens, the boundary stays put
// between steps, and each step buys that many tokens of cache hits. The
// request breathes between target and limit, which is the slack the
// arithmetic needs — FitTokens(150_000, 100_000) can send 149,999 tokens.
//
// When nothing old can be thinned — a transcript of one enormous turn — the
// request goes out as it is, and [WithOverflowRecovery] is what is left.
func FitTokens(limit, target int) Strategy {
	return fitTokens{limit: limit, target: target}
}

// TokenWindow is [FitTokens] with its bounds given as fractions of the model's
// context window, the one set with [WithContextWindow]:
//
//	wombat.WithContextWindow(200_000),
//	wombat.WithStrategy(wombat.TokenWindow(0.7, 0.5)),
//
// trims a request once it is estimated at 70% of the window, back to about
// 50%. With no window configured it does nothing at all, because a fraction of
// an unknown is not a number.
func TokenWindow(trigger, keep float64) Strategy {
	return fitTokens{trigger: trigger, keep: keep}
}

type fitTokens struct {
	limit, target int
	trigger, keep float64
}

func (f fitTokens) bounds(v View) (limit, target int, ok bool) {
	if f.trigger > 0 || f.keep > 0 {
		if v.Window <= 0 {
			return 0, 0, false
		}
		limit, target = int(f.trigger*float64(v.Window)), int(f.keep*float64(v.Window))
	} else {
		limit, target = f.limit, f.target
	}
	return limit, target, target > 0 && limit > target
}

func (f fitTokens) Apply(v View) []llm.Message {
	limit, target, ok := f.bounds(v)
	msgs := v.Messages
	if !ok || len(msgs) < 3 || v.EstimateTokens(msgs) < limit {
		return msgs
	}

	// The old part is shed in whole steps. Eliding leaves a note behind for
	// every result, so the first step that should suffice sometimes does not,
	// and the next one is tried: each candidate is still a multiple of the
	// step, which is what keeps the boundary from moving every turn.
	m := newMasses(v, msgs)
	step := limit - target
	for shed := (m.total - target) / step * step; ; shed += step {
		b := m.boundary(shed)
		if b <= 1 {
			return msgs
		}
		out := elideResults(v, msgs, b)
		if v.EstimateTokens(out) < limit {
			return out
		}
		if b >= len(msgs)-1 {
			return dropExchanges(out, b)
		}
	}
}

func (f fitTokens) String() string {
	if f.trigger > 0 || f.keep > 0 {
		return fmt.Sprintf("token_window(trigger=%g, keep=%g)", f.trigger, f.keep)
	}
	return fmt.Sprintf("fit_tokens(limit=%d, target=%d)", f.limit, f.target)
}

// masses is the estimated token mass of each message, for placing the
// boundary between the old part of a transcript and the recent one.
type masses struct {
	per   []int
	total int
}

func newMasses(v View, msgs []llm.Message) masses {
	base := v.EstimateTokens(nil)
	m := masses{per: make([]int, len(msgs)), total: base}
	for i := range msgs {
		m.per[i] = max(v.EstimateTokens(msgs[i:i+1])-base, 0)
		m.total += m.per[i]
	}
	return m
}

// boundary is the index before which the transcript is old when shed tokens'
// worth of it is to go: the first i at which the mass of msgs[:i] reaches
// shed. It depends only on the prefix's own mass and on shed, so for a given
// shed it stays where it is however the transcript grows after it. It is never
// past the last message, which is the one the model is answering.
func (m masses) boundary(shed int) int {
	if shed <= 0 {
		return 0
	}
	cum := 0
	for i := range m.per[:len(m.per)-1] {
		if cum >= shed {
			return i
		}
		cum += m.per[i]
	}
	return len(m.per) - 1
}

// elidedPrefix starts every elided tool result. The agent loop reads it too:
// an elided result is an evicted one, so a [tool.Reconciler] is told it is gone
// — a skill whose body was elided is retired, not left active with its
// knowledge removed.
const elidedPrefix = "[elided to save context: "

// elideResults replaces the body of every tool result before b, after the
// first message, with a note. The structure is untouched: every tool_use keeps
// its tool_result, so the transcript validates exactly as it did.
func elideResults(v View, msgs []llm.Message, b int) []llm.Message {
	out := make([]llm.Message, len(msgs))
	copy(out, msgs)
	for i := 1; i < b; i++ {
		m := msgs[i]
		if m.Role != llm.RoleUser {
			continue
		}
		var blocks []llm.ContentBlock
		for j, blk := range m.Content {
			tr, ok := blk.(llm.ToolResult)
			if !ok || strings.HasPrefix(tr.Content, elidedPrefix) {
				continue
			}
			if blocks == nil {
				blocks = append([]llm.ContentBlock(nil), m.Content...)
			}
			blocks[j] = llm.ToolResult{ToolUseID: tr.ToolUseID, IsError: tr.IsError, Content: elisionNote(v, tr)}
		}
		if blocks != nil {
			out[i] = llm.Message{Role: m.Role, Content: blocks}
		}
	}
	return out
}

func elisionNote(v View, tr llm.ToolResult) string {
	size := len(tr.Content)
	for _, m := range tr.Media {
		size += len(m.Data)
	}
	what := "this result"
	if info, ok := v.Results[tr.ToolUseID]; ok && info.Tool != "" {
		what = "this " + info.Tool + " result"
	}
	return fmt.Sprintf("%s%s (%d bytes) is no longer shown; call the tool again if you need it]", elidedPrefix, what, size)
}

// dropExchanges removes, before b and after the first message, every
// assistant turn whose following user turn holds nothing but its tool results —
// the pair together, so the roles still alternate and every remaining result
// still answers the turn right before it. A user turn that also says something
// is an instruction and is kept, and with it the turn it answers.
func dropExchanges(msgs []llm.Message, b int) []llm.Message {
	out := []llm.Message{msgs[0]}
	i := 1
	for ; i+1 < len(msgs) && i+1 < b; i += 2 {
		if msgs[i].Role == llm.RoleAssistant && onlyResults(msgs[i+1]) {
			continue
		}
		out = append(out, msgs[i], msgs[i+1])
	}
	return append(out, msgs[i:]...)
}

func onlyResults(m llm.Message) bool {
	if m.Role != llm.RoleUser || len(m.Content) == 0 {
		return false
	}
	for _, b := range m.Content {
		if _, ok := b.(llm.ToolResult); !ok {
			return false
		}
	}
	return true
}
