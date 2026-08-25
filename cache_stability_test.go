package wombat

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
)

// Prompt-cache stability, as three properties.
//
// Both providers cache on a prefix of the request: the leading bytes have to be
// identical to the previous call or the whole prompt is re-read at full price.
// Nothing reports a miss. There is no error, no warning, and no wrong answer —
// the only symptom is the bill, which means a regression here can live in the
// tree for as long as nobody happens to look at a usage dashboard.
//
// The OCaml original pinned this with a cache-stability test, and that
// test did not come across in the port. The consequence was immediate and
// went unnoticed: [SlidingWindowAt] recomputed its cut from len(msgs) on every
// call, so the surviving prefix advanced one message-pair per turn and every
// turn of a long run was a miss. These are that test, in Go, plus a fourth
// property that names the specific arithmetic that broke.
//
// Note what is NOT claimed: the default strategy is [Flat], and nothing in this
// repository wires up SlidingWindowAt, so the regression was latent rather than
// live. It was still wrong in a public API, and a caller reaching for the
// strategy whose entire reason to exist is prefix stability would have got the
// opposite.

// serialize renders messages the way a cache prefix comparison sees them: block
// by block, in order, as bytes.
func serialize(t *testing.T, msgs []llm.Message) []string {
	t.Helper()
	out := make([]string, len(msgs))
	for i, m := range msgs {
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatalf("marshal message %d: %v", i, err)
		}
		out[i] = string(b)
	}
	return out
}

// isPrefix reports whether short is a leading run of long.
func isPrefix(long, short []string) bool {
	if len(short) > len(long) {
		return false
	}
	for i, s := range short {
		if long[i] != s {
			return false
		}
	}
	return true
}

// alternating builds a legal transcript of n messages, user first.
func alternating(n int) []llm.Message {
	msgs := make([]llm.Message, 0, n)
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			msgs = append(msgs, llm.UserText(fmt.Sprintf("user %d", i)))
		} else {
			msgs = append(msgs, llm.Message{
				Role:    llm.RoleAssistant,
				Content: []llm.ContentBlock{llm.Text{Text: fmt.Sprintf("asst %d", i)}},
			})
		}
	}
	return msgs
}

// TestCacheStabilityPushTouchesOnlyTheTail: property 1.
//
// A push must not rewrite history. The exact statement has to account for
// [Convo.Append] merging a same-role push into the preceding turn — which is
// not a bug and cannot be avoided, because both providers reject two
// consecutive turns with the same role, so a same-role push has nowhere else to
// go. Merging touches the LAST turn, which is where new content belongs anyway.
//
// So the property is: everything before the last turn is byte-identical, and
// when the roles do alternate the push is purely additive.
func TestCacheStabilityPushTouchesOnlyTheTail(t *testing.T) {
	pushes := []struct {
		name      string
		fn        func(Convo) Convo
		alternate bool // true when this push's role differs from the base's last turn
	}{
		// alternating(6) ends on an assistant turn.
		{"PushUserText", func(c Convo) Convo { return c.PushUserText("next") }, true},
		{"PushToolResults", func(c Convo) Convo {
			return c.PushToolResults([]llm.ContentBlock{llm.ToolResult{ToolUseID: "t1", Content: "ok"}})
		}, true},
		{"PushAssistant merges into the assistant tail", func(c Convo) Convo {
			return c.PushAssistant([]llm.ContentBlock{llm.Text{Text: "reply"}})
		}, false},
	}

	for _, p := range pushes {
		t.Run(p.name, func(t *testing.T) {
			base := Convo(alternating(6))
			before := serialize(t, base)
			after := serialize(t, p.fn(base))

			if p.alternate {
				if len(after) != len(before)+1 {
					t.Fatalf("length = %d, want %d — an alternating push appends exactly one turn",
						len(after), len(before)+1)
				}
				if !isPrefix(after, before) {
					t.Errorf("the prior turns changed:\n before %v\n after  %v", before, after[:len(before)])
				}
				return
			}

			if len(after) != len(before) {
				t.Fatalf("length = %d, want %d — a same-role push merges", len(after), len(before))
			}
			// Everything but the tail is untouched.
			if !isPrefix(after, before[:len(before)-1]) {
				t.Errorf("a merge disturbed a turn before the tail:\n before %v\n after  %v", before, after)
			}
			if after[len(after)-1] == before[len(before)-1] {
				t.Error("the merge did not change the tail; the push was lost")
			}
		})
	}

	// The base itself must not be mutated — Convo is a value type and a push
	// that wrote through the backing array would corrupt a caller holding an
	// older snapshot, which is exactly what Run.Messages() hands out.
	base := Convo(alternating(6))
	snapshot := serialize(t, base)
	_ = base.PushUserText("x")
	_ = base.PushAssistant([]llm.ContentBlock{llm.Text{Text: "y"}})
	if got := serialize(t, base); !isPrefix(got, snapshot) || len(got) != len(snapshot) {
		t.Errorf("the base transcript was mutated by a push:\n was %v\n now %v", snapshot, got)
	}
}

// TestCacheStabilitySerializationIsDeterministic: property 2.
//
// The same message must produce the same bytes every time. Go map iteration is
// randomised per run, so a content block that serialised through a map would
// pass in development and miss the cache in production — which is exactly why
// llm.ToolSpec.InputSchema and llm.ToolUse.Input are json.RawMessage and not
// map[string]any.
func TestCacheStabilitySerializationIsDeterministic(t *testing.T) {
	msgs := []llm.Message{
		llm.UserText("hello"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.Text{Text: "let me look"},
			llm.ToolUse{ID: "t1", Name: "view_file", Input: json.RawMessage(`{"path":"/a","limit":10}`)},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "t1", Content: "contents"},
		}},
	}

	first := serialize(t, msgs)
	for i := 0; i < 64; i++ {
		if got := serialize(t, msgs); !isPrefix(got, first) || len(got) != len(first) {
			t.Fatalf("serialization %d differs from the first:\n %v\n %v", i, first, got)
		}
	}
}

// TestCacheStabilityWindowPrefixGrowsOnly: property 3, the one that broke.
//
// Stated the way the cache experiences it: whatever survived the window on turn
// N must still be a leading run of what survives on turn N+1. A window that
// merely "keeps the last K" fails this on every call — the transcript grew, so
// the survivor set slid, so the first surviving message is a different one.
func TestCacheStabilityWindowPrefixGrowsOnly(t *testing.T) {
	const (
		trigger = 20
		keep    = 10
	)
	s := SlidingWindowAt(trigger, keep)

	var (
		prev      []string
		reAnchors int
	)
	for n := 1; n <= 80; n++ {
		got := serialize(t, s.Apply(View{Messages: alternating(n)}))
		if prev != nil && !isPrefix(got, prev) {
			// A re-anchor is legal and expected; it is the point of delta. What
			// is not legal is re-anchoring on EVERY call, checked below.
			reAnchors++
			if !shrankOrMoved(prev, got) {
				t.Fatalf("n=%d: the kept suffix is neither an extension of the previous one nor a clean re-anchor", n)
			}
		}
		prev = got
	}

	// delta = trigger - keep = 10 messages per re-anchor, over the 61 calls that
	// actually trim (n = 20..80). Anything close to one re-anchor per call means
	// the anchor is not being held at all.
	const wantAtMost = 61/(trigger-keep) + 1
	if reAnchors > wantAtMost {
		t.Errorf("re-anchored %d times in 80 calls, want at most %d — the prefix is not being held",
			reAnchors, wantAtMost)
	}
	if reAnchors == 0 {
		t.Error("never re-anchored in 80 calls; the test is not exercising the window")
	}
}

// shrankOrMoved reports whether b looks like a fresh anchor rather than
// corruption. A re-anchor cuts more, so the survivor list gets shorter; a
// survivor list that grew while ceasing to extend the previous one means the
// window rewrote history, which is the failure this is looking for.
func shrankOrMoved(a, b []string) bool { return len(b) <= len(a) }

// TestCacheStabilityAnchorIsQuantized: property 4, Go-specific.
//
// Names the arithmetic directly, so a future edit to the formula fails here with
// a readable diff rather than showing up as a cost regression. The anchor must
// be constant across delta consecutive transcript lengths.
func TestCacheStabilityAnchorIsQuantized(t *testing.T) {
	const (
		trigger = 20
		keep    = 10
		delta   = trigger - keep
	)
	s := SlidingWindowAt(trigger, keep)

	// firstKept identifies the anchor: the text of the oldest surviving turn.
	firstKept := func(n int) string {
		out := s.Apply(View{Messages: alternating(n)})
		if len(out) == 0 {
			return "(empty)"
		}
		return llm.TextOf(out[0].Content)
	}

	runs := map[string][]int{}
	var order []string
	for n := trigger; n <= trigger+4*delta; n++ {
		k := firstKept(n)
		if _, seen := runs[k]; !seen {
			order = append(order, k)
		}
		runs[k] = append(runs[k], n)
	}

	if len(order) < 2 {
		t.Fatalf("the anchor never moved across %d lengths; the test is not exercising a re-anchor", 4*delta)
	}
	// Every run except possibly the first and last (which are clipped by the
	// loop bounds) must be exactly delta long.
	for _, k := range order[1 : len(order)-1] {
		if got := len(runs[k]); got != delta {
			t.Errorf("anchor %q held for %d lengths %v, want exactly %d", k, got, runs[k], delta)
		}
	}

	// And each anchor must own a CONTIGUOUS block of lengths. Summing the run
	// sizes proves nothing — one entry is appended per iteration, so the total
	// is the loop bound whatever the anchor does; an earlier version of this
	// check asserted exactly that and stayed green when the quantization was
	// removed entirely.
	for _, k := range order {
		ns := runs[k]
		for i := 1; i < len(ns); i++ {
			if ns[i] != ns[i-1]+1 {
				t.Errorf("anchor %q came back after moving away: lengths %v", k, ns)
				break
			}
		}
	}
}

// TestCacheStabilityDegenerateWindowNeverTrims: keepRecent >= triggerAt asks for
// "trim once we reach N, but always keep more than N", and the only coherent
// answer to a contradiction is to do nothing.
func TestCacheStabilityDegenerateWindowNeverTrims(t *testing.T) {
	for _, tc := range []struct{ trigger, keep int }{{10, 10}, {10, 11}, {10, 40}} {
		msgs := alternating(60)
		got := SlidingWindowAt(tc.trigger, tc.keep).Apply(View{Messages: msgs})
		if len(got) != len(msgs) {
			t.Errorf("SlidingWindowAt(%d, %d) trimmed %d to %d, want untouched",
				tc.trigger, tc.keep, len(msgs), len(got))
		}
	}
}

// TestSlidingWindowStillKeepsAtLeastKeepRecent: the quantized anchor must not
// weaken the floor. want is floored to a multiple of delta, which only ever
// makes it smaller, so at least keepRecent messages always survive.
func TestSlidingWindowStillKeepsAtLeastKeepRecent(t *testing.T) {
	const keep = 10
	s := SlidingWindowAt(20, keep)
	for n := 1; n <= 200; n++ {
		if got := len(s.Apply(View{Messages: alternating(n)})); got < min(n, keep) {
			t.Fatalf("n=%d kept %d, want at least %d", n, got, min(n, keep))
		}
	}
}

// withToolLoop grows a transcript by one full tool round trip: an assistant
// turn that calls a tool, then the user turn carrying the result.
//
// A user turn holding a tool_result is NOT a legal cut point, so this is the
// shape that makes SafeCut move the cut backward — and it is the shape a real
// long run actually has. A window tested only against plain text never
// exercises the interaction at all, because every user turn is a legal cut and
// SafeCut returns exactly what it was asked for.
func withToolLoop(base []llm.Message, rounds int) []llm.Message {
	out := append([]llm.Message(nil), base...)
	for i := 0; i < rounds; i++ {
		id := llm.ToolUseID(fmt.Sprintf("t%d", i))
		out = append(out,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.ToolUse{ID: id, Name: "grep_search", Input: json.RawMessage(`{"pattern":"x"}`)},
			}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{
				llm.ToolResult{ToolUseID: id, Content: fmt.Sprintf("match %d", i)},
			}},
		)
	}
	return out
}

// TestCacheStabilityHoldsThroughAToolLoop is the composition the doc claims and
// nothing tested.
//
// strategy.go argues that SafeCut needs no special handling because it is a
// pure function of msgs[0:anchor] and messages are only appended, so "a cut
// moved backward to avoid orphaning a tool_result stays moved to the same
// place". True, and worth pinning: this is where quantization and the cut
// interact, and the transcripts in the rest of this file — plain alternating
// text, where every user turn is a legal cut point — cannot reach it.
func TestCacheStabilityHoldsThroughAToolLoop(t *testing.T) {
	const (
		trigger = 20
		keep    = 10
		delta   = trigger - keep
	)
	s := SlidingWindowAt(trigger, keep)

	// One plain user turn, then nothing but tool round trips, so most candidate
	// cut points are illegal and SafeCut has to walk back.
	base := alternating(1)

	var (
		prev      []string
		anchors   []string
		lastFirst string
	)
	for rounds := 1; rounds <= 40; rounds++ {
		msgs := withToolLoop(base, rounds)
		if err := Convo(msgs).Validate(); err != nil {
			t.Fatalf("rounds=%d: the fixture itself is invalid: %v", rounds, err)
		}

		kept := s.Apply(View{Messages: msgs})

		// Whatever survives must still be sendable — that is the constraint
		// SafeCut exists for, and quantization must not break it.
		if err := Convo(kept).Validate(); err != nil {
			t.Fatalf("rounds=%d: the window produced an invalid transcript: %v", rounds, err)
		}
		if len(kept) < min(len(msgs), keep) {
			t.Errorf("rounds=%d: kept %d, want at least %d", rounds, len(kept), keep)
		}

		got := serialize(t, kept)
		if prev != nil && !isPrefix(got, prev) && len(got) > len(prev) {
			t.Errorf("rounds=%d: the survivors grew without extending the previous set", rounds)
		}
		prev = got

		if first := llm.TextOf(kept[0].Content) + string(firstToolID(kept[0])); first != lastFirst {
			anchors = append(anchors, first)
			lastFirst = first
		}
	}

	// Roughly 80 messages of growth at delta=10 gives single-figure re-anchors.
	// One per call would be the bug this file exists for.
	if len(anchors) > 16 {
		t.Errorf("re-anchored %d times over 40 rounds; the prefix is not being held", len(anchors))
	}
}

// firstToolID makes a tool-carrying turn distinguishable, since its text is
// empty and every such turn would otherwise look identical.
func firstToolID(m llm.Message) llm.ToolUseID {
	for _, b := range m.Content {
		switch v := b.(type) {
		case llm.ToolUse:
			return v.ID
		case llm.ToolResult:
			return v.ToolUseID
		}
	}
	return ""
}
