package core

import (
	"strconv"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
)

// toolLoopTail is a transcript whose last stretch is a tool loop: plain user
// turns early, then nothing but tool_use / tool_result pairs.
//
//	0 user   "q1"
//	1 asst   "a1"
//	2 user   "q2"        <- the only safe cut point below index 8
//	3 asst   tool_use 1
//	4 user   tool_result 1
//	5 asst   tool_use 2
//	6 user   tool_result 2
//	7 asst   tool_use 3
//	8 user   tool_result 3
func toolLoopTail() []llm.Message {
	return []llm.Message{
		userMsg("q1"),
		assistantText("a1"),
		userMsg("q2"),
		assistantUse("1", "t"),
		userResult("1", "r1"),
		assistantUse("2", "t"),
		userResult("2", "r2"),
		assistantUse("3", "t"),
		userResult("3", "r3"),
	}
}

// TestSafeCutSearchesBackward is a regression test.
//
// SafeCut must search BACKWARD from `want`. A forward search looks like the
// natural reading of "keep at most N recent messages", and it is wrong twice
// over: it can drop context the caller asked to keep, and — the failure that
// actually bit — on a transcript that ends in a long tool loop there is no
// safe boundary ahead of `want` at all, so it finds nothing, returns 0, and
// degrades to no truncation whatsoever. That is precisely the transcript for
// which truncation exists.
//
// The fixture below is that shape: index 2 is the last plain user turn, and
// everything after it is tool traffic. Forward from want=5 finds nothing.
func TestSafeCutSearchesBackward(t *testing.T) {
	msgs := toolLoopTail()

	got := SafeCut(msgs, 5)
	if got != 2 {
		t.Fatalf("SafeCut(toolLoopTail, 5) = %d, want 2 (the last plain user turn at or before 5); "+
			"0 means the search ran forward, found no boundary in the tool loop, and gave up on truncating", got)
	}

	// The contract that follows from searching backward: keepRecent is a lower
	// bound. The window keeps more than asked, never less.
	kept := len(msgs) - got
	if kept < 5 {
		t.Errorf("kept %d messages, want at least the 5 requested", kept)
	}
	if err := Convo(msgs[got:]).Validate(); err != nil {
		t.Errorf("cut transcript does not validate: %v", err)
	}
}

func TestSafeCutEdgeCases(t *testing.T) {
	msgs := toolLoopTail()

	tests := []struct {
		name string
		msgs []llm.Message
		want int
		got  int
	}{
		{name: "want <= 0", msgs: msgs, want: 0, got: SafeCut(msgs, 0)},
		{name: "negative want", msgs: msgs, want: 0, got: SafeCut(msgs, -3)},
		{name: "empty transcript", msgs: nil, want: 0, got: SafeCut(nil, 4)},
		// want beyond the end clamps to len-1 and then searches backward.
		{name: "want beyond the end", msgs: msgs, want: 2, got: SafeCut(msgs, 99)},
		// index 0 is never a cut point: cutting there removes nothing, and the
		// loop is explicitly `i > 0`.
		{name: "only the first turn is safe", msgs: []llm.Message{userMsg("a"), assistantText("b")},
			want: 0, got: SafeCut([]llm.Message{userMsg("a"), assistantText("b")}, 1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.got != tc.want {
				t.Errorf("SafeCut = %d, want %d", tc.got, tc.want)
			}
		})
	}
}

func TestFlatIsIdentity(t *testing.T) {
	msgs := toolLoopTail()
	got := Flat.Apply(View{Messages: msgs})
	if len(got) != len(msgs) {
		t.Errorf("Flat kept %d messages, want all %d", len(got), len(msgs))
	}
	if Flat.String() != "flat" {
		t.Errorf("Flat.String() = %q, want %q", Flat.String(), "flat")
	}
}

func TestSlidingWindowAlwaysProducesAValidTranscript(t *testing.T) {
	msgs := toolLoopTail()
	for keep := 1; keep <= len(msgs)+2; keep++ {
		t.Run("keep="+strconv.Itoa(keep), func(t *testing.T) {
			got := SlidingWindow(keep).Apply(View{Messages: msgs})
			if err := Convo(got).Validate(); err != nil {
				t.Fatalf("keep=%d produced an invalid transcript (%d msgs): %v", keep, len(got), err)
			}
			if len(got) < min(keep, len(msgs)) {
				t.Errorf("keep=%d kept only %d messages, want at least %d", keep, len(got), min(keep, len(msgs)))
			}
		})
	}
}

func TestSlidingWindowDegenerateCases(t *testing.T) {
	msgs := toolLoopTail()

	t.Run("keep <= 0 is identity", func(t *testing.T) {
		if got := SlidingWindow(0).Apply(View{Messages: msgs}); len(got) != len(msgs) {
			t.Errorf("got %d messages, want %d", len(got), len(msgs))
		}
	})

	t.Run("shorter than the window is identity", func(t *testing.T) {
		short := msgs[:2]
		if got := SlidingWindow(10).Apply(View{Messages: short}); len(got) != 2 {
			t.Errorf("got %d messages, want 2", len(got))
		}
	})

	// No safe boundary anywhere: sending an over-long request is recoverable,
	// an orphaned tool_result is a hard provider error.
	t.Run("no safe boundary sends everything", func(t *testing.T) {
		loop := []llm.Message{
			userMsg("q"),
			assistantUse("1", "t"), userResult("1", "r"),
			assistantUse("2", "t"), userResult("2", "r"),
		}
		got := SlidingWindow(1).Apply(View{Messages: loop})
		if len(got) != len(loop) {
			t.Errorf("got %d messages, want all %d", len(got), len(loop))
		}
	})

	t.Run("String", func(t *testing.T) {
		if got, want := SlidingWindow(8).String(), "sliding_window(keep=8)"; got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

func TestSlidingWindowAt(t *testing.T) {
	msgs := toolLoopTail() // 9 messages

	t.Run("below the trigger is identity", func(t *testing.T) {
		got := SlidingWindowAt(20, 3).Apply(View{Messages: msgs})
		if len(got) != len(msgs) {
			t.Errorf("got %d messages, want all %d (trigger not reached)", len(got), len(msgs))
		}
	})

	t.Run("at the trigger it windows", func(t *testing.T) {
		got := SlidingWindowAt(9, 3).Apply(View{Messages: msgs})
		if len(got) >= len(msgs) {
			t.Errorf("got %d messages, want fewer than %d", len(got), len(msgs))
		}
		if err := Convo(got).Validate(); err != nil {
			t.Errorf("windowed transcript does not validate: %v", err)
		}
	})

	t.Run("String", func(t *testing.T) {
		got := SlidingWindowAt(20, 4).String()
		if want := "sliding_window_at(trigger=20, keep=4)"; got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

func TestCompacted(t *testing.T) {
	msgs := toolLoopTail()
	summarize := func(dropped []llm.Message) string {
		return "earlier: " + strconv.Itoa(len(dropped)) + " messages"
	}

	t.Run("replaces the dropped prefix with a summary", func(t *testing.T) {
		got := Compacted(4, 5, summarize).Apply(View{Messages: msgs})
		if err := Convo(got).Validate(); err != nil {
			t.Fatalf("compacted transcript does not validate: %v", err)
		}
		if len(got) >= len(msgs) {
			t.Errorf("got %d messages, want fewer than %d", len(got), len(msgs))
		}
		head := llm.TextOf(got[0].Content)
		if !strings.Contains(head, "<conversation_summary>") || !strings.Contains(head, "earlier:") {
			t.Errorf("first turn: got %q, want it to carry the summary", head)
		}
		if got[0].Role != llm.RoleUser {
			t.Errorf("first turn role: got %s, want %s", got[0].Role, llm.RoleUser)
		}
	})

	t.Run("an empty summary just drops the prefix", func(t *testing.T) {
		got := Compacted(4, 5, func([]llm.Message) string { return "" }).Apply(View{Messages: msgs})
		if strings.Contains(llm.TextOf(got[0].Content), "conversation_summary") {
			t.Errorf("got a summary turn, want none")
		}
		if err := Convo(got).Validate(); err != nil {
			t.Errorf("does not validate: %v", err)
		}
	})

	t.Run("degenerate configurations are identity", func(t *testing.T) {
		cases := map[string]Strategy{
			"nil compactor":     Compacted(4, 5, nil),
			"keep <= 0":         Compacted(4, 0, summarize),
			"compactAt <= 0":    Compacted(0, 5, summarize),
			"below compactAt":   Compacted(100, 5, summarize),
			"shorter than keep": Compacted(4, 100, summarize),
		}
		for name, st := range cases {
			got := st.Apply(View{Messages: msgs})
			if len(got) != len(msgs) {
				t.Errorf("%s: got %d messages, want all %d", name, len(got), len(msgs))
			}
		}
	})

	t.Run("String", func(t *testing.T) {
		if got, want := Compacted(30, 6, summarize).String(), "compacted(at=30, keep=6)"; got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

// taggedView builds a transcript with two tool calls, one of which is tagged.
func taggedView() View {
	msgs := []llm.Message{
		userMsg("q1"),
		assistantUse("big", "load_skill"),
		userResult("big", "NINE-KILOBYTES-OF-SKILL"),
		assistantText("thanks"),
		userMsg("q2"),
		assistantUse("small", "grep"),
		userResult("small", "one line"),
		assistantText("done"),
		userMsg("q3"),
	}
	return View{
		Messages: msgs,
		Results: map[llm.ToolUseID]ResultInfo{
			"big":   {Tool: "load_skill", Tags: []string{"skill", "skill:demo"}, Bytes: 9000},
			"small": {Tool: "grep", Tags: nil, Bytes: 8},
		},
	}
}

func TestViewHasTagAndTagged(t *testing.T) {
	v := taggedView()

	tests := []struct {
		id   llm.ToolUseID
		tag  string
		want bool
	}{
		{id: "big", tag: "skill", want: true},
		{id: "big", tag: "skill:demo", want: true},
		{id: "big", tag: "other", want: false},
		{id: "small", tag: "skill", want: false},
		{id: "absent", tag: "skill", want: false},
	}
	for _, tc := range tests {
		if got := v.HasTag(tc.id, tc.tag); got != tc.want {
			t.Errorf("HasTag(%q, %q) = %v, want %v", tc.id, tc.tag, got, tc.want)
		}
	}

	t.Run("Tagged lists ids in transcript order", func(t *testing.T) {
		got := v.Tagged("skill")
		if len(got) != 1 || got[0] != "big" {
			t.Fatalf("Tagged(skill) = %v, want [big]", got)
		}
	})

	t.Run("Tagged matches any of several tags", func(t *testing.T) {
		if got := v.Tagged("nope", "skill:demo"); len(got) != 1 || got[0] != "big" {
			t.Errorf("Tagged(nope, skill:demo) = %v, want [big]", got)
		}
	})

	t.Run("no tags matches nothing", func(t *testing.T) {
		if got := v.Tagged(); len(got) != 0 {
			t.Errorf("Tagged() = %v, want none", got)
		}
	})
}

func TestDropTagged(t *testing.T) {
	v := taggedView()

	t.Run("evicts the whole pair and leaves the rest", func(t *testing.T) {
		got := DropTagged(0, 0, "skill").Apply(v)
		flat := allText(got)
		if strings.Contains(flat, "NINE-KILOBYTES-OF-SKILL") {
			t.Errorf("skill body survived: %q", flat)
		}
		if !strings.Contains(flat, "one line") {
			t.Errorf("untagged tool result was dropped too: %q", flat)
		}
		if err := Convo(got).Validate(); err != nil {
			t.Errorf("dropped transcript does not validate: %v", err)
		}
		// The tool_use half must go with the tool_result half.
		for _, m := range got {
			for _, b := range m.Content {
				if tu, ok := b.(llm.ToolUse); ok && tu.ID == "big" {
					t.Errorf("tool_use %q survived without its result", tu.ID)
				}
			}
		}
	})

	t.Run("keepRecent protects the tail", func(t *testing.T) {
		// The tagged pair sits at indices 1 and 2 of 9; keeping the last 8
		// protects everything from index 1 on.
		got := DropTagged(0, 8, "skill").Apply(v)
		if !strings.Contains(allText(got), "NINE-KILOBYTES-OF-SKILL") {
			t.Errorf("protected skill body was evicted anyway")
		}
	})

	t.Run("below the trigger is identity", func(t *testing.T) {
		got := DropTagged(100, 0, "skill").Apply(v)
		if len(got) != len(v.Messages) {
			t.Errorf("got %d messages, want all %d", len(got), len(v.Messages))
		}
	})

	t.Run("no tags or no results is identity", func(t *testing.T) {
		if got := DropTagged(0, 0).Apply(v); len(got) != len(v.Messages) {
			t.Errorf("no tags: got %d messages, want %d", len(got), len(v.Messages))
		}
		bare := View{Messages: v.Messages}
		if got := DropTagged(0, 0, "skill").Apply(bare); len(got) != len(bare.Messages) {
			t.Errorf("no results: got %d messages, want %d", len(got), len(bare.Messages))
		}
	})

	t.Run("String", func(t *testing.T) {
		got := DropTagged(4, 2, "skill", "bulk").String()
		if want := "drop_tagged(trigger=4, keep=2, tags=skill,bulk)"; got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

func TestDropPairs(t *testing.T) {
	t.Run("empty id set is identity", func(t *testing.T) {
		msgs := toolLoopTail()
		if got := DropPairs(msgs, nil); len(got) != len(msgs) {
			t.Errorf("got %d messages, want %d", len(got), len(msgs))
		}
	})

	// Surgical removal can strip the leading user turn entirely; normalize has
	// to restore "starts with a user turn" or the provider rejects the request.
	t.Run("normalizes back to a user-first alternating transcript", func(t *testing.T) {
		msgs := []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ToolResult{ToolUseID: "x", Content: "r"}}},
			assistantText("a"),
			userMsg("q"),
			assistantText("b"),
		}
		// Dropping x empties turn 0, leaving assistant-first.
		got := DropPairs(msgs, map[llm.ToolUseID]bool{"x": true})
		if len(got) == 0 {
			t.Fatalf("everything was dropped")
		}
		if got[0].Role != llm.RoleUser {
			t.Errorf("first turn role: got %s, want %s", got[0].Role, llm.RoleUser)
		}
		if err := Convo(got).Validate(); err != nil {
			t.Errorf("does not validate: %v", err)
		}
	})

	t.Run("merges adjacent same-role turns left behind", func(t *testing.T) {
		msgs := []llm.Message{
			userMsg("q"),
			assistantUse("x", "t"),
			userResult("x", "r"),
			userMsg("still user"), // deliberately un-merged input
		}
		got := DropPairs(msgs, map[llm.ToolUseID]bool{"x": true})
		if err := Convo(got).Validate(); err != nil {
			t.Fatalf("does not validate: %v", err)
		}
		if len(got) != 1 {
			t.Errorf("got %d turns, want 1 merged user turn: %v", len(got), got)
		}
	})
}

func TestSequence(t *testing.T) {
	v := taggedView()

	t.Run("applies left to right, each seeing the previous output", func(t *testing.T) {
		st := Sequence(DropTagged(0, 0, "skill"), SlidingWindow(3))
		got := st.Apply(v)
		if strings.Contains(allText(got), "NINE-KILOBYTES") {
			t.Errorf("first rung did not run")
		}
		if len(got) >= len(v.Messages) {
			t.Errorf("second rung did not run: got %d messages, want fewer than %d", len(got), len(v.Messages))
		}
		if err := Convo(got).Validate(); err != nil {
			t.Errorf("does not validate: %v", err)
		}
	})

	t.Run("empty sequence is identity", func(t *testing.T) {
		if got := Sequence().Apply(v); len(got) != len(v.Messages) {
			t.Errorf("got %d messages, want %d", len(got), len(v.Messages))
		}
	})

	t.Run("String names every rung", func(t *testing.T) {
		got := Sequence(Flat, SlidingWindow(4)).String()
		if want := "sequence(flat → sliding_window(keep=4))"; got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

func TestConvoPrepend(t *testing.T) {
	t.Run("empty convo", func(t *testing.T) {
		got := Convo(nil).prepend(userMsg("s"))
		if len(got) != 1 || llm.TextOf(got[0].Content) != "s" {
			t.Errorf("got %v, want the single prepended turn", got)
		}
	})

	t.Run("different role appends in front", func(t *testing.T) {
		c := Convo{assistantText("a")}
		got := c.prepend(userMsg("s"))
		if len(got) != 2 {
			t.Fatalf("got %d turns, want 2", len(got))
		}
		if got[0].Role != llm.RoleUser || got[1].Role != llm.RoleAssistant {
			t.Errorf("roles = %s,%s, want user,assistant", got[0].Role, got[1].Role)
		}
	})

	t.Run("same role merges without mutating the source", func(t *testing.T) {
		c := Convo{userMsg("q")}
		got := c.prepend(userMsg("s"))
		if len(got) != 1 {
			t.Fatalf("got %d turns, want 1 merged turn", len(got))
		}
		if want := "s\nq"; llm.TextOf(got[0].Content) != want {
			t.Errorf("merged = %q, want %q", llm.TextOf(got[0].Content), want)
		}
		if len(c[0].Content) != 1 {
			t.Errorf("the source turn was mutated: %d blocks, want 1", len(c[0].Content))
		}
	})
}
