package core

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
)

func TestConvoAppendMergesAndDoesNotAlias(t *testing.T) {
	t.Run("empty content is ignored", func(t *testing.T) {
		var c Convo
		if got := c.Append(llm.Message{Role: llm.RoleUser}); len(got) != 0 {
			t.Fatalf("Append(empty) grew the convo: got %d turns, want 0", len(got))
		}
	})

	t.Run("same role merges into the trailing turn", func(t *testing.T) {
		c := Convo{}.PushUserText("hi").PushUserText("and also")
		if len(c) != 1 {
			t.Fatalf("got %d turns, want 1 merged turn", len(c))
		}
		if len(c[0].Content) != 2 {
			t.Fatalf("got %d blocks in the merged turn, want 2", len(c[0].Content))
		}
		if got, want := llm.TextOf(c[0].Content), "hi\nand also"; got != want {
			t.Errorf("merged text: got %q, want %q", got, want)
		}
	})

	t.Run("different role appends", func(t *testing.T) {
		c := Convo{}.PushUserText("hi").PushAssistant([]llm.ContentBlock{llm.Text{Text: "hello"}})
		if len(c) != 2 {
			t.Fatalf("got %d turns, want 2", len(c))
		}
	})

	// Push returns a new value. If it aliased, a caller holding the earlier
	// transcript — Run.Messages does exactly that — would see it mutate under
	// them.
	t.Run("push does not alias the receiver", func(t *testing.T) {
		base := Convo{}.PushUserText("hi")
		a := base.PushAssistant([]llm.ContentBlock{llm.Text{Text: "A"}})
		b := base.PushAssistant([]llm.ContentBlock{llm.Text{Text: "B"}})

		if len(base) != 1 {
			t.Errorf("base grew: got %d turns, want 1", len(base))
		}
		if got := llm.TextOf(a[1].Content); got != "A" {
			t.Errorf("branch a: got %q, want %q", got, "A")
		}
		if got := llm.TextOf(b[1].Content); got != "B" {
			t.Errorf("branch b: got %q, want %q", got, "B")
		}
	})

	// Merging into the trailing turn must copy that turn's content, not append
	// into a slice two convos share.
	t.Run("merge does not alias the receiver's blocks", func(t *testing.T) {
		base := Convo{}.PushUserText("hi")
		a := base.PushUserText("A")
		b := base.PushUserText("B")

		if got, want := llm.TextOf(a[0].Content), "hi\nA"; got != want {
			t.Errorf("branch a: got %q, want %q", got, want)
		}
		if got, want := llm.TextOf(b[0].Content), "hi\nB"; got != want {
			t.Errorf("branch b: got %q, want %q", got, want)
		}
	})
}

func TestConvoDanglingAndClose(t *testing.T) {
	c := Convo{}.
		PushUserText("hi").
		PushAssistant([]llm.ContentBlock{
			llm.ToolUse{ID: "a", Name: "t"},
			llm.ToolUse{ID: "b", Name: "t"},
		})

	if got, want := len(c.Dangling()), 2; got != want {
		t.Fatalf("Dangling: got %d ids %v, want %d", got, c.Dangling(), want)
	}
	if got, want := c.Dangling(), []llm.ToolUseID{"a", "b"}; got[0] != want[0] || got[1] != want[1] {
		t.Errorf("Dangling order: got %v, want %v (transcript order)", got, want)
	}

	closed := c.PushToolResults([]llm.ContentBlock{
		llm.ToolResult{ToolUseID: "a", Content: "42"},
		llm.ToolResult{ToolUseID: "b", Content: "43"},
	})
	if got := closed.Dangling(); len(got) != 0 {
		t.Errorf("after answering both: got dangling %v, want none", got)
	}
	if err := closed.Validate(); err != nil {
		t.Errorf("Validate: got %v, want nil", err)
	}

	t.Run("CloseDangling answers every open use", func(t *testing.T) {
		out := c.CloseDangling("(cancelled)", llm.Text{Text: "never mind"})
		if err := out.Validate(); err != nil {
			t.Fatalf("Validate after CloseDangling: got %v, want nil", err)
		}
		if got := allText(out); !strings.Contains(got, "(cancelled)") ||
			!strings.Contains(got, "never mind") {
			t.Errorf("CloseDangling content: got %q, want the ack and the extra block", got)
		}
	})

	t.Run("CloseDangling on a clean convo with no extras is a no-op", func(t *testing.T) {
		clean := Convo{}.PushUserText("hi")
		out := clean.CloseDangling("(cancelled)")
		if len(out) != len(clean) {
			t.Errorf("got %d turns, want %d (unchanged)", len(out), len(clean))
		}
	})

	t.Run("CloseDangling with extras only still appends them", func(t *testing.T) {
		clean := Convo{}.PushUserText("hi").
			PushAssistant([]llm.ContentBlock{llm.Text{Text: "yes"}})
		out := clean.CloseDangling("(cancelled)", llm.Text{Text: "next"})
		if len(out) != 3 {
			t.Fatalf("got %d turns, want 3", len(out))
		}
		if err := out.Validate(); err != nil {
			t.Errorf("Validate: got %v, want nil", err)
		}
	})
}

func TestConvoLookup(t *testing.T) {
	c := Convo{}.
		PushUserText("hi").
		PushAssistant([]llm.ContentBlock{
			llm.ToolUse{ID: "ok", Name: "t"},
			llm.ToolUse{ID: "bad", Name: "t"},
		}).
		PushToolResults([]llm.ContentBlock{
			llm.ToolResult{ToolUseID: "ok", Content: "42"},
			llm.ToolResult{ToolUseID: "bad", Content: "no such file", IsError: true},
		})

	tests := []struct {
		name    string
		id      llm.ToolUseID
		want    string
		wantErr bool
		errFrag string
	}{
		{name: "present", id: "ok", want: "42"},
		{name: "errored result reports the failure", id: "bad", wantErr: true, errFrag: "no such file"},
		{name: "missing", id: "nope", wantErr: true, errFrag: "no tool_result"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := c.Lookup(tc.id)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Lookup(%q): got (%q, nil), want an error", tc.id, got)
				}
				if !strings.Contains(err.Error(), tc.errFrag) {
					t.Errorf("Lookup(%q) error: got %q, want it to mention %q", tc.id, err, tc.errFrag)
				}
				if got != "" {
					t.Errorf("Lookup(%q) content on error: got %q, want %q", tc.id, got, "")
				}
				return
			}
			if err != nil {
				t.Fatalf("Lookup(%q): got error %v, want nil", tc.id, err)
			}
			if got != tc.want {
				t.Errorf("Lookup(%q): got %q, want %q", tc.id, got, tc.want)
			}
		})
	}
}

// TestConvoValidateFailureModes covers all five invariant failures plus the
// happy path. Validate runs before every model call, so each of these is a
// provider rejection that never reaches the wire.
func TestConvoValidateFailureModes(t *testing.T) {
	tests := []struct {
		name string
		c    Convo
		want error
	}{
		{
			name: "empty",
			c:    nil,
			want: ErrEmptyConvo,
		},
		{
			name: "assistant first",
			c:    Convo{assistantText("x")},
			want: ErrNotUserFirst,
		},
		{
			name: "two user turns in a row",
			c:    Convo{userMsg("a"), userMsg("b")},
			want: ErrNotAlternating,
		},
		{
			name: "two assistant turns in a row",
			c:    Convo{userMsg("a"), assistantText("b"), assistantText("c")},
			want: ErrNotAlternating,
		},
		{
			name: "orphan tool_result",
			c:    Convo{userResult("z", "?")},
			want: ErrOrphanResult,
		},
		{
			name: "tool_result before its tool_use",
			c: Convo{
				userMsg("a"), assistantText("b"),
				userResult("later", "?"),
				assistantUse("later", "t"),
			},
			want: ErrOrphanResult,
		},
		{
			name: "dangling tool_use at the end",
			c:    Convo{userMsg("a"), assistantUse("a", "t")},
			want: ErrDanglingToolUse,
		},
		{
			name: "valid tool round trip",
			c:    Convo{userMsg("a"), assistantUse("a", "t"), userResult("a", "r")},
			want: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.c.Validate()
			if tc.want == nil {
				if err != nil {
					t.Fatalf("Validate: got %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, tc.want) {
				t.Fatalf("Validate: got %v, want it to wrap %v", err, tc.want)
			}
		})
	}
}

func TestConvoValidateErrorsAreDescriptive(t *testing.T) {
	c := Convo{userMsg("a"), assistantUse("id-7", "t")}
	err := c.Validate()
	if err == nil || !strings.Contains(err.Error(), "id-7") {
		t.Errorf("dangling error: got %v, want it to name the tool_use id %q", err, "id-7")
	}

	c2 := Convo{userMsg("a"), assistantText("b"), assistantText("c")}
	err2 := c2.Validate()
	if err2 == nil || !strings.Contains(err2.Error(), "assistant") {
		t.Errorf("alternation error: got %v, want it to name the offending role", err2)
	}
}

func TestConvoJSONRoundTrip(t *testing.T) {
	c := Convo{
		userMsg("hi"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.Text{Text: "let me look"},
			llm.Thinking{Text: "hmm", Signature: "sig"},
			llm.ToolUse{ID: "a", Name: "grep", Input: json.RawMessage(`{"q":"x"}`)},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "a", Content: "found", IsError: false},
		}},
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("fixture is not a valid convo: %v", err)
	}

	raw, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var back Convo
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(back) != len(c) {
		t.Fatalf("round trip length: got %d, want %d (%s)", len(back), len(c), raw)
	}
	if err := back.Validate(); err != nil {
		t.Errorf("round-tripped convo does not validate: %v", err)
	}

	// Every block type must survive, including the thinking signature the
	// provider rejects the next turn without.
	th, ok := back[1].Content[1].(llm.Thinking)
	if !ok {
		t.Fatalf("block 1 of turn 1: got %T, want llm.Thinking", back[1].Content[1])
	}
	if th.Signature != "sig" {
		t.Errorf("thinking signature: got %q, want %q", th.Signature, "sig")
	}
	tu, ok := back[1].Content[2].(llm.ToolUse)
	if !ok {
		t.Fatalf("block 2 of turn 1: got %T, want llm.ToolUse", back[1].Content[2])
	}
	if string(tu.Input) != `{"q":"x"}` {
		t.Errorf("tool input: got %s, want %s", tu.Input, `{"q":"x"}`)
	}
}

// TestValidateBlockRoleAndOrdering covers the three invariants Validate did not
// have.
//
// Every case here returned nil before, and every one of them is a 400 from the
// provider. They were invisible from inside the package because the Push
// helpers cannot construct any of them — the exposure is a transcript handed in
// through Continue or AnswerPause, which is precisely what Validate guards.
func TestValidateBlockRoleAndOrdering(t *testing.T) {
	const id = llm.ToolUseID("t1")
	tu := llm.ToolUse{ID: id, Name: "x", Input: json.RawMessage(`{}`)}
	tr := llm.ToolResult{ToolUseID: id, Content: "ok"}
	asst := func(b ...llm.ContentBlock) llm.Message {
		return llm.Message{Role: llm.RoleAssistant, Content: b}
	}
	usr := func(b ...llm.ContentBlock) llm.Message {
		return llm.Message{Role: llm.RoleUser, Content: b}
	}

	tests := []struct {
		name string
		c    Convo
		want error
	}{
		{
			// Roles alternate perfectly; the only thing wrong is the distance.
			// Anthropic answers this with a 400 whose body is prose. Reported
			// from the asking side — the call went unanswered when its window
			// closed — which names both turns instead of waiting for the late
			// answer to show up.
			name: "tool_result two turns after its tool_use",
			c: Convo{
				llm.UserText("q"), asst(tu), usr(llm.Text{Text: "never mind"}),
				asst(llm.Text{Text: "ok"}), usr(tr),
			},
			want: ErrDanglingToolUse,
		},
		{
			name: "tool_result inside an assistant turn",
			c:    Convo{llm.UserText("q"), asst(tu, tr)},
			want: ErrBlockInWrongRole,
		},
		{
			name: "tool_use inside a user turn",
			c: Convo{
				llm.UserText("q"), asst(llm.Text{Text: "a"}),
				usr(llm.ToolUse{ID: "t2", Name: "x", Input: json.RawMessage(`{}`)}),
			},
			want: ErrBlockInWrongRole,
		},
		{
			// Not last, so the dangling check cannot be what catches it.
			name: "tool_use inside a user turn, mid-conversation",
			c: Convo{
				llm.UserText("q"), asst(llm.Text{Text: "a"}),
				usr(llm.ToolUse{ID: "t2", Name: "x", Input: json.RawMessage(`{}`)}),
				asst(llm.Text{Text: "b"}), llm.UserText("q2"),
			},
			want: ErrBlockInWrongRole,
		},
		{
			name: "a second answer to a call already answered",
			c: Convo{
				llm.UserText("q"), asst(tu), usr(tr),
				asst(llm.Text{Text: "ok"}),
				usr(llm.ToolResult{ToolUseID: id, Content: "again"}),
			},
			want: ErrLateResult,
		},
		{
			name: "the same tool_use id twice",
			c: Convo{
				llm.UserText("q"), asst(tu), usr(tr),
				asst(llm.ToolUse{ID: id, Name: "x", Input: json.RawMessage(`{}`)}),
				usr(llm.ToolResult{ToolUseID: id, Content: "again"}),
			},
			want: ErrDuplicateToolUse,
		},
		{
			// Two calls asked, one answered: the unanswered one can never be
			// answered now, because invariant 5 closes the window.
			name: "a batch answered only in part",
			c: Convo{
				llm.UserText("q"),
				asst(tu, llm.ToolUse{ID: "t2", Name: "y", Input: json.RawMessage(`{}`)}),
				usr(tr),
				asst(llm.Text{Text: "done"}),
			},
			want: ErrDanglingToolUse,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.c.Validate()
			if !errors.Is(err, tt.want) {
				t.Errorf("Validate() = %v, want %v", err, tt.want)
			}
		})
	}
}

// TestValidateStillAcceptsRealTranscripts is the other half: the new checks must
// not reject anything the harness actually produces. A validator that fires on
// legitimate traffic is worse than one with holes, because it takes down runs
// that were working.
func TestValidateStillAcceptsRealTranscripts(t *testing.T) {
	const a, b = llm.ToolUseID("a"), llm.ToolUseID("b")
	asst := func(bl ...llm.ContentBlock) llm.Message {
		return llm.Message{Role: llm.RoleAssistant, Content: bl}
	}
	usr := func(bl ...llm.ContentBlock) llm.Message {
		return llm.Message{Role: llm.RoleUser, Content: bl}
	}

	tests := []struct {
		name string
		c    Convo
	}{
		{"plain chat", Convo{llm.UserText("hi"), asst(llm.Text{Text: "hello"})}},
		{"one tool round trip", Convo{
			llm.UserText("q"),
			asst(llm.Text{Text: "looking"}, llm.ToolUse{ID: a, Name: "x", Input: json.RawMessage(`{}`)}),
			usr(llm.ToolResult{ToolUseID: a, Content: "ok"}),
			asst(llm.Text{Text: "done"}),
		}},
		{"a parallel batch, all answered in one turn", Convo{
			llm.UserText("q"),
			asst(
				llm.ToolUse{ID: a, Name: "x", Input: json.RawMessage(`{}`)},
				llm.ToolUse{ID: b, Name: "y", Input: json.RawMessage(`{}`)},
			),
			usr(
				llm.ToolResult{ToolUseID: a, Content: "1"},
				llm.ToolResult{ToolUseID: b, Content: "2"},
			),
			asst(llm.Text{Text: "done"}),
		}},
		{"results answered out of order within the turn", Convo{
			llm.UserText("q"),
			asst(
				llm.ToolUse{ID: a, Name: "x", Input: json.RawMessage(`{}`)},
				llm.ToolUse{ID: b, Name: "y", Input: json.RawMessage(`{}`)},
			),
			usr(
				llm.ToolResult{ToolUseID: b, Content: "2"},
				llm.ToolResult{ToolUseID: a, Content: "1"},
			),
		}},
		{"a nudge stapled onto the results turn", Convo{
			llm.UserText("q"),
			asst(llm.ToolUse{ID: a, Name: "x", Input: json.RawMessage(`{}`)}),
			usr(llm.ToolResult{ToolUseID: a, Content: "ok"}, llm.Text{Text: "you have 2 turns left"}),
		}},
		{"thinking blocks alongside a call", Convo{
			llm.UserText("q"),
			asst(llm.Thinking{Text: "hmm"}, llm.ToolUse{ID: a, Name: "x", Input: json.RawMessage(`{}`)}),
			usr(llm.ToolResult{ToolUseID: a, Content: "ok"}),
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.c.Validate(); err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
		})
	}
}

// TestCloseDanglingStillValidates: the repair path has to produce something the
// stricter validator accepts, or resuming an abandoned pause is broken.
func TestCloseDanglingStillValidates(t *testing.T) {
	c := Convo{
		llm.UserText("deploy it"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.ToolUse{ID: "ask1", Name: "ask_user", Input: json.RawMessage(`{"question":"which env?"}`)},
		}},
	}
	if err := c.Validate(); !errors.Is(err, ErrDanglingToolUse) {
		t.Fatalf("Validate() = %v, want ErrDanglingToolUse before repair", err)
	}
	repaired := c.CloseDangling("(cancelled)", llm.Text{Text: "actually, do something else"})
	if err := repaired.Validate(); err != nil {
		t.Errorf("Validate() after CloseDangling = %v, want nil", err)
	}
}

func TestValidateRejectsMediaFromTheModel(t *testing.T) {
	c := Convo{
		llm.UserText("draw me something"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.Media{MediaType: "image/png", Data: []byte{1}}}},
	}
	if err := c.Validate(); !errors.Is(err, ErrBlockInWrongRole) {
		t.Errorf("Validate = %v, want ErrBlockInWrongRole for media in an assistant turn", err)
	}
	ok := Convo{{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.Text{Text: "look"}, llm.Media{MediaType: "image/png", Data: []byte{1}}}}}
	if err := ok.Validate(); err != nil {
		t.Errorf("Validate = %v, want media in a user turn accepted", err)
	}
}
