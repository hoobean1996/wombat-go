package core

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"wombat/pkg/core/llm"
)

// Convo is a conversation transcript.
//
// It is an ordinary slice, not an abstract type with smart constructors. Go
// cannot make illegal states unrepresentable without a package boundary per
// invariant, so this leans the other way: build freely, validate at the
// boundary. [Validate] runs before every model call, which is the only moment
// the invariants actually matter.
//
// The Push helpers maintain strict user/assistant alternation by merging into
// the trailing turn, so code that uses them cannot produce an invalid
// transcript in the first place.
type Convo []llm.Message

// Invariant failures.
var (
	ErrEmptyConvo      = errors.New("wombat: conversation is empty")
	ErrNotUserFirst    = errors.New("wombat: conversation must start with a user turn")
	ErrNotAlternating  = errors.New("wombat: conversation roles must alternate")
	ErrDanglingToolUse = errors.New("wombat: conversation ends with unanswered tool_use")
	ErrOrphanResult    = errors.New("wombat: tool_result has no matching tool_use")

	// ErrBlockInWrongRole is a tool_use in a user turn, or a tool_result or a
	// media block in an assistant turn. The block types are not
	// interchangeable: one is the model asking, the others are the harness
	// and the user answering.
	ErrBlockInWrongRole = errors.New("wombat: content block is in the wrong role")

	// ErrLateResult is a second tool_result for a call that was already
	// answered.
	//
	// Note what it is NOT: an answer that arrives two turns late never reaches
	// here, because the turn in between is already reported by
	// [ErrDanglingToolUse] — the call went unanswered when its window closed,
	// which is the more precise complaint and comes first. This sentinel exists
	// for the one shape that survives that check, a duplicate.
	ErrLateResult = errors.New("wombat: tool_result answers a call that was already answered")

	// ErrDuplicateToolUse is the same tool_use id declared twice. Ids key the
	// result lookup, so a duplicate silently makes one of the two calls
	// unaddressable.
	ErrDuplicateToolUse = errors.New("wombat: tool_use id used twice")
)

func (c Convo) clone() Convo {
	out := make(Convo, len(c))
	copy(out, c)
	return out
}

// Append adds a turn, merging into the trailing turn when the roles match so
// that alternation is preserved.
func (c Convo) Append(m llm.Message) Convo {
	if len(m.Content) == 0 {
		return c
	}
	if n := len(c); n > 0 && c[n-1].Role == m.Role {
		out := c.clone()
		merged := make([]llm.ContentBlock, 0, len(out[n-1].Content)+len(m.Content))
		merged = append(merged, out[n-1].Content...)
		merged = append(merged, m.Content...)
		out[n-1].Content = merged
		return out
	}
	return append(c.clone(), m)
}

// PushUserText appends a plain user turn.
func (c Convo) PushUserText(s string) Convo {
	return c.Append(llm.UserText(s))
}

// PushAssistant appends the model's reply verbatim.
func (c Convo) PushAssistant(blocks []llm.ContentBlock) Convo {
	return c.Append(llm.Message{Role: llm.RoleAssistant, Content: blocks})
}

// PushToolResults appends a user turn carrying tool_result blocks.
func (c Convo) PushToolResults(blocks []llm.ContentBlock) Convo {
	return c.Append(llm.Message{Role: llm.RoleUser, Content: blocks})
}

// Dangling lists tool_use ids that were never answered.
func (c Convo) Dangling() []llm.ToolUseID {
	open := map[llm.ToolUseID]bool{}
	var order []llm.ToolUseID
	for _, m := range c {
		for _, b := range m.Content {
			switch v := b.(type) {
			case llm.ToolUse:
				if !open[v.ID] {
					open[v.ID] = true
					order = append(order, v.ID)
				}
			case llm.ToolResult:
				delete(open, v.ToolUseID)
			}
		}
	}
	out := make([]llm.ToolUseID, 0, len(open))
	for _, id := range order {
		if open[id] {
			out = append(out, id)
		}
	}
	return out
}

// CloseDangling answers every unanswered tool_use with ack, then appends any
// extra blocks in the same user turn.
//
// Used when carrying a conversation forward past a pause the user abandoned:
// the provider rejects a request whose last assistant turn has an open
// tool_use, so something has to close it.
//
// It repairs a call left open by the LAST turn, and only that. A tool_use
// stranded in the middle of a transcript — the turn after it went by without an
// answer — cannot be repaired by appending, because the answer would arrive
// several turns late and [Validate] invariant 5 rejects it. That was always
// true of the provider; before invariant 5 existed this function produced a
// transcript Validate accepted and Anthropic did not, which is the worse of the
// two failures. Such a transcript has to be edited, not appended to.
func (c Convo) CloseDangling(ack string, extra ...llm.ContentBlock) Convo {
	pending := c.Dangling()
	if len(pending) == 0 && len(extra) == 0 {
		return c
	}
	blocks := make([]llm.ContentBlock, 0, len(pending)+len(extra))
	for _, id := range pending {
		blocks = append(blocks, llm.ToolResult{ToolUseID: id, Content: ack})
	}
	blocks = append(blocks, extra...)
	return c.Append(llm.Message{Role: llm.RoleUser, Content: blocks})
}

// Lookup returns the content of a previous tool_result.
//
// An errored result is reported as an error rather than returned as content:
// a caller asking for a prior observation wants the observation, and silently
// handing back a failure message is worse than saying so.
func (c Convo) Lookup(id llm.ToolUseID) (string, error) {
	for _, m := range c {
		for _, b := range m.Content {
			tr, ok := b.(llm.ToolResult)
			if !ok || tr.ToolUseID != id {
				continue
			}
			if tr.IsError {
				return "", fmt.Errorf("tool_use %s ran but failed: %s", id, tr.Content)
			}
			return tr.Content, nil
		}
	}
	return "", fmt.Errorf("no tool_result for tool_use %s in conversation", id)
}

// Validate checks the invariants both providers enforce.
//
// Six of them, and the last three were absent long enough to be worth naming,
// because the gap was invisible from inside the package: the Push helpers
// cannot produce any of these shapes, so ordinary code was safe and the tests
// that drove ordinary code all passed. The exposure was exactly the transcript
// this function exists to guard — one handed in from outside via [Continue] or
// [AnswerPause], where the caller assembled the blocks itself.
//
//  1. Not empty.
//  2. Opens with a user turn.
//  3. Roles alternate.
//  4. A tool_use rides in an assistant turn, a tool_result or media in a
//     user turn.
//  5. A tool_result answers a tool_use from the IMMEDIATELY preceding turn.
//  6. No tool_use is left unanswered, and no id is reused or answered twice.
//
// Invariant 5 is the one with teeth, and it is enforced from the asking side:
// whatever the previous turn called and this turn did not answer can never be
// answered, so it is reported here as [ErrDanglingToolUse] naming both turns
// rather than three turns later as a late result. A provider will not accept an
// answer that arrives two turns late — Anthropic returns a 400 whose body is
// prose — and nothing else in the harness would notice, because a late
// tool_result is perfectly well-formed JSON referring to a tool_use that really
// does exist.
func (c Convo) Validate() error {
	if len(c) == 0 {
		return ErrEmptyConvo
	}
	if c[0].Role != llm.RoleUser {
		return ErrNotUserFirst
	}

	seen := map[llm.ToolUseID]bool{}
	// open holds the tool_use ids declared by the PREVIOUS turn and not yet
	// answered. It is rebuilt each turn, which is what makes invariant 5 a
	// positional check rather than an existence check.
	open := map[llm.ToolUseID]bool{}

	for i, m := range c {
		if i > 0 && m.Role == c[i-1].Role {
			return fmt.Errorf("%w: turns %d and %d are both %s", ErrNotAlternating, i-1, i, m.Role)
		}

		declared := map[llm.ToolUseID]bool{}
		for _, b := range m.Content {
			switch v := b.(type) {
			case llm.ToolUse:
				if m.Role != llm.RoleAssistant {
					return fmt.Errorf("%w: tool_use %s is in a %s turn (turn %d); only the model calls tools",
						ErrBlockInWrongRole, v.ID, m.Role, i)
				}
				if seen[v.ID] {
					return fmt.Errorf("%w: %s (turn %d)", ErrDuplicateToolUse, v.ID, i)
				}
				seen[v.ID] = true
				declared[v.ID] = true

			case llm.ToolResult:
				if m.Role != llm.RoleUser {
					return fmt.Errorf("%w: tool_result for %s is in a %s turn (turn %d); only the harness answers tools",
						ErrBlockInWrongRole, v.ToolUseID, m.Role, i)
				}
				if !seen[v.ToolUseID] {
					return fmt.Errorf("%w: %s", ErrOrphanResult, v.ToolUseID)
				}
				if !open[v.ToolUseID] {
					return fmt.Errorf("%w: %s, again at turn %d", ErrLateResult, v.ToolUseID, i)
				}
				delete(open, v.ToolUseID)

			case llm.Media:
				if m.Role != llm.RoleUser {
					return fmt.Errorf("%w: a %s media block is in a %s turn (turn %d); media is something the model reads, not something it sends",
						ErrBlockInWrongRole, v.MediaType, m.Role, i)
				}
			}
		}

		// Anything the previous turn asked and this turn did not answer can
		// never be answered now, because invariant 5 closes the window.
		if len(open) > 0 {
			return fmt.Errorf("%w: %s (asked at turn %d, unanswered at turn %d)",
				ErrDanglingToolUse, joinIDs(open), i-1, i)
		}
		open = declared
	}

	if len(open) > 0 {
		return fmt.Errorf("%w: %s", ErrDanglingToolUse, joinIDs(open))
	}
	return nil
}

// joinIDs renders an id set for a diagnostic, sorted so two runs of the same
// broken transcript produce the same message.
func joinIDs(ids map[llm.ToolUseID]bool) string {
	out := make([]string, 0, len(ids))
	for id := range ids {
		out = append(out, string(id))
	}
	slices.Sort(out)
	return strings.Join(out, ", ")
}
