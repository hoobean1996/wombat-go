package wombat

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/hoobean1996/wombat-go/llm"
)

// Compactor writes the summary that stands in for the older part of a long
// transcript. It receives the messages being folded away — the first one is
// the original task, with any earlier summary in it — and returns prose.
//
// It is called with the run's context, so its model call is budgeted, traced
// and cancelled like every other; and it can fail, which costs the run nothing
// but the attempt. [Summarize] is the usual one.
type Compactor func(ctx context.Context, older []llm.Message) (string, error)

// WithCompaction folds the oldest part of the STORED transcript into a summary
// once a request is estimated at trigger tokens or more, keeping roughly the
// newest keep tokens of the conversation verbatim:
//
//	wombat.WithCompaction(150_000, 40_000, wombat.Summarize(client, ""))
//
// It is what [Compacted] should have been. That strategy calls its compactor
// on EVERY model call, over a prefix that grows by a turn each time, with no
// context to cancel it and no way to fail: with a model behind it, each
// iteration paid for a summary of the whole history, and because each summary
// came out different, the prompt cache missed on every call after the first.
// Compaction here happens once, when the threshold is crossed, and its result
// is written into the transcript itself, so every later call — and every later
// run that resumes the transcript — reuses the same bytes.
//
// The fold keeps the task. The first message, the instruction that started the
// run, stays at the head of the transcript with the summary appended to it,
// and the conversation resumes at an assistant turn, so no tool_result loses
// the call it answers. That also means it works on the run a window cannot cut
// — one instruction and a hundred tool calls — where there is no later user
// turn to cut before.
//
// Compaction changes what [Run.Messages] returns: the summarized turns are
// gone from the transcript, by design. They were all on the event stream as
// they happened, which is where a record of the whole run belongs.
//
// A failed summary is logged and the run carries on uncompacted; the
// transcript strategy and [WithOverflowRecovery] still apply. It is not
// attempted again until the transcript has grown by half the distance between
// keep and trigger, so a summarizer that is down is not paid for every turn.
func WithCompaction(trigger, keep int, c Compactor) Option {
	return func(cfg *config) {
		if c == nil || trigger <= 0 || keep <= 0 || keep >= trigger {
			cfg.compaction = nil
			return
		}
		cfg.compaction = &compaction{trigger: trigger, keep: keep, fn: c}
	}
}

type compaction struct {
	trigger, keep int
	fn            Compactor
}

// summaryOpen and summaryClose bracket the summary in the first message. They
// are also how a later compaction finds the earlier summary, to replace it
// rather than stack a second one on top.
const (
	summaryOpen  = "<conversation_summary>\n"
	summaryClose = "\n</conversation_summary>"
)

// compact folds convo if it has outgrown the threshold, returning the
// transcript to carry on with and whether it changed.
func (r *Run) compact(ctx context.Context, convo Convo) (Convo, bool) {
	cp := r.a.cfg.compaction
	if cp == nil || len(convo) < 3 {
		return convo, false
	}
	before := r.tokens.estimate(convo)
	if before < cp.trigger || before < r.compactRetryAt {
		return convo, false
	}

	cut := compactionCut(convo, cp.keep, r.tokens)
	if cut <= 1 {
		return convo, false
	}

	summary, err := cp.fn(ctx, convo[:cut])
	summary = strings.TrimSpace(summary)
	if err == nil && summary == "" {
		err = errors.New("the compactor returned an empty summary")
	}
	if err != nil {
		if ctx.Err() != nil {
			return convo, false // the run is ending; the loop says why
		}
		r.compactRetryAt = before + (cp.trigger-cp.keep)/2
		slog.WarnContext(ctx, "wombat: compaction failed; carrying on uncompacted",
			slog.String("agent", r.a.cfg.name), slog.Any("err", err))
		return convo, false
	}

	head := llm.Message{Role: llm.RoleUser}
	for _, b := range convo[0].Content {
		if t, ok := b.(llm.Text); ok && strings.HasPrefix(t.Text, summaryOpen) {
			continue // an earlier summary: the new one covers it
		}
		head.Content = append(head.Content, b)
	}
	head.Content = append(head.Content, llm.Text{Text: summaryOpen + summary + summaryClose})

	out := make(Convo, 0, len(convo)-cut+1)
	out = append(out, head)
	out = append(out, convo[cut:]...)
	if err := out.Validate(); err != nil {
		// compactionCut only cuts before an assistant turn whose results
		// follow it, so this is a bug; refuse to store the result rather
		// than poison the transcript with it.
		slog.ErrorContext(ctx, "wombat: compaction produced an invalid transcript; discarding it",
			slog.Any("err", err))
		return convo, false
	}
	r.compactRetryAt = 0
	r.emit(TranscriptCompacted{
		Folded:       cut - 1,
		TokensBefore: before,
		TokensAfter:  r.tokens.estimate(out),
		SummaryBytes: len(summary),
	})
	return out, true
}

// compactionCut is the index the kept part starts at: an assistant turn, so the
// user turn before it — which may hold tool results — is folded with its call,
// and so the folded head, a user turn, is followed by an assistant turn as the
// roles require. It is the latest such turn that still leaves at least keep
// tokens after it, or 0 when there is none past the first exchange.
func compactionCut(convo Convo, keep int, e *tokenEstimator) int {
	base := e.estimate(nil)
	kept := 0
	for i := len(convo) - 1; i >= 2; i-- {
		kept += max(e.estimate(convo[i:i+1])-base, 0)
		if kept >= keep && convo[i].Role == llm.RoleAssistant {
			return i
		}
	}
	return 0
}

// Summarize returns a [Compactor] that asks client for the summary, with
// model "" meaning the client's default and the call tagged
// llm.PurposeSummarizer so middleware can tell it apart.
//
// The folded turns are rendered as one plain-text transcript rather than sent
// as messages: the summarizer is not continuing the conversation, and a
// transcript passed as messages would have to be a valid conversation in its
// own right, ending where the model is meant to answer. Each tool result is
// cut to its first two kilobytes on the way, because what a summary needs from
// one is what it established, and the whole rendering is bounded so a summary
// request cannot itself overflow the window.
func Summarize(client llm.Client, model string) Compactor {
	return func(ctx context.Context, older []llm.Message) (string, error) {
		resp, err := client.Complete(ctx, llm.Request{
			System:    summarizerPrompt,
			Messages:  []llm.Message{llm.UserText(renderForSummary(older))},
			Model:     model,
			MaxTokens: 4096,
			Purpose:   llm.PurposeSummarizer,
		})
		if err != nil {
			return "", fmt.Errorf("wombat: summarize: %w", err)
		}
		return llm.TextOf(resp.Content), nil
	}
}

const summarizerPrompt = "You are compacting the transcript of an AI agent's work so that it can continue " +
	"in a fresh context window. Write a summary the agent will read in place of the transcript. " +
	"Keep: the task and every constraint the user stated; decisions made and why; what was tried " +
	"and failed; files, commands, identifiers and values that matter; the current state of the " +
	"work and what remains to do. Drop pleasantries and anything superseded. Write in the second " +
	"person, as notes to the agent. Do not invent anything that is not in the transcript."

const (
	summaryResultBytes = 2 << 10
	summaryInputBytes  = 400 << 10
)

// renderForSummary writes msgs as a role-tagged plain-text transcript.
func renderForSummary(msgs []llm.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		for _, blk := range m.Content {
			switch v := blk.(type) {
			case llm.Text:
				fmt.Fprintf(&b, "[%s] %s\n\n", m.Role, v.Text)
			case llm.ToolUse:
				fmt.Fprintf(&b, "[tool call %s] %s %s\n\n", v.ID, v.Name, v.Input)
			case llm.ToolResult:
				status := "result"
				if v.IsError {
					status = "error"
				}
				content := v.Content
				if len(content) > summaryResultBytes {
					content = content[:summaryResultBytes] + fmt.Sprintf("… (%d more bytes)", len(v.Content)-summaryResultBytes)
				}
				fmt.Fprintf(&b, "[tool %s %s] %s\n\n", status, v.ToolUseID, content)
				for _, md := range v.Media {
					fmt.Fprintf(&b, "[attached %s, %d bytes]\n\n", md.MediaType, len(md.Data))
				}
			case llm.Media:
				fmt.Fprintf(&b, "[%s attached %s, %d bytes]\n\n", m.Role, v.MediaType, len(v.Data))
			}
		}
	}
	s := strings.ToValidUTF8(b.String(), "�")
	if len(s) <= summaryInputBytes {
		return s
	}
	// Keep both ends: the start has the task, the end has where things stand.
	half := summaryInputBytes / 2
	return strings.ToValidUTF8(s[:half]+"\n\n[… the middle of the transcript is omitted …]\n\n"+s[len(s)-half:], "�")
}
