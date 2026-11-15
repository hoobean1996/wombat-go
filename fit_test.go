package wombat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// ===== the estimator =====

func TestEstimatorCalibratesAndIsSticky(t *testing.T) {
	e := newTokenEstimator()
	msgs := []llm.Message{llm.UserText(strings.Repeat("x", 996))} // 1000 bytes with framing
	if got := e.estimate(msgs); got != 250 {
		t.Fatalf("uncalibrated estimate = %d, want 250 at four bytes a token", got)
	}

	e.observe(msgs, 0, 500) // this run's text costs two bytes a token
	if got := e.estimate(msgs); got != 500 {
		t.Errorf("calibrated estimate = %d, want 500", got)
	}

	// Within a tenth of the current rate: no change, so boundaries hold still.
	e.observe(msgs, 0, 530)
	if got := e.estimate(msgs); got != 500 {
		t.Errorf("a 6%% wobble moved the estimate to %d", got)
	}
	// Past it: follow the measurement.
	e.observe(msgs, 0, 700)
	if got := e.estimate(msgs); got != 700 {
		t.Errorf("a 40%% change left the estimate at %d, want 700", got)
	}

	// A call that reported no usage teaches nothing.
	e.observe(msgs, 0, 0)
	if got := e.estimate(msgs); got != 700 {
		t.Errorf("an empty usage changed the estimate to %d", got)
	}
}

// An image is priced by what the provider charges for it, not by its bytes:
// five megabytes at four bytes a token would be a million tokens.
func TestEstimatorPricesMediaFlat(t *testing.T) {
	big := llm.Media{MediaType: "image/png", Data: make([]byte, 5<<20)}
	msgs := []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{big}}}
	if got := newTokenEstimator().estimate(msgs); got < imageTokens || got > imageTokens+10 {
		t.Errorf("estimate = %d, want about %d", got, imageTokens)
	}
}

func TestUsagePromptTokens(t *testing.T) {
	u := llm.Usage{InputTokens: 10, CacheReadTokens: 100, CacheWriteTokens: 5, OutputTokens: 7}
	if got := u.PromptTokens(); got != 115 {
		t.Errorf("PromptTokens = %d, want 115", got)
	}
}

// ===== FitTokens =====

// agentLoop is the transcript windows cannot cut: one instruction and then n
// exchanges, each result size bytes long.
func agentLoop(n, size int) []llm.Message {
	msgs := []llm.Message{llm.UserText("fix the failing test in pkg/foo")}
	for i := range n {
		id := llm.ToolUseID(fmt.Sprintf("t%d", i))
		msgs = append(msgs,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.Text{Text: fmt.Sprintf("step %d", i)},
				llm.ToolUse{ID: id, Name: "grep", Input: json.RawMessage(`{"q":"x"}`)},
			}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{
				llm.ToolResult{ToolUseID: id, Content: strings.Repeat("r", size)},
			}},
		)
	}
	return msgs
}

func TestSlidingWindowCannotCutAnAgentLoop(t *testing.T) {
	// The premise of FitTokens: one instruction, then nothing but exchanges.
	msgs := agentLoop(40, 4000)
	if got := SlidingWindow(10).Apply(View{Messages: msgs}); len(got) != len(msgs) {
		t.Fatalf("SlidingWindow cut an agent loop to %d messages; the test's premise is wrong", len(got))
	}
}

func TestFitTokensElidesOldResultsAndKeepsTheTask(t *testing.T) {
	msgs := agentLoop(40, 4000) // ~40k tokens at four bytes a token
	v := View{Messages: msgs}
	got := FitTokens(20_000, 10_000).Apply(v)

	if err := Convo(got).Validate(); err != nil {
		t.Fatalf("FitTokens broke the transcript: %v", err)
	}
	if !reflect.DeepEqual(got[0], msgs[0]) {
		t.Error("the task was changed")
	}
	if len(got) != len(msgs) {
		t.Errorf("len = %d, want %d: elision alone should have been enough", len(got), len(msgs))
	}
	if est := v.EstimateTokens(got); est >= 20_000 {
		t.Errorf("estimate after = %d, want under the 20000 limit", est)
	}
	last := got[len(got)-1].Content[0].(llm.ToolResult)
	if strings.HasPrefix(last.Content, elidedPrefix) {
		t.Error("the newest result was elided")
	}
	first := got[2].Content[0].(llm.ToolResult)
	if !strings.HasPrefix(first.Content, elidedPrefix) || !strings.Contains(first.Content, "call the tool again") {
		t.Errorf("the oldest result was not elided with a note: %q", first.Content)
	}
}

func TestFitTokensLeavesASmallTranscriptAlone(t *testing.T) {
	msgs := agentLoop(3, 100)
	if got := FitTokens(20_000, 10_000).Apply(View{Messages: msgs}); !reflect.DeepEqual(got, msgs) {
		t.Error("a transcript under the limit was changed")
	}
}

// The boundary is quantized: as the transcript grows by an exchange at a time,
// what is sent of the old part stays byte-identical until the next step, which
// is what keeps the provider's prompt cache hitting.
func TestFitTokensHoldsThePrefixStillBetweenSteps(t *testing.T) {
	s := FitTokens(20_000, 10_000)
	full := agentLoop(80, 4000)
	changes := 0
	var prev []llm.Message
	for n := 40; n <= 80; n++ {
		msgs := full[:2*n+1]
		got := s.Apply(View{Messages: msgs})
		if prev != nil && !reflect.DeepEqual(got[:len(prev)-2], prev[:len(prev)-2]) {
			changes++
		}
		prev = got
	}
	// 40 exchanges of ~1000 tokens with a 10000-token step: about four.
	if changes == 0 || changes > 6 {
		t.Errorf("the sent prefix changed %d times over 40 growth steps, want a handful", changes)
	}
}

func TestFitTokensDropsExchangesWhenElisionIsNotEnough(t *testing.T) {
	// Big assistant turns, tiny results: eliding saves nothing.
	msgs := []llm.Message{llm.UserText("task")}
	for i := range 30 {
		id := llm.ToolUseID(fmt.Sprintf("t%d", i))
		msgs = append(msgs,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.Text{Text: strings.Repeat("thinking out loud ", 250)},
				llm.ToolUse{ID: id, Name: "ls", Input: json.RawMessage(`{}`)},
			}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ToolResult{ToolUseID: id, Content: "ok"}}},
		)
	}
	got := FitTokens(20_000, 10_000).Apply(View{Messages: msgs})
	if err := Convo(got).Validate(); err != nil {
		t.Fatalf("FitTokens broke the transcript: %v", err)
	}
	if len(got) >= len(msgs) {
		t.Fatalf("len = %d, want old exchanges dropped from %d", len(got), len(msgs))
	}
	if !reflect.DeepEqual(got[0], msgs[0]) {
		t.Error("the task was dropped")
	}
}

// A user turn that says something is an instruction, and it is never dropped
// with the exchanges around it.
func TestFitTokensKeepsAnInstructionInTheOldPart(t *testing.T) {
	msgs := agentLoop(30, 10)
	for i := range msgs {
		if m := msgs[i]; m.Role == llm.RoleAssistant {
			msgs[i].Content = append([]llm.ContentBlock{llm.Text{Text: strings.Repeat("y", 4000)}}, m.Content[1:]...)
		}
	}
	msgs[4].Content = append(msgs[4].Content, llm.Text{Text: "ALSO: keep the public API unchanged"})
	got := FitTokens(20_000, 10_000).Apply(View{Messages: msgs})
	if err := Convo(got).Validate(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(allText(got), "keep the public API unchanged") {
		t.Error("an instruction in the old part was dropped")
	}
}

func TestTokenWindowNeedsAWindow(t *testing.T) {
	msgs := agentLoop(40, 4000)
	if got := TokenWindow(0.5, 0.25).Apply(View{Messages: msgs}); len(got) != len(msgs) || !reflect.DeepEqual(got, msgs) {
		t.Error("TokenWindow acted without a window")
	}
	got := TokenWindow(0.5, 0.25).Apply(View{Messages: msgs, Window: 40_000})
	if reflect.DeepEqual(got, msgs) {
		t.Error("TokenWindow did nothing at twice its trigger")
	}
}

// TestTokenStrategiesWorkInsideASequence: Sequence rebuilt the View for each
// step from Messages and Results alone, so a token strategy inside one had no
// window and no estimator, and did nothing.
func TestTokenStrategiesWorkInsideASequence(t *testing.T) {
	msgs := agentLoop(40, 4000)
	estimated := 0
	v := View{Messages: msgs, Window: 40_000, Tokens: func(m []llm.Message) int {
		estimated++
		return (*tokenEstimator)(nil).estimate(m)
	}}
	alone := TokenWindow(0.5, 0.25).Apply(v)
	if reflect.DeepEqual(alone, msgs) {
		t.Fatal("premise: TokenWindow alone did nothing")
	}
	estimated = 0
	if got := Sequence(Flat, TokenWindow(0.5, 0.25)).Apply(v); !reflect.DeepEqual(got, alone) {
		t.Error("TokenWindow inside a Sequence did not trim as it does alone")
	}
	if estimated == 0 {
		t.Error("the View's estimator did not reach a strategy inside a Sequence")
	}
}

// An elided result is an evicted one as far as a Reconciler is concerned.
func TestElidedResultsAreNotPresentObservations(t *testing.T) {
	msgs := elideResults(View{}, agentLoop(4, 10), 5)
	present := observationsIn(msgs)
	if present["t0"] || present["t1"] {
		t.Errorf("elided results reported present: %v", present)
	}
	if !present["t3"] {
		t.Errorf("a kept result reported absent: %v", present)
	}
}

// ===== Compacted, deprecated but bounded =====

func TestCompactedSummarizesOncePerStep(t *testing.T) {
	calls := 0
	s := Compacted(10, 4, func([]llm.Message) string { calls++; return "summary" })
	var msgs []llm.Message
	for i := range 30 {
		msgs = append(msgs, llm.UserText(fmt.Sprintf("q%d", i)), llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.Text{Text: "a"}}})
		s.Apply(View{Messages: msgs})
		s.Apply(View{Messages: msgs}) // a second call on the same transcript is free
	}
	// 60 messages, step 6: about ten distinct prefixes.
	if calls == 0 || calls > 12 {
		t.Errorf("the compactor ran %d times over 30 growing transcripts, want about one per step", calls)
	}
}

// ===== WithCompaction =====

// bigTool returns n bytes, so a few calls cross a compaction threshold.
func bigTool(name string, n int) tool.Def {
	return echoTool(name, strings.Repeat("z", n))
}

func TestWithCompactionFoldsTheStoredTranscript(t *testing.T) {
	var turns []llm.Response
	for i := range 8 {
		turns = append(turns, toolTurn(fmt.Sprintf("c%d", i), "big", `{}`))
	}
	turns = append(turns, textTurn("done"))
	cl := scripted(turns...)

	var folded [][]llm.Message
	compactor := func(_ context.Context, older []llm.Message) (string, error) {
		folded = append(folded, older)
		return fmt.Sprintf("summary #%d", len(folded)), nil
	}
	a := newAgent(t, cl, []tool.Def{bigTool("big", 8000)},
		WithCompaction(6000, 2500, compactor), WithForceTerminalLastN(0))

	run := a.Start(context.Background(), Ask("do the big thing"))
	_, evs := drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	var compactions []TranscriptCompacted
	for _, ev := range evs {
		if c, ok := ev.(TranscriptCompacted); ok {
			compactions = append(compactions, c)
		}
	}
	if len(compactions) == 0 || len(folded) != len(compactions) {
		t.Fatalf("%d compaction events for %d compactor calls, want some and equal", len(compactions), len(folded))
	}
	if c := compactions[0]; c.Folded == 0 || c.TokensAfter >= c.TokensBefore {
		t.Errorf("event = %+v, want messages folded and the estimate down", c)
	}

	stored := Convo(run.Messages())
	if err := stored.Validate(); err != nil {
		t.Fatalf("the compacted transcript is invalid: %v", err)
	}
	head := allText(stored[:1])
	if !strings.Contains(head, "do the big thing") {
		t.Error("compaction dropped the task")
	}
	want := fmt.Sprintf("summary #%d", len(folded))
	if !strings.Contains(head, want) || strings.Count(head, "<conversation_summary>") != 1 {
		t.Errorf("head = %q, want exactly one summary, the latest (%s)", head, want)
	}
	// The compactor saw the task, and a later compaction saw the earlier summary.
	if !strings.Contains(allText(folded[0][:1]), "do the big thing") {
		t.Error("the compactor was not given the task")
	}
	if len(folded) > 1 && !strings.Contains(allText(folded[1][:1]), "summary #1") {
		t.Error("the second compaction was not given the first summary to roll up")
	}
	// The model was sent the compacted transcript afterwards, not the original.
	last := cl.request(t, cl.calls()-1)
	if !strings.Contains(allText(last.Messages[:1]), want) {
		t.Error("the last request did not carry the summary")
	}
}

func TestWithCompactionSurvivesAFailingCompactor(t *testing.T) {
	var turns []llm.Response
	for i := range 6 {
		turns = append(turns, toolTurn(fmt.Sprintf("c%d", i), "big", `{}`))
	}
	turns = append(turns, textTurn("done"))
	calls := 0
	a := newAgent(t, scripted(turns...), []tool.Def{bigTool("big", 8000)},
		WithCompaction(6000, 2500, func(context.Context, []llm.Message) (string, error) {
			calls++
			return "", errors.New("summarizer down")
		}), WithForceTerminalLastN(0))
	run := a.Start(context.Background(), Ask("go"))
	drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatalf("a failing compactor failed the run: %v", err)
	}
	if calls == 0 || calls >= 6 {
		t.Errorf("compactor called %d times, want it retried only after the transcript grew", calls)
	}
}

func TestSummarizeAsksTheClientForProse(t *testing.T) {
	cl := scripted(textTurn("  the agent grepped and found nothing  "))
	got, err := Summarize(cl, "small-model")(context.Background(), agentLoop(3, 50_000))
	if err != nil {
		t.Fatal(err)
	}
	if got != "  the agent grepped and found nothing  " {
		t.Errorf("summary = %q", got)
	}
	req := cl.request(t, 0)
	if req.Purpose != llm.PurposeSummarizer || req.Model != "small-model" || len(req.Messages) != 1 {
		t.Errorf("request = purpose %q model %q, %d messages", req.Purpose, req.Model, len(req.Messages))
	}
	text := llm.TextOf(req.Messages[0].Content)
	if !strings.Contains(text, "fix the failing test") || !strings.Contains(text, "more bytes") {
		t.Errorf("the rendering lost the task or did not bound a result:\n%.300s", text)
	}
}

// ===== the events =====

func TestRetryIsReportedAsAnEvent(t *testing.T) {
	flaky := &flakyClient{fail: 1, next: scripted(textTurn("ok"))}
	cl := llm.Chain(flaky, llm.WithRetry(llm.RetryPolicy{MaxAttempts: 3, Base: time.Millisecond, Max: time.Millisecond}))
	run := newAgent(t, cl, nil).Start(context.Background(), Ask("hi"))
	_, evs := drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	var retries []LLMRetry
	for _, ev := range evs {
		if r, ok := ev.(LLMRetry); ok {
			retries = append(retries, r)
		}
	}
	if len(retries) != 1 || retries[0].Attempt != 2 || retries[0].Of != 3 || retries[0].Error == "" {
		t.Errorf("retries = %+v, want one, attempt 2 of 3, with the error", retries)
	}
}

type flakyClient struct {
	fail int
	next llm.Client
}

func (f *flakyClient) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	if f.fail > 0 {
		f.fail--
		return llm.Response{}, &llm.APIError{Class: llm.ErrOverloaded, Status: 529, Message: "overloaded"}
	}
	return f.next.Complete(ctx, req)
}

func TestContextTrimmedIsReportedOnChangeOnly(t *testing.T) {
	var turns []llm.Response
	for i := range 12 {
		turns = append(turns, toolTurn(fmt.Sprintf("c%d", i), "big", `{}`))
	}
	turns = append(turns, textTurn("done"))
	a := newAgent(t, scripted(turns...), []tool.Def{bigTool("big", 8000)},
		WithStrategy(FitTokens(8000, 4000)), WithForceTerminalLastN(0), WithMaxIters(20))
	run := a.Start(context.Background(), Ask("go"))
	_, evs := drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	var trims []ContextTrimmed
	for _, ev := range evs {
		if c, ok := ev.(ContextTrimmed); ok {
			trims = append(trims, c)
		}
	}
	if len(trims) == 0 {
		t.Fatal("no ContextTrimmed event although the strategy elided results")
	}
	for i := 1; i < len(trims); i++ {
		if trims[i].Dropped == trims[i-1].Dropped && trims[i].Elided == trims[i-1].Elided {
			t.Errorf("event %d repeats the one before it: %+v", i, trims[i])
		}
	}
}

func TestEmptyTurnIsReportedAsAnEvent(t *testing.T) {
	empty := llm.Response{StopReason: llm.StopEndTurn}
	run := newAgent(t, scripted(empty, textTurn("there")), nil).Start(context.Background(), Ask("hi"))
	kinds, _ := drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(kinds, ","), "empty_turn") {
		t.Errorf("events = %v, want an empty_turn", kinds)
	}
}

func TestOverflowIsReportedAsAnEvent(t *testing.T) {
	overflow := &overflowOnce{next: scripted(textTurn("ok"))}
	cl := llm.Chain(overflow, WithOverflowRecovery(SlidingWindow(1)))
	msgs := []llm.Message{llm.UserText("one"), {Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.Text{Text: "a"}}}, llm.UserText("two")}
	run := newAgent(t, cl, nil).Start(context.Background(), Continue(msgs))
	_, evs := drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	for _, ev := range evs {
		if o, ok := ev.(ContextOverflow); ok && o.Level == 1 && o.MessagesAfter < o.MessagesBefore && o.TokensAfter < o.TokensBefore {
			return
		}
	}
	t.Error("no ContextOverflow event for a request the ladder shrank")
}

type overflowOnce struct {
	done bool
	next llm.Client
}

func (o *overflowOnce) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	if !o.done {
		o.done = true
		return llm.Response{}, &llm.APIError{Class: llm.ErrContextWindow, Status: 400, Message: "prompt is too long"}
	}
	return o.next.Complete(ctx, req)
}

// TestEstimatorCalibratesOnWhatWasSent: with server-side clearing, the tokens
// the provider read are fewer than were sent, and calibrating on them taught
// the estimator that every message is smaller than it is.
func TestEstimatorCalibratesOnWhatWasSent(t *testing.T) {
	msgs := agentLoop(10, 4000)
	var plain, cleared tokenEstimator
	sent := (*tokenEstimator)(nil).estimate(msgs)
	plain.observe(msgs, 0, sent)
	cleared.observe(msgs, 0, llm.Usage{InputTokens: sent / 4, ClearedInputTokens: sent - sent/4}.RequestTokens())
	if a, b := plain.estimate(msgs), cleared.estimate(msgs); a != b {
		t.Errorf("estimate after a cleared request = %d, want %d: the server's clearing skewed the calibration", b, a)
	}
}

// TestRunCalibratesOnTheRequestItSent pins the call site: the loop must hand
// the estimator RequestTokens. A response that read one token of a request the
// server mostly cleared would otherwise calibrate every estimate to nothing.
func TestRunCalibratesOnTheRequestItSent(t *testing.T) {
	resp := textTurn("ok")
	resp.Usage = llm.Usage{InputTokens: 1, ClearedInputTokens: 100_000}
	task := strings.Repeat("word ", 2000)
	run := newAgent(t, scripted(resp), nil).Start(context.Background(), Ask(task))
	drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	if got := run.tokens.estimate([]llm.Message{llm.UserText(task)}); got < 1000 {
		t.Errorf("estimate after a cleared request = %d tokens for a 10 KB message: calibrated on what the server read, not on what was sent", got)
	}
}
