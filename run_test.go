package wombat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
	"github.com/hoobean1996/wombat-go/trace"
)

// ===== the happy path =====

func TestRunToolCallThenAnswerEventOrder(t *testing.T) {
	cl := scripted(
		llm.Response{
			Content: []llm.ContentBlock{
				llm.Text{Text: "let me compute that"},
				llm.ToolUse{ID: "tu_1", Name: "calc", Input: json.RawMessage(`{"expression":"2+3*4"}`)},
			},
			StopReason: llm.StopToolUse,
			Usage:      llm.Usage{InputTokens: 100, OutputTokens: 20},
			Model:      "test-model",
		},
		llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: "The answer is 14."}},
			StopReason: llm.StopEndTurn,
			Usage:      llm.Usage{InputTokens: 150, OutputTokens: 8},
			Model:      "test-model",
		},
	)
	a := newAgent(t, cl, []tool.Def{echoTool("calc", "14")})

	run := a.Start(context.Background(), Ask("what is 2+3*4?"))
	t.Cleanup(func() { _ = run.Close() })

	kinds, evs := drain(t, run)

	if err := run.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}

	wantKinds := []string{
		"iter_start", "llm_start", "text_delta", "llm_done",
		"tool_start", "tool_done",
		"iter_start", "llm_start", "text_delta", "llm_done",
	}
	if fmt.Sprint(kinds) != fmt.Sprint(wantKinds) {
		t.Errorf("event order:\ngot  %v\nwant %v", kinds, wantKinds)
	}

	ans, ok := run.Outcome().(Answer)
	if !ok {
		t.Fatalf("Outcome() = %T, want Answer", run.Outcome())
	}
	if got, want := ans.Text, "The answer is 14."; got != want {
		t.Errorf("Answer.Text = %q, want %q", got, want)
	}

	// Event payloads, not just their kinds.
	var (
		iterStarts []IterStart
		toolDone   ToolDone
		deltas     strings.Builder
	)
	for _, ev := range evs {
		switch e := ev.(type) {
		case IterStart:
			iterStarts = append(iterStarts, e)
		case ToolDone:
			toolDone = e
		case TextDelta:
			deltas.WriteString(e.Text)
		}
	}
	if len(iterStarts) != 2 || iterStarts[0].N != 1 || iterStarts[1].N != 2 {
		t.Errorf("iter_start numbering = %+v, want N=1 then N=2", iterStarts)
	}
	if iterStarts[0].Max != DefaultMaxIters {
		t.Errorf("iter_start Max = %d, want %d", iterStarts[0].Max, DefaultMaxIters)
	}
	if !toolDone.OK || toolDone.Output != "14" {
		t.Errorf("tool_done = %+v, want OK with output %q", toolDone, "14")
	}
	if got, want := deltas.String(), "let me compute thatThe answer is 14."; got != want {
		t.Errorf("streamed text = %q, want %q", got, want)
	}

	// The transcript must be a legal conversation to resume from.
	msgs := run.Messages()
	if len(msgs) != 4 {
		t.Errorf("transcript length = %d, want 4 (user, assistant, tool_result, assistant)", len(msgs))
	}
	if err := Convo(msgs).Validate(); err != nil {
		t.Errorf("transcript does not validate: %v", err)
	}

	// The second call must carry the tool result back.
	if cl.calls() != 2 {
		t.Fatalf("llm calls = %d, want 2", cl.calls())
	}
	second := cl.request(t, 1)
	found := false
	for _, m := range second.Messages {
		for _, b := range m.Content {
			if tr, ok := b.(llm.ToolResult); ok && tr.ToolUseID == "tu_1" {
				found = true
			}
		}
	}
	if !found {
		t.Error("second request carries no tool_result for tu_1")
	}
	if got, want := cl.request(t, 0).System, second.System; got != want {
		t.Errorf("system prompt drifted between calls:\ngot  %q\nwant %q", second.System, got)
	}
	if got := specNames(second.Tools); len(got) != 1 || got[0] != "calc" {
		t.Errorf("tool specs = %v, want [calc]", got)
	}
	if got, want := second.MaxTokens, DefaultMaxTokens; got != want {
		t.Errorf("MaxTokens = %d, want %d", got, want)
	}
	if got, want := second.Model, "test-model"; got != want {
		t.Errorf("Model = %q, want %q", got, want)
	}
}

func TestRunPause(t *testing.T) {
	cl := scripted(toolTurn("tu_ask", "ask_user", `{"question":"which branch?"}`))
	a := newAgent(t, cl, []tool.Def{echoTool("calc", "14"), pauseTool("ask_user")})

	out, err := a.Run(context.Background(), Ask("deploy it"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	p, ok := out.(Paused)
	if !ok {
		t.Fatalf("Outcome = %T, want Paused", out)
	}
	if p.ToolUseID != "tu_ask" {
		t.Errorf("ToolUseID = %q, want %q", p.ToolUseID, "tu_ask")
	}
	if p.Schema.Question != "which branch?" {
		t.Errorf("Question = %q, want %q", p.Schema.Question, "which branch?")
	}
	// The pause tool's handler must never be invoked, and the loop must stop
	// rather than asking the model again.
	if cl.calls() != 1 {
		t.Errorf("llm calls = %d, want 1", cl.calls())
	}
}

func TestRunTerminalTool(t *testing.T) {
	cl := scripted(toolTurn("tu_s", "submit", `{"summary":"done"}`))
	a := newAgent(t, cl, []tool.Def{echoTool("calc", "14"), terminalTool("submit")},
		WithTerminalTool("submit"), WithMaxIters(6))

	out, err := a.Run(context.Background(), Ask("do the thing"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	s, ok := out.(Submitted)
	if !ok {
		t.Fatalf("Outcome = %T, want Submitted", out)
	}
	if s.Tool != "submit" {
		t.Errorf("Tool = %q, want %q", s.Tool, "submit")
	}
	if got, want := string(s.Payload), `{"summary":"done"}`; got != want {
		t.Errorf("Payload = %s, want %s", got, want)
	}

	payload, err := ExpectSubmitted(out, "submit")
	if err != nil {
		t.Fatalf("ExpectSubmitted: %v", err)
	}
	if len(payload) == 0 {
		t.Error("ExpectSubmitted returned an empty payload")
	}
}

// A terminal tool takes precedence over a pause tool in the same batch: the
// run is over either way, and ending beats suspending.
func TestRunTerminalBeatsPause(t *testing.T) {
	cl := scripted(llm.Response{
		Content: []llm.ContentBlock{
			llm.ToolUse{ID: "p", Name: "ask_user", Input: json.RawMessage(`{"question":"?"}`)},
			llm.ToolUse{ID: "s", Name: "submit", Input: json.RawMessage(`{"summary":"done"}`)},
		},
		StopReason: llm.StopToolUse,
	})
	a := newAgent(t, cl, []tool.Def{pauseTool("ask_user"), terminalTool("submit")},
		WithTerminalTool("submit"))

	out, err := a.Run(context.Background(), Ask("go"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if _, ok := out.(Submitted); !ok {
		t.Fatalf("Outcome = %T, want Submitted", out)
	}
}

// ===== the failure modes =====

func TestRunStopReasons(t *testing.T) {
	tests := []struct {
		name    string
		resp    llm.Response
		wantErr error
		errFrag string
	}{
		{
			name: "refusal keeps the model's reason",
			resp: llm.Response{
				Content:    []llm.ContentBlock{llm.Text{Text: "I can't help with that."}},
				StopReason: llm.StopRefusal,
			},
			wantErr: ErrRefused,
			errFrag: "can't help",
		},
		{
			name:    "an unknown stop_reason is not silently continued",
			resp:    llm.Response{StopReason: llm.StopReason("content_filter")},
			wantErr: ErrUnexpectedStop,
			errFrag: "content_filter",
		},
		{
			name:    "stop_reason=tool_use with no tool_use block",
			resp:    llm.Response{StopReason: llm.StopToolUse},
			errFrag: "no tool_use block",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newAgent(t, scripted(tc.resp), nil)
			_, err := a.Run(context.Background(), Ask("x"))
			if err == nil {
				t.Fatal("got nil error, want one")
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Errorf("error = %v, want it to wrap %v", err, tc.wantErr)
			}
			if tc.errFrag != "" && !strings.Contains(err.Error(), tc.errFrag) {
				t.Errorf("error = %q, want it to contain %q", err, tc.errFrag)
			}
		})
	}
}

// stop_sequence ends the turn the same way end_turn does.
func TestRunStopSequenceIsAnAnswer(t *testing.T) {
	cl := scripted(llm.Response{
		Content:    []llm.ContentBlock{llm.Text{Text: "cut here"}},
		StopReason: llm.StopStopSequence,
	})
	a := newAgent(t, cl, nil)
	out, err := a.Run(context.Background(), Ask("x"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if ans, ok := out.(Answer); !ok || ans.Text != "cut here" {
		t.Errorf("Outcome = %#v, want Answer{cut here}", out)
	}
}

func TestRunMaxIterations(t *testing.T) {
	turns := make([]llm.Response, 10)
	for i := range turns {
		turns[i] = toolTurn(fmt.Sprintf("t%d", i), "echo", `{}`)
	}
	cl := scripted(turns...)
	a := newAgent(t, cl, []tool.Def{echoTool("echo", "again")}, WithMaxIters(3))

	_, err := a.Run(context.Background(), Ask("spin"))
	if !errors.Is(err, ErrMaxIterations) {
		t.Fatalf("error = %v, want ErrMaxIterations", err)
	}
	if !strings.Contains(err.Error(), "3") {
		t.Errorf("error = %q, want it to name the cap", err)
	}
	if got, want := cl.calls(), 3; got != want {
		t.Errorf("llm calls = %d, want %d (the cap, not one more)", got, want)
	}
}

func TestRunRejectsAnInvalidInputTranscript(t *testing.T) {
	cl := scripted(textTurn("never reached"))
	a := newAgent(t, cl, nil)

	_, err := a.Run(context.Background(), Continue([]llm.Message{assistantText("assistant first")}))
	if !errors.Is(err, ErrNotUserFirst) {
		t.Fatalf("error = %v, want it to wrap ErrNotUserFirst", err)
	}
	if !strings.Contains(err.Error(), "invalid input transcript") {
		t.Errorf("error = %q, want it to say the input was invalid", err)
	}
	if cl.calls() != 0 {
		t.Errorf("llm calls = %d, want 0 — a bad transcript must not reach the provider", cl.calls())
	}
}

func TestRunBudgetAbort(t *testing.T) {
	t.Run("step limit", func(t *testing.T) {
		turns := make([]llm.Response, 10)
		for i := range turns {
			turns[i] = toolTurn(fmt.Sprintf("t%d", i), "echo", `{}`)
		}
		a := newAgent(t, scripted(turns...), []tool.Def{echoTool("echo", "again")}, WithMaxIters(50))

		ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{Steps: 3})
		defer cancel()

		_, err := a.Run(ctx, Ask("loop"))
		if !errors.Is(err, governor.ErrStepLimit) {
			t.Fatalf("error = %v, want governor.ErrStepLimit", err)
		}
	})

	// A cost cap has to reach the loop through the client chain, which is the
	// only place that sees a response.
	t.Run("cost limit via TrackCost", func(t *testing.T) {
		turns := make([]llm.Response, 20)
		for i := range turns {
			turns[i] = llm.Response{
				Content: []llm.ContentBlock{
					llm.ToolUse{ID: llm.ToolUseID(fmt.Sprintf("tu_%d", i)), Name: "echo", Input: json.RawMessage(`{}`)},
				},
				StopReason: llm.StopToolUse,
				Usage:      llm.Usage{InputTokens: 100_000, OutputTokens: 100_000},
				Model:      "claude-opus-5",
			}
		}
		cl := llm.Chain(scripted(turns...), TrackCost(llm.DefaultPricing))
		a := newAgent(t, cl, []tool.Def{echoTool("echo", "again")}, WithMaxIters(50))

		ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{CostUSD: 5.00})
		defer cancel()

		_, err := a.Run(ctx, Ask("loop"))
		if !errors.Is(err, governor.ErrBudgetExhausted) {
			t.Fatalf("error = %v, want governor.ErrBudgetExhausted", err)
		}
	})

	// The abort has to be prompt, and it has to report the governor's cause
	// rather than the "context canceled" the interrupted call reports.
	t.Run("wall clock reports the cause, not context.Canceled", func(t *testing.T) {
		blocking := llm.ClientFunc(func(ctx context.Context, _ llm.Request) (llm.Response, error) {
			<-ctx.Done()
			return llm.Response{}, ctx.Err()
		})
		a := newAgent(t, blocking, nil)

		ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{Wall: 20 * time.Millisecond})
		defer cancel()

		start := time.Now()
		_, err := a.Run(ctx, Ask("hang"))
		if !errors.Is(err, governor.ErrWallClock) {
			t.Fatalf("error = %v, want governor.ErrWallClock", err)
		}
		if errors.Is(err, context.Canceled) {
			t.Errorf("error = %v, want the governor's cause rather than context.Canceled", err)
		}
		if elapsed := time.Since(start); elapsed > 2*time.Second {
			t.Errorf("took %v to abort, want prompt cancellation", elapsed)
		}
	})

	// Repeated identical calls are the backstop for an agent the advisory
	// dedup annotation failed to shake loose.
	t.Run("repeated identical tool calls", func(t *testing.T) {
		turns := make([]llm.Response, 10)
		for i := range turns {
			// Same tool, same arguments, different use id: a stuck agent.
			turns[i] = toolTurn(fmt.Sprintf("r%d", i), "echo", `{"expression":"1+1"}`)
		}
		a := newAgent(t, scripted(turns...), []tool.Def{echoTool("echo", "2")}, WithMaxIters(20))

		ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{RepeatedToolCalls: 3})
		defer cancel()

		_, err := a.Run(ctx, Ask("loop"))
		if !errors.Is(err, governor.ErrToolLoop) {
			t.Fatalf("error = %v, want governor.ErrToolLoop", err)
		}
	})
}

func TestRunToolTimeoutCancelsAndSurfacesToTheModel(t *testing.T) {
	cl := scripted(
		toolTurn("s1", "slow", `{}`),
		textTurn("gave up"),
	)
	a := newAgent(t, cl, []tool.Def{slowTool("slow", 40*time.Millisecond)})

	start := time.Now()
	run := a.Start(context.Background(), Ask("run it"))
	t.Cleanup(func() { _ = run.Close() })

	var toolErr string
	_, evs := drain(t, run)
	for _, ev := range evs {
		if td, ok := ev.(ToolDone); ok && !td.OK {
			toolErr = td.Error
		}
	}
	elapsed := time.Since(start)

	if err := run.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil — a tool timeout must not fail the run", err)
	}
	if elapsed > 5*time.Second {
		t.Errorf("run took %v; the tool's 40ms timeout did not cancel it", elapsed)
	}
	if !strings.Contains(toolErr, "timeout") {
		t.Errorf("tool_done error = %q, want it to report a timeout", toolErr)
	}
	// A killed subprocess's partial output is usually the most useful thing
	// the model can see; the normalisation must not throw it away.
	if !strings.Contains(toolErr, "abc") {
		t.Errorf("tool_done error = %q, want it to keep the tool's partial output", toolErr)
	}

	sawErrResult := false
	for _, m := range run.Messages() {
		for _, b := range m.Content {
			if tr, ok := b.(llm.ToolResult); ok && tr.IsError {
				sawErrResult = true
			}
		}
	}
	if !sawErrResult {
		t.Error("the failure never reached the model as an is_error tool_result")
	}
}

// ===== panic containment, at both levels =====

func TestToolPanicIsContained(t *testing.T) {
	oob := tool.Def{
		Name: "oob", Description: "index panic", InputSchema: json.RawMessage(objSchema),
		Fn: func(context.Context, json.RawMessage) (string, error) {
			var xs []int
			return fmt.Sprint(xs[3]), nil
		},
	}
	cl := scripted(
		toolTurn("p1", "boom", `{}`),
		toolTurn("p2", "oob", `{}`),
		textTurn("both tools crashed, giving up"),
	)
	a := newAgent(t, cl, []tool.Def{panicTool("boom", "kaboom"), oob})

	run := a.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	var toolErrs []string
	_, evs := drain(t, run)
	for _, ev := range evs {
		if td, ok := ev.(ToolDone); ok && !td.OK {
			toolErrs = append(toolErrs, td.Error)
		}
	}

	if err := run.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil — a panicking tool must not fail the run", err)
	}
	if _, ok := run.Outcome().(Answer); !ok {
		t.Fatalf("Outcome = %T, want Answer", run.Outcome())
	}
	if len(toolErrs) != 2 {
		t.Fatalf("got %d tool errors %v, want 2", len(toolErrs), toolErrs)
	}
	// The model gets a short message, not a stack: a screenful of goroutine
	// dump is tokens spent on nothing it can act on.
	if strings.Contains(toolErrs[0], "goroutine") || len(toolErrs[0]) > 300 {
		t.Errorf("tool error (%d bytes) = %q, want a short message with no stack", len(toolErrs[0]), toolErrs[0])
	}
	if !strings.Contains(toolErrs[1], "index out of range") {
		t.Errorf("runtime panic error = %q, want it to name the runtime error", toolErrs[1])
	}
	if !strings.Contains(allText(run.Messages()), "kaboom") {
		t.Error("the panic never reached the model as a tool_result")
	}
}

type panicStrategy struct{}

func (panicStrategy) Apply(View) []llm.Message { panic("strategy exploded") }
func (panicStrategy) String() string           { return "panic" }

func TestLoopPanicIsContained(t *testing.T) {
	a := newAgent(t, scripted(textTurn("never reached")), nil, WithStrategy(panicStrategy{}))

	out, err := a.Run(context.Background(), Ask("go"))
	if !errors.Is(err, ErrPanic) {
		t.Fatalf("error = %v, want ErrPanic", err)
	}
	if out != nil {
		t.Errorf("Outcome = %#v, want nil on a contained panic", out)
	}
	if !strings.Contains(err.Error(), "strategy exploded") {
		t.Errorf("error = %q, want it to carry the panic value", err)
	}
	// The operator, unlike the model, does need the stack.
	if !strings.Contains(err.Error(), "goroutine") {
		t.Errorf("error = %q, want it to carry a stack for the operator", err)
	}
}

func TestTruncateStack(t *testing.T) {
	short := []byte("goroutine 1 [running]:")
	if got := truncateStack(short); got != string(short) {
		t.Errorf("short stack was altered: got %q", got)
	}

	long := make([]byte, 8<<10)
	for i := range long {
		long[i] = 'x'
	}
	got := truncateStack(long)
	if len(got) >= len(long) {
		t.Errorf("long stack was not truncated: got %d bytes, want fewer than %d", len(got), len(long))
	}
	if !strings.HasSuffix(got, "stack truncated") {
		t.Errorf("truncated stack does not say so: %q", got[len(got)-40:])
	}
}

// ===== the partial-answer regression =====

// partialClient streams some text and then fails, the way a cancelled or
// dropped call does.
type partialClient struct {
	words []string
	fail  error
}

func (c *partialClient) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	for _, w := range c.words {
		if req.OnDelta != nil {
			req.OnDelta(llm.Delta{Text: w})
		}
	}
	return llm.Response{}, c.fail
}

// TestFailedCallKeepsThePartialAssistantTurn is a regression test.
//
// A call that dies part-way must leave what the user already watched arrive in
// the transcript. Dropping it makes "stop, then continue" — the most ordinary
// thing a chat UI does — impossible, because the resumed conversation has no
// record of the half-answer the user is looking at.
//
// The trailing whitespace matters just as much: the partial becomes the FINAL
// assistant turn, and Anthropic rejects a final assistant turn ending in
// whitespace. Keeping the text but leaving the trailing space turns a
// resumable transcript into a hard 400 on the next call.
func TestFailedCallKeepsThePartialAssistantTurn(t *testing.T) {
	cl := &partialClient{
		words: []string{"The answer ", "is going ", "to be "},
		fail:  errors.New("upstream died"),
	}
	a := newAgent(t, cl, nil)

	run := a.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	var shown strings.Builder
	_, evs := drain(t, run)
	for _, ev := range evs {
		if d, ok := ev.(TextDelta); ok {
			shown.WriteString(d.Text)
		}
	}

	if run.Err() == nil {
		t.Fatal("Err() = nil, want the upstream failure")
	}

	msgs := run.Messages()
	if len(msgs) != 2 {
		t.Fatalf("transcript length = %d, want 2 (the question and the half-answer)", len(msgs))
	}
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleAssistant {
		t.Errorf("last turn role = %s, want %s so the next call is a prefill", last.Role, llm.RoleAssistant)
	}

	got := llm.TextOf(last.Content)
	if want := "The answer is going to be"; got != want {
		t.Errorf("partial answer = %q, want %q", got, want)
	}
	if got != strings.TrimRight(got, " \t\r\n") {
		t.Errorf("partial answer = %q, want no trailing whitespace (Anthropic rejects it)", got)
	}
	if strings.TrimRight(shown.String(), " ") != got {
		t.Errorf("transcript %q does not match what was rendered %q", got, shown.String())
	}
	if err := Convo(msgs).Validate(); err != nil {
		t.Errorf("the stopped transcript is not resumable: %v", err)
	}
}

func TestFailedCallWithNoPartialLeavesTheTranscriptAlone(t *testing.T) {
	cl := &partialClient{fail: errors.New("died before a single token")}
	a := newAgent(t, cl, nil)

	run := a.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })
	drain(t, run)

	if run.Err() == nil {
		t.Fatal("Err() = nil, want the upstream failure")
	}
	msgs := run.Messages()
	if len(msgs) != 1 {
		t.Errorf("transcript length = %d, want 1 — an empty partial must not be pushed", len(msgs))
	}
}

// ===== turn notice =====

func TestTurnNotice(t *testing.T) {
	cl := scripted(
		toolTurn("n1", "echo", `{}`),
		textTurn("done"),
	)
	a := newAgent(t, cl, []tool.Def{echoTool("echo", "ok")},
		WithTurnNotice(func(_ context.Context, iter int) string {
			if iter < 2 {
				return ""
			}
			return "<budget_status>almost out</budget_status>"
		}))

	if _, err := a.Run(context.Background(), Ask("compute")); err != nil {
		t.Fatalf("Run: %v", err)
	}

	first := allText(cl.request(t, 0).Messages)
	second := allText(cl.request(t, 1).Messages)

	if strings.Contains(first, "almost out") {
		t.Error("the notice appeared on turn 1, where the hook returned \"\"")
	}
	if !strings.Contains(second, "almost out") {
		t.Errorf("turn 2 messages = %q, want the notice", second)
	}
	if strings.Contains(cl.request(t, 1).System, "almost out") {
		t.Error("the notice reached the system prompt, which must stay the stable cache prefix")
	}
	if got := strings.Count(second, "almost out"); got != 1 {
		t.Errorf("the notice appears %d times, want 1 — it must not accumulate", got)
	}

	// And it never enters the stored transcript.
	a2 := newAgent(t, scripted(textTurn("done")), nil,
		WithTurnNotice(func(context.Context, int) string { return "NOTICE" }))
	run := a2.Start(context.Background(), Ask("q"))
	t.Cleanup(func() { _ = run.Close() })
	drain(t, run)
	if strings.Contains(allText(run.Messages()), "NOTICE") {
		t.Errorf("the notice leaked into the stored transcript: %q", allText(run.Messages()))
	}
}

func TestWithNotice(t *testing.T) {
	t.Run("appends to the last user turn", func(t *testing.T) {
		msgs := []llm.Message{userMsg("hi")}
		got := withNotice(msgs, "NOTE")
		if len(got[0].Content) != 2 {
			t.Fatalf("got %d blocks, want 2", len(got[0].Content))
		}
		if !strings.Contains(llm.TextOf(got[0].Content), "NOTE") {
			t.Errorf("got %q, want the note", llm.TextOf(got[0].Content))
		}
		// The input must not be touched: msgs may alias the stored transcript.
		if len(msgs[0].Content) != 1 {
			t.Errorf("the source message was mutated: %d blocks, want 1", len(msgs[0].Content))
		}
	})

	t.Run("skipped when the last turn is not a user turn", func(t *testing.T) {
		msgs := []llm.Message{userMsg("hi"), assistantText("hello")}
		got := withNotice(msgs, "NOTE")
		if strings.Contains(allText(got), "NOTE") {
			t.Errorf("got %q, want no note on an assistant-final transcript", allText(got))
		}
	})

	t.Run("empty transcript", func(t *testing.T) {
		if got := withNotice(nil, "NOTE"); len(got) != 0 {
			t.Errorf("got %d messages, want 0", len(got))
		}
	})
}

// ===== small helpers on the loop =====

func TestObservationsIn(t *testing.T) {
	msgs := []llm.Message{
		userMsg("q"),
		assistantUse("a", "t"),
		userResult("a", "r"),
		assistantUse("b", "t"), // its result was evicted
		userMsg("next"),
	}
	got := observationsIn(msgs)
	if !got["a"] {
		t.Error(`observationsIn is missing "a", whose tool_result is present`)
	}
	if got["b"] {
		t.Error(`observationsIn contains "b", whose tool_result was evicted — ` +
			"keying on tool_use rather than tool_result would report the body as still present")
	}
	if len(got) != 1 {
		t.Errorf("got %d ids, want 1", len(got))
	}
}

// reconcilerSet records what the loop told it survived materialization.
type reconcilerSet struct {
	tool.Set
	mu    sync.Mutex
	calls []map[llm.ToolUseID]bool
}

func (s *reconcilerSet) Reconcile(_ context.Context, present map[llm.ToolUseID]bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, present)
}

func (s *reconcilerSet) snapshot() []map[llm.ToolUseID]bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]map[llm.ToolUseID]bool(nil), s.calls...)
}

func TestLoopReconcilesBeforeReadingTheSurface(t *testing.T) {
	set := &reconcilerSet{Set: tool.NewSet(echoTool("echo", "ok"))}
	cl := scripted(toolTurn("u1", "echo", `{}`), textTurn("done"))
	a := newAgent(t, cl, nil, WithToolSet(set))

	if _, err := a.Run(context.Background(), Ask("go")); err != nil {
		t.Fatalf("Run: %v", err)
	}

	calls := set.snapshot()
	if len(calls) != 2 {
		t.Fatalf("Reconcile called %d times, want once per iteration (2)", len(calls))
	}
	if len(calls[0]) != 0 {
		t.Errorf("iteration 1 saw %v, want no observations yet", calls[0])
	}
	if !calls[1]["u1"] {
		t.Errorf("iteration 2 saw %v, want u1 present", calls[1])
	}
}

// ===== the input constructors =====

func TestInputConstructors(t *testing.T) {
	t.Run("Ask", func(t *testing.T) {
		in := Ask("hi")
		if len(in.Messages) != 1 || in.Messages[0].Role != llm.RoleUser {
			t.Fatalf("Ask = %+v, want one user turn", in.Messages)
		}
		if err := Convo(in.Messages).Validate(); err != nil {
			t.Errorf("Ask does not validate: %v", err)
		}
	})

	t.Run("Continue passes the transcript through", func(t *testing.T) {
		prior := []llm.Message{userMsg("a"), assistantText("b")}
		if got := Continue(prior); len(got.Messages) != 2 {
			t.Errorf("got %d messages, want 2", len(got.Messages))
		}
	})

	t.Run("AnswerPause closes the tool_use", func(t *testing.T) {
		prior := []llm.Message{userMsg("q"), assistantUse("p", "ask_user")}
		in := AnswerPause(prior, "p", "main")
		if err := Convo(in.Messages).Validate(); err != nil {
			t.Fatalf("resumed transcript does not validate: %v", err)
		}
		if !strings.Contains(allText(in.Messages), "main") {
			t.Errorf("got %q, want the answer", allText(in.Messages))
		}
	})

	t.Run("Then on a clean transcript appends a user turn", func(t *testing.T) {
		prior := []llm.Message{userMsg("q"), assistantText("a")}
		in := Then(prior, "and now?")
		if len(in.Messages) != 3 {
			t.Fatalf("got %d messages, want 3", len(in.Messages))
		}
		if err := Convo(in.Messages).Validate(); err != nil {
			t.Errorf("does not validate: %v", err)
		}
	})

	t.Run("Then closes a dangling tool_use first", func(t *testing.T) {
		prior := []llm.Message{userMsg("q"), assistantUse("p", "t")}
		in := Then(prior, "never mind, do this")
		if err := Convo(in.Messages).Validate(); err != nil {
			t.Fatalf("does not validate: %v", err)
		}
		flat := allText(in.Messages)
		if !strings.Contains(flat, "(cancelled)") {
			t.Errorf("got %q, want the dangling tool_use acknowledged", flat)
		}
		if !strings.Contains(flat, "never mind") {
			t.Errorf("got %q, want the new instruction", flat)
		}
	})
}

// ===== Run mechanics =====

func TestRunMessagesIsASnapshot(t *testing.T) {
	cl := scripted(toolTurn("u1", "echo", `{}`), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{echoTool("echo", "ok")})

	run := a.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })

	// Grab a snapshot mid-run, then keep going. The snapshot must not grow.
	var early []llm.Message
	for run.Next() {
		if early == nil && run.Event().Kind() == "tool_done" {
			early = run.Messages()
		}
	}
	if early == nil {
		t.Fatal("never saw a tool_done event")
	}
	final := run.Messages()
	if len(early) >= len(final) {
		t.Errorf("early snapshot has %d messages and the final has %d; want the snapshot frozen", len(early), len(final))
	}
}

func TestRunCloseIsIdempotentAndUnblocksAnAbandonedRun(t *testing.T) {
	turns := make([]llm.Response, 30)
	for i := range turns {
		turns[i] = toolTurn(fmt.Sprintf("t%d", i), "echo", `{}`)
	}
	a := newAgent(t, scripted(turns...), []tool.Def{echoTool("echo", "ok")}, WithMaxIters(30))

	run := a.Start(context.Background(), Ask("go"))
	if !run.Next() {
		t.Fatal("Next() = false on the first event")
	}
	// Abandon the iteration with the producer parked on its next send.
	if err := run.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := run.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if run.Next() {
		t.Error("Next() = true after Close")
	}
}

func TestRunProgressReportsSpend(t *testing.T) {
	cl := scripted(llm.Response{
		Content:    []llm.ContentBlock{llm.Text{Text: "hi"}},
		StopReason: llm.StopEndTurn,
		Usage:      llm.Usage{InputTokens: 1000, OutputTokens: 100},
		Model:      "claude-opus-5",
	})
	a := newAgent(t, llm.Chain(cl, TrackCost(llm.DefaultPricing)), nil)

	ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{CostUSD: 1.00, Steps: 20})
	defer cancel()

	run := a.Start(ctx, Ask("hi"))
	t.Cleanup(func() { _ = run.Close() })
	kinds, _ := drain(t, run)

	if !contains(kinds, "spend") {
		t.Errorf("event kinds = %v, want a spend event from TrackCost", kinds)
	}

	p := run.Progress()
	if p.Calls != 1 {
		t.Errorf("Progress.Calls = %d, want 1", p.Calls)
	}
	if p.CostUSD <= 0 {
		t.Errorf("Progress.CostUSD = %v, want > 0", p.CostUSD)
	}
	if f := p.Fraction(); f <= 0 || f >= 1 {
		t.Errorf("Progress.Fraction() = %v, want a value strictly between 0 and 1", f)
	}
	if !strings.Contains(p.String(), "cost $") {
		t.Errorf("Progress.String() = %q, want it to name the cost", p.String())
	}
}

func TestRunProgressWithNoBudgetIsZero(t *testing.T) {
	a := newAgent(t, scripted(textTurn("hi")), nil)
	run := a.Start(context.Background(), Ask("hi"))
	t.Cleanup(func() { _ = run.Close() })
	drain(t, run)

	if p := run.Progress(); p.Calls != 0 || p.CostUSD != 0 {
		t.Errorf("Progress = %+v, want the zero value with no budget on ctx", p)
	}
}

func TestWithEventBufferDoesNotChangeBehaviour(t *testing.T) {
	unbuffered := newAgent(t, scripted(toolTurn("u", "echo", `{}`), textTurn("hi")),
		[]tool.Def{echoTool("echo", "ok")})
	buffered := newAgent(t, scripted(toolTurn("u", "echo", `{}`), textTurn("hi")),
		[]tool.Def{echoTool("echo", "ok")}, WithEventBuffer(64))

	runA := unbuffered.Start(context.Background(), Ask("x"))
	t.Cleanup(func() { _ = runA.Close() })
	kindsA, _ := drain(t, runA)

	runB := buffered.Start(context.Background(), Ask("x"))
	t.Cleanup(func() { _ = runB.Close() })
	kindsB, _ := drain(t, runB)

	if fmt.Sprint(kindsA) != fmt.Sprint(kindsB) {
		t.Errorf("buffered run produced different events:\ngot  %v\nwant %v", kindsB, kindsA)
	}
	if runB.Err() != nil {
		t.Errorf("Err() = %v, want nil", runB.Err())
	}
	if _, ok := runB.Outcome().(Answer); !ok {
		t.Errorf("Outcome = %T, want Answer", runB.Outcome())
	}
}

// ===== WithRunContext =====

type runCtxKey struct{}

func TestWithRunContextIsPerRunAndOrdered(t *testing.T) {
	var order []string
	var mu sync.Mutex
	record := func(s string) { mu.Lock(); order = append(order, s); mu.Unlock() }

	// Each run gets its own pointer, so two runs of one agent cannot share it.
	seen := make(chan *int, 4)
	a := newAgent(t, scripted(textTurn("hi")), nil,
		WithRunContext(func(ctx context.Context) context.Context {
			record("first")
			n := new(int)
			seen <- n
			return context.WithValue(ctx, runCtxKey{}, n)
		}),
		WithRunContext(func(ctx context.Context) context.Context {
			record("second")
			return ctx
		}),
	)

	for range 2 {
		cl := scripted(textTurn("hi"))
		derived, err := a.With(WithClient(cl))
		if err != nil {
			t.Fatalf("With: %v", err)
		}
		if _, err := derived.Run(context.Background(), Ask("x")); err != nil {
			t.Fatalf("Run: %v", err)
		}
	}

	mu.Lock()
	got := append([]string(nil), order...)
	mu.Unlock()
	want := []string{"first", "second", "first", "second"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("decorator order = %v, want %v", got, want)
	}

	close(seen)
	var ptrs []*int
	for p := range seen {
		ptrs = append(ptrs, p)
	}
	if len(ptrs) != 2 || ptrs[0] == ptrs[1] {
		t.Errorf("two runs shared per-run state: %v", ptrs)
	}
}

// ===== the run-span ordering regression =====

// orderedSink records spans and notices any that arrive after the consumer has
// declared the event channel closed.
type orderedSink struct {
	mu      sync.Mutex
	spans   []trace.Span
	sealed  bool
	lateRun []string
}

func (s *orderedSink) Emit(sp trace.Span) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sealed && sp.Kind == trace.KindRun {
		s.lateRun = append(s.lateRun, sp.Name)
	}
	s.spans = append(s.spans, sp)
}

// seal marks the moment the consumer observed the run's event channel close.
// A trace sink is closed exactly here in real code.
func (s *orderedSink) seal() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sealed = true
}

func (s *orderedSink) snapshot() ([]trace.Span, []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]trace.Span(nil), s.spans...), append([]string(nil), s.lateRun...)
}

// TestRunSpanIsEmittedBeforeTheEventChannelCloses is a regression test.
//
// loop registers three defers in reverse order so they run as: recover,
// span.End, close(events). The ordering of the last two is load-bearing.
// Run.Next reports false as soon as the channel closes, and a delegate tool
// takes that as its cue to return; if the span were still unemitted at that
// moment, a caller closing its trace sink right after the run — which is what
// `defer closer.Close()` does — could lose a whole sub-agent's run span and
// orphan every span beneath it. The original only reproduced under -race.
//
// Asserted directly rather than raced for: the sink is told the instant the
// consumer sees the channel close, and any run span arriving after that is
// recorded as late.
func TestRunSpanIsEmittedBeforeTheEventChannelCloses(t *testing.T) {
	sink := &orderedSink{}
	a := newAgent(t, scripted(textTurn("hi")), nil,
		WithName("root"), WithTracing(trace.New(sink)))

	run := a.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })
	for run.Next() { //nolint:revive // draining is the point
	}
	sink.seal()

	spans, late := sink.snapshot()
	if len(late) > 0 {
		t.Fatalf("run spans %v were emitted AFTER the event channel closed; "+
			"a caller closing its trace sink on run completion would lose them", late)
	}

	found := false
	for _, sp := range spans {
		if sp.Kind == trace.KindRun && sp.Name == "root" {
			found = true
		}
	}
	if !found {
		t.Fatalf("no run span in the sink when the channel closed; got %d spans", len(spans))
	}
}

// The sub-agent case is the one that actually lost data: the child's run span
// must be in the sink, parented to the subagent span, by the time the PARENT's
// run is over.
func TestSubagentRunSpanSurvivesTheHandoff(t *testing.T) {
	sink := &orderedSink{}
	child := newAgent(t, scripted(textTurn("child answer")), nil, WithName("child"))
	parentCl := scripted(
		toolTurn("d1", "delegate", `{"task":"do a thing"}`),
		textTurn("done"),
	)
	parent := newAgent(t, parentCl, []tool.Def{DelegateTool(child)},
		WithName("parent"), WithTracing(trace.New(sink)))

	run := parent.Start(context.Background(), Ask("go"))
	t.Cleanup(func() { _ = run.Close() })
	for run.Next() { //nolint:revive // draining is the point
	}
	sink.seal()

	spans, late := sink.snapshot()
	if len(late) > 0 {
		t.Fatalf("run spans %v arrived after the parent's channel closed", late)
	}

	byID := make(map[string]trace.Span, len(spans))
	for _, sp := range spans {
		byID[sp.ID] = sp
	}
	nested := 0
	for _, sp := range spans {
		if sp.Kind == trace.KindRun && byID[sp.ParentID].Kind == trace.KindSubagent {
			nested++
		}
	}
	if nested != 1 {
		t.Errorf("child run spans nested under a subagent span = %d, want 1 (of %d spans)", nested, len(spans))
	}
	for _, sp := range spans {
		if sp.ParentID != "" {
			if _, ok := byID[sp.ParentID]; !ok {
				t.Errorf("span %s (%s) is orphaned: parent %s is not in the sink", sp.Name, sp.Kind, sp.ParentID)
			}
		}
	}
}

// ===== TrackCost =====

func TestTrackCostChargesOnlyCompletedCalls(t *testing.T) {
	t.Run("a failed call is not charged and emits no Spend", func(t *testing.T) {
		failing := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
			return llm.Response{}, errors.New("upstream died")
		})
		a := newAgent(t, llm.Chain(failing, TrackCost(llm.DefaultPricing)), nil)

		ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{CostUSD: 1})
		defer cancel()

		run := a.Start(ctx, Ask("x"))
		t.Cleanup(func() { _ = run.Close() })
		kinds, _ := drain(t, run)

		if contains(kinds, "spend") {
			t.Errorf("event kinds = %v, want no spend event for a failed call", kinds)
		}
		if got := run.Progress().Calls; got != 0 {
			t.Errorf("Progress.Calls = %d, want 0", got)
		}
	})

	// The response's model wins over the request's, because a gateway can
	// route "claude-opus-5" somewhere else and bill accordingly.
	t.Run("nil pricing falls back to the default table", func(t *testing.T) {
		cl := scripted(llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: "hi"}},
			StopReason: llm.StopEndTurn,
			Usage:      llm.Usage{InputTokens: 1_000_000, OutputTokens: 0},
			// No Model: the request's model is used instead.
		})
		a := newAgent(t, llm.Chain(cl, TrackCost(nil)), nil, WithModel("claude-opus-5"))

		ctx, cancel := governor.WithBudget(context.Background(), governor.Limits{CostUSD: 100})
		defer cancel()

		run := a.Start(ctx, Ask("x"))
		t.Cleanup(func() { _ = run.Close() })
		drain(t, run)

		if got := run.Progress().CostUSD; got <= 0 {
			t.Errorf("Progress.CostUSD = %v, want the default table's price for 1M input tokens", got)
		}
	})
}

// emptyTurnResp is what the gateway actually sent: reasoning, finish_reason
// "stop", no text, no tool call, and no usage chunk at all.
func emptyTurnResp() llm.Response {
	return llm.Response{
		Content:    []llm.ContentBlock{llm.Thinking{Text: "let me work out which stages are idempotent…"}},
		StopReason: llm.StopEndTurn,
	}
}

// TestEmptyTurn covers the failure that made four benchmark episodes look like
// verifier problems.
//
// A reasoning model streamed its scratchpad for forty-five seconds, the stream
// ended with finish_reason "stop", and the response carried no text, no tool
// call and no usage. StopEndTurn said the turn was over, so the loop returned
// Answer{Text: ""} and reported success — for a run in which the model had
// never said anything.
func TestEmptyTurn(t *testing.T) {
	t.Run("is retried, not believed", func(t *testing.T) {
		cl := scripted(emptyTurnResp(), textTurn("Trim and Upper are idempotent; Prefix is not."))
		a := newAgent(t, cl, nil)

		out, err := a.Run(context.Background(), Ask("which stages are idempotent?"))
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		ans, ok := out.(Answer)
		if !ok {
			t.Fatalf("Outcome = %T, want Answer", out)
		}
		if ans.Text == "" {
			t.Error("Answer.Text is empty; the retry's answer was lost")
		}
		if cl.calls() != 2 {
			t.Errorf("calls = %d, want 2 — the empty turn must be retried", cl.calls())
		}
	})

	t.Run("the empty turn is not left in the transcript", func(t *testing.T) {
		// A lone thinking block is worse than useless on the next request:
		// Anthropic rejects one whose signature is missing, and every provider
		// reads it as a turn the model already took. The retry has to re-ask
		// the identical question.
		cl := scripted(emptyTurnResp(), textTurn("done"))
		a := newAgent(t, cl, nil)
		if _, err := a.Run(context.Background(), Ask("go")); err != nil {
			t.Fatalf("Run: %v", err)
		}

		first, second := cl.request(t, 0), cl.request(t, 1)
		if len(second.Messages) != len(first.Messages) {
			t.Errorf("retry sent %d messages, want the original %d — nothing may be pushed",
				len(second.Messages), len(first.Messages))
		}
	})

	t.Run("gives up loudly rather than answering nothing", func(t *testing.T) {
		cl := scripted(emptyTurnResp(), emptyTurnResp(), emptyTurnResp(), emptyTurnResp())
		a := newAgent(t, cl, nil)

		out, err := a.Run(context.Background(), Ask("go"))
		if !errors.Is(err, ErrEmptyTurn) {
			t.Fatalf("err = %v, want ErrEmptyTurn", err)
		}
		if out != nil {
			t.Errorf("Outcome = %#v, want nil — silence is not an answer", out)
		}
		if cl.calls() != maxEmptyTurns+1 {
			t.Errorf("calls = %d, want %d", cl.calls(), maxEmptyTurns+1)
		}
		var ete *EmptyTurnError
		if !errors.As(err, &ete) || ete.StopReason != llm.StopEndTurn {
			t.Errorf("err = %v, want an EmptyTurnError naming the stop reason", err)
		}
	})

	t.Run("the counter is consecutive, not cumulative", func(t *testing.T) {
		// Two empty turns separated by real work must not add up to a failure:
		// a provider hiccuping twice in a long run is not a mute model.
		cl := scripted(
			emptyTurnResp(),
			toolTurn("1", "calc", `{}`),
			emptyTurnResp(),
			emptyTurnResp(),
			textTurn("finished"),
		)
		a := newAgent(t, cl, []tool.Def{echoTool("calc", "14")})

		out, err := a.Run(context.Background(), Ask("go"))
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if ans, ok := out.(Answer); !ok || ans.Text != "finished" {
			t.Errorf("Outcome = %#v, want Answer{finished}", out)
		}
	})

	t.Run("a whitespace-only turn is an empty turn", func(t *testing.T) {
		blank := llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: "  \n\t "}},
			StopReason: llm.StopEndTurn,
		}
		cl := scripted(blank, textTurn("real answer"))
		a := newAgent(t, cl, nil)

		out, err := a.Run(context.Background(), Ask("go"))
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if ans, _ := out.(Answer); ans.Text != "real answer" {
			t.Errorf("Answer = %q, want the retry's answer", ans.Text)
		}
	})

	t.Run("other stop reasons keep their own handling", func(t *testing.T) {
		// max_tokens with no text is empty too, and must NOT be swallowed by
		// the empty-turn path: it has its own retry, which appends a notice so
		// the next request differs, and its own error. See
		// TestTruncatedReplyIsRetriedShorter.
		cl := scripted(
			llm.Response{StopReason: llm.StopMaxTokens},
			llm.Response{StopReason: llm.StopMaxTokens},
			llm.Response{StopReason: llm.StopMaxTokens},
		)
		a := newAgent(t, cl, nil)
		if _, err := a.Run(context.Background(), Ask("go")); !errors.Is(err, ErrMaxTokens) {
			t.Errorf("err = %v, want ErrMaxTokens", err)
		}
	})
}

// TestResumeAnswersAPause covers the case wombat.Then gets wrong.
//
// A run that stopped on a pause tool has a dangling tool_use. Then() closes it
// with "(cancelled)" and appends the text as a new user turn, so the model is
// told its question was withdrawn and then handed the answer with nothing
// saying what it answers. Resume() routes the same text to AnswerPause instead.
func TestResumeAnswersAPause(t *testing.T) {
	set := tool.NewSet(pauseTool("ask_user"), echoTool("calc", "14"))

	paused := []llm.Message{
		llm.UserText("deploy it"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.ToolUse{ID: "ask1", Name: "ask_user", Input: json.RawMessage(`{"question":"which env?"}`)},
		}},
	}

	t.Run("PendingPause finds it", func(t *testing.T) {
		id, ok := PendingPause(paused, set)
		if !ok || id != "ask1" {
			t.Fatalf("PendingPause = (%q, %v), want (ask1, true)", id, ok)
		}
	})

	t.Run("the instruction becomes the answer", func(t *testing.T) {
		in := Resume(paused, "staging", set)
		last := in.Messages[len(in.Messages)-1]
		if last.Role != llm.RoleUser {
			t.Fatalf("last turn is %s, want a user turn carrying the tool_result", last.Role)
		}
		tr, ok := last.Content[0].(llm.ToolResult)
		if !ok {
			t.Fatalf("last block is %T, want a ToolResult", last.Content[0])
		}
		if tr.ToolUseID != "ask1" || tr.Content != "staging" {
			t.Errorf("tool_result = {%s, %q}, want {ask1, \"staging\"}", tr.ToolUseID, tr.Content)
		}
		if err := Convo(in.Messages).Validate(); err != nil {
			t.Errorf("Validate() = %v, want nil", err)
		}
		// The regression: Then() would have put this in the transcript.
		for _, m := range in.Messages {
			for _, b := range m.Content {
				if r, ok := b.(llm.ToolResult); ok && r.Content == "(cancelled)" {
					t.Error("the pause was cancelled instead of answered")
				}
			}
		}
	})

	t.Run("an abandoned tool call is still cancelled", func(t *testing.T) {
		// Not a pause tool, so the run died mid-call; there is no question to
		// answer and the new text really is a new instruction.
		abandoned := []llm.Message{
			llm.UserText("q"),
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.ToolUse{ID: "c1", Name: "calc", Input: json.RawMessage(`{}`)},
			}},
		}
		in := Resume(abandoned, "never mind, do this instead", set)
		if err := Convo(in.Messages).Validate(); err != nil {
			t.Errorf("Validate() = %v, want nil", err)
		}
		found := false
		for _, m := range in.Messages {
			for _, b := range m.Content {
				if r, ok := b.(llm.ToolResult); ok && r.ToolUseID == "c1" && r.Content == "(cancelled)" {
					found = true
				}
			}
		}
		if !found {
			t.Error("the abandoned call was not closed; the provider would reject this")
		}
	})

	t.Run("no instruction just carries on", func(t *testing.T) {
		done := []llm.Message{llm.UserText("q"),
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.Text{Text: "a"}}}}
		if got := Resume(done, "", set); len(got.Messages) != len(done) {
			t.Errorf("Resume with no instruction changed the transcript: %d -> %d",
				len(done), len(got.Messages))
		}
	})

	t.Run("a paused run resumed end to end", func(t *testing.T) {
		// The whole loop: pause, save, resume with the answer, finish.
		cl := scripted(
			llm.Response{
				Content: []llm.ContentBlock{llm.ToolUse{
					ID: "ask1", Name: "ask_user", Input: json.RawMessage(`{"question":"which env?"}`),
				}},
				StopReason: llm.StopToolUse,
			},
			textTurn("deployed to staging"),
		)
		a := newAgent(t, cl, []tool.Def{pauseTool("ask_user"), echoTool("calc", "14")})

		r := a.Start(context.Background(), Ask("deploy it"))
		for r.Next() { //nolint:revive // draining is the point
		}
		if err := r.Err(); err != nil {
			t.Fatalf("first turn: %v", err)
		}
		p, ok := r.Outcome().(Paused)
		if !ok {
			t.Fatalf("Outcome = %T, want Paused", r.Outcome())
		}

		// Exactly what -session persists, round-tripped through JSON.
		saved, err := json.Marshal(r.Messages())
		_ = r.Close()
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var prior []llm.Message
		if err := json.Unmarshal(saved, &prior); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if id, ok := PendingPause(prior, tool.NewSet(a.Tools(context.Background())...)); !ok || id != p.ToolUseID {
			t.Fatalf("the pause did not survive the JSON round trip: (%q, %v)", id, ok)
		}

		out2, err := a.Run(context.Background(),
			Resume(prior, "staging", tool.NewSet(a.Tools(context.Background())...)))
		if err != nil {
			t.Fatalf("second turn: %v", err)
		}
		if ans, ok := out2.(Answer); !ok || ans.Text != "deployed to staging" {
			t.Errorf("Outcome = %#v, want the answer", out2)
		}
	})
}

// TestAnswerPauseClosesSiblings is the bug that bricks a session.
//
// A model that asks a question routinely asks it alongside other work: one
// assistant turn carrying [tool_use grep_search, tool_use ask_user] is ordinary.
// The loop returns Paused the moment it recognises the pause tool — before
// dispatching anything — so the siblings never ran.
//
// Answering only the pause leaves them dangling. Validate rejects the resumed
// transcript, the turn fails before reaching the provider, and a caller
// persisting the conversation saves that same unrepairable transcript, so every
// later turn fails identically.
func TestAnswerPauseClosesSiblings(t *testing.T) {
	prior := []llm.Message{
		llm.UserText("deploy it"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.ToolUse{ID: "g1", Name: "grep_search", Input: json.RawMessage(`{"pattern":"x"}`)},
			llm.ToolUse{ID: "ask1", Name: "ask_user", Input: json.RawMessage(`{"question":"which env?"}`)},
			llm.ToolUse{ID: "g2", Name: "grep_search", Input: json.RawMessage(`{"pattern":"y"}`)},
		}},
	}

	in := AnswerPause(prior, "ask1", "staging")

	if err := Convo(in.Messages).Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil — the resumed transcript must be sendable", err)
	}

	got := map[llm.ToolUseID]llm.ToolResult{}
	for _, b := range in.Messages[len(in.Messages)-1].Content {
		if tr, ok := b.(llm.ToolResult); ok {
			got[tr.ToolUseID] = tr
		}
	}
	if len(got) != 3 {
		t.Fatalf("%d tool_results, want 3 — every call in the turn has to be closed", len(got))
	}
	if a := got["ask1"]; a.Content != "staging" || a.IsError {
		t.Errorf("the pause answer = %#v, want {staging, not an error}", a)
	}
	for _, id := range []llm.ToolUseID{"g1", "g2"} {
		if !got[id].IsError {
			t.Errorf("%s closed as a success; the model cannot tell it never ran", id)
		}
	}

	// And the whole point: doing it twice does not degrade.
	if err := Convo(AnswerPause(in.Messages, "ask1", "again").Messages).Validate(); err == nil {
		t.Error("answering an already-answered pause was accepted; want an orphan or a late result")
	}
}

// TestResumeSurvivesRepeatedTurns: the session-file loop must not accumulate
// damage. A transcript that fails to validate is one that gets saved and then
// fails forever.
func TestResumeSurvivesRepeatedTurns(t *testing.T) {
	set := tool.NewSet(pauseTool("ask_user"), echoTool("calc", "14"))
	msgs := []llm.Message{llm.UserText("go")}

	for turn := 1; turn <= 4; turn++ {
		// Alternate: a plain reply, then a paused one with a live sibling.
		if turn%2 == 1 {
			msgs = Convo(msgs).PushAssistant([]llm.ContentBlock{llm.Text{Text: "ok"}})
		} else {
			msgs = Convo(msgs).PushAssistant([]llm.ContentBlock{
				llm.ToolUse{ID: llm.ToolUseID(fmt.Sprintf("c%d", turn)), Name: "calc", Input: json.RawMessage(`{}`)},
				llm.ToolUse{ID: llm.ToolUseID(fmt.Sprintf("a%d", turn)), Name: "ask_user", Input: json.RawMessage(`{}`)},
			})
		}

		// Round-trip through JSON the way -session does.
		b, err := json.Marshal(msgs)
		if err != nil {
			t.Fatalf("turn %d marshal: %v", turn, err)
		}
		var prior []llm.Message
		if err := json.Unmarshal(b, &prior); err != nil {
			t.Fatalf("turn %d unmarshal: %v", turn, err)
		}

		in := Resume(prior, "next", set)
		if err := Convo(in.Messages).Validate(); err != nil {
			t.Fatalf("turn %d: Validate() = %v — the session is now unrepairable", turn, err)
		}
		msgs = in.Messages
	}
}

// TestEmptyTurnBlamesTheTokenCap covers a real run.
//
// A hard task made a reasoning model want 17,577 output tokens against the
// default cap of 8,192. It streamed reasoning for two minutes, the gateway ran
// out mid-thought, and the response came back with finish_reason "end_turn", no
// content, and no usage at all — the same wire shape as a model that simply had
// nothing to say. The harness retried twice more, spent another four minutes
// reaching the identical wall, and reported "no text and no tool call", which
// is true and useless.
func TestEmptyTurnBlamesTheTokenCap(t *testing.T) {
	const maxTokens = 8192

	// A response that streams a budget's worth of scratchpad and then stops.
	truncated := llm.Response{
		Content:    []llm.ContentBlock{llm.Thinking{Text: "…"}},
		StopReason: llm.StopEndTurn,
	}
	cl := &reasoningClient{resp: truncated, reasoning: strings.Repeat("x", 4*maxTokens)}
	a := newAgent(t, cl, nil, WithMaxTokens(maxTokens))

	_, err := a.Run(context.Background(), Ask("design an NFA simulator"))
	if !errors.Is(err, ErrEmptyTurn) {
		t.Fatalf("err = %v, want ErrEmptyTurn", err)
	}

	var ete *EmptyTurnError
	if !errors.As(err, &ete) {
		t.Fatalf("err = %v, want an EmptyTurnError", err)
	}
	if ete.Attempts != 1 {
		t.Errorf("Attempts = %d, want 1 — an exhausted budget must not be retried, "+
			"the next attempt spends it identically", ete.Attempts)
	}
	if cl.calls != 1 {
		t.Errorf("model called %d times, want 1", cl.calls)
	}
	if ete.MaxTokens != maxTokens {
		t.Errorf("MaxTokens = %d, want %d", ete.MaxTokens, maxTokens)
	}
	for _, want := range []string{"8192", "reasoning", "MaxTokens"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the message does not mention %q:\n%s", want, err)
		}
	}
}

// TestEmptyTurnWithNoReasoningIsStillRetried: a provider hiccup streams nothing,
// and that one is worth another go.
func TestEmptyTurnWithNoReasoningIsStillRetried(t *testing.T) {
	cl := &reasoningClient{
		resp:      llm.Response{StopReason: llm.StopEndTurn},
		reasoning: "", // silent, not truncated
		then:      textTurn("here you go"),
	}
	a := newAgent(t, cl, nil, WithMaxTokens(8192))

	out, err := a.Run(context.Background(), Ask("go"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if ans, _ := out.(Answer); ans.Text != "here you go" {
		t.Errorf("Outcome = %#v, want the retry's answer", out)
	}
}

// reasoningClient streams a fixed amount of scratchpad before answering, which
// is what distinguishes a truncated turn from a silent one.
type reasoningClient struct {
	resp      llm.Response
	reasoning string
	then      llm.Response // returned from the second call, when set
	calls     int
}

func (c *reasoningClient) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	c.calls++
	if c.calls > 1 && c.then.StopReason != "" {
		return c.then, nil
	}
	if req.OnDelta != nil && c.reasoning != "" {
		req.OnDelta(llm.Delta{Reasoning: c.reasoning})
	}
	return c.resp, nil
}

// A truncated reply is retried with a notice, not failed outright.
//
// The old contract failed immediately, reasoning that an identical request
// truncates identically. The reasoning was sound and the conclusion did not
// follow: the request does not have to be identical. This pins the new one,
// including the half of the old contract that is still true — after the retry
// budget, ErrMaxTokens surfaces.
func TestTruncatedReplyIsRetriedShorter(t *testing.T) {
	t.Run("a shorter reply after truncation is accepted", func(t *testing.T) {
		cl := scripted(
			llm.Response{StopReason: llm.StopMaxTokens},
			llm.Response{
				StopReason: llm.StopEndTurn,
				Content:    []llm.ContentBlock{llm.Text{Text: "the short version"}},
			},
		)
		a := newAgent(t, cl, nil)
		out, err := a.Run(context.Background(), Ask("write me an essay"))
		if err != nil {
			t.Fatalf("run failed instead of retrying: %v", err)
		}
		ans, ok := out.(Answer)
		if !ok || ans.Text != "the short version" {
			t.Fatalf("outcome = %#v, want the second reply", out)
		}

		// The retry has to be a DIFFERENT request, or it truncates the same
		// way. The notice is what makes it different.
		reqs := cl.requests()
		if len(reqs) != 2 {
			t.Fatalf("made %d calls, want 2", len(reqs))
		}
		second := llm.TextOf(reqs[1].Messages[len(reqs[1].Messages)-1].Content)
		if !strings.Contains(second, "cut off") {
			t.Errorf("the retry does not tell the model its reply was truncated: %q", second)
		}
		// And it must not leak into the transcript, where it would accumulate
		// one copy per iteration.
		first := llm.TextOf(reqs[0].Messages[len(reqs[0].Messages)-1].Content)
		if strings.Contains(first, "cut off") {
			t.Error("the notice appeared in the first request too; it is not being cleared")
		}
	})

	t.Run("the retry is bounded and the error still surfaces", func(t *testing.T) {
		cl := scripted(
			llm.Response{StopReason: llm.StopMaxTokens},
			llm.Response{StopReason: llm.StopMaxTokens},
			llm.Response{StopReason: llm.StopMaxTokens},
			llm.Response{StopReason: llm.StopMaxTokens},
		)
		a := newAgent(t, cl, nil)
		_, err := a.Run(context.Background(), Ask("go"))
		if !errors.Is(err, ErrMaxTokens) {
			t.Fatalf("err = %v, want ErrMaxTokens once the retries are spent", err)
		}
		if n := cl.calls(); n != maxTruncatedTurns+1 {
			t.Errorf("made %d calls, want %d: one attempt plus %d retries",
				n, maxTruncatedTurns+1, maxTruncatedTurns)
		}
	})

	t.Run("a truncated reply is announced", func(t *testing.T) {
		cl := scripted(
			llm.Response{StopReason: llm.StopMaxTokens},
			llm.Response{StopReason: llm.StopEndTurn, Content: []llm.ContentBlock{llm.Text{Text: "ok"}}},
		)
		a := newAgent(t, cl, nil)
		run := a.Start(context.Background(), Ask("go"))
		defer run.Close()
		seen := 0
		for run.Next() {
			if ev, ok := run.Event().(TurnTruncated); ok {
				seen++
				if ev.Attempt != 1 {
					t.Errorf("attempt = %d, want 1", ev.Attempt)
				}
			}
		}
		if seen != 1 {
			t.Errorf("emitted %d TurnTruncated events, want 1: a retry that costs a full "+
				"generation at the cap must not be invisible", seen)
		}
	})
}
