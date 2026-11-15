package wire_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"
	"time"

	"wombat/eval/wire"
	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/llm/openai"
)

// agentOn builds the REAL OpenAI client against the fault server. Nothing here
// substitutes an llm.Client: the point is to exercise the HTTP and SSE code
// that ships, which a scripted client never reaches.
func agentOn(t *testing.T, s *wire.Server, opts ...core.Option) *core.Agent {
	t.Helper()
	c, err := openai.New(openai.Config{
		APIKey:    s.APIKey,
		BaseURL:   s.BaseURL,
		Model:     "wire-model",
		MaxTokens: 256,
	})
	if err != nil {
		t.Fatalf("building the client: %v", err)
	}
	base := []core.Option{
		core.WithClient(c),
		core.WithLogger(slog.New(slog.NewTextHandler(io.Discard, nil))),
		core.WithMaxIters(10),
		// On the AGENT, not only on the client. The reply cap the empty-turn
		// heuristic reasons about is the agent's — it is what goes into
		// llm.Request.MaxTokens and therefore what was actually asked for.
		// Setting only openai.Config.MaxTokens leaves the agent at its 8192
		// default, the request carries 8192, and a test that expected the
		// smaller number would be asserting against a cap nobody used.
		core.WithMaxTokens(256),
	}
	a, err := core.New(append(base, opts...)...)
	if err != nil {
		t.Fatalf("building the agent: %v", err)
	}
	return a
}

func messageCount(t *testing.T, c wire.Captured) int {
	t.Helper()
	msgs, ok := c.Body["messages"].([]any)
	if !ok {
		t.Fatalf("request %s carried no messages array", c.Behavior)
	}
	return len(msgs)
}

// The flagship. run.go's empty-turn handling was written for a shape nothing in
// this repository has ever produced on the wire: a stream that finishes with
// finish_reason "stop", no content, no tool call, and no usage chunk at all.
// Every existing test hands the loop a synthetic llm.Response instead, which
// skips the decoder entirely.
func TestEmptyTurnIsRetriedAndThenAnswers(t *testing.T) {
	s := wire.Start([]wire.Behavior{
		wire.EmptyNoUsage(),
		wire.EmptyNoUsage(),
		wire.Success("the real answer"),
	}, wire.Options{})
	defer s.Close()

	out, err := agentOn(t, s).Run(context.Background(), core.Ask("hello"))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	ans, ok := out.(core.Answer)
	if !ok {
		t.Fatalf("outcome = %T, want Answer", out)
	}
	if !strings.Contains(ans.Text, "the real answer") {
		t.Errorf("answer = %q, want the third turn's text", ans.Text)
	}
	if got := s.Consumed(); got != 3 {
		t.Errorf("consumed %d behaviors, want 3 (two empties then the answer)", got)
	}
	if got := s.Rejected(); got != 0 {
		t.Errorf("%d requests were rejected; the adapter and this server disagree on the protocol", got)
	}
}

// The retry is bounded, and the bound is what stops a genuinely mute model
// from spinning for free.
func TestEmptyTurnGivesUpAfterThreeAttempts(t *testing.T) {
	s := wire.Start([]wire.Behavior{
		wire.EmptyNoUsage(), wire.EmptyNoUsage(), wire.EmptyNoUsage(), wire.EmptyNoUsage(),
	}, wire.Options{})
	defer s.Close()

	_, err := agentOn(t, s).Run(context.Background(), core.Ask("hello"))
	if !errors.Is(err, core.ErrEmptyTurn) {
		t.Fatalf("err = %v, want ErrEmptyTurn", err)
	}
	if got := s.Consumed(); got != 3 {
		t.Errorf("made %d calls, want 3: two retries and then the verdict", got)
	}
}

// The discrimination that makes the retry safe. A turn that streamed a
// budget's worth of reasoning and then stopped is not a provider hiccup — the
// request is unchanged, so the next attempt spends the same budget the same
// way. It must fail on the FIRST call, not the third.
//
// The loop can only tell the two apart by how much reasoning streamed, which
// means the deltas have to have actually arrived through the decoder. That is
// exactly what a substituted llm.Client cannot test.
func TestReasoningBudgetExhaustedIsNotRetried(t *testing.T) {
	// agentOn caps the agent at 256 tokens and the loop's ratio is 2 chars of
	// reasoning per token of budget, so 4000 chars is comfortably over the
	// line at 512.
	s := wire.Start([]wire.Behavior{
		wire.EmptyAfterReasoning(4000),
		wire.Success("should never be reached"),
	}, wire.Options{})
	defer s.Close()

	_, err := agentOn(t, s).Run(context.Background(), core.Ask("hard question"))
	if !errors.Is(err, core.ErrEmptyTurn) {
		t.Fatalf("err = %v, want ErrEmptyTurn", err)
	}
	if got := s.Consumed(); got != 1 {
		t.Errorf("made %d calls, want 1: a budget spent on reasoning must not be retried", got)
	}
	if !strings.Contains(err.Error(), "reasoning") {
		t.Errorf("error does not mention reasoning, so it cannot tell the operator what to change: %v", err)
	}
}

// A stream that dies half way still leaves the user having watched text
// arrive. The transcript has to keep it, or "stop, then continue" — the most
// ordinary thing a chat UI does — has nothing to continue from.
func TestPartialAnswerSurvivesAMidStreamReset(t *testing.T) {
	s := wire.Start([]wire.Behavior{wire.PartialThenReset("half an answer before the wire died")},
		wire.Options{})
	defer s.Close()

	run := agentOn(t, s).Start(context.Background(), core.Ask("hello"))
	defer run.Close()
	for run.Next() { //nolint:revive // draining
	}
	if run.Err() == nil {
		t.Fatal("run succeeded; a reset mid-stream is a failure")
	}

	msgs := run.Messages()
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleAssistant {
		t.Fatalf("transcript ends with a %s turn; the partial answer was dropped", last.Role)
	}
	if got := llm.TextOf(last.Content); !strings.Contains(got, "half an answer") {
		t.Errorf("kept text = %q, want the streamed prefix", got)
	}
}

// Overflow is the one error the harness can actually fix, and it can only fix
// it if the error is CLASSIFIED as overflow. Providers give it no distinct
// status, so the whole mechanism hangs on recognising the prose in a 400 body.
// Get that wrong and the error is a plain bad request, which is not retryable,
// so the ladder never engages and a long run simply dies.
func TestContextOverflowEngagesTheLadder(t *testing.T) {
	s := wire.Start([]wire.Behavior{
		wire.ContextOverflow(),
		wire.Success("answered after shrinking"),
	}, wire.Options{})
	defer s.Close()

	c, err := openai.New(openai.Config{
		APIKey: s.APIKey, BaseURL: s.BaseURL, Model: "wire-model", MaxTokens: 256,
	})
	if err != nil {
		t.Fatal(err)
	}
	a, err := core.New(
		core.WithClient(llm.Chain(c, core.WithOverflowRecovery())),
		core.WithLogger(slog.New(slog.NewTextHandler(io.Discard, nil))),
		core.WithMaxIters(4),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Long enough that the first rung of the default ladder — a 24-message
	// window — has something to remove.
	var prior []llm.Message
	for i := range 20 {
		prior = append(prior,
			llm.UserText(strings.Repeat("q", 40)),
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.Text{Text: strings.Repeat("a", 40)},
			}},
		)
		_ = i
	}
	if _, err := a.Run(context.Background(), core.Then(prior, "the new question")); err != nil {
		t.Fatalf("run failed: %v", err)
	}

	reqs := s.Requests()
	if len(reqs) != 2 {
		t.Fatalf("made %d calls, want 2: the rejection and the shrunken retry", len(reqs))
	}
	before, after := messageCount(t, reqs[0]), messageCount(t, reqs[1])
	t.Logf("ladder shrank the request from %d messages to %d", before, after)
	if after >= before {
		t.Errorf("retry carried %d messages against the original %d; the ladder did not shrink anything",
			after, before)
	}
}

// A 429 is transient and must be retried; the class has to survive the wire.
func TestRateLimitIsRetried(t *testing.T) {
	s := wire.Start([]wire.Behavior{
		wire.RateLimit(0),
		wire.Success("answered after the backoff"),
	}, wire.Options{})
	defer s.Close()

	c, err := openai.New(openai.Config{
		APIKey: s.APIKey, BaseURL: s.BaseURL, Model: "wire-model", MaxTokens: 256,
	})
	if err != nil {
		t.Fatal(err)
	}
	policy := llm.DefaultRetryPolicy
	policy.Base = time.Millisecond
	a, err := core.New(
		core.WithClient(llm.Chain(c, llm.WithRetry(policy))),
		core.WithLogger(slog.New(slog.NewTextHandler(io.Discard, nil))),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Run(context.Background(), core.Ask("hello")); err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if got := s.Consumed(); got != 2 {
		t.Errorf("consumed %d behaviors, want 2", got)
	}
}

// The script must not shift when a request is rejected, or one adapter bug
// turns into a cascade of unrelated failures with the real cause several
// assertions back.
func TestRejectedRequestsDoNotConsumeTheScript(t *testing.T) {
	s := wire.Start([]wire.Behavior{wire.Success("first")}, wire.Options{})
	defer s.Close()

	bad, err := openai.New(openai.Config{
		APIKey: "wrong-key", BaseURL: s.BaseURL, Model: "wire-model", MaxTokens: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	badAgent, err := core.New(
		core.WithClient(bad),
		core.WithLogger(slog.New(slog.NewTextHandler(io.Discard, nil))),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := badAgent.Run(context.Background(), core.Ask("hello")); err == nil {
		t.Fatal("a call with the wrong bearer token succeeded")
	}
	if got := s.Consumed(); got != 0 {
		t.Errorf("a rejected request consumed %d behaviors, want 0", got)
	}
	if got := s.Rejected(); got != 1 {
		t.Errorf("rejected count = %d, want 1", got)
	}

	// The script is still where it was, so the good client gets behavior 0.
	out, err := agentOn(t, s).Run(context.Background(), core.Ask("hello"))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if ans, ok := out.(core.Answer); !ok || !strings.Contains(ans.Text, "first") {
		t.Errorf("outcome = %#v, want the first scripted behavior", out)
	}
}
