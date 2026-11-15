package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"
)

// countingClient records every call and replays a scripted sequence of
// results. A nil entry in errs means "succeed".
type countingClient struct {
	mu     sync.Mutex
	calls  int
	seen   []Request
	err    error                // returned on every call unless script is set
	script []error              // per-attempt errors, last entry repeats
	onCall func(n int)          // side effect keyed on the 1-based call number
	resp   func(n int) Response // optional response builder
}

func (c *countingClient) Complete(_ context.Context, req Request) (Response, error) {
	c.mu.Lock()
	c.calls++
	n := c.calls
	c.seen = append(c.seen, req)
	c.mu.Unlock()

	if c.onCall != nil {
		c.onCall(n)
	}

	err := c.err
	if c.script != nil {
		i := n - 1
		if i >= len(c.script) {
			i = len(c.script) - 1
		}
		err = c.script[i]
	}
	var resp Response
	if c.resp != nil {
		resp = c.resp(n)
	}
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}

func (c *countingClient) n() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

func retryable() error { return &APIError{Class: ErrOverloaded, Status: 529} }
func permanent() error {
	return &APIError{Class: ErrBadRequest, Status: 400, Message: "bad temperature"}
}
func fastPolicy(attempts int) RetryPolicy {
	return RetryPolicy{MaxAttempts: attempts, Base: time.Millisecond, Max: time.Millisecond, Jitter: 0}
}

// ===== RetryPolicy arithmetic =====

func TestRetryPolicyNormalized(t *testing.T) {
	tests := []struct {
		name string
		in   RetryPolicy
		want RetryPolicy
	}{
		{
			name: "zero base and max get the defaults",
			in:   RetryPolicy{MaxAttempts: 2},
			want: RetryPolicy{MaxAttempts: 2, Base: time.Second, Max: 30 * time.Second},
		},
		{
			name: "MaxAttempts=0 is left alone: no retry is a legitimate request",
			in:   RetryPolicy{},
			want: RetryPolicy{MaxAttempts: 0, Base: time.Second, Max: 30 * time.Second},
		},
		{
			name: "max below base is raised to base",
			in:   RetryPolicy{MaxAttempts: 3, Base: 5 * time.Second, Max: time.Second},
			want: RetryPolicy{MaxAttempts: 3, Base: 5 * time.Second, Max: 5 * time.Second},
		},
		{
			name: "negative jitter clamps to 0",
			in:   RetryPolicy{MaxAttempts: 3, Base: time.Second, Max: time.Second, Jitter: -1},
			want: RetryPolicy{MaxAttempts: 3, Base: time.Second, Max: time.Second, Jitter: 0},
		},
		{
			name: "jitter above 1 clamps to 1",
			in:   RetryPolicy{MaxAttempts: 3, Base: time.Second, Max: time.Second, Jitter: 7},
			want: RetryPolicy{MaxAttempts: 3, Base: time.Second, Max: time.Second, Jitter: 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.in.normalized(); got != tt.want {
				t.Errorf("normalized():\ngot  %+v\nwant %+v", got, tt.want)
			}
		})
	}
}

func TestRetryPolicyBackoff(t *testing.T) {
	p := RetryPolicy{Base: time.Second, Max: 30 * time.Second}
	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{-1, time.Second},
		{0, time.Second},
		{1, 2 * time.Second},
		{2, 4 * time.Second},
		{3, 8 * time.Second},
		{4, 16 * time.Second},
		{5, 30 * time.Second}, // 32s clipped to Max
		{6, 30 * time.Second},
		// A large attempt must saturate at Max, not overflow the shift into a
		// small positive duration and turn the backoff into a hot loop.
		{62, 30 * time.Second},
		{63, 30 * time.Second},
		{1000, 30 * time.Second},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("attempt=%d", tt.attempt), func(t *testing.T) {
			if got := p.Backoff(tt.attempt); got != tt.want {
				t.Errorf("Backoff(%d): got %v, want %v", tt.attempt, got, tt.want)
			}
		})
	}
}

func TestRetryPolicyDelayHonoursHint(t *testing.T) {
	p := RetryPolicy{MaxAttempts: 5, Base: time.Second, Max: 30 * time.Second, Jitter: 0}.normalized()

	t.Run("a larger hint wins over the computed backoff", func(t *testing.T) {
		if got := p.delay(0, 90*time.Second); got != 90*time.Second {
			t.Errorf("delay: got %v, want 90s — a provider hint is not clipped by Max", got)
		}
	})
	t.Run("a smaller hint loses", func(t *testing.T) {
		if got := p.delay(2, time.Millisecond); got != 4*time.Second {
			t.Errorf("delay: got %v, want 4s", got)
		}
	})
	t.Run("no hint uses the backoff", func(t *testing.T) {
		if got := p.delay(1, 0); got != 2*time.Second {
			t.Errorf("delay: got %v, want 2s", got)
		}
	})
	t.Run("jitter is additive, never subtractive", func(t *testing.T) {
		// Additive matters: a hint must stay a floor, so jitter cannot pull the
		// wait back under what the provider asked for.
		j := RetryPolicy{MaxAttempts: 5, Base: time.Second, Max: 30 * time.Second, Jitter: 1}.normalized()
		for range 200 {
			got := j.delay(0, 10*time.Second)
			if got < 10*time.Second {
				t.Fatalf("delay: got %v, want >= the 10s hint", got)
			}
			if got > 20*time.Second {
				t.Fatalf("delay: got %v, want <= 2x the hint with Jitter=1", got)
			}
		}
	})
}

// ===== WithRetry =====

func TestWithRetrySucceedsWithoutRetrying(t *testing.T) {
	c := &countingClient{resp: func(int) Response { return Response{StopReason: StopEndTurn} }}
	cl := Chain(c, WithRetry(fastPolicy(4)))

	resp, err := cl.Complete(context.Background(), Request{})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if resp.StopReason != StopEndTurn {
		t.Errorf("StopReason: got %q, want %q", resp.StopReason, StopEndTurn)
	}
	if c.n() != 1 {
		t.Errorf("calls: got %d, want 1", c.n())
	}
}

func TestWithRetryRecoversFromTransientFailures(t *testing.T) {
	c := &countingClient{
		script: []error{retryable(), retryable(), nil},
		resp:   func(int) Response { return Response{StopReason: StopEndTurn, Model: "m"} },
	}
	cl := Chain(c, WithRetry(fastPolicy(4)))

	resp, err := cl.Complete(context.Background(), Request{})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if resp.Model != "m" {
		t.Errorf("Model: got %q, want %q", resp.Model, "m")
	}
	if c.n() != 3 {
		t.Errorf("calls: got %d, want 3 (two failures then a success)", c.n())
	}
}

func TestWithRetryGivesUpAfterMaxAttempts(t *testing.T) {
	// MaxAttempts counts the FIRST call, so N means N total calls, not 1+N.
	for _, attempts := range []int{1, 2, 3, 5} {
		t.Run(fmt.Sprintf("MaxAttempts=%d", attempts), func(t *testing.T) {
			c := &countingClient{err: retryable()}
			cl := Chain(c, WithRetry(fastPolicy(attempts)))

			_, err := cl.Complete(context.Background(), Request{})
			if !errors.Is(err, ErrOverloaded) {
				t.Errorf("error: got %v, want ErrOverloaded", err)
			}
			if c.n() != attempts {
				t.Errorf("calls: got %d, want %d", c.n(), attempts)
			}
		})
	}
}

func TestWithRetryZeroOrNegativeMaxAttemptsCallsOnce(t *testing.T) {
	// 0 or less means the same as 1: call once, never retry. It must not mean
	// "retry forever".
	for _, attempts := range []int{0, -1} {
		t.Run(fmt.Sprint(attempts), func(t *testing.T) {
			c := &countingClient{err: retryable()}
			cl := Chain(c, WithRetry(fastPolicy(attempts)))

			if _, err := cl.Complete(context.Background(), Request{}); err == nil {
				t.Fatal("Complete: got nil error, want ErrOverloaded")
			}
			if c.n() != 1 {
				t.Errorf("calls: got %d, want 1", c.n())
			}
		})
	}
}

func TestWithRetryNeverRetriesNonRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		// Every one of these fails identically on a second attempt, and the
		// attempt costs input tokens.
		{"bad request", permanent()},
		{"auth", &APIError{Class: ErrAuth, Status: 401}},
		{"not found", &APIError{Class: ErrNotFound, Status: 404}},
		{"context window", &APIError{Class: ErrContextWindow, Status: 400, Message: "prompt is too long"}},
		{"a plain error is conservatively non-retryable", errors.New("who knows")},
		{"a bare sentinel is not an APIError", ErrRateLimit},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &countingClient{err: tt.err}
			cl := Chain(c, WithRetry(fastPolicy(5)))

			_, err := cl.Complete(context.Background(), Request{})
			if !errors.Is(err, tt.err) {
				t.Errorf("error: got %v, want %v", err, tt.err)
			}
			if c.n() != 1 {
				t.Errorf("calls: got %d, want 1 — %s must not be retried", c.n(), tt.name)
			}
		})
	}
}

func TestWithRetryRetriesEveryRetryableClass(t *testing.T) {
	for _, class := range []error{ErrRateLimit, ErrOverloaded, ErrServer, ErrTransport} {
		t.Run(class.Error(), func(t *testing.T) {
			c := &countingClient{err: &APIError{Class: class}}
			cl := Chain(c, WithRetry(fastPolicy(3)))
			if _, err := cl.Complete(context.Background(), Request{}); err == nil {
				t.Fatal("Complete: got nil error, want a failure")
			}
			if c.n() != 3 {
				t.Errorf("calls: got %d, want 3", c.n())
			}
		})
	}
}

func TestWithRetryHonoursRetryAfter(t *testing.T) {
	// The provider hint is a floor: with a 1ms backoff and a 60ms hint, the
	// wait must follow the hint. Ignoring it guarantees another rejection.
	const hint = 60 * time.Millisecond
	c := &countingClient{
		script: []error{&APIError{Class: ErrRateLimit, Status: 429, RetryAfter: hint}, nil},
		resp:   func(int) Response { return Response{StopReason: StopEndTurn} },
	}
	cl := Chain(c, WithRetry(RetryPolicy{MaxAttempts: 2, Base: time.Millisecond, Max: time.Millisecond, Jitter: 0}))

	start := time.Now()
	if _, err := cl.Complete(context.Background(), Request{}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	elapsed := time.Since(start)

	if c.n() != 2 {
		t.Fatalf("calls: got %d, want 2", c.n())
	}
	if elapsed < hint {
		t.Errorf("elapsed: got %v, want >= the %v Retry-After hint (Max must not clip it)", elapsed, hint)
	}
}

func TestWithRetryCancellableSleepReturnsContextCause(t *testing.T) {
	// An oversized Retry-After hint must not outlive the run. A budget-exhausted
	// or cancelled context returns context.Cause immediately instead of
	// sleeping through the abort.
	cause := errors.New("budget exhausted")
	ctx, cancel := context.WithCancelCause(context.Background())
	t.Cleanup(func() { cancel(cause) })

	c := &countingClient{
		err: &APIError{Class: ErrRateLimit, Status: 429, RetryAfter: time.Hour},
		onCall: func(n int) {
			if n == 1 {
				time.AfterFunc(5*time.Millisecond, func() { cancel(cause) })
			}
		},
	}
	cl := Chain(c, WithRetry(RetryPolicy{MaxAttempts: 10, Base: time.Hour, Max: time.Hour, Jitter: 0}))

	start := time.Now()
	_, err := cl.Complete(ctx, Request{})
	elapsed := time.Since(start)

	if !errors.Is(err, cause) {
		t.Errorf("error: got %v, want the context cause %v", err, cause)
	}
	if elapsed > 5*time.Second {
		t.Errorf("elapsed: got %v, want the sleep to be cancelled promptly", elapsed)
	}
	if c.n() != 1 {
		t.Errorf("calls: got %d, want 1 — a cancelled run must not attempt again", c.n())
	}
}

func TestWithRetryAlreadyCancelledContextReturnsCause(t *testing.T) {
	// A cancelled run can surface as a retryable transport error, because the
	// HTTP client wraps ctx.Err(). WithRetry must check the context itself so
	// an abort is never mistaken for a flaky network and retried.
	cause := errors.New("wall clock exceeded")
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(cause)

	c := &countingClient{err: ClassifyTransport(context.Canceled)}
	cl := Chain(c, WithRetry(RetryPolicy{MaxAttempts: 5, Base: time.Hour, Max: time.Hour}))

	_, err := cl.Complete(ctx, Request{})
	if !errors.Is(err, cause) {
		t.Errorf("error: got %v, want the context cause %v", err, cause)
	}
	if c.n() != 1 {
		t.Errorf("calls: got %d, want 1", c.n())
	}
}

func TestWithRetryForwardsTheRequestUnchanged(t *testing.T) {
	// OnDelta is an observability sink; middleware must forward it untouched,
	// and must not influence the bytes sent upstream.
	c := &countingClient{script: []error{retryable(), nil}}
	cl := Chain(c, WithRetry(fastPolicy(3)))

	req := Request{System: "sys", Messages: []Message{UserText("hi")}, Model: "m", OnDelta: func(Delta) {}}
	if _, err := cl.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if len(c.seen) != 2 {
		t.Fatalf("requests seen: got %d, want 2", len(c.seen))
	}
	for i, got := range c.seen {
		if got.System != "sys" || got.Model != "m" || len(got.Messages) != 1 {
			t.Errorf("attempt %d: request was altered: got %+v", i, got)
		}
		if got.OnDelta == nil {
			t.Errorf("attempt %d: OnDelta was dropped", i)
		}
	}
}

func TestSleepCtx(t *testing.T) {
	t.Run("non-positive duration returns immediately", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		// Even a cancelled context returns nil for a zero wait: there is
		// nothing to wait for.
		if err := sleepCtx(ctx, 0); err != nil {
			t.Errorf("sleepCtx(ctx, 0): got %v, want nil", err)
		}
		if err := sleepCtx(ctx, -time.Second); err != nil {
			t.Errorf("sleepCtx(ctx, -1s): got %v, want nil", err)
		}
	})

	t.Run("completes when the timer wins", func(t *testing.T) {
		if err := sleepCtx(context.Background(), time.Millisecond); err != nil {
			t.Errorf("sleepCtx: got %v, want nil", err)
		}
	})

	t.Run("returns the cause when the context wins", func(t *testing.T) {
		cause := errors.New("aborted")
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(cause)
		start := time.Now()
		err := sleepCtx(ctx, time.Hour)
		if !errors.Is(err, cause) {
			t.Errorf("sleepCtx: got %v, want %v", err, cause)
		}
		if d := time.Since(start); d > time.Second {
			t.Errorf("elapsed: got %v, want an immediate return", d)
		}
	})

	t.Run("a plain cancel reports context.Canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := sleepCtx(ctx, time.Hour); !errors.Is(err, context.Canceled) {
			t.Errorf("sleepCtx: got %v, want context.Canceled", err)
		}
	})
}

// ===== WithValidation =====

func TestWithValidationRejectsBadRequests(t *testing.T) {
	oneTool := []ToolSpec{{Name: "calculator", InputSchema: json.RawMessage(`{"type":"object"}`)}}

	tests := []struct {
		name    string
		req     Request
		wantSub string
	}{
		{
			name:    "no messages",
			req:     Request{},
			wantSub: "no messages",
		},
		{
			name:    "a message with no content blocks",
			req:     Request{Messages: []Message{UserText("hi"), {Role: RoleAssistant}}},
			wantSub: "message 1 (assistant) has no content blocks",
		},
		{
			name: "an unnamed tool",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    []ToolSpec{{Name: "", Description: "d"}},
			},
			wantSub: "tool 0 has an empty name",
		},
		{
			name: "a schema serialized as a JSON string",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    []ToolSpec{{Name: "t", InputSchema: json.RawMessage(`"{\"type\":\"object\"}"`)}},
			},
			wantSub: "not a JSON object",
		},
		{
			name: "a schema that is a JSON array",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    []ToolSpec{{Name: "t", InputSchema: json.RawMessage(`[]`)}},
			},
			wantSub: "not a JSON object",
		},
		{
			name: "a syntactically invalid schema",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    []ToolSpec{{Name: "t", InputSchema: json.RawMessage(`{"type":`)}},
			},
			wantSub: "not a JSON object",
		},
		{
			name: "tool_choice=tool with no name",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    oneTool,
				Choice:   ToolChoice{Mode: ChoiceTool},
			},
			wantSub: "needs a tool name",
		},
		{
			name: "tool_choice names a tool that is not offered",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    oneTool,
				Choice:   ForceTool("nonexistent"),
			},
			wantSub: `tool_choice names "nonexistent"`,
		},
		{
			name: "tool_choice=any with no tools",
			req: Request{
				Messages: []Message{UserText("hi")},
				Choice:   ToolChoice{Mode: ChoiceAny},
			},
			wantSub: "with no tools offered",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &countingClient{}
			cl := Chain(c, WithValidation)

			_, err := cl.Complete(context.Background(), tt.req)
			if err == nil {
				t.Fatal("Complete: got nil error, want ErrBadRequest")
			}
			if !errors.Is(err, ErrBadRequest) {
				t.Errorf("error: got %v, want it to wrap ErrBadRequest", err)
			}
			if !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("error: got %q, want it to contain %q", err, tt.wantSub)
			}
			// The point of validating outbound is that the call never costs a
			// round trip or input tokens.
			if c.n() != 0 {
				t.Errorf("upstream calls: got %d, want 0 — a bad request must never reach the wire", c.n())
			}
		})
	}
}

func TestWithValidationAcceptsGoodRequests(t *testing.T) {
	tests := []struct {
		name string
		req  Request
	}{
		{
			name: "plain user turn",
			req:  Request{Messages: []Message{UserText("hi")}},
		},
		{
			name: "an empty schema means not supplied, which is allowed",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    []ToolSpec{{Name: "t"}},
			},
		},
		{
			name: "a schema with leading whitespace is still an object",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    []ToolSpec{{Name: "t", InputSchema: json.RawMessage("  \n{\"type\":\"object\"}")}},
			},
		},
		{
			name: "tool_choice=tool naming an offered tool",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    []ToolSpec{{Name: "calculator"}},
				Choice:   ForceTool("calculator"),
			},
		},
		{
			name: "tool_choice=any with tools",
			req: Request{
				Messages: []Message{UserText("hi")},
				Tools:    []ToolSpec{{Name: "calculator"}},
				Choice:   ToolChoice{Mode: ChoiceAny},
			},
		},
		{
			name: "tool_choice=none needs no tools",
			req: Request{
				Messages: []Message{UserText("hi")},
				Choice:   ToolChoice{Mode: ChoiceNone},
			},
		},
		{
			name: "a tool_result turn",
			req: Request{Messages: []Message{
				UserText("hi"),
				{Role: RoleAssistant, Content: []ContentBlock{ToolUse{ID: "u", Name: "t"}}},
				{Role: RoleUser, Content: []ContentBlock{ToolResult{ToolUseID: "u", Content: "r"}}},
			}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &countingClient{resp: func(int) Response { return Response{StopReason: StopEndTurn} }}
			cl := Chain(c, WithValidation)

			if _, err := cl.Complete(context.Background(), tt.req); err != nil {
				t.Fatalf("Complete: got error %v, want nil", err)
			}
			if c.n() != 1 {
				t.Errorf("upstream calls: got %d, want 1", c.n())
			}
		})
	}
}

func TestWithValidationRejectsBadResponses(t *testing.T) {
	tests := []struct {
		name    string
		resp    Response
		wantErr bool
		wantSub string
	}{
		{
			// Untreated, the loop sees an empty tool batch, sends an empty user
			// turn back, and burns iterations until the cap.
			name:    "stop_reason=tool_use with no tool_use block",
			resp:    Response{StopReason: StopToolUse, Content: []ContentBlock{Text{Text: "I will use a tool"}}},
			wantErr: true,
			wantSub: "no tool_use block",
		},
		{
			name:    "stop_reason=tool_use with no content at all",
			resp:    Response{StopReason: StopToolUse},
			wantErr: true,
			wantSub: "no tool_use block",
		},
		{
			// A blank id produces an orphan tool_result that makes the NEXT turn
			// fail, far from the cause.
			name:    "a tool_use with an empty id",
			resp:    Response{StopReason: StopToolUse, Content: []ContentBlock{ToolUse{Name: "t"}}},
			wantErr: true,
			wantSub: "empty id or name",
		},
		{
			name:    "a tool_use with an empty name",
			resp:    Response{StopReason: StopToolUse, Content: []ContentBlock{ToolUse{ID: "u"}}},
			wantErr: true,
			wantSub: "empty id or name",
		},
		{
			name:    "one good use and one blank one",
			resp:    Response{StopReason: StopToolUse, Content: []ContentBlock{ToolUse{ID: "a", Name: "t"}, ToolUse{ID: "", Name: "t"}}},
			wantErr: true,
			wantSub: "empty id or name",
		},
		{
			name: "a well-formed tool_use passes",
			resp: Response{StopReason: StopToolUse, Content: []ContentBlock{ToolUse{ID: "u", Name: "t"}}},
		},
		{
			name: "an empty end_turn is not our business",
			resp: Response{StopReason: StopEndTurn},
		},
		{
			name: "max_tokens with no blocks is not our business",
			resp: Response{StopReason: StopMaxTokens},
		},
		{
			name: "an unknown stop reason passes through",
			resp: Response{StopReason: StopReason("gateway_invented_this")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &countingClient{resp: func(int) Response { return tt.resp }}
			cl := Chain(c, WithValidation)

			_, err := cl.Complete(context.Background(), Request{Messages: []Message{UserText("hi")}})
			if !tt.wantErr {
				if err != nil {
					t.Fatalf("Complete: got error %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatal("Complete: got nil error, want a provider-bug error")
			}
			// Classified as ErrServer, not ErrBadRequest: the fault is not ours,
			// and ErrServer is retryable, so a resample may come back well formed.
			if !errors.Is(err, ErrServer) {
				t.Errorf("error: got %v, want it to wrap ErrServer", err)
			}
			if !Retryable(err) {
				t.Error("Retryable(err): got false, want true — a malformed reply is worth resampling")
			}
			if !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("error: got %q, want it to contain %q", err, tt.wantSub)
			}
		})
	}
}

func TestWithValidationPassesUpstreamErrorsThrough(t *testing.T) {
	// A transport failure must not be reclassified by response validation.
	upstream := ClassifyTransport(errors.New("connection reset"))
	c := &countingClient{err: upstream}
	cl := Chain(c, WithValidation)

	_, err := cl.Complete(context.Background(), Request{Messages: []Message{UserText("hi")}})
	if !errors.Is(err, ErrTransport) {
		t.Errorf("error: got %v, want ErrTransport", err)
	}
}

func TestWithValidationInsideRetryResamplesAMalformedReply(t *testing.T) {
	// The documented composition: validation INSIDE retry, so a provider bug
	// gets a second sample rather than failing the run.
	calls := 0
	leaf := ClientFunc(func(context.Context, Request) (Response, error) {
		calls++
		if calls == 1 {
			return Response{StopReason: StopToolUse, Content: []ContentBlock{Text{Text: "oops"}}}, nil
		}
		return Response{StopReason: StopToolUse, Content: []ContentBlock{ToolUse{ID: "u", Name: "t"}}}, nil
	})
	cl := Chain(leaf, WithValidation, WithRetry(fastPolicy(3)))

	resp, err := cl.Complete(context.Background(), Request{Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if calls != 2 {
		t.Errorf("calls: got %d, want 2 (the malformed reply was resampled)", calls)
	}
	if len(ToolUses(resp.Content)) != 1 {
		t.Errorf("tool uses: got %d, want 1", len(ToolUses(resp.Content)))
	}
}

func TestIsJSONObject(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{`{}`, true},
		{`{"type":"object"}`, true},
		{"  \t\n{\"a\":1}\n ", true},
		{`[]`, false},
		{`"a string"`, false},
		{`null`, false},
		{`123`, false},
		{``, false},
		{`   `, false},
		{`{"a":`, false},
		{`{} garbage`, false},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			if got := isJSONObject([]byte(tt.in)); got != tt.want {
				t.Errorf("isJSONObject(%q): got %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

// ===== WithModelRouting =====

func namedClient(name string, seen *[]string) Client {
	return ClientFunc(func(_ context.Context, req Request) (Response, error) {
		*seen = append(*seen, name)
		return Response{Model: name}, nil
	})
}

func TestWithModelRouting(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  string
	}{
		{"exact key", "claude-haiku-4-5", "haiku"},
		{"a dated id matches its family prefix", "claude-haiku-4-5-20251001", "haiku"},
		{"the longest prefix wins", "claude-opus-5-20250101", "opus"},
		{"a shorter prefix catches the rest of the family", "claude-sonnet-5", "claude"},
		{"a different provider prefix", "gpt-4o", "openai"},
		// An empty model means "whatever the client defaults to", and the table
		// cannot know what that resolves to.
		{"an empty model falls through to next", "", "fallback"},
		{"an unmatched model falls through to next", "llama-3.3-70b", "fallback"},
		{"a model shorter than every prefix falls through", "gpt", "fallback"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var seen []string
			routes := map[string]Client{
				"claude":           namedClient("claude", &seen),
				"claude-opus-5":    namedClient("opus", &seen),
				"claude-haiku-4-5": namedClient("haiku", &seen),
				"gpt-":             namedClient("openai", &seen),
			}
			next := namedClient("fallback", &seen)
			cl := Chain(next, WithModelRouting(routes))

			resp, err := cl.Complete(context.Background(), Request{Model: tt.model})
			if err != nil {
				t.Fatalf("Complete: got error %v, want nil", err)
			}
			if resp.Model != tt.want {
				t.Errorf("routed to %q, want %q", resp.Model, tt.want)
			}
			if len(seen) != 1 {
				t.Errorf("clients called: got %v, want exactly one", seen)
			}
		})
	}
}

func TestWithModelRoutingBypassesNextEntirely(t *testing.T) {
	// The documented corollary: a routed client bypasses next, so middleware
	// installed between the leaf and this layer applies only to unrouted calls.
	var seen []string
	next := namedClient("fallback", &seen)
	cl := Chain(next, WithModelRouting(map[string]Client{"gpt-": namedClient("openai", &seen)}))

	if _, err := cl.Complete(context.Background(), Request{Model: "gpt-4o"}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if strings.Join(seen, ",") != "openai" {
		t.Errorf("clients called: got %v, want [openai] only", seen)
	}
}

func TestWithModelRoutingEmptyRouteKeyNeverMatches(t *testing.T) {
	// The empty key must not become a catch-all; next is the fallback.
	var seen []string
	cl := Chain(namedClient("fallback", &seen), WithModelRouting(map[string]Client{
		"": namedClient("empty", &seen),
	}))

	resp, err := cl.Complete(context.Background(), Request{Model: "anything"})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if resp.Model != "fallback" {
		t.Errorf("routed to %q, want %q", resp.Model, "fallback")
	}
}

func TestWithModelRoutingSnapshotsTheTable(t *testing.T) {
	// The chain is built once and shared by every goroutine in a fan-out, so a
	// caller must not be able to mutate routing after construction.
	var seen []string
	routes := map[string]Client{"gpt-": namedClient("openai", &seen)}
	cl := Chain(namedClient("fallback", &seen), WithModelRouting(routes))

	routes["gpt-"] = namedClient("hijacked", &seen)
	delete(routes, "gpt-")
	routes["claude"] = namedClient("late", &seen)

	resp, _ := cl.Complete(context.Background(), Request{Model: "gpt-4o"})
	if resp.Model != "openai" {
		t.Errorf("routed to %q, want %q — the table must be a snapshot", resp.Model, "openai")
	}
	resp, _ = cl.Complete(context.Background(), Request{Model: "claude-opus-5"})
	if resp.Model != "fallback" {
		t.Errorf("routed to %q, want %q — a late route must not appear", resp.Model, "fallback")
	}
}

func TestWithModelRoutingPanicsOnNilClient(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("got no panic, want a panic on a nil route")
		}
		if !strings.Contains(fmt.Sprint(r), "nil client for route claude-") {
			t.Errorf("panic message: got %v, want it to name the route", r)
		}
	}()
	_ = WithModelRouting(map[string]Client{"claude-": nil})
}

func TestWithModelRoutingEmptyTable(t *testing.T) {
	var seen []string
	cl := Chain(namedClient("fallback", &seen), WithModelRouting(nil))
	resp, err := cl.Complete(context.Background(), Request{Model: "claude-opus-5"})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if resp.Model != "fallback" {
		t.Errorf("routed to %q, want %q", resp.Model, "fallback")
	}
}

func TestLongestPrefixMatch(t *testing.T) {
	m := map[string]int{"a": 1, "ab": 2, "abcd": 4, "": 99}
	tests := []struct {
		key      string
		want     int
		wantOK   bool
		wantName string
	}{
		{key: "abcde", want: 4, wantOK: true},
		{key: "abcd", want: 4, wantOK: true},
		{key: "abc", want: 2, wantOK: true},
		{key: "ab", want: 2, wantOK: true},
		{key: "a", want: 1, wantOK: true},
		{key: "z", want: 0, wantOK: false},
		{key: "", want: 0, wantOK: false},
	}
	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			got, ok := longestPrefixMatch(m, tt.key)
			if ok != tt.wantOK {
				t.Fatalf("ok: got %v, want %v", ok, tt.wantOK)
			}
			if got != tt.want {
				t.Errorf("value: got %d, want %d", got, tt.want)
			}
		})
	}
}

// ===== WithLogging =====

func logBuffer(t *testing.T) (*bytes.Buffer, *slog.Logger) {
	t.Helper()
	var buf bytes.Buffer
	h := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	return &buf, slog.New(h)
}

func TestWithLoggingNeverLogsMessageContent(t *testing.T) {
	// Transcripts carry file contents, pasted credentials, tool output. A
	// diagnostic log is the last place that should hold any of it, and a log is
	// exactly the artifact that gets shipped to a vendor for support.
	const (
		userSecret     = "SECRET-USER-PROMPT-9f3a"
		toolSecret     = "SECRET-TOOL-RESULT-4b2c"
		thinkingSecret = "SECRET-THINKING-7d1e"
		replySecret    = "SECRET-MODEL-REPLY-1a5b"
		systemSecret   = "SECRET-SYSTEM-PROMPT-3c8d"
		schemaSecret   = "SECRET-SCHEMA-KEY-6e4f"
		toolNameSecret = "internal_deploy_tool"
	)

	buf, log := logBuffer(t)
	leaf := ClientFunc(func(context.Context, Request) (Response, error) {
		return Response{
			Content:    []ContentBlock{Thinking{Text: thinkingSecret}, Text{Text: replySecret}},
			StopReason: StopEndTurn,
			Usage:      Usage{InputTokens: 100, OutputTokens: 20, CacheWriteTokens: 5, CacheReadTokens: 7},
			Model:      "claude-opus-5",
		}, nil
	})
	cl := Chain(leaf, WithLogging(log))

	req := Request{
		System: systemSecret,
		Messages: []Message{
			UserText(userSecret),
			{Role: RoleAssistant, Content: []ContentBlock{Thinking{Text: thinkingSecret, Signature: "sig"}}},
			{Role: RoleUser, Content: []ContentBlock{ToolResult{ToolUseID: "u", Content: toolSecret}}},
		},
		Tools:     []ToolSpec{{Name: toolNameSecret, Description: "deploys", InputSchema: json.RawMessage(`{"` + schemaSecret + `":1}`)}},
		Model:     "claude-opus-5",
		MaxTokens: 4096,
		Purpose:   PurposeExecutor,
	}
	if _, err := cl.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}

	out := buf.String()
	if out == "" {
		t.Fatal("log output: got empty, want request and response lines")
	}
	for _, secret := range []string{userSecret, toolSecret, thinkingSecret, replySecret, systemSecret, schemaSecret} {
		if strings.Contains(out, secret) {
			t.Errorf("log leaked content %q:\n%s", secret, out)
		}
	}
	// Tool NAMES are also content in the sense that matters: they are supplied
	// by MCP servers and extensions, not by this package.
	if strings.Contains(out, toolNameSecret) {
		t.Errorf("log leaked the tool name %q:\n%s", toolNameSecret, out)
	}

	// What it must log: metadata a support engineer can act on.
	for _, want := range []string{
		`model=claude-opus-5`, `purpose=executor`, `messages=3`, `tools=1`,
		`max_tokens=4096`, `stream=false`, `stop=end_turn`, `in=100`, `out=20`,
		`cache_write=5`, `cache_read=7`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("log is missing %q:\n%s", want, out)
		}
	}
}

func TestWithLoggingRecordsFailures(t *testing.T) {
	buf, log := logBuffer(t)
	cl := Chain(&countingClient{err: &APIError{Class: ErrOverloaded, Status: 529}}, WithLogging(log))

	if _, err := cl.Complete(context.Background(), Request{Model: "m", Purpose: PurposePlanner}); err == nil {
		t.Fatal("Complete: got nil error, want ErrOverloaded")
	}

	out := buf.String()
	for _, want := range []string{"level=ERROR", "llm call failed", "retryable=true", "purpose=planner"} {
		if !strings.Contains(out, want) {
			t.Errorf("log is missing %q:\n%s", want, out)
		}
	}
}

func TestWithLoggingReportsRetryableFalseForPermanentFailures(t *testing.T) {
	buf, log := logBuffer(t)
	cl := Chain(&countingClient{err: permanent()}, WithLogging(log))
	_, _ = cl.Complete(context.Background(), Request{})

	if !strings.Contains(buf.String(), "retryable=false") {
		t.Errorf("log is missing retryable=false:\n%s", buf.String())
	}
}

func TestWithLoggingFallsBackToTheRequestModel(t *testing.T) {
	// Some gateways omit the model from the reply; the log must still name one.
	buf, log := logBuffer(t)
	leaf := ClientFunc(func(context.Context, Request) (Response, error) {
		return Response{StopReason: StopEndTurn}, nil // no Model
	})
	cl := Chain(leaf, WithLogging(log))

	if _, err := cl.Complete(context.Background(), Request{Model: "requested-model"}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if !strings.Contains(buf.String(), "model=requested-model") {
		t.Errorf("log is missing the fallback model:\n%s", buf.String())
	}
}

func TestWithLoggingReportsStreaming(t *testing.T) {
	buf, log := logBuffer(t)
	cl := Chain(&countingClient{resp: func(int) Response { return Response{StopReason: StopEndTurn} }}, WithLogging(log))

	_, err := cl.Complete(context.Background(), Request{Model: "m", OnDelta: func(Delta) {}})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if !strings.Contains(buf.String(), "stream=true") {
		t.Errorf("log is missing stream=true:\n%s", buf.String())
	}
}

func TestWithLoggingNilLoggerUsesTheDefault(t *testing.T) {
	buf, log := logBuffer(t)
	prev := slog.Default()
	slog.SetDefault(log)
	t.Cleanup(func() { slog.SetDefault(prev) })

	cl := Chain(&countingClient{resp: func(int) Response { return Response{StopReason: StopEndTurn} }}, WithLogging(nil))
	if _, err := cl.Complete(context.Background(), Request{Model: "m"}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if !strings.Contains(buf.String(), "llm response") {
		t.Errorf("a nil logger should use slog.Default:\n%s", buf.String())
	}
}

func TestWithLoggingOutermostSeesOnlyTheFinalVerdict(t *testing.T) {
	// Install it outermost, so it reports the verdict a caller actually
	// receives rather than each attempt inside WithRetry.
	buf, log := logBuffer(t)
	c := &countingClient{
		script: []error{retryable(), retryable(), nil},
		resp:   func(int) Response { return Response{StopReason: StopEndTurn, Model: "m"} },
	}
	cl := Chain(c, WithRetry(fastPolicy(4)), WithLogging(log))

	if _, err := cl.Complete(context.Background(), Request{Model: "m"}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	out := buf.String()
	if n := strings.Count(out, "llm response"); n != 1 {
		t.Errorf("response lines: got %d, want 1", n)
	}
	if n := strings.Count(out, "llm call failed"); n != 0 {
		t.Errorf("failure lines: got %d, want 0 — the retries are invisible from outside", n)
	}
	if c.n() != 3 {
		t.Errorf("upstream calls: got %d, want 3", c.n())
	}
}
