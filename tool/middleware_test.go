package tool

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
	"unicode/utf8"

	"github.com/hoobean1996/wombat-go/llm"
)

// countingHandler records how many times it was invoked and returns a fixed
// verdict.
type countingHandler struct {
	mu       sync.Mutex
	calls    int
	out      string
	err      error
	perCall  []error // when set, indexed by attempt
	sawInput []string
}

func (c *countingHandler) handle(_ context.Context, _ Def, u llm.ToolUse) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	c.sawInput = append(c.sawInput, string(u.Input))
	if c.perCall != nil {
		if i := c.calls - 1; i < len(c.perCall) {
			if c.perCall[i] != nil {
				return "", c.perCall[i]
			}
			return c.out, nil
		}
	}
	return c.out, c.err
}

func (c *countingHandler) n() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

// ===== WithValidation =====

func TestWithValidation(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"empty object", `{}`, false},
		{"populated object", `{"path":"/tmp/x"}`, false},
		{"nested object", `{"a":{"b":[1,2]}}`, false},
		{"whitespace-padded object", "  {\n\"a\":1}\t", false},
		{"array", `[1,2,3]`, true},
		{"bare string", `"hello"`, true},
		{"number", `42`, true},
		{"bool", `true`, true},
		// null unmarshals into a map without error but leaves it nil, which is
		// exactly the case the explicit probe == nil check exists for.
		{"null", `null`, true},
		{"malformed", `{"a":`, true},
		{"trailing garbage", `{} {}`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := &countingHandler{out: "ran"}
			h := WithValidation(inner.handle)
			out, err := h(context.Background(), Def{Name: "view_file"}, use("1", "view_file", tt.input))

			if tt.wantErr {
				if err == nil {
					t.Fatalf("input %s: err = nil, want an invalid-input error", tt.input)
				}
				if !errors.Is(err, ErrInvalidInput) {
					t.Errorf("input %s: err = %v, want it to match ErrInvalidInput", tt.input, err)
				}
				if !strings.Contains(err.Error(), "view_file") {
					t.Errorf("input %s: err = %q, want it to name the tool", tt.input, err)
				}
				if !strings.Contains(err.Error(), "expects a JSON object") {
					t.Errorf("input %s: err = %q, want it to say what was expected", tt.input, err)
				}
				if out != "" {
					t.Errorf("input %s: out = %q, want %q", tt.input, out, "")
				}
				if inner.n() != 0 {
					t.Errorf("input %s: inner handler ran %d times, want 0", tt.input, inner.n())
				}
				return
			}

			if err != nil {
				t.Fatalf("input %s: err = %v, want nil", tt.input, err)
			}
			if out != "ran" {
				t.Errorf("input %s: out = %q, want %q", tt.input, out, "ran")
			}
			if inner.n() != 1 {
				t.Errorf("input %s: inner handler ran %d times, want 1", tt.input, inner.n())
			}
		})
	}
}

// Empty input means "no arguments" and must pass through: zero-argument tools
// are routinely called with nothing at all.
func TestWithValidationAllowsEmptyInput(t *testing.T) {
	for _, raw := range []json.RawMessage{nil, {}} {
		inner := &countingHandler{out: "ran"}
		out, err := WithValidation(inner.handle)(context.Background(), Def{Name: "current_time"},
			llm.ToolUse{ID: "1", Name: "current_time", Input: raw})
		if err != nil || out != "ran" {
			t.Errorf("input %q: (%q, %v), want (ran, nil)", raw, out, err)
		}
	}
}

// The message previews the offending input so the model can see what it sent,
// but must not paste back a megabyte of it.
func TestWithValidationPreviewsLongInput(t *testing.T) {
	long := `"` + strings.Repeat("x", 5000) + `"`
	_, err := WithValidation((&countingHandler{}).handle)(context.Background(), Def{Name: "t"}, use("1", "t", long))
	if err == nil {
		t.Fatal("err = nil, want an invalid-input error")
	}
	if len(err.Error()) > 400 {
		t.Errorf("len(err) = %d, want the input previewed, not echoed in full", len(err.Error()))
	}
	if !strings.HasSuffix(err.Error(), "...") {
		t.Errorf("err = %q, want it to end with the truncation marker", err)
	}
}

// ===== WithTimeout =====

func TestWithTimeout(t *testing.T) {
	t.Run("no cap when both are zero", func(t *testing.T) {
		inner := &countingHandler{out: "slow but allowed"}
		h := WithTimeout(0)(func(ctx context.Context, d Def, u llm.ToolUse) (string, error) {
			if _, hasDeadline := ctx.Deadline(); hasDeadline {
				t.Error("ctx has a deadline, want none when Def.Timeout and the fallback are both 0")
			}
			return inner.handle(ctx, d, u)
		})
		out, err := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if err != nil || out != "slow but allowed" {
			t.Errorf("(%q, %v), want (slow but allowed, nil)", out, err)
		}
	})

	t.Run("Def.Timeout wins over the fallback", func(t *testing.T) {
		var budget time.Duration
		h := WithTimeout(time.Hour)(func(ctx context.Context, _ Def, _ llm.ToolUse) (string, error) {
			dl, _ := ctx.Deadline()
			budget = time.Until(dl)
			return "", nil
		})
		h(context.Background(), Def{Name: "t", Timeout: 50 * time.Millisecond}, use("1", "t", `{}`))
		if budget > 50*time.Millisecond || budget < 10*time.Millisecond {
			t.Errorf("deadline budget = %v, want ~50ms (Def.Timeout), not the 1h fallback", budget)
		}
	})

	t.Run("the fallback applies when the Def declares none", func(t *testing.T) {
		var budget time.Duration
		h := WithTimeout(40 * time.Millisecond)(func(ctx context.Context, _ Def, _ llm.ToolUse) (string, error) {
			dl, _ := ctx.Deadline()
			budget = time.Until(dl)
			return "", nil
		})
		h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if budget > 40*time.Millisecond || budget <= 0 {
			t.Errorf("deadline budget = %v, want ~40ms (the fallback)", budget)
		}
	})

	// The point of enforcing the cap with a context deadline rather than by
	// abandoning a goroutine: a tool that honours ctx is actually STOPPED.
	t.Run("a tool honouring ctx is really stopped", func(t *testing.T) {
		observed := make(chan struct{})
		d := Def{Name: "sleepy", Timeout: 30 * time.Millisecond}
		h := WithTimeout(0)(func(ctx context.Context, _ Def, _ llm.ToolUse) (string, error) {
			select {
			case <-ctx.Done():
				close(observed)
				return "", ctx.Err()
			case <-time.After(10 * time.Second):
				return "finished", nil
			}
		})

		start := time.Now()
		out, err := h(context.Background(), d, use("1", "sleepy", `{}`))
		elapsed := time.Since(start)

		select {
		case <-observed:
		default:
			t.Error("the tool never saw ctx.Done, want the deadline delivered to it")
		}
		if elapsed > 2*time.Second {
			t.Errorf("elapsed = %v, want it to abort at ~30ms rather than run to completion", elapsed)
		}
		if !errors.Is(err, ErrTimeout) {
			t.Errorf("err = %v, want it to match ErrTimeout", err)
		}
		if !strings.Contains(err.Error(), "sleepy") || !strings.Contains(err.Error(), "30ms") {
			t.Errorf("err = %q, want it to name the tool and its budget", err)
		}
		if out != "" {
			t.Errorf("out = %q, want %q", out, "")
		}
	})

	// A killed shell reports its partial output in the error, and that is
	// usually the most useful thing the model can see. Replacing it with a bare
	// "timeout" throws the evidence away.
	t.Run("the tool's own message is preserved alongside the timeout", func(t *testing.T) {
		d := Def{Name: "shell", Timeout: 20 * time.Millisecond}
		h := WithTimeout(0)(func(ctx context.Context, _ Def, _ llm.ToolUse) (string, error) {
			<-ctx.Done()
			return "", errors.New("command timed out, partial output:\nabc")
		})

		_, err := h(context.Background(), d, use("1", "shell", `{}`))
		if !errors.Is(err, ErrTimeout) {
			t.Fatalf("err = %v, want it to match ErrTimeout", err)
		}
		if !strings.Contains(err.Error(), "abc") {
			t.Errorf("err = %q, want it to keep the tool's partial output %q", err, "abc")
		}
		if !strings.Contains(err.Error(), "tool reported:") {
			t.Errorf("err = %q, want the tool's message marked as such", err)
		}
	})

	// A tool that just hands the context error back should NOT be quoted; that
	// would read as "tool reported: context deadline exceeded", which is noise.
	t.Run("a bare context error is normalised, not quoted", func(t *testing.T) {
		d := Def{Name: "ctxonly", Timeout: 20 * time.Millisecond}
		h := WithTimeout(0)(func(ctx context.Context, _ Def, _ llm.ToolUse) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		})
		_, err := h(context.Background(), d, use("1", "ctxonly", `{}`))
		if !errors.Is(err, ErrTimeout) {
			t.Fatalf("err = %v, want ErrTimeout", err)
		}
		if strings.Contains(err.Error(), "tool reported") {
			t.Errorf("err = %q, want no 'tool reported' clause for a bare context error", err)
		}
	})

	// A parent abort must surface as the parent's cause, not as our deadline.
	t.Run("a cancelled parent surfaces its own cause", func(t *testing.T) {
		errGovernor := errors.New("budget exhausted")
		parent, cancel := context.WithCancelCause(context.Background())
		h := WithTimeout(time.Hour)(func(ctx context.Context, _ Def, _ llm.ToolUse) (string, error) {
			cancel(errGovernor)
			<-ctx.Done()
			return "", ctx.Err()
		})
		defer cancel(nil)

		_, err := h(parent, Def{Name: "t"}, use("1", "t", `{}`))
		if !errors.Is(err, errGovernor) {
			t.Errorf("err = %v, want the parent's cause %v", err, errGovernor)
		}
		if errors.Is(err, ErrTimeout) {
			t.Errorf("err = %v, want it NOT to be reported as our own timeout", err)
		}
	})

	// A tool that failed for its own reasons before the deadline is untouched.
	t.Run("an ordinary failure passes through unwrapped", func(t *testing.T) {
		boom := errors.New("no such file")
		h := WithTimeout(time.Hour)(func(context.Context, Def, llm.ToolUse) (string, error) {
			return "", boom
		})
		_, err := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if err != boom {
			t.Errorf("err = %v, want the tool's own error unchanged", err)
		}
	})
}

// ===== WithRetry =====

// TestWithRetryNeedsBothIdempotentAndRetryable is a regression test. The rule
// is a conjunction and neither half is redundant: a non-idempotent write must
// never be replayed just because the error looked transient (a timed-out write
// may well have landed), and a nil classifier means the tool has declared none
// of its failures worth another go.
func TestWithRetryNeedsBothIdempotentAndRetryable(t *testing.T) {
	always := func(error) bool { return true }

	tests := []struct {
		name        string
		idempotent  bool
		retryable   func(error) bool
		wantCalls   int
		description string
	}{
		{"both set", true, always, 3, "retries to the attempt cap"},
		{"idempotent but no classifier", true, nil, 1, "nil means never"},
		{"classifier but not idempotent", false, always, 1, "a write must not be replayed"},
		{"neither", false, nil, 1, "no retry"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := &countingHandler{err: errors.New("transient")}
			d := Def{Name: "write_thing", Idempotent: tt.idempotent, Retryable: tt.retryable}
			h := WithRetry(RetryPolicy{MaxAttempts: 3, Base: time.Millisecond, Jitter: -1})(inner.handle)

			_, err := h(context.Background(), d, use("1", "write_thing", `{}`))
			if err == nil {
				t.Fatal("err = nil, want the tool's failure")
			}
			if got := inner.n(); got != tt.wantCalls {
				t.Errorf("attempts = %d, want %d (%s)", got, tt.wantCalls, tt.description)
			}
		})
	}
}

func TestWithRetry(t *testing.T) {
	transient := errors.New("transient")

	t.Run("stops as soon as it succeeds", func(t *testing.T) {
		inner := &countingHandler{out: "recovered", perCall: []error{transient, transient, nil, nil}}
		d := Def{Name: "flaky", Idempotent: true, Retryable: func(error) bool { return true }}
		h := WithRetry(RetryPolicy{MaxAttempts: 5, Base: time.Millisecond, Jitter: -1})(inner.handle)

		out, err := h(context.Background(), d, use("1", "flaky", `{}`))
		if err != nil || out != "recovered" {
			t.Fatalf("(%q, %v), want (recovered, nil)", out, err)
		}
		if inner.n() != 3 {
			t.Errorf("attempts = %d, want 3 (two failures then a success)", inner.n())
		}
	})

	t.Run("a classifier that declines stops the loop", func(t *testing.T) {
		permanent := errors.New("no such file")
		inner := &countingHandler{err: permanent}
		d := Def{Name: "read", Idempotent: true, Retryable: func(err error) bool { return errors.Is(err, transient) }}
		h := WithRetry(RetryPolicy{MaxAttempts: 5, Base: time.Millisecond, Jitter: -1})(inner.handle)

		_, err := h(context.Background(), d, use("1", "read", `{}`))
		if !errors.Is(err, permanent) {
			t.Errorf("err = %v, want %v", err, permanent)
		}
		if inner.n() != 1 {
			t.Errorf("attempts = %d, want 1: the classifier declined", inner.n())
		}
	})

	t.Run("MaxAttempts=1 disables retry", func(t *testing.T) {
		inner := &countingHandler{err: transient}
		d := Def{Name: "t", Idempotent: true, Retryable: func(error) bool { return true }}
		h := WithRetry(RetryPolicy{MaxAttempts: 1, Base: time.Millisecond, Jitter: -1})(inner.handle)
		h(context.Background(), d, use("1", "t", `{}`))
		if inner.n() != 1 {
			t.Errorf("attempts = %d, want 1", inner.n())
		}
	})

	// A run aborted mid-backoff must report the cause that stopped it, not the
	// stale tool error it was about to retry past.
	t.Run("cancellation during backoff returns the cause", func(t *testing.T) {
		abort := errors.New("run aborted")
		ctx, cancel := context.WithCancelCause(context.Background())
		defer cancel(nil)

		calls := 0
		d := Def{Name: "t", Idempotent: true, Retryable: func(error) bool { return true }}
		h := WithRetry(RetryPolicy{MaxAttempts: 5, Base: 2 * time.Second, Jitter: -1})(
			func(context.Context, Def, llm.ToolUse) (string, error) {
				calls++
				cancel(abort)
				return "", transient
			})

		start := time.Now()
		out, err := h(ctx, d, use("1", "t", `{}`))
		if !errors.Is(err, abort) {
			t.Errorf("err = %v, want the cancellation cause %v", err, abort)
		}
		if errors.Is(err, transient) {
			t.Errorf("err = %v, want the cause, not the stale tool error", err)
		}
		if out != "" {
			t.Errorf("out = %q, want %q", out, "")
		}
		if calls != 1 {
			t.Errorf("attempts = %d, want 1", calls)
		}
		if elapsed := time.Since(start); elapsed > time.Second {
			t.Errorf("elapsed = %v, want it to abandon the 2s sleep immediately", elapsed)
		}
	})
}

func TestRetryPolicyWithDefaultsAndBackoff(t *testing.T) {
	t.Run("zero fields take the defaults, negative jitter means none", func(t *testing.T) {
		got := RetryPolicy{Jitter: -1}.withDefaults()
		if got.MaxAttempts != DefaultRetryPolicy.MaxAttempts ||
			got.Base != DefaultRetryPolicy.Base ||
			got.Max != DefaultRetryPolicy.Max {
			t.Errorf("withDefaults = %+v, want the defaults for every zero field", got)
		}
		if got.Jitter != -1 {
			t.Errorf("Jitter = %v, want -1 preserved (negative is how you ask for none)", got.Jitter)
		}
	})

	t.Run("zero jitter is indistinguishable from unset and takes the default", func(t *testing.T) {
		if got := (RetryPolicy{}).withDefaults().Jitter; got != DefaultRetryPolicy.Jitter {
			t.Errorf("Jitter = %v, want the default %v", got, DefaultRetryPolicy.Jitter)
		}
	})

	t.Run("doubling, capping and clamping", func(t *testing.T) {
		p := RetryPolicy{MaxAttempts: 10, Base: 100 * time.Millisecond, Max: time.Second, Jitter: -1}
		tests := []struct {
			attempt int
			want    time.Duration
		}{
			{-5, 100 * time.Millisecond}, // negative clamps to attempt 0
			{0, 100 * time.Millisecond},
			{1, 200 * time.Millisecond},
			{2, 400 * time.Millisecond},
			{3, 800 * time.Millisecond},
			{4, time.Second}, // 1600ms capped at Max
			{20, time.Second},
		}
		for _, tt := range tests {
			if got := p.Backoff(tt.attempt); got != tt.want {
				t.Errorf("Backoff(%d) = %v, want %v", tt.attempt, got, tt.want)
			}
		}
	})

	t.Run("jitter stays within the band", func(t *testing.T) {
		p := RetryPolicy{MaxAttempts: 3, Base: time.Second, Max: time.Minute, Jitter: 0.5}
		for i := 0; i < 200; i++ {
			got := p.Backoff(0)
			if got < 500*time.Millisecond || got > 1500*time.Millisecond {
				t.Fatalf("Backoff(0) = %v, want it inside [500ms, 1500ms]", got)
			}
		}
	})
}

// ===== WithCircuitBreaker =====

func TestWithCircuitBreaker(t *testing.T) {
	t.Run("opens after N consecutive failures and fails fast", func(t *testing.T) {
		inner := &countingHandler{err: errors.New("dependency down")}
		h := WithCircuitBreaker(3, time.Minute)(inner.handle)
		d := Def{Name: "search"}

		for i := 1; i <= 3; i++ {
			_, err := h(context.Background(), d, use("1", "search", `{}`))
			if errors.Is(err, ErrCircuitOpen) {
				t.Fatalf("call %d tripped the breaker early, want the tool's own error", i)
			}
		}
		if inner.n() != 3 {
			t.Fatalf("attempts = %d, want 3 before the breaker opens", inner.n())
		}

		_, err := h(context.Background(), d, use("1", "search", `{}`))
		if !errors.Is(err, ErrCircuitOpen) {
			t.Fatalf("call 4: err = %v, want ErrCircuitOpen", err)
		}
		if inner.n() != 3 {
			t.Errorf("attempts = %d, want the 4th call rejected WITHOUT being attempted", inner.n())
		}
		if !strings.Contains(err.Error(), "search") {
			t.Errorf("err = %q, want it to name the tool", err)
		}
	})

	t.Run("state is per tool name", func(t *testing.T) {
		inner := &countingHandler{err: errors.New("down")}
		h := WithCircuitBreaker(2, time.Minute)(inner.handle)
		for i := 0; i < 2; i++ {
			h(context.Background(), Def{Name: "bash"}, use("1", "bash", `{}`))
		}
		if _, err := h(context.Background(), Def{Name: "bash"}, use("1", "bash", `{}`)); !errors.Is(err, ErrCircuitOpen) {
			t.Fatalf("bash: err = %v, want ErrCircuitOpen", err)
		}
		// A broken shell must not silence the file reader.
		if _, err := h(context.Background(), Def{Name: "view_file"}, use("2", "view_file", `{}`)); errors.Is(err, ErrCircuitOpen) {
			t.Error("view_file was rejected by bash's breaker, want a per-tool breaker")
		}
	})

	t.Run("a success closes it", func(t *testing.T) {
		inner := &countingHandler{perCall: []error{
			errors.New("f1"), errors.New("f2"), // 2 of 3
			nil,                                // success resets the count
			errors.New("f3"), errors.New("f4"), // 2 of 3 again
		}, out: "recovered"}
		h := WithCircuitBreaker(3, time.Minute)(inner.handle)
		d := Def{Name: "t"}

		for i := 0; i < 5; i++ {
			if _, err := h(context.Background(), d, use("1", "t", `{}`)); errors.Is(err, ErrCircuitOpen) {
				t.Fatalf("call %d tripped the breaker; the success at call 3 should have reset the count", i+1)
			}
		}
		if inner.n() != 5 {
			t.Errorf("attempts = %d, want all 5 to reach the tool", inner.n())
		}
	})

	t.Run("reopens for business after the cooldown", func(t *testing.T) {
		inner := &countingHandler{perCall: []error{errors.New("down"), errors.New("down")}, out: "back up"}
		h := WithCircuitBreaker(2, 20*time.Millisecond)(inner.handle)
		d := Def{Name: "t"}

		h(context.Background(), d, use("1", "t", `{}`))
		h(context.Background(), d, use("1", "t", `{}`))
		if _, err := h(context.Background(), d, use("1", "t", `{}`)); !errors.Is(err, ErrCircuitOpen) {
			t.Fatalf("err = %v, want ErrCircuitOpen right after the trip", err)
		}

		time.Sleep(40 * time.Millisecond)
		out, err := h(context.Background(), d, use("1", "t", `{}`))
		if err != nil || out != "back up" {
			t.Fatalf("after the cooldown: (%q, %v), want (back up, nil)", out, err)
		}
	})

	t.Run("threshold or cooldown at zero disables it", func(t *testing.T) {
		for _, tc := range []struct {
			name      string
			threshold int
			cooldown  time.Duration
		}{
			{"threshold 0", 0, time.Minute},
			{"cooldown 0", 3, 0},
			{"both negative", -1, -time.Second},
		} {
			t.Run(tc.name, func(t *testing.T) {
				inner := &countingHandler{err: errors.New("down")}
				h := WithCircuitBreaker(tc.threshold, tc.cooldown)(inner.handle)
				for i := 0; i < 10; i++ {
					if _, err := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`)); errors.Is(err, ErrCircuitOpen) {
						t.Fatalf("call %d: breaker tripped, want it disabled", i+1)
					}
				}
				if inner.n() != 10 {
					t.Errorf("attempts = %d, want 10", inner.n())
				}
			})
		}
	})

	// The run is being torn down and the tool is not at fault. Counting these
	// would leave the breaker open at the start of the next run.
	t.Run("a failure with the context already dead is not counted", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		inner := &countingHandler{err: errors.New("cancelled mid-flight")}
		h := WithCircuitBreaker(2, time.Minute)(inner.handle)
		for i := 0; i < 5; i++ {
			if _, err := h(ctx, Def{Name: "t"}, use("1", "t", `{}`)); errors.Is(err, ErrCircuitOpen) {
				t.Fatalf("call %d tripped the breaker on cancellations, want them ignored", i+1)
			}
		}
		// And a live call afterwards still finds a closed breaker.
		if _, err := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`)); errors.Is(err, ErrCircuitOpen) {
			t.Error("the breaker was left open by cancellations, want it closed")
		}
	})
}

// ===== WithDedupRepeats =====

// TestWithDedupRepeatsEnrichesWithoutChangingIdentity: it wraps with %w, so
// errors.Is and the error's identity survive. Only the message the model reads
// changes — a caller matching on the sentinel must not be broken by the notice.
func TestWithDedupRepeatsEnrichesWithoutChangingIdentity(t *testing.T) {
	wall := errors.New("wall")
	inner := &countingHandler{err: wall}
	h := WithDedupRepeats(3)(inner.handle)
	ctx := WithCallStats(context.Background(), NewCallStats())
	d := Def{Name: "stuck"}

	var last error
	for i := 1; i <= 3; i++ {
		_, err := h(ctx, d, use(fmt.Sprint(i), "stuck", `{}`))
		if !errors.Is(err, wall) {
			t.Fatalf("call %d: errors.Is(err, wall) = false, want true", i)
		}
		if i < 3 && err.Error() != "wall" {
			t.Errorf("call %d: err = %q, want the bare %q below the threshold", i, err, "wall")
		}
		last = err
	}

	if last.Error() == "wall" {
		t.Fatalf("err = %q, want it enriched after 3 identical failures", last)
	}
	if !strings.Contains(last.Error(), "[repeat]") {
		t.Errorf("err = %q, want it to carry the [repeat] notice", last)
	}
	if !strings.Contains(last.Error(), "3 times in a row") {
		t.Errorf("err = %q, want it to say how many times", last)
	}
	if !strings.Contains(last.Error(), "stuck") {
		t.Errorf("err = %q, want it to name the tool", last)
	}
	if !errors.Is(last, wall) {
		t.Error("errors.Is(enriched, wall) = false, want the identity preserved through %w")
	}
	// Still advisory: the call was dispatched every time.
	if inner.n() != 3 {
		t.Errorf("attempts = %d, want 3 — dedup must not block the call", inner.n())
	}
}

// TestWithDedupRepeatsIsPerRun is a regression test for a real bug: the
// counters used to be captured in the middleware closure, and the chain is
// built once per Agent. Two runs of one agent pooled their frustration, so run
// B could be told it had hit a wall three times on its first attempt.
func TestWithDedupRepeatsIsPerRun(t *testing.T) {
	wall := errors.New("wall")
	inner := &countingHandler{err: wall}
	h := WithDedupRepeats(3)(inner.handle)
	d := Def{Name: "stuck"}

	hit := func(ctx context.Context, n int) string {
		_, err := h(ctx, d, use(fmt.Sprint(n), "stuck", `{}`))
		return err.Error()
	}

	runA := WithCallStats(context.Background(), NewCallStats())
	runB := WithCallStats(context.Background(), NewCallStats())

	var lastA string
	for i := 1; i <= 3; i++ {
		lastA = hit(runA, i)
	}
	if !strings.Contains(lastA, "[repeat]") {
		t.Fatalf("run A after 3 failures: err = %q, want it escalated", lastA)
	}

	if firstB := hit(runB, 9); firstB != "wall" {
		t.Errorf("run B's FIRST call: err = %q, want the bare %q — run B must inherit nothing", firstB, "wall")
	}
}

func TestWithDedupRepeats(t *testing.T) {
	wall := errors.New("wall")

	t.Run("threshold at zero disables it", func(t *testing.T) {
		inner := &countingHandler{err: wall}
		h := WithDedupRepeats(0)(inner.handle)
		ctx := WithCallStats(context.Background(), NewCallStats())
		for i := 0; i < 10; i++ {
			_, err := h(ctx, Def{Name: "t"}, use("1", "t", `{}`))
			if err.Error() != "wall" {
				t.Fatalf("err = %q, want the bare error with dedup disabled", err)
			}
		}
	})

	t.Run("a different input is a different wall", func(t *testing.T) {
		inner := &countingHandler{err: wall}
		h := WithDedupRepeats(2)(inner.handle)
		ctx := WithCallStats(context.Background(), NewCallStats())
		d := Def{Name: "t"}
		for i := 0; i < 5; i++ {
			_, err := h(ctx, d, use(fmt.Sprint(i), "t", fmt.Sprintf(`{"i":%d}`, i)))
			if strings.Contains(err.Error(), "[repeat]") {
				t.Fatalf("call %d with a distinct input was flagged as a repeat: %q", i, err)
			}
		}
	})

	t.Run("a success clears the counter", func(t *testing.T) {
		inner := &countingHandler{perCall: []error{wall, wall, nil, wall, wall}, out: "fine"}
		h := WithDedupRepeats(3)(inner.handle)
		ctx := WithCallStats(context.Background(), NewCallStats())
		d := Def{Name: "t"}

		var last error
		for i := 0; i < 5; i++ {
			_, last = h(ctx, d, use(fmt.Sprint(i), "t", `{}`))
		}
		if last == nil {
			t.Fatal("last err = nil, want the tool's failure")
		}
		if strings.Contains(last.Error(), "[repeat]") {
			t.Errorf("err = %q, want no escalation: the success at call 3 reset the count", last)
		}
	})

	t.Run("cancellations are not counted", func(t *testing.T) {
		ctx, cancel := context.WithCancel(WithCallStats(context.Background(), NewCallStats()))
		cancel()
		inner := &countingHandler{err: wall}
		h := WithDedupRepeats(2)(inner.handle)
		for i := 0; i < 5; i++ {
			_, err := h(ctx, Def{Name: "t"}, use("1", "t", `{}`))
			if strings.Contains(err.Error(), "[repeat]") {
				t.Fatalf("call %d escalated on a cancelled context: %q", i, err)
			}
		}
	})
}

// CallStatsFrom must never return nil: a chain driven directly in a unit test
// has nothing on the context, and every caller would otherwise nil-check.
// The documented consequence is that dedup simply never fires.
func TestCallStatsFromNeverNil(t *testing.T) {
	if got := CallStatsFrom(context.Background()); got == nil {
		t.Fatal("CallStatsFrom(background) = nil, want a throwaway")
	}
	if got := CallStatsFrom(WithCallStats(context.Background(), nil)); got == nil {
		t.Fatal("CallStatsFrom(ctx with a nil *CallStats) = nil, want a throwaway")
	}

	wall := errors.New("wall")
	inner := &countingHandler{err: wall}
	h := WithDedupRepeats(2)(inner.handle)
	for i := 0; i < 5; i++ {
		_, err := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if err.Error() != "wall" {
			t.Fatalf("call %d: err = %q, want the bare error — a throwaway counts one call and is then garbage", i, err)
		}
	}
}

func TestCallStatsOverflowDropsCounters(t *testing.T) {
	s := NewCallStats()
	for i := 0; i < maxDedupKeys; i++ {
		s.fail(fmt.Sprintf("tool%d", i), "boom")
	}
	if got := len(s.counts); got != maxDedupKeys {
		t.Fatalf("len(counts) = %d, want %d", got, maxDedupKeys)
	}
	// The next distinct call trips the bound and clears the table; the fresh
	// key is then the only one, counted from 1.
	if got := s.fail("overflow", "boom"); got != 1 {
		t.Errorf("fail after overflow = %d, want 1", got)
	}
	if got := len(s.counts); got != 1 {
		t.Errorf("len(counts) = %d, want 1 after the clear", got)
	}
}

// ===== WithTruncation =====

func TestWithTruncation(t *testing.T) {
	t.Run("short output is untouched", func(t *testing.T) {
		h := WithTruncation(100)((&countingHandler{out: "short"}).handle)
		out, err := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if err != nil || out != "short" {
			t.Errorf("(%q, %v), want (short, nil)", out, err)
		}
	})

	t.Run("output exactly at the limit is untouched", func(t *testing.T) {
		exact := strings.Repeat("a", 10)
		h := WithTruncation(10)((&countingHandler{out: exact}).handle)
		out, _ := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if out != exact {
			t.Errorf("out = %q, want it untouched at exactly the limit", out)
		}
	})

	// Splitting a multi-byte rune produces invalid UTF-8, and an invalid
	// tool_result is rejected by the provider on the NEXT call — a failure that
	// surfaces far away from its cause.
	t.Run("cuts on a rune boundary", func(t *testing.T) {
		// A limit of 203 against two-byte runes puts the head cut at 101, which
		// lands mid-rune and must back up to 100. The old version of this used a
		// limit of 5, which no longer
		// clips at all: shrinking twenty bytes to five costs a ninety-byte
		// marker, so Clip leaves it alone rather than returning MORE than it was
		// given. Realistic sizes are what exercise the boundary logic now.
		src := strings.Repeat("é", 500)
		h := WithTruncation(203)((&countingHandler{out: src}).handle)
		out, err := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if !utf8.ValidString(out) {
			t.Errorf("out = %q, want valid UTF-8", out[:min(len(out), 60)])
		}
		if len(out) >= len(src) {
			t.Errorf("len(out) = %d, want less than the %d it was given", len(out), len(src))
		}
		if !strings.Contains(out, "omitted from the middle") {
			t.Errorf("out has no gap marker: %q", out[:min(len(out), 120)])
		}
		// Both halves must land on rune starts; a sweep, so the check does not
		// depend on one lucky alignment.
		for limit := 128; limit < 160; limit++ {
			h := WithTruncation(limit)((&countingHandler{out: src}).handle)
			got, _ := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
			if !utf8.ValidString(got) {
				t.Errorf("limit %d produced invalid UTF-8", limit)
			}
		}
	})

	t.Run("honours exempt", func(t *testing.T) {
		// A half-loaded skill is worse than no skill: the model acts on
		// instructions whose second half it never saw.
		body := strings.Repeat("x", 500)
		h := WithTruncation(10, "load_skill", "read_docs")((&countingHandler{out: body}).handle)

		for _, name := range []string{"load_skill", "read_docs"} {
			out, err := h(context.Background(), Def{Name: name}, use("1", name, `{}`))
			if err != nil {
				t.Fatalf("%s: err = %v, want nil", name, err)
			}
			if out != body {
				t.Errorf("%s: output was truncated (%d bytes), want all %d", name, len(out), len(body))
			}
		}

		out, _ := h(context.Background(), Def{Name: "grep_search"}, use("1", "grep_search", `{}`))
		if !strings.Contains(out, "[truncated") {
			t.Errorf("grep_search: out = %q, want it truncated — only exempt names are spared", out[:min(60, len(out))])
		}
	})

	// Truncating an error would cut off the note WithDedupRepeats appends, and
	// re-wrapping would destroy the identity errors.Is depends on.
	t.Run("errors are left alone", func(t *testing.T) {
		long := errors.New(strings.Repeat("e", 500))
		h := WithTruncation(10)((&countingHandler{err: long}).handle)
		_, err := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if !errors.Is(err, long) {
			t.Fatalf("err = %v, want the identity preserved", err)
		}
		if len(err.Error()) != 500 {
			t.Errorf("len(err) = %d, want 500: errors are never truncated", len(err.Error()))
		}
	})

	t.Run("limit at zero disables it", func(t *testing.T) {
		long := strings.Repeat("x", 5000)
		for _, limit := range []int{0, -1} {
			h := WithTruncation(limit)((&countingHandler{out: long}).handle)
			out, _ := h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
			if out != long {
				t.Errorf("WithTruncation(%d): output was cut, want it disabled", limit)
			}
		}
	})
}

func TestBoundary(t *testing.T) {
	tests := []struct {
		name string
		s    string
		n    int
		want int
	}{
		{"n past the end", "abc", 10, 3},
		{"n at the end", "abc", 3, 3},
		{"ascii mid-string", "abcdef", 3, 3},
		{"zero", "abc", 0, 0},
		{"mid two-byte rune", "éé", 1, 0},
		{"at a two-byte rune start", "éé", 2, 2},
		{"mid three-byte rune", "中文", 1, 0},
		{"mid three-byte rune, later byte", "中文", 4, 3},
		{"at a three-byte rune start", "中文", 3, 3},
		{"four-byte rune", "😀x", 2, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := boundary(tt.s, tt.n)
			if got != tt.want {
				t.Errorf("boundary(%q, %d) = %d, want %d", tt.s, tt.n, got, tt.want)
			}
			if !utf8.ValidString(tt.s[:got]) {
				t.Errorf("boundary(%q, %d) = %d leaves invalid UTF-8 %q", tt.s, tt.n, got, tt.s[:got])
			}
		})
	}
}

// ===== WithLogging =====

func TestWithLogging(t *testing.T) {
	// Diagnostics only: the semantic events go through the observer, and
	// conflating the two is what forces a UI to parse log strings.
	newLogger := func() (*slog.Logger, *bytes.Buffer) {
		var buf bytes.Buffer
		return slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})), &buf
	}

	t.Run("success logs an entry line and an ok line", func(t *testing.T) {
		l, buf := newLogger()
		h := WithLogging(l)((&countingHandler{out: "observation"}).handle)
		if _, err := h(context.Background(), Def{Name: "view_file", Category: "file_io"}, use("1", "view_file", `{"path":"/x"}`)); err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		got := buf.String()
		for _, want := range []string{`level=DEBUG`, `msg="tool call"`, `msg="tool ok"`, `tool=view_file`, `category=file_io`, `ok=true`, `bytes=11`} {
			if !strings.Contains(got, want) {
				t.Errorf("log = %q, want it to contain %q", got, want)
			}
		}
	})

	t.Run("failure logs a warning carrying the error", func(t *testing.T) {
		l, buf := newLogger()
		h := WithLogging(l)((&countingHandler{err: errors.New("no such file")}).handle)
		if _, err := h(context.Background(), Def{Name: "view_file"}, use("1", "view_file", `{}`)); err == nil {
			t.Fatal("err = nil, want the tool's failure")
		}
		got := buf.String()
		for _, want := range []string{`level=WARN`, `msg="tool failed"`, `ok=false`, `no such file`} {
			if !strings.Contains(got, want) {
				t.Errorf("log = %q, want it to contain %q", got, want)
			}
		}
		if strings.Contains(got, `msg="tool ok"`) {
			t.Errorf("log = %q, want no success line", got)
		}
	})

	// A log line is a diagnostic; a tool observation can be megabytes, and the
	// full text already reaches the model and the event stream.
	t.Run("previews rather than dumping", func(t *testing.T) {
		l, buf := newLogger()
		h := WithLogging(l)((&countingHandler{out: strings.Repeat("o", 100_000)}).handle)
		h(context.Background(), Def{Name: "t"}, use("1", "t", `{"q":"`+strings.Repeat("i", 100_000)+`"}`))
		if got := buf.Len(); got > 2000 {
			t.Errorf("log is %d bytes, want the input and output previewed", got)
		}
		if !strings.Contains(buf.String(), "bytes=100000") {
			t.Errorf("log = %q..., want the true byte count reported alongside the preview", preview(buf.String(), 200))
		}
	})

	// A nil logger falls back to slog.Default rather than disabling logging,
	// matching the rest of the harness.
	t.Run("a nil logger falls back to the default", func(t *testing.T) {
		var buf bytes.Buffer
		prev := slog.Default()
		slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
		t.Cleanup(func() { slog.SetDefault(prev) })

		h := WithLogging(nil)((&countingHandler{out: "x"}).handle)
		h(context.Background(), Def{Name: "t"}, use("1", "t", `{}`))
		if !strings.Contains(buf.String(), `msg="tool ok"`) {
			t.Errorf("log = %q, want WithLogging(nil) to write to slog.Default", buf.String())
		}
	})
}

func TestPreview(t *testing.T) {
	if got := preview("short", 100); got != "short" {
		t.Errorf("preview(short, 100) = %q, want %q", got, "short")
	}
	if got := preview("abcdef", 3); got != "abc..." {
		t.Errorf("preview(abcdef, 3) = %q, want %q", got, "abc...")
	}
	if got := preview("中文中文", 4); got != "中..." {
		t.Errorf("preview(中文中文, 4) = %q, want %q", got, "中...")
	}
	if got := preview("abc", 3); got != "abc" {
		t.Errorf("preview(abc, 3) = %q, want it untouched at exactly the limit", got)
	}
}

// TestClipNeverGrowsTheInput: a function called Clip must not return something
// longer than what it was asked to shorten.
//
// It did. Clip(8001 bytes, 8000) returned 8142 — a 141-byte gap marker spent to
// elide a single byte. Every observation between limit and roughly limit+150
// got bigger, at both call sites.
func TestClipNeverGrowsTheInput(t *testing.T) {
	const limit = 8000
	for _, over := range []int{1, 2, 50, 140, 141, 200, 1000, 100000} {
		in := strings.Repeat("a", limit+over)
		got := Clip(in, limit)
		if len(got) > len(in) {
			t.Errorf("Clip(%d bytes, %d) = %d bytes — longer than the input", len(in), limit, len(got))
		}
	}
	// And the same for the head-only path.
	for _, over := range []int{1, 50, 1000} {
		in := strings.Repeat("a", limit+over)
		if got := ClipHead(in, limit); len(got) > len(in) {
			t.Errorf("ClipHead(%d bytes, %d) = %d bytes — longer than the input", len(in), limit, len(got))
		}
	}
}

// TestClipHeadIsContiguous: the head-only variant must never introduce a gap.
//
// It is what view_file uses, and a model that reads a region in order to edit it
// composes a str_replace from what it saw. Two fragments that look adjacent
// produce an old_str spanning the invisible middle, which does not match.
func TestClipHeadIsContiguous(t *testing.T) {
	in := strings.Repeat("line\n", 4000)
	got := ClipHead(in, 1000)

	body, marker, found := strings.Cut(got, "\n\n[truncated")
	if !found {
		t.Fatalf("no marker in %q", got[max(0, len(got)-80):])
	}
	if !strings.HasPrefix(in, body) {
		t.Error("the kept text is not a prefix of the input")
	}
	if strings.Contains(marker, "omitted from the middle") {
		t.Error("ClipHead used the head-and-tail marker")
	}
	if strings.Contains(body, "[") {
		t.Error("a marker leaked into the body")
	}
}

// TestClipTinyLimitFallsBackToHead: below two useful halves there is nothing to
// split, and a two-byte tail behind a hundred-byte marker is not a tail.
func TestClipTinyLimitFallsBackToHead(t *testing.T) {
	in := strings.Repeat("x", 1000)
	for _, limit := range []int{1, 10, 63, 127} {
		if got := Clip(in, limit); strings.Contains(got, "omitted from the middle") {
			t.Errorf("Clip(_, %d) split the budget; want the head-only fallback", limit)
		}
	}
	if got := Clip(in, 128); !strings.Contains(got, "omitted from the middle") {
		t.Errorf("Clip(_, 128) did not split; 2*minClipSide should be enough")
	}
}
