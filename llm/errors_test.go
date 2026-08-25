package llm

import (
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestClassifyStatus(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		body       string
		retryAfter time.Duration
		wantClass  error
	}{
		{"400 plain is a bad request", 400, "invalid parameter 'temperature'", 0, ErrBadRequest},
		{"401 is auth", 401, "invalid x-api-key", 0, ErrAuth},
		{"403 is auth", 403, "forbidden", 0, ErrAuth},
		{"404 is not found", 404, "model not found", 0, ErrNotFound},
		{"422 is a bad request", 422, "unprocessable", 0, ErrBadRequest},
		{"429 is a rate limit", 429, "slow down", 2 * time.Second, ErrRateLimit},
		{"500 is a server error", 500, "internal", 0, ErrServer},
		{"502 is a server error", 502, "bad gateway", 0, ErrServer},
		{"503 is a server error", 503, "unavailable", 0, ErrServer},
		{"529 is Anthropic overload, not a generic 5xx", 529, "overloaded_error", 0, ErrOverloaded},
		{"599 is a server error", 599, "who knows", 0, ErrServer},
		// Anything unclassifiable is our fault by default.
		{"402 falls through to bad request", 402, "payment required", 0, ErrBadRequest},
		{"418 falls through to bad request", 418, "teapot", 0, ErrBadRequest},
		{"304 below 200 falls through", 199, "informational", 0, ErrBadRequest},
		{"0 falls through", 0, "", 0, ErrBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := ClassifyStatus(tt.status, tt.body, tt.retryAfter)
			if e.Class != tt.wantClass {
				t.Errorf("Class: got %v, want %v", e.Class, tt.wantClass)
			}
			if e.Status != tt.status {
				t.Errorf("Status: got %d, want %d", e.Status, tt.status)
			}
			if e.Message != tt.body {
				t.Errorf("Message: got %q, want %q", e.Message, tt.body)
			}
			if e.RetryAfter != tt.retryAfter {
				t.Errorf("RetryAfter: got %v, want %v", e.RetryAfter, tt.retryAfter)
			}
			if !errors.Is(e, tt.wantClass) {
				t.Errorf("errors.Is(err, %v): got false, want true", tt.wantClass)
			}
		})
	}
}

func TestClassifyStatusDetectsContextWindowIn400(t *testing.T) {
	// A context-window overflow arrives as a plain 400 with no distinct code;
	// the only signal is a substring of the body. Getting this wrong makes the
	// overflow-recovery ladder unreachable, so every marker is pinned here.
	tests := []struct {
		name      string
		body      string
		wantClass error
	}{
		{"anthropic prompt too long", "prompt is too long: 300000 tokens > 200000 maximum", ErrContextWindow},
		{"literal context window", "requested tokens exceed the context window of this model", ErrContextWindow},
		{"input is too long", "input is too long for requested model", ErrContextWindow},
		{"exceeds the maximum", "the request exceeds the maximum allowed size", ErrContextWindow},
		{"max_tokens_to_sample", "max_tokens_to_sample: must be less than context length", ErrContextWindow},
		{"too many tokens", "too many tokens in the request", ErrContextWindow},
		{"case insensitive", "PROMPT IS TOO LONG", ErrContextWindow},
		{"mixed case in the middle of a JSON body", `{"error":{"message":"Prompt Is Too Long"}}`, ErrContextWindow},
		{"unrelated 400 stays a bad request", "temperature must be between 0 and 1", ErrBadRequest},
		{"empty body stays a bad request", "", ErrBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := ClassifyStatus(400, tt.body, 0)
			if e.Class != tt.wantClass {
				t.Errorf("Class for body %q: got %v, want %v", tt.body, e.Class, tt.wantClass)
			}
		})
	}
}

// TestClassifyStatusDetectsOpenAIContextOverflow pins a real gap, not a
// hypothetical one.
//
// contextWindowMarkers is written against Anthropic's phrasing. OpenAI (and
// every OpenAI-compatible gateway) reports an oversized prompt as a 400 whose
// body reads "This model's maximum context length is N tokens. However, your
// messages resulted in M tokens." That string contains none of the markers, so
// llm/openai/client.go:426 classifies it as ErrBadRequest — which is NOT
// retryable and, more importantly, is not ErrContextWindow, so
// wombat.WithOverflowRecovery never engages. On OpenAI an overflow is a hard
// run failure instead of a compaction.
//
// The fix is one entry in contextWindowMarkers ("maximum context length"), but
// that is a production change, so this test is skipped rather than made to
// pass by weakening the assertion.
func TestClassifyStatusDetectsOpenAIContextOverflow(t *testing.T) {

	bodies := []string{
		"This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens.",
		`{"error":{"message":"This model's maximum context length is 8192 tokens","code":"context_length_exceeded"}}`,
	}
	for _, body := range bodies {
		e := ClassifyStatus(400, body, 0)
		if e.Class != ErrContextWindow {
			t.Errorf("body %q: got %v, want %v", body, e.Class, ErrContextWindow)
		}
	}
}

func TestClassifyStatusOnlySniffsContextWindowFor400(t *testing.T) {
	// The marker text must not reclassify a 500 or a 429: those are retryable
	// and a context-window error is not, so a false positive here would turn a
	// transient blip into a permanent failure.
	for _, status := range []int{429, 500, 529} {
		e := ClassifyStatus(status, "prompt is too long: 300000 tokens", 0)
		if e.Class == ErrContextWindow {
			t.Errorf("status %d: got ErrContextWindow, want the status-based class", status)
		}
		if !e.Retryable() {
			t.Errorf("status %d: Retryable() got false, want true", status)
		}
	}
}

func TestClassifyStatusPanicsOnSuccess(t *testing.T) {
	for _, status := range []int{200, 201, 204, 299} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("ClassifyStatus(%d): got no panic, want a panic", status)
				}
				if !strings.Contains(fmt.Sprint(r), "success status") {
					t.Errorf("panic message: got %v, want it to mention \"success status\"", r)
				}
			}()
			_ = ClassifyStatus(status, "ok", 0)
		})
	}
}

func TestClassifyTransport(t *testing.T) {
	cause := errors.New("dial tcp: connection refused")
	e := ClassifyTransport(cause)

	if e.Class != ErrTransport {
		t.Errorf("Class: got %v, want %v", e.Class, ErrTransport)
	}
	if e.Status != 0 {
		t.Errorf("Status: got %d, want 0 for a transport failure", e.Status)
	}
	if !errors.Is(e, ErrTransport) {
		t.Error("errors.Is(err, ErrTransport): got false, want true")
	}
	if !errors.Is(e, cause) {
		t.Error("errors.Is(err, cause): got false, want true — Unwrap must expose the transport cause")
	}
	if !e.Retryable() {
		t.Error("Retryable(): got false, want true for a transport failure")
	}
}

func TestAPIErrorUnwrapExposesBothClassAndCause(t *testing.T) {
	cause := errors.New("EOF")
	e := &APIError{Class: ErrServer, Status: 500, Cause: cause}

	got := e.Unwrap()
	if len(got) != 2 {
		t.Fatalf("Unwrap: got %d errors, want 2 (class and cause)", len(got))
	}
	if !errors.Is(e, ErrServer) {
		t.Error("errors.Is(err, ErrServer): got false, want true")
	}
	if !errors.Is(e, cause) {
		t.Error("errors.Is(err, cause): got false, want true")
	}
	if errors.Is(e, ErrRateLimit) {
		t.Error("errors.Is(err, ErrRateLimit): got true, want false")
	}

	noCause := &APIError{Class: ErrAuth, Status: 401}
	if len(noCause.Unwrap()) != 1 {
		t.Errorf("Unwrap with no cause: got %d errors, want 1", len(noCause.Unwrap()))
	}
}

func TestAPIErrorIsFindableThroughWrapping(t *testing.T) {
	// The whole point of the sentinel design: an APIError several fmt.Errorf
	// layers up still answers errors.Is and errors.As correctly.
	inner := &APIError{Class: ErrRateLimit, Status: 429, RetryAfter: 7 * time.Second}
	wrapped := fmt.Errorf("iteration 3: %w", fmt.Errorf("llm call: %w", inner))

	if !errors.Is(wrapped, ErrRateLimit) {
		t.Error("errors.Is(wrapped, ErrRateLimit): got false, want true")
	}
	var ae *APIError
	if !errors.As(wrapped, &ae) {
		t.Fatal("errors.As(wrapped, &*APIError): got false, want true")
	}
	if ae != inner {
		t.Errorf("errors.As target: got %p, want %p", ae, inner)
	}
	if !Retryable(wrapped) {
		t.Error("Retryable(wrapped): got false, want true")
	}
	if got := RetryAfter(wrapped); got != 7*time.Second {
		t.Errorf("RetryAfter(wrapped): got %v, want 7s", got)
	}
}

func TestAPIErrorRetryable(t *testing.T) {
	tests := []struct {
		class error
		want  bool
	}{
		{ErrRateLimit, true},
		{ErrOverloaded, true},
		{ErrServer, true},
		{ErrTransport, true},
		// Retrying these burns money for an identical failure.
		{ErrBadRequest, false},
		{ErrAuth, false},
		{ErrNotFound, false},
		// Context-window overflow is deliberately NOT retryable: the loop has to
		// compact instead. If this ever flips, WithOverflowRecovery is bypassed
		// and the run just retries itself to death.
		{ErrContextWindow, false},
	}
	for _, tt := range tests {
		t.Run(tt.class.Error(), func(t *testing.T) {
			e := &APIError{Class: tt.class}
			if got := e.Retryable(); got != tt.want {
				t.Errorf("(&APIError{Class: %v}).Retryable(): got %v, want %v", tt.class, got, tt.want)
			}
			if got := Retryable(e); got != tt.want {
				t.Errorf("Retryable(%v): got %v, want %v", tt.class, got, tt.want)
			}
		})
	}
}

func TestRetryableAndRetryAfterOnNonAPIErrors(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{"nil", nil},
		{"plain error", errors.New("something went wrong")},
		{"a bare class sentinel is not an APIError", ErrRateLimit},
		{"wrapped plain error", fmt.Errorf("ctx: %w", errors.New("x"))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if Retryable(tt.err) {
				t.Errorf("Retryable(%v): got true, want false (conservative answer)", tt.err)
			}
			if got := RetryAfter(tt.err); got != 0 {
				t.Errorf("RetryAfter(%v): got %v, want 0", tt.err, got)
			}
		})
	}
}

func TestAPIErrorMessage(t *testing.T) {
	tests := []struct {
		name string
		err  *APIError
		want string
	}{
		{
			name: "class only",
			err:  &APIError{Class: ErrServer},
			want: "llm: server error",
		},
		{
			name: "with status",
			err:  &APIError{Class: ErrServer, Status: 500},
			want: "llm: server error (HTTP 500)",
		},
		{
			name: "with retry-after",
			err:  &APIError{Class: ErrRateLimit, Status: 429, RetryAfter: 3 * time.Second},
			want: "llm: rate limited (HTTP 429) [retry-after 3s]",
		},
		{
			name: "with body",
			err:  &APIError{Class: ErrBadRequest, Status: 400, Message: "bad temperature"},
			want: "llm: bad request (HTTP 400): bad temperature",
		},
		{
			name: "transport failure carries the cause and no status",
			err:  &APIError{Class: ErrTransport, Cause: errors.New("connection reset")},
			want: "llm: transport failure: connection reset",
		},
		{
			name: "everything at once",
			err:  &APIError{Class: ErrServer, Status: 503, RetryAfter: time.Second, Message: "down", Cause: errors.New("eof")},
			want: "llm: server error (HTTP 503) [retry-after 1s]: down: eof",
		},
		{
			name: "a zero retry-after is omitted",
			err:  &APIError{Class: ErrRateLimit, Status: 429, RetryAfter: 0},
			want: "llm: rate limited (HTTP 429)",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error():\ngot  %q\nwant %q", got, tt.want)
			}
		})
	}
}

func TestClassSentinelsAreDistinct(t *testing.T) {
	all := []error{
		ErrRateLimit, ErrOverloaded, ErrServer, ErrTransport,
		ErrBadRequest, ErrAuth, ErrNotFound, ErrContextWindow,
	}
	for i, a := range all {
		for j, b := range all {
			if i == j {
				continue
			}
			if errors.Is(a, b) {
				t.Errorf("errors.Is(%v, %v): got true, want false — sentinels must not alias", a, b)
			}
		}
	}
}
