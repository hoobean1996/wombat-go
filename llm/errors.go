package llm

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// Failure classes. Match with errors.Is:
//
//	if errors.Is(err, llm.ErrContextWindow) { compactAndRetry() }
//
// They are sentinels rather than a Kind enum so that callers can compare
// without importing a type, and so that a wrapped error still answers
// correctly several layers up.
var (
	ErrRateLimit     = errors.New("llm: rate limited")
	ErrOverloaded    = errors.New("llm: overloaded")
	ErrServer        = errors.New("llm: server error")
	ErrTransport     = errors.New("llm: transport failure")
	ErrBadRequest    = errors.New("llm: bad request")
	ErrAuth          = errors.New("llm: authentication failed")
	ErrNotFound      = errors.New("llm: not found")
	ErrContextWindow = errors.New("llm: context window exceeded")
)

// APIError is a failed model call. Class is one of the sentinels above;
// Unwrap exposes both it and any transport cause, so errors.Is works for
// either.
type APIError struct {
	Class      error
	Status     int           // 0 for transport failures
	Message    string        // response body, truncated by the caller
	RetryAfter time.Duration // 0 when the provider gave no hint
	Cause      error         // underlying transport error, if any
}

// Error implements error.
func (e *APIError) Error() string {
	var b strings.Builder
	b.WriteString(e.Class.Error())
	if e.Status != 0 {
		fmt.Fprintf(&b, " (HTTP %d)", e.Status)
	}
	if e.RetryAfter > 0 {
		fmt.Fprintf(&b, " [retry-after %s]", e.RetryAfter)
	}
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}
	if e.Cause != nil {
		b.WriteString(": ")
		b.WriteString(e.Cause.Error())
	}
	return b.String()
}

// Unwrap exposes the class sentinel and the transport cause.
func (e *APIError) Unwrap() []error {
	if e.Cause != nil {
		return []error{e.Class, e.Cause}
	}
	return []error{e.Class}
}

// Retryable reports whether waiting and trying again could plausibly help.
//
// Context-window overflow is deliberately NOT retryable: the same request
// will fail identically. The agent loop has to compact or split instead.
func (e *APIError) Retryable() bool {
	switch e.Class {
	case ErrRateLimit, ErrOverloaded, ErrServer, ErrTransport:
		return true
	default:
		return false
	}
}

// Retryable reports whether err is a retryable API failure. Non-APIError
// values are treated as non-retryable, which is the conservative answer.
func Retryable(err error) bool {
	var ae *APIError
	if errors.As(err, &ae) {
		return ae.Retryable()
	}
	return false
}

// RetryAfter returns the provider's backoff hint, or 0.
func RetryAfter(err error) time.Duration {
	var ae *APIError
	if errors.As(err, &ae) {
		return ae.RetryAfter
	}
	return 0
}

// contextWindowMarkers are substrings that turn a generic 400 into a
// context-window diagnosis. Providers do not give this a distinct status.
//
// The list has to span dialects, and getting that wrong is expensive in a way
// that is invisible: an overflow classified as a plain bad request is not
// retryable, so wombat.WithOverflowRecovery never engages and a run that
// could have compacted and carried on simply dies. The first six entries are
// Anthropic's phrasings; the rest are OpenAI's and are shared by every
// OpenAI-compatible gateway, which is most of them.
var contextWindowMarkers = []string{
	// Anthropic
	"context window",
	"input is too long",
	"prompt is too long",
	"exceeds the maximum",
	"max_tokens_to_sample",
	"too many tokens",

	// OpenAI and compatible gateways. The prose form is
	// "This model's maximum context length is N tokens. However, your
	// messages resulted in M tokens"; the machine-readable form is the
	// error code, which some gateways send instead of any prose at all.
	"maximum context length",
	"context_length_exceeded",
	"reduce the length of the messages",
}

// ClassifyStatus buckets a non-2xx response. Panics on a success status:
// calling it there is a bug in the caller, not a runtime condition.
func ClassifyStatus(status int, body string, retryAfter time.Duration) *APIError {
	e := &APIError{Status: status, Message: body, RetryAfter: retryAfter}
	switch {
	case status >= 200 && status < 300:
		panic(fmt.Sprintf("llm.ClassifyStatus called on success status %d", status))
	case status == 400:
		lower := strings.ToLower(body)
		e.Class = ErrBadRequest
		for _, m := range contextWindowMarkers {
			if strings.Contains(lower, m) {
				e.Class = ErrContextWindow
				break
			}
		}
	case status == 401, status == 403:
		e.Class = ErrAuth
	case status == 404:
		e.Class = ErrNotFound
	case status == 422:
		e.Class = ErrBadRequest
	case status == 429:
		e.Class = ErrRateLimit
	case status == 529: // Anthropic-specific
		e.Class = ErrOverloaded
	case status >= 500:
		e.Class = ErrServer
	default:
		e.Class = ErrBadRequest
	}
	return e
}

// ClassifyTransport wraps a network/DNS/TLS/timeout failure.
func ClassifyTransport(cause error) *APIError {
	return &APIError{Class: ErrTransport, Cause: cause}
}
