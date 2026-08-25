package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"time"
)

// ===== Retry =====

// RetryPolicy parameterizes [WithRetry].
type RetryPolicy struct {
	// MaxAttempts is the total number of attempts INCLUDING the first, so 1
	// means "call once, never retry" and 0 or less means the same. Counting
	// total attempts rather than retries is the reading that matches the
	// budget question a caller actually asks: how many times can this cost me?
	MaxAttempts int

	// Base is the first backoff, doubled per attempt.
	Base time.Duration

	// Max caps the doubling. It does NOT cap a provider's Retry-After hint —
	// see [WithRetry].
	Max time.Duration

	// Jitter is the fraction of the delay added at random, clamped to 0..1.
	// It is additive, never subtractive, so a Retry-After hint is honoured as
	// a floor rather than averaged away.
	Jitter float64
}

// DefaultRetryPolicy is four attempts, one second, capped at thirty.
//
// The values are load-bearing. Four attempts with a 1s base spans roughly
// 1+2+4 = 7s of waiting, which covers the overwhelmingly common case — a
// single 429 or 529 during a burst — without leaving an interactive agent
// silent for a minute. The 30s cap matters only when a provider hint pushes
// the backoff up; past that a run should fail and let the caller decide.
//
// Jitter is deliberately large. Sub-agents fan out across goroutines sharing
// one client, so they hit a rate limit within microseconds of each other; with
// no jitter they would retry in lockstep and re-trip it together.
var DefaultRetryPolicy = RetryPolicy{
	MaxAttempts: 4,
	Base:        time.Second,
	Max:         30 * time.Second,
	Jitter:      0.5,
}

// normalized fills in unusable zero values so that a caller who set only
// MaxAttempts still gets sane waits instead of a hot loop. MaxAttempts itself
// is left alone: zero there means "no retry", which is a legitimate request.
func (p RetryPolicy) normalized() RetryPolicy {
	if p.Base <= 0 {
		p.Base = DefaultRetryPolicy.Base
	}
	if p.Max <= 0 {
		p.Max = DefaultRetryPolicy.Max
	}
	if p.Max < p.Base {
		p.Max = p.Base
	}
	switch {
	case p.Jitter < 0:
		p.Jitter = 0
	case p.Jitter > 1:
		p.Jitter = 1
	}
	return p
}

// Backoff returns the deterministic wait before the retry that follows a
// 0-based attempt number: Base doubled attempt times, capped at Max.
//
// Jitter is NOT applied here. Keeping this function pure means a caller can
// reason about — and test — the schedule, while the randomness that actually
// matters for decorrelating concurrent retries is added by [WithRetry].
func (p RetryPolicy) Backoff(attempt int) time.Duration {
	p = p.normalized()
	if attempt <= 0 {
		return p.Base
	}
	// Compare before shifting rather than after: Base<<attempt overflows into
	// a small positive duration for large attempts, which would silently turn
	// a long backoff into a busy retry.
	if attempt >= 63 || p.Base > p.Max>>uint(attempt) {
		return p.Max
	}
	return p.Base << uint(attempt)
}

// delay is the wait actually slept after a failed attempt: the larger of the
// provider's hint and our own backoff, plus jitter.
func (p RetryPolicy) delay(attempt int, hint time.Duration) time.Duration {
	d := p.Backoff(attempt)
	if hint > d {
		d = hint
	}
	if p.Jitter > 0 {
		d += time.Duration(rand.Float64() * p.Jitter * float64(d))
	}
	return d
}

// WithRetry retries transient failures with capped exponential backoff.
//
// Only [Retryable] errors are retried: rate limits, overload, 5xx and
// transport faults. A context-window overflow, a bad request or an auth
// failure comes straight back, because the identical request will fail
// identically and the attempt costs money.
//
// A provider's Retry-After hint wins over the computed backoff when it is
// larger, and is not clipped by Max — the provider is telling us when it will
// serve us again, and ignoring it guarantees another rejection. The sleep is
// cancellable, so an oversized hint cannot outlive the run: a cancelled or
// budget-exhausted context returns context.Cause immediately rather than
// sleeping through the abort.
//
// Streaming caveat: a retried call re-runs Request.OnDelta from the start, so
// a consumer that renders deltas may see the beginning of an answer twice.
// OnDelta is an observability sink, not a transcript; the authoritative text
// is in the returned Response.
func WithRetry(p RetryPolicy) Middleware {
	p = p.normalized()
	return func(next Client) Client {
		return ClientFunc(func(ctx context.Context, req Request) (Response, error) {
			for attempt := 0; ; attempt++ {
				resp, err := next.Complete(ctx, req)
				if err == nil {
					return resp, nil
				}
				if attempt+1 >= p.MaxAttempts || !Retryable(err) {
					return resp, err
				}
				// A cancelled run can surface as a retryable transport error
				// (the HTTP client wraps ctx.Err()). Check the context itself
				// so an abort is never mistaken for a flaky network.
				if ctx.Err() != nil {
					return resp, context.Cause(ctx)
				}
				if err := sleepCtx(ctx, p.delay(attempt, RetryAfter(err))); err != nil {
					return resp, err
				}
			}
		})
	}
}

// sleepCtx waits for d, or returns the cause of the context ending first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return context.Cause(ctx)
	case <-t.C:
		return nil
	}
}

// ===== Validation =====

// WithValidation rejects a malformed exchange at the boundary.
//
// It is a [Middleware] despite the plain signature — `Chain(leaf,
// WithValidation)` compiles — because it takes no configuration and a
// parameterless constructor would be noise.
//
// On the way out it catches the requests a provider answers with a 400: no
// messages, an empty message, an unnamed tool, a tool_choice that names a tool
// the request does not offer. Those are our bug, so they wrap [ErrBadRequest]
// and never reach the wire, where they would cost a round trip and (for large
// prompts) input tokens.
//
// On the way back it catches the one provider answer the agent loop cannot
// act on: StopToolUse with no ToolUse block. Untreated, the loop sees an empty
// tool batch, sends an empty user turn back, and burns iterations until the
// cap. Surfacing it as an error makes the failure legible at the call that
// caused it.
func WithValidation(next Client) Client {
	return ClientFunc(func(ctx context.Context, req Request) (Response, error) {
		if err := validateRequest(req); err != nil {
			return Response{}, err
		}
		resp, err := next.Complete(ctx, req)
		if err != nil {
			return resp, err
		}
		if err := validateResponse(resp); err != nil {
			return resp, err
		}
		return resp, nil
	})
}

func validateRequest(req Request) error {
	if len(req.Messages) == 0 {
		return fmt.Errorf("%w: request has no messages", ErrBadRequest)
	}
	for i, m := range req.Messages {
		if len(m.Content) == 0 {
			return fmt.Errorf("%w: message %d (%s) has no content blocks", ErrBadRequest, i, m.Role)
		}
	}

	names := make(map[string]bool, len(req.Tools))
	for i, t := range req.Tools {
		if t.Name == "" {
			return fmt.Errorf("%w: tool %d has an empty name", ErrBadRequest, i)
		}
		// An empty schema is allowed: it means "not supplied", and the
		// provider encoder substitutes the empty-object schema. Non-empty and
		// not an object is a real mistake — usually a schema serialized as a
		// JSON string, which the provider rejects with an unhelpful message.
		if len(t.InputSchema) > 0 && !isJSONObject(t.InputSchema) {
			return fmt.Errorf("%w: tool %q has an input schema that is not a JSON object", ErrBadRequest, t.Name)
		}
		names[t.Name] = true
	}

	switch req.Choice.Mode {
	case ChoiceTool:
		if req.Choice.Name == "" {
			return fmt.Errorf("%w: tool_choice mode %q needs a tool name", ErrBadRequest, ChoiceTool)
		}
		if !names[req.Choice.Name] {
			return fmt.Errorf("%w: tool_choice names %q, which is not among the %d tools offered", ErrBadRequest, req.Choice.Name, len(req.Tools))
		}
	case ChoiceAny:
		if len(req.Tools) == 0 {
			return fmt.Errorf("%w: tool_choice mode %q with no tools offered", ErrBadRequest, ChoiceAny)
		}
	}
	return nil
}

// validateResponse reports provider bugs.
//
// They are classified as [ErrServer] rather than [ErrBadRequest] for two
// reasons: the fault is not ours, and ErrServer is retryable — sampling is
// nondeterministic, so a second attempt at a malformed reply may well come
// back well-formed. That only happens when validation sits inside retry, which
// is the documented order.
func validateResponse(resp Response) error {
	if resp.StopReason != StopToolUse {
		return nil
	}
	uses := ToolUses(resp.Content)
	if len(uses) == 0 {
		return &APIError{
			Class:   ErrServer,
			Message: "provider returned stop_reason=tool_use with no tool_use block",
		}
	}
	for _, u := range uses {
		// An unnamed or unidentified tool_use is worse than useless: dispatch
		// keys the tool_result off the id, so a blank one produces an orphan
		// result that makes the NEXT turn fail, far from the cause.
		if u.Name == "" || u.ID == "" {
			return &APIError{
				Class:   ErrServer,
				Message: "provider returned a tool_use block with an empty id or name",
			}
		}
	}
	return nil
}

// isJSONObject reports whether b is a syntactically valid JSON object. It
// avoids unmarshalling into a map, which would both allocate and lose the
// author's key order — the property [ToolSpec.InputSchema] exists to preserve.
func isJSONObject(b []byte) bool {
	t := bytes.TrimSpace(b)
	return len(t) > 0 && t[0] == '{' && json.Valid(t)
}

// ===== Logging =====

// WithLogging records the shape of every call. A nil logger means
// [slog.Default].
//
// Never the content. Transcripts carry user data — file contents, credentials
// the user pasted, whatever the tools read — and a diagnostic log is the last
// place that should hold it. What is logged is metadata a support engineer can
// act on: which model, how big, what came back, how long it took.
//
// Requests log at Debug and results at Info, because in a healthy run the
// interesting line is the outcome; the request shape only matters once you are
// already debugging.
//
// Install it outermost, so it reports the verdict a caller actually receives
// rather than each attempt inside [WithRetry].
func WithLogging(l *slog.Logger) Middleware {
	if l == nil {
		l = slog.Default()
	}
	return func(next Client) Client {
		return ClientFunc(func(ctx context.Context, req Request) (Response, error) {
			l.LogAttrs(ctx, slog.LevelDebug, "llm request",
				slog.String("model", req.Model),
				slog.String("purpose", string(req.Purpose)),
				slog.Int("messages", len(req.Messages)),
				slog.Int("tools", len(req.Tools)),
				slog.Int("max_tokens", req.MaxTokens),
				slog.Bool("stream", req.OnDelta != nil),
			)

			start := time.Now()
			resp, err := next.Complete(ctx, req)
			dur := time.Since(start)

			if err != nil {
				l.LogAttrs(ctx, slog.LevelError, "llm call failed",
					slog.String("model", req.Model),
					slog.String("purpose", string(req.Purpose)),
					slog.Duration("dur", dur),
					slog.Bool("retryable", Retryable(err)),
					slog.String("err", err.Error()),
				)
				return resp, err
			}

			model := resp.Model
			if model == "" {
				model = req.Model
			}
			l.LogAttrs(ctx, slog.LevelInfo, "llm response",
				slog.String("model", model),
				slog.String("purpose", string(req.Purpose)),
				slog.String("stop", string(resp.StopReason)),
				slog.Int("in", resp.Usage.InputTokens),
				slog.Int("out", resp.Usage.OutputTokens),
				slog.Int("cache_write", resp.Usage.CacheWriteTokens),
				slog.Int("cache_read", resp.Usage.CacheReadTokens),
				slog.Duration("dur", dur),
			)
			return resp, nil
		})
	}
}

// ===== Model routing =====

// WithModelRouting sends a call to a different client based on
// [Request.Model], matching by longest prefix exactly as [Table] prices — so
// "claude-haiku-4-5" routes a dated "claude-haiku-4-5-20251001" without a
// route per release. An exact key always wins over a prefix.
//
// This is what lets one chain drive an Opus planner and a Haiku executor, or
// mix providers in a single run, while retry, cost tracking and logging stay
// uniform: install the routing layer INNERMOST and every shared behavior
// wraps it.
//
//	client := llm.Chain(
//	    llm.Chain(defaultLeaf, llm.WithModelRouting(map[string]llm.Client{
//	        "claude-": anthropicLeaf,
//	        "gpt-":    openaiLeaf,
//	    })),
//	    wombat.TrackCost(pricing), llm.WithRetry(p), llm.WithLogging(log),
//	)
//
// The corollary is that a routed client bypasses next entirely: any middleware
// installed between the leaf and this layer applies only to unrouted calls.
//
// A request with an empty Model falls through to next, because an empty model
// means "whatever the client defaults to" and the table cannot know what that
// resolves to. Panics on a nil route, which is a construction-time bug.
func WithModelRouting(routes map[string]Client) Middleware {
	// Snapshot: the chain is built once and shared by every goroutine in a
	// fan-out, so the routing table must not be something a caller can still
	// write to.
	table := make(map[string]Client, len(routes))
	for prefix, c := range routes {
		if c == nil {
			panic("llm: WithModelRouting has a nil client for route " + prefix)
		}
		table[prefix] = c
	}

	return func(next Client) Client {
		return ClientFunc(func(ctx context.Context, req Request) (Response, error) {
			if c, ok := longestPrefixMatch(table, req.Model); ok {
				return c.Complete(ctx, req)
			}
			return next.Complete(ctx, req)
		})
	}
}

// longestPrefixMatch finds the entry whose key is the longest prefix of key.
// The empty key never matches: use next for the fallback.
func longestPrefixMatch[T any](m map[string]T, key string) (T, bool) {
	var zero T
	if key == "" {
		return zero, false
	}
	if v, ok := m[key]; ok {
		return v, true
	}
	best, bestLen, found := zero, 0, false
	for prefix, v := range m {
		if prefix == "" || len(prefix) <= bestLen {
			continue
		}
		if len(prefix) <= len(key) && key[:len(prefix)] == prefix {
			best, bestLen, found = v, len(prefix), true
		}
	}
	return best, found
}
