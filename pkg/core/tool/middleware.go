package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand/v2"
	"sync"
	"time"
	"unicode/utf8"

	"wombat/pkg/core/llm"
)

// Failure classes produced by the middleware itself, as opposed to by a tool.
// Match with errors.Is; every layer below wraps with %w, so the identity
// survives enrichment by WithDedupRepeats.
var (
	// ErrInvalidInput is a call whose arguments are not a JSON object. It is a
	// [CallerFault]: the tool never ran, and nothing about it is broken.
	ErrInvalidInput = CallerError(errors.New("tool: invalid input"))

	// ErrTimeout is a call that outran its wall-clock budget.
	ErrTimeout = errors.New("tool: timeout")

	// ErrCircuitOpen is a call rejected without being attempted, because the
	// same tool has just failed repeatedly. It is transient by nature — the
	// request was never the problem, the clock is — so a tool's Retryable
	// classifier may treat errors.Is(err, ErrCircuitOpen) as worth retrying
	// later. Nothing downstream retries it immediately: the breaker sits
	// OUTSIDE WithRetry precisely so a trip is not burned through by the
	// retry loop it exists to interrupt.
	ErrCircuitOpen = errors.New("tool: circuit breaker open")
)

// ===== Retry policy =====

// RetryPolicy parameterises exponential backoff with jitter.
//
// A zero field means "use the default", so a caller can override one knob
// without restating the others:
//
//	tool.WithRetry(tool.RetryPolicy{MaxAttempts: 5})
//
// Jitter is the one exception worth knowing: because zero is indistinguishable
// from unset, a NEGATIVE Jitter is how you ask for none. Deterministic backoff
// is almost never what you want when a batch of tool calls fails together —
// they would all wake at the same instant and hammer the same failing
// dependency.
type RetryPolicy struct {
	// MaxAttempts is the total number of calls, not the number of extra ones.
	// 1 disables retry.
	MaxAttempts int

	// Base is the first delay; each further attempt doubles it.
	Base time.Duration

	// Max caps a single delay after doubling, before jitter.
	Max time.Duration

	// Jitter is the fraction of the delay randomised in each direction: 0.5
	// spreads a 1s delay over [0.5s, 1.5s]. Negative disables it.
	Jitter float64
}

// DefaultRetryPolicy is the policy used for any field left zero.
//
// The values are tuned for tools, not for the model API: tool failures worth
// retrying are usually a flaky subprocess or a socket, which recover in
// milliseconds, so Base is short and Max is low enough that a stuck retry loop
// cannot eat a meaningful slice of the run's wall clock.
var DefaultRetryPolicy = RetryPolicy{
	MaxAttempts: 3,
	Base:        200 * time.Millisecond,
	Max:         10 * time.Second,
	Jitter:      0.5,
}

func (p RetryPolicy) withDefaults() RetryPolicy {
	if p.MaxAttempts <= 0 {
		p.MaxAttempts = DefaultRetryPolicy.MaxAttempts
	}
	if p.Base <= 0 {
		p.Base = DefaultRetryPolicy.Base
	}
	if p.Max <= 0 {
		p.Max = DefaultRetryPolicy.Max
	}
	if p.Jitter == 0 {
		p.Jitter = DefaultRetryPolicy.Jitter
	}
	return p
}

// Backoff returns the delay before the attempt after attempt (0-based), with
// zero fields defaulted and jitter already applied. It is exported so a tool
// that wants to pace its own internal retries can reuse the harness's curve.
func (p RetryPolicy) Backoff(attempt int) time.Duration {
	p = p.withDefaults()
	if attempt < 0 {
		attempt = 0
	}

	d := float64(p.Base) * math.Pow(2, float64(attempt))
	if d > float64(p.Max) {
		d = float64(p.Max)
	}
	if p.Jitter > 0 {
		// rand/v2's global source is safe for concurrent use, which matters:
		// one chain is shared by every goroutine of a fanned-out dispatch.
		d *= 1 + p.Jitter*(2*rand.Float64()-1)
	}
	if d < 0 {
		d = 0
	}
	return time.Duration(d)
}

// ===== Validation =====

// WithValidation rejects arguments that are not a JSON object.
//
// It is a Handler decorator rather than a Middleware factory because it takes
// no configuration; it is still usable anywhere a Middleware is, since the
// signatures are identical.
//
// Both providers send an object whenever the input schema declares one, so a
// non-object here means a malformed or hallucinated call. Catching it at the
// boundary gives the model a clearer message than whatever the tool's own
// decoder would have produced. Empty input is allowed and means "no
// arguments": zero-argument tools are routinely called with nothing at all,
// and [Typed] already treats that as the zero value.
func WithValidation(next Handler) Handler {
	return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
		if len(use.Input) > 0 {
			var probe map[string]json.RawMessage
			if err := json.Unmarshal(use.Input, &probe); err != nil || probe == nil {
				return "", fmt.Errorf("%w: tool %s expects a JSON object, got %s",
					ErrInvalidInput, d.Name, preview(string(use.Input), inputPreview))
			}
		}
		return next(ctx, d, use)
	}
}

// ===== Timeout =====

// WithTimeout caps a call's wall clock at Def.Timeout, falling back to the
// argument when the tool declares none. When both are zero there is no cap at
// all — an unbounded tool is a deliberate choice, not an accident.
//
// The cap is a context deadline, so a tool that honours ctx is actually
// STOPPED: exec.CommandContext kills the child, an http.Request is aborted, a
// select on ctx.Done returns. This is one of the concrete wins over the OCaml
// original, which ran the tool on a thread it had no safe way to interrupt and
// simply abandoned it on timeout — the tool kept running, kept holding its
// file handles and sockets, and the middleware was excluded from the default
// chain anyway because effect handlers do not cross threads.
//
// Position matters: this sits INSIDE WithRetry, so the budget is per attempt.
func WithTimeout(fallback time.Duration) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
			budget := d.Timeout
			if budget <= 0 {
				budget = fallback
			}
			if budget <= 0 {
				return next(ctx, d, use)
			}

			cause := fmt.Errorf("%w: tool %s exceeded its %s budget", ErrTimeout, d.Name, budget)
			ctx, cancel := context.WithTimeoutCause(ctx, budget, cause)
			defer cancel()

			out, err := next(ctx, d, use)
			if err != nil && ctx.Err() != nil {
				// Normalise: a tool may report a cancelled context as anything
				// it likes (or wrap it), and callers upstream need one shape to
				// match on. context.Cause distinguishes our deadline from a
				// governor abort of the parent, and returns the right sentinel
				// either way.
				//
				// But keep what the tool managed to say. A killed shell command
				// reports its partial output in the error, and that is usually
				// the most useful thing the model can see — replacing it with a
				// bare "timeout" throws away the evidence.
				cause := context.Cause(ctx)
				if msg := err.Error(); msg != "" &&
					!errors.Is(err, cause) &&
					!errors.Is(err, context.DeadlineExceeded) &&
					!errors.Is(err, context.Canceled) {
					return "", fmt.Errorf("%w (tool reported: %s)", cause, msg)
				}
				return "", cause
			}
			return out, err
		}
	}
}

// ===== Retry =====

// WithRetry re-runs a failed call with exponential backoff.
//
// Two conditions must BOTH hold, and neither is redundant:
//
//   - Def.Idempotent — a non-idempotent write must never be replayed merely
//     because the error looked transient. A timed-out write may well have
//     landed.
//   - Def.Retryable != nil and Def.Retryable(err) — the tool itself decides
//     which of its failures are worth another go. Nil means never, which is
//     the conservative default: most tool errors ("no such file", "missing
//     field") are perfectly deterministic.
//
// A provider hint (llm.RetryAfter, present when the tool wrapped an HTTP call)
// raises the delay but never lowers it. The sleep is cancellable; if the run
// is aborted mid-backoff the cancellation cause is returned rather than the
// stale tool error, because the cause is what actually stopped the run.
func WithRetry(p RetryPolicy) Middleware {
	p = p.withDefaults()
	return func(next Handler) Handler {
		return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
			if !d.Idempotent || d.Retryable == nil {
				return next(ctx, d, use)
			}

			for attempt := 0; ; attempt++ {
				out, err := next(ctx, d, use)
				// A call waiting for a decision or a result has not failed,
				// whatever a classifier says: trying it again would ask twice,
				// or start the work it is waiting on a second time.
				if err == nil || pending(err) || !d.Retryable(err) || attempt+1 >= p.MaxAttempts {
					return out, err
				}

				delay := p.Backoff(attempt)
				if hint := llm.RetryAfter(err); hint > delay {
					delay = hint
				}
				if serr := sleepCtx(ctx, delay); serr != nil {
					return "", serr
				}
			}
		}
	}
}

// sleepCtx waits for d, or returns the cancellation cause if ctx ends first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
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

// ===== Circuit breaker =====

type breakerState struct {
	failures  int
	openUntil time.Time

	// trips counts how many times the breaker has opened. A call notes it on
	// the way in, and an outcome that arrives after the count moved on belongs
	// to the time before the trip; see the bookkeeping in WithCircuitBreaker.
	trips int
}

// WithCircuitBreaker stops calling a tool that keeps failing.
//
// After threshold CONSECUTIVE failures the breaker opens for cooldown and
// further calls fail fast with [ErrCircuitOpen]; one success closes it — a
// success from a call that was already in flight when the breaker opened does
// not, since it says nothing about the tool after the trip. State is per tool
// name, so a broken shell does not silence the file reader.
//
// The state is PER PROCESS, deliberately, and this is the opposite choice from
// [WithDedupRepeats] — read the two together, because it is an easy pair to get
// backwards. A breaker guards an external dependency, and a dependency that is
// down is down for every run: if run A has just established that the search
// index refuses ten connections in a row, run B starting a second later gains
// nothing by rediscovering it, and the cooldown exists precisely to stop that
// rediscovery. Dedup counts something else entirely — how stuck ONE
// conversation is — which is why it lives on the context.
//
// The counters therefore live in this closure, captured when the chain is
// built, and the chain is built once per agent. Two agents get two breakers,
// which is the right granularity: they may well be pointed at different hosts.
//
// Position matters: it sits OUTSIDE WithRetry, so the count is of logical
// failures rather than of attempts — otherwise a single retried call could
// trip a threshold-3 breaker by itself.
//
// A failure that arrives with the context already dead is NOT counted: the run
// is being torn down and the tool is not at fault. Counting those would leave
// the breaker open at the start of the next run that shares the chain.
//
// Neither is a [CallerFault] — a rejected path, a malformed argument, a
// command that exited non-zero. Those report a healthy tool answering a bad
// question, and counting them is how a breaker ends up punishing the one
// participant who can still fix the situation. See [IsCallerFault] for why the
// default runs the other way.
//
// threshold or cooldown at zero disables the breaker entirely.
func WithCircuitBreaker(threshold int, cooldown time.Duration) Middleware {
	if threshold <= 0 || cooldown <= 0 {
		return func(next Handler) Handler { return next }
	}

	// One mutex guards the map and every state in it. The critical sections
	// are a handful of field writes and never span the tool call, so a shared
	// lock costs nothing measurable and cannot deadlock; per-tool locks would
	// buy contention we do not have. The chain is built once and shared by
	// every goroutine of a fanned-out dispatch, so this must be safe.
	var (
		mu       sync.Mutex
		breakers = make(map[string]*breakerState)
	)

	return func(next Handler) Handler {
		return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
			mu.Lock()
			b := breakers[d.Name]
			if b == nil {
				b = &breakerState{}
				breakers[d.Name] = b
			}
			remaining := time.Until(b.openUntil)
			trips := b.trips
			mu.Unlock()

			if remaining > 0 {
				return "", fmt.Errorf("%w: %s has failed %d times in a row; not retrying for another %s",
					ErrCircuitOpen, d.Name, threshold, remaining.Round(time.Millisecond))
			}

			out, err := next(ctx, d, use)

			// Two kinds of failure are neither counted nor forgiven, because
			// neither is evidence either way about the tool's health: one that
			// arrives with the run already being torn down, and one the caller
			// authored. Leaving the counters alone — rather than resetting
			// them — is the point: four real failures, a bad path, then a
			// fifth real failure should still trip a threshold of five.
			// A call waiting for a decision or a result is neither either: it
			// has not finished at all.
			if err != nil && (ctx.Err() != nil || IsCallerFault(err) || pending(err)) {
				return out, err
			}

			mu.Lock()
			switch {
			case b.trips != trips:
				// The breaker opened while this call was in flight, so its
				// outcome predates the trip and is neither counted nor
				// forgiven. Without this a slow call that started before
				// the trip and happened to succeed closed the breaker that
				// its siblings' failures had just opened, and the next call
				// went straight back to the tool the breaker was protecting.
			case err == nil:
				b.failures, b.openUntil = 0, time.Time{}
			default:
				b.failures++
				if b.failures >= threshold {
					b.openUntil = time.Now().Add(cooldown)
					b.failures = 0
					b.trips++
				}
			}
			mu.Unlock()
			return out, err
		}
	}
}

// ===== Dedup =====

// maxDedupKeys bounds the repeat counters. A run can be long — thirty
// iterations of a batching model is a lot of distinct calls — and the stats
// outlive every one of its tool calls, so an unbounded map is a slow leak
// inside a single run. Counters are advisory, so dropping them all on overflow
// costs at most a delayed notice.
const maxDedupKeys = 1024

// CallStats is per-run bookkeeping for the dedup middleware: how many times in
// a row each (tool, input) pair has failed with each error text.
//
// PER RUN, not per process, and that is the correction this type exists to
// make. The chain is built once in core.New and the Agent is shared, so
// counters captured in the middleware closure were shared by every run of that
// agent — two sequential runs, or two concurrent ones, pooled their
// frustration, and run B could be told it had hit a wall three times on its
// first attempt. Being stuck in a loop is a property of ONE conversation.
// Contrast [WithCircuitBreaker], which is per process on purpose.
//
// It rides on the context for the same reason [Info], [Lookup] and
// skill.State do: the Agent is immutable and shared, so anything that changes
// during a run cannot live on it. A sub-agent that inherits the context
// inherits the counters, which is what you want — a loop that spans a fork is
// still one loop.
type CallStats struct {
	// Guarded because one chain serves every goroutine of a fanned-out
	// dispatch. The critical section is a map lookup and an increment, never
	// the tool call itself.
	mu     sync.Mutex
	counts map[string]map[string]int
}

// NewCallStats builds empty counters. One per run; see core.WithRunContext.
func NewCallStats() *CallStats {
	return &CallStats{counts: make(map[string]map[string]int, 8)}
}

type callStatsKey struct{}

// WithCallStats attaches s to ctx.
func WithCallStats(ctx context.Context, s *CallStats) context.Context {
	return context.WithValue(ctx, callStatsKey{}, s)
}

// CallStatsFrom retrieves the counters. It NEVER returns nil.
//
// Outside a run — a unit test driving a chain directly, a REPL, a tool called
// by hand — there is nothing on the context, and every caller would otherwise
// have to nil-check. A throwaway is returned instead, so a chain still works
// anywhere. The consequence is deliberate: a throwaway counts one call and is
// then garbage, so dedup simply never fires. Advisory bookkeeping degrading to
// silence is correct; silently sharing a process-global would not be.
func CallStatsFrom(ctx context.Context) *CallStats {
	if s, ok := ctx.Value(callStatsKey{}).(*CallStats); ok && s != nil {
		return s
	}
	return NewCallStats()
}

// fail records one failure of call with the error text msg, returning how many
// times in a row that exact pair has now been seen.
func (s *CallStats) fail(call, msg string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.counts == nil {
		s.counts = make(map[string]map[string]int, 8)
	}
	if len(s.counts) >= maxDedupKeys {
		clear(s.counts)
	}
	byErr := s.counts[call]
	if byErr == nil {
		byErr = make(map[string]int, 1)
		s.counts[call] = byErr
	}
	n := byErr[msg] + 1
	byErr[msg] = n
	return n
}

// succeed clears every error bucket for call.
//
// Two levels of map rather than one flat key exist for this method: a success
// has to reset the counts for EVERY error this call has produced, and nesting
// turns the OCaml version's prefix sweep over the whole table into one delete.
func (s *CallStats) succeed(call string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.counts, call)
}

// WithDedupRepeats tells the model when it is stuck in a loop.
//
// After threshold consecutive identical (tool name, input, error text) tuples
// it ENRICHES the error message with a note that the same wall has now been
// hit N times, so the model reads it on the next iteration of the ReAct loop
// and pivots — a different tool, different arguments, or giving up honestly.
//
// Advisory only, in two senses that are both load-bearing:
//
//   - It does not block the call. Hard escalation was tried in the OCaml
//     original and rejected: identical errors in a row are exactly what a
//     transient outage looks like, so refusing to dispatch would cut off the
//     retries and the self-correction that would have fixed it.
//   - It wraps with %w, so errors.Is and the error's identity are untouched.
//     Only the message the model reads changes.
//
// A success clears every counter for that (tool, input), so a recovered loop
// does not carry a stale escalation. Cancellations are not counted, for the
// same reason as in the breaker.
//
// The counters are PER RUN, read off the context via [CallStatsFrom] — see
// [CallStats] for why, and [WithCircuitBreaker] for the deliberately opposite
// choice. The middleware itself stays stateless, so one chain built at
// construction time still serves every run.
func WithDedupRepeats(threshold int) Middleware {
	if threshold <= 0 {
		return func(next Handler) Handler { return next }
	}

	return func(next Handler) Handler {
		return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
			out, err := next(ctx, d, use)
			if err != nil && (ctx.Err() != nil || pending(err)) {
				return out, err
			}

			call := d.Name + "|" + string(use.Input)

			stats := CallStatsFrom(ctx)
			var n int
			if err == nil {
				stats.succeed(call)
			} else {
				n = stats.fail(call, err.Error())
			}

			if n < threshold {
				return out, err
			}
			return out, fmt.Errorf("%w\n\n[repeat] this exact call (tool=%s, same input, same error) "+
				"has now failed %d times in a row — try a different tool or different arguments, "+
				"or stop and report the failure", err, d.Name, n)
		}
	}
}

// ===== Truncation =====

// WithTruncation caps observation length at limit bytes, keeping the head and
// the tail; see [Clip] for why both.
//
// The marker says how much was cut and where the gap is, so the model knows it
// is reading two fragments and can page the middle in with a more specific call
// instead of assuming they are contiguous.
//
// exempt names tools whose output must never be cut. load_skill is the
// motivating case: a half-loaded skill is worse than no skill at all, because
// the model acts on instructions whose second half it never saw.
//
// Errors are deliberately left alone. Truncating an error string could cut off
// the note WithDedupRepeats appends, and would mean re-wrapping, which destroys
// the identity errors.Is depends on. Tool errors are short; observations are
// what blow up.
//
// limit at zero disables truncation.
func WithTruncation(limit int, exempt ...string) Middleware {
	if limit <= 0 {
		return func(next Handler) Handler { return next }
	}

	skip := make(map[string]struct{}, len(exempt))
	for _, name := range exempt {
		skip[name] = struct{}{}
	}

	return func(next Handler) Handler {
		return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
			out, err := next(ctx, d, use)
			if err != nil || len(out) <= limit {
				return out, err
			}
			if _, ok := skip[d.Name]; ok {
				return out, nil
			}

			return Clip(out, limit), nil
		}
	}
}

// Clip shortens s to limit bytes of content, keeping the HEAD and the TAIL and
// dropping the middle.
//
// Not head-only, and the difference decides whether a tool call was worth
// making. Command output is back-loaded: `go build` prints the compiler errors
// last, a test suite prints the failing assertion after every passing test
// name, a stack trace ends at the throw. Keeping only the front of a 200 KB
// failing test run hands the model eight kilobytes of test names that passed
// and a note that the rest is gone — every byte of it true, and none of it the
// answer. The head still earns its half: it carries the command echo, the
// header, the first error, and the shape of whatever is being read.
//
// This is the OCaml original's rule, which splits its cap down the
// middle for the same stated reason.
//
// The marker is explicit that the two halves are NOT contiguous. A model handed
// a gap it does not know about will reason across it — concluding a function is
// missing, or that a list has ended — which is a worse failure than being told
// the output was too long.
//
// The budget is on content: the marker is added on top, so the result may
// exceed limit by the length of one line. That keeps the arithmetic a caller
// can predict, and a caller sizing a context window is not counting to the byte.
func Clip(s string, limit int) string {
	if limit <= 0 || len(s) <= limit {
		return s
	}

	// Too small a budget to split: a tail of a handful of bytes is not a tail,
	// it is a fragment of a word behind a hundred-byte marker. 64 bytes is about
	// one terminal line, the smallest piece that can still carry
	// "FAIL: TestFoo" or "exit status 2".
	if limit < 2*minClipSide {
		return ClipHead(s, limit)
	}

	head := boundary(s, limit/2)
	start := len(s) - (limit - head)
	// Forward, not back: advancing to the next rune start drops at most three
	// more bytes, while backing up would re-admit the tail of a rune whose head
	// is in the omitted middle, which is exactly the invalid UTF-8 that a
	// provider rejects on the NEXT call — a failure that surfaces nowhere near
	// its cause.
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	out := fmt.Sprintf("%s\n\n[... %d of %d bytes omitted from the middle — what follows is the TAIL, "+
		"not a continuation. Narrow the request to see between them ...]\n\n%s",
		s[:head], start-head, len(s), s[start:])
	return shorterOf(s, out, ClipHead(s, limit))
}

// shorterOf returns the first candidate that actually beats the original, or
// the original.
//
// Comparing the finished strings rather than guessing a threshold, because the
// marker's length depends on the numbers printed in it and a threshold that is
// nearly right is a function called Clip that returns MORE than it was given.
// That is not hypothetical: Clip(8001 bytes, 8000) returned 8142, spending a
// 141-byte notice to elide a single byte. Every observation within a marker's
// length of the limit got bigger, at both call sites.
//
// A clip that cannot shrink its input has nothing useful to do, and leaving the
// bytes alone is the honest answer — the cap is soft by at most one marker,
// which no caller sizing a context window is counting.
func shorterOf(original string, candidates ...string) string {
	for _, c := range candidates {
		if len(c) < len(original) {
			return c
		}
	}
	return original
}

// minClipSide is the smallest half [Clip] will produce before it gives up on
// splitting and keeps the head alone.
const minClipSide = 64

// ClipHead shortens s to limit bytes by keeping only the front.
//
// The right choice when the output is a REGION the caller asked for rather than
// a report: view_file returns lines the model is about to edit, and a
// head-and-tail clip hands it two fragments that look adjacent. The model then
// composes a str_replace whose old_str spans the invisible gap, which does not
// match, and it retries. A contiguous prefix is smaller and more useful, and
// the tool that produced it has offset and limit parameters for reaching the
// rest. Use [Clip] for command output, where the answer is at the end.
func ClipHead(s string, limit int) string {
	if limit <= 0 || len(s) <= limit {
		return s
	}
	cut := boundary(s, limit)
	out := fmt.Sprintf("%s\n\n[truncated: showing the first %d of %d bytes, %d omitted — "+
		"narrow the request to see the rest]", s[:cut], cut, len(s), len(s)-cut)
	return shorterOf(s, out)
}

// boundary backs n up to the nearest rune start, so a cut never leaves a
// mangled half-character for the tokenizer to choke on.
func boundary(s string, n int) int {
	if n >= len(s) {
		return len(s)
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return n
}

// ===== Logging =====

// Preview budgets for logs. Small on purpose: a log line is a diagnostic, and
// a tool observation can be megabytes. The full text reaches the model and the
// event stream; it has no business in a log sink too.
const (
	inputPreview  = 200
	outputPreview = 200
)

// WithLogging emits one Debug line on entry and one Info (or Warn, on failure)
// line on the verdict.
//
// Diagnostics only — semantic events go through the observer, and conflating
// the two is what forces a UI to parse log strings. A nil logger falls back to
// slog.Default rather than disabling logging, matching the rest of the
// harness.
func WithLogging(l *slog.Logger) Middleware {
	if l == nil {
		l = slog.Default()
	}
	return func(next Handler) Handler {
		return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
			l.DebugContext(ctx, "tool call",
				slog.String("tool", d.Name),
				slog.String("category", d.Category),
				slog.String("input", preview(string(use.Input), inputPreview)),
			)

			start := time.Now()
			out, err := next(ctx, d, use)
			dur := time.Since(start)

			if reason, deferred := DeferralReason(err); deferred {
				// Not a failure: the call waits for a person's decision.
				l.InfoContext(ctx, "tool deferred",
					slog.String("tool", d.Name),
					slog.String("category", d.Category),
					slog.String("reason", reason),
				)
				return out, err
			}
			if err != nil {
				l.WarnContext(ctx, "tool failed",
					slog.String("tool", d.Name),
					slog.String("category", d.Category),
					slog.Duration("dur", dur),
					slog.Bool("ok", false),
					slog.String("error", preview(err.Error(), outputPreview)),
				)
				return out, err
			}

			l.InfoContext(ctx, "tool ok",
				slog.String("tool", d.Name),
				slog.String("category", d.Category),
				slog.Duration("dur", dur),
				slog.Bool("ok", true),
				slog.Int("bytes", len(out)),
				slog.String("output", preview(out, outputPreview)),
			)
			return out, nil
		}
	}
}

// preview clips s to at most n bytes on a rune boundary, marking the cut.
func preview(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:boundary(s, n)] + "..."
}
