package rl

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"slices"
	"sync"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/permission"
)

// Defaults applied by [Rollout] when the corresponding option is not given.
const (
	// DefaultConcurrency is how many samples run at once.
	DefaultConcurrency = 4

	// DefaultSuccessThreshold is the reward at or above which a cleanly
	// finished episode counts as [Success].
	//
	// 1.0 because the convention this package is built around is that a
	// task's verifier weights sum to 1: [FileExists] 0.3 plus [Shell] 0.7 is a
	// task, and 1.0 means all of it worked. Anything less means something did
	// not verify, which is [VerifierFailed] — the row that says the agent ran
	// fine and produced the wrong thing.
	//
	// Penalties break that convention on purpose: a run with
	// TurnPenalty(0.01) can do everything right and still score 0.94. Lower
	// the threshold with [WithSuccessThreshold] whenever the score can be
	// docked, or the histogram will read as if nothing ever passed.
	DefaultSuccessThreshold = 1.0
)

// ErrNoSamples reports a rollout asked for a non-positive number of samples.
var ErrNoSamples = errors.New("rl: sample count must be positive")

type config struct {
	concurrency int
	limits      governor.Limits
	progress    func(Task, *Episode)
	threshold   float64
	keepAll     bool
	pricing     llm.Pricing
	log         *slog.Logger
}

// Option configures a [Rollout].
type Option func(*config)

// WithConcurrency runs up to n samples at once. Default [DefaultConcurrency].
// Non-positive values are ignored, so the zero value of a caller's own
// variable cannot silently serialize the whole rollout.
func WithConcurrency(n int) Option {
	return func(c *config) {
		if n > 0 {
			c.concurrency = n
		}
	}
}

// WithBudget gives each episode these limits.
//
// EACH episode, not the rollout: [governor.WithBudget] is called once per
// sample, so eight samples of a $0.50 task may cost $4.00 and each one gets
// the full $0.50 to work with. A shared budget would make the samples
// interfere — the first to start would spend it and the rest would be
// guillotined on their first call — and the resulting failure histogram would
// measure the scheduler rather than the agent. That is the difference between
// eight samples and one sample run eight times badly.
//
// Cap the total from outside instead, with the concurrency setting and a
// wall-clock limit on the caller's own context.
func WithBudget(l governor.Limits) Option { return func(c *config) { c.limits = l } }

// WithProgress calls f as each episode finishes.
//
// Called from the finishing episode's own goroutine, so f runs concurrently
// with itself when [WithConcurrency] is above 1: guard anything it touches. It
// is called for every episode including the ones that never started, because a
// progress bar that skips the broken samples finishes early and lies.
func WithProgress(f func(Task, *Episode)) Option { return func(c *config) { c.progress = f } }

// WithSuccessThreshold sets the reward at or above which a cleanly finished
// episode counts as [Success]. Default [DefaultSuccessThreshold].
//
// This is the line between [Success] and [VerifierFailed], and it is the only
// place the two are distinguished — everything below it is an episode that ran
// without error and produced the wrong thing.
func WithSuccessThreshold(r float64) Option { return func(c *config) { c.threshold = r } }

// WithKeepWorkspaces preserves every episode's artifacts, passing and failing
// alike.
//
// Without it, [Rollout] still asks the environment to keep the FAILED ones —
// see [WithKeep] — because a failed episode's directory is the first thing
// anyone will want to look at, and a successful one is just disk. Set this
// when you want the successful trajectories too, for training data or for
// diffing a pass against a fail.
func WithKeepWorkspaces() Option { return func(c *config) { c.keepAll = true } }

// WithPricing tells the rollout which pricing the agent's cost tally was
// computed with, so that [Episode.Priced] can say whether that tally is a
// price or a gap in a table.
//
// It has to be passed rather than read off the agent, for two reasons. A
// [wombat.Agent] does not expose the pricing it was built with; and the number
// that lands in Episode.Spend does not come from the agent at all — it comes
// from whichever [wombat.TrackCost] middleware is in the client chain, which
// the agent cannot see either. The caller is the only party that knows, so the
// caller says.
//
// Without it a rollout falls back to the honest proxy — tokens were spent and
// the tally is still zero, so the zero is not a measurement — which catches
// the common case but cannot name the model. Pass the same [llm.Pricing] you
// gave TrackCost and the report can name it.
func WithPricing(p llm.Pricing) Option { return func(c *config) { c.pricing = p } }

// WithLogger sets the diagnostic logger. Default [slog.Default].
//
// Diagnostics only: everything semantic is on the [Episode].
func WithLogger(l *slog.Logger) Option {
	return func(c *config) {
		if l != nil {
			c.log = l
		}
	}
}

type keepKey struct{}

// WithKeep marks ctx as an episode whose artifacts are worth preserving.
// [Rollout] sets it on the context it hands to [Env.Cleanup].
//
// It rides on the context rather than being an argument because Env.Cleanup
// takes only a [Task], deliberately: most environments tear down the same way
// whatever happened, and widening the interface for the one that does not
// would make every implementation answer a question it does not care about.
func WithKeep(ctx context.Context, keep bool) context.Context {
	return context.WithValue(ctx, keepKey{}, keep)
}

// Keep reports whether the caller of [Env.Cleanup] asked for the task's
// artifacts to be preserved. False when nothing said either way, so an Env
// used outside a rollout cleans up as it normally would.
func Keep(ctx context.Context) bool {
	keep, _ := ctx.Value(keepKey{}).(bool)
	return keep
}

// Rollout runs n samples of env's task and returns them as one [Group].
//
// Every sample gets its own workspace from Env.Reset, its own agent from mk
// and its own budget from [WithBudget]; see the package doc for why all three
// have to be per-sample. Samples run concurrently ([WithConcurrency], default
// [DefaultConcurrency]) and land in Group.Episodes at their own index, so the
// result does not depend on the order they finished in.
//
// A rollout does not fail because an episode did. A sample whose Reset broke,
// whose agent would not build or which panicked mid-run comes back as an
// [Episode] carrying that error and a [FailureKind], because losing the other
// seven samples to one broken one is the worst possible response to a flaky
// environment. The returned error is reserved for the two cases where there is
// no group to speak of: arguments that cannot be run at all, and a caller
// context that was cancelled underneath the whole thing.
func Rollout(ctx context.Context, mk AgentFunc, env Env, n int, opts ...Option) (*Group, error) {
	if mk == nil {
		return nil, errors.New("rl: nil AgentFunc")
	}
	if env == nil {
		return nil, errors.New("rl: nil Env")
	}
	if n <= 0 {
		return nil, fmt.Errorf("%w, got %d", ErrNoSamples, n)
	}

	cfg := config{
		concurrency: DefaultConcurrency,
		threshold:   DefaultSuccessThreshold,
		log:         slog.Default(),
	}
	for _, o := range opts {
		o(&cfg)
	}

	g := &Group{Env: env.Name(), Episodes: make([]*Episode, n), Threshold: cfg.threshold}

	// A counting semaphore rather than a worker pool: the episodes are
	// independent and indexed, so there is no queue to drain and nothing to
	// hand between goroutines.
	sem := make(chan struct{}, min(cfg.concurrency, n))
	var wg sync.WaitGroup
	for sample := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			ep := runEpisode(ctx, mk, env, sample, &cfg)
			// Distinct indices from distinct goroutines: no lock needed, and
			// the slice header itself is never written after construction.
			g.Episodes[sample] = ep
			if cfg.progress != nil {
				cfg.progress(ep.Task, ep)
			}
		}()
	}
	wg.Wait()

	for _, ep := range g.Episodes {
		if ep.Task.ID != "" {
			g.TaskID = ep.Task.ID
			break
		}
	}

	// context.Cause rather than ctx.Err, so a governed caller learns WHY.
	if cause := context.Cause(ctx); cause != nil {
		return g, fmt.Errorf("rl: rollout of %q was cut short: %w", g.TaskID, cause)
	}
	return g, nil
}

// runEpisode is one sample, start to finish: reset, build, run, score,
// classify, clean up.
//
// It never panics out. A panic here is on a rollout goroutine, which means an
// unrecovered one takes down the whole process and every other sample with it
// — including the ones that had already finished and would have told you what
// went wrong.
func runEpisode(parent context.Context, mk AgentFunc, env Env, sample int, cfg *config) (ep *Episode) {
	started := time.Now()
	ep = &Episode{Task: Task{Sample: sample}}

	// The models that answered, filled in by drive. Declared out here so the
	// defer below sees whatever was recorded before a panic.
	var models []string
	defer func() {
		if p := recover(); p != nil {
			ep.Err = fmt.Errorf("rl: episode panicked: %v\n%s", p, debug.Stack())
			ep.Failure = Panicked
		}
		ep.Wall = time.Since(started)
		// Last, and on every path including the ones that never ran an agent:
		// it reads the final spend, and an episode that reported no cost
		// because it never started is honestly reporting zero.
		setPriced(ep, cfg.pricing, models)
	}()

	t, err := env.Reset(parent, sample)
	if err != nil {
		ep.Err = fmt.Errorf("rl: env %q could not reset sample %d: %w", env.Name(), sample, err)
		ep.Failure = classify(ep, cfg.threshold)
		return ep
	}
	// Stamped rather than trusted: Group.Episodes is indexed by sample, and an
	// Env that forgot to fill the field in would mislabel every row of the
	// report without anything failing.
	t.Sample = sample
	ep.Task = t

	// From here on the workspace exists, so every exit has to go past cleanup.
	defer func() { cleanup(parent, env, ep, cfg) }()

	a, err := mk(t)
	if err != nil {
		ep.Err = fmt.Errorf("rl: building the agent for sample %d: %w", sample, err)
		ep.Failure = classify(ep, cfg.threshold)
		return ep
	}
	if a == nil {
		ep.Err = fmt.Errorf("rl: AgentFunc returned a nil agent for sample %d", sample)
		ep.Failure = classify(ep, cfg.threshold)
		return ep
	}

	// One budget per episode. See WithBudget.
	ectx, cancel := governor.WithBudget(parent, cfg.limits)
	defer cancel()

	models = drive(ectx, a, ep)
	ep.Spend = governor.FromContext(ectx).Progress()

	// Before scoring, not only in the defer, so that a [Verifier] reading the
	// cost — [CostPenalty] — can see whether that cost means anything. The
	// defer recomputes it, which is what covers the paths that never got here.
	setPriced(ep, cfg.pricing, models)

	// Scored on the CALLER's context, not the episode's. An episode that died
	// of budget exhaustion has a cancelled context, and a verifier that shells
	// out under it would be killed before it could report what the agent
	// managed to build — turning every budget failure into a zero and hiding
	// the partial credit that makes a benchmark diagnosable.
	reward, breakdown, serr := env.Score(parent, ep)
	ep.Reward, ep.Breakdown = reward, breakdown
	if serr != nil && ep.Err == nil {
		ep.Err = fmt.Errorf("rl: env %q could not score sample %d: %w", env.Name(), sample, serr)
	}

	ep.Failure = classify(ep, cfg.threshold)
	return ep
}

// drive runs the agent and folds its event stream into the episode.
//
// Through Start and the event stream rather than Agent.Run, because that is
// the only place the per-turn facts exist: Run returns an outcome and throws
// away which tools were called, which failed, which were refused and what each
// turn cost. Reconstructing a [Step] from the transcript afterwards cannot
// recover any of it — a tool_result does not say whether the gate refused the
// call or the tool itself blew up.
//
// It returns the models that answered, which is the other fact only the stream
// carries: a cost cannot be judged without knowing what was being charged for.
func drive(ctx context.Context, a *wombat.Agent, ep *Episode) []string {
	run := a.Start(ctx, wombat.Ask(ep.Task.Prompt))
	defer run.Close()

	var f folder
	for run.Next() {
		f.fold(run.Event(), "")
	}

	ep.Steps = f.steps
	ep.Messages = run.Messages()
	ep.Outcome = run.Outcome()
	ep.Err = run.Err()
	return f.models
}

// cleanup disposes of the episode's world, keeping the artifacts of anything
// that did not succeed.
//
// On a context detached from the caller's cancellation: a rollout aborted
// half-way still has to release what it allocated, and a Cleanup that is
// skipped because the context is already dead leaks a container or a
// directory for every sample in flight.
func cleanup(parent context.Context, env Env, ep *Episode, cfg *config) {
	keep := cfg.keepAll || ep.Failure != Success
	ctx := WithKeep(context.WithoutCancel(parent), keep)
	if err := env.Cleanup(ctx, ep.Task); err != nil {
		// Logged, not returned: a cleanup failure is the operator's problem,
		// not evidence about the agent, and putting it in ep.Err would
		// reclassify a perfectly good episode.
		cfg.log.WarnContext(ctx, "rl: cleanup failed",
			slog.String("env", env.Name()),
			slog.String("task", ep.Task.ID),
			slog.Int("sample", ep.Task.Sample),
			slog.Any("err", err))
	}
}

// ===== folding the event stream into Steps =====

// folder accumulates a run's events into [Step]s.
//
// Not safe for concurrent use, and does not need to be: a Run's events arrive
// on one channel, read by one goroutine.
type folder struct {
	steps []Step

	// models are the models that answered and charged for it, in first-seen
	// order and deduplicated. Sub-agents included: a child's tokens are the
	// parent's money, so a child on an unpriced model makes the parent's cost
	// wrong too.
	models []string

	// asked is the model of the most recent request, used when a response does
	// not name one. Same fallback the metric middleware makes: the resolved
	// model is what answered and therefore what the bill follows, and the
	// requested one is the best guess when the provider stayed quiet.
	asked string

	// denied remembers which calls the permission gate refused, keyed by
	// tool_use id. It exists for streams where the error VALUE is gone —
	// events replayed from JSON, or forwarded through a transport — since
	// permission.Decided carries the verdict structurally and survives the
	// round trip.
	denied map[llm.ToolUseID]bool
}

// fold folds one event into the open step. prefix names the sub-agent an event
// came from, empty for the parent's own.
func (f *folder) fold(ev wombat.Event, prefix string) {
	switch e := ev.(type) {
	case wombat.IterStart:
		// Opens a step — but only the parent's iterations do. A sub-agent
		// running its own ReAct loop inside one delegate call is not a turn
		// the parent took.
		if prefix == "" {
			f.steps = append(f.steps, Step{Iteration: e.N})
		}

	case wombat.LLMStart:
		// Kept only as the fallback for a response that does not name its
		// model; it opens no step, because a request that fails costs nothing
		// and a turn is opened by IterStart.
		f.asked = e.Model

	case wombat.LLMDone:
		s := f.current()
		// A child's tokens are the parent's money, so they count. A child's
		// latency is not added, because the delegate tool's own ToolDone
		// already spans it and adding both would double-count the turn.
		s.Usage.Add(e.Usage)
		if prefix == "" {
			s.Millis += e.Millis
		}
		// Only a call that consumed tokens: a model that charged nothing
		// cannot have produced a misleading zero.
		if tokens(e.Usage) > 0 {
			f.model(e.Model)
		}

	case wombat.ToolStart:
		s := f.current()
		s.Tools = append(s.Tools, prefix+e.Name)

	case wombat.ToolDone:
		if e.OK {
			return
		}
		s := f.current()
		name := prefix + e.Name
		s.Failed = append(s.Failed, name)
		if f.refused(e) {
			// Denied is a subset of Failed on purpose: a refused call did not
			// happen, which is a failure, and the extra list says why.
			s.Denied = append(s.Denied, name)
		}

	case permission.Decided:
		if !e.Allowed {
			if f.denied == nil {
				f.denied = make(map[llm.ToolUseID]bool)
			}
			f.denied[e.UseID] = true
		}

	case wombat.SubagentEvent:
		// A delegated call is ONE action from the parent's point of view, so
		// the child's events fold into the step that is already open. Naming
		// them "child/tool" keeps them attributable without inventing a turn
		// the parent never took. Recursive, so a grandchild reads
		// "child/grandchild/tool".
		f.fold(e.Inner, prefix+e.Name+"/")
	}
}

// refused reports whether a failed call was a permission refusal.
//
// errors.Is against the sentinel, never a substring of the message: the text
// of a refusal is prose written for the model and it is rewritten whenever
// somebody improves the wording, while the sentinel is the contract. The
// Decided fallback covers a stream whose error values did not survive — see
// [folder.denied].
func (f *folder) refused(e wombat.ToolDone) bool {
	return errors.Is(e.Err, permission.ErrDenied) || f.denied[e.UseID]
}

// model records that name answered, falling back to the requested model and
// then to nothing. A name this stream has already seen is not recorded twice —
// the list is what a report prints, and thirty copies of one model id is not a
// report.
func (f *folder) model(name string) {
	if name == "" {
		name = f.asked
	}
	if name == "" || slices.Contains(f.models, name) {
		return
	}
	f.models = append(f.models, name)
}

// current returns the open step, opening one if the stream produced work
// before any IterStart.
//
// The fallback should be unreachable against this harness, which always emits
// IterStart first. It is here because the alternative is dropping the events
// on the floor, and a benchmark that silently under-counts tool calls is worse
// than one that reports an odd-looking extra turn.
func (f *folder) current() *Step {
	if len(f.steps) == 0 {
		f.steps = append(f.steps, Step{Iteration: 1})
	}
	return &f.steps[len(f.steps)-1]
}

// ===== pricing =====

// setPriced records the verdict on the episode. Idempotent, and called twice
// on the ordinary path — see [runEpisode].
func setPriced(ep *Episode, p llm.Pricing, models []string) {
	ep.Priced, ep.Unpriced = pricedSpend(p, models, ep)
}

// pricedSpend decides whether an episode's cost tally is a measurement, and
// names the models it could not price.
//
// Two checks, because either alone can be fooled.
//
// The table check is the direct one: a model the pricing does not know is
// charged at zero, and llm.Priced is the only way to tell that zero from a
// free model. It needs a pricing to consult, which the caller supplies with
// [WithPricing], and it is the only check that can NAME the model.
//
// The proxy is the check that works without one: tokens went out and the tally
// is still zero. That is not a price under any table — either nothing was
// costing the run anything, or nothing was counting — and after a real run the
// second is overwhelmingly likelier. It fires even when the table says the
// model is priced, which is deliberate: a priced model with a zero tally means
// the cost middleware is not in the chain, and reporting $0.0000 for that is
// the same lie by a different route.
func pricedSpend(p llm.Pricing, models []string, ep *Episode) (bool, []string) {
	if tokens(ep.Usage()) > 0 && ep.Spend.CostUSD == 0 {
		// By evidence, none of them: whatever a table claims, this run's spend
		// was not priced.
		return false, models
	}

	var unpriced []string
	for _, m := range models {
		// A nil pricing makes no claim, so it cannot contradict the tally.
		if p != nil && !llm.Priced(p, m) {
			unpriced = append(unpriced, m)
		}
	}
	return len(unpriced) == 0, unpriced
}

// tokens totals a usage across all four kinds.
func tokens(u llm.Usage) int {
	return u.InputTokens + u.OutputTokens + u.CacheWriteTokens + u.CacheReadTokens
}

// ===== classification =====

// classify decides the one [FailureKind] an episode ended as.
//
// Order is the whole content of this function, because errors nest: a
// budget-exhausted run reports the governor's cause, a refused tool call
// inside a run that then ran out of iterations reports the iteration cap. The
// rule is most-proximate-cause-first — what actually stopped the loop — with
// the standing conditions (a run that never got past permission, an
// unrecognised error) as fallbacks underneath.
func classify(ep *Episode, threshold float64) FailureKind {
	if ep.Err == nil {
		if ep.Reward >= threshold {
			return Success
		}
		// Nothing broke. The agent simply did not do the task, and this is the
		// row worth reading.
		return VerifierFailed
	}

	err := ep.Err
	switch {
	// The governor first: it cancels the context, so its cause is the real
	// reason and whatever error the interrupted call reported is noise.
	case errors.Is(err, governor.ErrBudgetExhausted):
		return BudgetExceeded
	case errors.Is(err, governor.ErrWallClock), errors.Is(err, context.DeadlineExceeded):
		return WallClock
	case errors.Is(err, governor.ErrToolLoop), errors.Is(err, governor.ErrToolCallLimit):
		return ToolLoop
	case errors.Is(err, governor.ErrStepLimit):
		// A step is an iteration; the governor's cap and the agent's own cap
		// are the same failure seen from two places, and splitting them would
		// put one bug in two rows of the histogram.
		return MaxIterations

	// Then the loop's own verdicts.
	case errors.Is(err, wombat.ErrMaxIterations):
		return MaxIterations
	case errors.Is(err, wombat.ErrMaxTokens):
		return MaxTokens
	case errors.Is(err, wombat.ErrRefused):
		return Refused
	case errors.Is(err, wombat.ErrPanic):
		return Panicked

	// Then the provider. Context-window overflow is called out separately
	// because it is the one provider failure that is the agent's own doing.
	case errors.Is(err, llm.ErrContextWindow):
		return ContextWindow
	case errors.Is(err, llm.ErrRateLimit), errors.Is(err, llm.ErrOverloaded),
		errors.Is(err, llm.ErrServer), errors.Is(err, llm.ErrTransport),
		errors.Is(err, llm.ErrBadRequest), errors.Is(err, llm.ErrAuth),
		errors.Is(err, llm.ErrNotFound):
		return ProviderError

	case errors.Is(err, permission.ErrDenied):
		return Denied
	case errors.Is(err, context.Canceled):
		return Cancelled
	}

	// No sentinel matched. If the run spent its time being refused, that is
	// what happened to it, whatever the final error says.
	if denials(ep) > 0 {
		return Denied
	}
	return Other
}

// denials counts the permission refusals across an episode.
func denials(ep *Episode) int {
	n := 0
	for _, s := range ep.Steps {
		n += len(s.Denied)
	}
	return n
}
