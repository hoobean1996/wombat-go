// Package eval measures whether the HARNESS's mechanisms do their job against
// a real model.
//
// This is not the benchmark suite and it does not replace it. benchmarks/ asks
// "can the agent solve the task" and answers with pass@k; every task there runs
// on a bare agent, so the mechanisms this project is actually about — skill
// gating, delegation, the permission gate, transcript strategies — are measured
// by nothing at all. Their only coverage is unit tests against a scripted
// client, and a scripted client can prove that given a fake response the code
// takes a branch. It cannot prove that against a live model the mechanism
// changed the outcome in the direction it claims.
//
// # The shape is a contrast, not a score
//
// A mechanism claim always has the form "with M on, X; with M off, not X". So
// the unit here is a [Probe]: one task, two [Arm]s that differ ONLY in the
// mechanism, n samples of each, compared on a mechanism-specific observable.
//
// That observable is deliberately never pass@k. A skill gate that halves the
// tool list does not move a task score, and neither does a window that keeps
// the prompt prefix byte-stable — the whole point of the second one is that its
// only symptom is the bill. Measuring mechanisms with a task score is how a
// regression in them stays invisible for a year.
//
// # Two kinds of expectation
//
//   - [Invariant] must hold in EVERY sample. "A gated tool never executes
//     before its skill is loaded" is not a statistic; one violation is a bug.
//   - [Contrast] compares aggregates ACROSS arms. "The anchored window reads
//     more from cache than the sliding one" is only meaningful as a difference,
//     and only over enough samples to mean something.
//
// A [Verdict] must always carry its numbers. A pass with no measurement in it
// is not evidence, and this suite exists precisely because "the tests are
// green" was not evidence either.
package eval

import (
	"time"

	"github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/skill"
)

// Deps is everything an [Arm] needs to build its agent.
//
// Workspace is per SAMPLE, not per probe: two samples of the same arm must not
// share a directory or the second one starts in the first one's leftovers.
type Deps struct {
	Client    llm.Client
	Model     string
	Workspace string
	Skills    []skill.Skill
}

// Arm is one configuration of the agent under test.
//
// Two arms of a probe must differ in exactly one thing — the mechanism. Any
// other difference makes the contrast measure something else, and nothing here
// can detect that for you.
type Arm struct {
	Name  string
	Build func(Deps) (*wombat.Agent, error)
}

// Kind distinguishes an expectation that must always hold from one that only
// means something as a difference between arms.
type Kind int

// Expectation kinds.
const (
	Invariant Kind = iota
	Contrast
)

// String implements fmt.Stringer.
func (k Kind) String() string {
	if k == Contrast {
		return "contrast"
	}
	return "invariant"
}

// Verdict is one expectation's answer.
//
// Detail is not optional in spirit: it must name the numbers the verdict was
// reached from, because a bare pass tells a reader nothing they can check.
type Verdict struct {
	Pass   bool
	Detail string
}

// Runs holds every sample of every arm, keyed by arm name.
type Runs map[string][]Obs

// Arm returns the samples for one arm, or nil.
func (r Runs) Arm(name string) []Obs { return r[name] }

// Expect is one checkable claim about a mechanism.
type Expect struct {
	Name  string
	Kind  Kind
	Check func(Runs) Verdict
}

// Probe is one mechanism under test.
type Probe struct {
	// ID names the probe on the command line.
	ID string

	// Mechanism is the thing being measured, for grouping in the report.
	Mechanism string

	// Claim is the sentence this probe is trying to falsify, in the words the
	// design uses. It is printed next to the verdict so a failure reads as
	// "this claim is no longer true" rather than as a broken test.
	Claim string

	// Prompt is the task, identical across arms.
	Prompt string

	// Setup builds one sample's workspace. Called once per sample per arm,
	// with a fresh empty directory.
	Setup func(dir string) error

	// Skills are offered to arms that ask for them; an arm is free to ignore
	// the list, which is how the ungated baseline of a gating probe is built.
	Skills []skill.Skill

	// Arms are the configurations to compare. One arm is legal and means the
	// probe carries only invariants.
	Arms []Arm

	// Expect is what has to be true.
	Expect []Expect

	// MaxIters caps every arm's iterations, applied on top of whatever the
	// arm built so the arms stay identical in it; 0 leaves each agent's own
	// cap. Limits is the run's budget, the same for every sample.
	MaxIters int
	Limits   governor.Limits

	// Expensive marks a probe that needs a long run to observe anything —
	// a window only trims once a transcript is big enough. Skipped unless
	// asked for.
	Expensive bool
}

// NamedVerdict is one expectation's result.
type NamedVerdict struct {
	Name string
	Kind Kind
	Verdict
}

// Result is one probe's outcome.
type Result struct {
	Probe    *Probe
	Runs     Runs
	Verdicts []NamedVerdict
	Wall     time.Duration

	// Err is set when the probe could not be evaluated at all — a workspace
	// that would not build, an arm that would not construct. Distinct from a
	// failing expectation, which is a real answer.
	Err error
}

// Pass reports whether every expectation held.
func (r *Result) Pass() bool {
	if r.Err != nil {
		return false
	}
	for _, v := range r.Verdicts {
		if !v.Pass {
			return false
		}
	}
	return true
}
