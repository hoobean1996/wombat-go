package judge

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/hoobean1996/wombat-go/rl"
)

// Oracle judges correctness from the verifiers that actually checked the work.
//
// It exists because of a calibration result rather than a hunch. Run against a
// trajectory with a verified spec deviation — the agent said the state file
// lived next to the binary and wrote it relative to the working directory — a
// model judge scored correctness 5 with confidence 1.00, and cited as evidence
// a functional test that was run inside the workspace and therefore could not
// have shown the difference either way.
//
// That failure is not a weak model. It is structural: a judge reading a
// transcript sees the agent's own verification, and an agent's verification is
// blind in exactly the places the agent was blind. Correctness has to come
// from outside the trajectory, which is what a verifier is — benchmarks.GoProbe
// writes its test into a private copy of the workspace after the agent has
// stopped, from bytes it never had access to.
//
// So this judge reads [rl.Episode.Reward] and [rl.Episode.Breakdown], and
// refuses when no verifier ran. Refusing is the point: an unjudged correctness
// is visibly unjudged, while a guessed one is indistinguishable from a
// measured one in every report it reaches.
type Oracle struct {
	// Pass is the reward at or above which correctness is the top level.
	// Zero uses DefaultOraclePass.
	Pass float64

	// Partial is the reward at or above which correctness is the middle level.
	// Zero uses DefaultOraclePartial.
	Partial float64
}

// Thresholds used when an Oracle leaves them zero.
const (
	DefaultOraclePass    = 0.999
	DefaultOraclePartial = 0.5
)

// ErrNoOracle is returned when no verifier scored the episode.
//
// Distinct from a zero reward, and the distinction is the whole reason this
// error exists: "the verifiers ran and it failed" and "nothing checked it" are
// the same number and opposite facts.
var ErrNoOracle = fmt.Errorf("judge: no verifier scored this episode")

// Name implements Judge.
func (Oracle) Name() string { return "oracle" }

// Assess implements Judge. It rules on Correctness and nothing else.
func (o Oracle) Assess(_ context.Context, ep *rl.Episode, r Rubric) (Assessment, error) {
	if ep == nil {
		return Assessment{}, fmt.Errorf("judge: nil episode")
	}
	switch ep.Failure {
	case rl.Panicked, rl.Cancelled:
		return Assessment{}, ErrUnjudgeable
	}
	if r.Criterion != Correctness {
		return Assessment{}, fmt.Errorf("%w: the oracle only rules on correctness", ErrNotRulable)
	}
	if len(ep.Breakdown) == 0 {
		return Assessment{}, ErrNoOracle
	}

	pass, partial := o.Pass, o.Partial
	if pass <= 0 {
		pass = DefaultOraclePass
	}
	if partial <= 0 {
		partial = DefaultOraclePartial
	}

	a := Assessment{
		Criterion: Correctness, Judge: o.Name(), FirstBadStep: -1, Confidence: 1,
	}
	switch {
	case ep.Reward >= pass:
		a.Score = 5
	case ep.Reward >= partial:
		a.Score = 3
		a.Failure = WrongResult
	default:
		a.Score = 1
		a.Failure = WrongResult
	}
	a.Pass = a.Score >= r.PassAt
	if a.Pass {
		a.Failure = None
	} else {
		a.Repair = RepairFor(a.Failure)
		for _, line := range failedChecks(ep) {
			a.Evidence = append(a.Evidence, Evidence{Step: -1, Note: line})
		}
		if len(a.Evidence) == 0 {
			a.Evidence = append(a.Evidence, Evidence{
				Step: -1,
				Note: fmt.Sprintf("the verifiers scored %.2f against a passing bar of %.2f", ep.Reward, pass),
			})
		}
	}
	a.Rationale = fmt.Sprintf("reward %.2f from %d verifier(s)", ep.Reward, len(ep.Breakdown))
	return a, nil
}

// failedChecks names the verifiers that scored nothing, in a stable order.
func failedChecks(ep *rl.Episode) []string {
	var names []string
	for name, score := range ep.Breakdown {
		if score <= 0 {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	out := make([]string, 0, len(names))
	for _, n := range names {
		out = append(out, fmt.Sprintf("verifier %q scored 0", n))
	}
	if len(out) > 3 {
		out = append(out[:3], fmt.Sprintf("… and %d more", len(out)-3))
	}
	return out
}

// String renders the thresholds, so a report can say what bar it used.
func (o Oracle) String() string {
	pass, partial := o.Pass, o.Partial
	if pass <= 0 {
		pass = DefaultOraclePass
	}
	if partial <= 0 {
		partial = DefaultOraclePartial
	}
	return fmt.Sprintf("oracle(pass>=%.2f, partial>=%.2f)", pass, partial)
}

// StandardPanel is the panel this package recommends.
//
// Order is the design: the oracle settles correctness because only it can, the
// rules settle everything mechanical because they are free and reproducible,
// and the model is left with the two criteria that genuinely need a reader.
// Ordering it any other way spends model calls on questions a regex already
// answers, and — worse — lets a transcript reader overrule ground truth.
func StandardPanel(model Judge) Panel {
	p := Panel{Oracle{}, Rules{}}
	if model != nil {
		p = append(p, model)
	}
	return p
}

// describePanel renders a panel for a report header.
func describePanel(p Panel) string {
	names := make([]string, 0, len(p))
	for _, j := range p {
		names = append(names, j.Name())
	}
	return strings.Join(names, " → ")
}
