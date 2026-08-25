// Package judge evaluates what an agent DID, from the trajectory it left
// behind.
//
// It is the second half of a pipeline whose first half already exists:
// rl/ supplies the task set, the oracle, the execution harness and the trace;
// benchmarks/ supplies the tasks. What was missing was everything after the
// reward — a scalar says a run was bad and cannot say what to change.
//
//	Goal → Criteria → Rubric → Judge → Score → Aggregate/Slice → Diagnose → Repair
//
// # The rubric defines the levels; a judge only picks one
//
// That split is the load-bearing decision here. A [Rubric] states, in prose,
// what a 5 and a 3 and a 1 look like on one dimension. A [Judge]'s entire job
// is to choose the level and say why. Everything else — scoring, gating,
// aggregation, diagnosis — reads the chosen level and never re-litigates it.
//
// The payoff is that a rule-based judge and a model-based one produce the same
// [Assessment] over the same anchors, so they can be run on the same episode
// and DISAGREED with each other. An LLM judge nobody can check is a scalar
// with extra steps; one that can be scored against deterministic rules on the
// cases the rules cover is a measurement.
//
// # Two things this deliberately does not do
//
// It does not replace rl.Verifier. A verifier answers "did the tests pass",
// from the oracle, programmatically, and is the ground truth this package is
// judged against rather than a competitor to it. [Verifier] adapts one into
// the other direction so both land in the same report.
//
// It does not evaluate the harness. That is eval/shape and eval/wire, which
// measure mechanisms rather than behaviour and share none of this vocabulary
// on purpose — a rubric over a cache prefix would be inventing a judgement
// where an assertion is available.
package judge

import (
	"context"
	"fmt"
	"sort"

	"github.com/hoobean1996/wombat-go/rl"
)

// Criterion is one dimension of quality.
//
// A string rather than an enum so a caller can add a dimension their domain
// needs without forking this package; [Known] lists the ones with rubrics
// shipped here.
type Criterion string

// The dimensions.
const (
	// TaskSuccess: did it accomplish what was asked?
	TaskSuccess Criterion = "task_success"

	// Correctness: is the result right? Distinct from success — an agent can
	// finish confidently and be wrong, and the two have different repairs.
	Correctness Criterion = "correctness"

	// Faithfulness: is every claim backed by something it actually observed?
	Faithfulness Criterion = "faithfulness"

	// ReasoningQuality: do the steps follow from each other?
	ReasoningQuality Criterion = "reasoning_quality"

	// ToolUseQuality: right tools, right arguments, adapted to what came back?
	ToolUseQuality Criterion = "tool_use_quality"

	// Efficiency: did it get there without wasted work?
	Efficiency Criterion = "efficiency"

	// Robustness: did it handle what went wrong?
	Robustness Criterion = "robustness"

	// Safety: did it avoid destructive, leaking or escalating actions?
	Safety Criterion = "safety"

	// UserAlignment: did it serve the intent, not just the letter?
	UserAlignment Criterion = "user_alignment"
)

// Level is one anchor on a rubric's scale.
//
// Anchors are prose on purpose. A number with no anchor is not reproducible —
// two judges, or the same judge a week apart, will not mean the same thing by
// "3" — and reproducibility is the only property that makes a rubric worth
// more than an opinion.
type Level struct {
	Score  int
	Anchor string
}

// Rubric is one criterion's scale.
type Rubric struct {
	Criterion Criterion

	// Question is what this dimension asks, in one line.
	Question string

	// Levels are the anchors, highest score first.
	Levels []Level

	// PassAt is the lowest score that counts as a pass.
	PassAt int

	// Weight scales this criterion in the weighted overall.
	Weight float64

	// Gating means a failure here fails the episode whatever else scored.
	// Safety is gating; efficiency is not.
	Gating bool
}

// Max is the top of the scale.
func (r Rubric) Max() int {
	m := 0
	for _, l := range r.Levels {
		if l.Score > m {
			m = l.Score
		}
	}
	return m
}

// Min is the bottom of the scale.
func (r Rubric) Min() int {
	if len(r.Levels) == 0 {
		return 0
	}
	m := r.Levels[0].Score
	for _, l := range r.Levels {
		if l.Score < m {
			m = l.Score
		}
	}
	return m
}

// Normalized maps a score onto 0..1, so criteria with different scales can be
// averaged without one of them quietly dominating.
func (r Rubric) Normalized(score int) float64 {
	lo, hi := r.Min(), r.Max()
	if hi <= lo {
		return 0
	}
	if score < lo {
		score = lo
	}
	if score > hi {
		score = hi
	}
	return float64(score-lo) / float64(hi-lo)
}

// Anchor returns the prose for a score, or "" when the scale has no such level.
func (r Rubric) Anchor(score int) string {
	for _, l := range r.Levels {
		if l.Score == score {
			return l.Anchor
		}
	}
	return ""
}

// Evidence is one observation supporting an assessment.
//
// Step is the trajectory index it was seen at, or -1 for something true of the
// episode as a whole. A judge that cannot point at a step for a step-level
// claim is guessing, and the field exists to make that visible.
type Evidence struct {
	Step int    `json:"step"`
	Note string `json:"note"`
}

// Assessment is one judge's verdict on one criterion.
type Assessment struct {
	Criterion Criterion `json:"criterion"`

	// Score is the level the judge chose.
	Score int `json:"score"`

	// Pass is Score >= Rubric.PassAt.
	Pass bool `json:"pass"`

	// Evidence is what the verdict rests on. An assessment below the top level
	// with no evidence is not a judgement, and [Validate] rejects it.
	Evidence []Evidence `json:"evidence,omitempty"`

	// Failure classifies a non-passing assessment.
	Failure Failure `json:"failure,omitempty"`

	// Repair names the component to change. Defaulted from Failure when the
	// judge does not set it.
	Repair RepairTarget `json:"repair,omitempty"`

	// FirstBadStep is the earliest step anything went wrong at, -1 when
	// nothing did. Reported even on a passing assessment: a run that recovered
	// from a stumble is not the same as one that never stumbled.
	FirstBadStep int `json:"first_bad_step"`

	// Rationale is the judge's own sentence.
	Rationale string `json:"rationale,omitempty"`

	// Judge names who produced this, so two judges' verdicts stay
	// distinguishable in a report.
	Judge string `json:"judge"`

	// Confidence is 0..1. A rule that fired a definite check reports 1; a
	// heuristic or a model reports what it means. Aggregation can then weight
	// or filter, instead of averaging a guess with a measurement.
	Confidence float64 `json:"confidence"`
}

// Validate reports whether an assessment is well formed against its rubric.
//
// Enforced rather than trusted because the two judges that produce these are a
// pile of regexes and a language model, and both fail in the same direction:
// a confident verdict with nothing behind it.
func (a Assessment) Validate(r Rubric) error {
	if a.Criterion != r.Criterion {
		return fmt.Errorf("judge: assessment is for %q but the rubric is for %q", a.Criterion, r.Criterion)
	}
	if r.Anchor(a.Score) == "" {
		return fmt.Errorf("judge: %s scored %d, which is not a level on its scale", a.Criterion, a.Score)
	}
	if a.Score < r.Max() && len(a.Evidence) == 0 {
		return fmt.Errorf("judge: %s scored %d with no evidence; a verdict below the top level has to point at something",
			a.Criterion, a.Score)
	}
	if !a.Pass && a.Failure == None {
		return fmt.Errorf("judge: %s did not pass and carries no failure classification", a.Criterion)
	}
	if !Valid(a.Failure) {
		return fmt.Errorf("judge: %s reported the unknown failure %q", a.Criterion, a.Failure)
	}
	if a.Confidence < 0 || a.Confidence > 1 {
		return fmt.Errorf("judge: %s reported confidence %.2f, want 0..1", a.Criterion, a.Confidence)
	}
	return nil
}

// Judge assesses one criterion of one episode.
type Judge interface {
	Name() string

	// Assess picks a level from r and justifies it. An error means the judge
	// could not reach a verdict at all — which is different from a low score
	// and must not be recorded as one.
	Assess(ctx context.Context, ep *rl.Episode, r Rubric) (Assessment, error)
}

// Diagnosis is the pipeline's answer to "what should change".
type Diagnosis struct {
	// RootCause is the failure behind the worst gating or lowest-scoring
	// criterion.
	RootCause Failure `json:"root_cause,omitempty"`

	// FirstBadStep is the earliest step any criterion flagged.
	FirstBadStep int `json:"first_bad_step"`

	// Repair is where to act.
	Repair RepairTarget `json:"repair,omitempty"`

	// Also lists the other repairs implicated, most severe first, so a loop
	// can see whether one change addresses several criteria.
	Also []RepairTarget `json:"also,omitempty"`

	Summary string `json:"summary"`
}

// Report is every assessment of one episode, plus what to do about it.
type Report struct {
	Task   string `json:"task"`
	Sample int    `json:"sample"`

	Assessments []Assessment `json:"assessments"`

	// Overall is the weighted mean of normalized scores, 0..1.
	Overall float64 `json:"overall"`

	// Pass is a conjunction, not a threshold on Overall: an episode that is
	// unsafe but efficient must not average its way to a pass.
	Pass bool `json:"pass"`

	// Unjudged lists criteria no judge in the panel would rule on.
	//
	// Visible rather than absent, and excluded from Overall rather than
	// scored zero or five. A criterion nobody judged is a hole in the
	// measurement, and a report that hides it is claiming coverage it does
	// not have — which is how "we evaluate faithfulness" survives for months
	// after the only judge that could stopped being run.
	Unjudged []Criterion `json:"unjudged,omitempty"`

	Diagnosis Diagnosis `json:"diagnosis"`

	// Reward carries the oracle's own verdict through, so a report can be read
	// against ground truth rather than only against itself.
	Reward float64 `json:"reward"`

	// RunFailure is how the run stopped — rl's vocabulary, kept separate from
	// the judged Failure on purpose. See the note on [Failure].
	RunFailure rl.FailureKind `json:"run_failure,omitempty"`
}

// Of returns the assessment for a criterion, if there is one.
func (r *Report) Of(c Criterion) (Assessment, bool) {
	for _, a := range r.Assessments {
		if a.Criterion == c {
			return a, true
		}
	}
	return Assessment{}, false
}

// Failures lists the non-passing assessments, worst score first.
func (r *Report) Failures() []Assessment {
	var out []Assessment
	for _, a := range r.Assessments {
		if !a.Pass {
			out = append(out, a)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score < out[j].Score })
	return out
}
