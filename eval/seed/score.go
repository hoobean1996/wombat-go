package seed

import (
	"context"
	"fmt"
	"sort"

	"wombat/eval/judge"
	"wombat/eval/rl"
	"wombat/pkg/core/llm"
)

// Clone deep-copies an episode so a mutation cannot reach the original.
//
// Explicit rather than a JSON round trip, because Outcome is an interface and
// would not survive one. Content blocks are value types, so copying the slices
// is enough for them.
func Clone(ep *rl.Episode) *rl.Episode {
	out := *ep
	out.Messages = make([]llm.Message, len(ep.Messages))
	for i, m := range ep.Messages {
		content := make([]llm.ContentBlock, len(m.Content))
		copy(content, m.Content)
		out.Messages[i] = llm.Message{Role: m.Role, Content: content}
	}
	out.Steps = append([]rl.Step(nil), ep.Steps...)
	if ep.Breakdown != nil {
		out.Breakdown = make(map[string]float64, len(ep.Breakdown))
		for k, v := range ep.Breakdown {
			out.Breakdown[k] = v
		}
	}
	return &out
}

// Case is one episode to judge, and what is known to be true about it.
type Case struct {
	Episode *rl.Episode

	// Defect is empty for a control.
	Defect string

	// Expect is the failure a judge should attribute, empty for a control.
	Expect judge.Failure

	// Criterion is where it should surface.
	Criterion judge.Criterion

	// Detectable records whether the panel is even capable of finding it.
	Detectable bool
}

// Build produces one control plus one mutant per defect, from each base.
//
// The controls are not optional. A corpus of only-bad episodes measures recall
// and hides precision, and precision is the half that has been wrong here far
// more often.
func Build(bases []*rl.Episode, defects []Defect) ([]Case, error) {
	var out []Case
	for i, base := range bases {
		ctrl := Clone(base)
		ctrl.Task.ID = fmt.Sprintf("%s/control", base.Task.ID)
		ctrl.Task.Sample = i
		out = append(out, Case{Episode: ctrl})

		for _, d := range defects {
			m := Clone(base)
			if err := d.Apply(m); err != nil {
				return nil, fmt.Errorf("seed: applying %s to %s: %w", d.Name, base.Task.ID, err)
			}
			m.Task.ID = fmt.Sprintf("%s/%s", base.Task.ID, d.Name)
			m.Task.Sample = i
			out = append(out, Case{
				Episode: m, Defect: d.Name, Expect: d.Expect,
				Criterion: d.Criterion, Detectable: d.Detectable,
			})
		}
	}
	return out, nil
}

// Outcome is what happened to one case.
type Outcome struct {
	Case

	// Failed is whether any authoritative assessment failed the episode.
	Failed bool

	// Attributed is whether some assessment named the EXPECTED failure.
	//
	// This is the measure that counts. A repair acts on an attribution, so a
	// judge that notices something is wrong and calls it the wrong thing has
	// not helped — it has sent somebody to the wrong component. Scoring on
	// Failed alone would give full marks to a judge that fails everything.
	Attributed bool

	// Got is what the judges actually said, for reading the misses.
	Got []judge.Failure
}

// Report is the corpus-level result.
type Report struct {
	Outcomes []Outcome

	// Over the mutants the panel is supposed to be able to catch.
	DetectableTotal   int
	DetectableFound   int // right attribution
	DetectableNoticed int // failed, but possibly the wrong attribution

	// Over the mutants nobody claims to catch. Reported so the blind spot is
	// a number.
	BlindTotal int
	BlindFound int

	// Over the untouched controls.
	Controls    int
	FalseAlarms int
}

// Recall is right-attribution over the detectable mutants.
func (r Report) Recall() float64 {
	if r.DetectableTotal == 0 {
		return 0
	}
	return float64(r.DetectableFound) / float64(r.DetectableTotal)
}

// NoticeRate is how many detectable mutants failed at all, right or wrong.
//
// The gap between this and Recall is the misattribution rate: defects the
// panel saw and named incorrectly. That gap is invisible in any report that
// scores on pass/fail alone, and it is exactly the population that sends
// people to fix the wrong thing.
func (r Report) NoticeRate() float64 {
	if r.DetectableTotal == 0 {
		return 0
	}
	return float64(r.DetectableNoticed) / float64(r.DetectableTotal)
}

// FalseAlarmRate is the share of untouched controls the panel failed.
func (r Report) FalseAlarmRate() float64 {
	if r.Controls == 0 {
		return 0
	}
	return float64(r.FalseAlarms) / float64(r.Controls)
}

// Run judges every case and scores the panel against the answer key.
func Run(ctx context.Context, cases []Case, cfg judge.Config) (Report, error) {
	var rep Report
	for _, c := range cases {
		out := Outcome{Case: c}

		r, err := judge.Evaluate(ctx, c.Episode, cfg)
		if err != nil {
			return rep, fmt.Errorf("seed: judging %s: %w", c.Episode.Task.ID, err)
		}
		out.Failed = !r.Pass
		for _, a := range r.Assessments {
			if a.Pass {
				continue
			}
			out.Got = append(out.Got, a.Failure)
			if a.Failure == c.Expect && c.Expect != "" {
				out.Attributed = true
			}
		}
		sort.Slice(out.Got, func(i, j int) bool { return out.Got[i] < out.Got[j] })

		switch {
		case c.Defect == "":
			rep.Controls++
			if out.Failed {
				rep.FalseAlarms++
			}
		case c.Detectable:
			rep.DetectableTotal++
			if out.Attributed {
				rep.DetectableFound++
			}
			if out.Failed {
				rep.DetectableNoticed++
			}
		default:
			rep.BlindTotal++
			if out.Attributed {
				rep.BlindFound++
			}
		}
		rep.Outcomes = append(rep.Outcomes, out)
	}
	return rep, nil
}
