package judge

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/hoobean1996/wombat-go/rl"
)

// Panel is the judges to try, in order.
//
// The first one that will rule on a criterion decides it. That ordering is the
// hybrid evaluator: put [Rules] first and a model second, and every criterion a
// rule can settle is settled deterministically and for free, while the model is
// spent only on the ones that genuinely need reading — correctness, reasoning
// quality, user alignment.
//
// Ordering it the other way is also legitimate and means something different:
// the model rules and the rules become a check on it. Both are useful; what is
// not useful is a panel whose order nobody chose.
type Panel []Judge

// Config drives one evaluation.
type Config struct {
	Rubrics []Rubric
	Panel   Panel

	// Strict rejects an assessment that does not [Assessment.Validate].
	// Default is to reject: a judge that returns a verdict with no evidence
	// has produced a number, not a judgement, and letting it through is how a
	// report fills up with unfalsifiable scores.
	Lenient bool

	// Authority is the confidence a non-passing assessment needs before it is
	// allowed to fail the episode on its own.
	//
	// Without it, a judge that has honestly reported low confidence still
	// gates everything downstream, and the pipeline acts on a verdict its own
	// author disowned. Measured: the rule judge's token-grounding check caps
	// itself at 0.50 because it has three known false-positive modes — a value
	// rendered differently in two places, an explanatory example, and usage
	// instructions that describe how to run something rather than reporting
	// that it was run. It nonetheless failed an episode that was correct in
	// every requirement, and put a component in the repair queue that had
	// nothing wrong with it.
	//
	// Below this bar an assessment is still recorded, still shown, and still
	// counted in the aggregate — it is demoted from a verdict to a concern,
	// not hidden. Zero keeps every assessment gating, which is the old
	// behaviour and is right for a panel of judges that never report low
	// confidence.
	Authority float64

	// Escalate is the confidence below which the panel keeps asking.
	//
	// Without it the panel is first-answer-wins, and a judge that says "I
	// looked, within my means, and found nothing" silently blocks every judge
	// behind it. That is not hypothetical: on the labelled pair in testdata/
	// the rule judge answers faithfulness at confidence 0.50 with a rationale
	// that literally says it cannot examine claims made in prose — and a panel
	// with a model behind it never asked the model. A separation test built on
	// that panel passed while measuring the judge documented as unable to
	// separate anything.
	//
	// Set it to just above what a self-limiting judge reports (0.6 suits the
	// judges here) and the panel escalates instead of settling. The
	// higher-confidence verdict wins; ties keep the earlier judge, so a
	// deterministic answer is preferred over a model one at equal confidence.
	//
	// Zero keeps first-answer-wins, which is right when the panel holds one
	// judge per criterion by construction.
	Escalate float64
}

// Errors from evaluation.
var (
	// ErrNoVerdict means the panel would not rule on anything at all.
	ErrNoVerdict = errors.New("judge: no judge in the panel would rule on any criterion")
)

// Evaluate runs the panel over the rubrics and assembles a report.
//
// An episode that died for a reason outside the agent's control comes back
// wrapped in [ErrUnjudgeable] rather than as a report full of ones.
func Evaluate(ctx context.Context, ep *rl.Episode, cfg Config) (*Report, error) {
	if ep == nil {
		return nil, fmt.Errorf("judge: nil episode")
	}
	if len(cfg.Rubrics) == 0 {
		cfg.Rubrics = DefaultRubrics()
	}
	if len(cfg.Panel) == 0 {
		cfg.Panel = Panel{Rules{}}
	}

	rep := &Report{
		Task:       ep.Task.ID,
		Sample:     ep.Task.Sample,
		Reward:     ep.Reward,
		RunFailure: ep.Failure,
	}

	for _, r := range cfg.Rubrics {
		a, err := assessOne(ctx, ep, r, cfg)
		switch {
		case errors.Is(err, ErrUnjudgeable):
			return nil, fmt.Errorf("%w: %s stopped as %q", ErrUnjudgeable, ep.Label(), ep.Failure)
		case err != nil:
			rep.Unjudged = append(rep.Unjudged, r.Criterion)
			continue
		}
		rep.Assessments = append(rep.Assessments, a)
	}
	if len(rep.Assessments) == 0 {
		return nil, ErrNoVerdict
	}

	rep.Overall, rep.Pass = score(rep, cfg.Rubrics, cfg.Authority)
	rep.Diagnosis = Diagnose(rep, cfg.Rubrics, cfg.Authority)
	return rep, nil
}

// assessOne walks the panel, escalating past a judge that is not confident.
func assessOne(ctx context.Context, ep *rl.Episode, r Rubric, cfg Config) (Assessment, error) {
	var (
		last error
		best Assessment
		have bool
	)
	for _, j := range cfg.Panel {
		a, err := j.Assess(ctx, ep, r)
		if errors.Is(err, ErrUnjudgeable) {
			return Assessment{}, err
		}
		if err != nil {
			last = err
			continue // this judge declines; try the next
		}
		if a.Judge == "" {
			a.Judge = j.Name()
		}
		if verr := a.Validate(r); verr != nil {
			if !cfg.Lenient {
				last = verr
				continue
			}
		}
		// Strictly greater, so an equal-confidence later judge does not
		// displace an earlier one: at the same confidence a deterministic
		// verdict is worth more than a model's, because it can be re-derived.
		if !have || a.Confidence > best.Confidence {
			best, have = a, true
		}
		if best.Confidence >= cfg.Escalate {
			return best, nil
		}
	}
	if have {
		return best, nil
	}
	if last == nil {
		last = fmt.Errorf("judge: nobody ruled on %s", r.Criterion)
	}
	return Assessment{}, last
}

// score computes the weighted overall and the pass conjunction.
//
// Pass is a conjunction and not a threshold on Overall, deliberately. An
// episode that was unsafe but fast must not average its way to a pass, and any
// single number over nine dimensions can be made to do exactly that.
func score(rep *Report, rubrics []Rubric, authority float64) (float64, bool) {
	byCrit := map[Criterion]Rubric{}
	for _, r := range rubrics {
		byCrit[r.Criterion] = r
	}

	var sum, weight float64
	pass := true
	for _, a := range rep.Assessments {
		r, ok := byCrit[a.Criterion]
		if !ok {
			continue
		}
		w := r.Weight
		if w <= 0 {
			w = 1
		}
		sum += r.Normalized(a.Score) * w
		weight += w

		// A failing assessment the judge itself is unsure of does not gate.
		// It stays in Assessments and in the aggregate; it just does not get
		// to decide the episode alone.
		if !a.Pass && a.Confidence >= authority {
			pass = false
		}
	}
	if weight == 0 {
		return 0, false
	}
	return sum / weight, pass
}

// Diagnose picks the root cause and where to repair it.
//
// The worst GATING failure wins, then the lowest score, then the earliest step.
// That ordering is the point: a safety failure at step 9 outranks a task
// failure at step 2, because a repair queue sorted by when things went wrong
// puts the cheapest bug first and the dangerous one last.
func Diagnose(rep *Report, rubrics []Rubric, authority float64) Diagnosis {
	gating := map[Criterion]bool{}
	for _, r := range rubrics {
		gating[r.Criterion] = r.Gating
	}

	d := Diagnosis{FirstBadStep: -1}
	for _, a := range rep.Assessments {
		if a.FirstBadStep >= 0 && (d.FirstBadStep < 0 || a.FirstBadStep < d.FirstBadStep) {
			d.FirstBadStep = a.FirstBadStep
		}
	}

	var failures []Assessment
	for _, f := range rep.Failures() {
		if f.Confidence >= authority {
			failures = append(failures, f)
		}
	}
	if len(failures) == 0 {
		if n := len(rep.Failures()); n > 0 {
			d.Summary = fmt.Sprintf("%d low-confidence concern(s), none authoritative enough to fail the episode", n)
		} else {
			d.Summary = fmt.Sprintf("all %d judged criteria passed", len(rep.Assessments))
		}
		return d
	}

	sort.SliceStable(failures, func(i, j int) bool {
		gi, gj := gating[failures[i].Criterion], gating[failures[j].Criterion]
		if gi != gj {
			return gi
		}
		if failures[i].Score != failures[j].Score {
			return failures[i].Score < failures[j].Score
		}
		return failures[i].FirstBadStep >= 0 && failures[i].FirstBadStep < failures[j].FirstBadStep
	})

	worst := failures[0]
	d.RootCause, d.Repair = worst.Failure, worst.Repair
	if d.Repair == NoRepair {
		d.Repair = RepairFor(worst.Failure)
	}

	// When the worst failure names no component — an oracle rejecting a result
	// without a view of how it was produced — take the repair from the next
	// failure that does, while keeping the root cause honest. Guessing a
	// target for an undetermined cause is how a queue recommends changing the
	// model because one verifier failed.
	if d.Repair == NoRepair {
		for _, f := range failures[1:] {
			t := f.Repair
			if t == NoRepair {
				t = RepairFor(f.Failure)
			}
			if t != NoRepair {
				d.Repair = t
				break
			}
		}
	}

	seen := map[RepairTarget]bool{d.Repair: true}
	for _, f := range failures[1:] {
		t := f.Repair
		if t == NoRepair {
			t = RepairFor(f.Failure)
		}
		if t != NoRepair && !seen[t] {
			seen[t] = true
			d.Also = append(d.Also, t)
		}
	}

	names := make([]string, 0, len(failures))
	for _, f := range failures {
		names = append(names, string(f.Criterion))
	}
	repair := string(d.Repair)
	if d.Repair == NoRepair {
		repair = "undetermined — run a judge that reads the trajectory"
	}
	d.Summary = fmt.Sprintf("%s (%s) on %s; repair %s",
		worst.Failure, worst.Criterion, strings.Join(names, ", "), repair)
	return d
}
