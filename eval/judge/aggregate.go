package judge

import (
	"fmt"
	"math"
	"sort"
)

// Corpus is a set of reports to aggregate over.
type Corpus []*Report

// Interval is a two-sided confidence interval on a proportion.
//
// Reported everywhere a pass rate is, because a rate with no interval is the
// single most misleading number an eval produces: 3 of 4 samples passing reads
// as 75% and is consistent with a true rate anywhere from about 30% to 95%. A
// team that ships on the first number and not the second will chase noise for
// weeks.
type Interval struct {
	Lo, Hi float64
}

// String renders the interval for a report column.
func (i Interval) String() string { return fmt.Sprintf("[%.2f, %.2f]", i.Lo, i.Hi) }

// Width is the interval's span; a wide one means the sample is too small to
// conclude anything from.
func (i Interval) Width() float64 { return i.Hi - i.Lo }

// wilson is the Wilson score interval at 95%.
//
// Wilson rather than the normal approximation, which is the one everybody
// writes first and which is wrong exactly where an agent eval lives: with
// small n and a rate near 0 or 1 it produces intervals that run past the ends
// of the scale, so a 4-for-4 result gets an upper bound above 1.0 and a lower
// bound that pretends to more certainty than four samples can carry.
func wilson(successes, n int) Interval {
	if n <= 0 {
		return Interval{}
	}
	const z = 1.959963985 // 95%
	p := float64(successes) / float64(n)
	nf := float64(n)
	denom := 1 + z*z/nf
	center := (p + z*z/(2*nf)) / denom
	half := (z / denom) * math.Sqrt(p*(1-p)/nf+z*z/(4*nf*nf))
	return Interval{Lo: math.Max(0, center-half), Hi: math.Min(1, center+half)}
}

// Summary aggregates one criterion across a corpus.
type Summary struct {
	Criterion Criterion

	N        int
	Passed   int
	PassRate float64
	CI       Interval

	MeanScore float64
	MinScore  int

	// Unjudged is how many reports had no verdict on this criterion. Carried
	// next to N rather than folded into it: a 100% pass rate over two judged
	// samples out of forty is not a 100% pass rate.
	Unjudged int

	// Failures counts the taxonomy across the non-passing assessments.
	Failures map[Failure]int
}

// Aggregate summarizes every criterion present in the corpus, in Known order.
func (c Corpus) Aggregate() []Summary {
	byCrit := map[Criterion]*Summary{}
	ensure := func(cr Criterion) *Summary {
		s, ok := byCrit[cr]
		if !ok {
			s = &Summary{Criterion: cr, MinScore: math.MaxInt32, Failures: map[Failure]int{}}
			byCrit[cr] = s
		}
		return s
	}

	for _, rep := range c {
		for _, u := range rep.Unjudged {
			ensure(u).Unjudged++
		}
		for _, a := range rep.Assessments {
			s := ensure(a.Criterion)
			s.N++
			s.MeanScore += float64(a.Score)
			if a.Score < s.MinScore {
				s.MinScore = a.Score
			}
			if a.Pass {
				s.Passed++
			} else {
				s.Failures[a.Failure]++
			}
		}
	}

	order := append([]Criterion(nil), Known...)
	for cr := range byCrit {
		found := false
		for _, k := range order {
			if k == cr {
				found = true
				break
			}
		}
		if !found {
			order = append(order, cr)
		}
	}

	out := make([]Summary, 0, len(byCrit))
	for _, cr := range order {
		s, ok := byCrit[cr]
		if !ok {
			continue
		}
		if s.N > 0 {
			s.MeanScore /= float64(s.N)
			s.PassRate = float64(s.Passed) / float64(s.N)
			s.CI = wilson(s.Passed, s.N)
		} else {
			s.MinScore = 0
		}
		out = append(out, *s)
	}
	return out
}

// PassRate is the share of reports where every judged criterion passed.
func (c Corpus) PassRate() (float64, Interval) {
	if len(c) == 0 {
		return 0, Interval{}
	}
	passed := 0
	for _, r := range c {
		if r.Pass {
			passed++
		}
	}
	return float64(passed) / float64(len(c)), wilson(passed, len(c))
}

// ===== slicing =====

// Slicer maps a report to the bucket it belongs in.
type Slicer func(*Report) string

// ByTask slices by task id.
func ByTask(r *Report) string { return r.Task }

// ByRunFailure slices by how the run stopped — rl's vocabulary.
func ByRunFailure(r *Report) string {
	if r.RunFailure == "" {
		return "unknown"
	}
	return string(r.RunFailure)
}

// ByRootCause slices by the diagnosed failure.
func ByRootCause(r *Report) string {
	if r.Diagnosis.RootCause == None {
		return "none"
	}
	return string(r.Diagnosis.RootCause)
}

// ByRepair slices by where the diagnosis says to act.
func ByRepair(r *Report) string {
	if r.Diagnosis.Repair == NoRepair {
		return "none"
	}
	return string(r.Diagnosis.Repair)
}

// ByOracle slices by whether the oracle accepted the episode, which is the one
// slice that shows the judges disagreeing with ground truth.
//
// The two interesting buckets are the disagreements. "rewarded but judged
// failing" is usually a real defect the oracle cannot see — a correct patch
// arrived at by unsafe means. "judged passing but unrewarded" is usually a
// defect in the rubric, and is the strongest evidence available that the
// measurement rather than the agent needs repairing.
func ByOracle(threshold float64) Slicer {
	return func(r *Report) string {
		ok := r.Reward >= threshold
		switch {
		case ok && r.Pass:
			return "agreed_pass"
		case !ok && !r.Pass:
			return "agreed_fail"
		case ok && !r.Pass:
			return "rewarded_but_judged_failing"
		default:
			return "judged_passing_but_unrewarded"
		}
	}
}

// Slice buckets a corpus. Keys come back sorted, so two runs render alike.
func (c Corpus) Slice(by Slicer) ([]string, map[string]Corpus) {
	out := map[string]Corpus{}
	for _, r := range c {
		k := by(r)
		out[k] = append(out[k], r)
	}
	keys := make([]string, 0, len(out))
	for k := range out {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys, out
}

// ===== the optimization decision =====

// Repair is one entry in the queue: a component, and what fixing it would buy.
type Repair struct {
	Target RepairTarget

	// Episodes is how many failing episodes name this component.
	Episodes int

	// Share is Episodes over the whole corpus.
	Share float64

	// Causes counts the failures that pointed here, so a target backed by one
	// failure mode can be told from a target that is merely popular.
	Causes map[Failure]int
}

// RepairQueue ranks the components to change, most episodes first.
//
// This is the pipeline's actual output. Everything upstream — criteria,
// rubrics, judges, scores, aggregation — exists so that this list can be
// produced without a person re-reading the traces, and a pipeline that stops
// at a score has not finished.
//
// Only the ROOT cause of each report counts toward a target, not every
// implicated one. A failing episode usually lights up three or four criteria
// downstream of a single mistake, and counting them all would rank the
// symptoms above the cause.
func (c Corpus) RepairQueue() []Repair {
	byTarget := map[RepairTarget]*Repair{}
	for _, r := range c {
		if r.Pass || r.Diagnosis.Repair == NoRepair {
			continue
		}
		t := r.Diagnosis.Repair
		e, ok := byTarget[t]
		if !ok {
			e = &Repair{Target: t, Causes: map[Failure]int{}}
			byTarget[t] = e
		}
		e.Episodes++
		e.Causes[r.Diagnosis.RootCause]++
	}

	out := make([]Repair, 0, len(byTarget))
	for _, e := range byTarget {
		if len(c) > 0 {
			e.Share = float64(e.Episodes) / float64(len(c))
		}
		out = append(out, *e)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Episodes != out[j].Episodes {
			return out[i].Episodes > out[j].Episodes
		}
		return out[i].Target < out[j].Target
	})
	return out
}

// Undiagnosed returns the failing reports that name no component to repair.
//
// They must never be summarised away. An episode that failed with an
// undetermined cause is the most important one in a corpus — it is the one
// nobody knows what to do about — and a queue that silently skips it reports
// "nothing to fix" over a real failure. That happened here: adding an honest
// "cause undetermined" verdict made a failing episode vanish from the queue,
// which is a worse answer than the invented repair target it replaced.
func (c Corpus) Undiagnosed() Corpus {
	var out Corpus
	for _, r := range c {
		if !r.Pass && r.Diagnosis.Repair == NoRepair {
			out = append(out, r)
		}
	}
	return out
}

// Concerns returns assessments that did not pass but were not confident enough
// to gate their episode.
//
// Shown rather than dropped. A judge saying "this looks wrong and I am not
// sure" is worth reading — it is often where a real defect and a false
// positive are indistinguishable without a human — but it is not worth acting
// on, and the two need different places in a report.
func (c Corpus) Concerns() []Assessment {
	var out []Assessment
	for _, r := range c {
		for _, a := range r.Assessments {
			if !a.Pass && a.Confidence < 0.7 {
				out = append(out, a)
			}
		}
	}
	return out
}

// FailureHistogram counts root causes across the corpus.
func (c Corpus) FailureHistogram() map[Failure]int {
	out := map[Failure]int{}
	for _, r := range c {
		if !r.Pass && r.Diagnosis.RootCause != None {
			out[r.Diagnosis.RootCause]++
		}
	}
	return out
}
