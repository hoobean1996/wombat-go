package judge_test

import (
	"math"
	"testing"

	"wombat/eval/judge"
)

func report(task string, pass bool, cause judge.Failure, repair judge.RepairTarget, reward float64) *judge.Report {
	return &judge.Report{
		Task: task, Pass: pass, Reward: reward,
		Diagnosis: judge.Diagnosis{RootCause: cause, Repair: repair, FirstBadStep: -1},
	}
}

// The interval is the whole reason this package reports one. A 4-for-4 result
// is NOT certainty, and the normal approximation everybody writes first says
// it is — producing an upper bound above 1 and a lower bound that pretends to
// more than four samples can carry.
func TestWilsonIntervalIsHonestAtSmallN(t *testing.T) {
	perfect := judge.Corpus{
		report("a", true, judge.None, judge.NoRepair, 1),
		report("a", true, judge.None, judge.NoRepair, 1),
		report("a", true, judge.None, judge.NoRepair, 1),
		report("a", true, judge.None, judge.NoRepair, 1),
	}
	rate, ci := perfect.PassRate()
	if rate != 1 {
		t.Fatalf("rate = %.2f, want 1", rate)
	}
	if ci.Hi > 1.0000001 || ci.Lo < 0 {
		t.Errorf("interval %s escapes [0,1]", ci)
	}
	// Wilson at 95% for 4/4 is about [0.51, 1.00].
	if math.Abs(ci.Lo-0.51) > 0.02 {
		t.Errorf("lower bound = %.3f, want about 0.51", ci.Lo)
	}
	if ci.Width() < 0.4 {
		t.Errorf("interval %s is too tight for four samples; a report reading that "+
			"would invite shipping on noise", ci)
	}

	three := judge.Corpus{
		report("a", true, judge.None, judge.NoRepair, 1),
		report("a", true, judge.None, judge.NoRepair, 1),
		report("a", true, judge.None, judge.NoRepair, 1),
		report("a", false, judge.ToolMisuse, judge.RepairToolPolicy, 0),
	}
	rate, ci = three.PassRate()
	if math.Abs(rate-0.75) > 1e-9 {
		t.Fatalf("rate = %.2f, want 0.75", rate)
	}
	// About [0.30, 0.95]: 75% over four samples says almost nothing.
	if math.Abs(ci.Lo-0.30) > 0.03 || math.Abs(ci.Hi-0.95) > 0.03 {
		t.Errorf("interval = %s, want about [0.30, 0.95]", ci)
	}
}

// The queue is the pipeline's actual output: not a score, a list of components
// ranked by how many episodes changing them would address.
func TestRepairQueueRanksComponentsByEpisodesAffected(t *testing.T) {
	c := judge.Corpus{
		report("a", false, judge.ToolMisuse, judge.RepairToolPolicy, 0),
		report("b", false, judge.ToolMisuse, judge.RepairToolPolicy, 0),
		report("c", false, judge.TunnelVision, judge.RepairPlanner, 0),
		report("d", false, judge.ToolMisuse, judge.RepairToolPolicy, 0),
		report("e", true, judge.None, judge.NoRepair, 1),
	}
	q := c.RepairQueue()
	if len(q) != 2 {
		t.Fatalf("queue has %d entries, want 2", len(q))
	}
	if q[0].Target != judge.RepairToolPolicy || q[0].Episodes != 3 {
		t.Errorf("head = %+v, want tool_policy with 3 episodes", q[0])
	}
	if math.Abs(q[0].Share-0.6) > 1e-9 {
		t.Errorf("share = %.2f, want 0.60 of five episodes", q[0].Share)
	}
	if q[0].Causes[judge.ToolMisuse] != 3 {
		t.Errorf("causes = %v, want tool_misuse x3", q[0].Causes)
	}
	if q[1].Target != judge.RepairPlanner {
		t.Errorf("second = %q, want planner", q[1].Target)
	}
}

// The disagreement slices are where a rubric gets audited. "judged passing but
// unrewarded" is the strongest available evidence that the MEASUREMENT rather
// than the agent needs repairing.
func TestOracleSliceSurfacesJudgeDisagreement(t *testing.T) {
	c := judge.Corpus{
		report("a", true, judge.None, judge.NoRepair, 1.0),
		report("b", false, judge.UnsafeAction, judge.RepairSafetyPolicy, 1.0),
		report("c", true, judge.None, judge.NoRepair, 0.0),
		report("d", false, judge.ToolMisuse, judge.RepairToolPolicy, 0.0),
	}
	keys, buckets := c.Slice(judge.ByOracle(0.5))
	want := map[string]int{
		"agreed_pass": 1, "agreed_fail": 1,
		"rewarded_but_judged_failing": 1, "judged_passing_but_unrewarded": 1,
	}
	if len(keys) != len(want) {
		t.Fatalf("slices = %v, want %d buckets", keys, len(want))
	}
	for k, n := range want {
		if len(buckets[k]) != n {
			t.Errorf("bucket %q has %d, want %d", k, len(buckets[k]), n)
		}
	}
}

// A criterion nobody judged must not vanish. A 100% pass rate over two judged
// samples out of forty is not a 100% pass rate.
func TestUnjudgedCriteriaSurviveAggregation(t *testing.T) {
	c := judge.Corpus{
		{Task: "a", Assessments: []judge.Assessment{
			{Criterion: judge.TaskSuccess, Score: 5, Pass: true, Judge: "rules", Confidence: 1},
		}, Unjudged: []judge.Criterion{judge.Correctness}},
		{Task: "b", Assessments: []judge.Assessment{
			{Criterion: judge.TaskSuccess, Score: 1, Pass: false, Judge: "rules",
				Failure: judge.PlanningFailure, Confidence: 1},
		}, Unjudged: []judge.Criterion{judge.Correctness}},
	}
	var seenCorrectness bool
	for _, s := range c.Aggregate() {
		if s.Criterion != judge.Correctness {
			continue
		}
		seenCorrectness = true
		if s.N != 0 || s.Unjudged != 2 {
			t.Errorf("correctness: N=%d unjudged=%d, want 0 and 2", s.N, s.Unjudged)
		}
	}
	if !seenCorrectness {
		t.Error("a criterion nobody judged disappeared from the aggregate entirely")
	}

	for _, s := range c.Aggregate() {
		if s.Criterion != judge.TaskSuccess {
			continue
		}
		if s.N != 2 || s.Passed != 1 {
			t.Errorf("task_success: N=%d passed=%d, want 2 and 1", s.N, s.Passed)
		}
		if s.Failures[judge.PlanningFailure] != 1 {
			t.Errorf("failures = %v, want one planning_failure", s.Failures)
		}
	}
}
