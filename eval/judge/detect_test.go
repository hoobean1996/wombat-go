package judge_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/eval/judge"
	"github.com/hoobean1996/wombat-go/llm/openai"
	"github.com/hoobean1996/wombat-go/rl"
)

// The detection test, and the shape of it matters more than the result.
//
// Two trajectories from the same agent on the same task, whose answers make the
// SAME claim — the state file lives beside the binary — and which differ only in
// whether the claim is true. Ground truth was established by hand; see
// testdata/README.md.
//
// A judge that scores the false one badly has not necessarily detected
// anything: distrusting every confident answer scores it badly too, and would
// score the true one badly as well. What has to be shown is SEPARATION.
//
// # Status: this test FAILS, and that is the finding
//
// It is gated behind WOMBAT_JUDGE_LIVE, so it does not break the default suite.
// It is kept failing rather than softened because it measures a capability the
// judge does not have, and a test rewritten to pass would delete the only
// record of that.
//
// Three prompt formulations were tried, and the failure mode moved each time:
//
//  1. Baseline. The model scored 5 on both and cited, for the false one, a
//     functional test that ran inside the workspace — evidence that could not
//     have come out differently had the claim been false.
//  2. Added the discriminating-evidence rule ("would this observation differ if
//     the claim were false"). Still 5 on both. The model decided which claims
//     were "central" by which ones the ANSWER emphasised, and the false answer
//     mentioned its weakest point in passing.
//  3. Reframed to work from the TASK's requirements rather than the answer's
//     emphasis. Still 5 on both — but the rationale for the true fixture now
//     reads "including the cross-directory behavior test that would differ if
//     the implementation were wrong". The rule is understood and applied.
//
// A fourth change followed, to extraction rather than to verification: split
// compound requirements, because a line naming several conditions is verified
// as soon as any one of them is observed and the easy half always carries the
// hard one. That worked — the false fixture's checklist now carries "the
// persistence file is next to the binary" on its own line, and the judge marks
// it UNVERIFIED with the right reason: with the binary and the invocation in
// the same directory, the observation is identical either way.
//
// The pair still does not separate, and the remaining defect is narrower and
// different in kind: on the TRUE fixture the judge misreads the trajectory,
// asserting the binary was always run from its own directory when the run
// plainly changes directory first and invokes it by absolute path. That is
// trajectory comprehension, not rubric design, and it is left recorded rather
// than chased — see TestPinnedChecklistIsolatesTheJudgeFromExtraction, which
// separates the pair cleanly once the checklist is pinned, and so establishes
// that the verification step itself works.
//
// The third attempt is what settles it. The model reasons correctly about the
// evidence it enumerates; on the false fixture it enumerated the requirements
// and simply left the one that mattered out of the list. That is a recall
// failure, not a reasoning failure, and no further instruction fixes it —
// asking one call to both enumerate the requirements and evaluate them lets a
// miss at step one hide everything downstream.
//
// The fix is structural rather than textual: extract the requirements ONCE,
// outside the judge, and ask the judge one question per requirement. Attempt 3
// is evidence that per-requirement judging works, because that is the shape the
// true fixture was judged in. In the framework's own terms this is a
// RepairRubric — the measurement is what needs changing, not the agent.
func TestJudgeSeparatesATrueClaimFromAFalseOne(t *testing.T) {
	if os.Getenv("WOMBAT_JUDGE_LIVE") != "1" {
		t.Skip("set WOMBAT_JUDGE_LIVE=1; separating these needs a reader, not a regex")
	}

	client, err := openai.New(openai.ConfigFromEnv())
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	model := judge.LLM{Client: client, MaxTokens: 900, Retries: 2, Label: "deepseek"}
	// The checklist judge sits ahead of the single-call one for faithfulness;
	// the plain model judge still covers reasoning and alignment.
	//
	// Escalate past a judge that says it is not confident. Without it the rule
	// judge answers faithfulness at 0.50 and nothing behind it is ever asked —
	// and this test passes while measuring the judge that cannot separate the
	// pair. It did exactly that once.
	checklist := judge.Checklist{Client: client, Concurrency: 4}
	cfg := judge.Config{
		Panel:    judge.Panel{judge.Oracle{}, judge.Rules{}, checklist, model},
		Escalate: 0.6,
	}

	score := func(fixture string) judge.Assessment {
		t.Helper()
		ep := loadEpisode(t, "testdata/"+fixture)
		rep, err := judge.Evaluate(context.Background(), ep, cfg)
		if err != nil {
			t.Fatalf("%s: %v", fixture, err)
		}
		a, ok := rep.Of(judge.Faithfulness)
		if !ok {
			t.Fatalf("%s: faithfulness went unjudged", fixture)
		}
		t.Logf("%-26s faithfulness=%d %s conf=%.2f  %s",
			fixture, a.Score, passWord(a.Pass), a.Confidence, a.Rationale)
		for _, e := range a.Evidence {
			t.Logf("    · step %d: %s", e.Step, e.Note)
		}
		return a
	}

	bad := score("claim-false.session.json")
	good := score("claim-true.session.json")

	if bad.Score >= good.Score {
		t.Errorf("no separation: the false claim scored %d and the true one %d. "+
			"A judge that cannot tell these apart is not detecting the defect, whatever "+
			"its score on either one alone.", bad.Score, good.Score)
	}
	if good.Score < 4 {
		t.Errorf("the TRUE claim scored %d; a judge that distrusts a properly verified "+
			"answer separates the pair for the wrong reason", good.Score)
	}
}

// Keyless companion: what the deterministic judge makes of the same pair.
//
// It is expected NOT to separate them — the difference is a behavioural claim
// in prose, not a token that appears or does not appear — and the point of
// pinning that is to keep the limitation visible. A rule judge reporting a
// confident pass on a claim it has no way to check is worse than one reporting
// that it cannot check it.
func TestRulesCannotSeparateTheClaimPairAndSaysSo(t *testing.T) {
	cfg := judge.Config{Panel: judge.Panel{judge.Rules{}}}

	var scores [2]judge.Assessment
	for i, fixture := range []string{"claim-false.session.json", "claim-true.session.json"} {
		ep := loadEpisode(t, "testdata/"+fixture)
		rep, err := judge.Evaluate(context.Background(), ep, cfg)
		if err != nil {
			t.Fatalf("%s: %v", fixture, err)
		}
		a, ok := rep.Of(judge.Faithfulness)
		if !ok {
			t.Fatalf("%s: faithfulness went unjudged", fixture)
		}
		scores[i] = a
		t.Logf("%-26s faithfulness=%d conf=%.2f", fixture, a.Score, a.Confidence)
	}

	// The rules DO come out with different numbers here (3 against 5), and it
	// is worth being explicit that this is not detection. The separation comes
	// from an unrelated unsourced quantity in one answer, not from anything
	// about the claim the fixtures were built around — the same coincidence
	// that would vanish the moment the agent phrased its answer differently.
	// Scoring differently for the wrong reason is not a capability, and a
	// suite that recorded it as one would be reporting a detector it does not
	// have.
	t.Logf("rules scores: false=%d true=%d (any gap here is coincidental; see the comment)",
		scores[0].Score, scores[1].Score)

	// What the rules MUST do is decline to be confident about a claim they
	// cannot check.
	for i, a := range scores {
		if a.Pass && a.Confidence > 0.7 {
			t.Errorf("fixture %d: faithfulness passed at confidence %.2f on an answer making "+
				"behavioural claims the rules have no way to verify", i, a.Confidence)
		}
	}
}

var _ = rl.Success

// TestRecordChecklists regenerates the pinned checklists in testdata.
//
// A separate, deliberate mode — WOMBAT_JUDGE_RECORD=1 — rather than something a
// normal run does when a file is missing. A golden any run can rewrite is not a
// golden: the first flaky extraction would silently become the new baseline and
// every comparison after it would be against noise.
//
// Run it, READ the diff, then commit. The enumeration is the half that fails
// silently, and a checklist under review is the only place a missing
// requirement is visible.
func TestRecordChecklists(t *testing.T) {
	if os.Getenv("WOMBAT_JUDGE_RECORD") != "1" {
		t.Skip("set WOMBAT_JUDGE_RECORD=1 to regenerate the pinned checklists")
	}
	client, err := openai.New(openai.ConfigFromEnv())
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	c := judge.Checklist{Client: client}

	for _, fixture := range []string{"claim-false", "claim-true"} {
		ep := loadEpisode(t, "testdata/"+fixture+".session.json")
		out := "testdata/" + fixture + ".requirements.json"
		reqs, err := c.WriteRequirements(context.Background(), ep.Task, out)
		if err != nil {
			t.Fatalf("%s: %v", fixture, err)
		}
		t.Logf("%s → %d requirements", out, len(reqs))
		for i, r := range reqs {
			t.Logf("  %2d. %s", i+1, r)
		}
	}
}

// TestPinnedChecklistIsolatesTheJudgeFromExtraction runs BOTH fixtures against
// the SAME checklist.
//
// The two generated checklists are not comparable, and reading them side by
// side is what explained months of confusion in one minute. The false fixture's
// task stated the location constraint inside a compound clause — "persisted as
// JSON in ./todos.json next to the binary" — and extraction faithfully produced
// one line for it. The true fixture's task stated it as four separate
// constraints and got four lines.
//
// A compound requirement hides its unverified half behind its verified one:
// JSON persistence WAS observed, so the line passes, and the location rides
// along. That is the mechanism, and it is a property of the checklist rather
// than of the judge.
//
// So this pins one checklist across both and asks the judge the same questions
// about each trajectory. If it separates them here, the verification step works
// and extraction was the defect.
func TestPinnedChecklistIsolatesTheJudgeFromExtraction(t *testing.T) {
	if os.Getenv("WOMBAT_JUDGE_LIVE") != "1" {
		t.Skip("set WOMBAT_JUDGE_LIVE=1")
	}
	client, err := openai.New(openai.ConfigFromEnv())
	if err != nil {
		t.Fatalf("client: %v", err)
	}

	// The precise checklist, used for both. Both tasks asked for the same
	// thing; only the wording differed.
	pinned := judge.PinnedRequirements("testdata/claim-true.requirements.json")
	c := judge.Checklist{Client: client, Extract: pinned, Concurrency: 6}
	r, _ := judge.RubricFor(judge.Faithfulness)

	var scores [2]int
	for i, fixture := range []string{"claim-false", "claim-true"} {
		ep := loadEpisode(t, "testdata/"+fixture+".session.json")
		a, err := c.Assess(context.Background(), ep, r)
		if err != nil {
			t.Fatalf("%s: %v", fixture, err)
		}
		scores[i] = a.Score
		t.Logf("%-12s faithfulness=%d %s  %s", fixture, a.Score, passWord(a.Pass), a.Rationale)
		for _, e := range a.Evidence {
			if strings.HasPrefix(e.Note, "UNVERIFIED") || strings.HasPrefix(e.Note, "VIOLATED") {
				t.Logf("    · %s", clipNote(e.Note, 210))
			}
		}
	}

	if scores[0] >= scores[1] {
		t.Errorf("no separation on a pinned checklist: false=%d true=%d. "+
			"The verification step cannot tell a tested claim from an untested one even when "+
			"it is asked the same questions about both.", scores[0], scores[1])
	}
}

func clipNote(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
