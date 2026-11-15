package review

import (
	"errors"
	"strings"
	"testing"

	"wombat/pkg/core"
)

type wombatAgent = core.Agent

var errNoAgent = errors.New("no agent")

func TestLensRotatesSoSweepsDoNotRepeatEachOther(t *testing.T) {
	seen := map[string]bool{}
	for sweep := 1; sweep <= len(lenses); sweep++ {
		l := lensFor(sweep)
		if seen[l.name] {
			t.Fatalf("sweep %d reused lens %q", sweep, l.name)
		}
		seen[l.name] = true
	}
	if got := lensFor(len(lenses) + 1); got.name != lenses[0].name {
		t.Errorf("lens did not wrap: got %q, want %q", got.name, lenses[0].name)
	}
	if got := lensFor(0); got.name != lenses[0].name {
		t.Errorf("lensFor(0) = %q, want the open lens", got.name)
	}
	// Every lens after the first must actually narrow the search, or rotating
	// them buys nothing.
	for _, l := range lenses[1:] {
		if strings.TrimSpace(l.text) == "" {
			t.Errorf("lens %q has no text", l.name)
		}
	}
}

// Reusing one agent across sweeps is a single pass wearing a hat: it reports
// that it already looked. Refusing is better than reporting sweeps that could
// not have been independent.
func TestSecondSweepRefusesToReuseOneReviewer(t *testing.T) {
	cfg := Config{Reviewer: nil}
	if _, err := cfg.reviewerFor(2); err == nil {
		t.Fatal("sweep 2 accepted a single Reviewer")
	}
	calls := 0
	cfg = Config{NewReviewer: func() (*wombatAgent, error) { calls++; return nil, errNoAgent }}
	_, _ = cfg.reviewerFor(1)
	_, _ = cfg.reviewerFor(2)
	if calls != 2 {
		t.Fatalf("NewReviewer called %d time(s), want one per sweep", calls)
	}
}

// A run that used up its sweep budget has not shown the code is clean; a run
// that converged has. The report must not read the same either way.
func TestNotConvergedIsNotTheSameAsFoundNothing(t *testing.T) {
	ranOut := &Report{Ran: true, Sweeps: 3, Converged: false}
	if s := ranOut.Summary(); !strings.Contains(s, "never converged") {
		t.Errorf("a run that used up its budget reads as clean: %q", s)
	}
	converged := &Report{Ran: true, Sweeps: 3, Converged: true}
	if s := converged.Summary(); s != "review found nothing" {
		t.Errorf("converged summary = %q", s)
	}
	if !converged.Clean() {
		t.Error("a converged review with no findings is not Clean")
	}
}

// A truncated reviewer leaves an Err. That must defeat Clean(): the measured
// case reported one shallow finding after being cut off at its iteration cap,
// which reads exactly like a thorough review that found little.
func TestATruncatedSweepIsNotClean(t *testing.T) {
	rep := &Report{Ran: true, Sweeps: 1, Converged: true, Err: errNoAgent}
	if rep.Clean() {
		t.Fatal("a review that did not finish reported Clean")
	}
}

func TestPromptCarriesTheLensAndRefusesRepeats(t *testing.T) {
	known := []Finding{
		{Test: "TestReviewDeadLetter", Message: "claim returned an error\nsecond line", Fixed: true},
	}
	p := reviewPrompt("t", "", nil, lensFor(2), known)
	if !strings.Contains(p, "THIS SWEEP: WHAT HAPPENS WHEN THINGS FAIL") {
		t.Error("the sweep's lens is missing from the prompt")
	}
	if !strings.Contains(p, "TestReviewDeadLetter (repaired)") {
		t.Error("an already-found defect was not listed as already found")
	}
	if strings.Contains(p, "second line") {
		t.Error("known findings are not trimmed to one line; they will crowd out the instructions")
	}
	if !strings.Contains(p, "do not report these again") {
		t.Error("nothing tells the sweep to skip what is already known")
	}
}

// A dry sweep before every lens has run is a claim about the lenses that ran,
// not about the code. The measured case: sweeps 1 and 2 came back empty, the
// run reported convergence, and the defect the third lens finds was still
// there — silently returning another task's value.
func TestDryStreakRequiresEveryLensToHaveRun(t *testing.T) {
	if minToConverge := len(lenses) + defaultDryStreak - 1; minToConverge <= len(lenses) {
		t.Fatalf("the minimum to converge (%d) does not exceed the lens count (%d); "+
			"a run could converge without trying every lens", minToConverge, len(lenses))
	}
}

// Convergence must be reachable at the documented minimum and no sooner.
func TestConvergenceIsUnreachableBelowFullLensCoverage(t *testing.T) {
	n := len(lenses)
	for sweeps := 1; sweeps < n+defaultDryStreak-1; sweeps++ {
		dry := 0
		for sweep := 1; sweep <= sweeps; sweep++ {
			if sweep < n {
				continue // not counted: lenses still untried
			}
			dry++
		}
		if dry >= defaultDryStreak {
			t.Fatalf("%d sweeps reached a %d-sweep dry streak; every lens has not run",
				sweeps, defaultDryStreak)
		}
	}
	sweeps := n + defaultDryStreak - 1
	dry := 0
	for sweep := 1; sweep <= sweeps; sweep++ {
		if sweep < n {
			continue
		}
		dry++
	}
	if dry < defaultDryStreak {
		t.Fatalf("%d sweeps (the documented minimum) still cannot converge: dry=%d", sweeps, dry)
	}
}
