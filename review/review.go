// Package review runs an adversarial reviewer over code an agent just wrote,
// and turns what it finds into failing tests.
//
// # Why the artefact is a test
//
// Every earlier attempt in this project at model-produced findings reported
// prose, and prose has to be sorted by a person: of seven such findings, four
// were wrong, and each took a hand check to tell which. Requiring a finding to
// arrive as a test that FAILS against the code under review removes the sort.
// A claim that cannot be compiled into a failing test cannot be filed, and one
// that can is checked by running it.
//
// Measured on a labelled set of three defects a human review had already
// confirmed: the reviewer found two, filed no false positives, and deleted the
// one test it wrote that passed. Two rounds, sixteen tool calls, five cents.
//
// # What it is not
//
// It is not a second opinion on whether the task was done. The build agent's
// own tests already answer that, and they passed on every defect found here.
// It is a pass over what the requirements do not say — the consequence of a
// requirement that nobody wrote down, which is where every defect in this
// project's measurements has lived.
package review

import (
	"fmt"
	"strings"
	"time"

	"github.com/hoobean1996/wombat-go"
)

// Finding is one defect, identified by the test that demonstrates it.
type Finding struct {
	// Test is the Go test function name.
	Test string

	// File is the file it was written to, relative to the workspace.
	File string

	// Message is the failure output the test produced, which is the reviewer's
	// actual claim: what was expected and what happened.
	Message string

	// Fixed reports whether the test passes after a repair round.
	Fixed bool
}

// Report is the outcome of a review.
type Report struct {
	Findings []Finding

	// Ran is whether the reviewer completed. A review that failed to run is
	// not a clean review, and the distinction has to survive into the report:
	// zero findings from a reviewer that never started reads exactly like zero
	// findings from one that looked.
	Ran bool

	// Err is why the reviewer could not finish, if it could not.
	Err error

	// Unfinished is the text of every unticked box in the workspace's TODO
	// file. It is the author's own admission that something it planned was not
	// built, so it needs no model to confirm and cannot be a false positive.
	Unfinished []string

	// Discarded lists test-file sites that throw away a call's last return
	// value with `_`. Unlike Unfinished these do NOT gate Clean(): most are
	// harmless, and a rule that made every one of them count would be ignored
	// within a day. They are leads handed to the reviewer, and a count a person
	// can watch drift.
	Discarded []string

	// PlanEdited names planning artifacts the reviewer or the repair agent
	// changed during the run. They are inputs describing what the author
	// intended; an agent that rewrites them is editing the yardstick.
	PlanEdited []string

	// Sweeps is how many review passes actually ran, and Converged reports
	// whether the run ended because consecutive sweeps stopped finding
	// anything — as opposed to running out of its sweep budget, which is not
	// evidence of anything.
	Sweeps    int
	Converged bool

	Rounds    int
	Wall      time.Duration
	Workspace string
}

// Clean reports whether the review ran and found nothing.
// Clean reports that the reviewer looked and nothing is outstanding.
//
// An unticked TODO counts against it. The planner skill has the author write
// down what it intends to build before it builds; a box it left unticked is a
// gap it found itself, and a report that called that clean would be hiding the
// cheapest signal available.
func (r *Report) Clean() bool {
	return r.Ran && r.Err == nil && len(r.Findings) == 0 &&
		len(r.Unfinished) == 0 && len(r.PlanEdited) == 0
}

// Outstanding returns the findings still failing after any repair rounds.
func (r *Report) Outstanding() []Finding {
	var out []Finding
	for _, f := range r.Findings {
		if !f.Fixed {
			out = append(out, f)
		}
	}
	return out
}

// Summary renders one line for a caller to print.
func (r *Report) Summary() string {
	switch {
	case r.Err != nil:
		return fmt.Sprintf("review did not complete: %v", r.Err)
	case !r.Ran:
		return "review did not run"
	case len(r.PlanEdited) > 0 && len(r.Findings) == 0:
		return fmt.Sprintf("review found nothing, but the run rewrote %s", strings.Join(r.PlanEdited, ", "))
	case len(r.Findings) == 0 && len(r.Unfinished) == 0 && !r.Converged && r.Sweeps > 1:
		return fmt.Sprintf("review found nothing in %d sweep(s), but never converged", r.Sweeps)
	case len(r.Findings) == 0 && len(r.Unfinished) == 0:
		return "review found nothing"
	case len(r.Findings) == 0:
		return fmt.Sprintf("review found no defect, but %d planned item(s) were left unfinished", len(r.Unfinished))
	}
	fixed := 0
	for _, f := range r.Findings {
		if f.Fixed {
			fixed++
		}
	}
	var tail string
	if len(r.PlanEdited) > 0 {
		tail += fmt.Sprintf(", the run rewrote %s", strings.Join(r.PlanEdited, ", "))
	}
	if n := len(r.Unfinished); n > 0 {
		tail = fmt.Sprintf(", %d planned item(s) unfinished", n)
	}
	if r.Rounds > 0 {
		return fmt.Sprintf("review found %d defect(s), %d fixed, %d outstanding%s",
			len(r.Findings), fixed, len(r.Findings)-fixed, tail)
	}
	return fmt.Sprintf("review found %d defect(s)%s", len(r.Findings), tail)
}

// Config drives a review.
type Config struct {
	// Reviewer is the agent that reads the code. Required.
	//
	// A separate agent rather than another turn of the builder: the builder
	// has just spent its whole context arguing itself into believing the code
	// is right, and asking it to find its own defects asks it to overturn the
	// conclusion it just reached. The measured defects were all in what the
	// author had not thought of, which is exactly what a fresh reader is for.
	Reviewer *wombat.Agent

	// NewReviewer builds a fresh reviewer for each sweep. Required for more
	// than one sweep: asking the same agent to look again gets a report that
	// it already looked, not a second opinion.
	NewReviewer func() (*wombat.Agent, error)

	// Sweeps is the maximum number of review passes. Each gets a fresh agent
	// and a different lens. One (the default) is a single pass.
	Sweeps int

	// DryStreak is how many consecutive sweeps must find nothing new before
	// the search stops early. Zero uses the default.
	DryStreak int

	// Fixer repairs what the review finds. Nil reports findings without
	// attempting repair, which is the right default when a person is going to
	// read them.
	Fixer *wombat.Agent

	// Rounds is how many repair-and-recheck cycles to run. Zero reports only.
	Rounds int

	// Workspace is the directory holding the code under review.
	Workspace string

	// Task is what the code was supposed to do, passed to the reviewer so it
	// can tell a defect from a design decision.
	Task string

	// PlanFile and TodoFile are the planner skill's artifacts, read from
	// Workspace. Empty means the defaults; neither has to exist.
	//
	// The plan is given to the reviewer so it tests the code against the
	// invariants the AUTHOR wrote, not only against the task. That is aimed at
	// one specific measured failure: the author states a correct invariant and
	// then narrows it in the same breath, and every later step is consistent
	// with the narrowed version. Reviewing against the task cannot see that.
	// Reviewing against the author's own words makes it a visible mismatch.
	PlanFile string
	TodoFile string

	// TestTimeout bounds each `go test` invocation.
	TestTimeout time.Duration

	// Log receives progress lines. Nil discards them.
	Log func(string)
}

// DefaultTestTimeout bounds one test run.
const DefaultTestTimeout = 3 * time.Minute

// TestPrefix is the name every reviewer-written test must start with, so that
// a review's output can be run, re-run and shipped separately from the code's
// own tests.
const TestPrefix = "TestReview"

func (c Config) log(format string, args ...any) {
	if c.Log != nil {
		c.Log(fmt.Sprintf(format, args...))
	}
}

func (c Config) testTimeout() time.Duration {
	if c.TestTimeout > 0 {
		return c.TestTimeout
	}
	return DefaultTestTimeout
}
