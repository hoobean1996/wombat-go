package review

import (
	"fmt"
	"strings"
)

// reviewPrompt is the contract that makes a finding checkable.
//
// The load-bearing clause is that a finding must arrive as a test that fails.
// Without it a reviewer reports prose, and prose has to be sorted by a person:
// of seven such findings earlier in this project, four were wrong. With it,
// the reviewer sorts its own output — the measured run wrote a third test,
// watched it pass, and deleted it rather than filing it.
func reviewPrompt(task, plan string, discarded []string, ln lens, known []Finding) string {
	var b strings.Builder

	b.WriteString(`This directory holds finished Go code. It builds, it vets, its own tests pass. Treat none of that as evidence that it is correct: every defect worth finding here was in code whose tests were green.

Your job is adversarial review. Find defects the existing tests do not.

`)

	if strings.TrimSpace(task) != "" {
		b.WriteString("WHAT THE CODE WAS ASKED TO DO\n\n")
		b.WriteString(strings.TrimSpace(task))
		b.WriteString("\n\n")
	}

	if strings.TrimSpace(plan) != "" {
		b.WriteString("WHAT THE AUTHOR PLANNED, IN ITS OWN WORDS\n\n")
		b.WriteString(strings.TrimSpace(plan))
		b.WriteString(`

USING THE PLAN

The plan is a lead, not an authority. Where the plan and the task disagree,
the task is right and the plan is a defect in itself.

Three things to do with it:

1. Take each invariant AT FULL STRENGTH, exactly as written, and test that.
   The specific failure this is here to catch: an invariant stated as a fact
   that holds at every moment ("owned by at most one worker at any moment"),
   implemented as if it constrained a single transition (only the code that
   TAKES ownership is guarded). Every such invariant has paths that give and
   paths that take. Find the paths that take. Test those.
2. The plan names tests. Check each one exists, and that it tests what the
   plan says it tests rather than something weaker with the same name.
3. An item the plan describes and the code does not contain is worth a test
   even if nothing else points at it.

`)
	}

	b.WriteString(`HOW TO REPORT A FINDING

A finding is only a finding if it comes with a test that fails.

1. Write it in a file whose name starts with "review" and ends in "_test.go",
   in the package it tests.
2. Name the function ` + TestPrefix + `Something.
3. Run it. It MUST FAIL against this code, and its message must say what was
   expected and what happened instead.
4. It must PASS against a correct implementation. A test that fails because it
   asserts something the task never asked for is not a finding: delete it.

Run every test you write. Report only the ones you have seen fail. If you
suspect something and cannot make a test fail, say so in one line as a
suspicion and do not count it.

Do not modify the code under review. Your output is tests.

`)

	if ln.text != "" {
		b.WriteString(strings.TrimSpace(ln.text))
		b.WriteString("\n\n")
	}

	if len(known) > 0 {
		b.WriteString("ALREADY FOUND BY EARLIER SWEEPS — do not report these again\n\n")
		for _, f := range known {
			state := "outstanding"
			if f.Fixed {
				state = "repaired"
			}
			fmt.Fprintf(&b, "  %s (%s): %s\n", f.Test, state, firstLine(f.Message))
		}
		b.WriteString(`
Their tests are in the tree. A repaired one may have been fixed narrowly, so
the same cause elsewhere is still worth a test — but it must be a DIFFERENT
failure, not a restatement of one above.

`)
	}

	b.WriteString(`A GREEN TEST IS NOT A TESTED PATH

Read the existing tests before writing new ones. The highest-value defect found
in this project so far was in code that already HAD a correct test: the test
drove exactly the right path and passed anyway, because the call's error was
assigned to the blank identifier and only the other return value was asserted.
The function was returning an error that killed its caller, and the suite was
green.

For each existing test, ask two questions:

- What does it throw away? A result assigned to the blank identifier is a
  result nobody checked, and a call can fail for the wrong reason underneath a
  passing assertion.
- Does it assert what its NAME claims? A test called ...DeadLetters that only
  checks that a boolean is false will pass when the call errored instead.

When a test passes over a broken path, the finding is a NEW failing test. Do
not edit the existing one.

`)

	if len(discarded) > 0 {
		b.WriteString("Sites where a test discards a call's last return value (by Go convention\nthat is the error). These are leads, not defects — most are harmless:\n\n")
		n := min(len(discarded), 40)
		for _, d := range discarded[:n] {
			b.WriteString("  " + d + "\n")
		}
		if len(discarded) > n {
			fmt.Fprintf(&b, "  ... and %d more\n", len(discarded)-n)
		}
		b.WriteString("\n")
	}

	b.WriteString(`WHERE DEFECTS LIVE

Requirements say what should happen. Defects gather in what FOLLOWS from a
requirement and is not written in it. In particular:

- A resource acquired in one place and released in another. The check that
  guards acquisition is usually there; the matching check on release is
  usually not.
- A requirement of the form "after T, X becomes true". It almost always
  implies something that must STOP being allowed once X is true, and that
  second half is rarely written down or tested.
- Two mechanisms that touch the same state. Each is usually right alone.
- A function that reports how much work it did. Check that it did it.
- An operation that changes memory and then persists. Ask what the caller is
  told, and what is true, when the persist fails.

KEEP REPLIES SHORT

One or two sentences per finding. A long reply gets truncated and ends the
run before you have reported anything.`)

	return b.String()
}

// fixPrompt hands the failing tests back to a repair agent.
//
// It forbids touching the tests, and that prohibition is enforced rather than
// trusted: a finding is credited as fixed only when its test still exists and
// now passes. Deleting the demonstration is the cheapest way to make a red
// test green, and it is the one repair that leaves the defect in place.
func fixPrompt(task string, findings []Finding) string {
	var b strings.Builder

	b.WriteString(`An adversarial review of this code found defects. Each one is demonstrated by a failing test that is already in the tree.

Fix the CODE so that these tests pass.

`)
	for i, f := range findings {
		fmt.Fprintf(&b, "%d. %s (%s)\n   %s\n", i+1, f.Test, f.File, f.Message)
	}

	b.WriteString(`
RULES

- Do not edit, weaken, rename or delete any test. They are the specification
  of what is wrong. A finding counts as fixed only when its own test still
  exists and passes, so removing one repairs nothing and hides the defect.
- Do not break the code's existing tests. Run them.
- Do not edit PLAN.md or TODO.md. They record what the author set out to build
  and are the yardstick this repair is measured against. Rewriting a plan to
  match the code, or deleting checklist items so the list looks finished,
  changes the measurement instead of the defect. This is checked.
- Fix the cause, not the symptom the test happens to observe. If two of these
  tests have the same cause, one repair should satisfy both.

Keep replies short: a long reply gets truncated and ends the run.`)

	if strings.TrimSpace(task) != "" {
		b.WriteString("\n\nWHAT THE CODE WAS ASKED TO DO\n\n")
		b.WriteString(strings.TrimSpace(task))
	}
	return b.String()
}

// firstLine keeps the known-findings list to one line per entry: the full
// messages of several sweeps would crowd out the instructions.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 160 {
		s = s[:160] + "..."
	}
	return s
}
