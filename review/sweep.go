package review

import (
	"fmt"

	wombat "github.com/hoobean1996/wombat-go"
)

// defaultDryStreak is how many consecutive sweeps must find nothing new before
// the search is treated as converged.
//
// One is too few: a sweep can come back empty because its lens pointed away
// from the remaining defects, not because there are none. Two is the cheapest
// number that requires two different lenses to agree.
const defaultDryStreak = 2

// lens is what one sweep is told to look for. Sweeps with the same prompt tend
// to rediscover the same defect: the five defects a hand review found after the
// first sweep were in Nack, the CLI and the schema, while the sweep had been
// reading the dead-letter path. Rotating the lens is what makes a second sweep
// worth its cost.
type lens struct {
	name string
	text string
}

var lenses = []lens{
	{
		name: "open",
		text: "", // the standing instructions, unnarrowed
	},
	{
		name: "error paths",
		text: `THIS SWEEP: WHAT HAPPENS WHEN THINGS FAIL

Go through every exported function and ask what it returns when the thing it
depends on fails — the write fails, the row is missing, the deadline passes,
the input is malformed.

For each, check two things separately:

- What is the caller TOLD? An error that is wrapped into the wrong type, or a
  sentinel that differs from the one a sibling function returns for the same
  condition, is a defect the happy path will never reveal.
- What is TRUE afterwards? A call that half-applied its change and then
  reported failure leaves state no test asserted on.`,
	},
	{
		name: "shared state",
		text: `THIS SWEEP: TWO MECHANISMS, ONE PIECE OF STATE

Find every piece of state written by more than one code path, and every field
or column that is written but never read.

- A value written and never read means an intended check was designed and then
  not implemented. Find what it was for, and test whether that thing holds.
- A read followed by a write, outside a transaction or lock, is a decision made
  on a value that can be stale by the time it is used. The race detector does
  not see this: it is a logical race, not a data race.`,
	},
	{
		name: "entry points",
		text: `THIS SWEEP: WHAT A USER ACTUALLY RUNS

Review the command-line entry points, argument parsing and the output a person
sees, not the library underneath.

- An argument that fails to parse and is silently treated as a default.
- A flag or subcommand documented in the task and not implemented, or
  implemented with different behaviour.
- What is printed on failure: an exit code, and a message a user can act on
  rather than a raw driver error.`,
	},
	{
		name: "coverage gaps",
		text: `THIS SWEEP: WHAT NOTHING LOOKS AT

Do not start from the code. Start from the list of behaviours the task
requires, and find the ones no test exercises at all.

Then, for each existing test, ask whether the path it drives is the path its
name claims. A surface with no test is a lead; a surface with a test that
passes for the wrong reason is a better one.`,
	},
}

func lensFor(sweep int) lens {
	if sweep <= 0 {
		return lenses[0]
	}
	return lenses[(sweep-1)%len(lenses)]
}

// reviewerFor returns the agent for a sweep. A fresh agent per sweep is the
// point: the same agent asked to look again reports that it already looked.
func (c Config) reviewerFor(sweep int) (*wombat.Agent, error) {
	if c.NewReviewer != nil {
		a, err := c.NewReviewer()
		if err != nil {
			return nil, err
		}
		if a == nil {
			return nil, fmt.Errorf("NewReviewer returned no agent")
		}
		return a, nil
	}
	if sweep > 1 {
		// Reusing one agent across sweeps degrades to a single pass; say so
		// rather than reporting sweeps that could not have been independent.
		return nil, fmt.Errorf("sweep %d needs NewReviewer: one Reviewer cannot review twice independently", sweep)
	}
	return c.Reviewer, nil
}
