---
name: planner
install: true
description: How to turn a task into a written plan, a test plan, and a TODO file before writing code — and how to use that file while implementing. Installed: this is standing method, not reference material, and applies to any task that will produce more than one file.
---

# Plan first, in a file

A plan that exists only in your reasoning cannot be checked, cannot be
returned to after twenty tool calls, and cannot be compared against what you
actually built. Write it down.

Two files, both at the root of the workspace, both written **before** the
first line of implementation:

- `PLAN.md` — what is true when this is done, and how you will know.
- `TODO.md` — the same plan as an ordered checklist you tick off.

## 1. State the invariants as facts about state

This is the step that decides whether the rest is worth anything.

An invariant is a statement that is true of the system **at every moment**. It
is not a statement about a transition. The difference is the single most
common way a correct plan turns into an incorrect program:

    GOOD  A job is owned by at most one worker at any moment.
    BAD   The ready -> claimed transition happens exactly once.

The second sentence sounds like the first and is much weaker. It constrains
one edge. Everything that takes ownership away — release, timeout, crash,
retry, shutdown — is outside it, and code written against it will guard
acquisition perfectly and leave release unguarded.

So write each invariant as `<subject> is <predicate> at all times`, and then,
for each one, write the line that follows from it:

    Invariant: a job is owned by at most one worker at any moment.
    Therefore: every path that GIVES ownership must be exclusive,
               and every path that TAKES ownership away must verify
               the caller is the current owner.
    Paths that give:  Claim, stale-reclaim
    Paths that take:  Ack, Nack, timeout expiry

List both columns. If one column is empty you have written a transition, not
an invariant; rewrite it.

## 2. Write the test plan against the invariants, not the features

For every invariant, name the test that would fail if it broke — including
the half you are least likely to implement. A test plan that only tests the
paths in the "gives" column is the same bug as the bad invariant above, one
level up.

Also name, for each one, the test you would write if you were trying to
**break** it rather than confirm it: the concurrent case, the out-of-order
case, the crash-between-two-writes case, the input a thousand times bigger
than the example.

Write this into `PLAN.md` as a table: invariant, test name, what makes it
fail.

## 3. Turn the plan into TODO.md

`TODO.md` is the plan in the only form you will actually reread. Plain
markdown checkboxes, ordered so that nothing depends on something below it:

    # TODO
    - [ ] queue.go: schema + Open/Close
    - [ ] Claim: exclusive under concurrency (invariant 1, gives)
    - [ ] Ack: verify caller is current owner (invariant 1, takes)
    - [ ] Nack: verify caller is current owner (invariant 1, takes)
    - [ ] test: TestConcurrentClaimIsExclusive
    - [ ] test: TestAckByNonOwnerIsRejected

Every invariant's "takes" column must appear here as its own item. That is the
whole point of the file: the items you would otherwise forget are written down
before you are deep enough in the code to forget them.

Tests from the test plan are TODO items too.

## 4. While implementing: read it, tick it, do not edit around it

- Reread `TODO.md` when you finish an item. Not from memory — read the file.
- Tick an item only when the code for it exists and its test passes:
  `- [x]`.
- If you find the plan was wrong, change the plan **and say so in PLAN.md**
  under a `## Revisions` heading, with the reason. Silently dropping an item
  is how the release half disappears.
- Never delete an item to make the list look finished.

## 5. Verifier.md, if one is present

A `Verifier.md` in the workspace is a list of things that must be TRUE of the
finished work, written before the code existed. Each item looks like this:

    ## V1. dividing by zero reports the line it happened on
    Given: a program whose third line divides by zero
    Expect: an *Error with Line == 3
    Status: [ ]
    Test:

Your job is to settle every item, and there are exactly two ways.

**Prove it.** Write a test that decides the item, run it, and record it:

    Status: [x]
    Test: TestDivideByZeroCarriesItsLine

The tick is checked. An item whose test does not exist, or does not pass, is
not settled — writing `[x]` next to a test you did not run fails the same way
as leaving it blank, and more expensively, because it looks finished.

**Withdraw it, with an argument.** An item written before the code existed can
ask for something the task never required. That is a defect in the item, not in
your work, and you may refuse it:

    Status: [-] withdrawn: the task specifies only "64-bit signed" and says
    nothing about overflow behaviour, so wrapping is not a defect

Quote the task. "This seems unnecessary" is not an argument, and a withdrawal
without one is treated as an unsettled item.

Do not delete items. Do not reword an item so that a test you already have
happens to satisfy it. Both are ways of changing the measurement instead of
doing the work, and both are checked.

## 6. Before you say you are done

Read `TODO.md` one more time and check that every box is ticked, and
`Verifier.md` for any item still on `[ ]`. If one is not done, you are not
done: either finish it, or leave it unticked and say plainly in your final
message which item is unfinished and why. An unticked box is a better outcome
than a false one — it will be read.
