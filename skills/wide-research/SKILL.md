---
name: wide-research
install: true
description: How to open a build task — survey the domain, name the invariants, and write acceptance criteria before writing code. Installed: this is standing method, not reference material, and applies to any task with concurrency, persistence, failure recovery, or more than one moving part.
---

# Opening a build task

Most defects in a finished system are not mistakes in the code that was
written. They are consequences nobody worked out before the code was written,
and they survive testing because the tests were written from the same
understanding that produced the bug.

So the work before the first line of code is not overhead. It is the only phase
in which those consequences are cheap to find.

## 1. Survey before deciding

Find out how this problem is normally solved before inventing an approach.

- What is the standard technique for this, and what does it protect against?
  A technique usually exists because a specific thing went wrong for somebody.
  Knowing what it protects against tells you whether you need it.
- What does the tooling you are about to use already guarantee, and what does
  it explicitly not? Read the documentation for the guarantee, not for the API.
- What are the known sharp edges? Most mature libraries and protocols have a
  short list of things everyone gets wrong the first time.

Write down what you learned in one or two sentences per point. If a decision
later contradicts one of them, you will notice.

## 2. Name the invariants

An invariant is a statement that must be true at every instant, not just at the
end of a successful run. Write them as sentences that could be false.

Good: "a unit of work is owned by at most one worker at any moment."
Useless: "the system handles concurrency correctly."

Aim for the smallest set that, if all held, would make the system correct.
Usually three to six.

## 3. Attack each invariant

For every invariant, ask what could make it false. Work through these
deliberately rather than trusting that nothing comes to mind:

- **Concurrency.** Two actors doing this at the same instant. Two actors in
  different processes. One actor arriving while another is halfway through.
- **Interruption.** The process dies between any two steps. Which two steps is
  the worst pair? What state is left behind, and who cleans it up?
- **Time.** Something takes far longer than expected. A deadline expires while
  the work it bounded is still running. Two clocks disagree.
- **Repetition.** The same request arrives twice. An operation is retried after
  it already partly succeeded.
- **The unhappy path.** An operation fails after changing some state but not
  all of it.

For each attack that would break the invariant, decide now: prevent it, detect
it, or accept it and say so.

## 4. Look hardest where two mechanisms meet

Single mechanisms are usually implemented correctly, because the requirement
that describes one is easy to read and easy to test. The defects cluster where
two of them interact, because no requirement describes the interaction and so
nothing tests it.

Take every pair of mechanisms in your design and ask what happens when both are
active at once. Especially: when one mechanism reverses or reassigns something
the other one is relying on, does the other one find out?

A requirement of the form "X happens after T" almost always implies a second
requirement about what may no longer happen once X has happened. That second
requirement is rarely written down, and it is where the defect lives.

## 5. Turn invariants into acceptance criteria

For each invariant, write the check that would catch it being false, before
writing the code it checks.

A criterion is only worth writing if you can answer: **what observation would
come out differently if this were broken?** If the answer is "nothing I can
run", the criterion is a wish. Either find an observation or write down that
this one is unverified.

Prefer checks that exercise the attack from step 3 — concurrent actors, an
interrupted process, an expired deadline, a duplicate request — over checks
that confirm the happy path. The happy path is what the implementation was
written to do; it is the least informative thing to test.

## 6. Then build, and check the criteria you wrote

Implement, then run the criteria from step 5 as written. If a criterion is
inconvenient to check now, that is information about the design, not a reason
to drop the criterion.

When you report what you did, separate what you **verified** from what you
**believe**. A claim you did not test is worth stating as untested; stating it
as verified is the one failure that makes every other number in your report
untrustworthy.
