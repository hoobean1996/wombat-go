# Reviewer agent — first measurement

Does a model reading the CODE adversarially find what a model writing the code
misses? Measured against a labelled set: three defects I had already confirmed
by hand in the job queue, each with a reproducing probe.

The distinction that made this worth testing: every judge experiment in this
project fed the judge a **trajectory**. All six defects were found by reading
**source**. Concluding from the first that model review cannot work was an
overreach, and this measurement refutes it.

## Result

| defect | found | how |
|---|---|---|
| D3 poison pill retried forever | **yes** | `TestReviewDeadOnRepeatedCrash` |
| D2 Nack not scoped to the claim holder | **yes** | `TestReviewStaleNackStealsClaim` |
| D1 Ack not scoped to the claim holder | no | — |

**Recall 2/3. Precision 2/2.** Two rounds, 22 tool calls, about five cents.

Both tests were run independently before being counted. Both assert something
the specification requires, so neither is an over-reach:

- D3: the implementation itself increments `attempts` on the stale-reclaim
  path, so its own logic treats a crash as an attempt. It simply never checks
  the count there.
- D2: the visibility timeout exists so a dead worker loses nothing. If a stale
  caller can pull the job back out of the new worker's hands, the reclaim it
  performed means nothing.

D3 is the one I predicted it would miss, on the grounds that finding it means
reading `Claim` and `Nack` together and noticing that a requirement's
consequence was never written down. It found it, and its test is three orders
of magnitude faster than mine: it backdates `updated_at` to expire the claim
instead of sleeping through the timeout.

## The constraint that made precision free

Every finding had to arrive with a Go test that FAILS against this
implementation and would pass against a correct one, run and shown.

It wrote a third test — cross-instance claim atomicity — watched it pass, and
**deleted it**. No finding was reported for it.

That is the first time in this project that a model-produced finding did not
need a human to sort the real ones from the false ones. Seven earlier
attributions from the transcript-reading judges needed exactly that, and four
of the seven were wrong. Making falsifiability a **deliverable** rather than an
instruction dissolves the problem: a claim that cannot be compiled into a
failing test cannot be filed.

Worth carrying back: the `Checklist` judge currently reports prose. Under this
constraint none of its three known false-positive modes — a value rendered two
ways, an explanatory example, usage instructions read as claims — could produce
a finding at all.

## What it did not do

It found one half of a pair. `Ack` and `Nack` carry the identical defect in
adjacent functions, and once the `Nack` scoping was in hand, checking `Ack` was
one line of reading away. It stopped.

So the reviewer is not a replacement for a second pass; it is a very cheap
first pass that reliably finds things the author missed. The obvious next
experiments are whether a second reviewer round with the first round's findings
in hand closes the gap, and whether several reviewers with different lenses
cover more than one with a general brief.

## Caveats

- One task, three defects, one run per round. The jobq family showed zero
  variance across six generation samples, but that says nothing about the
  variance of review.
- I told it where to look in round two: "the paths that RELEASE a job rather
  than acquire it". That is a real hint and it precedes the D2 finding. Round
  one, with no such hint, found D3 on its own. The honest split is 1/3
  unaided and 2/3 with one lens supplied.
