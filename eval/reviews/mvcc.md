# Code review — MVCC key/value store

Task: a transactional KV store with snapshot isolation, write-write conflict
detection, GC of dead versions, and crash-safe persistence. 16 iterations,
15 tool calls, $0.05.

Two passes. The first checked the requirements with eight probes written
**before the implementation existed**, from the task statement alone. The
second read the 12 KB of implementation looking for what the requirements do
not describe. The two passes found different things, and only the second found
defects.

| pass | probes | result |
|---|---|---|
| requirements, written blind | 9 | 9 pass |
| code review | 3 | **3 confirmed defects** |

Probes are kept beside this file as `.txt` so they do not compile into the
suite; drop them into the package to reproduce.

---

## Confirmed defects

### 1. GC counts versions it does not release

```go
db.versions[key] = vs[cutoff:]
```

`vs` is `[]*version`. Reslicing forward shares the backing array, so every
`*version` before the cutoff is still referenced by it and stays alive. GC
returns a count of slice entries dropped, not of memory freed.

Measured: after 20 overwrites of one key, GC reported reclaiming 19 versions
and **all 19 were still reachable** through the original slice header.

This is the whole purpose of the function. A long-lived key never gives its
history back, and the number GC reports is the number that makes it look like
it did.

Fix: copy into a fresh slice, or nil the dropped entries before reslicing.

Pinned by `TestReviewGCActuallyReleasesTheVersionsItCounts`.

**A note on how this was nearly missed.** The first version of that probe
asserted on `cap()`, saw it fall from 32 to 13, and passed. That drop is just
`origCap - offset` arithmetic and says nothing about whether the prefix was
released. A green tick stood for a claim it had not checked, and the finding
was only recovered by rewriting the probe to hold the original slice header
and look.

### 2. A failed persist leaves the commit applied

`finish` appends the new versions and bumps the clock, and only then calls
`persistLocked`. When persistence fails the error is returned — and the writes
are already visible to every transaction that begins afterwards.

Measured: with the database directory made unwritable, `Commit` returned
`permission denied` and the value it wrote was readable from the next
transaction.

A caller cannot handle this. Retrying re-applies a write that already landed;
reporting failure tells the user their data is gone when it is not. The write
should be staged and applied only after the persist succeeds, or the in-memory
state should be rolled back on failure.

Pinned by `TestReviewFailedPersistDoesNotLeaveTheCommitApplied`.

### 3. Deleted keys are never forgotten

A delete writes a tombstone version. GC keeps the newest version at or below
the oldest live snapshot, so a tombstone survives as the head forever and the
map entry with it.

Measured: 500 keys created, all 500 deleted, GC run — **500 map entries
remain**.

A create/delete workload grows without bound however often GC is called, and
the persisted file grows with it. A tombstone older than every live snapshot,
with nothing newer, can be dropped along with its key.

Pinned by `TestReviewDeletedKeysAreEventuallyForgotten`.

---

## Observations, not defects

**Each key of a multi-key transaction gets its own commit version.**
`db.curVersion++` is inside the per-key loop, so a transaction writing A and B
produces two version numbers, assigned in Go map iteration order — which is
randomised. No reader can observe the split, because `Begin` and `finish` both
hold `db.mu` and no snapshot can land between the two increments. But the same
logical transaction produces different version numbers on different runs,
which makes a persisted file non-reproducible and would defeat a byte-for-byte
comparison of two runs of the same workload.

**Every commit rewrites the whole database.** `persistLocked` serialises all
head state and `writeAtomic` writes and fsyncs it. Correct, and O(size) per
commit.

**The persistence itself is well done.** Temp file in the same directory,
write, fsync, close, rename, then fsync the directory. That is the correct
sequence and it is rarely written correctly first time.

---

## What the agent claimed

The final answer opens "Everything builds, vets, and passes `-race` cleanly",
which is true, and lists the required API item by item, which is accurate. It
also volunteers that it added one method beyond the specification —
`(*DB) Snapshot()`, for the CLI's `dump` — explains why, and offers to remove
it. Declaring a deviation rather than hiding it is the right behaviour and is
worth recording alongside the defects.

None of the three defects above are mentioned, and none contradicts anything
the answer claims: they live in what the requirements do not say.
