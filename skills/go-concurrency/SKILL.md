---
name: go-concurrency
description: Reviewing Go code for data races, goroutine leaks and context misuse. Load before auditing concurrent code; not needed for ordinary Go questions.
---

# Reviewing Go concurrency

## Goroutine leaks

A goroutine blocked on a send that nobody will receive never exits, and the
process leaks it. Every `go func()` needs an answer to: what unblocks this if
the consumer disappears?

The two correct shapes:

```go
select {
case ch <- v:
case <-ctx.Done():
}
```

or a buffered channel sized to the number of sends, so the send cannot block.

## Shared mutable state

Look for a field written by one goroutine and read by another with no
synchronisation. Common carriers:

- a map used as a cache
- a counter incremented per call
- a slice appended to from a fan-out

`go test -race` finds these only on paths the test exercises, so read for them
as well as running for them.

## Context

- `context.Context` carries cancellation and request-scoped values. It does
  not carry dependencies; a struct field does.
- A function that takes a `ctx` must pass it to everything it calls that can
  block. A `ctx` accepted and ignored is worse than none, because it looks
  cancellable.
- `context.WithCancel` without a matching `defer cancel()` leaks a timer.

## What not to flag

- A goroutine that runs for the process's lifetime and is meant to (a
  background flusher, a signal handler) is not a leak.
- Unsynchronised access is fine when a value is written once before any
  goroutine starts and only read afterwards. Say so instead of flagging it.
