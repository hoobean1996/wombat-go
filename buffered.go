package wombat

import (
	"context"
	"iter"
	"slices"
	"sync"

	"github.com/hoobean1996/wombat-go/llm"
)

// bufferCap is how many events one [Buffered] retains.
//
// Sized for a whole run rather than for a screenful. A reasoning model emits
// on the order of 500 deltas per turn and the usual iteration cap is around a
// dozen, so 8192 holds a normal run end to end: a client that reconnects gets
// everything it missed, not a window. The memory is bounded by the same
// number — a few megabytes per session in the worst case, which is what makes
// it safe to keep finished sessions around for a TTL.
//
// Overflow drops the OLDEST retained events, never the newest. Dropping the
// newest would stall a live follower behind a permanent gap; dropping the
// oldest costs a resuming client only the part of the transcript it is most
// likely to have already rendered. See [Buffered.Follow] for how a follower
// detects that it happened.
const bufferCap = 8192

// Buffered decouples a Run from its consumers.
//
// It gives up the backpressure that [Run] provides. A Run is unbuffered and
// single-consumer, so the agent runs exactly as fast as whoever calls
// [Run.Next]; wrapping it here means a goroutine drains it as fast as the
// provider produces, and the events pile up in memory (bounded — see below)
// whether or not anyone is reading. That is the right trade for a server: a
// run that stops when nobody is watching cannot be resumed, and resumption is
// the entire point. A dropped connection has to leave the work going, or the
// reconnect has nothing to reconnect to.
//
// What is given up is real, so it is bounded rather than ignored. At most
// bufferCap (8192) events are retained; past that the oldest are evicted and
// their sequence numbers can no longer be served.
//
// Every method is safe to call from any goroutine, including concurrently
// with [Buffered.Follow].
type Buffered struct {
	run *Run

	mu sync.Mutex
	// events[i] carries sequence base+i. base rises only on eviction.
	events []Event
	base   int
	// changed is closed — and replaced — on every append, and closed once
	// more when the run ends. A per-wait channel rather than a sync.Cond
	// because a follower has to be able to wake on its own ctx too, and
	// Cond.Wait cannot be selected on.
	changed  chan struct{}
	finished bool

	done chan struct{}
}

// Buffer starts draining r immediately, on its own goroutine, and returns a
// view that any number of consumers can follow independently.
//
// Draining starts now, not on the first [Buffered.Follow]: the reason to
// buffer at all is that the run must survive having no consumer.
//
// The returned Buffered takes ownership of r — it will close it — so the
// caller must not use r afterwards. Panics if r is nil.
func Buffer(r *Run) *Buffered {
	if r == nil {
		panic("wombat: Buffer(nil)")
	}
	b := &Buffered{
		run:     r,
		changed: make(chan struct{}),
		done:    make(chan struct{}),
	}
	go b.drain()
	return b
}

// Follow yields (sequence, event) pairs starting at from, blocking for events
// that have not been produced yet.
//
// It returns when the run ends and the buffer is exhausted, or when ctx is
// done, whichever comes first. Breaking out of the loop early is fine and
// affects nothing else: followers are independent, several can run at once at
// different positions, and a reconnect that overlaps the previous
// connection's teardown is the normal case rather than an edge one.
//
// Sequence numbers start at 0, are dense, and never change — that is what a
// Last-Event-ID resume is built on. from is clamped: a negative from starts at
// 0, and a from below the oldest retained sequence starts at the oldest
// retained one. That clamp is the only way eviction is observable, so a
// consumer that cares whether it missed anything compares the first sequence
// it is yielded against the one it asked for.
//
// A from beyond the end is not an error; it simply blocks until the run
// produces that many events, or ends.
func (b *Buffered) Follow(ctx context.Context, from int) iter.Seq2[int, Event] {
	return func(yield func(int, Event) bool) {
		seq := max(from, 0)
		for {
			b.mu.Lock()
			// Clamp inside the loop, not once up front: eviction can pass a
			// slow follower at any point, not just before its first read.
			seq = max(seq, b.base)
			var batch []Event
			if i := seq - b.base; i < len(b.events) {
				batch = slices.Clone(b.events[i:])
			}
			// Both taken under the same lock as the snapshot. Reading
			// "finished" any later would race with the final append: the run
			// can end between the unlock and the wait, and a follower that
			// concluded "done, nothing pending" from a stale pair would drop
			// the last events of the run on the floor.
			wait, finished := b.changed, b.finished
			b.mu.Unlock()

			for _, ev := range batch {
				if !yield(seq, ev) {
					return
				}
				seq++
			}
			if len(batch) > 0 {
				continue
			}
			if finished {
				return
			}

			select {
			case <-wait:
			case <-ctx.Done():
				return
			}
		}
	}
}

// Len reports how many events the run has produced, so sequence numbers run
// from 0 to Len()-1. Some of the early ones may already have been evicted;
// this is the position of the tail, not the size of the buffer.
func (b *Buffered) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.base + len(b.events)
}

// Done is closed once the run has ended and every event it produced is in the
// buffer. [Buffered.Outcome] and [Buffered.Err] are meaningful from then on.
func (b *Buffered) Done() <-chan struct{} { return b.done }

// Outcome reports how the run ended, or nil if it failed. Valid once
// [Buffered.Done] is closed.
func (b *Buffered) Outcome() Outcome { return b.run.Outcome() }

// Err reports why the run failed, or nil. Valid once [Buffered.Done] is
// closed.
func (b *Buffered) Err() error { return b.run.Err() }

// Messages returns a snapshot of the transcript so far. Safe at any time.
func (b *Buffered) Messages() []llm.Message { return b.run.Messages() }

// Close cancels the run and waits for the draining goroutine to exit.
// Idempotent, and safe to call after the run has finished on its own.
//
// It waits rather than just signalling so that a caller dropping the last
// reference — a session registry evicting an entry, say — knows the goroutine
// and the run's context are gone by the time Close returns.
func (b *Buffered) Close() error {
	err := b.run.Close()
	<-b.done
	return err
}

// drain is the goroutine that keeps the run moving with no consumer attached.
func (b *Buffered) drain() {
	defer func() {
		// Close the run before publishing the end, so that by the time a
		// follower observes Done the run's context is already cancelled and
		// its resources released. Idempotent, and a no-op cost here because
		// the event channel is already closed.
		b.run.Close()

		b.mu.Lock()
		b.finished = true
		b.wake()
		b.mu.Unlock()

		close(b.done)
	}()

	for b.run.Next() {
		b.append(b.run.Event())
	}
}

func (b *Buffered) append(ev Event) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.events = append(b.events, ev)
	if over := len(b.events) - bufferCap; over > 0 {
		copy(b.events, b.events[over:])
		// Clear the vacated tail so the evicted events — a tool_done can
		// carry a large output — are actually collectable.
		clear(b.events[bufferCap:])
		b.events = b.events[:bufferCap]
		b.base += over
	}
	b.wake()
}

// wake releases every follower parked on the current generation. Callers hold
// b.mu.
func (b *Buffered) wake() {
	close(b.changed)
	b.changed = make(chan struct{})
}
