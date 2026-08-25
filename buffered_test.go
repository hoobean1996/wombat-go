package wombat

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

// countingClient emits n text deltas and then answers, with no dependence on
// wall clock.
func countingClient(n int) llm.Client {
	return llm.ClientFunc(func(_ context.Context, req llm.Request) (llm.Response, error) {
		for i := range n {
			if req.OnDelta != nil {
				req.OnDelta(llm.Delta{Text: strconv.Itoa(i) + " "})
			}
		}
		return llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: "done"}},
			StopReason: llm.StopEndTurn,
		}, nil
	})
}

func TestBufferNilPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Buffer(nil) did not panic")
		}
	}()
	Buffer(nil)
}

// The whole point of Buffered: the run must make progress and finish with
// nobody following it. A Run alone is consumer-paced and would stall.
func TestBufferedRunAdvancesWithNoConsumer(t *testing.T) {
	a := newAgent(t, countingClient(20), nil)
	b := Buffer(a.Start(context.Background(), Ask("count")))
	t.Cleanup(func() { _ = b.Close() })

	<-b.Done()

	if got := b.Len(); got == 0 {
		t.Fatal("Len() = 0 after the run finished; nothing was buffered")
	}
	if err := b.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
	ans, ok := b.Outcome().(Answer)
	if !ok {
		t.Fatalf("Outcome() = %T, want Answer", b.Outcome())
	}
	if ans.Text != "done" {
		t.Errorf("Answer.Text = %q, want %q", ans.Text, "done")
	}
	if len(b.Messages()) == 0 {
		t.Error("Messages() is empty")
	}
}

func TestBufferedFollowResumeAndReplay(t *testing.T) {
	a := newAgent(t, countingClient(20), nil)
	b := Buffer(a.Start(context.Background(), Ask("count")))
	t.Cleanup(func() { _ = b.Close() })

	<-b.Done()
	total := b.Len()
	if total < 6 {
		t.Fatalf("the run produced %d events, want at least 6 for this test", total)
	}

	// Follow from the start and stop early, the way a dropped connection does.
	var firstLeg []int
	ctx1, cancel1 := context.WithCancel(context.Background())
	defer cancel1()
	for seq := range b.Follow(ctx1, 0) {
		firstLeg = append(firstLeg, seq)
		if len(firstLeg) == 5 {
			cancel1()
			break
		}
	}
	if len(firstLeg) != 5 {
		t.Fatalf("first leg read %d events, want 5", len(firstLeg))
	}
	if firstLeg[0] != 0 || firstLeg[4] != 4 {
		t.Errorf("first leg sequences = %v, want a dense run from 0", firstLeg)
	}

	// Reconnect exactly where we left off.
	var secondLeg []int
	for seq := range b.Follow(context.Background(), 5) {
		secondLeg = append(secondLeg, seq)
	}
	if len(secondLeg) == 0 || secondLeg[0] != 5 {
		t.Fatalf("second leg starts at %v, want 5", secondLeg[:min(3, len(secondLeg))])
	}
	if got, want := len(firstLeg)+len(secondLeg), total; got != want {
		t.Errorf("the two legs cover %d events, want %d — there is a gap or a duplicate", got, want)
	}
	for i, seq := range secondLeg {
		if seq != 5+i {
			t.Fatalf("second leg is not dense: got %v", secondLeg)
		}
	}

	// A late follower replays everything.
	late := 0
	for range b.Follow(context.Background(), 0) {
		late++
	}
	if late != total {
		t.Errorf("a late follower saw %d events, want the whole run (%d)", late, total)
	}
}

// A follower attached while the run is still going must see the rest of it,
// including the events produced after it started waiting.
func TestBufferedFollowIsLive(t *testing.T) {
	gate := make(chan struct{})
	cl := llm.ClientFunc(func(_ context.Context, req llm.Request) (llm.Response, error) {
		req.OnDelta(llm.Delta{Text: "before "})
		<-gate
		req.OnDelta(llm.Delta{Text: "after "})
		return llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: "done"}},
			StopReason: llm.StopEndTurn,
		}, nil
	})
	a := newAgent(t, cl, nil)
	b := Buffer(a.Start(context.Background(), Ask("go")))
	t.Cleanup(func() { _ = b.Close() })

	type result struct {
		seqs  []int
		texts []string
	}
	out := make(chan result, 1)
	go func() {
		var r result
		for seq, ev := range b.Follow(context.Background(), 0) {
			r.seqs = append(r.seqs, seq)
			if d, ok := ev.(TextDelta); ok {
				r.texts = append(r.texts, d.Text)
			}
		}
		out <- r
	}()

	close(gate)

	var r result
	select {
	case r = <-out:
	case <-time.After(10 * time.Second):
		t.Fatal("the live follower never returned")
	}

	if len(r.texts) != 2 || r.texts[0] != "before " || r.texts[1] != "after " {
		t.Errorf("deltas = %v, want [before  after ]", r.texts)
	}
	for i, seq := range r.seqs {
		if seq != i {
			t.Fatalf("sequences are not dense from 0: %v", r.seqs)
		}
	}
	if len(r.seqs) != b.Len() {
		t.Errorf("follower saw %d events, want %d", len(r.seqs), b.Len())
	}
}

func TestBufferedFollowStopsWhenTheContextIsDone(t *testing.T) {
	// A client that never returns keeps the run open, so Follow can only
	// return because its own ctx was cancelled.
	block := make(chan struct{})
	t.Cleanup(func() { close(block) })
	cl := llm.ClientFunc(func(ctx context.Context, _ llm.Request) (llm.Response, error) {
		select {
		case <-block:
		case <-ctx.Done():
		}
		return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: "x"}}, StopReason: llm.StopEndTurn}, nil
	})
	a := newAgent(t, cl, nil)
	b := Buffer(a.Start(context.Background(), Ask("go")))
	t.Cleanup(func() { _ = b.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan int, 1)
	go func() {
		n := 0
		for range b.Follow(ctx, 0) {
			n++
		}
		done <- n
	}()
	cancel()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Follow did not return when its context was cancelled")
	}
}

// from beyond the end is not an error; it blocks until the run gets there or
// ends. Here the run ends first, so it yields nothing and returns.
func TestBufferedFollowFromBeyondTheEnd(t *testing.T) {
	a := newAgent(t, countingClient(3), nil)
	b := Buffer(a.Start(context.Background(), Ask("go")))
	t.Cleanup(func() { _ = b.Close() })
	<-b.Done()

	n := 0
	for range b.Follow(context.Background(), b.Len()+100) {
		n++
	}
	if n != 0 {
		t.Errorf("got %d events past the end, want 0", n)
	}
}

func TestBufferedCloseIsIdempotent(t *testing.T) {
	a := newAgent(t, countingClient(5), nil)
	b := Buffer(a.Start(context.Background(), Ask("go")))

	if err := b.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// Close waits for the draining goroutine, so Done is already closed.
	select {
	case <-b.Done():
	default:
		t.Error("Done is not closed after Close returned")
	}
	if err := b.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// TestBufferedEvictionDropsTheOldest exercises the ring directly: producing
// 8000+ events through a real run would be slow and would tell us nothing the
// buffer's own bookkeeping does not.
func TestBufferedEvictionDropsTheOldest(t *testing.T) {
	b := &Buffered{
		changed: make(chan struct{}),
		done:    make(chan struct{}),
	}
	const over = 10
	for i := range bufferCap + over {
		b.append(TextDelta{Text: strconv.Itoa(i)})
	}

	// Len is the position of the tail, not the size of the buffer.
	if got, want := b.Len(), bufferCap+over; got != want {
		t.Errorf("Len() = %d, want %d", got, want)
	}
	b.mu.Lock()
	base, held := b.base, len(b.events)
	b.mu.Unlock()
	if base != over {
		t.Errorf("base = %d, want %d", base, over)
	}
	if held != bufferCap {
		t.Errorf("retained %d events, want the cap %d", held, bufferCap)
	}

	// The oldest went, not the newest: dropping the newest would stall a live
	// follower behind a permanent gap.
	b.mu.Lock()
	b.finished = true
	b.mu.Unlock()

	var firstSeq = -1
	var firstText string
	for seq, ev := range b.Follow(context.Background(), 0) {
		firstSeq = seq
		firstText = ev.(TextDelta).Text
		break
	}
	if firstSeq != over {
		t.Errorf("a follower asking for 0 was clamped to %d, want %d (the oldest retained)", firstSeq, over)
	}
	if firstText != strconv.Itoa(over) {
		t.Errorf("first retained event = %q, want %q", firstText, strconv.Itoa(over))
	}

	// And the newest is still there.
	last := ""
	for _, ev := range b.Follow(context.Background(), b.Len()-1) {
		last = ev.(TextDelta).Text
	}
	if want := strconv.Itoa(bufferCap + over - 1); last != want {
		t.Errorf("last event = %q, want %q", last, want)
	}
}

func TestBufferedFollowClampsNegativeFrom(t *testing.T) {
	a := newAgent(t, countingClient(3), nil)
	b := Buffer(a.Start(context.Background(), Ask("go")))
	t.Cleanup(func() { _ = b.Close() })
	<-b.Done()

	first := -1
	for seq := range b.Follow(context.Background(), -5) {
		first = seq
		break
	}
	if first != 0 {
		t.Errorf("Follow(-5) started at %d, want 0", first)
	}
}

// Several followers at different positions must not interfere.
func TestBufferedFollowersAreIndependent(t *testing.T) {
	a := newAgent(t, countingClient(10), nil)
	b := Buffer(a.Start(context.Background(), Ask("go")))
	t.Cleanup(func() { _ = b.Close() })
	<-b.Done()

	total := b.Len()
	counts := make(chan int, 3)
	for _, from := range []int{0, 2, total - 1} {
		go func() {
			n := 0
			for range b.Follow(context.Background(), from) {
				n++
			}
			counts <- n
		}()
	}

	got := map[int]bool{}
	for range 3 {
		select {
		case n := <-counts:
			got[n] = true
		case <-time.After(10 * time.Second):
			t.Fatal("a follower never returned")
		}
	}
	for _, want := range []int{total, total - 2, 1} {
		if !got[want] {
			t.Errorf("no follower reported %d events; got %v", want, got)
		}
	}
}
