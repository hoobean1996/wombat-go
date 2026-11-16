package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Several daemons sharing one store: each session is run by one of them, an
// event put in a session's inbox through any of them is run by the one that
// holds it, and a daemon that dies has its sessions taken over.

func daemonOn(t *testing.T, st Store, name string, cfg ManagerConfig) *Manager {
	t.Helper()
	cfg.Store, cfg.Name, cfg.Addr = st, name, "http://"+name
	return newManager(t, cfg)
}

func say(t *testing.T, m *Manager, session, text string) Event {
	t.Helper()
	payload, _ := json.Marshal(MessagePayload{Text: text})
	e, _, err := m.Receive(testCtx(t), Inbound{Session: session, Kind: KindMessage, Payload: payload})
	if err != nil {
		t.Fatalf("Receive through %s: %v", m.Name(), err)
	}
	return e
}

func replies(t *testing.T, m *Manager, session string) []string {
	t.Helper()
	msgs, err := m.Outbox(testCtx(t), OutboxFilter{Session: session}, 0)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, msg := range msgs {
		if msg.Kind == MessageReply {
			out = append(out, msg.Text)
		}
	}
	return out
}

func TestDaemonsShareAStore(t *testing.T) {
	st := MemoryStore()
	a := daemonOn(t, st, "a", ManagerConfig{Build: builder(script(textTurn("from a")))})
	b := daemonOn(t, st, "b", ManagerConfig{Build: builder(script(textTurn("from b")))})

	sa, err := a.Create(testCtx(t), SessionOptions{Title: "a's"})
	if err != nil {
		t.Fatal(err)
	}
	sb, err := b.Create(testCtx(t), SessionOptions{Title: "b's"})
	if err != nil {
		t.Fatal(err)
	}

	// Each holds its own; asking the other for it says who has it, and where.
	_, err = b.Get(testCtx(t), sa.ID())
	var owned *OwnedError
	if !errors.As(err, &owned) || owned.Lease.Owner != "a" || owned.Lease.Addr != "http://a" {
		t.Fatalf("b.Get(a's session) = %v, want an OwnedError naming a", err)
	}

	// An event through either daemon is run by the one holding the session.
	ea := say(t, b, sa.ID(), "hello a")
	eb := say(t, a, sb.ID(), "hello b")
	if got := waitEvent(t, sa, ea.ID); got.Answer != "from a" {
		t.Fatalf("a's session ran %+v", got)
	}
	if got := waitEvent(t, sb, eb.ID); got.Answer != "from b" {
		t.Fatalf("b's session ran %+v", got)
	}

	// Either lists both.
	for _, m := range []*Manager{a, b} {
		list, err := m.List(testCtx(t))
		if err != nil || len(list) != 2 {
			t.Fatalf("%s lists %d sessions (%v), want 2", m.Name(), len(list), err)
		}
	}
}

// A daemon that dies mid-turn stops renewing its leases. Once one runs out,
// another takes the session over, puts the dead turn's events back and runs
// them. Should the first come back to life and finish its turn, its
// settlement is refused: the session is answered once, by the daemon that
// holds it.
func TestADeadDaemonsSessionIsTakenOver(t *testing.T) {
	st := MemoryStore()
	held := &heldClient{scripted: script(textTurn("from a")), entered: make(chan struct{}, 1), release: make(chan struct{})}
	const lease = 300 * time.Millisecond
	a := daemonOn(t, st, "a", ManagerConfig{Build: builder(held), Lease: lease})
	b := daemonOn(t, st, "b", ManagerConfig{Build: builder(script(textTurn("from b"))), Lease: lease})

	s, err := a.Create(testCtx(t), SessionOptions{Title: "t"})
	if err != nil {
		t.Fatal(err)
	}
	e := say(t, a, s.ID(), "go")
	<-held.entered

	// a "dies": its loops stop, the renewer among them; its turn hangs on.
	a.cancel()

	until(t, "b to answer the session a was running", func() bool {
		return strings.Join(replies(t, b, s.ID()), ",") == "from b"
	})
	rec, err := st.LoadSession(testCtx(t), s.ID())
	if err != nil || rec.Lease.Owner != "b" {
		t.Fatalf("the session is held by %+v (%v), want b", rec.Lease, err)
	}

	// a comes back, and finishes: refused, so nothing of it lands.
	close(held.release)
	until(t, "a's turn to end", func() bool {
		a.mu.Lock()
		defer a.mu.Unlock()
		_, still := a.sessions[s.ID()]
		return !still
	})
	if got := replies(t, b, s.ID()); len(got) != 1 || got[0] != "from b" {
		t.Fatalf("replies = %q, want only b's", got)
	}
	rec, _ = st.LoadSession(testCtx(t), s.ID())
	if text := transcriptOf(rec); strings.Contains(text, "from a") || !strings.Contains(text, "from b") {
		t.Fatalf("the transcript has a's turn in it, or not b's: %s", text)
	}
	if got := eventStatus(t, st, e.ID); got.Status != EventDone || got.Attempts != 2 {
		t.Fatalf("the event = %+v, want done, on its second attempt", got)
	}
}

// A daemon that stops cleanly lets go of its sessions, and the next event for
// one is run by whichever daemon takes it.
func TestAStoppedDaemonsSessionsMoveOn(t *testing.T) {
	st := MemoryStore()
	a := daemonOn(t, st, "a", ManagerConfig{Build: builder(script())})
	b := daemonOn(t, st, "b", ManagerConfig{Build: builder(script(textTurn("from b")))})
	s, err := a.Create(testCtx(t), SessionOptions{Title: "t"})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if rec, _ := st.LoadSession(testCtx(t), s.ID()); rec.Lease.Owner != "" {
		t.Fatalf("a stopped and still holds the session: %+v", rec.Lease)
	}
	say(t, b, s.ID(), "anyone there?")
	until(t, "b to answer", func() bool { return strings.Join(replies(t, b, s.ID()), ",") == "from b" })
}

// Only one daemon sends the outbox, however many share the store.
func TestOneDispatcher(t *testing.T) {
	st := MemoryStore()
	var sent atomic.Int32
	deliver := func(_ context.Context, _ Message) error { sent.Add(1); return nil }
	a := daemonOn(t, st, "a", ManagerConfig{Build: builder(script(textTurn("hi"))), Deliver: deliver})
	daemonOn(t, st, "b", ManagerConfig{Build: builder(script()), Deliver: deliver})
	s, err := a.Create(testCtx(t), SessionOptions{Title: "t"})
	if err != nil {
		t.Fatal(err)
	}
	say(t, a, s.ID(), "hi")
	until(t, "the reply to be delivered", func() bool {
		msgs, _ := st.Outbox(testCtx(t), OutboxFilter{Session: s.ID(), Status: MessageDelivered})
		return len(msgs) == 1
	})
	time.Sleep(50 * time.Millisecond) // time enough for a second dispatcher to send it again
	if n := sent.Load(); n != 1 {
		t.Fatalf("the reply was sent %d times, want once", n)
	}
}

func transcriptOf(rec SessionRecord) string {
	b, _ := json.Marshal(rec.Messages)
	return string(b)
}

func eventStatus(t *testing.T, st Store, id int64) Event {
	t.Helper()
	all, err := st.Events(testCtx(t), EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range all {
		if e.ID == id {
			return e
		}
	}
	t.Fatalf("no event %d", id)
	return Event{}
}

// A message for a session whose daemon has died is taken all the same, and
// answered by the daemon that takes the session over once the lease runs out.
func TestAMessageWaitsOutADeadDaemon(t *testing.T) {
	st := MemoryStore()
	const lease = 300 * time.Millisecond
	a := daemonOn(t, st, "a", ManagerConfig{Build: builder(script()), Lease: lease})
	b := daemonOn(t, st, "b", ManagerConfig{Build: builder(script(textTurn("from b"))), Lease: lease})
	s, err := a.Create(testCtx(t), SessionOptions{Title: "t"})
	if err != nil {
		t.Fatal(err)
	}
	a.cancel() // a dies holding the session

	if _, err := b.SendAs(testCtx(t), s.ID(), "", "still there?"); err != nil {
		t.Fatalf("SendAs through b while a's lease runs = %v, want it taken", err)
	}
	until(t, "b to answer once a's lease ran out", func() bool {
		return strings.Join(replies(t, b, s.ID()), ",") == "from b"
	})
}
