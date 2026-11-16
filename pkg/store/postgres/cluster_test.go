package postgres_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/daemon"
	"wombat/pkg/store/postgres"
)

// Two daemons on one schema: each runs the sessions it holds, an event put
// in through either is run by the holder, and a daemon that stops cleanly
// hands its sessions on.
func TestTwoDaemonsShareASchema(t *testing.T) {
	ctx := context.Background()
	where := schema(t)
	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	start := func(name string, answer string) *daemon.Manager {
		st := open(t, where)
		if err := st.Lock(ctx, name); err != nil {
			t.Fatal(err)
		}
		build := func(daemon.SessionOptions) (*core.Agent, error) {
			return core.New(core.WithClient(&replies{next: nil, always: answer}), core.WithModel("m"), core.WithLogger(quiet))
		}
		m, err := daemon.NewManager(daemon.ManagerConfig{Store: st, Build: build, Name: name, Addr: "http://" + name,
			Poll: 5 * time.Millisecond, Logger: quiet})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = m.Close() })
		return m
	}
	a, b := start("a", "from a"), start("b", "from b")

	s, err := a.Create(ctx, daemon.SessionOptions{Title: "a's"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = b.Get(ctx, s.ID())
	var owned *daemon.OwnedError
	if !errors.As(err, &owned) || owned.Lease.Owner != "a" || owned.Lease.Addr != "http://a" {
		t.Fatalf("b.Get(a's session) = %v, want an OwnedError naming a", err)
	}

	payload, _ := json.Marshal(daemon.MessagePayload{Text: "hi"})
	e, _, err := b.Receive(ctx, daemon.Inbound{Session: s.ID(), Kind: daemon.KindMessage, Payload: payload})
	if err != nil {
		t.Fatal(err)
	}
	if got := waitFor(t, s, e.ID); got.Answer != "from a" {
		t.Fatalf("a's session ran %+v", got)
	}

	// a stops cleanly, and the next message is b's to answer.
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := b.Receive(ctx, daemon.Inbound{Session: s.ID(), Kind: daemon.KindMessage, Payload: payload}); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		msgs, err := b.Outbox(ctx, daemon.OutboxFilter{Session: s.ID()}, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(msgs) == 2 && msgs[1].Text == "from b" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("outbox = %+v, want a's reply and then b's", msgs)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := postgres.Open(ctx, where); err != nil {
		t.Fatal(err)
	}
}

// waitFor waits for the turn that handles event id to end.
func waitFor(t *testing.T, s *daemon.Session, id int64) daemon.TurnEnded {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for _, f := range s.Follow(ctx, 0) {
		if te, ok := f.Event.(daemon.TurnEnded); ok && len(te.Events) > 0 && te.Events[0] == id {
			return te
		}
	}
	t.Fatalf("the turn for event %d never ended", id)
	return daemon.TurnEnded{}
}
