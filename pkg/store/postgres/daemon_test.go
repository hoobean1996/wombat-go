package postgres_test

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/tool"
	"wombat/pkg/daemon"
	"wombat/pkg/store/postgres"
)

// replies answers each model call with the next response, across daemons,
// and then with always, if set.
type replies struct {
	mu     sync.Mutex
	next   []llm.Response
	always string
}

func (r *replies) Complete(context.Context, llm.Request) (llm.Response, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.next) == 0 {
		text := r.always
		if text == "" {
			text = "out of replies"
		}
		return say(text), nil
	}
	out := r.next[0]
	r.next = r.next[1:]
	return out, nil
}

func say(text string) llm.Response {
	return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: text}}, StopReason: llm.StopEndTurn}
}

func ended(t *testing.T, s *daemon.Session, turn int) daemon.TurnEnded {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	for _, f := range s.Follow(ctx, 0) {
		if e, ok := f.Event.(daemon.TurnEnded); ok && e.Turn == turn {
			return e
		}
	}
	t.Fatalf("turn %d never ended (state %s)", turn, s.Info().State)
	return daemon.TurnEnded{}
}

// The daemon on Postgres, through a restart: a message whose turn stops on a
// call that needs a decision; the daemon goes away; the next one finds the
// session still waiting, takes the decision, and carries the turn on; and a
// timer wakes the session after.
func TestADaemonOnPostgresCarriesOnAfterARestart(t *testing.T) {
	ctx := context.Background()
	where := schema(t)
	var writes atomic.Int64
	write := tool.Def{Name: "write", Description: "writes", InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn: func(context.Context, json.RawMessage) (string, error) { writes.Add(1); return "written", nil }}
	cl := &replies{next: []llm.Response{
		{Content: []llm.ContentBlock{llm.ToolUse{ID: "w1", Name: "write", Input: json.RawMessage(`{"path":"b"}`)}}, StopReason: llm.StopToolUse},
		say("wrote it"),
		say("woke up"),
	}}
	build := func(daemon.SessionOptions) (*core.Agent, error) {
		return core.New(core.WithClient(cl), core.WithModel("m"), core.WithTools(write),
			core.WithToolMiddleware(permission.Gate(permission.AskEverything(), permission.Defer())),
			core.WithLogger(slog.New(slog.NewTextHandler(io.Discard, nil))))
	}
	start := func() (*postgres.Store, *daemon.Manager) {
		st := open(t, where)
		if err := st.Lock(ctx, "d1"); err != nil {
			t.Fatal(err)
		}
		// The same name each time, so the daemon that starts again takes back
		// at once what the one before it held.
		m, err := daemon.NewManager(daemon.ManagerConfig{Store: st, Build: build, Name: "d1", Poll: 5 * time.Millisecond,
			Logger: slog.New(slog.NewTextHandler(io.Discard, nil))})
		if err != nil {
			t.Fatal(err)
		}
		return st, m
	}

	st, m := start()
	s, err := m.Create(ctx, daemon.SessionOptions{Title: "on postgres"})
	if err != nil {
		t.Fatal(err)
	}
	msg, err := s.Send(ctx, "write b")
	if err != nil {
		t.Fatal(err)
	}
	if stopped := ended(t, s, 1); stopped.Outcome != "deferred" {
		t.Fatalf("turn 1 = %+v, want it stopped on the write", stopped)
	}
	m.Close()
	st.Close()

	st, m = start()
	defer st.Close()
	defer m.Close()
	s, err = m.Get(ctx, s.ID())
	if err != nil {
		t.Fatal(err)
	}
	if info := s.Info(); info.State != daemon.Approving || len(s.Pending()) != 1 {
		t.Fatalf("after the restart: %+v, pending %+v; want it still waiting on the write", info, s.Pending())
	}
	if _, err := s.Resolve(ctx, "w1", true); err != nil {
		t.Fatal(err)
	}
	if done := ended(t, s, 2); done.Answer != "wrote it" {
		t.Fatalf("turn 2 = %+v", done)
	}
	if n := writes.Load(); n != 1 {
		t.Fatalf("the write ran %d times, want once", n)
	}

	if _, err := m.AddTimer(ctx, daemon.Timer{Session: s.ID(), Note: "look again", Due: time.Now().Add(30 * time.Millisecond)}); err != nil {
		t.Fatal(err)
	}
	if woke := ended(t, s, 3); woke.Answer != "woke up" {
		t.Fatalf("the timer's turn = %+v", woke)
	}

	msgs, err := m.Outbox(ctx, daemon.OutboxFilter{Session: s.ID()}, 0)
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, m := range msgs {
		kinds = append(kinds, m.Kind)
	}
	if strings.Join(kinds, ",") != "approval,reply" || msgs[1].Text != "wrote it" || msgs[1].InReplyTo[0] != msg.ID {
		t.Fatalf("outbox = %+v, want the approval and the reply to the message; the timer's turn said nothing", msgs)
	}
	events, _ := m.Inbox(ctx, daemon.EventFilter{Session: s.ID()})
	for _, e := range events {
		if e.Status != daemon.EventDone {
			t.Errorf("event %d (%s) is %s, want done", e.ID, e.Kind, e.Status)
		}
	}
	if len(events) != 3 {
		t.Errorf("%d events, want the message, the decision and the timer", len(events))
	}
}
