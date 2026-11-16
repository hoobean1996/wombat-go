package daemon

import (
	"context"
	"slices"
	"sync"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/permission"
)

// byPermission builds a session's agent the way a server would: under
// PermissionFull nothing is asked, and under anything else every call waits
// for a decision. It records the permissions it was asked to build for.
type byPermission struct {
	cl   llm.Client
	echo *countingTool

	mu    sync.Mutex
	asked []string
}

func (b *byPermission) build(o SessionOptions) (*core.Agent, error) {
	b.mu.Lock()
	b.asked = append(b.asked, o.Permission)
	b.mu.Unlock()
	opts := []core.Option{
		core.WithClient(b.cl), core.WithModel("test-model"), core.WithLogger(quietLogger),
		core.WithTools(b.echo.def("echo")),
	}
	if o.Permission != PermissionFull {
		opts = append(opts, core.WithToolMiddleware(permission.Gate(permission.AskEverything(), permission.Defer())))
	}
	return core.New(opts...)
}

// Full access is a session's to switch to while it waits: the calls it waits
// on are allowed with it, the turn resumes as if someone had said yes, and the
// agent is built again under the new policy, which the store keeps.
func TestFullAccessAllowsWhatIsWaiting(t *testing.T) {
	b := &byPermission{cl: script(toolTurn("u1", "echo", `{"x":1}`), textTurn("all done")), echo: &countingTool{}}
	m := newManager(t, ManagerConfig{Build: b.build})
	s, err := m.Create(testCtx(t), SessionOptions{Title: "t"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	msg, err := s.Send(testCtx(t), "run the tool")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if stopped := waitEvent(t, s, msg.ID); stopped.State != Approving {
		t.Fatalf("turn 1 = %+v, want it stopped on the call", stopped)
	}
	// Like the reply it is on the way to, the request for a decision says
	// which messages it is part of answering.
	if asked := outboxOf(t, m, s.ID(), MessageApproval); !slices.Equal(asked.InReplyTo, []int64{msg.ID}) {
		t.Fatalf("the approval is in reply to %v, want [%d]", asked.InReplyTo, msg.ID)
	}

	if err := s.SetPermission(testCtx(t), PermissionFull); err != nil {
		t.Fatalf("SetPermission: %v", err)
	}
	if got := s.Info().Options.Permission; got != PermissionFull {
		t.Fatalf("permission = %q, want full", got)
	}
	if reply := outboxOf(t, m, s.ID(), MessageReply); reply.Text != "all done" {
		t.Fatalf("the reply = %+v, want the resumed turn's answer", reply)
	}
	until(t, "the session to be idle", func() bool { return s.Info().State == Idle })
	if b.echo.calls() != 1 {
		t.Fatalf("the allowed call ran %d times, want 1", b.echo.calls())
	}
	if len(s.Pending()) != 0 {
		t.Fatalf("still pending: %+v", s.Pending())
	}
	rec, err := m.store.LoadSession(testCtx(t), s.ID())
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if rec.Options.Permission != PermissionFull {
		t.Fatalf("the store kept permission %q, want full", rec.Options.Permission)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if !slices.Equal(b.asked, []string{"", PermissionFull}) {
		t.Fatalf("built for %q, want the default and then full", b.asked)
	}
}

// heldClient holds its first call until released, then answers from a script.
type heldClient struct {
	*scripted
	entered, release chan struct{}
	once             sync.Once
}

func (h *heldClient) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	first := false
	h.once.Do(func() { first = true })
	if first {
		h.entered <- struct{}{}
		select {
		case <-h.release:
		case <-ctx.Done():
			return llm.Response{}, ctx.Err()
		}
	}
	return h.scripted.Complete(ctx, req)
}

// A switch made while a turn runs leaves that turn under the policy it began
// with, so it may still stop on a call. Under full access nobody is asked, so
// the call is allowed as soon as it does, and the turn's own record — written
// before the switch — does not undo it.
func TestFullAccessDuringATurn(t *testing.T) {
	cl := &heldClient{
		scripted: script(toolTurn("u1", "echo", `{"x":1}`), textTurn("all done")),
		entered:  make(chan struct{}, 1), release: make(chan struct{}),
	}
	b := &byPermission{cl: cl, echo: &countingTool{}}
	m := newManager(t, ManagerConfig{Build: b.build})
	s, err := m.Create(testCtx(t), SessionOptions{Title: "t"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := s.Send(testCtx(t), "run the tool"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	<-cl.entered
	if err := s.SetPermission(testCtx(t), PermissionFull); err != nil {
		t.Fatalf("SetPermission: %v", err)
	}
	close(cl.release)

	if reply := outboxOf(t, m, s.ID(), MessageReply); reply.Text != "all done" {
		t.Fatalf("the reply = %+v, want the answer once the call was allowed", reply)
	}
	until(t, "the session to be idle", func() bool { return s.Info().State == Idle })
	if b.echo.calls() != 1 {
		t.Fatalf("the call ran %d times, want 1", b.echo.calls())
	}
	rec, err := m.store.LoadSession(testCtx(t), s.ID())
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if rec.Options.Permission != PermissionFull {
		t.Fatalf("the store kept permission %q, want full", rec.Options.Permission)
	}
}

// Going back is as plain: the next turn asks again, and the store says so.
func TestPermissionBackToDefault(t *testing.T) {
	b := &byPermission{cl: script(textTurn("hi")), echo: &countingTool{}}
	m := newManager(t, ManagerConfig{Build: b.build})
	s, err := m.Create(testCtx(t), SessionOptions{Permission: PermissionFull})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := s.SetPermission(testCtx(t), ""); err != nil {
		t.Fatalf("SetPermission: %v", err)
	}
	rec, err := m.store.LoadSession(testCtx(t), s.ID())
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if rec.Options.Permission != "" || s.Info().Options.Permission != "" {
		t.Fatalf("permission = %q in the store, %q in memory, want the default", rec.Options.Permission, s.Info().Options.Permission)
	}
	if err := m.Delete(testCtx(t), s.ID()); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if err := s.SetPermission(testCtx(t), PermissionFull); err == nil {
		t.Fatal("SetPermission on a deleted session succeeded")
	}
}
