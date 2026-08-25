package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/permission"
)

// errNoQueue reports a gated run with no session behind it. Always a wiring
// mistake — an agent built by [ManagerConfig.Build] and started by anything
// other than [Session.Send] — so it is not exported: no client can act on it.
var errNoQueue = errors.New("httpapi: this run has no approval queue attached")

// approvals is one session's set of tool calls parked on a human.
//
// Per session, because a tool_use id is only unique within a run and because
// the session id in the URL is the only thing standing between one browser and
// another's questions.
//
// Every method is safe to call from any goroutine. They have to be: the waiter
// runs on the agent's dispatch goroutine and the answer arrives on an HTTP
// handler's.
type approvals struct {
	mu       sync.Mutex
	waiting  map[llm.ToolUseID]*parked
	answered map[llm.ToolUseID]bool
	closed   bool
}

// parked is one question and the channel its answer arrives on.
type parked struct {
	// ch is buffered so resolve never blocks on a waiter that has already
	// given up on its context: the HTTP handler answering must not be able to
	// hang.
	ch chan bool
	ap Approval
}

func newApprovals() *approvals {
	return &approvals{
		waiting:  map[llm.ToolUseID]*parked{},
		answered: map[llm.ToolUseID]bool{},
	}
}

// register parks a request and returns the channel its verdict arrives on.
func (a *approvals) register(r permission.Request) (*parked, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, errNoQueue
	}
	if _, dup := a.waiting[r.Use.ID]; dup {
		// A tool_use id is unique within a run, so this cannot happen without
		// the gate being invoked twice for one call — which would mean an
		// answer could satisfy the wrong waiter. Refuse instead.
		return nil, ErrAlreadyAnswered
	}
	p := &parked{
		ch: make(chan bool, 1),
		ap: Approval{
			UseID:  string(r.Use.ID),
			Tool:   r.Tool.Name,
			Reason: r.Reason,
			Input:  jsonOrNil(r.Use.Input),
			Since:  time.Now(),
		},
	}
	a.waiting[r.Use.ID] = p
	return p, nil
}

// forget drops a wait that has ended, answered or abandoned.
//
// It checks identity rather than just the key: register refuses a duplicate,
// but a future caller re-parking the same id must not have its channel deleted
// by the previous waiter's deferred cleanup.
func (a *approvals) forget(id llm.ToolUseID, p *parked) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.waiting[id] == p {
		delete(a.waiting, id)
	}
}

// resolve delivers a verdict.
func (a *approvals) resolve(id llm.ToolUseID, allow bool) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	p, ok := a.waiting[id]
	if !ok {
		if a.answered[id] {
			return ErrAlreadyAnswered
		}
		return ErrNoSuchApproval
	}
	delete(a.waiting, id)
	a.answered[id] = true
	p.ch <- allow // buffered; the waiter may already be gone
	return nil
}

// pending lists what is waiting, oldest first — which is the order a person
// should answer in, and the order a list rendered from it stays stable in.
func (a *approvals) pending() []Approval {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]Approval, 0, len(a.waiting))
	for _, p := range a.waiting {
		out = append(out, p.ap)
	}
	slices.SortFunc(out, func(x, y Approval) int { return x.Since.Compare(y.Since) })
	return out
}

// count reports how many calls are parked. Its own lock, because the caller
// holding a session's mutex must not reach into this one's fields directly.
func (a *approvals) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.waiting)
}

// deny releases every parked call, refusing it, and leaves the queue open.
//
// Cancelling the turn's context would unblock the waiters on its own, so this
// is not what makes a cancelled turn stop; it is what makes the answer
// deterministic — denied — instead of a race between the cancel and a verdict
// arriving in the same instant.
func (a *approvals) deny() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.release()
}

// close denies everything parked and refuses anything new. For a session that
// is going away, not for a turn that is being cancelled.
func (a *approvals) close() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.closed = true
	a.release()
}

// release denies every waiter. Callers hold a.mu.
func (a *approvals) release() {
	for id, p := range a.waiting {
		delete(a.waiting, id)
		a.answered[id] = true
		p.ch <- false
	}
}

type approvalsKey struct{}

// withApprovals attaches a session's approval queue to its turn's context.
//
// It travels on the context rather than in a closure because the agent —
// including the permission gate inside its middleware chain — is built once
// per session by [ManagerConfig.Build], which never sees the [Session]. Same
// rule as [wombat.WithEmitter].
func withApprovals(ctx context.Context, a *approvals) context.Context {
	return context.WithValue(ctx, approvalsKey{}, a)
}

// Approver returns the [permission.Approver] that parks a gated tool call on
// the session running it and blocks until a human answers over
// POST /api/sessions/{id}/approvals/{use_id}.
//
// Install it in the gate that [ManagerConfig.Build] puts on the agent:
//
//	permission.Gate(permission.Workspace(root), httpapi.Approver())
//
// Blocking is the whole design, and it is [permission.Gate]'s contract rather
// than this package's invention: the alternative — suspending the loop and
// answering the tool_use with the verdict — cannot work, because then the tool
// never runs and the model reads "approved" as the tool's output. Staying
// parked means an approved call goes on to execute and its real result enters
// the transcript, which never mentions the approval at all.
//
// What unblocks a wait other than an answer is cancellation: the per-turn
// budget, [Session.Cancel], [Manager.Delete] or the sweeper. There is no
// timeout, because a question the user is still reading is not a question that
// should expire.
func Approver() permission.Approver { return permission.ApproverFunc(approve) }

func approve(ctx context.Context, r permission.Request) (bool, error) {
	a, _ := ctx.Value(approvalsKey{}).(*approvals)
	if a == nil {
		return false, errNoQueue
	}

	p, err := a.register(r)
	if err != nil {
		return false, err
	}
	defer a.forget(r.Use.ID, p)

	select {
	case allow := <-p.ch:
		return allow, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

// jsonOrNil keeps an empty argument list out of the payload: an empty non-nil
// json.RawMessage is not valid JSON and would fail the encode for the whole
// response, taking the prompt with it.
func jsonOrNil(in json.RawMessage) json.RawMessage {
	if len(in) == 0 {
		return nil
	}
	return in
}
