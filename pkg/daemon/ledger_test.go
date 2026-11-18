package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/tool"
)

// ===== Fixtures =====

// stepModel is a model that answers each call with the next of its steps — a
// response, or an error — and records what it was asked.
type stepModel struct {
	mu    sync.Mutex
	steps []func() (llm.Response, error)
	seen  []llm.Request
}

func (r *stepModel) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seen = append(r.seen, req)
	if len(r.steps) == 0 {
		return llm.Response{}, errors.New("stepModel: out of steps")
	}
	step := r.steps[0]
	r.steps = r.steps[1:]
	return step()
}

func (r *stepModel) asked() []llm.Request {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.seen)
}

func reply(resp llm.Response) func() (llm.Response, error) {
	return func() (llm.Response, error) { return resp, nil }
}

// overloaded is the provider falling over: a 529, which a turn is tried
// again for.
func overloaded() (llm.Response, error) {
	return llm.Response{}, &llm.APIError{Class: llm.ErrOverloaded, Message: "overloaded"}
}

// errCrashed is every store call of a daemon whose process has died.
var errCrashed = errors.New("the process died")

// crashStore is a store a daemon dies on: after crash, every call fails, as
// it would for a process that is gone — nothing it had not written by then
// is written. after, when set, crashes it once the named call has been made.
type crashStore struct {
	Store
	down  atomic.Bool
	after string
}

func (c *crashStore) crash() { c.down.Store(true) }

func (c *crashStore) dead(call string) error {
	if c.down.Load() {
		return errCrashed
	}
	return nil
}

// made is called after a call went through, to crash after the one named.
func (c *crashStore) made(call string) {
	if c.after == call {
		c.crash()
	}
}

func (c *crashStore) CreateSession(ctx context.Context, rec SessionRecord) error {
	if err := c.dead("CreateSession"); err != nil {
		return err
	}
	return c.Store.CreateSession(ctx, rec)
}
func (c *crashStore) SaveSession(ctx context.Context, rec SessionRecord) error {
	if err := c.dead("SaveSession"); err != nil {
		return err
	}
	return c.Store.SaveSession(ctx, rec)
}
func (c *crashStore) LoadSession(ctx context.Context, id string) (SessionRecord, error) {
	if err := c.dead("LoadSession"); err != nil {
		return SessionRecord{}, err
	}
	return c.Store.LoadSession(ctx, id)
}
func (c *crashStore) SessionByThread(ctx context.Context, thread string) (string, error) {
	if err := c.dead("SessionByThread"); err != nil {
		return "", err
	}
	return c.Store.SessionByThread(ctx, thread)
}
func (c *crashStore) ListSessions(ctx context.Context) ([]SessionRecord, error) {
	if err := c.dead("ListSessions"); err != nil {
		return nil, err
	}
	return c.Store.ListSessions(ctx)
}
func (c *crashStore) DeleteSession(ctx context.Context, id string) error {
	if err := c.dead("DeleteSession"); err != nil {
		return err
	}
	return c.Store.DeleteSession(ctx, id)
}
func (c *crashStore) Enqueue(ctx context.Context, e Event) (Event, bool, error) {
	if err := c.dead("Enqueue"); err != nil {
		return Event{}, false, err
	}
	return c.Store.Enqueue(ctx, e)
}
func (c *crashStore) Ready(ctx context.Context, now time.Time, owner string, busy []string, limit int) ([]string, error) {
	if err := c.dead("Ready"); err != nil {
		return nil, err
	}
	return c.Store.Ready(ctx, now, owner, busy, limit)
}
func (c *crashStore) Claim(ctx context.Context, session, owner string, now time.Time, limit int) ([]Event, error) {
	if err := c.dead("Claim"); err != nil {
		return nil, err
	}
	return c.Store.Claim(ctx, session, owner, now, limit)
}
func (c *crashStore) Settle(ctx context.Context, s Settlement) error {
	if err := c.dead("Settle"); err != nil {
		return err
	}
	return c.Store.Settle(ctx, s)
}
func (c *crashStore) Acquire(ctx context.Context, session string, l Lease, now time.Time, attempts int) (int, error) {
	if err := c.dead("Acquire"); err != nil {
		return 0, err
	}
	return c.Store.Acquire(ctx, session, l, now, attempts)
}
func (c *crashStore) Renew(ctx context.Context, owner string, sessions []string, until time.Time) ([]string, error) {
	if err := c.dead("Renew"); err != nil {
		return nil, err
	}
	return c.Store.Renew(ctx, owner, sessions, until)
}
func (c *crashStore) Release(ctx context.Context, owner string, sessions []string) error {
	if err := c.dead("Release"); err != nil {
		return err
	}
	return c.Store.Release(ctx, owner, sessions)
}
func (c *crashStore) Recover(ctx context.Context, owner string, attempts int) (int, error) {
	if err := c.dead("Recover"); err != nil {
		return 0, err
	}
	return c.Store.Recover(ctx, owner, attempts)
}
func (c *crashStore) Lead(ctx context.Context, role, owner string, until time.Time) (bool, error) {
	if err := c.dead("Lead"); err != nil {
		return false, err
	}
	return c.Store.Lead(ctx, role, owner, until)
}
func (c *crashStore) Events(ctx context.Context, f EventFilter) ([]Event, error) {
	if err := c.dead("Events"); err != nil {
		return nil, err
	}
	return c.Store.Events(ctx, f)
}
func (c *crashStore) AddTimer(ctx context.Context, t Timer) (Timer, error) {
	if err := c.dead("AddTimer"); err != nil {
		return Timer{}, err
	}
	return c.Store.AddTimer(ctx, t)
}
func (c *crashStore) CancelTimer(ctx context.Context, id int64, session string) error {
	if err := c.dead("CancelTimer"); err != nil {
		return err
	}
	return c.Store.CancelTimer(ctx, id, session)
}
func (c *crashStore) Timers(ctx context.Context, f TimerFilter) ([]Timer, error) {
	if err := c.dead("Timers"); err != nil {
		return nil, err
	}
	return c.Store.Timers(ctx, f)
}
func (c *crashStore) Fire(ctx context.Context, now time.Time) ([]Event, error) {
	if err := c.dead("Fire"); err != nil {
		return nil, err
	}
	return c.Store.Fire(ctx, now)
}
func (c *crashStore) Post(ctx context.Context, m Message) (Message, bool, error) {
	if err := c.dead("Post"); err != nil {
		return Message{}, false, err
	}
	defer c.made("Post")
	return c.Store.Post(ctx, m)
}
func (c *crashStore) Outbox(ctx context.Context, f OutboxFilter) ([]Message, error) {
	if err := c.dead("Outbox"); err != nil {
		return nil, err
	}
	return c.Store.Outbox(ctx, f)
}
func (c *crashStore) Delivered(ctx context.Context, id int64, at time.Time) error {
	if err := c.dead("Delivered"); err != nil {
		return err
	}
	return c.Store.Delivered(ctx, id, at)
}
func (c *crashStore) Undelivered(ctx context.Context, id int64, reason string, retry time.Time) error {
	if err := c.dead("Undelivered"); err != nil {
		return err
	}
	return c.Store.Undelivered(ctx, id, reason, retry)
}
func (c *crashStore) OpenStep(ctx context.Context, o StepOpen) (int, error) {
	if err := c.dead("OpenStep"); err != nil {
		return 0, err
	}
	return c.Store.OpenStep(ctx, o)
}
func (c *crashStore) MoveCalls(ctx context.Context, m CallMoves) error {
	if err := c.dead("MoveCalls"); err != nil {
		return err
	}
	return c.Store.MoveCalls(ctx, m)
}
func (c *crashStore) ToolCalls(ctx context.Context, f CallFilter) ([]ToolCall, error) {
	if err := c.dead("ToolCalls"); err != nil {
		return nil, err
	}
	return c.Store.ToolCalls(ctx, f)
}
func (c *crashStore) SubRun(ctx context.Context, session, parent string, n int) (SubRun, bool, error) {
	if err := c.dead("SubRun"); err != nil {
		return SubRun{}, false, err
	}
	return c.Store.SubRun(ctx, session, parent, n)
}
func (c *crashStore) PruneCalls(ctx context.Context, before time.Time) (int, error) {
	if err := c.dead("PruneCalls"); err != nil {
		return 0, err
	}
	return c.Store.PruneCalls(ctx, before)
}

// effect is a tool with an effect that is not safe to have twice — a write,
// a payment — that counts how many times it took effect. With hold set, it
// takes effect, says so on took, and then does not return until hold is
// closed: what a process dying mid-call looks like from outside.
type effect struct {
	n    atomic.Int64
	took chan struct{}
	hold chan struct{}
}

func newEffect() *effect { return &effect{took: make(chan struct{}, 16)} }

func (e *effect) def(name string) tool.Def {
	return tool.Def{
		Name: name, Description: "takes effect", InputSchema: json.RawMessage(objSchema), Caps: tool.CapMutating,
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			e.n.Add(1)
			e.took <- struct{}{}
			if e.hold != nil {
				select {
				case <-e.hold:
				case <-ctx.Done():
				}
			}
			return "took effect", nil
		},
	}
}

func readTool(n *atomic.Int64) tool.Def {
	return tool.Def{
		Name: "read", Description: "reads", InputSchema: json.RawMessage(objSchema), Caps: tool.CapReadOnly,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			n.Add(1)
			return "the file says 42", nil
		},
	}
}

func uses(resp ...llm.ToolUse) llm.Response {
	out := llm.Response{StopReason: llm.StopToolUse}
	for _, u := range resp {
		out.Content = append(out.Content, u)
	}
	return out
}

func call(id, name string) llm.ToolUse {
	return llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(`{}`)}
}

// sentText is everything the model was sent in req, as text.
func sentText(req llm.Request) string {
	b, _ := json.Marshal(req.Messages)
	return string(b)
}

func ledgerOf(t *testing.T, st Store, session string) []ToolCall {
	t.Helper()
	calls, err := st.ToolCalls(testCtx(t), CallFilter{Session: session})
	if err != nil {
		t.Fatal(err)
	}
	return calls
}

func callNamed(t *testing.T, calls []ToolCall, use string) ToolCall {
	t.Helper()
	for _, c := range calls {
		if string(c.UseID) == use {
			return c
		}
	}
	t.Fatalf("no call %s in the ledger: %+v", use, calls)
	return ToolCall{}
}

// gatedBuild serves every session from one model, with tools that a person
// approves before they run.
func gatedBuild(cl llm.Client, tools ...tool.Def) func(SessionOptions) (*core.Agent, error) {
	return builder(cl, core.WithTools(tools...),
		core.WithToolMiddleware(permission.Gate(permission.AskEverything(), permission.Defer())))
}

// ===== The three ways a call used to run twice =====

// An approved call ran, and then the model fell over: the turn is tried
// again, and the call — which the ledger says finished — is answered with
// what it returned, not run a second time.
func TestAnApprovedCallRunsOnceWhenTheModelFailsAfterIt(t *testing.T) {
	pay := newEffect()
	cl := &stepModel{steps: []func() (llm.Response, error){
		reply(uses(call("p1", "pay"))),
		overloaded,
		reply(textTurn("paid")),
	}}
	m := newManager(t, ManagerConfig{Build: gatedBuild(cl, pay.def("pay"))})
	s, _ := m.Create(testCtx(t), SessionOptions{})
	waitEvent(t, s, mustSend(t, s, "pay the bill").ID)

	decision, err := s.Resolve(testCtx(t), "p1", true)
	if err != nil {
		t.Fatal(err)
	}
	if ended := waitEvent(t, s, decision.ID); ended.Answer != "paid" {
		t.Fatalf("the resumed turn = %+v", ended)
	}
	if n := pay.n.Load(); n != 1 {
		t.Fatalf("the approved call took effect %d times, want once", n)
	}
	asked := cl.asked()
	if last := sentText(asked[len(asked)-1]); !strings.Contains(last, "took effect") {
		t.Errorf("the model was not shown what the call returned: %s", last)
	}
	c := callNamed(t, ledgerOf(t, m.store, s.ID()), "p1")
	if c.State != tool.StateSucceeded || c.Attempts != 1 || c.Output != "took effect" || c.Decision == nil || !c.Decision.Allow {
		t.Errorf("the ledger has the call as %+v", c)
	}
}

// An approved call took effect, and the daemon died before it could write
// down that it had: the next daemon does not run it again, because the
// ledger says it was running, and so may have taken effect — it tells the
// model so.
func TestAnApprovedCallRunsOnceAcrossARestart(t *testing.T) {
	st := MemoryStore()
	pay := newEffect()
	pay.hold = make(chan struct{})
	defer close(pay.hold)
	dying := &crashStore{Store: st}
	first := newManager(t, ManagerConfig{Name: "d1", Store: dying, Build: gatedBuild(&stepModel{steps: []func() (llm.Response, error){
		reply(uses(call("p1", "pay"))),
	}}, pay.def("pay"))})
	s, _ := first.Create(testCtx(t), SessionOptions{})
	waitEvent(t, s, mustSend(t, s, "pay the bill").ID)
	decision, err := s.Resolve(testCtx(t), "p1", true)
	if err != nil {
		t.Fatal(err)
	}
	<-pay.took
	dying.crash()

	cl := &stepModel{steps: []func() (llm.Response, error){reply(textTurn("I will check the bank before paying again"))}}
	next := newManager(t, ManagerConfig{Name: "d1", Store: st, Build: gatedBuild(cl, pay.def("pay"))})
	again, err := next.Get(testCtx(t), s.ID())
	if err != nil {
		t.Fatal(err)
	}
	if ended := waitEvent(t, again, decision.ID); ended.State != Idle {
		t.Fatalf("the turn carried on = %+v", ended)
	}
	if n := pay.n.Load(); n != 1 {
		t.Fatalf("the approved call took effect %d times, want once", n)
	}
	if sent := sentText(cl.asked()[0]); !strings.Contains(sent, "may or may not have taken effect") {
		t.Errorf("the model was not told the call may have taken effect: %s", sent)
	}
	c := callNamed(t, ledgerOf(t, st, s.ID()), "p1")
	if c.State != tool.StateFailed || c.Attempts != 1 || !strings.Contains(c.Output, "interrupted") {
		t.Errorf("the ledger has the call as %+v, want it failed as interrupted, run once", c)
	}
}

// A daemon died in the middle of a turn: the next one carries the turn on
// from its last step rather than starting it over — the read it had done is
// not done again, the prompt is not asked again, and the write that was
// running when it died is reported, not run.
func TestATurnCarriesOnAfterARestart(t *testing.T) {
	st := MemoryStore()
	var reads atomic.Int64
	write := newEffect()
	write.hold = make(chan struct{})
	defer close(write.hold)
	dying := &crashStore{Store: st}
	tools := func() []tool.Def { return []tool.Def{readTool(&reads), write.def("write")} }
	first := newManager(t, ManagerConfig{Name: "d1", Store: dying, Build: builder(&stepModel{steps: []func() (llm.Response, error){
		reply(uses(call("r1", "read"))),
		reply(uses(call("w1", "write"))),
	}}, core.WithTools(tools()...))})
	s, _ := first.Create(testCtx(t), SessionOptions{})
	e := mustSend(t, s, "read it, then write it down")
	<-write.took
	dying.crash()

	cl := &stepModel{steps: []func() (llm.Response, error){reply(textTurn("done, though the write may not have landed"))}}
	next := newManager(t, ManagerConfig{Name: "d1", Store: st, Build: builder(cl, core.WithTools(tools()...))})
	again, err := next.Get(testCtx(t), s.ID())
	if err != nil {
		t.Fatal(err)
	}
	ended := waitEvent(t, again, e.ID)
	if ended.State != Idle || ended.Turn != 1 {
		t.Fatalf("the turn carried on = %+v, want turn 1 answered", ended)
	}
	if reads.Load() != 1 || write.n.Load() != 1 {
		t.Fatalf("read %d times and wrote %d; want each once", reads.Load(), write.n.Load())
	}
	asked := cl.asked()
	if len(asked) != 1 {
		t.Fatalf("the next daemon asked the model %d times, want once", len(asked))
	}
	sent := sentText(asked[0])
	if !strings.Contains(sent, "the file says 42") || !strings.Contains(sent, "may or may not have taken effect") {
		t.Errorf("the model was not carried on from where the turn got to: %s", sent)
	}
	if got := strings.Count(transcript(again), "read it, then write it down"); got != 1 {
		t.Errorf("the prompt is in the transcript %d times, want once", got)
	}
	var resumed bool
	for _, f := range collect(t, again, 0, endOfTurn(1)) {
		if ts, ok := f.Event.(TurnStarted); ok && ts.Resumed && ts.Turn == 1 && slices.Equal(ts.Events, []int64{e.ID}) {
			resumed = true
		}
	}
	if !resumed {
		t.Error("no turn_started saying the turn was resumed")
	}
	info := again.Info()
	if info.Turns != 1 || info.State != Idle {
		t.Errorf("the session = %+v, want one turn, answered", info)
	}
	if rec, _ := st.LoadSession(testCtx(t), s.ID()); rec.Inflight != nil {
		t.Errorf("the settled turn is still in flight: %+v", rec.Inflight)
	}
}

// A call safe to run again — a message sent, keyed by the call — is run
// again by the turn carried on, and still sends once.
func TestAMessageSentBeforeARestartIsSentOnce(t *testing.T) {
	st := MemoryStore()
	dying := &crashStore{Store: st, after: "Post"}
	first := newManager(t, ManagerConfig{Name: "d1", Store: dying, Build: builder(&stepModel{steps: []func() (llm.Response, error){
		reply(uses(llm.ToolUse{ID: "m1", Name: ToolSendMessage, Input: json.RawMessage(`{"text":"started"}`)})),
	}}, core.WithTools(Tools()...))})
	s, _ := first.Create(testCtx(t), SessionOptions{})
	e := mustSend(t, s, "start, and say so")
	until(t, "the message to be posted", func() bool { return dying.down.Load() })

	cl := &stepModel{steps: []func() (llm.Response, error){reply(textTurn("said it"))}}
	next := newManager(t, ManagerConfig{Name: "d1", Store: st, Build: builder(cl, core.WithTools(Tools()...))})
	again, _ := next.Get(testCtx(t), s.ID())
	waitEvent(t, again, e.ID)
	var sent int
	msgs, _ := st.Outbox(testCtx(t), OutboxFilter{Session: s.ID()})
	for _, m := range msgs {
		if m.Kind == MessageSent {
			sent++
		}
	}
	if sent != 1 {
		t.Fatalf("the message was sent %d times, want once: %+v", sent, msgs)
	}
	if c := callNamed(t, ledgerOf(t, st, s.ID()), "m1"); c.State != tool.StateSucceeded || c.Attempts != 1 {
		t.Errorf("the ledger has the send as %+v", c)
	}
}

// ===== Spend =====

// A turn tried again is one turn's spend: what each of its model calls
// cost, counted once.
func TestATurnTriedAgainIsCountedOnce(t *testing.T) {
	var reads atomic.Int64
	used := func(r llm.Response) llm.Response {
		r.Usage, r.Model = llm.Usage{InputTokens: 10, OutputTokens: 1}, "test-model"
		return r
	}
	cl := &stepModel{steps: []func() (llm.Response, error){
		reply(used(uses(call("r1", "read")))),
		overloaded,
		reply(used(textTurn("42"))),
	}}
	priced := llm.Chain(cl, core.TrackCost(llm.Table{"test-model": {In: 1000, Out: 1000}}))
	m := newManager(t, ManagerConfig{Build: builder(priced, core.WithTools(readTool(&reads)))})
	s, _ := m.Create(testCtx(t), SessionOptions{})
	e := mustSend(t, s, "what does the file say?")
	var spent []Spend
	for _, f := range s.Follow(testCtx(t), 0) {
		if te, ok := f.Event.(TurnEnded); ok {
			spent = append(spent, te.Spend)
			if te.RetryAt.IsZero() && slices.Contains(te.Events, e.ID) {
				break
			}
		}
	}
	if len(spent) != 2 || spent[0].InputTokens != 10 || spent[1].InputTokens != 10 {
		t.Fatalf("the turn's two runs spent %+v; want 10 input tokens each", spent)
	}
	if got := s.Info().Spend; got.InputTokens != 20 || got.Calls != 2 || got.CostUSD < 0.0219 || got.CostUSD > 0.0221 {
		t.Fatalf("the session spent %+v; want two calls, 20 input tokens, $0.022", got)
	}
	if reads.Load() != 1 {
		t.Errorf("the read ran %d times; the run tried again carries on after it", reads.Load())
	}
	if ts := s.Info().Turns; ts != 1 {
		t.Errorf("%d turns, want the one, tried again", ts)
	}
}

// ===== Waiting for a result =====

// A call that waits for a result from outside ends the turn; the result,
// posted for it, is what the call returns, and the turn carries on.
func TestAResultFromOutsideResumesTheTurn(t *testing.T) {
	cl := &stepModel{steps: []func() (llm.Response, error){
		reply(uses(llm.ToolUse{ID: "j1", Name: ToolWaitForResult, Input: json.RawMessage(`{"handle":"build 7","note":"pass or fail"}`)})),
		reply(textTurn("the build passed")),
	}}
	m := newManager(t, ManagerConfig{Build: builder(cl, core.WithTools(Tools()...))})
	s, _ := m.Create(testCtx(t), SessionOptions{})
	msg := mustSend(t, s, "wait for the build")
	stopped := waitEvent(t, s, msg.ID)
	if stopped.State != Awaiting || len(stopped.Awaiting) != 1 || stopped.Awaiting[0].Handle != "build 7" || len(stopped.Approvals) != 0 {
		t.Fatalf("the turn = %+v, want it awaiting the build", stopped)
	}
	if s.Info().State != Awaiting || len(s.Pending()) != 0 || len(s.Awaited()) != 1 {
		t.Fatalf("state %s, pending %v, awaited %v", s.Info().State, s.Pending(), s.Awaited())
	}
	said := outboxOf(t, m, s.ID(), MessageAwaiting)
	if !strings.Contains(said.Text, "j1") || !strings.Contains(said.Text, "build 7") || !slices.Equal(said.InReplyTo, []int64{msg.ID}) {
		t.Fatalf("the awaiting message = %+v", said)
	}
	if c := callNamed(t, ledgerOf(t, m.store, s.ID()), "j1"); c.State != tool.StateWaiting || c.Handle != "build 7" || c.Reason != "pass or fail" {
		t.Fatalf("the ledger has the call as %+v", c)
	}

	if _, err := s.Resolve(testCtx(t), "j1", true); !errors.Is(err, ErrNoSuchApproval) {
		t.Fatalf("deciding a call that waits for a result = %v, want ErrNoSuchApproval", err)
	}
	if _, err := s.Complete(testCtx(t), "nope", "x", false); !errors.Is(err, ErrNoSuchApproval) {
		t.Fatalf("a result for nothing = %v, want ErrNoSuchApproval", err)
	}
	res, err := s.CompleteAs(testCtx(t), "ci", "j1", "build 7 passed", false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Complete(testCtx(t), "j1", "again", false); !errors.Is(err, ErrAlreadyAnswered) {
		t.Fatalf("a second result = %v, want ErrAlreadyAnswered", err)
	}
	if ended := waitEvent(t, s, res.ID); ended.Answer != "the build passed" {
		t.Fatalf("the resumed turn = %+v", ended)
	}
	asked := cl.asked()
	if last := sentText(asked[len(asked)-1]); !strings.Contains(last, "build 7 passed") {
		t.Errorf("the model was not given the result: %s", last)
	}
	if c := callNamed(t, ledgerOf(t, m.store, s.ID()), "j1"); c.State != tool.StateSucceeded || c.Output != "build 7 passed" {
		t.Errorf("the ledger has the call as %+v", c)
	}
	if reply := outboxOf(t, m, s.ID(), MessageReply); !slices.Equal(reply.InReplyTo, []int64{msg.ID}) {
		t.Errorf("the reply = %+v, want it in reply to the message that started it", reply)
	}
}

// A result is posted through its own door, not as any event.
func TestAResultIsNotReceivedAsAnEvent(t *testing.T) {
	m, s := session(t, script())
	if _, _, err := m.Receive(testCtx(t), Inbound{Session: s.ID(), Kind: KindResult, Payload: json.RawMessage(`{}`)}); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("Receive(result) = %v, want ErrBadRequest", err)
	}
}

// ===== The ledger =====

// Every call a turn makes is in the ledger: what it was, how it ended, and
// how many times it ran; the turn's answer is a step of its own.
func TestTheLedgerHasEveryCall(t *testing.T) {
	var reads atomic.Int64
	cl := &stepModel{steps: []func() (llm.Response, error){
		reply(uses(call("r1", "read"), call("r2", "read"))),
		reply(textTurn("42")),
	}}
	m := newManager(t, ManagerConfig{Build: builder(cl, core.WithTools(readTool(&reads)))})
	s, _ := m.Create(testCtx(t), SessionOptions{})
	send(t, s, "read twice")
	calls, err := m.ToolCalls(testCtx(t), CallFilter{Session: s.ID()})
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 {
		t.Fatalf("the ledger has %+v, want both reads", calls)
	}
	for _, c := range calls {
		if c.State != tool.StateSucceeded || c.Attempts != 1 || c.Output != "the file says 42" || c.Turn != 1 ||
			c.Step != 1 || c.Started.IsZero() || c.Finished.IsZero() || c.Owner != m.Name() {
			t.Errorf("a read in the ledger: %+v", c)
		}
	}
	if rec, _ := m.store.LoadSession(testCtx(t), s.ID()); rec.Inflight != nil {
		t.Errorf("an answered turn is still in flight: %+v", rec.Inflight)
	}
	// The answer is step 2: the next turn's calls are numbered after it.
	cl.mu.Lock()
	cl.steps = append(cl.steps, reply(uses(call("r3", "read"))), reply(textTurn("still 42")))
	cl.mu.Unlock()
	send(t, s, "again")
	if c := callNamed(t, ledgerOf(t, m.store, s.ID()), "r3"); c.Step != 3 || c.Turn != 2 {
		t.Errorf("the next turn's call: %+v", c)
	}
}

// A decision is written on the call it decides, when it is made.
func TestADecisionIsInTheLedger(t *testing.T) {
	pay := newEffect()
	cl := &stepModel{steps: []func() (llm.Response, error){
		reply(uses(call("a", "pay"), call("b", "pay"))),
		reply(textTurn("one paid")),
	}}
	m := newManager(t, ManagerConfig{Build: gatedBuild(cl, pay.def("pay"))})
	s, _ := m.Create(testCtx(t), SessionOptions{})
	waitEvent(t, s, mustSend(t, s, "pay both").ID)
	if c := callNamed(t, ledgerOf(t, m.store, s.ID()), "a"); c.State != tool.StateDeferred || c.Decision != nil {
		t.Fatalf("before any decision: %+v", c)
	}
	if _, err := s.ResolveAs(testCtx(t), "carol", "a", false); err != nil {
		t.Fatal(err)
	}
	until(t, "the decision on the ledger", func() bool {
		c := callNamed(t, ledgerOf(t, m.store, s.ID()), "a")
		return c.Decision != nil
	})
	c := callNamed(t, ledgerOf(t, m.store, s.ID()), "a")
	if c.State != tool.StateDeferred || c.Decision.Allow || c.Decision.By.From != "carol" || c.Decision.By.Kind != KindApproval {
		t.Fatalf("decided, not yet resumed: %+v", c)
	}
	last, err := s.Resolve(testCtx(t), "b", true)
	if err != nil {
		t.Fatal(err)
	}
	waitEvent(t, s, last.ID)
	calls := ledgerOf(t, m.store, s.ID())
	a, b := callNamed(t, calls, "a"), callNamed(t, calls, "b")
	if a.State != tool.StateDenied || a.Decision == nil || a.Decision.By.From != "carol" {
		t.Errorf("the refused call: %+v", a)
	}
	if b.State != tool.StateSucceeded || b.Decision == nil || !b.Decision.Allow || b.Attempts != 1 {
		t.Errorf("the allowed call: %+v", b)
	}
}

// The ledger forgets what has ended once it is older than it keeps.
func TestTheLedgerIsPruned(t *testing.T) {
	var reads atomic.Int64
	cl := &stepModel{steps: []func() (llm.Response, error){reply(uses(call("r1", "read"))), reply(textTurn("42"))}}
	m := newManager(t, ManagerConfig{Build: builder(cl, core.WithTools(readTool(&reads))), LedgerRetention: time.Hour})
	s, _ := m.Create(testCtx(t), SessionOptions{})
	send(t, s, "read")
	m.pruneAt(time.Now())
	if calls := ledgerOf(t, m.store, s.ID()); len(calls) != 1 {
		t.Fatalf("pruned a call that ended just now: %+v", calls)
	}
	m.pruneAt(time.Now().Add(2 * time.Hour))
	if calls := ledgerOf(t, m.store, s.ID()); len(calls) != 0 {
		t.Fatalf("kept %+v past its retention", calls)
	}
}

// A turn given up on — its events failed — is not carried on by the next
// event: that one starts a turn of its own, and the calls the turn left
// open are closed.
func TestATurnGivenUpOnIsNotCarriedOn(t *testing.T) {
	st := MemoryStore()
	write := newEffect()
	write.hold = make(chan struct{})
	defer close(write.hold)
	dying := &crashStore{Store: st}
	first := newManager(t, ManagerConfig{Name: "d1", Store: dying, Attempts: 1, Build: builder(&stepModel{steps: []func() (llm.Response, error){
		reply(uses(call("w1", "write"))),
	}}, core.WithTools(write.def("write")))})
	s, _ := first.Create(testCtx(t), SessionOptions{})
	e := mustSend(t, s, "write")
	<-write.took
	dying.crash()

	// Started again after one attempt of one: the event is failed, not put
	// back.
	cl := &stepModel{steps: []func() (llm.Response, error){reply(textTurn("hello again"))}}
	next := newManager(t, ManagerConfig{Name: "d1", Store: st, Attempts: 1, Build: builder(cl, core.WithTools(write.def("write")))})
	if got := eventStatus(t, st, e.ID); got.Status != EventFailed {
		t.Fatalf("the dead turn's event is %s, want failed", got.Status)
	}
	again, err := next.Get(testCtx(t), s.ID())
	if err != nil {
		t.Fatal(err)
	}
	if ended := send(t, again, "hello"); ended.Answer != "hello again" || ended.Turn != 2 {
		t.Fatalf("the next turn = %+v, want a turn of its own", ended)
	}
	if write.n.Load() != 1 {
		t.Fatalf("the write ran %d times", write.n.Load())
	}
	if c := callNamed(t, ledgerOf(t, st, s.ID()), "w1"); c.State != tool.StateInterrupted {
		t.Errorf("the call the dead turn left running: %+v, want it closed as interrupted", c)
	}
	if strings.Contains(sentText(cl.asked()[0]), "took effect") {
		t.Error("the new turn was given the dead turn's work")
	}
}
