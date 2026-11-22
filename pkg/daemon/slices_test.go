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
	"wombat/pkg/core/tool"
)

// ===== Turns in slices =====

// aNanosecond is a slice every turn outruns: each does one unit of work —
// a model call, or a batch of calls — and stops.
const aNanosecond = time.Nanosecond

// readWriteAnswer is a model that reads, writes and answers, with the usage
// a priced model reports.
func readWriteAnswer() *stepModel {
	used := func(r llm.Response) llm.Response {
		r.Usage, r.Model = llm.Usage{InputTokens: 10, OutputTokens: 1}, "test-model"
		return r
	}
	return &stepModel{steps: []func() (llm.Response, error){
		reply(used(uses(call("r1", "read")))),
		reply(used(uses(call("w1", "write")))),
		reply(used(textTurn("written"))),
	}}
}

func priced(cl llm.Client) llm.Client {
	return llm.Chain(cl, core.TrackCost(llm.Table{"test-model": {In: 1000, Out: 1000}}))
}

// framesOf is every frame a session's log has, read the way a client
// following it reads them.
func framesOf(t *testing.T, s *Session, stop func(Frame) bool) []Frame {
	t.Helper()
	return collect(t, s, 0, stop)
}

func kinds(frames []Frame) []string {
	var out []string
	for _, f := range frames {
		out = append(out, f.Event.Kind())
	}
	return out
}

func count(xs []string, x string) int {
	n := 0
	for _, y := range xs {
		if y == x {
			n++
		}
	}
	return n
}

// A turn run in slices — each as short as a slice can be — does what the
// turn in one go does: each call once, one turn, its spend counted once, its
// events done and tried once; the log says where the slices ended.
func TestATurnRunsInSlices(t *testing.T) {
	var reads atomic.Int64
	write := newEffect()
	m := newManager(t, ManagerConfig{Slice: aNanosecond,
		Build: builder(priced(readWriteAnswer()), core.WithTools(readTool(&reads), write.def("write")))})
	s, _ := m.Create(testCtx(t), SessionOptions{})
	e := mustSend(t, s, "read it, then write it")
	ended := waitEvent(t, s, e.ID)
	if ended.Answer != "written" || ended.Turn != 1 {
		t.Fatalf("the turn = %+v", ended)
	}
	if reads.Load() != 1 || write.n.Load() != 1 {
		t.Fatalf("read %d times and wrote %d; want each once", reads.Load(), write.n.Load())
	}
	info := s.Info()
	if info.Turns != 1 || info.State != Idle || info.Spend.InputTokens != 30 || info.Spend.Calls != 3 {
		t.Errorf("the session = %+v; want one turn, idle, three calls of 10 tokens counted once", info)
	}
	got := kinds(framesOf(t, s, endOfTurn(1)))
	if count(got, "turn_started") != 1 || count(got, "turn_suspended") < 4 || count(got, "turn_ended") != 1 {
		t.Errorf("the log = %v; want one turn_started, a turn_suspended per slice, one turn_ended", got)
	}
	if ev := eventStatus(t, m.store, e.ID); ev.Status != EventDone || ev.Attempts != 1 {
		t.Errorf("the event = %+v; want done, tried once", ev)
	}
	if rec, _ := m.store.LoadSession(testCtx(t), s.ID()); rec.Inflight != nil {
		t.Errorf("an answered turn is still in flight: %+v", rec.Inflight)
	}
	if c := callNamed(t, ledgerOf(t, m.store, s.ID()), "w1"); c.State != tool.StateSucceeded || c.Attempts != 1 {
		t.Errorf("the write in the ledger: %+v", c)
	}
}

// wakes records what a Stateless daemon asks to be woken for.
type wakes struct {
	mu  sync.Mutex
	got []string
}

func (w *wakes) wake(_ context.Context, session string, _ time.Time) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.got = append(w.got, session)
	return nil
}

func (w *wakes) n() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.got)
}

// stateless is a daemon that keeps nothing between calls, on st.
func stateless(t *testing.T, st Store, name string, cl llm.Client, w *wakes, tools ...tool.Def) *Manager {
	t.Helper()
	cfg := ManagerConfig{Store: st, Name: name, Stateless: true, Slice: aNanosecond,
		Build: builder(cl, core.WithTools(tools...))}
	if w != nil {
		cfg.Wake = w.wake
	}
	return newManager(t, cfg)
}

// advance runs a session's slices, one call at a time, alternating between
// daemons as a queue spreading calls over instances would, until it has no
// more work; it reports how many slices it took.
func advance(t *testing.T, id string, daemons ...*Manager) int {
	t.Helper()
	for n := 1; n <= 100; n++ {
		more, err := daemons[n%len(daemons)].Advance(testCtx(t), id)
		if err != nil {
			t.Fatalf("Advance %d: %v", n, err)
		}
		if !more {
			return n
		}
	}
	t.Fatal("the session never ran out of work")
	return 0
}

// A daemon that keeps nothing between calls runs a turn one call at a
// time — each call a slice, on whichever instance gets it — and every call
// asks to be woken for the next. A client reads the log from the store.
func TestAStatelessDaemonRunsATurnCallByCall(t *testing.T) {
	st := MemoryStore()
	var reads atomic.Int64
	write := newEffect()
	cl := priced(readWriteAnswer())
	w := &wakes{}
	a := stateless(t, st, "a", cl, w, readTool(&reads), write.def("write"))
	b := stateless(t, st, "b", cl, w, readTool(&reads), write.def("write"))

	s, err := a.Create(testCtx(t), SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	e := mustSend(t, s, "read it, then write it")
	if w.n() != 1 {
		t.Fatalf("%d wake-ups for the message, want one", w.n())
	}
	slicesTaken := advance(t, s.ID(), a, b)
	if slicesTaken < 5 {
		t.Errorf("%d calls; want a slice per unit of work", slicesTaken)
	}
	if w.n() < slicesTaken {
		t.Errorf("%d wake-ups for %d slices; each slice that left work asks for the next", w.n(), slicesTaken)
	}
	if reads.Load() != 1 || write.n.Load() != 1 {
		t.Fatalf("read %d times and wrote %d; want each once", reads.Load(), write.n.Load())
	}
	if ev := eventStatus(t, st, e.ID); ev.Status != EventDone || ev.Attempts != 1 {
		t.Errorf("the event = %+v; want done, tried once", ev)
	}
	rec, _ := st.LoadSession(testCtx(t), s.ID())
	if rec.State != Idle || rec.Answer != "written" || rec.Lease.Owner != "" || rec.Inflight != nil {
		t.Fatalf("the session = state %s answer %q lease %+v inflight %v; want idle, answered, held by nobody", rec.State, rec.Answer, rec.Lease, rec.Inflight)
	}
	if rec.Spend.InputTokens != 30 {
		t.Errorf("spend %+v; want three calls' tokens, counted once", rec.Spend)
	}

	// A client reads the log from the store, as a view: in order, no gap,
	// ending with the turn.
	view, err := b.Get(testCtx(t), s.ID())
	if err != nil {
		t.Fatal(err)
	}
	frames := framesOf(t, view, endOfTurn(1))
	for i, f := range frames {
		if f.Seq != i {
			t.Fatalf("frame %d has seq %d: %v", i, f.Seq, kinds(frames))
		}
	}
	got := kinds(frames)
	if got[0] != "turn_started" || got[len(got)-1] != "turn_ended" || count(got, "turn_suspended") < 4 || count(got, "tool_done") != 2 {
		t.Errorf("the log = %v", got)
	}
}

// A view takes no lease, runs nothing, and says what the store says.
func TestAViewHoldsNothing(t *testing.T) {
	st := MemoryStore()
	a := stateless(t, st, "a", script(textTurn("hi")), nil)
	s, _ := a.Create(testCtx(t), SessionOptions{Title: "t"})
	view, err := a.Get(testCtx(t), s.ID())
	if err != nil {
		t.Fatal(err)
	}
	if rec, _ := st.LoadSession(testCtx(t), s.ID()); rec.Lease.Owner != "" {
		t.Fatalf("reading the session took it: %+v", rec.Lease)
	}
	if view.Info().Title != "t" || view.Info().State != Idle {
		t.Errorf("the view = %+v", view.Info())
	}
	if _, err := a.Get(testCtx(t), "nope"); !errors.Is(err, ErrNoSuchSession) {
		t.Errorf("Get(unknown) = %v", err)
	}
}

// A message steering a turn that runs elsewhere, or between its slices, goes
// through the inbox, and the turn takes it at its next model call.
func TestSteeringATurnBetweenSlices(t *testing.T) {
	st := MemoryStore()
	var reads atomic.Int64
	cl := &stepModel{steps: []func() (llm.Response, error){
		reply(uses(call("r1", "read"))),
		reply(textTurn("done, in v2")),
	}}
	a := stateless(t, st, "a", cl, nil, readTool(&reads))
	s, _ := a.Create(testCtx(t), SessionOptions{})
	mustSend(t, s, "read it")
	// The first slice calls the model, and stops before the read.
	if more, err := a.Advance(testCtx(t), s.ID()); err != nil || !more {
		t.Fatalf("Advance = %v, %v", more, err)
	}
	view, _ := a.Get(testCtx(t), s.ID())
	if view.Info().State != Running {
		t.Fatalf("between slices the session is %s, want running", view.Info().State)
	}
	if err := view.Steer("use the v2 api"); err != nil {
		t.Fatalf("Steer between slices = %v", err)
	}
	advance(t, s.ID(), a)
	asked := cl.asked()
	if last := sentText(asked[len(asked)-1]); !strings.Contains(last, "use the v2 api") {
		t.Errorf("the steering message did not reach the model: %s", last)
	}
	again, _ := a.Get(testCtx(t), s.ID())
	if got := kinds(framesOf(t, again, endOfTurn(1))); count(got, "steer_sent") != 1 {
		t.Errorf("the log = %v; want the message echoed once", got)
	}
	if rec, _ := st.LoadSession(testCtx(t), s.ID()); rec.Answer != "done, in v2" {
		t.Errorf("answer %q", rec.Answer)
	}
}

// A turn cancelled between its slices ends there, without running again,
// and the calls it left open are closed.
func TestCancellingATurnBetweenSlices(t *testing.T) {
	st := MemoryStore()
	write := newEffect()
	cl := &stepModel{steps: []func() (llm.Response, error){reply(uses(call("w1", "write")))}}
	a := stateless(t, st, "a", cl, nil, write.def("write"))
	s, _ := a.Create(testCtx(t), SessionOptions{})
	e := mustSend(t, s, "write it")
	if more, _ := a.Advance(testCtx(t), s.ID()); !more {
		t.Fatal("the first slice left nothing to do")
	}
	view, _ := a.Get(testCtx(t), s.ID())
	view.Cancel()
	advance(t, s.ID(), a)
	if write.n.Load() != 0 {
		t.Fatalf("the cancelled turn wrote %d times", write.n.Load())
	}
	rec, _ := st.LoadSession(testCtx(t), s.ID())
	if rec.State != Failed || rec.ErrorKind != "cancelled" || rec.Inflight != nil {
		t.Fatalf("the session = %s %s, inflight %v; want failed as cancelled, nothing in flight", rec.State, rec.ErrorKind, rec.Inflight)
	}
	if ev := eventStatus(t, st, e.ID); ev.Status != EventFailed {
		t.Errorf("the turn's event = %+v, want failed", ev)
	}
	if c := callNamed(t, ledgerOf(t, st, s.ID()), "w1"); c.State != tool.StateCancelled {
		t.Errorf("the write in the ledger: %+v, want cancelled", c)
	}
	events, _ := st.Events(testCtx(t), EventFilter{Session: s.ID()})
	for _, ev := range events {
		if ev.Kind == KindCancel && ev.Status != EventDone {
			t.Errorf("the cancellation = %+v, want done", ev)
		}
	}
}

// Tick finds what a lost wake-up left — a session with events to take, a
// timer due — and asks for it again.
func TestTickFindsWhatAWakeUpLost(t *testing.T) {
	st := MemoryStore()
	w := &wakes{}
	a := stateless(t, st, "a", script(textTurn("hi")), w)
	s, _ := a.Create(testCtx(t), SessionOptions{})
	other, _ := a.Create(testCtx(t), SessionOptions{})
	payload, _ := json.Marshal(MessagePayload{Text: "hello"})
	// Straight into the store: nobody was told.
	if _, _, err := st.Enqueue(testCtx(t), Event{Session: s.ID(), Kind: KindMessage, Source: SourceAPI, Payload: payload, Created: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if _, err := st.AddTimer(testCtx(t), Timer{Session: other.ID(), Note: "look", Due: time.Now().Add(-time.Second), CreatedBy: "api"}); err != nil {
		t.Fatal(err)
	}
	ready, err := a.Tick(testCtx(t))
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(ready)
	want := []string{s.ID(), other.ID()}
	slices.Sort(want)
	if !slices.Equal(ready, want) {
		t.Fatalf("Tick found %v, want both sessions", ready)
	}
	w.mu.Lock()
	got := slices.Clone(w.got)
	w.mu.Unlock()
	if !slices.Contains(got, s.ID()) || !slices.Contains(got, other.ID()) {
		t.Errorf("woke %v", got)
	}
}

// A slice that died — its instance gone mid-call — is carried on by the next
// once its lease is let go: from where it was written down, the call that
// was running reported, not run again.
func TestAStatelessSliceThatDiedIsCarriedOn(t *testing.T) {
	st := MemoryStore()
	write := newEffect()
	write.hold = make(chan struct{})
	defer close(write.hold)
	dying := &crashStore{Store: st}
	cl := &stepModel{steps: []func() (llm.Response, error){
		reply(uses(call("w1", "write"))),
		reply(textTurn("checked, and carried on")),
	}}
	a := newManager(t, ManagerConfig{Store: dying, Name: "a", Stateless: true, Slice: time.Hour,
		Build: builder(cl, core.WithTools(write.def("write")))})
	b := stateless(t, st, "b", cl, nil, write.def("write"))
	s, _ := b.Create(testCtx(t), SessionOptions{})
	mustSend(t, s, "write it")
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = a.Advance(context.Background(), s.ID())
	}()
	<-write.took
	dying.crash()
	// Its lease would run out; let it go now.
	if err := st.Release(testCtx(t), "a", []string{s.ID()}); err != nil {
		t.Fatal(err)
	}
	advance(t, s.ID(), b)
	rec, _ := st.LoadSession(testCtx(t), s.ID())
	if rec.Answer != "checked, and carried on" || write.n.Load() != 1 {
		t.Fatalf("answer %q after %d writes", rec.Answer, write.n.Load())
	}
	if c := callNamed(t, ledgerOf(t, st, s.ID()), "w1"); c.State != tool.StateFailed || !strings.Contains(c.Output, "interrupted") {
		t.Errorf("the write in the ledger: %+v", c)
	}
}

// A permission changed while a turn runs somewhere would be written over
// when it ends: a Stateless daemon refuses it until then.
func TestAPermissionWaitsForTheTurn(t *testing.T) {
	st := MemoryStore()
	cl := &stepModel{steps: []func() (llm.Response, error){reply(uses(call("r1", "read"))), reply(textTurn("done"))}}
	var reads atomic.Int64
	a := stateless(t, st, "a", cl, nil, readTool(&reads))
	s, _ := a.Create(testCtx(t), SessionOptions{})
	mustSend(t, s, "read")
	if more, _ := a.Advance(testCtx(t), s.ID()); !more {
		t.Fatal("nothing left after the first slice")
	}
	view, _ := a.Get(testCtx(t), s.ID())
	if err := view.SetPermission(testCtx(t), "readonly"); !errors.Is(err, ErrBusy) {
		t.Fatalf("SetPermission mid-turn = %v, want ErrBusy", err)
	}
	advance(t, s.ID(), a)
	view, _ = a.Get(testCtx(t), s.ID())
	if err := view.SetPermission(testCtx(t), "readonly"); err != nil {
		t.Fatalf("SetPermission between turns = %v", err)
	}
	if again, _ := a.Get(testCtx(t), s.ID()); again.Info().Options.Permission != "readonly" {
		t.Errorf("options = %+v", again.Info().Options)
	}
}
