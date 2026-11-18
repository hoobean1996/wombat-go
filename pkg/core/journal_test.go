package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// memJournal is a Journal in memory: it keeps every write, and the calls of
// the step it has open as they were last written, which is what a run
// resumed from it is given back.
type memJournal struct {
	mu     sync.Mutex
	steps  []Step
	tracks []Call
	closes [][]Call
	open   map[int]Call

	failOpen, failTrack, failClose error
	// failRunning fails only the writes of a call as running.
	failRunning error
	keys        map[llm.ToolUseID]string
}

func (j *memJournal) Open(_ context.Context, s Step) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.failOpen != nil {
		return j.failOpen
	}
	j.steps = append(j.steps, s)
	j.open = map[int]Call{}
	for _, c := range s.Calls {
		j.open[c.Index] = c
	}
	return nil
}

func (j *memJournal) Track(_ context.Context, c Call) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.failTrack != nil {
		return j.failTrack
	}
	if j.failRunning != nil && c.State == tool.StateRunning {
		return j.failRunning
	}
	j.tracks = append(j.tracks, c)
	if j.open == nil {
		j.open = map[int]Call{} // a resumed run's step, opened before
	}
	j.open[c.Index] = c
	return nil
}

func (j *memJournal) Close(_ context.Context, calls []Call) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.failClose != nil {
		return j.failClose
	}
	j.closes = append(j.closes, calls)
	if j.open == nil {
		j.open = map[int]Call{}
	}
	for _, c := range calls {
		j.open[c.Index] = c
	}
	return nil
}

// checkpoint is what the journal has written: what a run resumed from it
// starts with.
func (j *memJournal) checkpoint() Checkpoint {
	j.mu.Lock()
	defer j.mu.Unlock()
	calls := slices.Collect(maps.Values(j.open))
	slices.SortFunc(calls, func(a, b Call) int { return a.Index - b.Index })
	cp := Checkpoint{Calls: calls}
	if len(j.steps) > 0 {
		last := j.steps[len(j.steps)-1]
		cp.Messages, cp.Iterations = last.Messages, last.Iteration
	}
	return cp
}

func (j *memJournal) states() map[llm.ToolUseID]tool.State {
	j.mu.Lock()
	defer j.mu.Unlock()
	out := map[llm.ToolUseID]tool.State{}
	for _, c := range j.open {
		out[c.Use.ID] = c.State
	}
	return out
}

// keyedJournal names its calls.
type keyedJournal struct{ *memJournal }

func (j keyedJournal) Key(id llm.ToolUseID) string { return "call-" + string(id) }

// counting is a tool that records how often it ran and what it saw.
type counting struct {
	n    atomic.Int64
	keys sync.Map
}

func (c *counting) def(name string, caps tool.Cap) tool.Def {
	return tool.Def{
		Name: name, Description: name, InputSchema: json.RawMessage(objSchema), Caps: caps,
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			c.n.Add(1)
			c.keys.Store(tool.CallKey(ctx), true)
			return name + " done", nil
		},
	}
}

func use(id, name string) llm.ToolUse {
	return llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(objSchema)}
}

func journaled(j Journal) context.Context { return WithJournal(context.Background(), j) }

// The reply and its calls are written down before any of them runs, and a
// call whose tool is not safe to run again is written as running first.
func TestTheStepIsWrittenDownBeforeItsCallsRun(t *testing.T) {
	j := &memJournal{}
	var seen map[llm.ToolUseID]tool.State
	var steps int
	writeTool := tool.Def{
		Name: "write", Description: "writes", InputSchema: json.RawMessage(objSchema), Caps: tool.CapMutating,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			seen, steps = j.states(), len(j.steps)
			return "written", nil
		},
	}
	reads := &counting{}
	cl := scripted(toolsTurn(use("r1", "read"), use("w1", "write")), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{reads.def("read", tool.CapReadOnly), writeTool})
	if _, err := a.Run(journaled(j), Ask("go")); err != nil {
		t.Fatal(err)
	}

	if steps != 1 {
		t.Fatalf("%d steps were written when the write ran, want its own", steps)
	}
	if seen["w1"] != tool.StateRunning {
		t.Errorf("the write ran while the journal had it %q, want running", seen["w1"])
	}
	s := j.steps[0]
	if len(s.Calls) != 2 || s.Calls[0].State != tool.StatePending || s.Calls[1].Use.ID != "w1" || s.Iteration != 1 {
		t.Errorf("step = %+v, want both calls pending, at iteration 1", s.Calls)
	}
	if last := s.Messages[len(s.Messages)-1]; last.Role != llm.RoleAssistant || len(llm.ToolUses(last.Content)) != 2 {
		t.Errorf("the step's transcript does not end on the reply: %+v", last)
	}
	// The read is run again should the run be resumed, so nothing was written
	// before it ran; the write was, and again once it had.
	var tracked []string
	for _, c := range j.tracks {
		tracked = append(tracked, fmt.Sprintf("%s:%s", c.Use.ID, c.State))
	}
	if !slices.Equal(tracked, []string{"w1:running", "w1:succeeded"}) {
		t.Errorf("tracked %v, want the write running and then done", tracked)
	}
	if len(j.closes) != 1 || j.closes[0][0].State != tool.StateSucceeded || j.closes[0][1].Result.Content != "written" {
		t.Errorf("close = %+v, want both calls done with their results", j.closes)
	}
	// And the answer is written down too, as a step with no calls.
	if len(j.steps) != 2 || len(j.steps[1].Calls) != 0 || j.steps[1].Iteration != 2 {
		t.Errorf("steps = %d, want the answer written as a second step", len(j.steps))
	}
}

// What a gate refuses, defers, or what fails, is written down as such.
func TestCloseSaysHowEachCallEnded(t *testing.T) {
	j := &memJournal{}
	failing := tool.Def{Name: "fail", Description: "fails", InputSchema: json.RawMessage(objSchema),
		Fn: func(context.Context, json.RawMessage) (string, error) { return "", errors.New("boom") }}
	refuse := func(next tool.Handler) tool.Handler {
		return func(ctx context.Context, d tool.Def, u llm.ToolUse) (string, error) {
			if d.Name == "secret" {
				return "", errors.New("refused by policy")
			}
			return next(ctx, d, u)
		}
	}
	w := &writes{}
	cl := scripted(toolsTurn(use("f1", "fail"), use("s1", "secret"), writeCall))
	a := newAgent(t, cl, []tool.Def{failing, echoTool("secret", "x"), w.tool()},
		WithToolMiddleware(deferWrites, refuse))
	r := a.Start(journaled(j), Ask("go"))
	drain(t, r)
	if _, ok := r.Outcome().(Deferred); !ok {
		t.Fatalf("outcome = %#v (%v), want Deferred", r.Outcome(), r.Err())
	}
	got := j.closes[0]
	want := []tool.State{tool.StateFailed, tool.StateDenied, tool.StateDeferred}
	for i, c := range got {
		if c.State != want[i] {
			t.Errorf("call %d (%s) = %s, want %s", i, c.Use.ID, c.State, want[i])
		}
	}
	if got[2].Reason != "a write needs a person" || got[2].Result != nil {
		t.Errorf("the deferred call = %+v, want its reason and no result", got[2])
	}
	if got[0].Result == nil || !got[0].Result.IsError || got[0].Result.Content != "boom" {
		t.Errorf("the failed call = %+v, want its error as its result", got[0])
	}
	if !got[0].Ran || got[1].Ran || got[2].Ran {
		t.Errorf("Ran = %v, %v, %v; want only the call that got to its tool", got[0].Ran, got[1].Ran, got[2].Ran)
	}
}

// A journal that cannot write the step down stops the run before anything
// runs; one that cannot write a call as running stops it before that call.
func TestAJournalThatCannotWriteStopsTheRun(t *testing.T) {
	t.Run("the step", func(t *testing.T) {
		w := &writes{}
		j := &memJournal{failOpen: errors.New("disk full")}
		a := newAgent(t, scripted(toolsTurn(writeCall)), []tool.Def{w.tool()})
		_, err := a.Run(journaled(j), Ask("go"))
		if !errors.Is(err, ErrJournal) || w.n.Load() != 0 {
			t.Fatalf("err = %v, ran %d times; want ErrJournal before the write ran", err, w.n.Load())
		}
	})
	t.Run("a call", func(t *testing.T) {
		w := &writes{}
		reads := &counting{}
		j := &memJournal{failRunning: errors.New("disk full")}
		cl := scripted(toolsTurn(use("r1", "read"), writeCall))
		a := newAgent(t, cl, []tool.Def{reads.def("read", tool.CapReadOnly), w.tool()})
		r := a.Start(journaled(j), Ask("go"))
		drain(t, r)
		if !errors.Is(r.Err(), ErrJournal) || w.n.Load() != 0 {
			t.Fatalf("err = %v, ran %d times; want ErrJournal, and the write not run", r.Err(), w.n.Load())
		}
		// The read ran, and the transcript keeps what really happened.
		if !strings.Contains(allText(r.Messages()), "read done") || !strings.Contains(allText(r.Messages()), "did not run") {
			t.Errorf("transcript = %q, want the read's result and the write's refusal", allText(r.Messages()))
		}
	})
}

// A step resumed from its journal: a call that finished is answered with what
// it returned, and not run again; one that never started runs.
func TestRecoverAnswersFinishedCallsFromTheRecord(t *testing.T) {
	w := &writes{}
	reads := &counting{}
	cl := scripted(textTurn("all done"))
	a := newAgent(t, cl, []tool.Def{reads.def("read", tool.CapReadOnly), w.tool()})
	reply := llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{writeCall, use("r1", "read")}}
	cp := Checkpoint{
		Messages: []llm.Message{userMsg("copy"), reply},
		Calls: []Call{
			{Index: 0, Use: writeCall, State: tool.StateSucceeded, Result: &llm.ToolResult{ToolUseID: "w1", Content: "written before the crash"}},
			{Index: 1, Use: use("r1", "read"), State: tool.StatePending},
		},
		Iterations: 1,
	}
	r := a.Start(context.Background(), Recover(cp))
	_, evs := drain(t, r)
	if ans, ok := r.Outcome().(Answer); !ok || ans.Text != "all done" {
		t.Fatalf("outcome = %#v (%v)", r.Outcome(), r.Err())
	}
	if w.n.Load() != 0 || reads.n.Load() != 1 {
		t.Errorf("write ran %d times, read %d; want the write not again and the read once", w.n.Load(), reads.n.Load())
	}
	sent := cl.request(t, 0).Messages
	results := sent[len(sent)-1].Content
	if res := results[0].(llm.ToolResult); res.Content != "written before the crash" {
		t.Errorf("first result = %+v, want the recorded one", res)
	}
	if res := results[1].(llm.ToolResult); res.Content != "read done" {
		t.Errorf("second result = %+v, want the read's", res)
	}
	if !hasRecovered(evs, "w1", "replayed") {
		t.Errorf("no tool_recovered event replaying the write: %v", evs)
	}
}

func hasRecovered(evs []Event, id llm.ToolUseID, action string) bool {
	for _, e := range evs {
		if tr, ok := e.(ToolRecovered); ok && tr.UseID == id && tr.Action == action {
			return true
		}
	}
	return false
}

// A call that was running when the run stopped is recovered as its tool says.
func TestRecoverHandlesAnInterruptedCallAsItsToolSays(t *testing.T) {
	reply := func(u llm.ToolUse) Checkpoint {
		return Checkpoint{
			Messages: []llm.Message{userMsg("go"), {Role: llm.RoleAssistant, Content: []llm.ContentBlock{u}}},
			Calls:    []Call{{Index: 0, Use: u, State: tool.StateRunning}},
		}
	}
	for _, tc := range []struct {
		name     string
		caps     tool.Cap
		recovery tool.Recovery
		ran      int64
		action   string
		want     string
	}{
		{"a write is reported", tool.CapMutating, tool.RecoverAuto, 0, "reported", "may or may not have taken effect"},
		{"a read is run again", tool.CapReadOnly, tool.RecoverAuto, 1, "rerun", "x done"},
		{"one that says rerun is", tool.CapMutating, tool.RecoverRerun, 1, "rerun", "x done"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &counting{}
			def := c.def("x", tc.caps)
			def.Recovery = tc.recovery
			cl := scripted(textTurn("ok"))
			a := newAgent(t, cl, []tool.Def{def})
			j := &memJournal{}
			r := a.Start(journaled(j), Recover(reply(use("x1", "x"))))
			_, evs := drain(t, r)
			if r.Err() != nil {
				t.Fatal(r.Err())
			}
			if c.n.Load() != tc.ran {
				t.Errorf("ran %d times, want %d", c.n.Load(), tc.ran)
			}
			// The journal is told which: a call reported is not one that ran.
			if len(j.closes) == 0 || j.closes[0][0].Ran != (tc.ran == 1) {
				t.Errorf("closed with %+v; want Ran %v", j.closes, tc.ran == 1)
			}
			if got := allText(cl.request(t, 0).Messages); !strings.Contains(got, tc.want) {
				t.Errorf("the model read %q, want %q", got, tc.want)
			}
			if !hasRecovered(evs, "x1", tc.action) {
				t.Errorf("no tool_recovered %s event", tc.action)
			}
		})
	}
}

// Ask defers the interrupted call; a person's yes runs it, a no answers it.
func TestAnInterruptedCallCanBeLeftToAPerson(t *testing.T) {
	c := &counting{}
	def := c.def("deploy", tool.CapExec)
	def.Recovery = tool.RecoverAsk
	u := use("d1", "deploy")
	cp := Checkpoint{
		Messages: []llm.Message{userMsg("deploy"), {Role: llm.RoleAssistant, Content: []llm.ContentBlock{u}}},
		Calls:    []Call{{Index: 0, Use: u, State: tool.StateInterrupted}},
	}
	j := &memJournal{}
	a := newAgent(t, scripted(textTurn("deployed"), textTurn("left alone")), []tool.Def{def})
	r := a.Start(journaled(j), Recover(cp))
	drain(t, r)
	d, ok := r.Outcome().(Deferred)
	if !ok || len(d.Calls) != 1 || d.Calls[0].UseID != "d1" || !strings.Contains(d.Calls[0].Reason, "run it again") {
		t.Fatalf("outcome = %#v (%v), want the call deferred with the question", r.Outcome(), r.Err())
	}
	if c.n.Load() != 0 {
		t.Fatal("it ran before anyone decided")
	}
	if got := j.closes[0][0]; got.State != tool.StateInterrupted || got.Reason == "" {
		t.Errorf("the journal has %+v, want it still interrupted, waiting for the decision", got)
	}

	recorded := j.checkpoint().Calls
	r = a.Start(context.Background(), Decide(cp.Messages, d, map[llm.ToolUseID]bool{"d1": true}).Recorded(recorded))
	drain(t, r)
	if c.n.Load() != 1 {
		t.Errorf("allowed, it ran %d times, want once", c.n.Load())
	}

	r = a.Start(context.Background(), Decide(cp.Messages, d, map[llm.ToolUseID]bool{"d1": false}).Recorded(recorded))
	drain(t, r)
	if c.n.Load() != 1 {
		t.Errorf("refused, it ran anyway")
	}
	if msgs := r.Messages(); !strings.Contains(allText(msgs), "decided it should not run a second time") {
		t.Errorf("the refused call's answer: %q", allText(msgs))
	}
}

// A tool that can tell whether an interrupted call landed is asked first.
func TestReconcileSaysWhetherAnInterruptedCallLanded(t *testing.T) {
	for _, landed := range []bool{true, false} {
		t.Run(fmt.Sprint("landed=", landed), func(t *testing.T) {
			c := &counting{}
			def := c.def("edit", tool.CapMutating)
			def.Reconcile = func(context.Context, json.RawMessage) (string, bool, error) {
				return "edited (it had landed)", landed, nil
			}
			u := use("e1", "edit")
			cp := Checkpoint{
				Messages: []llm.Message{userMsg("edit"), {Role: llm.RoleAssistant, Content: []llm.ContentBlock{u}}},
				Calls:    []Call{{Index: 0, Use: u, State: tool.StateRunning}},
			}
			cl := scripted(textTurn("ok"))
			a := newAgent(t, cl, []tool.Def{def})
			r := a.Start(context.Background(), Recover(cp))
			_, evs := drain(t, r)
			wantRan, wantText, wantAction := int64(0), "edited (it had landed)", "reconciled"
			if !landed {
				wantRan, wantText, wantAction = 1, "edit done", "rerun"
			}
			if c.n.Load() != wantRan || !strings.Contains(allText(cl.request(t, 0).Messages), wantText) {
				t.Errorf("ran %d, the model read %q; want %d and %q", c.n.Load(), allText(cl.request(t, 0).Messages), wantRan, wantText)
			}
			if !hasRecovered(evs, "e1", wantAction) {
				t.Errorf("no tool_recovered %s", wantAction)
			}
		})
	}
}

// The bug the journal exists to fix: an approved call ran, and the run after
// it was cut short. Resumed with the same decisions, the call is answered from
// what it returned, not run a second time.
func TestAnApprovedCallThatRanIsNotRunAgain(t *testing.T) {
	w := &writes{}
	cl := scripted(toolsTurn(readCall, writeCall), textTurn("copied"))
	a, msgs, d := deferredRun(t, cl, w)

	// The first resume runs the write and is cut short before the model
	// reads it: the journal has it.
	j := &memJournal{}
	verdicts := map[llm.ToolUseID]bool{"w1": true}
	ctx, cancel := context.WithCancel(journaled(j))
	cut := newAgent(t, llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		cancel()
		return llm.Response{}, context.Canceled
	}), []tool.Def{echoTool("read", "the contents of a"), w.tool()}, WithToolMiddleware(deferWrites))
	drain(t, cut.Start(ctx, Decide(msgs, d, verdicts)))
	if w.n.Load() != 1 {
		t.Fatalf("the write ran %d times on the first resume", w.n.Load())
	}
	if j.states()["w1"] != tool.StateSucceeded {
		t.Fatalf("the journal has the write %q", j.states()["w1"])
	}

	r := a.Start(context.Background(), Decide(msgs, d, verdicts).Recorded(j.checkpoint().Calls))
	drain(t, r)
	if ans, ok := r.Outcome().(Answer); !ok || ans.Text != "copied" {
		t.Fatalf("outcome = %#v (%v)", r.Outcome(), r.Err())
	}
	if w.n.Load() != 1 {
		t.Errorf("the approved write ran %d times, want once", w.n.Load())
	}
}

// The answer is written down before it is checked: a run resumed after it
// does not ask for it again.
func TestRecoverDoesNotAskForAnAnswerItHas(t *testing.T) {
	cl := scripted()
	a := newAgent(t, cl, nil)
	r := a.Start(context.Background(), Recover(Checkpoint{
		Messages:   []llm.Message{userMsg("hi"), assistantText("hello")},
		Iterations: 1,
	}))
	drain(t, r)
	if ans, ok := r.Outcome().(Answer); !ok || ans.Text != "hello" {
		t.Fatalf("outcome = %#v (%v), want the answer it had", r.Outcome(), r.Err())
	}
	if cl.calls() != 0 {
		t.Errorf("the model was asked %d times", cl.calls())
	}
}

// The iteration cap counts the iterations of the run a resumed one carries on.
func TestRecoverCountsTheIterationsItCarriesOn(t *testing.T) {
	cl := scripted(textTurn("never"))
	a := newAgent(t, cl, []tool.Def{echoTool("read", "x")}, WithMaxIters(3))
	u := use("r1", "read")
	r := a.Start(context.Background(), Recover(Checkpoint{
		Messages:   []llm.Message{userMsg("go"), {Role: llm.RoleAssistant, Content: []llm.ContentBlock{u}}},
		Calls:      []Call{{Index: 0, Use: u, State: tool.StatePending}},
		Iterations: 3,
	}))
	drain(t, r)
	if !errors.Is(r.Err(), ErrMaxIterations) || cl.calls() != 0 {
		t.Errorf("err = %v after %d model calls, want the cap reached at once", r.Err(), cl.calls())
	}
}

// A run being stopped writes down that the call it stopped was interrupted.
func TestAStoppedCallIsWrittenDownAsInterrupted(t *testing.T) {
	j := &memJournal{}
	ctx, cancel := context.WithCancel(journaled(j))
	slow := tool.Def{Name: "build", Description: "builds", InputSchema: json.RawMessage(objSchema), Caps: tool.CapExec,
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			cancel()
			<-ctx.Done()
			return "", ctx.Err()
		}}
	a := newAgent(t, scripted(toolsTurn(use("b1", "build"))), []tool.Def{slow})
	drain(t, a.Start(ctx, Ask("build")))
	if got := j.states()["b1"]; got != tool.StateInterrupted {
		t.Errorf("the journal has the build %q, want interrupted", got)
	}
}

// A sub-agent does not write into its parent's journal.
func TestASubagentDoesNotInheritTheJournal(t *testing.T) {
	j := &memJournal{}
	child := newAgent(t, scripted(toolsTurn(use("c1", "read")), textTurn("found it")),
		[]tool.Def{echoTool("read", "x")}, WithName("child"))
	parent := newAgent(t, scripted(toolTurn("d1", "delegate", `{"task":"look"}`), textTurn("done")),
		[]tool.Def{DelegateTool(child)})
	if _, err := parent.Run(journaled(j), Ask("go")); err != nil {
		t.Fatal(err)
	}
	for _, s := range j.steps {
		for _, c := range s.Calls {
			if c.Use.ID == "c1" {
				t.Fatalf("the child's call is in the parent's journal: %+v", s)
			}
		}
	}
}

// A journal that names its calls hands the names to the tools.
func TestToolsSeeTheJournalsNameForTheirCall(t *testing.T) {
	c := &counting{}
	a := newAgent(t, scripted(toolsTurn(use("x1", "x")), textTurn("ok")), []tool.Def{c.def("x", tool.CapMutating)})
	if _, err := a.Run(journaled(keyedJournal{&memJournal{}}), Ask("go")); err != nil {
		t.Fatal(err)
	}
	if _, ok := c.keys.Load("call-x1"); !ok {
		t.Errorf("the tool saw keys %v, want the journal's", keysOf(&c.keys))
	}
	c2 := &counting{}
	a = newAgent(t, scripted(toolsTurn(use("x2", "x")), textTurn("ok")), []tool.Def{c2.def("x", tool.CapMutating)})
	if _, err := a.Run(context.Background(), Ask("go")); err != nil {
		t.Fatal(err)
	}
	if _, ok := c2.keys.Load("x2"); !ok {
		t.Errorf("without a journal the tool saw %v, want the tool_use id", keysOf(&c2.keys))
	}
}

func keysOf(m *sync.Map) []string {
	var out []string
	m.Range(func(k, _ any) bool { out = append(out, k.(string)); return true })
	return out
}

// A call that started something finishing elsewhere ends the run awaiting
// it, and the result, once posted, is what the model reads.
func TestAnAwaitedCallIsAnsweredWithItsPostedResult(t *testing.T) {
	var started atomic.Int64
	job := tool.Def{Name: "ci", Description: "starts a CI run", InputSchema: json.RawMessage(objSchema), Caps: tool.CapNetwork,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			started.Add(1)
			return "", tool.Await("run 42", "the CI run to finish")
		}}
	cl := scripted(toolsTurn(use("c1", "ci"), use("r1", "read")), textTurn("CI passed"))
	j := &memJournal{}
	a := newAgent(t, cl, []tool.Def{job, echoTool("read", "the log")})
	r := a.Start(journaled(j), Ask("run CI"))
	drain(t, r)
	d, ok := r.Outcome().(Deferred)
	if !ok || len(d.Awaited()) != 1 || d.Calls[0].Handle != "run 42" || d.Calls[0].Reason != "the CI run to finish" || len(d.Decisions()) != 0 {
		t.Fatalf("outcome = %#v (%v), want the CI call awaiting", r.Outcome(), r.Err())
	}
	if len(d.Done) != 1 || d.Done[0].Content != "the log" {
		t.Errorf("done = %+v, want the read kept", d.Done)
	}
	if got := j.closes[0][0]; got.State != tool.StateWaiting || got.Handle != "run 42" {
		t.Errorf("the journal has %+v, want it waiting", got)
	}

	msgs := r.Messages()
	// Resumed with nothing posted: still waiting.
	r = a.Start(context.Background(), Decide(msgs, d, nil))
	drain(t, r)
	if again, ok := r.Outcome().(Deferred); !ok || len(again.Awaited()) != 1 {
		t.Fatalf("resumed without a result: %#v (%v), want still waiting", r.Outcome(), r.Err())
	}

	r = a.Start(context.Background(), Decide(msgs, d, nil).Completed(map[llm.ToolUseID]llm.ToolResult{
		"c1": {Content: "all 312 tests passed"},
	}))
	drain(t, r)
	if ans, ok := r.Outcome().(Answer); !ok || ans.Text != "CI passed" {
		t.Fatalf("outcome = %#v (%v)", r.Outcome(), r.Err())
	}
	if started.Load() != 1 {
		t.Errorf("the CI run was started %d times, want once", started.Load())
	}
	sent := cl.request(t, 1).Messages
	results := sent[len(sent)-1].Content
	if res := results[0].(llm.ToolResult); res.Content != "all 312 tests passed" || res.IsError {
		t.Errorf("the awaited call's result = %+v, want the posted one", res)
	}
}

// nestJournal journals sub-agents: one memJournal per call and child, and a
// checkpoint for each once it has written something.
type nestJournal struct {
	*memJournal
	mu       sync.Mutex
	children map[string]*memJournal
}

func (j *nestJournal) Nest(_ context.Context, key string, n int) (Journal, *Checkpoint, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.children == nil {
		j.children = map[string]*memJournal{}
	}
	name := fmt.Sprintf("%s#%d", key, n)
	if child, ok := j.children[name]; ok && len(child.steps) > 0 {
		cp := child.checkpoint()
		return child, &cp, nil
	}
	child := &memJournal{}
	j.children[name] = child
	return child, nil, nil
}

// A delegate call running again after its parent was resumed carries its
// sub-agent on from where it got to.
func TestADelegateCallCarriesItsSubagentOn(t *testing.T) {
	j := &nestJournal{memJournal: &memJournal{}}
	// The child reads, and the run is cut short while it waits for its
	// second model call.
	reads := &counting{}
	ctx, cancel := context.WithCancel(journaled(j))
	cutChild := newAgent(t, llm.ClientFunc(func() func(context.Context, llm.Request) (llm.Response, error) {
		n := 0
		return func(ctx context.Context, _ llm.Request) (llm.Response, error) {
			if n++; n == 1 {
				return toolsTurn(use("c1", "read")), nil
			}
			cancel()
			return llm.Response{}, context.Canceled
		}
	}()), []tool.Def{reads.def("read", tool.CapReadOnly)}, WithName("child"))
	parent := newAgent(t, scripted(toolTurn("d1", "delegate", `{"task":"look"}`)), []tool.Def{DelegateTool(cutChild)})
	drain(t, parent.Start(ctx, Ask("go")))
	if reads.n.Load() != 1 {
		t.Fatalf("the child read %d times before the cut", reads.n.Load())
	}
	if got := j.states()["d1"]; got != tool.StateInterrupted && got != tool.StateRunning {
		t.Fatalf("the parent's journal has the delegate call %q", got)
	}

	// Resumed: the delegate call runs again, and the child picks up after its
	// read — it is not asked for it again, and does not read twice.
	childCl := scripted(textTurn("found it"))
	child := newAgent(t, childCl, []tool.Def{reads.def("read", tool.CapReadOnly)}, WithName("child"))
	parentCl := scripted(textTurn("the child found it"))
	parent = newAgent(t, parentCl, []tool.Def{DelegateTool(child)})
	r := parent.Start(journaled(j), Recover(j.checkpoint()))
	_, evs := drain(t, r)
	if ans, ok := r.Outcome().(Answer); !ok || ans.Text != "the child found it" {
		t.Fatalf("outcome = %#v (%v)", r.Outcome(), r.Err())
	}
	if reads.n.Load() != 1 {
		t.Errorf("the child read %d times, want once", reads.n.Load())
	}
	if childCl.calls() != 1 {
		t.Errorf("the child's model was asked %d times, want once, after its read", childCl.calls())
	}
	resumed := false
	for _, e := range evs {
		if s, ok := e.(SubagentStart); ok && s.Resumed {
			resumed = true
		}
	}
	if !resumed {
		t.Error("no subagent_start said the child was resumed")
	}
}

// Without a journal that nests, an interrupted delegate call is reported:
// running it again would start its child over.
func TestAnInterruptedDelegateCallWithoutNestingIsReported(t *testing.T) {
	childCl := scripted(textTurn("never"))
	child := newAgent(t, childCl, nil)
	u := llm.ToolUse{ID: "d1", Name: "delegate", Input: json.RawMessage(`{"task":"look"}`)}
	cl := scripted(textTurn("ok"))
	parent := newAgent(t, cl, []tool.Def{DelegateTool(child)})
	r := parent.Start(context.Background(), Recover(Checkpoint{
		Messages: []llm.Message{userMsg("go"), {Role: llm.RoleAssistant, Content: []llm.ContentBlock{u}}},
		Calls:    []Call{{Index: 0, Use: u, State: tool.StateRunning}},
	}))
	drain(t, r)
	if childCl.calls() != 0 || !strings.Contains(allText(cl.request(t, 0).Messages), "may or may not have taken effect") {
		t.Errorf("child ran %d times; the model read %q", childCl.calls(), allText(cl.request(t, 0).Messages))
	}
}

// A pause ends the run with the pause call done and its siblings cancelled,
// in the words AnswerPause closes them with.
func TestAPauseClosesItsStep(t *testing.T) {
	j := &memJournal{}
	cl := scripted(toolsTurn(use("r1", "read"), use("p1", "ask")))
	a := newAgent(t, cl, []tool.Def{echoTool("read", "x"), pauseTool("ask")})
	r := a.Start(journaled(j), Ask("go"))
	drain(t, r)
	if _, ok := r.Outcome().(Paused); !ok {
		t.Fatalf("outcome = %#v (%v)", r.Outcome(), r.Err())
	}
	got := j.closes[0]
	if got[0].State != tool.StateCancelled || !strings.Contains(got[0].Result.Content, "stopped to ask the user") {
		t.Errorf("the sibling = %+v, want it cancelled", got[0])
	}
	if got[1].State != tool.StateSucceeded {
		t.Errorf("the pause call = %+v, want it done", got[1])
	}
}
