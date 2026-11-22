package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// past is a time every slice is already over by: each slice does its one
// unit of work and stops.
var past = time.Unix(1, 0)

// slicedJournal is a journal that a run stopped at the end of its slice
// writes down where it got to in, and that journals sub-agents the same way.
type slicedJournal struct {
	*memJournal
	mu        sync.Mutex
	suspended *Checkpoint
	children  map[string]*slicedJournal
	suspends  int
}

func newSlicedJournal() *slicedJournal { return &slicedJournal{memJournal: &memJournal{}} }

func (j *slicedJournal) Open(ctx context.Context, s Step) error {
	j.mu.Lock()
	j.suspended = nil
	j.mu.Unlock()
	return j.memJournal.Open(ctx, s)
}

func (j *slicedJournal) Suspend(_ context.Context, cp Checkpoint) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.suspended = &cp
	j.suspends++
	return nil
}

// resumeFrom is where a run resumed from the journal starts: where it
// stopped, or its last step.
func (j *slicedJournal) resumeFrom() *Checkpoint {
	j.mu.Lock()
	s := j.suspended
	j.mu.Unlock()
	if s != nil {
		return s
	}
	if len(j.memJournal.steps) == 0 {
		return nil
	}
	cp := j.memJournal.checkpoint()
	cp.State = j.memJournal.steps[len(j.memJournal.steps)-1].State
	return &cp
}

func (j *slicedJournal) Nest(_ context.Context, key string, n int) (Journal, *Checkpoint, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.children == nil {
		j.children = map[string]*slicedJournal{}
	}
	name := fmt.Sprintf("%s#%d", key, n)
	if child, ok := j.children[name]; ok {
		return child, child.resumeFrom(), nil
	}
	child := newSlicedJournal()
	j.children[name] = child
	return child, nil, nil
}

// sliceRun is how a run carried on in slices ended.
type sliceRun struct {
	outcome  Outcome
	err      error
	slices   int
	messages []llm.Message
}

// runSliced runs in to its end in slices, each as short as a slice can be,
// carrying the budget on as a host would; with j, every slice writes into it.
func runSliced(t *testing.T, a *Agent, in Input, limits governor.Limits, j Journal) sliceRun {
	t.Helper()
	var used governor.Used
	for n := 1; n <= 200; n++ {
		ctx := context.Background()
		if j != nil {
			ctx = WithJournal(ctx, j)
		}
		ctx, cancel := governor.WithBudgetFrom(ctx, limits, used)
		r := a.Start(ctx, in.Until(past))
		drain(t, r)
		cancel()
		if s, ok := r.Outcome().(Suspended); ok {
			used = s.Checkpoint.Used
			in = Recover(s.Checkpoint)
			continue
		}
		return sliceRun{outcome: r.Outcome(), err: r.Err(), slices: n, messages: r.Messages()}
	}
	t.Fatal("the run never ended")
	return sliceRun{}
}

// runWhole runs in to its end in one go.
func runWhole(t *testing.T, a *Agent, in Input, limits governor.Limits) sliceRun {
	t.Helper()
	ctx, cancel := governor.WithBudget(context.Background(), limits)
	defer cancel()
	r := a.Start(ctx, in)
	drain(t, r)
	return sliceRun{outcome: r.Outcome(), err: r.Err(), slices: 1, messages: r.Messages()}
}

func sentMessages(cl *scriptedClient) [][]llm.Message {
	var out [][]llm.Message
	for _, req := range cl.requests() {
		out = append(out, req.Messages)
	}
	return out
}

// sameRun asserts that a run carried on in slices did what the run in one
// go did: the same requests to the model, the same transcript, the same end.
func sameRun(t *testing.T, whole, sliced sliceRun, wholeCl, slicedCl *scriptedClient) {
	t.Helper()
	if !reflect.DeepEqual(sentMessages(wholeCl), sentMessages(slicedCl)) {
		t.Errorf("the model was asked different things\nwhole  %s\nsliced %s", jsonOf(sentMessages(wholeCl)), jsonOf(sentMessages(slicedCl)))
	}
	if !reflect.DeepEqual(whole.messages, sliced.messages) {
		t.Errorf("transcripts differ\nwhole  %s\nsliced %s", jsonOf(whole.messages), jsonOf(sliced.messages))
	}
	if !reflect.DeepEqual(whole.outcome, sliced.outcome) || errText(whole.err) != errText(sliced.err) {
		t.Errorf("ended differently: whole %#v (%v), sliced %#v (%v)", whole.outcome, whole.err, sliced.outcome, sliced.err)
	}
}

func jsonOf(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// A run carried on in slices ends exactly as the run in one go: the model is
// asked the same things, the tools run as often, the transcript is the same.
func TestARunInSlicesIsTheRunInOneGo(t *testing.T) {
	script := func() *scriptedClient {
		return scripted(
			toolsTurn(use("r1", "read")),
			toolsTurn(use("w1", "write"), use("r2", "read")),
			textTurn("done"),
		)
	}
	agent := func(cl llm.Client, reads, writes *counting) *Agent {
		return newAgent(t, cl, []tool.Def{reads.def("read", tool.CapReadOnly), writes.def("write", tool.CapMutating)})
	}
	wr, ww, sr, sw := &counting{}, &counting{}, &counting{}, &counting{}
	wholeCl, slicedCl := script(), script()
	whole := runWhole(t, agent(wholeCl, wr, ww), Ask("go"), governor.Limits{})
	sliced := runSliced(t, agent(slicedCl, sr, sw), Ask("go"), governor.Limits{}, nil)
	sameRun(t, whole, sliced, wholeCl, slicedCl)
	if sr.n.Load() != 2 || sw.n.Load() != 1 {
		t.Errorf("in slices, read ran %d times and write %d; want 2 and 1", sr.n.Load(), sw.n.Load())
	}
	// Three model calls and two batches, a unit to a slice.
	if sliced.slices != 5 {
		t.Errorf("%d slices, want 5: one unit each", sliced.slices)
	}
}

// What the loop keeps while it goes is carried across slices: a reply cut
// off at max_tokens is asked for again with the notice, even when a slice
// ends in between.
func TestATruncationNoticeIsCarried(t *testing.T) {
	script := func() *scriptedClient {
		return scripted(
			llm.Response{Content: []llm.ContentBlock{llm.Text{Text: "a very long"}}, StopReason: llm.StopMaxTokens},
			textTurn("short"),
		)
	}
	wholeCl, slicedCl := script(), script()
	whole := runWhole(t, newAgent(t, wholeCl, nil), Ask("go"), governor.Limits{})
	sliced := runSliced(t, newAgent(t, slicedCl, nil), Ask("go"), governor.Limits{}, nil)
	sameRun(t, whole, sliced, wholeCl, slicedCl)
	if got := allText(slicedCl.request(t, 1).Messages); !strings.Contains(got, "cut off") {
		t.Errorf("the second request had no notice: %s", got)
	}
}

// The guards that bound a run bound it in slices too: a model that keeps
// answering nothing fails after the same number of tries, not never.
func TestTheEmptyTurnGuardHoldsAcrossSlices(t *testing.T) {
	empty := llm.Response{StopReason: llm.StopEndTurn}
	script := func() *scriptedClient { return scripted(empty, empty, empty, empty, empty, textTurn("late")) }
	wholeCl, slicedCl := script(), script()
	whole := runWhole(t, newAgent(t, wholeCl, nil), Ask("go"), governor.Limits{})
	sliced := runSliced(t, newAgent(t, slicedCl, nil), Ask("go"), governor.Limits{}, nil)
	var e *EmptyTurnError
	if !errors.As(sliced.err, &e) || slicedCl.calls() != wholeCl.calls() {
		t.Fatalf("in slices: %v after %d calls; in one go: %v after %d", sliced.err, slicedCl.calls(), whole.err, wholeCl.calls())
	}
}

// A completion check that sends the model back gives up after the same
// number of tries in slices.
func TestCompletionChecksAreCountedAcrossSlices(t *testing.T) {
	never := func(context.Context, Completion) (string, error) { return "not yet", nil }
	script := func() *scriptedClient {
		return scripted(textTurn("one"), textTurn("two"), textTurn("three"), textTurn("four"))
	}
	wholeCl, slicedCl := script(), script()
	whole := runWhole(t, newAgent(t, wholeCl, nil, WithCompletionCheck(2, never)), Ask("go"), governor.Limits{})
	sliced := runSliced(t, newAgent(t, slicedCl, nil, WithCompletionCheck(2, never)), Ask("go"), governor.Limits{}, nil)
	sameRun(t, whole, sliced, wholeCl, slicedCl)
	if ans, ok := sliced.outcome.(Answer); !ok || ans.Text != "three" {
		t.Errorf("in slices the run ended %#v, want the third answer once the check gave up", sliced.outcome)
	}
}

// Loop detection, carried with the budget, trips at the same call. The
// model reuses its tool_use id, as some gateways do: a resumed run does not
// refuse the transcript it wrote itself for it.
func TestLoopDetectionHoldsAcrossSlices(t *testing.T) {
	same := toolTurn("x", "read", `{"q":1}`)
	script := func() *scriptedClient { return scripted(same, same, same, same, same, textTurn("never")) }
	limits := governor.Limits{RepeatedToolCalls: 3}
	wr, sr := &counting{}, &counting{}
	wholeCl, slicedCl := script(), script()
	whole := runWhole(t, newAgent(t, wholeCl, []tool.Def{wr.def("read", tool.CapReadOnly)}), Ask("go"), limits)
	sliced := runSliced(t, newAgent(t, slicedCl, []tool.Def{sr.def("read", tool.CapReadOnly)}), Ask("go"), limits, nil)
	if !errors.Is(whole.err, governor.ErrToolLoop) || !errors.Is(sliced.err, governor.ErrToolLoop) {
		t.Fatalf("whole: %v; sliced: %v; want both stopped as a loop", whole.err, sliced.err)
	}
	if sr.n.Load() != wr.n.Load() {
		t.Errorf("in slices the call ran %d times, in one go %d", sr.n.Load(), wr.n.Load())
	}
}

// counter is per-run state a decorator installs and a tool bumps.
type counter struct{ n atomic.Int64 }

func (c *counter) MarshalState() (json.RawMessage, error) { return json.Marshal(c.n.Load()) }
func (c *counter) UnmarshalState(b json.RawMessage) error {
	var n int64
	if err := json.Unmarshal(b, &n); err != nil {
		return err
	}
	c.n.Store(n)
	return nil
}

type counterKey struct{}

// What a host registers with Carry goes from slice to slice.
func TestCarriedStateGoesFromSliceToSlice(t *testing.T) {
	var last atomic.Int64
	bump := tool.Def{Name: "bump", Description: "bump", InputSchema: json.RawMessage(objSchema), Caps: tool.CapReadOnly,
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			c := ctx.Value(counterKey{}).(*counter)
			last.Store(c.n.Add(1))
			return "bumped", nil
		}}
	cl := scripted(toolTurn("b1", "bump", `{}`), toolTurn("b2", "bump", `{}`), toolTurn("b3", "bump", `{}`), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{bump}, WithRunContext(func(ctx context.Context) context.Context {
		c := &counter{}
		return Carry(context.WithValue(ctx, counterKey{}, c), "test.counter", c)
	}))
	runSliced(t, a, Ask("go"), governor.Limits{}, nil)
	if last.Load() != 3 {
		t.Errorf("the counter reached %d, want 3: carried from slice to slice", last.Load())
	}
}

// A slice stops before a batch it may not finish: the calls are written
// down, pending, and run as the next slice's first unit.
func TestASliceStopsBeforeABatch(t *testing.T) {
	writes := &counting{}
	cl := scripted(toolsTurn(use("w1", "write")), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{writes.def("write", tool.CapMutating)})
	j := newSlicedJournal()
	r := a.Start(journaled(j), Ask("go").Until(past))
	drain(t, r)
	s, ok := r.Outcome().(Suspended)
	if !ok || writes.n.Load() != 0 {
		t.Fatalf("outcome %#v, %d writes; want it stopped before the write", r.Outcome(), writes.n.Load())
	}
	if len(s.Checkpoint.Calls) != 1 || s.Checkpoint.Calls[0].State != tool.StatePending || s.Checkpoint.Iterations != 1 || len(s.Checkpoint.State) == 0 {
		t.Fatalf("checkpoint = %+v", s.Checkpoint)
	}
	if j.suspends != 1 || !reflect.DeepEqual(*j.suspended, s.Checkpoint) {
		t.Errorf("the journal was not given the checkpoint: %d suspends", j.suspends)
	}
	next := a.Start(journaled(j), Recover(s.Checkpoint))
	drain(t, next)
	if ans, ok := next.Outcome().(Answer); !ok || ans.Text != "done" || writes.n.Load() != 1 {
		t.Fatalf("carried on to %#v (%v) with %d writes", next.Outcome(), next.Err(), writes.n.Load())
	}
}

// A call with no time bound is only started at the start of a slice, however
// much time is left.
func TestAnUnboundedBatchStartsASlice(t *testing.T) {
	slow := tool.Def{Name: "slow", Description: "slow", InputSchema: json.RawMessage(objSchema), Caps: tool.CapReadOnly,
		Fn: func(context.Context, json.RawMessage) (string, error) { return "slow done", nil }}
	cl := scripted(toolTurn("s1", "slow", `{}`), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{slow}, WithToolTimeoutFallback(0))
	r := a.Start(context.Background(), Ask("go").Until(time.Now().Add(time.Hour)))
	drain(t, r)
	if _, ok := r.Outcome().(Suspended); !ok {
		t.Fatalf("outcome %#v; want it stopped before the unbounded call", r.Outcome())
	}
	// With a slice to itself, it runs.
	r = a.Start(context.Background(), Recover(r.Outcome().(Suspended).Checkpoint).Until(time.Now().Add(time.Hour)))
	drain(t, r)
	if ans, ok := r.Outcome().(Answer); !ok || ans.Text != "done" {
		t.Fatalf("outcome %#v (%v)", r.Outcome(), r.Err())
	}
}

// With time enough, a run with a time to stop by does not stop.
func TestASliceWithTimeEnoughDoesNotStop(t *testing.T) {
	reads := &counting{}
	cl := scripted(toolsTurn(use("r1", "read")), textTurn("done"))
	a := newAgent(t, cl, []tool.Def{reads.def("read", tool.CapReadOnly)})
	r := a.Start(context.Background(), Ask("go").Until(time.Now().Add(time.Hour)))
	drain(t, r)
	if ans, ok := r.Outcome().(Answer); !ok || ans.Text != "done" {
		t.Fatalf("outcome %#v (%v)", r.Outcome(), r.Err())
	}
}

// Messages queued for a run carried on, and messages from outside the
// process, reach the model at its next call.
func TestSteeringFromOutsideTheProcess(t *testing.T) {
	cl := scripted(toolTurn("r1", "read", `{}`), textTurn("done"))
	reads := &counting{}
	a := newAgent(t, cl, []tool.Def{reads.def("read", tool.CapReadOnly)})
	asked := 0
	src := func(context.Context) []string {
		asked++
		if asked == 2 {
			return []string{"use the v2 api"}
		}
		return nil
	}
	r := a.Start(context.Background(), Ask("go").Steer("be brief").SteerFrom(src))
	drain(t, r)
	if got := allText(cl.request(t, 0).Messages); !strings.Contains(got, "be brief") {
		t.Errorf("the queued message did not reach the first call: %s", got)
	}
	if got := allText(cl.request(t, 1).Messages); !strings.Contains(got, "use the v2 api") {
		t.Errorf("the message from outside did not reach the second call: %s", got)
	}
}

// A delegate call's sub-agent stops with the slice, and the call carries on
// in the next: the sub-agent does each of its steps once, and the run ends
// as it would have in one go.
func TestASubagentStopsWithTheSlice(t *testing.T) {
	childScript := func() *scriptedClient {
		return scripted(toolsTurn(use("c1", "read")), toolsTurn(use("c2", "read")), textTurn("the child's answer"))
	}
	parentScript := func() *scriptedClient {
		return scripted(toolTurn("d1", "delegate", `{"task":"look twice"}`), textTurn("the parent's answer"))
	}
	build := func(parentCl, childCl llm.Client, reads *counting) *Agent {
		child := newAgent(t, childCl, []tool.Def{reads.def("read", tool.CapReadOnly)}, WithName("child"))
		return newAgent(t, parentCl, []tool.Def{DelegateTool(child)})
	}
	wReads, sReads := &counting{}, &counting{}
	wParent, wChild, sParent, sChild := parentScript(), childScript(), parentScript(), childScript()
	whole := runWhole(t, build(wParent, wChild, wReads), Ask("go"), governor.Limits{})
	j := newSlicedJournal()
	sliced := runSliced(t, build(sParent, sChild, sReads), Ask("go"), governor.Limits{}, j)
	sameRun(t, whole, sliced, wParent, sParent)
	if !reflect.DeepEqual(sentMessages(wChild), sentMessages(sChild)) {
		t.Errorf("the sub-agent was asked different things\nwhole  %s\nsliced %s", jsonOf(sentMessages(wChild)), jsonOf(sentMessages(sChild)))
	}
	if sReads.n.Load() != 2 {
		t.Errorf("the sub-agent read %d times, want twice", sReads.n.Load())
	}
	if sliced.slices < 4 {
		t.Errorf("%d slices: the sub-agent did not stop with them", sliced.slices)
	}
}

// Parallel sub-agents stop with the slice too: a branch that finished
// answers from where its journal wrote it down, the others carry on.
func TestParallelSubagentsStopWithTheSlice(t *testing.T) {
	branch := func(name string) llm.Client {
		return llm.ClientFunc(func() func(context.Context, llm.Request) (llm.Response, error) {
			var mu sync.Mutex
			calls := map[string]int{}
			return func(_ context.Context, req llm.Request) (llm.Response, error) {
				task := allText(req.Messages[:1])
				mu.Lock()
				defer mu.Unlock()
				calls[task]++
				switch {
				case strings.Contains(task, "quick"):
					return textTurn(name + ": quick answer"), nil
				case calls[task] == 1:
					return toolsTurn(use("c-"+task[:5], "read")), nil
				default:
					return textTurn(name + ": slow answer"), nil
				}
			}
		}())
	}
	reads := &counting{}
	child := newAgent(t, branch("child"), []tool.Def{reads.def("read", tool.CapReadOnly)}, WithName("child"))
	cl := scripted(toolTurn("p1", "parallel_delegate", `{"tasks":["quick one","slow one"]}`), textTurn("both in"))
	a := newAgent(t, cl, []tool.Def{ParallelDelegateTool(child)})
	sliced := runSliced(t, a, Ask("go"), governor.Limits{}, newSlicedJournal())
	if ans, ok := sliced.outcome.(Answer); !ok || ans.Text != "both in" {
		t.Fatalf("ended %#v (%v)", sliced.outcome, sliced.err)
	}
	result := allText(cl.request(t, 1).Messages)
	if !strings.Contains(result, "child: quick answer") || !strings.Contains(result, "child: slow answer") {
		t.Errorf("the parent read %s", result)
	}
	if reads.n.Load() != 1 {
		t.Errorf("the slow branch read %d times, want once", reads.n.Load())
	}
}

// A sub-agent whose run cannot write down where it got to runs to its end
// within the call: nothing would carry it on.
func TestASubagentWithoutASuspenderDoesNotStop(t *testing.T) {
	reads := &counting{}
	child := newAgent(t, scripted(toolsTurn(use("c1", "read")), textTurn("child done")),
		[]tool.Def{reads.def("read", tool.CapReadOnly)}, WithName("child"))
	cl := scripted(toolTurn("d1", "delegate", `{"task":"look"}`), textTurn("parent done"))
	a := newAgent(t, cl, []tool.Def{DelegateTool(child)})
	sliced := runSliced(t, a, Ask("go"), governor.Limits{}, &nestJournal{memJournal: &memJournal{}})
	if ans, ok := sliced.outcome.(Answer); !ok || ans.Text != "parent done" || reads.n.Load() != 1 {
		t.Fatalf("ended %#v (%v), %d reads", sliced.outcome, sliced.err, reads.n.Load())
	}
}
