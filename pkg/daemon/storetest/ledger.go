package storetest

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
	"wombat/pkg/daemon"
)

// ===== The ledger =====

func call(idx int, id, name, input string) daemon.ToolCall {
	return daemon.ToolCall{Index: idx, UseID: llm.ToolUseID(id), Tool: name, Input: json.RawMessage(input)}
}

func openStep(t *testing.T, st daemon.Store, o daemon.StepOpen) int {
	t.Helper()
	if o.Owner == "" {
		o.Owner = owner
	}
	step, err := st.OpenStep(ctx, o)
	if err != nil {
		t.Fatalf("OpenStep(%s): %v", o.Session, err)
	}
	return step
}

func move(t *testing.T, st daemon.Store, session string, step int, at time.Time, moves ...daemon.CallMove) {
	t.Helper()
	if err := st.MoveCalls(ctx, daemon.CallMoves{Session: session, Owner: owner, Step: step, Moves: moves, At: at}); err != nil {
		t.Fatalf("MoveCalls(%s, step %d): %v", session, step, err)
	}
}

func toolCalls(t *testing.T, st daemon.Store, f daemon.CallFilter) []daemon.ToolCall {
	t.Helper()
	got, err := st.ToolCalls(ctx, f)
	if err != nil {
		t.Fatalf("ToolCalls(%+v): %v", f, err)
	}
	return got
}

func keys(calls []daemon.ToolCall) []string {
	out := make([]string, len(calls))
	for i, c := range calls {
		out[i] = c.Key()
	}
	return out
}

// A step's calls are written down pending, numbered by the session's steps,
// and come back as they went in: a tool's input byte for byte.
func ledgerWritesStepsDown(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	create(t, st, record("s2"))
	if _, err := st.OpenStep(ctx, daemon.StepOpen{Session: "nope", Owner: owner}); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("OpenStep for an unknown session = %v, want ErrNoSuchSession", err)
	}

	first := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, At: t0, Calls: []daemon.ToolCall{
		call(0, "u1", "read", `{"z": 1, "a": [2]}`),
		// Whatever comes with a call it is handed is not what came of it.
		{Index: 1, UseID: "u2", Tool: "write", Input: json.RawMessage(`{"path":"b"}`), State: tool.StateSucceeded, Output: "early", Attempts: 3},
	}})
	answer := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, At: t0.Add(time.Second)})
	other := openStep(t, st, daemon.StepOpen{Session: "s2", Turn: 4, At: t0, Calls: []daemon.ToolCall{call(0, "u1", "read", `{}`)}})
	if first != 1 || answer != 2 || other != 1 {
		t.Fatalf("steps numbered %d, %d and %d; want 1 and 2 for s1, 1 for s2", first, answer, other)
	}

	got := toolCalls(t, st, daemon.CallFilter{Session: "s1"})
	want := []daemon.ToolCall{
		{Session: "s1", Step: 1, Index: 1, Turn: 1, UseID: "u2", Tool: "write", Input: json.RawMessage(`{"path":"b"}`),
			State: tool.StatePending, Owner: owner, Created: t0},
		{Session: "s1", Step: 1, Index: 0, Turn: 1, UseID: "u1", Tool: "read", Input: json.RawMessage(`{"z": 1, "a": [2]}`),
			State: tool.StatePending, Owner: owner, Created: t0},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("s1's calls\ngot  %#v\nwant %#v", got, want)
	}
	if got[0].Key() != "s1/1.1" {
		t.Errorf("Key = %q", got[0].Key())
	}
}

// A call moves through its states, and what came of it is kept; an ended
// one moves only to another ending, and a batch with a move that cannot be
// made makes none.
func ledgerMovesCalls(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	step := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, At: t0, Calls: []daemon.ToolCall{
		call(0, "u1", "read", `{}`), call(1, "u2", "write", `{}`), call(2, "u3", "fetch", `{}`),
	}})
	by := daemon.Origin{Event: 9, Kind: daemon.KindApproval, Source: daemon.SourceAPI, From: "carol", At: t0}
	media := []llm.Media{{MediaType: "image/png", Data: []byte{0x89, 'P', 'N', 'G'}, Name: "shot"}}

	move(t, st, "s1", step, t0.Add(time.Second),
		daemon.CallMove{Index: 0, State: tool.StateRunning, Attempt: true},
		daemon.CallMove{Index: 2, State: tool.StateRunning, Attempt: true})
	move(t, st, "s1", step, t0.Add(2*time.Second),
		daemon.CallMove{Index: 0, State: tool.StateSucceeded, Output: "contents", Media: media, Tags: []string{"cached", "big"}},
		daemon.CallMove{Index: 1, State: tool.StateDenied, Output: "not allowed", IsError: true, Reason: "outside the workspace",
			Decision: &daemon.Decision{Allow: false, By: by}},
		daemon.CallMove{Index: 2, State: tool.StateInterrupted})
	// Carried on: it runs again, and ends.
	move(t, st, "s1", step, t0.Add(3*time.Second), daemon.CallMove{Index: 2, State: tool.StateRunning, Attempt: true})
	move(t, st, "s1", step, t0.Add(4*time.Second), daemon.CallMove{Index: 2, State: tool.StateFailed, Output: "timeout", IsError: true})

	got := toolCalls(t, st, daemon.CallFilter{Session: "s1", Step: step})
	want := []daemon.ToolCall{
		{Session: "s1", Step: 1, Index: 2, Turn: 1, UseID: "u3", Tool: "fetch", Input: json.RawMessage(`{}`),
			State: tool.StateFailed, Attempts: 2, Output: "timeout", IsError: true, Owner: owner,
			Created: t0, Started: t0.Add(3 * time.Second), Finished: t0.Add(4 * time.Second)},
		{Session: "s1", Step: 1, Index: 1, Turn: 1, UseID: "u2", Tool: "write", Input: json.RawMessage(`{}`),
			State: tool.StateDenied, Output: "not allowed", IsError: true, Reason: "outside the workspace",
			Decision: &daemon.Decision{Allow: false, By: by}, Owner: owner, Created: t0, Finished: t0.Add(2 * time.Second)},
		{Session: "s1", Step: 1, Index: 0, Turn: 1, UseID: "u1", Tool: "read", Input: json.RawMessage(`{}`),
			State: tool.StateSucceeded, Attempts: 1, Output: "contents", Media: media, Tags: []string{"cached", "big"},
			Owner: owner, Created: t0, Started: t0.Add(time.Second), Finished: t0.Add(2 * time.Second)},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("after the moves\ngot  %#v\nwant %#v", got, want)
	}

	// None of a batch with a move that cannot be made is made.
	for _, bad := range [][]daemon.CallMove{
		{{Index: 1, State: tool.StateCancelled}, {Index: 0, State: tool.StateRunning, Attempt: true}}, // ended
		{{Index: 1, State: tool.StateCancelled}, {Index: 7, State: tool.StateFailed}},                 // not there
		{{Index: 1, State: tool.StateCancelled}, {Index: 0, State: "lost"}},                           // no such state
	} {
		if err := st.MoveCalls(ctx, daemon.CallMoves{Session: "s1", Owner: owner, Step: step, Moves: bad, At: t0.Add(time.Hour)}); err == nil {
			t.Errorf("MoveCalls(%+v) made a move that cannot be made", bad)
		}
	}
	if again := toolCalls(t, st, daemon.CallFilter{Session: "s1", Step: step}); !reflect.DeepEqual(again, want) {
		t.Fatalf("a refused batch changed the ledger\ngot  %#v\nwant %#v", again, want)
	}

	// An ending moves to another — the more exact one a step closes with —
	// and keeps when it ended. The decision stays unless another is made.
	move(t, st, "s1", step, t0.Add(time.Hour), daemon.CallMove{Index: 1, State: tool.StateCancelled, Reason: "the turn stopped"})
	if c := toolCalls(t, st, daemon.CallFilter{Session: "s1", Step: step})[1]; c.State != tool.StateCancelled ||
		!c.Finished.Equal(t0.Add(2*time.Second)) || c.Decision == nil || c.Reason != "the turn stopped" || c.Output != "" {
		t.Errorf("re-ended: %+v", c)
	}
	if err := st.MoveCalls(ctx, daemon.CallMoves{Session: "s1", Owner: owner, Step: step + 1,
		Moves: []daemon.CallMove{{Index: 0, State: tool.StateFailed}}}); err == nil {
		t.Error("MoveCalls on a step that is not there succeeded")
	}
}

// Every state the harness has is one the store keeps.
func ledgerKeepsEveryState(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	for i, state := range tool.States() {
		step := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, At: t0.Add(time.Duration(i) * time.Second),
			Calls: []daemon.ToolCall{call(0, "u", "x", `{}`)}})
		move(t, st, "s1", step, t0, daemon.CallMove{Index: 0, State: state, Handle: "h"})
		if got := toolCalls(t, st, daemon.CallFilter{Session: "s1", Step: step}); len(got) != 1 || got[0].State != state {
			t.Errorf("moved to %s, the call is %+v", state, got)
		}
	}
}

func ledgerFilters(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	create(t, st, record("s2"))
	a := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, At: t0, Calls: []daemon.ToolCall{
		call(0, "u1", "delegate", `{}`), call(1, "u2", "read", `{}`),
	}})
	// The delegate's sub-agent takes a step of its own.
	sub := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, Parent: "s1/1.0", Run: 1, At: t0.Add(time.Second),
		Calls: []daemon.ToolCall{call(0, "c1", "grep", `{}`)}})
	b := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 2, At: t0.Add(2 * time.Second), Calls: []daemon.ToolCall{call(0, "u1", "read", `{}`)}})
	openStep(t, st, daemon.StepOpen{Session: "s2", Turn: 1, At: t0.Add(3 * time.Second), Calls: []daemon.ToolCall{call(0, "u1", "read", `{}`)}})
	move(t, st, "s1", a, t0, daemon.CallMove{Index: 1, State: tool.StateSucceeded})
	move(t, st, "s1", b, t0, daemon.CallMove{Index: 0, State: tool.StateWaiting, Handle: "job 7"})

	for _, tc := range []struct {
		f    daemon.CallFilter
		want []string
	}{
		{daemon.CallFilter{}, []string{"s2/1.0", "s1/3.0", "s1/2.0", "s1/1.1", "s1/1.0"}},
		{daemon.CallFilter{Limit: 2}, []string{"s2/1.0", "s1/3.0"}},
		{daemon.CallFilter{Session: "s1", Turn: 1}, []string{"s1/2.0", "s1/1.1", "s1/1.0"}},
		{daemon.CallFilter{Session: "s1", Step: sub}, []string{"s1/2.0"}},
		{daemon.CallFilter{Session: "s1", Parent: "s1/1.0"}, []string{"s1/2.0"}},
		{daemon.CallFilter{Session: "s1", Parent: "-"}, []string{"s1/3.0", "s1/1.1", "s1/1.0"}},
		{daemon.CallFilter{States: []tool.State{tool.StateSucceeded, tool.StateWaiting}}, []string{"s1/3.0", "s1/1.1"}},
		{daemon.CallFilter{Session: "s2", States: []tool.State{tool.StateWaiting}}, []string{}},
		{daemon.CallFilter{Session: "nope"}, []string{}},
	} {
		if got := keys(toolCalls(t, st, tc.f)); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("ToolCalls(%+v) = %v, want %v", tc.f, got, tc.want)
		}
	}
	if c := toolCalls(t, st, daemon.CallFilter{Session: "s1", Step: sub})[0]; c.Parent != "s1/1.0" || c.Run != 1 || c.Turn != 1 {
		t.Errorf("a sub-agent's call: %+v", c)
	}
}

// A daemon that lost a session writes none of its ledger, and nor does one
// that let it go and took it back: each tenure is an epoch of its own.
func ledgerIsFenced(t *testing.T, st daemon.Store) {
	create(t, st, record("s1")) // d1's, until t0+24h
	if got, _ := st.LoadSession(ctx, "s1"); got.Lease.Epoch != 1 {
		t.Fatalf("a new session's epoch is %d, want 1", got.Lease.Epoch)
	}
	open := func(owner string, epoch int64) (int, error) {
		return st.OpenStep(ctx, daemon.StepOpen{Session: "s1", Owner: owner, Epoch: epoch, Turn: 1, At: t0,
			Calls: []daemon.ToolCall{call(0, "u1", "read", `{}`)}})
	}
	moved := func(owner string, epoch int64, step int) error {
		return st.MoveCalls(ctx, daemon.CallMoves{Session: "s1", Owner: owner, Epoch: epoch, Step: step, At: t0,
			Moves: []daemon.CallMove{{Index: 0, State: tool.StateRunning, Attempt: true}}})
	}
	var owned *daemon.OwnedError
	if _, err := open("d2", 0); !errors.As(err, &owned) || owned.Lease.Owner != owner {
		t.Fatalf("OpenStep by a daemon that does not hold it = %v, want an OwnedError naming d1", err)
	}
	if _, err := open(owner, 2); !errors.Is(err, daemon.ErrOwned) {
		t.Fatalf("OpenStep under an epoch to come = %v, want ErrOwned", err)
	}
	if _, err := open(owner, 1); err != nil {
		t.Fatalf("OpenStep by the owner, under its epoch = %v", err)
	}
	step, err := open(owner, 0)
	if err != nil {
		t.Fatalf("OpenStep by the owner, naming no epoch = %v", err)
	}

	// Taken over: the epoch moves on, and the old owner is fenced off.
	later := t0.Add(25 * time.Hour)
	if _, err := st.Acquire(ctx, "s1", lease("d2", later.Add(time.Hour)), later, 3); err != nil {
		t.Fatal(err)
	}
	if got, _ := st.LoadSession(ctx, "s1"); got.Lease.Epoch != 2 || got.Lease.Owner != "d2" {
		t.Fatalf("taken over, the lease is %+v; want d2's, in epoch 2", got.Lease)
	}
	if err := moved(owner, 1, step); !errors.As(err, &owned) || owned.Lease.Owner != "d2" || owned.Lease.Epoch != 2 {
		t.Fatalf("MoveCalls by the old owner = %v, want an OwnedError naming d2 in epoch 2", err)
	}
	if _, err := open(owner, 0); !errors.Is(err, daemon.ErrOwned) {
		t.Fatalf("OpenStep by the old owner = %v, want ErrOwned", err)
	}
	if c := toolCalls(t, st, daemon.CallFilter{Session: "s1", Step: step}); c[0].State != tool.StatePending {
		t.Fatalf("the old owner's move was made: %+v", c[0])
	}

	// Let go and taken back by the same daemon: a new tenure, so what the
	// last one had in flight is fenced off too.
	if err := st.Release(ctx, "d2", []string{"s1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Acquire(ctx, "s1", lease("d2", later.Add(time.Hour)), later, 3); err != nil {
		t.Fatal(err)
	}
	if err := moved("d2", 2, step); !errors.Is(err, daemon.ErrOwned) {
		t.Fatalf("MoveCalls from the last tenure = %v, want ErrOwned", err)
	}
	stale := record("s1")
	stale.Lease = daemon.Lease{Owner: "d2", Epoch: 2}
	stale.Answer = "late"
	if err := st.SaveSession(ctx, stale); !errors.Is(err, daemon.ErrOwned) {
		t.Fatalf("SaveSession from the last tenure = %v, want ErrOwned", err)
	}
	if err := st.Settle(ctx, daemon.Settlement{Session: stale}); !errors.Is(err, daemon.ErrOwned) {
		t.Fatalf("Settle from the last tenure = %v, want ErrOwned", err)
	}
	if err := moved("d2", 3, step); err != nil {
		t.Fatalf("MoveCalls under the epoch it holds = %v", err)
	}
	fresh := stale
	fresh.Lease.Epoch = 3
	if err := st.SaveSession(ctx, fresh); err != nil {
		t.Fatalf("SaveSession under the epoch it holds = %v", err)
	}

	// Keeping it — renewing, or acquiring what it holds — is the same tenure.
	if _, err := st.Renew(ctx, "d2", []string{"s1"}, later.Add(2*time.Hour)); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Acquire(ctx, "s1", lease("d2", later.Add(3*time.Hour)), later, 3); err != nil {
		t.Fatal(err)
	}
	if got, _ := st.LoadSession(ctx, "s1"); got.Lease.Epoch != 3 {
		t.Errorf("kept, the epoch moved to %d", got.Lease.Epoch)
	}
	if all, _ := st.ListSessions(ctx); len(all) != 1 || all[0].Lease.Epoch != 3 {
		t.Errorf("ListSessions' lease: %+v", all)
	}
}

// Where a turn in flight had got to goes with its session: written by the
// step that got it there, kept by a save that carries it, and gone with the
// save that settles the turn.
func inflightGoesWithTheSession(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	if got, _ := st.LoadSession(ctx, "s1"); got.Inflight != nil {
		t.Fatalf("a new session has a turn in flight: %+v", got.Inflight)
	}
	f := &daemon.Inflight{
		Turn: 2, Base: 3, Iterations: 4, Events: []int64{5, 6}, ReplyTo: []int64{5}, At: 3, Folds: []int{2},
		Messages: []llm.Message{{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.ToolUse{ID: "u1", Name: "read", Input: json.RawMessage(`{"z":1,"a":2}`)},
		}}},
		Origins: []daemon.Origin{{Event: 5, Kind: daemon.KindMessage, Source: "slack", From: "alice", At: t0}},
		Used:    governor.Used{CostUSD: 0.5, Calls: 2, Steps: 2, ToolCalls: 1, InputTokens: 100, OutputTokens: 20, Elapsed: 3 * time.Second},
		Counted: daemon.Spend{CostUSD: 0.25, Calls: 1},
	}
	step := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 2, Inflight: f, At: t0,
		Calls: []daemon.ToolCall{call(0, "u1", "read", `{"z": 1, "a": 2}`)}})
	got, err := st.LoadSession(ctx, "s1")
	if err != nil {
		t.Fatal(err)
	}
	want := *f
	want.Step = step
	if !reflect.DeepEqual(got.Inflight, &want) {
		t.Fatalf("in flight\ngot  %#v\nwant %#v", got.Inflight, &want)
	}

	// A sub-agent's step leaves it be.
	openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 2, Parent: "s1/1.0", Run: 1, At: t0, Sub: &daemon.SubRun{Iterations: 1}})
	if got, _ := st.LoadSession(ctx, "s1"); got.Inflight == nil || got.Inflight.Step != step {
		t.Fatalf("after a sub-agent's step: %+v", got.Inflight)
	}

	got.Answer = "half way"
	if err := st.SaveSession(ctx, got); err != nil {
		t.Fatal(err)
	}
	if again, _ := st.LoadSession(ctx, "s1"); !reflect.DeepEqual(again.Inflight, &want) || again.Answer != "half way" {
		t.Fatalf("a save carrying it lost it: %+v", again.Inflight)
	}
	got.Inflight = nil
	if err := st.Settle(ctx, daemon.Settlement{Session: got}); err != nil {
		t.Fatal(err)
	}
	if again, _ := st.LoadSession(ctx, "s1"); again.Inflight != nil {
		t.Fatalf("settled, the turn is still in flight: %+v", again.Inflight)
	}
}

// Where a sub-agent had got to is kept by the call that started it, and
// which of its sub-agents it is.
func subRunsAreKept(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	m1 := []llm.Message{llm.UserText("find it"), {Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		llm.ToolUse{ID: "c1", Name: "grep", Input: json.RawMessage(`{"q":"x","a":1}`)},
	}}}
	step := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, Parent: "s1/1.0", Run: 1, At: t0,
		Sub: &daemon.SubRun{Messages: m1, Iterations: 1}})
	got, ok, err := st.SubRun(ctx, "s1", "s1/1.0", 1)
	want := daemon.SubRun{Session: "s1", Parent: "s1/1.0", Run: 1, Step: step, Messages: m1, Iterations: 1, Updated: t0}
	if err != nil || !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("SubRun = %#v, %v, %v\nwant %#v", got, ok, err, want)
	}

	m2 := append(m1, llm.UserText("more"))
	next := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, Parent: "s1/1.0", Run: 1, At: t0.Add(time.Second),
		Sub: &daemon.SubRun{Messages: m2, Iterations: 2}})
	got, _, _ = st.SubRun(ctx, "s1", "s1/1.0", 1)
	if got.Step != next || got.Iterations != 2 || len(got.Messages) != 3 || !got.Updated.Equal(t0.Add(time.Second)) {
		t.Errorf("after its next step: %+v", got)
	}
	for _, k := range []struct {
		parent string
		n      int
	}{{"s1/1.0", 2}, {"s1/1.1", 1}} {
		if _, ok, err := st.SubRun(ctx, "s1", k.parent, k.n); ok || err != nil {
			t.Errorf("SubRun(%s, %d) = %v, %v; want nothing", k.parent, k.n, ok, err)
		}
	}
	if _, _, err := st.SubRun(ctx, "nope", "nope/1.0", 1); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("SubRun of an unknown session = %v, want ErrNoSuchSession", err)
	}
}

// Pruning forgets what nothing will carry on: the calls of sessions with
// no turn in flight and nothing waiting, ended — or, never ending, made —
// before the cut, and where the sub-agents of ended calls had got to.
func pruneForgetsWhatNothingCarriesOn(t *testing.T, st daemon.Store) {
	waiting := record("w")
	waiting.State = daemon.Approving
	waiting.Deferred = &core.Deferred{Calls: []core.DeferredCall{{UseID: "u2", Tool: "write"}}}
	for _, rec := range []daemon.SessionRecord{record("q"), record("f"), waiting} {
		create(t, st, rec)
		var f *daemon.Inflight
		if rec.ID == "f" {
			f = &daemon.Inflight{Turn: 1}
		}
		old := openStep(t, st, daemon.StepOpen{Session: rec.ID, Turn: 1, At: t0, Inflight: f,
			Calls: []daemon.ToolCall{call(0, "u1", "delegate", `{}`), call(1, "u2", "write", `{}`)}})
		move(t, st, rec.ID, old, t0.Add(time.Minute), daemon.CallMove{Index: 0, State: tool.StateSucceeded})
		move(t, st, rec.ID, old, t0, daemon.CallMove{Index: 1, State: tool.StateRunning, Attempt: true})
		openStep(t, st, daemon.StepOpen{Session: rec.ID, Turn: 1, Parent: rec.ID + "/1.0", Run: 1, At: t0, Sub: &daemon.SubRun{}})
	}
	// Of q's, a call too new to go, and one still going whose sub-agent
	// stays with it; a sub-agent of a call the ledger has not got goes.
	recent := openStep(t, st, daemon.StepOpen{Session: "q", Turn: 2, At: t0.Add(2 * time.Hour),
		Calls: []daemon.ToolCall{call(0, "u1", "read", `{}`), call(1, "u2", "delegate", `{}`)}})
	move(t, st, "q", recent, t0.Add(2*time.Hour), daemon.CallMove{Index: 0, State: tool.StateSucceeded})
	move(t, st, "q", recent, t0.Add(2*time.Hour), daemon.CallMove{Index: 1, State: tool.StateRunning, Attempt: true})
	openStep(t, st, daemon.StepOpen{Session: "q", Turn: 2, Parent: daemon.CallKey("q", recent, 1), Run: 1, At: t0, Sub: &daemon.SubRun{}})
	openStep(t, st, daemon.StepOpen{Session: "q", Turn: 2, Parent: "q/99.0", Run: 1, At: t0, Sub: &daemon.SubRun{}})

	n, err := st.PruneCalls(ctx, t0.Add(time.Hour))
	if err != nil || n != 2 {
		t.Fatalf("PruneCalls = %d, %v; want q's two old calls", n, err)
	}
	if got := keys(toolCalls(t, st, daemon.CallFilter{Session: "q"})); !reflect.DeepEqual(got, []string{"q/3.1", "q/3.0"}) {
		t.Errorf("q kept %v, want only its recent step", got)
	}
	for _, s := range []string{"f", "w"} {
		if got := toolCalls(t, st, daemon.CallFilter{Session: s}); len(got) != 2 {
			t.Errorf("%s, not quiet, lost calls: %v", s, keys(got))
		}
		if _, ok, _ := st.SubRun(ctx, s, s+"/1.0", 1); !ok {
			t.Errorf("%s, not quiet, lost a sub-agent", s)
		}
	}
	for parent, keep := range map[string]bool{"q/1.0": false, "q/99.0": false, daemon.CallKey("q", recent, 1): true} {
		if _, ok, _ := st.SubRun(ctx, "q", parent, 1); ok != keep {
			t.Errorf("the sub-agent of %s: kept %v, want %v", parent, ok, keep)
		}
	}
	if n, _ := st.PruneCalls(ctx, t0.Add(time.Hour)); n != 0 {
		t.Errorf("pruning again forgot %d", n)
	}
}

// A call that sets a timer, run again, sets it once.
func timersDedup(t *testing.T, st daemon.Store) {
	create(t, st, record("a"))
	first, err := st.AddTimer(ctx, daemon.Timer{Session: "a", Note: "first", Due: t0.Add(time.Hour), CreatedBy: "agent", Created: t0, DedupKey: "a/1.0"})
	if err != nil {
		t.Fatal(err)
	}
	again, err := st.AddTimer(ctx, daemon.Timer{Session: "a", Note: "again", Due: t0.Add(2 * time.Hour), CreatedBy: "agent", Created: t0, DedupKey: "a/1.0"})
	if err != nil || again.ID != first.ID || again.Note != "first" {
		t.Fatalf("the same key again = %+v, %v; want the first timer", again, err)
	}
	if _, err := st.AddTimer(ctx, daemon.Timer{Session: "a", Note: "other", Due: t0, CreatedBy: "agent", Created: t0, DedupKey: "a/1.1"}); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if _, err := st.AddTimer(ctx, daemon.Timer{Session: "a", Note: "keyless", Due: t0, CreatedBy: "api", Created: t0}); err != nil {
			t.Fatal(err)
		}
	}
	if all, _ := st.Timers(ctx, daemon.TimerFilter{Session: "a"}); len(all) != 4 {
		t.Fatalf("%d timers, want 4: one per key, and both without one", len(all))
	}
	// Cancelled, it is still the one that call set.
	if err := st.CancelTimer(ctx, first.ID, "a"); err != nil {
		t.Fatal(err)
	}
	if got, _ := st.AddTimer(ctx, daemon.Timer{Session: "a", Note: "again", Due: t0, CreatedBy: "agent", DedupKey: "a/1.0"}); got.ID != first.ID || got.Status != daemon.TimerCancelled {
		t.Errorf("after cancelling, the key gave %+v", got)
	}
}
