package storetest

import (
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"testing"
	"time"

	"wombat/pkg/core/llm"
	"wombat/pkg/daemon"
)

// ===== What a turn run in slices needs =====

// Where a run stopped at the end of its slice is written without a step of
// its own: the session's turn beside its record, a sub-agent's under its
// call — and only by the daemon that holds the session.
func progressIsSaved(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	step := openStep(t, st, daemon.StepOpen{Session: "s1", Turn: 1, At: t0, Inflight: &daemon.Inflight{Turn: 1},
		Calls: []daemon.ToolCall{call(0, "u1", "read", `{}`)}})
	f := &daemon.Inflight{Turn: 1, Step: step, Base: -1, Iterations: 2, Suspended: true,
		Messages: []llm.Message{llm.UserText("go on")}, State: json.RawMessage(`{"rejections":1}`)}
	if err := st.SaveProgress(ctx, daemon.Progress{Session: "s1", Owner: owner, Inflight: f, At: t0}); err != nil {
		t.Fatal(err)
	}
	got, _ := st.LoadSession(ctx, "s1")
	if !reflect.DeepEqual(got.Inflight, f) {
		t.Fatalf("in flight\ngot  %#v\nwant %#v", got.Inflight, f)
	}
	if c := toolCalls(t, st, daemon.CallFilter{Session: "s1"}); len(c) != 1 {
		t.Fatalf("saving progress opened a step: %v", keys(c))
	}

	sub := &daemon.SubRun{Step: 7, Iterations: 3, Messages: []llm.Message{llm.UserText("look")}, State: json.RawMessage(`{"truncated":1}`)}
	if err := st.SaveProgress(ctx, daemon.Progress{Session: "s1", Owner: owner, Parent: "s1/1.0", Run: 2, Sub: sub, At: t0.Add(time.Second)}); err != nil {
		t.Fatal(err)
	}
	gotSub, ok, err := st.SubRun(ctx, "s1", "s1/1.0", 2)
	want := daemon.SubRun{Session: "s1", Parent: "s1/1.0", Run: 2, Step: 7, Iterations: 3, Messages: sub.Messages,
		State: sub.State, Updated: t0.Add(time.Second)}
	if err != nil || !ok || !reflect.DeepEqual(gotSub, want) {
		t.Fatalf("SubRun = %#v, %v, %v\nwant %#v", gotSub, ok, err, want)
	}
	if again, _ := st.LoadSession(ctx, "s1"); !reflect.DeepEqual(again.Inflight, f) {
		t.Error("a sub-agent's progress touched the session's turn")
	}

	var owned *daemon.OwnedError
	if err := st.SaveProgress(ctx, daemon.Progress{Session: "s1", Owner: "d2", Inflight: &daemon.Inflight{}}); !errors.As(err, &owned) {
		t.Errorf("SaveProgress by a daemon that does not hold it = %v, want an OwnedError", err)
	}
	if err := st.SaveProgress(ctx, daemon.Progress{Session: "s1", Owner: owner, Epoch: 9, Sub: sub, Parent: "x", Run: 1}); !errors.Is(err, daemon.ErrOwned) {
		t.Errorf("SaveProgress under another epoch = %v, want ErrOwned", err)
	}
	if _, ok, _ := st.SubRun(ctx, "s1", "x", 1); ok {
		t.Error("a fenced-off sub-agent's progress was written")
	}
	if err := st.SaveProgress(ctx, daemon.Progress{Session: "nope", Owner: owner, Inflight: &daemon.Inflight{}}); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("SaveProgress for an unknown session = %v, want ErrNoSuchSession", err)
	}
}

// A running turn takes the events of the kinds it asks for as it goes, and
// leaves the rest.
func claimKinds(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	msg := enqueue(t, st, message("s1", "later", t0))
	steer := enqueue(t, st, daemon.Event{Session: "s1", Kind: daemon.KindSteer, Source: daemon.SourceAPI,
		Payload: json.RawMessage(`{"text":"use v2"}`), Created: t0})
	stop := enqueue(t, st, daemon.Event{Session: "s1", Kind: daemon.KindCancel, Source: daemon.SourceAPI, Created: t0})
	if got, _ := st.ClaimKinds(ctx, "s1", "d2", []string{daemon.KindSteer}, t0); len(got) != 0 {
		t.Fatalf("a daemon that does not hold the session claimed %v", ids(got))
	}
	got, err := st.ClaimKinds(ctx, "s1", owner, []string{daemon.KindSteer, daemon.KindCancel}, t0)
	if err != nil || !slices.Equal(ids(got), []int64{steer.ID, stop.ID}) || got[0].Attempts != 1 || got[0].Status != daemon.EventRunning {
		t.Fatalf("ClaimKinds = %+v, %v; want the steer and the cancel, running", got, err)
	}
	if got, _ := st.ClaimKinds(ctx, "s1", owner, []string{daemon.KindSteer}, t0); len(got) != 0 {
		t.Errorf("claimed again: %v", ids(got))
	}
	if got, _ := st.ClaimKinds(ctx, "s1", owner, nil, t0); len(got) != 0 {
		t.Errorf("no kinds claimed %v", ids(got))
	}
	if e := statusOf(t, st, msg.ID); e.Status != daemon.EventPending {
		t.Errorf("the message = %+v, want it left pending", e)
	}
}

// A turn that stopped at the end of its slice puts its events back at once,
// without the claim counting as a try.
func continueDoesNotCountATry(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	e := enqueue(t, st, message("s1", "go", t0))
	if got, _ := st.Claim(ctx, "s1", owner, t0, 10); len(got) != 1 || got[0].Attempts != 1 {
		t.Fatalf("Claim = %+v", got)
	}
	if err := st.Settle(ctx, daemon.Settlement{Continue: []int64{e.ID}}); err != nil {
		t.Fatal(err)
	}
	got := statusOf(t, st, e.ID)
	if got.Status != daemon.EventPending || got.Attempts != 0 {
		t.Fatalf("continued, the event is %+v; want pending, no try counted", got)
	}
	if again, _ := st.Claim(ctx, "s1", owner, t0, 10); len(again) != 1 || again[0].Attempts != 1 {
		t.Fatalf("claimed again: %+v", again)
	}
}

// A session's log is kept: each frame once, read back after a sequence
// number, forgotten with the session or once old.
func framesAreKept(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	create(t, st, record("s2"))
	frame := func(seq int, kind string, at time.Time) daemon.StoredFrame {
		return daemon.StoredFrame{Seq: seq, Turn: 1, Kind: kind, Data: json.RawMessage(`{"type":"` + kind + `","n":1}`), At: at}
	}
	if last, err := st.LastFrame(ctx, "s1"); err != nil || last != -1 {
		t.Fatalf("LastFrame of an empty log = %d, %v", last, err)
	}
	first := []daemon.StoredFrame{frame(0, "turn_started", t0), frame(1, "text_delta", t0), frame(2, "text_delta", t0.Add(time.Hour))}
	if err := st.AppendFrames(ctx, "s1", owner, 0, first); err != nil {
		t.Fatal(err)
	}
	// Kept once: the same sequence number again changes nothing.
	if err := st.AppendFrames(ctx, "s1", owner, 1, []daemon.StoredFrame{frame(2, "other", t0), frame(3, "turn_ended", t0.Add(time.Hour))}); err != nil {
		t.Fatal(err)
	}
	if err := st.AppendFrames(ctx, "s2", "", 0, []daemon.StoredFrame{frame(0, "turn_started", t0)}); err != nil {
		t.Fatal(err)
	}
	all, err := st.Frames(ctx, daemon.FrameFilter{Session: "s1", After: -1})
	if err != nil || len(all) != 4 || all[2].Kind != "text_delta" || all[3].Seq != 3 {
		t.Fatalf("Frames = %+v, %v", all, err)
	}
	if !reflect.DeepEqual(all[:3], first) {
		t.Errorf("frames came back\ngot  %+v\nwant %+v", all[:3], first)
	}
	if after, _ := st.Frames(ctx, daemon.FrameFilter{Session: "s1", After: 1, Limit: 1}); len(after) != 1 || after[0].Seq != 2 {
		t.Errorf("after 1, one = %+v", after)
	}
	if last, _ := st.LastFrame(ctx, "s1"); last != 3 {
		t.Errorf("LastFrame = %d, want 3", last)
	}
	if _, err := st.Frames(ctx, daemon.FrameFilter{Session: "nope"}); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("Frames of an unknown session = %v, want ErrNoSuchSession", err)
	}
	if err := st.AppendFrames(ctx, "nope", owner, 0, first); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("AppendFrames to an unknown session = %v, want ErrNoSuchSession", err)
	}
	// A daemon that does not hold the session, or holds it no more under
	// the epoch it names, has nothing to say in it.
	late := []daemon.StoredFrame{frame(9, "turn_ended", t0)}
	if err := st.AppendFrames(ctx, "s1", "d2", 0, late); !errors.Is(err, daemon.ErrOwned) {
		t.Errorf("AppendFrames by another daemon = %v, want ErrOwned", err)
	}
	if err := st.AppendFrames(ctx, "s1", owner, 7, late); !errors.Is(err, daemon.ErrOwned) {
		t.Errorf("AppendFrames under another epoch = %v, want ErrOwned", err)
	}
	if last, _ := st.LastFrame(ctx, "s1"); last != 3 {
		t.Errorf("a fenced-off frame was kept: last %d", last)
	}

	if n, err := st.PruneFrames(ctx, t0.Add(time.Minute)); err != nil || n != 3 {
		t.Fatalf("PruneFrames = %d, %v; want the three before the cut", n, err)
	}
	if left, _ := st.Frames(ctx, daemon.FrameFilter{Session: "s1", After: -1}); len(left) != 2 {
		t.Errorf("left %+v", left)
	}
	if err := st.DeleteSession(ctx, "s1"); err != nil {
		t.Fatal(err)
	}
	if _, err := st.LastFrame(ctx, "s1"); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("a deleted session's log: %v", err)
	}
}
