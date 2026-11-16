// Package storetest is what a [daemon.Store] has to do, as tests. A store
// that passes [Run] is one a [daemon.Manager] can be given; [daemon.MemoryStore]
// and the Postgres store both do, which is what lets a daemon test written
// against the one stand for the other.
package storetest

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/daemon"
)

// Run runs the suite. open returns an empty store, a new one for every test;
// closing it is open's business.
func Run(t *testing.T, open func(t *testing.T) daemon.Store) {
	for _, tc := range []struct {
		name string
		fn   func(*testing.T, daemon.Store)
	}{
		{"SessionRoundTrip", sessionRoundTrip},
		{"SessionKeepsToolInputsByteForByte", toolInputsByteForByte},
		{"SaveOnlyReplaces", saveOnlyReplaces},
		{"Threads", threads},
		{"ListLeavesTheTranscriptOut", listLeavesTheTranscriptOut},
		{"DeleteTakesEverythingWithIt", deleteTakesEverything},
		{"NewerRecordIsRefused", newerRecordIsRefused},
		{"EnqueueAndDedup", enqueueAndDedup},
		{"ReadyAndClaim", readyAndClaim},
		{"AnAwaitingSessionTakesOnlyApprovals", awaitingTakesOnlyApprovals},
		{"SettleIsAllOrNothing", settleIsAllOrNothing},
		{"Recover", recoverRunning},
		{"Leases", leases},
		{"WritesAreFenced", writesAreFenced},
		{"Lead", lead},
		{"EventsNewestFirst", eventsNewestFirst},
		{"TimersAndCancel", timersAndCancel},
		{"FireOnceAndRepeat", fireOnceAndRepeat},
		{"Outbox", outbox},
		{"Delivery", delivery},
	} {
		t.Run(tc.name, func(t *testing.T) { tc.fn(t, open(t)) })
	}
}

var (
	ctx = context.Background()
	t0  = time.Date(2026, 11, 15, 9, 0, 0, 0, time.UTC)
)

// owner is the daemon the suite's sessions are held by, unless a test says
// otherwise; its lease runs well past anything a test does at t0.
const owner = "d1"

func lease(name string, until time.Time) daemon.Lease {
	return daemon.Lease{Owner: name, Addr: "http://" + name + ":8080", Until: until}
}

func record(id string) daemon.SessionRecord {
	return daemon.SessionRecord{
		Version: daemon.RecordVersion, ID: id, State: daemon.Idle,
		Options: daemon.SessionOptions{Title: "t " + id},
		Created: t0, Updated: t0, Lease: lease(owner, t0.Add(24*time.Hour)),
	}
}

func create(t *testing.T, st daemon.Store, rec daemon.SessionRecord) {
	t.Helper()
	if err := st.CreateSession(ctx, rec); err != nil {
		t.Fatalf("CreateSession(%s): %v", rec.ID, err)
	}
}

func enqueue(t *testing.T, st daemon.Store, e daemon.Event) daemon.Event {
	t.Helper()
	got, created, err := st.Enqueue(ctx, e)
	if err != nil || !created {
		t.Fatalf("Enqueue(%+v) = %v, %v", e, created, err)
	}
	return got
}

func message(session, text string, at time.Time) daemon.Event {
	payload, _ := json.Marshal(map[string]string{"text": text})
	return daemon.Event{Session: session, Kind: daemon.KindMessage, Source: daemon.SourceAPI,
		Payload: payload, Created: at}
}

func ids(events []daemon.Event) []int64 {
	out := make([]int64, len(events))
	for i, e := range events {
		out[i] = e.ID
	}
	return out
}

func statusOf(t *testing.T, st daemon.Store, id int64) daemon.Event {
	t.Helper()
	all, err := st.Events(ctx, daemon.EventFilter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range all {
		if e.ID == id {
			return e
		}
	}
	t.Fatalf("event %d is not in the inbox", id)
	return daemon.Event{}
}

// ===== Sessions =====

func sessionRoundTrip(t *testing.T, st daemon.Store) {
	rec := record("s1")
	rec.Thread = "github:o/r#1"
	rec.Turns, rec.State, rec.Events = 2, daemon.Approving, 41
	rec.Messages = []llm.Message{
		llm.UserText("copy a to b"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.ToolUse{ID: "w1", Name: "write", Input: json.RawMessage(`{"path":"b"}`)},
		}},
	}
	rec.Spend = daemon.Spend{CostUSD: 0.25, InputTokens: 10, OutputTokens: 5, Calls: 2}
	rec.PauseID = "p1"
	rec.Deferred = &core.Deferred{
		Calls: []core.DeferredCall{{UseID: "w1", Tool: "write", Input: json.RawMessage(`{"path":"b"}`), Reason: "outside"}},
		Done:  []llm.ToolResult{{ToolUseID: "r1", Content: "contents"}},
	}
	rec.Verdicts = map[llm.ToolUseID]bool{"w1": true}
	rec.DeferredAt = t0.Add(time.Minute)
	rec.DecidedBy = []daemon.Origin{{Event: 5, Kind: daemon.KindApproval, Source: daemon.SourceAPI, From: "carol", At: t0.Add(time.Second)}}
	rec.ReplyTo = []int64{3, 4}
	rec.Origins = map[int][]daemon.Origin{0: {
		{Event: 3, Kind: daemon.KindMessage, Source: "slack", From: "alice", At: t0},
		{Event: 4, Kind: "push", Source: "github", At: t0.Add(time.Millisecond)},
	}}
	create(t, st, rec)

	got, err := st.LoadSession(ctx, "s1")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, rec) {
		t.Errorf("round trip\ngot  %#v\nwant %#v", got, rec)
	}
	if _, err := st.LoadSession(ctx, "nope"); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("LoadSession(unknown) = %v, want ErrNoSuchSession", err)
	}
	if err := st.CreateSession(ctx, record("s1")); err == nil {
		t.Error("a second session with the same id was created")
	}
}

// The transcript goes back to the provider on the next turn, and a prompt
// cache matches it byte for byte. Encoding compacts a tool's input, which is
// the same every time; a store that reorders its keys — as Postgres's jsonb
// does — makes every turn after a reload a cache miss.
func toolInputsByteForByte(t *testing.T, st daemon.Store) {
	rec := record("s1")
	input := json.RawMessage(`{"z": 1,  "a":[2, 1], "m": {"y": true, "b": null}}`)
	rec.Messages = []llm.Message{{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		llm.ToolUse{ID: "u1", Name: "t", Input: input},
	}}}
	create(t, st, rec)
	got, err := st.LoadSession(ctx, "s1")
	if err != nil {
		t.Fatal(err)
	}
	want := `{"z":1,"a":[2,1],"m":{"y":true,"b":null}}`
	if in := got.Messages[0].Content[0].(llm.ToolUse).Input; string(in) != want {
		t.Errorf("the tool's input came back as %s, want %s", in, want)
	}
}

func saveOnlyReplaces(t *testing.T, st daemon.Store) {
	rec := record("s1")
	if err := st.SaveSession(ctx, rec); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Fatalf("SaveSession of a session never created = %v, want ErrNoSuchSession", err)
	}
	if _, err := st.LoadSession(ctx, "s1"); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Fatalf("a save brought a session into being: %v", err)
	}
	create(t, st, rec)
	rec.Turns, rec.Answer = 1, "done"
	rec.Messages = []llm.Message{llm.UserText("hi")}
	if err := st.SaveSession(ctx, rec); err != nil {
		t.Fatal(err)
	}
	got, err := st.LoadSession(ctx, "s1")
	if err != nil || got.Turns != 1 || got.Answer != "done" || len(got.Messages) != 1 {
		t.Errorf("after a save: %+v, %v", got, err)
	}
}

func threads(t *testing.T, st daemon.Store) {
	a, b := record("a"), record("b")
	a.Thread, b.Thread = "slack:C1/1700000000.0001", "slack:C1/1700000000.0001"
	create(t, st, a)
	if err := st.CreateSession(ctx, b); !errors.Is(err, daemon.ErrThreadTaken) {
		t.Fatalf("a second session for one thread = %v, want ErrThreadTaken", err)
	}
	if id, err := st.SessionByThread(ctx, a.Thread); err != nil || id != "a" {
		t.Errorf("SessionByThread = %q, %v; want a", id, err)
	}
	if _, err := st.SessionByThread(ctx, "slack:other"); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("SessionByThread(unknown) = %v, want ErrNoSuchSession", err)
	}
	// Sessions without a thread do not collide with each other.
	create(t, st, record("c"))
	create(t, st, record("d"))
}

func listLeavesTheTranscriptOut(t *testing.T, st daemon.Store) {
	for i, id := range []string{"late", "early", "middle"} {
		rec := record(id)
		rec.Created = t0.Add([]time.Duration{3, 1, 2}[i] * time.Minute)
		rec.Messages = []llm.Message{llm.UserText("a long transcript")}
		create(t, st, rec)
	}
	list, err := st.ListSessions(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var order []string
	for _, rec := range list {
		order = append(order, rec.ID)
		if rec.Messages != nil {
			t.Errorf("%s was listed with its transcript", rec.ID)
		}
		if rec.Options.Title != "t "+rec.ID {
			t.Errorf("%s was listed without the rest of its record: %+v", rec.ID, rec)
		}
	}
	if want := []string{"early", "middle", "late"}; !slices.Equal(order, want) {
		t.Errorf("listed %v, want %v (oldest first)", order, want)
	}
}

func deleteTakesEverything(t *testing.T, st daemon.Store) {
	create(t, st, record("gone"))
	create(t, st, record("kept"))
	for _, s := range []string{"gone", "kept"} {
		enqueue(t, st, message(s, "hi", t0))
		if _, err := st.AddTimer(ctx, daemon.Timer{Session: s, Note: "n", Due: t0.Add(time.Hour), CreatedBy: "api"}); err != nil {
			t.Fatal(err)
		}
		if _, _, err := st.Post(ctx, daemon.Message{Session: s, Kind: daemon.MessageSent, Text: "x"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := st.DeleteSession(ctx, "gone"); err != nil {
		t.Fatal(err)
	}
	if err := st.DeleteSession(ctx, "gone"); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("deleting twice = %v, want ErrNoSuchSession", err)
	}
	events, _ := st.Events(ctx, daemon.EventFilter{})
	timers, _ := st.Timers(ctx, daemon.TimerFilter{})
	msgs, _ := st.Outbox(ctx, daemon.OutboxFilter{})
	if len(events) != 1 || len(timers) != 1 || len(msgs) != 1 {
		t.Fatalf("left %d events, %d timers, %d messages; want one of each, the kept session's", len(events), len(timers), len(msgs))
	}
	if events[0].Session != "kept" || timers[0].Session != "kept" || msgs[0].Session != "kept" {
		t.Error("the delete took the wrong session's rows")
	}
	if _, _, err := st.Enqueue(ctx, message("gone", "hi", t0)); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("Enqueue to a deleted session = %v, want ErrNoSuchSession", err)
	}
}

func newerRecordIsRefused(t *testing.T, st daemon.Store) {
	rec := record("s1")
	rec.Version = daemon.RecordVersion + 1
	create(t, st, rec)
	if _, err := st.LoadSession(ctx, "s1"); !errors.Is(err, daemon.ErrNewerRecord) {
		t.Errorf("LoadSession of a newer record = %v, want ErrNewerRecord", err)
	}
}

// ===== The inbox =====

func enqueueAndDedup(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	if _, _, err := st.Enqueue(ctx, message("nope", "hi", t0)); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("Enqueue to an unknown session = %v, want ErrNoSuchSession", err)
	}
	a := enqueue(t, st, message("s1", "one", t0))
	b := enqueue(t, st, message("s1", "two", t0))
	if a.ID <= 0 || b.ID <= a.ID {
		t.Errorf("ids %d then %d, want increasing", a.ID, b.ID)
	}
	if a.Status != daemon.EventPending || a.Attempts != 0 || !a.Created.Equal(t0) || !a.NotBefore.Equal(t0) {
		t.Errorf("a new event: %+v", a)
	}
	if string(a.Payload) != `{"text":"one"}` {
		t.Errorf("payload = %s", a.Payload)
	}

	hook := daemon.Event{Session: "s1", Kind: "push", Source: "github", From: "octocat", DedupKey: "delivery-7", Created: t0}
	first := enqueue(t, st, hook)
	again, created, err := st.Enqueue(ctx, hook)
	if err != nil || created || again.ID != first.ID {
		t.Errorf("the same delivery again = %d, %v, %v; want event %d back, not created", again.ID, created, err, first.ID)
	}
	if first.From != "octocat" || again.From != "octocat" {
		t.Errorf("who sent the event came back as %q, then %q", first.From, again.From)
	}
	listed, err := st.Events(ctx, daemon.EventFilter{Session: "s1"})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range listed {
		if want := map[int64]string{first.ID: "octocat"}[e.ID]; e.From != want {
			t.Errorf("event %d listed from %q, want %q", e.ID, e.From, want)
		}
	}
	claimed, err := st.Claim(ctx, "s1", owner, t0.Add(time.Hour), 10)
	if err != nil {
		t.Fatal(err)
	}
	if i := slices.IndexFunc(claimed, func(e daemon.Event) bool { return e.ID == first.ID }); i < 0 || claimed[i].From != "octocat" {
		t.Errorf("claimed %+v, want event %d from octocat among them", claimed, first.ID)
	}
	hook.Source = "gitlab"
	enqueue(t, st, hook) // a key is unique per source, not across them
}

func readyAndClaim(t *testing.T, st daemon.Store) {
	for _, s := range []string{"a", "b", "c"} {
		create(t, st, record(s))
	}
	b1 := enqueue(t, st, message("b", "b1", t0))
	a1 := enqueue(t, st, message("a", "a1", t0))
	a2 := enqueue(t, st, message("a", "a2", t0))
	later := message("c", "later", t0)
	later.NotBefore = t0.Add(time.Hour)
	enqueue(t, st, later)

	ready, err := st.Ready(ctx, t0, owner, nil, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ready, []string{"b", "a"}) {
		t.Errorf("Ready = %v, want [b a]: longest waiting first, and c's event held back", ready)
	}
	if ready, _ := st.Ready(ctx, t0, owner, []string{"b"}, 10); !slices.Equal(ready, []string{"a"}) {
		t.Errorf("Ready with b busy = %v, want [a]", ready)
	}
	if ready, _ := st.Ready(ctx, t0, owner, nil, 1); !slices.Equal(ready, []string{"b"}) {
		t.Errorf("Ready(limit 1) = %v, want [b]", ready)
	}
	if ready, _ := st.Ready(ctx, t0.Add(time.Hour), owner, nil, 10); !slices.Equal(ready, []string{"b", "a", "c"}) {
		t.Errorf("Ready an hour later = %v, want c too", ready)
	}

	got, err := st.Claim(ctx, "a", owner, t0, 1)
	if err != nil || !slices.Equal(ids(got), []int64{a1.ID}) {
		t.Fatalf("Claim(limit 1) = %v, %v; want [%d]", ids(got), err, a1.ID)
	}
	if got[0].Status != daemon.EventRunning || got[0].Attempts != 1 {
		t.Errorf("a claimed event: %+v", got[0])
	}
	got, _ = st.Claim(ctx, "a", owner, t0, 10)
	if !slices.Equal(ids(got), []int64{a2.ID}) {
		t.Errorf("the rest of a = %v, want [%d]", ids(got), a2.ID)
	}
	if got, _ := st.Claim(ctx, "a", owner, t0, 10); len(got) != 0 {
		t.Errorf("claimed %v twice", ids(got))
	}
	if ready, _ := st.Ready(ctx, t0, owner, nil, 10); !slices.Equal(ready, []string{"b"}) {
		t.Errorf("Ready once a is claimed = %v, want [b]", ready)
	}
	_ = b1
}

func awaitingTakesOnlyApprovals(t *testing.T, st daemon.Store) {
	rec := record("s1")
	rec.State = daemon.Approving
	rec.Deferred = &core.Deferred{Calls: []core.DeferredCall{{UseID: "w1", Tool: "write"}}}
	create(t, st, rec)
	msg := enqueue(t, st, message("s1", "never mind", t0))
	if ready, _ := st.Ready(ctx, t0, owner, nil, 10); len(ready) != 0 {
		t.Fatalf("Ready = %v; a session waiting on a decision took a message", ready)
	}
	ok := enqueue(t, st, daemon.Event{Session: "s1", Kind: daemon.KindApproval, Source: daemon.SourceAPI,
		Payload: json.RawMessage(`{"use_id":"w1","allow":true}`), Created: t0})
	if ready, _ := st.Ready(ctx, t0, owner, nil, 10); !slices.Equal(ready, []string{"s1"}) {
		t.Fatalf("Ready = %v, want the session, for its approval", ready)
	}
	if got, _ := st.Claim(ctx, "s1", owner, t0, 10); !slices.Equal(ids(got), []int64{ok.ID}) {
		t.Fatalf("Claim = %v, want only the approval %d", ids(got), ok.ID)
	}

	// Decided, it takes everything again.
	rec.State, rec.Deferred = daemon.Idle, nil
	if err := st.Settle(ctx, daemon.Settlement{Session: rec, Done: []int64{ok.ID}}); err != nil {
		t.Fatal(err)
	}
	if got, _ := st.Claim(ctx, "s1", owner, t0, 10); !slices.Equal(ids(got), []int64{msg.ID}) {
		t.Errorf("Claim after the decision = %v, want the message %d", ids(got), msg.ID)
	}
}

func settleIsAllOrNothing(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	var claimed []daemon.Event
	for _, text := range []string{"done", "retry", "fail"} {
		enqueue(t, st, message("s1", text, t0))
	}
	claimed, _ = st.Claim(ctx, "s1", owner, t0, 10)
	if len(claimed) != 3 {
		t.Fatalf("claimed %d", len(claimed))
	}

	// A settlement for a session that is not there changes nothing at all.
	ghost := record("ghost")
	err := st.Settle(ctx, daemon.Settlement{Session: ghost, Done: ids(claimed),
		Outbox: []daemon.Message{{Session: "s1", Kind: daemon.MessageReply, Text: "lost"}}})
	if !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Fatalf("Settle for a missing session = %v, want ErrNoSuchSession", err)
	}
	if e := statusOf(t, st, claimed[0].ID); e.Status != daemon.EventRunning {
		t.Errorf("a refused settlement settled an event: %+v", e)
	}
	if msgs, _ := st.Outbox(ctx, daemon.OutboxFilter{}); len(msgs) != 0 {
		t.Errorf("a refused settlement posted %+v", msgs)
	}

	rec := record("s1")
	rec.Turns, rec.Answer = 1, "the answer"
	retryAt := t0.Add(time.Minute)
	err = st.Settle(ctx, daemon.Settlement{
		Session: rec,
		Done:    []int64{claimed[0].ID},
		Retry:   []int64{claimed[1].ID}, RetryAt: retryAt,
		Failed: []int64{claimed[2].ID},
		Error:  "the provider fell over",
		Outbox: []daemon.Message{{Session: "s1", Kind: daemon.MessageReply, Text: "the answer", InReplyTo: []int64{claimed[0].ID}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := st.LoadSession(ctx, "s1"); got.Answer != "the answer" {
		t.Errorf("the session was not saved: %+v", got)
	}
	if e := statusOf(t, st, claimed[0].ID); e.Status != daemon.EventDone || e.Error != "" {
		t.Errorf("done: %+v", e)
	}
	if e := statusOf(t, st, claimed[1].ID); e.Status != daemon.EventPending || !e.NotBefore.Equal(retryAt) || e.Error == "" || e.Attempts != 1 {
		t.Errorf("retry: %+v", e)
	}
	if e := statusOf(t, st, claimed[2].ID); e.Status != daemon.EventFailed || e.Error != "the provider fell over" {
		t.Errorf("failed: %+v", e)
	}
	msgs, _ := st.Outbox(ctx, daemon.OutboxFilter{})
	if len(msgs) != 1 || msgs[0].Text != "the answer" || !slices.Equal(msgs[0].InReplyTo, []int64{claimed[0].ID}) {
		t.Errorf("outbox = %+v", msgs)
	}
	if ready, _ := st.Ready(ctx, t0, owner, nil, 10); len(ready) != 0 {
		t.Errorf("the retry was ready before its time: %v", ready)
	}
	if got, _ := st.Claim(ctx, "s1", owner, retryAt, 10); len(got) != 1 || got[0].Attempts != 2 {
		t.Errorf("the retry, claimed at its time: %+v", got)
	}

	// No session: only the events.
	enqueue(t, st, message("s1", "x", t0))
	got, _ := st.Claim(ctx, "s1", owner, t0, 10)
	if err := st.Settle(ctx, daemon.Settlement{Failed: ids(got), Error: "cannot build"}); err != nil {
		t.Fatal(err)
	}
	if e := statusOf(t, st, got[0].ID); e.Status != daemon.EventFailed {
		t.Errorf("settled without a session: %+v", e)
	}
}

func recoverRunning(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	theirs := record("s2")
	theirs.Lease = lease("d2", t0.Add(24*time.Hour))
	create(t, st, theirs)
	enqueue(t, st, message("s1", "once", t0))
	enqueue(t, st, message("s2", "theirs", t0))
	first, _ := st.Claim(ctx, "s1", owner, t0, 10)
	other, _ := st.Claim(ctx, "s2", "d2", t0, 10)
	enqueue(t, st, message("s1", "pending", t0))

	n, err := st.Recover(ctx, owner, 3)
	if err != nil || n != 1 {
		t.Fatalf("Recover = %d, %v; want the one running event back", n, err)
	}
	if e := statusOf(t, st, first[0].ID); e.Status != daemon.EventPending || e.Attempts != 1 {
		t.Errorf("recovered: %+v", e)
	}
	if e := statusOf(t, st, other[0].ID); e.Status != daemon.EventRunning {
		t.Errorf("another daemon's running event was touched: %+v", e)
	}
	if got, _ := st.LoadSession(ctx, "s1"); got.Lease.Owner != "" {
		t.Errorf("after Recover the session is held by %q, want nobody", got.Lease.Owner)
	}

	if _, err := st.Acquire(ctx, "s1", lease(owner, t0.Add(time.Hour)), t0, 3); err != nil {
		t.Fatal(err)
	}
	got, _ := st.Claim(ctx, "s1", owner, t0, 10)
	if len(got) != 2 {
		t.Fatalf("claimed %d", len(got))
	}
	if n, err := st.Recover(ctx, owner, 2); err != nil || n != 1 {
		t.Fatalf("Recover(2) = %d, %v; want one back and the twice-tried one failed", n, err)
	}
	if e := statusOf(t, st, first[0].ID); e.Status != daemon.EventFailed || e.Error == "" {
		t.Errorf("tried twice: %+v", e)
	}
}

// A session is held by one daemon at a time: another is refused while the
// lease runs, renewing keeps it, releasing frees it, and one that runs out is
// anyone's — whoever takes it puts back what the last owner left running.
func leases(t *testing.T, st daemon.Store) {
	create(t, st, record("s1")) // held by d1 until t0+24h
	enqueue(t, st, message("s1", "hi", t0))
	running, _ := st.Claim(ctx, "s1", owner, t0, 10)
	if len(running) != 1 {
		t.Fatalf("the owner claimed %d", len(running))
	}

	_, err := st.Acquire(ctx, "s1", lease("d2", t0.Add(time.Hour)), t0, 3)
	var owned *daemon.OwnedError
	if !errors.As(err, &owned) || owned.Session != "s1" || owned.Lease.Owner != owner || owned.Lease.Addr != "http://d1:8080" {
		t.Fatalf("Acquire by another daemon = %v, want an OwnedError naming d1 and where it is", err)
	}
	if got, _ := st.Claim(ctx, "s1", "d2", t0, 10); len(got) != 0 {
		t.Fatalf("a daemon that does not hold the session claimed %d", len(got))
	}
	if n, err := st.Acquire(ctx, "s1", lease(owner, t0.Add(time.Hour)), t0, 3); err != nil || n != 0 {
		t.Fatalf("Acquire by the owner = %d, %v; want it kept, nothing put back", n, err)
	}
	if _, err := st.Acquire(ctx, "nope", lease(owner, t0), t0, 3); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("Acquire(unknown) = %v, want ErrNoSuchSession", err)
	}

	// Renewing: only what the daemon holds, and the answer says which.
	kept, err := st.Renew(ctx, owner, []string{"s1", "nope"}, t0.Add(2*time.Hour))
	if err != nil || !slices.Equal(kept, []string{"s1"}) {
		t.Fatalf("Renew = %v, %v; want [s1]", kept, err)
	}
	if kept, _ := st.Renew(ctx, "d2", []string{"s1"}, t0.Add(time.Hour)); len(kept) != 0 {
		t.Fatalf("another daemon renewed %v", kept)
	}
	if got, _ := st.LoadSession(ctx, "s1"); got.Lease.Owner != owner || !got.Lease.Until.Equal(t0.Add(2*time.Hour)) {
		t.Fatalf("lease = %+v, want d1's, until t0+2h", got.Lease)
	}

	// Run out, it is anyone's, and the running event goes back.
	later := t0.Add(3 * time.Hour)
	if ready, _ := st.Ready(ctx, later, "d2", nil, 10); !slices.Equal(ready, []string{"s1"}) {
		t.Fatalf("Ready for d2 once the lease ran out = %v, want the session left running", ready)
	}
	n, err := st.Acquire(ctx, "s1", lease("d2", later.Add(time.Hour)), later, 3)
	if err != nil || n != 1 {
		t.Fatalf("Acquire after the lease ran out = %d, %v; want it taken and the running event back", n, err)
	}
	if e := statusOf(t, st, running[0].ID); e.Status != daemon.EventPending {
		t.Fatalf("the dead owner's event = %+v, want pending", e)
	}
	if got, _ := st.Claim(ctx, "s1", "d2", later, 10); len(got) != 1 || got[0].Attempts != 2 {
		t.Fatalf("the new owner claimed %+v", got)
	}
	if kept, _ := st.Renew(ctx, owner, []string{"s1"}, later); len(kept) != 0 {
		t.Fatal("the old owner still renews a session taken from it")
	}

	// Released, it is anyone's at once.
	if err := st.Release(ctx, "d2", []string{"s1"}); err != nil {
		t.Fatal(err)
	}
	if got, _ := st.LoadSession(ctx, "s1"); got.Lease.Owner != "" {
		t.Fatalf("after Release the session is held by %+v", got.Lease)
	}
	if _, err := st.Acquire(ctx, "s1", lease("d3", later.Add(time.Hour)), later, 3); err != nil {
		t.Fatalf("Acquire of a released session = %v", err)
	}
}

// A daemon that lost a session writes nothing of it: not a save, not a
// settlement, and a refused settlement changes nothing at all.
func writesAreFenced(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	e := enqueue(t, st, message("s1", "hi", t0))
	running, _ := st.Claim(ctx, "s1", owner, t0, 10)
	if len(running) != 1 {
		t.Fatal("not claimed")
	}
	later := t0.Add(25 * time.Hour)
	if _, err := st.Acquire(ctx, "s1", lease("d2", later.Add(time.Hour)), later, 3); err != nil {
		t.Fatal(err)
	}

	stale := record("s1") // d1's
	stale.Answer = "late"
	if err := st.SaveSession(ctx, stale); !errors.Is(err, daemon.ErrOwned) {
		t.Fatalf("SaveSession by the old owner = %v, want ErrOwned", err)
	}
	err := st.Settle(ctx, daemon.Settlement{Session: stale, Done: []int64{e.ID},
		Outbox: []daemon.Message{{Session: "s1", Kind: daemon.MessageReply, Text: "late", Created: later}}})
	if !errors.Is(err, daemon.ErrOwned) {
		t.Fatalf("Settle by the old owner = %v, want ErrOwned", err)
	}
	if got, _ := st.LoadSession(ctx, "s1"); got.Answer == "late" {
		t.Fatal("the old owner's record was written")
	}
	if got := statusOf(t, st, e.ID); got.Status == daemon.EventDone {
		t.Fatal("the old owner's settlement settled the event")
	}
	if msgs, _ := st.Outbox(ctx, daemon.OutboxFilter{Session: "s1"}); len(msgs) != 0 {
		t.Fatalf("the old owner's settlement posted %+v", msgs)
	}

	// The new owner writes; a record naming no owner is written as before.
	fresh := record("s1")
	fresh.Lease = daemon.Lease{Owner: "d2"}
	fresh.Answer = "now"
	if err := st.SaveSession(ctx, fresh); err != nil {
		t.Fatalf("SaveSession by the owner = %v", err)
	}
	anyone := record("s1")
	anyone.Lease = daemon.Lease{}
	if err := st.SaveSession(ctx, anyone); err != nil {
		t.Fatalf("SaveSession naming no owner = %v", err)
	}
}

// Lead: whoever asks first leads, and goes on leading.
func lead(t *testing.T, st daemon.Store) {
	for range 2 {
		if ok, err := st.Lead(ctx, "deliver", owner, time.Now().Add(time.Hour)); err != nil || !ok {
			t.Fatalf("Lead = %v, %v; want the first to ask to lead, and keep leading", ok, err)
		}
	}
}

func eventsNewestFirst(t *testing.T, st daemon.Store) {
	create(t, st, record("a"))
	create(t, st, record("b"))
	a1 := enqueue(t, st, message("a", "1", t0))
	b1 := enqueue(t, st, message("b", "1", t0))
	a2 := enqueue(t, st, message("a", "2", t0))
	st.Claim(ctx, "a", owner, t0, 1)

	all, _ := st.Events(ctx, daemon.EventFilter{})
	if !slices.Equal(ids(all), []int64{a2.ID, b1.ID, a1.ID}) {
		t.Errorf("Events = %v, want newest first", ids(all))
	}
	if got, _ := st.Events(ctx, daemon.EventFilter{Session: "a"}); !slices.Equal(ids(got), []int64{a2.ID, a1.ID}) {
		t.Errorf("Events(a) = %v", ids(got))
	}
	if got, _ := st.Events(ctx, daemon.EventFilter{Status: daemon.EventRunning}); !slices.Equal(ids(got), []int64{a1.ID}) {
		t.Errorf("Events(running) = %v", ids(got))
	}
	if got, _ := st.Events(ctx, daemon.EventFilter{Limit: 2}); !slices.Equal(ids(got), []int64{a2.ID, b1.ID}) {
		t.Errorf("Events(limit 2) = %v", ids(got))
	}
}

// ===== Timers =====

func timersAndCancel(t *testing.T, st daemon.Store) {
	create(t, st, record("a"))
	create(t, st, record("b"))
	if _, err := st.AddTimer(ctx, daemon.Timer{Session: "nope", Due: t0}); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("AddTimer for an unknown session = %v", err)
	}
	late, err := st.AddTimer(ctx, daemon.Timer{Session: "a", Note: "late", Due: t0.Add(2 * time.Hour), CreatedBy: "agent", Created: t0})
	if err != nil {
		t.Fatal(err)
	}
	soon, _ := st.AddTimer(ctx, daemon.Timer{Session: "b", Note: "soon", Due: t0.Add(time.Hour), Every: 24 * time.Hour, CreatedBy: "api", Created: t0})
	if late.Status != daemon.TimerActive || late.ID <= 0 || late.CreatedBy != "agent" || !late.Created.Equal(t0) {
		t.Errorf("a new timer: %+v", late)
	}

	all, _ := st.Timers(ctx, daemon.TimerFilter{})
	if len(all) != 2 || all[0].ID != soon.ID || all[0].Every != 24*time.Hour || all[1].ID != late.ID {
		t.Errorf("Timers = %+v, want the soonest first", all)
	}
	if err := st.CancelTimer(ctx, late.ID, "b"); !errors.Is(err, daemon.ErrNoSuchTimer) {
		t.Errorf("cancelling another session's timer = %v, want ErrNoSuchTimer", err)
	}
	if err := st.CancelTimer(ctx, late.ID, "a"); err != nil {
		t.Fatal(err)
	}
	if err := st.CancelTimer(ctx, late.ID, ""); !errors.Is(err, daemon.ErrNoSuchTimer) {
		t.Errorf("cancelling twice = %v, want ErrNoSuchTimer", err)
	}
	if got, _ := st.Timers(ctx, daemon.TimerFilter{Status: daemon.TimerActive}); len(got) != 1 || got[0].ID != soon.ID {
		t.Errorf("active timers = %+v", got)
	}
	if got, _ := st.Timers(ctx, daemon.TimerFilter{Session: "a"}); len(got) != 1 || got[0].Status != daemon.TimerCancelled {
		t.Errorf("a's timers = %+v", got)
	}
}

func fireOnceAndRepeat(t *testing.T, st daemon.Store) {
	create(t, st, record("s1"))
	once, _ := st.AddTimer(ctx, daemon.Timer{Session: "s1", Note: "check the deploy", Due: t0, CreatedBy: "agent"})
	daily, _ := st.AddTimer(ctx, daemon.Timer{Session: "s1", Note: "standup", Due: t0.Add(-71 * time.Hour), Every: 24 * time.Hour, CreatedBy: "api"})
	future, _ := st.AddTimer(ctx, daemon.Timer{Session: "s1", Note: "later", Due: t0.Add(time.Hour), CreatedBy: "api"})
	gone, _ := st.AddTimer(ctx, daemon.Timer{Session: "s1", Note: "cancelled", Due: t0, CreatedBy: "api"})
	st.CancelTimer(ctx, gone.ID, "")

	fired, err := st.Fire(ctx, t0)
	if err != nil {
		t.Fatal(err)
	}
	// The daily one was due three days running and fires once: a daemon
	// that was down for a while catches up, it does not stampede.
	if len(fired) != 2 {
		t.Fatalf("fired %d events, want one per due timer: %+v", len(fired), fired)
	}
	for _, e := range fired {
		if e.Kind != daemon.KindTimer || e.Source != daemon.SourceTimer || e.Status != daemon.EventPending || e.Session != "s1" {
			t.Errorf("a timer's event: %+v", e)
		}
	}
	var p daemon.TimerPayload
	if err := json.Unmarshal(fired[0].Payload, &p); err != nil || p.Timer != daily.ID || p.Note != "standup" || p.Every != "24h0m0s" {
		t.Errorf("the first event, the timer due longest ago: %s (%v)", fired[0].Payload, err)
	}

	byID := map[int64]daemon.Timer{}
	all, _ := st.Timers(ctx, daemon.TimerFilter{})
	for _, tm := range all {
		byID[tm.ID] = tm
	}
	if tm := byID[once.ID]; tm.Status != daemon.TimerFired || tm.Fired != 1 {
		t.Errorf("the one-shot timer: %+v", tm)
	}
	if tm := byID[daily.ID]; tm.Status != daemon.TimerActive || tm.Fired != 1 || !tm.Due.Equal(t0.Add(time.Hour)) {
		t.Errorf("the daily timer: %+v, want it due next at %v", tm, t0.Add(time.Hour))
	}
	if tm := byID[future.ID]; tm.Status != daemon.TimerActive || tm.Fired != 0 {
		t.Errorf("the timer not due yet: %+v", tm)
	}

	if again, _ := st.Fire(ctx, t0); len(again) != 0 {
		t.Errorf("firing again at the same time enqueued %+v", again)
	}
	if next, _ := st.Fire(ctx, t0.Add(time.Hour)); len(next) != 2 {
		t.Errorf("an hour later fired %d, want the daily one and the later one", len(next))
	}
}

// ===== The outbox =====

func outbox(t *testing.T, st daemon.Store) {
	create(t, st, record("a"))
	create(t, st, record("b"))
	if _, _, err := st.Post(ctx, daemon.Message{Session: "nope", Kind: daemon.MessageSent}); !errors.Is(err, daemon.ErrNoSuchSession) {
		t.Errorf("Post for an unknown session = %v", err)
	}
	m1, created, err := st.Post(ctx, daemon.Message{Session: "a", Thread: "th", Kind: daemon.MessageApproval, Text: "may I?",
		Data: json.RawMessage(`{"calls": [1]}`), Created: t0})
	if err != nil || !created {
		t.Fatal(err)
	}
	if m1.Status != daemon.MessagePending || m1.ID <= 0 || !m1.Created.Equal(t0) || string(m1.Data) != `{"calls": [1]}` || m1.Thread != "th" {
		t.Errorf("a new message: %+v", m1)
	}
	m2, _, _ := st.Post(ctx, daemon.Message{Session: "b", Kind: daemon.MessageSent, Text: "hi", DedupKey: "send:b:u1"})
	again, created, _ := st.Post(ctx, daemon.Message{Session: "b", Kind: daemon.MessageSent, Text: "hi", DedupKey: "send:b:u1"})
	if created || again.ID != m2.ID {
		t.Errorf("the same send twice = %d, %v; want %d back", again.ID, created, m2.ID)
	}
	m3, _, _ := st.Post(ctx, daemon.Message{Session: "a", Kind: daemon.MessageReply, Text: "done"})

	all, _ := st.Outbox(ctx, daemon.OutboxFilter{})
	if len(all) != 3 || all[0].ID != m1.ID || all[1].ID != m2.ID || all[2].ID != m3.ID {
		t.Errorf("Outbox = %+v, want oldest first", all)
	}
	if got, _ := st.Outbox(ctx, daemon.OutboxFilter{After: m1.ID}); len(got) != 2 || got[0].ID != m2.ID {
		t.Errorf("Outbox(after %d) = %+v", m1.ID, got)
	}
	if got, _ := st.Outbox(ctx, daemon.OutboxFilter{Session: "a", Limit: 1}); len(got) != 1 || got[0].ID != m1.ID {
		t.Errorf("Outbox(a, limit 1) = %+v", got)
	}
	st.Delivered(ctx, m1.ID, t0)
	if got, _ := st.Outbox(ctx, daemon.OutboxFilter{Status: daemon.MessagePending}); len(got) != 2 || got[0].ID != m2.ID {
		t.Errorf("Outbox(pending) = %+v", got)
	}
}

func delivery(t *testing.T, st daemon.Store) {
	create(t, st, record("a"))
	m, _, _ := st.Post(ctx, daemon.Message{Session: "a", Kind: daemon.MessageSent, Text: "x"})
	if err := st.Delivered(ctx, 999, t0); !errors.Is(err, daemon.ErrNoSuchMessage) {
		t.Errorf("Delivered(unknown) = %v", err)
	}
	if err := st.Undelivered(ctx, 999, "x", t0); !errors.Is(err, daemon.ErrNoSuchMessage) {
		t.Errorf("Undelivered(unknown) = %v", err)
	}

	retry := t0.Add(time.Minute)
	if err := st.Undelivered(ctx, m.ID, "502 from the webhook", retry); err != nil {
		t.Fatal(err)
	}
	got, _ := st.Outbox(ctx, daemon.OutboxFilter{})
	if got[0].Status != daemon.MessagePending || got[0].Attempts != 1 || !got[0].NotBefore.Equal(retry) || got[0].Error != "502 from the webhook" {
		t.Errorf("after a failed delivery: %+v", got[0])
	}
	if err := st.Delivered(ctx, m.ID, t0.Add(2*time.Minute)); err != nil {
		t.Fatal(err)
	}
	if err := st.Delivered(ctx, m.ID, t0.Add(time.Hour)); err != nil {
		t.Errorf("delivered twice = %v, want no error", err)
	}
	st.Undelivered(ctx, m.ID, "late failure", time.Time{})
	got, _ = st.Outbox(ctx, daemon.OutboxFilter{})
	if got[0].Status != daemon.MessageDelivered || !got[0].Delivered.Equal(t0.Add(2*time.Minute)) || got[0].Error != "" {
		t.Errorf("a delivered message: %+v, want it delivered once, at the first time", got[0])
	}

	f, _, _ := st.Post(ctx, daemon.Message{Session: "a", Kind: daemon.MessageSent, Text: "y"})
	st.Undelivered(ctx, f.ID, "410 gone", time.Time{})
	got, _ = st.Outbox(ctx, daemon.OutboxFilter{After: m.ID})
	if got[0].Status != daemon.MessageFailed || got[0].Attempts != 1 {
		t.Errorf("a message given up on: %+v", got[0])
	}
}
