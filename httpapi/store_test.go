package httpapi

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/llm"
)

func TestDirStoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	st, err := DirStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	rec := SessionRecord{ID: newID(), Turns: 2, State: Idle, Messages: []llm.Message{llm.UserText("hi")}, Events: 7}
	if err := st.Save(rec); err != nil {
		t.Fatal(err)
	}
	rec.Turns = 3
	if err := st.Save(rec); err != nil {
		t.Fatal(err)
	}
	got, err := st.Load()
	if err != nil || len(got) != 1 || got[0].Turns != 3 || got[0].Events != 7 || llm.TextOf(got[0].Messages[0].Content) != "hi" {
		t.Fatalf("Load = %+v, %v", got, err)
	}
	if tmps, _ := filepath.Glob(filepath.Join(dir, "*.tmp")); len(tmps) != 0 {
		t.Errorf("temporary files left behind: %v", tmps)
	}
	if err := st.Delete(rec.ID); err != nil {
		t.Fatal(err)
	}
	if err := st.Delete(rec.ID); err != nil {
		t.Errorf("deleting twice: %v", err)
	}
	if got, _ := st.Load(); len(got) != 0 {
		t.Errorf("Load after Delete = %+v", got)
	}
	if err := st.Save(SessionRecord{ID: "../escape"}); err == nil {
		t.Error("an id that is not one of ours named a path")
	}
}

func TestDirStoreLoadSkipsAnUnreadableFile(t *testing.T) {
	dir := t.TempDir()
	st, _ := DirStore(dir)
	good := SessionRecord{ID: newID(), State: Idle}
	if err := st.Save(good); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, newID()+".json"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := st.Load()
	if err == nil || len(got) != 1 || got[0].ID != good.ID {
		t.Errorf("Load = %d records, %v; want the good one and an error naming the bad", len(got), err)
	}
}

func TestSessionsSurviveARestart(t *testing.T) {
	dir := t.TempDir()
	st, _ := DirStore(dir)
	cl := script(textTurn("eight three one seven"), textTurn("24951"))

	m1 := newManager(t, ManagerConfig{Build: builder(cl), Store: st})
	s, err := m1.Create(SessionOptions{Title: "numbers"})
	if err != nil {
		t.Fatal(err)
	}
	send(t, s, "remember 8317")
	events := s.Info().Events
	id := s.ID()
	if err := m1.Close(); err != nil {
		t.Fatal(err)
	}

	if data, err := os.ReadFile(filepath.Join(dir, id+".json")); err != nil || !strings.HasPrefix(string(data), fmt.Sprintf(`{"version":%d,`, RecordVersion)) {
		t.Errorf("the saved record does not lead with its version: %.60s, %v", data, err)
	}

	m2 := newManager(t, ManagerConfig{Build: builder(cl), Store: st})
	back, ok := m2.Get(id)
	if !ok {
		t.Fatal("the session did not come back")
	}
	info := back.Info()
	if info.State != Idle || info.Turns != 1 || info.Title != "numbers" || info.Events != events {
		t.Errorf("restored info = %+v, want idle after 1 turn, events numbered from %d", info, events)
	}
	if len(back.Messages()) != 2 {
		t.Errorf("restored transcript = %d messages, want 2", len(back.Messages()))
	}
	end := send(t, back, "times three?")
	if end.Answer != "24951" {
		t.Errorf("answer = %q", end.Answer)
	}
	// The model saw the conversation from before the restart.
	if got := cl.request(t, 1); len(got.Messages) != 3 {
		t.Errorf("the turn after the restart sent %d messages, want the whole conversation (3)", len(got.Messages))
	}
	// Numbering carried on past the restart.
	for _, f := range collect(t, back, 0, endOfTurn(2)) {
		if f.Seq < events {
			t.Errorf("a frame after the restart reuses sequence %d (< %d)", f.Seq, events)
		}
	}
}

func TestATurnCutShortByShutdownIsRestoredAsInterrupted(t *testing.T) {
	dir := t.TempDir()
	st, _ := DirStore(dir)
	blocking := newBlockingClient()
	m1 := newManager(t, ManagerConfig{Build: builder(blocking), Store: st})
	s, _ := m1.Create(SessionOptions{})
	if _, err := s.Send("a long job"); err != nil {
		t.Fatal(err)
	}
	<-blocking.entered
	id := s.ID()
	_ = m1.Close() // cancels the turn in flight

	m2 := newManager(t, ManagerConfig{Build: builder(script(textTurn("resumed"))), Store: st})
	back, ok := m2.Get(id)
	if !ok {
		t.Fatal("the session did not come back")
	}
	if info := back.Info(); info.State != Failed || info.ErrorKind != KindInterrupted {
		t.Errorf("restored = %s/%s, want failed/%s", info.State, info.ErrorKind, KindInterrupted)
	}
	if !strings.Contains(allText(back.Messages()), "a long job") {
		t.Error("the interrupted turn's prompt was lost")
	}
	if end := send(t, back, "try again"); end.State != Idle {
		t.Errorf("the next turn ended %s, want idle", end.State)
	}
}

func TestAPausedSessionIsStillWaitingAfterARestart(t *testing.T) {
	dir := t.TempDir()
	st, _ := DirStore(dir)
	cl := script(toolTurn("p1", "ask_user", `{"question":"which branch?"}`), textTurn("on main, then"))
	build := builder(cl, wombat.WithTools(pauseTool("ask_user")))

	m1 := newManager(t, ManagerConfig{Build: build, Store: st})
	s, _ := m1.Create(SessionOptions{})
	if end := send(t, s, "deploy"); end.State != Waiting {
		t.Fatalf("first turn ended %s, want waiting", end.State)
	}
	id := s.ID()
	_ = m1.Close()

	m2 := newManager(t, ManagerConfig{Build: build, Store: st})
	back, _ := m2.Get(id)
	if back.Info().State != Waiting {
		t.Fatalf("restored state = %s, want waiting", back.Info().State)
	}
	send(t, back, "main")
	// The answer went in as the pause's tool_result, not as a new instruction.
	last := cl.request(t, 1).Messages
	tr, ok := last[len(last)-1].Content[0].(llm.ToolResult)
	if !ok || tr.ToolUseID != "p1" || tr.Content != "main" {
		t.Errorf("the answer after the restart was sent as %#v", last[len(last)-1].Content)
	}
}

func TestDeletedAndExpiredSessionsLeaveTheStore(t *testing.T) {
	dir := t.TempDir()
	st, _ := DirStore(dir)
	m := newManager(t, ManagerConfig{Build: builder(script()), Store: st})
	s, _ := m.Create(SessionOptions{})
	m.Delete(s.ID())
	if got, _ := st.Load(); len(got) != 0 {
		t.Errorf("a deleted session is still stored: %+v", got)
	}

	old := SessionRecord{ID: newID(), State: Idle, Updated: time.Now().Add(-2 * DefaultTTL)}
	if err := st.Save(old); err != nil {
		t.Fatal(err)
	}
	m2 := newManager(t, ManagerConfig{Build: builder(script()), Store: st})
	if _, ok := m2.Get(old.ID); ok {
		t.Error("an expired session was restored")
	}
	if got, _ := st.Load(); len(got) != 0 {
		t.Errorf("an expired session is still stored: %+v", got)
	}
}

// A record from a newer layout is reported as that, not as whatever its new
// shape fails to decode as — and stays where it is.
func TestDirStoreRefusesANewerRecord(t *testing.T) {
	dir := t.TempDir()
	st, _ := DirStore(dir)
	good := SessionRecord{Version: RecordVersion, ID: newID(), State: Idle}
	if err := st.Save(good); err != nil {
		t.Fatal(err)
	}
	newer := filepath.Join(dir, newID()+".json")
	if err := os.WriteFile(newer, fmt.Appendf(nil, `{"version":%d,"messages":"a shape this build has never seen"}`, RecordVersion+1), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := st.Load()
	if !errors.Is(err, ErrNewerRecord) || len(got) != 1 || got[0].ID != good.ID {
		t.Errorf("Load = %d records, %v; want the current one, and ErrNewerRecord for the other", len(got), err)
	}
	if _, err := os.Stat(newer); err != nil {
		t.Errorf("the newer record is gone: %v", err)
	}
}

// memStore is a Store that checks nothing, which a Store may do.
type memStore struct {
	mu      sync.Mutex
	recs    []SessionRecord
	deleted []string
}

func (m *memStore) Save(rec SessionRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.recs = append(m.recs, rec)
	return nil
}

func (m *memStore) Load() ([]SessionRecord, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]SessionRecord(nil), m.recs...), nil
}

func (m *memStore) Delete(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deleted = append(m.deleted, id)
	return nil
}

// The manager does not restore a record it cannot read the layout of, from a
// Store that handed it over anyway — and does not expire it either.
func TestARecordFromANewerVersionIsLeftAlone(t *testing.T) {
	st := &memStore{recs: []SessionRecord{
		{Version: RecordVersion + 1, ID: newID(), State: Idle, Updated: time.Now()},
		{Version: RecordVersion + 1, ID: newID(), State: Idle, Updated: time.Now().Add(-2 * DefaultTTL)},
	}}
	m := newManager(t, ManagerConfig{Build: builder(script()), Store: st})
	for _, rec := range st.recs {
		if _, ok := m.Get(rec.ID); ok {
			t.Errorf("a version %d record was restored by a build that reads %d", rec.Version, RecordVersion)
		}
	}
	if len(st.deleted) != 0 {
		t.Errorf("deleted %q: a record in a newer layout was expired by this one's rules", st.deleted)
	}
}

func allText(msgs []llm.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		for _, blk := range m.Content {
			switch v := blk.(type) {
			case llm.Text:
				b.WriteString(v.Text + "\n")
			case llm.ToolResult:
				b.WriteString(v.Content + "\n")
			}
		}
	}
	return b.String()
}
