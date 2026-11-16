package daemon

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"time"
)

// MemoryStore returns a [Store] that keeps everything in this process, and
// loses all of it when the process ends: for tests, and for a daemon that
// does not need to outlive itself. It behaves as the Postgres store does —
// both pass the same suite — so a test against one is a test of the other.
func MemoryStore() Store {
	return &memStore{sessions: map[string]*memSession{}, leaders: map[string]Lease{}}
}

type memStore struct {
	mu       sync.Mutex
	sessions map[string]*memSession
	leaders  map[string]Lease // by role
	events   []Event          // in id order
	timers   []Timer          // in id order
	outbox   []Message
	lastID   struct{ event, timer, message int64 }
}

// memSession is a record as Postgres keeps one: encoded, so nothing a caller
// holds aliases it, and the transcript apart from the rest.
type memSession struct {
	thread   string
	created  time.Time
	awaiting bool
	meta     []byte
	messages []byte
	lease    Lease
}

// ownedBy reports the session held, at now, by a daemon other than owner.
func (s *memSession) ownedBy(owner string, now time.Time) bool {
	return s.lease.Owner != "" && s.lease.Owner != owner && s.lease.Until.After(now)
}

// micro is a time as a store keeps it.
func micro(t time.Time) time.Time {
	if t.IsZero() {
		return t
	}
	return t.UTC().Truncate(time.Microsecond)
}

func encodeRecord(rec SessionRecord) (meta, messages []byte, err error) {
	if messages, err = json.Marshal(rec.Messages); err != nil {
		return nil, nil, fmt.Errorf("daemon: encoding session %s: %w", rec.ID, err)
	}
	rec.Messages = nil
	if meta, err = json.Marshal(rec); err != nil {
		return nil, nil, fmt.Errorf("daemon: encoding session %s: %w", rec.ID, err)
	}
	return meta, messages, nil
}

// DecodeRecord rebuilds a record from the two halves a store keeps it in:
// everything but the transcript, and the transcript, which may be nil. For
// a [Store] implementation.
func DecodeRecord(meta, messages []byte) (SessionRecord, error) {
	var rec SessionRecord
	if err := json.Unmarshal(meta, &rec); err != nil {
		return SessionRecord{}, fmt.Errorf("daemon: decoding a session: %w", err)
	}
	if rec.Version > RecordVersion {
		return SessionRecord{}, fmt.Errorf("%w (session %s, version %d; this build reads up to %d)",
			ErrNewerRecord, rec.ID, rec.Version, RecordVersion)
	}
	rec.Messages = nil
	if messages != nil {
		if err := json.Unmarshal(messages, &rec.Messages); err != nil {
			return SessionRecord{}, fmt.Errorf("daemon: decoding session %s: %w", rec.ID, err)
		}
	}
	return rec, nil
}

// EncodeRecord splits a record into the two halves a store keeps it in. For a
// [Store] implementation; see [DecodeRecord].
func EncodeRecord(rec SessionRecord) (meta, messages []byte, err error) {
	return encodeRecord(rec)
}

func (m *memStore) CreateSession(_ context.Context, rec SessionRecord) error {
	meta, msgs, err := encodeRecord(rec)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, dup := m.sessions[rec.ID]; dup {
		return fmt.Errorf("daemon: session %s exists already", rec.ID)
	}
	if rec.Thread != "" {
		for _, s := range m.sessions {
			if s.thread == rec.Thread {
				return fmt.Errorf("%w: %q", ErrThreadTaken, rec.Thread)
			}
		}
	}
	lease := rec.Lease
	lease.Until = micro(lease.Until)
	m.sessions[rec.ID] = &memSession{
		thread: rec.Thread, created: micro(rec.Created), awaiting: rec.Deferred != nil,
		meta: meta, messages: msgs, lease: lease,
	}
	return nil
}

func (m *memStore) SaveSession(_ context.Context, rec SessionRecord) error {
	meta, msgs, err := encodeRecord(rec)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.saveLocked(rec, meta, msgs)
}

func (m *memStore) saveLocked(rec SessionRecord, meta, msgs []byte) error {
	if err := m.writableLocked(rec); err != nil {
		return err
	}
	s := m.sessions[rec.ID]
	s.awaiting, s.meta, s.messages = rec.Deferred != nil, meta, msgs
	return nil
}

// writableLocked is whether rec may be written: the session is there, and
// held by rec.Lease.Owner when that is set.
func (m *memStore) writableLocked(rec SessionRecord) error {
	s, ok := m.sessions[rec.ID]
	switch {
	case !ok:
		return fmt.Errorf("%w: %s", ErrNoSuchSession, rec.ID)
	case rec.Lease.Owner != "" && s.lease.Owner != rec.Lease.Owner:
		return &OwnedError{Session: rec.ID, Lease: s.lease}
	}
	return nil
}

func (m *memStore) LoadSession(_ context.Context, id string) (SessionRecord, error) {
	m.mu.Lock()
	s, ok := m.sessions[id]
	var meta, msgs []byte
	var lease Lease
	if ok {
		meta, msgs, lease = s.meta, s.messages, s.lease
	}
	m.mu.Unlock()
	if !ok {
		return SessionRecord{}, fmt.Errorf("%w: %s", ErrNoSuchSession, id)
	}
	rec, err := DecodeRecord(meta, msgs)
	rec.Lease = lease
	return rec, err
}

func (m *memStore) SessionByThread(_ context.Context, thread string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if thread != "" {
		for id, s := range m.sessions {
			if s.thread == thread {
				return id, nil
			}
		}
	}
	return "", fmt.Errorf("%w: thread %q", ErrNoSuchSession, thread)
}

func (m *memStore) ListSessions(context.Context) ([]SessionRecord, error) {
	m.mu.Lock()
	type entry struct {
		id      string
		created time.Time
		meta    []byte
		lease   Lease
	}
	all := make([]entry, 0, len(m.sessions))
	for id, s := range m.sessions {
		all = append(all, entry{id, s.created, s.meta, s.lease})
	}
	m.mu.Unlock()
	slices.SortFunc(all, func(a, b entry) int {
		return cmp.Or(a.created.Compare(b.created), cmp.Compare(a.id, b.id))
	})
	out := make([]SessionRecord, 0, len(all))
	for _, e := range all {
		rec, err := DecodeRecord(e.meta, nil)
		if err != nil {
			return nil, err
		}
		rec.Lease = e.lease
		out = append(out, rec)
	}
	return out, nil
}

func (m *memStore) DeleteSession(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.sessions[id]; !ok {
		return fmt.Errorf("%w: %s", ErrNoSuchSession, id)
	}
	delete(m.sessions, id)
	m.events = slices.DeleteFunc(m.events, func(e Event) bool { return e.Session == id })
	m.timers = slices.DeleteFunc(m.timers, func(t Timer) bool { return t.Session == id })
	m.outbox = slices.DeleteFunc(m.outbox, func(msg Message) bool { return msg.Session == id })
	return nil
}

// ===== The inbox =====

func (m *memStore) Enqueue(_ context.Context, e Event) (Event, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.enqueueLocked(e)
}

func (m *memStore) enqueueLocked(e Event) (Event, bool, error) {
	if _, ok := m.sessions[e.Session]; !ok {
		return Event{}, false, fmt.Errorf("%w: %s", ErrNoSuchSession, e.Session)
	}
	if e.DedupKey != "" {
		for _, old := range m.events {
			if old.Source == e.Source && old.DedupKey == e.DedupKey {
				return copyEvent(old), false, nil
			}
		}
	}
	m.lastID.event++
	e.ID = m.lastID.event
	e.Status, e.Attempts, e.Error = EventPending, 0, ""
	e.Created = micro(cmp.Or(e.Created, time.Now()))
	e.NotBefore = micro(cmp.Or(e.NotBefore, e.Created))
	e.Payload = clone(e.Payload)
	m.events = append(m.events, e)
	return copyEvent(e), true, nil
}

// claimable is the one rule Ready and Claim share.
func (m *memStore) claimable(e Event, now time.Time) bool {
	if e.Status != EventPending || e.NotBefore.After(now) {
		return false
	}
	s, ok := m.sessions[e.Session]
	return ok && (!s.awaiting || e.Kind == KindApproval)
}

// orphaned is a running event whose session nobody holds any more, at now:
// its turn died with the daemon that ran it.
func (m *memStore) orphaned(e Event, owner string, now time.Time) bool {
	s, ok := m.sessions[e.Session]
	return ok && e.Status == EventRunning && s.lease.Owner != owner && !s.lease.Until.After(now)
}

func (m *memStore) Ready(_ context.Context, now time.Time, owner string, busy []string, limit int) ([]string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	now = micro(now)
	var out []string
	for _, e := range m.events {
		if len(out) >= limit {
			break
		}
		s, ok := m.sessions[e.Session]
		if !ok || s.ownedBy(owner, now) || slices.Contains(busy, e.Session) || slices.Contains(out, e.Session) {
			continue
		}
		if m.claimable(e, now) || m.orphaned(e, owner, now) {
			out = append(out, e.Session)
		}
	}
	return out, nil
}

func (m *memStore) Claim(_ context.Context, session, owner string, now time.Time, limit int) ([]Event, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	now = micro(now)
	if s, ok := m.sessions[session]; !ok || s.lease.Owner != owner {
		return nil, nil
	}
	var out []Event
	for i := range m.events {
		if len(out) >= limit {
			break
		}
		e := &m.events[i]
		if e.Session != session || !m.claimable(*e, now) {
			continue
		}
		e.Status = EventRunning
		e.Attempts++
		out = append(out, copyEvent(*e))
	}
	return out, nil
}

func (m *memStore) Settle(_ context.Context, s Settlement) error {
	var meta, msgs []byte
	if s.Session.ID != "" {
		var err error
		if meta, msgs, err = encodeRecord(s.Session); err != nil {
			return err
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	// Everything that can refuse is checked before anything is changed.
	if s.Session.ID != "" {
		if err := m.writableLocked(s.Session); err != nil {
			return err
		}
	}
	for _, msg := range s.Outbox {
		if _, ok := m.sessions[msg.Session]; !ok {
			return fmt.Errorf("%w: %s", ErrNoSuchSession, msg.Session)
		}
	}

	if s.Session.ID != "" {
		_ = m.saveLocked(s.Session, meta, msgs)
	}
	m.setStatus(s.Done, EventDone, "", time.Time{})
	m.setStatus(s.Retry, EventPending, s.Error, micro(s.RetryAt))
	m.setStatus(s.Failed, EventFailed, s.Error, time.Time{})
	for _, msg := range s.Outbox {
		_, _, _ = m.postLocked(msg)
	}
	return nil
}

func (m *memStore) setStatus(ids []int64, status EventStatus, why string, notBefore time.Time) {
	for i := range m.events {
		e := &m.events[i]
		if !slices.Contains(ids, e.ID) {
			continue
		}
		e.Status, e.Error = status, why
		if !notBefore.IsZero() {
			e.NotBefore = notBefore
		}
	}
}

// recoverLocked puts the running events of the sessions in back to pending,
// or fails those tried attempts times already, and says how many went back.
func (m *memStore) recoverLocked(in func(session string) bool, attempts int) int {
	n := 0
	for i := range m.events {
		e := &m.events[i]
		if e.Status != EventRunning || !in(e.Session) {
			continue
		}
		if e.Attempts >= attempts {
			e.Status, e.Error = EventFailed, RecoverError(e.Attempts)
			continue
		}
		e.Status = EventPending
		n++
	}
	return n
}

func (m *memStore) Acquire(_ context.Context, session string, l Lease, now time.Time, attempts int) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[session]
	if !ok {
		return 0, fmt.Errorf("%w: %s", ErrNoSuchSession, session)
	}
	if s.ownedBy(l.Owner, micro(now)) {
		return 0, &OwnedError{Session: session, Lease: s.lease}
	}
	n := 0
	if s.lease.Owner != l.Owner {
		// Whatever the last owner left running died with it.
		n = m.recoverLocked(func(id string) bool { return id == session }, attempts)
	}
	l.Until = micro(l.Until)
	s.lease = l
	return n, nil
}

func (m *memStore) Renew(_ context.Context, owner string, sessions []string, until time.Time) ([]string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var kept []string
	for _, id := range sessions {
		if s, ok := m.sessions[id]; ok && s.lease.Owner == owner {
			s.lease.Until = micro(until)
			kept = append(kept, id)
		}
	}
	return kept, nil
}

func (m *memStore) Release(_ context.Context, owner string, sessions []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, id := range sessions {
		if s, ok := m.sessions[id]; ok && s.lease.Owner == owner {
			s.lease = Lease{}
		}
	}
	return nil
}

func (m *memStore) Recover(_ context.Context, owner string, attempts int) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	mine := func(id string) bool { s, ok := m.sessions[id]; return ok && s.lease.Owner == owner }
	n := m.recoverLocked(mine, attempts)
	for _, s := range m.sessions {
		if s.lease.Owner == owner {
			s.lease = Lease{}
		}
	}
	return n, nil
}

// Lead: a lease on the role, as on a session, so that Managers sharing one
// memory store — in one process — have one dispatcher too.
func (m *memStore) Lead(_ context.Context, role, owner string, until time.Time) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	l := m.leaders[role]
	if l.Owner != "" && l.Owner != owner && l.Until.After(time.Now()) {
		return false, nil
	}
	m.leaders[role] = Lease{Owner: owner, Until: until}
	return true, nil
}

// RecoverError is what an event [Store.Recover] or [Store.Acquire] gives up
// on says. For a [Store] implementation, so both say the same.
func RecoverError(attempts int) string {
	return fmt.Sprintf("the daemon stopped while this event's turn was running, %d times", attempts)
}

func (m *memStore) Events(_ context.Context, f EventFilter) ([]Event, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []Event
	for i := len(m.events) - 1; i >= 0; i-- {
		e := m.events[i]
		if (f.Session != "" && e.Session != f.Session) || (f.Status != "" && e.Status != f.Status) {
			continue
		}
		out = append(out, copyEvent(e))
		if f.Limit > 0 && len(out) == f.Limit {
			break
		}
	}
	return out, nil
}

// ===== Timers =====

func (m *memStore) AddTimer(_ context.Context, t Timer) (Timer, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.sessions[t.Session]; !ok {
		return Timer{}, fmt.Errorf("%w: %s", ErrNoSuchSession, t.Session)
	}
	m.lastID.timer++
	t.ID = m.lastID.timer
	t.Status, t.Fired = TimerActive, 0
	t.Due = micro(t.Due)
	t.Created = micro(cmp.Or(t.Created, time.Now()))
	m.timers = append(m.timers, t)
	return t, nil
}

func (m *memStore) CancelTimer(_ context.Context, id int64, session string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i := range m.timers {
		t := &m.timers[i]
		if t.ID == id && t.Status == TimerActive && (session == "" || t.Session == session) {
			t.Status = TimerCancelled
			return nil
		}
	}
	return fmt.Errorf("%w: %d", ErrNoSuchTimer, id)
}

func (m *memStore) Timers(_ context.Context, f TimerFilter) ([]Timer, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []Timer
	for _, t := range m.timers {
		if (f.Session != "" && t.Session != f.Session) || (f.Status != "" && t.Status != f.Status) {
			continue
		}
		out = append(out, t)
	}
	slices.SortStableFunc(out, func(a, b Timer) int { return a.Due.Compare(b.Due) })
	return out, nil
}

func (m *memStore) Fire(_ context.Context, now time.Time) ([]Event, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	now = micro(now)
	due := make([]int, 0)
	for i, t := range m.timers {
		if t.Status == TimerActive && !t.Due.After(now) {
			due = append(due, i)
		}
	}
	slices.SortStableFunc(due, func(a, b int) int { return m.timers[a].Due.Compare(m.timers[b].Due) })

	var out []Event
	for _, i := range due {
		t := &m.timers[i]
		e, created, err := m.enqueueLocked(TimerEvent(*t, now))
		if err != nil {
			return nil, err
		}
		if created {
			out = append(out, e)
		}
		t.Fired++
		if next, again := NextDue(*t, now); again {
			t.Due = micro(next)
		} else {
			t.Status = TimerFired
		}
	}
	return out, nil
}

// TimerPayload is the payload of a [KindTimer] event.
type TimerPayload struct {
	Timer int64     `json:"timer"`
	Note  string    `json:"note"`
	Due   time.Time `json:"due"`
	Every string    `json:"every,omitempty"`
}

// TimerEvent is the event a timer due by now fires. Its DedupKey is the timer
// and the time it was due, so firing the same due time twice enqueues once.
// For a [Store] implementation, so both fire the same events.
func TimerEvent(t Timer, now time.Time) Event {
	p := TimerPayload{Timer: t.ID, Note: t.Note, Due: t.Due.UTC()}
	if t.Every > 0 {
		p.Every = t.Every.String()
	}
	payload, _ := json.Marshal(p) // cannot fail: plain fields
	return Event{
		Session: t.Session, Kind: KindTimer, Source: SourceTimer,
		DedupKey: fmt.Sprintf("%d@%d", t.ID, t.Due.UnixMicro()),
		Payload:  payload, Created: now, NotBefore: now,
	}
}

// NextDue is when a timer that has just fired at now is due again — its first
// period after now, so a daemon that was down for a week fires a daily timer
// once, not seven times — and false for one that is done. For a [Store]
// implementation.
func NextDue(t Timer, now time.Time) (time.Time, bool) {
	if t.Every <= 0 {
		return time.Time{}, false
	}
	periods := now.Sub(t.Due)/t.Every + 1
	return t.Due.Add(periods * t.Every), true
}

// ===== The outbox =====

func (m *memStore) Post(_ context.Context, msg Message) (Message, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.postLocked(msg)
}

func (m *memStore) postLocked(msg Message) (Message, bool, error) {
	if _, ok := m.sessions[msg.Session]; !ok {
		return Message{}, false, fmt.Errorf("%w: %s", ErrNoSuchSession, msg.Session)
	}
	if msg.DedupKey != "" {
		for _, old := range m.outbox {
			if old.DedupKey == msg.DedupKey {
				return copyMessage(old), false, nil
			}
		}
	}
	m.lastID.message++
	msg.ID = m.lastID.message
	msg.Status, msg.Attempts, msg.Error, msg.Delivered = MessagePending, 0, "", time.Time{}
	msg.Created = micro(cmp.Or(msg.Created, time.Now()))
	msg.NotBefore = micro(msg.NotBefore)
	msg.Data, msg.InReplyTo = clone(msg.Data), slices.Clone(msg.InReplyTo)
	m.outbox = append(m.outbox, msg)
	return copyMessage(msg), true, nil
}

func (m *memStore) Outbox(_ context.Context, f OutboxFilter) ([]Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []Message
	for _, msg := range m.outbox {
		if (f.Session != "" && msg.Session != f.Session) || (f.Status != "" && msg.Status != f.Status) || msg.ID <= f.After {
			continue
		}
		out = append(out, copyMessage(msg))
		if f.Limit > 0 && len(out) == f.Limit {
			break
		}
	}
	return out, nil
}

func (m *memStore) Delivered(_ context.Context, id int64, at time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i := range m.outbox {
		msg := &m.outbox[i]
		if msg.ID != id {
			continue
		}
		if msg.Status != MessageDelivered {
			msg.Status, msg.Delivered, msg.Error = MessageDelivered, micro(at), ""
		}
		return nil
	}
	return fmt.Errorf("%w: %d", ErrNoSuchMessage, id)
}

func (m *memStore) Undelivered(_ context.Context, id int64, reason string, retry time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i := range m.outbox {
		msg := &m.outbox[i]
		if msg.ID != id {
			continue
		}
		if msg.Status != MessagePending {
			return nil
		}
		msg.Attempts++
		msg.Error = reason
		if retry.IsZero() {
			msg.Status = MessageFailed
		} else {
			msg.NotBefore = micro(retry)
		}
		return nil
	}
	return fmt.Errorf("%w: %d", ErrNoSuchMessage, id)
}

func (m *memStore) Close() error { return nil }

func clone(b json.RawMessage) json.RawMessage {
	if len(b) == 0 {
		return nil
	}
	return slices.Clone(b)
}

func copyEvent(e Event) Event {
	e.Payload = clone(e.Payload)
	return e
}

func copyMessage(m Message) Message {
	m.Data, m.InReplyTo = clone(m.Data), slices.Clone(m.InReplyTo)
	return m
}
