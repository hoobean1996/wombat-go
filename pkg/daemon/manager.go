package daemon

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
)

// Defaults [NewManager] fills in for a zero-valued [ManagerConfig] field.
const (
	// DefaultIdle is how long a session with nothing to do stays in memory.
	// Measured from its last activity, and never while a turn runs or a
	// stream follows it; it stays in the store either way.
	DefaultIdle = 30 * time.Minute

	// DefaultBufferSize is how many events one session's log retains. Sized
	// for a conversation rather than a screenful: a reasoning model emits
	// hundreds of deltas per turn, and a client that reconnects should get
	// what it missed rather than a window.
	DefaultBufferSize = 16384

	// DefaultPoll is how often the worker, the timers and the dispatcher look
	// at the store. An event posted through this process wakes the worker at
	// once; the poll is for timers, retries, and what arrives otherwise.
	DefaultPoll = time.Second

	// DefaultMaxRuns is how many turns run at once, across sessions.
	DefaultMaxRuns = 4

	// DefaultAttempts is how many times an event is tried before it fails.
	DefaultAttempts = 3

	// DefaultLease is how long a daemon's hold on a session lasts without
	// being renewed: how long the sessions of a daemon that died wait before
	// another takes them over. Renewed every third of it.
	DefaultLease = 30 * time.Second

	// DefaultRetryDelay is the backoff before an event's second try; it
	// doubles for each try after.
	DefaultRetryDelay = 10 * time.Second
)

// Bounds the daemon holds what it is asked to do to.
const (
	// MaxBatch is the most events one turn is given; the rest wait for the
	// next.
	MaxBatch = 32

	// MaxTimers is the most active timers one session may have.
	MaxTimers = 32

	// MinEvery is the shortest a repeating timer may repeat at.
	MinEvery = time.Minute

	// maxDeliveries is how many times a message is sent before the
	// dispatcher gives up on it, and moves on to the ones behind it.
	maxDeliveries = 10

	// storeTimeout bounds a store call the daemon makes on its own behalf.
	storeTimeout = 30 * time.Second

	// deliverTimeout bounds one delivery.
	deliverTimeout = 30 * time.Second

	// crashGap is how far a session's log numbering jumps when it is loaded
	// after a crash cut a turn short: the frames that turn issued were never
	// saved, and a client may hold their numbers. A gap is how a client is
	// told it missed something.
	crashGap = 1 << 20

	// The sweeper's period is Idle/10, bounded by these, so a short Idle — a
	// test's — is observable and a long one costs nothing.
	minSweep = 10 * time.Millisecond
	maxSweep = 30 * time.Second
)

// ManagerConfig configures a [Manager].
type ManagerConfig struct {
	// Build makes the agent for one session. Required.
	//
	// Per session and not once per process, because the interesting choices —
	// which directory the file tools are rooted at, which permission policy
	// gates them — are per session. A [Manager] never inspects the options it
	// passes: clamping what a client asked for against what the deployment
	// allows is the host's job, and the host is the only party that knows.
	//
	// The agent's permission gate should defer what needs a person — see
	// [permission.Defer] — and its tools should include [Tools].
	Build func(SessionOptions) (*core.Agent, error)

	// Store keeps everything. Default: a [MemoryStore].
	//
	// Several Managers may share one — daemons on several machines, one
	// Postgres. Each session is run by one of them at a time; see
	// [Store.Acquire].
	Store Store

	// Name is this daemon's, among those sharing the Store: unique, and the
	// same when it starts again, so that it takes back at once the sessions
	// it was running when it stopped. Default: one made up at random, whose
	// sessions wait out Lease after a crash before another daemon takes them.
	Name string

	// Addr is where this daemon answers HTTP, as another daemon sharing the
	// Store would reach it — "http://10.0.0.5:8080" — to forward it the
	// requests for the sessions it runs. Optional; see [OwnedError].
	Addr string

	// Lease is how long a hold on a session lasts unrenewed. Default
	// DefaultLease.
	Lease time.Duration

	// Limits caps ONE TURN. A long conversation is bounded per exchange; a
	// cumulative cap would kill a session mid-sentence on its fourth question
	// for reasons the user cannot see.
	Limits governor.Limits

	// Idle is how long a session with nothing to do stays in memory.
	// Default DefaultIdle.
	Idle time.Duration

	// BufferSize is how many events one session's log retains.
	// Default DefaultBufferSize.
	BufferSize int

	// Logger receives lifecycle lines. Default: a handler that discards.
	Logger *slog.Logger

	// Poll is how often the store is looked at. Default DefaultPoll.
	Poll time.Duration

	// MaxRuns is how many turns run at once. Default DefaultMaxRuns.
	MaxRuns int

	// Attempts is how many times an event is tried. Default DefaultAttempts.
	Attempts int

	// RetryDelay is the backoff before a second try. Default
	// DefaultRetryDelay.
	RetryDelay time.Duration

	// Deliver, when set, sends each outbox message, in order; a message it
	// returns an error for is tried again after a backoff. nil leaves the
	// messages to a client, which acks them. See [Webhook].
	Deliver func(context.Context, Message) error
}

// Manager runs sessions: it takes their events from the inbox, runs their
// turns, fires their timers and delivers what they say.
//
// Every method is safe to call from any goroutine.
type Manager struct {
	cfg     ManagerConfig
	store   Store
	started time.Time

	// ctx ends with Close, and with it the loops and the store calls they
	// make. A turn settles on a context of its own: being stopped is when it
	// most needs to write down that it was.
	ctx    context.Context
	cancel context.CancelFunc
	loops  sync.WaitGroup
	turns  sync.WaitGroup

	kick     chan struct{} // the worker has something to look at
	dispatch chan struct{} // the dispatcher has something to send

	mu       sync.Mutex
	sessions map[string]*Session
	// busy are the sessions a turn has been claimed for, from the claim to
	// the settlement.
	busy map[string]bool
	// deleted remembers a deletion for long enough that a load racing it
	// cannot bring the session back into memory.
	deleted map[string]time.Time
	// posted is closed, and replaced, whenever the outbox grows.
	posted chan struct{}
	closed bool
}

// NewManager returns a Manager, and starts it: it puts back the events a
// daemon that did not stop cleanly left running, and starts the worker, the
// timers, the dispatcher and the sweeper.
//
// The caller must call [Manager.Close], and closes the Store itself after.
func NewManager(cfg ManagerConfig) (*Manager, error) {
	if cfg.Build == nil {
		return nil, errors.New("daemon: ManagerConfig.Build is required")
	}
	if cfg.Store == nil {
		cfg.Store = MemoryStore()
	}
	if cfg.Idle <= 0 {
		cfg.Idle = DefaultIdle
	}
	if cfg.BufferSize <= 0 {
		cfg.BufferSize = DefaultBufferSize
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	if cfg.Poll <= 0 {
		cfg.Poll = DefaultPoll
	}
	if cfg.MaxRuns <= 0 {
		cfg.MaxRuns = DefaultMaxRuns
	}
	if cfg.Attempts <= 0 {
		cfg.Attempts = DefaultAttempts
	}
	if cfg.RetryDelay <= 0 {
		cfg.RetryDelay = DefaultRetryDelay
	}
	if cfg.Lease <= 0 {
		cfg.Lease = DefaultLease
	}
	if cfg.Name == "" {
		cfg.Name = "wombatd-" + newID()[:12]
	}

	// The turns this daemon was running when it last stopped, under this
	// name, died with it; nobody else's are touched.
	ctx, cancel := context.WithTimeout(context.Background(), storeTimeout)
	n, err := cfg.Store.Recover(ctx, cfg.Name, cfg.Attempts)
	cancel()
	if err != nil {
		return nil, fmt.Errorf("daemon: recovering the inbox: %w", err)
	}
	if n > 0 {
		cfg.Logger.Info("events this daemon left running when it stopped are back in the inbox", "events", n)
	}

	m := &Manager{
		cfg:      cfg,
		store:    cfg.Store,
		started:  time.Now(),
		kick:     make(chan struct{}, 1),
		dispatch: make(chan struct{}, 1),
		sessions: map[string]*Session{},
		busy:     map[string]bool{},
		deleted:  map[string]time.Time{},
		posted:   make(chan struct{}),
	}
	m.ctx, m.cancel = context.WithCancel(context.Background())
	m.loops.Add(4)
	go m.work()
	go m.schedule()
	go m.sweep()
	go m.renew()
	if cfg.Deliver != nil {
		m.loops.Add(1)
		go m.deliver()
	}
	return m, nil
}

// ===== Sessions =====

// Create makes a session. It runs nothing: a turn needs an event, and
// [Session.Send] is the one a person sends.
func (m *Manager) Create(ctx context.Context, opts SessionOptions) (*Session, error) {
	return m.create(ctx, opts, "")
}

func (m *Manager) create(ctx context.Context, opts SessionOptions, thread string) (*Session, error) {
	if m.isClosed() {
		return nil, ErrClosed
	}
	agent, err := m.cfg.Build(opts)
	if err != nil {
		return nil, fmt.Errorf("daemon: building the agent for this session: %w", err)
	}
	now := time.Now()
	s := m.newSession(SessionRecord{
		Version: RecordVersion, ID: newID(), Thread: thread, Options: opts,
		Created: now, State: Idle, Lease: m.leaseAt(now),
	}, agent)
	s.updated = s.created
	rec := s.record()
	rec.Lease = m.leaseAt(now) // made here, so held here
	if err := m.store.CreateSession(ctx, rec); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	m.sessions[s.id] = s
	m.cfg.Logger.Info("session created", "session", s.id, "title", opts.Title, "thread", thread)
	return s, nil
}

// errBuild marks a session whose agent cannot be built any more: its
// workspace is gone, its options are no longer allowed.
var errBuild = errors.New("daemon: cannot rebuild the session's agent")

// Get returns a session, loading it from the store if it is not in memory —
// and taking it, so that this daemon runs it from now on. A session another
// daemon runs is an *[OwnedError], which names that daemon.
func (m *Manager) Get(ctx context.Context, id string) (*Session, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, ErrClosed
	}
	if s, ok := m.sessions[id]; ok {
		m.mu.Unlock()
		return s, nil
	}
	m.mu.Unlock()

	// Held first, loaded after: what is loaded is then what the last owner
	// left, and no other daemon can change it until this one lets go.
	now := time.Now()
	n, err := m.store.Acquire(ctx, id, m.leaseAt(now), now, m.cfg.Attempts)
	if err != nil {
		return nil, err
	}
	if n > 0 {
		m.cfg.Logger.Info("took over a session whose daemon stopped; its events are back in the inbox", "session", id, "events", n)
	}
	s, err := m.load(ctx, id)
	if err != nil {
		m.mu.Lock()
		_, loaded := m.sessions[id] // by a Get that raced this one, and won
		m.mu.Unlock()
		if !loaded {
			m.release(id)
		}
		return nil, err
	}
	return s, nil
}

// load builds a session this daemon now holds from its record.
func (m *Manager) load(ctx context.Context, id string) (*Session, error) {
	rec, err := m.store.LoadSession(ctx, id)
	if err != nil {
		return nil, err
	}
	if rec.State == Running {
		// A turn was running when the daemon that ran it died; its events
		// are back in the inbox, and will run again. Fixed here and not in
		// the store: the next turn saves it, and a save from here could land
		// on top of one from a turn this process has started since.
		rec = interrupted(rec)
		rec.Events += crashGap
	}
	agent, err := m.cfg.Build(rec.Options)
	if err != nil {
		return nil, fmt.Errorf("%w %s: %w", errBuild, id, err)
	}
	s := m.newSession(rec, agent)

	m.mu.Lock()
	defer m.mu.Unlock()
	switch {
	case m.closed:
		return nil, ErrClosed
	case !m.deleted[id].IsZero():
		return nil, fmt.Errorf("%w: %s", ErrNoSuchSession, id)
	}
	if other, ok := m.sessions[id]; ok {
		return other, nil
	}
	m.sessions[id] = s
	return s, nil
}

// interrupted is a record saved as running, as it stands now that the turn is
// not: still stopped on its decision if it was resuming one, failed
// otherwise.
func interrupted(rec SessionRecord) SessionRecord {
	if rec.Deferred != nil {
		rec.State = Approving
		return rec
	}
	rec.State, rec.Outcome, rec.Answer = Failed, "", ""
	rec.Error, rec.ErrorKind = "the daemon stopped while this turn was running; its events will run again", KindInterrupted
	return rec
}

// List returns every session, newest first.
func (m *Manager) List(ctx context.Context) ([]SessionInfo, error) {
	recs, err := m.store.ListSessions(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]SessionInfo, 0, len(recs))
	for _, rec := range recs {
		m.mu.Lock()
		s, ok := m.sessions[rec.ID]
		m.mu.Unlock()
		// Info takes the session's own lock, so it is called outside the
		// manager's.
		if ok {
			out = append(out, s.Info())
			continue
		}
		if rec.State == Running && !m.heldElsewhere(rec.Lease) {
			rec = interrupted(rec)
		}
		out = append(out, infoOf(rec))
	}
	slices.Reverse(out)
	return out, nil
}

// Delete cancels a session's turn, if it has one, and forgets the session:
// its record, its events, its timers and its messages.
func (m *Manager) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	s, ok := m.sessions[id]
	delete(m.sessions, id)
	m.deleted[id] = time.Now()
	m.mu.Unlock()
	if ok {
		// Outside the lock: drop waits for the turn to settle.
		s.drop()
	}
	if err := m.store.DeleteSession(ctx, id); err != nil {
		return err
	}
	m.cfg.Logger.Info("session deleted", "session", id)
	return nil
}

// forget drops s from memory, if it is still the one there.
func (m *Manager) forget(s *Session) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sessions[s.id] == s {
		delete(m.sessions, s.id)
	}
}

// ===== Events =====

// Inbound is an event from outside, and where it goes: to the session named,
// or to the one tied to the thread — made, with Options, if there is none —
// or, with neither, to a new session.
type Inbound struct {
	Session string
	Thread  string
	Options SessionOptions

	Kind     string
	Source   string
	From     string // who, at Source, sent it; optional
	DedupKey string
	Payload  json.RawMessage
}

// validName is what a kind, a source and a thread may look like: short, and
// nothing that needs escaping wherever it is shown.
var validName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:/#@-]{0,127}$`)

// Receive puts an event from outside in a session's inbox, and reports
// whether it was new: an event with the Source and DedupKey of one received
// before is not received again, and that one is returned.
//
// Approvals and timers are not received this way: they have their own doors,
// [Session.Resolve] and [Manager.AddTimer], which check what they are about.
func (m *Manager) Receive(ctx context.Context, in Inbound) (Event, bool, error) {
	if in.Source == "" {
		in.Source = SourceAPI
	}
	switch {
	case !validName.MatchString(in.Kind):
		return Event{}, false, fmt.Errorf("%w: kind %q is not a name", ErrBadRequest, in.Kind)
	case in.Kind == KindApproval || in.Kind == KindTimer:
		return Event{}, false, fmt.Errorf("%w: a %s is not posted as an event", ErrBadRequest, in.Kind)
	case !validName.MatchString(in.Source):
		return Event{}, false, fmt.Errorf("%w: source %q is not a name", ErrBadRequest, in.Source)
	case in.Thread != "" && !validName.MatchString(in.Thread):
		return Event{}, false, fmt.Errorf("%w: thread %q is not a name", ErrBadRequest, in.Thread)
	case len(in.DedupKey) > 256:
		return Event{}, false, fmt.Errorf("%w: dedup_key is longer than 256 bytes", ErrBadRequest)
	case len(in.Payload) > 0 && !json.Valid(in.Payload):
		return Event{}, false, fmt.Errorf("%w: payload is not JSON", ErrBadRequest)
	}
	if err := CheckFrom(in.From); err != nil {
		return Event{}, false, err
	}
	if in.Kind == KindMessage {
		if text, ok := messageText(Event{Payload: in.Payload}); !ok || strings.TrimSpace(text) == "" {
			return Event{}, false, fmt.Errorf(`%w: a message's payload is {"text": "..."}`, ErrBadRequest)
		}
	}

	// The session is not taken: an event goes in its inbox, and whichever
	// daemon runs it — this one, or another — takes it from there.
	id := in.Session
	made := false
	var err error
	switch {
	case in.Session != "":
	case in.Thread != "":
		id, made, err = m.threadSession(ctx, in.Thread, in.Options)
	default:
		var s *Session
		if s, err = m.create(ctx, in.Options, ""); err == nil {
			id, made = s.id, true
		}
	}
	if err != nil {
		return Event{}, false, err
	}
	if state, err := m.stateOf(ctx, id); err != nil {
		return Event{}, false, err
	} else if state == Done {
		return Event{}, false, ErrDone
	}
	e, created, err := m.enqueue(ctx, Event{
		Session: id, Kind: in.Kind, Source: in.Source, From: in.From, DedupKey: in.DedupKey, Payload: in.Payload,
	})
	if err == nil && !created && made {
		// A delivery of an event received before, which went to the session
		// it went to then: the one just made for it has nothing in it.
		_ = m.Delete(ctx, id)
	}
	return e, created, err
}

// stateOf is where a session stands; see [Manager.Peek].
func (m *Manager) stateOf(ctx context.Context, id string) (State, error) {
	info, err := m.Peek(ctx, id)
	return info.State, err
}

// Peek is a session's info, from memory if this daemon runs it and from the
// store if not — without taking it.
func (m *Manager) Peek(ctx context.Context, id string) (SessionInfo, error) {
	m.mu.Lock()
	s, ok := m.sessions[id]
	m.mu.Unlock()
	if ok {
		return s.Info(), nil
	}
	rec, err := m.store.LoadSession(ctx, id)
	if err != nil {
		return SessionInfo{}, err
	}
	if rec.State == Running && !m.heldElsewhere(rec.Lease) {
		rec = interrupted(rec)
	}
	return infoOf(rec), nil
}

// SendAs puts a message in a session's inbox, for its next turn, wherever the
// session runs: [Session.SendAs] when this daemon holds it, and otherwise
// straight into the store, where the daemon that runs the session takes it —
// or, should that one have died, the one that takes the session over. A
// message needs nothing of the session's memory, so it is never refused for
// where the session is.
func (m *Manager) SendAs(ctx context.Context, session, from, prompt string) (Event, error) {
	m.mu.Lock()
	s, ok := m.sessions[session]
	m.mu.Unlock()
	if ok {
		return s.SendAs(ctx, from, prompt)
	}
	if err := CheckFrom(from); err != nil {
		return Event{}, err
	}
	if state, err := m.stateOf(ctx, session); err != nil {
		return Event{}, err
	} else if state == Done {
		return Event{}, ErrDone
	}
	payload, _ := json.Marshal(MessagePayload{Text: prompt})
	e, _, err := m.enqueue(ctx, Event{Session: session, Kind: KindMessage, Source: SourceAPI, From: from, Payload: payload})
	return e, err
}

// threadSession finds the session tied to thread, or makes it, and says
// which it did.
func (m *Manager) threadSession(ctx context.Context, thread string, opts SessionOptions) (string, bool, error) {
	for range 2 {
		id, err := m.store.SessionByThread(ctx, thread)
		if err == nil {
			return id, false, nil
		}
		if !errors.Is(err, ErrNoSuchSession) {
			return "", false, err
		}
		s, err := m.create(ctx, opts, thread)
		if err == nil {
			return s.id, true, nil
		}
		if !errors.Is(err, ErrThreadTaken) {
			return "", false, err
		}
		// Another event for the thread made its session first: use that.
	}
	return "", false, fmt.Errorf("%w: %q", ErrThreadTaken, thread)
}

// enqueue puts an event in the inbox and wakes the worker.
func (m *Manager) enqueue(ctx context.Context, e Event) (Event, bool, error) {
	if m.isClosed() {
		return Event{}, false, ErrClosed
	}
	e.Created = time.Now()
	got, created, err := m.store.Enqueue(ctx, e)
	if err == nil && created {
		m.poke(m.kick)
	}
	return got, created, err
}

// Inbox lists events, newest first.
func (m *Manager) Inbox(ctx context.Context, f EventFilter) ([]Event, error) {
	return m.store.Events(ctx, f)
}

// ===== Timers =====

// AddTimer sets a timer for a session: to fire at t.Due, and — with
// t.Every set — every Every after.
func (m *Manager) AddTimer(ctx context.Context, t Timer) (Timer, error) {
	t.Note = strings.TrimSpace(t.Note)
	switch {
	case t.Note == "":
		return Timer{}, fmt.Errorf("%w: a timer needs a note, to say what it is for", ErrBadRequest)
	case len(t.Note) > 4000:
		return Timer{}, fmt.Errorf("%w: a timer's note is at most 4000 bytes", ErrBadRequest)
	case t.Due.IsZero():
		return Timer{}, fmt.Errorf("%w: a timer needs a time to fire at", ErrBadRequest)
	case t.Every < 0 || (t.Every > 0 && t.Every < MinEvery):
		return Timer{}, fmt.Errorf("%w: a timer repeats every %v at the most often", ErrBadRequest, MinEvery)
	}
	state, err := m.stateOf(ctx, t.Session)
	if err != nil {
		return Timer{}, err
	}
	if state == Done {
		return Timer{}, ErrDone
	}
	active, err := m.store.Timers(ctx, TimerFilter{Session: t.Session, Status: TimerActive})
	if err != nil {
		return Timer{}, err
	}
	if len(active) >= MaxTimers {
		return Timer{}, fmt.Errorf("%w: the session has %d timers set, which is the most it may", ErrBadRequest, len(active))
	}
	if t.CreatedBy == "" {
		t.CreatedBy = SourceAPI
	}
	t.Created = time.Now()
	return m.store.AddTimer(ctx, t)
}

// CancelTimer cancels an active timer — of that session, when session is
// not empty.
func (m *Manager) CancelTimer(ctx context.Context, id int64, session string) error {
	return m.store.CancelTimer(ctx, id, session)
}

// Timers lists timers, the soonest due first.
func (m *Manager) Timers(ctx context.Context, f TimerFilter) ([]Timer, error) {
	return m.store.Timers(ctx, f)
}

// ===== The outbox =====

// Outbox lists messages, oldest first. With wait, a list that would be empty
// waits up to that long for a message to arrive: a client long-polls with it,
// passing the last id it has as f.After.
func (m *Manager) Outbox(ctx context.Context, f OutboxFilter, wait time.Duration) ([]Message, error) {
	deadline := time.Now().Add(wait)
	for {
		// The channel is taken before the store is read, so a message posted
		// between the read and the wait still wakes it.
		m.mu.Lock()
		posted := m.posted
		m.mu.Unlock()
		msgs, err := m.store.Outbox(ctx, f)
		left := time.Until(deadline)
		if err != nil || len(msgs) > 0 || left <= 0 {
			return msgs, err
		}
		// Woken by a message this daemon posts, or at the next poll for one
		// another daemon sharing the store did.
		t := time.NewTimer(min(left, m.cfg.Poll))
		select {
		case <-posted:
		case <-t.C:
		case <-ctx.Done():
		case <-m.ctx.Done():
		}
		t.Stop()
		if ctx.Err() != nil || m.ctx.Err() != nil {
			return nil, nil
		}
	}
}

// Ack marks a message delivered: a client has handled it.
func (m *Manager) Ack(ctx context.Context, id int64) error {
	return m.store.Delivered(ctx, id, time.Now())
}

// post adds a message to the outbox, and tells whoever is waiting for one.
func (m *Manager) post(ctx context.Context, msg Message) (Message, bool, error) {
	msg.Created = time.Now()
	got, created, err := m.store.Post(ctx, msg)
	if err == nil && created {
		m.posts()
	}
	return got, created, err
}

// posts wakes the outbox's long polls and the dispatcher.
func (m *Manager) posts() {
	m.mu.Lock()
	close(m.posted)
	m.posted = make(chan struct{})
	m.mu.Unlock()
	m.poke(m.dispatch)
}

// ===== Lifecycle =====

// Close stops the loops and every turn, and waits for them. A turn it stops
// puts its events back in the inbox, so the next daemon runs them. Idempotent.
func (m *Manager) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	live := slices.Collect(maps.Values(m.sessions))
	clear(m.sessions)
	m.mu.Unlock()

	m.cancel()
	m.loops.Wait()
	for _, s := range live {
		s.close()
	}
	m.turns.Wait()
	// Settled, every one: another daemon may take them at once.
	ids := make([]string, 0, len(live))
	for _, s := range live {
		ids = append(ids, s.id)
	}
	m.release(ids...)
	return nil
}

// Name is this daemon's name among those sharing its store.
func (m *Manager) Name() string { return m.cfg.Name }

// leaseAt is the hold this daemon takes, or renews, at now.
func (m *Manager) leaseAt(now time.Time) Lease {
	return Lease{Owner: m.cfg.Name, Addr: m.cfg.Addr, Until: now.Add(m.cfg.Lease)}
}

// heldElsewhere reports a lease another daemon holds, live now.
func (m *Manager) heldElsewhere(l Lease) bool {
	return l.Owner != "" && l.Owner != m.cfg.Name && l.Until.After(time.Now())
}

// release gives up this daemon's hold on sessions, best effort: a hold not
// given up runs out on its own.
func (m *Manager) release(ids ...string) {
	if len(ids) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), storeTimeout)
	defer cancel()
	if err := m.store.Release(ctx, m.cfg.Name, ids); err != nil {
		m.cfg.Logger.Warn("daemon: releasing sessions failed; their leases will run out", "sessions", len(ids), "err", err)
	}
}

// Uptime is how long this Manager has been running, for GET /api/health.
func (m *Manager) Uptime() time.Duration { return time.Since(m.started) }

func (m *Manager) isClosed() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.closed
}

func (m *Manager) logger() *slog.Logger { return m.cfg.Logger }

// poke wakes a loop without blocking: one wake-up pending is as good as ten.
func (m *Manager) poke(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}

// settle writes a turn's settlement, on a context of its own: a turn being
// stopped is when it most needs to write down that it was.
func (m *Manager) settle(s Settlement) error {
	ctx, cancel := context.WithTimeout(context.Background(), storeTimeout)
	defer cancel()
	err := m.store.Settle(ctx, s)
	if errors.Is(err, ErrOwned) {
		m.cfg.Logger.Warn("daemon: another daemon took this session over while its turn ran; the turn is dropped, and its events run there",
			"session", s.Session.ID)
		return err
	}
	if err != nil && !errors.Is(err, ErrNoSuchSession) {
		m.cfg.Logger.Warn("daemon: settling a turn failed; its events will run again when the daemon restarts",
			"session", s.Session.ID, "err", err)
	}
	if err == nil && len(s.Outbox) > 0 {
		m.posts()
	}
	return err
}

// save writes a session's record, best effort: what is lost is only what a
// crash before the next save would have needed.
func (m *Manager) save(rec SessionRecord) {
	ctx, cancel := context.WithTimeout(context.Background(), storeTimeout)
	defer cancel()
	if err := m.store.SaveSession(ctx, rec); err != nil && !errors.Is(err, ErrNoSuchSession) && !errors.Is(err, ErrOwned) {
		m.cfg.Logger.Warn("daemon: saving a session failed", "session", rec.ID, "err", err)
	}
}

// newID is a session identifier: opaque, unguessable, and short enough to
// paste into a URL. Unguessable matters — the id is the only thing standing
// between one client and another's conversation, transcript and approvals.
func newID() string {
	var b [16]byte
	rand.Read(b[:]) // crypto/rand.Read cannot fail as of Go 1.24
	return hex.EncodeToString(b[:])
}
