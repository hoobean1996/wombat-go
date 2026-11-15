package httpapi

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"sync"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
)

// Defaults [NewManager] fills in for a zero-valued [ManagerConfig] field.
const (
	// DefaultTTL is how long an untouched session survives.
	//
	// Measured from the last activity rather than from the end of the last
	// turn, because a conversation nobody came back to is as abandoned as one
	// that finished. It has to cover the worst reconnect a browser actually
	// performs, not the typical one: EventSource retries in seconds, but a
	// phone that changed network or a laptop whose lid was shut comes back
	// minutes later, and a client that reconnects to a reaped session sees its
	// whole conversation vanish.
	DefaultTTL = 30 * time.Minute

	// DefaultMax bounds concurrent sessions.
	DefaultMax = 32

	// DefaultBufferSize is how many events one session retains. Sized for a
	// conversation rather than a screenful: a reasoning model emits hundreds
	// of deltas per turn, and a client that reconnects should get what it
	// missed rather than a window.
	DefaultBufferSize = 16384

	// The reaper's period is TTL/10, bounded by these, so expiry is accurate to
	// a tenth of the TTL whether that TTL is half an hour or — in a test — a
	// moment. One goroutine for the whole manager, rather than a timer per
	// session.
	//
	// Derived rather than fixed because a constant cannot serve both ends: 30
	// seconds is right for the default TTL and makes a short one unobservable,
	// which is how a reaper ends up shipped untested.
	minSweep = 10 * time.Millisecond
	maxSweep = 30 * time.Second
)

// ErrDone reports a [Session.Send] to a conversation that has ended.
//
// Only [Done] is final, and only a terminal tool produces it. A FAILED turn
// does not end the conversation: a rate limit, a blown budget or a cancelled
// request must not cost the user the transcript, so the session stays
// continuable and another Send picks it up where it stopped.
//
// [New] should report it as 409 with kind "session_done": like [ErrBusy] it
// means "not in this state", not "not found" and not "you sent nonsense".
var ErrDone = errors.New("httpapi: the session has finished")

// ManagerConfig configures a [Manager].
type ManagerConfig struct {
	// Build makes the agent for one session. Required.
	//
	// Per session and not once per process, because the interesting choices —
	// which directory the file tools are rooted at, which permission policy
	// gates them — are per session. A [Manager] never inspects the options it
	// passes: clamping what a client asked for against what the deployment
	// allows is the host's job, and the host is the only party that knows.
	Build func(SessionOptions) (*wombat.Agent, error)

	// Limits caps ONE TURN. A long conversation is bounded per exchange; a
	// cumulative cap would kill a session mid-sentence on its fourth question
	// for reasons the user cannot see.
	Limits governor.Limits

	// TTL is how long an untouched session stays readable. Default DefaultTTL.
	TTL time.Duration

	// Max is the number of concurrent sessions. Default DefaultMax.
	Max int

	// BufferSize is how many events one session retains.
	// Default DefaultBufferSize.
	BufferSize int

	// Logger receives lifecycle lines. Default: a handler that discards.
	Logger *slog.Logger

	// Store, when set, keeps every session across restarts: each is saved
	// when it is created and at the start and end of every turn, and
	// [NewManager] restores whatever the store holds. nil keeps sessions in
	// memory only. See [Store] for what a restart does and does not keep.
	Store Store
}

// Manager owns sessions.
//
// It is HTTP-free on purpose. Everything interesting here is a lifecycle
// problem — a turn that outlives the request that started it, a stream that
// resumes across a turn boundary, a reaper that must not evict something a
// person is still reading — and a lifecycle you can only test through a socket
// is one nobody tests.
//
// Every method is safe to call from any goroutine.
type Manager struct {
	cfg     ManagerConfig
	started time.Time
	stop    chan struct{}
	swept   chan struct{}

	mu       sync.Mutex
	sessions map[string]*Session
	// order is creation order, so a list is stable and a UI's sidebar does not
	// reshuffle itself on every poll.
	order  []string
	closed bool
}

// NewManager returns a Manager and starts its reaper.
//
// The caller must call [Manager.Close]; until it does, one goroutine and every
// live session's run are still going. Reports an error rather than panicking
// on a missing Build because the config usually comes from flags, and a
// process that exits with a message beats one that prints a stack.
func NewManager(cfg ManagerConfig) (*Manager, error) {
	if cfg.Build == nil {
		return nil, errors.New("httpapi: ManagerConfig.Build is required")
	}
	if cfg.TTL <= 0 {
		cfg.TTL = DefaultTTL
	}
	if cfg.Max <= 0 {
		cfg.Max = DefaultMax
	}
	if cfg.BufferSize <= 0 {
		cfg.BufferSize = DefaultBufferSize
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}

	m := &Manager{
		cfg:      cfg,
		started:  time.Now(),
		stop:     make(chan struct{}),
		swept:    make(chan struct{}),
		sessions: map[string]*Session{},
	}
	if cfg.Store != nil {
		m.restore()
	}
	go m.sweep()
	return m, nil
}

// restore brings back what the store holds. A session that has expired is
// forgotten; one whose agent cannot be built any more — its workspace is gone,
// its options are no longer allowed — is skipped and kept, so a fixed
// configuration can still restore it.
func (m *Manager) restore() {
	recs, err := m.cfg.Store.Load()
	if err != nil {
		m.cfg.Logger.Warn("session store: some sessions could not be read", "err", err)
	}
	slices.SortFunc(recs, func(a, b SessionRecord) int { return a.Created.Compare(b.Created) })
	for _, rec := range recs {
		if rec.Version > RecordVersion {
			// From a Store that does not check, which DirStore does. Left
			// where it is, expiry included: its layout is not this one.
			m.cfg.Logger.Warn("session store: a session saved by a newer version is left on disk",
				"session", rec.ID, "version", rec.Version, "reads", RecordVersion)
			continue
		}
		if time.Since(rec.Updated) >= m.cfg.TTL {
			_ = m.cfg.Store.Delete(rec.ID)
			continue
		}
		if len(m.sessions) >= m.cfg.Max {
			m.cfg.Logger.Warn("session store: more saved sessions than -max allows; the rest stay on disk", "max", m.cfg.Max)
			break
		}
		agent, err := m.cfg.Build(rec.Options)
		if err != nil {
			m.cfg.Logger.Warn("session store: cannot rebuild a saved session's agent", "session", rec.ID, "err", err)
			continue
		}
		s := m.newSession(rec.ID, agent, rec.Options, rec.Created)
		s.updated = rec.Updated
		s.turns, s.state, s.msgs, s.spend = rec.Turns, rec.State, rec.Messages, rec.Spend
		s.outcome, s.answer, s.errText, s.errKind, s.pauseID = rec.Outcome, rec.Answer, rec.Error, rec.ErrorKind, rec.PauseID
		s.base = rec.Events
		if s.state == Running || s.state == Approving {
			// The turn died with the process that ran it. Its transcript is
			// the one it started from, so the next prompt carries on from
			// there — Then closes the tool call it may have left open.
			s.state, s.outcome, s.answer = Failed, "", ""
			s.errText, s.errKind = "the server restarted while this turn was running", KindInterrupted
		}
		m.sessions[s.id] = s
		m.order = append(m.order, s.id)
	}
	if n := len(m.sessions); n > 0 {
		m.cfg.Logger.Info("sessions restored", "count", n)
	}
}

func (m *Manager) newSession(id string, agent *wombat.Agent, opts SessionOptions, created time.Time) *Session {
	return &Session{
		id:        id,
		m:         m,
		agent:     agent,
		opts:      opts,
		created:   created,
		updated:   created,
		bufCap:    m.cfg.BufferSize,
		approvals: newApprovals(),
		state:     Idle,
		changed:   make(chan struct{}),
	}
}

// Create makes a session. It does NOT start a turn — [Session.Send] does — so
// that a client can hold an id before the user has finished typing.
func (m *Manager) Create(opts SessionOptions) (*Session, error) {
	agent, err := m.cfg.Build(opts)
	if err != nil {
		return nil, fmt.Errorf("httpapi: building the agent for this session: %w", err)
	}

	s := m.newSession(newID(), agent, opts, time.Now())

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, ErrClosed
	}
	if len(m.sessions) >= m.cfg.Max {
		m.mu.Unlock()
		return nil, fmt.Errorf("%w (%d live)", ErrTooManySessions, m.cfg.Max)
	}
	m.sessions[s.id] = s
	m.order = append(m.order, s.id)
	m.mu.Unlock()

	m.cfg.Logger.Info("session created", "session", s.id, "title", opts.Title)
	s.mu.Lock()
	rec, seq := s.recordLocked(nil)
	s.mu.Unlock()
	s.persist(rec, seq)
	return s, nil
}

// Get looks a session up by id.
func (m *Manager) Get(id string) (*Session, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	return s, ok
}

// List returns every live session, newest first — the order a sidebar wants,
// because the one you are working in is the one you just made.
func (m *Manager) List() []SessionInfo {
	m.mu.Lock()
	live := make([]*Session, 0, len(m.sessions))
	for _, id := range m.order {
		if s, ok := m.sessions[id]; ok {
			live = append(live, s)
		}
	}
	m.mu.Unlock()

	// Info takes the session's own lock, so it is called outside the
	// manager's: holding both would order two locks that are otherwise
	// unordered, and a Session never reaches for the Manager's.
	out := make([]SessionInfo, 0, len(live))
	for _, s := range live {
		out = append(out, s.Info())
	}
	slices.Reverse(out)
	return out
}

// Delete cancels a session and drops it, reporting whether it existed.
//
// Disconnecting cancels nothing — that is the point of a session outliving its
// connection — so cancellation had to become explicit somewhere, and this is
// it.
func (m *Manager) Delete(id string) bool {
	m.mu.Lock()
	s, ok := m.sessions[id]
	if ok {
		delete(m.sessions, id)
		m.order = slices.DeleteFunc(m.order, func(x string) bool { return x == id })
	}
	m.mu.Unlock()

	if !ok {
		return false
	}
	// Outside the lock: close waits for a goroutine, and holding the registry
	// while doing that would block every other request.
	s.drop()
	m.cfg.Logger.Info("session dropped", "session", id)
	return true
}

// Close stops the reaper and every session, and waits for their runs to exit.
// Idempotent.
func (m *Manager) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	live := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		live = append(live, s)
	}
	clear(m.sessions)
	m.order = nil
	m.mu.Unlock()

	close(m.stop)
	<-m.swept
	for _, s := range live {
		s.close()
	}
	return nil
}

// Uptime is how long this Manager has been running, for GET /api/health.
func (m *Manager) Uptime() time.Duration { return time.Since(m.started) }

func (m *Manager) logger() *slog.Logger { return m.cfg.Logger }

// sweep evicts sessions nobody has touched for a TTL.
//
// A session with a turn in flight is never swept: it ends on its own, because
// every turn is capped by the per-turn budget and the iteration limit, and
// evicting one mid-run would bill the user for work they can no longer read.
func (m *Manager) sweep() {
	defer close(m.swept)
	t := time.NewTicker(min(max(m.cfg.TTL/10, minSweep), maxSweep))
	defer t.Stop()

	for {
		select {
		case <-m.stop:
			return
		case now := <-t.C:
			m.reap(now)
		}
	}
}

func (m *Manager) reap(now time.Time) {
	m.mu.Lock()
	var dead []*Session
	for id, s := range m.sessions {
		updated, idle := s.idleSince()
		if !idle || now.Sub(updated) < m.cfg.TTL {
			continue
		}
		delete(m.sessions, id)
		m.order = slices.DeleteFunc(m.order, func(x string) bool { return x == id })
		dead = append(dead, s)
	}
	m.mu.Unlock()

	for _, s := range dead {
		s.drop()
		m.cfg.Logger.Info("session expired", "session", s.id, "events", s.Info().Events)
	}
}

// newID is a session identifier: opaque, unguessable, and short enough to
// paste into a URL. Unguessable matters — the id is the only thing standing
// between one browser and another's conversation, transcript and approvals.
func newID() string {
	var b [16]byte
	rand.Read(b[:]) // crypto/rand.Read cannot fail as of Go 1.24
	return hex.EncodeToString(b[:])
}
