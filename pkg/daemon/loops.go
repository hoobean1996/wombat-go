package daemon

import (
	"context"
	"errors"
	"maps"
	"slices"
	"time"
)

// The six loops a Manager runs: the worker, which turns events into turns;
// the timers, which turn time into events; the dispatcher, which sends the
// outbox; the sweeper, which drops what memory no longer needs; the renewer,
// which keeps this daemon's hold on the sessions it has; and the pruner, which
// forgets the tool calls of the ledger once they are old enough.

// work claims what is ready, whenever it is told there is something and at
// every poll.
func (m *Manager) work() {
	defer m.loops.Done()
	t := time.NewTicker(m.cfg.Poll)
	defer t.Stop()
	for {
		m.claim()
		select {
		case <-m.ctx.Done():
			return
		case <-m.kick:
		case <-t.C:
		}
	}
}

// claim starts a turn for every session with events ready, as many as there
// is room for. One goroutine does this, so a session is claimed once by this
// daemon — busy is how the sessions already running are left out — and the
// store's leases see that no other daemon claims it meanwhile.
func (m *Manager) claim() {
	m.mu.Lock()
	free := m.cfg.MaxRuns - len(m.busy)
	busy := slices.Collect(maps.Keys(m.busy))
	m.mu.Unlock()
	if free <= 0 {
		return
	}
	now := time.Now()
	ready, err := m.store.Ready(m.ctx, now, m.cfg.Name, busy, free)
	if err != nil {
		if m.ctx.Err() == nil {
			m.cfg.Logger.Warn("daemon: reading the inbox failed", "err", err)
		}
		return
	}
	for _, id := range ready {
		// Taken before it is claimed: Claim takes events only for the
		// session's owner, and taking one a dead daemon left puts back what it
		// was running, to be claimed here.
		if _, err := m.Get(m.ctx, id); err != nil {
			if !errors.Is(err, ErrOwned) && m.ctx.Err() == nil {
				m.cfg.Logger.Warn("daemon: taking a session failed", "session", id, "err", err)
			}
			continue
		}
		batch, err := m.store.Claim(m.ctx, id, m.cfg.Name, now, MaxBatch)
		if err != nil {
			if m.ctx.Err() == nil {
				m.cfg.Logger.Warn("daemon: claiming events failed", "session", id, "err", err)
			}
			return
		}
		if len(batch) == 0 {
			continue
		}
		m.mu.Lock()
		m.busy[id] = true
		m.mu.Unlock()
		m.turns.Add(1)
		go m.handle(id, batch)
	}
}

// handle runs one claimed batch on its session.
func (m *Manager) handle(id string, batch []Event) {
	defer m.turns.Done()
	defer func() {
		m.mu.Lock()
		delete(m.busy, id)
		m.mu.Unlock()
		// The session may have had more arrive while it ran.
		m.poke(m.kick)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), storeTimeout)
	s, err := m.Get(ctx, id)
	cancel()
	switch {
	case err == nil:
		s.handle(batch)
	case errors.Is(err, ErrClosed):
		m.settle(Settlement{Retry: eventIDs(batch), Error: "the daemon stopped before this event's turn began"})
	case errors.Is(err, ErrNoSuchSession), errors.Is(err, ErrNewerRecord), errors.Is(err, errBuild):
		m.cfg.Logger.Warn("daemon: events for a session that cannot run", "session", id, "err", err)
		m.settle(Settlement{Failed: eventIDs(batch), Error: err.Error()})
	default:
		// The store, most likely: try again later rather than give up on
		// events because of a blip.
		m.cfg.Logger.Warn("daemon: loading a session failed", "session", id, "err", err)
		m.settle(Settlement{Retry: eventIDs(batch), RetryAt: time.Now().Add(m.cfg.RetryDelay), Error: err.Error()})
	}
}

// schedule fires the timers that are due.
func (m *Manager) schedule() {
	defer m.loops.Done()
	t := time.NewTicker(m.cfg.Poll)
	defer t.Stop()
	for {
		fired, err := m.store.Fire(m.ctx, time.Now())
		switch {
		case err != nil && m.ctx.Err() == nil:
			m.cfg.Logger.Warn("daemon: firing timers failed", "err", err)
		case len(fired) > 0:
			m.cfg.Logger.Info("timers fired", "events", len(fired))
			m.poke(m.kick)
		}
		select {
		case <-m.ctx.Done():
			return
		case <-t.C:
		}
	}
}

// deliver sends the outbox, in order, whenever there is something new and at
// every poll.
func (m *Manager) deliver() {
	defer m.loops.Done()
	t := time.NewTicker(m.cfg.Poll)
	defer t.Stop()
	for {
		m.deliverPending()
		select {
		case <-m.ctx.Done():
			return
		case <-m.dispatch:
		case <-t.C:
		}
	}
}

// deliverPending sends what is pending, oldest first. A message that fails
// holds back the ones behind it until it is sent or given up on: a reply that
// arrives before the question it answers is worse than one that is late.
func (m *Manager) deliverPending() {
	// One dispatcher among the daemons sharing the store, so a message goes
	// out once, and in order.
	if lead, err := m.store.Lead(m.ctx, "deliver", m.cfg.Name, time.Now().Add(m.cfg.Lease)); err != nil || !lead {
		if err != nil && m.ctx.Err() == nil {
			m.cfg.Logger.Warn("daemon: asking to lead the dispatcher failed", "err", err)
		}
		return
	}
	const page = 32
	for {
		msgs, err := m.store.Outbox(m.ctx, OutboxFilter{Status: MessagePending, Limit: page})
		if err != nil || len(msgs) == 0 {
			return
		}
		for _, msg := range msgs {
			if msg.NotBefore.After(time.Now()) {
				return
			}
			ctx, cancel := context.WithTimeout(m.ctx, deliverTimeout)
			err := m.cfg.Deliver(ctx, msg)
			cancel()
			if m.ctx.Err() != nil {
				return
			}
			if err == nil {
				if err := m.store.Delivered(m.ctx, msg.ID, time.Now()); err != nil && !errors.Is(err, ErrNoSuchMessage) {
					m.cfg.Logger.Warn("daemon: marking a message delivered failed", "message", msg.ID, "err", err)
					return
				}
				continue
			}
			var retry time.Time
			if msg.Attempts+1 < maxDeliveries {
				retry = time.Now().Add(m.backoff(msg.Attempts + 1))
			}
			m.cfg.Logger.Warn("daemon: delivering a message failed", "message", msg.ID, "attempt", msg.Attempts+1,
				"giving_up", retry.IsZero(), "err", err)
			if err := m.store.Undelivered(m.ctx, msg.ID, err.Error(), retry); err != nil || !retry.IsZero() {
				return
			}
		}
		if len(msgs) < page {
			return
		}
	}
}

// backoff is how long to wait before try n+1: RetryDelay, doubling, at most
// an hour.
func (m *Manager) backoff(n int) time.Duration {
	d := m.cfg.RetryDelay
	for i := 1; i < n && d < time.Hour; i++ {
		d *= 2
	}
	return min(d, time.Hour)
}

// sweep drops from memory the sessions that have had nothing to do for Idle.
// They stay in the store, and are loaded again when something wants them.
func (m *Manager) sweep() {
	defer m.loops.Done()
	t := time.NewTicker(min(max(m.cfg.Idle/10, minSweep), maxSweep))
	defer t.Stop()
	for {
		select {
		case <-m.ctx.Done():
			return
		case now := <-t.C:
			m.unload(now)
		}
	}
}

func (m *Manager) unload(now time.Time) {
	m.mu.Lock()
	var idle []*Session
	for id, s := range m.sessions {
		if !m.busy[id] && s.idle(now, m.cfg.Idle) {
			delete(m.sessions, id)
			idle = append(idle, s)
		}
	}
	for id, at := range m.deleted {
		if now.Sub(at) > time.Minute {
			delete(m.deleted, id)
		}
	}
	m.mu.Unlock()
	ids := make([]string, 0, len(idle))
	for _, s := range idle {
		s.close()
		ids = append(ids, s.id)
		m.cfg.Logger.Debug("session unloaded", "session", s.id)
	}
	// Out of memory, and so free for any daemon to take.
	m.release(ids...)
}

// renew keeps this daemon's hold on the sessions in its memory, every third
// of a lease. One it has lost — deleted, or taken over after a lease ran out
// while this process could not reach the store — is dropped: another daemon
// runs it now, and a turn still running here would only be refused when it
// settles.
func (m *Manager) renew() {
	defer m.loops.Done()
	t := time.NewTicker(m.cfg.Lease / 3)
	defer t.Stop()
	for {
		select {
		case <-m.ctx.Done():
			return
		case now := <-t.C:
			m.renewAt(now)
		}
	}
}

func (m *Manager) renewAt(now time.Time) {
	m.mu.Lock()
	ids := slices.Collect(maps.Keys(m.sessions))
	m.mu.Unlock()
	if len(ids) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(m.ctx, storeTimeout)
	kept, err := m.store.Renew(ctx, m.cfg.Name, ids, now.Add(m.cfg.Lease))
	cancel()
	if err != nil {
		if m.ctx.Err() == nil {
			m.cfg.Logger.Warn("daemon: renewing leases failed", "sessions", len(ids), "err", err)
		}
		return
	}
	var lost []*Session
	m.mu.Lock()
	for _, id := range ids {
		if s, ok := m.sessions[id]; ok && !slices.Contains(kept, id) {
			delete(m.sessions, id)
			lost = append(lost, s)
		}
	}
	m.mu.Unlock()
	for _, s := range lost {
		m.cfg.Logger.Warn("daemon: lost a session to another daemon, or to a deletion", "session", s.id)
		go s.close() // it waits for the turn to stop; the renewer does not
	}
}

// prune forgets, every prunePeriod, the calls of the ledger that ended
// longer than LedgerRetention ago. One daemon of those sharing the store does
// it: the one that leads "prune".
func (m *Manager) prune() {
	defer m.loops.Done()
	t := time.NewTicker(prunePeriod)
	defer t.Stop()
	for {
		m.pruneAt(time.Now())
		select {
		case <-m.ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (m *Manager) pruneAt(now time.Time) {
	ctx, cancel := context.WithTimeout(m.ctx, storeTimeout)
	defer cancel()
	if lead, err := m.store.Lead(ctx, "prune", m.cfg.Name, now.Add(2*prunePeriod)); err != nil || !lead {
		return
	}
	n, err := m.store.PruneCalls(ctx, now.Add(-m.cfg.LedgerRetention))
	switch {
	case err != nil && m.ctx.Err() == nil:
		m.cfg.Logger.Warn("daemon: pruning the ledger failed", "err", err)
	case n > 0:
		m.cfg.Logger.Info("ledger pruned", "calls", n)
	}
	n, err = m.store.PruneFrames(ctx, now.Add(-m.cfg.LedgerRetention))
	switch {
	case err != nil && m.ctx.Err() == nil:
		m.cfg.Logger.Warn("daemon: pruning the session logs failed", "err", err)
	case n > 0:
		m.cfg.Logger.Info("session logs pruned", "frames", n)
	}
}

// ===== Driven from outside =====
//
// What the loops do, one call at a time, for a host that runs none: a
// serverless function, invoked by a queue and a cron.

// wake says a session has work at at: to Wake, when the host gave one, or
// to this daemon's own loop.
func (m *Manager) wake(session string, at time.Time) {
	if m.cfg.Wake == nil {
		if !at.After(time.Now()) {
			m.poke(m.kick)
		}
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), storeTimeout)
	defer cancel()
	if err := m.cfg.Wake(ctx, session, at); err != nil {
		m.cfg.Logger.Warn("daemon: waking a session failed; a tick will find it", "session", session, "err", err)
	}
}

// Advance runs a session's next slice of work, if it has any: takes it,
// claims its events, runs the turn they make — carried on from where it
// stopped, if one is in flight — until it ends or its slice is up, and
// settles. A Stateless daemon lets the session go after. It reports whether
// the session has more work now, which has been asked for (Wake) already.
//
// A session another daemon holds, or this one is advancing already, is left
// alone: false, and no error.
func (m *Manager) Advance(ctx context.Context, id string) (bool, error) {
	m.mu.Lock()
	switch {
	case m.closed:
		m.mu.Unlock()
		return false, ErrClosed
	case m.busy[id]:
		m.mu.Unlock()
		return false, nil
	}
	m.busy[id] = true
	m.turns.Add(1)
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		delete(m.busy, id)
		m.mu.Unlock()
		m.turns.Done()
	}()

	s, err := m.take(ctx, id)
	if errors.Is(err, ErrOwned) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	batch, err := m.store.Claim(ctx, id, m.cfg.Name, time.Now(), MaxBatch)
	if err == nil && len(batch) > 0 {
		s.handle(batch)
	}
	if m.cfg.Stateless {
		s.close()
		m.release(id)
	}
	if err != nil {
		return false, err
	}
	more := m.hasWork(ctx, s)
	if more {
		m.wake(id, time.Now())
	}
	return more, nil
}

// hasWork reports whether a session has events it could take now.
func (m *Manager) hasWork(ctx context.Context, s *Session) bool {
	pending, err := m.store.Events(ctx, EventFilter{Session: s.id, Status: EventPending, Limit: MaxBatch})
	if err != nil {
		return false
	}
	now := time.Now()
	s.mu.Lock()
	awaiting := s.deferred != nil
	s.mu.Unlock()
	for _, e := range pending {
		if !e.NotBefore.After(now) && (!awaiting || e.Kind == KindApproval || e.Kind == KindResult) {
			return true
		}
	}
	return false
}

// Tick does once what the loops do continuously: fires the timers that are
// due, wakes every session with work it could take now — those whose
// daemon died included — delivers the outbox, when Deliver is set, and
// prunes the ledger. It returns the sessions it found work for. A host that
// runs no loop calls it now and then — a cron — so that nothing a lost
// wake-up left waits for longer.
func (m *Manager) Tick(ctx context.Context) ([]string, error) {
	if m.isClosed() {
		return nil, ErrClosed
	}
	now := time.Now()
	fired, err := m.store.Fire(ctx, now)
	if err != nil {
		return nil, err
	}
	for _, e := range fired {
		m.wake(e.Session, now)
	}
	m.mu.Lock()
	busy := slices.Collect(maps.Keys(m.busy))
	m.mu.Unlock()
	ready, err := m.store.Ready(ctx, now, m.cfg.Name, busy, 100)
	if err != nil {
		return nil, err
	}
	for _, id := range ready {
		m.wake(id, now)
	}
	if m.cfg.Deliver != nil {
		m.deliverPending()
	}
	if m.cfg.LedgerRetention > 0 {
		m.pruneAt(now)
	}
	return ready, nil
}
