package daemon

import (
	"context"
	"errors"
	"maps"
	"slices"
	"time"
)

// The four loops a Manager runs: the worker, which turns events into turns;
// the timers, which turn time into events; the dispatcher, which sends the
// outbox; and the sweeper, which drops what memory no longer needs.

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
// is room for. One goroutine does this, so a session is claimed once: busy is
// how the sessions already running are left out.
func (m *Manager) claim() {
	m.mu.Lock()
	free := m.cfg.MaxRuns - len(m.busy)
	busy := slices.Collect(maps.Keys(m.busy))
	m.mu.Unlock()
	if free <= 0 {
		return
	}
	now := time.Now()
	ready, err := m.store.Ready(m.ctx, now, busy, free)
	if err != nil {
		if m.ctx.Err() == nil {
			m.cfg.Logger.Warn("daemon: reading the inbox failed", "err", err)
		}
		return
	}
	for _, id := range ready {
		batch, err := m.store.Claim(m.ctx, id, now, MaxBatch)
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
	for _, s := range idle {
		s.close()
		m.cfg.Logger.Debug("session unloaded", "session", s.id)
	}
}
