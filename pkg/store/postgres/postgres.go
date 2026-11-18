// Package postgres is a [daemon.Store] in Postgres: the sessions, the
// inbox, timer and outbox tables an always-on daemon runs on, and the ledger
// of every tool call the sessions' turns make (tool_calls, and tool_runs for
// where their sub-agents had got to).
//
//	st, err := postgres.Open(ctx, "postgres://wombat:wombat@127.0.0.1:5432/wombat")
//	...
//	defer st.Close()
//	if err := st.Lock(ctx, "wombatd-1"); err != nil { ... } // one daemon per name
//
// Open creates the tables it needs, in the schema the connection's
// search_path names — add ?search_path=name to the DSN to keep one
// deployment's tables apart from another's — and is safe to run against a
// database that has them already, from any number of processes at once.
//
// Any number of daemons may share the schema, each under a name of its own:
// a session is run by the one holding its lease (daemon.Store.Acquire), a
// timer is fired by the first to lock it, and the outbox is sent by the one
// that holds the dispatcher's lock (Store.Lead).
//
// A transcript is kept as json, not jsonb, on purpose: jsonb reorders an
// object's keys, a tool call's input would come back different from what the
// model wrote, and the prompt cache that matches a transcript byte for byte
// would miss on every turn after a reload.
package postgres

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"wombat/pkg/daemon"
)

// ErrLocked reports a daemon name another daemon on the schema is running
// under.
var ErrLocked = errors.New("postgres: another daemon is running on this schema under that name")

// Store is a [daemon.Store] in Postgres.
type Store struct {
	pool *pgxpool.Pool
	cfg  *pgxpool.Config

	mu    sync.Mutex
	lock  *pgx.Conn            // holds the advisory lock Lock took
	roles map[string]*leadConn // by role: the connections Lead asks on
}

// leadConn is a connection of Lead's, and whether it holds its role's lock.
type leadConn struct {
	conn *pgx.Conn
	held bool
}

// Open connects to the database dsn names and brings its schema up to date.
func Open(ctx context.Context, dsn string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("postgres: %w", err)
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("postgres: %w", err)
	}
	st := &Store{pool: pool, cfg: cfg, roles: map[string]*leadConn{}}
	if err := st.migrate(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return st, nil
}

// Lock makes this the only daemon on the schema running under name, for as
// long as the store is open: a second one's Lock is [ErrLocked]. Sessions are
// held by name, and a daemon starting again under a name takes back the
// sessions it held — puts back what it left running (daemon.Store.Recover) —
// so two under one name would each run the other's.
//
// The lock is Postgres's, held by a connection of its own, and goes when that
// connection does — when the store is closed, or the process dies.
func (s *Store) Lock(ctx context.Context, name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lock != nil {
		return nil
	}
	conn, err := pgx.ConnectConfig(ctx, s.cfg.ConnConfig.Copy())
	if err != nil {
		return fmt.Errorf("postgres: %w", err)
	}
	var got bool
	err = conn.QueryRow(ctx, `SELECT pg_try_advisory_lock(hashtext('wombatd:' || current_schema() || ':' || $1))`, name).Scan(&got)
	if err == nil && !got {
		err = ErrLocked
	}
	if err != nil {
		conn.Close(ctx)
		return fmt.Errorf("postgres: %w", err)
	}
	s.lock = conn
	return nil
}

// Close releases the locks this store holds, and the connections.
func (s *Store) Close() error {
	s.mu.Lock()
	if s.lock != nil {
		s.lock.Close(context.Background())
		s.lock = nil
	}
	for role, lc := range s.roles {
		lc.conn.Close(context.Background())
		delete(s.roles, role)
	}
	s.mu.Unlock()
	s.pool.Close()
	return nil
}

// ===== Sessions =====

func (s *Store) CreateSession(ctx context.Context, rec daemon.SessionRecord) error {
	meta, msgs, err := daemon.EncodeRecord(rec)
	if err != nil {
		return err
	}
	inflight, err := daemon.EncodeInflight(rec.Inflight)
	if err != nil {
		return err
	}
	_, err = s.pool.Exec(ctx, `
		INSERT INTO sessions (id, thread, created, awaiting, meta, messages, owner, owner_addr, lease_until, epoch, inflight)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 1, $10)`,
		rec.ID, nullString(rec.Thread), rec.Created, rec.Deferred != nil, meta, msgs,
		nullString(rec.Lease.Owner), rec.Lease.Addr, nullTime(rec.Lease.Until), inflight)
	if isUnique(err, "sessions_thread_key") {
		return fmt.Errorf("%w: %q", daemon.ErrThreadTaken, rec.Thread)
	}
	return wrap(err)
}

func (s *Store) SaveSession(ctx context.Context, rec daemon.SessionRecord) error {
	return saveSession(ctx, s.pool, rec)
}

func saveSession(ctx context.Context, q querier, rec daemon.SessionRecord) error {
	meta, msgs, err := daemon.EncodeRecord(rec)
	if err != nil {
		return err
	}
	inflight, err := daemon.EncodeInflight(rec.Inflight)
	if err != nil {
		return err
	}
	// With an owner named, written only while that daemon holds it — under
	// the epoch named, when one is.
	tag, err := q.Exec(ctx, `
		UPDATE sessions SET awaiting = $2, meta = $3, messages = $4, inflight = $7
		WHERE id = $1 AND `+fenced(5, 6),
		rec.ID, rec.Deferred != nil, meta, msgs, rec.Lease.Owner, rec.Lease.Epoch, inflight)
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() > 0 {
		return nil
	}
	return ownedOrGone(ctx, q, rec.ID)
}

// fenced is the condition a write by a daemon is made under: with an owner
// named — $owner — only while that daemon holds the session, and with an
// epoch — $epoch — only under it.
func fenced(owner, epoch int) string {
	return fmt.Sprintf(`($%[1]d::text = '' OR (owner = $%[1]d AND ($%[2]d::bigint = 0 OR epoch = $%[2]d)))`, owner, epoch)
}

// ownedOrGone is why a fenced write of a session wrote nothing: it is gone,
// or another lease holds it.
func ownedOrGone(ctx context.Context, q querier, id string) error {
	l, err := leaseOf(ctx, q, id)
	if err != nil {
		return err
	}
	return &daemon.OwnedError{Session: id, Lease: l}
}

// leaseCols read a session's lease, as scanLease takes it.
const leaseCols = `coalesce(owner, ''), owner_addr, lease_until, epoch`

func scanLease(owner, addr string, until *time.Time, epoch int64) daemon.Lease {
	l := daemon.Lease{Owner: owner, Addr: addr, Epoch: epoch}
	if until != nil {
		l.Until = until.UTC()
	}
	return l
}

// leaseOf is who holds a session, or ErrNoSuchSession.
func leaseOf(ctx context.Context, q querier, id string) (daemon.Lease, error) {
	var owner, addr string
	var until *time.Time
	var epoch int64
	err := q.QueryRow(ctx, `SELECT `+leaseCols+` FROM sessions WHERE id = $1`, id).Scan(&owner, &addr, &until, &epoch)
	if errors.Is(err, pgx.ErrNoRows) {
		return daemon.Lease{}, fmt.Errorf("%w: %s", daemon.ErrNoSuchSession, id)
	}
	if err != nil {
		return daemon.Lease{}, wrap(err)
	}
	return scanLease(owner, addr, until, epoch), nil
}

func (s *Store) LoadSession(ctx context.Context, id string) (daemon.SessionRecord, error) {
	var meta, msgs, inflight []byte
	var owner, addr string
	var until *time.Time
	var epoch int64
	err := s.pool.QueryRow(ctx, `SELECT meta, messages, inflight, `+leaseCols+` FROM sessions WHERE id = $1`, id).
		Scan(&meta, &msgs, &inflight, &owner, &addr, &until, &epoch)
	if errors.Is(err, pgx.ErrNoRows) {
		return daemon.SessionRecord{}, fmt.Errorf("%w: %s", daemon.ErrNoSuchSession, id)
	}
	if err != nil {
		return daemon.SessionRecord{}, wrap(err)
	}
	rec, err := daemon.DecodeRecord(meta, msgs)
	if err != nil {
		return rec, err
	}
	rec.Lease = scanLease(owner, addr, until, epoch)
	rec.Inflight, err = daemon.DecodeInflight(inflight)
	return rec, err
}

func (s *Store) SessionByThread(ctx context.Context, thread string) (string, error) {
	var id string
	err := s.pool.QueryRow(ctx, `SELECT id FROM sessions WHERE thread = $1`, thread).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", fmt.Errorf("%w: thread %q", daemon.ErrNoSuchSession, thread)
	}
	return id, wrap(err)
}

func (s *Store) ListSessions(ctx context.Context) ([]daemon.SessionRecord, error) {
	rows, err := s.pool.Query(ctx, `SELECT meta, `+leaseCols+` FROM sessions ORDER BY created, id`)
	if err != nil {
		return nil, wrap(err)
	}
	type row struct {
		Meta  []byte
		Owner string
		Addr  string
		Until *time.Time
		Epoch int64
	}
	all, err := pgx.CollectRows(rows, pgx.RowToStructByPos[row])
	if err != nil {
		return nil, wrap(err)
	}
	out := make([]daemon.SessionRecord, 0, len(all))
	for _, r := range all {
		rec, err := daemon.DecodeRecord(r.Meta, nil)
		if err != nil {
			return nil, err
		}
		rec.Lease = scanLease(r.Owner, r.Addr, r.Until, r.Epoch)
		out = append(out, rec)
	}
	return out, nil
}

func (s *Store) DeleteSession(ctx context.Context, id string) error {
	tag, err := s.pool.Exec(ctx, `DELETE FROM sessions WHERE id = $1`, id)
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("%w: %s", daemon.ErrNoSuchSession, id)
	}
	return nil
}

// ===== The inbox =====

const eventCols = `id, session_id, kind, source, sender, coalesce(dedup_key, ''), payload, status, attempts, error, created, not_before`

func scanEvent(row pgx.CollectableRow) (daemon.Event, error) {
	var e daemon.Event
	err := row.Scan(&e.ID, &e.Session, &e.Kind, &e.Source, &e.From, &e.DedupKey, (*[]byte)(&e.Payload),
		&e.Status, &e.Attempts, &e.Error, &e.Created, &e.NotBefore)
	e.Created, e.NotBefore = e.Created.UTC(), e.NotBefore.UTC()
	return e, err
}

func (s *Store) Enqueue(ctx context.Context, e daemon.Event) (daemon.Event, bool, error) {
	return enqueue(ctx, s.pool, e)
}

func enqueue(ctx context.Context, q querier, e daemon.Event) (daemon.Event, bool, error) {
	created := orNow(e.Created)
	notBefore := e.NotBefore
	if notBefore.IsZero() {
		notBefore = created
	}
	rows, err := q.Query(ctx, `
		INSERT INTO inbox (session_id, kind, source, sender, dedup_key, payload, created, not_before)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (source, dedup_key) DO NOTHING
		RETURNING `+eventCols,
		e.Session, e.Kind, e.Source, e.From, nullString(e.DedupKey), nullJSON(e.Payload), created, notBefore)
	if err != nil {
		return daemon.Event{}, false, noSession(err, e.Session)
	}
	got, err := pgx.CollectRows(rows, scanEvent)
	if err != nil {
		return daemon.Event{}, false, noSession(err, e.Session)
	}
	if len(got) == 1 {
		return got[0], true, nil
	}
	rows, err = q.Query(ctx, `SELECT `+eventCols+` FROM inbox WHERE source = $1 AND dedup_key = $2`, e.Source, e.DedupKey)
	if err != nil {
		return daemon.Event{}, false, wrap(err)
	}
	old, err := pgx.CollectExactlyOneRow(rows, scanEvent)
	return old, false, wrap(err)
}

// claimable is the rule Ready and Claim share, over inbox e joined to
// sessions s; $1 is now.
const claimable = `e.status = 'pending' AND e.not_before <= $1 AND (NOT s.awaiting OR e.kind IN ('approval', 'result'))`

// takeable is a session that owner, $4, holds or could take at now, $1.
const takeable = `(s.owner IS NULL OR s.owner = $4 OR s.lease_until <= $1)`

// orphaned is a running event whose session nobody holds any more: its turn
// died with the daemon that ran it.
const orphaned = `(e.status = 'running' AND s.owner IS DISTINCT FROM $4 AND (s.owner IS NULL OR s.lease_until <= $1))`

func (s *Store) Ready(ctx context.Context, now time.Time, owner string, busy []string, limit int) ([]string, error) {
	if busy == nil {
		busy = []string{} // NULL would make the whole condition NULL
	}
	rows, err := s.pool.Query(ctx, `
		SELECT e.session_id FROM inbox e JOIN sessions s ON s.id = e.session_id
		WHERE (`+claimable+` OR `+orphaned+`) AND `+takeable+` AND NOT (e.session_id = ANY($2))
		GROUP BY e.session_id ORDER BY min(e.id) LIMIT $3`, now, busy, limit, owner)
	if err != nil {
		return nil, wrap(err)
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	return ids, wrap(err)
}

func (s *Store) Claim(ctx context.Context, session, owner string, now time.Time, limit int) ([]daemon.Event, error) {
	rows, err := s.pool.Query(ctx, `
		UPDATE inbox SET status = 'running', attempts = attempts + 1, claimed_at = $1
		WHERE id IN (
			SELECT e.id FROM inbox e JOIN sessions s ON s.id = e.session_id
			WHERE e.session_id = $2 AND s.owner = $4 AND `+claimable+`
			ORDER BY e.id LIMIT $3
			FOR UPDATE OF e SKIP LOCKED)
		RETURNING `+eventCols, now, session, limit, owner)
	if err != nil {
		return nil, wrap(err)
	}
	got, err := pgx.CollectRows(rows, scanEvent)
	slices.SortFunc(got, func(a, b daemon.Event) int { return int(a.ID - b.ID) })
	return got, wrap(err)
}

func (s *Store) Settle(ctx context.Context, set daemon.Settlement) error {
	return pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		if set.Session.ID != "" {
			if err := saveSession(ctx, tx, set.Session); err != nil {
				return err
			}
		}
		for _, m := range set.Outbox {
			if _, _, err := post(ctx, tx, m); err != nil {
				return err
			}
		}
		for _, st := range []struct {
			ids []int64
			sql string
			arg []any
		}{
			{set.Done, `UPDATE inbox SET status = 'done', error = '' WHERE id = ANY($1)`, nil},
			{set.Retry, `UPDATE inbox SET status = 'pending', error = $2, not_before = coalesce($3, not_before) WHERE id = ANY($1)`,
				[]any{set.Error, nullTime(set.RetryAt)}},
			{set.Failed, `UPDATE inbox SET status = 'failed', error = $2 WHERE id = ANY($1)`, []any{set.Error}},
		} {
			if len(st.ids) == 0 {
				continue
			}
			if _, err := tx.Exec(ctx, st.sql, append([]any{st.ids}, st.arg...)...); err != nil {
				return wrap(err)
			}
		}
		return nil
	})
}

// recoverWhere puts the running events of the sessions where names back to
// pending, or fails those tried attempts times already, and says how many
// went back. where is a condition on session_id, with $1 its argument.
func recoverWhere(ctx context.Context, tx pgx.Tx, where string, arg any, attempts int) (int, error) {
	rows, err := tx.Query(ctx, `SELECT id, attempts FROM inbox WHERE status = 'running' AND `+where+` FOR UPDATE`, arg)
	if err != nil {
		return 0, wrap(err)
	}
	type running struct {
		ID       int64
		Attempts int
	}
	all, err := pgx.CollectRows(rows, pgx.RowToStructByPos[running])
	if err != nil {
		return 0, wrap(err)
	}
	n := 0
	for _, r := range all {
		if r.Attempts >= attempts {
			_, err = tx.Exec(ctx, `UPDATE inbox SET status = 'failed', error = $2 WHERE id = $1`,
				r.ID, daemon.RecoverError(r.Attempts))
		} else {
			_, err = tx.Exec(ctx, `UPDATE inbox SET status = 'pending' WHERE id = $1`, r.ID)
			n++
		}
		if err != nil {
			return 0, wrap(err)
		}
	}
	return n, nil
}

func (s *Store) Recover(ctx context.Context, owner string, attempts int) (int, error) {
	n := 0
	err := pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		var err error
		if n, err = recoverWhere(ctx, tx, `session_id IN (SELECT id FROM sessions WHERE owner = $1)`, owner, attempts); err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `UPDATE sessions SET owner = NULL, owner_addr = '', lease_until = NULL WHERE owner = $1`, owner)
		return wrap(err)
	})
	return n, err
}

func (s *Store) Acquire(ctx context.Context, session string, l daemon.Lease, now time.Time, attempts int) (int, error) {
	n := 0
	err := pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		n = 0
		var owner, addr string
		var until *time.Time
		var epoch int64
		err := tx.QueryRow(ctx, `SELECT `+leaseCols+` FROM sessions WHERE id = $1 FOR UPDATE`, session).Scan(&owner, &addr, &until, &epoch)
		if errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("%w: %s", daemon.ErrNoSuchSession, session)
		}
		if err != nil {
			return wrap(err)
		}
		held := scanLease(owner, addr, until, epoch)
		if held.Owner != "" && held.Owner != l.Owner && held.Until.After(now) {
			return &daemon.OwnedError{Session: session, Lease: held}
		}
		moved := held.Owner != l.Owner
		if moved {
			// Whatever the last owner left running died with it.
			if n, err = recoverWhere(ctx, tx, `session_id = $1`, session, attempts); err != nil {
				return err
			}
		}
		// Changing hands, from a daemon or from nobody, moves the epoch on:
		// whatever the last tenure had in flight is fenced off.
		_, err = tx.Exec(ctx, `
			UPDATE sessions SET owner = $2, owner_addr = $3, lease_until = $4,
				epoch = epoch + CASE WHEN $5 THEN 1 ELSE 0 END
			WHERE id = $1`,
			session, l.Owner, l.Addr, l.Until, moved)
		return wrap(err)
	})
	return n, err
}

func (s *Store) Renew(ctx context.Context, owner string, sessions []string, until time.Time) ([]string, error) {
	rows, err := s.pool.Query(ctx, `UPDATE sessions SET lease_until = $3 WHERE owner = $1 AND id = ANY($2) RETURNING id`,
		owner, sessions, until)
	if err != nil {
		return nil, wrap(err)
	}
	kept, err := pgx.CollectRows(rows, pgx.RowTo[string])
	return kept, wrap(err)
}

func (s *Store) Release(ctx context.Context, owner string, sessions []string) error {
	_, err := s.pool.Exec(ctx, `
		UPDATE sessions SET owner = NULL, owner_addr = '', lease_until = NULL
		WHERE owner = $1 AND id = ANY($2)`, owner, sessions)
	return wrap(err)
}

// Lead holds role's lock on a connection of its own: owner and until are for
// stores without one, and are not needed here. The lock goes with the
// connection, so a leader that dies, or whose connection drops, hands it to
// the next to ask; a connection that has dropped is replaced on the next ask.
func (s *Store) Lead(ctx context.Context, role, _ string, _ time.Time) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	lc := s.roles[role]
	if lc != nil && lc.conn.Ping(ctx) != nil {
		lc.conn.Close(context.Background())
		delete(s.roles, role)
		lc = nil
	}
	if lc == nil {
		conn, err := pgx.ConnectConfig(ctx, s.cfg.ConnConfig.Copy())
		if err != nil {
			return false, fmt.Errorf("postgres: %w", err)
		}
		lc = &leadConn{conn: conn}
		s.roles[role] = lc
	}
	if lc.held {
		return true, nil
	}
	err := lc.conn.QueryRow(ctx, `SELECT pg_try_advisory_lock(hashtext('wombat-lead:' || current_schema() || ':' || $1))`, role).Scan(&lc.held)
	if err != nil {
		lc.conn.Close(context.Background())
		delete(s.roles, role)
		return false, fmt.Errorf("postgres: %w", err)
	}
	return lc.held, nil
}

func (s *Store) Events(ctx context.Context, f daemon.EventFilter) ([]daemon.Event, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT `+eventCols+` FROM inbox
		WHERE ($1 = '' OR session_id = $1) AND ($2 = '' OR status = $2)
		ORDER BY id DESC LIMIT $3`, f.Session, string(f.Status), nullLimit(f.Limit))
	if err != nil {
		return nil, wrap(err)
	}
	got, err := pgx.CollectRows(rows, scanEvent)
	return got, wrap(err)
}

// ===== Timers =====

const timerCols = `id, session_id, note, due, every_ns, status, created_by, created, fired`

func scanTimer(row pgx.CollectableRow) (daemon.Timer, error) {
	var t daemon.Timer
	var every int64
	err := row.Scan(&t.ID, &t.Session, &t.Note, &t.Due, &every, &t.Status, &t.CreatedBy, &t.Created, &t.Fired)
	t.Every = time.Duration(every)
	t.Due, t.Created = t.Due.UTC(), t.Created.UTC()
	return t, err
}

func (s *Store) AddTimer(ctx context.Context, t daemon.Timer) (daemon.Timer, error) {
	rows, err := s.pool.Query(ctx, `
		INSERT INTO timers (session_id, note, due, every_ns, created_by, created, dedup_key)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (dedup_key) DO NOTHING
		RETURNING `+timerCols,
		t.Session, t.Note, t.Due, int64(t.Every), t.CreatedBy, orNow(t.Created), nullString(t.DedupKey))
	if err != nil {
		return daemon.Timer{}, noSession(err, t.Session)
	}
	got, err := pgx.CollectRows(rows, scanTimer)
	if err != nil {
		return daemon.Timer{}, noSession(err, t.Session)
	}
	if len(got) == 1 {
		return got[0], nil
	}
	rows, err = s.pool.Query(ctx, `SELECT `+timerCols+` FROM timers WHERE dedup_key = $1`, t.DedupKey)
	if err != nil {
		return daemon.Timer{}, wrap(err)
	}
	old, err := pgx.CollectExactlyOneRow(rows, scanTimer)
	return old, wrap(err)
}

func (s *Store) CancelTimer(ctx context.Context, id int64, session string) error {
	tag, err := s.pool.Exec(ctx, `
		UPDATE timers SET status = 'cancelled'
		WHERE id = $1 AND status = 'active' AND ($2 = '' OR session_id = $2)`, id, session)
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("%w: %d", daemon.ErrNoSuchTimer, id)
	}
	return nil
}

func (s *Store) Timers(ctx context.Context, f daemon.TimerFilter) ([]daemon.Timer, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT `+timerCols+` FROM timers
		WHERE ($1 = '' OR session_id = $1) AND ($2 = '' OR status = $2)
		ORDER BY due, id`, f.Session, string(f.Status))
	if err != nil {
		return nil, wrap(err)
	}
	got, err := pgx.CollectRows(rows, scanTimer)
	return got, wrap(err)
}

func (s *Store) Fire(ctx context.Context, now time.Time) ([]daemon.Event, error) {
	var out []daemon.Event
	err := pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		out = nil
		rows, err := tx.Query(ctx, `
			SELECT `+timerCols+` FROM timers
			WHERE status = 'active' AND due <= $1
			ORDER BY due, id FOR UPDATE SKIP LOCKED`, now)
		if err != nil {
			return wrap(err)
		}
		due, err := pgx.CollectRows(rows, scanTimer)
		if err != nil {
			return wrap(err)
		}
		for _, t := range due {
			e, created, err := enqueue(ctx, tx, daemon.TimerEvent(t, now))
			if err != nil {
				return err
			}
			if created {
				out = append(out, e)
			}
			if next, again := daemon.NextDue(t, now); again {
				_, err = tx.Exec(ctx, `UPDATE timers SET fired = fired + 1, due = $2 WHERE id = $1`, t.ID, next)
			} else {
				_, err = tx.Exec(ctx, `UPDATE timers SET fired = fired + 1, status = 'fired' WHERE id = $1`, t.ID)
			}
			if err != nil {
				return wrap(err)
			}
		}
		return nil
	})
	return out, err
}

// ===== The outbox =====

const messageCols = `id, session_id, thread, kind, text, data, in_reply_to, coalesce(dedup_key, ''),
	status, attempts, error, created, delivered, not_before`

func scanMessage(row pgx.CollectableRow) (daemon.Message, error) {
	var m daemon.Message
	var delivered, notBefore *time.Time
	err := row.Scan(&m.ID, &m.Session, &m.Thread, &m.Kind, &m.Text, (*[]byte)(&m.Data), &m.InReplyTo, &m.DedupKey,
		&m.Status, &m.Attempts, &m.Error, &m.Created, &delivered, &notBefore)
	m.Created = m.Created.UTC()
	if delivered != nil {
		m.Delivered = delivered.UTC()
	}
	if notBefore != nil {
		m.NotBefore = notBefore.UTC()
	}
	if len(m.InReplyTo) == 0 {
		m.InReplyTo = nil
	}
	return m, err
}

func (s *Store) Post(ctx context.Context, m daemon.Message) (daemon.Message, bool, error) {
	return post(ctx, s.pool, m)
}

func post(ctx context.Context, q querier, m daemon.Message) (daemon.Message, bool, error) {
	replyTo := m.InReplyTo
	if replyTo == nil {
		replyTo = []int64{}
	}
	rows, err := q.Query(ctx, `
		INSERT INTO outbox (session_id, thread, kind, text, data, in_reply_to, dedup_key, created, not_before)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		ON CONFLICT (dedup_key) DO NOTHING
		RETURNING `+messageCols,
		m.Session, m.Thread, m.Kind, m.Text, nullJSON(m.Data), replyTo, nullString(m.DedupKey),
		orNow(m.Created), nullTime(m.NotBefore))
	if err != nil {
		return daemon.Message{}, false, noSession(err, m.Session)
	}
	got, err := pgx.CollectRows(rows, scanMessage)
	if err != nil {
		return daemon.Message{}, false, noSession(err, m.Session)
	}
	if len(got) == 1 {
		return got[0], true, nil
	}
	rows, err = q.Query(ctx, `SELECT `+messageCols+` FROM outbox WHERE dedup_key = $1`, m.DedupKey)
	if err != nil {
		return daemon.Message{}, false, wrap(err)
	}
	old, err := pgx.CollectExactlyOneRow(rows, scanMessage)
	return old, false, wrap(err)
}

func (s *Store) Outbox(ctx context.Context, f daemon.OutboxFilter) ([]daemon.Message, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT `+messageCols+` FROM outbox
		WHERE ($1 = '' OR session_id = $1) AND ($2 = '' OR status = $2) AND id > $3
		ORDER BY id LIMIT $4`, f.Session, string(f.Status), f.After, nullLimit(f.Limit))
	if err != nil {
		return nil, wrap(err)
	}
	got, err := pgx.CollectRows(rows, scanMessage)
	return got, wrap(err)
}

func (s *Store) Delivered(ctx context.Context, id int64, at time.Time) error {
	tag, err := s.pool.Exec(ctx, `
		UPDATE outbox SET
			delivered = CASE WHEN status = 'delivered' THEN delivered ELSE $2 END,
			error = CASE WHEN status = 'delivered' THEN error ELSE '' END,
			status = 'delivered'
		WHERE id = $1`, id, at)
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("%w: %d", daemon.ErrNoSuchMessage, id)
	}
	return nil
}

func (s *Store) Undelivered(ctx context.Context, id int64, reason string, retry time.Time) error {
	tag, err := s.pool.Exec(ctx, `
		UPDATE outbox SET attempts = attempts + 1, error = $2,
			status = CASE WHEN $3::timestamptz IS NULL THEN 'failed' ELSE status END,
			not_before = coalesce($3, not_before)
		WHERE id = $1 AND status = 'pending'`, id, reason, nullTime(retry))
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() > 0 {
		return nil
	}
	var exists bool
	if err := s.pool.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM outbox WHERE id = $1)`, id).Scan(&exists); err != nil {
		return wrap(err)
	}
	if !exists {
		return fmt.Errorf("%w: %d", daemon.ErrNoSuchMessage, id)
	}
	return nil
}

// ===== Plumbing =====

// querier is what the pool and a transaction share.
type querier interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

func wrap(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("postgres: %w", err)
}

// noSession turns the foreign key a row for a missing session trips over
// into the error the interface promises.
func noSession(err error, session string) error {
	var pg *pgconn.PgError
	if errors.As(err, &pg) && pg.Code == "23503" {
		return fmt.Errorf("%w: %s", daemon.ErrNoSuchSession, session)
	}
	return wrap(err)
}

func isUnique(err error, constraint string) bool {
	var pg *pgconn.PgError
	return errors.As(err, &pg) && pg.Code == "23505" && pg.ConstraintName == constraint
}

func orNow(t time.Time) time.Time {
	if t.IsZero() {
		return time.Now()
	}
	return t
}

func nullString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func nullTime(t time.Time) any {
	if t.IsZero() {
		return nil
	}
	return t
}

// nullJSON is a payload as pgx sends it to a json column, verbatim.
func nullJSON(b []byte) any {
	if len(b) == 0 {
		return nil
	}
	return b
}

func nullLimit(n int) any {
	if n <= 0 {
		return nil
	}
	return n
}
