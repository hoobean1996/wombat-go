package postgres

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// migrations are the schema, one version at a time. Each runs once, in a
// transaction, and a database remembers in wombat_schema how far it has got:
// a change to the schema is a new entry at the end, never an edit of an old
// one, which some database somewhere has already run.
var migrations = []string{
	// 1: the sessions, and the three tables.
	`
	CREATE TABLE sessions (
		id       text PRIMARY KEY,
		thread   text UNIQUE,
		created  timestamptz NOT NULL,
		-- set while the session waits on a decision, and takes only approvals
		awaiting boolean NOT NULL DEFAULT false,
		-- the record without its transcript, and the transcript; json, not
		-- jsonb, so both come back byte for byte
		meta     json NOT NULL,
		messages json NOT NULL
	);

	CREATE TABLE inbox (
		id         bigserial PRIMARY KEY,
		session_id text NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
		kind       text NOT NULL,
		source     text NOT NULL,
		dedup_key  text,
		payload    json,
		status     text NOT NULL DEFAULT 'pending'
		           CHECK (status IN ('pending', 'running', 'done', 'failed')),
		attempts   integer NOT NULL DEFAULT 0,
		error      text NOT NULL DEFAULT '',
		created    timestamptz NOT NULL,
		not_before timestamptz NOT NULL,
		claimed_at timestamptz,
		UNIQUE (source, dedup_key)
	);
	CREATE INDEX inbox_pending ON inbox (session_id, id) WHERE status = 'pending';
	CREATE INDEX inbox_running ON inbox (id) WHERE status = 'running';
	CREATE INDEX inbox_session ON inbox (session_id, id);

	CREATE TABLE timers (
		id         bigserial PRIMARY KEY,
		session_id text NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
		note       text NOT NULL,
		due        timestamptz NOT NULL,
		every_ns   bigint NOT NULL DEFAULT 0,
		status     text NOT NULL DEFAULT 'active'
		           CHECK (status IN ('active', 'fired', 'cancelled')),
		created_by text NOT NULL,
		created    timestamptz NOT NULL,
		fired      integer NOT NULL DEFAULT 0
	);
	CREATE INDEX timers_due ON timers (due) WHERE status = 'active';
	CREATE INDEX timers_session ON timers (session_id);

	CREATE TABLE outbox (
		id          bigserial PRIMARY KEY,
		session_id  text NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
		thread      text NOT NULL DEFAULT '',
		kind        text NOT NULL,
		text        text NOT NULL,
		data        json,
		in_reply_to bigint[] NOT NULL DEFAULT '{}',
		dedup_key   text UNIQUE,
		status      text NOT NULL DEFAULT 'pending'
		            CHECK (status IN ('pending', 'delivered', 'failed')),
		attempts    integer NOT NULL DEFAULT 0,
		error       text NOT NULL DEFAULT '',
		created     timestamptz NOT NULL,
		delivered   timestamptz,
		not_before  timestamptz
	);
	CREATE INDEX outbox_pending ON outbox (id) WHERE status = 'pending';
	CREATE INDEX outbox_session ON outbox (session_id, id);
	`,

	// 2: who sent an event, beside where it came from: daemon.Event.From.
	// "sender" because FROM is SQL's.
	`
	ALTER TABLE inbox ADD COLUMN sender text NOT NULL DEFAULT '';
	`,

	// 3: which daemon runs a session, and until when, so that any number of
	// daemons can share the schema: daemon.Store.Acquire.
	`
	ALTER TABLE sessions
		ADD COLUMN owner       text,
		ADD COLUMN owner_addr  text NOT NULL DEFAULT '',
		ADD COLUMN lease_until timestamptz;
	CREATE INDEX sessions_owner ON sessions (owner) WHERE owner IS NOT NULL;
	`,

	// 4: the ledger — every tool call a session's turns make, from the moment
	// the model asks for it until it has its result, and where the sub-agents
	// those calls start had got to — and where a turn in flight had got to,
	// for a daemon that takes it over to carry it on from: daemon.ToolCall,
	// daemon.Store.OpenStep. epoch fences a daemon's writes to the tenure it
	// made them in: daemon.Lease.Epoch. A timer's dedup_key makes a call that
	// sets one, run again, set it once.
	`
	ALTER TABLE sessions
		ADD COLUMN epoch    bigint NOT NULL DEFAULT 1,
		ADD COLUMN steps    integer NOT NULL DEFAULT 0,
		ADD COLUMN inflight json;

	CREATE TABLE tool_calls (
		session_id text NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
		step       integer NOT NULL,
		idx        integer NOT NULL,
		turn       integer NOT NULL,
		-- the call whose sub-agent made this one, and which of its sub-agents
		parent     text NOT NULL DEFAULT '',
		run        integer NOT NULL DEFAULT 0,
		use_id     text NOT NULL,
		tool       text NOT NULL,
		input      json,
		state      text NOT NULL
		           CHECK (state IN ('pending', 'running', 'succeeded', 'failed', 'denied',
		                            'deferred', 'waiting', 'interrupted', 'cancelled')),
		attempts   integer NOT NULL DEFAULT 0,
		output     text NOT NULL DEFAULT '',
		is_error   boolean NOT NULL DEFAULT false,
		media      json,
		tags       text[] NOT NULL DEFAULT '{}',
		reason     text NOT NULL DEFAULT '',
		handle     text NOT NULL DEFAULT '',
		decision   json,
		owner      text NOT NULL DEFAULT '',
		created    timestamptz NOT NULL,
		started    timestamptz,
		finished   timestamptz,
		PRIMARY KEY (session_id, step, idx)
	);
	CREATE INDEX tool_calls_newest ON tool_calls (created DESC, step DESC, idx DESC);
	CREATE INDEX tool_calls_parent ON tool_calls (session_id, parent) WHERE parent <> '';
	CREATE INDEX tool_calls_ended ON tool_calls ((coalesce(finished, created)));

	CREATE TABLE tool_runs (
		session_id text NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
		parent     text NOT NULL,
		run        integer NOT NULL,
		step       integer NOT NULL,
		messages   json NOT NULL,
		iterations integer NOT NULL,
		updated    timestamptz NOT NULL,
		PRIMARY KEY (session_id, parent, run)
	);

	ALTER TABLE timers ADD COLUMN dedup_key text UNIQUE;
	`,
}

// migrate brings the schema up to date. A transaction-scoped advisory lock
// serializes daemons starting at once against a new database, so the second
// waits for the first and finds nothing left to do.
func (s *Store) migrate(ctx context.Context) error {
	err := pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtext('wombat-migrate:' || current_schema()))`); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `CREATE TABLE IF NOT EXISTS wombat_schema (version integer NOT NULL)`); err != nil {
			return err
		}
		var have int
		if err := tx.QueryRow(ctx, `SELECT coalesce(max(version), 0) FROM wombat_schema`).Scan(&have); err != nil {
			return err
		}
		if have > len(migrations) {
			return fmt.Errorf("the schema is at version %d, and this build knows %d: it was made by a newer wombat", have, len(migrations))
		}
		for v := have + 1; v <= len(migrations); v++ {
			if _, err := tx.Exec(ctx, migrations[v-1]); err != nil {
				return fmt.Errorf("version %d: %w", v, err)
			}
			if _, err := tx.Exec(ctx, `INSERT INTO wombat_schema (version) VALUES ($1)`, v); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("postgres: migrating the schema: %w", err)
	}
	return nil
}
