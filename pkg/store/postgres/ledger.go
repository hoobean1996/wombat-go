package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	"wombat/pkg/core/llm"
	"wombat/pkg/daemon"
)

// ===== The ledger =====

// final is the states a call has ended in, for SQL.
const final = `('succeeded', 'failed', 'denied', 'cancelled')`

const callCols = `session_id, step, idx, turn, parent, run, use_id, tool, input, state, attempts,
	output, is_error, media, tags, reason, handle, decision, owner, created, started, finished`

func scanCall(row pgx.CollectableRow) (daemon.ToolCall, error) {
	var c daemon.ToolCall
	var input, media, decision []byte
	var started, finished *time.Time
	err := row.Scan(&c.Session, &c.Step, &c.Index, &c.Turn, &c.Parent, &c.Run, &c.UseID, &c.Tool, &input,
		&c.State, &c.Attempts, &c.Output, &c.IsError, &media, &c.Tags, &c.Reason, &c.Handle, &decision,
		&c.Owner, &c.Created, &started, &finished)
	if err != nil {
		return c, err
	}
	if len(input) > 0 {
		c.Input = json.RawMessage(input)
	}
	if len(media) > 0 {
		if err := json.Unmarshal(media, &c.Media); err != nil {
			return c, fmt.Errorf("decoding call %s's media: %w", c.Key(), err)
		}
	}
	if len(decision) > 0 {
		c.Decision = new(daemon.Decision)
		if err := json.Unmarshal(decision, c.Decision); err != nil {
			return c, fmt.Errorf("decoding call %s's decision: %w", c.Key(), err)
		}
	}
	if len(c.Tags) == 0 {
		c.Tags = nil
	}
	c.Created = c.Created.UTC()
	if started != nil {
		c.Started = started.UTC()
	}
	if finished != nil {
		c.Finished = finished.UTC()
	}
	return c, nil
}

// fence takes a share of the session's row, held while o.Owner holds it
// under the epoch named: a daemon taking the session over waits for the
// write to land, and a write that comes after finds the session gone from
// under it.
func fence(ctx context.Context, tx pgx.Tx, session, owner string, epoch int64) error {
	var one int
	err := tx.QueryRow(ctx, `SELECT 1 FROM sessions WHERE id = $1 AND `+fenced(2, 3)+` FOR KEY SHARE`,
		session, owner, epoch).Scan(&one)
	if errors.Is(err, pgx.ErrNoRows) {
		return ownedOrGone(ctx, tx, session)
	}
	return wrap(err)
}

func (s *Store) OpenStep(ctx context.Context, o daemon.StepOpen) (int, error) {
	at := orNow(o.At)
	step := 0
	err := pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		err := tx.QueryRow(ctx, `UPDATE sessions SET steps = steps + 1 WHERE id = $1 AND `+fenced(2, 3)+` RETURNING steps`,
			o.Session, o.Owner, o.Epoch).Scan(&step)
		if errors.Is(err, pgx.ErrNoRows) {
			return ownedOrGone(ctx, tx, o.Session)
		}
		if err != nil {
			return wrap(err)
		}
		b := &pgx.Batch{}
		if o.Inflight != nil {
			f := *o.Inflight
			f.Step = step
			inflight, err := daemon.EncodeInflight(&f)
			if err != nil {
				return err
			}
			b.Queue(`UPDATE sessions SET inflight = $2 WHERE id = $1`, o.Session, inflight)
		}
		for _, c := range o.Calls {
			b.Queue(`
				INSERT INTO tool_calls (session_id, step, idx, turn, parent, run, use_id, tool, input, state, owner, created)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending', $10, $11)`,
				o.Session, step, c.Index, o.Turn, o.Parent, o.Run, string(c.UseID), c.Tool, nullJSON(c.Input), o.Owner, at)
		}
		if o.Sub != nil {
			msgs, err := json.Marshal(o.Sub.Messages)
			if err != nil {
				return err
			}
			b.Queue(`
				INSERT INTO tool_runs (session_id, parent, run, step, messages, iterations, updated)
				VALUES ($1, $2, $3, $4, $5, $6, $7)
				ON CONFLICT (session_id, parent, run) DO UPDATE
				SET step = excluded.step, messages = excluded.messages,
					iterations = excluded.iterations, updated = excluded.updated`,
				o.Session, o.Parent, o.Run, step, msgs, o.Sub.Iterations, at)
		}
		return wrap(tx.SendBatch(ctx, b).Close())
	})
	if err != nil {
		return 0, err
	}
	return step, nil
}

func (s *Store) MoveCalls(ctx context.Context, mv daemon.CallMoves) error {
	at := orNow(mv.At)
	return pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		if err := fence(ctx, tx, mv.Session, mv.Owner, mv.Epoch); err != nil {
			return err
		}
		idx := make([]int, 0, len(mv.Moves))
		for _, m := range mv.Moves {
			idx = append(idx, m.Index)
		}
		rows, err := tx.Query(ctx, `
			SELECT `+callCols+` FROM tool_calls
			WHERE session_id = $1 AND step = $2 AND idx = ANY($3)
			FOR UPDATE`, mv.Session, mv.Step, idx)
		if err != nil {
			return wrap(err)
		}
		got, err := pgx.CollectRows(rows, scanCall)
		if err != nil {
			return wrap(err)
		}
		calls := make(map[int]daemon.ToolCall, len(got))
		for _, c := range got {
			calls[c.Index] = c
		}
		// All or nothing: every move is checked before any is made.
		for _, m := range mv.Moves {
			c, ok := calls[m.Index]
			if !ok {
				return fmt.Errorf("daemon: no call %s in the ledger", daemon.CallKey(mv.Session, mv.Step, m.Index))
			}
			if err := daemon.CheckMove(c, m); err != nil {
				return err
			}
		}
		b := &pgx.Batch{}
		for _, m := range mv.Moves {
			c := calls[m.Index]
			daemon.ApplyMove(&c, m, mv.Owner, at)
			calls[m.Index] = c
			media, err := mediaJSON(c.Media)
			if err != nil {
				return err
			}
			var decision []byte
			if c.Decision != nil {
				if decision, err = json.Marshal(c.Decision); err != nil {
					return err
				}
			}
			b.Queue(`
				UPDATE tool_calls SET state = $4, attempts = $5, output = $6, is_error = $7, media = $8,
					tags = $9, reason = $10, handle = $11, decision = $12, owner = $13, started = $14, finished = $15
				WHERE session_id = $1 AND step = $2 AND idx = $3`,
				mv.Session, mv.Step, c.Index, string(c.State), c.Attempts, c.Output, c.IsError, media,
				textArray(c.Tags), c.Reason, c.Handle, nullJSON(decision), c.Owner, nullTime(c.Started), nullTime(c.Finished))
		}
		return wrap(tx.SendBatch(ctx, b).Close())
	})
}

// mediaJSON is a call's media as its column keeps it: null for none.
func mediaJSON(m []llm.Media) ([]byte, error) {
	if len(m) == 0 {
		return nil, nil
	}
	return json.Marshal(m)
}

// textArray is tags as a text[] that is never null.
func textArray(tags []string) []string {
	if tags == nil {
		return []string{}
	}
	return tags
}

func (s *Store) ToolCalls(ctx context.Context, f daemon.CallFilter) ([]daemon.ToolCall, error) {
	states := make([]string, 0, len(f.States))
	for _, st := range f.States {
		states = append(states, string(st))
	}
	rows, err := s.pool.Query(ctx, `
		SELECT `+callCols+` FROM tool_calls
		WHERE ($1 = '' OR session_id = $1) AND ($2 = 0 OR step = $2) AND ($3 = 0 OR turn = $3)
			AND ($4 = '' OR ($4 = '-' AND parent = '') OR parent = $4)
			AND (cardinality($5::text[]) = 0 OR state = ANY($5))
		ORDER BY created DESC, step DESC, idx DESC, session_id
		LIMIT $6`, f.Session, f.Step, f.Turn, f.Parent, states, nullLimit(f.Limit))
	if err != nil {
		return nil, wrap(err)
	}
	got, err := pgx.CollectRows(rows, scanCall)
	return got, wrap(err)
}

func (s *Store) SubRun(ctx context.Context, session, parent string, n int) (daemon.SubRun, bool, error) {
	sub := daemon.SubRun{Session: session, Parent: parent, Run: n}
	var msgs []byte
	err := s.pool.QueryRow(ctx, `
		SELECT step, messages, iterations, updated FROM tool_runs
		WHERE session_id = $1 AND parent = $2 AND run = $3`, session, parent, n).
		Scan(&sub.Step, &msgs, &sub.Iterations, &sub.Updated)
	if errors.Is(err, pgx.ErrNoRows) {
		if _, err := leaseOf(ctx, s.pool, session); err != nil {
			return daemon.SubRun{}, false, err
		}
		return daemon.SubRun{}, false, nil
	}
	if err != nil {
		return daemon.SubRun{}, false, wrap(err)
	}
	if err := json.Unmarshal(msgs, &sub.Messages); err != nil {
		return daemon.SubRun{}, false, fmt.Errorf("postgres: decoding the sub-agent of %s: %w", parent, err)
	}
	sub.Updated = sub.Updated.UTC()
	return sub, true, nil
}

// quiet is the sessions whose ledger nothing will carry on: no turn in
// flight, and nothing waiting.
const quiet = `(SELECT id FROM sessions WHERE NOT awaiting AND inflight IS NULL)`

func (s *Store) PruneCalls(ctx context.Context, before time.Time) (int, error) {
	n := 0
	err := pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		tag, err := tx.Exec(ctx, `
			DELETE FROM tool_calls
			WHERE session_id IN `+quiet+` AND coalesce(finished, created) < $1`, before)
		if err != nil {
			return wrap(err)
		}
		n = int(tag.RowsAffected())
		_, err = tx.Exec(ctx, `
			DELETE FROM tool_runs r
			WHERE r.session_id IN `+quiet+` AND NOT EXISTS (
				SELECT 1 FROM tool_calls c
				WHERE c.session_id = r.session_id AND c.state NOT IN `+final+`
					AND c.session_id || '/' || c.step || '.' || c.idx = r.parent)`)
		return wrap(err)
	})
	return n, err
}
