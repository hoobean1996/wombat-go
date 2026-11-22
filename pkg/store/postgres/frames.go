package postgres

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"

	"wombat/pkg/daemon"
)

// ===== The event log =====

func (s *Store) AppendFrames(ctx context.Context, session, owner string, epoch int64, frames []daemon.StoredFrame) error {
	return pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		if err := fence(ctx, tx, session, owner, epoch); err != nil {
			return err
		}
		if len(frames) == 0 {
			return nil
		}
		b := &pgx.Batch{}
		for _, f := range frames {
			b.Queue(`
				INSERT INTO frames (session_id, seq, turn, kind, data, at) VALUES ($1, $2, $3, $4, $5, $6)
				ON CONFLICT (session_id, seq) DO NOTHING`,
				session, f.Seq, f.Turn, f.Kind, []byte(f.Data), orNow(f.At))
		}
		return wrap(tx.SendBatch(ctx, b).Close())
	})
}

func (s *Store) Frames(ctx context.Context, f daemon.FrameFilter) ([]daemon.StoredFrame, error) {
	if _, err := leaseOf(ctx, s.pool, f.Session); err != nil {
		return nil, err
	}
	rows, err := s.pool.Query(ctx, `
		SELECT seq, turn, kind, data, at FROM frames
		WHERE session_id = $1 AND seq > $2 ORDER BY seq LIMIT $3`, f.Session, f.After, nullLimit(f.Limit))
	if err != nil {
		return nil, wrap(err)
	}
	got, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (daemon.StoredFrame, error) {
		var fr daemon.StoredFrame
		var data []byte
		err := row.Scan(&fr.Seq, &fr.Turn, &fr.Kind, &data, &fr.At)
		fr.Data, fr.At = data, fr.At.UTC()
		return fr, err
	})
	return got, wrap(err)
}

func (s *Store) LastFrame(ctx context.Context, session string) (int, error) {
	if _, err := leaseOf(ctx, s.pool, session); err != nil {
		return -1, err
	}
	var last int
	err := s.pool.QueryRow(ctx, `SELECT coalesce(max(seq), -1) FROM frames WHERE session_id = $1`, session).Scan(&last)
	return last, wrap(err)
}

func (s *Store) PruneFrames(ctx context.Context, before time.Time) (int, error) {
	tag, err := s.pool.Exec(ctx, `DELETE FROM frames WHERE at < $1`, before)
	if err != nil {
		return 0, wrap(err)
	}
	return int(tag.RowsAffected()), nil
}
