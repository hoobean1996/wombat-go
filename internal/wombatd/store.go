package wombatd

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"wombat/pkg/daemon"
	"wombat/pkg/store/postgres"
)

// openTimeout bounds connecting to the store and bringing its schema up to
// date, so a database that is not there fails the start instead of hanging it.
const openTimeout = 30 * time.Second

// openStore opens where the sessions and their inbox, timers and outbox are
// kept. Any number of wombatd may share a Postgres schema, each under a name
// of its own; a second one under this one's name would take its sessions for
// its own, and is refused at the start instead.
func openStore(cfg storeConfig, name string, logger *slog.Logger) (daemon.Store, error) {
	if cfg.Driver != driverPostgres {
		logger.Warn("store.driver is memory: sessions, timers and the outbox are gone when wombatd stops")
		return daemon.MemoryStore(), nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), openTimeout)
	defer cancel()
	st, err := postgres.Open(ctx, cfg.DSN)
	if err != nil {
		return nil, fmt.Errorf("store: %w", err)
	}
	if err := st.Lock(ctx, name); err != nil {
		st.Close()
		return nil, fmt.Errorf("store: daemon %q: %w", name, err)
	}
	return st, nil
}
