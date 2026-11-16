package postgres_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"wombat/pkg/daemon"
	"wombat/pkg/daemon/storetest"
	"wombat/pkg/store/postgres"
)

// dsn names the Postgres these tests run against; without one they skip.
// `make pg-up` starts one in podman, and `make test-pg` passes it.
var dsn = flag.String("dsn", "", "a Postgres to test against, e.g. postgres://wombat:wombat@127.0.0.1:55432/wombat")

// schema makes a schema of its own for one test, dropped when it ends, and
// returns a DSN whose search_path is it: the tests share a database and
// nothing else.
func schema(t *testing.T) string {
	t.Helper()
	if *dsn == "" {
		t.Skip("no -dsn: these tests need a Postgres (make test-pg)")
	}
	var b [6]byte
	rand.Read(b[:])
	name := "wombat_test_" + hex.EncodeToString(b[:])

	ctx := context.Background()
	admin, err := pgx.Connect(ctx, *dsn)
	if err != nil {
		t.Fatalf("connecting to %s: %v", *dsn, err)
	}
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+name); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		admin.Exec(ctx, "DROP SCHEMA "+name+" CASCADE")
		admin.Close(ctx)
	})
	return withSearchPath(t, *dsn, name)
}

func withSearchPath(t *testing.T, dsn, schema string) string {
	if !strings.Contains(dsn, "://") {
		return dsn + " search_path=" + schema
	}
	u, err := url.Parse(dsn)
	if err != nil {
		t.Fatal(err)
	}
	q := u.Query()
	q.Set("search_path", schema)
	u.RawQuery = q.Encode()
	return u.String()
}

func open(t *testing.T, dsn string) *postgres.Store {
	t.Helper()
	st, err := postgres.Open(context.Background(), dsn)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	return st
}

func TestStore(t *testing.T) {
	storetest.Run(t, func(t *testing.T) daemon.Store { return open(t, schema(t)) })
}

// A second Open finds the schema made and changes nothing; the data the first
// wrote is still there.
func TestOpenIsIdempotent(t *testing.T) {
	where := schema(t)
	first := open(t, where)
	rec := daemon.SessionRecord{Version: daemon.RecordVersion, ID: "s1", State: daemon.Idle}
	if err := first.CreateSession(context.Background(), rec); err != nil {
		t.Fatal(err)
	}
	second := open(t, where)
	if _, err := second.LoadSession(context.Background(), "s1"); err != nil {
		t.Errorf("after a second Open: %v", err)
	}
}

// One daemon to a name: a second under the same name is told, and gets its
// turn once the first is gone. Daemons under other names share the schema.
func TestLockIsOneDaemonPerName(t *testing.T) {
	ctx := context.Background()
	where := schema(t)
	first, second := open(t, where), open(t, where)
	if err := first.Lock(ctx, "d1"); err != nil {
		t.Fatal(err)
	}
	if err := first.Lock(ctx, "d1"); err != nil {
		t.Errorf("locking twice from one store = %v", err)
	}
	if err := second.Lock(ctx, "d1"); !errors.Is(err, postgres.ErrLocked) {
		t.Fatalf("a second daemon's Lock under the same name = %v, want ErrLocked", err)
	}
	if err := open(t, where).Lock(ctx, "d2"); err != nil {
		t.Errorf("a daemon under another name on the same schema = %v, want the lock", err)
	}
	first.Close()
	if err := second.Lock(ctx, "d1"); err != nil {
		t.Errorf("once the first is closed = %v, want the lock", err)
	}
}

// Lead: one store leads a role; the next to ask leads once it is closed.
func TestLeadIsOneAtATime(t *testing.T) {
	ctx := context.Background()
	where := schema(t)
	first, second := open(t, where), open(t, where)
	if ok, err := first.Lead(ctx, "deliver", "d1", time.Time{}); err != nil || !ok {
		t.Fatalf("the first Lead = %v, %v", ok, err)
	}
	if ok, err := second.Lead(ctx, "deliver", "d2", time.Time{}); err != nil || ok {
		t.Fatalf("a second store's Lead = %v, %v; want false", ok, err)
	}
	if ok, err := second.Lead(ctx, "other", "d2", time.Time{}); err != nil || !ok {
		t.Fatalf("another role = %v, %v; want it led", ok, err)
	}
	first.Close()
	if ok, err := second.Lead(ctx, "deliver", "d2", time.Time{}); err != nil || !ok {
		t.Fatalf("once the leader is closed = %v, %v; want the second to lead", ok, err)
	}
}
