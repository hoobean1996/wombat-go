package wombatd

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"wombat/pkg/daemon"
	"wombat/pkg/httpapi"
)

func env(vars map[string]string) func(string) string {
	return func(k string) string { return vars[k] }
}

// The configuration of a serverless wombatd comes from the environment, its
// ${VARS} expanded, and is refused when it cannot work there.
func TestTheServerlessConfigComesFromTheEnvironment(t *testing.T) {
	if _, err := configFromEnv(env(nil)); err == nil || !strings.Contains(err.Error(), ConfigEnv) {
		t.Errorf("no configuration = %v, want it to say where it is looked for", err)
	}
	if _, err := configFromEnv(env(map[string]string{ConfigEnv: "store:\n  driver: memory\n"})); err == nil || !strings.Contains(err.Error(), "postgres") {
		t.Errorf("a memory store = %v, want it refused", err)
	}
	cfg, err := configFromEnv(env(map[string]string{
		ConfigEnv:      "store:\n  driver: postgres\n  dsn: ${DATABASE_URL}\nserver:\n  token: ${WOMBATD_TOKEN}\n",
		"DATABASE_URL": "postgres://u:p@db/x", "WOMBATD_TOKEN": "secret",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Store.DSN != "postgres://u:p@db/x" || cfg.Server.Token != "secret" {
		t.Errorf("not expanded: dsn %q, token %q", cfg.Store.DSN, cfg.Server.Token)
	}
	if cfg.Daemon.Slice != serverlessSlice || cfg.Serverless.StreamLimit != serverlessSlice || cfg.Server.Addr != "0.0.0.0:0" {
		t.Errorf("serverless defaults: slice %v, stream limit %v, addr %q", cfg.Daemon.Slice, cfg.Serverless.StreamLimit, cfg.Server.Addr)
	}
	kept, _ := configFromEnv(env(map[string]string{ConfigEnv: "store:\n  driver: postgres\n  dsn: x\ndaemon:\n  slice: 30s\n"}))
	if kept.Daemon.Slice != 30*time.Second {
		t.Errorf("a slice set is %v, want it kept", kept.Daemon.Slice)
	}
}

// pgSchema is a schema of its own in the Postgres WOMBAT_TEST_DSN names
// (make test-pg passes it), dropped when the test ends, and a DSN for it.
func pgSchema(t *testing.T) string {
	t.Helper()
	dsn := os.Getenv("WOMBAT_TEST_DSN")
	if dsn == "" {
		t.Skip("WOMBAT_TEST_DSN is not set: this test needs a Postgres (make test-pg)")
	}
	var b [6]byte
	rand.Read(b[:])
	name := "wombat_test_" + hex.EncodeToString(b[:])
	ctx := context.Background()
	admin, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+name); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		admin.Exec(ctx, "DROP SCHEMA "+name+" CASCADE")
		admin.Close(ctx)
	})
	u, err := url.Parse(dsn)
	if err != nil {
		t.Fatal(err)
	}
	q := u.Query()
	q.Set("search_path", name)
	u.RawQuery = q.Encode()
	return u.String()
}

// A serverless wombatd, end to end over HTTP: the first message asks the
// queue to run it, each POST /api/advance runs a slice — a fresh read of
// the store, nothing kept — and the turn ends as it would in a wombatd that
// stays up.
func TestTheServerlessHandlerRunsATurn(t *testing.T) {
	dsn := pgSchema(t)
	var mu sync.Mutex
	calls := 0
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"t1","type":"function","function":{"name":"list_timers","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"c2","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"no timers, hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer provider.Close()
	var woke []daemon.WakeRequest
	queue := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req daemon.WakeRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		mu.Lock()
		woke = append(woke, req)
		mu.Unlock()
		if r.Header.Get("Authorization") != "Bearer queue-token" {
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer queue.Close()

	h, err := newServerless(env(map[string]string{
		ConfigEnv: `
store: {driver: postgres, dsn: "${DSN}"}
server: {token: tok, log: error}
model: {provider: openai, name: m, base_url: "${PROVIDER}/v1", api_key: k, stream: never}
tools: {working_dir: "${DIR}"}
daemon: {slice: 1ns}
serverless:
  wake:
    url: "${QUEUE}"
    headers:
      Authorization: "Bearer queue-token"
`,
		"DSN": dsn, "PROVIDER": provider.URL, "QUEUE": queue.URL, "DIR": t.TempDir(),
	}))
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(h)
	defer srv.Close()
	do := func(method, path, body string, out any) int {
		t.Helper()
		req, _ := http.NewRequest(method, srv.URL+path, bytes.NewReader([]byte(body)))
		req.Header.Set("Authorization", "Bearer tok")
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if out != nil {
			_ = json.NewDecoder(resp.Body).Decode(out)
		}
		return resp.StatusCode
	}

	var info daemon.SessionInfo
	if code := do(http.MethodPost, "/api/sessions", `{"prompt":"any timers?"}`, &info); code != http.StatusCreated {
		t.Fatalf("POST /api/sessions = %d", code)
	}
	mu.Lock()
	if len(woke) != 1 || woke[0].SessionID != info.ID {
		t.Fatalf("the queue was asked %+v", woke)
	}
	mu.Unlock()
	for n := 0; ; n++ {
		var adv httpapi.Advanced
		if code := do(http.MethodPost, "/api/advance", `{"session_id":"`+info.ID+`"}`, &adv); code != http.StatusOK {
			t.Fatalf("POST /api/advance = %d", code)
		}
		if !adv.More {
			break
		}
		if n > 20 {
			t.Fatal("the session never ran out of work")
		}
	}
	var got daemon.SessionInfo
	do(http.MethodGet, "/api/sessions/"+info.ID, "", &got)
	if got.State != daemon.Idle || got.Answer != "no timers, hello" || got.Turns != 1 {
		t.Fatalf("the session = %+v", got)
	}
	var ticked httpapi.Ticked
	if code := do(http.MethodGet, "/api/tick", "", &ticked); code != http.StatusOK {
		t.Errorf("GET /api/tick = %d", code)
	}
	var calls2 httpapi.ToolCallList
	do(http.MethodGet, "/api/tool-calls?session="+info.ID, "", &calls2)
	if len(calls2.Calls) != 1 || calls2.Calls[0].Tool != "list_timers" || calls2.Calls[0].Attempts != 1 {
		t.Errorf("the ledger = %+v", calls2.Calls)
	}
}
