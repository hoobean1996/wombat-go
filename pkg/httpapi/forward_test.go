package httpapi_test

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/daemon"
	"wombat/pkg/httpapi"
)

// serveDaemon is one daemon of several sharing st, served at an address it
// gives its leases, so the others can forward to it.
func serveDaemon(t *testing.T, st daemon.Store, name string, c llm.Client, addr func(string) string) *apiHarness {
	t.Helper()
	var h atomic.Pointer[http.Handler]
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { (*h.Load()).ServeHTTP(w, r) }))
	t.Cleanup(srv.Close)
	agent, err := core.New(core.WithName("test"), core.WithClient(c), core.WithMaxIters(4), core.WithLogger(apiQuiet))
	if err != nil {
		t.Fatal(err)
	}
	m, err := daemon.NewManager(daemon.ManagerConfig{
		Store: st, Name: name, Addr: addr(srv.URL),
		Build:  func(daemon.SessionOptions) (*core.Agent, error) { return agent, nil },
		Limits: governor.Limits{Steps: 8}, Poll: 5 * time.Millisecond, Logger: apiQuiet,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	handler := httpapi.New(m)
	h.Store(&handler)
	return &apiHarness{t: t, srv: srv, mgr: m, agent: agent}
}

func itself(url string) string { return url }

// A request about a session, made to a daemon that does not run it, is
// answered by the one that does — its stream included — so a load balancer
// can send any request to any daemon.
func TestRequestsGoWhereTheSessionRuns(t *testing.T) {
	t.Parallel()
	st := daemon.MemoryStore()
	a := serveDaemon(t, st, "a", apiScript(apiText("from a"), apiText("again from a")), itself)
	b := serveDaemon(t, st, "b", apiScript(apiText("from b")), itself)

	info := a.start(`{"prompt":"hi"}`)
	// Asked of b, answered by a.
	if got := b.awaitState(info.ID, daemon.Idle); got.Answer != "from a" {
		t.Fatalf("through b: %+v, want a's answer", got)
	}
	var msgs struct {
		Messages []llm.Message `json:"messages"`
	}
	b.decode(b.do(http.MethodGet, "/api/sessions/"+info.ID+"/messages", ""), http.StatusOK, &msgs)
	if len(msgs.Messages) != 2 {
		t.Fatalf("the transcript through b has %d messages, want 2", len(msgs.Messages))
	}

	// A message through b runs on a; b's stream of the session is a's log,
	// passed on frame by frame.
	b.decode(b.do(http.MethodPost, "/api/sessions/"+info.ID+"/messages", `{"prompt":"more"}`), http.StatusAccepted, nil)
	// Until the second turn's answer: awaitState can see the session idle
	// just before that turn begins.
	var done daemon.SessionInfo
	for deadline := time.Now().Add(10 * time.Second); ; {
		if done = a.awaitState(info.ID, daemon.Idle); done.Turns >= 2 || time.Now().After(deadline) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if done.Answer != "again from a" {
		t.Fatalf("the second turn = %+v, want it run on a", done)
	}
	frames := b.readStream(info.ID, nil, 0, done.Events)
	ended := 0
	for _, f := range frames {
		if f.name == "turn_ended" {
			ended++
		}
	}
	if ended != 2 {
		t.Fatalf("b's stream of a's session has %d turn_ended frames, want 2: %+v", ended, frames)
	}

	// Forwarded once at most: a forwarded request b does not hold is refused
	// rather than passed on.
	resp := b.raw(http.MethodGet, "/api/sessions/"+info.ID, "", func(r *http.Request) { r.Header.Set("X-Wombat-Forwarded-By", "c") })
	b.wantErr(resp, http.StatusServiceUnavailable, "elsewhere")
}

// A daemon with no address cannot be forwarded to: the others say where the
// session is instead.
func TestNoAddressNoForwarding(t *testing.T) {
	t.Parallel()
	st := daemon.MemoryStore()
	a := serveDaemon(t, st, "a", apiScript(apiText("from a")), func(string) string { return "" })
	b := serveDaemon(t, st, "b", apiScript(), itself)
	info := a.start(`{"prompt":"hi"}`)
	b.wantErr(b.do(http.MethodGet, "/api/sessions/"+info.ID, ""), http.StatusServiceUnavailable, "elsewhere")
}
