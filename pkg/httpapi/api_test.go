package httpapi_test

// Black-box: the handler is a wire contract, and a test that reaches inside
// the package stops describing the contract the first time the internals move.
// Everything here goes through httptest and reads bytes back.
//
// Helper names are prefixed "api" so they cannot collide with the Manager's
// own tests in this directory.

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/tool"
	"wombat/pkg/daemon"
	"wombat/pkg/httpapi"
)

// ===== fakes =====

// apiClient is the scripted model client every test here runs on. No network,
// and no second fake.
//
// Safe for concurrent use: a session's turn runs on its own goroutine while
// the test drives HTTP from another, and the whole file runs under -race.
type apiClient struct {
	mu    sync.Mutex
	turns []llm.Response
	calls int

	// gate, when non-nil, holds every model call until it is closed. It is how
	// a test keeps a turn genuinely in flight — a sleep would be a flake, and
	// a turn that has already finished cannot demonstrate 409 or a dropped
	// stream.
	gate chan struct{}

	// free is how many calls go through before gate holds the rest, and
	// started how many have begun.
	free, started int

	// entered reports that a model call has started. Buffered and sent to
	// without blocking, so a test can wait for the turn to reach the provider
	// without the client depending on anyone reading.
	entered chan struct{}
}

func apiScript(turns ...llm.Response) *apiClient {
	return &apiClient{turns: turns, entered: make(chan struct{}, 64)}
}

// heldAfter is held for every call after the first n.
func (c *apiClient) heldAfter(n int) (*apiClient, func()) {
	c.free = n
	return c.held()
}

// held returns the same client with every call parked until release is called.
func (c *apiClient) held() (*apiClient, func()) {
	c.gate = make(chan struct{})
	var once sync.Once
	return c, func() { once.Do(func() { close(c.gate) }) }
}

// Complete implements llm.Client.
func (c *apiClient) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	select {
	case c.entered <- struct{}{}:
	default:
	}
	c.mu.Lock()
	n := c.started
	c.started++
	c.mu.Unlock()
	if c.gate != nil && n >= c.free {
		select {
		case <-c.gate:
		case <-ctx.Done():
			return llm.Response{}, ctx.Err()
		}
	}

	c.mu.Lock()
	if c.calls >= len(c.turns) {
		n := c.calls
		c.mu.Unlock()
		return llm.Response{}, fmt.Errorf("apiClient: out of turns after %d calls", n)
	}
	r := c.turns[c.calls]
	c.calls++
	c.mu.Unlock()

	// Stream the text the way a real client does, so the event plumbing the
	// SSE tests assert on is exercised rather than bypassed.
	if req.OnDelta != nil {
		if t := llm.TextOf(r.Content); t != "" {
			req.OnDelta(llm.Delta{Text: t})
		}
	}
	return r, nil
}

func apiText(s string) llm.Response {
	return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: s}}, StopReason: llm.StopEndTurn}
}

func apiToolCall(id, name, input string) llm.Response {
	return llm.Response{
		Content:    []llm.ContentBlock{llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(input)}},
		StopReason: llm.StopToolUse,
	}
}

// apiTool is a tool with no effect, so a test can watch dispatch and the
// permission gate without touching the machine.
func apiTool(name string) tool.Def {
	return tool.Def{
		Name:        name,
		Description: name + " does nothing, loudly",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		Category:    "test",
		Caps:        tool.CapReadOnly,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			return "ok", nil
		},
	}
}

// apiTerminalTool ends a run: the model calling it produces core.Submitted,
// which is the one outcome that finishes a CONVERSATION rather than a turn.
// It is how these tests reach a terminal session without waiting for a reaper.
func apiTerminalTool(name string) tool.Def {
	return tool.Def{
		Name:        name,
		Description: name + " ends the task",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"result":{"type":"string"}}}`),
		Caps:        tool.CapTerminal,
	}
}

var apiQuiet = slog.New(slog.NewTextHandler(io.Discard, nil))

// ===== harness =====

type apiHarness struct {
	t     *testing.T
	srv   *httptest.Server
	mgr   *daemon.Manager
	agent *core.Agent
}

// apiServe wires a Manager over a fake builder and serves it.
//
// The builder is a closure over one agent rather than a fresh one per session,
// which is exactly how a real server works: one Agent, many conversations.
func apiServe(t *testing.T, c llm.Client, tools []tool.Def, tweak func(*daemon.ManagerConfig), opts ...httpapi.Option) *apiHarness {
	t.Helper()

	agent, err := core.New(
		core.WithName("test"),
		core.WithClient(c),
		core.WithTools(tools...),
		core.WithMaxIters(4),
		core.WithLogger(apiQuiet),
	)
	if err != nil {
		t.Fatalf("building the agent: %v", err)
	}

	cfg := daemon.ManagerConfig{
		// The builder is what turns SessionOptions.Permission into a gate, and
		// permission.Defer is what sends what it asks about to the approvals
		// endpoints. Wired here rather than hidden in the harness because it
		// is the wiring a real server has to get right, and a test that
		// skipped it would leave the approval routes untested.
		Build: func(o daemon.SessionOptions) (*core.Agent, error) {
			if !strings.EqualFold(strings.TrimSpace(o.Permission), "ask") {
				return agent, nil
			}
			return agent.With(core.WithToolMiddleware(
				permission.Gate(permission.AskEverything(), permission.Defer())))
		},
		Limits:     governor.Limits{Steps: 8},
		Poll:       5 * time.Millisecond,
		RetryDelay: time.Millisecond,
		Logger:     apiQuiet,
	}
	if tweak != nil {
		tweak(&cfg)
	}
	m, err := daemon.NewManager(cfg)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })

	srv := httptest.NewServer(httpapi.New(m, opts...))
	t.Cleanup(srv.Close)

	return &apiHarness{t: t, srv: srv, mgr: m, agent: agent}
}

func (h *apiHarness) do(method, path, body string) *http.Response {
	h.t.Helper()
	var rdr io.Reader
	if body != "" {
		rdr = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, h.srv.URL+path, rdr)
	if err != nil {
		h.t.Fatalf("building %s %s: %v", method, path, err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		h.t.Fatalf("%s %s: %v", method, path, err)
	}
	h.t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

// decode reads a JSON body, insisting on the status and the content type.
func (h *apiHarness) decode(resp *http.Response, want int, dst any) {
	h.t.Helper()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != want {
		h.t.Fatalf("status = %d, want %d (body %s)", resp.StatusCode, want, body)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		h.t.Fatalf("Content-Type = %q, want application/json", ct)
	}
	if dst == nil {
		return
	}
	if err := json.Unmarshal(body, dst); err != nil {
		h.t.Fatalf("decoding %s: %v", body, err)
	}
}

type apiErrBody struct {
	Error struct {
		Kind    string `json:"kind"`
		Message string `json:"message"`
	} `json:"error"`
}

// wantErr asserts the pinned envelope: the status AND the bounded kind, since
// a UI branches on the kind and a human reads the status.
func (h *apiHarness) wantErr(resp *http.Response, status int, kind string) apiErrBody {
	h.t.Helper()
	var got apiErrBody
	h.decode(resp, status, &got)
	if got.Error.Kind != kind {
		h.t.Fatalf("kind = %q, want %q (message %q)", got.Error.Kind, kind, got.Error.Message)
	}
	if got.Error.Message == "" {
		h.t.Fatal("error message is empty; a bounded kind still needs prose for a human")
	}
	return got
}

// start creates a session and returns its info.
func (h *apiHarness) start(body string) daemon.SessionInfo {
	h.t.Helper()
	var info daemon.SessionInfo
	h.decode(h.do(http.MethodPost, "/api/sessions", body), http.StatusCreated, &info)
	return info
}

// awaitState polls until the session has nothing left in its inbox to do and
// is in one of want. Polling rather than sleeping: a turn's duration is the
// scheduler's business — and the inbox has to be empty too, because a session
// that has just been sent a message is still idle until its turn begins.
func (h *apiHarness) awaitState(id string, want ...daemon.State) daemon.SessionInfo {
	h.t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	var info daemon.SessionInfo
	for time.Now().Before(deadline) {
		h.decode(h.do(http.MethodGet, "/api/sessions/"+id, ""), http.StatusOK, &info)
		if slices.Contains(want, info.State) && !h.busy(id) {
			return info
		}
		time.Sleep(2 * time.Millisecond)
	}
	h.t.Fatalf("session %s stuck in %q, wanted one of %v", id, info.State, want)
	return info
}

// busy reports whether the session has events still to run.
func (h *apiHarness) busy(id string) bool {
	h.t.Helper()
	var inbox httpapi.EventList
	h.decode(h.do(http.MethodGet, "/api/inbox?session="+id, ""), http.StatusOK, &inbox)
	for _, e := range inbox.Events {
		if e.Status == daemon.EventPending || e.Status == daemon.EventRunning {
			return true
		}
	}
	return false
}

// ===== health, config, routing =====

func TestHealth(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithVersion("v1.2.3"))

	var got struct {
		Status    string  `json:"status"`
		Version   string  `json:"version"`
		UptimeSec float64 `json:"uptime_sec"`
	}
	h.decode(h.do(http.MethodGet, "/api/health", ""), http.StatusOK, &got)

	if got.Status != "ok" || got.Version != "v1.2.3" {
		t.Fatalf("health = %+v", got)
	}
	if got.UptimeSec < 0 {
		t.Fatalf("uptime_sec = %v", got.UptimeSec)
	}
}

func TestConfigReportsTheToolsTheBuilderInstalled(t *testing.T) {
	t.Parallel()
	tools := []tool.Def{apiTool("look"), apiTool("poke")}
	h := apiServe(t, apiScript(apiText("hi")), tools, nil)

	// Described from the built agent, not from a list typed a second time —
	// that is the whole point of the endpoint.
	h2 := httptest.NewServer(httpapi.New(h.mgr,
		httpapi.WithVersion("v9"),
		httpapi.WithCapabilities(httpapi.Capabilities{
			DefaultModel: "test-model",
			Approvals:    true,
			Tools:        httpapi.ToolsOf(h.agent),
		})))
	defer h2.Close()
	h.srv = h2

	var got httpapi.Capabilities
	h.decode(h.do(http.MethodGet, "/api/config", ""), http.StatusOK, &got)

	if got.DefaultModel != "test-model" || !got.Approvals || got.Version != "v9" {
		t.Fatalf("config = %+v", got)
	}
	if len(got.Tools) != 2 || got.Tools[0].Name != "look" || got.Tools[1].Name != "poke" {
		t.Fatalf("tools = %+v, want the two the builder installed", got.Tools)
	}
	if !got.Tools[0].ReadOnly || got.Tools[0].Category != "test" {
		t.Fatalf("tool metadata lost: %+v", got.Tools[0])
	}
	if want := []string{"off", "readonly", "workspace", "ask", "full"}; !equalStrings(got.PermissionModes, want) {
		t.Fatalf("permission_modes = %v, want %v", got.PermissionModes, want)
	}
}

func TestConfigDefaultsWithoutCapabilities(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)

	var got httpapi.Capabilities
	h.decode(h.do(http.MethodGet, "/api/config", ""), http.StatusOK, &got)
	if got.Tools == nil {
		t.Fatal("tools is null; a client iterating the field should not have to nil-check")
	}
	if len(got.PermissionModes) == 0 {
		t.Fatal("permission_modes must always be reported")
	}
}

func TestUnknownAPIPathIsJSON(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)
	h.wantErr(h.do(http.MethodGet, "/api/nope", ""), http.StatusNotFound, "not_found")
	// Including a verb no route claims: the envelope holds for the whole
	// namespace, which is what lets a client parse failures uniformly.
	h.wantErr(h.do(http.MethodPut, "/api/sessions/x", ""), http.StatusNotFound, "not_found")
}

func TestMetricsMount(t *testing.T) {
	t.Parallel()
	metrics := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "# HELP wombat_up\n")
	})
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithMetrics(metrics))

	resp := h.do(http.MethodGet, "/metrics", "")
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(string(body), "# HELP") {
		t.Fatalf("metrics: %d %s", resp.StatusCode, body)
	}

	// There is nothing outside /api/ and /metrics: a path there is the same
	// JSON 404 as an unknown API path, not a page.
	h.wantErr(h.do(http.MethodGet, "/", ""), http.StatusNotFound, "not_found")
	h.wantErr(h.do(http.MethodGet, "/api/nope", ""), http.StatusNotFound, "not_found")
}

// ===== who may call =====

// raw sends a request with headers the harness's do would not set.
func (h *apiHarness) raw(method, path, body string, set func(*http.Request)) *http.Response {
	h.t.Helper()
	var rdr io.Reader
	if body != "" {
		rdr = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, h.srv.URL+path, rdr)
	if err != nil {
		h.t.Fatal(err)
	}
	if set != nil {
		set(req)
	}
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		h.t.Fatalf("%s %s: %v", method, path, err)
	}
	h.t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

// TestBodiesMustBeDeclaredJSON is a regression test for a cross-site request
// forgery. text/plain is a type any page may POST to any origin without a
// preflight, and the body was decoded whatever its type: a page on any site
// the operator visited could start a session here, with a prompt of its
// choosing, and the session ran tools on the operator's machine. That it
// could not read the 201 did not matter.
func TestBodiesMustBeDeclaredJSON(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("ran"), apiText("ran")), nil, nil)
	prompt := `{"prompt":"do what the page says"}`
	for _, ct := range []string{"text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", ""} {
		resp := h.raw(http.MethodPost, "/api/sessions", prompt, func(r *http.Request) {
			r.Header.Set("Origin", "https://elsewhere.example")
			if ct != "" {
				r.Header.Set("Content-Type", ct)
			}
		})
		h.wantErr(resp, http.StatusUnsupportedMediaType, "unsupported_media_type")
	}
	var list httpapi.SessionList
	h.decode(h.do(http.MethodGet, "/api/sessions", ""), http.StatusOK, &list)
	if len(list.Sessions) != 0 {
		t.Fatalf("a body not declared JSON started %d session(s)", len(list.Sessions))
	}

	resp := h.raw(http.MethodPost, "/api/sessions", prompt, func(r *http.Request) {
		r.Header.Set("Content-Type", "application/json; charset=utf-8")
	})
	h.decode(resp, http.StatusCreated, nil)
}

func TestBearerToken(t *testing.T) {
	t.Parallel()
	metrics := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "# HELP x\n") })
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithBearerToken("s3cret"),
		httpapi.WithMetrics(metrics), httpapi.WithCORS("http://ui.example"))
	bearer := func(tok string) func(*http.Request) {
		return func(r *http.Request) {
			r.Header.Set("Authorization", "Bearer "+tok)
			r.Header.Set("Content-Type", "application/json")
		}
	}

	for _, set := range []func(*http.Request){nil, bearer("wrong"), bearer("s3cret-and-more")} {
		resp := h.raw(http.MethodGet, "/api/sessions", "", set)
		if resp.Header.Get("WWW-Authenticate") == "" {
			t.Error("a 401 without WWW-Authenticate does not say what it wants")
		}
		h.wantErr(resp, http.StatusUnauthorized, "unauthorized")
	}
	h.wantErr(h.raw(http.MethodGet, "/metrics", "", nil), http.StatusUnauthorized, "unauthorized")
	h.wantErr(h.raw(http.MethodGet, "/./api/../api/sessions", "", nil), http.StatusUnauthorized, "unauthorized")

	// Open: the probe, and a preflight.
	if resp := h.raw(http.MethodGet, "/api/health", "", nil); resp.StatusCode != http.StatusOK {
		t.Errorf("GET /api/health without a token = %d, want it open", resp.StatusCode)
	}
	pre := h.raw(http.MethodOptions, "/api/sessions", "", func(r *http.Request) {
		r.Header.Set("Origin", "http://ui.example")
		r.Header.Set("Access-Control-Request-Method", "POST")
		r.Header.Set("Access-Control-Request-Headers", "authorization, content-type")
	})
	if pre.StatusCode != http.StatusNoContent || !strings.Contains(pre.Header.Get("Access-Control-Allow-Headers"), "Authorization") {
		t.Errorf("preflight = %d, Allow-Headers %q; want 204 allowing Authorization", pre.StatusCode, pre.Header.Get("Access-Control-Allow-Headers"))
	}

	var info daemon.SessionInfo
	h.decode(h.raw(http.MethodPost, "/api/sessions", `{"prompt":"hello"}`, bearer("s3cret")), http.StatusCreated, &info)
	lower := h.raw(http.MethodGet, "/api/sessions", "", func(r *http.Request) { r.Header.Set("Authorization", "bearer s3cret") })
	h.decode(lower, http.StatusOK, nil) // the scheme is case-insensitive

	// EventSource cannot set a header, so the stream takes the token in its
	// URL — and only the stream does.
	ev := h.raw(http.MethodGet, "/api/sessions/"+info.ID+"/events?access_token=s3cret", "", nil)
	if ev.StatusCode != http.StatusOK || ev.Header.Get("Content-Type") != "text/event-stream" {
		t.Errorf("events with ?access_token = %d %s, want the stream", ev.StatusCode, ev.Header.Get("Content-Type"))
	}
	_ = ev.Body.Close()
	h.wantErr(h.raw(http.MethodGet, "/api/sessions/"+info.ID+"/events?access_token=wrong", "", nil), http.StatusUnauthorized, "unauthorized")
	h.wantErr(h.raw(http.MethodGet, "/api/sessions?access_token=s3cret", "", nil), http.StatusUnauthorized, "unauthorized")
}

// TestAllowedHosts: DNS rebinding makes a page on another site same-origin
// with a loopback server, responses and all. What it cannot change is the
// Host header, which still names the attacker's domain.
func TestAllowedHosts(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithAllowedHosts("localhost", "127.0.0.1", "[::1]"))
	host := func(name string) func(*http.Request) { return func(r *http.Request) { r.Host = name } }

	for _, bad := range []string{"rebound.example", "rebound.example:8080", "localhost.rebound.example", "127.0.0.1.nip.io"} {
		h.wantErr(h.raw(http.MethodGet, "/api/sessions", "", host(bad)), http.StatusMisdirectedRequest, "misdirected")
	}
	for _, good := range []string{"localhost:8080", "LOCALHOST", "127.0.0.1:1", "[::1]:8080"} {
		if resp := h.raw(http.MethodGet, "/api/sessions", "", host(good)); resp.StatusCode != http.StatusOK {
			t.Errorf("Host %q = %d, want it served", good, resp.StatusCode)
		}
	}
}

// ===== sessions =====

func TestCreateSessionRunsTheFirstTurn(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("the answer")), nil, nil)

	resp := h.do(http.MethodPost, "/api/sessions", `{"prompt":"hello","title":"greeting"}`)
	var info daemon.SessionInfo
	h.decode(resp, http.StatusCreated, &info)

	if info.ID == "" {
		t.Fatal("no session id")
	}
	if info.Title != "greeting" {
		t.Fatalf("title = %q", info.Title)
	}
	if want := "/api/sessions/" + info.ID; resp.Header.Get("Location") != want {
		t.Fatalf("Location = %q, want %q", resp.Header.Get("Location"), want)
	}

	done := h.awaitState(info.ID, daemon.Done, daemon.Idle)
	if done.Answer != "the answer" || done.Turns != 1 {
		t.Fatalf("after the first message: %+v", done)
	}
}

func TestCreateSessionRejectsUnknownField(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)

	// The typo case this exists for: "max_iter" instead of "max_iters" would
	// otherwise be accepted and silently ignored.
	got := h.wantErr(h.do(http.MethodPost, "/api/sessions",
		`{"prompt":"hi","max_iter":3}`), http.StatusBadRequest, "bad_request")
	if !strings.Contains(got.Error.Message, "max_iter") {
		t.Fatalf("message %q should name the offending field", got.Error.Message)
	}
}

func TestCreateSessionRejectsMalformedAndEmptyBodies(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)

	h.wantErr(h.do(http.MethodPost, "/api/sessions", `{"prompt":`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/sessions", `{}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/sessions", `{"prompt":"   "}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/sessions", `{"prompt":"a"}{"prompt":"b"}`), http.StatusBadRequest, "bad_request")
}

func TestCreateSessionCapsTheBody(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)

	huge := `{"prompt":"` + strings.Repeat("a", 2<<20) + `"}`
	h.wantErr(h.do(http.MethodPost, "/api/sessions", huge), http.StatusBadRequest, "body_too_large")
}

func TestListGetDeleteSession(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("a"), apiText("b")), nil, nil)

	first := h.start(`{"prompt":"one"}`)
	second := h.start(`{"prompt":"two"}`)

	var list struct {
		Sessions []daemon.SessionInfo `json:"sessions"`
	}
	h.decode(h.do(http.MethodGet, "/api/sessions", ""), http.StatusOK, &list)
	if len(list.Sessions) != 2 {
		t.Fatalf("listed %d sessions, want 2", len(list.Sessions))
	}

	var one daemon.SessionInfo
	h.decode(h.do(http.MethodGet, "/api/sessions/"+first.ID, ""), http.StatusOK, &one)
	if one.ID != first.ID {
		t.Fatalf("got session %s, wanted %s", one.ID, first.ID)
	}

	if resp := h.do(http.MethodDelete, "/api/sessions/"+second.ID, ""); resp.StatusCode != http.StatusNoContent {
		t.Fatalf("DELETE = %d, want 204", resp.StatusCode)
	}
	h.wantErr(h.do(http.MethodGet, "/api/sessions/"+second.ID, ""), http.StatusNotFound, "no_such_session")
	h.wantErr(h.do(http.MethodDelete, "/api/sessions/"+second.ID, ""), http.StatusNotFound, "no_such_session")
}

func TestUnknownSessionOnEveryRoute(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)

	for _, tc := range []struct{ method, path, body string }{
		{http.MethodGet, "/api/sessions/nope", ""},
		{http.MethodDelete, "/api/sessions/nope", ""},
		{http.MethodGet, "/api/sessions/nope/events", ""},
		{http.MethodGet, "/api/sessions/nope/messages", ""},
		{http.MethodPost, "/api/sessions/nope/messages", `{"prompt":"x"}`},
		{http.MethodPost, "/api/sessions/nope/steer", `{"prompt":"x"}`},
		{http.MethodGet, "/api/sessions/nope/approvals", ""},
		{http.MethodPost, "/api/sessions/nope/approvals/u1", `{"allow":true}`},
		{http.MethodPost, "/api/sessions/nope/results/u1", `{"output":"x"}`},
		{http.MethodPost, "/api/events", `{"session_id":"nope","kind":"push"}`},
		{http.MethodPost, "/api/timers", `{"session_id":"nope","note":"x","in":"1m"}`},
	} {
		t.Run(tc.method+" "+tc.path, func(t *testing.T) {
			h.wantErr(h.do(tc.method, tc.path, tc.body), http.StatusNotFound, "no_such_session")
		})
	}
}

// ===== multi-turn =====

func TestSecondTurnContinuesTheConversationAndTheLog(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("first answer"), apiText("second answer")), nil, nil)

	info := h.start(`{"prompt":"one"}`)
	first := h.awaitState(info.ID, daemon.Done, daemon.Idle)

	var second httpapi.Accepted
	h.decode(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/messages", `{"prompt":"two"}`),
		http.StatusAccepted, &second)
	if second.Event.Kind != daemon.KindMessage || second.Event.Session != info.ID || second.Event.ID == 0 {
		t.Fatalf("accepted = %+v, want the message's event", second)
	}

	done := h.awaitState(info.ID, daemon.Done, daemon.Idle)
	if done.Answer != "second answer" || done.Turns != 2 {
		t.Fatalf("after the second message: %+v", done)
	}

	// The point of a session owning a conversation: the transcript grew and
	// the sequence numbers kept counting rather than restarting at 0.
	if done.Events <= first.Events {
		t.Fatalf("events went %d -> %d; the log must continue across a turn", first.Events, done.Events)
	}
	frames := framesOf(h.readStream(info.ID, nil, 0, done.Events))
	if frames[0].id != 0 || frames[len(frames)-1].id != done.Events-1 {
		t.Fatalf("sequence ran %d..%d over %d events", frames[0].id, frames[len(frames)-1].id, done.Events)
	}

	var msgs struct {
		Messages []llm.Message `json:"messages"`
	}
	h.decode(h.do(http.MethodGet, "/api/sessions/"+info.ID+"/messages", ""), http.StatusOK, &msgs)
	if len(msgs.Messages) < 4 {
		t.Fatalf("transcript has %d messages, want both turns", len(msgs.Messages))
	}
	if !strings.Contains(fmt.Sprint(msgs.Messages), "two") {
		t.Fatalf("transcript lost the second prompt: %v", msgs.Messages)
	}
}

// A message sent while a turn runs is not refused: it waits in the inbox, and
// is the next turn.
func TestAMessageDuringATurnIsTheNextTurn(t *testing.T) {
	t.Parallel()
	c, release := apiScript(apiText("one done"), apiText("two done")).held()
	h := apiServe(t, c, nil, nil)

	info := h.start(`{"prompt":"one"}`)
	<-c.entered // the turn is genuinely inside the provider

	var queued httpapi.Accepted
	h.decode(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/messages", `{"prompt":"two"}`),
		http.StatusAccepted, &queued)
	if queued.Event.Status != daemon.EventPending || queued.Session.State != daemon.Running {
		t.Fatalf("accepted = %+v, want the message pending behind the running turn", queued)
	}

	release()
	done := h.awaitState(info.ID, daemon.Idle)
	if done.Turns != 2 || done.Answer != "two done" {
		t.Fatalf("after both: %+v", done)
	}
}

func TestPostMessageValidatesTheBody(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("a")), nil, nil)
	info := h.start(`{"prompt":"one"}`)
	h.awaitState(info.ID, daemon.Done, daemon.Idle)

	path := "/api/sessions/" + info.ID + "/messages"
	h.wantErr(h.do(http.MethodPost, path, `{"promt":"typo"}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, path, `{}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, path, `{"prompt":"`+strings.Repeat("a", 2<<20)+`"}`),
		http.StatusBadRequest, "body_too_large")
}

// TestSendToAFinishedConversation pins the two statuses the pinned table did
// not name, because the sentinels they come from arrived with the Manager.
//
// Both are conflicts with state rather than faults: a terminal tool ended the
// conversation, and a closed Manager is shutting down. Either falling through
// to 500 would tell a client to file a bug about a server doing its job.
func TestSendToAFinishedConversation(t *testing.T) {
	t.Parallel()
	tools := []tool.Def{apiTerminalTool("submit")}
	h := apiServe(t, apiScript(apiToolCall("t1", "submit", `{"result":"42"}`)), tools, nil)

	info := h.start(`{"prompt":"go"}`)
	h.awaitState(info.ID, daemon.Done)

	h.wantErr(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/messages", `{"prompt":"more"}`),
		http.StatusConflict, "done")

	if err := h.mgr.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	h.wantErr(h.do(http.MethodPost, "/api/sessions", `{"prompt":"after close"}`),
		http.StatusServiceUnavailable, "closed")
}

// ===== approvals =====

func TestApprovalLifecycle(t *testing.T) {
	t.Parallel()
	tools := []tool.Def{apiTool("look")}
	// The turn the decision resumes is held at its model call, so the double
	// click below lands while the decision is being acted on — and not after
	// the turn it resumed has ended, when nothing waits on the call at all.
	c, release := apiScript(apiToolCall("use-1", "look", `{"q":"x"}`), apiText("all done")).heldAfter(1)
	t.Cleanup(release)
	h := apiServe(t, c, tools, nil)

	info := h.start(`{"prompt":"look at it","permission":"ask"}`)
	stopped := h.awaitState(info.ID, daemon.Approving)
	if stopped.Outcome != "deferred" {
		t.Fatalf("the turn stopped as %+v, want deferred", stopped)
	}

	var pending httpapi.ApprovalList
	h.decode(h.do(http.MethodGet, "/api/sessions/"+info.ID+"/approvals", ""), http.StatusOK, &pending)
	if len(pending.Approvals) != 1 {
		t.Fatalf("approvals = %+v, want the call under permission=ask", pending.Approvals)
	}
	got := pending.Approvals[0]
	if got.UseID != "use-1" || got.Tool != "look" || string(got.Input) != `{"q":"x"}` {
		t.Fatalf("approval = %+v", got)
	}
	if got.Since.IsZero() {
		t.Fatal("approval carries no timestamp; a UI cannot age a prompt without one")
	}

	// The outbox says so too, for a client that is not watching the stream.
	var out httpapi.MessageList
	h.decode(h.do(http.MethodGet, "/api/outbox?session="+info.ID, ""), http.StatusOK, &out)
	if len(out.Messages) != 1 || out.Messages[0].Kind != daemon.MessageApproval || !strings.Contains(out.Messages[0].Text, "use-1") {
		t.Fatalf("outbox = %+v, want the approval", out.Messages)
	}

	path := "/api/sessions/" + info.ID + "/approvals/" + got.UseID
	h.wantErr(h.do(http.MethodPost, path, `{}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, path, `{"alow":true}`), http.StatusBadRequest, "bad_request")

	var accepted httpapi.Accepted
	h.decode(h.do(http.MethodPost, path, `{"allow":true}`), http.StatusAccepted, &accepted)
	if accepted.Event.Kind != daemon.KindApproval {
		t.Fatalf("accepted = %+v, want the decision's event", accepted)
	}

	// A double click on a slow network is ordinary, not exceptional.
	h.wantErr(h.do(http.MethodPost, path, `{"allow":true}`), http.StatusConflict, "already_answered")

	release()
	done := h.awaitState(info.ID, daemon.Idle)
	if done.Answer != "all done" {
		t.Fatalf("the approved call did not carry the run to the end: %+v", done)
	}
	h.decode(h.do(http.MethodGet, "/api/outbox?session="+info.ID+"&after="+strconv.FormatInt(out.Messages[0].ID, 10), ""), http.StatusOK, &out)
	if len(out.Messages) != 1 || out.Messages[0].Kind != daemon.MessageReply || out.Messages[0].Text != "all done" {
		t.Fatalf("outbox after = %+v, want the reply", out.Messages)
	}
}

// Full Access is a switch a UI offers on a session that is waiting: the calls
// it waits on are allowed, the turn carries on, and the session keeps it.
func TestFullAccessAllowsWhatIsWaiting(t *testing.T) {
	t.Parallel()
	tools := []tool.Def{apiTool("look")}
	c := apiScript(apiToolCall("use-1", "look", `{"q":"x"}`), apiText("all done"))
	h := apiServe(t, c, tools, nil, httpapi.WithCapabilities(httpapi.Capabilities{FullAccess: true}))

	info := h.start(`{"prompt":"look at it","permission":"ask"}`)
	h.awaitState(info.ID, daemon.Approving)

	// The request for a decision answers the message, as the reply will, so
	// a client can tell it belongs to that conversation.
	var out httpapi.MessageList
	h.decode(h.do(http.MethodGet, "/api/outbox?session="+info.ID, ""), http.StatusOK, &out)
	if len(out.Messages) != 1 || out.Messages[0].Kind != daemon.MessageApproval || len(out.Messages[0].InReplyTo) != 1 {
		t.Fatalf("outbox = %+v, want the approval, in reply to the message", out.Messages)
	}

	var patched daemon.SessionInfo
	h.decode(h.do(http.MethodPatch, "/api/sessions/"+info.ID, `{"permission":"full"}`), http.StatusOK, &patched)
	if patched.Options.Permission != daemon.PermissionFull {
		t.Fatalf("patched = %+v, want permission full", patched.Options)
	}
	done := h.awaitState(info.ID, daemon.Idle)
	if done.Answer != "all done" || done.Options.Permission != daemon.PermissionFull {
		t.Fatalf("after the switch = %+v, want the turn carried to its answer under full access", done)
	}
}

// Where the server does not offer full access, asking for it is refused —
// not clamped to a policy that would go on asking about every call.
func TestFullAccessIsRefusedWhereNotOffered(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)

	h.wantErr(h.do(http.MethodPost, "/api/sessions", `{"prompt":"hi","permission":"full"}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/events", `{"kind":"note","options":{"permission":"full"}}`), http.StatusBadRequest, "bad_request")

	info := h.start(`{"prompt":"hi"}`)
	h.awaitState(info.ID, daemon.Idle)
	path := "/api/sessions/" + info.ID
	h.wantErr(h.do(http.MethodPatch, path, `{"permission":"full"}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPatch, path, `{"permission":"root"}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPatch, path, `{}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPatch, "/api/sessions/nope", `{"permission":"ask"}`), http.StatusNotFound, "no_such_session")

	var got daemon.SessionInfo
	h.decode(h.do(http.MethodPatch, path, `{"permission":"ask"}`), http.StatusOK, &got)
	if got.Options.Permission != "ask" {
		t.Fatalf("permission = %q, want ask", got.Options.Permission)
	}
}

func TestApprovalUnknownUseID(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)
	info := h.start(`{"prompt":"one"}`)
	h.awaitState(info.ID, daemon.Done, daemon.Idle)

	var pending struct {
		Approvals []daemon.Approval `json:"approvals"`
	}
	h.decode(h.do(http.MethodGet, "/api/sessions/"+info.ID+"/approvals", ""), http.StatusOK, &pending)
	if pending.Approvals == nil {
		t.Fatal("approvals is null, want an empty array")
	}

	h.wantErr(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/approvals/ghost", `{"allow":false}`),
		http.StatusNotFound, "no_such_approval")
}

// ===== SSE =====

type sseEvent struct {
	id    int // -1 when the frame carried none
	name  string
	data  string
	ping  bool
	retry int
}

// readStream attaches to a session's stream and parses what comes back.
//
// It stops after wantFrames id-bearing frames, or at EOF, whichever is first;
// wantFrames of -1 means "read to EOF". Bounded on purpose: the stream belongs
// to a CONVERSATION, so it stays open between turns waiting for the next one,
// and a helper that read to EOF unconditionally would hang on every healthy
// session. The context deadline is a backstop for a stream that never produces
// what the test asked for.
func (h *apiHarness) readStream(id string, header http.Header, from, wantFrames int) []sseEvent {
	h.t.Helper()
	path := "/api/sessions/" + id + "/events"
	if from > 0 {
		path += "?from=" + strconv.Itoa(from)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.srv.URL+path, nil)
	if err != nil {
		h.t.Fatal(err)
	}
	for k, v := range header {
		req.Header[k] = v
	}
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		h.t.Fatalf("GET %s: %v", path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		h.t.Fatalf("GET %s = %d %s", path, resp.StatusCode, body)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		h.t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	events := parseSSE(h.t, resp.Body, wantFrames)
	if ctx.Err() != nil {
		h.t.Fatalf("stream stalled: wanted %d frames, got %d", wantFrames, len(framesOf(events)))
	}
	return events
}

// parseSSE turns the wire format back into events. It insists on the shape:
// an unrecognised line fails the test rather than being skipped, because a
// stray line is how a stream silently stops being SSE.
func parseSSE(t *testing.T, r io.Reader, stopAfter int) []sseEvent {
	t.Helper()
	var out []sseEvent
	frames := 0
	cur := sseEvent{id: -1}
	dirty := false

	flush := func() bool {
		if !dirty {
			return false
		}
		out = append(out, cur)
		if cur.id >= 0 {
			frames++
		}
		cur, dirty = sseEvent{id: -1}, false
		return stopAfter >= 0 && frames >= stopAfter
	}

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			if flush() {
				return out
			}
		case strings.HasPrefix(line, ": "):
			cur.ping, dirty = true, true
		case strings.HasPrefix(line, "id: "):
			n, err := strconv.Atoi(strings.TrimPrefix(line, "id: "))
			if err != nil {
				t.Fatalf("bad id line %q", line)
			}
			cur.id, dirty = n, true
		case strings.HasPrefix(line, "event: "):
			cur.name, dirty = strings.TrimPrefix(line, "event: "), true
		case strings.HasPrefix(line, "data: "):
			cur.data, dirty = strings.TrimPrefix(line, "data: "), true
		case strings.HasPrefix(line, "retry: "):
			n, err := strconv.Atoi(strings.TrimPrefix(line, "retry: "))
			if err != nil {
				t.Fatalf("bad retry line %q", line)
			}
			cur.retry, dirty = n, true
		default:
			t.Fatalf("unparseable SSE line %q", line)
		}
	}
	if err := sc.Err(); err != nil && !isDisconnect(err) {
		t.Fatalf("reading the stream: %v", err)
	}
	flush()
	return out
}

func isDisconnect(err error) bool {
	s := err.Error()
	return strings.Contains(s, "context canceled") ||
		strings.Contains(s, "context deadline exceeded") ||
		strings.Contains(s, "connection reset")
}

func TestEventStreamShape(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("streamed answer")), nil, nil)
	info := h.start(`{"prompt":"go"}`)
	final := h.awaitState(info.ID, daemon.Done, daemon.Idle)

	events := h.readStream(info.ID, nil, 0, final.Events)
	if len(events) < 3 {
		t.Fatalf("stream had %d events: %+v", len(events), events)
	}
	if events[0].retry == 0 {
		t.Fatal("no retry: hint; EventSource would use the browser default backoff")
	}

	frames := framesOf(events)
	if len(frames) != final.Events {
		t.Fatalf("streamed %d frames, session reports %d events", len(frames), final.Events)
	}
	for i, f := range frames {
		if f.id != i {
			t.Fatalf("frame %d carried id %d; ids must be dense and start at 0", i, f.id)
		}
		if f.name == "" {
			t.Fatalf("frame %d has no event name", i)
		}
		// data is the event's own JSON, and its "type" is what named the
		// event field. A UI that trusted one and not the other would render
		// against a discriminator nobody checked.
		var envelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(f.data), &envelope); err != nil {
			t.Fatalf("frame %d data is not JSON: %q", i, f.data)
		}
		if envelope.Type != f.name {
			t.Fatalf("frame %d: event %q but data type %q", i, f.name, envelope.Type)
		}
	}

}

// TestStreamEndsOnceTheConversationIs pins the terminal event.
//
// EventSource reconnects on its own, so a stream that just closes leaves a
// finished session in a reconnect loop: attach, resume past the end, close,
// back off, repeat. The id-less "end" frame is how a client learns to stop —
// id-less so a reconnect that races it still resumes from the last real frame
// rather than past it.
func TestStreamEndsOnceTheConversationIs(t *testing.T) {
	t.Parallel()
	// A terminal tool ends the conversation, not merely the turn — which is
	// the only thing that makes the stream finite.
	tools := []tool.Def{apiTerminalTool("submit")}
	h := apiServe(t, apiScript(apiToolCall("t1", "submit", `{"result":"42"}`)), tools, nil)

	info := h.start(`{"prompt":"go"}`)
	final := h.awaitState(info.ID, daemon.Done)

	events := h.readStream(info.ID, nil, 0, -1)
	if len(events) == 0 {
		t.Fatal("empty stream")
	}
	last := events[len(events)-1]
	if last.name != "end" || last.id != -1 {
		t.Fatalf("stream ended with %+v; want an id-less terminal event", last)
	}
	var ended daemon.SessionInfo
	if err := json.Unmarshal([]byte(last.data), &ended); err != nil {
		t.Fatalf("terminal event data: %v", err)
	}
	if ended.ID != info.ID || ended.State != final.State {
		t.Fatalf("terminal event = %+v, want session %s in %q", ended, info.ID, final.State)
	}
}

func TestEventStreamResume(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("resumable")), nil, nil)
	info := h.start(`{"prompt":"go"}`)
	settled := h.awaitState(info.ID, daemon.Done, daemon.Idle)

	all := framesOf(h.readStream(info.ID, nil, 0, settled.Events))
	if len(all) < 3 {
		t.Fatalf("need at least three frames to test a resume, got %d", len(all))
	}
	want := all[2:]

	// Last-Event-ID says "I have 1", so the stream restarts at 2.
	byHeader := framesOf(h.readStream(info.ID, http.Header{"Last-Event-Id": {"1"}}, 0, len(want)))
	// ?from= says "send me 2 onward" — the same place, spelled the way a
	// browser can actually manage, since script cannot set a header on an
	// EventSource.
	byQuery := framesOf(h.readStream(info.ID, nil, 2, len(want)))

	for name, got := range map[string][]sseEvent{"Last-Event-ID": byHeader, "?from=": byQuery} {
		if len(got) != len(want) {
			t.Fatalf("%s resumed with %d frames, want %d", name, len(got), len(want))
		}
		for i := range got {
			if got[i].id != want[i].id || got[i].data != want[i].data {
				t.Fatalf("%s frame %d = %+v, want %+v", name, i, got[i], want[i])
			}
		}
		if got[0].id != 2 {
			t.Fatalf("%s restarted at %d, want 2", name, got[0].id)
		}
	}
}

func TestEventStreamRejectsBadResumePoints(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)
	info := h.start(`{"prompt":"go"}`)
	h.awaitState(info.ID, daemon.Done, daemon.Idle)

	base := "/api/sessions/" + info.ID + "/events"
	h.wantErr(h.do(http.MethodGet, base+"?from=abc", ""), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, base+"?from=-4", ""), http.StatusBadRequest, "bad_request")

	req, _ := http.NewRequest(http.MethodGet, h.srv.URL+base, nil)
	req.Header.Set("Last-Event-ID", "junk")
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	h.wantErr(resp, http.StatusBadRequest, "bad_request")
}

// TestHeartbeatKeepsTheStreamWarm pins the keep-alive comment.
//
// A turn can spend minutes inside one tool call, or waiting on the provider, and a
// proxy that sees nothing on the socket closes it. The comment is invisible to
// EventSource and is the only thing keeping such a stream alive.
func TestHeartbeatKeepsTheStreamWarm(t *testing.T) {
	if testing.Short() {
		t.Skip("the heartbeat period is 15s, so this costs real wall clock")
	}
	t.Parallel()

	// Held, so the turn produces its opening frames and then goes quiet —
	// which is exactly the situation the heartbeat exists for.
	c, release := apiScript(apiText("eventually")).held()
	defer release()
	h := apiServe(t, c, nil, nil)
	info := h.start(`{"prompt":"go"}`)
	<-c.entered

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		h.srv.URL+"/api/sessions/"+info.ID+"/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), ": ") {
			return
		}
	}
	t.Fatal("no heartbeat comment within 25s; a proxy would have dropped this stream")
}

func TestDroppedStreamLeavesTheSessionRunning(t *testing.T) {
	t.Parallel()
	c, release := apiScript(apiText("survived")).held()
	h := apiServe(t, c, nil, nil)

	info := h.start(`{"prompt":"go"}`)
	<-c.entered

	// Attach, read the retry preamble so the response is genuinely open, then
	// walk away mid-turn.
	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		h.srv.URL+"/api/sessions/"+info.ID+"/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, len("retry: 1000\n\n"))
	if _, err := io.ReadFull(resp.Body, buf); err != nil {
		t.Fatalf("reading the preamble: %v", err)
	}
	cancel()
	_ = resp.Body.Close()

	// The turn was never cancelled with the connection: that is the whole
	// reason a session outlives a request.
	release()
	done := h.awaitState(info.ID, daemon.Done, daemon.Idle)
	if done.Answer != "survived" {
		t.Fatalf("session did not finish after the client left: %+v", done)
	}

	// And a fresh connection replays everything it missed.
	if frames := framesOf(h.readStream(info.ID, nil, 0, done.Events)); len(frames) != done.Events {
		t.Fatalf("reattached stream had %d frames, want %d", len(frames), done.Events)
	}
}

// ===== CORS =====

func TestCORSPreflight(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithCORS("http://ui.example"))

	req, _ := http.NewRequest(http.MethodOptions, h.srv.URL+"/api/sessions", nil)
	req.Header.Set("Origin", "http://ui.example")
	req.Header.Set("Access-Control-Request-Method", "POST")
	req.Header.Set("Access-Control-Request-Headers", "content-type")
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("preflight = %d, want 204", resp.StatusCode)
	}
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "http://ui.example" {
		t.Fatalf("Allow-Origin = %q", got)
	}
	for _, want := range []string{"GET", "POST", "DELETE"} {
		if !strings.Contains(resp.Header.Get("Access-Control-Allow-Methods"), want) {
			t.Fatalf("Allow-Methods %q is missing %s", resp.Header.Get("Access-Control-Allow-Methods"), want)
		}
	}
	// Without this a cross-origin EventSource cannot resume.
	if !strings.Contains(resp.Header.Get("Access-Control-Allow-Headers"), "Last-Event-ID") {
		t.Fatalf("Allow-Headers %q must include Last-Event-ID", resp.Header.Get("Access-Control-Allow-Headers"))
	}
	if resp.Header.Get("Access-Control-Max-Age") == "" {
		t.Fatal("no Max-Age; every request would pay for a preflight")
	}
}

func TestCORSActualRequest(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithCORS("http://ui.example"))

	req, _ := http.NewRequest(http.MethodGet, h.srv.URL+"/api/health", nil)
	req.Header.Set("Origin", "http://ui.example")
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "http://ui.example" {
		t.Fatalf("Allow-Origin = %q", got)
	}
	if !strings.Contains(resp.Header.Get("Vary"), "Origin") {
		t.Fatalf("Vary = %q, want Origin — a cache must not serve one origin's answer to another",
			resp.Header.Get("Vary"))
	}
	// A cross-origin client cannot read Location unless it is exposed, and
	// Location is how it learns a new session's URL.
	if !strings.Contains(resp.Header.Get("Access-Control-Expose-Headers"), "Location") {
		t.Fatalf("Expose-Headers = %q", resp.Header.Get("Access-Control-Expose-Headers"))
	}
	if resp.Header.Get("Access-Control-Allow-Credentials") != "" {
		t.Fatal("credentialed CORS is never offered: a bearer token rides in a header the page sets, not in a cookie the browser adds")
	}
}

func TestCORSRefusesUnlistedOriginAndWildcardAllowsAll(t *testing.T) {
	t.Parallel()

	strict := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithCORS("http://ui.example"))
	req, _ := http.NewRequest(http.MethodGet, strict.srv.URL+"/api/health", nil)
	req.Header.Set("Origin", "http://evil.example")
	resp, err := strict.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Allow-Origin = %q for an unlisted origin", got)
	}

	open := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithCORS("*"))
	req2, _ := http.NewRequest(http.MethodGet, open.srv.URL+"/api/health", nil)
	req2.Header.Set("Origin", "http://anywhere.example")
	resp2, err := open.srv.Client().Do(req2)
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	if got := resp2.Header.Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf(`Allow-Origin = %q, want "*"`, got)
	}
}

func TestNoCORSHeadersWithoutTheOption(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)

	req, _ := http.NewRequest(http.MethodGet, h.srv.URL+"/api/health", nil)
	req.Header.Set("Origin", "http://ui.example")
	resp, err := h.srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Allow-Origin = %q with CORS off", got)
	}
}

// ===== construction =====

func TestNewPanicsOnNilManager(t *testing.T) {
	t.Parallel()
	defer func() {
		if recover() == nil {
			t.Fatal("New(nil) must panic: a handler with nothing to serve is a wiring bug")
		}
	}()
	_ = httpapi.New(nil)
}

// ===== small helpers =====

func framesOf(events []sseEvent) []sseEvent {
	out := make([]sseEvent, 0, len(events))
	for _, e := range events {
		if e.id >= 0 {
			out = append(out, e)
		}
	}
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ===== steering =====

func TestSteerReachesTheTurnInProgress(t *testing.T) {
	t.Parallel()
	c, release := apiScript(apiText("v1 plan"), apiText("v2 plan")).held()
	h := apiServe(t, c, nil, nil)

	// Idle sessions take a new turn, not a steering message.
	idle := apiServe(t, apiScript(apiText("hi")), nil, nil)
	first := idle.start(`{"prompt":"one"}`)
	idle.awaitState(first.ID, daemon.Idle, daemon.Done)
	idle.wantErr(idle.do(http.MethodPost, "/api/sessions/"+first.ID+"/steer", `{"prompt":"x"}`),
		http.StatusConflict, "not_running")

	info := h.start(`{"prompt":"plan it"}`)
	<-c.entered // the turn is inside the provider
	h.wantErr(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/steer", `{"prompt":""}`),
		http.StatusBadRequest, "bad_request")
	resp := h.do(http.MethodPost, "/api/sessions/"+info.ID+"/steer", `{"prompt":"use v2"}`)
	var got daemon.SessionInfo
	h.decode(resp, http.StatusAccepted, &got)

	release()
	h.awaitState(info.ID, daemon.Idle, daemon.Done)
	c.mu.Lock()
	calls := c.calls
	c.mu.Unlock()
	if calls != 2 {
		t.Errorf("model calls = %d, want 2: the steering message is answered too", calls)
	}
}

// ===== the inbox, the outbox, the timers =====

func TestEventsFromOutside(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("on it")), nil, nil)

	body := `{"thread":"github:o/r#7","kind":"issue_comment","source":"github","dedup_key":"d-1",
		"payload":{"body":"please look"},"options":{"title":"issue 7"}}`
	var first httpapi.Accepted
	h.decode(h.do(http.MethodPost, "/api/events", body), http.StatusAccepted, &first)
	if first.Event.Kind != "issue_comment" || first.Session.Thread != "github:o/r#7" || first.Session.Title != "issue 7" {
		t.Fatalf("accepted = %+v, want a new session for the thread", first)
	}
	var again httpapi.Accepted
	h.decode(h.do(http.MethodPost, "/api/events", body), http.StatusOK, &again)
	if !again.Duplicate || again.Event.ID != first.Event.ID {
		t.Fatalf("the same delivery again = %+v, want the first one back as a duplicate", again)
	}

	done := h.awaitState(first.Event.Session, daemon.Idle)
	if done.Answer != "on it" {
		t.Fatalf("the event's turn = %+v", done)
	}
	var inbox httpapi.EventList
	h.decode(h.do(http.MethodGet, "/api/inbox?session="+done.ID, ""), http.StatusOK, &inbox)
	if len(inbox.Events) != 1 || inbox.Events[0].Status != daemon.EventDone || string(inbox.Events[0].Payload) != `{"body":"please look"}` {
		t.Fatalf("inbox = %+v", inbox.Events)
	}
	// A turn no person asked for says nothing on its own.
	var out httpapi.MessageList
	h.decode(h.do(http.MethodGet, "/api/outbox?session="+done.ID, ""), http.StatusOK, &out)
	if len(out.Messages) != 0 {
		t.Fatalf("outbox = %+v, want nothing: the agent did not send anything", out.Messages)
	}

	h.wantErr(h.do(http.MethodPost, "/api/events", `{"kind":"approval"}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/events", `{"kind":"no spaces"}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/events", `{"kind":"message","payload":{"text":""}}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, "/api/inbox?limit=-1", ""), http.StatusBadRequest, "bad_request")
}

// A client reads the outbox by long-polling it with the last id it has, and
// acks what it handled.
func TestTheOutboxIsLongPolledAndAcked(t *testing.T) {
	t.Parallel()
	c, release := apiScript(apiText("the reply")).held()
	h := apiServe(t, c, nil, nil)
	info := h.start(`{"prompt":"hello"}`)
	<-c.entered

	got := make(chan httpapi.MessageList, 1)
	go func() {
		var out httpapi.MessageList
		h.decode(h.do(http.MethodGet, "/api/outbox?after=0&wait=10s", ""), http.StatusOK, &out)
		got <- out
	}()
	time.Sleep(20 * time.Millisecond)
	release()

	out := <-got
	if len(out.Messages) != 1 || out.Messages[0].Kind != daemon.MessageReply || out.Messages[0].Text != "the reply" ||
		out.Messages[0].Session != info.ID || len(out.Messages[0].InReplyTo) != 1 {
		t.Fatalf("the long poll returned %+v", out.Messages)
	}
	id := strconv.FormatInt(out.Messages[0].ID, 10)
	if resp := h.do(http.MethodPost, "/api/outbox/"+id+"/ack", ""); resp.StatusCode != http.StatusNoContent {
		t.Fatalf("ack = %d, want 204", resp.StatusCode)
	}
	var pending httpapi.MessageList
	h.decode(h.do(http.MethodGet, "/api/outbox?status=pending", ""), http.StatusOK, &pending)
	if len(pending.Messages) != 0 {
		t.Fatalf("pending after the ack = %+v", pending.Messages)
	}

	h.wantErr(h.do(http.MethodPost, "/api/outbox/999/ack", ""), http.StatusNotFound, "no_such_message")
	h.wantErr(h.do(http.MethodPost, "/api/outbox/x/ack", ""), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, "/api/outbox?after=x", ""), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, "/api/outbox?wait=soon", ""), http.StatusBadRequest, "bad_request")
}

func TestTimersOverTheAPI(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hello"), apiText("woke up")), nil, nil)
	info := h.start(`{"prompt":"hi"}`)
	h.awaitState(info.ID, daemon.Idle)

	var timer daemon.Timer
	h.decode(h.do(http.MethodPost, "/api/timers", `{"session_id":"`+info.ID+`","note":"check in","in":"30ms"}`),
		http.StatusCreated, &timer)
	if timer.ID == 0 || timer.Status != daemon.TimerActive || timer.CreatedBy != "api" {
		t.Fatalf("timer = %+v", timer)
	}
	var later daemon.Timer
	h.decode(h.do(http.MethodPost, "/api/timers", `{"session_id":"`+info.ID+`","note":"daily","in":"1h","every":"24h"}`),
		http.StatusCreated, &later)
	if later.Every != 24*time.Hour {
		t.Fatalf("a repeating timer = %+v", later)
	}

	deadline := time.Now().Add(10 * time.Second)
	for {
		var got daemon.SessionInfo
		h.decode(h.do(http.MethodGet, "/api/sessions/"+info.ID, ""), http.StatusOK, &got)
		if got.Turns == 2 && got.State == daemon.Idle {
			if got.Answer != "woke up" {
				t.Fatalf("the timer's turn = %+v", got)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the timer never woke the session: %+v", got)
		}
		time.Sleep(5 * time.Millisecond)
	}

	var timers httpapi.TimerList
	h.decode(h.do(http.MethodGet, "/api/timers?session="+info.ID+"&status=active", ""), http.StatusOK, &timers)
	if len(timers.Timers) != 1 || timers.Timers[0].ID != later.ID {
		t.Fatalf("active timers = %+v, want the daily one", timers.Timers)
	}
	id := strconv.FormatInt(later.ID, 10)
	if resp := h.do(http.MethodDelete, "/api/timers/"+id, ""); resp.StatusCode != http.StatusNoContent {
		t.Fatalf("DELETE = %d, want 204", resp.StatusCode)
	}
	h.wantErr(h.do(http.MethodDelete, "/api/timers/"+id, ""), http.StatusNotFound, "no_such_timer")

	for _, bad := range []string{
		`{"session_id":"` + info.ID + `","note":"x"}`,
		`{"session_id":"` + info.ID + `","note":"x","in":"1m","at":"2026-11-16T09:00:00Z"}`,
		`{"session_id":"` + info.ID + `","note":"x","in":"1m","every":"1s"}`,
		`{"session_id":"` + info.ID + `","note":"","in":"1m"}`,
		`{"session_id":"` + info.ID + `","note":"x","in":"later"}`,
	} {
		h.wantErr(h.do(http.MethodPost, "/api/timers", bad), http.StatusBadRequest, "bad_request")
	}
}

// ===== who said what =====

// transcriptBody is GET /api/sessions/{id}/messages.
type transcriptBody struct {
	Messages []llm.Message           `json:"messages"`
	Origins  map[int][]daemon.Origin `json:"origins"`
}

// Who said what goes in with every way a session hears from a person — its
// first message, a later one, a chat relayed through an event — and comes
// back out beside the transcript, keyed by the message it made.
func TestWhoSaidWhat(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("one"), apiText("two"), apiText("three")), nil, nil)

	info := h.start(`{"prompt":"hello","from":"alice"}`)
	h.awaitState(info.ID, daemon.Idle)

	var sent httpapi.Accepted
	h.decode(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/messages", `{"prompt":"and you?","from":"bob"}`),
		http.StatusAccepted, &sent)
	if sent.Event.From != "bob" || sent.Event.Source != daemon.SourceAPI {
		t.Fatalf("accepted = %+v, want bob's message", sent.Event)
	}
	h.awaitState(info.ID, daemon.Idle)

	var relayed httpapi.Accepted
	h.decode(h.do(http.MethodPost, "/api/events",
		`{"session_id":"`+info.ID+`","kind":"message","source":"slack","from":"carol","payload":{"text":"me too"}}`),
		http.StatusAccepted, &relayed)
	if relayed.Event.From != "carol" || relayed.Event.Source != "slack" {
		t.Fatalf("accepted = %+v, want carol's message", relayed.Event)
	}
	h.awaitState(info.ID, daemon.Idle)

	var got transcriptBody
	h.decode(h.do(http.MethodGet, "/api/sessions/"+info.ID+"/messages", ""), http.StatusOK, &got)
	if len(got.Messages) != 6 {
		t.Fatalf("the transcript has %d messages, want 6", len(got.Messages))
	}
	for i, want := range map[int]struct{ source, from string }{0: {"api", "alice"}, 2: {"api", "bob"}, 4: {"slack", "carol"}} {
		o := got.Origins[i]
		if len(o) != 1 || o[0].Kind != daemon.KindMessage || o[0].Source != want.source || o[0].From != want.from {
			t.Errorf("origins of message %d = %+v, want a message from %s through %s", i, o, want.from, want.source)
		}
	}
	if o := got.Origins[2]; len(o) == 1 && o[0].Event != sent.Event.ID {
		t.Errorf("message 2 names event %d, want %d", o[0].Event, sent.Event.ID)
	}
	if len(got.Origins) != 3 {
		t.Errorf("origins = %+v, want three: the model's turns have none", got.Origins)
	}
}

// A decision says who made it, and the result of the call it decided keeps
// that.
func TestADecisionSaysWhoMadeIt(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiToolCall("use-1", "look", `{}`), apiText("done")), []tool.Def{apiTool("look")}, nil)
	info := h.start(`{"prompt":"look","permission":"ask"}`)
	h.awaitState(info.ID, daemon.Approving)

	path := "/api/sessions/" + info.ID + "/approvals/use-1"
	h.wantErr(h.do(http.MethodPost, path, `{"allow":true,"from":"a\tb"}`), http.StatusBadRequest, "bad_request")
	var accepted httpapi.Accepted
	h.decode(h.do(http.MethodPost, path, `{"allow":true,"from":"dave"}`), http.StatusAccepted, &accepted)
	if accepted.Event.From != "dave" || accepted.Event.Kind != daemon.KindApproval {
		t.Fatalf("accepted = %+v, want dave's decision", accepted.Event)
	}
	h.awaitState(info.ID, daemon.Idle)

	var got transcriptBody
	h.decode(h.do(http.MethodGet, "/api/sessions/"+info.ID+"/messages", ""), http.StatusOK, &got)
	if o := got.Origins[2]; len(o) != 1 || o[0].Kind != daemon.KindApproval || o[0].From != "dave" || o[0].Event != accepted.Event.ID {
		t.Fatalf("origins of the call's result = %+v, want dave's decision (all: %+v)", o, got.Origins)
	}
}

// A from that is not a short line of text is refused wherever it comes in,
// before anything is made; steer, which is not an event, takes none at all.
func TestAFromThatIsNotANameIsRefused(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("ok")), nil, nil)

	h.wantErr(h.do(http.MethodPost, "/api/sessions", `{"prompt":"hi","from":"two\nlines"}`), http.StatusBadRequest, "bad_request")
	var list httpapi.SessionList
	h.decode(h.do(http.MethodGet, "/api/sessions", ""), http.StatusOK, &list)
	if len(list.Sessions) != 0 {
		t.Fatalf("a refused create left %d sessions behind", len(list.Sessions))
	}

	info := h.start(`{"prompt":"hi"}`)
	h.awaitState(info.ID, daemon.Idle)
	long := strings.Repeat("x", 200)
	h.wantErr(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/messages", `{"prompt":"x","from":"`+long+`"}`),
		http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/events", `{"session_id":"`+info.ID+`","kind":"ping","from":" padded"}`),
		http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/steer", `{"prompt":"x","from":"bob"}`),
		http.StatusBadRequest, "bad_request")
}

// ===== results and the ledger =====

// A call that waits for a result from outside is listed with the approvals,
// takes its result over the API, and the turn carries on with it.
func TestAResultOverTheAPI(t *testing.T) {
	t.Parallel()
	// Held at the model call the result resumes, so the second post lands
	// while the result is being acted on, as the approvals' double click does.
	c, release := apiScript(apiToolCall("j1", daemon.ToolWaitForResult, `{"handle":"build 7","note":"did it pass?"}`), apiText("it passed")).heldAfter(1)
	t.Cleanup(release)
	h := apiServe(t, c, daemon.Tools(), nil)
	info := h.start(`{"prompt":"wait for the build"}`)
	h.awaitState(info.ID, daemon.Awaiting)

	var waiting httpapi.ApprovalList
	h.decode(h.do(http.MethodGet, "/api/sessions/"+info.ID+"/approvals", ""), http.StatusOK, &waiting)
	if len(waiting.Approvals) != 0 || len(waiting.Awaiting) != 1 || waiting.Awaiting[0].UseID != "j1" ||
		waiting.Awaiting[0].Handle != "build 7" || !waiting.Awaiting[0].Awaiting {
		t.Fatalf("approvals = %+v, want the call awaiting its result", waiting)
	}
	path := "/api/sessions/" + info.ID + "/results/j1"
	h.wantErr(h.do(http.MethodPost, path, `{}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/results/nope", `{"output":"x"}`), http.StatusNotFound, "no_such_approval")
	h.wantErr(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/approvals/j1", `{"allow":true}`), http.StatusNotFound, "no_such_approval")

	var accepted httpapi.Accepted
	h.decode(h.do(http.MethodPost, path, `{"output":"build 7: green","from":"ci"}`), http.StatusAccepted, &accepted)
	if accepted.Event.Kind != daemon.KindResult || accepted.Event.From != "ci" {
		t.Fatalf("accepted = %+v, want the result's event", accepted)
	}
	h.wantErr(h.do(http.MethodPost, path, `{"output":"again"}`), http.StatusConflict, "already_answered")
	release()
	if done := h.awaitState(info.ID, daemon.Idle); done.Answer != "it passed" {
		t.Fatalf("the turn carried on to %+v", done)
	}
}

// The ledger is listed over the API, filtered by session, state, turn and
// step.
func TestToolCallsOverTheAPI(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiToolCall("u1", "look", `{"q":"x"}`), apiText("seen")), []tool.Def{apiTool("look")}, nil)
	info := h.start(`{"prompt":"look"}`)
	h.awaitState(info.ID, daemon.Idle)

	var list httpapi.ToolCallList
	h.decode(h.do(http.MethodGet, "/api/tool-calls?session="+info.ID, ""), http.StatusOK, &list)
	if len(list.Calls) != 1 {
		t.Fatalf("the ledger = %+v, want the one call", list.Calls)
	}
	c := list.Calls[0]
	if c.UseID != "u1" || c.Tool != "look" || string(c.Input) != `{"q":"x"}` || c.State != tool.StateSucceeded ||
		c.Output != "ok" || c.Attempts != 1 || c.Turn != 1 || c.Step != 1 {
		t.Fatalf("the call = %+v", c)
	}
	for q, n := range map[string]int{
		"state=succeeded":               1,
		"state=running,waiting":         0,
		"state=running&state=succeeded": 1,
		"turn=1&step=1":                 1,
		"turn=2":                        0,
		"parent=-":                      1,
	} {
		h.decode(h.do(http.MethodGet, "/api/tool-calls?session="+info.ID+"&"+q, ""), http.StatusOK, &list)
		if len(list.Calls) != n {
			t.Errorf("?%s = %d calls, want %d", q, len(list.Calls), n)
		}
	}
	for _, q := range []string{"state=lost", "step=x", "turn=0", "limit=-1"} {
		h.wantErr(h.do(http.MethodGet, "/api/tool-calls?"+q, ""), http.StatusBadRequest, "bad_request")
	}
}
