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
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/httpapi"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/permission"
	"github.com/hoobean1996/wombat-go/tool"
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

	// entered reports that a model call has started. Buffered and sent to
	// without blocking, so a test can wait for the turn to reach the provider
	// without the client depending on anyone reading.
	entered chan struct{}
}

func apiScript(turns ...llm.Response) *apiClient {
	return &apiClient{turns: turns, entered: make(chan struct{}, 64)}
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
	if c.gate != nil {
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

// apiTerminalTool ends a run: the model calling it produces wombat.Submitted,
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
	mgr   *httpapi.Manager
	agent *wombat.Agent
}

// apiServe wires a Manager over a fake builder and serves it.
//
// The builder is a closure over one agent rather than a fresh one per session,
// which is exactly how a real server works: one Agent, many conversations.
func apiServe(t *testing.T, c llm.Client, tools []tool.Def, tweak func(*httpapi.ManagerConfig), opts ...httpapi.Option) *apiHarness {
	t.Helper()

	agent, err := wombat.New(
		wombat.WithName("test"),
		wombat.WithClient(c),
		wombat.WithTools(tools...),
		wombat.WithMaxIters(4),
		wombat.WithLogger(apiQuiet),
	)
	if err != nil {
		t.Fatalf("building the agent: %v", err)
	}

	cfg := httpapi.ManagerConfig{
		// The builder is what turns SessionOptions.Permission into a gate, and
		// httpapi.Approver is what connects that gate to the approvals
		// endpoints. Wired here rather than hidden in the harness because it
		// is the wiring a real server has to get right, and a test that
		// skipped it would leave the approval routes untested.
		Build: func(o httpapi.SessionOptions) (*wombat.Agent, error) {
			if !strings.EqualFold(strings.TrimSpace(o.Permission), "ask") {
				return agent, nil
			}
			return agent.With(wombat.WithToolMiddleware(
				permission.Gate(permission.AskEverything(), httpapi.Approver())))
		},
		Limits: governor.Limits{Steps: 8},
		TTL:    time.Minute,
		Max:    8,
		Logger: apiQuiet,
	}
	if tweak != nil {
		tweak(&cfg)
	}
	m, err := httpapi.NewManager(cfg)
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
func (h *apiHarness) start(body string) httpapi.SessionInfo {
	h.t.Helper()
	var info httpapi.SessionInfo
	h.decode(h.do(http.MethodPost, "/api/sessions", body), http.StatusCreated, &info)
	return info
}

// awaitState polls until the session reaches one of want. Polling rather than
// sleeping: a turn's duration is the scheduler's business.
func (h *apiHarness) awaitState(id string, want ...httpapi.State) httpapi.SessionInfo {
	h.t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	var info httpapi.SessionInfo
	for time.Now().Before(deadline) {
		h.decode(h.do(http.MethodGet, "/api/sessions/"+id, ""), http.StatusOK, &info)
		for _, w := range want {
			if info.State == w {
				return info
			}
		}
		time.Sleep(2 * time.Millisecond)
	}
	h.t.Fatalf("session %s stuck in %q, wanted one of %v", id, info.State, want)
	return info
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
	if want := []string{"off", "readonly", "workspace", "ask"}; !equalStrings(got.PermissionModes, want) {
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
	h.wantErr(h.do(http.MethodPatch, "/api/sessions/x", ""), http.StatusNotFound, "not_found")
}

func TestUIAndMetricsMount(t *testing.T) {
	t.Parallel()
	ui := fstest.MapFS{"index.html": &fstest.MapFile{Data: []byte("<h1>wombat</h1>")}}
	metrics := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "# HELP wombat_up\n")
	})
	h := apiServe(t, apiScript(apiText("hi")), nil, nil,
		httpapi.WithUI(ui), httpapi.WithMetrics(metrics))

	resp := h.do(http.MethodGet, "/", "")
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), "wombat") {
		t.Fatalf("UI: %d %s", resp.StatusCode, body)
	}

	resp = h.do(http.MethodGet, "/metrics", "")
	body, _ = io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(string(body), "# HELP") {
		t.Fatalf("metrics: %d %s", resp.StatusCode, body)
	}

	// The file server must not shadow the API's own 404.
	h.wantErr(h.do(http.MethodGet, "/api/nope", ""), http.StatusNotFound, "not_found")
}

// ===== sessions =====

func TestCreateSessionRunsTheFirstTurn(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("the answer")), nil, nil)

	resp := h.do(http.MethodPost, "/api/sessions", `{"prompt":"hello","title":"greeting"}`)
	var info httpapi.SessionInfo
	h.decode(resp, http.StatusCreated, &info)

	if info.ID == "" {
		t.Fatal("no session id")
	}
	if info.Turns != 1 {
		t.Fatalf("turns = %d, want 1", info.Turns)
	}
	if info.Title != "greeting" {
		t.Fatalf("title = %q", info.Title)
	}
	if want := "/api/sessions/" + info.ID; resp.Header.Get("Location") != want {
		t.Fatalf("Location = %q, want %q", resp.Header.Get("Location"), want)
	}

	done := h.awaitState(info.ID, httpapi.Done, httpapi.Idle)
	if done.Answer != "the answer" {
		t.Fatalf("answer = %q", done.Answer)
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

func TestCreateSessionTooMany(t *testing.T) {
	t.Parallel()
	c, release := apiScript(apiText("one"), apiText("two")).held()
	defer release()
	h := apiServe(t, c, nil, func(cfg *httpapi.ManagerConfig) { cfg.Max = 1 })

	h.start(`{"prompt":"first"}`)
	h.wantErr(h.do(http.MethodPost, "/api/sessions", `{"prompt":"second"}`),
		http.StatusTooManyRequests, "too_many_sessions")
}

func TestListGetDeleteSession(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("a"), apiText("b")), nil, nil)

	first := h.start(`{"prompt":"one"}`)
	second := h.start(`{"prompt":"two"}`)

	var list struct {
		Sessions []httpapi.SessionInfo `json:"sessions"`
	}
	h.decode(h.do(http.MethodGet, "/api/sessions", ""), http.StatusOK, &list)
	if len(list.Sessions) != 2 {
		t.Fatalf("listed %d sessions, want 2", len(list.Sessions))
	}

	var one httpapi.SessionInfo
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
		{http.MethodGet, "/api/sessions/nope/approvals", ""},
		{http.MethodPost, "/api/sessions/nope/approvals/u1", `{"allow":true}`},
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
	first := h.awaitState(info.ID, httpapi.Done, httpapi.Idle)

	var second httpapi.SessionInfo
	h.decode(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/messages", `{"prompt":"two"}`),
		http.StatusAccepted, &second)
	if second.Turns != 2 {
		t.Fatalf("turns = %d, want 2", second.Turns)
	}

	done := h.awaitState(info.ID, httpapi.Done, httpapi.Idle)
	if done.Answer != "second answer" {
		t.Fatalf("answer = %q", done.Answer)
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

func TestSecondSendWhileRunningIsBusy(t *testing.T) {
	t.Parallel()
	c, release := apiScript(apiText("slow")).held()
	h := apiServe(t, c, nil, nil)

	info := h.start(`{"prompt":"one"}`)
	<-c.entered // the turn is genuinely inside the provider

	h.wantErr(h.do(http.MethodPost, "/api/sessions/"+info.ID+"/messages", `{"prompt":"two"}`),
		http.StatusConflict, "busy")

	release()
	h.awaitState(info.ID, httpapi.Done, httpapi.Idle)
}

func TestPostMessageValidatesTheBody(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("a")), nil, nil)
	info := h.start(`{"prompt":"one"}`)
	h.awaitState(info.ID, httpapi.Done, httpapi.Idle)

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
	h.awaitState(info.ID, httpapi.Done)

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
	c := apiScript(apiToolCall("use-1", "look", `{"q":"x"}`), apiText("all done"))
	h := apiServe(t, c, tools, nil)

	info := h.start(`{"prompt":"look at it","permission":"ask"}`)

	var pending struct {
		Approvals []httpapi.Approval `json:"approvals"`
	}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		h.decode(h.do(http.MethodGet, "/api/sessions/"+info.ID+"/approvals", ""), http.StatusOK, &pending)
		if len(pending.Approvals) > 0 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	if len(pending.Approvals) != 1 {
		t.Fatal("no tool call ever parked on a human under permission=ask")
	}
	got := pending.Approvals[0]
	if got.UseID != "use-1" || got.Tool != "look" {
		t.Fatalf("approval = %+v", got)
	}
	if got.Since.IsZero() {
		t.Fatal("approval carries no timestamp; a UI cannot age a prompt without one")
	}

	path := "/api/sessions/" + info.ID + "/approvals/" + got.UseID
	h.wantErr(h.do(http.MethodPost, path, `{}`), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodPost, path, `{"alow":true}`), http.StatusBadRequest, "bad_request")

	if resp := h.do(http.MethodPost, path, `{"allow":true}`); resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("approve = %d %s, want 204", resp.StatusCode, body)
	}

	// A double click on a slow network is ordinary, not exceptional.
	h.wantErr(h.do(http.MethodPost, path, `{"allow":true}`), http.StatusConflict, "already_answered")

	done := h.awaitState(info.ID, httpapi.Done, httpapi.Idle)
	if done.Answer != "all done" {
		t.Fatalf("the approved call did not carry the run to the end: %+v", done)
	}
}

func TestApprovalUnknownUseID(t *testing.T) {
	t.Parallel()
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)
	info := h.start(`{"prompt":"one"}`)
	h.awaitState(info.ID, httpapi.Done, httpapi.Idle)

	var pending struct {
		Approvals []httpapi.Approval `json:"approvals"`
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
	final := h.awaitState(info.ID, httpapi.Done, httpapi.Idle)

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
	final := h.awaitState(info.ID, httpapi.Done)

	events := h.readStream(info.ID, nil, 0, -1)
	if len(events) == 0 {
		t.Fatal("empty stream")
	}
	last := events[len(events)-1]
	if last.name != "end" || last.id != -1 {
		t.Fatalf("stream ended with %+v; want an id-less terminal event", last)
	}
	var ended httpapi.SessionInfo
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
	settled := h.awaitState(info.ID, httpapi.Done, httpapi.Idle)

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
	h.awaitState(info.ID, httpapi.Done, httpapi.Idle)

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
// A turn can spend minutes inside one tool call, or parked on a human, and a
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
	done := h.awaitState(info.ID, httpapi.Done, httpapi.Idle)
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
		t.Fatal("credentialed CORS is never offered: there is no auth here to carry")
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
