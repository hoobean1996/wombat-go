package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/metric"
	"wombat/pkg/core/sandbox"
	"wombat/pkg/daemon"
	"wombat/pkg/httpapi"
)

// The model section is all a provider client is built from — no environment —
// so each of its keys has to reach the request.
func TestTheModelSectionReachesTheProvider(t *testing.T) {
	for _, tc := range []struct {
		provider, path, keyHeader, keyValue, reply string
	}{
		{
			provider: "openai", path: "/v1/chat/completions",
			keyHeader: "Authorization", keyValue: "Bearer sk-from-the-file",
			reply: `{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		},
		{
			provider: "anthropic", path: "/v1/messages",
			keyHeader: "X-Api-Key", keyValue: "sk-from-the-file",
			reply: `{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
		},
	} {
		t.Run(tc.provider, func(t *testing.T) {
			t.Setenv("OPENAI_API_KEY", "sk-from-the-environment")
			t.Setenv("ANTHROPIC_API_KEY", "sk-from-the-environment")
			var got *http.Request
			var body string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				got, body = r, string(b)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.reply)
			}))
			defer srv.Close()

			base := srv.URL
			if tc.provider == "openai" {
				base += "/v1"
			}
			client, err := newClient(modelConfig{
				Provider:  tc.provider,
				Name:      "the-configured-model",
				BaseURL:   base,
				APIKey:    "sk-from-the-file",
				Headers:   map[string]string{"X-Route": "my-team"},
				Stream:    "never",
				MaxTokens: 100,
			}, slog.New(slog.NewTextHandler(io.Discard, nil)))
			if err != nil {
				t.Fatal(err)
			}
			// A delta sink would make an auto client stream; never must not.
			_, err = client.Complete(context.Background(), llm.Request{
				Messages: []llm.Message{llm.UserText("hello")},
				OnDelta:  func(llm.Delta) {},
			})
			if err != nil {
				t.Fatal(err)
			}
			if got.URL.Path != tc.path {
				t.Errorf("path = %s, want %s: base_url was not used", got.URL.Path, tc.path)
			}
			if v := got.Header.Get(tc.keyHeader); v != tc.keyValue {
				t.Errorf("%s = %q, want the file's key, not the environment's", tc.keyHeader, v)
			}
			if v := got.Header.Get("X-Route"); v != "my-team" {
				t.Errorf("X-Route = %q: headers were not sent", v)
			}
			if !strings.Contains(body, `"the-configured-model"`) {
				t.Errorf("the request does not name the configured model: %s", body)
			}
			if strings.Contains(body, `"stream":true`) {
				t.Errorf("stream: never streamed: %s", body)
			}
		})
	}
}

// newFactory turns the configuration into the shared resources, and refuses
// a working directory that is not there.
func TestNewFactory(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg := defaults()
	cfg.Tools.WorkingDir = t.TempDir()
	cfg.Tools.Confine, cfg.Tools.ConfineNetwork = true, "none"
	cfg.Tools.DenyRead = []string{"/etc/wombat", "/srv/wombatd.yaml"}
	f, err := newFactory(cfg, &recorders{}, logger)
	if err != nil {
		t.Fatal(err)
	}
	defer f.close()
	if f.root != resolved(t, cfg.Tools.WorkingDir) {
		t.Errorf("root = %s, want the working directory with its links resolved", f.root)
	}
	if f.registry == nil || f.metrics == nil {
		t.Error("server.metrics is on by default, and there is no registry")
	}
	if f.sandbox == nil || *f.sandbox != sandbox.NetNone {
		t.Errorf("sandbox = %v, want the none network", f.sandbox)
	}
	if !slices.Equal(f.deny, cfg.Tools.DenyRead) {
		t.Errorf("deny = %q, want tools.deny_read %q", f.deny, cfg.Tools.DenyRead)
	}

	cfg.Tools.WorkingDir = filepath.Join(t.TempDir(), "missing")
	if _, err := newFactory(cfg, &recorders{}, logger); err == nil || !strings.Contains(err.Error(), "tools.working_dir") {
		t.Errorf("err = %v, want a refusal naming tools.working_dir", err)
	}
}

// The recorders write what record asks for: the model's calls on the tape,
// and on close, the trace rendered as a report.
// A tools.disable entry has to name a tool: a misspelt one would leave the
// tool it meant on offer, so wombatd does not start.
func TestADisabledToolHasToBeATool(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg := defaults()
	cfg.Tools.WorkingDir = t.TempDir()
	cfg.Tools.Disable = []string{"write_file", "edit_file"}
	if _, err := newFactory(cfg, &recorders{}, logger); err == nil || !strings.Contains(err.Error(), `tools.disable: "edit_file" names no tool`) {
		t.Fatalf("err = %v, want edit_file refused by name", err)
	}
	cfg.Tools.Disable = []string{"write_file", "str_replace", "view_media"}
	f, err := newFactory(cfg, &recorders{}, logger)
	if err != nil {
		t.Fatal(err)
	}
	f.close()
}

func TestRecordersKeepTheTapeAndRenderTheTrace(t *testing.T) {
	dir := t.TempDir()
	cfg := recordConfig{
		Tape:      filepath.Join(dir, "tape.jsonl"),
		TapeMode:  "record",
		Trace:     filepath.Join(dir, "trace.ndjson"),
		TraceHTML: filepath.Join(dir, "trace.html"),
	}
	rec, err := openRecorders(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	if rec.tracer == nil {
		t.Error("record.trace is set, and there is no tracer")
	}
	client := rec.wrap(llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) { return say("hi"), nil }))
	if _, err := client.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.UserText("hello")}}); err != nil {
		t.Fatal(err)
	}
	rec.close()

	if b, err := os.ReadFile(cfg.Tape); err != nil || !strings.Contains(string(b), `"kind":"llm"`) {
		t.Errorf("the tape did not record the model's call: %v\n%s", err, b)
	}
	if b, err := os.ReadFile(cfg.TraceHTML); err != nil || !strings.Contains(string(b), "<html") {
		t.Errorf("the trace was not rendered on close: %v", err)
	}
}

// What the server section says is what the handler enforces: the token is
// required, and /metrics is served.
func TestTheHandlerDoesWhatTheServerSectionSays(t *testing.T) {
	f, _ := testFactory(t, &scripted{turn: func(int, llm.Request) llm.Response { return say("ok") }}, func(c *config) {
		c.Server.Token = "s3cret"
	})
	f.registry = metric.NewRegistry()
	probe, err := f.build(daemon.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	mgr, err := newManager(f.cfg, daemon.MemoryStore(), "test", "", f.build, f.logger)
	if err != nil {
		t.Fatal(err)
	}
	defer mgr.Close()
	srv := httptest.NewServer(httpapi.New(mgr, handlerOptions(f.cfg, f, probe, f.logger)...))
	defer srv.Close()

	get := func(path, token string) int {
		req, _ := http.NewRequest(http.MethodGet, srv.URL+path, nil)
		req.Host = "localhost:8080"
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if got := get("/api/sessions", ""); got != http.StatusUnauthorized {
		t.Errorf("without the token: %d, want 401", got)
	}
	if got := get("/api/sessions", "s3cret"); got != http.StatusOK {
		t.Errorf("with the token: %d, want 200", got)
	}
	if got := get("/metrics", "s3cret"); got != http.StatusOK {
		t.Errorf("/metrics: %d, want 200", got)
	}
}

// turn.budget_usd is what the manager holds each turn to: the call that
// spends past it is the last one.
func TestATurnStopsAtItsBudget(t *testing.T) {
	cl := &scripted{turn: func(n int, req llm.Request) llm.Response {
		r := call(fmt.Sprintf("v%d", n), "list_files", map[string]string{"path": "."})
		r.Model = "claude-sonnet-4" // priced: 100k tokens each way is $1.80
		r.Usage = llm.Usage{InputTokens: 100_000, OutputTokens: 100_000}
		return r
	}}
	f, _ := testFactory(t, nil, func(c *config) { c.Turn.BudgetUSD = 0.5 })
	f.client = llm.Chain(cl, core.TrackCost(llm.DefaultPricing))
	mgr, err := newManager(f.cfg, daemon.MemoryStore(), "test", "", f.build, f.logger)
	if err != nil {
		t.Fatal(err)
	}
	defer mgr.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	s, err := mgr.Create(ctx, daemon.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Send(ctx, "look around"); err != nil {
		t.Fatal(err)
	}
	for _, fr := range s.Follow(ctx, 0) {
		if end, ok := fr.Event.(daemon.TurnEnded); ok {
			if end.ErrorKind != "budget_exhausted" {
				t.Errorf("the turn ended %q (%s), want budget_exhausted", end.ErrorKind, end.Error)
			}
			if n := len(cl.requests()); n != 1 {
				t.Errorf("%d model calls, want the one that crossed the budget", n)
			}
			return
		}
	}
	t.Fatal("the turn never ended")
}

// Under the workspace policy a command waits for a person: the turn stops,
// the outbox asks, and once it is decided the turn carries on and the command
// runs. The agent's own tools never ask — a message it sends goes out at once.
func TestACommandWaitsForADecisionAndTheAgentsOwnToolsDoNot(t *testing.T) {
	f, root := testFactory(t, nil, func(c *config) { c.Daemon.Poll = 5 * time.Millisecond })
	cl := &scripted{turn: func(n int, req llm.Request) llm.Response {
		switch n {
		case 0:
			return call("m1", "send_message", map[string]string{"text": "starting"})
		case 1:
			return call("b1", "bash", map[string]string{"command": "echo ok > made-by-bash", "exec_dir": root})
		default:
			return say("done")
		}
	}}
	f.client = cl
	mgr, err := newManager(f.cfg, daemon.MemoryStore(), "test", "", f.build, f.logger)
	if err != nil {
		t.Fatal(err)
	}
	defer mgr.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	s, err := mgr.Create(ctx, daemon.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Send(ctx, "make the file"); err != nil {
		t.Fatal(err)
	}
	ended := func(turn int) daemon.TurnEnded {
		for _, fr := range s.Follow(ctx, 0) {
			if end, ok := fr.Event.(daemon.TurnEnded); ok && end.Turn == turn {
				return end
			}
		}
		t.Fatalf("turn %d never ended", turn)
		return daemon.TurnEnded{}
	}

	stopped := ended(1)
	if stopped.Outcome != "deferred" || len(stopped.Approvals) != 1 || stopped.Approvals[0].Tool != "bash" {
		t.Fatalf("turn 1 = %+v, want it stopped on the command", stopped)
	}
	if _, err := os.Stat(filepath.Join(root, "made-by-bash")); err == nil {
		t.Fatal("the command ran before anyone decided")
	}
	msgs, _ := mgr.Outbox(ctx, daemon.OutboxFilter{Session: s.ID()}, 0)
	if len(msgs) != 2 || msgs[0].Kind != daemon.MessageSent || msgs[0].Text != "starting" || msgs[1].Kind != daemon.MessageApproval {
		t.Fatalf("outbox = %+v, want the agent's message, sent without asking, then the approval", msgs)
	}

	if _, err := s.Resolve(ctx, "b1", true); err != nil {
		t.Fatal(err)
	}
	if done := ended(2); done.Answer != "done" {
		t.Fatalf("turn 2 = %+v, want the answer", done)
	}
	if _, err := os.Stat(filepath.Join(root, "made-by-bash")); err != nil {
		t.Fatalf("the approved command did not run: %v", err)
	}
}
