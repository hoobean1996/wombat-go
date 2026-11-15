package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/daemon"
	"wombat/pkg/httpapi"
)

func resolved(t *testing.T, dir string) string {
	t.Helper()
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatal(err)
	}
	return real
}

// TestWorkspaceFollowsSymlinks: the check was lexical, so a link inside the
// root to anywhere at all passed it, and the file tools, the permission rules
// and the sandbox then all worked in the directory it pointed at.
func TestWorkspaceFollowsSymlinks(t *testing.T) {
	root, outside := resolved(t, t.TempDir()), resolved(t, t.TempDir())
	if err := os.Mkdir(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "out")); err != nil {
		t.Skip(err)
	}
	if err := os.Symlink(filepath.Join(root, "sub"), filepath.Join(root, "alias")); err != nil {
		t.Fatal(err)
	}
	f := &factory{root: root}

	for _, tc := range []struct {
		requested, want string
	}{
		{"", root},
		{"sub", filepath.Join(root, "sub")},
		{"alias", filepath.Join(root, "sub")}, // a link that stays inside resolves to where it points
		{"sub/not-yet", filepath.Join(root, "sub", "not-yet")},
	} {
		got, err := f.workspace(tc.requested)
		if err != nil || got != tc.want {
			t.Errorf("workspace(%q) = %q, %v; want %q", tc.requested, got, err, tc.want)
		}
	}
	for _, bad := range []string{"out", "out/deeper", "../escape", outside} {
		if got, err := f.workspace(bad); !errors.Is(err, daemon.ErrBadRequest) {
			t.Errorf("workspace(%q) = %q, %v; want it refused as a bad request", bad, got, err)
		}
	}
}

// TestSessionWithoutAPolicyReadsOnlyItsWorkspace: with permission off the
// shell is taken away, and the read-only tools used to be left free to read
// any file the server could — from a browser, with no one to ask.
func TestSessionWithoutAPolicyReadsOnlyItsWorkspace(t *testing.T) {
	root, outside := resolved(t, t.TempDir()), resolved(t, t.TempDir())
	inside := filepath.Join(root, "notes.txt")
	secret := filepath.Join(outside, "id_ed25519")
	if err := os.WriteFile(inside, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(secret, []byte("PRIVATE KEY"), 0o600); err != nil {
		t.Fatal(err)
	}

	view := func(id, path string) llm.ContentBlock {
		in, _ := json.Marshal(map[string]string{"path": path})
		return llm.ToolUse{ID: llm.ToolUseID(id), Name: "view_file", Input: in}
	}
	client := llm.ClientFunc(func(_ context.Context, req llm.Request) (llm.Response, error) {
		if len(req.Messages) == 1 {
			return llm.Response{Content: []llm.ContentBlock{view("in", inside), view("out", secret)}, StopReason: llm.StopToolUse}, nil
		}
		return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: "done"}}, StopReason: llm.StopEndTurn}, nil
	})
	cfg := defaults()
	cfg.Model.Name, cfg.Tools.Permission, cfg.Turn.MaxIters = "m", "off", 4
	f := &factory{cfg: cfg, client: client, root: root, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	a, err := f.build(daemon.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	r := a.Start(context.Background(), core.Ask("read both"))
	for r.Next() {
	}
	defer r.Close()
	if err := r.Err(); err != nil {
		t.Fatal(err)
	}
	results := map[llm.ToolUseID]llm.ToolResult{}
	for _, m := range r.Messages() {
		for _, b := range m.Content {
			if tr, ok := b.(llm.ToolResult); ok {
				results[tr.ToolUseID] = tr
			}
		}
	}
	if got := results["in"]; got.IsError || !strings.Contains(got.Content, "hello") {
		t.Errorf("reading inside the workspace: %+v, want the file", got)
	}
	if got := results["out"]; !got.IsError || strings.Contains(got.Content, "PRIVATE") {
		t.Errorf("reading outside the workspace: %+v, want it refused", got)
	}
}

// TestAccess: a loopback server answers only to the names this machine uses
// for it, a token is asked for wherever one is set, and a server anyone can
// reach without one says so.
func TestAccess(t *testing.T) {
	mgr, err := daemon.NewManager(daemon.ManagerConfig{
		Build:  func(daemon.SessionOptions) (*core.Agent, error) { return nil, errors.New("unused") },
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer mgr.Close()

	for _, tc := range []struct {
		addr, token           string
		hostChecked, needsTok bool
		warns                 bool
	}{
		{addr: "127.0.0.1:8080", hostChecked: true},
		{addr: "localhost:8080", hostChecked: true},
		{addr: "[::1]:8080", hostChecked: true},
		{addr: "127.0.0.1:8080", token: "t", hostChecked: true, needsTok: true},
		{addr: ":8080", warns: true},
		{addr: "0.0.0.0:8080", token: "t", needsTok: true},
	} {
		var log bytes.Buffer
		opts := access(tc.addr, tc.token, slog.New(slog.NewTextHandler(&log, nil)))
		srv := httptest.NewServer(httpapi.New(mgr, opts...))
		get := func(host string, tok string) int {
			req, _ := http.NewRequest(http.MethodGet, srv.URL+"/api/sessions", nil)
			req.Host = host
			if tok != "" {
				req.Header.Set("Authorization", "Bearer "+tok)
			}
			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			return resp.StatusCode
		}

		if got := get("rebound.example:8080", tc.token); (got == http.StatusMisdirectedRequest) != tc.hostChecked {
			t.Errorf("%s: a request for another host = %d; host checked should be %v", tc.addr, got, tc.hostChecked)
		}
		if got := get("localhost:8080", ""); (got == http.StatusUnauthorized) != tc.needsTok {
			t.Errorf("%s: a request without the token = %d; token needed should be %v", tc.addr, got, tc.needsTok)
		}
		if tc.token != "" {
			if got := get("localhost:8080", tc.token); got != http.StatusOK {
				t.Errorf("%s: a request with the token = %d, want 200", tc.addr, got)
			}
		}
		if warned := strings.Contains(log.String(), "no server.token"); warned != tc.warns {
			t.Errorf("%s with token %q: warned = %v, want %v\n%s", tc.addr, tc.token, warned, tc.warns, log.String())
		}
		srv.Close()
	}
}
