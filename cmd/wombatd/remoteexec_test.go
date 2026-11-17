package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"wombat/pkg/core/sandbox"
)

// fakeExec stands in for the app's /api/agent/sandbox/exec. It records the last
// request and answers with whatever the test set.
type fakeExec struct {
	srv     *httptest.Server
	auth    string
	lastReq remoteExecRequest
	status  int
	reply   remoteExecResponse
}

func newFakeExec(t *testing.T) *fakeExec {
	f := &fakeExec{status: 200}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.auth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &f.lastReq)
		w.WriteHeader(f.status)
		_ = json.NewEncoder(w).Encode(f.reply)
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func newRunner(f *fakeExec) *remoteRunner {
	return newRemoteRunner(f.srv.URL, "s3cret", "task-123")
}

func TestRemoteRunSuccess(t *testing.T) {
	f := newFakeExec(t)
	f.reply = remoteExecResponse{ExitCode: 0, Stdout: "hello\n"}
	out, err := newRunner(f).Run(context.Background(), "grep", "-rn", "foo", "/vercel/repo")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "hello\n" {
		t.Fatalf("stdout = %q", out)
	}
	if f.auth != "Bearer s3cret" {
		t.Fatalf("auth = %q", f.auth)
	}
	if f.lastReq.TaskKey != "task-123" {
		t.Fatalf("task_key = %q", f.lastReq.TaskKey)
	}
	if got := strings.Join(f.lastReq.Argv, " "); got != "grep -rn foo /vercel/repo" {
		t.Fatalf("argv = %q", got)
	}
}

func TestRemoteRunNonZeroExitIsCommandError(t *testing.T) {
	f := newFakeExec(t)
	f.reply = remoteExecResponse{ExitCode: 2, Stdout: "partial", Stderr: "  boom  "}
	out, err := newRunner(f).Run(context.Background(), "grep", "bad[")
	if out != "partial" {
		t.Fatalf("stdout = %q (a failing command's output must survive)", out)
	}
	var ce *sandbox.CommandError
	if !errors.As(err, &ce) {
		t.Fatalf("want *sandbox.CommandError, got %T: %v", err, err)
	}
	if ce.Name != "grep" || ce.Stderr != "boom" {
		t.Fatalf("CommandError = %+v (stderr should be trimmed)", ce)
	}
}

func TestRemoteRunTransportErrorIsCommandError(t *testing.T) {
	f := newFakeExec(t)
	f.status = 503
	f.reply = remoteExecResponse{Error: "sandbox exec is off"}
	_, err := newRunner(f).Run(context.Background(), "ls")
	var ce *sandbox.CommandError
	if !errors.As(err, &ce) {
		t.Fatalf("want *sandbox.CommandError, got %T: %v", err, err)
	}
	if !strings.Contains(err.Error(), "sandbox exec is off") {
		t.Fatalf("error should carry the endpoint's message: %v", err)
	}
}

func TestRemoteShellCombinesStreams(t *testing.T) {
	f := newFakeExec(t)
	f.reply = remoteExecResponse{ExitCode: 0, Stdout: "out", Stderr: "warn"}
	out, err := newRunner(f).Shell(context.Background(), "echo hi")
	if err != nil {
		t.Fatalf("Shell: %v", err)
	}
	if out != "out\n[stderr]\nwarn" {
		t.Fatalf("combined = %q", out)
	}
	if got := strings.Join(f.lastReq.Argv, " "); got != "/bin/sh -c echo hi" {
		t.Fatalf("argv = %q", got)
	}
}

func TestRemoteShellNonZeroExit(t *testing.T) {
	f := newFakeExec(t)
	f.reply = remoteExecResponse{ExitCode: 1, Stdout: "", Stderr: "nope"}
	out, err := newRunner(f).Shell(context.Background(), "false")
	if out != "nope" {
		t.Fatalf("combined = %q", out)
	}
	var ce *sandbox.CommandError
	if !errors.As(err, &ce) || ce.Name != "sh" {
		t.Fatalf("want sh CommandError, got %T: %v", err, err)
	}
}
