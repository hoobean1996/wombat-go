package wombatd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"wombat/pkg/core/sandbox"
)

// remoteWorkdir is the home directory inside a task's sandbox, where the
// repository sits at ./repo. The model is told this is its working directory,
// exactly as the local workspace path is in the on-host case.
const remoteWorkdir = "/vercel"

// remoteRunner runs a session's commands in a remote sandbox instead of on this
// host. It is a drop-in sandbox.Runner: Run and Shell forward the command to an
// HTTP endpoint (the ondemandagent app's /api/agent/sandbox/exec), which relays
// it to the task's own microVM and returns the exit code and output. Nothing
// about the command runs locally, so this host's files and secrets are never in
// reach — the point of the whole arrangement.
//
// It mirrors osRunner's contract exactly: Run returns stdout and a
// *sandbox.CommandError on a non-zero exit (with Stderr), Shell returns the two
// streams combined with the [stderr] marker, and a transport failure is an
// error with the output captured so far.
type remoteRunner struct {
	url    string // the exec endpoint
	token  string // bearer for it
	task   string // which task's sandbox (its key)
	client *http.Client
}

func newRemoteRunner(url, token, task string) *remoteRunner {
	return &remoteRunner{
		url:   url,
		token: token,
		task:  task,
		// Above the endpoint's own per-command cap, so the server's timeout
		// (which returns a clean result) wins over the transport's.
		client: &http.Client{Timeout: 150 * time.Second},
	}
}

type remoteExecRequest struct {
	TaskKey string   `json:"task_key"`
	Argv    []string `json:"argv"`
	Cwd     string   `json:"cwd,omitempty"`
}

type remoteExecResponse struct {
	ExitCode int    `json:"exit_code"`
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	Error    string `json:"error"`
}

// call sends one command and returns the result, or an error if the endpoint
// could not be reached or refused the request (as opposed to the command
// itself failing, which comes back with a non-zero ExitCode).
func (r *remoteRunner) call(ctx context.Context, argv []string) (*remoteExecResponse, error) {
	body, err := json.Marshal(remoteExecRequest{TaskKey: r.task, Argv: argv})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+r.token)
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	var out remoteExecResponse
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &out); err != nil {
			return nil, fmt.Errorf("sandbox exec: bad response (%d): %s", resp.StatusCode, strings.TrimSpace(string(raw)))
		}
	}
	if resp.StatusCode/100 != 2 {
		msg := out.Error
		if msg == "" {
			msg = strings.TrimSpace(string(raw))
		}
		return nil, fmt.Errorf("sandbox exec: %s (%d)", msg, resp.StatusCode)
	}
	return &out, nil
}

// Run runs cmd with args and returns its stdout. A non-zero exit is a
// *sandbox.CommandError carrying the trimmed stderr, as osRunner does.
func (r *remoteRunner) Run(ctx context.Context, name string, args ...string) (string, error) {
	resp, err := r.call(ctx, append([]string{name}, args...))
	if err != nil {
		return "", &sandbox.CommandError{Name: name, Err: err}
	}
	if resp.ExitCode != 0 {
		return resp.Stdout, &sandbox.CommandError{Name: name, Err: fmt.Errorf("exit status %d", resp.ExitCode), Stderr: strings.TrimSpace(resp.Stderr)}
	}
	return resp.Stdout, nil
}

// Shell runs a script through sh -c and returns the two streams combined, with
// the [stderr] marker between them, matching osRunner.Shell so the bash tool
// reads the same shape wherever the command ran.
func (r *remoteRunner) Shell(ctx context.Context, script string) (string, error) {
	resp, err := r.call(ctx, []string{"/bin/sh", "-c", script})
	if err != nil {
		return "", &sandbox.CommandError{Name: "sh", Err: err}
	}
	out, errOut := resp.Stdout, resp.Stderr
	var combined string
	switch {
	case out != "" && errOut != "":
		combined = out + "\n[stderr]\n" + errOut
	case out != "":
		combined = out
	default:
		combined = errOut
	}
	if resp.ExitCode != 0 {
		return combined, &sandbox.CommandError{Name: "sh", Err: fmt.Errorf("exit status %d", resp.ExitCode)}
	}
	return combined, nil
}
