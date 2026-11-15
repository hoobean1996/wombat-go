package builtin

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/tool"
)

// exitErr is the minimal thing the Runner contract asks for: an error whose
// chain carries an ExitCode() int. Tools use it to tell "grep found nothing"
// (exit 1) apart from "grep failed" (exit 2), and matching on the METHOD
// rather than on *exec.ExitError is what lets a remote runner participate.
type exitErr struct{ code int }

func (e exitErr) Error() string { return fmt.Sprintf("exit status %d", e.code) }
func (e exitErr) ExitCode() int { return e.code }

// fakeRunner records the argv it was handed and returns a scripted result.
type fakeRunner struct {
	cmd    string
	args   []string
	script string

	out string
	err error
}

func (r *fakeRunner) Run(_ context.Context, cmd string, args ...string) (string, error) {
	r.cmd, r.args = cmd, args
	return r.out, r.err
}

func (r *fakeRunner) Shell(_ context.Context, script string) (string, error) {
	r.script = script
	return r.out, r.err
}

func TestExitCode(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantCode int
		wantOK   bool
	}{
		{"bare", exitErr{2}, 2, true},
		{"wrapped by a Runner", &sandbox.CommandError{Name: "grep", Err: exitErr{1}}, 1, true},
		{"wrapped with %w", fmt.Errorf("context: %w", exitErr{7}), 7, true},
		{"no exit status", errors.New("fork/exec: resource temporarily unavailable"), 0, false},
		{"nil", nil, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, hasCode := exitCode(tt.err)
			if code != tt.wantCode || hasCode != tt.wantOK {
				t.Errorf("exitCode(%v) = (%d, %v), want (%d, %v)", tt.err, code, hasCode, tt.wantCode, tt.wantOK)
			}
		})
	}
}

// ===== grep_search =====

// TestGrepExitOneMeansNoMatches is the classification that matters: grep exits
// 1 for "no matches", which is an ANSWER, not a failure. Reporting it as an
// error would make the model believe the search itself broke and retry it.
func TestGrepExitOneMeansNoMatches(t *testing.T) {
	r := &fakeRunner{err: &sandbox.CommandError{Name: "grep", Err: exitErr{1}}}
	out, err := call(t, GrepSearch(r), `{"pattern":"needle","path":"/work"}`)
	if err != nil {
		t.Fatalf("err = %v, want nil: exit 1 is an answer", err)
	}
	if out != "(no matches)" {
		t.Errorf("out = %q, want %q", out, "(no matches)")
	}
}

func TestGrepSearch(t *testing.T) {
	t.Run("returns matching lines", func(t *testing.T) {
		r := &fakeRunner{out: "/work/a.go:3:hit one\n/work/b.go:9:hit two\n"}
		out, err := call(t, GrepSearch(r), `{"pattern":"hit","path":"/work"}`)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if want := "/work/a.go:3:hit one\n/work/b.go:9:hit two"; out != want {
			t.Errorf("out = %q, want %q", out, want)
		}
	})

	// The pattern is a single argv element and never a shell word, and `--`
	// terminates the options so a pattern starting with `-` is a pattern.
	t.Run("builds an argv, not a shell string", func(t *testing.T) {
		r := &fakeRunner{out: "x\n"}
		if _, err := call(t, GrepSearch(r), `{"pattern":"-v; rm -rf /","path":"/work"}`); err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if r.cmd != "grep" {
			t.Errorf("cmd = %q, want grep", r.cmd)
		}
		if r.script != "" {
			t.Errorf("Shell was used with %q, want Run with an argv", r.script)
		}
		joined := strings.Join(r.args, " ")
		if !strings.HasPrefix(joined, "-rEnH --binary-files=without-match") {
			t.Errorf("args = %v, want them to start with -rEnH --binary-files=without-match", r.args)
		}
		n := len(r.args)
		if n < 3 || r.args[n-3] != "--" {
			t.Errorf("args = %v, want `--` immediately before the pattern", r.args)
		}
		if r.args[n-2] != "-v; rm -rf /" {
			t.Errorf("pattern arg = %q, want it passed through verbatim as one element", r.args[n-2])
		}
		if r.args[n-1] != "/work" {
			t.Errorf("path arg = %q, want /work", r.args[n-1])
		}
		for _, want := range grepExcludes {
			if !strings.Contains(joined, want) {
				t.Errorf("args = %v, want them to contain %q", r.args, want)
			}
		}
	})

	t.Run("case_insensitive adds -i", func(t *testing.T) {
		r := &fakeRunner{out: "x\n"}
		call(t, GrepSearch(r), `{"pattern":"p","path":"/w","case_insensitive":true}`)
		if !contains(r.args, "-i") {
			t.Errorf("args = %v, want -i", r.args)
		}

		r2 := &fakeRunner{out: "x\n"}
		call(t, GrepSearch(r2), `{"pattern":"p","path":"/w"}`)
		if contains(r2.args, "-i") {
			t.Errorf("args = %v, want no -i by default", r2.args)
		}
	})

	t.Run("max_results defaults and is hard-capped", func(t *testing.T) {
		tests := []struct {
			in   string
			want string
		}{
			{`{"pattern":"p","path":"/w"}`, fmt.Sprint(grepDefaultResults)},
			{`{"pattern":"p","path":"/w","max_results":0}`, fmt.Sprint(grepDefaultResults)},
			{`{"pattern":"p","path":"/w","max_results":-5}`, fmt.Sprint(grepDefaultResults)},
			{`{"pattern":"p","path":"/w","max_results":7}`, "7"},
			{`{"pattern":"p","path":"/w","max_results":99999}`, fmt.Sprint(grepMaxResults)},
		}
		for _, tt := range tests {
			r := &fakeRunner{out: "x\n"}
			call(t, GrepSearch(r), tt.in)
			got := argAfter(r.args, "-m")
			if got != tt.want {
				t.Errorf("%s: -m %s, want -m %s", tt.in, got, tt.want)
			}
		}
	})

	// grep's -m is per FILE, so a recursive search can return far more than
	// max_results lines. The total is clamped here.
	t.Run("clamps the total line count and says how many were dropped", func(t *testing.T) {
		var lines []string
		for i := 0; i < 10; i++ {
			lines = append(lines, fmt.Sprintf("/w/f%d.go:1:hit", i))
		}
		r := &fakeRunner{out: strings.Join(lines, "\n") + "\n"}
		out, err := call(t, GrepSearch(r), `{"pattern":"hit","path":"/w","max_results":3}`)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		body, note, found := strings.Cut(out, "\n[...")
		if !found {
			t.Fatalf("out = %q, want the clamp note", out)
		}
		if got := len(strings.Split(body, "\n")); got != 3 {
			t.Errorf("returned %d lines, want 3", got)
		}
		if !strings.Contains(note, "7 more matching lines") {
			t.Errorf("note = %q, want it to report 7 more lines", note)
		}
	})

	t.Run("empty stdout is no matches", func(t *testing.T) {
		for _, out := range []string{"", "\n"} {
			r := &fakeRunner{out: out}
			got, err := call(t, GrepSearch(r), `{"pattern":"p","path":"/w"}`)
			if err != nil || got != "(no matches)" {
				t.Errorf("out %q: got (%q, %v), want ((no matches), nil)", out, got, err)
			}
		}
	})

	// exit 2 is a bad regex or an unreadable path: a real failure.
	t.Run("any other exit code is a real failure", func(t *testing.T) {
		r := &fakeRunner{err: &sandbox.CommandError{Name: "grep", Err: exitErr{2}, Stderr: "grep: unmatched ( or \\("}}
		_, err := call(t, GrepSearch(r), `{"pattern":"(","path":"/w"}`)
		if err == nil {
			t.Fatal("err = nil, want exit 2 reported")
		}
		if !strings.Contains(err.Error(), "unmatched") {
			t.Errorf("err = %q, want the stderr diagnostic carried through", err)
		}
	})

	t.Run("a spawn failure with no exit code is a real failure", func(t *testing.T) {
		r := &fakeRunner{err: &sandbox.CommandError{Name: "grep", Err: syscall.ENOENT}}
		if _, err := call(t, GrepSearch(r), `{"pattern":"p","path":"/w"}`); err == nil {
			t.Fatal("err = nil, want the spawn failure")
		}
	})

	t.Run("validates its inputs", func(t *testing.T) {
		r := &fakeRunner{out: "x\n"}
		d := GrepSearch(r)
		if _, err := call(t, d, `{"pattern":"","path":"/w"}`); err == nil ||
			err.Error() != "field 'pattern' must not be empty" {
			t.Errorf("err = %v, want the empty-pattern message", err)
		}
		if _, err := call(t, d, `{"pattern":"p","path":"rel"}`); err == nil ||
			!strings.Contains(err.Error(), "must be an absolute path") {
			t.Errorf("err = %v, want the absolute-path message", err)
		}
		if r.cmd != "" {
			t.Errorf("the runner was invoked (%q) despite invalid input, want it refused first", r.cmd)
		}
	})

	// Read-only despite spawning a process: the argv is fixed and grep cannot
	// mutate anything. NeedExec is what hides it on a host with no subprocess.
	t.Run("is CapReadOnly but NeedExec", func(t *testing.T) {
		d := GrepSearch(&fakeRunner{})
		if !d.Has(tool.CapReadOnly) {
			t.Errorf("Caps = %b, want CapReadOnly so it survives a verifier filter", d.Caps)
		}
		if d.Has(tool.CapExec) {
			t.Errorf("Caps = %b, want no CapExec: Cap is what a tool DOES", d.Caps)
		}
		if d.Needs&tool.NeedExec == 0 {
			t.Errorf("Needs = %b, want NeedExec", d.Needs)
		}
		if !d.Idempotent || d.Retryable == nil {
			t.Error("want Idempotent with a classifier so the retry actually engages")
		}
	})
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func argAfter(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

// ===== bash =====

func TestBash(t *testing.T) {
	t.Run("runs the command in exec_dir", func(t *testing.T) {
		r := &fakeRunner{out: "hello\n"}
		out, err := call(t, Bash(r), `{"command":"echo hello","exec_dir":"/work/proj"}`)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if out != "hello\n" {
			t.Errorf("out = %q, want %q", out, "hello\n")
		}
		// `cd DIR && (cmd)`: the subshell makes every nested shell inherit the
		// directory, and the cd gives a clear error with no TOCTOU pre-check.
		if want := "cd '/work/proj' && (echo hello)"; r.script != want {
			t.Errorf("script = %q, want %q", r.script, want)
		}
	})

	t.Run("exec_dir is single-quoted", func(t *testing.T) {
		r := &fakeRunner{out: "x"}
		call(t, Bash(r), `{"command":"pwd","exec_dir":"/work/it's here"}`)
		if want := `cd '/work/it'\''s here' && (pwd)`; r.script != want {
			t.Errorf("script = %q, want %q", r.script, want)
		}
	})

	t.Run("exec_dir is required and must be absolute", func(t *testing.T) {
		r := &fakeRunner{}
		d := Bash(r)
		for _, in := range []string{
			`{"command":"ls"}`,
			`{"command":"ls","exec_dir":""}`,
			`{"command":"ls","exec_dir":"relative"}`,
		} {
			if _, err := call(t, d, in); err == nil || !strings.Contains(err.Error(), "exec_dir") {
				t.Errorf("%s: err = %v, want an exec_dir validation error", in, err)
			}
		}
		if _, err := call(t, d, `{"command":"","exec_dir":"/w"}`); err == nil ||
			err.Error() != "field 'command' must not be empty" {
			t.Errorf("err = %v, want the empty-command message", err)
		}
		if r.script != "" {
			t.Errorf("the runner ran %q despite invalid input, want nothing", r.script)
		}
	})

	t.Run("silent success reports (no output)", func(t *testing.T) {
		r := &fakeRunner{out: ""}
		out, err := call(t, Bash(r), `{"command":"true","exec_dir":"/w"}`)
		if err != nil || out != "(no output)" {
			t.Errorf("(%q, %v), want ((no output), nil)", out, err)
		}
	})

	t.Run("a non-zero exit carries the code and the output", func(t *testing.T) {
		r := &fakeRunner{out: "boom\n", err: &sandbox.CommandError{Name: "sh", Err: exitErr{3}}}
		_, err := call(t, Bash(r), `{"command":"false","exec_dir":"/w"}`)
		if err == nil {
			t.Fatal("err = nil, want the exit reported")
		}
		if want := "exit 3:\nboom\n"; err.Error() != want {
			t.Errorf("err = %q, want %q", err, want)
		}
	})

	t.Run("a failure with no exit code still carries the output", func(t *testing.T) {
		r := &fakeRunner{out: "partial", err: errors.New("spawn failed")}
		_, err := call(t, Bash(r), `{"command":"x","exec_dir":"/w"}`)
		if err == nil || !strings.Contains(err.Error(), "spawn failed") || !strings.Contains(err.Error(), "partial") {
			t.Errorf("err = %v, want both the cause and the captured output", err)
		}
	})

	// A dead context outranks the exit status: the process was SIGKILLed by the
	// cancellation, so "exit -1" would describe the symptom and hide the cause.
	t.Run("a deadline outranks the exit status", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
		defer cancel()
		<-ctx.Done()

		r := &fakeRunner{out: "abc", err: &sandbox.CommandError{Name: "sh", Err: exitErr{-1}}}
		_, err := Bash(r).Fn(ctx, []byte(`{"command":"sleep 100","exec_dir":"/w"}`))
		if err == nil {
			t.Fatal("err = nil, want a timeout error")
		}
		if !strings.Contains(err.Error(), "timed out") {
			t.Errorf("err = %q, want it to say the command timed out", err)
		}
		if !strings.Contains(err.Error(), "abc") {
			t.Errorf("err = %q, want it to keep the partial output", err)
		}
		if strings.Contains(err.Error(), "exit -1") {
			t.Errorf("err = %q, want the cause, not the SIGKILL exit status", err)
		}
	})

	t.Run("a cancellation reports the cause", func(t *testing.T) {
		abort := errors.New("budget exhausted")
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(abort)

		r := &fakeRunner{out: "", err: errors.New("signal: killed")}
		_, err := Bash(r).Fn(ctx, []byte(`{"command":"sleep 100","exec_dir":"/w"}`))
		if !errors.Is(err, abort) {
			t.Errorf("err = %v, want it to wrap the cancellation cause %v", err, abort)
		}
	})

	// bash can curl, so declaring CapNetwork is what makes an offline agent
	// built with tool.OnlyCaps actually exclude it. Needs deliberately omits
	// NeedNetwork: a sandbox without a network should still get a shell.
	t.Run("capabilities and needs", func(t *testing.T) {
		d := Bash(&fakeRunner{})
		for _, c := range []tool.Cap{tool.CapExec, tool.CapMutating, tool.CapNetwork} {
			if !d.Has(c) {
				t.Errorf("Caps = %b, want it to include %b", d.Caps, c)
			}
		}
		if d.Needs&tool.NeedNetwork != 0 {
			t.Errorf("Needs = %b, want NO NeedNetwork", d.Needs)
		}
		if d.Needs != tool.NeedExec|tool.NeedFSRead|tool.NeedFSWrite {
			t.Errorf("Needs = %b, want NeedExec|NeedFSRead|NeedFSWrite", d.Needs)
		}
		if d.Idempotent {
			t.Error("Idempotent = true, want false: a replayed `git push` is not a free retry")
		}
		// The classifier is inert here (Idempotent=false wins) but is kept as
		// declarative metadata, so its absence would be a regression.
		if d.Retryable == nil {
			t.Error("Retryable = nil, want retryExec kept as the truthful answer")
		}
		if d.Timeout != bashTimeout {
			t.Errorf("Timeout = %v, want %v to match the description", d.Timeout, bashTimeout)
		}
		if !strings.Contains(d.Description, "60s") {
			t.Errorf("description does not quote the 60s timeout, want it to match Def.Timeout")
		}
	})

	t.Run("a nil Runner panics at construction", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("Bash(nil) did not panic, want a construction-time panic")
			}
		}()
		Bash(nil)
	})
}

func TestShQuote(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"/plain/path", "'/plain/path'"},
		{"/with space", "'/with space'"},
		{"/it's", `'/it'\''s'`},
		{"", "''"},
		{"$(rm -rf /)", "'$(rm -rf /)'"},
		{"`whoami`", "'`whoami`'"},
		{"a'b'c", `'a'\''b'\''c'`},
	}
	for _, tt := range tests {
		if got := shQuote(tt.in); got != tt.want {
			t.Errorf("shQuote(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestBashNonZeroExitIsCallerFault pins the distinction that keeps a coding
// agent alive.
//
// bash reports a non-zero exit as an error, and it has to: a model that read a
// failing build as a successful one would ship the break. But the SHELL is
// fine — it spawned the process, waited, and told the truth. Every coding
// agent generates a long string of these by doing its job correctly
// (`go build` on half-finished code, `grep` finding nothing, a test suite that
// is red until the fix lands), and when they counted as tool failures the
// circuit breaker took bash away at exactly the moment the agent needed it to
// prove the fix worked.
//
// A process that never started is the other case, and it IS the tool failing.
func TestBashNonZeroExitIsCallerFault(t *testing.T) {
	t.Run("a command that ran and failed", func(t *testing.T) {
		r := &fakeRunner{out: "./x.go:4:2: undefined: foo\n", err: exitErr{2}}
		_, err := call(t, Bash(r), `{"command":"go build ./...","exec_dir":"/work"}`)
		if err == nil {
			t.Fatal("err = nil, want the exit status reported")
		}
		if !tool.IsCallerFault(err) {
			t.Errorf("IsCallerFault(%v) = false — a failing build would trip the breaker", err)
		}
		if !strings.Contains(err.Error(), "undefined: foo") {
			t.Errorf("err = %q, want the compiler output kept", err)
		}
	})

	t.Run("a command that never ran", func(t *testing.T) {
		r := &fakeRunner{err: errors.New("fork/exec: resource temporarily unavailable")}
		_, err := call(t, Bash(r), `{"command":"true","exec_dir":"/work"}`)
		if err == nil {
			t.Fatal("err = nil, want the spawn failure reported")
		}
		if tool.IsCallerFault(err) {
			t.Errorf("IsCallerFault(%v) = true, want false — the shell itself failed", err)
		}
	})
}

// The end-to-end version: bash over a real runner in a real directory.
func TestBashOverARealRunner(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("spawns POSIX utilities")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "here.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	d := Bash(sandbox.Unconfined(""))

	t.Run("runs in exec_dir", func(t *testing.T) {
		out, err := call(t, d, fmt.Sprintf(`{"command":"ls","exec_dir":%q}`, dir))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if strings.TrimSpace(out) != "here.txt" {
			t.Errorf("out = %q, want %q", out, "here.txt")
		}
	})

	// The cd gives a clear error if exec_dir does not exist, with no TOCTOU
	// pre-check.
	t.Run("a missing exec_dir fails with the shell's own message", func(t *testing.T) {
		_, err := call(t, d, fmt.Sprintf(`{"command":"ls","exec_dir":%q}`, filepath.Join(dir, "nope")))
		if err == nil {
			t.Fatal("err = nil, want the cd failure")
		}
		if !strings.Contains(err.Error(), "exit ") {
			t.Errorf("err = %q, want a non-zero exit reported", err)
		}
	})

	t.Run("a directory with a quote in its name is handled", func(t *testing.T) {
		odd := filepath.Join(dir, "it's here")
		if err := os.Mkdir(odd, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(odd, "inside.txt"), nil, 0o644); err != nil {
			t.Fatal(err)
		}
		out, err := call(t, d, fmt.Sprintf(`{"command":"ls","exec_dir":%q}`, odd))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if strings.TrimSpace(out) != "inside.txt" {
			t.Errorf("out = %q, want %q", out, "inside.txt")
		}
	})
}
