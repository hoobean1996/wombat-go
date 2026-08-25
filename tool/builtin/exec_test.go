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
	"unicode/utf8"
)

func TestLimitBuf(t *testing.T) {
	t.Run("under the cap", func(t *testing.T) {
		b := &limitBuf{max: 100}
		n, err := b.Write([]byte("hello"))
		if n != 5 || err != nil {
			t.Fatalf("Write = (%d, %v), want (5, nil)", n, err)
		}
		if got := b.String(); got != "hello" {
			t.Errorf("String() = %q, want %q", got, "hello")
		}
	})

	// Write never reports an error: os/exec turns a writer error into the
	// result of Wait, so refusing bytes here would masquerade as a command
	// failure. We take everything and drop what does not fit.
	t.Run("over the cap keeps a prefix and counts the rest", func(t *testing.T) {
		b := &limitBuf{max: 10}
		n, err := b.Write([]byte(strings.Repeat("a", 25)))
		if n != 25 || err != nil {
			t.Fatalf("Write = (%d, %v), want (25, nil): a short write would look like a command failure", n, err)
		}
		n, err = b.Write([]byte(strings.Repeat("b", 5)))
		if n != 5 || err != nil {
			t.Fatalf("second Write = (%d, %v), want (5, nil)", n, err)
		}

		got := b.String()
		if !strings.HasPrefix(got, strings.Repeat("a", 10)) {
			t.Errorf("String() = %q, want it to start with the first 10 bytes", got)
		}
		if !strings.Contains(got, "20 more bytes dropped at the 10-byte capture limit") {
			t.Errorf("String() = %q, want it to report 20 dropped bytes", got)
		}
	})

	// A command that emits binary produces a tool_result the provider rejects,
	// and the rejection lands on the NEXT call — far from the cause.
	t.Run("scrubs to valid UTF-8", func(t *testing.T) {
		b := &limitBuf{max: 100}
		b.Write([]byte{'o', 'k', 0xff, 0xfe, 'z'})
		got := b.String()
		if !utf8.ValidString(got) {
			t.Errorf("String() = %q, which is not valid UTF-8", got)
		}
		if !strings.Contains(got, "�") {
			t.Errorf("String() = %q, want the replacement character", got)
		}
	})

	t.Run("a cut mid-rune is scrubbed too", func(t *testing.T) {
		b := &limitBuf{max: 2} // "中" is three bytes
		b.Write([]byte("中文"))
		if got := b.String(); !utf8.ValidString(got) {
			t.Errorf("String() = %q, which is not valid UTF-8", got)
		}
	})
}

func TestCommandError(t *testing.T) {
	inner := exitErr{128}

	t.Run("without stderr", func(t *testing.T) {
		e := &commandError{name: "git", err: inner}
		if want := "git: exit status 128"; e.Error() != want {
			t.Errorf("Error() = %q, want %q", e.Error(), want)
		}
	})

	// The model reading the is_error tool_result needs the diagnostic, not just
	// "exit status 128".
	t.Run("with stderr", func(t *testing.T) {
		e := &commandError{name: "git", err: inner, stderr: "fatal: not a git repository"}
		if want := "git: exit status 128: fatal: not a git repository"; e.Error() != want {
			t.Errorf("Error() = %q, want %q", e.Error(), want)
		}
	})

	t.Run("unwraps so exitCode and errors.Is keep working", func(t *testing.T) {
		e := &commandError{name: "git", err: syscall.EAGAIN}
		if !errors.Is(e, syscall.EAGAIN) {
			t.Error("errors.Is(commandError, EAGAIN) = false, want true")
		}
		if code, hasCode := exitCode(&commandError{name: "g", err: inner}); !hasCode || code != 128 {
			t.Errorf("exitCode = (%d, %v), want (128, true)", code, hasCode)
		}
	})
}

func TestOSRunner(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the runner tests spawn POSIX utilities")
	}

	r := OSRunner("")
	ctx := context.Background()

	t.Run("Run captures stdout", func(t *testing.T) {
		out, err := r.Run(ctx, "echo", "hello world")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if out != "hello world\n" {
			t.Errorf("out = %q, want %q", out, "hello world\n")
		}
	})

	// Nothing is interpreted by a shell, so a model-supplied value is safe to
	// pass as an argument with no quoting.
	t.Run("Run does not interpret its arguments", func(t *testing.T) {
		out, err := r.Run(ctx, "echo", "$HOME; rm -rf /")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if out != "$HOME; rm -rf /\n" {
			t.Errorf("out = %q, want the argument passed through verbatim", out)
		}
	})

	t.Run("a non-zero exit carries an ExitCode and the stderr", func(t *testing.T) {
		out, err := r.Run(ctx, "/bin/sh", "-c", "echo to-stdout; echo to-stderr >&2; exit 7")
		if err == nil {
			t.Fatal("err = nil, want a non-zero exit")
		}
		code, hasCode := exitCode(err)
		if !hasCode || code != 7 {
			t.Errorf("exitCode = (%d, %v), want (7, true)", code, hasCode)
		}
		if !strings.Contains(err.Error(), "to-stderr") {
			t.Errorf("err = %q, want the stderr carried into the message", err)
		}
		// Whatever was captured comes back even when err is non-nil.
		if out != "to-stdout\n" {
			t.Errorf("out = %q, want the captured stdout %q", out, "to-stdout\n")
		}
	})

	t.Run("a missing binary is an error", func(t *testing.T) {
		if _, err := r.Run(ctx, "definitely-not-a-real-binary-xyz"); err == nil {
			t.Fatal("err = nil, want a spawn failure")
		}
	})

	// Streams are captured separately so the [stderr] marker tells the model
	// which half is the diagnostic.
	t.Run("Shell marks the stderr half", func(t *testing.T) {
		out, err := r.Shell(ctx, "echo out; echo err >&2")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if want := "out\n\n[stderr]\nerr\n"; out != want {
			t.Errorf("out = %q, want %q", out, want)
		}
	})

	t.Run("Shell with only stdout has no marker", func(t *testing.T) {
		out, err := r.Shell(ctx, "echo only")
		if err != nil || out != "only\n" {
			t.Errorf("(%q, %v), want (only\\n, nil)", out, err)
		}
	})

	t.Run("Shell with only stderr returns it bare", func(t *testing.T) {
		out, err := r.Shell(ctx, "echo bad >&2")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if out != "bad\n" {
			t.Errorf("out = %q, want %q", out, "bad\n")
		}
	})

	t.Run("dir is the working directory", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "marker.txt"), nil, 0o644); err != nil {
			t.Fatal(err)
		}
		out, err := OSRunner(dir).Run(ctx, "ls")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if strings.TrimSpace(out) != "marker.txt" {
			t.Errorf("ls = %q, want %q", out, "marker.txt")
		}
	})

	// Cancellation is exec.CommandContext's: the child is SIGKILLed, and the
	// process group goes with it. The point is that the call RETURNS — an
	// abandoned goroutine parked on Wait is the failure mode this replaces.
	t.Run("cancellation stops the child promptly", func(t *testing.T) {
		cctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		start := time.Now()
		_, err := r.Shell(cctx, "sleep 30")
		elapsed := time.Since(start)

		if err == nil {
			t.Fatal("err = nil, want the kill reported")
		}
		if elapsed > 5*time.Second {
			t.Errorf("elapsed = %v, want the call to return promptly after cancellation", elapsed)
		}
		if cctx.Err() == nil {
			t.Error("the context is not done, want the deadline to have fired")
		}
	})

	// A descendant that outlives its parent shell would otherwise hold the
	// pipes open and park Wait forever; WaitDelay closes them.
	t.Run("an orphan holding the pipes cannot park the call", func(t *testing.T) {
		if testing.Short() {
			t.Skip("waits out the WaitDelay")
		}
		cctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		done := make(chan struct{})
		go func() {
			defer close(done)
			// The subshell escapes the process group and keeps stdout open.
			r.Shell(cctx, "(setsid sleep 30 || sleep 30) & sleep 30")
		}()

		select {
		case <-done:
		case <-time.After(waitDelay + 5*time.Second):
			t.Fatalf("Shell did not return within WaitDelay+5s, want it bounded by WaitDelay=%v", waitDelay)
		}
	})

	t.Run("no stdin: a reader gets EOF rather than the user's keystrokes", func(t *testing.T) {
		out, err := r.Shell(ctx, "cat")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if out != "" {
			t.Errorf("out = %q, want empty: the child must see an immediate EOF", out)
		}
	})

	t.Run("capture is bounded", func(t *testing.T) {
		out, err := r.Shell(ctx, fmt.Sprintf("head -c %d /dev/zero | tr '\\0' 'a'", maxCapture+5000))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if len(out) > maxCapture+200 {
			t.Errorf("len(out) = %d, want it bounded near %d", len(out), maxCapture)
		}
		if !strings.Contains(out, "capture limit") {
			t.Errorf("out does not report the capture limit, want the note")
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

	d := Bash(OSRunner(""))

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
