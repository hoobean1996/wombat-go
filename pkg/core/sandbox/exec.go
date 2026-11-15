package sandbox

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// Runner is how the exec-flavoured tools spawn processes.
//
// Two methods, because the two use cases are genuinely different and
// collapsing them is where command injection comes from:
//
//   - Run takes an argv. Nothing is interpreted by a shell, so a model-supplied
//     value can be passed as an argument with no quoting and no thought. Every
//     tool that takes structured input (grep_search, git_log, git_show) uses it.
//   - Shell takes a script. Exactly one tool needs it — bash, whose entire
//     purpose is to run what the model wrote.
//
// Implementations must be safe for concurrent use: one Runner is shared by
// every tool and every sub-agent goroutine.
//
// Contract for errors: a command that ran and exited non-zero should return an
// error whose chain contains a value with an `ExitCode() int` method (as
// *exec.ExitError does). Tools use it to tell "grep found nothing" (exit 1)
// apart from "grep failed" (exit 2). Both methods return whatever output was
// captured even when err is non-nil — a failing command's output is usually
// the most useful thing it produced.
type Runner interface {
	Run(ctx context.Context, cmd string, args ...string) (stdout string, err error)
	Shell(ctx context.Context, script string) (combined string, err error)
}

// maxCapture bounds what one command may buffer, per stream. Well above the
// 8000 chars a tool will actually return, so the truncation the model sees is
// decided by the tool and not by an accident of memory management — but low
// enough that `cat /dev/urandom` cannot exhaust the process.
const maxCapture = 1 << 20

// waitDelay bounds the wait after cancellation.
//
// Cancellation kills the child's whole process group (see
// isolateProcessGroup), but that is best effort: a descendant that changed
// its own group, or a platform without groups, can still hold the pipes open,
// and Wait blocks on the read side until they close. WaitDelay closes them
// for us, so cancelling a run cannot leave a goroutine parked forever on an
// orphan.
const waitDelay = 2 * time.Second

// Unconfined returns a Runner that spawns real processes with dir as their
// working directory. dir == "" inherits the harness process's directory.
//
// Cancellation is exec.CommandContext's: when ctx is done the child is
// SIGKILLed. That is the whole implementation of the timeout — no polling
// loop, no select over pipes, no manual kill — because the tool's Def.Timeout
// is enforced by cancelling the context and the standard library already
// propagates that all the way to the process.
//
// # The environment is filtered
//
// A child sees only an allowlisted part of the harness's environment — what a
// build, a test or git needs, and none of the harness's credentials. It used to
// inherit all of it, and every process the model can start runs code the model
// chose: one `go test` over a test file the model wrote was enough to print
// the provider key, with no permission prompt anywhere on the way. See
// [WithEnvAllow], [WithEnv] and [WithInheritedEnv] for the ways to change it.
//
// The filter is applied to the environment as it is when each command starts,
// not when the Runner is built, so a variable the process sets later — the way
// benchmarks.ApplyGoEnv pins the Go toolchain — still reaches the commands that
// follow.
//
// None of this confines what a command can touch. [Confined] does.
func Unconfined(dir string, opts ...Option) Runner {
	return osRunner{dir: dir, env: newEnvConfig(opts)}
}

type osRunner struct {
	dir string
	env envConfig
	// wrap rewrites a command into the one actually spawned — the sandbox
	// launcher with the original argv as its tail. nil spawns it as is.
	wrap func(name string, args []string) (string, []string)
}

func (r osRunner) Run(ctx context.Context, name string, args ...string) (string, error) {
	out, errOut, err := r.capture(ctx, name, args...)
	if err != nil {
		return out, &CommandError{Name: name, Err: err, Stderr: strings.TrimSpace(errOut)}
	}
	return out, nil
}

func (r osRunner) Shell(ctx context.Context, script string) (string, error) {
	out, errOut, err := r.capture(ctx, "/bin/sh", "-c", script)

	// Streams are captured separately and joined here rather than being
	// interleaved into one buffer: the [stderr] marker tells the model which
	// half is the diagnostic, which is the difference between "the command
	// printed a warning" and "the command printed a result".
	var combined string
	switch {
	case out != "" && errOut != "":
		combined = out + "\n[stderr]\n" + errOut
	case out != "":
		combined = out
	default:
		combined = errOut
	}

	if err != nil {
		return combined, &CommandError{Name: "sh", Err: err}
	}
	return combined, nil
}

func (r osRunner) capture(ctx context.Context, name string, args ...string) (string, string, error) {
	stdout := &limitBuf{max: maxCapture}
	stderr := &limitBuf{max: maxCapture}

	if r.wrap != nil {
		name, args = r.wrap(name, args)
	}
	c := exec.CommandContext(ctx, name, args...)
	c.Dir = r.dir
	c.Env = r.env.environ()
	c.Stdout = stdout
	c.Stderr = stderr
	c.WaitDelay = waitDelay
	isolateProcessGroup(c)

	// No stdin. A child that reads from the harness's stdin would either
	// steal the user's keystrokes or block forever; an immediate EOF is the
	// only sane answer for a non-interactive tool.
	c.Stdin = nil

	err := c.Run()
	return stdout.String(), stderr.String(), err
}

// CommandError carries the failing command's stderr into the error text, so
// the model reading the is_error tool_result sees the diagnostic and not just
// "exit status 128". It unwraps to the underlying *exec.ExitError, keeping
// exitCode and errors.Is working through it.
type CommandError struct {
	Name   string // the program, or "sh" for a script
	Err    error  // usually an *exec.ExitError
	Stderr string // trimmed; empty when the output was combined
}

func (e *CommandError) Error() string {
	if e.Stderr == "" {
		return fmt.Sprintf("%s: %v", e.Name, e.Err)
	}
	return fmt.Sprintf("%s: %v: %s", e.Name, e.Err, e.Stderr)
}

func (e *CommandError) Unwrap() error { return e.Err }

// limitBuf accumulates up to max bytes and counts the rest.
//
// Write never reports an error: os/exec turns a writer error into the result
// of Wait, so refusing bytes here would masquerade as a command failure. We
// take everything and drop what does not fit.
type limitBuf struct {
	b       bytes.Buffer
	max     int
	dropped int
}

func (l *limitBuf) Write(p []byte) (int, error) {
	if room := l.max - l.b.Len(); room > 0 {
		if len(p) <= room {
			l.b.Write(p)
		} else {
			l.b.Write(p[:room])
			l.dropped += len(p) - room
		}
	} else {
		l.dropped += len(p)
	}
	return len(p), nil
}

// String returns the captured output, scrubbed to valid UTF-8.
//
// Scrubbing is not cosmetic: a command that emits binary (or a capture cut
// mid-rune at the limit) produces a tool_result the provider rejects, and the
// rejection lands on the next call — far from the command that caused it.
func (l *limitBuf) String() string {
	s := strings.ToValidUTF8(l.b.String(), "�")
	if l.dropped == 0 {
		return s
	}
	return s + fmt.Sprintf("\n[... %d more bytes dropped at the %d-byte capture limit ...]", l.dropped, l.max)
}
