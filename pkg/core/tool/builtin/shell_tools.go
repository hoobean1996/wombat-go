package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"syscall"
	"time"

	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/tool"
)

// bashTimeout is both the Def.Timeout the middleware enforces and the number
// quoted in the description, so the model's expectation matches reality.
const bashTimeout = 60 * time.Second

type bashIn struct {
	Command string `json:"command"`
	ExecDir string `json:"exec_dir"`
}

// Bash runs a shell command and returns its combined output.
//
// exec_dir is required, not optional. Without it the command inherits the
// harness process's working directory, which is wherever the user happened to
// launch from — so `find . -name '*.go'` would silently search the wombat
// source tree instead of the project. Making the model state the directory
// turns an invisible wrong answer into an explicit choice.
//
// # Bash is NOT confined by the filesystem root
//
// [OSFS] confines the file tools to a subtree. Nothing here does, and nothing
// here can: exec_dir says where the command STARTS, and `cat ../../secrets`
// leaves from there in one hop. A deployment that gives an agent a rooted FS
// and a bash tool in the same set has confined its file reader and left its
// shell wide open, which is worse than either honest position because the
// rooted FS makes it look contained.
//
// This is not theoretical. In a benchmark run, two agents were correctly
// refused by the rooted FS when they reached for a sibling workspace, and both
// then read the same files through bash on the next call.
//
// The seam for fixing it is [sandbox.Runner]. Everything about how a command reaches
// the world goes through that interface, so a Runner that wraps the argv in
// sandbox-exec, bwrap, a container or an ssh hop confines the shell for real —
// without this file knowing. [sandbox.Confined] is that Runner for macOS and
// Linux; a container or a remote host is the same seam with a different
// Runner behind it. What must NOT be attempted is confinement by
// inspecting the command string: a shell has $(), backticks, eval, aliases and
// a hundred ways to spell a path, and a filter over it is a filter an
// adversary — or a prompt-injected model behaving like one — walks straight
// through while everyone downstream believes the boundary holds.
//
// Short of that, [permission.Policy] is the available control, and it is a
// real one: it decides per call, with the command in hand, and can hold the
// call until a human answers.
//
// # Long output
//
// A result is clipped to its first and last few kilobytes. Give Bash a [Spill]
// and the complete output of a clipped command is saved to a file the result
// names, so the model can grep the middle instead of re-running the command.
// [Default] gives it one.
func Bash(r sandbox.Runner, opts ...BashOption) tool.Def {
	mustNotBeNil(r != nil, "Bash requires a non-nil Runner")
	var cfg bashConfig
	for _, o := range opts {
		o(&cfg)
	}

	return tool.Typed(tool.Def{
		Name: "bash",
		// The description does NOT name a fallback directory. An earlier
		// version suggested /tmp for one-off commands, and a model given a
		// working directory in its system prompt took the suggestion, ran pwd
		// in /tmp, found nothing, and spent its whole iteration budget hunting
		// for the project. Telling the model where to look is the caller's
		// job — see the working_directory block the CLI sets.
		Description: "Execute a shell command and return its combined stdout+stderr. " +
			"60s timeout. Output truncated to ~8000 chars; when it is, the result names a file " +
			"holding the complete output. " +
			"REQUIRES exec_dir, an ABSOLUTE path — the command runs there, so " +
			"relative paths in your command resolve against it. If the system " +
			"prompt names a working directory, use that.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "command": {
      "type": "string",
      "description": "Shell command to execute (passed to sh -c)"
    },
    "exec_dir": {
      "type": "string",
      "description": "Working directory (absolute path). The command runs in this directory."
    }
  },
  "required": ["command", "exec_dir"]
}`),
		// CapNetwork alongside the obvious two: the script is the model's, and
		// `curl`, `npm install` and `git push` are network I/O however the
		// harness feels about it. Declaring it means an offline agent built
		// with tool.OnlyCaps actually excludes bash instead of quietly keeping
		// its one uncontrolled route out. Needs is deliberately NOT given
		// NeedNetwork: Need is what the tool requires FROM THE HOST, and bash
		// runs perfectly well on a machine with no network — a sandbox without
		// one should still get a shell.
		Caps:  tool.CapExec | tool.CapMutating | tool.CapNetwork,
		Needs: tool.NeedExec | tool.NeedFSRead | tool.NeedFSWrite,
		// Not idempotent: shell commands have side effects (mkdir, npm
		// install, file writes). Conservative default — a replayed
		// `git push` is not a free retry.
		Idempotent: false,
		Timeout:    bashTimeout,
		Category:   "exec",
		// Deliberately inert as a retry policy, and kept anyway. Idempotent=false
		// is what forbids replaying a script the model wrote, and it wins:
		// WithRetry requires both flags, so this classifier never fires here.
		// It stays as declarative metadata — it is the truthful answer to "which
		// of this tool's failures are transient?", and other middleware (a
		// breaker that should not count EAGAIN against a tool, a report that
		// separates flaky spawns from real exits) can read it without having to
		// re-derive it. Removing it would not change dispatch; it would only
		// delete the fact.
		Retryable: retryExec,
	}, func(ctx context.Context, in bashIn) (string, error) {
		if in.Command == "" {
			return "", tool.CallerError(errors.New("field 'command' must not be empty"))
		}
		if err := requireAbs("exec_dir", in.ExecDir); err != nil {
			return "", err
		}

		// `cd DIR && (cmd)` rather than setting the Runner's directory: the
		// Runner's directory is fixed at construction (that is the point),
		// and the subshell form makes every nested shell the command spawns
		// inherit the directory too. The cd also gives a clear error if
		// exec_dir does not exist, with no TOCTOU pre-check.
		//
		// This is the ONE place in the package where model input reaches a
		// shell string, and it is unavoidable: running what the model wrote
		// is what bash is for. exec_dir is single-quoted; `command` is
		// deliberately not, because escaping it would break the tool.
		script := "cd " + shQuote(in.ExecDir) + " && (" + in.Command + ")"

		out, err := r.Shell(ctx, script)
		// tool.Clip, not truncate: this is the one tool whose answer is at the
		// END. `go build` prints its errors after everything else, a test suite
		// prints the failing assertion after every passing test name, a stack
		// trace ends at the throw. Keeping only the head of a 200 KB failing
		// test run hands the model eight kilobytes of test names that PASSED.
		out = clipAndSpill(cfg.spill, "bash", out, maxToolOutput, tool.Clip)
		if out == "" {
			out = "(no output)"
		}

		if err != nil {
			// A dead context outranks the exit status: the process was
			// SIGKILLed by the cancellation, so "exit -1" would describe the
			// symptom and hide the cause. The deadline is not quoted here
			// because it belongs to whoever set it — the tool's own 60s under
			// the harness, something shorter under a run budget.
			switch {
			case errors.Is(ctx.Err(), context.DeadlineExceeded):
				return "", fmt.Errorf("command timed out, partial output:\n%s", out)
			case ctx.Err() != nil:
				return "", fmt.Errorf("command cancelled: %w", context.Cause(ctx))
			}
			if code, ok := exitCode(err); ok {
				// A non-zero exit is reported as an error so the model cannot
				// mistake a failed build for a successful one — but it is a
				// [tool.CallerFault], and the distinction is what keeps a
				// coding agent alive. The shell did its whole job here: it
				// spawned the process, waited, and reported the truth. `grep`
				// finding nothing exits 1. `go build` on broken code exits 2.
				// An agent in the edit-build-edit loop that every coding agent
				// runs will collect a long string of these BY DOING THE TASK
				// CORRECTLY, and if they counted as tool failures the circuit
				// breaker would take the shell away at exactly the moment the
				// agent needed it to prove the fix worked.
				return "", tool.CallerError(fmt.Errorf("exit %d:\n%s", code, out))
			}
			// No exit code means the process never ran — a missing binary, a
			// fork failure, a broken pipe. That IS the tool failing.
			return "", fmt.Errorf("%w:\n%s", err, out)
		}
		return out, nil
	})
}

// BashOption configures [Bash].
type BashOption func(*bashConfig)

type bashConfig struct {
	spill *Spill
}

// WithSpill saves the complete output of a clipped command; see [Spill].
func WithSpill(s *Spill) BashOption { return func(c *bashConfig) { c.spill = s } }

// retryExec classifies a subprocess failure.
//
// Subprocess errors are almost all deterministic — a bad exit code, a missing
// binary, a syntax error — and retrying them just burns budget to reach the
// same conclusion. Only the three genuinely flaky spawn failures are worth a
// second attempt. Matching on syscall errnos rather than on substrings of the
// message is the Go version of the OCaml's exec classifier, and it does not
// break when the locale changes.
func retryExec(err error) bool {
	return errors.Is(err, syscall.EINTR) || // interrupted system call
		errors.Is(err, syscall.EAGAIN) || // resource temporarily unavailable
		errors.Is(err, syscall.ETXTBSY) // text file busy: binary still being written
}

// ===== grep_search =====

type grepIn struct {
	Pattern         string `json:"pattern"`
	Path            string `json:"path"`
	CaseInsensitive bool   `json:"case_insensitive"`
	MaxResults      int    `json:"max_results"`
}

const (
	grepDefaultResults = 100
	grepMaxResults     = 500
)

// grepExcludes are the directories never worth searching. Machine-generated
// or vendored trees dominate the match count and drown the real hits.
var grepExcludes = []string{
	"--exclude-dir=node_modules",
	"--exclude-dir=.git",
	"--exclude-dir=.venv",
	"--exclude-dir=__pycache__",
	"--exclude-dir=dist",
	"--exclude-dir=build",
	"--exclude-dir=.next",
	"--exclude-dir=target",
}

// GrepSearch searches a file or tree for an extended regex.
//
// It shells out to grep through [Runner.Run] as an argv, so the model's
// pattern is a single argument and never a shell word. `--` terminates the
// options, so a pattern starting with `-` is a pattern and not a flag.
func GrepSearch(r sandbox.Runner) tool.Def {
	mustNotBeNil(r != nil, "GrepSearch requires a non-nil Runner")

	return tool.Typed(tool.Def{
		Name: "grep_search",
		Description: "Search a file or directory for an extended-regex pattern (POSIX " +
			"ERE, like `grep -E`). Returns matching lines with their file " +
			"paths and line numbers. PATH MUST BE ABSOLUTE. Default scope is " +
			"recursive. Output capped at [max_results] lines (default 100, " +
			"hard cap 500) and ~8000 chars total. Excludes node_modules / " +
			".git / .venv / dist / build by default. For literal-string " +
			"searches, escape regex metacharacters or use simple alphanumeric " +
			"patterns.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Extended regex pattern (grep -E syntax)"
    },
    "path": {
      "type": "string",
      "description": "Absolute path to a file or directory"
    },
    "case_insensitive": {
      "type": "boolean",
      "description": "Match case-insensitively (default false)"
    },
    "max_results": {
      "type": "integer",
      "description": "Max matching lines to return (1-500)"
    }
  },
  "required": ["pattern", "path"]
}`),
		// Read-only despite spawning a process: the argv is fixed and grep
		// cannot mutate anything. NeedExec is what tells a host without a
		// subprocess facility to hide it. See the package doc.
		Caps:       tool.CapReadOnly,
		Needs:      tool.NeedFSRead | tool.NeedExec,
		Idempotent: true,
		Timeout:    10 * time.Second,
		Category:   "file_io",
		// Unlike bash, this argv is built here and grep cannot mutate
		// anything, so Idempotent holds and the retry actually engages. Only
		// the spawn can fail transiently: "no matches" is exit 1 and handled
		// below, and every other exit code (2 = bad regex, unreadable path) is
		// deterministic and excluded by retryExec.
		Retryable: retryExec,
	}, func(ctx context.Context, in grepIn) (string, error) {
		if in.Pattern == "" {
			return "", tool.CallerError(errors.New("field 'pattern' must not be empty"))
		}
		if err := requireAbs("path", in.Path); err != nil {
			return "", err
		}

		limit := in.MaxResults
		if limit <= 0 {
			limit = grepDefaultResults
		}
		limit = min(limit, grepMaxResults)

		args := []string{"-rEnH", "--binary-files=without-match"}
		if in.CaseInsensitive {
			args = append(args, "-i")
		}
		// grep's -m is per FILE, so a recursive search can return far more
		// than max_results lines. Keep it (it stops grep early on a huge
		// single file) and clamp the total below.
		args = append(args, "-m", fmt.Sprint(limit))
		args = append(args, grepExcludes...)
		args = append(args, "--", in.Pattern, in.Path)

		out, err := r.Run(ctx, "grep", args...)
		if err != nil {
			// grep exits 1 for "no matches", which is an answer, not a
			// failure. Anything else (2 = unreadable path, bad regex) is real.
			if code, ok := exitCode(err); ok && code == 1 {
				return "(no matches)", nil
			}
			return "", err
		}

		lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
		if len(lines) == 1 && lines[0] == "" {
			return "(no matches)", nil
		}
		clamped := ""
		if len(lines) > limit {
			clamped = fmt.Sprintf("\n[... %d more matching lines; raise max_results or narrow the pattern ...]",
				len(lines)-limit)
			lines = lines[:limit]
		}
		return truncate(strings.Join(lines, "\n")+clamped, maxToolOutput), nil
	})
}

// shQuote wraps s in single quotes for /bin/sh.
//
// Only bash uses this, and only for its exec_dir. Everything else in this
// package passes argv through [sandbox.Runner.Run], where quoting is not a
// concept and therefore not a bug.
func shQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}
