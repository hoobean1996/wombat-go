// Package builtin is the built-in tool surface: the dozen tools an agent
// starts with.
//
// Every tool is a constructor, and every constructor takes its dependencies:
//
//	builtin.ViewFile(fsys)        // reads through an FS
//	builtin.Bash(runner)          // spawns through a Runner
//	builtin.HTTPGet(httpClient)   // fetches through an *http.Client
//	builtin.CurrentTime(time.Now) // reads a clock
//
// Nothing is ambient and nothing is global. A tool.Fn only ever receives a
// context and its JSON arguments, so whatever a tool needs to touch the
// outside world has to be captured when it is built. That is what makes a
// tool testable without a harness — builtin.ViewFile(fakeFS) is the whole
// setup — and what makes a restricted agent honest: hand it a sandboxed FS
// and there is no second path to the disk.
//
// The usual entry point is [Default], which builds all of them:
//
//	defs := builtin.Default(builtin.Deps{
//	    FS:   builtin.OSFS("/work/project"),
//	    Exec: sandbox.Unconfined("/work/project"),
//	})
//	a, err := core.New(core.WithClient(c), core.WithTools(defs...))
//
// For a read-only sub-agent (verifier, planner), filter the result rather
// than hand-picking constructors — the tool list stays the single authority:
//
//	core.WithTools(tool.Filter(defs, tool.OnlyCaps(tool.CapReadOnly))...)
//
// which leaves view_file, grep_search, git_log, git_show, calculator and
// current_time.
//
// # Caps versus Needs
//
// grep_search, git_log and git_show all spawn a subprocess, yet none of them
// declares [tool.CapExec]. The distinction is deliberate and it is the one in
// tool.go: Cap is what a tool DOES, Need is what it requires from the host.
// These three run a fixed argv with a read-only effect, so they are
// CapReadOnly (and therefore survive the verifier filter above) while
// declaring NeedExec so a host without a subprocess facility hides them.
// bash, which runs whatever the model wrote, is CapExec|CapMutating|CapNetwork
// — it can curl — but declares only NeedExec|NeedFSRead|NeedFSWrite, because a
// host with no network can still offer a perfectly useful shell.
//
// # Which tools are retried
//
// [tool.WithRetry] fires only when a Def is BOTH Idempotent and carries a
// Retryable classifier, so the two fields together are the whole policy:
//
//   - The idempotent tools that touch the OS classify. view_file uses retryFS;
//     grep_search, git_log and git_show use retryExec. All four are read-only
//     with an argv or a path this package builds, so replaying one is free.
//   - The pure tools — calculator, current_time — leave Retryable nil. There is
//     no transient failure mode to catch; a retry would buy a second identical
//     answer at the price of wall clock.
//   - The mutating tools — write_file, str_replace, save_tool_result — and bash
//     are not idempotent, which is what actually forbids the replay. A write
//     that timed out may well have landed.
//
// http_get is the one classifier that is not about errnos: see retryHTTP.
package builtin

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/tool"
)

// maxToolOutput caps what any single tool returns to the model, in bytes.
//
// The value is load-bearing and is carried over from the OCaml: 8000 chars is
// about 2k tokens, roughly the most a single observation can cost before it
// crowds out the rest of the transcript.
//
// The harness has its own, much larger cap in tool.WithTruncation (64 KiB), and
// the two are not redundant: this one bounds what the TOOL is willing to
// produce whatever the harness is configured to accept, so a tool used outside
// an agent — in a test, a script, a REPL — is still bounded.
//
// Which end survives is per tool, not global: [truncate] keeps the head for
// everything that returns a region or a list, and bash reaches for [tool.Clip]
// to keep both ends because its answer is at the bottom.
const maxToolOutput = 8000

// defaultHTTPTimeout is the client timeout [Default] installs when Deps.HTTP
// is nil. It matches http_get's own Def.Timeout, so a caller who forgets the
// per-call middleware still cannot hang a run on a stalled socket.
const defaultHTTPTimeout = 15 * time.Second

// Deps are what the built-in tools reach the world through.
// A nil field is filled in by [Default] with an OS-backed implementation.
type Deps struct {
	// FS backs view_file, write_file, str_replace and save_tool_result.
	// Default: OSFS("") — the whole filesystem.
	FS FS

	// Exec backs bash, grep_search, git_log and git_show.
	// Default: sandbox.Unconfined("") — the process working directory, unconfined, with
	// the environment filtered. SandboxRunner confines it.
	Exec sandbox.Runner

	// HTTP backs http_get. Default: a client with a 15s timeout. Note that
	// http.DefaultClient is deliberately NOT the default: it has no timeout,
	// and a tool that can block forever defeats the run budget.
	HTTP *http.Client

	// Now backs current_time. Default: time.Now. Injected rather than called
	// directly so a replayed or recorded run can pin the clock.
	Now func() time.Time

	// Spill keeps the complete output of a bash command too long to return.
	// Default: NewSpill("", 0), under the system temp directory. A deployment
	// whose agent cannot read the temp directory points it somewhere it can.
	Spill *Spill
}

// Default builds every built-in tool, in the order the OCaml registry used.
//
// Order matters only in that it is the order the model sees the tools in, and
// therefore part of the prompt-cache prefix — keep it stable.
func Default(d Deps) []tool.Def {
	if d.FS == nil {
		d.FS = OSFS("")
	}
	if d.Exec == nil {
		d.Exec = sandbox.Unconfined("")
	}
	if d.HTTP == nil {
		d.HTTP = &http.Client{Timeout: defaultHTTPTimeout}
	}
	if d.Now == nil {
		d.Now = time.Now
	}
	if d.Spill == nil {
		d.Spill = NewSpill("", 0)
	}
	return []tool.Def{
		Calculator(),
		HTTPGet(d.HTTP),
		CurrentTime(d.Now),
		Bash(d.Exec, WithSpill(d.Spill)),
		ViewFile(d.FS),
		GrepSearch(d.Exec),
		GitLog(d.Exec),
		GitShow(d.Exec),
		WriteFile(d.FS),
		SaveToolResult(d.FS),
		StrReplace(d.FS),
		AskUser(),
	}
}

// ===== shared helpers =====

// truncate clips s to max bytes, keeping the head.
//
// The default for a tool's output, and the exception is bash — see the call
// site there. The head-and-tail split in [tool.Clip] is justified by COMMAND
// output, where the tail carries the failing assertion and the head only the
// echo. It is the wrong shape for everything else here:
//
//   - view_file returns a region the model is about to edit. Two fragments that
//     look adjacent get a str_replace composed across the invisible gap, which
//     does not match; a contiguous prefix does not. The tool has offset and
//     limit for reaching the rest.
//   - a directory listing, git_log and grep_search are ordered lists whose
//     front is the part that was asked for.
//   - a fetched document is front-loaded; the tail of an HTML page is a footer.
func truncate(s string, max int) string { return tool.ClipHead(s, max) }

// requireAbs rejects a relative or empty path.
//
// Every file-touching tool calls this. A relative path resolves against the
// harness process's working directory, which is almost never what the model
// intended; silently writing to the wrong place is the worst failure mode
// available to this package. The wording is verbatim from the OCaml because
// it is what teaches the model to retry with an absolute path.
func requireAbs(field, path string) error {
	if path == "" {
		return fmt.Errorf("field '%s' must not be empty", field)
	}
	if !strings.HasPrefix(path, "/") {
		return fmt.Errorf("field '%s' must be an absolute path (starts with /), got: %q", field, path)
	}
	return nil
}

// exitCode reports a command's exit status, if the error carries one.
//
// It matches on the method rather than on *exec.ExitError so that any Runner
// implementation participates — a remote or containerised runner only has to
// return an error with an ExitCode method. See [sandbox.Runner].
func exitCode(err error) (int, bool) {
	var ec interface{ ExitCode() int }
	if errors.As(err, &ec) {
		return ec.ExitCode(), true
	}
	return 0, false
}

// mustNotBeNil guards a constructor argument.
//
// Panicking is right here and only here: a nil dependency is a programmer
// error at construction time, discovered on the first line of main rather
// than three hours into a run. Silently substituting a default would be worse
// than a panic — a caller who passes a sandboxed FS and gets an unrestricted
// one because of a typo has lost the guardrail without being told.
func mustNotBeNil(cond bool, what string) {
	if !cond {
		panic("builtin: " + what)
	}
}
