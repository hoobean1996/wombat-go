// Package accept is the deterministic gate between an agent saying it is done
// and the run treating it as done.
//
// It exists because of one observation. Sorting every failure of a measured
// build task into "the agent got feedback about this" and "it did not" splits
// them almost perfectly:
//
//	fed back      compile errors, failing tests, vet     — all clean at the end
//	not fed back  the tests the task required            — none written
//	              a 17-item TODO list                    — none ticked in 80 turns
//	              a rule not to rewrite the plan         — broken twice
//
// Nothing in the second column is a reasoning mistake. Each is something no
// signal ever pointed at. The same run produced a correct deterministic
// scheduler and a cross-process atomic claim, so the gap is not capability.
//
// So this package does not advise. It runs cheap, deterministic checks and
// hands the failures back as work, which is the form the agent already
// responds to.
package accept

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"wombat/internal/safeenv"
	"wombat/internal/shellquote"
)

// Check is one deterministic verdict.
type Check struct {
	Name   string
	OK     bool
	Detail string // what is wrong, phrased so the agent can act on it
}

// Report is the outcome of one gate pass.
type Report struct {
	Checks  []Check
	Skipped []string // checks that could not run, and why
}

// Failed returns the checks that did not pass.
func (r *Report) Failed() []Check {
	var out []Check
	for _, c := range r.Checks {
		if !c.OK {
			out = append(out, c)
		}
	}
	return out
}

// OK reports whether every check that ran passed.
func (r *Report) OK() bool { return len(r.Failed()) == 0 }

// Summary is one line for a log or an event stream.
func (r *Report) Summary() string {
	if len(r.Checks) == 0 {
		return "acceptance: nothing to check"
	}
	if r.OK() {
		return fmt.Sprintf("acceptance: %d/%d checks pass", len(r.Checks), len(r.Checks))
	}
	var names []string
	for _, c := range r.Failed() {
		names = append(names, c.Name)
	}
	return fmt.Sprintf("acceptance: %d/%d checks pass, failed: %s",
		len(r.Checks)-len(names), len(r.Checks), strings.Join(names, ", "))
}

// Config selects what the gate enforces.
type Config struct {
	// Workspace is the directory to check.
	Workspace string

	// RequireTests fails the gate when the tree has no test files at all.
	//
	// This is the check the measured run needed and did not have: `go test`
	// printed "no test files", exited zero, and the run was reported as an
	// answer. An absent test suite is the one defect that cannot show up as a
	// failing test.
	RequireTests bool

	// VerifierFile, when it exists in Workspace, must have every item settled:
	// each one either proven by a test that exists and passes, or withdrawn
	// with an argument from the task text. Empty disables the check.
	VerifierFile string

	// RegressionDir holds one test file per defect ever found on this task.
	// They are installed into the workspace before the build and must pass at
	// the gate. Empty disables the check.
	RegressionDir string

	// TodoFile, when it exists in Workspace, must have no unticked items.
	// Empty disables the check.
	TodoFile string

	// Timeout bounds each go invocation.
	Timeout time.Duration

	// Shell, when set, runs every command the gate needs — go build, go vet,
	// go test, a verifier item's own command — in place of a plain
	// subprocess, and returns what it printed, stdout and stderr together.
	// It must run its script in Workspace.
	//
	// Pass the runner the agent's own commands go through. Every one of these
	// runs code the agent wrote, and a gate that runs it outside the sandbox
	// the agent is held in is a way out of that sandbox. Nil runs them
	// directly, with the filtered environment and nothing else.
	Shell func(ctx context.Context, script string) (string, error)

	// Log receives progress lines. Nil discards them.
	Log func(string)
}

func (c Config) log(format string, args ...any) {
	if c.Log != nil {
		c.Log(fmt.Sprintf(format, args...))
	}
}

func (c Config) timeout() time.Duration {
	if c.Timeout <= 0 {
		return 5 * time.Minute
	}
	return c.Timeout
}

// Run executes the gate. It never returns an error: a check that cannot run is
// recorded as skipped rather than failing the agent for something outside its
// control.
func Run(ctx context.Context, cfg Config) *Report {
	rep := &Report{}
	if cfg.Workspace == "" {
		rep.Skipped = append(rep.Skipped, "no workspace")
		return rep
	}
	if !isGoModule(cfg.Workspace) {
		rep.Skipped = append(rep.Skipped, "not a Go module: the go checks do not apply")
		rep.checkTodo(cfg)
		return rep
	}

	build := run(ctx, cfg, "build", "./...")
	rep.add(Check{
		Name:   "go build",
		OK:     build.ok,
		Detail: build.detail("the code does not compile"),
	})

	// vet and the tests are only meaningful once it compiles; running them on
	// a broken tree produces the same compile errors a second time and buries
	// the one that matters.
	if !build.ok {
		rep.Skipped = append(rep.Skipped, "go vet and go test: skipped while the build is broken")
		rep.checkTodo(cfg)
		return rep
	}

	vet := run(ctx, cfg, "vet", "./...")
	rep.add(Check{Name: "go vet", OK: vet.ok, Detail: vet.detail("go vet reports problems")})

	test := run(ctx, cfg, "test", "./...")
	rep.add(Check{Name: "go test", OK: test.ok, Detail: test.detail("tests fail")})

	if cfg.RequireTests {
		n := countTestFiles(cfg.Workspace)
		rep.add(Check{
			Name: "tests exist",
			OK:   n > 0,
			Detail: "the tree contains no _test.go files at all. `go test` printing " +
				"\"no test files\" is not a passing test suite: it means nothing was checked.",
		})
	}

	rep.checkTodo(cfg)
	rep.checkVerifier(ctx, cfg)
	rep.checkRegressions(ctx, cfg)
	rep.checkCensus(cfg)
	cfg.log("%s", rep.Summary())
	return rep
}

func (r *Report) add(c Check) { r.Checks = append(r.Checks, c) }

// checkTodo enforces that the plan's own checklist is finished. The list is
// the agent's own statement of what it set out to build, so an unticked box is
// its own admission, and needs no judgement to read.
func (r *Report) checkTodo(cfg Config) {
	if cfg.TodoFile == "" {
		return
	}
	path := filepath.Join(cfg.Workspace, cfg.TodoFile)
	b, err := os.ReadFile(path)
	if err != nil {
		return // no plan file is not a failure; the planner may not have run
	}
	open := untickedItems(string(b))
	if len(open) == 0 {
		r.add(Check{Name: "todo complete", OK: true})
		return
	}
	shown := open
	if len(shown) > 12 {
		shown = shown[:12]
	}
	detail := fmt.Sprintf("%s still has %d unticked item(s):\n  - %s",
		cfg.TodoFile, len(open), strings.Join(shown, "\n  - "))
	if len(open) > len(shown) {
		detail += fmt.Sprintf("\n  ... and %d more", len(open)-len(shown))
	}
	detail += "\n\nTick an item only when the work for it exists. Do not delete " +
		"items to shorten the list: the list is what this is measured against."
	r.add(Check{Name: "todo complete", OK: false, Detail: detail})
}

// untickedItems parses markdown checkboxes in the forms a model writes.
func untickedItems(content string) []string {
	var out []string
	for _, line := range strings.Split(content, "\n") {
		t := strings.TrimSpace(line)
		if len(t) < 6 || (t[0] != '-' && t[0] != '*') {
			continue
		}
		rest := strings.TrimSpace(t[1:])
		if len(rest) < 4 || rest[0] != '[' || rest[2] != ']' || rest[1] != ' ' {
			continue
		}
		if text := strings.TrimSpace(rest[3:]); text != "" {
			out = append(out, text)
		}
	}
	return out
}

func isGoModule(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "go.mod"))
	return err == nil
}

func countTestFiles(dir string) int {
	n := 0
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if name := d.Name(); name != "." && (strings.HasPrefix(name, ".") || name == "vendor") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(d.Name(), "_test.go") {
			n++
		}
		return nil
	})
	return n
}

type result struct {
	ok     bool
	output string
}

// detail keeps the tail of the output: compilers and test runners put the
// summary last, and a head-truncated log usually holds only the first error.
func (r result) detail(headline string) string {
	if r.ok {
		return ""
	}
	out := strings.TrimSpace(r.output)
	const cap = 3000
	if len(out) > cap {
		out = "..." + out[len(out)-cap:]
	}
	return headline + ":\n\n" + out
}

func run(ctx context.Context, cfg Config, args ...string) result {
	tctx, cancel := context.WithTimeout(ctx, cfg.timeout())
	defer cancel()

	out, err := cfg.exec(tctx, append([]string{"go"}, args...)...)
	cfg.log("acceptance: go %s -> %v", strings.Join(args, " "), err == nil)
	return result{ok: err == nil, output: out}
}

// exec runs argv in the workspace — through Shell when there is one — and
// returns its combined output.
func (c Config) exec(ctx context.Context, argv ...string) (string, error) {
	if c.Shell != nil {
		return c.Shell(ctx, shellquote.Join(argv...))
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = c.Workspace
	cmd.Env = safeenv.Environ() // runs the agent's code: no harness credentials
	out, err := cmd.CombinedOutput()
	return string(out), err
}
