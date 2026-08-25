package accept

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// A regression ledger is a directory of test files, one per defect ever found
// on this task, that every later run must keep passing.
//
// Without it each run starts from nothing and the same defect comes back. It
// did: a duplicate-parameter-name defect was found and repaired on one run of a
// task, and reappeared in the next run of the same task, where six review
// sweeps missed it and the run reported that it had converged. Nothing carried
// the knowledge between them except a person remembering.
//
// The ledger is that carrier. Its tests are copied into the workspace before
// the build starts, so the defect is a failing test from the first moment
// rather than something a review has to rediscover — and rediscovery is exactly
// what proved unreliable.
var reTestFunc = regexp.MustCompile(`(?m)^func (Test\w+)\(`)

type ledger struct {
	Dir   string
	Files []string
}

// LoadLedger lists the test files in dir. A missing directory is empty, not an
// error: the first run of a task has no history.
func LoadLedger(dir string) (*ledger, error) {
	l := &ledger{Dir: dir}
	if dir == "" {
		return l, nil
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return l, nil
		}
		return nil, err
	}
	for _, e := range ents {
		if !e.IsDir() && strings.HasSuffix(e.Name(), "_test.go") {
			l.Files = append(l.Files, e.Name())
		}
	}
	return l, nil
}

// Install copies the ledger's tests into the workspace so the build sees them
// from the start.
func (l *ledger) Install(workspace string) error {
	for _, name := range l.Files {
		b, err := os.ReadFile(filepath.Join(l.Dir, name))
		if err != nil {
			return fmt.Errorf("regression %s: %w", name, err)
		}
		if err := os.WriteFile(filepath.Join(workspace, name), b, 0o644); err != nil {
			return fmt.Errorf("installing regression %s: %w", name, err)
		}
	}
	return nil
}

// Promote records a test file as a permanent regression. Called after a review
// finding has been repaired: the test that demonstrated it becomes the thing
// that stops it coming back.
func Promote(ledgerDir, workspace, testFile string) error {
	if ledgerDir == "" {
		return nil
	}
	if err := os.MkdirAll(ledgerDir, 0o755); err != nil {
		return err
	}
	b, err := os.ReadFile(filepath.Join(workspace, testFile))
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(ledgerDir, testFile), b, 0o644)
}

// checkRegressions fails the gate when a previously-repaired defect is back, or
// when its test has gone missing from the workspace.
//
// A missing file counts as a failure on purpose. Deleting the test is the
// cheapest way to make it stop failing, and it is the one repair that leaves
// the defect in place — the same reason a review credits a fix only when its
// test survives.
func (r *Report) checkRegressions(ctx context.Context, cfg Config) {
	l, err := LoadLedger(cfg.RegressionDir)
	if err != nil || len(l.Files) == 0 {
		return
	}

	var missing []string
	var names []string
	for _, f := range l.Files {
		if _, err := os.Stat(filepath.Join(cfg.Workspace, f)); err != nil {
			missing = append(missing, f)
			continue
		}
		names = append(names, testNamesIn(filepath.Join(cfg.Workspace, f))...)
	}

	results := runNamedTests(ctx, cfg, names)
	var broken []string
	for _, n := range names {
		switch results[n] {
		case "PASS":
		case "":
			broken = append(broken, n+" (did not run)")
		default:
			broken = append(broken, n+" ("+strings.ToLower(results[n])+"ed)")
		}
	}

	if len(missing) == 0 && len(broken) == 0 {
		cfg.log("acceptance: %d regression test(s) still pass", len(names))
		r.add(Check{Name: "no regressions", OK: true})
		return
	}
	var b strings.Builder
	b.WriteString("These cover defects already found and repaired on this task. They must keep passing.\n")
	if len(missing) > 0 {
		fmt.Fprintf(&b, "\nRemoved from the tree: %s\n", strings.Join(missing, ", "))
		b.WriteString("Restore them. Deleting a regression test does not repair anything.\n")
	}
	if len(broken) > 0 {
		fmt.Fprintf(&b, "\nFailing: %s\n", strings.Join(broken, ", "))
		b.WriteString("A defect that was fixed before is back. Fix the cause, not the test.\n")
	}
	r.add(Check{Name: "no regressions", OK: false, Detail: b.String()})
}

// testNamesIn lists the Test functions declared in a file. Kept here rather
// than shared with the review package: the two read the same syntax for
// different reasons, and a shared helper would couple a gate to a reviewer.
func testNamesIn(path string) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []string
	for _, m := range reTestFunc.FindAllStringSubmatch(string(b), -1) {
		out = append(out, m[1])
	}
	return out
}
