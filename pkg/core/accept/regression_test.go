package accept

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const passing = `package x

import "testing"

func TestNoDuplicateParams(t *testing.T) {}
`

const failing = `package x

import "testing"

func TestNoDuplicateParams(t *testing.T) { t.Fatal("the defect is back") }
`

// The case this exists for: a defect found and repaired on one run came back on
// the next, six review sweeps missed it, and the run reported convergence.
func TestARepairedDefectComingBackFailsTheGate(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	led := t.TempDir()
	write(t, led, "regression_dupparam_test.go", passing)

	l, err := LoadLedger(led)
	if err != nil {
		t.Fatal(err)
	}
	if err := l.Install(dir); err != nil {
		t.Fatal(err)
	}
	cfg := Config{Workspace: dir, RequireTests: true, RegressionDir: led}

	if rep := Run(context.Background(), cfg); !rep.OK() {
		t.Fatalf("a tree where the regression passes was failed: %+v", rep.Failed())
	}

	// The defect returns.
	write(t, dir, "regression_dupparam_test.go", failing)
	rep := Run(context.Background(), cfg)
	if rep.OK() {
		t.Fatal("a returning defect passed the gate")
	}
	if !strings.Contains(rep.Failed()[0].Detail, "back") {
		t.Errorf("detail does not say the defect returned: %s", rep.Failed()[0].Detail)
	}
}

// Deleting the test is the cheapest way to stop it failing, and it repairs
// nothing — the same rule the reviewer's markFixed enforces.
func TestDeletingARegressionTestIsNotAPass(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	led := t.TempDir()
	write(t, led, "regression_dupparam_test.go", passing)

	l, _ := LoadLedger(led)
	if err := l.Install(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "regression_dupparam_test.go")); err != nil {
		t.Fatal(err)
	}

	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, RegressionDir: led})
	if rep.OK() {
		t.Fatal("removing the regression test passed the gate")
	}
	if !strings.Contains(rep.Failed()[0].Detail, "Restore them") {
		t.Errorf("detail does not demand restoration: %s", rep.Failed()[0].Detail)
	}
}

// The first run of a task has no history, and that is not a failure.
func TestAnEmptyLedgerIsSilent(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	for _, led := range []string{"", filepath.Join(t.TempDir(), "never-created")} {
		rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, RegressionDir: led})
		if !rep.OK() {
			t.Fatalf("ledger %q failed the gate: %+v", led, rep.Failed())
		}
		for _, c := range rep.Checks {
			if c.Name == "no regressions" {
				t.Errorf("ledger %q added a check with nothing in it", led)
			}
		}
	}
}

// Promote is what turns a repaired finding into a permanent guard.
func TestPromoteCarriesTheTestIntoTheLedger(t *testing.T) {
	dir := t.TempDir()
	led := filepath.Join(t.TempDir(), "ledger")
	write(t, dir, "review_thing_test.go", passing)

	if err := Promote(led, dir, "review_thing_test.go"); err != nil {
		t.Fatal(err)
	}
	l, err := LoadLedger(led)
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Files) != 1 || l.Files[0] != "review_thing_test.go" {
		t.Fatalf("ledger holds %v", l.Files)
	}
}
