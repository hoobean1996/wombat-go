package accept

import (
	"context"
	"strings"
	"testing"
)

// Counting unticked boxes cannot see a deleted one. Measured three times: 17→8,
// 22→7, 90→89, each fully ticked and each reported complete.
func TestShorteningThePlanFailsTheGate(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	write(t, dir, "TODO.md", "- [ ] a\n- [ ] b\n- [ ] c\n")
	write(t, dir, "Verifier.md", "## V1. one\nStatus: [ ]\nTest:\n\n## V2. two\nStatus: [ ]\nTest:\n")

	if err := RecordCensus(dir, "TODO.md", "Verifier.md"); err != nil {
		t.Fatal(err)
	}
	cfg := Config{Workspace: dir, RequireTests: true, TodoFile: "TODO.md", VerifierFile: "Verifier.md"}

	// The shortcut: delete two items and tick the rest.
	write(t, dir, "TODO.md", "- [x] a\n")
	write(t, dir, "Verifier.md", "## V1. one\nStatus: [x]\nTest: TestReal\n")

	rep := Run(context.Background(), cfg)
	if rep.OK() {
		t.Fatal("a plan shortened from 3+2 items to 1+1 passed the gate")
	}
	var got *Check
	for i := range rep.Checks {
		if rep.Checks[i].Name == "plan intact" {
			got = &rep.Checks[i]
		}
	}
	if got == nil || got.OK {
		t.Fatalf("the census check did not fail: %+v", rep.Checks)
	}
	for _, want := range []string{"TODO.md went from 3 items to 1", "Verifier.md went from 2 items to 1"} {
		if !strings.Contains(got.Detail, want) {
			t.Errorf("detail missing %q:\n%s", want, got.Detail)
		}
	}
}

// Noticing more work is not the failure mode.
func TestGrowingThePlanIsAllowed(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	write(t, dir, "TODO.md", "- [ ] a\n")
	if err := RecordCensus(dir, "TODO.md", ""); err != nil {
		t.Fatal(err)
	}
	write(t, dir, "TODO.md", "- [x] a\n- [x] b\n- [x] c\n")

	if rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, TodoFile: "TODO.md"}); !rep.OK() {
		t.Fatalf("growing the plan failed the gate: %+v", rep.Failed())
	}
}

// A run with no plan at all records nothing and is not failed for it.
func TestNoPlanRecordsNoCensus(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	if err := RecordCensus(dir, "TODO.md", "Verifier.md"); err != nil {
		t.Fatal(err)
	}
	if rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, TodoFile: "TODO.md"}); !rep.OK() {
		t.Fatalf("a run with no plan was failed: %+v", rep.Failed())
	}
}
