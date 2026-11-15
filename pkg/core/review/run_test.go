package review

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// scratch builds a tiny Go module with the given files.
func scratch(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module scratch\n\ngo 1.21\n")
	for name, body := range files {
		write(name, body)
	}
	return dir
}

const failingTest = `package scratch

import "testing"

func TestReviewFindsSomething(t *testing.T) {
	t.Errorf("released the claim but the old holder can still ack it")
}
`

const passingTest = `package scratch

import "testing"

func TestReviewCheckedAndFine(t *testing.T) {}
`

// A test the reviewer wrote and watched PASS is a hypothesis it discarded, not
// a defect. Counting it would reward writing tests rather than finding
// anything, which is the failure every prose-reporting judge in this project
// fell into from the other direction.
func TestPassingReviewTestIsNotAFinding(t *testing.T) {
	dir := scratch(t, map[string]string{
		"review_ok_test.go":  passingTest,
		"review_bad_test.go": failingTest,
	})
	cfg := Config{Workspace: dir}

	got := collect(context.Background(), cfg)
	if len(got) != 1 {
		t.Fatalf("collected %d finding(s), want 1: %+v", len(got), got)
	}
	if got[0].Test != "TestReviewFindsSomething" {
		t.Errorf("collected %q, want the failing test", got[0].Test)
	}
	if !strings.Contains(got[0].Message, "old holder can still ack it") {
		t.Errorf("the finding carries %q; it should carry what the test actually reported",
			got[0].Message)
	}
}

// The cheapest way to turn a red test green is to delete it, and that repair
// leaves the defect exactly where it was.
func TestDeletingTheTestDoesNotCountAsFixed(t *testing.T) {
	dir := scratch(t, map[string]string{"review_bad_test.go": failingTest})
	cfg := Config{Workspace: dir}

	rep := &Report{Workspace: dir, Ran: true, Findings: collect(context.Background(), cfg)}
	if len(rep.Findings) != 1 {
		t.Fatalf("setup: %d finding(s), want 1", len(rep.Findings))
	}

	// The "repair": remove the evidence.
	if err := os.Remove(filepath.Join(dir, "review_bad_test.go")); err != nil {
		t.Fatal(err)
	}
	markFixed(context.Background(), cfg, rep)

	if rep.Findings[0].Fixed {
		t.Error("a finding whose test was deleted was credited as fixed")
	}
	if !strings.Contains(rep.Findings[0].Message, "removed during repair") {
		t.Errorf("the report does not say the demonstration was removed: %q",
			rep.Findings[0].Message)
	}
	if len(rep.Outstanding()) != 1 {
		t.Error("the deleted finding is not outstanding")
	}
}

// A genuine repair is credited, and only when the test that demonstrated the
// defect still exists and now passes.
func TestRepairIsCreditedOnlyByTheTestItself(t *testing.T) {
	dir := scratch(t, map[string]string{"review_bad_test.go": failingTest})
	cfg := Config{Workspace: dir}
	rep := &Report{Workspace: dir, Ran: true, Findings: collect(context.Background(), cfg)}
	if len(rep.Findings) != 1 {
		t.Fatalf("setup: %d finding(s), want 1", len(rep.Findings))
	}

	fixed := strings.Replace(failingTest,
		`t.Errorf("released the claim but the old holder can still ack it")`, "", 1)
	if err := os.WriteFile(filepath.Join(dir, "review_bad_test.go"), []byte(fixed), 0o644); err != nil {
		t.Fatal(err)
	}
	markFixed(context.Background(), cfg, rep)

	if !rep.Findings[0].Fixed {
		t.Error("a finding whose test now passes was not credited as fixed")
	}
	if len(rep.Outstanding()) != 0 {
		t.Error("a fixed finding is still outstanding")
	}
}

// Zero findings from a reviewer that never started reads exactly like zero
// findings from one that looked, and a caller acting on the second while
// holding the first is the whole reason this distinction exists.
func TestAReviewThatDidNotRunIsNotACleanReview(t *testing.T) {
	never := &Report{Ran: false}
	if never.Clean() {
		t.Error("a review that never ran reports itself clean")
	}
	looked := &Report{Ran: true}
	if !looked.Clean() {
		t.Error("a review that ran and found nothing does not report itself clean")
	}
	if !strings.Contains(never.Summary(), "did not run") {
		t.Errorf("summary is %q; it should say the review did not run", never.Summary())
	}
}

func TestRunRefusesWithoutAReviewer(t *testing.T) {
	rep := Run(context.Background(), Config{Workspace: t.TempDir()})
	if rep.Err == nil {
		t.Error("Run accepted a config with no reviewer")
	}
	if rep.Clean() {
		t.Error("a refused review reports itself clean")
	}
}

// TestShellRunsTheReviewTests: the tests a reviewer writes are code an agent
// wrote, and with Shell set they run through it rather than on the side.
func TestShellRunsTheReviewTests(t *testing.T) {
	dir := scratch(t, map[string]string{"review_findings_test.go": failingTest})
	var scripts []string
	cfg := Config{Workspace: dir, Shell: func(ctx context.Context, script string) (string, error) {
		scripts = append(scripts, script)
		return "--- FAIL: TestReviewFindsSomething (0.00s)\n    x_test.go:5: released the claim but the old holder can still ack it\nFAIL\n", errors.New("exit status 1")
	}}
	failed := runTests(context.Background(), cfg)
	if len(scripts) != 1 || scripts[0] != "go test -run TestReview -v ./..." {
		t.Errorf("Shell ran %q, want the one go test", scripts)
	}
	if _, ok := failed["TestReviewFindsSomething"]; !ok {
		t.Errorf("failed = %v, want the failure Shell reported", failed)
	}
}
