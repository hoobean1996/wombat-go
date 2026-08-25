package accept

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, dir, name, body string) {
	t.Helper()
	full := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func goModule(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	write(t, dir, "go.mod", "module example.com/x\n\ngo 1.24\n")
	write(t, dir, "x.go", "package x\n\nfunc Add(a, b int) int { return a + b }\n")
	return dir
}

// The check the measured run needed: `go test` exits zero and prints "no test
// files", and that was accepted as a finished answer.
func TestNoTestsIsNotAPassingSuite(t *testing.T) {
	dir := goModule(t)
	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true})

	if rep.OK() {
		t.Fatalf("a tree with no tests passed the gate: %s", rep.Summary())
	}
	var found bool
	for _, c := range rep.Failed() {
		if c.Name == "tests exist" {
			found = true
			if !strings.Contains(c.Detail, "nothing was checked") {
				t.Errorf("detail does not say why it matters: %q", c.Detail)
			}
		}
	}
	if !found {
		t.Fatalf("no 'tests exist' failure: %+v", rep.Failed())
	}

	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {\n\tif Add(1, 2) != 3 {\n\t\tt.Fatal(\"bad\")\n\t}\n}\n")
	if rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true}); !rep.OK() {
		t.Fatalf("a tree with a passing test failed the gate: %s", rep.Summary())
	}
}

// A broken build must not also report vet and test failures: they are the same
// compile error a second time, and they bury the one that matters.
func TestABrokenBuildReportsOnlyTheBuild(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "go.mod", "module example.com/x\n\ngo 1.24\n")
	write(t, dir, "x.go", "package x\n\nfunc Add(a, b int) int { return a + }\n")

	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true})
	failed := rep.Failed()
	if len(failed) != 1 || failed[0].Name != "go build" {
		t.Fatalf("expected only the build to fail, got %+v", failed)
	}
	if len(rep.Skipped) == 0 {
		t.Error("the skipped checks were not recorded")
	}
}

func TestUntickedTodoFailsTheGate(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {}\n")
	write(t, dir, "TODO.md", "- [x] lexer\n- [ ] scheduler: FIFO ready queue\n- [ ] tests\n")

	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, TodoFile: "TODO.md"})
	failed := rep.Failed()
	if len(failed) != 1 || failed[0].Name != "todo complete" {
		t.Fatalf("expected the todo check to fail, got %+v", failed)
	}
	if !strings.Contains(failed[0].Detail, "scheduler: FIFO ready queue") {
		t.Errorf("the unticked item is not named: %q", failed[0].Detail)
	}
	if !strings.Contains(failed[0].Detail, "Do not delete") {
		t.Error("nothing warns against shortening the list, which is how the last two runs 'finished' it")
	}
}

// A missing plan file is not a failure: the planner may not have run at all.
func TestMissingTodoIsNotAFailure(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {}\n")
	if rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, TodoFile: "TODO.md"}); !rep.OK() {
		t.Fatalf("a missing TODO file failed the gate: %+v", rep.Failed())
	}
}

// Something that is not a Go module must be skipped, not failed: the gate is
// not entitled to fail an agent for a language it cannot check.
func TestNonGoTreeIsSkippedNotFailed(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "main.py", "print('hi')\n")
	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true})
	if !rep.OK() {
		t.Fatalf("a non-Go tree was failed: %+v", rep.Failed())
	}
	if len(rep.Skipped) == 0 {
		t.Error("the skip was not recorded, so the report reads as a real pass")
	}
}

func TestFeedbackIsWorkNotAdvice(t *testing.T) {
	rep := &Report{Checks: []Check{
		{Name: "tests exist", OK: false, Detail: "the tree contains no _test.go files at all."},
		{Name: "todo complete", OK: false, Detail: "TODO.md still has 17 unticked item(s)"},
	}}
	f := Feedback(rep)
	if !strings.Contains(f, "You are not done yet") {
		t.Error("the feedback does not say the work is unfinished")
	}
	if !strings.Contains(f, "2 of them failed") {
		t.Error("the failure count is missing")
	}
	if !strings.Contains(f, "no _test.go files") || !strings.Contains(f, "17 unticked") {
		t.Error("the specifics did not survive into the feedback")
	}
	if !strings.Contains(f, "removing what it measures") {
		t.Error("nothing forbids passing a check by deleting what it measures")
	}
	if Feedback(&Report{}) != "" {
		t.Error("a clean report produced feedback")
	}
}
