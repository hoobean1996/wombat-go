package review

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseTodosAcceptsTheFormsModelsWrite(t *testing.T) {
	items := parseTodos(`# TODO
- [x] queue.go: schema
- [ ] Ack: verify caller is current owner
  - [X] nested and ticked
* [ ] star bullet
- [~] not a checkbox
- [] too short
plain line
- [x]
`)
	if len(items) != 4 {
		t.Fatalf("parsed %d items, want 4: %+v", len(items), items)
	}
	if !items[0].Done || items[0].Text != "queue.go: schema" {
		t.Errorf("item 0 = %+v", items[0])
	}
	if items[1].Done || items[1].Text != "Ack: verify caller is current owner" {
		t.Errorf("item 1 = %+v", items[1])
	}
	if !items[2].Done {
		t.Errorf("uppercase X should count as ticked: %+v", items[2])
	}
	if items[3].Done || items[3].Text != "star bullet" {
		t.Errorf("item 3 = %+v", items[3])
	}
}

func TestUnfinishedTodosReportsOnlyUntickedItems(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, DefaultTodoFile, "- [x] done thing\n- [ ] Nack: owner check\n- [ ] test: TestAckByNonOwner\n")

	open := unfinishedTodos(dir, DefaultTodoFile)
	if len(open) != 2 {
		t.Fatalf("got %d unfinished, want 2: %v", len(open), open)
	}
	if open[0] != "Nack: owner check" {
		t.Errorf("open[0] = %q", open[0])
	}
}

// A missing TODO file must not look like an incomplete plan, and must not be
// an error: the planner skill may not have run at all.
func TestMissingPlanFilesAreNotFailures(t *testing.T) {
	dir := t.TempDir()
	if got := unfinishedTodos(dir, DefaultTodoFile); got != nil {
		t.Errorf("missing TODO gave %v, want nil", got)
	}
	if got := readPlan(dir, DefaultPlanFile); got != "" {
		t.Errorf("missing PLAN gave %q, want empty", got)
	}
}

// The whole point of reading TODO.md: an author that ticked nothing has told
// us it did not finish, and that must not be reported as a clean review.
func TestUnfinishedTodoIsNotACleanReview(t *testing.T) {
	rep := &Report{Ran: true, Unfinished: []string{"Nack: owner check"}}
	if rep.Clean() {
		t.Fatal("a review with an unfinished planned item reported Clean")
	}
	if !strings.Contains(rep.Summary(), "unfinished") {
		t.Fatalf("summary hides the unfinished item: %q", rep.Summary())
	}
	rep.Unfinished = nil
	if !rep.Clean() {
		t.Fatal("a review that ran with nothing outstanding should be Clean")
	}
}

func TestPlanIsCappedNotDropped(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, DefaultPlanFile, strings.Repeat("invariant line\n", 2000))

	got := readPlan(dir, DefaultPlanFile)
	if len(got) <= planCap {
		t.Fatalf("expected the cap marker to be appended, got %d bytes", len(got))
	}
	if !strings.HasSuffix(got, "[plan truncated]") {
		t.Errorf("truncation is silent; tail = %q", got[len(got)-40:])
	}
}

// The reviewer must be told the plan is a lead, never the authority — a plan
// that narrowed its own invariant would otherwise license the same narrowing
// in review.
func TestReviewPromptCarriesThePlanAndSubordinatesIt(t *testing.T) {
	p := reviewPrompt("build a queue", "Invariant: owned by at most one worker at any moment.", nil, lensFor(1), nil)
	if !strings.Contains(p, "at most one worker at any moment") {
		t.Fatal("plan text missing from the review prompt")
	}
	if !strings.Contains(p, "the task is right and the plan is a defect in itself") {
		t.Error("prompt does not subordinate the plan to the task")
	}
	if bare := reviewPrompt("build a queue", "", nil, lensFor(1), nil); strings.Contains(bare, "USING THE PLAN") {
		t.Error("plan section appears even with no plan")
	}
}

func write(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// The repair agent rewrote TODO.md from 22 items to 7 and ticked them all. The
// instruction not to is in fixPrompt; this is what makes it checkable.
func TestPlanRewriteIsDetectedAndIsNotClean(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, DefaultTodoFile, "- [ ] a\n- [ ] b\n- [ ] c\n")
	write(t, dir, DefaultPlanFile, "Invariant: one owner at a time.\n")

	before := planSnapshot(dir, DefaultPlanFile, DefaultTodoFile)
	if got := planEdits(dir, before); len(got) != 0 {
		t.Fatalf("untouched files reported as edited: %v", got)
	}

	write(t, dir, DefaultTodoFile, "- [x] a\n") // shrunk and ticked
	got := planEdits(dir, before)
	if len(got) != 1 || got[0] != DefaultTodoFile {
		t.Fatalf("planEdits = %v, want [%s]", got, DefaultTodoFile)
	}

	rep := &Report{Ran: true, PlanEdited: got}
	if rep.Clean() {
		t.Error("a run that rewrote the plan reported Clean")
	}
	if !strings.Contains(rep.Summary(), DefaultTodoFile) {
		t.Errorf("summary hides the rewrite: %q", rep.Summary())
	}
}

// A plan file that never existed must not look like one that was deleted.
func TestPlanSnapshotHandlesAbsentFiles(t *testing.T) {
	dir := t.TempDir()
	before := planSnapshot(dir, DefaultPlanFile)
	if got := planEdits(dir, before); len(got) != 0 {
		t.Fatalf("absent file reported as edited: %v", got)
	}
	write(t, dir, DefaultPlanFile, "created mid-run\n")
	if got := planEdits(dir, before); len(got) != 1 {
		t.Fatalf("a file created during the run was not reported: %v", got)
	}
}

func TestFixPromptForbidsEditingThePlan(t *testing.T) {
	p := fixPrompt("t", []Finding{{Test: "TestX", File: "review_x_test.go", Message: "m"}})
	if !strings.Contains(p, "Do not edit PLAN.md or TODO.md") {
		t.Error("the fixer is not told to leave the plan alone")
	}
}
