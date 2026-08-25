package benchmarks

import (
	"context"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/rl"
)

// hermetic applies GoEnv for the duration of a test and silences the verifier
// logging, which is a warn line with the whole of a failing build's output and
// is exactly what several of these tests are asserting SHOULD happen.
func hermetic(t *testing.T) {
	t.Helper()
	for _, kv := range GoEnv() {
		k, v, _ := strings.Cut(kv, "=")
		t.Setenv(k, v)
	}
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Cleanup(func() { slog.SetDefault(prev) })
}

// materialize writes a task's fixture into a fresh directory and returns the
// episode a verifier would be handed for it. The episode did nothing: this is
// the state the agent STARTS from, which is the baseline every fixture
// assertion below is about.
func materialize(t *testing.T, task Task) *rl.Episode {
	t.Helper()
	dir := t.TempDir()
	if err := writeTree(dir, task.Files); err != nil {
		t.Fatalf("writing the %s fixture: %v", task.ID, err)
	}
	return &rl.Episode{Task: rl.Task{ID: task.ID, Workspace: dir}}
}

// grade runs a task's verifiers over an episode.
func grade(t *testing.T, task Task, ep *rl.Episode) (float64, map[string]float64) {
	t.Helper()
	total, breakdown, err := rl.Score(task.Verifiers...)(context.Background(), ep)
	if err != nil {
		t.Fatalf("scoring %s: %v", task.ID, err)
	}
	return total, breakdown
}

func mustTask(t *testing.T, id string) Task {
	t.Helper()
	task, ok := Lookup(id)
	if !ok {
		t.Fatalf("no task %q", id)
	}
	return task
}

// ===== the suite as data =====

// TestWeightsSumToOne pins the arithmetic every task's verifier weights are
// written to satisfy.
//
// It matters because the default success threshold is 1.0: a task whose weights
// sum to 0.95 can never be passed, and one that sums to 1.1 is passed by doing
// most of it. A verifier only reports its weight when it PASSES, so the sum
// cannot be read back off the values — arranging a workspace where a greenfield
// task passes everything means writing the todo app. The literals are restated
// here instead, which at least fails loudly when a weight is edited and its
// neighbour is not.
func TestWeightsSumToOne(t *testing.T) {
	want := map[string]float64{
		"read-only-qa": 0.2 + 0.8,
		"fix-bug":      0.6 + 0.4,
		"add-feature":  0.20 + 0.10 + 0.10 + 0.40 + 0.20,
		"todo-app":     0.15 + 0.15 + 0.20 + 0.10 + 0.40,

		"refactor-interface": 0.10 + 0.20 + 0.70,
		"cross-file-bug":     0.25 + 0.15 + 0.15 + 0.45,
		"needle-in-haystack": 0.10 + 0.60 + 0.30,

		// The graded verifier contributes its MAXIMUM, which is what "sum to
		// 1.0" has to mean once a verifier can return something between zero
		// and its weight: the top of the scale is what the success threshold
		// is compared against.
		"ambiguous-spec": 0.05 + 0.05 + 0.10 + 0.80,
	}
	for id, sum := range want {
		if sum < 0.999 || sum > 1.001 {
			t.Errorf("%s: weights sum to %v, want 1.0", id, sum)
		}
	}
	if len(want) != len(All()) {
		t.Errorf("the suite has %d tasks but this test knows about %d", len(All()), len(want))
	}
}

func TestSelect(t *testing.T) {
	if got, err := Select(nil); err != nil || len(got) != len(All()) {
		t.Fatalf("Select(nil) = %d tasks, %v; want the whole suite", len(got), err)
	}

	got, err := Select([]string{"todo-app", " fix-bug "})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if len(got) != 2 || got[0].ID != "todo-app" || got[1].ID != "fix-bug" {
		t.Fatalf("Select did not preserve order: %v", ids(got))
	}

	if _, err := Select([]string{"nope"}); err == nil {
		t.Fatal("Select accepted an unknown task id; a run that silently skips the task you " +
			"asked about reports a clean sweep over the ones you did not")
	}
}

// TestSelectTiers covers the tier aliases, including the de-duplication that
// keeps `-tasks easy,fix-bug` from putting two rows with the same task id in
// one report.
func TestSelectTiers(t *testing.T) {
	hard, err := Select([]string{"hard"})
	if err != nil {
		t.Fatalf("Select(hard): %v", err)
	}
	if got, want := ids(hard), ids(Hard()); !slices.Equal(got, want) {
		t.Errorf("Select(hard) = %v, want %v", got, want)
	}

	mixed, err := Select([]string{"easy", "cross-file-bug", "fix-bug"})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	want := append(ids(Easy()), "cross-file-bug")
	if got := ids(mixed); !slices.Equal(got, want) {
		t.Errorf("Select(easy,cross-file-bug,fix-bug) = %v, want %v (fix-bug is already in easy)", got, want)
	}

	if got, err := Select([]string{"all"}); err != nil || !slices.Equal(ids(got), ids(All())) {
		t.Errorf("Select(all) = %v, %v; want the whole suite", ids(got), err)
	}
}

// TestTierNamesDoNotCollide holds the line [Select] leans on: it checks tiers
// before task ids, so a task called "hard" would become unreachable from the
// command line without anything failing.
func TestTierNamesDoNotCollide(t *testing.T) {
	tiers := Tiers()
	for _, task := range All() {
		if _, clash := tiers[task.ID]; clash {
			t.Errorf("task %q has the same name as a tier, so -tasks %s can never select it",
				task.ID, task.ID)
		}
	}
}

// TestEveryTaskIsInExactlyOneTier stops a task being added to All by way of a
// tier and then quietly dropped from the other, or added to both and run
// twice.
func TestEveryTaskIsInExactlyOneTier(t *testing.T) {
	count := map[string]int{}
	for _, t := range append(Easy(), Hard()...) {
		count[t.ID]++
	}
	for _, task := range All() {
		if count[task.ID] != 1 {
			t.Errorf("%s appears in %d tiers, want exactly 1", task.ID, count[task.ID])
		}
	}
	if len(count) != len(All()) {
		t.Errorf("the tiers hold %d distinct tasks but All returns %d", len(count), len(All()))
	}
}

// TestTaskIDsAreUnique guards Lookup, which returns the first match: two tasks
// sharing an id would make one of them unrunnable and would collide in the
// report, which is keyed by task id.
func TestTaskIDsAreUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, task := range All() {
		if seen[task.ID] {
			t.Errorf("two tasks share the id %q", task.ID)
		}
		seen[task.ID] = true
	}
}

func ids(ts []Task) []string {
	out := make([]string, len(ts))
	for i, t := range ts {
		out[i] = t.ID
	}
	return out
}

func TestFixtureWriteIsDeterministic(t *testing.T) {
	for _, task := range All() {
		a, b := t.TempDir(), t.TempDir()
		for _, dir := range []string{a, b} {
			if err := writeTree(dir, task.Files); err != nil {
				t.Fatalf("%s: %v", task.ID, err)
			}
		}
		for rel := range task.Files {
			ba := read(t, filepath.Join(a, filepath.FromSlash(rel)))
			bb := read(t, filepath.Join(b, filepath.FromSlash(rel)))
			if ba != bb {
				t.Errorf("%s: %s differs between two Resets", task.ID, rel)
			}
		}
	}
}

func read(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(b)
}

// ===== the checksum verifier =====

func TestUnchanged(t *testing.T) {
	const body = "package p\n\nfunc F() int { return 1 }\n"
	ep := &rl.Episode{Task: rl.Task{Workspace: t.TempDir()}}
	path := filepath.Join(ep.Task.Workspace, "p.go")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	v := Unchanged("untouched", "p.go", body, 0.4)

	if _, got := v(context.Background(), ep); got != 0.4 {
		t.Errorf("identical file scored %v, want 0.4", got)
	}

	// One byte. The whole point of the verifier is that "I only deleted the
	// failing case" is not a smaller edit than any other.
	if err := os.WriteFile(path, []byte(body+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, got := v(context.Background(), ep); got != 0 {
		t.Errorf("edited file scored %v, want 0", got)
	}

	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, got := v(context.Background(), ep); got != 0 {
		t.Errorf("deleted file scored %v, want 0", got)
	}
}

// ===== the fixtures themselves =====

// TestFixBugFixtureIsBrokenAndFixable pins both halves of the task.
//
// Broken: the shipped fixture must fail its test, or the agent is asked to fix
// something that already works. Fixable: the ONE documented edit must make it
// pass, or the task is unsolvable and the benchmark measures nothing but the
// agent's stamina.
func TestFixBugFixtureIsBrokenAndFixable(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "fix-bug")
	ep := materialize(t, task)

	total, breakdown := grade(t, task, ep)
	if breakdown["test"] != 0 {
		t.Error("fix-bug ships with a passing test; there is nothing to fix")
	}
	if breakdown["test_untouched"] != 0.4 {
		t.Errorf("the untouched fixture failed its own checksum: %v", breakdown)
	}
	if total != 0.4 {
		t.Errorf("unsolved fix-bug scored %v, want 0.4 (checksum only)", total)
	}

	// The intended fix, and nothing else: the loop bound.
	fixed := strings.Replace(fixBugSource, "i < len(xs)-1", "i < len(xs)", 1)
	if fixed == fixBugSource {
		t.Fatal("the off-by-one this task is built around is no longer in window.go")
	}
	if err := os.WriteFile(filepath.Join(ep.Task.Workspace, "window.go"), []byte(fixed), 0o644); err != nil {
		t.Fatal(err)
	}
	if total, _ := grade(t, task, ep); total != 1.0 {
		t.Errorf("the fixed fixture scored %v, want 1.0", total)
	}
}

// TestFixBugRewardsFixingNotDeleting is the verifier this suite exists to
// demonstrate.
//
// An agent that replaces the test file with one that passes gets `go test` to
// exit 0, which is 0.6 of 1.0 — and would be a full pass under any scorer that
// only runs the tests. The checksum caps it below the success threshold.
func TestFixBugRewardsFixingNotDeleting(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "fix-bug")
	ep := materialize(t, task)

	cheat := "package window\n\nimport \"testing\"\n\nfunc TestMax(t *testing.T) {}\n"
	if err := os.WriteFile(filepath.Join(ep.Task.Workspace, "window_test.go"), []byte(cheat), 0o644); err != nil {
		t.Fatal(err)
	}

	total, breakdown := grade(t, task, ep)
	if breakdown["test"] != 0.6 {
		t.Fatalf("the gutted test suite did not pass, so this test is not testing what it says: %v", breakdown)
	}
	if breakdown["test_untouched"] != 0 {
		t.Error("the checksum accepted a rewritten test file")
	}
	if total >= 1.0 {
		t.Errorf("deleting the test scored %v, which is a pass", total)
	}
}

// TestAddFeatureFixtureWorksAndLacksTheFeature pins the starting state: the
// program builds, its tests pass, its current output is what the verifier
// expects to survive, and the one thing being asked for is missing.
func TestAddFeatureFixtureWorksAndLacksTheFeature(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "add-feature")
	ep := materialize(t, task)

	_, breakdown := grade(t, task, ep)
	for _, name := range []string{"existing_tests", "tests_untouched", "build", "default_unchanged"} {
		if breakdown[name] == 0 {
			t.Errorf("%s failed on the untouched fixture; the task starts broken", name)
		}
	}
	if breakdown["top_flag"] != 0 {
		t.Error("the fixture already implements -top; there is no feature to add")
	}
}

// TestTodoAppEmptyWorkspaceScoresZero guards the freebie.
//
// `go build ./...` in a module with no Go files exits 0 with a warning, so a
// naive build check would award an agent that did nothing at all. Every
// verifier must score zero here.
func TestTodoAppEmptyWorkspaceScoresZero(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "todo-app")
	ep := materialize(t, task)

	total, breakdown := grade(t, task, ep)
	if total != 0 {
		t.Errorf("an empty todo-app workspace scored %v: %v", total, breakdown)
	}
}

// TestReadOnlyQANeedsBothFiles pins the two hops.
//
// If the answer could be grepped out of one file the task would measure
// grepping. Neither file contains both the port and the service name.
func TestReadOnlyQANeedsBothFiles(t *testing.T) {
	task := mustTask(t, "read-only-qa")
	const answer = "ledger-api"

	ports := task.Files["docs/ports.md"]
	services := task.Files["docs/services.md"]

	if !strings.Contains(ports, "8081") {
		t.Fatal("ports.md no longer mentions the port the question asks about")
	}
	if strings.Contains(ports, answer) {
		t.Error("ports.md contains the answer, so one file is enough")
	}
	if strings.Contains(services, "8081") {
		t.Error("services.md contains the port, so one file is enough")
	}
	if !strings.Contains(services, answer) {
		t.Fatal("services.md no longer contains the answer")
	}

	// And the join has to be findable: the codename must appear in both.
	if !strings.Contains(ports, "basalt") || !strings.Contains(services, "basalt") {
		t.Error("the codename that joins the two tables is missing from one of them")
	}
}

// TestReadOnlyQAScoresARightAnswer walks the verifiers over an episode that
// did the task, and over one that answered a different service.
func TestReadOnlyQAScoresARightAnswer(t *testing.T) {
	task := mustTask(t, "read-only-qa")

	right := materialize(t, task)
	write(t, right.Task.Workspace, "ANSWER.txt", "ledger-api\n")
	if total, breakdown := grade(t, task, right); total != 1.0 {
		t.Errorf("the right answer scored %v: %v", total, breakdown)
	}

	wrong := materialize(t, task)
	write(t, wrong.Task.Workspace, "ANSWER.txt", "web-frontend\n")
	total, breakdown := grade(t, task, wrong)
	if breakdown["answer_file"] != 0.2 {
		t.Error("writing the file at all should be worth something")
	}
	if breakdown["answer"] != 0 || total != 0.2 {
		t.Errorf("a wrong answer scored %v: %v", total, breakdown)
	}
}

func write(t *testing.T, dir, rel, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, rel), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestFixturesAreGofmtClean keeps the fixtures honest: an agent asked to run
// `go vet` on code that is not even formatted is being taught that the
// starting state is untrustworthy.
func TestFixturesAreGofmtClean(t *testing.T) {
	if _, err := exec.LookPath("gofmt"); err != nil {
		t.Skip("gofmt not on PATH")
	}
	for _, task := range All() {
		for rel, body := range task.Files {
			if filepath.Ext(rel) != ".go" {
				continue
			}
			cmd := exec.Command("gofmt", "-l")
			cmd.Stdin = strings.NewReader(body)
			out, err := cmd.Output()
			if err != nil {
				t.Fatalf("%s %s: gofmt: %v", task.ID, rel, err)
			}
			if len(out) > 0 {
				t.Errorf("%s: %s is not gofmt-clean", task.ID, rel)
			}
		}
	}
}
