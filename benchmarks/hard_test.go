package benchmarks

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/rl"
	"github.com/hoobean1996/wombat-go/tool"
)

// The hard tier's fixture tests.
//
// Each task gets the same three questions, and the third is the one that
// matters. A hard task with a soft verifier is a slow easy task, and the only
// way to know which one you have is to write the cheat down and check that it
// is capped:
//
//  1. Broken as shipped — the achievement verifier scores 0 on the untouched
//     fixture, so the agent is not being paid for the fixture's own work.
//  2. Fixable — the intended edit, applied mechanically here, scores 1.0. A
//     task nobody can pass measures stamina.
//  3. Cheat-capped — the cheapest wrong way to green scores below 1.0, and the
//     BREAKDOWN says which verifier stopped it, so a future edit that removes
//     the cap fails here rather than in a $40 benchmark run.

// approx compares two rewards.
//
// Exact equality is wrong for a total: the rewards are sums of literals like
// 0.10 and 0.20 and the sum of those two is not 0.30 in binary floating point.
// A test that fails on the last bit of a float teaches whoever hits it to
// loosen the assertion, which is how a real regression gets waved through.
func approx(got, want float64) bool { return got-want < 1e-9 && want-got < 1e-9 }

// ===== refactor-interface =====

// refactorFix is the intended refactor, applied to a materialized fixture:
// the interface gains the method and all five implementations follow.
//
// upper, trim and prefix are the literals those three stages return from
// Idempotent, so one function produces both the correct refactor and the
// "return true everywhere" cheat.
func refactorFix(t *testing.T, dir string, upper, trim, prefix string) {
	t.Helper()

	iface := strings.Replace(refactorStage,
		"	// Process transforms one string.\n	Process(in string) string\n",
		"	// Process transforms one string.\n	Process(in string) string\n\n"+
			"	// Idempotent reports whether running Process twice on the same input\n"+
			"	// gives the same result as running it once.\n	Idempotent() bool\n", 1)
	if iface == refactorStage {
		t.Fatal("stage.go no longer has the shape this test edits; update refactorFix")
	}

	pipelineImpl := `
// Idempotent implements [Stage].
func (p *Pipeline) Idempotent() bool {
	for _, s := range p.stages {
		if !s.Idempotent() {
			return false
		}
	}
	return true
}
`
	for rel, body := range map[string]string{
		"stage.go":         iface,
		"upper.go":         refactorUpper + "\n// Idempotent implements [Stage].\nfunc (Upper) Idempotent() bool { return " + upper + " }\n",
		"trim.go":          refactorTrim + "\n// Idempotent implements [Stage].\nfunc (Trim) Idempotent() bool { return " + trim + " }\n",
		"prefix.go":        refactorPrefix + "\n// Idempotent implements [Stage].\nfunc (Prefix) Idempotent() bool { return " + prefix + " }\n",
		"pipeline.go":      refactorPipeline + pipelineImpl,
		"pipeline_test.go": refactorTest + "\nfunc (r recorder) Idempotent() bool { return true }\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, rel), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRefactorInterfaceIsBrokenAndFixable(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "refactor-interface")

	// Shipped: it compiles and its own tests pass — this is a refactor, not a
	// repair — but the method does not exist, so the probe does not compile.
	ep := materialize(t, task)
	total, breakdown := grade(t, task, ep)
	if breakdown["build"] != 0.10 || breakdown["existing_tests"] != 0.20 {
		t.Errorf("the shipped fixture does not build or does not pass its own tests: %v", breakdown)
	}
	if breakdown["new_method"] != 0 {
		t.Error("the fixture already has Idempotent; there is nothing to refactor")
	}
	if !approx(total, 0.30) {
		t.Errorf("the untouched fixture scored %v, want 0.30", total)
	}

	// Fixed: Upper and Trim are idempotent, Prefix is not.
	fixed := materialize(t, task)
	refactorFix(t, fixed.Task.Workspace, "true", "true", "false")
	if total, breakdown := grade(t, task, fixed); !approx(total, 1.0) {
		t.Errorf("the intended refactor scored %v: %v", total, breakdown)
	}
}

// TestRefactorInterfaceRewardsReadingNotGuessing is the anti-cheat.
//
// "Add the method, return true, make it compile" is the fastest way to a green
// `go test ./...`, and it is what an agent that never opened prefix.go would
// do. It has to be capped, or the task measures whether the agent can satisfy
// a type checker rather than whether it read the code.
func TestRefactorInterfaceRewardsReadingNotGuessing(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "refactor-interface")

	ep := materialize(t, task)
	refactorFix(t, ep.Task.Workspace, "true", "true", "true")

	total, breakdown := grade(t, task, ep)
	if breakdown["build"] != 0.10 || breakdown["existing_tests"] != 0.20 {
		t.Fatalf("the cheat did not even compile, so this test is not testing what it says: %v", breakdown)
	}
	if breakdown["new_method"] != 0 {
		t.Error("the probe accepted Idempotent() = true for a stage that is not idempotent")
	}
	if total >= 1.0 {
		t.Errorf("guessing scored %v, which is a pass", total)
	}
}

// TestRefactorInterfaceNeedsTheInterfaceItself pins the other half of the
// probe's job.
//
// The method on the three concrete stages, with the interface left exactly as
// it shipped, builds and passes every test the agent can see. It is not the
// task: nothing holding a Stage can call the new method, which is the only
// reason anybody adds a method to an interface. The probe catches it because
// it ranges over a []Stage, and that does not compile unless the method is on
// the interface.
func TestRefactorInterfaceNeedsTheInterfaceItself(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "refactor-interface")

	ep := materialize(t, task)
	for rel, body := range map[string]string{
		"upper.go":  refactorUpper + "\nfunc (Upper) Idempotent() bool { return true }\n",
		"trim.go":   refactorTrim + "\nfunc (Trim) Idempotent() bool { return true }\n",
		"prefix.go": refactorPrefix + "\nfunc (Prefix) Idempotent() bool { return false }\n",
	} {
		if err := os.WriteFile(filepath.Join(ep.Task.Workspace, rel), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	_, breakdown := grade(t, task, ep)
	if breakdown["build"] != 0.10 || breakdown["existing_tests"] != 0.20 {
		t.Fatalf("the half-refactor does not compile, so this test proves nothing: %v", breakdown)
	}
	if breakdown["new_method"] != 0 {
		t.Error("the probe passed without Idempotent on the Stage interface")
	}
}

// TestRefactorProbeIsNotInTheFixture states the property that makes a probe a
// probe: the agent cannot read it, so it cannot be written to.
func TestRefactorProbeIsNotInTheFixture(t *testing.T) {
	task := mustTask(t, "refactor-interface")
	if _, shipped := task.Files[refactorProbeFile]; shipped {
		t.Errorf("%s ships with the fixture; the agent can see the grader", refactorProbeFile)
	}
	if strings.Contains(strings.Join(fixtureBodies(task), "\n"), "TestWombatProbeIdempotent") {
		t.Error("the fixture names the probe test, which is most of the way to leaking it")
	}
}

// ===== cross-file-bug =====

// crossFileRealFix is the one-line repair in index.go: normalizeKey finally
// does what its own doc comment says.
const crossFileRealFix = `func normalizeKey(key string) string {
	return strings.ToLower(strings.Join(strings.Fields(key), " "))
}
`

// crossFileNaiveFix is the attractor: canonicalise in Store and leave Index
// broken. It turns the failing test green in one edit.
const crossFileNaiveFix = `func (s *Store) Put(key, value string) int {
	key = strings.Join(strings.Fields(key), " ")
	if id, ok := s.idx.Lookup(key); ok {`

func TestCrossFileBugIsBrokenAndFixable(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "cross-file-bug")

	ep := materialize(t, task)
	total, breakdown := grade(t, task, ep)
	if breakdown["test"] != 0 {
		t.Error("cross-file-bug ships with a passing test suite; there is nothing to fix")
	}
	if breakdown["store_untouched"] != 0.15 || breakdown["tests_untouched"] != 0.15 {
		t.Errorf("the untouched fixture failed its own checksums: %v", breakdown)
	}
	if breakdown["index_normalises"] != 0 {
		t.Error("the shipped normalizeKey already collapses internal whitespace")
	}
	if !approx(total, 0.30) {
		t.Errorf("the untouched fixture scored %v, want 0.30 (the two checksums)", total)
	}

	fixed := materialize(t, task)
	writeIndexFix(t, fixed.Task.Workspace, crossFileRealFix)
	if total, breakdown := grade(t, task, fixed); !approx(total, 1.0) {
		t.Errorf("fixing normalizeKey scored %v: %v", total, breakdown)
	}
}

// TestCrossFileBugCapsTheSymptomPatch is the verifier pair this task exists
// for.
//
// Normalising the key in Store.Put makes `go test ./...` pass — the whole of
// the visible evidence says the bug is fixed — and leaves Index, which is
// exported and has its own callers, exactly as broken as it was. It must score
// well under a real fix, and the breakdown must say why.
func TestCrossFileBugCapsTheSymptomPatch(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "cross-file-bug")

	ep := materialize(t, task)
	patched := strings.Replace(crossFileStore,
		"func (s *Store) Put(key, value string) int {\n	if id, ok := s.idx.Lookup(key); ok {",
		crossFileNaiveFix, 1)
	if patched == crossFileStore {
		t.Fatal("store.go no longer has the shape this test patches; update crossFileNaiveFix")
	}
	// Store.Get needs it too, for the ORDER TOTAL lookup.
	patched = strings.Replace(patched,
		"func (s *Store) Get(key string) (Record, error) {\n	id, ok := s.idx.Lookup(key)",
		"func (s *Store) Get(key string) (Record, error) {\n	key = strings.Join(strings.Fields(key), \" \")\n	id, ok := s.idx.Lookup(key)", 1)
	patched = strings.Replace(patched, `import "errors"`, "import (\n\t\"errors\"\n\t\"strings\"\n)", 1)
	if err := os.WriteFile(filepath.Join(ep.Task.Workspace, "store.go"), []byte(patched), 0o644); err != nil {
		t.Fatal(err)
	}

	total, breakdown := grade(t, task, ep)
	if breakdown["test"] != 0.25 {
		t.Fatalf("the symptom patch did not make the tests pass, so this test is not testing "+
			"what it says: %v", breakdown)
	}
	if breakdown["store_untouched"] != 0 {
		t.Error("the checksum accepted an edited store.go")
	}
	if breakdown["index_normalises"] != 0 {
		t.Error("the probe passed against a store.go patch; it must be reaching Index through Store")
	}
	if total >= 1.0 {
		t.Errorf("the symptom patch scored %v, which is a pass", total)
	}
	if total >= 0.60 {
		t.Errorf("the symptom patch scored %v; partial credit should be well short of half", total)
	}
}

// TestCrossFileBugCapsDeletingTheTest closes the other door. It is the same
// cheat fix-bug is built around, and it has to be capped here too: this task's
// failing test is one t.Fatalf away from never running.
func TestCrossFileBugCapsDeletingTheTest(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "cross-file-bug")

	ep := materialize(t, task)
	gutted := "package store\n\nimport \"testing\"\n\nfunc TestStoreRoundTrip(t *testing.T) {}\n"
	if err := os.WriteFile(filepath.Join(ep.Task.Workspace, "store_test.go"), []byte(gutted), 0o644); err != nil {
		t.Fatal(err)
	}

	total, breakdown := grade(t, task, ep)
	if breakdown["test"] != 0.25 {
		t.Fatalf("the gutted test suite did not pass: %v", breakdown)
	}
	if breakdown["tests_untouched"] != 0 {
		t.Error("the checksum accepted a rewritten store_test.go")
	}
	if total >= 1.0 {
		t.Errorf("deleting the test scored %v, which is a pass", total)
	}
}

// TestCrossFileBugSymptomAndCauseAreInDifferentFiles pins the property the
// task is named after. If the bug ever migrates into store.go the task becomes
// unsolvable, because store.go is checksummed.
func TestCrossFileBugSymptomAndCauseAreInDifferentFiles(t *testing.T) {
	task := mustTask(t, "cross-file-bug")

	if !strings.Contains(task.Files["index.go"], "func normalizeKey") {
		t.Fatal("normalizeKey is no longer in index.go, which is the whole premise")
	}
	if strings.Contains(task.Files["store.go"], "normalizeKey") ||
		strings.Contains(task.Files["store.go"], "strings.") {
		t.Error("store.go now normalises keys itself; the cause has moved into the checksummed file")
	}
	if !strings.Contains(task.Files["store_test.go"], "TestStoreSpellingsAreOneRecord") {
		t.Fatal("the failing test is gone from store_test.go")
	}

	// The fix must be reachable without touching either checksummed file.
	dir := t.TempDir()
	if err := writeTree(dir, task.Files); err != nil {
		t.Fatal(err)
	}
	writeIndexFix(t, dir, crossFileRealFix)
	for _, rel := range []string{"store.go", "store_test.go"} {
		if read(t, filepath.Join(dir, rel)) != task.Files[rel] {
			t.Errorf("the intended fix changed %s, which the prompt forbids", rel)
		}
	}
}

// writeIndexFix replaces normalizeKey's body in a materialized index.go.
func writeIndexFix(t *testing.T, dir, replacement string) {
	t.Helper()
	const broken = `func normalizeKey(key string) string {
	return strings.ToLower(strings.TrimSpace(key))
}
`
	fixed := strings.Replace(crossFileIndex, broken, replacement, 1)
	if fixed == crossFileIndex {
		t.Fatal("index.go no longer has the off-spec normalizeKey this test replaces")
	}
	if err := os.WriteFile(filepath.Join(dir, "index.go"), []byte(fixed), 0o644); err != nil {
		t.Fatal(err)
	}
}

// ===== ambiguous-spec =====

// mergeImpl renders a merge.go whose Merge has the given doc comment and
// collision rule, so a test can vary the two independently — which is exactly
// the pair this task grades.
func mergeImpl(doc, collision string) string {
	return `// Package mergemap combines score tables keyed by player name.
package mergemap

` + doc + `func Merge(a, b map[string]int) map[string]int {
	out := make(map[string]int, len(a)+len(b))
	for k, v := range a {
		out[k] = v
	}
	for k, v := range b {
		` + collision + `
	}
	return out
}
`
}

const (
	// mergeLastWins is what AskUserAnswer says to do.
	mergeLastWins = `out[k] = v`

	// mergeFirstWins is a perfectly defensible other choice — and the wrong
	// one for an agent that asked and was told otherwise.
	mergeFirstWins = `if _, ok := out[k]; !ok {
			out[k] = v
		}`

	mergeSilentDoc = `// Merge returns a new table holding every key from a and b.
//
// The result is a fresh map: neither argument is modified.
`
	mergeDocumentedDoc = `// Merge returns a new table holding every key from a and b.
//
// The result is a fresh map: neither argument is modified.
//
// On a collision — a key present in both tables — b's value wins. The task did
// not say which should, and picking silently would leave the next reader
// unable to tell the rule from an accident.
`
)

// askedEpisode returns an episode whose transcript contains an ask_user call.
//
// Built from llm.Message values rather than by driving an agent, because what
// the verifier reads is the transcript and this is what a transcript with an
// ask in it looks like.
func askedEpisode(t *testing.T, task Task, question string) *rl.Episode {
	t.Helper()
	ep := materialize(t, task)
	in, err := json.Marshal(map[string]string{"question": question})
	if err != nil {
		t.Fatal(err)
	}
	ep.Messages = []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.Text{Text: task.Prompt}}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.ToolUse{ID: "u1", Name: AskUserName, Input: in},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "u1", Content: AskUserAnswer},
		}},
	}
	return ep
}

func TestAmbiguousSpecShipsUnfinished(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "ambiguous-spec")

	ep := materialize(t, task)
	total, breakdown := grade(t, task, ep)
	if breakdown["build"] != 0.05 {
		t.Errorf("the shipped fixture does not build: %v", breakdown)
	}
	for _, name := range []string{"existing_tests", "collision_behaviour", "judgement"} {
		if breakdown[name] != 0 {
			t.Errorf("%s scored %v on an untouched fixture", name, breakdown[name])
		}
	}
	if !approx(total, 0.05) {
		t.Errorf("the untouched fixture scored %v, want 0.05 (it compiles, nothing else)", total)
	}

	// The gap has to be a real gap: no shipped test may pin the collision
	// rule, or the answer is in the workspace and the task is not ambiguous.
	for rel, body := range task.Files {
		if !strings.HasSuffix(rel, "_test.go") {
			continue
		}
		if strings.Contains(body, `"x": 1`) || strings.Contains(body, "collision") {
			t.Errorf("%s looks like it tests the collision case; the spec is no longer ambiguous", rel)
		}
	}
}

// TestAmbiguousSpecGradesJudgement is the task, in one function: three agents
// that all wrote working code, ranked by what they did about the thing nobody
// told them.
func TestAmbiguousSpecGradesJudgement(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "ambiguous-spec")

	silent := materialize(t, task)
	write(t, silent.Task.Workspace, "merge.go", mergeImpl(mergeSilentDoc, mergeLastWins))
	silentTotal, silentBreak := grade(t, task, silent)
	if silentBreak["existing_tests"] != 0.05 || silentBreak["collision_behaviour"] != 0.10 {
		t.Fatalf("the silent implementation does not work, so the comparison below is "+
			"about something else: %v", silentBreak)
	}
	if silentBreak["judgement"] != 0 {
		t.Errorf("choosing silently scored %v on judgement, want 0", silentBreak["judgement"])
	}

	documented := materialize(t, task)
	write(t, documented.Task.Workspace, "merge.go", mergeImpl(mergeDocumentedDoc, mergeLastWins))
	docTotal, docBreak := grade(t, task, documented)
	if docBreak["judgement"] != 0.30 {
		t.Errorf("documenting the choice scored %v on judgement, want 0.30", docBreak["judgement"])
	}

	askedEp := askedEpisode(t, task, "Merge(a, b): what should happen to a key that is in both maps?")
	write(t, askedEp.Task.Workspace, "merge.go", mergeImpl(mergeDocumentedDoc, mergeLastWins))
	askedTotal, askedBreak := grade(t, task, askedEp)
	if askedBreak["judgement"] != 0.80 {
		t.Errorf("asking scored %v on judgement, want 0.80", askedBreak["judgement"])
	}
	if !approx(askedTotal, 1.0) {
		t.Errorf("asking and then doing what it was told scored %v: %v", askedTotal, askedBreak)
	}

	// The ordering is the measurement. Assert it as an ordering, not only as
	// three numbers, so a future reweighting that inverts it fails here.
	if !(askedTotal > docTotal && docTotal > silentTotal) {
		t.Errorf("asked %v, documented %v, silent %v; want strictly decreasing",
			askedTotal, docTotal, silentTotal)
	}
}

// TestAmbiguousSpecHoldsAnAskerToTheAnswer covers the case that makes the
// canned reply worth having: an agent that asked, was told the second map
// wins, and implemented first-wins anyway.
func TestAmbiguousSpecHoldsAnAskerToTheAnswer(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "ambiguous-spec")

	ep := askedEpisode(t, task, "What should Merge do on a key collision?")
	write(t, ep.Task.Workspace, "merge.go", mergeImpl(mergeDocumentedDoc, mergeFirstWins))

	total, breakdown := grade(t, task, ep)
	if breakdown["judgement"] != 0.80 {
		t.Fatalf("it asked, so judgement should still be full: %v", breakdown)
	}
	if breakdown["collision_behaviour"] != 0 {
		t.Error("an agent that ignored the answer it asked for still scored the behaviour check")
	}
	if total >= 1.0 {
		t.Errorf("ignoring the answer scored %v, which is a pass", total)
	}

	// The same implementation from an agent that never asked is a legitimate
	// choice, and scores for it.
	quiet := materialize(t, task)
	write(t, quiet.Task.Workspace, "merge.go", mergeImpl(mergeDocumentedDoc, mergeFirstWins))
	if _, b := grade(t, task, quiet); b["collision_behaviour"] != 0.10 {
		t.Errorf("first-wins from an agent that did not ask scored %v; it was never told "+
			"otherwise: %v", b["collision_behaviour"], b)
	}
}

// TestAmbiguousSpecRejectsANonDeterministicChoice pins what "chose" means for
// an agent that did not ask. Dropping the key, or letting map iteration decide,
// is not a choice.
func TestAmbiguousSpecRejectsANonDeterministicChoice(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "ambiguous-spec")

	ep := materialize(t, task)
	write(t, ep.Task.Workspace, "merge.go", mergeImpl(mergeDocumentedDoc, `if _, dup := a[k]; !dup {
			out[k] = v
		} else {
			delete(out, k)
		}`))

	_, breakdown := grade(t, task, ep)
	if breakdown["collision_behaviour"] != 0 {
		t.Error("dropping the colliding key counted as deciding what to do with it")
	}
}

// TestAskedReadsTheTranscriptAndNotTheWorkspace states the property that makes
// the judgement score hard to fake: it is read from the harness's own record
// of what the model emitted.
func TestAskedReadsTheTranscriptAndNotTheWorkspace(t *testing.T) {
	task := mustTask(t, "ambiguous-spec")

	ep := materialize(t, task)
	// Everything an agent could write for itself, all at once.
	write(t, ep.Task.Workspace, "ask_user.log", "I asked about the key collision, honest")
	write(t, ep.Task.Workspace, "NOTES.md", "asked the user: what happens on a collision?")
	if asked(ep, mergeAskWords) {
		t.Error("a file in the workspace counted as having asked")
	}

	if !asked(askedEpisode(t, task, "which value wins when a key is in both?"), mergeAskWords) {
		t.Error("a real ask_user call in the transcript did not count")
	}

	// A different tool with the same words does not count either.
	other := materialize(t, task)
	other.Messages = []llm.Message{{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		llm.ToolUse{ID: "u1", Name: "bash", Input: json.RawMessage(`{"command":"echo which value wins on a collision"}`)},
	}}}
	if asked(other, mergeAskWords) {
		t.Error("a bash call counted as an ask_user call")
	}
}

// TestAskUserToolDoesNotPause is why this package has its own ask_user rather
// than using the builtin. A pause ends the episode, and an episode that ended
// cannot then be graded on what the agent did with the answer.
func TestAskUserToolDoesNotPause(t *testing.T) {
	def := AskUserTool(AskUserAnswer)
	if def.Name != AskUserName {
		t.Errorf("tool name = %q, want %q", def.Name, AskUserName)
	}
	if def.Has(tool.CapPause) {
		t.Error("the benchmark's ask_user carries CapPause, which ends the run")
	}
	if def.Needs != 0 {
		t.Error("ask_user needs something from the host, so a sandbox filter could strip it")
	}

	got, err := def.Fn(context.Background(), json.RawMessage(`{"question":"anything?"}`))
	if err != nil {
		t.Fatalf("ask_user returned an error: %v", err)
	}
	if got != AskUserAnswer {
		t.Errorf("ask_user answered %q, want the canned answer", got)
	}
	if !strings.Contains(strings.ToLower(AskUserAnswer), "second") {
		t.Error("the canned answer no longer settles the collision rule; the last-wins probe " +
			"is now grading an answer nobody gave")
	}
}

// TestAmbiguousSpecIsTheOnlyTaskWithTools keeps the ask_user tool out of the
// other seven. Handing it to a task with nothing to ask about would let an
// agent spend a turn on it, and the system prompt would be promising an
// answerer to an agent that does not need one.
func TestAmbiguousSpecIsTheOnlyTaskWithTools(t *testing.T) {
	for _, task := range All() {
		if task.ID == "ambiguous-spec" {
			if len(task.Tools) == 0 {
				t.Error("ambiguous-spec has no ask_user tool; the judgement axis is unpassable")
			}
			continue
		}
		if len(task.Tools) != 0 {
			t.Errorf("%s carries extra tools: %v", task.ID, task.Tools)
		}
	}
}

// ===== needle-in-haystack =====

func TestNeedleAppearsExactlyOnce(t *testing.T) {
	task := mustTask(t, "needle-in-haystack")

	var holding []string
	for rel, body := range task.Files {
		if strings.Contains(body, NeedleAnswer) {
			holding = append(holding, rel)
		}
	}
	if len(holding) != 1 {
		t.Fatalf("%s appears in %d files (%v), want exactly 1", NeedleAnswer, len(holding), holding)
	}
	if !strings.HasPrefix(holding[0], "logs/") {
		t.Errorf("the needle is in %s, not in the haystack", holding[0])
	}

	// The prompt must not contain the answer, which sounds obvious and is one
	// careless edit away.
	if strings.Contains(task.Prompt, NeedleAnswer) {
		t.Error("the prompt contains the answer")
	}
}

// TestNeedleHasDistractors is what stops this being a one-grep task.
//
// If only one file said "rollback", `grep -l rollback logs/` would answer the
// question and the task would measure grep. Several files match each of the
// question's terms, and only one matches both, so the agent has to intersect
// them.
func TestNeedleHasDistractors(t *testing.T) {
	task := mustTask(t, "needle-in-haystack")

	rollback, settlement, both := 0, 0, 0
	for rel, body := range task.Files {
		if !strings.HasPrefix(rel, "logs/") {
			continue
		}
		r := strings.Contains(body, "rolled back")
		s := strings.Contains(body, "settlement")
		if r {
			rollback++
		}
		if s {
			settlement++
		}
		if r && s {
			both++
		}
	}
	if rollback < 4 {
		t.Errorf("%d logs mention a rollback; one grep would shortlist them all", rollback)
	}
	if settlement < 4 {
		t.Errorf("%d logs mention settlement; one grep would shortlist them all", settlement)
	}
	if both != 1 {
		t.Errorf("%d logs mention both, want exactly 1 — the answer has to be unambiguous", both)
	}
}

func TestNeedleHaystackIsBigEnough(t *testing.T) {
	task := mustTask(t, "needle-in-haystack")

	logs, bytes := 0, 0
	for rel, body := range task.Files {
		if strings.HasPrefix(rel, "logs/") {
			logs++
		}
		bytes += len(body)
	}
	if logs < 40 {
		t.Errorf("%d log files; the task is meant to be too big to read", logs)
	}
	if logs != NeedleFileCount {
		t.Errorf("%d log files but the prompt says %d", logs, NeedleFileCount)
	}
	if bytes < 20_000 {
		t.Errorf("the haystack is %d bytes; that fits in a transcript comfortably", bytes)
	}
}

func TestNeedleScoresTheAnswer(t *testing.T) {
	hermetic(t)
	task := mustTask(t, "needle-in-haystack")

	nothing := materialize(t, task)
	if total, breakdown := grade(t, task, nothing); total != 0 {
		t.Errorf("doing nothing scored %v: %v", total, breakdown)
	}

	exact := materialize(t, task)
	write(t, exact.Task.Workspace, "ANSWER.txt", NeedleAnswer+"\n")
	if total, breakdown := grade(t, task, exact); !approx(total, 1.0) {
		t.Errorf("the right answer scored %v: %v", total, breakdown)
	}

	// Right answer, wrong format. Worth most of the marks and not all of them:
	// the prompt asked for the id alone.
	wordy := materialize(t, task)
	write(t, wordy.Task.Workspace, "ANSWER.txt", "The deployment was "+NeedleAnswer+".\n")
	total, breakdown := grade(t, task, wordy)
	if breakdown["answer"] != 0.60 || breakdown["answer_exact"] != 0 {
		t.Errorf("a wordy right answer graded oddly: %v", breakdown)
	}
	if !approx(total, 0.70) {
		t.Errorf("a wordy right answer scored %v, want 0.70", total)
	}

	wrong := materialize(t, task)
	write(t, wrong.Task.Workspace, "ANSWER.txt", "deploy-2c19ee0\n")
	if total, breakdown := grade(t, task, wrong); !approx(total, 0.10) {
		t.Errorf("a plausible wrong answer scored %v: %v", total, breakdown)
	}
}

// ===== GoProbe itself =====

// TestGoProbeCleansUpAndCannotBeSpoofed covers the three ways a probe could
// quietly stop being a grader.
func TestGoProbeCleansUpAndCannotBeSpoofed(t *testing.T) {
	hermetic(t)

	const mod = "module probe\n\ngo 1.25\n"
	const src = "package probe\n\n// F is the thing under test.\nfunc F() int { return 1 }\n"
	const body = "package probe\n\nimport \"testing\"\n\nfunc TestWombatProbeF(t *testing.T) {\n" +
		"\tif F() != 1 {\n\t\tt.Fatal(\"F\")\n\t}\n}\n"

	newDir := func() *rl.Episode {
		dir := t.TempDir()
		write(t, dir, "go.mod", mod)
		write(t, dir, "probe.go", src)
		return &rl.Episode{Task: rl.Task{ID: "probe", Workspace: dir}}
	}

	v := GoProbe("probe", "zz_probe_test.go", body, "TestWombatProbeF", 0.5)

	ep := newDir()
	if _, got := v(context.Background(), ep); got != 0.5 {
		t.Errorf("a passing probe scored %v, want 0.5", got)
	}
	if _, err := os.Stat(filepath.Join(ep.Task.Workspace, "zz_probe_test.go")); !os.IsNotExist(err) {
		t.Error("the probe file survived scoring; a kept workspace would show a test the agent never wrote")
	}

	// An empty module. `go test ./...` would exit 0 here; the probe must not.
	empty := &rl.Episode{Task: rl.Task{ID: "probe", Workspace: t.TempDir()}}
	write(t, empty.Task.Workspace, "go.mod", mod)
	if _, got := v(context.Background(), empty); got != 0 {
		t.Errorf("a probe against an empty module scored %v, want 0", got)
	}

	// A -run pattern that matches nothing also exits 0. The "--- PASS" check
	// is the only thing between that and a free score.
	missing := GoProbe("probe", "zz_probe_test.go", body, "TestNoSuchTest", 0.5)
	if _, got := missing(context.Background(), newDir()); got != 0 {
		t.Errorf("a probe whose test never ran scored %v, want 0", got)
	}

	// An existing file at the probe's path is the agent's, not ours.
	occupied := newDir()
	write(t, occupied.Task.Workspace, "zz_probe_test.go", "package probe\n")
	if _, got := v(context.Background(), occupied); got != 0 {
		t.Errorf("the probe overwrote a file the agent had put there and scored %v", got)
	}
	if read(t, filepath.Join(occupied.Task.Workspace, "zz_probe_test.go")) != "package probe\n" {
		t.Error("the probe clobbered the agent's file")
	}
}

// TestProbesAreGofmtClean extends the fixture formatting rule to the probes.
//
// TestFixturesAreGofmtClean cannot see these: a probe is deliberately not in
// task.Files. They still land in the workspace and still get compiled, and a
// probe that does not parse fails every episode of its task with a compile
// error that looks like the agent's fault.
func TestProbesAreGofmtClean(t *testing.T) {
	if _, err := exec.LookPath("gofmt"); err != nil {
		t.Skip("gofmt not on PATH")
	}
	for name, body := range map[string]string{
		"refactorProbe":            refactorProbe,
		"crossFileProbe":           crossFileProbe,
		"ambiguousProbeLastWins":   ambiguousProbeLastWins,
		"ambiguousProbeDecided":    ambiguousProbeDecided,
		"crossFileRealFix (index)": strings.Replace(crossFileIndex, "func normalizeKey(key string) string {\n\treturn strings.ToLower(strings.TrimSpace(key))\n}\n", crossFileRealFix, 1),
	} {
		cmd := exec.Command("gofmt", "-l")
		cmd.Stdin = strings.NewReader(body)
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("%s: gofmt: %v", name, err)
		}
		if len(out) > 0 {
			t.Errorf("%s is not gofmt-clean", name)
		}
	}
}

// ===== shared =====

// fixtureBodies returns every file in a fixture, for a scan.
func fixtureBodies(task Task) []string {
	out := make([]string, 0, len(task.Files))
	for _, body := range task.Files {
		out = append(out, body)
	}
	return out
}

// TestGoProbeLeavesTheWorkspaceAlone locks the answer-key leak shut.
//
// The probe file says what the correct answer is. Samples of one task run
// concurrently in sibling directories and an agent WILL go looking in them —
// in the run that motivated this, one spent nine consecutive calls trying, and
// reached for another sample's check file by name. Grading in place put the
// answer in a readable location for the length of a `go test`.
func TestGoProbeLeavesTheWorkspaceAlone(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "go.mod", "module probe\n\ngo 1.25\n")
	write(t, dir, "probe.go", "package probe\n\nfunc F() int { return 7 }\n")
	ep := &rl.Episode{Task: rl.Task{ID: "probe", Workspace: dir}}

	const body = `package probe

import "testing"

func TestWombatProbeF(t *testing.T) {
	if F() != 7 {
		t.Fatal("no")
	}
}
`
	before := entries(t, dir)
	if _, got := GoProbe("probe", "zz_probe_test.go", body, "TestWombatProbeF", 0.5)(context.Background(), ep); got != 0.5 {
		t.Fatalf("score = %v, want 0.5 — the probe must still grade", got)
	}
	if after := entries(t, dir); !slices.Equal(before, after) {
		t.Errorf("workspace changed during grading:\n before %v\n after  %v", before, after)
	}
}

// entries lists dir's immediate children, sorted.
func entries(t *testing.T, dir string) []string {
	t.Helper()
	des, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]string, len(des))
	for i, de := range des {
		out[i] = de.Name()
	}
	slices.Sort(out)
	return out
}

// TestFixtureIsAGitRepo: the agent is handed git_log, git_diff and git_status,
// so the workspace has to be a repository or all three fail on contact.
func TestFixtureIsAGitRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed; the fixture degrades to a plain directory by design")
	}
	task := fixBug()
	env := task.Env(t.TempDir())
	got, err := env.Reset(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command("git", "log", "--oneline")
	cmd.Dir = got.Workspace
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git log in the fixture: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "fixture") {
		t.Errorf("git log = %q, want the fixture commit", out)
	}

	// And nothing uncommitted, so the agent's first `git diff` shows its own
	// work rather than the fixture's arrival.
	cmd = exec.Command("git", "status", "--porcelain")
	cmd.Dir = got.Workspace
	if out, err := cmd.CombinedOutput(); err != nil || len(out) != 0 {
		t.Errorf("git status = %q (err %v), want a clean tree", out, err)
	}
}
