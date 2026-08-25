// Package benchmarks is the wombat benchmark suite: eight coding tasks defined
// as data, all programmatically verifiable and all fast.
//
// A task is a [Task] value — an id, a prompt, a fixture and a list of
// verifiers — and nothing else. There is no per-task code path, no interface to
// implement and no registration hook, because the moment a benchmark suite
// grows those, adding a task means reading the harness first and the suite
// stops growing.
//
//	for _, t := range benchmarks.All() {
//	    g, err := rl.Rollout(ctx, mk, t.Env(root), 8)
//	}
//
// # Two tiers
//
// [Easy] is the original four: single file, standard library, unambiguous
// spec. A live model scored 4/4 on every one of them, which is why they are
// still here. They are the FLOOR — the thing you look at to find out whether
// the harness itself is working — and a suite with no tier that is supposed to
// pass cannot tell "the tasks are hard" from "the fixtures never got written".
//
// [Hard] is the four that discriminate, and each one breaks a different
// assumption the easy tier lets an agent get away with:
//
//   - refactor-interface — the files that have to change are not named in the
//     prompt, and one of them is a _test.go file.
//   - cross-file-bug — the failing test and the broken function are in
//     different files, and the cheap fix at the call site is priced.
//   - ambiguous-spec — the prompt genuinely underspecifies something, an
//     ask_user tool is available, and most of the score is what the agent did
//     about that rather than what it wrote.
//   - needle-in-haystack — 44 files, one fact, and a transcript that will not
//     hold them all.
//
// # Grading against bytes the agent never saw
//
// The easy tier grades with [rl.Shell] over the fixture's own tests plus
// [Unchanged] checksums. That stops an agent editing the specification, but on
// a refactor the agent legitimately has to edit the tests, and then a checksum
// is not available. [GoProbe] is the answer: a test written into the workspace
// AFTER the agent has stopped, run once, and deleted. It cannot be gutted,
// because it was never there to read.
//
// # Hermeticity
//
// Every fixture is standard-library-only Go with its own go.mod and no
// requires, so nothing a verifier runs can reach a module proxy. [ApplyGoEnv]
// nails that shut with GOPROXY=off, GOWORK=off and GOTOOLCHAIN=local, applied
// to the benchmark PROCESS so the agent's own bash tool inherits it too — a
// fixture that is hermetic only for the verifier is not hermetic, because the
// agent is the one running `go get`.
//
// What is NOT hermetic, stated plainly rather than left to be discovered:
//
//   - The Go toolchain itself. Six of the eight tasks shell out to `go build`,
//     `go test` or `go run`, so a machine without Go scores every one of them
//     zero. read-only-qa and needle-in-haystack are the two that do not, which
//     makes them the pair to run when you are testing the harness.
//   - The build cache. GOCACHE is shared with the host, deliberately: a cold
//     cache turns a 2-second verifier into a 40-second one, and the tasks are
//     supposed to be fast enough to run eight samples of each.
//   - The model provider. That is the point of the benchmark, not a leak.
//   - /bin/sh and the POSIX text utilities the shell verifiers use.
//   - The workspace boundary, for bash. The file tools are confined to the
//     sample's directory; [builtin.Bash] is not confined by anything, so an
//     agent can read a sibling sample's workspace and does — two of four
//     samples in one run went looking, and both got there through the shell
//     after the file tools refused them. Two consequences are closed by
//     construction rather than by hoping: the results directory is a SIBLING of
//     the workspace root and never an ancestor of it, so episodes.jsonl is not
//     reachable by climbing; and [GoProbe] grades in a private copy, so the
//     answer key never exists inside a workspace at all. What remains reachable
//     is another sample's copy of the same fixture, which is the same bytes the
//     reader already has. Closing even that needs an OS-level sandbox behind
//     [builtin.Runner]; see [builtin.Bash].
package benchmarks

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/hoobean1996/wombat-go/rl"
	"github.com/hoobean1996/wombat-go/tool"
)

// Task is one benchmark, as data.
type Task struct {
	// ID names the task on the command line and in the report.
	ID string

	// Summary is a one-line description for the task listing.
	Summary string

	// Prompt is what the agent is asked to do. It is deliberately specific
	// about names and output formats: a verifier can only check what the
	// prompt actually asked for, and grading an agent on a requirement nobody
	// stated measures the prompt, not the agent.
	Prompt string

	// Files is the fixture, keyed by workspace-relative path. Written before
	// the agent starts, identically for every sample.
	Files map[string]string

	// Verifiers grade the finished episode. See [Task.Env] for how they are
	// combined.
	Verifiers []rl.Verifier

	// Tools are extra tools this task's agent needs, on top of the file and
	// shell builtins the runner always supplies.
	//
	// Nil for every task but [ambiguousSpec], and it is a task field rather
	// than a runner flag because the tool set is part of the task's
	// definition: "can this agent ask a question" is the thing that task
	// measures, and a runner that forgot to pass the flag would score the
	// judgement axis of a task the agent had no way to pass.
	Tools []tool.Def
}

// Easy returns the four original tasks: one file each, standard library,
// unambiguous spec.
//
// Kept in the suite after every one of them was solved 4/4 by a live model,
// which sounds like a reason to delete them and is the opposite. These are the
// floor. A harness bug — a fixture that does not get written, a workspace
// mounted read-only, a provider returning nonsense — shows up here first and
// unmistakably, and without a tier that is SUPPOSED to be 1.000 you cannot
// tell "the hard tasks are hard" from "the harness is broken".
func Easy() []Task {
	return []Task{
		readOnlyQA(),
		fixBug(),
		addFeature(),
		todoApp(),
	}
}

// Hard returns the four discriminating tasks.
//
// Each one breaks a different assumption the easy tier lets an agent get away
// with: refactor-interface needs files the prompt never names,
// cross-file-bug's symptom is two hops from its cause, ambiguous-spec has no
// right answer to write down, and needle-in-haystack does not fit in the
// context window.
func Hard() []Task {
	return []Task{
		refactorInterface(),
		crossFileBug(),
		ambiguousSpec(),
		needleInHaystack(),
	}
}

// All returns the suite, easy tier first, each tier in increasing order of
// difficulty.
func All() []Task {
	return append(Easy(), Hard()...)
}

// Tiers are the group names [Select] accepts in place of a task id.
//
// A map and not a switch so that [Select]'s error message can list them: an id
// that is neither a task nor a tier is a typo, and a typo you have to go and
// read the source to correct is a bad error message.
func Tiers() map[string][]Task {
	return map[string][]Task{
		"all":  All(),
		"easy": Easy(),
		"hard": Hard(),
	}
}

// IDs returns every task id, for a usage message.
func IDs() []string {
	all := All()
	out := make([]string, len(all))
	for i, t := range all {
		out[i] = t.ID
	}
	return out
}

// Lookup finds one task by id.
func Lookup(id string) (Task, bool) {
	for _, t := range All() {
		if t.ID == id {
			return t, true
		}
	}
	return Task{}, false
}

// Select resolves a list of ids, preserving the caller's order. An empty list
// means the whole suite.
//
// A name from [Tiers] — "easy", "hard", "all" — expands to that tier in
// place, so `-tasks hard` and `-tasks easy,cross-file-bug` both work. Tiers
// are checked before task ids, which costs nothing as long as no task is ever
// named after a tier; [TestTierNamesDoNotCollide] holds that line.
//
// A duplicate is dropped rather than run twice: `-tasks easy,fix-bug` reads as
// "the easy tier, and fix-bug" and running fix-bug twice would put two rows
// with the same task id in a report whose rows are keyed by task id.
//
// An unknown id is an error rather than a warning: a benchmark run that
// silently skips the task you asked about reports a clean sweep over the three
// tasks you did not care about.
func Select(ids []string) ([]Task, error) {
	if len(ids) == 0 {
		return All(), nil
	}
	tiers := Tiers()
	seen := make(map[string]bool, len(ids))
	out := make([]Task, 0, len(ids))
	add := func(t Task) {
		if seen[t.ID] {
			return
		}
		seen[t.ID] = true
		out = append(out, t)
	}

	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if tier, ok := tiers[id]; ok {
			for _, t := range tier {
				add(t)
			}
			continue
		}
		t, ok := Lookup(id)
		if !ok {
			return nil, fmt.Errorf("benchmarks: unknown task %q (have %s; or a tier: %s)",
				id, strings.Join(IDs(), ", "), strings.Join(TierNames(), ", "))
		}
		add(t)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("benchmarks: no tasks selected (have %s)", strings.Join(IDs(), ", "))
	}
	return out, nil
}

// TierNames returns the [Tiers] keys, sorted, for a usage message.
func TierNames() []string {
	return slices.Sorted(maps.Keys(Tiers()))
}

// With returns a copy of t with extra verifiers appended.
//
// A copy and not a mutation: [All] hands out fresh values every call, and a
// caller that appended in place would still be surprising the next one that
// held a slice of them.
func (t Task) With(extra ...rl.Verifier) Task {
	t.Verifiers = append(append([]rl.Verifier(nil), t.Verifiers...), extra...)
	return t
}

// Penalties are the ranking terms: small negative scores for taking too many
// turns, breaking too many tools and spending too much.
//
// Deliberately NOT part of a task's own verifiers. A task's weights sum to 1.0
// and 1.0 means "did the job", which is a yes/no question; a penalty makes the
// total a preference ordering instead, and mixing the two gives you a success
// column that depends on how expensive the run was. Add these when you are
// comparing two agents that both work, and lower the success threshold to
// match — see wombat-bench's -penalize.
//
// The magnitudes are chosen so a well-behaved run loses a few percent and a
// thrashing one loses a tenth: a penalty that can outweigh the task teaches an
// agent that giving up immediately scores better than trying.
func Penalties() []rl.Verifier {
	return []rl.Verifier{
		rl.TurnPenalty(0.005),
		rl.ToolErrorPenalty(0.01),
		rl.CostPenalty(0.10),
	}
}

// Env builds the [rl.Env] for this task, rooted at root.
//
// It is [rl.Dir] — one scratch directory per sample — with the fixture written
// into that directory on every Reset. Writing it in Reset rather than once, up
// front, is what makes the samples independent: sample 3 must start from the
// same bytes as sample 0 however thoroughly sample 0 wrecked its copy.
func (t Task) Env(root string) rl.Env {
	return fixtureEnv{
		Env:   rl.Dir(root, t.ID, t.Prompt, rl.Score(t.Verifiers...)),
		files: t.Files,
	}
}

// fixtureEnv decorates an [rl.Env] with fixture writing.
//
// Embedding the interface rather than reimplementing it: Name, Score and
// Cleanup are exactly Dir's, and a copy of them here would be three more places
// to forget when Dir changes.
type fixtureEnv struct {
	rl.Env
	files map[string]string
}

// Reset implements rl.Env.
func (e fixtureEnv) Reset(ctx context.Context, sample int) (rl.Task, error) {
	t, err := e.Env.Reset(ctx, sample)
	if err != nil {
		return t, err
	}
	if err := writeTree(t.Workspace, e.files); err != nil {
		return t, fmt.Errorf("benchmarks: %s sample %d: writing the fixture: %w", t.ID, sample, err)
	}
	gitInit(ctx, t.Workspace)
	return t, nil
}

// gitInit turns the fixture into a repository with one commit.
//
// Not decoration. The agent is handed git_log, git_diff and git_status because
// a real coding agent has them, and in a directory that is not a repository
// every one of them fails with "fatal: not a git repository" — which costs a
// turn, costs a tool-error penalty, and teaches the model nothing except that
// its tools are unreliable. Both samples of the run that prompted this burned a
// call on exactly that. Committing the fixture makes the tools work: `git diff`
// now shows the agent its own changes, which is the single most useful thing a
// coding agent can ask for and the one thing it could not have here.
//
// Failure is logged and ignored. git is not a build dependency of this suite,
// and a machine without it should run the benchmark with three tools that error
// rather than not run it at all — the same state as before this existed.
//
// Identity and signing are forced on the command line rather than read from the
// environment, because a scratch commit must not depend on, or be blocked by,
// whatever ~/.gitconfig happens to say.
func gitInit(ctx context.Context, dir string) {
	ident := []string{
		"-c", "user.name=wombat-bench",
		"-c", "user.email=bench@wombat.invalid",
		"-c", "commit.gpgsign=false",
	}
	steps := [][]string{
		{"init", "-q", "-b", "main"},
		{"add", "-A"},
		append(append([]string{}, ident...), "commit", "-q", "-m", "fixture"),
	}
	for _, args := range steps {
		cmd := exec.CommandContext(ctx, "git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			slog.WarnContext(ctx, "benchmarks: git fixture setup failed; the git tools will error in this workspace",
				slog.String("dir", dir),
				slog.String("step", args[0]),
				slog.Any("err", err),
				slog.String("output", clip(string(out), rl.VerifierOutputLimit)))
			return
		}
	}
}

// writeTree writes files under dir, creating parent directories.
//
// Deterministic on purpose, right down to the order: the paths are sorted so
// that two runs of the same fixture produce the same sequence of syscalls, and
// a fixture whose content depends on map iteration order cannot sneak in.
func writeTree(dir string, files map[string]string) error {
	for _, rel := range slices.Sorted(maps.Keys(files)) {
		path := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(path, []byte(files[rel]), 0o644); err != nil {
			return err
		}
	}
	return nil
}

// Unchanged scores weight when rel still holds exactly the bytes the fixture
// put there.
//
// This is the interesting verifier in the suite. "Make the test pass" and "make
// the code correct" are the same task only as long as the test is fixed; the
// moment an agent may edit the test, the cheapest way to a green build is to
// delete the assertion, and a scorer that only runs `go test` awards full marks
// for it. Checking the file is byte-identical closes that door — and does it
// without a heuristic, because the fixture's own bytes are the reference.
//
// A checksum rather than a re-read of the constant so the failure message can
// say WHAT changed to what, and so the comparison is constant-time in the file
// size for a fixture that grows.
func Unchanged(name, rel, want string, weight float64) rl.Verifier {
	sum := sha256.Sum256([]byte(want))
	return func(_ context.Context, ep *rl.Episode) (string, float64) {
		got, err := os.ReadFile(filepath.Join(ep.Task.Workspace, filepath.FromSlash(rel)))
		if err != nil {
			return name, 0
		}
		if sha256.Sum256(got) != sum {
			return name, 0
		}
		return name, weight
	}
}

// ProbeTimeout caps a [GoProbe] compile-and-run.
//
// The same budget [rl.DefaultShellTimeout] gives a shell verifier, restated
// here because a probe is not a shell verifier and importing the constant only
// to change it later in one place and not the other is how they drift.
const ProbeTimeout = rl.DefaultShellTimeout

// GoProbe scores weight when a test the fixture never shipped compiles and
// passes against whatever the agent built.
//
// This is the verifier that makes a hard task hard. `go test ./...` grades the
// agent against tests the agent could see, and on a refactor it can see all of
// them — so the cheapest green build is to make the visible tests agree with
// the code rather than the code correct. A probe is written AFTER the agent
// has stopped, from bytes it never had access to.
//
// Three details are load-bearing:
//
//   - Grading happens in a temporary COPY of the workspace. The probe file is
//     the answer key, samples of one task run concurrently in sibling
//     directories, and a probe written in place would be readable by a sample
//     that has not finished yet. The copy also keeps the grader's artefacts out
//     of the workspace that -keep leaves behind.
//
//   - It runs `go test -run '^testName$' -count=1 .`, on the root package and
//     not ./..., because ./... in a module whose packages were all deleted
//     matches nothing and exits 0 — a full score for an empty directory.
//
//   - It requires "--- PASS: testName" in the -v output, not just exit 0,
//     because a -run pattern that matches no test ALSO exits 0. Without this,
//     an agent that renamed the package's identifiers out from under the probe
//     would score full marks for a test that never ran.
//
// An agent that put its own file at the probe's path scores 0 rather than
// having it overwritten in the copy: a name collision means the probe is not
// grading what it thinks it is, and guessing which one to keep is worse than
// saying so.
func GoProbe(name, rel, body, testName string, weight float64) rl.Verifier {
	return func(ctx context.Context, ep *rl.Episode) (string, float64) {
		if ep.Task.Workspace == "" {
			slog.WarnContext(ctx, "benchmarks: probe has no workspace", slog.String("verifier", name))
			return name, 0
		}
		if _, err := os.Stat(filepath.Join(ep.Task.Workspace, filepath.FromSlash(rel))); err == nil {
			slog.WarnContext(ctx, "benchmarks: the agent created a file at the probe's path",
				slog.String("verifier", name), slog.String("path", rel))
			return name, 0
		}

		// The probe is graded in a private copy of the workspace, never in the
		// workspace itself, and the reason is an answer leak with a real
		// window. The probe file IS the answer key — it is the one artefact
		// that says what each stage's Idempotent must return — and samples of
		// the same task run concurrently in sibling directories. Written in
		// place, it would sit in sample-0's directory for as long as `go test`
		// takes while sample-1 is still working, one `cat ../sample-0/*_test.go`
		// away. That is not hypothetical: in the run that prompted this, an
		// agent spent nine consecutive calls trying to read sibling workspaces,
		// and one of the paths it reached for was another sample's check file.
		//
		// Copying also stops grading from mutating the thing being graded,
		// which matters for -keep: the workspace left behind is what the agent
		// actually produced, with nothing of the grader's in it.
		tmp, err := os.MkdirTemp("", "wombat-probe-")
		if err != nil {
			slog.WarnContext(ctx, "benchmarks: creating the probe sandbox",
				slog.String("verifier", name), slog.Any("err", err))
			return name, 0
		}
		defer os.RemoveAll(tmp)

		work := filepath.Join(tmp, "work")
		if err := os.CopyFS(work, os.DirFS(ep.Task.Workspace)); err != nil {
			// os.CopyFS refuses irregular files, so an agent that left a
			// symlink in the tree lands here. Scoring zero is the safe answer
			// and the log says why, rather than the probe silently not running.
			slog.WarnContext(ctx, "benchmarks: copying the workspace for the probe",
				slog.String("verifier", name), slog.Any("err", err))
			return name, 0
		}
		if err := os.WriteFile(filepath.Join(work, filepath.FromSlash(rel)), []byte(body), 0o644); err != nil {
			slog.WarnContext(ctx, "benchmarks: writing the probe",
				slog.String("verifier", name), slog.Any("err", err))
			return name, 0
		}

		ctx, cancel := context.WithTimeout(ctx, ProbeTimeout)
		defer cancel()

		cmd := exec.CommandContext(ctx, "go", "test", "-run", "^"+testName+"$", "-count=1", "-v", ".")
		cmd.Dir = work
		out, err := cmd.CombinedOutput()
		if err == nil && strings.Contains(string(out), "--- PASS: "+testName) {
			return name, weight
		}

		slog.WarnContext(ctx, "benchmarks: probe failed",
			slog.String("verifier", name),
			slog.String("test", testName),
			slog.String("workspace", ep.Task.Workspace),
			slog.Any("err", err),
			slog.String("output", clip(string(out), rl.VerifierOutputLimit)))
		return name, 0
	}
}

// clip bounds a diagnostic, saying how much it dropped so nobody reads the
// tail of a compile log as the end of the failure.
func clip(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return s[:limit] + fmt.Sprintf("\n… %d more bytes", len(s)-limit)
}

// GoEnv is the environment every fixture, every verifier and the agent's own
// shell must run under.
//
// Four variables, each closing one route to the network or to a machine's
// personality:
//
//   - GOPROXY=off — no module downloads, ever. A fixture needs none; an agent
//     that types `go get` should fail loudly here rather than succeed on the
//     maintainer's laptop and fail in CI.
//   - GOFLAGS=-mod=mod — go.mod is writable, so a stray import produces a
//     resolution error against GOPROXY=off instead of the readonly-mode error,
//     which reads like a tooling misconfiguration and sends the agent down a
//     rabbit hole.
//   - GOWORK=off — a go.work file anywhere above the scratch root would pull
//     the fixture module into someone else's workspace.
//   - GOTOOLCHAIN=local — a `go` line the local toolchain does not satisfy
//     otherwise triggers a toolchain DOWNLOAD, which is a network call in the
//     middle of a benchmark that claims not to make any.
func GoEnv() []string {
	return []string{
		"GOPROXY=off",
		"GOFLAGS=-mod=mod",
		"GOWORK=off",
		"GOTOOLCHAIN=local",
	}
}

// ApplyGoEnv sets [GoEnv] on the current process.
//
// The process and not a per-command environment, because the agent reaches the
// shell through builtin.OSRunner, which inherits os.Environ() and offers no
// seam to add to it. Setting it here is the only place that covers BOTH the
// agent's `go test` and the verifier's.
func ApplyGoEnv() error {
	for _, kv := range GoEnv() {
		k, v, _ := strings.Cut(kv, "=")
		if err := os.Setenv(k, v); err != nil {
			return fmt.Errorf("benchmarks: setting %s: %w", k, err)
		}
	}
	return nil
}
