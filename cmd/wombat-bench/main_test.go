package main

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/benchmarks"
)

// quiet restores the global state run() reaches for, so one test cannot leak
// into the next.
func quiet(t *testing.T) {
	t.Helper()
	prev := slog.Default()
	t.Cleanup(func() { slog.SetDefault(prev) })

	// run calls benchmarks.ApplyGoEnv, which sets these on the process.
	// Registering them with t.Setenv first means the test framework puts them
	// back.
	for _, k := range []string{"GOPROXY", "GOFLAGS", "GOWORK", "GOTOOLCHAIN"} {
		t.Setenv(k, os.Getenv(k))
	}
}

// TestEndToEnd drives the whole command against the fake provider: two tasks,
// two samples each, one sample of each scripted to succeed and one to fail.
//
// The asymmetry is the point. A suite where every sample passes cannot tell a
// correct pass@k from a constant 1.0, and one where every sample fails cannot
// tell it from a constant 0. With c=1 of n=2 the estimator has to produce
// pass@1 = 0.5 and pass@2 = 1.0, which is a number no bug returns by accident.
func TestEndToEnd(t *testing.T) {
	quiet(t)
	newFake(t, script)

	out := t.TempDir()
	var stdout, stderr bytes.Buffer

	code := run([]string{
		"-n", "2",
		"-c", "2",
		"-tasks", "read-only-qa,fix-bug",
		"-out", out,
		"-temp", "1.0",
		"-max-iters", "8",
		"-budget", "0",
		"-log", "error",
	}, &stdout, &stderr)

	if code != exitOK {
		t.Fatalf("exit %d\nstdout:\n%s\nstderr:\n%s", code, stdout.String(), stderr.String())
	}
	t.Logf("stdout:\n%s", stdout.String())

	report := stdout.String()
	for _, want := range []string{
		"read-only-qa", "fix-bug",
		"PASS@1", "PASS@K",
		"failures (4 episodes)",
	} {
		if !strings.Contains(report, want) {
			t.Errorf("the report does not mention %q:\n%s", want, report)
		}
	}

	// One success and one failure per task, so pass@1 = 0.5 and pass@2 = 1.0
	// from the unbiased estimator: 1 - C(1,1)/C(2,1) = 0.5, 1 - C(1,2)/C(2,2)
	// = 1 - 0 = 1.
	if !strings.Contains(report, "0.500") || !strings.Contains(report, "1.000") {
		t.Errorf("expected pass@1 0.500 and pass@2 1.000 in:\n%s", report)
	}

	// Both artifacts, and both written under -out rather than wherever the
	// test happened to be run from.
	text, err := os.ReadFile(filepath.Join(out, "report.txt"))
	if err != nil {
		t.Fatalf("report.txt: %v", err)
	}
	if !strings.Contains(string(text), "PASS@1") {
		t.Errorf("report.txt is not the report:\n%s", text)
	}

	lines := jsonl(t, filepath.Join(out, "episodes.jsonl"))
	if len(lines) != 4 {
		t.Fatalf("episodes.jsonl has %d lines, want 4", len(lines))
	}

	byLabel := map[string]episodeLine{}
	for _, l := range lines {
		byLabel[l.Task+"#"+strconv.Itoa(l.Sample)] = l
	}
	for label, wantSuccess := range map[string]bool{
		"read-only-qa#0": true,
		"read-only-qa#1": false,
		"fix-bug#0":      true,
		"fix-bug#1":      false,
	} {
		got, ok := byLabel[label]
		if !ok {
			t.Fatalf("no episode %s in the JSONL", label)
		}
		if got.Success != wantSuccess {
			t.Errorf("%s: success = %v, want %v (reward %v, %s, %v)",
				label, got.Success, wantSuccess, got.Reward, got.Failure, got.Breakdown)
		}
		if len(got.Messages) == 0 {
			t.Errorf("%s: no transcript in the JSONL, which is the point of the file", label)
		}
		if got.Turns == 0 || got.ToolCalls == 0 {
			t.Errorf("%s: turns=%d tools=%d, so the step log was not reconstructed",
				label, got.Turns, got.ToolCalls)
		}
		if got.CostUSD == 0 {
			t.Errorf("%s: cost is zero, so token accounting did not reach the episode", label)
		}
		// The fake answers as a model DefaultPricing knows, so that cost is a
		// real number and the line has to say so.
		if !got.Priced {
			t.Errorf("%s: a priced model was recorded as unpriced (%v)", label, got.Unpriced)
		}
	}

	// The failing sample of fix-bug must fail for the RIGHT reason: it applied
	// a wrong fix, so the test verifier scores 0 while the checksum still
	// scores — an agent that had cheated by editing the test would show the
	// opposite.
	bad := byLabel["fix-bug#1"]
	if bad.Breakdown["test"] != 0 {
		t.Errorf("fix-bug#1 was supposed to leave the test failing: %v", bad.Breakdown)
	}
	if bad.Breakdown["test_untouched"] == 0 {
		t.Errorf("fix-bug#1 was not supposed to touch the test file: %v", bad.Breakdown)
	}
	if bad.Failure != "verifier" {
		t.Errorf("fix-bug#1 failure = %q, want verifier: it ran cleanly and produced the wrong thing", bad.Failure)
	}
}

// TestTemperatureReachesTheProvider pins the -temp flag all the way to the
// wire.
//
// It is worth its own test because the path is not obvious: the agent
// materializes its own llm.Request and never sets Temperature, so the flag can
// only arrive through middleware, and a refactor that reorders the client chain
// would drop it silently. A benchmark sampling at the provider's default
// temperature reports a pass@k for an experiment nobody ran.
func TestTemperatureReachesTheProvider(t *testing.T) {
	quiet(t)

	f := newFake(t, func(string, int) reply { return answer("done") })

	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := run([]string{
		"-n", "1", "-c", "1", "-tasks", "read-only-qa",
		"-out", out, "-temp", "0.7", "-max-iters", "2", "-budget", "0", "-log", "error",
	}, &stdout, &stderr)
	if code != exitOK {
		t.Fatalf("exit %d\n%s\n%s", code, stdout.String(), stderr.String())
	}
	temps := f.Temperatures()
	if len(temps) == 0 {
		t.Fatal("no request carried a temperature; -temp never reached the wire")
	}
	for _, v := range temps {
		if v != 0.7 {
			t.Errorf("temperature = %v, want 0.7", v)
		}
	}
}

// TestWholeSuiteRunsWithAnAgentThatDoesNothing walks all eight tasks through
// the real command with a model that answers immediately and touches nothing.
//
// It is a floor test, and the floor is where benchmarks break. Every task must
// reset, run, score and clean up without erroring, and no task may PASS.
//
// Not "must score zero", which would be the wrong assertion for two of them:
// fix-bug and add-feature start from a working fixture, and their preservation
// verifiers — the checksum, the pre-existing tests, the unchanged default
// output — are supposed to score on an untouched workspace. They are regression
// guards, not achievements. What must be zero is the achievement verifier of
// each task, listed below, and todo-app had exactly such a hole once:
// `go build ./...` in a module with no Go files exits 0.
func TestWholeSuiteRunsWithAnAgentThatDoesNothing(t *testing.T) {
	quiet(t)
	newFake(t, func(string, int) reply { return answer("I did nothing.") })

	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := run([]string{
		"-n", "1", "-c", "4", "-out", out, "-max-iters", "2", "-budget", "0", "-log", "error",
	}, &stdout, &stderr)
	if code != exitOK {
		t.Fatalf("exit %d\n%s\n%s", code, stdout.String(), stderr.String())
	}

	lines := jsonl(t, filepath.Join(out, "episodes.jsonl"))
	if want := len(benchmarks.All()); len(lines) != want {
		t.Fatalf("%d episodes, want one per task (%d)", len(lines), want)
	}
	// The verifier that can only score if the agent actually did the task.
	mustBeZero := map[string][]string{
		"read-only-qa": {"answer_file", "answer"},
		"fix-bug":      {"test"},
		"add-feature":  {"top_flag"},
		"todo-app":     {"vet", "build", "test", "has_test_file", "add_then_list"},

		// The hard tier. Each one names the verifier that is the task, not the
		// preservation checks around it: refactor-interface and cross-file-bug
		// both ship a fixture that builds, and both are SUPPOSED to score
		// their compile and checksum terms on an untouched workspace.
		"refactor-interface": {"new_method"},
		"cross-file-bug":     {"test", "index_normalises"},
		"ambiguous-spec":     {"existing_tests", "collision_behaviour", "judgement"},
		"needle-in-haystack": {"answer_file", "answer", "answer_exact"},
	}
	seen := map[string]bool{}
	for _, l := range lines {
		seen[l.Task] = true
		if l.Success || l.Reward >= 1.0 {
			t.Errorf("%s passed without the agent doing anything: reward %v, %v",
				l.Task, l.Reward, l.Breakdown)
		}
		if l.Failure != "verifier" {
			t.Errorf("%s failed as %q; a clean run that produced nothing is a verifier failure",
				l.Task, l.Failure)
		}
		for _, name := range mustBeZero[l.Task] {
			if l.Breakdown[name] != 0 {
				t.Errorf("%s: %s scored %v for free", l.Task, name, l.Breakdown[name])
			}
		}
	}
	for id := range mustBeZero {
		if !seen[id] {
			t.Errorf("no episode for %s; the default -tasks is not the whole suite", id)
		}
	}
}

// TestUnpricedModelIsReportedAsSuch is the bug, end to end and through the
// files a person actually reads: a gateway model nobody has a rate for spends
// real tokens, DefaultPricing charges it nothing, and neither report.txt nor
// the JSONL is allowed to present that nothing as a cost.
//
// No rate is invented for it. An estimate that looks like a measurement is the
// failure being fixed, so the honest output is "n/a" plus the model's name.
func TestUnpricedModelIsReportedAsSuch(t *testing.T) {
	quiet(t)
	const gateway = "some-gateway-model"
	f := newFake(t, func(string, int) reply { return answer("I did nothing.") })
	f.answerAs(gateway)

	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := run([]string{
		"-n", "1", "-c", "1", "-tasks", "read-only-qa", "-out", out,
		"-model", gateway, "-max-iters", "2", "-budget", "0", "-log", "error",
	}, &stdout, &stderr)
	if code != exitOK {
		t.Fatalf("exit %d\n%s\n%s", code, stdout.String(), stderr.String())
	}

	report, err := os.ReadFile(filepath.Join(out, "report.txt"))
	if err != nil {
		t.Fatalf("report.txt: %v", err)
	}
	if strings.Contains(string(report), "$0.0000") {
		t.Errorf("report.txt prices an unpriced run at zero:\n%s", report)
	}
	if !strings.Contains(string(report), "unpriced: "+gateway) {
		t.Errorf("report.txt does not name the model it could not price:\n%s", report)
	}
	// Tokens are what is left to compare runs by, so they have to be on the
	// page whatever the cost column says.
	if !strings.Contains(string(report), "TOK/IN") {
		t.Errorf("report.txt has no token column:\n%s", report)
	}
	// The live progress line is what an operator watches for ten minutes.
	if !strings.Contains(stdout.String(), "n/a") {
		t.Errorf("the progress line still shows a dollar figure:\n%s", stdout.String())
	}

	lines := jsonl(t, filepath.Join(out, "episodes.jsonl"))
	if len(lines) != 1 {
		t.Fatalf("%d episodes, want 1", len(lines))
	}
	// The raw zero stays — a consumer summing cost_usd keeps working — and the
	// flag beside it says the zero is not a measurement.
	if lines[0].CostUSD != 0 {
		t.Errorf("cost_usd = %v, want 0", lines[0].CostUSD)
	}
	if lines[0].Priced {
		t.Error("the JSONL claims an unpriced episode was priced")
	}
	if want := []string{gateway}; !slices.Equal(lines[0].Unpriced, want) {
		t.Errorf("unpriced_models = %v, want %v", lines[0].Unpriced, want)
	}
}

func TestListAndUsage(t *testing.T) {
	quiet(t)

	var stdout, stderr bytes.Buffer
	if code := run([]string{"-list"}, &stdout, &stderr); code != exitOK {
		t.Fatalf("-list exit %d: %s", code, stderr.String())
	}
	for _, task := range benchmarks.All() {
		if !strings.Contains(stdout.String(), task.ID) {
			t.Errorf("-list did not mention %q:\n%s", task.ID, stdout.String())
		}
	}
	for _, tier := range benchmarks.TierNames() {
		if tier == "all" {
			continue // not a heading; it is the default
		}
		if !strings.Contains(stdout.String(), tier) {
			t.Errorf("-list did not name the %q tier, so -tasks %s is undiscoverable:\n%s",
				tier, tier, stdout.String())
		}
	}

	stdout.Reset()
	stderr.Reset()
	newFake(t, script)
	if code := run([]string{"-tasks", "no-such-task", "-out", t.TempDir()}, &stdout, &stderr); code != exitUsage {
		t.Errorf("an unknown -tasks id exited %d, want %d", code, exitUsage)
	}

	stdout.Reset()
	stderr.Reset()
	if code := run([]string{"-n", "0", "-out", t.TempDir()}, &stdout, &stderr); code != exitUsage {
		t.Errorf("-n 0 exited %d, want %d", code, exitUsage)
	}
}

// ===== the scripted trajectories =====

// script is the canned agent: it solves read-only-qa and fix-bug on sample 0
// and gets each of them wrong, in a different and realistic way, on sample 1.
//
// The two failures are chosen to exercise different halves of the scoring.
// read-only-qa#1 writes a confidently wrong answer, which is the FileContains
// verifier's whole job. fix-bug#1 applies a plausible but incorrect edit — a
// bug fix that is off by one in the other direction — which leaves the checksum
// happy and the test failing, and is the shape of a real bad patch.
func script(workspace string, turn int) reply {
	sample := sampleOf(workspace)

	switch {
	case strings.Contains(workspace, "read-only-qa"):
		switch turn {
		case 0:
			return call("view_file", map[string]any{"path": workspace + "/docs/ports.md"})
		case 1:
			return call("view_file", map[string]any{"path": workspace + "/docs/services.md"})
		case 2:
			body := "ledger-api\n"
			if sample != 0 {
				body = "web-frontend\n"
			}
			return call("write_file", map[string]any{
				"path":    workspace + "/ANSWER.txt",
				"content": body,
			})
		default:
			return answer("Port 8081 belongs to codename basalt, which is the ledger-api service.")
		}

	case strings.Contains(workspace, "fix-bug"):
		switch turn {
		case 0:
			return call("bash", map[string]any{
				"command":  "go test ./... 2>&1 | head -20",
				"exec_dir": workspace,
			})
		case 1:
			fixed := "i < len(xs)"
			if sample != 0 {
				// Plausible and wrong: still misses the last window.
				fixed = "i <= len(xs)-2"
			}
			return call("str_replace", map[string]any{
				"path":    workspace + "/window.go",
				"old_str": "i < len(xs)-1",
				"new_str": fixed,
			})
		case 2:
			return call("bash", map[string]any{
				"command":  "go test ./... 2>&1 | tail -5",
				"exec_dir": workspace,
			})
		default:
			return answer("The window loop stopped one element early, so the final window was never scored.")
		}
	}
	return answer("I do not know how to do this task.")
}

// ===== helpers =====

// episodeLine is the subset of the JSONL this test reads back. Decoding into a
// narrow struct rather than a map keeps the assertions readable and still fails
// if a field is renamed.
type episodeLine struct {
	Task      string             `json:"task"`
	Sample    int                `json:"sample"`
	Reward    float64            `json:"reward"`
	Breakdown map[string]float64 `json:"breakdown"`
	Failure   string             `json:"failure"`
	Success   bool               `json:"success"`
	Turns     int                `json:"turns"`
	ToolCalls int                `json:"tool_calls"`
	CostUSD   float64            `json:"cost_usd"`
	Priced    bool               `json:"priced"`
	Unpriced  []string           `json:"unpriced_models"`
	Messages  []json.RawMessage  `json:"messages"`
}

func jsonl(t *testing.T, path string) []episodeLine {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("episodes.jsonl: %v", err)
	}
	var out []episodeLine
	for line := range strings.SplitSeq(strings.TrimSpace(string(b)), "\n") {
		if line == "" {
			continue
		}
		var e episodeLine
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("decoding a JSONL line: %v\n%s", err, line)
		}
		out = append(out, e)
	}
	return out
}
