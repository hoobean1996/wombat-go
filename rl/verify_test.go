package rl

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
)

// episodeIn builds an episode rooted at ws with a few plausible steps.
func episodeIn(ws string) *Episode {
	ep := &Episode{
		Task: Task{ID: "t", Sample: 0, Prompt: "go", Workspace: ws},
		Steps: []Step{
			{Iteration: 1, Tools: []string{"read", "write"}, Failed: []string{"write"},
				Denied: []string{"write"}, Usage: llm.Usage{InputTokens: 10}},
			{Iteration: 2, Tools: []string{"bash"}, Failed: []string{"bash"}},
			{Iteration: 3},
		},
	}
	ep.Spend = governor.Progress{CostUSD: 0.25}
	return ep
}

func TestVerifiers(t *testing.T) {
	ws := t.TempDir()
	if err := os.WriteFile(filepath.Join(ws, "main.go"), []byte("package main // hello"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(ws, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	ep := episodeIn(ws)

	tests := []struct {
		name     string
		v        Verifier
		wantName string
		want     float64
	}{
		{"shell exit 0 scores the weight", Shell("ok", "exit 0", 0.4), "ok", 0.4},
		{"shell non-zero scores nothing", Shell("bad", "exit 3", 0.4), "bad", 0},
		{"shell runs in the workspace", Shell("cwd", "test -f main.go", 1), "cwd", 1},
		{"shell gets a real /bin/sh", Shell("pipe", "echo a | grep -q a", 1), "pipe", 1},
		{"file that exists", FileExists("main", "main.go", 0.3), "main", 0.3},
		{"file that does not", FileExists("gone", "nope.go", 0.3), "gone", 0},
		{"a directory counts as existing", FileExists("dir", "sub", 1), "dir", 1},
		{"escaping the workspace scores nothing",
			FileExists("escape", "../../../etc/hosts", 1), "escape", 0},
		{"file with the substring", FileContains("pkg", "main.go", "package main", 0.5), "pkg", 0.5},
		{"file without it", FileContains("nope", "main.go", "package other", 0.5), "nope", 0},
		{"missing file cannot contain it", FileContains("miss", "gone.go", "x", 0.5), "miss", 0},
		{"turn penalty counts steps", TurnPenalty(0.01), "turn_penalty", -0.03},
		{"cost penalty reads the episode's own spend", CostPenalty(2), "cost_penalty", -0.5},
		{"tool error penalty counts refusals too",
			ToolErrorPenalty(0.1), "tool_error_penalty", -0.2},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			name, score := tc.v(t.Context(), ep)
			if name != tc.wantName {
				t.Errorf("name = %q, want %q", name, tc.wantName)
			}
			if math.Abs(score-tc.want) > 1e-12 {
				t.Errorf("score = %v, want %v", score, tc.want)
			}
		})
	}
}

func TestShellWithoutAWorkspaceScoresZero(t *testing.T) {
	name, score := Shell("ok", "exit 0", 1)(t.Context(), &Episode{})
	if name != "ok" || score != 0 {
		t.Fatalf("got %q/%v, want ok/0 — a verifier with nowhere to run must not pass", name, score)
	}
}

func TestScoreSumsAndExplains(t *testing.T) {
	ws := t.TempDir()
	if err := os.WriteFile(filepath.Join(ws, "main.go"), []byte("package main"), 0o600); err != nil {
		t.Fatal(err)
	}
	ep := episodeIn(ws)

	total, breakdown, err := Score(
		FileExists("main", "main.go", 0.6),
		Shell("build", "exit 0", 0.4),
		TurnPenalty(0.01),
	)(t.Context(), ep)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}

	want := map[string]float64{"main": 0.6, "build": 0.4, "turn_penalty": -0.03}
	if !reflect.DeepEqual(breakdown, want) {
		t.Errorf("breakdown = %v, want %v", breakdown, want)
	}
	if math.Abs(total-0.97) > 1e-12 {
		t.Errorf("total = %v, want 0.97", total)
	}
}

func TestScoreSurvivesAPanickingVerifier(t *testing.T) {
	boom := Verifier(func(context.Context, *Episode) (string, float64) {
		panic("verifier is broken")
	})

	total, breakdown, err := Score(
		Verifier(func(context.Context, *Episode) (string, float64) { return "good", 1 }),
		boom,
		Verifier(func(context.Context, *Episode) (string, float64) { return "also_good", 0.5 }),
	)(t.Context(), &Episode{Task: Task{ID: "t"}})

	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	// The panicking one scores zero and the ones after it still run.
	if math.Abs(total-1.5) > 1e-12 {
		t.Fatalf("total = %v, want 1.5 — a broken check must not cost the whole rollout", total)
	}
	if breakdown["good"] != 1 || breakdown["also_good"] != 0.5 {
		t.Fatalf("breakdown = %v, want the surviving verifiers intact", breakdown)
	}
	if _, ok := breakdown["verifier#1"]; !ok {
		t.Errorf("breakdown = %v, want a placeholder key for the panicking verifier", breakdown)
	}
}

func TestScoreDisambiguatesDuplicateNames(t *testing.T) {
	same := func(name string, w float64) Verifier {
		return func(context.Context, *Episode) (string, float64) { return name, w }
	}
	total, breakdown, err := Score(same("build", 1), same("build", 2))(t.Context(), &Episode{})
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if len(breakdown) != 2 {
		t.Fatalf("breakdown = %v, want two keys so it adds up to the total", breakdown)
	}
	sum := 0.0
	for _, v := range breakdown {
		sum += v
	}
	if sum != total || total != 3 {
		t.Fatalf("breakdown sums to %v but total is %v", sum, total)
	}
}

func TestScoreWithNoVerifiers(t *testing.T) {
	total, breakdown, err := Score()(t.Context(), &Episode{})
	if err != nil || total != 0 || len(breakdown) != 0 {
		t.Fatalf("got %v/%v/%v, want 0 and an empty breakdown", total, breakdown, err)
	}
}

// A shell verifier runs code the agent wrote — `go test` over the agent's own
// test files — so it must not see the harness's credentials any more than the
// agent's commands do.
func TestShellVerifierDoesNotSeeTheHarnessCredentials(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-leak")
	t.Setenv("GOFLAGS", "-mod=mod")
	ep := episodeIn(t.TempDir())
	if _, score := Shell("no-key", `test -z "${OPENAI_API_KEY:-}"`, 1)(t.Context(), ep); score != 1 {
		t.Error("the verifier's command could read OPENAI_API_KEY")
	}
	if _, score := Shell("go-env", `test "$GOFLAGS" = "-mod=mod"`, 1)(t.Context(), ep); score != 1 {
		t.Error("the verifier's command lost the Go toolchain's settings")
	}
}
