package judge_test

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/eval/judge"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/openai"
	"github.com/hoobean1996/wombat-go/rl"
)

// Live calibration against a real model.
//
// Skipped without WOMBAT_JUDGE_LIVE=1, so the default suite stays keyless and
// deterministic; this lane costs money and needs a network.
//
// It runs both judges over one transcript and reports where they disagree.
// That number is the only cheap evidence there is that a model judge has
// learned the anchors rather than pattern-matched the prose, and a model judge
// shipped without it is an opinion with a JSON schema.
func TestLiveJudgeAgainstRules(t *testing.T) {
	if os.Getenv("WOMBAT_JUDGE_LIVE") != "1" {
		t.Skip("set WOMBAT_JUDGE_LIVE=1 to run the live calibration lane")
	}
	path := os.Getenv("WOMBAT_JUDGE_SESSION")
	if path == "" {
		t.Skip("set WOMBAT_JUDGE_SESSION to a session transcript")
	}

	ep := loadEpisode(t, path)
	t.Logf("episode: %d messages, %d turns", len(ep.Messages), ep.Turns())

	client, err := openai.New(openai.ConfigFromEnv())
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	model := judge.LLM{Client: client, MaxTokens: 800, Retries: 2, Label: "deepseek"}

	rulesOnly, err := judge.Evaluate(context.Background(), ep,
		judge.Config{Panel: judge.Panel{judge.Rules{}}})
	if err != nil {
		t.Fatalf("rules: %v", err)
	}
	modelOnly, err := judge.Evaluate(context.Background(), ep,
		judge.Config{Panel: judge.Panel{model}})
	if err != nil {
		t.Fatalf("model: %v", err)
	}

	t.Log("\n--- rules ---")
	logReport(t, rulesOnly)
	t.Log("\n--- model ---")
	logReport(t, modelOnly)

	t.Log("\n--- agreement on the criteria both judged ---")
	exact, compared := 0, 0
	for _, c := range judge.Rulable {
		ag := judge.Compare(c, judge.Corpus{rulesOnly}, judge.Corpus{modelOnly})
		if ag.N == 0 {
			continue
		}
		compared++
		ra, _ := rulesOnly.Of(c)
		ma, _ := modelOnly.Of(c)
		mark := "agree"
		if ra.Score != ma.Score {
			mark = "DISAGREE"
		} else {
			exact++
		}
		t.Logf("  %-18s %-9s rules=%d model=%d (delta %.0f)", c, mark, ra.Score, ma.Score, ag.MeanAbsDelta)
	}
	if compared > 0 {
		t.Logf("  exact agreement on %d of %d rulable criteria", exact, compared)
	}

	// The two the rules refuse and the model is FOR. Correctness is not on this
	// list any more: it belongs to the oracle, and a session transcript carries
	// no verifier scores, so it stays visibly unjudged rather than guessed at.
	for _, c := range []judge.Criterion{judge.ReasoningQuality, judge.UserAlignment} {
		a, ok := modelOnly.Of(c)
		if !ok {
			t.Errorf("  %s: the model judge produced no verdict, which is what it exists for", c)
			continue
		}
		t.Logf("  %-18s model scored %d (%s) conf=%.2f — %s",
			c, a.Score, passWord(a.Pass), a.Confidence, a.Rationale)
	}
	if _, ok := modelOnly.Of(judge.Correctness); ok {
		t.Error("  correctness: the model judged it; that is the oracle's job")
	}
	if !slices.Contains(modelOnly.Unjudged, judge.Correctness) {
		t.Errorf("  correctness is neither judged nor reported unjudged: %v", modelOnly.Unjudged)
	}
}

func passWord(b bool) string {
	if b {
		return "pass"
	}
	return "FAIL"
}

func logReport(t *testing.T, r *judge.Report) {
	t.Helper()
	for _, a := range r.Assessments {
		line := "  " + string(a.Criterion)
		for len(line) < 22 {
			line += " "
		}
		t.Logf("%s %d  %-4s conf=%.2f  %s", line, a.Score, passWord(a.Pass), a.Confidence, a.Failure)
		for _, e := range a.Evidence {
			t.Logf("      · step %d: %s", e.Step, e.Note)
		}
	}
	t.Logf("  overall %.2f  pass=%v  unjudged=%v", r.Overall, r.Pass, r.Unjudged)
	t.Logf("  diagnosis: %s", r.Diagnosis.Summary)
}

// loadEpisode rebuilds an Episode from a saved transcript.
//
// A transcript is less than an Episode — it has no steps, so it cannot say
// whether the permission gate refused a call — and the judges degrade
// accordingly rather than inventing the missing half.
func loadEpisode(t *testing.T, path string) *rl.Episode {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var msgs []llm.Message
	if err := json.Unmarshal(raw, &msgs); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	if len(msgs) == 0 {
		t.Fatalf("%s holds no messages", path)
	}

	ep := &rl.Episode{Messages: msgs, Failure: rl.Success}
	ep.Task = rl.Task{ID: "session", Prompt: strings.TrimSpace(llm.TextOf(msgs[0].Content))}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != llm.RoleAssistant {
			continue
		}
		if txt := strings.TrimSpace(llm.TextOf(msgs[i].Content)); txt != "" {
			ep.Outcome = wombat.Answer{Text: txt}
			break
		}
	}
	for i, m := range msgs {
		if m.Role != llm.RoleAssistant {
			continue
		}
		step := rl.Step{Iteration: len(ep.Steps) + 1}
		for _, b := range m.Content {
			if tu, ok := b.(llm.ToolUse); ok {
				step.Tools = append(step.Tools, tu.Name)
			}
		}
		if len(step.Tools) > 0 {
			ep.Steps = append(ep.Steps, step)
		}
		_ = i
	}
	return ep
}
