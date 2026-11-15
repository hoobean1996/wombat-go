package rl

import (
	"bytes"
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
)

func sampleReport() *Report {
	ep := func(id string, n int, reward float64, k FailureKind, cost float64, turns int) *Episode {
		e := &Episode{
			Task:      Task{ID: id, Sample: n, Prompt: "do " + id, Workspace: "/tmp/" + id + "/sample-" + string(rune('0'+n))},
			Reward:    reward,
			Breakdown: map[string]float64{"build": reward},
			Failure:   k,
			Wall:      time.Duration(n+1) * time.Second,
			Outcome:   core.Answer{Text: "answered " + id},
			// A real run against a model the table knows: the cost column is a
			// measurement. The unpriced case is TestReportWriteTextUnpriced.
			Priced: true,
		}
		e.Spend = governor.Progress{CostUSD: cost}
		for i := range turns {
			e.Steps = append(e.Steps, Step{
				Iteration: i + 1,
				Tools:     []string{"read"},
				Usage:     llm.Usage{InputTokens: 100, OutputTokens: 10},
			})
		}
		e.Messages = []llm.Message{llm.UserText("do " + id)}
		return e
	}

	r := &Report{}
	r.Add(&Group{Env: "dir:todo", TaskID: "todo", Episodes: []*Episode{
		ep("todo", 0, 1, Success, 0.10, 4),
		ep("todo", 1, 0, VerifierFailed, 0.20, 6),
		ep("todo", 2, 1, Success, 0.30, 5),
		ep("todo", 3, 0, MaxIterations, 0.40, 30),
	}})
	r.Add(&Group{Env: "dir:fizz", TaskID: "fizzbuzz", Episodes: []*Episode{
		ep("fizzbuzz", 0, 1, Success, 0.01, 2),
		ep("fizzbuzz", 1, 1, Success, 0.02, 3),
	}})
	return r
}

func TestReportWriteTextIsDeterministic(t *testing.T) {
	r := sampleReport()

	var a, b bytes.Buffer
	if err := r.WriteText(&a); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	if err := r.WriteText(&b); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	if a.String() != b.String() {
		t.Fatal("WriteText is not deterministic across calls")
	}

	got := a.String()
	for _, want := range []string{
		"TASK", "PASS@1", "PASS@K", "MEAN", "STD", "TURNS", "TOK/IN", "TOK/OUT", "COST",
		"todo", "fizzbuzz",
		"failures (6 episodes)",
		"success", "verifier", "max_iterations",
		"worst:",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("WriteText output is missing %q:\n%s", want, got)
		}
	}

	// The worst episode is the one a human opens, so its workspace has to be
	// on the page.
	if !strings.Contains(got, "/tmp/todo/sample-1") && !strings.Contains(got, "/tmp/todo/sample-3") {
		t.Errorf("WriteText did not print the worst episode's workspace:\n%s", got)
	}

	// Alignment: every row of the table has the same width.
	lines := strings.Split(strings.TrimRight(got, "\n"), "\n")
	head := lines[0]
	for _, l := range lines[1 : 1+len(r.Groups)] {
		if len(l) != len(head) {
			t.Errorf("table row %q is %d wide, header is %d", l, len(l), len(head))
		}
	}
}

// unpricedEpisode is one episode that burned tokens and reported no cost,
// which is what an unpriced model looks like from here.
func unpricedEpisode(model string) *Episode {
	return &Episode{
		Task:     Task{ID: "gateway", Prompt: "do it"},
		Reward:   1,
		Failure:  Success,
		Steps:    []Step{{Iteration: 1, Usage: llm.Usage{InputTokens: 120_000, OutputTokens: 8_000}}},
		Spend:    governor.Progress{CostUSD: 0},
		Priced:   false,
		Unpriced: []string{model},
	}
}

// TestReportWriteTextPricedShowsDollars is the control for the test below: a
// group whose spend was priced still prints a number.
func TestReportWriteTextPricedShowsDollars(t *testing.T) {
	var buf bytes.Buffer
	if err := sampleReport().WriteText(&buf); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	got := buf.String()
	if !strings.Contains(got, "$0.2500") {
		t.Errorf("priced group did not print its median cost:\n%s", got)
	}
	if strings.Contains(got, "n/a") || strings.Contains(got, "unpriced") {
		t.Errorf("priced report claims something was unpriced:\n%s", got)
	}
}

// TestReportWriteTextUnpriced is the bug: sixteen episodes of a gateway model
// reported $0.0000 and were believed. A zero in a money column is a claim; an
// absence has to look like one.
func TestReportWriteTextUnpriced(t *testing.T) {
	r := &Report{}
	r.Add(&Group{Env: "dir:gateway", TaskID: "gateway", Episodes: []*Episode{
		unpricedEpisode("some-gateway-model"),
		unpricedEpisode("some-gateway-model"),
	}})

	var buf bytes.Buffer
	if err := r.WriteText(&buf); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	got := buf.String()

	if strings.Contains(got, "$0.0000") {
		t.Errorf("an unpriced group still renders a dollar zero:\n%s", got)
	}
	if !strings.Contains(got, "n/a") {
		t.Errorf("COST does not read n/a:\n%s", got)
	}
	// Naming the model is the part that makes the n/a actionable: the fix is a
	// table entry, and the reader cannot make it without the id.
	if !strings.Contains(got, "unpriced: some-gateway-model") {
		t.Errorf("the report does not name the model it could not price:\n%s", got)
	}
	// The fallback metric has to be on the page, or an unpriced run is not
	// comparable to the one before it at all.
	if !strings.Contains(got, "TOK/IN") || !strings.Contains(got, "120.0k") {
		t.Errorf("token columns are missing from an unpriced report:\n%s", got)
	}
}

// TestReportWriteTextUnpricedWithNoModelName covers the honest-proxy case with
// nothing to name: the report still has to say the cost is not a number.
func TestReportWriteTextUnpricedWithNoModelName(t *testing.T) {
	ep := unpricedEpisode("")
	ep.Unpriced = nil

	r := &Report{}
	r.Add(&Group{Env: "e", TaskID: "t", Episodes: []*Episode{ep}})

	var buf bytes.Buffer
	if err := r.WriteText(&buf); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	if got := buf.String(); !strings.Contains(got, "unpriced: unnamed model") {
		t.Errorf("no unpriced line for an episode with no model name:\n%s", got)
	}
}

// TestReportMixedPricingIsUnpricedAsAWhole: one silent zero among three real
// numbers makes the median meaningless, so the column says so.
func TestReportMixedPricingIsUnpricedAsAWhole(t *testing.T) {
	priced := &Episode{
		Task:    Task{ID: "mixed"},
		Failure: Success,
		Steps:   []Step{{Iteration: 1, Usage: llm.Usage{InputTokens: 1000, OutputTokens: 100}}},
		Spend:   governor.Progress{CostUSD: 0.12},
		Priced:  true,
	}
	g := &Group{Env: "e", TaskID: "mixed", Episodes: []*Episode{priced, unpricedEpisode("local-llama")}}
	if g.Priced() {
		t.Fatal("a group with one unpriced episode reports itself priced")
	}
	if got, want := g.UnpricedModels(), []string{"local-llama"}; !slices.Equal(got, want) {
		t.Errorf("UnpricedModels = %v, want %v", got, want)
	}
}

func TestReportUnpricedModelsAreSortedAndDeduplicated(t *testing.T) {
	r := &Report{}
	r.Add(
		&Group{Env: "e", TaskID: "a", Episodes: []*Episode{
			unpricedEpisode("some-gateway-model"), unpricedEpisode("local-llama"),
		}},
		&Group{Env: "e", TaskID: "b", Episodes: []*Episode{
			unpricedEpisode("some-gateway-model"),
		}},
	)
	want := []string{"local-llama", "some-gateway-model"}
	if got := r.UnpricedModels(); !slices.Equal(got, want) {
		t.Errorf("UnpricedModels = %v, want %v — the line gets diffed between runs", got, want)
	}
}

func TestReportWriteTextEmpty(t *testing.T) {
	var buf bytes.Buffer
	if err := (&Report{}).WriteText(&buf); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	if got := buf.String(); got != "no groups\n" {
		t.Fatalf("empty report rendered %q", got)
	}
}

func TestReportWriteJSONL(t *testing.T) {
	r := sampleReport()

	var a, b bytes.Buffer
	if err := r.WriteJSONL(&a); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	if err := r.WriteJSONL(&b); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	if a.String() != b.String() {
		t.Fatal("WriteJSONL is not deterministic across calls")
	}

	lines := strings.Split(strings.TrimRight(a.String(), "\n"), "\n")
	if got, want := len(lines), 6; got != want {
		t.Fatalf("wrote %d lines, want one per episode (%d)", got, want)
	}

	var first map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("line 1 is not JSON: %v", err)
	}
	// The keys a trainer reads without being handed a schema.
	for _, k := range []string{
		"env", "task", "sample", "prompt", "reward", "breakdown", "failure",
		"success", "outcome", "turns", "tool_calls", "wall_ms", "cost_usd",
		"usage", "steps", "messages",
	} {
		if _, ok := first[k]; !ok {
			t.Errorf("episode JSON is missing key %q: %s", k, lines[0])
		}
	}
	if first["success"] != true {
		t.Errorf("sample 0 of todo succeeded but success = %v", first["success"])
	}
	if first["outcome"] != "answer" {
		t.Errorf("outcome = %v, want %q", first["outcome"], "answer")
	}

	// Order is group then sample, so two runs of the same benchmark diff.
	var last map[string]any
	if err := json.Unmarshal([]byte(lines[5]), &last); err != nil {
		t.Fatal(err)
	}
	if last["task"] != "fizzbuzz" || last["sample"] != float64(1) {
		t.Errorf("last line = %v/%v, want fizzbuzz/1", last["task"], last["sample"])
	}
}

// TestReportJSONLCarriesTheZeroAndTheFlag: the raw number stays, so a reader
// that already sums cost_usd keeps working, and the flag sits next to it so
// that reader can tell which of those zeroes are not zeroes.
func TestReportJSONLCarriesTheZeroAndTheFlag(t *testing.T) {
	r := &Report{}
	r.Add(&Group{Env: "dir:gateway", TaskID: "gateway", Episodes: []*Episode{
		unpricedEpisode("some-gateway-model"),
	}})

	var buf bytes.Buffer
	if err := r.WriteJSONL(&buf); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}

	var line map[string]any
	if err := json.Unmarshal([]byte(strings.TrimRight(buf.String(), "\n")), &line); err != nil {
		t.Fatalf("not JSON: %v", err)
	}
	if got, ok := line["cost_usd"].(float64); !ok || got != 0 {
		t.Errorf("cost_usd = %v, want the raw 0 to still be there", line["cost_usd"])
	}
	if line["priced"] != false {
		t.Errorf("priced = %v, want false", line["priced"])
	}
	if got, want := line["unpriced_models"], []any{"some-gateway-model"}; !reflect.DeepEqual(got, want) {
		t.Errorf("unpriced_models = %v, want %v", got, want)
	}
	// The tokens are the number a downstream consumer falls back to.
	usage, _ := line["usage"].(map[string]any)
	if usage["input_tokens"] != float64(120_000) {
		t.Errorf("usage.input_tokens = %v, want 120000", usage["input_tokens"])
	}

	// A priced episode says so, and says it in the same field: a consumer must
	// not have to infer the flag from the presence of another one.
	var priced bytes.Buffer
	if err := sampleReport().WriteJSONL(&priced); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	var first map[string]any
	if err := json.Unmarshal([]byte(strings.SplitN(priced.String(), "\n", 2)[0]), &first); err != nil {
		t.Fatalf("not JSON: %v", err)
	}
	if first["priced"] != true {
		t.Errorf("priced = %v, want true", first["priced"])
	}
	if got, want := first["unpriced_models"], []any{}; !reflect.DeepEqual(got, want) {
		t.Errorf("unpriced_models = %v, want an empty list", got)
	}
}

// TestReportJSONLNeverEmitsNullLists pins the shape a trainer relies on: an
// empty list is [], not null, in every field that is a list.
func TestReportJSONLNeverEmitsNullLists(t *testing.T) {
	r := &Report{}
	r.Add(&Group{Env: "e", TaskID: "t", Episodes: []*Episode{{
		Task:    Task{ID: "t"},
		Steps:   []Step{{Iteration: 1}},
		Failure: VerifierFailed,
	}}})

	var buf bytes.Buffer
	if err := r.WriteJSONL(&buf); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	if strings.Contains(buf.String(), "null") {
		t.Fatalf("JSONL contains a null list: %s", buf.String())
	}
}

// TestReportJSONLDoesNotEscapeHTML matches the rest of the project: a
// transcript is full of angle brackets and a corpus of \u003c helps nobody.
func TestReportJSONLDoesNotEscapeHTML(t *testing.T) {
	r := &Report{}
	r.Add(&Group{Env: "e", TaskID: "t", Episodes: []*Episode{{
		Task:    Task{ID: "t", Prompt: "write <html> & such"},
		Failure: VerifierFailed,
	}}})

	var buf bytes.Buffer
	if err := r.WriteJSONL(&buf); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	if !strings.Contains(buf.String(), "<html>") {
		t.Fatalf("JSONL escaped the prompt: %s", buf.String())
	}
}

func TestReportAddSkipsNil(t *testing.T) {
	r := &Report{}
	r.Add(nil, &Group{Env: "e", TaskID: "t"}, nil)
	if len(r.Groups) != 1 {
		t.Fatalf("Groups = %d, want 1", len(r.Groups))
	}
}

// TestReportSolvedButUnfinished covers the distinction pass@k cannot draw.
//
// From a live run: one of four samples of a refactor task scored 1.000 —
// including the anti-cheat probe, so the work was genuinely done — and then
// ran to the iteration cap without ever saying it was finished. The table
// reported PASS@1 0.000, PASS@K 0.000, and the only trace of the solve was a
// standard deviation of 0.433. "This agent cannot do the task" and "this agent
// cannot tell when it is done" are different bugs, and they rendered
// identically.
func TestReportSolvedButUnfinished(t *testing.T) {
	g := &Group{
		Env: "dir:refactor", TaskID: "refactor", Threshold: 1.0,
		Episodes: []*Episode{
			{Task: Task{ID: "refactor", Sample: 0}, Reward: 0, Failure: WallClock},
			{Task: Task{ID: "refactor", Sample: 1}, Reward: 0, Failure: MaxIterations},
			{Task: Task{ID: "refactor", Sample: 2}, Reward: 1, Failure: MaxIterations},
			{Task: Task{ID: "refactor", Sample: 3}, Reward: 0, Failure: MaxIterations},
		},
	}
	r := &Report{Groups: []*Group{g}}

	var buf bytes.Buffer
	if err := r.WriteText(&buf); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	got := buf.String()

	if !strings.Contains(got, "solved but did not finish (1)") {
		t.Errorf("report does not name the solved-but-unfinished episode:\n%s", got)
	}
	if !strings.Contains(got, "refactor#2") {
		t.Errorf("report does not identify which sample:\n%s", got)
	}

	// And the numbers stay strict. An agent that never terminates has not
	// succeeded, whatever its workspace ends up containing.
	if p := g.PassAt(1); p != 0 {
		t.Errorf("PassAt(1) = %v, want 0 — the line reports, it does not forgive", p)
	}
	if g.Successes() != 0 {
		t.Errorf("Successes = %d, want 0", g.Successes())
	}
}

// TestReportNoSolvedButUnfinishedLine keeps the report quiet when there is
// nothing to say: a section that always prints is a section nobody reads.
func TestReportNoSolvedButUnfinishedLine(t *testing.T) {
	var buf bytes.Buffer
	if err := sampleReport().WriteText(&buf); err != nil {
		t.Fatalf("WriteText: %v", err)
	}
	if strings.Contains(buf.String(), "solved but did not finish") {
		t.Error("the section printed with no qualifying episode")
	}
}
