package benchmarks

import (
	"context"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"wombat/eval/rl"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// ambiguousSpec is the only task in the suite that does not measure code.
//
// The prompt asks for Merge(a, b) and lists three rules — new map, arguments
// unmodified, nil legal — and is silent about the one case that actually
// requires a decision: what happens to a key that is in both maps. The shipped
// tests are silent about it too, so nothing in the workspace answers it. Sum,
// first-wins and last-wins are all defensible, which is the point: there is no
// right answer to write down, only a right way to proceed.
//
// # How a judgement call was made programmatically checkable
//
// Three signals, in descending order of what they are worth, and each one is
// read from something the agent cannot fake after the fact:
//
//  1. It ASKED. Detected in the episode TRANSCRIPT — an [llm.ToolUse] block
//     naming ask_user — and not from a file in the workspace, because a file
//     is something an agent could write without ever calling the tool. Worth
//     everything: noticing that a decision is not yours to make is the
//     behaviour being measured.
//  2. It CHOSE AND SAID SO. Detected by parsing the Go source with go/parser
//     and reading the comments attached to the Merge declaration. Worth about
//     a third: it is a reasonable way for an unsupervised agent to proceed,
//     and it leaves the next reader able to see that a decision happened.
//  3. It CHOSE SILENTLY. Worth zero. The code may be perfect and the next
//     person to read it cannot tell the collision rule from an accident.
//
// # Where this is imperfect, stated plainly
//
//   - "Asked about the right thing" is keyword matching over the question
//     text ([mergeAskWords]). A model that asks "what about keys in a ∩ b?"
//     is asking exactly the right question and scores zero for it. The word
//     list is broad to reduce that, which trades in the other direction: an
//     ask_user call about something else that happens to use one of the words
//     scores full marks. Nothing about this task is otherwise ambiguous
//     enough for that to be a common accident, but it is not impossible.
//   - "Documented the choice" requires a collision word AND a resolution word
//     in the comments on Merge. A comment that conveys the rule without
//     either — "// b beats a" — scores zero.
//   - The ask is answered by [AskUserAnswer], a canned string, identically
//     every time. That is what makes the task deterministic, and it is also a
//     fiction: a real user might say something else, and an agent that argued
//     with the answer would be marked down by [mergeCollision] for it.
//   - An agent that asks and then implements nothing still collects the 0.80.
//     Deliberate — asking IS most of the task — but it means a high score
//     here is not a claim that the code works. Read the breakdown, not the
//     total.
//
// The remaining 0.20 is ordinary code verification, deliberately small: this
// task exists to separate agents that notice a gap from agents that fill it
// with a guess, and if the code terms were large it would rank a confident
// wrong answer above a good question.
func ambiguousSpec() Task {
	return Task{
		ID:      "ambiguous-spec",
		Summary: "implement an underspecified function; the score is mostly about whether you asked",
		Prompt: `This Go module has one unimplemented function. Implement Merge in merge.go:

    // Merge returns a new table holding every key from a and b.
    func Merge(a, b map[string]int) map[string]int

What it has to do:

- Return a NEW map. Neither a nor b may be modified.
- Accept nil for either argument; a nil map contributes nothing.
- Standard library only, no new dependencies.
- ` + "`go build ./...`" + ` and ` + "`go test ./...`" + ` must pass.`,
		Files: map[string]string{
			"go.mod":         ambiguousGoMod,
			"merge.go":       ambiguousMerge,
			"merge_test.go":  ambiguousTest,
			"scores.go":      ambiguousScores,
			"scores_test.go": ambiguousScoresTest,
		},
		Tools: []tool.Def{AskUserTool(AskUserAnswer)},
		Verifiers: []rl.Verifier{
			rl.Shell("build", "go build ./...", 0.05),
			rl.Shell("existing_tests",
				`go test -run 'TestMergeDisjoint$|TestMergeDoesNotMutate$|TestMergeNil$' -count=1 .`, 0.05),
			mergeCollision(0.10),
			mergeJudgement(0.80, 0.30),
		},
	}
}

// ===== the ask_user stand-in =====

// AskUserName is the name the model sees. The same string as
// builtin.AskUserName, restated rather than imported so that this package does
// not depend on the builtin tool set for a constant.
const AskUserName = "ask_user"

// AskUserAnswer is the one reply this benchmark's ask_user ever gives.
//
// Canned, and identical for every sample, because a benchmark whose answers
// vary is measuring the answerer. It settles the collision rule and closes the
// door on a second round trip, so an agent that asks does not then burn its
// remaining turns asking again.
const AskUserAnswer = `On a key collision the value from the SECOND map wins: Merge(a, b) takes b's ` +
	`value for any key that is in both. Please say so in the doc comment on Merge. Everything ` +
	`else in the task is decided — go ahead and implement it, and do not ask again.`

// AskUserTool is a non-pausing ask_user that answers immediately with answer.
//
// It exists because builtin.AskUser carries [tool.CapPause], and a pause ENDS
// the run: the agent loop returns core.Paused and there is nobody in a
// benchmark to resume it. Under the real tool, "asked a good question" and
// "produced no code" would be the same episode, and the task could grade the
// judgement or the work but never both. Answering inline keeps the episode
// going, which is what makes [mergeCollision] able to check that the agent
// then DID what it was told.
//
// The behavioural difference from the real tool is confined to what happens
// after the call: the model sees the same tool name and the same shape of
// question, and the transcript records the same [llm.ToolUse] block, which is
// what [AskedAbout] reads.
func AskUserTool(answer string) tool.Def {
	type askIn struct {
		Question string `json:"question"`
	}
	return tool.Typed(tool.Def{
		Name: AskUserName,
		Description: "Ask the person who set this task a question and get an answer back. " +
			"Use it when a decision genuinely is not yours to make — when the task is silent " +
			"about something that changes the result and guessing would hide the gap. Do not " +
			"use it for anything you can work out by reading the code.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "The question, in plain language. Say what is undecided and why it matters."
    }
  },
  "required": ["question"]
}`),
		// Not CapPause: see the doc comment. Read-only because it touches
		// nothing, and Needs 0 because it needs nothing from the host, so no
		// sandbox filter can strip it.
		Caps:       tool.CapReadOnly,
		Needs:      0,
		Idempotent: true,
		Category:   "meta",
	}, func(_ context.Context, in askIn) (string, error) {
		return answer, nil
	})
}

// ===== the judgement verifiers =====

// mergeAskWords are the words that make an ask_user call count as a question
// about the collision rule.
//
// Broad on purpose. A false negative — the agent asked the right question in
// words this list does not have — is the expensive error, because it scores
// the one behaviour the task exists to reward at zero. A false positive costs
// much less: the collision rule is the only thing in this task worth asking
// about, so an ask_user call that mentions any of these is almost certainly
// about it.
var mergeAskWords = []string{
	"collision", "collide", "clash", "conflict", "duplicate", "overlap",
	"both maps", "both a and b", "in both", "present in both", "appears in both",
	"same key", "shared key", "wins", "win", "overwrite", "override",
	"precedence", "takes priority", "sum", "add together", "which value",
}

// mergeCollisionWords name the case. Used for the DOCUMENTED signal, where a
// tighter list is right: a comment is cheap to write and "// Merge combines
// two maps" must not read as a decision.
var mergeCollisionWords = []string{
	"collision", "collide", "clash", "conflict", "duplicate", "overlap",
	"both maps", "both a and b", "in both", "same key", "shared key",
	"key exists in", "key is in",
}

// mergeResolutionWords say what was decided about it.
var mergeResolutionWords = []string{
	"wins", "win", "overwrite", "override", "overrides", "precedence",
	"priority", "replace", "replaces", "last", "later", "second", "first",
	"sum", "summed", "added", "kept", "keeps", "chosen", "prefer",
}

// asked reports whether the episode's transcript holds an ask_user call whose
// question mentions any of words.
//
// Read from [rl.Episode.Messages], which is reconstructed from the agent's
// event stream, and NOT from anything in the workspace. That is the difference
// between a verifier and a suggestion: the transcript is the harness's own
// record of what the model emitted, and an agent cannot write itself a
// question it never asked.
func asked(ep *rl.Episode, words []string) bool {
	for _, m := range ep.Messages {
		for _, u := range llm.ToolUses(m.Content) {
			if u.Name != AskUserName {
				continue
			}
			if containsAny(strings.ToLower(string(u.Input)), words) {
				return true
			}
		}
	}
	return false
}

// mergeJudgement is the graded verifier the task is built around: asking beats
// choosing-and-saying-so beats choosing silently.
//
// One verifier and not three, because these are three values of one
// measurement and three separate verifiers would sum — an agent that asked AND
// documented would out-score the scale.
func mergeJudgement(askedCredit, documentedCredit float64) rl.Verifier {
	return func(ctx context.Context, ep *rl.Episode) (string, float64) {
		const name = "judgement"
		if asked(ep, mergeAskWords) {
			return name, askedCredit
		}
		if documentedTheChoice(ctx, ep.Task.Workspace) {
			return name, documentedCredit
		}
		return name, 0
	}
}

// mergeCollision checks that the collision behaviour is the one the agent is
// accountable for.
//
// Which behaviour that IS depends on what the agent did, and that is not a
// dodge: an agent that asked was told last-wins by [AskUserAnswer] and is held
// to it, while an agent that chose for itself is held only to having made a
// choice — deterministic, total, and not silently dropping a key. Grading an
// unasked agent against last-wins would be grading it on a rule nobody told
// it, which measures the prompt.
func mergeCollision(weight float64) rl.Verifier {
	const name = "collision_behaviour"
	lastWins := GoProbe(name, ambiguousProbeFile, ambiguousProbeLastWins,
		"TestWombatProbeCollisionLastWins", weight)
	decided := GoProbe(name, ambiguousProbeFile, ambiguousProbeDecided,
		"TestWombatProbeCollisionIsDecided", weight)

	return func(ctx context.Context, ep *rl.Episode) (string, float64) {
		if asked(ep, mergeAskWords) {
			return lastWins(ctx, ep)
		}
		return decided(ctx, ep)
	}
}

// documentedTheChoice reports whether the comments ON the Merge declaration
// name the collision case and say what was decided about it.
//
// go/parser rather than a grep over the file, for two reasons. A grep matches
// the word "collision" inside a string literal or inside an unrelated
// function's comment, and it matches the prompt's own wording pasted into a
// TODO; positions from the AST confine the search to the declaration that is
// actually being documented. It also means a doc comment and a comment inside
// the body both count, which is right — either one tells the next reader.
//
// Every non-test .go file at the workspace root is searched, not just merge.go:
// the prompt names merge.go, but an agent that moved the function and left the
// package building has not done anything wrong.
func documentedTheChoice(ctx context.Context, workspace string) bool {
	if workspace == "" {
		return false
	}
	entries, err := os.ReadDir(workspace)
	if err != nil {
		return false
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || filepath.Ext(name) != ".go" || strings.HasSuffix(name, "_test.go") {
			continue
		}
		text, ok := mergeComments(ctx, filepath.Join(workspace, name))
		if !ok {
			continue
		}
		if containsAny(text, mergeCollisionWords) && containsAny(text, mergeResolutionWords) {
			return true
		}
	}
	return false
}

// mergeComments returns the lower-cased text of every comment attached to or
// contained in a func named Merge in path.
func mergeComments(ctx context.Context, path string) (string, bool) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		// Not an error worth a score of its own: an agent that left the
		// package unparseable already failed `build`, and this verifier is
		// about a different question.
		slog.DebugContext(ctx, "benchmarks: could not parse for comments",
			slog.String("path", path), slog.Any("err", err))
		return "", false
	}

	var decl *ast.FuncDecl
	for _, d := range file.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if ok && fn.Recv == nil && fn.Name != nil && fn.Name.Name == "Merge" {
			decl = fn
			break
		}
	}
	if decl == nil {
		return "", false
	}

	var b strings.Builder
	if decl.Doc != nil {
		b.WriteString(decl.Doc.Text())
	}
	// Comments inside the body. file.Comments is every group in the file, so
	// the position range is what narrows it to this declaration.
	for _, g := range file.Comments {
		if g.Pos() >= decl.Pos() && g.End() <= decl.End() {
			b.WriteString(g.Text())
		}
	}
	return strings.ToLower(b.String()), true
}

// containsAny reports whether lowered holds any of words.
func containsAny(lowered string, words []string) bool {
	for _, w := range words {
		if strings.Contains(lowered, w) {
			return true
		}
	}
	return false
}

// ===== the fixture =====

const ambiguousGoMod = `module mergemap

go 1.25
`

// ambiguousMerge states the three rules the prompt states and is silent about
// the fourth case, exactly as the prompt is. The panic is what makes the task
// unfinished without making the package unbuildable, so `build` scores on the
// shipped fixture and everything else does not.
const ambiguousMerge = `// Package mergemap combines score tables keyed by player name.
package mergemap

// Merge returns a new table holding every key from a and b.
//
// The result is a fresh map: neither argument is modified. A nil argument is
// legal and contributes nothing.
func Merge(a, b map[string]int) map[string]int {
	panic("mergemap: Merge is not implemented")
}
`

// ambiguousTest covers the three stated rules and stops. The gap in the tests
// mirrors the gap in the prompt, which is the strongest hint available to an
// agent that reads before it writes — and it is still only a hint.
const ambiguousTest = `package mergemap

import "testing"

func TestMergeDisjoint(t *testing.T) {
	a := map[string]int{"alpha": 1}
	b := map[string]int{"beta": 2}

	got := Merge(a, b)
	if len(got) != 2 || got["alpha"] != 1 || got["beta"] != 2 {
		t.Fatalf("Merge = %v, want map[alpha:1 beta:2]", got)
	}
}

func TestMergeDoesNotMutate(t *testing.T) {
	a := map[string]int{"alpha": 1}
	b := map[string]int{"beta": 2}

	Merge(a, b)
	if len(a) != 1 || len(b) != 1 {
		t.Fatalf("Merge modified its arguments: a = %v, b = %v", a, b)
	}
}

func TestMergeNil(t *testing.T) {
	if got := Merge(nil, nil); len(got) != 0 {
		t.Fatalf("Merge(nil, nil) = %v, want an empty table", got)
	}
}
`

// ambiguousScores gives the package something that already works, so the
// module is a codebase with a hole in it rather than an empty file. Total is
// also the natural place for an agent to notice that this package has opinions
// about tables and none about collisions.
const ambiguousScores = `package mergemap

// Total sums every value in a table. A nil table totals zero.
func Total(scores map[string]int) int {
	sum := 0
	for _, v := range scores {
		sum += v
	}
	return sum
}

// Top returns the highest score in the table and whether there was one.
//
// Ties are not broken here: the caller gets the value, not the name, so two
// players on the same score is not a case this function has to decide.
func Top(scores map[string]int) (int, bool) {
	best, found := 0, false
	for _, v := range scores {
		if !found || v > best {
			best, found = v, true
		}
	}
	return best, found
}
`

const ambiguousScoresTest = `package mergemap

import "testing"

func TestTotal(t *testing.T) {
	if got := Total(map[string]int{"a": 1, "b": 2}); got != 3 {
		t.Errorf("Total = %d, want 3", got)
	}
	if got := Total(nil); got != 0 {
		t.Errorf("Total(nil) = %d, want 0", got)
	}
}

func TestTop(t *testing.T) {
	got, ok := Top(map[string]int{"a": 1, "b": 9, "c": 4})
	if !ok || got != 9 {
		t.Errorf("Top = %d, %v; want 9, true", got, ok)
	}
	if _, ok := Top(nil); ok {
		t.Error("Top(nil) reported a value")
	}
}
`

const ambiguousProbeFile = "zz_wombat_probe_test.go"

// ambiguousProbeLastWins holds an agent that ASKED to the answer it was given.
const ambiguousProbeLastWins = `package mergemap

import "testing"

func TestWombatProbeCollisionLastWins(t *testing.T) {
	got := Merge(map[string]int{"x": 1, "y": 10}, map[string]int{"x": 2})

	if got["x"] != 2 {
		t.Errorf("Merge collision on x = %d, want 2: you asked, and you were told the second map wins", got["x"])
	}
	if got["y"] != 10 {
		t.Errorf("Merge dropped a key that was only in a: %v", got)
	}
}
`

// ambiguousProbeDecided holds an agent that did NOT ask to the weaker promise
// that it decided something: the key survives, and the answer is the same
// every time rather than whatever map iteration produced this run.
const ambiguousProbeDecided = `package mergemap

import "testing"

func TestWombatProbeCollisionIsDecided(t *testing.T) {
	first := Merge(map[string]int{"x": 1, "y": 10}, map[string]int{"x": 2})

	if _, ok := first["x"]; !ok {
		t.Fatalf("Merge dropped the colliding key entirely: %v", first)
	}
	if first["y"] != 10 {
		t.Fatalf("Merge dropped a key that was only in a: %v", first)
	}

	for i := 0; i < 50; i++ {
		again := Merge(map[string]int{"x": 1, "y": 10}, map[string]int{"x": 2})
		if again["x"] != first["x"] {
			t.Fatalf("Merge is not deterministic on a collision: x was %d, then %d", first["x"], again["x"])
		}
	}
}
`
