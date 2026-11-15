package benchmarks

import "wombat/eval/rl"

// refactorInterface is the "edit the files nobody told you about" task.
//
// The prompt names exactly one identifier — the Stage interface — and one
// method to add to it. It names none of the five types that satisfy Stage, and
// one of those five lives in the TEST file, which an agent that only greps the
// non-test sources will miss and then spend its remaining turns failing to
// build. Finding them is the task; the Go compiler is the grader, so there is
// nothing to argue about.
//
// The new method is chosen so its correct value is DERIVABLE rather than
// decreed. `Idempotent() bool` is a property of code the agent can read, not a
// number the prompt has to supply per type, which is what lets the probe check
// the answers mechanically: for every stage it compares the declared value
// against what actually happens when Process runs twice. An agent that adds
// the method and returns true everywhere compiles, passes the shipped tests,
// and scores zero on the probe.
//
// The probe is also what forces the INTERFACE change rather than five
// unrelated methods: it calls s.Idempotent() through a []Stage, which does not
// compile unless the method is on the interface.
func refactorInterface() Task {
	return Task{
		ID:      "refactor-interface",
		Summary: "add a method to an interface and chase every implementation and call site",
		Prompt: `This Go module is a small text pipeline. Read it before you change anything.

Add one method to the ` + "`Stage`" + ` interface:

    // Idempotent reports whether running Process twice on the same input
    // gives the same result as running it once.
    Idempotent() bool

Every type that satisfies Stage has to gain it, and the value each one returns
must be the truth about that type: a stage whose Process really is idempotent
returns true, one whose Process is not returns false. Work that out by reading
what each Process does.

This prompt does not tell you which types those are, or which files they are
in. Find them.

Do not change what Name or Process already do, and do not add dependencies.
` + "`go build ./...`" + ` and ` + "`go test ./...`" + ` must both pass when you
are done, including the tests that are already there.`,
		Files: map[string]string{
			"go.mod":           refactorGoMod,
			"stage.go":         refactorStage,
			"upper.go":         refactorUpper,
			"trim.go":          refactorTrim,
			"prefix.go":        refactorPrefix,
			"pipeline.go":      refactorPipeline,
			"pipeline_test.go": refactorTest,
		},
		Verifiers: []rl.Verifier{
			rl.Shell("build", "go build ./...", 0.10),

			// -run pins the three tests that shipped. The test FILE cannot be
			// checksummed on this task the way fix-bug's is, because it holds
			// an implementation of Stage and the refactor genuinely has to
			// edit it — so the anti-cheat here is the probe, which re-checks
			// the old behaviour from bytes the agent never saw.
			rl.Shell("existing_tests",
				`go test -run 'TestPipelineProcess$|TestPipelineNames$|TestPipelineOrder$' -count=1 .`, 0.20),

			GoProbe("new_method", refactorProbeFile, refactorProbe, "TestWombatProbeIdempotent", 0.70),
		},
	}
}

const refactorGoMod = `module pipeline

go 1.25
`

const refactorStage = `// Package pipeline runs text through a chain of named stages.
package pipeline

// Stage is one step of a text pipeline.
//
// Implementations must be pure: Process may depend on nothing but its input,
// because a pipeline reuses one stage value across every input it is given
// and makes no promise about the order.
type Stage interface {
	// Name identifies the stage in diagnostics and in [Pipeline.Name].
	Name() string

	// Process transforms one string.
	Process(in string) string
}
`

const refactorUpper = `package pipeline

import "strings"

// Upper folds its input to upper case.
type Upper struct{}

// Name implements [Stage].
func (Upper) Name() string { return "upper" }

// Process implements [Stage].
func (Upper) Process(in string) string { return strings.ToUpper(in) }
`

const refactorTrim = `package pipeline

import "strings"

// Trim removes leading and trailing whitespace.
type Trim struct{}

// Name implements [Stage].
func (Trim) Name() string { return "trim" }

// Process implements [Stage].
func (Trim) Process(in string) string { return strings.TrimSpace(in) }
`

const refactorPrefix = `package pipeline

// Prefix puts a fixed string in front of its input.
type Prefix struct {
	// Text is prepended to every input, the empty one included.
	Text string
}

// Name implements [Stage].
func (Prefix) Name() string { return "prefix" }

// Process implements [Stage].
func (p Prefix) Process(in string) string { return p.Text + in }
`

// refactorPipeline holds the fourth implementation and the call sites. The
// compile-time assertion is what makes "a pipeline is a stage" a requirement
// the compiler enforces rather than a comment the agent can ignore.
const refactorPipeline = `package pipeline

import "strings"

// Pipeline runs a sequence of stages in order.
//
// A Pipeline is itself a [Stage], so pipelines nest. The assertion below is
// what keeps that true when somebody changes the interface.
type Pipeline struct {
	stages []Stage
}

var _ Stage = (*Pipeline)(nil)

// New builds a pipeline from stages, in the order they will run.
//
// The slice is copied so a caller that keeps its own reference cannot reorder
// a pipeline that is already running.
func New(stages ...Stage) *Pipeline {
	return &Pipeline{stages: append([]Stage(nil), stages...)}
}

// Add appends one stage to the end of the pipeline.
func (p *Pipeline) Add(s Stage) { p.stages = append(p.stages, s) }

// Name implements [Stage]: every stage name joined by "|".
func (p *Pipeline) Name() string {
	parts := make([]string, len(p.stages))
	for i, s := range p.stages {
		parts[i] = s.Name()
	}
	return strings.Join(parts, "|")
}

// Process implements [Stage]: each stage in turn, feeding one's output to the
// next.
func (p *Pipeline) Process(in string) string {
	for _, s := range p.stages {
		in = s.Process(in)
	}
	return in
}
`

// refactorTest ships the fifth implementation of Stage, and it is in a _test.go
// file on purpose: an agent that greps the non-test sources for "Process(" and
// stops finds four of the five and cannot build.
const refactorTest = `package pipeline

import "testing"

// recorder is a stage that remembers what it was handed, so a test can assert
// the ORDER the stages ran in rather than only the final string.
type recorder struct {
	label string
	seen  *[]string
}

func (r recorder) Name() string { return r.label }

func (r recorder) Process(in string) string {
	*r.seen = append(*r.seen, r.label+":"+in)
	return in
}

func TestPipelineProcess(t *testing.T) {
	p := New(Trim{}, Upper{}, Prefix{Text: "> "})
	if got, want := p.Process("  hello  "), "> HELLO"; got != want {
		t.Errorf("Process = %q, want %q", got, want)
	}
}

func TestPipelineNames(t *testing.T) {
	p := New(Trim{}, Upper{})
	p.Add(Prefix{Text: "> "})
	if got, want := p.Name(), "trim|upper|prefix"; got != want {
		t.Errorf("Name = %q, want %q", got, want)
	}
}

func TestPipelineOrder(t *testing.T) {
	var seen []string
	p := New(recorder{label: "a", seen: &seen}, Trim{}, recorder{label: "b", seen: &seen})
	p.Process(" x ")
	if len(seen) != 2 || seen[0] != "a: x " || seen[1] != "b:x" {
		t.Errorf("stages ran in the wrong order or on the wrong input: %q", seen)
	}
}
`

// refactorProbeFile is where [GoProbe] writes refactorProbe. The zz_ prefix
// keeps it last in a directory listing and out of the way of anything an agent
// would plausibly create.
const refactorProbeFile = "zz_wombat_probe_test.go"

// refactorProbe is the grader, and it never reaches the agent.
//
// It checks the DECLARED value of Idempotent against the OBSERVED behaviour of
// Process, so there is no table of expected answers to leak and no way to
// satisfy it except by having read each stage. Ranging over a []Stage is also
// the interface check: this file does not compile unless Idempotent is on the
// interface itself.
const refactorProbe = `package pipeline

import "testing"

func TestWombatProbeIdempotent(t *testing.T) {
	inputs := []string{"", " ", "hello", "  Mixed Case  ", "> already"}

	stages := []Stage{
		Upper{},
		Trim{},
		Prefix{Text: "> "},
		New(Trim{}, Upper{}),
		New(Trim{}, Prefix{Text: "> "}),
	}

	for _, s := range stages {
		stable := true
		for _, in := range inputs {
			once := s.Process(in)
			if s.Process(once) != once {
				stable = false
			}
		}
		if got := s.Idempotent(); got != stable {
			t.Errorf("%s: Idempotent() = %v, but running Process twice %v the result",
				s.Name(), got, map[bool]string{true: "does not change", false: "changes"}[stable])
		}
	}

	// The refactor must not have moved anything else. A gutted pipeline_test.go
	// would hide this; the probe cannot be gutted, because the agent never
	// sees it.
	if got, want := New(Trim{}, Upper{}, Prefix{Text: "> "}).Process("  hello  "), "> HELLO"; got != want {
		t.Errorf("Process = %q, want %q", got, want)
	}
	if got, want := New(Trim{}, Upper{}).Name(), "trim|upper"; got != want {
		t.Errorf("Name = %q, want %q", got, want)
	}
}
`
