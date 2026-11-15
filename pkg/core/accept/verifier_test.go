package accept

import (
	"context"
	"strings"
	"testing"
)

const verifierBody = `# Verifier

## V1. await of a plain integer is an error
Given: let n = 1 colliding with a live task id
Expect: an *Error, not the other task's value
Status: [x]
Test: TestAwaitPlainInt

## V2. comparisons do not chain
Status: [ ]
Test:

## V3. trailing hash starts a comment
Status: [-] withdrawn: the task never mentions comments at all, so this asks for behaviour it does not specify
Test:
`

func TestParseVerifierReadsAllThreeStates(t *testing.T) {
	items := ParseVerifier(verifierBody)
	if len(items) != 3 {
		t.Fatalf("parsed %d items, want 3: %+v", len(items), items)
	}
	if items[0].ID != "V1" || items[0].State != Verified || items[0].Test != "TestAwaitPlainInt" {
		t.Errorf("V1 = %+v", items[0])
	}
	if !strings.Contains(items[0].Title, "plain integer") {
		t.Errorf("V1 title lost: %q", items[0].Title)
	}
	if items[1].State != Unverified {
		t.Errorf("V2 state = %v, want Unverified", items[1].State)
	}
	if items[2].State != Withdrawn || !strings.Contains(items[2].Note, "never mentions comments") {
		t.Errorf("V3 = %+v", items[2])
	}
	if items[0].Line != 3 {
		t.Errorf("V1 line = %d, want 3", items[0].Line)
	}
}

// The whole contract: a tick is worth nothing unless a real test backs it.
func TestATickWithoutAPassingTestIsNotVerified(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", `package x

import "testing"

func TestReal(t *testing.T)    {}
func TestBroken(t *testing.T)  { t.Fatal("this one fails") }
`)
	cases := []struct {
		name, body, wantIn string
		ok                 bool
	}{
		{"names a passing test", "## V1. thing\nStatus: [x]\nTest: TestReal\n", "", true},
		{"names a failing test", "## V1. thing\nStatus: [x]\nTest: TestBroken\n", "which failed", false},
		{"names a test that does not exist", "## V1. thing\nStatus: [x]\nTest: TestGhost\n", "does not exist", false},
		{"ticked but names nothing", "## V1. thing\nStatus: [x]\nTest:\n", "neither a test nor a command", false},
		{"not ticked at all", "## V1. thing\nStatus: [ ]\nTest:\n", "not verified", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			write(t, dir, "Verifier.md", tc.body)
			rep := Run(context.Background(), Config{
				Workspace: dir, RequireTests: true, VerifierFile: "Verifier.md",
			})
			var got *Check
			for i := range rep.Checks {
				if rep.Checks[i].Name == "verifier" {
					got = &rep.Checks[i]
				}
			}
			if got == nil {
				t.Fatalf("no verifier check ran: %+v", rep.Checks)
			}
			if got.OK != tc.ok {
				t.Fatalf("verifier OK = %v, want %v (detail: %s)", got.OK, tc.ok, got.Detail)
			}
			if tc.wantIn != "" && !strings.Contains(got.Detail, tc.wantIn) {
				t.Errorf("detail does not say %q:\n%s", tc.wantIn, got.Detail)
			}
		})
	}
}

// A wrong item must be escapable — a planner that has not seen the code can
// require something the task never asked for — but not for free, or the escape
// hatch becomes the way the checklist gets deleted.
func TestWithdrawalCostsAnArgument(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")

	write(t, dir, "Verifier.md", "## V1. thing\nStatus: [-] withdrawn: nope\nTest:\n")
	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, VerifierFile: "Verifier.md"})
	if rep.OK() {
		t.Fatal("a bare 'nope' withdrew an item")
	}
	if !strings.Contains(rep.Failed()[0].Detail, "Quote the part of the task") {
		t.Errorf("the detail does not say what an argument needs: %s", rep.Failed()[0].Detail)
	}

	write(t, dir, "Verifier.md",
		"## V1. thing\nStatus: [-] withdrawn: the task text never mentions this behaviour anywhere, so the item asks for something unspecified\nTest:\n")
	if rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, VerifierFile: "Verifier.md"}); !rep.OK() {
		t.Fatalf("a reasoned withdrawal was rejected: %+v", rep.Failed())
	}
}

// An empty file is the cheapest way to pass a check that reads a file.
func TestAnEmptyVerifierIsNotAPass(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	write(t, dir, "Verifier.md", "# Verifier\n\nNothing to check here.\n")

	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, VerifierFile: "Verifier.md"})
	if rep.OK() {
		t.Fatal("a verifier file with no items passed")
	}
}

func TestMissingVerifierIsNotAFailure(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	if rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, VerifierFile: "Verifier.md"}); !rep.OK() {
		t.Fatalf("a missing verifier file failed the gate: %+v", rep.Failed())
	}
}

// Some conditions a task states are commands, not tests. Forcing them through
// a test function is what produced a suite that ran itself as a subprocess and
// then exceeded the time the stated requirement allowed.
func TestACommandCanSettleAnItem(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")

	write(t, dir, "Verifier.md", "## V1. the build succeeds\nStatus: [x]\nCommand: go build ./...\n")
	if rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, VerifierFile: "Verifier.md"}); !rep.OK() {
		t.Fatalf("a passing command did not settle the item: %+v", rep.Failed())
	}

	write(t, dir, "Verifier.md", "## V1. impossible\nStatus: [x]\nCommand: test -f definitely-not-here\n")
	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, VerifierFile: "Verifier.md"})
	if rep.OK() {
		t.Fatal("a failing command settled the item anyway")
	}
	if !strings.Contains(rep.Failed()[0].Detail, "exited non-zero") {
		t.Errorf("the detail does not say the command failed: %s", rep.Failed()[0].Detail)
	}
}

// A ticked item that names nothing at all must not pass.
func TestATickNeedsEitherATestOrACommand(t *testing.T) {
	dir := goModule(t)
	write(t, dir, "x_test.go", "package x\n\nimport \"testing\"\n\nfunc TestReal(t *testing.T) {}\n")
	write(t, dir, "Verifier.md", "## V1. thing\nStatus: [x]\nTest:\nCommand:\n")

	rep := Run(context.Background(), Config{Workspace: dir, RequireTests: true, VerifierFile: "Verifier.md"})
	if rep.OK() {
		t.Fatal("an item backed by nothing passed")
	}
	if !strings.Contains(rep.Failed()[0].Detail, "neither a test nor a command") {
		t.Errorf("detail = %s", rep.Failed()[0].Detail)
	}
}
