package review

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The exact construct that let a correct test go green over a call that was
// returning a caller-killing error.
func TestDiscardedErrorsFindsTheRealCase(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "queue_test.go", `package jobq

import "testing"

func TestRedeliveryExhaustionDeadLetters(t *testing.T) {
	q, _ := tempDB(t)
	if _, ok, _ := q.Claim("w"); ok {
		t.Fatal("should not claim after dead-lettering")
	}
}
`)
	got := discardedErrors(dir)
	if len(got) != 2 {
		t.Fatalf("found %d sites, want 2: %v", len(got), got)
	}
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "q.Claim(...)") {
		t.Errorf("the Claim site was not reported: %v", got)
	}
	if !strings.Contains(joined, "queue_test.go:6") {
		t.Errorf("expected the tempDB site at line 6: %v", got)
	}
}

func TestDiscardedErrorsIgnoresWhatIsNotADiscardedResult(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "x_test.go", `package p

import "testing"

func TestX(t *testing.T) {
	_ = compute()          // single-value: not the error convention
	a, b := 1, 2           // not a call
	v, _ := lookup()       // discarded, but not last-of-three — still counts
	_, err := lookup()     // error is KEPT
	if err != nil || a+b+v == 0 {
		t.Fatal("x")
	}
}
`)
	got := discardedErrors(dir)
	if len(got) != 1 {
		t.Fatalf("found %d sites, want 1 (only `v, _ := lookup()`): %v", len(got), got)
	}
	if !strings.Contains(got[0], "lookup(...)") {
		t.Errorf("wrong site reported: %v", got)
	}
}

// The reviewer's own files are not the author's tests, and a non-test file is
// not in scope: production code discards results for reasons this cannot judge.
func TestDiscardedErrorsSkipsReviewAndNonTestFiles(t *testing.T) {
	dir := t.TempDir()
	body := `package p

func f() { a, _ := g(); _ = a }
`
	write(t, dir, "review_probe_test.go", body)
	write(t, dir, "main.go", body)
	if got := discardedErrors(dir); len(got) != 0 {
		t.Fatalf("expected no sites, got %v", got)
	}
}

// A tree that does not compile must not take the review down with it.
func TestDiscardedErrorsSurvivesUnparseableFiles(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, "broken_test.go", "package p\nfunc {{{")
	write(t, dir, "ok_test.go", "package p\nfunc f() { a, _ := g(); _ = a }\n")

	got := discardedErrors(dir)
	if len(got) != 1 {
		t.Fatalf("a broken neighbour changed the result: %v", got)
	}
}

func TestDiscardedErrorsOnMissingDirIsEmpty(t *testing.T) {
	if got := discardedErrors(filepath.Join(t.TempDir(), "nope")); got != nil {
		t.Fatalf("got %v, want nil", got)
	}
	if got := discardedErrors(""); got != nil {
		t.Fatalf("got %v, want nil", got)
	}
}

func TestReviewPromptCarriesDiscardedLeadsAsLeads(t *testing.T) {
	p := reviewPrompt("t", "", []string{"queue_test.go:7: q.Claim(...) — last result discarded"}, lensFor(1), nil)
	if !strings.Contains(p, "queue_test.go:7") {
		t.Fatal("lead missing from the prompt")
	}
	if !strings.Contains(p, "leads, not defects") {
		t.Error("leads are not marked as leads; the reviewer will file them as findings")
	}
	if !strings.Contains(p, "A GREEN TEST IS NOT A TESTED PATH") {
		t.Error("the green-test lens is missing")
	}
}

var _ = os.WriteFile
