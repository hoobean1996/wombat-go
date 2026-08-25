package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// The warning has to arrive while there is still room to act on it. At two
// iterations left the only way to obey is to stop, which is the failure it is
// meant to prevent.
func TestIterationNoticeArrivesWhileThereIsRoomToAct(t *testing.T) {
	n := iterationNotice(80)
	ctx := context.Background()

	if got := n(ctx, 40); got != "" {
		t.Errorf("warned at half the budget: %q", got)
	}
	if got := n(ctx, 53); got != "" {
		t.Errorf("warned just under two thirds: %q", got)
	}
	got := n(ctx, 54)
	if got == "" {
		t.Fatal("silent at two thirds of the budget")
	}
	if !strings.Contains(got, "26 left") {
		t.Errorf("the remaining count is wrong or missing: %q", got)
	}
	if !strings.Contains(got, "does not exist yet") {
		t.Error("the notice says how much is left but not to check for missing work")
	}
	if !strings.Contains(got, "Do not shorten a checklist") {
		t.Error("nothing forbids the shortcut both measured runs took")
	}
}

func TestIterationNoticeDoesNotGoNegativeOrPanic(t *testing.T) {
	n := iterationNotice(10)
	for _, iter := range []int{9, 10, 11, 50} {
		got := n(context.Background(), iter)
		if got == "" {
			t.Fatalf("iteration %d produced no notice", iter)
		}
		var left int
		if _, err := fmt.Sscanf(got[strings.Index(got, "; "):], "; %d left", &left); err != nil {
			t.Fatalf("iteration %d: could not read the remainder from %q", iter, got)
		}
		if left < 0 {
			t.Errorf("iteration %d reported %d left", iter, left)
		}
	}
	if iterationNotice(0) != nil {
		t.Error("an unbounded run should get no iteration notice")
	}
}
