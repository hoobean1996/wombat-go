package wombat

import (
	"strings"
	"testing"
)

// A model cut off once and told "be shorter" has been observed to resend the
// same oversized reply verbatim until the run died with nothing written. The
// second notice must therefore say something different from the first.
func TestTruncationNoticeEscalates(t *testing.T) {
	r1 := &Run{truncated: 1} // the counter is incremented before the notice is built
	r2 := &Run{truncated: 2}
	a := &Agent{cfg: config{maxTokens: 16000}}

	first := truncationNotice(a, r1)
	second := truncationNotice(a, r2)

	if first == second {
		t.Fatal("the second notice repeats the first; the measured failure was three identical replies")
	}
	if !strings.Contains(second, "third time") {
		t.Error("the second notice does not name what is happening")
	}
	if !strings.Contains(second, "FRACTION") {
		t.Error("the second notice does not demand a partial reply")
	}
	for _, n := range []string{first, second} {
		if !strings.Contains(n, "16000") {
			t.Errorf("notice omits the actual cap: %q", n)
		}
	}
}
