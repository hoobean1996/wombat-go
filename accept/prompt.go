package accept

import (
	"fmt"
	"strings"
)

// Feedback is what the agent is told when the gate rejects its answer.
//
// Phrased as remaining work, not as a complaint. The measured failures were
// never refusals — the agent had simply stopped, believing it was finished, so
// what it needs is the same thing a failing test gives it: a specific, checkable
// thing that is not done yet.
func Feedback(rep *Report) string {
	failed := rep.Failed()
	if len(failed) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("You are not done yet. Automatic checks ran against your work and ")
	if len(failed) == 1 {
		b.WriteString("one of them failed.\n\n")
	} else {
		fmt.Fprintf(&b, "%d of them failed.\n\n", len(failed))
	}

	for i, c := range failed {
		fmt.Fprintf(&b, "%d. %s\n", i+1, c.Name)
		if c.Detail != "" {
			for _, line := range strings.Split(strings.TrimRight(c.Detail, "\n"), "\n") {
				b.WriteString("   " + line + "\n")
			}
		}
		b.WriteString("\n")
	}

	b.WriteString(`Fix these, then answer again. The same checks will run on your next answer.

Two things not to do:

- Do not make a check pass by removing what it measures. Deleting a test,
  dropping a checklist item, or narrowing a build target changes the
  measurement rather than the work, and each of those is detected.
- Do not report this as blocked or as someone else's problem. Every check here
  is about code in this directory.

Keep your reply short: the work goes in the files, not in the message.`)

	return b.String()
}
