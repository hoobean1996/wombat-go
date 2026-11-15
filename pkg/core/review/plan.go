package review

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// DefaultPlanFile and DefaultTodoFile are what the planner skill writes.
const (
	DefaultPlanFile = "PLAN.md"
	DefaultTodoFile = "TODO.md"
)

// planCap bounds how much of the plan reaches the reviewer's prompt. The plan
// is author-written and can run long; the review instructions matter more than
// the tail of a plan, and a prompt that overflows costs the whole run.
const planCap = 6000

// readPlan returns the contents of name under dir, trimmed and capped. A
// missing file is not an error: the plan is an input the reviewer uses when it
// exists, not a precondition for reviewing.
func readPlan(dir, name string) string {
	if dir == "" || name == "" {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return ""
	}
	s := strings.TrimSpace(string(b))
	if len(s) > planCap {
		s = s[:planCap] + "\n\n[plan truncated]"
	}
	return s
}

// todoItem is one checklist line from TODO.md.
type todoItem struct {
	Text string
	Done bool
}

// parseTodos extracts markdown checkboxes. It accepts the forms a model
// actually writes: any leading whitespace, "-" or "*" as the bullet, and
// "x" or "X" as the tick.
func parseTodos(content string) []todoItem {
	var out []todoItem
	for _, line := range strings.Split(content, "\n") {
		t := strings.TrimSpace(line)
		if len(t) < 6 {
			continue
		}
		if t[0] != '-' && t[0] != '*' {
			continue
		}
		rest := strings.TrimSpace(t[1:])
		if len(rest) < 4 || rest[0] != '[' || rest[2] != ']' {
			continue
		}
		var done bool
		switch rest[1] {
		case ' ':
			done = false
		case 'x', 'X':
			done = true
		default:
			continue
		}
		text := strings.TrimSpace(rest[3:])
		if text == "" {
			continue
		}
		out = append(out, todoItem{Text: text, Done: done})
	}
	return out
}

// unfinishedTodos returns the text of every unticked item in dir's TODO file.
//
// This is deliberately not a job for the reviewing model. An unticked box is
// the author's own admission that something in its plan was not built, and
// reading it costs nothing and cannot be wrong. The reviewer's budget is
// better spent on the items that ARE ticked.
func unfinishedTodos(dir, name string) []string {
	content := readPlan(dir, name)
	if content == "" {
		return nil
	}
	var open []string
	for _, it := range parseTodos(content) {
		if !it.Done {
			open = append(open, it.Text)
		}
	}
	return open
}

// planSnapshot fingerprints the planning artifacts so a later comparison can
// tell whether an agent rewrote them.
//
// The repair agent is the reason this exists. Told to make failing tests pass,
// one of them rewrote TODO.md from 22 items to 7 and ticked all of them, then
// fixed the code. The report was still honest — Unfinished is captured before
// any repair — but the file it was captured from no longer describes the work,
// and the next review of that tree would read a tidy 7/7 list with 15 items
// silently gone.
//
// The prohibition is stated in fixPrompt. This is what makes it checkable, on
// the same principle as crediting a repair only when its test survives: an
// instruction the system cannot verify is a hope.
func planSnapshot(dir string, names ...string) map[string]string {
	out := make(map[string]string, len(names))
	for _, name := range names {
		if name == "" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			out[name] = "" // absent; a later non-empty value means it was created
			continue
		}
		sum := sha256.Sum256(b)
		out[name] = hex.EncodeToString(sum[:])
	}
	return out
}

// planEdits returns the names whose contents differ from the snapshot.
func planEdits(dir string, before map[string]string) []string {
	var changed []string
	for name, was := range before {
		now := planSnapshot(dir, name)[name]
		if now != was {
			changed = append(changed, name)
		}
	}
	sort.Strings(changed)
	return changed
}
