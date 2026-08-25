package accept

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// A census is the number of items the plan had when the build started.
//
// Counting unticked boxes is not enough on its own, because a deleted box is
// not an unticked one. Measured three times across two tasks: a 17-item list
// came back with 8, a 22-item list with 7, and a 90-item list with 89 — each
// time fully ticked, each time reported as complete. Shortening the list is
// cheaper than finishing it and looks identical afterwards.
//
// So the count is recorded before the build and compared at the gate. This is
// the same rule as crediting a repair only when its test survives: whatever is
// being measured must not be able to edit the measurement.
type census struct {
	Todo     int `json:"todo"`
	Verifier int `json:"verifier"`
}

const censusFile = ".plan-census.json"

// RecordCensus counts the plan's items and stores the counts in the workspace.
// Called once, before the build starts.
func RecordCensus(workspace, todoFile, verifierFile string) error {
	c := census{
		Todo:     countTodoBoxes(read(workspace, todoFile)),
		Verifier: len(ParseVerifier(read(workspace, verifierFile))),
	}
	if c.Todo == 0 && c.Verifier == 0 {
		return nil // no plan to protect
	}
	b, err := json.Marshal(c)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(workspace, censusFile), b, 0o644)
}

func read(dir, name string) string {
	if dir == "" || name == "" {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return ""
	}
	return string(b)
}

func countTodoBoxes(content string) int {
	n := 0
	for _, line := range strings.Split(content, "\n") {
		t := strings.TrimSpace(line)
		if len(t) < 4 || (t[0] != '-' && t[0] != '*') {
			continue
		}
		rest := strings.TrimSpace(t[1:])
		if len(rest) >= 3 && rest[0] == '[' && rest[2] == ']' {
			n++
		}
	}
	return n
}

// checkCensus fails the gate when the plan has fewer items than it started
// with. Growth is fine — noticing more work is not the failure mode.
func (r *Report) checkCensus(cfg Config) {
	b, err := os.ReadFile(filepath.Join(cfg.Workspace, censusFile))
	if err != nil {
		return // nothing was recorded; nothing to compare against
	}
	var was census
	if json.Unmarshal(b, &was) != nil {
		return
	}
	now := census{
		Todo:     countTodoBoxes(read(cfg.Workspace, cfg.TodoFile)),
		Verifier: len(ParseVerifier(read(cfg.Workspace, cfg.VerifierFile))),
	}

	var lost []string
	if cfg.TodoFile != "" && now.Todo < was.Todo {
		lost = append(lost, fmt.Sprintf("%s went from %d items to %d", cfg.TodoFile, was.Todo, now.Todo))
	}
	if cfg.VerifierFile != "" && now.Verifier < was.Verifier {
		lost = append(lost, fmt.Sprintf("%s went from %d items to %d", cfg.VerifierFile, was.Verifier, now.Verifier))
	}
	if len(lost) == 0 {
		r.add(Check{Name: "plan intact", OK: true})
		return
	}
	r.add(Check{Name: "plan intact", OK: false, Detail: strings.Join(lost, "\n") +
		"\n\nItems were removed from the plan during this build. Restore them and " +
		"either\ndo the work or, for a verifier item, withdraw it with an argument " +
		"from the task.\nA shorter list is not a finished one."})
}
