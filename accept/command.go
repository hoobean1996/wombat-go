package accept

import (
	"context"
	"fmt"
	"os/exec"
	"strings"

	"github.com/hoobean1996/wombat-go/internal/safeenv"
)

// A verifier item can be settled by a COMMAND as well as by a test.
//
// The format started with tests only, and that was a mistake with a concrete
// cost. A task said "'go test -race ./...' must pass"; the plan turned it into
// item V1; and because every item needed a Go test function to back it, the
// build wrote a test that ran the entire suite as a subprocess. The suite then
// took longer than six minutes under -race and the stated requirement was not
// met — by the code written to prove it was met.
//
// Some conditions are commands. "The build succeeds", "vet is clean", "the CLI
// exits 0 on this input" are checked by running something and looking at the
// exit status, and forcing them through a test function adds a layer that can
// itself fail.
type command struct {
	Line string // as written in the item
}

// runCommand executes one Command: line in the workspace and reports whether it
// succeeded, with the tail of its output when it did not.
func runCommand(ctx context.Context, cfg Config, line string) (bool, string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return false, "the Command: line is empty"
	}
	tctx, cancel := context.WithTimeout(ctx, cfg.timeout())
	defer cancel()

	// Through a shell: these are the commands the task states, written the way
	// a person would type them, pipes and all.
	cmd := exec.CommandContext(tctx, "sh", "-c", line)
	cmd.Dir = cfg.Workspace
	cmd.Env = safeenv.Environ() // runs the agent's code: no harness credentials
	out, err := cmd.CombinedOutput()
	if err == nil {
		return true, ""
	}
	text := strings.TrimSpace(string(out))
	const cap = 1500
	if len(text) > cap {
		text = "..." + text[len(text)-cap:]
	}
	if tctx.Err() != nil {
		return false, fmt.Sprintf("did not finish within %s. A condition the task states must be\n"+
			"reachable in reasonable time; if a test suite has grown too slow to run, that is\n"+
			"itself the defect.\n\n%s", cfg.timeout(), text)
	}
	return false, fmt.Sprintf("exited non-zero: %v\n\n%s", err, text)
}
