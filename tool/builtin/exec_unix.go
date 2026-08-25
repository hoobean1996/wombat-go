//go:build unix

package builtin

import (
	"os/exec"
	"syscall"
)

// isolateProcessGroup puts the child in its own process group and makes
// cancellation kill the whole group.
//
// exec.CommandContext's default cancel kills only the direct child. For
// `sh -c "npm install"` that is the shell, and every process it forked keeps
// running — still holding the pipes, so cmd.Wait then blocks until WaitDelay
// expires, and the orphaned tree outlives the run that was supposed to own
// it. Signalling the negated pgid gets the descendants too.
//
// A dedicated group also means the harness's own SIGINT (Ctrl-C in a
// terminal) is not delivered to the child, so a cancelled tool cannot take
// the agent down with it.
func isolateProcessGroup(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	c.Cancel = func() error {
		if c.Process == nil {
			return nil
		}
		// Negative pid means "the group". Fall back to the single process if
		// the group is already gone, which is not an error worth reporting.
		if err := syscall.Kill(-c.Process.Pid, syscall.SIGKILL); err != nil {
			return c.Process.Kill()
		}
		return nil
	}
}
