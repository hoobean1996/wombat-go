//go:build unix

package mcp

import (
	"os"
	"os/exec"
	"syscall"
)

// isolate starts the server in a process group of its own, so stopping it
// stops whatever it started, and a Ctrl-C meant for the harness is not
// delivered to it as well.
func isolate(c *exec.Cmd) { c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} }

// stopProcess sends SIGTERM (terminate) or SIGKILL to the server's process group,
// falling back to the process alone if the group is gone.
func stopProcess(p *os.Process, terminate bool) error {
	sig := syscall.SIGKILL
	if terminate {
		sig = syscall.SIGTERM
	}
	if err := syscall.Kill(-p.Pid, sig); err != nil {
		return p.Signal(sig)
	}
	return nil
}
