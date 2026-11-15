//go:build !unix

package mcp

import (
	"os"
	"os/exec"
)

// isolate is a no-op where there is no process group to put the server in.
func isolate(*exec.Cmd) {}

// stopProcess kills the server: there is no SIGTERM to ask with first.
func stopProcess(p *os.Process, _ bool) error { return p.Kill() }
