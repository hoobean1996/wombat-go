//go:build !unix

package builtin

import "os/exec"

// isolateProcessGroup is a no-op off unix: there is no portable process group
// to signal, so cancellation falls back to exec.CommandContext's default —
// kill the direct child, then let WaitDelay close the pipes on anything it
// spawned.
func isolateProcessGroup(*exec.Cmd) {}
