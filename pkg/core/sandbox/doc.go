// Package sandbox runs the commands an agent's tools spawn: a [Runner], and
// the two ways of getting one.
//
// [Unconfined] runs a command as an ordinary child process, in a directory,
// with an environment filtered down to what a build needs — so the provider
// key the host holds does not reach a command the model wrote. [Confined]
// does the same inside the platform sandbox, sandbox-exec on macOS and
// bubblewrap on Linux: writes only under the working directory and the temp
// directory, credential directories unreadable, and as much of the network as
// [Config.Network] allows.
//
// Everything that runs a command on an agent's behalf takes a Runner — the
// shell tools, a completion check, the acceptance gate, a review — so one
// decision about confinement covers all of them.
package sandbox
