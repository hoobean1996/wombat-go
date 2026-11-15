// Package shellquote spells a command line for /bin/sh.
//
// The gates that run an agent's code — accept, review — can hand their
// commands to a host's runner, which takes a script rather than an argument
// vector, so the arguments have to survive a shell's parsing intact: a test
// pattern such as ^(TestA|TestB)$ is a pipeline and a glob to sh unless it is
// quoted.
package shellquote

import "strings"

// Quote returns s as a single sh word: unchanged when it holds nothing the
// shell treats specially, single-quoted otherwise.
func Quote(s string) string {
	if s != "" && strings.Trim(s, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./=:-") == "" {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// Join quotes each argument and joins them with spaces.
func Join(args ...string) string {
	q := make([]string, len(args))
	for i, a := range args {
		q[i] = Quote(a)
	}
	return strings.Join(q, " ")
}
