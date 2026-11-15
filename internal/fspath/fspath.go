// Package fspath holds the one path-containment test that the file tools, the
// permission rules and the benchmark runner share.
//
// One function rather than a copy in each package, because the copies are how
// it broke: each wrote `clean == root || strings.HasPrefix(clean,
// root+separator)`, which is right for every root except the one that already
// ends in a separator. For "/" the prefix becomes "//", nothing but "/" itself
// matches, and a filesystem rooted at "/" reported /etc/hosts as outside "/".
package fspath

import (
	"path/filepath"
	"strings"
)

// Within reports whether path is root or lies beneath it.
//
// Both arguments must already be cleaned (filepath.Clean), and the test is
// purely textual: nothing touches the disk, so a symlink under root that points
// outside it is "within". Callers document that limit where it matters.
//
// The separator is what makes this a path test rather than a string test —
// /tmp/runs is not an ancestor of /tmp/runs-old — and the only root that
// already ends in one is a filesystem root ("/", or a volume root such as
// `C:\` on Windows), which filepath.Clean leaves with its trailing separator.
func Within(root, path string) bool {
	if path == root {
		return true
	}
	prefix := root
	if !strings.HasSuffix(prefix, string(filepath.Separator)) {
		prefix += string(filepath.Separator)
	}
	return strings.HasPrefix(path, prefix)
}
