package fspath

import (
	"path/filepath"
	"testing"
)

func TestWithin(t *testing.T) {
	sep := string(filepath.Separator)
	j := func(parts ...string) string { return filepath.Join(append([]string{sep}, parts...)...) }

	tests := []struct {
		name       string
		root, path string
		want       bool
	}{
		{"root itself", j("work"), j("work"), true},
		{"a child", j("work"), j("work", "main.go"), true},
		{"a deep child", j("work"), j("work", "a", "b", "c"), true},
		{"a sibling sharing a prefix", j("work"), j("workspace"), false},
		{"a sibling sharing a prefix, deeper", j("work"), j("workspace", "x"), false},
		{"the parent", j("work"), sep, false},
		{"elsewhere", j("work"), j("etc", "hosts"), false},

		// The case every hand-written copy got wrong: a root that already
		// ends in a separator. Everything absolute is inside "/".
		{"the filesystem root contains a file", sep, j("etc", "hosts"), true},
		{"the filesystem root contains itself", sep, sep, true},
		{"the filesystem root contains a directory", sep, j("tmp"), true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Within(tc.root, tc.path); got != tc.want {
				t.Errorf("Within(%q, %q) = %v, want %v", tc.root, tc.path, got, tc.want)
			}
		})
	}
}
