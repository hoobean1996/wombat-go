package shellquote

import (
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// The argument vector sh hands its command is the one Join was given.
func TestJoinSurvivesTheShell(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	args := []string{"-run", "^(TestA|TestB)$", "-v", "./...", "it's", "", "$HOME", "a b", "*", "x;y", "-tags=a,b"}
	out, err := exec.Command("/bin/sh", "-c", "printf '%s\\n' "+Join(args...)).Output()
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Split(strings.TrimSuffix(string(out), "\n"), "\n")
	if !slices.Equal(got, args) {
		t.Errorf("sh saw %q, want %q", got, args)
	}
	if Quote("./...") != "./..." || Quote("go") != "go" {
		t.Error("a plain word was quoted")
	}
}
