package builtin

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestBashSpillsAClippedOutput(t *testing.T) {
	dir := t.TempDir()
	d := Bash(OSRunner(""), WithSpill(NewSpill(dir, 0)))
	in, _ := json.Marshal(map[string]string{
		"command":  `i=0; while [ $i -lt 4000 ]; do echo "line $i"; i=$((i+1)); done`,
		"exec_dir": dir,
	})
	out, err := d.Fn(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	// tool.Clip's budget is on content, with its one-line marker on top; the
	// spill note comes out of the budget rather than on top of it.
	if len(out) > maxToolOutput+200 {
		t.Errorf("result is %d bytes, well over the %d cap", len(out), maxToolOutput)
	}
	m := regexp.MustCompile(`saved in (\S+\.log)`).FindStringSubmatch(out)
	if m == nil {
		t.Fatalf("the clipped result does not say where the output went:\n...%s", out[len(out)-300:])
	}
	full, err := os.ReadFile(m[1])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(full), "line 2000\n") || !strings.HasSuffix(strings.TrimSpace(string(full)), "line 3999") {
		t.Error("the saved file is not the complete output")
	}
	if strings.Contains(out, "line 2000\n") {
		t.Error("the middle of the output reached the result; the test's premise is wrong")
	}
	if !strings.Contains(out, "line 3999") {
		t.Error("the clipped result lost the end of the output")
	}
}

func TestBashDoesNotSpillAShortOutput(t *testing.T) {
	dir := t.TempDir()
	d := Bash(OSRunner(""), WithSpill(NewSpill(dir, 0)))
	in, _ := json.Marshal(map[string]string{"command": "echo hi", "exec_dir": dir})
	if out, err := d.Fn(context.Background(), in); err != nil || strings.TrimSpace(out) != "hi" {
		t.Fatalf("got %q, %v", out, err)
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Errorf("a short output was spilled: %v", entries)
	}
}

func TestSpillKeepsOnlyTheNewest(t *testing.T) {
	dir := t.TempDir()
	s := NewSpill(dir, 3)
	for range 7 {
		if _, err := s.save("bash", "x"); err != nil {
			t.Fatal(err)
		}
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.log"))
	if len(matches) != 3 {
		t.Errorf("%d files kept, want 3", len(matches))
	}
}
