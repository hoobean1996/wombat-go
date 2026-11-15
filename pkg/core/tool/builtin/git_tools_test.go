package builtin

import (
	"fmt"
	"strings"
	"testing"

	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/tool"
)

// TestRequireSafeRef pins the refusal. git has no `--` escape that works for
// revisions — `git show -- x` means the PATH x — so a ref beginning with `-`
// would be parsed as a flag, and git has flags that execute things
// (--upload-pack, --exec).
func TestRequireSafeRef(t *testing.T) {
	tests := []struct {
		name    string
		ref     string
		wantErr string
	}{
		{"empty", "", "field 'ref' must not be empty"},
		{"single dash flag", "-v", `field 'ref' must not start with '-' (it would be read as a git option), got: "-v"`},
		{"long flag", "--upload-pack=/bin/sh", `field 'ref' must not start with '-' (it would be read as a git option), got: "--upload-pack=/bin/sh"`},
		{"exec flag", "--exec=whoami", `field 'ref' must not start with '-' (it would be read as a git option), got: "--exec=whoami"`},
		{"a bare dash", "-", `field 'ref' must not start with '-' (it would be read as a git option), got: "-"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := requireSafeRef("ref", tt.ref)
			if err == nil {
				t.Fatalf("requireSafeRef(ref, %q) = nil, want an error", tt.ref)
			}
			if err.Error() != tt.wantErr {
				t.Errorf("requireSafeRef(ref, %q) = %q, want %q", tt.ref, err, tt.wantErr)
			}
		})
	}

	// No real ref starts with a dash, so refusing the whole class is cheap.
	for _, ref := range []string{"HEAD", "HEAD~3", "main", "v1.2.3", "abc1234", "origin/main", "HEAD^{tree}"} {
		if err := requireSafeRef("ref", ref); err != nil {
			t.Errorf("requireSafeRef(ref, %q) = %v, want nil", ref, err)
		}
	}
}

func TestGitShow(t *testing.T) {
	t.Run("rejects a ref starting with '-' before spawning git", func(t *testing.T) {
		r := &fakeRunner{out: "should never run"}
		out, err := call(t, GitShow(r), `{"cwd":"/repo","ref":"--upload-pack=/bin/sh"}`)
		if err == nil {
			t.Fatalf("out = %q, want a refusal", out)
		}
		if !strings.Contains(err.Error(), "must not start with '-'") {
			t.Errorf("err = %q, want the dash refusal", err)
		}
		if r.cmd != "" || r.args != nil {
			t.Errorf("git was invoked with %q %v, want nothing spawned", r.cmd, r.args)
		}
	})

	t.Run("builds the argv with -C", func(t *testing.T) {
		r := &fakeRunner{out: "commit abc\n\ndiff --git a/x b/x\n"}
		out, err := call(t, GitShow(r), `{"cwd":"/repo","ref":"HEAD~3"}`)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		want := []string{"-C", "/repo", "show", "--no-color", "HEAD~3"}
		if r.cmd != "git" || strings.Join(r.args, " ") != strings.Join(want, " ") {
			t.Errorf("argv = %s %v, want git %v", r.cmd, r.args, want)
		}
		if out != "commit abc\n\ndiff --git a/x b/x" {
			t.Errorf("out = %q, want the output trimmed", out)
		}
	})

	t.Run("cwd must be absolute", func(t *testing.T) {
		r := &fakeRunner{}
		if _, err := call(t, GitShow(r), `{"cwd":"repo","ref":"HEAD"}`); err == nil ||
			!strings.Contains(err.Error(), "field 'cwd' must be an absolute path") {
			t.Errorf("err = %v, want the absolute-path message", err)
		}
		if r.cmd != "" {
			t.Error("git was invoked despite an invalid cwd")
		}
	})

	t.Run("empty output is reported, not returned blank", func(t *testing.T) {
		r := &fakeRunner{out: "   \n\t\n"}
		out, err := call(t, GitShow(r), `{"cwd":"/repo","ref":"HEAD"}`)
		if err != nil || out != "(empty output)" {
			t.Errorf("(%q, %v), want ((empty output), nil)", out, err)
		}
	})

	t.Run("a git failure surfaces", func(t *testing.T) {
		r := &fakeRunner{err: &sandbox.CommandError{Name: "git", Err: exitErr{128}, Stderr: "fatal: bad revision"}}
		_, err := call(t, GitShow(r), `{"cwd":"/repo","ref":"nosuchref"}`)
		if err == nil || !strings.Contains(err.Error(), "bad revision") {
			t.Errorf("err = %v, want the git diagnostic", err)
		}
	})

	t.Run("output is capped", func(t *testing.T) {
		r := &fakeRunner{out: strings.Repeat("+a line of diff\n", 2000)}
		out, err := call(t, GitShow(r), `{"cwd":"/repo","ref":"HEAD"}`)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if len(out) > maxToolOutput+clipMarkerMax {
			t.Errorf("len(out) = %d, want it capped near %d", len(out), maxToolOutput)
		}
	})

	t.Run("is read-only with a fixed argv", func(t *testing.T) {
		d := GitShow(&fakeRunner{})
		if !d.Has(tool.CapReadOnly) || d.Has(tool.CapExec) {
			t.Errorf("Caps = %b, want CapReadOnly and no CapExec", d.Caps)
		}
		if d.Needs&tool.NeedExec == 0 {
			t.Errorf("Needs = %b, want NeedExec", d.Needs)
		}
		if !d.Idempotent || d.Retryable == nil {
			t.Error("want Idempotent with retryExec: the argv is built here, not by the model")
		}
	})
}

func TestGitLog(t *testing.T) {
	t.Run("builds the argv", func(t *testing.T) {
		r := &fakeRunner{out: "abc123 2024-01-01 Ada: initial\n"}
		out, err := call(t, GitLog(r), `{"cwd":"/repo"}`)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		want := []string{"-C", "/repo", "log", "--pretty=format:%h %ad %an: %s", "--date=short", "-n", fmt.Sprint(gitLogDefaultN)}
		if r.cmd != "git" || strings.Join(r.args, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("argv = %s %v, want git %v", r.cmd, r.args, want)
		}
		if out != "abc123 2024-01-01 Ada: initial" {
			t.Errorf("out = %q, want it trimmed", out)
		}
	})

	t.Run("n defaults and is hard-capped", func(t *testing.T) {
		tests := []struct {
			in   string
			want string
		}{
			{`{"cwd":"/r"}`, fmt.Sprint(gitLogDefaultN)},
			{`{"cwd":"/r","n":0}`, fmt.Sprint(gitLogDefaultN)},
			{`{"cwd":"/r","n":-3}`, fmt.Sprint(gitLogDefaultN)},
			{`{"cwd":"/r","n":5}`, "5"},
			{`{"cwd":"/r","n":100000}`, fmt.Sprint(gitLogMaxN)},
		}
		for _, tt := range tests {
			r := &fakeRunner{out: "x\n"}
			call(t, GitLog(r), tt.in)
			if got := argAfter(r.args, "-n"); got != tt.want {
				t.Errorf("%s: -n %s, want -n %s", tt.in, got, tt.want)
			}
		}
	})

	// After `--` git reads everything as a pathspec, so a path beginning with
	// `-` is harmless — which is why git_log takes no requireSafeRef.
	t.Run("a path filter goes after --", func(t *testing.T) {
		r := &fakeRunner{out: "x\n"}
		call(t, GitLog(r), `{"cwd":"/repo","path":"-weird-name.txt"}`)
		n := len(r.args)
		if n < 2 || r.args[n-2] != "--" || r.args[n-1] != "-weird-name.txt" {
			t.Errorf("argv tail = %v, want [-- -weird-name.txt]", r.args[max(0, n-2):])
		}
	})

	t.Run("no path filter means no --", func(t *testing.T) {
		r := &fakeRunner{out: "x\n"}
		call(t, GitLog(r), `{"cwd":"/repo"}`)
		if contains(r.args, "--") {
			t.Errorf("argv = %v, want no `--` when no path filter is given", r.args)
		}
	})

	t.Run("an empty log is reported", func(t *testing.T) {
		r := &fakeRunner{out: "\n  \n"}
		out, err := call(t, GitLog(r), `{"cwd":"/repo"}`)
		if err != nil || out != "(no commits)" {
			t.Errorf("(%q, %v), want ((no commits), nil)", out, err)
		}
	})

	t.Run("cwd must be absolute", func(t *testing.T) {
		r := &fakeRunner{}
		if _, err := call(t, GitLog(r), `{"cwd":"repo"}`); err == nil ||
			!strings.Contains(err.Error(), "must be an absolute path") {
			t.Errorf("err = %v, want the absolute-path message", err)
		}
		if r.cmd != "" {
			t.Error("git was invoked despite an invalid cwd")
		}
	})

	t.Run("a nil Runner panics at construction", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("GitLog(nil) did not panic, want a construction-time panic")
			}
		}()
		GitLog(nil)
	})
}
