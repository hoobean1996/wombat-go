package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// fakeFS is the "struct with four fields" the FS doc promises. A nil field
// means the method is never expected to be called.
type fakeFS struct {
	readFile  func(ctx context.Context, path string) ([]byte, error)
	writeFile func(ctx context.Context, path string, data []byte) error
	stat      func(ctx context.Context, path string) (Kind, error)
	listDir   func(ctx context.Context, path string) ([]string, error)
}

func (f fakeFS) ReadFile(ctx context.Context, p string) ([]byte, error) {
	if f.readFile == nil {
		return nil, fmt.Errorf("fakeFS: unexpected ReadFile(%q)", p)
	}
	return f.readFile(ctx, p)
}

func (f fakeFS) WriteFile(ctx context.Context, p string, d []byte) error {
	if f.writeFile == nil {
		return fmt.Errorf("fakeFS: unexpected WriteFile(%q)", p)
	}
	return f.writeFile(ctx, p, d)
}

func (f fakeFS) Stat(ctx context.Context, p string) (Kind, error) {
	if f.stat == nil {
		return KindMissing, fmt.Errorf("fakeFS: unexpected Stat(%q)", p)
	}
	return f.stat(ctx, p)
}

func (f fakeFS) ListDir(ctx context.Context, p string) ([]string, error) {
	if f.listDir == nil {
		return nil, fmt.Errorf("fakeFS: unexpected ListDir(%q)", p)
	}
	return f.listDir(ctx, p)
}

func call(t *testing.T, d tool.Def, input string) (string, error) {
	t.Helper()
	return d.Fn(context.Background(), json.RawMessage(input))
}

// ===== requireAbs =====

// The wording is what teaches the model to retry with an absolute path, so it
// is pinned rather than paraphrased.
func TestRequireAbs(t *testing.T) {
	tests := []struct {
		name    string
		field   string
		path    string
		wantErr string
	}{
		{"empty", "path", "", "field 'path' must not be empty"},
		{"relative", "path", "src/main.go", `field 'path' must be an absolute path (starts with /), got: "src/main.go"`},
		{"dot", "path", ".", `field 'path' must be an absolute path (starts with /), got: "."`},
		{"tilde is not absolute", "path", "~/notes.md", `field 'path' must be an absolute path (starts with /), got: "~/notes.md"`},
		{"names the field it was given", "exec_dir", "build", `field 'exec_dir' must be an absolute path (starts with /), got: "build"`},
		{"names the field when empty", "cwd", "", "field 'cwd' must not be empty"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := requireAbs(tt.field, tt.path)
			if err == nil {
				t.Fatalf("requireAbs(%q, %q) = nil, want an error", tt.field, tt.path)
			}
			if err.Error() != tt.wantErr {
				t.Errorf("requireAbs(%q, %q) = %q, want %q", tt.field, tt.path, err, tt.wantErr)
			}
		})
	}

	for _, p := range []string{"/", "/tmp/x", "/a/b/../c"} {
		if err := requireAbs("path", p); err != nil {
			t.Errorf("requireAbs(path, %q) = %v, want nil", p, err)
		}
	}
}

// Every file-touching tool must reject a relative path with the same message.
func TestFileToolsValidateAbsolutePaths(t *testing.T) {
	fsys := fakeFS{}
	tests := []struct {
		name  string
		def   tool.Def
		input string
		field string
	}{
		{"view_file", ViewFile(fsys), `{"path":"relative.txt"}`, "path"},
		{"write_file", WriteFile(fsys), `{"path":"relative.txt","content":"x"}`, "path"},
		{"str_replace", StrReplace(fsys), `{"path":"relative.txt","old_str":"a","new_str":"b"}`, "path"},
		{"save_tool_result", SaveToolResult(fsys), `{"tool_use_id":"u1","path":"relative.txt"}`, "path"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := call(t, tt.def, tt.input)
			if err == nil {
				t.Fatalf("%s = %q, want a path validation error", tt.name, out)
			}
			want := fmt.Sprintf("field '%s' must be an absolute path (starts with /), got: %q", tt.field, "relative.txt")
			if err.Error() != want {
				t.Errorf("err = %q, want %q", err, want)
			}
		})
	}
}

// ===== view_file =====

// TestNumberLinesTrailingNewline is a regression test. The OCaml did not trim
// the terminator, so "foo\n" was reported as TWO lines with an empty second
// one, and the model would routinely ask to edit a line that does not exist.
func TestNumberLines(t *testing.T) {
	tests := []struct {
		name      string
		content   string
		viewRange []int
		want      string
	}{
		{"no trailing newline", "foo", nil, "    1\tfoo"},
		{"one trailing newline is a terminator, not a line", "foo\n", nil, "    1\tfoo"},
		{"two lines", "a\nb", nil, "    1\ta\n    2\tb"},
		{"two lines terminated", "a\nb\n", nil, "    1\ta\n    2\tb"},
		{"a genuinely blank last line survives", "a\nb\n\n", nil, "    1\ta\n    2\tb\n    3\t"},
		{"empty content is one empty line", "", nil, "    1\t"},
		{"a lone newline is one empty line", "\n", nil, "    1\t"},
		{"range selects an interior slice", "a\nb\nc\nd", []int{2, 3}, "    2\tb\n    3\tc"},
		{"range of one", "a\nb\nc", []int{2, 2}, "    2\tb"},
		{"start below 1 clamps", "a\nb", []int{-5, 2}, "    1\ta\n    2\tb"},
		{"end past the last line clamps", "a\nb", []int{1, 99}, "    1\ta\n    2\tb"},
		{"a range entirely past the end yields nothing", "a\nb", []int{5, 9}, ""},
		{"an inverted range yields nothing", "a\nb\nc", []int{3, 1}, ""},
		// Anything that is not exactly two elements is ignored, matching the
		// tolerant decode the schema invites.
		{"a one-element range is ignored", "a\nb", []int{2}, "    1\ta\n    2\tb"},
		{"a three-element range is ignored", "a\nb", []int{1, 1, 1}, "    1\ta\n    2\tb"},
		{"an empty range is ignored", "a\nb", []int{}, "    1\ta\n    2\tb"},
		{"line numbers are right-aligned in five columns", strings.Repeat("x\n", 10), []int{10, 10}, "   10\tx"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := numberLines(tt.content, tt.viewRange); got != tt.want {
				t.Errorf("numberLines(%q, %v) = %q, want %q", tt.content, tt.viewRange, got, tt.want)
			}
		})
	}
}

func TestViewFile(t *testing.T) {
	dir := t.TempDir()
	fsys := OSFS(dir)
	d := ViewFile(fsys)

	write := func(name, content string) string {
		t.Helper()
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	t.Run("reads a file with line numbers", func(t *testing.T) {
		p := write("a.txt", "first\nsecond\nthird\n")
		out, err := call(t, d, fmt.Sprintf(`{"path":%q}`, p))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		want := "    1\tfirst\n    2\tsecond\n    3\tthird"
		if out != want {
			t.Errorf("out = %q, want %q", out, want)
		}
	})

	t.Run("honours view_range", func(t *testing.T) {
		p := write("b.txt", "1\n2\n3\n4\n5\n")
		out, err := call(t, d, fmt.Sprintf(`{"path":%q,"view_range":[2,4]}`, p))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if want := "    2\t2\n    3\t3\n    4\t4"; out != want {
			t.Errorf("out = %q, want %q", out, want)
		}
	})

	t.Run("lists a directory instead of reading it", func(t *testing.T) {
		sub := filepath.Join(dir, "listme")
		if err := os.MkdirAll(filepath.Join(sub, "nested"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(sub, "file.txt"), nil, 0o644); err != nil {
			t.Fatal(err)
		}

		out, err := call(t, d, fmt.Sprintf(`{"path":%q}`, sub))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		want := "Directory: " + sub + "\n  file.txt\n  nested/"
		if out != want {
			t.Errorf("out = %q, want %q", out, want)
		}
	})

	t.Run("an empty directory still names itself", func(t *testing.T) {
		sub := filepath.Join(dir, "emptydir")
		if err := os.Mkdir(sub, 0o755); err != nil {
			t.Fatal(err)
		}
		out, err := call(t, d, fmt.Sprintf(`{"path":%q}`, sub))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if want := "Directory: " + sub; out != want {
			t.Errorf("out = %q, want %q", out, want)
		}
	})

	t.Run("a missing path is a readable error", func(t *testing.T) {
		p := filepath.Join(dir, "nope.txt")
		_, err := call(t, d, fmt.Sprintf(`{"path":%q}`, p))
		if err == nil {
			t.Fatal("err = nil, want a not-found error")
		}
		if want := "no such file or directory: " + p; err.Error() != want {
			t.Errorf("err = %q, want %q", err, want)
		}
	})

	t.Run("output is capped", func(t *testing.T) {
		p := write("big.txt", strings.Repeat("line of text\n", 5000))
		out, err := call(t, d, fmt.Sprintf(`{"path":%q}`, p))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if len(out) > maxToolOutput+clipMarkerMax {
			t.Errorf("len(out) = %d, want it capped near %d", len(out), maxToolOutput)
		}
		if !strings.Contains(out, "omitted") {
			t.Errorf("out does not mention truncation, want the marker")
		}
	})

	t.Run("a Stat failure surfaces", func(t *testing.T) {
		boom := errors.New("filesystem offline")
		bad := ViewFile(fakeFS{stat: func(context.Context, string) (Kind, error) { return KindMissing, boom }})
		if _, err := call(t, bad, `{"path":"/x"}`); !errors.Is(err, boom) {
			t.Errorf("err = %v, want %v", err, boom)
		}
	})

	t.Run("metadata", func(t *testing.T) {
		if !d.Idempotent {
			t.Error("Idempotent = false, want true")
		}
		if d.Retryable == nil {
			t.Fatal("Retryable = nil, want retryFS")
		}
		if d.Needs != tool.NeedFSRead {
			t.Errorf("Needs = %b, want NeedFSRead", d.Needs)
		}
		if !d.Has(tool.CapReadOnly) {
			t.Errorf("Caps = %b, want CapReadOnly", d.Caps)
		}
	})

	t.Run("a nil FS panics at construction", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("ViewFile(nil) did not panic, want a construction-time panic")
			}
		}()
		ViewFile(nil)
	})
}

// ===== write_file =====

func TestWriteFile(t *testing.T) {
	dir := t.TempDir()
	d := WriteFile(OSFS(dir))

	t.Run("writes and reports the byte count", func(t *testing.T) {
		p := filepath.Join(dir, "out.txt")
		out, err := call(t, d, fmt.Sprintf(`{"path":%q,"content":"hello"}`, p))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if want := fmt.Sprintf("wrote 5 bytes to %s", p); out != want {
			t.Errorf("out = %q, want %q", out, want)
		}
		if data, _ := os.ReadFile(p); string(data) != "hello" {
			t.Errorf("file = %q, want %q", data, "hello")
		}
	})

	t.Run("the byte count is bytes, not runes", func(t *testing.T) {
		p := filepath.Join(dir, "utf8.txt")
		out, err := call(t, d, fmt.Sprintf(`{"path":%q,"content":"中文"}`, p))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if !strings.HasPrefix(out, "wrote 6 bytes") {
			t.Errorf("out = %q, want it to report 6 bytes", out)
		}
	})

	t.Run("overwrites", func(t *testing.T) {
		p := filepath.Join(dir, "over.txt")
		call(t, d, fmt.Sprintf(`{"path":%q,"content":"aaaaaaaa"}`, p))
		call(t, d, fmt.Sprintf(`{"path":%q,"content":"b"}`, p))
		if data, _ := os.ReadFile(p); string(data) != "b" {
			t.Errorf("file = %q, want %q", data, "b")
		}
	})

	t.Run("a write outside the root is refused", func(t *testing.T) {
		_, err := call(t, d, `{"path":"/etc/definitely-not-writable-by-this-test","content":"x"}`)
		if !errors.Is(err, ErrOutsideRoot) {
			t.Errorf("err = %v, want ErrOutsideRoot", err)
		}
	})

	// A write that timed out may well have landed, so it must never be replayed.
	t.Run("is not idempotent", func(t *testing.T) {
		if d.Idempotent {
			t.Error("Idempotent = true, want false: a replayed write can clobber a later edit")
		}
		if !d.Has(tool.CapMutating) {
			t.Errorf("Caps = %b, want CapMutating", d.Caps)
		}
	})
}

// ===== str_replace =====

func TestStrReplace(t *testing.T) {
	dir := t.TempDir()
	fsys := OSFS(dir)
	d := StrReplace(fsys)

	write := func(name, content string) string {
		t.Helper()
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	t.Run("replaces a unique occurrence", func(t *testing.T) {
		p := write("one.txt", "alpha\nbeta\ngamma\n")
		out, err := call(t, d, fmt.Sprintf(`{"path":%q,"old_str":"beta","new_str":"BETA!"}`, p))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if data, _ := os.ReadFile(p); string(data) != "alpha\nBETA!\ngamma\n" {
			t.Errorf("file = %q, want %q", data, "alpha\nBETA!\ngamma\n")
		}
		if want := fmt.Sprintf("replaced 1 occurrence in %s (17 → 18 bytes)", p); out != want {
			t.Errorf("out = %q, want %q", out, want)
		}
	})

	// An ambiguous old_str means the model does not know which site it is
	// editing, and guessing is worse than refusing.
	t.Run("refuses an ambiguous old_str and says how many matched", func(t *testing.T) {
		p := write("many.txt", "x\nx\nx\n")
		_, err := call(t, d, fmt.Sprintf(`{"path":%q,"old_str":"x","new_str":"y"}`, p))
		if err == nil {
			t.Fatal("err = nil, want a uniqueness failure")
		}
		if want := fmt.Sprintf("old_str matches 3 times in %s — must be unique", p); err.Error() != want {
			t.Errorf("err = %q, want %q", err, want)
		}
		if data, _ := os.ReadFile(p); string(data) != "x\nx\nx\n" {
			t.Errorf("file = %q, want it UNCHANGED after a refusal", data)
		}
	})

	t.Run("reports a missing old_str", func(t *testing.T) {
		p := write("none.txt", "hello\n")
		_, err := call(t, d, fmt.Sprintf(`{"path":%q,"old_str":"absent","new_str":"y"}`, p))
		if want := "old_str not found in " + p; err == nil || err.Error() != want {
			t.Errorf("err = %v, want %q", err, want)
		}
	})

	// An empty old_str would match everywhere, so it counts as no match at all.
	t.Run("an empty old_str is no match, not every match", func(t *testing.T) {
		p := write("empty.txt", "hello\n")
		_, err := call(t, d, fmt.Sprintf(`{"path":%q,"old_str":"","new_str":"INJECTED"}`, p))
		if want := "old_str not found in " + p; err == nil || err.Error() != want {
			t.Errorf("err = %v, want %q", err, want)
		}
		if data, _ := os.ReadFile(p); string(data) != "hello\n" {
			t.Errorf("file = %q, want it unchanged", data)
		}
	})

	t.Run("matching is whitespace-sensitive", func(t *testing.T) {
		p := write("ws.txt", "  indented\n")
		if _, err := call(t, d, fmt.Sprintf(`{"path":%q,"old_str":"indented","new_str":"flush"}`, p)); err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if data, _ := os.ReadFile(p); string(data) != "  flush\n" {
			t.Errorf("file = %q, want %q — the surrounding whitespace is preserved", data, "  flush\n")
		}
	})

	t.Run("a multi-line old_str works", func(t *testing.T) {
		p := write("multi.txt", "a\nb\nc\n")
		if _, err := call(t, d, fmt.Sprintf(`{"path":%q,"old_str":"a\nb","new_str":"A"}`, p)); err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if data, _ := os.ReadFile(p); string(data) != "A\nc\n" {
			t.Errorf("file = %q, want %q", data, "A\nc\n")
		}
	})

	t.Run("deleting is replacing with nothing", func(t *testing.T) {
		p := write("del.txt", "keep\ndrop\n")
		out, err := call(t, d, fmt.Sprintf(`{"path":%q,"old_str":"drop\n","new_str":""}`, p))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if data, _ := os.ReadFile(p); string(data) != "keep\n" {
			t.Errorf("file = %q, want %q", data, "keep\n")
		}
		if !strings.Contains(out, "10 → 5 bytes") {
			t.Errorf("out = %q, want it to report 10 → 5 bytes", out)
		}
	})

	t.Run("a read failure surfaces and nothing is written", func(t *testing.T) {
		_, err := call(t, d, fmt.Sprintf(`{"path":%q,"old_str":"a","new_str":"b"}`, filepath.Join(dir, "ghost.txt")))
		if err == nil {
			t.Fatal("err = nil, want the read failure")
		}
		if !strings.Contains(err.Error(), "no such file") {
			t.Errorf("err = %q, want it to report the missing file", err)
		}
	})

	// A replayed edit finds no match and reports failure, which would read to
	// the model as "my edit was lost".
	t.Run("is not idempotent", func(t *testing.T) {
		if d.Idempotent {
			t.Error("Idempotent = true, want false")
		}
	})
}

// ===== save_tool_result =====

func TestSaveToolResult(t *testing.T) {
	dir := t.TempDir()
	d := SaveToolResult(OSFS(dir))

	// The whole point of the tool is that the bytes never leave the harness.
	t.Run("writes a previous observation without re-emitting it", func(t *testing.T) {
		p := filepath.Join(dir, "saved.json")
		payload := strings.Repeat("payload ", 100)
		ctx := tool.WithLookup(context.Background(), func(id llm.ToolUseID) (string, error) {
			if id != "toolu_01ABC" {
				return "", fmt.Errorf("no tool_result with id %s", id)
			}
			return payload, nil
		})

		out, err := d.Fn(ctx, json.RawMessage(fmt.Sprintf(`{"tool_use_id":"toolu_01ABC","path":%q}`, p)))
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if data, _ := os.ReadFile(p); string(data) != payload {
			t.Errorf("file has %d bytes, want the %d-byte payload", len(data), len(payload))
		}
		if !strings.Contains(out, fmt.Sprintf("saved %d bytes", len(payload))) {
			t.Errorf("out = %q, want it to report %d bytes", out, len(payload))
		}
		if !strings.Contains(out, "no LLM regen") {
			t.Errorf("out = %q, want it to say the content was not regenerated", out)
		}
	})

	// Outside an agent loop there is no transcript, and the tool must say so
	// and point at the alternative rather than write an empty file.
	t.Run("a nil Lookup is reported, not a nil-deref", func(t *testing.T) {
		p := filepath.Join(dir, "never.json")
		out, err := call(t, d, fmt.Sprintf(`{"tool_use_id":"u1","path":%q}`, p))
		if err == nil {
			t.Fatalf("out = %q, want ErrNoTranscript", out)
		}
		if !errors.Is(err, ErrNoTranscript) {
			t.Errorf("err = %v, want it to match ErrNoTranscript", err)
		}
		if !strings.Contains(err.Error(), "write_file") {
			t.Errorf("err = %q, want it to name the alternative tool", err)
		}
		if _, statErr := os.Stat(p); statErr == nil {
			t.Error("the file was created despite the failure, want nothing written")
		}
	})

	t.Run("an unknown tool_use_id is reported with the id", func(t *testing.T) {
		ctx := tool.WithLookup(context.Background(), func(llm.ToolUseID) (string, error) {
			return "", errors.New("not in the transcript")
		})
		_, err := d.Fn(ctx, json.RawMessage(fmt.Sprintf(`{"tool_use_id":"toolu_missing","path":%q}`, filepath.Join(dir, "x"))))
		if err == nil {
			t.Fatal("err = nil, want the lookup failure")
		}
		if !strings.Contains(err.Error(), "toolu_missing") || !strings.Contains(err.Error(), "not in the transcript") {
			t.Errorf("err = %q, want it to name the id and carry the cause", err)
		}
	})

	t.Run("an empty tool_use_id is refused before the lookup", func(t *testing.T) {
		_, err := call(t, d, fmt.Sprintf(`{"tool_use_id":"","path":%q}`, filepath.Join(dir, "x")))
		if err == nil || err.Error() != "field 'tool_use_id' must not be empty" {
			t.Errorf("err = %v, want the empty-id message", err)
		}
	})

	// It ends in the same overwriting WriteFile as write_file, and shares its
	// budget so an identical payload cannot time out here and succeed there.
	t.Run("metadata matches write_file", func(t *testing.T) {
		w := WriteFile(OSFS(dir))
		if d.Idempotent {
			t.Error("Idempotent = true, want false")
		}
		if d.Timeout != w.Timeout {
			t.Errorf("Timeout = %v, want write_file's %v", d.Timeout, w.Timeout)
		}
	})
}

// ===== reconciling an interrupted call =====

func TestWriteFileReconcilesFromTheFile(t *testing.T) {
	dir := t.TempDir()
	d := WriteFile(OSFS(dir))
	p := filepath.Join(dir, "a.txt")
	in := json.RawMessage(fmt.Sprintf(`{"path":%q,"content":"new"}`, p))

	if _, landed, err := d.Reconcile(context.Background(), in); err != nil || landed {
		t.Errorf("no file yet: landed=%v err=%v, want not landed", landed, err)
	}
	if err := os.WriteFile(p, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, landed, err := d.Reconcile(context.Background(), in); err != nil || landed {
		t.Errorf("other content: landed=%v err=%v, want not landed", landed, err)
	}
	if err := os.WriteFile(p, []byte("new"), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, landed, err := d.Reconcile(context.Background(), in); err != nil || !landed || !strings.Contains(out, "wrote 3 bytes") {
		t.Errorf("the content it was to write: %q landed=%v err=%v, want landed", out, landed, err)
	}
}

func TestStrReplaceReconcilesFromTheFile(t *testing.T) {
	dir := t.TempDir()
	d := StrReplace(OSFS(dir))
	p := filepath.Join(dir, "a.go")
	edit := func(old, new string) json.RawMessage {
		return json.RawMessage(fmt.Sprintf(`{"path":%q,"old_str":%q,"new_str":%q}`, p, old, new))
	}
	for _, tc := range []struct {
		name, file, old, new string
		landed, cannot       bool
	}{
		{"old text still there once", "a := 1", "a := 1", "a := 2", false, false},
		{"replaced", "a := 2", "a := 1", "a := 2", true, false},
		{"deleted", "x\n", "y\n", "", true, false},
		{"new contains old", "foo bar", "foo", "foo bar", false, true},
		{"neither there", "z", "a := 1", "a := 2", false, true},
		{"old twice", "a a", "a", "b", false, true},
	} {
		if err := os.WriteFile(p, []byte(tc.file), 0o644); err != nil {
			t.Fatal(err)
		}
		_, landed, err := d.Reconcile(context.Background(), edit(tc.old, tc.new))
		if landed != tc.landed || (err != nil) != tc.cannot {
			t.Errorf("%s: landed=%v err=%v, want landed=%v cannot-tell=%v", tc.name, landed, err, tc.landed, tc.cannot)
		}
	}
}
