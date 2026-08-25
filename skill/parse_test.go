package skill

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParse(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		source   string
		wantName string
		wantDesc string
		wantBody string
	}{
		{
			name:     "plain scalars",
			text:     frontmatter("name: demo\ndescription: A demo skill.", "# Demo\n\nBody text.\n"),
			wantName: "demo",
			wantDesc: "A demo skill.",
			wantBody: "# Demo\n\nBody text.",
		},
		{
			name: "literal block scalar keeps its newlines",
			text: frontmatter(
				"name: pdf-forms\ndescription: |\n  Filling and flattening AcroForm PDFs.\n  Use when the task mentions a fillable PDF.",
				"body"),
			wantName: "pdf-forms",
			wantDesc: "Filling and flattening AcroForm PDFs.\nUse when the task mentions a fillable PDF.",
			wantBody: "body",
		},
		{
			name: "folded block scalar joins with spaces",
			text: frontmatter(
				"name: sql\ndescription: >\n  Reading EXPLAIN output\n  and fixing slow queries.",
				"body"),
			wantName: "sql",
			wantDesc: "Reading EXPLAIN output and fixing slow queries.",
			wantBody: "body",
		},
		{
			name: "a blank line inside a block is a paragraph break",
			text: frontmatter(
				"name: x\ndescription: |\n  Para one.\n\n  Para two.",
				"body"),
			wantName: "x",
			wantDesc: "Para one.\n\nPara two.",
			wantBody: "body",
		},
		{
			name:     "trailing blank lines in a block are trimmed",
			text:     frontmatter("name: x\ndescription: |\n  Only line.\n\n", "body"),
			wantName: "x",
			wantDesc: "Only line.",
			wantBody: "body",
		},
		{
			name: "a block keeps relative indentation",
			text: frontmatter(
				"name: x\ndescription: |\n  Steps:\n    - one\n    - two",
				"body"),
			wantName: "x",
			wantDesc: "Steps:\n  - one\n  - two",
			wantBody: "body",
		},
		{
			name:     "double quotes are stripped",
			text:     frontmatter(`name: "demo"`+"\n"+`description: "A quoted description."`, "body"),
			wantName: "demo",
			wantDesc: "A quoted description.",
			wantBody: "body",
		},
		{
			name:     "single quotes are stripped",
			text:     frontmatter("name: 'demo'\ndescription: 'A quoted description.'", "body"),
			wantName: "demo",
			wantDesc: "A quoted description.",
			wantBody: "body",
		},
		{
			name:     "unbalanced quotes are left alone",
			text:     frontmatter(`name: demo`+"\n"+`description: "half quoted`, "body"),
			wantName: "demo",
			wantDesc: `"half quoted`,
			wantBody: "body",
		},
		{
			// SKILL.md files are shared with other harnesses that add their own
			// keys; failing on one we do not need would make this package the
			// reason a perfectly good skill cannot be used.
			name: "unknown keys are ignored",
			text: frontmatter(
				"name: demo\nversion: 3\nallowed-tools: bash, grep\nlicense: MIT\ndescription: A demo skill.",
				"body"),
			wantName: "demo",
			wantDesc: "A demo skill.",
			wantBody: "body",
		},
		{
			name:     "a value may contain colons",
			text:     frontmatter("name: demo\ndescription: Use when: the PDF is fillable.", "body"),
			wantName: "demo",
			wantDesc: "Use when: the PDF is fillable.",
			wantBody: "body",
		},
		{
			name:     "comments and blank lines are skipped",
			text:     frontmatter("# a comment\n\nname: demo\n\n  # indented comment\ndescription: A demo skill.", "body"),
			wantName: "demo",
			wantDesc: "A demo skill.",
			wantBody: "body",
		},
		{
			name:     "a repeated key takes the last value",
			text:     frontmatter("name: first\nname: second\ndescription: A demo skill.", "body"),
			wantName: "second",
			wantDesc: "A demo skill.",
			wantBody: "body",
		},
		{
			// An editor-written BOM would otherwise make the first line
			// "<BOM>---" and fail the delimiter check for a reason nobody can
			// see in a diff.
			name:     "a leading BOM is stripped",
			text:     "\ufeff" + frontmatter("name: demo\ndescription: A demo skill.", "body"),
			wantName: "demo",
			wantDesc: "A demo skill.",
			wantBody: "body",
		},
		{
			name: "CRLF line endings are normalized",
			text: strings.ReplaceAll(
				frontmatter("name: demo\ndescription: |\n  One.\n  Two.", "Body line.\n"), "\n", "\r\n"),
			wantName: "demo",
			wantDesc: "One.\nTwo.",
			wantBody: "Body line.",
		},
		{
			name:     "a BOM in front of CRLF frontmatter",
			text:     "\ufeff" + strings.ReplaceAll(frontmatter("name: demo\ndescription: A demo skill.", "body"), "\n", "\r\n"),
			wantName: "demo",
			wantDesc: "A demo skill.",
			wantBody: "body",
		},
		{
			name:     "the delimiter may carry trailing whitespace",
			text:     "---  \nname: demo\ndescription: A demo skill.\n---\t\nbody",
			wantName: "demo",
			wantDesc: "A demo skill.",
			wantBody: "body",
		},
		{
			// The on-disk layout already names the skill; requiring the author
			// to repeat it buys nothing but a chance to disagree with itself.
			name:     "a missing name falls back to the directory",
			text:     frontmatter("description: A demo skill.", "body"),
			source:   filepath.Join("skills", "pdf-forms", FileName),
			wantName: "pdf-forms",
			wantDesc: "A demo skill.",
			wantBody: "body",
		},
		{
			name:     "an explicit name beats the directory",
			text:     frontmatter("name: real-name\ndescription: A demo skill.", "body"),
			source:   filepath.Join("skills", "dir-name", FileName),
			wantName: "real-name",
			wantDesc: "A demo skill.",
			wantBody: "body",
		},
		{
			name:     "an empty body is fine",
			text:     frontmatter("name: demo\ndescription: A demo skill.", "   \n\n"),
			wantName: "demo",
			wantDesc: "A demo skill.",
			wantBody: "",
		},
		{
			name:     "the body may itself contain a --- rule",
			text:     frontmatter("name: demo\ndescription: A demo skill.", "before\n\n---\n\nafter\n"),
			wantName: "demo",
			wantDesc: "A demo skill.",
			wantBody: "before\n\n---\n\nafter",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Parse(tt.text, tt.source)
			if err != nil {
				t.Fatalf("Parse() error = %v, want nil", err)
			}
			if got.Name != tt.wantName {
				t.Errorf("Name = %q, want %q", got.Name, tt.wantName)
			}
			if got.Description != tt.wantDesc {
				t.Errorf("Description = %q, want %q", got.Description, tt.wantDesc)
			}
			if got.Body != tt.wantBody {
				t.Errorf("Body = %q, want %q", got.Body, tt.wantBody)
			}
			if got.Source != tt.source {
				t.Errorf("Source = %q, want %q", got.Source, tt.source)
			}
		})
	}
}

func TestParseErrors(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		source  string
		want    error
		wantMsg string // a substring the message must carry
	}{
		{
			name:    "no frontmatter at all",
			text:    "no frontmatter here\n",
			want:    ErrNoFrontmatter,
			wantMsg: "<memory>",
		},
		{
			name: "empty file",
			text: "",
			want: ErrNoFrontmatter,
		},
		{
			name: "frontmatter is not the first line",
			text: "# Title\n" + frontmatter("name: demo\ndescription: d", "body"),
			want: ErrNoFrontmatter,
		},
		{
			name:    "unterminated frontmatter",
			text:    "---\nname: demo\ndescription: A demo skill.\n",
			want:    ErrNoFrontmatter,
			wantMsg: "no closing",
		},
		{
			name:    "a line that is not key: value",
			text:    frontmatter("name: demo\nthis line has no colon\ndescription: d", "body"),
			want:    ErrBadFrontmatter,
			wantMsg: "this line has no colon",
		},
		{
			// The grammar is a fixed subset: block sequences are not in it, and
			// the continuation line trips the "no colon" check. Pinned because
			// it is the most likely thing an author reaches for.
			name: "YAML block sequences are not supported",
			text: frontmatter("name: demo\ndescription: d\ntools:\n  - bash", "body"),
			want: ErrBadFrontmatter,
		},
		{
			name:    "empty key",
			text:    frontmatter("name: demo\n: orphan\ndescription: d", "body"),
			want:    ErrBadFrontmatter,
			wantMsg: "empty key",
		},
		{
			name: "no name and no path to fall back to",
			text: frontmatter("description: A demo skill.", "body"),
			want: ErrNoName,
		},
		{
			name:   "no name and a path with no directory",
			text:   frontmatter("description: A demo skill.", "body"),
			source: FileName,
			want:   ErrNoName,
		},
		{
			// Fatal rather than defaulted: the description is the ONLY thing
			// the model sees before deciding to load, so a skill without one is
			// either never loaded or loaded at random.
			name:    "no description",
			text:    frontmatter("name: demo", "body"),
			want:    ErrNoDescription,
			wantMsg: `"demo"`,
		},
		{
			name: "a whitespace-only description",
			text: frontmatter("name: demo\ndescription:    ", "body"),
			want: ErrNoDescription,
		},
		{
			name: "an empty block scalar description",
			text: frontmatter("name: demo\ndescription: |", "body"),
			want: ErrNoDescription,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Parse(tt.text, tt.source)
			if !errors.Is(err, tt.want) {
				t.Fatalf("Parse() error = %v, want one wrapping %v", err, tt.want)
			}
			if tt.wantMsg != "" && !strings.Contains(err.Error(), tt.wantMsg) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tt.wantMsg)
			}
			if got != (Skill{}) {
				t.Errorf("Parse() skill = %+v, want the zero Skill on error", got)
			}
		})
	}
}

// TestParseNamesTheFile: a parse error is read by an operator staring at a
// directory of user content, so it has to say which file was at fault.
func TestParseNamesTheFile(t *testing.T) {
	path := filepath.Join("skills", "broken", FileName)
	_, err := Parse("nope\n", path)
	if err == nil {
		t.Fatal("Parse() error = nil, want ErrNoFrontmatter")
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("error = %q, want it to name %q", err.Error(), path)
	}
}

func TestParseFile(t *testing.T) {
	root := t.TempDir()
	path := mkSkill(t, root, "demo", frontmatter("name: demo\ndescription: A demo skill.", demoBody))

	got, err := ParseFile(path)
	if err != nil {
		t.Fatalf("ParseFile() error = %v, want nil", err)
	}
	if got.Name != "demo" || got.Body != demoBody {
		t.Errorf("ParseFile() = %+v, want name demo with the demo body", got)
	}
	if got.Source != path {
		t.Errorf("Source = %q, want %q", got.Source, path)
	}

	t.Run("missing file", func(t *testing.T) {
		_, err := ParseFile(filepath.Join(root, "nope", FileName))
		if err == nil {
			t.Fatal("ParseFile() error = nil, want a read error")
		}
		if !errors.Is(err, os.ErrNotExist) {
			t.Errorf("error = %v, want one wrapping os.ErrNotExist", err)
		}
	})
}

// ===== LoadDir =====

func TestLoadDir(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "zebra", frontmatter("name: zebra\ndescription: Last alphabetically.", "z body"))
	mkSkill(t, root, "demo", frontmatter("name: demo\ndescription: A demo skill.", demoBody))
	mkSkill(t, root, "alpha", frontmatter("description: Named after its directory.", "a body"))

	// A directory with no SKILL.md is not a skill at all: silent, not an error.
	if err := os.MkdirAll(filepath.Join(root, "notaskill"), 0o755); err != nil {
		t.Fatal(err)
	}
	// A loose file at the root is not a skill either.
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}

	var reported []error
	got, err := LoadDir(root, func(e error) { reported = append(reported, e) })
	if err != nil {
		t.Fatalf("LoadDir() error = %v, want nil", err)
	}
	if len(reported) != 0 {
		t.Errorf("onError called %d times (%v), want 0", len(reported), reported)
	}

	var gotNames []string
	for _, s := range got {
		gotNames = append(gotNames, s.Name)
	}
	want := []string{"alpha", "demo", "zebra"}
	if strings.Join(gotNames, ",") != strings.Join(want, ",") {
		t.Errorf("names = %v, want %v (sorted by name)", gotNames, want)
	}
	if got[1].Body != demoBody {
		t.Errorf("demo body = %q, want %q", got[1].Body, demoBody)
	}
}

// TestLoadDirSkipsBroken: skills are user-authored content dropped into a
// directory. Refusing to start the agent because the seventeenth one has a
// typo punishes the wrong person.
func TestLoadDirSkipsBroken(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "good", frontmatter("name: good\ndescription: Fine.", "body"))
	mkSkill(t, root, "broken", "no frontmatter here\n")
	mkSkill(t, root, "nodesc", frontmatter("name: nodesc", "body"))

	var reported []error
	got, err := LoadDir(root, func(e error) { reported = append(reported, e) })
	if err != nil {
		t.Fatalf("LoadDir() error = %v, want nil: one broken skill must not stall startup", err)
	}
	if len(got) != 1 || got[0].Name != "good" {
		t.Fatalf("LoadDir() = %v, want just the good skill", got)
	}
	if len(reported) != 2 {
		t.Fatalf("onError called %d times, want 2 (broken + nodesc): %v", len(reported), reported)
	}
	for _, e := range reported {
		if !strings.Contains(e.Error(), "skipping") {
			t.Errorf("reported error = %q, want it to say the skill was skipped", e)
		}
	}
	if !errors.Is(reported[0], ErrNoFrontmatter) {
		t.Errorf("reported[0] = %v, want one wrapping %v", reported[0], ErrNoFrontmatter)
	}
	if !errors.Is(reported[1], ErrNoDescription) {
		t.Errorf("reported[1] = %v, want one wrapping %v", reported[1], ErrNoDescription)
	}
}

func TestLoadDirDuplicateNames(t *testing.T) {
	root := t.TempDir()
	// Two directories declaring the same frontmatter name. os.ReadDir is
	// sorted, so "a-first" is the one that survives.
	mkSkill(t, root, "a-first", frontmatter("name: dup\ndescription: The winner.", "first"))
	mkSkill(t, root, "b-second", frontmatter("name: dup\ndescription: The loser.", "second"))

	var reported []error
	got, err := LoadDir(root, func(e error) { reported = append(reported, e) })
	if err != nil {
		t.Fatalf("LoadDir() error = %v, want nil", err)
	}
	if len(got) != 1 {
		t.Fatalf("LoadDir() returned %d skills, want 1", len(got))
	}
	if got[0].Body != "first" {
		t.Errorf("kept the body %q, want %q (first wins)", got[0].Body, "first")
	}
	if len(reported) != 1 {
		t.Fatalf("onError called %d times, want 1: %v", len(reported), reported)
	}
	if !strings.Contains(reported[0].Error(), "duplicate skill name") {
		t.Errorf("reported = %q, want it to mention a duplicate skill name", reported[0])
	}
}

// TestLoadDirMissingRoot: no skills directory means "no skills configured",
// which is the common case for an agent that ships without any.
func TestLoadDirMissingRoot(t *testing.T) {
	got, err := LoadDir(filepath.Join(t.TempDir(), "does-not-exist"), func(e error) {
		t.Errorf("onError(%v), want no report for a missing root", e)
	})
	if err != nil {
		t.Errorf("LoadDir() error = %v, want nil for a missing root", err)
	}
	if got != nil {
		t.Errorf("LoadDir() = %v, want nil", got)
	}
}

// TestLoadDirUnreadableRoot: a root that exists but cannot be read IS an
// error, because that is a misconfiguration the operator wants to hear about.
func TestLoadDirUnreadableRoot(t *testing.T) {
	root := t.TempDir()
	notADir := filepath.Join(root, "skills.txt")
	if err := os.WriteFile(notADir, []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := LoadDir(notADir, nil)
	if err == nil {
		t.Fatalf("LoadDir(%q) error = nil, want a scan error", notADir)
	}
	if errors.Is(err, os.ErrNotExist) {
		t.Errorf("error = %v, want something other than ErrNotExist: the path exists", err)
	}
	if !strings.Contains(err.Error(), notADir) {
		t.Errorf("error = %q, want it to name the root", err)
	}
	if got != nil {
		t.Errorf("LoadDir() = %v, want nil on error", got)
	}
}

func TestLoadDirNilOnError(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "broken", "no frontmatter here\n")
	mkSkill(t, root, "good", frontmatter("name: good\ndescription: Fine.", "body"))

	got, err := LoadDir(root, nil) // must not panic
	if err != nil {
		t.Fatalf("LoadDir() error = %v, want nil", err)
	}
	if len(got) != 1 || got[0].Name != "good" {
		t.Errorf("LoadDir() = %v, want just the good skill", got)
	}
}

func TestLoadDirEmptyRoot(t *testing.T) {
	got, err := LoadDir(t.TempDir(), nil)
	if err != nil {
		t.Fatalf("LoadDir() error = %v, want nil", err)
	}
	if len(got) != 0 {
		t.Errorf("LoadDir() = %v, want no skills", got)
	}
}

// ===== unexported helpers =====

func TestFirstLine(t *testing.T) {
	tests := []struct{ in, want string }{
		{"one line", "one line"},
		{"first\nsecond", "first"},
		{"\n\n  padded  \nsecond", "padded"},
		{"", ""},
		{"\n\n", ""},
		{"   \n real ", "real"},
	}
	for _, tt := range tests {
		if got := firstLine(tt.in); got != tt.want {
			t.Errorf("firstLine(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestUnquote(t *testing.T) {
	tests := []struct{ in, want string }{
		{`"x"`, "x"},
		{`'x'`, "x"},
		{`x`, "x"},
		{``, ``},
		{`"`, `"`},
		{`""`, ``},
		{`"mixed'`, `"mixed'`},
		{`"outer "inner""`, `outer "inner"`},
		{`he said "hi"`, `he said "hi"`},
	}
	for _, tt := range tests {
		if got := unquote(tt.in); got != tt.want {
			t.Errorf("unquote(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestReadBlock(t *testing.T) {
	tests := []struct {
		name      string
		lines     []string
		start     int
		wantText  string
		wantAfter int
	}{
		{
			name:      "consumes the indented run and stops at the dedent",
			lines:     []string{"description: |", "  one", "  two", "name: demo"},
			start:     1,
			wantText:  "one\ntwo",
			wantAfter: 3,
		},
		{
			name:      "an empty block",
			lines:     []string{"description: |", "name: demo"},
			start:     1,
			wantText:  "",
			wantAfter: 1,
		},
		{
			name:      "runs to the end of the frontmatter",
			lines:     []string{"description: |", "  only"},
			start:     1,
			wantText:  "only",
			wantAfter: 2,
		},
		{
			name:      "a tab-indented continuation",
			lines:     []string{"description: |", "\tone", "\ttwo"},
			start:     1,
			wantText:  "one\ntwo",
			wantAfter: 3,
		},
		{
			name:      "leading blanks are separators, not content",
			lines:     []string{"description: |", "", "  one"},
			start:     1,
			wantText:  "one",
			wantAfter: 3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, after := readBlock(tt.lines, tt.start)
			if text != tt.wantText {
				t.Errorf("readBlock text = %q, want %q", text, tt.wantText)
			}
			if after != tt.wantAfter {
				t.Errorf("readBlock next = %d, want %d", after, tt.wantAfter)
			}
		})
	}
}
