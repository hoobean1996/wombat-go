package builtin

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/tool"
)

func TestKindString(t *testing.T) {
	tests := []struct {
		k    Kind
		want string
	}{
		{KindMissing, "missing"},
		{KindFile, "file"},
		{KindDir, "dir"},
		{Kind(99), "missing"},
	}
	for _, tt := range tests {
		if got := tt.k.String(); got != tt.want {
			t.Errorf("Kind(%d).String() = %q, want %q", tt.k, got, tt.want)
		}
	}
	// The zero value is KindMissing, so a failed Stat and an absent file agree.
	var zero Kind
	if zero != KindMissing {
		t.Errorf("zero Kind = %v, want KindMissing", zero)
	}
}

// ===== containment =====

func TestOSFSRootContainment(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "sub", "in.txt"), []byte("inside"), 0o644); err != nil {
		t.Fatal(err)
	}

	outsideDir := t.TempDir()
	outside := filepath.Join(outsideDir, "out.txt")
	if err := os.WriteFile(outside, []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}

	fsys := OSFS(root)
	ctx := context.Background()

	t.Run("a path inside the root works", func(t *testing.T) {
		got, err := fsys.ReadFile(ctx, filepath.Join(root, "sub", "in.txt"))
		if err != nil {
			t.Fatalf("ReadFile err = %v, want nil", err)
		}
		if string(got) != "inside" {
			t.Errorf("ReadFile = %q, want %q", got, "inside")
		}
	})

	t.Run("the root itself is inside the root", func(t *testing.T) {
		kind, err := fsys.Stat(ctx, root)
		if err != nil || kind != KindDir {
			t.Errorf("Stat(root) = (%v, %v), want (dir, nil)", kind, err)
		}
	})

	// Every method must enforce it; a hole in one of the four is a hole.
	t.Run("every method rejects a path outside the root", func(t *testing.T) {
		checks := []struct {
			name string
			run  func(string) error
		}{
			{"ReadFile", func(p string) error { _, err := fsys.ReadFile(ctx, p); return err }},
			{"WriteFile", func(p string) error { return fsys.WriteFile(ctx, p, []byte("x")) }},
			{"Stat", func(p string) error { _, err := fsys.Stat(ctx, p); return err }},
			{"ListDir", func(p string) error { _, err := fsys.ListDir(ctx, p); return err }},
		}
		for _, c := range checks {
			err := c.run(outside)
			if !errors.Is(err, ErrOutsideRoot) {
				t.Errorf("%s(%q) err = %v, want ErrOutsideRoot", c.name, outside, err)
			}
			if err != nil && !strings.Contains(err.Error(), root) {
				t.Errorf("%s err = %q, want it to name the root %q", c.name, err, root)
			}
		}
		// The escape attempt must not have written anything.
		if data, _ := os.ReadFile(outside); string(data) != "outside" {
			t.Errorf("the file outside the root was modified: %q, want %q", data, "outside")
		}
	})

	// Clean resolves ".." lexically, which is what makes the prefix test work.
	t.Run("a lexical .. escape is rejected", func(t *testing.T) {
		for _, p := range []string{
			root + "/sub/../../escaped.txt",
			root + "/../escaped.txt",
			root + "/./../escaped.txt",
		} {
			if _, err := fsys.ReadFile(ctx, p); !errors.Is(err, ErrOutsideRoot) {
				t.Errorf("ReadFile(%q) err = %v, want ErrOutsideRoot", p, err)
			}
		}
	})

	// A .. that stays inside is fine: the check is on the cleaned path.
	t.Run("a .. that lands back inside is allowed", func(t *testing.T) {
		got, err := fsys.ReadFile(ctx, root+"/sub/../sub/in.txt")
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if string(got) != "inside" {
			t.Errorf("ReadFile = %q, want %q", got, "inside")
		}
	})

	// The separator in the prefix test is load-bearing: without it "/work/proj"
	// would contain "/work/project-secrets".
	t.Run("a sibling sharing the root's string prefix is rejected", func(t *testing.T) {
		sibling := root + "-secrets/file.txt"
		if _, err := fsys.ReadFile(ctx, sibling); !errors.Is(err, ErrOutsideRoot) {
			t.Errorf("ReadFile(%q) err = %v, want ErrOutsideRoot", sibling, err)
		}
	})

	t.Run("root == \"\" is unrestricted", func(t *testing.T) {
		got, err := OSFS("").ReadFile(ctx, outside)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if string(got) != "outside" {
			t.Errorf("ReadFile = %q, want %q", got, "outside")
		}
	})

	t.Run("the root is cleaned at construction", func(t *testing.T) {
		messy := OSFS(root + "/sub/..")
		if _, err := messy.ReadFile(ctx, filepath.Join(root, "sub", "in.txt")); err != nil {
			t.Errorf("err = %v, want nil: %q should clean to the root", err, root+"/sub/..")
		}
	})

	// A regression test. The containment check was root+"/" as a prefix, which
	// for a root of "/" is "//" — so OSFS("/") refused every path but "/"
	// itself, reporting /etc/hosts as outside "/".
	t.Run("a root of / contains every absolute path", func(t *testing.T) {
		slash := OSFS(string(filepath.Separator))
		target := filepath.Join(root, "sub", "in.txt")
		got, err := slash.ReadFile(ctx, target)
		if err != nil {
			t.Fatalf("OSFS(/).ReadFile(%s) = %v, want the file", target, err)
		}
		if string(got) != "inside" {
			t.Errorf("ReadFile = %q, want %q", got, "inside")
		}
		if k, err := slash.Stat(ctx, root); err != nil || k != KindDir {
			t.Errorf("OSFS(/).Stat(%s) = %v, %v; want dir, nil", root, k, err)
		}
		if err := slash.WriteFile(ctx, filepath.Join(root, "new.txt"), []byte("x")); err != nil {
			t.Errorf("OSFS(/).WriteFile = %v, want nil", err)
		}
	})
}

// ===== atomic write =====

func TestOSFSWriteFileIsAtomicAndPreservesMode(t *testing.T) {
	dir := t.TempDir()
	fsys := OSFS(dir)
	ctx := context.Background()

	t.Run("preserves the mode of an existing file", func(t *testing.T) {
		p := filepath.Join(dir, "script.sh")
		if err := os.WriteFile(p, []byte("old"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(p, 0o750); err != nil { // beat the umask
			t.Fatal(err)
		}

		if err := fsys.WriteFile(ctx, p, []byte("new content")); err != nil {
			t.Fatalf("WriteFile err = %v, want nil", err)
		}

		st, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if got := st.Mode().Perm(); got != 0o750 {
			t.Errorf("mode = %o, want 0750 preserved across the rename", got)
		}
		if data, _ := os.ReadFile(p); string(data) != "new content" {
			t.Errorf("content = %q, want %q", data, "new content")
		}
	})

	// CreateTemp makes 0600, which would leave every file the agent writes
	// unreadable to the rest of the system.
	t.Run("a fresh file gets 0644, not CreateTemp's 0600", func(t *testing.T) {
		p := filepath.Join(dir, "fresh.txt")
		if err := fsys.WriteFile(ctx, p, []byte("hello")); err != nil {
			t.Fatalf("WriteFile err = %v, want nil", err)
		}
		st, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if got := st.Mode().Perm(); got != 0o644 {
			t.Errorf("mode = %o, want 0644", got)
		}
	})

	t.Run("missing parents are created", func(t *testing.T) {
		p := filepath.Join(dir, "a", "b", "c", "deep.txt")
		if err := fsys.WriteFile(ctx, p, []byte("deep")); err != nil {
			t.Fatalf("WriteFile err = %v, want nil", err)
		}
		if data, err := os.ReadFile(p); err != nil || string(data) != "deep" {
			t.Errorf("ReadFile = (%q, %v), want (deep, nil)", data, err)
		}
	})

	t.Run("the temp file is not left behind", func(t *testing.T) {
		sub := filepath.Join(dir, "clean")
		p := filepath.Join(sub, "only.txt")
		if err := fsys.WriteFile(ctx, p, []byte("x")); err != nil {
			t.Fatal(err)
		}
		ents, err := os.ReadDir(sub)
		if err != nil {
			t.Fatal(err)
		}
		if len(ents) != 1 || ents[0].Name() != "only.txt" {
			var got []string
			for _, e := range ents {
				got = append(got, e.Name())
			}
			t.Errorf("directory contains %v, want just [only.txt] — a temp file leaked", got)
		}
	})

	t.Run("an empty write truncates", func(t *testing.T) {
		p := filepath.Join(dir, "empty.txt")
		if err := fsys.WriteFile(ctx, p, []byte("content")); err != nil {
			t.Fatal(err)
		}
		if err := fsys.WriteFile(ctx, p, nil); err != nil {
			t.Fatal(err)
		}
		if data, _ := os.ReadFile(p); len(data) != 0 {
			t.Errorf("content = %q, want empty", data)
		}
	})
}

// ===== Stat / ListDir =====

func TestOSFSStat(t *testing.T) {
	dir := t.TempDir()
	fsys := OSFS(dir)
	ctx := context.Background()

	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "d"), 0o755); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		path string
		want Kind
	}{
		{"file", filepath.Join(dir, "f.txt"), KindFile},
		{"dir", filepath.Join(dir, "d"), KindDir},
		// "does it exist" is the question being asked, so absence is an answer.
		{"missing", filepath.Join(dir, "nope"), KindMissing},
		{"missing under a missing parent", filepath.Join(dir, "no", "such", "path"), KindMissing},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := fsys.Stat(ctx, tt.path)
			if err != nil {
				t.Fatalf("Stat err = %v, want nil", err)
			}
			if got != tt.want {
				t.Errorf("Stat = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOSFSListDir(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"zebra.txt", "apple.txt", "middle.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "subdir"), 0o755); err != nil {
		t.Fatal(err)
	}

	got, err := OSFS(dir).ListDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("ListDir err = %v, want nil", err)
	}
	// Sorted by name, with a trailing slash on directories so the model does
	// not have to stat to find out.
	want := []string{"apple.txt", "middle.txt", "subdir/", "zebra.txt"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("ListDir = %v, want %v", got, want)
	}

	t.Run("a missing directory is an error, unlike Stat", func(t *testing.T) {
		_, err := OSFS(dir).ListDir(context.Background(), filepath.Join(dir, "nope"))
		if !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("err = %v, want fs.ErrNotExist", err)
		}
	})

	t.Run("an empty directory lists nothing", func(t *testing.T) {
		empty := t.TempDir()
		got, err := OSFS("").ListDir(context.Background(), empty)
		if err != nil || len(got) != 0 {
			t.Errorf("ListDir = (%v, %v), want ([], nil)", got, err)
		}
	})
}

// Cancellation is checked before each syscall, so a torn-down run does not
// keep hitting the disk.
func TestOSFSHonoursACancelledContext(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	fsys := OSFS(dir)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	checks := map[string]func() error{
		"ReadFile":  func() error { _, err := fsys.ReadFile(ctx, p); return err },
		"WriteFile": func() error { return fsys.WriteFile(ctx, p, []byte("y")) },
		"Stat":      func() error { _, err := fsys.Stat(ctx, p); return err },
		"ListDir":   func() error { _, err := fsys.ListDir(ctx, dir); return err },
	}
	for name, run := range checks {
		if err := run(); !errors.Is(err, context.Canceled) {
			t.Errorf("%s err = %v, want context.Canceled", name, err)
		}
	}
	if data, _ := os.ReadFile(p); string(data) != "x" {
		t.Errorf("the file was written despite cancellation: %q, want %q", data, "x")
	}
}

// ===== retry classifiers =====

// TestRetryClassifiers pins the errno matching. It is errors.Is against the
// errno and never a substring of the message: os wraps errnos in
// *fs.PathError, and the errno survives translation and locale while the
// message does not.
func TestRetryClassifiers(t *testing.T) {
	classifiers := map[string]func(error) bool{
		"retryFS":   retryFS,
		"retryExec": retryExec,
	}

	transient := []struct {
		name string
		err  error
	}{
		{"EINTR", syscall.EINTR},
		{"EAGAIN", syscall.EAGAIN},
		{"ETXTBSY", syscall.ETXTBSY},
	}
	// "Not on the transient list" has to be the safe side of this decision: a
	// retry only delays the same answer, and the file stays missing.
	permanent := []struct {
		name string
		err  error
	}{
		{"ENOENT", syscall.ENOENT},
		{"EACCES", syscall.EACCES},
		{"EISDIR", syscall.EISDIR},
		{"EPERM", syscall.EPERM},
		{"ENOTDIR", syscall.ENOTDIR},
		{"ErrOutsideRoot", ErrOutsideRoot},
		{"fs.ErrNotExist", fs.ErrNotExist},
		{"a plain error", errors.New("something went wrong")},
		{"nil", nil},
	}

	for cname, classify := range classifiers {
		for _, tt := range transient {
			t.Run(cname+"/"+tt.name+" is transient", func(t *testing.T) {
				if !classify(tt.err) {
					t.Errorf("%s(%v) = false, want true", cname, tt.err)
				}
				// The real shape: os wraps the errno in a *fs.PathError.
				wrapped := &fs.PathError{Op: "open", Path: "/x", Err: tt.err}
				if !classify(wrapped) {
					t.Errorf("%s(*fs.PathError{%v}) = false, want true — errors.Is must unwrap to the errno", cname, tt.err)
				}
				// And a Runner may wrap it again.
				if !classify(&sandbox.CommandError{Name: "grep", Err: wrapped}) {
					t.Errorf("%s(commandError{PathError{%v}}) = false, want true", cname, tt.err)
				}
			})
		}
		for _, tt := range permanent {
			t.Run(cname+"/"+tt.name+" is not transient", func(t *testing.T) {
				if classify(tt.err) {
					t.Errorf("%s(%v) = true, want false", cname, tt.err)
				}
				if tt.err != nil {
					wrapped := &fs.PathError{Op: "open", Path: "/x", Err: tt.err}
					if classify(wrapped) {
						t.Errorf("%s(*fs.PathError{%v}) = true, want false", cname, tt.err)
					}
				}
			})
		}
	}
}

// A real ENOENT from the OS must not be classified as retryable, which is what
// the errno matching above is actually protecting.
func TestRetryFSAgainstARealMissingFile(t *testing.T) {
	_, err := os.ReadFile(filepath.Join(t.TempDir(), "definitely-not-here"))
	if err == nil {
		t.Fatal("os.ReadFile of a missing file returned nil, want an error")
	}
	if !errors.Is(err, syscall.ENOENT) {
		t.Fatalf("err = %v, want it to unwrap to ENOENT", err)
	}
	if retryFS(err) {
		t.Errorf("retryFS(%v) = true, want false: a missing file stays missing", err)
	}
}

// TestOutsideRootIsCallerFault: the refusal is the filesystem working.
//
// An agent exploring an unfamiliar tree guesses wrong about the boundary
// several times in a row — that is what exploration looks like — and the
// guesses must not add up to a verdict that the file reader is broken.
func TestOutsideRootIsCallerFault(t *testing.T) {
	f := OSFS("/proj")
	_, err := f.ReadFile(context.Background(), "/proj/../etc/passwd")
	if !errors.Is(err, ErrOutsideRoot) {
		t.Fatalf("err = %v, want ErrOutsideRoot", err)
	}
	if !tool.IsCallerFault(err) {
		t.Errorf("IsCallerFault(%v) = false, want true", err)
	}
}
