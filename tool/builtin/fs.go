package builtin

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"

	"github.com/hoobean1996/wombat-go/internal/fspath"
	"github.com/hoobean1996/wombat-go/tool"
)

// Kind is what a path is. The zero value is KindMissing, so a failed Stat and
// an absent file agree.
type Kind int

// Path kinds.
const (
	KindMissing Kind = iota
	KindFile
	KindDir
)

// String implements fmt.Stringer.
func (k Kind) String() string {
	switch k {
	case KindFile:
		return "file"
	case KindDir:
		return "dir"
	default:
		return "missing"
	}
}

// FS is the filesystem the file tools reach the world through.
//
// It is four methods rather than io/fs.FS because these tools write, and
// because they work in absolute paths: io/fs is rooted and read-only, and
// wedging "write /etc/hosts" into it buys nothing. Substituting a fake is a
// struct with four fields.
//
// Every method takes a context so that an implementation which is not local —
// a network filesystem, a sandbox daemon — can honour cancellation. [OSFS]
// only checks it before each syscall: a read of a local file is not
// interruptible anyway, and pretending otherwise would be theatre.
type FS interface {
	ReadFile(ctx context.Context, path string) ([]byte, error)
	WriteFile(ctx context.Context, path string, data []byte) error

	// Stat reports what path is. A missing path is (KindMissing, nil) — not
	// an error, because "does it exist" is the question being asked.
	Stat(ctx context.Context, path string) (Kind, error)

	// ListDir returns entry names (not paths), sorted.
	ListDir(ctx context.Context, path string) ([]string, error)
}

// ErrOutsideRoot is returned by a rooted [OSFS] for a path outside its root.
//
// A [tool.CallerFault]: the refusal IS the filesystem working. An agent
// exploring an unfamiliar tree will guess wrong about the boundary several
// times in a row, and the guesses must not add up to a verdict that the file
// reader is broken.
var ErrOutsideRoot = tool.CallerError(errors.New("builtin: path escapes the filesystem root"))

// retryFS classifies a filesystem failure, for the idempotent tools that read
// through an [FS].
//
// It is the FS-side counterpart of retryExec, and separate from it on purpose:
// the two reach the world by different routes, and what is transient on a
// spawn is not the same question as what is transient on a read. Three errnos
// qualify, and they are the ones the caller can do nothing about:
//
//   - EINTR — a signal landed mid-syscall; the read never happened.
//   - EAGAIN — the descriptor would block (a fifo, a slow network mount).
//   - ETXTBSY — the file is being written right now, typically by a build the
//     agent itself just kicked off.
//
// Everything else is deterministic and a retry only delays the same answer:
// ENOENT stays missing, EACCES stays forbidden, EISDIR stays a directory, and
// [ErrOutsideRoot] stays outside the root. They are excluded by the default
// rather than listed, because "not on the transient list" has to be the safe
// side of this decision — a classifier that guesses wrong about an unknown
// errno should decline to retry, not insist on it.
//
// Matching is errors.Is against the errno, never a substring of the message:
// os wraps errnos in *fs.PathError, which unwraps to the syscall.Errno, and
// the errno survives translation and locale while the message does not.
func retryFS(err error) bool {
	return errors.Is(err, syscall.EINTR) ||
		errors.Is(err, syscall.EAGAIN) ||
		errors.Is(err, syscall.ETXTBSY)
}

// OSFS returns an FS backed by the real filesystem.
//
// root == "" is unrestricted. Any other value confines every operation to
// that subtree.
//
// # The containment check is a UX guardrail, NOT a security boundary
//
// Confinement is a textual prefix test after filepath.Clean. Clean resolves
// "." and ".." lexically, so /root/../etc/passwd is rejected — but nothing
// here touches the disk to find out what a path really is. Symlinks are NOT
// resolved. A symlink anywhere under root pointing outside it WILL escape,
// and so will a hardlink, a bind mount, or a race between this check and the
// syscall that follows it (TOCTOU).
//
// Use it to stop an agent from casually rewriting /etc/hosts or ~/.bashrc
// during a dev run. Do NOT use it against an adversary, or against a model
// that has been prompt-injected into behaving like one. Real isolation is an
// OS-level facility: a container, a chroot, a mount namespace, a seccomp
// profile. If you need that, implement FS on top of one and pass it in —
// which is exactly why FS is an interface.
func OSFS(root string) FS {
	if root != "" {
		root = filepath.Clean(root)
	}
	return osFS{root: root}
}

type osFS struct{ root string }

// resolve cleans path and enforces containment. It returns the cleaned path,
// which is what the caller must then use: rejecting /proj/../etc but passing
// the uncleaned string through would leave the syscall and the check looking
// at different names.
func (f osFS) resolve(path string) (string, error) {
	clean := filepath.Clean(path)
	if f.root == "" {
		return clean, nil
	}
	if !fspath.Within(f.root, clean) {
		return "", fmt.Errorf("%w: %q is outside %q", ErrOutsideRoot, path, f.root)
	}
	return clean, nil
}

func (f osFS) ReadFile(ctx context.Context, path string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p, err := f.resolve(path)
	if err != nil {
		return nil, err
	}
	return os.ReadFile(p)
}

// WriteFile writes atomically: a sibling temp file, fsync, rename over the
// target, then fsync the parent directory.
//
// POSIX makes the rename atomic, so a reader — the next tool call, an editor,
// a build watching the tree — never observes a half-written file, and a crash
// leaves either the old content or the new. The directory fsync is what makes
// the rename itself durable; without it the metadata can be lost on power
// failure even after the file's own fsync.
//
// Missing parent directories are created. The model asked for a path, not for
// a lecture about mkdir -p, and the alternative is a two-call dance through
// bash.
func (f osFS) WriteFile(ctx context.Context, path string, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	p, err := f.resolve(path)
	if err != nil {
		return err
	}

	dir := filepath.Dir(p)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create parent of %s: %w", p, err)
	}

	// Preserve the mode of an existing file; a fresh one gets 0644 rather
	// than CreateTemp's 0600, which would otherwise make every file the
	// agent writes unreadable to the rest of the system.
	mode := os.FileMode(0o644)
	if st, serr := os.Stat(p); serr == nil {
		mode = st.Mode().Perm()
	}

	tmp, err := os.CreateTemp(dir, "."+filepath.Base(p)+".tmp*")
	if err != nil {
		return fmt.Errorf("create temp for %s: %w", p, err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename has succeeded

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write %s: %w", p, err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("sync %s: %w", p, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close %s: %w", p, err)
	}
	if err := os.Chmod(tmpName, mode); err != nil {
		return fmt.Errorf("chmod %s: %w", p, err)
	}
	if err := os.Rename(tmpName, p); err != nil {
		return fmt.Errorf("rename onto %s: %w", p, err)
	}
	syncDir(dir)
	return nil
}

func (f osFS) Stat(ctx context.Context, path string) (Kind, error) {
	if err := ctx.Err(); err != nil {
		return KindMissing, err
	}
	p, err := f.resolve(path)
	if err != nil {
		return KindMissing, err
	}
	st, err := os.Stat(p)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return KindMissing, nil
	case err != nil:
		return KindMissing, err
	case st.IsDir():
		return KindDir, nil
	default:
		return KindFile, nil
	}
}

func (f osFS) ListDir(ctx context.Context, path string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p, err := f.resolve(path)
	if err != nil {
		return nil, err
	}
	ents, err := os.ReadDir(p) // already sorted by name
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(ents))
	for _, e := range ents {
		name := e.Name()
		if e.IsDir() {
			name += "/" // the model should not have to stat to find out
		}
		names = append(names, name)
	}
	return names, nil
}

// syncDir flushes a directory entry. Best effort: some filesystems (and every
// Windows build) refuse to open a directory for reading, and a failure here
// costs durability, not correctness.
func syncDir(dir string) {
	d, err := os.Open(dir)
	if err != nil {
		return
	}
	defer d.Close()
	_ = d.Sync()
}
