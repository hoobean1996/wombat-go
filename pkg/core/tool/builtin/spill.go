package builtin

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Spill keeps the complete output of a command that was too long to return,
// so the model can read the part it needs instead of losing it.
//
// bash returns at most a few kilobytes — the start and the end, because a
// failing test run's assertion is at the end. Everything in between used to be
// gone: a model that needed the one stack frame in the middle of a 200 KB log
// could only run the command again and pipe it through grep, if it thought to,
// and pay for the whole run a second time. With a Spill the clipped result
// says where the whole output is, and grep, sed -n or head over that file
// costs nothing.
//
// The files live in one directory and the oldest are removed past a count, so
// a long run cannot fill a disk with logs. They are not removed when the run
// ends, because a transcript can still point at one when the conversation is
// resumed.
type Spill struct {
	dir  string
	keep int

	mu   sync.Mutex
	made bool
	seq  atomic.Uint64
}

// DefaultSpillKeep is how many spilled outputs a [Spill] keeps.
const DefaultSpillKeep = 100

// NewSpill keeps outputs in dir — the system temp directory's wombat-output
// when dir is "" — and at most keep of them, DefaultSpillKeep when keep <= 0.
func NewSpill(dir string, keep int) *Spill {
	if dir == "" {
		dir = filepath.Join(os.TempDir(), "wombat-output")
	}
	if keep <= 0 {
		keep = DefaultSpillKeep
	}
	return &Spill{dir: dir, keep: keep}
}

// save writes out under a new name and returns its path.
func (s *Spill) save(name, out string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.made {
		if err := os.MkdirAll(s.dir, 0o700); err != nil {
			return "", err
		}
		s.made = true
	}
	path := filepath.Join(s.dir, fmt.Sprintf("%s-%s-%d.log", name, time.Now().Format("20060102-150405"), s.seq.Add(1)))
	if err := os.WriteFile(path, []byte(out), 0o600); err != nil {
		return "", err
	}
	s.prune()
	return path, nil
}

// prune removes the oldest spilled files past keep. Best effort: a file that
// cannot be removed is left, and the next save tries again.
func (s *Spill) prune() {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return
	}
	type file struct {
		path string
		mod  time.Time
	}
	var files []file
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".log") {
			continue
		}
		if info, err := e.Info(); err == nil {
			files = append(files, file{filepath.Join(s.dir, e.Name()), info.ModTime()})
		}
	}
	if len(files) <= s.keep {
		return
	}
	slices.SortFunc(files, func(a, b file) int { return a.mod.Compare(b.mod) })
	for _, f := range files[:len(files)-s.keep] {
		_ = os.Remove(f.path)
	}
}

// clipAndSpill returns out cut to max bytes by clip, plus — when it had to cut
// and a spill is configured — where the whole of it was saved.
func clipAndSpill(s *Spill, name, out string, max int, clip func(string, int) string) string {
	if len(out) <= max || s == nil {
		return clip(out, max)
	}
	path, err := s.save(name, out)
	if err != nil {
		return clip(out, max)
	}
	note := fmt.Sprintf("\n[output clipped: the complete %d bytes are saved in %s — read the part you need with grep, sed -n or tail instead of running the command again]", len(out), path)
	return clip(out, max-len(note)) + note
}
