package httpapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

// Store persists sessions, so a server that restarts picks up the
// conversations it had instead of losing every one of them — and every
// question a model had asked and was waiting on.
//
// What survives is the conversation: the transcript, where it stands, what it
// cost. What does not is anything that only exists while a process runs —
// the event log a stream resumes from, a turn in flight, a tool call parked on
// an approval. A turn the restart interrupted is restored as failed, with its
// transcript as it was when the turn began, and the next prompt carries on
// from there; a model that was waiting on a pause is still waiting on it.
//
// Implementations must be safe for concurrent use.
type Store interface {
	// Save records the session's current state, replacing any earlier one.
	Save(SessionRecord) error
	// Load returns every saved session.
	Load() ([]SessionRecord, error)
	// Delete forgets a session. Deleting one that is not there is not an
	// error.
	Delete(id string) error
}

// SessionRecord is what a [Store] keeps of one session.
type SessionRecord struct {
	ID       string         `json:"id"`
	Options  SessionOptions `json:"options"`
	Created  time.Time      `json:"created"`
	Updated  time.Time      `json:"updated"`
	Turns    int            `json:"turns"`
	State    State          `json:"state"`
	Messages []llm.Message  `json:"messages"`
	Spend    Spend          `json:"spend"`

	Outcome   string        `json:"outcome,omitempty"`
	Answer    string        `json:"answer,omitempty"`
	Error     string        `json:"error,omitempty"`
	ErrorKind string        `json:"error_kind,omitempty"`
	PauseID   llm.ToolUseID `json:"pause_id,omitempty"`

	// Events is the next sequence number the session's log would have
	// issued. A restored log starts there, so a client resuming with a
	// Last-Event-ID from before the restart is told it missed events rather
	// than being handed new ones under old numbers.
	Events int `json:"events"`
}

// ErrorKind values a restored session can report.
const (
	// KindInterrupted is a turn the server restarted in the middle of.
	KindInterrupted = "interrupted"
)

// DirStore keeps one JSON file per session in dir, which is created if it
// does not exist. Each save is written to a temporary file, synced and renamed
// over the old one, so a crash mid-write leaves the previous version rather
// than half of a new one.
func DirStore(dir string) (Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("httpapi: session store: %w", err)
	}
	return dirStore{dir: dir}, nil
}

type dirStore struct{ dir string }

// validID is the shape newID produces. Anything else is not a session this
// package made, and is refused before it can name a path.
var validID = regexp.MustCompile(`^[0-9a-f]{32}$`)

func (d dirStore) path(id string) (string, error) {
	if !validID.MatchString(id) {
		return "", fmt.Errorf("httpapi: session store: refusing id %q", id)
	}
	return filepath.Join(d.dir, id+".json"), nil
}

func (d dirStore) Save(rec SessionRecord) error {
	path, err := d.path(rec.ID)
	if err != nil {
		return err
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("httpapi: session store: %w", err)
	}
	tmp, err := os.CreateTemp(d.dir, rec.ID+".*.tmp")
	if err != nil {
		return fmt.Errorf("httpapi: session store: %w", err)
	}
	defer os.Remove(tmp.Name()) // a no-op once renamed
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("httpapi: session store: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("httpapi: session store: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("httpapi: session store: %w", err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return fmt.Errorf("httpapi: session store: %w", err)
	}
	return nil
}

func (d dirStore) Load() ([]SessionRecord, error) {
	entries, err := os.ReadDir(d.dir)
	if err != nil {
		return nil, fmt.Errorf("httpapi: session store: %w", err)
	}
	var out []SessionRecord
	var errs []error
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".json") || !validID.MatchString(strings.TrimSuffix(name, ".json")) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(d.dir, name))
		if err != nil {
			errs = append(errs, err)
			continue
		}
		var rec SessionRecord
		if err := json.Unmarshal(data, &rec); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", name, err))
			continue
		}
		out = append(out, rec)
	}
	// A file that cannot be read costs that session, not the others.
	return out, errors.Join(errs...)
}

func (d dirStore) Delete(id string) error {
	path, err := d.path(id)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("httpapi: session store: %w", err)
	}
	return nil
}
