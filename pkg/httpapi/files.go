package httpapi

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io/fs"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"wombat/pkg/daemon"
)

/*
 * The files a session's agent works on, for a UI to show beside the
 * conversation: the page it wrote, the chart, the report.
 *
 * Two routes. GET /api/sessions/{id}/file?path= is the API's, behind the
 * token like the rest: it finds the file in the session's workspace and
 * answers with what it is and a URL to read it from. That URL is under
 * /files/, outside /api, because an <iframe> or an <img> cannot send a token
 * — so it carries its own permission instead: a key, an HMAC of the session
 * and the folder the file is in, under a secret this process made up when it
 * started. It opens that file and the ones under its folder, in that session,
 * until wombatd restarts; nothing else.
 *
 * What the agent wrote is not to be trusted with the UI's origin: a page it
 * made runs scripts, and the UI's origin holds the token. Every response under
 * /files/ is therefore sandboxed by its own Content-Security-Policy — an
 * opaque origin, whether it is framed or opened on its own — while scripts,
 * styles and fonts from anywhere still load, so the page works. PDFs are the
 * exception: a sandboxed document cannot show one, and a PDF cannot script
 * this origin anyway.
 */

// errNoSuchFile marks a path the session's workspace has no file at.
var errNoSuchFile = errors.New("httpapi: no such file")

const kindNoSuchFile = "no_such_file"

// WithFiles lets a client read the files in a session's workspace: root
// returns the directory a session's tools are rooted at, given its options —
// the same directory the agent builder roots them at. Without it the file
// routes answer 404, and GET /api/config reports files false.
func WithFiles(root func(daemon.SessionOptions) (string, error)) Option {
	return func(c *apiConfig) { c.files = root }
}

// FileInfo is the body of GET /api/sessions/{id}/file: a file in the
// session's workspace, and where to read it.
type FileInfo struct {
	// Path is relative to the workspace, with forward slashes.
	Path     string    `json:"path"`
	Name     string    `json:"name"`
	Size     int64     `json:"size"`
	Modified time.Time `json:"modified"`

	// Type is the media type it is served as.
	Type string `json:"type"`

	// URL reads it, with no token: it carries its own key, good for this file
	// and the ones under its folder until the server restarts. Relative to
	// this server.
	URL string `json:"url"`
}

func newFileSecret() []byte {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		panic(fmt.Sprintf("httpapi: no randomness for the file key: %v", err))
	}
	return b
}

// fileKey is the key for the files under dir — "." for the workspace itself —
// in one session.
func (a *api) fileKey(session, dir string) string {
	m := hmac.New(sha256.New, a.fileSecret)
	m.Write([]byte(session))
	m.Write([]byte{0})
	m.Write([]byte(dir))
	return base64.RawURLEncoding.EncodeToString(m.Sum(nil)[:18])
}

// fileURL is where the file at rel in session's workspace is read.
func (a *api) fileURL(session, rel string) string {
	segs := strings.Split(rel, "/")
	for i, s := range segs {
		segs[i] = url.PathEscape(s)
	}
	return "/files/" + a.fileKey(session, path.Dir(rel)) + "/" + url.PathEscape(session) + "/" + strings.Join(segs, "/")
}

// hidden reports a path with a dotfile or dot-directory in it: .env, .git,
// .ssh. They are not shown, as the UI's own server does not serve them.
func hidden(rel string) bool {
	for _, s := range strings.Split(rel, "/") {
		if strings.HasPrefix(s, ".") {
			return true
		}
	}
	return false
}

// relPath turns what a client asks for — relative to the workspace, or an
// absolute path the agent printed — into a clean path inside root.
func relPath(root, raw string) (string, error) {
	p := strings.TrimSpace(raw)
	if p == "" {
		return "", fmt.Errorf("%w: path is required", daemon.ErrBadRequest)
	}
	if filepath.IsAbs(p) {
		r, err := filepath.Rel(root, filepath.Clean(p))
		if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
			// The agent may have printed it through a symlink: /tmp for
			// /private/tmp, a linked home directory.
			if real, rerr := filepath.EvalSymlinks(p); rerr == nil {
				r, err = filepath.Rel(root, real)
			}
		}
		if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
			return "", fmt.Errorf("%w: %s is outside the session's workspace", daemon.ErrBadRequest, raw)
		}
		p = r
	}
	rel := path.Clean(filepath.ToSlash(p))
	switch {
	case rel == "." || rel == ".." || strings.HasPrefix(rel, "../") || path.IsAbs(rel):
		return "", fmt.Errorf("%w: %s is outside the session's workspace", daemon.ErrBadRequest, raw)
	case hidden(rel):
		return "", fmt.Errorf("%w: %s is a hidden file", daemon.ErrBadRequest, raw)
	}
	return rel, nil
}

// workspaceOf is the directory a session's files are read from.
func (a *api) workspaceOf(s *daemon.Session) (string, error) {
	if a.cfg.files == nil {
		return "", fmt.Errorf("%w: this server does not serve files", errNotFound)
	}
	root, err := a.cfg.files(s.Info().Options)
	if err != nil {
		return "", fmt.Errorf("%w: the session's workspace: %w", daemon.ErrBadRequest, err)
	}
	return root, nil
}

// openFile opens rel inside root, refusing a directory, and a path that would
// leave root — through "..", or a symlink pointing out.
func openFile(root, rel string) (*os.File, fs.FileInfo, error) {
	r, err := os.OpenRoot(root)
	if err != nil {
		return nil, nil, err
	}
	defer r.Close()
	f, err := r.Open(filepath.FromSlash(rel))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil, fmt.Errorf("%w: %s", errNoSuchFile, rel)
		}
		return nil, nil, fmt.Errorf("%w: %s: %w", daemon.ErrBadRequest, rel, err)
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	if st.IsDir() {
		f.Close()
		return nil, nil, fmt.Errorf("%w: %s is a directory", errNoSuchFile, rel)
	}
	return f, st, nil
}

// textTypes are files that are text whatever the system's MIME table says —
// .ts is a video there — so a browser shows them rather than downloading.
var textTypes = map[string]bool{
	".md": true, ".markdown": true, ".txt": true, ".log": true, ".csv": true, ".tsv": true,
	".go": true, ".mod": true, ".sum": true, ".ts": true, ".tsx": true, ".jsx": true, ".py": true,
	".rs": true, ".java": true, ".kt": true, ".c": true, ".h": true, ".cc": true, ".cpp": true, ".hpp": true,
	".rb": true, ".php": true, ".swift": true, ".sh": true, ".bash": true, ".zsh": true, ".sql": true,
	".yaml": true, ".yml": true, ".toml": true, ".ini": true, ".cfg": true, ".conf": true, ".env": true,
	".proto": true, ".graphql": true, ".lua": true, ".r": true, ".scala": true, ".vue": true, ".svelte": true,
}

// typeOf is the media type a file is served as.
func typeOf(name string) string {
	ext := strings.ToLower(path.Ext(name))
	switch {
	case ext == ".html" || ext == ".htm":
		return "text/html; charset=utf-8"
	case ext == ".json":
		return "application/json"
	case textTypes[ext]:
		return "text/plain; charset=utf-8"
	}
	if t := mime.TypeByExtension(ext); t != "" {
		return t
	}
	return "application/octet-stream"
}

// fileCSP sandboxes what /files/ serves: an opaque origin with scripts,
// framed only by this server's own pages. Not for a PDF, which a sandboxed
// document cannot show.
func fileCSP(ctype string) string {
	if strings.HasPrefix(ctype, "application/pdf") {
		return "frame-ancestors 'self'"
	}
	return "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals allow-downloads; frame-ancestors 'self'"
}

// getFile answers what a file in the session's workspace is, and where to
// read it.
func (a *api) getFile(w http.ResponseWriter, r *http.Request) {
	s, err := a.session(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	root, err := a.workspaceOf(s)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	rel, err := relPath(root, r.URL.Query().Get("path"))
	if err != nil {
		a.fail(w, r, err)
		return
	}
	f, st, err := openFile(root, rel)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	f.Close()
	writeJSON(w, http.StatusOK, FileInfo{
		Path: rel, Name: path.Base(rel), Size: st.Size(), Modified: st.ModTime().UTC(),
		Type: typeOf(rel), URL: a.fileURL(s.ID(), rel),
	})
}

// serveFile serves a file under a key getFile handed out. Anything wrong —
// the key, the path, the session — is the same plain 404, so the route says
// nothing to someone guessing at it.
func (a *api) serveFile(w http.ResponseWriter, r *http.Request) {
	key, id, rel := r.PathValue("key"), r.PathValue("session"), r.PathValue("path")
	nope := func() { http.Error(w, "404 page not found", http.StatusNotFound) }
	if a.cfg.files == nil || rel == "" || path.Clean(rel) != rel || strings.HasPrefix(rel, "../") || hidden(rel) {
		nope()
		return
	}
	// The key is for a folder; any folder the file is in will do.
	ok := false
	for d := path.Dir(rel); ; d = path.Dir(d) {
		if hmac.Equal([]byte(key), []byte(a.fileKey(id, d))) {
			ok = true
			break
		}
		if d == "." {
			break
		}
	}
	if !ok {
		nope()
		return
	}
	s, err := a.m.Get(r.Context(), id)
	if err != nil {
		nope()
		return
	}
	root, err := a.workspaceOf(s)
	if err != nil {
		nope()
		return
	}
	f, st, err := openFile(root, rel)
	if err != nil {
		nope()
		return
	}
	defer f.Close()
	ctype := typeOf(rel)
	h := w.Header()
	h.Set("Content-Type", ctype)
	h.Set("Content-Security-Policy", fileCSP(ctype))
	h.Set("X-Content-Type-Options", "nosniff")
	// The key is in this URL: a page that loads a font from a CDN must not
	// hand it over in a Referer.
	h.Set("Referrer-Policy", "no-referrer")
	// The agent rewrites files as it works; a preview shows what is there now.
	h.Set("Cache-Control", "no-store")
	http.ServeContent(w, r, st.Name(), st.ModTime(), f)
}
