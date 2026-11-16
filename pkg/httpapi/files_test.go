package httpapi_test

import (
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"wombat/pkg/daemon"
	"wombat/pkg/httpapi"
)

// workspace lays out a session's workspace: a page and an image beside it, a
// file at the top, a dotfile, and a link pointing out of it.
func workspace(t *testing.T) (root, outside string) {
	t.Helper()
	root, outside = t.TempDir(), t.TempDir()
	if r, err := filepath.EvalSymlinks(root); err == nil {
		root = r
	}
	write := func(rel, body string) {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("diagrams/page.html", "<h1>hi</h1>")
	write("diagrams/img/chart.png", "\x89PNG")
	write("diagrams/doc.pdf", "%PDF-1.4")
	write("notes.md", "# top")
	write(".env", "KEY=secret")
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("nope"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(outside, "secret.txt"), filepath.Join(root, "diagrams", "escape.txt")); err != nil {
		t.Fatal(err)
	}
	return root, outside
}

func fileInfo(t *testing.T, h *apiHarness, id, p string) httpapi.FileInfo {
	t.Helper()
	var fi httpapi.FileInfo
	h.decode(h.do(http.MethodGet, "/api/sessions/"+id+"/file?path="+url.QueryEscape(p), ""), http.StatusOK, &fi)
	return fi
}

func get(t *testing.T, h *apiHarness, u string) (*http.Response, string) {
	t.Helper()
	resp, err := http.Get(h.srv.URL + u)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, string(b)
}

// A file the agent wrote is found by its path in the workspace, or by the
// absolute path the agent printed, and read from the URL that comes back —
// sandboxed, so a page it wrote runs in no origin of this server's.
func TestFilesAreShownSandboxed(t *testing.T) {
	t.Parallel()
	root, _ := workspace(t)
	h := apiServe(t, apiScript(apiText("hi")), nil, nil,
		httpapi.WithFiles(func(daemon.SessionOptions) (string, error) { return root, nil }))
	info := h.start(`{"prompt":"hi"}`)

	var caps httpapi.Capabilities
	h.decode(h.do(http.MethodGet, "/api/config", ""), http.StatusOK, &caps)
	if !caps.Files {
		t.Fatal("config does not report files")
	}

	fi := fileInfo(t, h, info.ID, filepath.Join(root, "diagrams", "page.html"))
	if fi.Path != "diagrams/page.html" || fi.Name != "page.html" || fi.Size != 11 || !strings.HasPrefix(fi.Type, "text/html") {
		t.Fatalf("file = %+v", fi)
	}
	if rel := fileInfo(t, h, info.ID, "./diagrams/page.html"); rel.URL != fi.URL {
		t.Fatalf("the relative path gave %q, the absolute one %q", rel.URL, fi.URL)
	}

	resp, body := get(t, h, fi.URL)
	if resp.StatusCode != http.StatusOK || body != "<h1>hi</h1>" {
		t.Fatalf("GET %s = %d %q", fi.URL, resp.StatusCode, body)
	}
	csp := resp.Header.Get("Content-Security-Policy")
	if !strings.Contains(csp, "sandbox allow-scripts") || strings.Contains(csp, "allow-same-origin") || !strings.Contains(csp, "frame-ancestors 'self'") {
		t.Fatalf("CSP = %q, want a sandbox without the same origin, framed only here", csp)
	}
	if resp.Header.Get("Referrer-Policy") != "no-referrer" || resp.Header.Get("X-Content-Type-Options") != "nosniff" {
		t.Fatalf("headers = %v", resp.Header)
	}

	// The key opens what is under the page's folder, as its relative links need.
	dir := fi.URL[:strings.LastIndex(fi.URL, "/")]
	if resp, body := get(t, h, dir+"/img/chart.png"); resp.StatusCode != http.StatusOK || body != "\x89PNG" {
		t.Fatalf("a file under the folder = %d %q", resp.StatusCode, body)
	}
	// A PDF is not sandboxed: a sandboxed document cannot show one.
	if resp, _ := get(t, h, dir+"/doc.pdf"); resp.StatusCode != http.StatusOK || strings.Contains(resp.Header.Get("Content-Security-Policy"), "sandbox") {
		t.Fatalf("pdf = %d, CSP %q", resp.StatusCode, resp.Header.Get("Content-Security-Policy"))
	}

	// And nothing else: not the folder above, not out through a link, not
	// with a key that was not handed out, not a dotfile.
	parts := strings.SplitN(strings.TrimPrefix(fi.URL, "/files/"), "/", 3)
	key, sid := parts[0], parts[1]
	for _, u := range []string{
		"/files/" + key + "/" + sid + "/notes.md",
		dir + "/escape.txt",
		"/files/" + key + "x/" + sid + "/diagrams/page.html",
		"/files/" + key + "/" + sid + "/diagrams/../notes.md",
		"/files/" + key + "/" + sid + "/.env",
		"/files/" + key + "/nope/diagrams/page.html",
	} {
		if resp, body := get(t, h, u); resp.StatusCode != http.StatusNotFound {
			t.Errorf("GET %s = %d %q, want 404", u, resp.StatusCode, body)
		}
	}
}

// What the API refuses to look up, and how it says so.
func TestFilesRefused(t *testing.T) {
	t.Parallel()
	root, outside := workspace(t)
	h := apiServe(t, apiScript(apiText("hi")), nil, nil,
		httpapi.WithFiles(func(daemon.SessionOptions) (string, error) { return root, nil }))
	info := h.start(`{"prompt":"hi"}`)
	base := "/api/sessions/" + info.ID + "/file?path="

	h.wantErr(h.do(http.MethodGet, base+url.QueryEscape("missing.html"), ""), http.StatusNotFound, "no_such_file")
	h.wantErr(h.do(http.MethodGet, base+url.QueryEscape("diagrams"), ""), http.StatusNotFound, "no_such_file")
	h.wantErr(h.do(http.MethodGet, base+url.QueryEscape("../x"), ""), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, base+url.QueryEscape(filepath.Join(outside, "secret.txt")), ""), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, base+url.QueryEscape(".env"), ""), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, base+url.QueryEscape("diagrams/escape.txt"), ""), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, base, ""), http.StatusBadRequest, "bad_request")
	h.wantErr(h.do(http.MethodGet, "/api/sessions/nope/file?path=notes.md", ""), http.StatusNotFound, "no_such_session")

	plain := apiServe(t, apiScript(apiText("hi")), nil, nil)
	other := plain.start(`{"prompt":"hi"}`)
	plain.wantErr(plain.do(http.MethodGet, "/api/sessions/"+other.ID+"/file?path=notes.md", ""), http.StatusNotFound, "not_found")
}

// The API asks for the token; the URL it hands out does not, since an iframe
// cannot send one.
func TestFilesURLNeedsNoToken(t *testing.T) {
	t.Parallel()
	root, _ := workspace(t)
	h := apiServe(t, apiScript(apiText("hi")), nil, nil,
		httpapi.WithBearerToken("s3cret"),
		httpapi.WithFiles(func(daemon.SessionOptions) (string, error) { return root, nil }))

	auth := func(r *http.Request) {
		r.Header.Set("Authorization", "Bearer s3cret")
		r.Header.Set("Content-Type", "application/json")
	}
	resp := h.raw(http.MethodPost, "/api/sessions", `{"prompt":"hi"}`, auth)
	var info daemon.SessionInfo
	h.decode(resp, http.StatusCreated, &info)

	h.wantErr(h.do(http.MethodGet, "/api/sessions/"+info.ID+"/file?path=notes.md", ""), http.StatusUnauthorized, "unauthorized")
	var fi httpapi.FileInfo
	h.decode(h.raw(http.MethodGet, "/api/sessions/"+info.ID+"/file?path=notes.md", "", auth), http.StatusOK, &fi)
	if resp, body := get(t, h, fi.URL); resp.StatusCode != http.StatusOK || body != "# top" {
		t.Fatalf("GET %s without the token = %d %q", fi.URL, resp.StatusCode, body)
	}
}
