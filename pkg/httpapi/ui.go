package httpapi

import (
	"io/fs"
	"net/http"
	"path"
	"strings"
)

// uiCSP keeps the page to its own files. A reply is model output, and the
// page renders it without HTML; this is the second line, for the day a
// renderer gets that wrong: no script, style sheet, frame or fetch from
// anywhere but this server, and the page cannot be framed by another site.
// 'unsafe-inline' for styles only, for the style attributes React writes.
const uiCSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; " +
	"connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'"

// uiMissing is served at every UI path when fsys has no index.html: a binary
// built without running the front end's build first.
const uiMissing = `<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>wombatd</title>
<body style="font:15px/1.6 system-ui,sans-serif;max-width:40em;margin:4em auto;padding:0 1em;color:#1f1f23">
<h1 style="font-size:20px">wombatd is running</h1>
<p>This binary was built without its web UI. Build it with <code>make ui</code>, then
rebuild wombatd with <code>make bin</code> — or use the HTTP API under <code>/api/</code>.</p>
</body>
`

// uiHandler serves the browser front end in fsys.
//
//   - A file that exists is served as it is. Files under assets/ carry a
//     content hash in their names, so they are cached for a year; everything
//     else is revalidated.
//   - Any other GET that does not look like a file is answered with
//     index.html, so a client-side route such as /c/<id> survives a reload.
//   - A path that looks like a file — it has an extension — and does not
//     exist is a 404, not index.html: a missing script answered with HTML
//     fails far more confusingly than a missing one.
//   - Dot files (dist/.gitkeep) are not served, and neither is /metrics when
//     metrics are off.
func (a *api) uiHandler(fsys fs.FS) http.Handler {
	index, err := fs.ReadFile(fsys, "index.html")
	built := err == nil
	files := http.FileServerFS(fsys)

	page := func(w http.ResponseWriter, r *http.Request, body []byte) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		if r.Method == http.MethodHead {
			return
		}
		_, _ = w.Write(body)
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/")
		if (r.Method != http.MethodGet && r.Method != http.MethodHead) || p == "metrics" {
			a.notFound(w, r)
			return
		}
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY") // the approve buttons must not be clickable through another site's frame
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("Content-Security-Policy", uiCSP)

		if !built {
			page(w, r, []byte(uiMissing))
			return
		}
		for _, seg := range strings.Split(p, "/") {
			if strings.HasPrefix(seg, ".") {
				http.NotFound(w, r)
				return
			}
		}
		if p == "" || p == "index.html" {
			page(w, r, index)
			return
		}
		if st, err := fs.Stat(fsys, p); err == nil && !st.IsDir() {
			if strings.HasPrefix(p, "assets/") {
				h.Set("Cache-Control", "public, max-age=31536000, immutable")
			} else {
				h.Set("Cache-Control", "no-cache")
			}
			files.ServeHTTP(w, r)
			return
		}
		if path.Ext(p) != "" {
			http.NotFound(w, r)
			return
		}
		page(w, r, index)
	})
}
