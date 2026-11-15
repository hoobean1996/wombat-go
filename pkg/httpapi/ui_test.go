package httpapi_test

import (
	"io"
	"net/http"
	"strings"
	"testing"
	"testing/fstest"

	"wombat/pkg/httpapi"
)

func uiFS() fstest.MapFS {
	return fstest.MapFS{
		"index.html":     {Data: []byte("<!doctype html><title>app</title>")},
		"favicon.svg":    {Data: []byte("<svg/>")},
		"assets/app.js":  {Data: []byte("console.log(1)")},
		".gitkeep":       {Data: nil},
		"assets/app.css": {Data: []byte("body{}")},
	}
}

func readBody(t *testing.T, resp *http.Response) string {
	t.Helper()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading body: %v", err)
	}
	return string(b)
}

func TestUIServesFilesAndFallsBackToIndex(t *testing.T) {
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithUI(uiFS()))

	for _, tc := range []struct {
		path, want, cache string
		status            int
	}{
		{"/", "<title>app</title>", "no-cache", 200},
		{"/index.html", "<title>app</title>", "no-cache", 200},
		{"/c/0123abcd", "<title>app</title>", "no-cache", 200}, // a client-side route survives a reload
		{"/outbox", "<title>app</title>", "no-cache", 200},
		{"/assets/app.js", "console.log(1)", "public, max-age=31536000, immutable", 200},
		{"/favicon.svg", "<svg/>", "no-cache", 200},
		{"/assets/missing.js", "", "", 404}, // a missing file is not answered with HTML
		{"/.gitkeep", "", "", 404},
	} {
		resp := h.do("GET", tc.path, "")
		body := readBody(t, resp)
		if resp.StatusCode != tc.status {
			t.Errorf("GET %s: status %d, want %d", tc.path, resp.StatusCode, tc.status)
			continue
		}
		if tc.status != 200 {
			continue
		}
		if !strings.Contains(body, tc.want) {
			t.Errorf("GET %s: body %q, want it to contain %q", tc.path, body, tc.want)
		}
		if got := resp.Header.Get("Cache-Control"); got != tc.cache {
			t.Errorf("GET %s: Cache-Control %q, want %q", tc.path, got, tc.cache)
		}
		if resp.Header.Get("X-Frame-Options") != "DENY" || !strings.Contains(resp.Header.Get("Content-Security-Policy"), "frame-ancestors 'none'") {
			t.Errorf("GET %s: the page can be framed by another site", tc.path)
		}
	}
}

func TestUIDoesNotShadowTheAPI(t *testing.T) {
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithUI(uiFS()))

	// An unknown API path is still the JSON error envelope, not the page.
	resp := h.do("GET", "/api/nope", "")
	if body := readBody(t, resp); resp.StatusCode != 404 || !strings.Contains(body, `"not_found"`) {
		t.Fatalf("GET /api/nope: %d %s", resp.StatusCode, body)
	}
	// /metrics without WithMetrics is not the page either.
	if resp := h.do("GET", "/metrics", ""); resp.StatusCode != 404 {
		t.Fatalf("GET /metrics: %d, want 404", resp.StatusCode)
	}
	// Writing to a UI path is not a thing.
	if resp := h.do("POST", "/c/abc", "{}"); resp.StatusCode == 200 {
		t.Fatalf("POST /c/abc: %d", resp.StatusCode)
	}
	// And the API itself still works.
	var list httpapi.SessionList
	h.decode(h.do("GET", "/api/sessions", ""), 200, &list)
}

func TestUIIsServedWithoutTheTokenButTheAPIIsNot(t *testing.T) {
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithUI(uiFS()), httpapi.WithBearerToken("s3cret"))

	if resp := h.do("GET", "/c/abc", ""); resp.StatusCode != 200 {
		t.Fatalf("GET /c/abc without a token: %d, want 200 — the page has to load to ask for the token", resp.StatusCode)
	}
	if resp := h.do("GET", "/api/sessions", ""); resp.StatusCode != 401 {
		t.Fatalf("GET /api/sessions without a token: %d, want 401", resp.StatusCode)
	}
}

func TestUINotBuiltSaysSo(t *testing.T) {
	h := apiServe(t, apiScript(apiText("hi")), nil, nil, httpapi.WithUI(fstest.MapFS{".gitkeep": {}}))

	for _, p := range []string{"/", "/c/abc"} {
		resp := h.do("GET", p, "")
		if body := readBody(t, resp); resp.StatusCode != 200 || !strings.Contains(body, "make ui") {
			t.Fatalf("GET %s without a build: %d %q", p, resp.StatusCode, body)
		}
	}
}

func TestWithoutUITheRootIsNotFound(t *testing.T) {
	h := apiServe(t, apiScript(apiText("hi")), nil, nil)
	resp := h.do("GET", "/", "")
	if body := readBody(t, resp); resp.StatusCode != 404 || !strings.Contains(body, `"not_found"`) {
		t.Fatalf("GET / without WithUI: %d %s", resp.StatusCode, body)
	}
}
