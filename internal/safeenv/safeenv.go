// Package safeenv decides which of the harness's environment variables a child
// process that runs model-written code is allowed to see.
//
// The harness holds secrets in its environment as a matter of course — the
// provider key, the gateway's routing headers, whatever cloud credentials the
// shell it was started from had exported — and every process it spawns used
// to inherit all of them. That made one tool call enough to read a key: a
// model, or a prompt injection steering one, writes a test that prints
// os.Getenv("OPENAI_API_KEY") and runs `go test`, which a workspace policy
// lets through without asking anyone. Filtering the environment at the spawn
// site closes that whatever the permission layer decides, because a secret the
// child never had is a secret it cannot leak.
//
// The filter is an ALLOWLIST. A denylist of names that look secret is the
// tempting alternative and the wrong one: it has to anticipate every name a
// credential will ever be given, and the first one it misses is the one that
// leaks.
package safeenv

import (
	"os"
	"strings"
)

// DefaultAllow names the variables a child process sees. An entry ending in
// "*" matches every name with that prefix.
//
// What is here is what a build, a test or a version-control command needs to
// find its tools, its caches and the network: the process basics, the Go
// toolchain's own settings, the C toolchain cgo drives, and proxy settings.
// The Go variables are listed by name rather than as "GO*", because that prefix
// also matches GOOGLE_API_KEY and GOOGLE_APPLICATION_CREDENTIALS.
//
// Two compromises are worth knowing about. The proxy variables can carry
// credentials (http://user:pass@proxy), and they are passed anyway: a build
// behind a corporate proxy cannot fetch a module without them, and a filter
// that breaks every build gets switched off. SSH_AUTH_SOCK is NOT passed, so
// git over ssh does not work from a child by default — the agent socket signs
// for every key the user has loaded, which is a capability and not a setting.
// A caller who wants either decision the other way adds or removes names.
//
// Leaving SSH_AUTH_SOCK out hides where the socket is, and no more: the
// socket itself is still there, usually under TMPDIR or /tmp, which a child
// can list as well as the harness can. An environment filter cannot close a
// socket; sandbox.Confined is what does.
var DefaultAllow = []string{
	// The process basics.
	"PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "TZ",
	"TMPDIR", "TMP", "TEMP",
	"LANG", "LANGUAGE", "LC_*",
	"XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR",

	// The Go toolchain.
	"GOPATH", "GOROOT", "GOBIN", "GOCACHE", "GOMODCACHE", "GOENV", "GOFLAGS",
	"GOPROXY", "GONOPROXY", "GOPRIVATE", "GONOSUMDB", "GOSUMDB", "GOINSECURE",
	"GOTOOLCHAIN", "GOTMPDIR", "GOWORK", "GOOS", "GOARCH", "GODEBUG",
	"GOEXPERIMENT", "GO111MODULE", "GOTELEMETRY", "GOFIPS140",
	"CGO_*", "CC", "CXX", "AR", "PKG_CONFIG_PATH",

	// Reaching the network at all.
	"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
	"SSL_CERT_FILE", "SSL_CERT_DIR",

	// What a process needs to start at all on Windows.
	"SYSTEMROOT", "SystemRoot", "COMSPEC", "ComSpec", "PATHEXT", "WINDIR", "windir",
}

// Allowed reports whether a variable called name passes allow.
func Allowed(name string, allow []string) bool {
	for _, a := range allow {
		if p, ok := strings.CutSuffix(a, "*"); ok {
			if strings.HasPrefix(name, p) {
				return true
			}
		} else if name == a {
			return true
		}
	}
	return false
}

// Filter returns the KEY=VALUE entries of environ whose names pass allow, in
// their original order. An entry with no name — Windows keeps per-drive
// directories as "=C:=C:\dir" — is dropped.
func Filter(environ, allow []string) []string {
	out := make([]string, 0, len(environ))
	for _, kv := range environ {
		name, _, _ := strings.Cut(kv, "=")
		if name != "" && Allowed(name, allow) {
			out = append(out, kv)
		}
	}
	return out
}

// Overlay returns base with every KEY=VALUE in set applied: an entry for a
// name already in base replaces it, a new name is appended. base is not
// modified.
func Overlay(base, set []string) []string {
	if len(set) == 0 {
		return base
	}
	replaced := make(map[string]bool, len(set))
	for _, kv := range set {
		name, _, _ := strings.Cut(kv, "=")
		replaced[name] = true
	}
	out := make([]string, 0, len(base)+len(set))
	for _, kv := range base {
		name, _, _ := strings.Cut(kv, "=")
		if !replaced[name] {
			out = append(out, kv)
		}
	}
	return append(out, set...)
}

// Environ is the harness's own environment filtered through [DefaultAllow],
// read at the moment of the call so that a variable the process sets later
// still reaches the children it spawns after that.
func Environ() []string { return Filter(os.Environ(), DefaultAllow) }
