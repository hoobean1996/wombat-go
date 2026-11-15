package permission

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/hoobean1996/wombat-go/internal/fspath"
	"github.com/hoobean1996/wombat-go/tool"
)

// AllowTools permits the named tools and declines on everything else.
//
// Names, not capabilities, because an allowlist is only trustworthy if you can
// read it and know what it admits. Place it early in a [Policy]: a rule that
// allows is an exception, and exceptions go first.
func AllowTools(names ...string) Rule {
	set := nameSet(names)
	return func(_ context.Context, r Request) (Decision, string) {
		if _, ok := set[r.Tool.Name]; ok {
			return Allow, fmt.Sprintf("%s is on this run's allow list", r.Tool.Name)
		}
		return Undecided, ""
	}
}

// DenyTools refuses the named tools outright.
//
// Place it before any rule that might allow them, since the first decisive
// rule wins. A deny that sits after an [AllowTools] naming the same tool never
// fires, and nothing warns you.
func DenyTools(names ...string) Rule {
	set := nameSet(names)
	return func(_ context.Context, r Request) (Decision, string) {
		if _, ok := set[r.Tool.Name]; ok {
			return Deny, fmt.Sprintf("%s is on this run's deny list and cannot be used at all", r.Tool.Name)
		}
		return Undecided, ""
	}
}

// AskFor puts every tool carrying ANY of caps in front of a person.
//
// Any rather than all, unlike [tool.Def.Has]: the argument is a mask of things
// that are individually worth a question, so AskFor(tool.CapMutating|
// tool.CapExec) means "ask about writes, and ask about exec", not "ask about
// calls that are both at once".
func AskFor(caps tool.Cap) Rule {
	return func(_ context.Context, r Request) (Decision, string) {
		if r.Tool.Caps&caps != 0 {
			return Ask, fmt.Sprintf("%s is a %s tool, which this policy always confirms with a person",
				r.Tool.Name, capNames(r.Tool.Caps&caps))
		}
		return Undecided, ""
	}
}

// DenyCaps refuses every tool carrying ANY of caps. Same any-not-all reading
// as [AskFor].
func DenyCaps(caps tool.Cap) Rule {
	return func(_ context.Context, r Request) (Decision, string) {
		if hit := r.Tool.Caps & caps; hit != 0 {
			return Deny, fmt.Sprintf("%s is a %s tool and this run does not permit that at all",
				r.Tool.Name, capNames(hit))
		}
		return Undecided, ""
	}
}

// pathArgs are the input keys FSRoot reads as filesystem paths.
//
// A fixed list rather than a heuristic over key names. Guessing from the key
// ("anything ending in _path") would silently change meaning when a tool is
// added, and the failure would be a permitted write nobody asked for. The
// order is fixed too, so the reason attached to a decision is the same on
// every run for the same call.
var pathArgs = []string{"path", "file", "dir", "exec_dir", "cwd", "dest", "src"}

// FSRoot confines path arguments to root, and refuses to guess about shells.
//
// For a tool that declares [tool.NeedFSRead] or [tool.NeedFSWrite], the input
// is decoded as an object and the string-valued keys named in [pathArgs] are
// compared against root: every path inside allows, any path outside denies. A
// declared-filesystem tool whose call names no path at all is [Undecided] —
// there is nothing to judge, and inventing a verdict would be a guess.
//
// For a [tool.CapExec] tool the answer is [Ask], always, and the reason says
// why in plain words: what a shell command will touch cannot be decided by
// reading it. `cat $F`, `make`, `git clean -xdf`, a script that reads a
// config, anything with a pipe — the paths are not in the string. Every
// attempt to pattern-match them is a filter an adversary (or a
// prompt-injected model, which behaves like one) walks around, and worse, it
// tells the operator they are protected. Asking is the honest verdict, and it
// is the reason this package exists at all: the -sandbox flag constrains the
// file tools and does nothing whatsoever to bash.
//
// # The containment check is textual and symlink-blind
//
// Loudly, because this is the same check [builtin.OSFS] does and it has the
// same limits. Containment is filepath.Clean followed by a prefix test.
// Clean resolves "." and ".." lexically, so /work/../etc/passwd is caught —
// but nothing here touches the disk. Symlinks are NOT resolved: a symlink
// under root pointing outside it escapes, and so do hardlinks, bind mounts,
// and the race between this check and the syscall that follows it (TOCTOU).
// A relative path is interpreted against root, which is what the harness tells
// the model to do with relative paths anyway.
//
// This is a guardrail against a model casually wandering out of its
// workspace. It is not an isolation boundary. Real isolation is a container, a
// chroot, a mount namespace — and if you have one, you do not need to read
// path arguments at all.
//
// root must not be empty and is resolved to an absolute path when the rule is
// built; both are construction-time programmer errors and panic.
func FSRoot(root string) Rule {
	if strings.TrimSpace(root) == "" {
		panic("permission: FSRoot needs a root directory")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		panic("permission: FSRoot cannot resolve " + root + ": " + err.Error())
	}
	abs = filepath.Clean(abs)

	return func(_ context.Context, r Request) (Decision, string) {
		if r.Tool.Caps&tool.CapExec != 0 {
			return Ask, fmt.Sprintf(
				"%s runs a shell command, and what a shell command will touch cannot be "+
					"decided by reading it — the paths may come from a variable, a pipe or a "+
					"script. It is not possible to tell whether this stays inside %s, so a "+
					"person has to look", r.Tool.Name, abs)
		}
		if r.Tool.Needs&(tool.NeedFSRead|tool.NeedFSWrite) == 0 {
			return Undecided, ""
		}

		paths := extractPaths(r.Use.Input)
		if len(paths) == 0 {
			// A filesystem tool called with no path argument. Nothing to judge:
			// say so instead of manufacturing a verdict from silence.
			return Undecided, ""
		}
		for _, p := range paths {
			if !contains(abs, resolve(abs, p)) {
				return Deny, fmt.Sprintf("%s is outside the permitted directory %s", p, abs)
			}
		}
		return Allow, fmt.Sprintf("every path (%s) is inside %s", strings.Join(paths, ", "), abs)
	}
}

// extractPaths pulls the string-valued path arguments out of a call, in
// [pathArgs] order.
//
// Decoding to map[string]json.RawMessage rather than to a typed struct is what
// makes this rule work for a tool the package has never heard of, including
// one a host registered itself. Input that is not a JSON object yields
// nothing, which lands the caller on Undecided — the right answer, since a
// call whose arguments cannot be read has no paths to check and
// tool.WithValidation will reject it moments later anyway.
func extractPaths(in json.RawMessage) []string {
	if len(in) == 0 {
		return nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(in, &obj); err != nil {
		return nil
	}
	var out []string
	for _, key := range pathArgs {
		raw, ok := obj[key]
		if !ok {
			continue
		}
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			continue // a non-string under a path key is not a path
		}
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// resolve makes p absolute, interpreting a relative path against root.
func resolve(root, p string) string {
	if !filepath.IsAbs(p) {
		return filepath.Join(root, p)
	}
	return filepath.Clean(p)
}

// contains reports whether clean is root or lives under it. Textual; see the
// warning on [FSRoot]. A root of "/" contains every absolute path, which a
// plain root+"/" prefix test got wrong; see [fspath.Within].
func contains(root, clean string) bool {
	return fspath.Within(root, clean)
}

// AllowHosts permits network calls to the named hosts and denies the rest.
//
// A bare name matches that host exactly ("example.com"). A leading dot matches
// the domain and every subdomain (".example.com" matches example.com and
// api.example.com), which is the form people already know from cookies and
// no_proxy. Matching is case-insensitive and ignores the port, because
// example.com:8443 is the same host as example.com.
//
// The URL is read from the "url" argument. A call without one is [Undecided]:
// this rule is about network destinations and has no opinion on a tool that
// names none. A url that is present but unparseable, or whose scheme is not
// http or https, is [Deny] — file:// and gopher:// through an HTTP tool are
// how a fetcher becomes a file reader, and a scheme this rule does not
// understand is one it cannot vouch for.
func AllowHosts(hosts ...string) Rule {
	allowed := make([]string, 0, len(hosts))
	for _, h := range hosts {
		if h = strings.ToLower(strings.TrimSpace(h)); h != "" {
			allowed = append(allowed, h)
		}
	}

	return func(_ context.Context, r Request) (Decision, string) {
		raw, ok := stringArg(r.Use.Input, "url")
		if !ok {
			return Undecided, ""
		}
		u, err := url.Parse(raw)
		if err != nil {
			return Deny, fmt.Sprintf("%q is not a URL this policy can read, so it cannot be checked against the allowed hosts", raw)
		}
		switch strings.ToLower(u.Scheme) {
		case "http", "https":
		default:
			return Deny, fmt.Sprintf("the %q scheme is not permitted; only http and https are", u.Scheme)
		}
		host := strings.ToLower(u.Hostname())
		if host == "" {
			return Deny, fmt.Sprintf("%q names no host", raw)
		}
		for _, a := range allowed {
			if hostMatches(host, a) {
				return Allow, fmt.Sprintf("%s matches the allowed host %s", host, a)
			}
		}
		return Deny, fmt.Sprintf("%s is not on the list of hosts this run may reach (%s)",
			host, strings.Join(allowed, ", "))
	}
}

// hostMatches applies the two forms: exact, and leading-dot suffix.
func hostMatches(host, pattern string) bool {
	if strings.HasPrefix(pattern, ".") {
		return host == pattern[1:] || strings.HasSuffix(host, pattern)
	}
	return host == pattern
}

// stringArg reads one string-valued key out of a call's arguments.
func stringArg(in json.RawMessage, key string) (string, bool) {
	if len(in) == 0 {
		return "", false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(in, &obj); err != nil {
		return "", false
	}
	raw, ok := obj[key]
	if !ok {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	if s = strings.TrimSpace(s); s == "" {
		return "", false
	}
	return s, true
}

func nameSet(names []string) map[string]struct{} {
	set := make(map[string]struct{}, len(names))
	for _, n := range names {
		set[n] = struct{}{}
	}
	return set
}

// capNames renders a capability mask for a human. The reason string is read by
// a person deciding under time pressure and by a model deciding what to do
// next; "exec" beats "4".
func capNames(c tool.Cap) string {
	var out []string
	for _, e := range []struct {
		bit  tool.Cap
		name string
	}{
		{tool.CapReadOnly, "read-only"},
		{tool.CapMutating, "mutating"},
		{tool.CapExec, "exec"},
		{tool.CapNetwork, "network"},
		{tool.CapMeta, "meta"},
		{tool.CapPause, "pause"},
		{tool.CapTerminal, "terminal"},
	} {
		if c&e.bit != 0 {
			out = append(out, e.name)
		}
	}
	if len(out) == 0 {
		return "capability-less"
	}
	return strings.Join(out, "+")
}
