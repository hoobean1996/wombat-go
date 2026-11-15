package permission

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// ===== fixtures =====

// root is the workspace every path test is written against. Absolute, because
// FSRoot resolves its argument and the tools demand absolute paths anyway.
var root = filepath.Join(string(filepath.Separator), "work")

func inRoot(rest ...string) string {
	return filepath.Join(append([]string{root}, rest...)...)
}

// calls counts how often the leaf handler actually ran, so every test can
// assert the thing that matters most: a refused call does not execute.
type calls struct{ n atomic.Int64 }

func def(name string, caps tool.Cap, needs tool.Need, c *calls) tool.Def {
	return tool.Def{
		Name:  name,
		Caps:  caps,
		Needs: needs,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			if c != nil {
				c.n.Add(1)
			}
			return "ran " + name, nil
		},
	}
}

func viewFile(c *calls) tool.Def {
	return def("view_file", tool.CapReadOnly, tool.NeedFSRead, c)
}
func writeFile(c *calls) tool.Def {
	return def("write_file", tool.CapMutating, tool.NeedFSWrite, c)
}
func bash(c *calls) tool.Def { return def("bash", tool.CapExec, tool.NeedExec, c) }
func fetchURL(c *calls) tool.Def {
	return def("fetch_url", tool.CapReadOnly|tool.CapNetwork, tool.NeedNetwork, c)
}
func calculator(c *calls) tool.Def { return def("calculator", tool.CapReadOnly, 0, c) }
func delegate(c *calls) tool.Def   { return def("delegate", tool.CapMeta, 0, c) }

func useOf(id string, args any) llm.ToolUse {
	var raw json.RawMessage
	if args != nil {
		b, err := json.Marshal(args)
		if err != nil {
			panic(err)
		}
		raw = b
	}
	return llm.ToolUse{ID: llm.ToolUseID(id), Input: raw}
}

func args(kv ...string) map[string]string {
	m := map[string]string{}
	for i := 0; i+1 < len(kv); i += 2 {
		m[kv[i]] = kv[i+1]
	}
	return m
}

func gated(p Policy, a Approver) tool.Handler {
	return tool.Chain(tool.Direct, Gate(p, a))
}

func yes() Approver {
	return ApproverFunc(func(context.Context, Request) (bool, error) { return true, nil })
}

func no() Approver {
	return ApproverFunc(func(context.Context, Request) (bool, error) { return false, nil })
}

// rule builds a constant rule, for the composition tests.
func rule(d Decision, why string) Rule {
	return func(context.Context, Request) (Decision, string) { return d, why }
}

// ===== decisions and policies =====

func TestDecisionString(t *testing.T) {
	for _, tc := range []struct {
		d    Decision
		want string
	}{
		{Undecided, "undecided"},
		{Allow, "allow"},
		{Deny, "deny"},
		{Ask, "ask"},
		{Decision(99), "undecided"},
	} {
		if got := tc.d.String(); got != tc.want {
			t.Errorf("Decision(%d).String() = %q, want %q", tc.d, got, tc.want)
		}
	}
}

func TestPolicyDecide(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
		want   Decision
		reason string // substring
	}{
		{
			name:   "the zero policy fails closed",
			policy: Policy{},
			want:   Deny,
			reason: "fails closed",
		},
		{
			name:   "no rules, explicit allow default",
			policy: Policy{Default: Allow},
			want:   Allow,
			reason: "allows by default",
		},
		{
			name:   "no rules, explicit deny default",
			policy: Policy{Default: Deny},
			want:   Deny,
			reason: "denies by default",
		},
		{
			name:   "no rules, ask default",
			policy: Policy{Default: Ask},
			want:   Ask,
			reason: "asks about anything",
		},
		{
			name:   "every rule declines, default applies",
			policy: Policy{Rules: []Rule{rule(Undecided, "nope"), rule(Undecided, "")}, Default: Ask},
			want:   Ask,
			reason: "no rule decided",
		},
		{
			name:   "the first decisive rule wins",
			policy: Policy{Rules: []Rule{rule(Allow, "first"), rule(Deny, "second")}, Default: Deny},
			want:   Allow,
			reason: "first",
		},
		{
			name:   "order matters the other way too",
			policy: Policy{Rules: []Rule{rule(Deny, "first"), rule(Allow, "second")}, Default: Allow},
			want:   Deny,
			reason: "first",
		},
		{
			name:   "undecided rules are skipped, not counted",
			policy: Policy{Rules: []Rule{rule(Undecided, "x"), rule(Ask, "the real one")}, Default: Deny},
			want:   Ask,
			reason: "the real one",
		},
		{
			name:   "a nil rule is skipped",
			policy: Policy{Rules: []Rule{nil, rule(Allow, "after the nil")}, Default: Deny},
			want:   Allow,
			reason: "after the nil",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, why := tc.policy.Decide(t.Context(), Request{Tool: viewFile(nil)})
			if got != tc.want {
				t.Errorf("Decide() = %v (%q), want %v", got, why, tc.want)
			}
			if !strings.Contains(why, tc.reason) {
				t.Errorf("Decide() reason = %q, want it to contain %q", why, tc.reason)
			}
		})
	}
}

// TestPolicyDecideNeverReturnsUndecided is the fail-closed invariant stated as
// a property: whatever the rules do, a caller always gets an actionable
// verdict.
func TestPolicyDecideNeverReturnsUndecided(t *testing.T) {
	for _, d := range []Decision{Undecided, Allow, Deny, Ask, Decision(42)} {
		p := Policy{Rules: []Rule{rule(Undecided, "")}, Default: d}
		if got, why := p.Decide(t.Context(), Request{}); got == Undecided {
			t.Errorf("Decide() with Default=%v = Undecided (%q)", d, why)
		}
	}
}

// ===== rules =====

func TestNameRules(t *testing.T) {
	c := &calls{}
	for _, tc := range []struct {
		name string
		r    Rule
		tool tool.Def
		want Decision
	}{
		{"allow hits", AllowTools("view_file", "bash"), viewFile(c), Allow},
		{"allow misses", AllowTools("view_file"), bash(c), Undecided},
		{"allow with no names decides nothing", AllowTools(), viewFile(c), Undecided},
		{"deny hits", DenyTools("bash"), bash(c), Deny},
		{"deny misses", DenyTools("bash"), viewFile(c), Undecided},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, why := tc.r(t.Context(), Request{Tool: tc.tool})
			if got != tc.want {
				t.Fatalf("rule = %v (%q), want %v", got, why, tc.want)
			}
			if tc.want != Undecided && !strings.Contains(why, tc.tool.Name) {
				t.Errorf("reason = %q, want it to name the tool %q", why, tc.tool.Name)
			}
			if tc.want == Undecided && why != "" {
				t.Errorf("an Undecided rule returned reason %q, want empty", why)
			}
		})
	}
}

func TestCapRules(t *testing.T) {
	c := &calls{}
	for _, tc := range []struct {
		name string
		r    Rule
		tool tool.Def
		want Decision
	}{
		{"ask for exec", AskFor(tool.CapExec), bash(c), Ask},
		{"ask for any of the mask", AskFor(tool.CapMutating | tool.CapExec), writeFile(c), Ask},
		{"ask matches on ANY bit, not all", AskFor(tool.CapExec | tool.CapNetwork), bash(c), Ask},
		{"ask declines on a miss", AskFor(tool.CapExec), viewFile(c), Undecided},
		{"deny caps hits", DenyCaps(tool.CapExec), bash(c), Deny},
		{"deny caps hits one bit of several", DenyCaps(tool.CapNetwork), fetchURL(c), Deny},
		{"deny caps declines", DenyCaps(tool.CapExec | tool.CapMutating), viewFile(c), Undecided},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, why := tc.r(t.Context(), Request{Tool: tc.tool})
			if got != tc.want {
				t.Fatalf("rule = %v (%q), want %v", got, why, tc.want)
			}
			if tc.want != Undecided && !strings.Contains(why, tc.tool.Name) {
				t.Errorf("reason = %q, want it to name the tool", why)
			}
		})
	}
}

func TestFSRoot(t *testing.T) {
	c := &calls{}
	r := FSRoot(root)

	for _, tc := range []struct {
		name   string
		tool   tool.Def
		args   any
		want   Decision
		reason string // substring
	}{
		{
			name: "a read inside the root",
			tool: viewFile(c), args: args("path", inRoot("src", "main.go")),
			want: Allow, reason: "inside",
		},
		{
			name: "the root itself",
			tool: viewFile(c), args: args("path", root),
			want: Allow, reason: "inside",
		},
		{
			name: "a read outside the root",
			tool: viewFile(c), args: args("path", "/etc/hosts"),
			want: Deny, reason: "/etc/hosts is outside",
		},
		{
			name: "a write inside the root",
			tool: writeFile(c), args: args("path", inRoot("out.txt")),
			want: Allow, reason: "inside",
		},
		{
			name: "traversal is cleaned before the check",
			tool: writeFile(c), args: args("path", inRoot("..", "etc", "passwd")),
			want: Deny, reason: "outside",
		},
		{
			name: "a prefix that is not a parent directory",
			tool: viewFile(c), args: args("path", root+"-secrets/key"),
			want: Deny, reason: "outside",
		},
		{
			name: "a relative path is taken against the root",
			tool: viewFile(c), args: args("path", "src/main.go"),
			want: Allow, reason: "inside",
		},
		{
			name: "a relative path can still escape",
			tool: viewFile(c), args: args("path", "../etc/passwd"),
			want: Deny, reason: "outside",
		},
		{
			name: "every path argument is checked, not just the first",
			tool: writeFile(c), args: args("src", inRoot("a"), "dest", "/tmp/b"),
			want: Deny, reason: "/tmp/b is outside",
		},
		{
			name: "all of several paths inside",
			tool: writeFile(c), args: args("src", inRoot("a"), "dest", inRoot("b")),
			want: Allow, reason: "inside",
		},
		{
			name: "the other spelling of a directory argument",
			tool: viewFile(c), args: args("dir", "/etc"),
			want: Deny, reason: "outside",
		},
		{
			name: "a filesystem tool with no path argument has no verdict",
			tool: viewFile(c), args: args("pattern", "TODO"),
			want: Undecided,
		},
		{
			name: "input that is not an object has no verdict",
			tool: viewFile(c), args: nil,
			want: Undecided,
		},
		{
			name: "a tool that needs no filesystem is not this rule's business",
			tool: fetchURL(c), args: args("path", "/etc/hosts"),
			want: Undecided,
		},
		{
			name: "a shell tool is always a question, even inside the root",
			tool: bash(c), args: args("exec_dir", inRoot(), "command", "cat /etc/hosts"),
			want: Ask, reason: "cannot be decided by reading it",
		},
		{
			name: "a shell tool with no arguments at all is still a question",
			tool: bash(c), args: nil,
			want: Ask, reason: "shell command",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, why := r(t.Context(), Request{Tool: tc.tool, Use: useOf("u1", tc.args)})
			if got != tc.want {
				t.Fatalf("FSRoot = %v (%q), want %v", got, why, tc.want)
			}
			if tc.reason != "" && !strings.Contains(why, tc.reason) {
				t.Errorf("reason = %q, want it to contain %q", why, tc.reason)
			}
		})
	}
}

// TestFilesystemRootContainsEverything is a regression test. The containment
// check prefixed root+"/", which for a root of "/" is "//": FSRoot("/") then
// denied /etc/hosts as outside "/" and denied every relative path too, since
// those resolve against the root, and SafeCommands("/") could never allow a
// command that named an absolute path.
func TestFilesystemRootContainsEverything(t *testing.T) {
	slash := string(filepath.Separator)
	abs := filepath.Join(slash, "etc", "hosts")

	r := FSRoot(slash)
	for _, p := range []string{abs, slash, "rel/x"} {
		if got, why := r(t.Context(), Request{Tool: viewFile(nil), Use: useOf("u1", args("path", p))}); got != Allow {
			t.Errorf("FSRoot(/) on %q = %v (%q), want Allow", p, got, why)
		}
	}

	s := SafeCommands(slash)
	for _, kv := range [][]string{
		{"command", "cat " + abs},
		{"command", "go test ./...", "exec_dir", filepath.Join(slash, "tmp")},
	} {
		if got, why := s(t.Context(), Request{Tool: bash(nil), Use: useOf("u1", args(kv...))}); got != Allow {
			t.Errorf("SafeCommands(/) on %v = %v (%q), want Allow", kv, got, why)
		}
	}
}

// TestFSRootIgnoresNonStringPathKeys: a tool whose "path" is an array is not a
// path, and reading one as a path would be a decision made on garbage.
func TestFSRootIgnoresNonStringPathKeys(t *testing.T) {
	r := FSRoot(root)
	in := json.RawMessage(`{"path":["/etc/hosts"],"dir":42,"file":""}`)
	if got, why := r(t.Context(), Request{Tool: viewFile(nil), Use: llm.ToolUse{Input: in}}); got != Undecided {
		t.Errorf("FSRoot = %v (%q), want Undecided", got, why)
	}
}

func TestFSRootPanicsOnAnEmptyRoot(t *testing.T) {
	for _, bad := range []string{"", "   "} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("FSRoot(%q) did not panic", bad)
				}
			}()
			FSRoot(bad)
		}()
	}
}

func TestAllowHosts(t *testing.T) {
	r := AllowHosts("example.com", ".corp.internal", "API.Example.ORG")

	for _, tc := range []struct {
		name   string
		url    any
		want   Decision
		reason string
	}{
		{"an exact host", args("url", "https://example.com/a"), Allow, "matches"},
		{"a port does not change the host", args("url", "https://example.com:8443/a"), Allow, "matches"},
		{"plain http is fine", args("url", "http://example.com"), Allow, "matches"},
		{"the pattern is case-insensitive", args("url", "https://api.example.org/x"), Allow, "matches"},
		{"the host is case-insensitive", args("url", "https://EXAMPLE.com/x"), Allow, "matches"},
		{"a leading dot matches a subdomain", args("url", "https://git.corp.internal/x"), Allow, "matches"},
		{"a leading dot matches the bare domain", args("url", "https://corp.internal/x"), Allow, "matches"},
		{"a subdomain of an exact entry does not match", args("url", "https://evil.example.com"), Deny, "not on the list"},
		{"a suffix that is not a subdomain", args("url", "https://notcorp.internal"), Deny, "not on the list"},
		{"an unlisted host", args("url", "https://evil.test/x"), Deny, "not on the list"},
		{"file:// is refused", args("url", "file:///etc/hosts"), Deny, `"file" scheme is not permitted`},
		{"a scheme-less url names no host", args("url", "example.com/x"), Deny, "scheme is not permitted"},
		{"an unparseable url", args("url", "http://a b.com/%zz"), Deny, "not a URL"},
		{"no url argument at all", args("path", "/work/x"), Undecided, ""},
		{"no arguments at all", nil, Undecided, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, why := r(t.Context(), Request{Tool: fetchURL(nil), Use: useOf("u1", tc.url)})
			if got != tc.want {
				t.Fatalf("AllowHosts = %v (%q), want %v", got, why, tc.want)
			}
			if tc.reason != "" && !strings.Contains(why, tc.reason) {
				t.Errorf("reason = %q, want it to contain %q", why, tc.reason)
			}
		})
	}
}

// ===== ready-made policies =====

func TestReadOnlyPolicy(t *testing.T) {
	c := &calls{}
	p := ReadOnly()
	for _, tc := range []struct {
		name string
		tool tool.Def
		want Decision
	}{
		{"a pure read", viewFile(c), Allow},
		{"a self-contained read", calculator(c), Allow},
		{"a write", writeFile(c), Deny},
		{"a shell", bash(c), Deny},
		{"read plus network is not read-only", fetchURL(c), Deny},
		{"orchestration", delegate(c), Deny},
		{"a tool that declares nothing", def("mystery", 0, 0, c), Deny},
		// Their handler never runs, so there is nothing to restrict.
		{"a pause tool", def("ask_user", tool.CapPause, 0, c), Allow},
		{"a terminal tool", def("submit", tool.CapTerminal, 0, c), Allow},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got, why := p.Decide(t.Context(), Request{Tool: tc.tool}); got != tc.want {
				t.Errorf("ReadOnly().Decide(%s) = %v (%q), want %v", tc.tool.Name, got, why, tc.want)
			}
		})
	}
}

func TestWorkspacePolicy(t *testing.T) {
	c := &calls{}
	p := Workspace(root)
	for _, tc := range []struct {
		name string
		tool tool.Def
		args any
		want Decision
	}{
		{"a read inside the workspace is free", viewFile(c), args("path", inRoot("a.go")), Allow},
		{"a write inside the workspace is free", writeFile(c), args("path", inRoot("a.go")), Allow},
		{"a read outside the workspace asks", viewFile(c), args("path", "/etc/hosts"), Ask},
		{"a write outside the workspace asks", writeFile(c), args("path", "/etc/hosts"), Ask},
		{"a shell command always asks", bash(c), args("exec_dir", inRoot()), Ask},
		{"orchestration is allowed", delegate(c), nil, Allow},
		{"a self-contained tool is allowed", calculator(c), args("expression", "2+2"), Allow},
		{"a network call asks", fetchURL(c), args("url", "https://example.com"), Ask},
		{"a filesystem tool with no path asks", viewFile(c), args("pattern", "TODO"), Ask},
		// Not "may I ask you something?" before asking it.
		{"a pause tool is allowed", def("ask_user", tool.CapPause, 0, c), args("question", "which env?"), Allow},
		{"a terminal tool is allowed", def("submit", tool.CapTerminal, 0, c), args("summary", "done"), Allow},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, why := p.Decide(t.Context(), Request{Tool: tc.tool, Use: useOf("u1", tc.args)})
			if got != tc.want {
				t.Errorf("Workspace().Decide(%s) = %v (%q), want %v", tc.tool.Name, got, why, tc.want)
			}
		})
	}
}

// A policy that must not let the model ask anything — a headless run nobody can
// answer — still says so by name, ahead of the ready-made rules.
func TestControlFlowCanStillBeDeniedByName(t *testing.T) {
	p := Workspace(root)
	p.Rules = append([]Rule{DenyTools("ask_user")}, p.Rules...)
	if got, why := p.Decide(t.Context(), Request{Tool: def("ask_user", tool.CapPause, 0, nil)}); got != Deny {
		t.Errorf("Decide(ask_user) = %v (%q), want Deny", got, why)
	}
}

// TestWorkspaceAsksAboutTheShellForTheRightReason is the bug this package was
// written for: the sandbox flag refused view_file for /etc/hosts and bash read
// the same file in the same run. Under Workspace both stop.
func TestWorkspaceAsksAboutTheShellForTheRightReason(t *testing.T) {
	p := Workspace(root)

	read, why := p.Decide(t.Context(), Request{
		Tool: viewFile(nil), Use: useOf("u1", args("path", "/etc/hosts")),
	})
	if read != Ask {
		t.Errorf("view_file /etc/hosts = %v (%q), want Ask", read, why)
	}

	shell, why := p.Decide(t.Context(), Request{
		Tool: bash(nil), Use: useOf("u2", args("command", "cat /etc/hosts", "exec_dir", inRoot())),
	})
	if shell != Ask {
		t.Fatalf("bash 'cat /etc/hosts' = %v (%q), want Ask", shell, why)
	}
	if !strings.Contains(why, "person") {
		t.Errorf("reason = %q, want it to say a person has to look", why)
	}
}

func TestAskEverythingPolicy(t *testing.T) {
	c := &calls{}
	p := AskEverything()
	for _, d := range []tool.Def{viewFile(c), writeFile(c), bash(c), calculator(c), delegate(c)} {
		if got, why := p.Decide(t.Context(), Request{Tool: d}); got != Ask {
			t.Errorf("AskEverything().Decide(%s) = %v (%q), want Ask", d.Name, got, why)
		}
	}
}

// ===== the gate =====

func TestGateAllows(t *testing.T) {
	c := &calls{}
	h := gated(Policy{Default: Allow}, nil)

	out, err := h(t.Context(), viewFile(c), useOf("u1", args("path", inRoot("a"))))
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if out != "ran view_file" {
		t.Errorf("output = %q, want the tool's own output", out)
	}
	if got := c.n.Load(); got != 1 {
		t.Errorf("the tool ran %d times, want 1", got)
	}
}

func TestGateDenies(t *testing.T) {
	c := &calls{}
	h := gated(Policy{Rules: []Rule{DenyTools("bash")}, Default: Allow}, yes())

	out, err := h(t.Context(), bash(c), useOf("u1", args("command", "rm -rf /")))
	if !errors.Is(err, ErrDenied) {
		t.Fatalf("err = %v, want it to wrap ErrDenied", err)
	}
	if out != "" {
		t.Errorf("output = %q, want empty", out)
	}
	if got := c.n.Load(); got != 0 {
		t.Fatalf("the tool ran %d times, want 0 — a denied call must not execute", got)
	}
}

// TestGateDenialMessageIsActionable: this string goes into a tool_result and
// the MODEL reads it. "denied" alone makes the model retry; it has to name
// what was refused, why, and what to do instead.
func TestGateDenialMessageIsActionable(t *testing.T) {
	for _, tc := range []struct {
		name     string
		policy   Policy
		approver Approver
		want     []string
	}{
		{
			name:   "refused by policy",
			policy: Policy{Rules: []Rule{FSRoot(root)}, Default: Deny},
			want: []string{
				"view_file",  // what was refused
				"/etc/hosts", // and on what
				"outside the permitted directory",
				"retrying the same call", // what not to do
				"tell the user",          // what to do instead
			},
		},
		{
			name:     "refused by a person",
			policy:   Policy{Default: Ask},
			approver: no(),
			want: []string{
				"view_file",
				"person supervising this run refused",
				"Do not retry",
				"tell the user",
			},
		},
		{
			name:   "nobody to ask",
			policy: Policy{Default: Ask},
			want: []string{
				"view_file",
				"needs a person to approve it",
				"nobody to ask",
				"tell the user",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &calls{}
			h := gated(tc.policy, tc.approver)
			_, err := h(t.Context(), viewFile(c), useOf("u1", args("path", "/etc/hosts")))
			if !errors.Is(err, ErrDenied) {
				t.Fatalf("err = %v, want it to wrap ErrDenied", err)
			}
			for _, want := range tc.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("the model reads %q\nwant it to contain %q", err.Error(), want)
				}
			}
			if c.n.Load() != 0 {
				t.Errorf("the tool ran despite the refusal")
			}
		})
	}
}

func TestGateNoApprover(t *testing.T) {
	c := &calls{}
	h := gated(AskEverything(), nil)

	_, err := h(t.Context(), bash(c), useOf("u1", args("command", "ls")))
	if !errors.Is(err, ErrNoApprover) {
		t.Errorf("err = %v, want it to wrap ErrNoApprover", err)
	}
	if !errors.Is(err, ErrDenied) {
		t.Errorf("err = %v, want it to wrap ErrDenied as well — the call did not run", err)
	}
	if c.n.Load() != 0 {
		t.Errorf("the tool ran with no approver installed")
	}
}

func TestGateApproverAllows(t *testing.T) {
	c := &calls{}
	var seen Request
	ap := ApproverFunc(func(_ context.Context, r Request) (bool, error) {
		seen = r
		return true, nil
	})
	h := gated(Policy{Rules: []Rule{FSRoot(root)}, Default: Deny}, ap)

	u := useOf("u1", args("command", "go test ./...", "exec_dir", inRoot()))
	out, err := h(t.Context(), bash(c), u)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if out != "ran bash" {
		t.Errorf("output = %q, want the tool's own output", out)
	}
	if seen.Tool.Name != "bash" || string(seen.Use.Input) != string(u.Input) {
		t.Errorf("the approver saw %+v, want the tool and its arguments", seen)
	}
	if !strings.Contains(seen.Reason, "shell command") {
		t.Errorf("Request.Reason = %q, want the deciding rule's reason", seen.Reason)
	}
}

// TestGateAsksOnce: the same question, asked twice, reaches a person once.
func TestGateAsksOnce(t *testing.T) {
	var asked atomic.Int64
	ap := ApproverFunc(func(context.Context, Request) (bool, error) {
		asked.Add(1)
		return true, nil
	})
	c := &calls{}
	h := gated(AskEverything(), ap)
	ctx := WithGrants(t.Context(), NewGrants())

	same := useOf("u1", args("command", "go build ./..."))
	for i := range 3 {
		if _, err := h(ctx, bash(c), same); err != nil {
			t.Fatalf("call %d: err = %v, want nil", i, err)
		}
	}
	if got := asked.Load(); got != 1 {
		t.Errorf("the approver was asked %d times about the same call, want 1", got)
	}
	if got := c.n.Load(); got != 3 {
		t.Errorf("the tool ran %d times, want 3 — a remembered grant still runs the call", got)
	}

	// A different call is a different question.
	if _, err := h(ctx, bash(c), useOf("u2", args("command", "rm -rf /"))); err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if got := asked.Load(); got != 2 {
		t.Errorf("the approver was asked %d times in total, want 2 — approving one command must not approve another", got)
	}

	// Whitespace is not a different question.
	if _, err := h(ctx, bash(c), llm.ToolUse{ID: "u3", Input: json.RawMessage(`{ "command" : "go build ./..." }`)}); err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if got := asked.Load(); got != 2 {
		t.Errorf("the approver was asked %d times, want 2 — reformatted arguments are the same question", got)
	}
}

// TestGateWithoutGrantsAsksEveryTime: GrantsFrom hands out a throwaway outside
// a run, and a throwaway must not silently behave like shared state.
func TestGateWithoutGrantsAsksEveryTime(t *testing.T) {
	var asked atomic.Int64
	ap := ApproverFunc(func(context.Context, Request) (bool, error) {
		asked.Add(1)
		return true, nil
	})
	h := gated(AskEverything(), ap)
	u := useOf("u1", args("command", "ls"))
	for range 2 {
		if _, err := h(t.Context(), bash(&calls{}), u); err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
	}
	if got := asked.Load(); got != 2 {
		t.Errorf("the approver was asked %d times with no Grants installed, want 2", got)
	}
}

// TestGateDoesNotRememberARefusal: a person who said no once may say yes when
// they see the model ask again, so a no is not cached.
func TestGateDoesNotRememberARefusal(t *testing.T) {
	var asked atomic.Int64
	ap := ApproverFunc(func(context.Context, Request) (bool, error) {
		return asked.Add(1) > 1, nil
	})
	h := gated(AskEverything(), ap)
	ctx := WithGrants(t.Context(), NewGrants())
	u := useOf("u1", args("command", "ls"))

	if _, err := h(ctx, bash(&calls{}), u); !errors.Is(err, ErrDenied) {
		t.Fatalf("first call err = %v, want ErrDenied", err)
	}
	if _, err := h(ctx, bash(&calls{}), u); err != nil {
		t.Fatalf("second call err = %v, want nil", err)
	}
	if got := asked.Load(); got != 2 {
		t.Errorf("the approver was asked %d times, want 2", got)
	}
}

// TestGateBlocksUntilCancelled is the load-bearing behaviour: the call is still
// pending while the human thinks, and cancellation — not a timeout, not a
// denial — is what ends the wait. The error must carry the CAUSE, so an
// operator learns the run was aborted rather than refused.
func TestGateBlocksUntilCancelled(t *testing.T) {
	errAbort := errors.New("governor: out of budget")

	entered := make(chan struct{})
	ap := ApproverFunc(func(ctx context.Context, _ Request) (bool, error) {
		close(entered)
		<-ctx.Done()
		return false, ctx.Err() // what a well-behaved approver returns
	})

	c := &calls{}
	h := gated(AskEverything(), ap)
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(nil)

	type outcome struct {
		out string
		err error
	}
	done := make(chan outcome, 1)
	go func() {
		out, err := h(ctx, bash(c), useOf("u1", args("command", "sleep 1")))
		done <- outcome{out, err}
	}()

	<-entered // the wait is real: we are inside Approve with no answer yet
	cancel(errAbort)
	got := <-done

	if !errors.Is(got.err, errAbort) {
		t.Errorf("err = %v, want it to carry the cancellation cause %v", got.err, errAbort)
	}
	if errors.Is(got.err, ErrDenied) {
		t.Errorf("err = %v, want an abandoned wait NOT to look like a refusal", got.err)
	}
	if !strings.Contains(got.err.Error(), "bash") {
		t.Errorf("err = %v, want it to name the tool", got.err)
	}
	if c.n.Load() != 0 {
		t.Errorf("the tool ran after the wait was cancelled")
	}
}

// TestGateReportsAnApproverFailure: a UI that cannot reach anybody is an error,
// not a verdict, and it must not be reported as a person saying no.
func TestGateReportsAnApproverFailure(t *testing.T) {
	boom := errors.New("prompt: the browser went away")
	h := gated(AskEverything(), ApproverFunc(func(context.Context, Request) (bool, error) {
		return false, boom
	}))
	_, err := h(t.Context(), bash(&calls{}), useOf("u1", args("command", "ls")))
	if !errors.Is(err, boom) {
		t.Errorf("err = %v, want it to wrap %v", err, boom)
	}
}

// ===== events =====

type recorder struct {
	mu     sync.Mutex
	events []core.Event
}

func (r *recorder) emit(e core.Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

func (r *recorder) kinds() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(r.events))
	for _, e := range r.events {
		out = append(out, e.Kind())
	}
	return out
}

func (r *recorder) decided(t *testing.T) Decided {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, e := range r.events {
		if d, ok := e.(Decided); ok {
			return d
		}
	}
	t.Fatalf("no Decided event in %v", r.events)
	return Decided{}
}

func TestGateEvents(t *testing.T) {
	for _, tc := range []struct {
		name     string
		policy   Policy
		approver Approver
		grants   bool
		twice    bool
		want     []string
		allowed  bool
		source   string
	}{
		{
			name:    "an allow announces nothing but the verdict",
			policy:  Policy{Default: Allow},
			want:    []string{"permission_decided"},
			allowed: true, source: sourcePolicy,
		},
		{
			name:   "a deny is still recorded",
			policy: Policy{Default: Deny},
			want:   []string{"permission_decided"},
			source: sourcePolicy,
		},
		{
			name: "an ask announces the request first", policy: AskEverything(), approver: yes(),
			want:    []string{"permission_requested", "permission_decided"},
			allowed: true, source: sourceApprover,
		},
		{
			name: "nobody to ask still records a verdict", policy: AskEverything(),
			want:   []string{"permission_decided"},
			source: sourceNoApprover,
		},
		{
			name: "a remembered grant asks nothing", policy: AskEverything(), approver: yes(),
			grants: true, twice: true,
			want: []string{
				"permission_requested", "permission_decided", // the first call
				"permission_decided", // the second, from memory
			},
			allowed: true, source: sourceApprover,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &recorder{}
			ctx := core.WithEmitter(t.Context(), rec.emit)
			if tc.grants {
				ctx = WithGrants(ctx, NewGrants())
			}
			h := gated(tc.policy, tc.approver)
			u := useOf("u1", args("command", "ls"))

			_, _ = h(ctx, bash(&calls{}), u)
			if tc.twice {
				_, _ = h(ctx, bash(&calls{}), u)
			}

			if got := rec.kinds(); strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("events = %v, want %v", got, tc.want)
			}
			d := rec.decided(t)
			if d.Tool != "bash" || d.UseID != "u1" {
				t.Errorf("Decided = %+v, want it to identify the call", d)
			}
			if d.Allowed != tc.allowed {
				t.Errorf("Decided.Allowed = %v, want %v", d.Allowed, tc.allowed)
			}
			if d.Source != tc.source {
				t.Errorf("Decided.Source = %q, want %q", d.Source, tc.source)
			}
			if d.Reason == "" {
				t.Error("Decided.Reason is empty; a verdict with no reason is not auditable")
			}
		})
	}
}

// TestGateRequestedCarriesTheArguments: "approve bash?" is not a question
// anybody can answer.
func TestGateRequestedCarriesTheArguments(t *testing.T) {
	rec := &recorder{}
	ctx := core.WithEmitter(t.Context(), rec.emit)
	h := gated(AskEverything(), yes())
	u := useOf("u7", args("command", "rm -rf build"))

	if _, err := h(ctx, bash(&calls{}), u); err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	req, ok := rec.events[0].(Requested)
	if !ok {
		t.Fatalf("first event = %T, want Requested", rec.events[0])
	}
	if req.UseID != "u7" || req.Tool != "bash" {
		t.Errorf("Requested = %+v, want it to identify the call", req)
	}
	if !strings.Contains(string(req.Input), "rm -rf build") {
		t.Errorf("Requested.Input = %s, want the arguments the person has to judge", req.Input)
	}
	if req.Reason == "" {
		t.Error("Requested.Reason is empty; the person is being asked why for no reason")
	}
}

// TestGateSurvivesEmptyInput: an empty json.RawMessage is not valid JSON, and
// marshalling one would take the whole event down.
func TestGateSurvivesEmptyInput(t *testing.T) {
	rec := &recorder{}
	ctx := core.WithEmitter(t.Context(), rec.emit)
	h := gated(AskEverything(), yes())

	if _, err := h(ctx, bash(&calls{}), llm.ToolUse{ID: "u1", Input: json.RawMessage{}}); err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	b, err := json.Marshal(rec.events[0])
	if err != nil {
		t.Fatalf("marshalling Requested: %v", err)
	}
	if strings.Contains(string(b), `"input"`) {
		t.Errorf("Requested = %s, want no input key when there are no arguments", b)
	}
}

func TestEventJSON(t *testing.T) {
	for _, tc := range []struct {
		name  string
		event core.Event
		want  string
	}{
		{
			name:  "requested",
			event: Requested{UseID: "u1", Tool: "bash", Reason: "why", Input: json.RawMessage(`{"a":1}`)},
			want:  `{"type":"permission_requested","use_id":"u1","tool":"bash","reason":"why","input":{"a":1}}`,
		},
		{
			name:  "decided",
			event: Decided{UseID: "u1", Tool: "bash", Allowed: true, Reason: "why", Source: sourcePolicy},
			want:  `{"type":"permission_decided","use_id":"u1","tool":"bash","allowed":true,"reason":"why","source":"policy"}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.event)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			if string(b) != tc.want {
				t.Errorf("Marshal = %s\nwant            %s", b, tc.want)
			}
		})
	}
}

func TestEventTypes(t *testing.T) {
	want := []string{"permission_requested", "permission_decided"}
	got := EventTypes()
	if len(got) != len(want) {
		t.Fatalf("EventTypes() = %v, want %d entries", got, len(want))
	}
	for i, e := range got {
		if e.Kind() != want[i] {
			t.Errorf("EventTypes()[%d].Kind() = %q, want %q", i, e.Kind(), want[i])
		}
	}
}

// ===== the audit log =====

// TestGateLogsEveryDecision: an unauditable permission system is not one.
func TestGateLogsEveryDecision(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
		level  string
	}{
		{"an allow is Info", Policy{Default: Allow}, "level=INFO"},
		{"a deny is Warn", Policy{Default: Deny}, "level=WARN"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			restore := slog.Default()
			slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})))
			t.Cleanup(func() { slog.SetDefault(restore) })

			h := gated(tc.policy, nil)
			_, _ = h(t.Context(), bash(&calls{}), useOf("u1", args("command", "ls")))

			line := buf.String()
			for _, want := range []string{tc.level, "tool=bash", "use_id=u1", "rule=policy", "reason="} {
				if !strings.Contains(line, want) {
					t.Errorf("log line %q\nwant it to contain %q", line, want)
				}
			}
			if n := strings.Count(line, "\n"); n != 1 {
				t.Errorf("logged %d lines, want exactly 1 per decision:\n%s", n, line)
			}
		})
	}
}

// ===== concurrency =====

// TestGrantsAreConcurrencySafe: the write side runs inside tool handlers, which
// the dispatcher may run in parallel.
func TestGrantsAreConcurrencySafe(t *testing.T) {
	var asked atomic.Int64
	ap := ApproverFunc(func(context.Context, Request) (bool, error) {
		asked.Add(1)
		return true, nil
	})
	h := gated(AskEverything(), ap)
	ctx := WithGrants(core.WithEmitter(t.Context(), (&recorder{}).emit), NewGrants())

	var wg sync.WaitGroup
	for i := range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			u := useOf(fmt.Sprintf("u%d", i), args("command", fmt.Sprintf("echo %d", i%4)))
			if _, err := h(ctx, bash(&calls{}), u); err != nil {
				t.Errorf("err = %v, want nil", err)
			}
		}()
	}
	wg.Wait()

	// Four distinct commands, so at most four questions — and at least one.
	// Not exactly four: two goroutines can be inside Approve for the same
	// command before either has remembered it, which is a duplicate question
	// and not a wrong answer.
	if got := asked.Load(); got < 1 || got > 16 {
		t.Errorf("the approver was asked %d times, want between 1 and 16", got)
	}
}

// TestDeniedIsCallerFault keeps a refused call from being read as a broken
// tool.
//
// A denial is the gate working. An agent that probes three commands the policy
// forbids has learned three true things, and the tool it was reaching for is
// in perfect health — so the circuit breaker must not count the refusals and
// then withhold the tool for the calls that WOULD have been allowed.
func TestDeniedIsCallerFault(t *testing.T) {
	if !tool.IsCallerFault(ErrDenied) {
		t.Error("IsCallerFault(ErrDenied) = false, want true")
	}
	wrapped := fmt.Errorf("%w: the bash tool is not allowed here", ErrDenied)
	if !errors.Is(wrapped, ErrDenied) {
		t.Error("errors.Is(wrapped, ErrDenied) = false, want true")
	}
	if !tool.IsCallerFault(wrapped) {
		t.Error("IsCallerFault(wrapped) = false, want true")
	}
}
