package permission

import (
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/tool"
)

// safeCase is one command put to the rule. want is Allow or Undecided and
// never Deny — see TestSafeCommandsNeverDenies, which asserts that over the
// whole table rather than case by case.
type safeCase struct {
	name string
	cmd  string
	dir  string // exec_dir; empty means the argument is absent
	want Decision
}

// safeCases is shared between the behaviour test and the never-denies test, so
// every bypass attempt is checked for both "is not allowed" and "is not a
// refusal either".
var safeCases = []safeCase{
	// ===== the whole point: the commands that were interrupting a person =====
	{"go test", "go test ./...", inRoot(), Allow},
	{"go test with flags in another order", "go test ./... -v", inRoot(), Allow},
	{"go test with a run filter", "go test -run TestSafeCommands ./permission", inRoot(), Allow},
	{"go build", "go build ./...", inRoot(), Allow},
	{"go vet", "go vet ./...", inRoot(), Allow},
	{"gofmt", "gofmt -l .", inRoot(), Allow},
	{"git status", "git status --short", inRoot(), Allow},
	{"git log", "git log --oneline -20", inRoot(), Allow},
	{"ls", "ls -la", inRoot(), Allow},
	{"pwd", "pwd", inRoot(), Allow},
	{"echo", "echo hello world", inRoot(), Allow},
	{"which", "which go", inRoot(), Allow},
	{"no exec_dir at all", "go test ./...", "", Allow},

	// ===== paths =====
	{"an absolute path inside root", "cat " + inRoot("main.go"), inRoot(), Allow},
	{"root itself", "ls " + root, inRoot(), Allow},
	{"a relative path", "cat sub/main.go", inRoot(), Allow},
	{"a flag value inside root", "grep --file=" + inRoot("pat") + " .", inRoot(), Allow},
	{"a glob", "find . -name *.go", inRoot(), Allow},
	{"odd spacing between words", "go   test  ./...", inRoot(), Allow},
	{"a leading space", "  go test ./...", inRoot(), Allow},

	// ===== metacharacter injection: the load-bearing check =====
	{"semicolon", "go test; rm -rf /", inRoot(), Undecided},
	{"double ampersand", "go test && rm -rf /", inRoot(), Undecided},
	{"single ampersand", "go test & rm -rf /", inRoot(), Undecided},
	{"pipe into a shell", "go test | sh", inRoot(), Undecided},
	{"backtick substitution", "go test `rm -rf /`", inRoot(), Undecided},
	{"dollar substitution", "go test $(rm -rf /)", inRoot(), Undecided},
	{"a variable", "echo $HOME", inRoot(), Undecided},
	{"a subshell", "(rm -rf /) ls", inRoot(), Undecided},
	{"output redirection", "echo pwned > " + inRoot("x"), inRoot(), Undecided},
	{"input redirection", "cat < /etc/passwd", inRoot(), Undecided},
	{"a newline", "go test\nrm -rf /", inRoot(), Undecided},
	{"a carriage return", "go test\r\nrm -rf /", inRoot(), Undecided},

	// ===== a prefix that is a longer word =====
	{"go testify is not go test", "go testify ./...", inRoot(), Undecided},
	{"gotest is not go test", "gotest ./...", inRoot(), Undecided},
	{"gofmtx is not gofmt", "gofmtx -l .", inRoot(), Undecided},
	{"lsof is not ls", "lsof -i", inRoot(), Undecided},
	{"a bare go is not a subcommand", "go", inRoot(), Undecided},
	{"go run is not on the list", "go run ./cmd/wombat-jsonl", inRoot(), Undecided},
	{"go generate is not on the list", "go generate ./...", inRoot(), Undecided},

	// ===== the allowlist has to be the FIRST word =====
	{"sudo in front", "sudo go test ./...", inRoot(), Undecided},
	{"env in front", "env GOFLAGS=-mod=mod go test ./...", inRoot(), Undecided},
	{"a leading assignment", "GOFLAGS=-mod=mod go test ./...", inRoot(), Undecided},

	// ===== path traversal =====
	{"a bare dotdot", "ls ..", inRoot(), Undecided},
	{"a relative climb", "cat ../secret", inRoot(), Undecided},
	{"a long relative climb", "cat ../../etc/passwd", inRoot(), Undecided},
	{"a climb through root", "cat " + root + "/../etc/passwd", inRoot(), Undecided},
	{"a climb in the middle", "cat sub/../../etc/passwd", inRoot(), Undecided},
	{"a trailing climb", "ls sub/..", inRoot(), Undecided},

	// ===== an absolute path outside root =====
	{"reading /etc/passwd", "cat /etc/passwd", inRoot(), Undecided},
	{"a double-quoted absolute path", `cat "/etc/passwd"`, inRoot(), Undecided},
	{"a single-quoted absolute path", "cat '/etc/passwd'", inRoot(), Undecided},
	{"an absolute path in a flag value", "grep --file=/etc/passwd .", inRoot(), Undecided},
	{"a sibling directory that shares a prefix", "ls /workspace", inRoot(), Undecided},
	{"grep pointed outside", "grep -rn secret /home", inRoot(), Undecided},

	// ===== tilde, which the shell rewrites into a path we never see =====
	{"a private key by tilde", "cat ~/.ssh/id_rsa", inRoot(), Undecided},
	{"a bare tilde", "ls ~", inRoot(), Undecided},

	// ===== exec_dir outside root =====
	{"exec_dir outside root", "go test ./...", "/etc", Undecided},
	{"exec_dir climbing out", "go test ./...", root + "/../etc", Undecided},
	{"exec_dir sharing a prefix", "go test ./...", "/workspace", Undecided},

	// ===== an allowlisted program used as a shell =====
	{"find -exec", "find . -exec rm -rf {} +", inRoot(), Undecided},
	{"find -execdir", "find . -execdir rm {} +", inRoot(), Undecided},
	{"find -delete", "find . -delete", inRoot(), Undecided},
	{"find -fprintf", "find . -fprintf out %p", inRoot(), Undecided},

	// ===== unicode lookalikes =====
	// A no-break space is a space to strings.Fields and NOT a space to a
	// shell, so a rule that used Fields would match the allowlist on a command
	// line that reads "go test" and executes something else entirely.
	{"a no-break space between the words", "go\u00a0test ./...", inRoot(), Undecided},
	{"a fullwidth g", "\uff47o test ./...", inRoot(), Undecided},
	{"a cyrillic o in go", "g\u043e test ./...", inRoot(), Undecided},
	{"an ideographic space", "go\u3000test ./...", inRoot(), Undecided},

	// ===== a flag that runs a program, in every spelling =====
	// Matched by NAME. The check used to compare the literal token, so the =
	// form walked straight past it and -toolexec was not on the list at all.
	{"go test -exec=", "go test -exec=./evil.sh ./...", inRoot(), Undecided},
	{"go test -exec", "go test -exec ./evil.sh ./...", inRoot(), Undecided},
	{"go test --exec=", "go test --exec=./evil.sh ./...", inRoot(), Undecided},
	{"go test -toolexec=", "go test -toolexec=./evil.sh ./...", inRoot(), Undecided},
	{"go test -toolexec", "go test -toolexec ./evil.sh ./...", inRoot(), Undecided},
	{"go build -toolexec=", "go build -toolexec=./evil.sh ./...", inRoot(), Undecided},
	{"go list -toolexec", "go list -toolexec ./evil ./...", inRoot(), Undecided},
	{"go vet -vettool=", "go vet -vettool=./evil ./...", inRoot(), Undecided},
	{"find -exec=", "find . -exec=rm", inRoot(), Undecided},
	{"find -fprint0", "find . -fprint0 out", inRoot(), Undecided},

	// ===== global configuration =====
	// go env -w rewrites the user's go env file: outside every workspace,
	// persistent, and applied to every later go command.
	{"go env -w", "go env -w GOPATH=evil", inRoot(), Undecided},
	{"go env -w=", "go env -w=GOPATH=evil", inRoot(), Undecided},
	{"go env --w", "go env --w GOFLAGS=-x", inRoot(), Undecided},
	{"go env -u", "go env -u GOFLAGS", inRoot(), Undecided},
	{"go env reading", "go env GOPATH GOFLAGS", inRoot(), Allow},
	{"go env -json", "go env -json", inRoot(), Allow},

	// ===== destroying work through a command listed for reading =====
	{"git branch -D", "git branch -D main", inRoot(), Undecided},
	{"git branch -d", "git branch -d feature", inRoot(), Undecided},
	{"git branch bundled -dr", "git branch -dr origin/x", inRoot(), Undecided},
	{"git branch --delete", "git branch --delete feature", inRoot(), Undecided},
	{"git branch abbreviated --del", "git branch --del feature", inRoot(), Undecided},
	{"git branch -m", "git branch -m old new", inRoot(), Undecided},
	{"git branch -f", "git branch -f main HEAD", inRoot(), Undecided},
	{"git branch --force", "git branch --force main HEAD", inRoot(), Undecided},
	{"git branch -vv", "git branch -vv", inRoot(), Allow},
	{"git branch -a sorted", "git branch -a --sort=-committerdate", inRoot(), Allow},
	{"git log --output", "git log --output=" + inRoot("main.go"), inRoot(), Undecided},
	{"git show abbreviated --out", "git show --out=x HEAD", inRoot(), Undecided},
	{"git diff --ext-diff", "git diff --ext-diff", inRoot(), Undecided},
	{"git diff --no-ext-diff", "git diff --no-ext-diff", inRoot(), Allow},
	{"git log --oneline", "git log --oneline -n 5", inRoot(), Allow},

	// ===== quoting: the shell's word, not the raw characters =====
	// Every check reads the argument the program will receive. Reading raw
	// tokens let the first two through as an unprompted `find -exec`.
	{"a flag split by double quotes", `find . -e"x"ec rm {} +`, inRoot(), Undecided},
	{"a flag split by a backslash", `find . -e\xec rm {} +`, inRoot(), Undecided},
	{"a quoted flag", `go test '-toolexec=./x' ./...`, inRoot(), Undecided},
	{"dotdot spelled with quotes", `cat '.'"."/secret`, inRoot(), Undecided},
	{"dotdot spelled with a backslash", `cat .\./secret`, inRoot(), Undecided},
	{"an absolute path behind a backslash", `cat \/etc/passwd`, inRoot(), Undecided},
	{"an unterminated quote", `cat "main.go`, inRoot(), Undecided},
	{"a trailing backslash", `cat main.go\`, inRoot(), Undecided},
	{"an escaped space makes one word", `go\ test ./...`, inRoot(), Undecided},
	{"a quoted program name is still the program", `"go" test ./...`, inRoot(), Allow},
	{"a quoted pattern", `grep -rn "func.*Handler" .`, inRoot(), Allow},

	// ===== expansions that can produce a path =====
	// /bin/sh on macOS is bash: .* still matches .. there, and braces expand.
	{"a glob that matches dotdot", "cat .*/secret", inRoot(), Undecided},
	{"a single-character glob that matches dotdot", "cat .?/secret", inRoot(), Undecided},
	{"brace expansion to dotdot", "cat {..,x}/secret", inRoot(), Undecided},
	{"a brace sequence", "cat {a..c}", inRoot(), Undecided},
	{"a quoted dot glob is literal", `grep -rn ".*foo" .`, inRoot(), Allow},
	{"a glob under a dot directory", "ls .git/*", inRoot(), Allow},
	{"a brace with nothing to expand", "git log @{u}", inRoot(), Allow},

	// ===== a short option with its value attached =====
	{"an attached absolute path", "grep -f/etc/passwd .", inRoot(), Undecided},
	{"an attached climb", "grep -f../secret .", inRoot(), Undecided},

	// ===== a letter that is dangerous to another program is fine here =====
	{"grep -w", "grep -w foo .", inRoot(), Allow},
	{"wc -w", "wc -w main.go", inRoot(), Allow},
	{"go doc -u", "go doc -u fmt", inRoot(), Allow},

	// ===== not on the list at all =====
	{"rm", "rm -rf " + inRoot("build"), inRoot(), Undecided},
	{"mv", "mv a b", inRoot(), Undecided},
	{"curl", "curl https://example.com", inRoot(), Undecided},
	{"chmod", "chmod +x " + inRoot("x"), inRoot(), Undecided},
	{"npm", "npm install", inRoot(), Undecided},
	{"pip", "pip install requests", inRoot(), Undecided},
	{"git push", "git push origin main", inRoot(), Undecided},
	{"git commit", "git commit -m wip", inRoot(), Undecided},
	{"git checkout", "git checkout main", inRoot(), Undecided},
	{"an empty command", "", inRoot(), Undecided},
	{"only spaces", "   ", inRoot(), Undecided},
}

func TestSafeCommands(t *testing.T) {
	rule := SafeCommands(root)
	for _, tc := range safeCases {
		t.Run(tc.name, func(t *testing.T) {
			kv := []string{"command", tc.cmd}
			if tc.dir != "" {
				kv = append(kv, "exec_dir", tc.dir)
			}
			got, why := rule(t.Context(), Request{Tool: bash(nil), Use: useOf("u1", args(kv...))})
			if got != tc.want {
				t.Errorf("SafeCommands(%s)(%q) = %v (%q), want %v", root, tc.cmd, got, why, tc.want)
			}
		})
	}
}

// TestSafeCommandsNeverDenies is the invariant that makes this rule safe to
// put in front of an asking rule: it can only ever turn a question into a
// pass, so a bug in the heuristic costs a prompt and never a broken run.
func TestSafeCommandsNeverDenies(t *testing.T) {
	rule := SafeCommands(root)
	for _, tc := range safeCases {
		kv := []string{"command", tc.cmd}
		if tc.dir != "" {
			kv = append(kv, "exec_dir", tc.dir)
		}
		if got, why := rule(t.Context(), Request{Tool: bash(nil), Use: useOf("u1", args(kv...))}); got == Deny {
			t.Errorf("SafeCommands(%q) = Deny (%q), want Allow or Undecided", tc.cmd, why)
		}
	}
}

// TestSafeCommandsOnlyJudgesShellTools guards the other half of the contract:
// a "command" key on something that is not a shell tool means whatever that
// tool decided it means, and is none of this rule's business.
func TestSafeCommandsOnlyJudgesShellTools(t *testing.T) {
	rule := SafeCommands(root)
	for _, d := range []tool.Def{viewFile(nil), writeFile(nil), calculator(nil), delegate(nil)} {
		got, why := rule(t.Context(), Request{Tool: d, Use: useOf("u1", args("command", "go test ./..."))})
		if got != Undecided {
			t.Errorf("SafeCommands()(%s) = %v (%q), want Undecided", d.Name, got, why)
		}
	}
}

func TestSafeCommandsWithoutACommandArgument(t *testing.T) {
	rule := SafeCommands(root)
	for _, in := range []any{
		nil,
		args("exec_dir", inRoot()),
		args("cmd", "go test ./..."), // a different key is a different tool
	} {
		if got, why := rule(t.Context(), Request{Tool: bash(nil), Use: useOf("u1", in)}); got != Undecided {
			t.Errorf("SafeCommands()(%v) = %v (%q), want Undecided", in, got, why)
		}
	}
}

// TestSafeCommandsIgnoresUnreadableInput covers the case FSRoot has: arguments
// that are not a JSON object at all. Nothing to read means nothing to vouch
// for.
func TestSafeCommandsIgnoresUnreadableInput(t *testing.T) {
	rule := SafeCommands(root)
	use := useOf("u1", nil)
	use.Input = []byte(`"go test ./..."`)
	if got, why := rule(t.Context(), Request{Tool: bash(nil), Use: use}); got != Undecided {
		t.Errorf("SafeCommands() on a JSON string = %v (%q), want Undecided", got, why)
	}
}

func TestSafeCommandsExtra(t *testing.T) {
	rule := SafeCommands(root, "make", "cargo check")
	for _, tc := range []struct {
		cmd  string
		want Decision
	}{
		{"make build", Allow},
		{"cargo check --all", Allow},
		{"cargo build", Undecided},    // only "cargo check" was added
		{"makefile x", Undecided},     // still a word boundary
		{"make; rm -rf /", Undecided}, // extras get the metacharacter check too
	} {
		got, why := rule(t.Context(), Request{
			Tool: bash(nil), Use: useOf("u1", args("command", tc.cmd, "exec_dir", inRoot())),
		})
		if got != tc.want {
			t.Errorf("SafeCommands(root, extra...)(%q) = %v (%q), want %v", tc.cmd, got, why, tc.want)
		}
	}
}

// TestSafeCommandsReasonNamesTheEntry matters because the reason is what ends
// up in the audit log and in the Decided event: "allowed" with no explanation
// is not reviewable.
// An extra entry that extends a flagged one inherits its flags: "go env -json"
// is still go env, and -w still rewrites the user's environment beneath it.
func TestSafeCommandsExtraEntryKeepsTheFlagsOfWhatItExtends(t *testing.T) {
	rule := SafeCommands(root, "go env -json", "git branch --list")
	for _, tc := range []struct {
		cmd  string
		want Decision
	}{
		{"go env -json", Allow},
		{"go env -json -w GOPATH=evil", Undecided},
		{"git branch --list", Allow},
		{"git branch --list -D main", Undecided},
	} {
		if got, why := rule(t.Context(), Request{Tool: bash(nil), Use: useOf("u1", args("command", tc.cmd))}); got != tc.want {
			t.Errorf("%q = %v (%q), want %v", tc.cmd, got, why, tc.want)
		}
	}
}

func TestSafeCommandsReasonNamesTheEntry(t *testing.T) {
	rule := SafeCommands(root)
	got, why := rule(t.Context(), Request{
		Tool: bash(nil), Use: useOf("u1", args("command", "git log --oneline", "exec_dir", inRoot())),
	})
	if got != Allow {
		t.Fatalf("git log = %v (%q), want Allow", got, why)
	}
	for _, want := range []string{"git log", root} {
		if !strings.Contains(why, want) {
			t.Errorf("reason = %q, want it to mention %q", why, want)
		}
	}
}

func TestSafeCommandsPanicsOnAnEmptyRoot(t *testing.T) {
	for _, bad := range []string{"", "   "} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("SafeCommands(%q) did not panic", bad)
				}
			}()
			SafeCommands(bad)
		}()
	}
}

// TestWorkspaceStopsAskingAboutGoTest is the measured problem this rule was
// added for, end to end through the policy: the routine build command no
// longer reaches a person, and everything the heuristic cannot vouch for still
// does.
func TestWorkspaceStopsAskingAboutGoTest(t *testing.T) {
	p := Workspace(root)
	for _, tc := range []struct {
		cmd  string
		want Decision
	}{
		{"go test ./...", Allow},
		{"go test ./... -v", Allow},
		{"go build ./...", Allow},
		{"gofmt -l .", Allow},

		// A pipeline of allowlisted stages, which is how a model actually
		// writes shell. Refusing these is what made the rule useless: in a
		// live run the agent could not test the code it had just written,
		// twice in a row, because it wrote `go test ./... 2>&1 | head -n 300`.
		{"go test -v ./... | head -100", Allow},
		{"go test ./... 2>&1 | head -n 300", Allow},
		{"go build ./... 2>&1", Allow},
		{"ls -la | head -20", Allow},

		// And the stage that is NOT on the list is still examined, which is
		// the thing a prefix match over the whole string never did.
		{"go test ./... | sh", Ask},
		{"go test ./... | xargs rm", Ask},
		{"cat /etc/hosts", Ask},
		{"rm -rf /", Ask},
	} {
		got, why := p.Decide(t.Context(), Request{
			Tool: bash(nil), Use: useOf("u1", args("command", tc.cmd, "exec_dir", inRoot())),
		})
		if got != tc.want {
			t.Errorf("Workspace().Decide(bash %q) = %v (%q), want %v", tc.cmd, got, why, tc.want)
		}
	}
}

// TestPipelineIsNotAnEscapeHatch is the adversarial half of admitting "|".
//
// Splitting on a metacharacter in order to allow more is exactly the move that
// turns an allowlist into a bypass, so every construct that could reattach a
// second command has its own line here. The rule NEVER returns Deny, so a
// failure looks like Ask — a person gets the question, which is where all of
// these went before the pipe was admitted and where they must stay.
func TestPipelineIsNotAnEscapeHatch(t *testing.T) {
	p := Workspace(root)
	for _, cmd := range []string{
		// Sequencing, in every spelling.
		"go test; rm -rf /",
		"go test && rm -rf /",
		"go test || rm -rf /",
		"go test & rm -rf /",
		"go test\nrm -rf /",
		"go test\r\nrm -rf /",

		// Substitution.
		"go test $(rm -rf /)",
		"go test `rm -rf /`",
		"go test ${HOME}",
		"echo $(cat /etc/passwd)",

		// Redirects that name a file, as opposed to 2>&1 which cannot.
		"go test > /etc/passwd",
		"go test >> ~/.bashrc",
		"go test < /etc/passwd",
		"echo pwned 1> /tmp/x",
		// Not a whole token, so not the fd dup it is pretending to be.
		"go test x2>&1",

		// A stage that is not on the list, in each position.
		"go test | sh",
		"go test | bash -c 'rm -rf /'",
		"sh | go test",
		"go test | head -5 | sh",
		"go test | xargs rm",

		// Degenerate pipes.
		"| rm -rf /",
		"go test |",
		"go test || ",
		"|",

		// Path expansion the rule cannot see through.
		"cat ~/.ssh/id_rsa",
		"cat ~/.ssh/id_rsa | head -1",

		// Escaping the root, in a stage that is otherwise fine.
		"cat ../../etc/passwd",
		"go test ./... | cat /etc/shadow",

		// Subshell.
		"(rm -rf /)",
		"go test | (rm -rf /)",
	} {
		got, why := p.Decide(t.Context(), Request{
			Tool: bash(nil), Use: useOf("u1", args("command", cmd, "exec_dir", inRoot())),
		})
		if got == Allow {
			t.Errorf("Workspace().Decide(bash %q) = ALLOW (%q); this must reach a person", cmd, why)
		}
	}
}

// TestPipelineChecksEveryStageFully: a stage is not just prefix-matched, it goes
// through the same path and flag checks as a lone command would.
func TestPipelineChecksEveryStageFully(t *testing.T) {
	p := Workspace(root)
	for _, tc := range []struct {
		cmd  string
		want Decision
	}{
		{"grep -r x . | head -5", Allow},
		// -exec makes an allowlisted program run another one; it is refused in
		// a pipeline exactly as it is alone.
		{"find . -name x -exec rm {} +", Ask},
		{"ls | find . -name x -exec rm {} +", Ask},
		// An absolute path outside the root, in the second stage.
		{"ls -la | wc -l", Allow},
		{"ls -la | grep -f /etc/passwd", Ask},
	} {
		got, why := p.Decide(t.Context(), Request{
			Tool: bash(nil), Use: useOf("u1", args("command", tc.cmd, "exec_dir", inRoot())),
		})
		if got != tc.want {
			t.Errorf("Decide(bash %q) = %v (%q), want %v", tc.cmd, got, why, tc.want)
		}
	}
}

// TestLex pins the shell's reading of a stage, which every other check in the
// rule is built on: a check is only as good as its picture of the arguments.
func TestLex(t *testing.T) {
	tests := []struct {
		in    string
		texts []string
		ok    bool
	}{
		{"go test ./...", []string{"go", "test", "./..."}, true},
		{"  go\t test  ", []string{"go", "test"}, true},
		{`-e"x"ec`, []string{"-exec"}, true},
		{`-e\xec`, []string{"-exec"}, true},
		{`'-exec'`, []string{"-exec"}, true},
		{`a'b c'd`, []string{"ab cd"}, true},
		{`"a\"b"`, []string{`a"b`}, true},
		{`"a\\b"`, []string{`a\b`}, true},
		{`"a\qb"`, []string{`a\qb`}, true}, // inside double quotes \q stays two bytes
		{`'a\qb'`, []string{`a\qb`}, true}, // inside single quotes nothing is special
		{`go\ test`, []string{"go test"}, true},
		{`''`, []string{""}, true}, // an empty argument is still an argument
		{`x '' y`, []string{"x", "", "y"}, true},
		{`"unterminated`, nil, false},
		{`'unterminated`, nil, false},
		{`trailing\`, nil, false},
		{"", nil, true},
	}
	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			words, ok := lex(tc.in)
			if ok != tc.ok {
				t.Fatalf("lex(%q) ok = %v, want %v", tc.in, ok, tc.ok)
			}
			var texts []string
			for _, w := range words {
				texts = append(texts, w.text)
				if len(w.mask) != len(w.text) {
					t.Errorf("word %q: mask is %d bytes and text %d; expandsUnseen indexes one by the other",
						w.text, len(w.mask), len(w.text))
				}
			}
			if strings.Join(texts, "|") != strings.Join(tc.texts, "|") || len(texts) != len(tc.texts) {
				t.Errorf("lex(%q) = %q, want %q", tc.in, texts, tc.texts)
			}
		})
	}

	t.Run("the mask keeps only what the shell may still expand", func(t *testing.T) {
		words, _ := lex(`*.go '*.go' \*.go "a*"b*`)
		want := []string{"*.go", "\x00\x00\x00\x00", "\x00.go", "\x00\x00b*"}
		for i, w := range words {
			if w.mask != want[i] {
				t.Errorf("word %d (%q): mask = %q, want %q", i, w.text, w.mask, want[i])
			}
		}
	})
}
