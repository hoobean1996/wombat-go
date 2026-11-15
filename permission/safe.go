package permission

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/hoobean1996/wombat-go/tool"
)

// DefaultSafeCommands is the allowlist SafeCommands uses.
//
// Every entry is a command PREFIX, matched whole-word: "go test" admits
// `go test ./...` and does not admit `go testify`. Entries are deliberately
// specific where a program has both harmless and destructive subcommands, so
// this list carries "git status" and "git log" rather than "git".
//
// What is missing is the interesting part. rm, mv and cp change the tree; curl
// and wget reach the network and can write a file; chmod and sudo change who
// may do what; npm and pip execute arbitrary code out of a lockfile before
// they print anything; git push, git commit and git checkout publish or
// destroy work. None of those is "recognisably harmless", so none is here and
// all of them keep going to a person.
//
// A caller that wants more passes extra arguments to [SafeCommands] rather
// than appending here: the variable is read once, when a rule is built, and a
// package-level slice mutated at runtime is a permission change nobody can
// review.
var DefaultSafeCommands = []string{
	"go build",
	"go test",
	"go vet",
	"go fmt",
	"go list",
	"go doc",
	"go env",
	"gofmt",
	"ls",
	"pwd",
	"cat",
	"head",
	"tail",
	"wc",
	"find",
	"grep",
	"stat",
	"file",
	"du",
	"df",
	"date",
	"echo",
	"which",
	"git status",
	"git log",
	"git diff",
	"git show",
	"git branch",
}

// shellMetacharacters are the bytes that let one command string become two, or
// become a different command than it reads as.
//
// Semicolon and newline sequence, & backgrounds and (with &&) sequences, | pipes,
// backtick and $ substitute — either the output of another command or the
// contents of a variable — parentheses subshell, and < and > redirect. The
// carriage return is here with the newline because a CRLF-terminated line is
// still two lines to a shell.
const shellMetacharacters = ";&|`$()<>\n\r"

// shellMetacharactersNoPipe is the same set without the pipe, for the check
// that runs after [pipeline] has already split on it. Derived by hand rather
// than by strings.ReplaceAll so that the two constants are both greppable and
// neither can silently lose a byte the other still has.
const shellMetacharactersNoPipe = ";&`$()<>\n\r"

// pathExpansionChars are characters a shell turns into a path this rule cannot
// see.
//
// Only tilde, and it is worth its own constant because it is the one that
// looks innocent. `cat ~/.ssh/id_rsa` contains no metacharacter, no absolute
// path and no "..", and it reads a private key. The containment check below is
// textual, so the only honest response to a token the shell will rewrite
// before the kernel sees it is to stop having an opinion.
const pathExpansionChars = "~"

// unsafeFlags make an allowlisted program run another program, delete a file or
// write one, whichever entry it was allowed under.
//
// This is a patch on the sharpest edges rather than a solution — see the
// limitations section of [SafeCommands]. `find . -exec rm -rf {} +` contains no
// shell metacharacter at all, so without this the allowlist would admit
// arbitrary code execution through an entry whose whole justification is that
// it only looks at files; `go test -toolexec=./x.sh` and `go vet -vettool=./x`
// are the same hole in the go command.
//
// Keys are flag NAMES, without dashes, and a word is matched by name however it
// is spelled: -exec, --exec, -exec=prog, "-exec" and -e\xec all reach the
// program as the same flag. Matching the literal token was the original bug —
// `go test -exec=./evil.sh` was allowed because "-exec=./evil.sh" is not
// "-exec".
var unsafeFlags = map[string]bool{
	// find(1)
	"exec":    true,
	"execdir": true,
	"ok":      true,
	"okdir":   true,
	"delete":  true,
	"fprint":  true,
	"fprint0": true,
	"fprintf": true,
	"fls":     true,

	// The go command. Each takes a program and runs it: -exec runs the test
	// binary under it (go test, go run), -toolexec runs every compiler and
	// linker invocation through it, -vettool replaces vet.
	"toolexec": true,
	"vettool":  true,
}

// flagRule lists flags that disqualify ONE allowlist entry: they turn a
// subcommand that is on the list for being harmless into one that is not.
type flagRule struct {
	// names are long option names, without dashes.
	names []string

	// abbrev reports that the program accepts any unambiguous prefix of a
	// long option, as git does: `git branch --del x` is `--delete`. A
	// double-dash word that is a prefix of a name therefore counts as it.
	abbrev bool

	// short holds single-letter options the program lets you bundle, as git
	// does: `git branch -dr` is `-d -r`. Every letter of a single-dash word is
	// checked against it. Empty for a program that does not bundle.
	short string
}

// entryFlags maps an allowlist entry, words joined by one space, to the flags
// that disqualify it.
//
// Per entry rather than global because the same letter means different things
// to different programs: -w is "write the user's global Go environment" to go
// env and "match whole words" to grep, and refusing it everywhere would put
// `grep -w` and `wc -w` in front of a person for nothing.
var entryFlags = map[string]flagRule{
	// Rewrites the user's go env file — global, persistent configuration
	// outside any workspace, applied to every later go command.
	"go env": {names: []string{"w", "u"}},

	// On the list for LISTING branches. These delete, rename, copy over,
	// force-reset or re-point one, which is destroying work exactly as much as
	// the git commands the list leaves out.
	"git branch": {
		names: []string{
			"delete", "move", "copy", "force",
			"set-upstream-to", "unset-upstream", "edit-description",
		},
		abbrev: true,
		short:  "dDmMcCfu",
	},

	// On the list for READING history. --output turns them into writers that
	// can overwrite any file in reach; --ext-diff runs the external diff
	// program named in git config.
	"git diff": {names: []string{"output", "ext-diff"}, abbrev: true},
	"git log":  {names: []string{"output", "ext-diff"}, abbrev: true},
	"git show": {names: []string{"output", "ext-diff"}, abbrev: true},
}

// SafeCommands allows a shell command that is recognisably harmless.
//
// A grant is keyed on the tool name and the FULL arguments (see [Grants]), so
// `go test -v ./... | head -100` and `go test ./... -v` are two different
// questions and both get asked. That is correct and it is exhausting: a
// measured run of one incremental coding task stopped for a human five times,
// and three of those were variations on `go test` and `go build`. This rule
// exists to delete that particular noise and nothing else.
//
// A command is allowed only if ALL of the following hold. Otherwise the answer
// is [Undecided] and the policy falls through to whatever asks a person. This
// rule NEVER returns [Deny]: its job is to remove noise, not to add refusals,
// and a heuristic that can refuse is a heuristic that can break a run it does
// not understand.
//
//  1. The command matches an allowlisted prefix on a word boundary. Matching
//     is by word, so "go test" admits `go test ./...` and not `go testify`,
//     and `sudo go test` matches nothing because the first word is not "go".
//
//  2. The command contains no shell metacharacter AT ALL — see
//     [shellMetacharacters]. This is the load-bearing check. `go test; rm -rf
//     /` starts with an allowlisted prefix, and without this the allowlist is
//     not a convenience, it is an exploit: every entry becomes a prefix an
//     attacker (or a prompt-injected model, which behaves like one) glues a
//     real command onto. Tilde goes the same way, for the reason on
//     [pathExpansionChars].
//
//  3. Every absolute path appearing as a token is inside root, and no token
//     contains a ".." element. Relative tokens are not resolved and do not
//     need to be: with ".." refused they cannot climb out of the directory the
//     command runs in, and rule 5 puts that directory inside root. No token
//     may be one the shell could still expand into such a path: a glob on a
//     dot element (`.*` matches ".." in older bash) or a brace expansion
//     (`{..,x}`).
//
//  4. No token is a flag that runs a program or writes where it should not —
//     [unsafeFlags] for every entry, and [entryFlags] for the entries that are
//     only harmless without them (`go env -w`, `git branch -D`). A flag is
//     matched by name in any spelling: `-exec`, `--exec` and `-exec=prog` are
//     one flag to the program and therefore one flag here.
//
//  5. exec_dir, if present, is inside root.
//
// Every rule reads the words the way the shell will hand them to the program —
// quotes removed, backslash escapes applied — not the raw characters, because
// `-e"x"ec` is -exec and `.\./x` is ../x to the shell. See [lex].
//
// # What this is NOT
//
// Loudly, because the failure mode of a safety feature nobody understands is
// an operator who thinks they are protected.
//
// This is a heuristic over a string. It reads words the way the shell does
// and nothing more: it does not know what any program on the list actually
// does with its arguments, and it cannot follow a path through a variable, a
// config file or a symlink. A program on the allowlist can itself be a shell:
// `echo` cannot run anything, `find` very much can, which is why [unsafeFlags]
// exists and why that map is a patch and not a proof. Configuration is the
// same story one step removed — git runs whatever core.fsmonitor or
// diff.external names, and a workspace the agent can write to includes
// .git/config — and so is `go test` itself, which runs whatever test code is in
// the tree. `tail -f` never returns. `grep -r` reads every file
// under the directory it is pointed at, and the whole point of rule 3 is that
// it can only be pointed inside root — textually, with no symlink resolution,
// exactly like [FSRoot], with exactly the same TOCTOU race.
//
// It exists so that nobody is asked about `go test` twenty times in one
// afternoon. Anything it cannot prove obvious falls through to a human, and
// real containment is a sandbox around the process — builtin.SandboxRunner, or
// a container — which does not have to read the command to hold. With one in
// place this rule is what it should be: a way to ask less, not a boundary.
//
// root must not be empty and is resolved to an absolute path when the rule is
// built; both are construction-time programmer errors and panic.
func SafeCommands(root string, extra ...string) Rule {
	if strings.TrimSpace(root) == "" {
		panic("permission: SafeCommands needs a root directory")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		panic("permission: SafeCommands cannot resolve " + root + ": " + err.Error())
	}
	abs = filepath.Clean(abs)

	// Tokenised once, at construction: the comparison in the hot path is then
	// word against word, and an entry with odd spacing ("git  log") behaves
	// like the one without it.
	allow := make([][]string, 0, len(DefaultSafeCommands)+len(extra))
	for _, entry := range DefaultSafeCommands {
		if words := shellWords(entry); len(words) > 0 {
			allow = append(allow, words)
		}
	}
	for _, entry := range extra {
		if words := shellWords(entry); len(words) > 0 {
			allow = append(allow, words)
		}
	}

	return func(_ context.Context, r Request) (Decision, string) {
		if r.Tool.Caps&tool.CapExec == 0 {
			return Undecided, "" // not a shell call; not this rule's department
		}
		cmd, ok := stringArg(r.Use.Input, "command")
		if !ok {
			return Undecided, ""
		}
		segments, ok := pipeline(cmd)
		if !ok {
			return Undecided, ""
		}

		var matched []string
		for _, seg := range segments {
			// Read the stage the way the shell will, quotes and escapes
			// resolved, so every check below sees the arguments the program
			// will actually receive rather than the characters that spell them.
			words, ok := lex(seg)
			if !ok {
				return Undecided, ""
			}
			texts := make([]string, len(words))
			for i, w := range words {
				texts[i] = w.text
			}
			m := longestPrefixMatch(allow, texts)
			if m == nil {
				return Undecided, ""
			}
			rules := rulesFor(m)
			for _, w := range words {
				if !safeWord(abs, rules, w) {
					return Undecided, ""
				}
			}
			matched = append(matched, strings.Join(m, " "))
		}

		if dir, ok := stringArg(r.Use.Input, "exec_dir"); ok {
			if !contains(abs, resolve(abs, dir)) {
				return Undecided, ""
			}
		}

		return Allow, fmt.Sprintf(
			"every stage of %q is on this policy's list of commands not worth interrupting "+
				"a person for (%s), it has no substitution or file redirect in it, and every "+
				"path it names is inside %s",
			cmd, strings.Join(matched, " | "), abs)
	}
}

// pipeline splits a command into the stages a shell would run, and reports
// false for anything it will not reason about.
//
// The reason this exists is that the rule it serves was, in practice, off. The
// original check refused a command containing ANY shell metacharacter, which is
// the safe reading and also excludes how models actually write shell:
// `go test ./... 2>&1 | head -n 300`. In a live run under the workspace policy,
// an agent that had just written its implementation could not run its own tests
// — two of those in a row, and the refusal did not say why, so it tried
// `echo hi; pwd; ls -la` next and hit the same wall. An allowlist that never
// fires is not a safety feature, it is a rule that turns every exec into a
// question a headless front end cannot answer.
//
// So two constructs are admitted, and only two:
//
//   - "|" between stages. Each stage is then checked in full and independently:
//     `go test ./... | head -300` needs BOTH "go test" and "head" on the list.
//     `go test ./... | sh` does not get past the second stage. This is safe in a
//     way that a prefix match over the whole string never was — the failure it
//     replaces is `go test; rm -rf /`, where the dangerous half was never
//     examined at all.
//   - The exact token "2>&1". It duplicates a file descriptor and cannot name a
//     file, which is what makes it different from every other redirect. It is
//     removed before the stage is tokenised so that ">" and "&" can stay
//     forbidden everywhere else.
//
// Everything else still fails: ";" and "&&" and "||" sequence, "$()" and
// backticks substitute, ">" and "<" touch files, "&" backgrounds, "~" expands
// to a path this rule cannot see. All of them return false, the rule returns
// Undecided, and a person gets asked — which is the behaviour every one of them
// had before.
//
// "||" deserves its own note: splitting naively on "|" would turn
// `go test || rm -rf /` into the stages "go test" and "| rm -rf /", and a
// leading empty stage is easy to mishandle. It is rejected outright, before any
// split, by looking for the doubled byte.
func pipeline(cmd string) ([]string, bool) {
	if strings.Contains(cmd, "||") {
		return nil, false
	}

	// Whole-token only: "2>&1" is a fd dup, but "x2>&1" is a redirect of a fd
	// belonging to a word this rule has not looked at.
	kept := make([]string, 0, 8)
	for _, f := range strings.FieldsFunc(cmd, func(r rune) bool { return r == ' ' || r == '\t' }) {
		if f != "2>&1" {
			kept = append(kept, f)
		}
	}
	stripped := strings.Join(kept, " ")

	if strings.ContainsAny(stripped, pathExpansionChars) {
		return nil, false
	}
	// Everything except the pipe. Checked on the stripped string so a rejected
	// ">" cannot be smuggled in as part of a "2>&1" that was not a whole token.
	if strings.ContainsAny(stripped, shellMetacharactersNoPipe) {
		return nil, false
	}

	var out []string
	for _, seg := range strings.Split(stripped, "|") {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			return nil, false // a leading, trailing or doubled pipe
		}
		out = append(out, seg)
	}
	return out, true
}

// shellWords splits on ASCII space and tab only, which is what a shell's
// default IFS does once newlines are already refused.
//
// Deliberately not strings.Fields, which splits on every Unicode space
// including U+00A0. `go<U+00A0>test` is ONE word to a shell — a program named
// "go test", which does not exist — and Fields would have reported it as
// the two words "go" and "test" and matched the allowlist on a command line
// that is not the one being run. Keeping the split rule identical to the
// shell's is cheaper than auditing every place the two could diverge.
func shellWords(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool { return r == ' ' || r == '\t' })
}

// longestPrefixMatch returns the allowlist entry that matches the most leading
// words of the command, or nil.
//
// Longest rather than first, so the reason a human or a model reads names the
// specific entry ("git log") even when a shorter one would also have matched,
// and so the answer does not depend on the order of [DefaultSafeCommands].
func longestPrefixMatch(allow [][]string, words []string) []string {
	var best []string
	for _, entry := range allow {
		if len(entry) > len(words) || len(entry) <= len(best) {
			continue
		}
		hit := true
		for i, w := range entry {
			if words[i] != w {
				hit = false
				break
			}
		}
		if hit {
			best = entry
		}
	}
	return best
}

// rulesFor collects the [entryFlags] rules that apply under a matched allowlist
// entry: its own, and those of every entry it extends. A caller's extra entry
// "go env -json" is still go env, and -w still rewrites the user's environment
// under it; looking up only the longest match would have let it through.
func rulesFor(entry []string) []flagRule {
	var out []flagRule
	for k := 1; k <= len(entry); k++ {
		if r, ok := entryFlags[strings.Join(entry[:k], " ")]; ok {
			out = append(out, r)
		}
	}
	return out
}

// safeWord reports whether one shell word is one this rule is willing to vouch
// for, given the flag rules of the allowlist entry its stage matched: not a
// flag that runs a program or writes, nothing the shell could expand into a
// path this rule never saw, no ".." element, and no absolute path outside root.
func safeWord(root string, rules []flagRule, w shellWord) bool {
	if flagDisqualifies(rules, w.text) {
		return false
	}
	if expandsUnseen(w) {
		return false
	}
	for _, cand := range pathCandidates(w.text) {
		if cand == "" {
			continue
		}
		for _, elem := range strings.Split(cand, "/") {
			if elem == ".." {
				return false
			}
		}
		if filepath.IsAbs(cand) && !contains(root, filepath.Clean(cand)) {
			return false
		}
	}
	return true
}

// flagDisqualifies reports whether word, as the program receives it, is a flag
// that [unsafeFlags] or one of rules refuses.
//
// The name is what is compared, never the token: leading dashes are dropped and
// a value attached with '=' is cut off, so -exec, --exec and -exec=prog are all
// "exec". Being stricter than the program here only ever sends a command to a
// person, which is the direction this rule is allowed to be wrong in.
func flagDisqualifies(rules []flagRule, word string) bool {
	if len(word) < 2 || word[0] != '-' || word == "--" {
		return false
	}
	name := strings.TrimLeft(word, "-")
	if i := strings.IndexByte(name, '='); i >= 0 {
		name = name[:i]
	}
	if unsafeFlags[name] {
		return true
	}

	double := strings.HasPrefix(word, "--")
	for _, rule := range rules {
		for _, n := range rule.names {
			if name == n || (rule.abbrev && double && name != "" && strings.HasPrefix(n, name)) {
				return true
			}
		}
		if !double && rule.short != "" && strings.ContainsAny(name, rule.short) {
			return true
		}
	}
	return false
}

// expandsUnseen reports whether the shell could turn w into words this rule
// never examined, through an expansion that survives the metacharacter check.
//
// Two survive it. Pathname expansion: an element beginning with a dot and
// holding an unquoted *, ? or [ can match "..", because a leading dot written
// in the pattern matches the dot entries too — `cat .*/secret` reads
// ../secret wherever /bin/sh is an older bash, macOS's included. And bash's
// brace expansion, which /bin/sh also does on macOS: `cat {..,x}/secret` is
// `cat ../secret x/secret`, with no ".." element anywhere in the text. A glob
// anywhere else (`*.go`) or a brace with nothing to expand (`@{u}`) is left
// alone; only the forms that can produce a path are refused.
func expandsUnseen(w shellWord) bool {
	if open := strings.IndexByte(w.mask, '{'); open >= 0 {
		if n := strings.IndexByte(w.mask[open:], '}'); n >= 0 {
			if inner := w.mask[open+1 : open+n]; strings.Contains(inner, ",") || strings.Contains(inner, "..") {
				return true
			}
		}
	}
	// text and mask have the same length, so one set of element boundaries,
	// taken from text, serves both: a quoted '/' still separates path elements.
	start := 0
	for i := 0; i <= len(w.text); i++ {
		if i < len(w.text) && w.text[i] != '/' {
			continue
		}
		if strings.HasPrefix(w.text[start:i], ".") && strings.ContainsAny(w.mask[start:i], "*?[") {
			return true
		}
		start = i + 1
	}
	return false
}

// pathCandidates yields the parts of one word that could name a file.
//
// The word itself; whatever follows the first '=', since `--file=/etc/passwd`
// is one token and the path in it is real; and, for a single-dash word, what
// follows the option letter, since `-f/etc/passwd` and `-I../include` attach
// the value directly. Extra candidates can only make this rule stricter (a false
// candidate that looks like an escape sends the call to a human), which is the
// direction errors are allowed to go here.
func pathCandidates(word string) []string {
	out := []string{word}
	if i := strings.IndexByte(word, '='); i >= 0 {
		out = append(out, word[i+1:])
	}
	if len(word) > 2 && word[0] == '-' && word[1] != '-' {
		out = append(out, word[2:])
	}
	return out
}

// shellWord is one word of a pipeline stage as the shell hands it to the
// program.
type shellWord struct {
	// text is the word after quote removal and backslash escapes: the
	// argument the program receives.
	text string

	// mask is text with every quoted or escaped byte replaced by 0. What is
	// left is exactly what the shell may still expand — an unquoted * or { —
	// and the two strings always have the same length.
	mask string
}

// lex splits one pipeline stage into words the way /bin/sh does, once
// [pipeline] has refused every metacharacter.
//
// It exists because every check in this file is only as good as its picture of
// the arguments, and splitting on spaces gives the wrong picture: to a shell,
// `-e"x"ec` and `-e\xec` are the word -exec, and `'.'"."/secret` and
// `.\./secret` are ../secret. A rule reading the raw tokens waved all four
// through, two of them as `find . -e"x"ec rm {} +`.
//
// The subset is the one that remains after pipeline's checks: blanks separate
// words; single quotes keep everything literally; double quotes keep everything
// but a backslash before \ or ", which escapes it ($, ` and newline are
// already refused); an unquoted backslash makes the next byte literal. ok is
// false for what it will not reason about — an unterminated quote, a trailing
// backslash. Bytes, not runes, which is safe because every byte it acts on is
// ASCII and can never occur inside a multi-byte UTF-8 sequence.
func lex(stage string) (words []shellWord, ok bool) {
	var text, mask strings.Builder
	inWord := false

	literal := func(b byte) { text.WriteByte(b); mask.WriteByte(0) }
	flush := func() {
		if inWord {
			words = append(words, shellWord{text: text.String(), mask: mask.String()})
		}
		text.Reset()
		mask.Reset()
		inWord = false
	}

	for i := 0; i < len(stage); i++ {
		switch c := stage[i]; c {
		case ' ', '\t':
			flush()

		case '\'':
			// An empty pair still makes a word: '' is an empty argument.
			inWord = true
			end := strings.IndexByte(stage[i+1:], '\'')
			if end < 0 {
				return nil, false
			}
			for _, b := range []byte(stage[i+1 : i+1+end]) {
				literal(b)
			}
			i += end + 1

		case '"':
			inWord = true
			for i++; ; i++ {
				if i >= len(stage) {
					return nil, false
				}
				if stage[i] == '"' {
					break
				}
				if stage[i] == '\\' && i+1 < len(stage) && (stage[i+1] == '\\' || stage[i+1] == '"') {
					i++
				}
				literal(stage[i])
			}

		case '\\':
			if i+1 >= len(stage) {
				return nil, false
			}
			inWord = true
			i++
			literal(stage[i])

		default:
			inWord = true
			text.WriteByte(c)
			mask.WriteByte(c)
		}
	}
	flush()
	return words, true
}
