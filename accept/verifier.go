package accept

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/hoobean1996/wombat-go/internal/safeenv"
)

// DefaultVerifierFile is where the planner records what must be verified.
const DefaultVerifierFile = "Verifier.md"

// A verifier item is one thing that must be true of the finished work.
//
// The format exists because of what TODO.md showed. Written in prose it is
// advice, and advice loses: the same checklist went unticked for eighty
// iterations in two runs and was only honoured once something parsed it and
// refused to accept the answer. So an item here is not satisfied by being
// described as done — it must name a test, and that test must exist and pass.
//
// This is the rule that made the reviewer's precision free, pointed the other
// way. There a finding only counted as a failing test; here a verification only
// counts as a passing one. Neither needs a person to adjudicate, and neither can
// be talked past.
type Item struct {
	ID    string
	Title string
	State ItemState
	Test  string // the test that proves it
	Cmd   string // the command that proves it; an alternative to Test
	Note  string // withdrawal reason, required when State is Withdrawn
	Line  int
}

// ItemState is the tick in an item's Status field.
type ItemState int

const (
	Unverified ItemState = iota // "[ ]"
	Verified                    // "[x]"
	Withdrawn                   // "[-]"
)

// withdrawnMinReason is how much justification a withdrawal must carry.
//
// Withdrawal is a necessary escape hatch: a planner that has not seen the code
// can write an item the specification does not support, and a wrong item blocks
// a correct implementation. Two of the probes written by hand in this project
// were exactly that. But an unguarded escape hatch is just a way to delete the
// checklist, which is what happened to TODO.md three times, so it costs an
// argument.
const withdrawnMinReason = 40

// h is horizontal whitespace only. Plain \s crosses newlines, and with these
// fields written one per line that is not a nicety: "Test:" followed by
// "Command:" on the next line parsed as a test NAMED Command, and the item was
// then reported as naming a test that does not exist.
const h = `[^\S\n]*`

var (
	reItemHead = regexp.MustCompile(`(?m)^#{1,6}` + h + `\s+(V\d+)[.:)]?` + h + `(.*)$`)
	reStatus   = regexp.MustCompile(`(?mi)^` + h + `Status:` + h + `\[([ xX\-])\]` + h + `(.*)$`)
	reTest     = regexp.MustCompile(`(?mi)^` + h + "Test:" + h + "`?" + `(\w*)` + "`?" + h + `$`)
	reCommand  = regexp.MustCompile("(?mi)^" + h + "Command:" + h + "`?([^`\n]*?)`?" + h + "$")
	rePassLine = regexp.MustCompile(`(?m)^\s*--- (PASS|FAIL|SKIP): (Test\w+)`)
)

// ParseVerifier reads the items out of a Verifier.md body.
func ParseVerifier(content string) []Item {
	heads := reItemHead.FindAllStringSubmatchIndex(content, -1)
	var items []Item
	for i, h := range heads {
		end := len(content)
		if i+1 < len(heads) {
			end = heads[i+1][0]
		}
		body := content[h[1]:end]
		it := Item{
			ID:    content[h[2]:h[3]],
			Title: strings.TrimSpace(content[h[4]:h[5]]),
			Line:  1 + strings.Count(content[:h[0]], "\n"),
		}
		if m := reStatus.FindStringSubmatch(body); m != nil {
			switch m[1] {
			case "x", "X":
				it.State = Verified
			case "-":
				it.State = Withdrawn
			}
			it.Note = strings.TrimSpace(m[2])
		}
		if m := reTest.FindStringSubmatch(body); m != nil {
			it.Test = m[1]
		}
		if m := reCommand.FindStringSubmatch(body); m != nil {
			it.Cmd = strings.TrimSpace(m[1])
		}
		items = append(items, it)
	}
	return items
}

// checkVerifier enforces the Verifier.md contract.
func (r *Report) checkVerifier(ctx context.Context, cfg Config) {
	if cfg.VerifierFile == "" {
		return
	}
	b, err := os.ReadFile(filepath.Join(cfg.Workspace, cfg.VerifierFile))
	if err != nil {
		return // no verifier file is not a failure; the planner may not have run
	}
	items := ParseVerifier(string(b))
	if len(items) == 0 {
		r.add(Check{
			Name: "verifier",
			OK:   false,
			Detail: cfg.VerifierFile + " has no items. Each one is a heading of the form\n" +
				"\"## V1. <what must be true>\" with a Status: line and a Test: line.",
		})
		return
	}

	// Collect the tests the ticked items claim, and find out what they really do.
	claimed := map[string][]Item{}
	for _, it := range items {
		if it.State == Verified && it.Test != "" && it.Cmd == "" {
			claimed[it.Test] = append(claimed[it.Test], it)
		}
	}
	results := runNamedTests(ctx, cfg, keysOf(claimed))

	var problems []string
	verified := 0
	for _, it := range items {
		switch it.State {
		case Unverified:
			problems = append(problems, fmt.Sprintf(
				"%s (%s:%d) %s\n     not verified: tick it only once a test proves it",
				it.ID, cfg.VerifierFile, it.Line, it.Title))
		case Withdrawn:
			reason := strings.TrimPrefix(strings.TrimSpace(it.Note), "withdrawn:")
			if len(strings.TrimSpace(reason)) < withdrawnMinReason {
				problems = append(problems, fmt.Sprintf(
					"%s (%s:%d) %s\n     withdrawn without an argument. Quote the part of the task that\n"+
						"     shows this item asks for something it never required.",
					it.ID, cfg.VerifierFile, it.Line, it.Title))
			}
		case Verified:
			// A command settles the item by exiting zero. Preferred for the
			// conditions a task states as commands: forcing those through a
			// test function adds a layer that can fail on its own.
			if it.Cmd != "" {
				ok, why := runCommand(ctx, cfg, it.Cmd)
				if ok {
					verified++
				} else {
					problems = append(problems, fmt.Sprintf(
						"%s (%s:%d) %s\n     `%s` %s",
						it.ID, cfg.VerifierFile, it.Line, it.Title, it.Cmd, why))
				}
				continue
			}
			switch {
			case it.Test == "":
				problems = append(problems, fmt.Sprintf(
					"%s (%s:%d) %s\n     ticked but names neither a test nor a command. Put a test\n"+
						"     name on the Test: line, or a shell command on a Command: line.",
					it.ID, cfg.VerifierFile, it.Line, it.Title))
			case results[it.Test] == "":
				problems = append(problems, fmt.Sprintf(
					"%s (%s:%d) %s\n     names %s, which does not exist in this tree.",
					it.ID, cfg.VerifierFile, it.Line, it.Title, it.Test))
			case results[it.Test] != "PASS":
				problems = append(problems, fmt.Sprintf(
					"%s (%s:%d) %s\n     names %s, which %sed.",
					it.ID, cfg.VerifierFile, it.Line, it.Title, it.Test,
					strings.ToLower(results[it.Test])))
			default:
				verified++
			}
		}
	}

	if len(problems) == 0 {
		cfg.log("acceptance: %d/%d verifier item(s) proven by a passing test", verified, len(items))
		r.add(Check{Name: "verifier", OK: true})
		return
	}
	detail := fmt.Sprintf("%d of %d item(s) in %s are not settled:\n\n  - %s",
		len(problems), len(items), cfg.VerifierFile, strings.Join(problems, "\n  - "))
	detail += "\n\nAn item is settled one of two ways: a test that exists and passes, or a\n" +
		"withdrawal that quotes the task to show the item was never required."
	r.add(Check{Name: "verifier", OK: false, Detail: detail})
}

// runNamedTests runs exactly the named tests and reports PASS/FAIL/SKIP for
// each. A name that never appears in the output does not exist, which is a
// different failure from one that ran and failed, and the two must not be
// reported as the same thing.
func runNamedTests(ctx context.Context, cfg Config, names []string) map[string]string {
	out := map[string]string{}
	if len(names) == 0 {
		return out
	}
	tctx, cancel := context.WithTimeout(ctx, cfg.timeout())
	defer cancel()

	pattern := "^(" + strings.Join(names, "|") + ")$"
	cmd := exec.CommandContext(tctx, "go", "test", "-run", pattern, "-v", "./...")
	cmd.Dir = cfg.Workspace
	cmd.Env = safeenv.Environ()     // runs the agent's code: no harness credentials
	text, _ := cmd.CombinedOutput() // a failing test is the normal case here

	for _, m := range rePassLine.FindAllStringSubmatch(string(text), -1) {
		out[m[2]] = m[1]
	}
	return out
}

func keysOf(m map[string][]Item) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}
