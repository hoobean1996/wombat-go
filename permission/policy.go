// Package permission decides whether a tool call is allowed to run, and asks a
// human when the answer is "it depends".
//
// # Why this exists
//
// The harness already had a guardrail: builtin.OSFS(root) confines the file
// tools to a subtree. It constrains view_file and write_file and it does
// exactly nothing to bash — which was demonstrated live, in one run, when
// view_file was refused for /etc/hosts and the model read the same file with
// `cat /etc/hosts` on the next turn. A guardrail that one tool walks around is
// not a guardrail, it is a UX affordance with a security-sounding name.
//
// The fix is not a better path parser. It is admitting that some calls cannot
// be judged by the harness at all and must be put in front of a person. So the
// verdict has three values, not two: [Allow], [Deny] and [Ask].
//
// # How it fits together
//
//	p := permission.Workspace("/work")
//	h := tool.Chain(inner, permission.Gate(p, approver))
//
// A [Policy] is an ordered list of [Rule]s and a fallback. [Gate] turns a
// policy into tool middleware: it resolves the verdict, blocks on the
// [Approver] when the verdict is [Ask], emits [Requested] and [Decided] for a
// UI, logs every decision through slog, and refuses the call with an error the
// model can act on.
//
// # Fail closed
//
// [Undecided] is the zero [Decision] and means "this rule has no opinion".
// A Policy whose rules all decline falls back to Policy.Default — and a zero
// Default is Undecided, which [Policy.Decide] resolves to [Deny]. So the
// zero Policy denies everything. That is deliberate: the failure mode of a
// half-configured permission system must be refusal, never permission.
package permission

import (
	"context"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// Decision is the verdict on one tool call.
type Decision int

// Verdicts. Undecided is the zero value so that a rule which forgets to answer
// declines rather than accidentally permitting.
const (
	Undecided Decision = iota // this rule has no opinion; fall through
	Allow
	Deny
	Ask
)

// String implements fmt.Stringer, so a decision reads as itself in a log line.
func (d Decision) String() string {
	switch d {
	case Allow:
		return "allow"
	case Deny:
		return "deny"
	case Ask:
		return "ask"
	default:
		return "undecided"
	}
}

// Request is one call awaiting a verdict.
//
// It carries the resolved [tool.Def] rather than a name, because the rules
// judge on Caps and Needs — what the tool DOES and what it wants FROM THE HOST
// — and re-deriving those from a name would need a second lookup and a second
// source of truth.
type Request struct {
	Tool tool.Def
	Use  llm.ToolUse

	// Reason is filled in by the rule that decided, for the human. It is empty
	// while the policy is still being evaluated and set by [Gate] before the
	// request reaches an [Approver], so the question a person is asked is the
	// question the policy actually asked.
	Reason string
}

// Rule decides about one call, or declines to.
//
// Returning ([Undecided], "") is the polite way to say "not my department" and
// is what every rule must do for calls it does not recognise: a rule that
// denies everything it has not been taught about cannot be composed with
// another rule.
//
// A rule receives the context so it can consult per-run state, and must not
// block on it — the one place that is allowed to wait for a human is the
// [Approver], behind an [Ask].
type Rule func(context.Context, Request) (Decision, string)

// Policy is an ordered list of rules and what happens when none decides.
//
// Order is the whole semantics: the first rule to return anything other than
// [Undecided] wins, so a DenyTools placed after an AllowTools naming the same
// tool never fires. Put the exceptions first and the broad strokes last.
type Policy struct {
	Rules   []Rule
	Default Decision
}

// Decide runs the rules in order and returns the first real verdict, together
// with the reason the deciding rule gave.
//
// When no rule decides, Policy.Default applies — and a zero Default (that is,
// [Undecided]) resolves to [Deny]. The zero Policy therefore refuses
// everything. Nil rules are skipped rather than panicking, so a caller can
// build a rule list with conditional entries.
func (p Policy) Decide(ctx context.Context, r Request) (Decision, string) {
	for _, rule := range p.Rules {
		if rule == nil {
			continue
		}
		if d, why := rule(ctx, r); d != Undecided {
			return d, why
		}
	}
	switch p.Default {
	case Allow:
		return Allow, "no rule objected and this policy allows by default"
	case Deny:
		return Deny, "no rule allowed this call and this policy denies by default"
	case Ask:
		return Ask, "no rule decided, and this policy asks about anything it was not told about"
	default:
		return Deny, "no rule decided and this policy has no default, so it fails closed"
	}
}

// ===== Ready-made policies =====

// ReadOnly permits tools that only read and denies everything else.
//
// "Only read" means exactly [tool.CapReadOnly] and nothing besides — a tool
// that is also CapNetwork is refused, because a call that can read the
// filesystem and reach the network is an exfiltration primitive whatever its
// individual halves are called. A tool that declares no capability at all is
// refused too: undeclared is not the same as harmless.
//
// Pause and terminal tools are allowed: their handler never runs, so they have
// no effect to restrict, and a read-only agent that cannot ask its user a
// question or hand in its result is a broken agent rather than a safe one.
func ReadOnly() Policy {
	return Policy{
		Rules: []Rule{
			controlFlow(),
			func(_ context.Context, r Request) (Decision, string) {
				if r.Tool.Caps == tool.CapReadOnly {
					return Allow, "the tool only reads"
				}
				return Undecided, ""
			},
		},
		Default: Deny,
	}
}

// Workspace confines a run to root: reads and writes inside it are free,
// everything else goes in front of a person.
//
// Precisely: pause and terminal tools allow, CapMeta allows, a self-contained
// read-only tool allows, an obviously harmless shell command allows, any other
// CapExec asks, paths inside root allow, and anything left over asks.
//
// # The order, and why it is this order
//
//  0. Pause and terminal tools first. Their handler never runs: a pause asks
//     the person a question and a terminal call hands in the result, and
//     asking that person for permission to ask them something is the purest
//     form of the approval fatigue step 1 is about.
//
//  1. CapMeta next. delegate and load_skill spawn a sub-agent or paste text
//     into the transcript, and the sub-agent's calls are gated too — by this
//     same policy, which [Gate] carries into the sub-agent with the delegate
//     call, and by any gate of its own. Asking about the orchestration itself
//     trains the human to click yes, which is the only real failure mode of an
//     approval prompt.
//
//  2. Then the tools that touch nothing — a calculator, a clock. They cannot
//     be "inside" or "outside" a directory, and without this they would fall
//     through to the default and interrupt a person to ask about arithmetic.
//
//  3. Then [SafeCommands], which allows `go test ./...` and its two hundred
//     spellings without a prompt. It must come before the CapExec rule below,
//     because that rule asks about every shell call and the first decisive
//     rule wins. SafeCommands never denies, so putting it here can only turn
//     an Ask into an Allow — never the reverse — and everything it does not
//     recognise falls straight into step 4. Read its limitations: it is a
//     heuristic over a string, and it is the one rule in this policy that
//     makes the answer less conservative than it was.
//
//  4. Then CapExec, which ASKS. It has to come before the path rule so the
//     reason a person reads is about the shell command rather than about
//     whichever incidental path argument the call happened to carry. See
//     [FSRoot] for why a shell command cannot be judged any other way.
//
//  5. Then the path check, which is [FSRoot] with its Deny softened to Ask:
//     inside root is allowed outright, and outside root is a question rather
//     than a refusal. That is the difference between this policy and
//     [FSRoot] used alone — a workspace is a default, not a wall.
//
//  6. Default Ask. Network calls, anything a rule did not recognise: the
//     human decides.
//
// Note that a read OUTSIDE root asks rather than being free. Reading
// ~/.aws/credentials is a read, and the whole point of a workspace is that it
// bounds the run in both directions.
//
// root is resolved to an absolute path when the policy is built; see [FSRoot]
// for the containment check's limits, which are real.
func Workspace(root string) Policy {
	return Policy{
		Rules: []Rule{
			controlFlow(),
			allowCaps(tool.CapMeta, "orchestration, not an effect on the world"),
			selfContained(),
			SafeCommands(root),
			AskFor(tool.CapExec),
			insideRoot(root),
		},
		Default: Ask,
	}
}

// AskEverything puts every call in front of a person. It has no rules at all:
// useful as a starting point, for a demo, and for the first hour of a run
// against an unfamiliar tool set.
func AskEverything() Policy {
	return Policy{Default: Ask}
}

// allowCaps allows a tool carrying ANY of caps. Unexported because the pinned
// API deliberately has no allow-by-capability rule: an allowlist by capability
// is how a policy grows a hole nobody notices, and the two places it is
// genuinely right are both in this file.
func allowCaps(caps tool.Cap, why string) Rule {
	return func(_ context.Context, r Request) (Decision, string) {
		if r.Tool.Caps&caps != 0 {
			return Allow, why
		}
		return Undecided, ""
	}
}

// controlFlow allows pause and terminal tools.
//
// They reach a policy at all only because the agent loop sends them through
// the dispatch chain for a verdict before it ends or suspends a run — and their
// handler never runs, whatever the verdict, so there is no effect on the world
// to judge. Any capability bit beside the control one is moot for the same
// reason. A policy that wants to forbid asking the user, say for a headless
// run nobody can answer, says so by name with [DenyTools] ahead of these.
func controlFlow() Rule {
	return func(_ context.Context, r Request) (Decision, string) {
		if r.Tool.Caps&(tool.CapPause|tool.CapTerminal) != 0 {
			return Allow, "the tool only suspends or ends the run, and its handler never runs"
		}
		return Undecided, ""
	}
}

// selfContained allows a read-only tool that asks nothing of the host: no
// filesystem, no subprocess, no network. Whatever it does happens inside this
// process and cannot be constrained by a path.
func selfContained() Rule {
	return func(_ context.Context, r Request) (Decision, string) {
		if r.Tool.Caps == tool.CapReadOnly && r.Tool.Needs == 0 {
			return Allow, "the tool reads nothing outside this process"
		}
		return Undecided, ""
	}
}

// insideRoot is [FSRoot] with its Deny softened to Ask.
//
// FSRoot alone is a wall: outside the root is refused and the model must find
// another way. Inside a workspace policy the same fact should be a question,
// because "read the file I just told you about, which happens to live in
// /tmp" is a normal thing for a person to want.
func insideRoot(root string) Rule {
	within := FSRoot(root)
	return func(ctx context.Context, r Request) (Decision, string) {
		d, why := within(ctx, r)
		if d == Deny {
			return Ask, why
		}
		return d, why
	}
}
