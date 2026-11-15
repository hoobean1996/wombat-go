package main

import (
	"context"
	"fmt"
	"strings"

	"wombat/pkg/core"
	"wombat/pkg/core/accept"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/review"
	"wombat/pkg/core/sandbox"
)

// When a turn is done.
//
// An answer is a claim, not a delivery. Everything here hangs off the one
// mechanism the loop has for doubting it, [core.WithCompletionCheck]: when
// the model ends its turn, the gates run — done.check's command, then
// done.accept's checks — and the first that is not satisfied sends the model
// back with what it said.
// Only an answer that passes them all is reviewed. The review never sends the
// builder back: its own fixer repairs what it can, and what it found reaches
// the stream as a [review.Reviewed] event, next to the answer it is about.
//
// All of it runs inside the turn, on the turn's context: it is held to
// turn.budget_usd and turn.wall, its calls are held to the session's policy,
// and every command it runs goes through the session's runner,
// tools.confine and all. A reviewer or a fixer cannot stop for a decision —
// the turn is not theirs to end — so a call of theirs the policy defers is
// not made, and the review reports what it could do without it.

// turn is what the session's own agent gets that its reviewers and fixers do
// not: the instructions a mode adds, what happens before each turn, and the
// gates an answer has to pass.
func (f *factory) turn(s session, runner sandbox.Runner) []core.Option {
	var opts []core.Option
	switch f.cfg.Mode {
	case modeReview:
		opts = append(opts, core.WithEnvBlock("review_only", reviewOnlyNote))
	case modePlan:
		opts = append(opts, core.WithEnvBlock("planning", planInstructions))
	}

	// No grants here: the daemon gives every turn its own, on the turn's
	// context, so a delegated task or a reviewer — which runs on a context
	// derived from it — is not asked again about a call the person already
	// approved.

	if f.cfg.Mode == modeBuild && (f.cfg.Done.Accept || f.cfg.Done.Regressions != "") {
		opts = append(opts, core.WithRunContext(func(ctx context.Context) context.Context {
			f.beforeTurn(s.workspace)
			return ctx
		}))
	}

	if check := f.completion(s, runner); check != nil {
		opts = append(opts, core.WithCompletionCheck(f.cfg.Done.Attempts, check))
	}
	return opts
}

// beforeTurn prepares the workspace for a turn the done.accept gate will judge.
//
// The plan census is counted first, so the gate can tell a finished list from
// a shortened one: a deleted item is not an unticked item, and shortening the
// list has been the shortcut taken on three runs across two tasks. The
// regression tests are installed rather than checked afterwards: a defect the
// ledger knows about should be a failing test the builder sees from its first
// `go test`, not something a review has to find again.
func (f *factory) beforeTurn(workspace string) {
	if f.cfg.Done.Accept {
		if err := accept.RecordCensus(workspace, review.DefaultTodoFile, accept.DefaultVerifierFile); err != nil {
			f.logger.Warn("could not record the plan census", "err", err)
		}
	}
	dir := f.cfg.Done.Regressions
	if dir == "" {
		return
	}
	led, err := accept.LoadLedger(dir)
	if err == nil {
		err = led.Install(workspace)
	}
	if err != nil {
		f.logger.Warn("could not install the regression tests", "from", dir, "err", err)
		return
	}
	if n := len(led.Files); n > 0 {
		f.logger.Info("installed regression tests", "count", n, "from", dir)
	}
}

// completion composes the gates and the review into the one check the agent
// runs when the model ends its turn, or nil when none is configured.
func (f *factory) completion(s session, runner sandbox.Runner) core.CompletionCheck {
	// A review session builds nothing for a gate to judge, and a plan is not
	// held to the checks a build is.
	var gates []core.CompletionCheck
	if f.cfg.Done.Check != "" && f.cfg.Mode != modeReview {
		gates = append(gates, core.CommandCheck(runner, f.cfg.Done.Check, 0))
	}
	if f.cfg.Done.Accept && f.cfg.Mode == modeBuild {
		gates = append(gates, f.acceptGate(s.workspace, runner))
	}
	var after core.CompletionCheck
	if f.cfg.Review.Enabled || f.cfg.Mode == modeReview {
		after = f.reviewAfter(s, runner)
	}
	if len(gates) == 0 && after == nil {
		return nil
	}
	return func(ctx context.Context, c core.Completion) (string, error) {
		for _, gate := range gates {
			if feedback, err := gate(ctx, c); err != nil || feedback != "" {
				return feedback, err
			}
		}
		if after == nil {
			return "", nil
		}
		return after(ctx, c)
	}
}

// acceptGate runs the deterministic checks, and turns a failing report into
// the work the model is sent back with.
func (f *factory) acceptGate(workspace string, runner sandbox.Runner) core.CompletionCheck {
	return func(ctx context.Context, _ core.Completion) (string, error) {
		rep := accept.Run(ctx, accept.Config{
			Workspace:     workspace,
			RequireTests:  true,
			TodoFile:      review.DefaultTodoFile,
			VerifierFile:  accept.DefaultVerifierFile,
			RegressionDir: f.cfg.Done.Regressions,
			Shell:         runner.Shell,
			Log:           func(line string) { f.logger.Info(line) },
		})
		if rep.OK() {
			return "", nil
		}
		return accept.Feedback(rep), nil
	}
}

// reviewAfter hands the workspace to fresh reviewers once an answer has
// passed every gate.
//
// Fresh agents rather than another turn of the builder: the builder has just
// argued itself into believing the code is correct, and asking it to find its
// own defects asks it to overturn the conclusion it reached one message ago.
// Measured against three defects a human review had confirmed, a fresh
// reviewer found two and filed no false positives, for about five cents.
func (f *factory) reviewAfter(s session, runner sandbox.Runner) core.CompletionCheck {
	return func(ctx context.Context, c core.Completion) (string, error) {
		cfg := review.Config{
			// A factory, not one agent: every sweep needs a reader that has not
			// already concluded the code is fine.
			NewReviewer: func() (*core.Agent, error) { return f.agent(s, reviewer) },
			Sweeps:      f.cfg.Review.Sweeps,
			DryStreak:   f.cfg.Review.Dry,
			Workspace:   s.workspace,
			Task:        taskOf(c.Messages),
			Rounds:      f.cfg.Review.Fix,
			Shell:       runner.Shell,
			Log:         func(line string) { f.logger.Info(line) },
		}
		if f.cfg.Review.Fix > 0 {
			fx, err := f.agent(s, fixer)
			if err != nil {
				f.logger.Warn("review: no fixer", "err", err)
			} else {
				cfg.Fixer = fx
			}
		}
		rep := review.Run(ctx, cfg)
		f.promote(rep, s.workspace)
		core.Emit(ctx, rep.Event())
		return "", nil
	}
}

// promote makes a repaired finding a permanent guard. The test that
// demonstrated the defect is exactly the thing that stops it returning, and
// promoting it is what makes the next run start from what this one learned
// instead of from nothing.
func (f *factory) promote(rep *review.Report, workspace string) {
	dir := f.cfg.Done.Regressions
	if dir == "" {
		return
	}
	seen := map[string]bool{}
	for _, fd := range rep.Findings {
		if !fd.Fixed || fd.File == "" || seen[fd.File] {
			continue
		}
		seen[fd.File] = true
		if err := accept.Promote(dir, workspace, fd.File); err != nil {
			f.logger.Warn("could not promote regression", "file", fd.File, "err", err)
			continue
		}
		f.logger.Info("promoted to the regression ledger", "file", fd.File)
	}
}

// taskOf is what a reviewer is told the code was meant to do: the session's
// first message, which is the task a build session was given.
func taskOf(msgs []llm.Message) string {
	for _, m := range msgs {
		if m.Role != llm.RoleUser {
			continue
		}
		if t := strings.TrimSpace(llm.TextOf(m.Content)); t != "" {
			return t
		}
	}
	return ""
}

// iterationNotice tells the model how much of its iteration budget is left.
//
// It had no way to know. A measured run spent all 80 of its iterations writing
// correct code and stopped mid-work, with the tests the task explicitly asked
// for never written — not a misjudgement, since nothing in the loop said how
// much room was left or that anything was still outstanding.
//
// It starts at two thirds rather than at the end. A warning delivered with two
// iterations left can only be obeyed by stopping; one delivered with a third of
// the budget left can be obeyed by finishing.
//
// The note goes into the materialized request only, never the transcript, so it
// cannot accumulate.
func iterationNotice(maxIters int) func(context.Context, int) string {
	if maxIters <= 0 {
		return nil
	}
	const startAt = 2.0 / 3.0
	return func(_ context.Context, iter int) string {
		if float64(iter) < startAt*float64(maxIters) {
			return ""
		}
		left := max(maxIters-iter, 0)
		return fmt.Sprintf(
			"<budget_status>Iteration %d of %d; %d left, and the run ends when they "+
				"are gone.\n\nBefore doing anything else, check what the task asked for "+
				"that does not exist yet — re-read your checklist file if you wrote one, "+
				"and the task's own list of requirements. Finish what is missing before "+
				"improving what is already there. Do not shorten a checklist to make it "+
				"look complete.</budget_status>",
			iter, maxIters, left)
	}
}

// reviewOnlyNote is what review mode tells the session's own agent, which has
// no tools: the review that follows its answer is the session's work.
const reviewOnlyNote = `This session reviews code; it writes none. The user's message says what the code in the working directory was meant to do. Reply with one sentence restating that task, and end your turn: a review of the code starts as soon as you do, by reviewers who have not seen it before.`

// planInstructions turns every session's task into a planning task.
//
// The split exists because the two halves need different things. Knowing what
// must be verified is a question of breadth — how many ways this kind of system
// has failed before — and a measured run showed the cost of missing one: five of
// nine defects in one build came from a single unconsidered question, whether a
// task handle could be confused with an integer. Knowing how to write the code
// needs the code in front of you, which planning does not have.
//
// So the plan is deliberately not asked for test code. Before any of the API
// exists, a test file is a guess at names and signatures, and a wrong guess does
// not compile and blocks the build. Prose that names its own test is checkable
// without being written blind.
const planInstructions = `Plan the task the user gives you. Write no implementation code and no test code.

Produce exactly three files in the working directory, then answer with a one
paragraph summary.

PLAN.md — the invariants, stated as facts that hold at every moment, each with
the paths that establish it and the paths that could break it. Architecture and
file layout.

TODO.md — an ordered markdown checklist, "- [ ] item", covering everything the
task requires, including a test for each invariant. Nothing in it may depend on
something below it.

Verifier.md — what must be TRUE of the finished work, one item per heading:

## V1. <the single thing that must hold>
Given: <the situation that exercises it>
Expect: <what must happen, specifically>
Status: [ ]
Test:

An item that a COMMAND decides — the build succeeds, vet is clean, a suite
passes, the CLI exits zero on some input — says so instead, and needs no Given
or Expect:

## V2. the three acceptance commands succeed
Command: go test -race ./...
Status: [ ]

Use Command for anything the task states AS a command. Forcing those through a
test function adds a layer that can fail on its own: asked to prove "go test
-race ./... passes" with a test, one build wrote a test that ran the whole
suite as a subprocess, and the suite then took longer than the requirement
allowed — defeated by the code written to prove it.

Whoever builds this will fill in Status and Test. An item is settled only by a
test that exists and passes, so write each one so a test could decide it: not
"errors are handled well" but "dividing by zero returns an *Error whose Line is
the line of the division".

Spend your effort on the items nobody would think to write down. Requirements
get implemented; what gets shipped broken is the consequence of a requirement
that nobody stated. For each value the system represents, ask whether two kinds
of it can be confused. For each thing acquired, ask what releases it. For each
"after X, Y is true", ask what must stop being allowed once Y is true. For each
operation that reports success, ask what is true when its last step fails.

Cover the task's stated requirements too, but those are the cheap half.`
