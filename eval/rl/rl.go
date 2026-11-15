// Package rl evaluates an agent by running it many times and scoring what it
// did.
//
// The unit is an episode: one agent, one isolated workspace, one prompt, run
// to completion and then graded. n episodes of the same task form a [Group],
// which is the unit a pass@k is computed over, and a [Report] aggregates
// groups into the table a human reads.
//
//	env := rl.Dir("/tmp/bench", "todo-app", "write a todo app in main.go",
//	    rl.Score(
//	        rl.FileExists("main", "main.go", 0.3),
//	        rl.Shell("build", "go build ./...", 0.3),
//	        rl.Shell("test", "go test ./...", 0.4),
//	        rl.TurnPenalty(0.01),
//	    ))
//
//	g, err := rl.Rollout(ctx, mk, env, 8,
//	    rl.WithConcurrency(4),
//	    rl.WithBudget(governor.Limits{CostUSD: 0.50, Wall: 5 * time.Minute}))
//
// Three properties are load-bearing and everything else follows from them.
//
// Isolation. Samples run concurrently and each gets its own workspace and its
// OWN budget. Sharing either turns eight samples into one sample run eight
// times badly: a shared directory means two samples overwrite each other's
// work, and a shared budget means the first sample to spend it starves the
// rest, so the failure histogram measures the harness rather than the agent.
//
// Reconstruction from the event stream. An episode is driven through
// [core.Agent.Start] rather than [core.Agent.Run], because the per-turn
// facts — which tools were called, which failed, which permission refused, how
// many tokens each turn cost — exist only as events. Run discards them.
//
// Honest cost. A model with no entry in the pricing table costs $0.00, which
// after a real run reads as good news rather than as a missing table entry —
// sixteen episodes of a gateway model were once reported as free and the
// number was believed. So an episode records whether its spend was actually
// priced ([Episode.Priced], [WithPricing]), the report renders an unpriced
// COST as "n/a" and names the model, and the token columns are always there to
// compare runs by when the dollars cannot be. No rate is ever guessed: an
// estimate that looks like a measurement is the failure, not the fix.
package rl

import (
	"context"
	"strconv"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
)

// Task is one episode's work, already isolated from every other episode's.
type Task struct {
	// ID is stable across samples of the same task: "todo-app".
	ID string

	// Sample is which of the n samples this is, 0-based.
	Sample int

	// Prompt is what the agent is asked to do.
	Prompt string

	// Workspace is absolute, and this sample's alone.
	Workspace string
}

// Env is a resettable world.
//
// Reset must hand out a workspace no other concurrent sample can see; Score
// grades a finished episode; Cleanup disposes of the world afterwards. Reset
// is called once per sample and may run on several goroutines at once, so an
// implementation with shared state must say so or lock.
type Env interface {
	// Name identifies the environment in reports.
	Name() string

	// Reset prepares the world for one sample and returns its task.
	Reset(ctx context.Context, sample int) (Task, error)

	// Score grades a finished episode: a total reward and the per-verifier
	// breakdown that explains it.
	Score(ctx context.Context, ep *Episode) (float64, map[string]float64, error)

	// Cleanup disposes of the task's world. [Rollout] calls it on a context
	// marked with [WithKeep] when the artifacts are worth preserving.
	Cleanup(ctx context.Context, t Task) error
}

// Step is one ReAct turn, reconstructed from the event stream.
type Step struct {
	// Iteration is the loop's own 1-based turn number.
	Iteration int

	// Tools are the tool names called, in order. A call made by a sub-agent
	// appears as "sub/tool", since a delegated call is still work this turn
	// caused.
	Tools []string

	// Failed are the tools of this turn that errored.
	Failed []string

	// Denied are the tools of this turn that the permission gate refused.
	// A subset of Failed: a refusal is a failure with a known cause.
	Denied []string

	// Millis is how long the model call for this turn took.
	Millis int64

	// Usage is the token accounting for this turn, including any tokens a
	// sub-agent spent inside it.
	Usage llm.Usage
}

// Episode is one rollout: what happened and what it was worth.
type Episode struct {
	Task     Task
	Messages []llm.Message
	Steps    []Step

	// Outcome is how the run ended successfully, nil when it did not.
	Outcome core.Outcome

	// Err is why the run failed, nil when it did not.
	Err error

	// Reward is the total the environment scored, and Breakdown is the
	// per-verifier detail that adds up to it.
	Reward    float64
	Breakdown map[string]float64

	// Spend is the episode's own budget tally. Its own, not the rollout's:
	// see the package doc.
	Spend governor.Progress

	// Priced reports whether Spend.CostUSD is a measurement rather than the
	// absence of one.
	//
	// False when a model that answered is not in the pricing the rollout was
	// given, and false when tokens were spent and the tally is nonetheless
	// zero. Either way the number is not a price, and a report renders it as
	// "n/a": $0.0000 after a real run reads as good news rather than as a
	// missing table entry, which is exactly how sixteen episodes of a gateway
	// model were once reported as free and believed.
	//
	// The zero value is false, so an Episode nobody filled in reports its cost
	// as unknown rather than as zero. [Rollout] sets it on every episode
	// including the ones that never started — see [WithPricing].
	Priced bool

	// Unpriced names the models whose spend could not be priced, empty when
	// Priced. It can also be empty while Priced is false: a run whose stream
	// never named a model leaves nothing to name.
	Unpriced []string

	// Wall is how long the episode took end to end: reset, run, scoring and
	// cleanup. Wider than the agent's own time on purpose — it is what a
	// rollout of n samples actually costs in minutes.
	Wall time.Duration

	// Failure is why the episode did not succeed, [Success] when it worked.
	Failure FailureKind
}

// Turns reports how many ReAct iterations the episode took.
func (e *Episode) Turns() int { return len(e.Steps) }

// ToolCalls reports how many tool calls the episode made, sub-agent calls
// included.
func (e *Episode) ToolCalls() int {
	n := 0
	for _, s := range e.Steps {
		n += len(s.Tools)
	}
	return n
}

// ToolErrors reports how many tool calls failed, refusals included.
func (e *Episode) ToolErrors() int {
	n := 0
	for _, s := range e.Steps {
		n += len(s.Failed)
	}
	return n
}

// Usage reports the token accounting summed over every turn.
func (e *Episode) Usage() llm.Usage {
	var u llm.Usage
	for _, s := range e.Steps {
		u.Add(s.Usage)
	}
	return u
}

// PromptTokens is every token the episode sent to a model: fresh input plus
// cache reads plus cache writes.
//
// All three, because on a long agentic run the cached ones are most of the
// volume — a prompt-token column that counted only the fresh ones would report
// a fraction of what was actually processed, and this column exists to be
// comparable across runs when the cost column cannot be.
func (e *Episode) PromptTokens() int {
	u := e.Usage()
	return u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens
}

// OutputTokens is every token the episode's models generated.
func (e *Episode) OutputTokens() int { return e.Usage().OutputTokens }

// Label names the episode for a human: "task#sample".
func (e *Episode) Label() string {
	return e.Task.ID + "#" + strconv.Itoa(e.Task.Sample)
}

// FailureKind is why an episode did not succeed. Bounded on purpose: an error
// string is unbounded and a taxonomy you cannot count is not a taxonomy.
type FailureKind string

// The taxonomy. Every episode ends as exactly one of these.
const (
	// Success means the run finished cleanly and scored at or above the
	// success threshold.
	Success FailureKind = "success"

	// MaxIterations means the loop hit its cap, or the governor's step limit,
	// without a final answer.
	MaxIterations FailureKind = "max_iterations"

	// BudgetExceeded means the cost cap tripped.
	BudgetExceeded FailureKind = "budget"

	// WallClock means the episode ran out of time.
	WallClock FailureKind = "wall_clock"

	// ToolLoop means the agent kept making the same call, or exhausted the
	// tool-call cap.
	ToolLoop FailureKind = "tool_loop"

	// ContextWindow means the transcript outgrew the model.
	ContextWindow FailureKind = "context_window"

	// Refused means the model declined.
	Refused FailureKind = "refused"

	// MaxTokens means a reply was truncated.
	MaxTokens FailureKind = "max_tokens"

	// Denied means the episode ran out of road on permission.
	Denied FailureKind = "denied"

	// Panicked means something in the run panicked and was contained.
	Panicked FailureKind = "panic"

	// Cancelled means the caller's context ended the episode.
	Cancelled FailureKind = "cancelled"

	// VerifierFailed means the episode finished cleanly and produced the
	// wrong thing. The most interesting row in the table: nothing broke, the
	// agent simply did not do the task.
	VerifierFailed FailureKind = "verifier"

	// ProviderError means the model API failed in a way the harness could not
	// continue from.
	ProviderError FailureKind = "provider"

	// Other is everything else, and a large Other column means this taxonomy
	// is missing a case.
	Other FailureKind = "other"
)

// Kinds returns every [FailureKind] in report order: Success first, then the
// failures roughly by how much they implicate the agent rather than the
// harness.
//
// It exists because Go cannot check a switch over a string type for
// exhaustiveness, and a histogram that silently omits a kind reads as zero
// rather than as missing.
func Kinds() []FailureKind {
	return []FailureKind{
		Success, VerifierFailed, MaxIterations, ToolLoop, Denied, Refused,
		MaxTokens, ContextWindow, BudgetExceeded, WallClock, Cancelled,
		ProviderError, Panicked, Other,
	}
}

// AgentFunc builds the agent for one episode.
//
// A factory rather than one shared *Agent, and that is forced rather than
// stylistic: a tool captures its dependencies when it is CONSTRUCTED, so the
// workspace has to reach tool construction. Two samples sharing an agent would
// share a working directory and overwrite each other.
type AgentFunc func(t Task) (*core.Agent, error)
