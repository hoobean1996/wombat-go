package wombat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/tool"
	"github.com/hoobean1996/wombat-go/trace"
)

// ===== Events =====

// SubagentStart opens one delegated task.
//
// Depth is the nesting level the child runs at, counted by the governor, so a
// front end can indent without tracking the tree itself.
type SubagentStart struct {
	Name  string `json:"name"`
	Task  string `json:"task"`
	Depth int    `json:"depth"`
}

// SubagentEnd closes one delegated task.
type SubagentEnd struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Error  string `json:"error,omitempty"`
	Millis int64  `json:"ms"`
}

// SubagentEvent is one of the child's own events, forwarded to the parent
// stream with the child's identity attached.
//
// Wrapped rather than re-emitted bare, because the parent's stream is a
// timeline of the parent's work: a child's TextDelta interleaved raw would
// read as the parent talking, and under a fan-out four children's deltas would
// read as one incoherent voice. Wrapping lets a UI nest or collapse the child,
// and lets a consumer that does not care skip one case in its type switch.
type SubagentEvent struct {
	Name  string `json:"name"`
	Depth int    `json:"depth"`
	Inner Event  `json:"inner"`
}

// Kind implements Event.
func (SubagentStart) Kind() string { return "subagent_start" }

// Kind implements Event.
func (SubagentEnd) Kind() string { return "subagent_end" }

// Kind implements Event.
func (SubagentEvent) Kind() string { return "subagent_event" }

// MarshalJSON implements json.Marshaler.
func (e SubagentStart) MarshalJSON() ([]byte, error) {
	type r SubagentStart
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements json.Marshaler.
func (e SubagentEnd) MarshalJSON() ([]byte, error) {
	type r SubagentEnd
	return eventJSON(e.Kind(), r(e))
}

// MarshalJSON implements json.Marshaler.
//
// Inner is an interface, so it marshals through its own MarshalJSON and keeps
// its own "type" discriminator: the wrapper nests rather than flattens, and a
// consumer reads the inner event with exactly the code it already has.
func (e SubagentEvent) MarshalJSON() ([]byte, error) {
	type r SubagentEvent
	return eventJSON(e.Kind(), r(e))
}

// ===== Errors =====

// Reasons a delegated task did not produce an answer. Match with errors.Is.
var (
	// ErrDelegateTooDeep means the governor refused another nesting level.
	ErrDelegateTooDeep = errors.New("wombat: delegation is nested too deep")

	// ErrSubagentPaused means the child called a tool with [tool.CapPause].
	// A sub-agent has no user to ask, and inventing an answer on the user's
	// behalf would be worse than failing.
	ErrSubagentPaused = errors.New("wombat: sub-agent asked for user input")

	// ErrEmptyTask means the model called a delegate tool with nothing to do.
	ErrEmptyTask = errors.New("wombat: delegate called with an empty task")
)

// ===== Options =====

// Default names and limits for the delegate tools.
const (
	DefaultDelegateName         = "delegate"
	DefaultParallelDelegateName = "parallel_delegate"
	DefaultMaxBranches          = 4
)

type delegateConfig struct {
	name        string
	description string
	maxBranches int
}

// DelegateOption configures [DelegateTool] and [ParallelDelegateTool].
type DelegateOption func(*delegateConfig)

// WithDelegateName renames the tool. Default [DefaultDelegateName] or
// [DefaultParallelDelegateName].
//
// Worth setting when an agent has more than one child: "delegate_to_reviewer"
// and "delegate_to_researcher" tell the model what each child is FOR, which no
// amount of description text conveys as cheaply as the name it has to type.
func WithDelegateName(name string) DelegateOption {
	return func(c *delegateConfig) { c.name = name }
}

// WithDelegateDescription replaces the description shown to the model.
//
// The default describes delegation in general. A specialised child deserves
// better: say what this particular sub-agent knows and what a good task for it
// looks like.
func WithDelegateDescription(text string) DelegateOption {
	return func(c *delegateConfig) { c.description = text }
}

// WithMaxBranches caps how many branches of one [ParallelDelegateTool] call
// run concurrently. Default [DefaultMaxBranches]. Ignored by [DelegateTool].
//
// Panics on n < 1: a fan-out that can never start a branch is a construction
// bug, and discovering it as a hung tool call at run time is much worse.
func WithMaxBranches(n int) DelegateOption {
	if n < 1 {
		panic("wombat: WithMaxBranches requires n >= 1")
	}
	return func(c *delegateConfig) { c.maxBranches = n }
}

// ===== delegate =====

const delegateDescription = "Hand one self-contained sub-task to a fresh sub-agent and get back its " +
	"final answer.\n\n" +
	"The sub-agent starts from an EMPTY conversation. It sees NONE of this conversation: not the " +
	"user's request, not your instructions, not the files you have read, not your earlier tool " +
	"results. Everything it needs must be spelled out in `task` — full paths, full names, the exact " +
	"question, and what the answer should contain. A task that begins \"also check the other one\" " +
	"is guaranteed to fail.\n\n" +
	"Delegate when the work is (a) self-contained, (b) several tool calls deep, and (c) noisy — " +
	"searching a large tree, reading many files, exploratory debugging. The intermediate output " +
	"then lands in the sub-agent's context instead of yours, and you get back only the conclusion.\n\n" +
	"Do NOT delegate a single tool call you could make yourself; do NOT delegate work that depends " +
	"on context you cannot write down; do NOT delegate anything that needs a decision from the " +
	"user, because the sub-agent has no user to ask and will fail.\n\n" +
	"You get back the sub-agent's final answer as text. If it fails you get its error, and you can " +
	"retry with a sharper task or do the work yourself."

const delegateSchema = `{
  "type": "object",
  "properties": {
    "task": {
      "type": "string",
      "description": "A complete, self-contained task description. Assume the reader knows nothing about this conversation. State the goal, the inputs (full paths, exact names, literal values) and the shape of the answer you want back."
    }
  },
  "required": ["task"]
}`

type delegateIn struct {
	Task string `json:"task"`
}

// DelegateTool exposes child as a tool the parent's model can call.
//
// Panics on a nil child: an agent whose delegate tool has nothing to delegate
// to is a wiring bug, and the only honest time to report it is construction.
func DelegateTool(child *Agent, opts ...DelegateOption) tool.Def {
	if child == nil {
		panic("wombat: DelegateTool requires a non-nil child agent")
	}
	cfg := delegateConfig{name: DefaultDelegateName, description: delegateDescription}
	for _, o := range opts {
		o(&cfg)
	}

	return tool.Typed(tool.Def{
		Name:        cfg.name,
		Description: cfg.description,
		InputSchema: json.RawMessage(delegateSchema),

		// Meta: this tool's effect is on the agent's own structure. Its Needs
		// stay zero because the child's tools declare their own — the host
		// gates those where they are defined, not here.
		Caps: tool.CapMeta,

		// Not idempotent, and not close: a delegated task can write files,
		// spend money and page someone. Replaying it because the first attempt
		// looked transient would do all of that twice.
		Idempotent: false,

		// No timeout of its own. The child has an iteration cap and the run
		// budget bounds wall clock, cost and steps for the whole tree; a
		// second, tighter clock here would kill useful work for no new
		// guarantee. See the package note about the dispatcher's fallback.
		Timeout:  0,
		Category: "meta",
	}, func(ctx context.Context, in delegateIn) (string, error) {
		task := strings.TrimSpace(in.Task)
		if task == "" {
			return "", fmt.Errorf("%w: describe the whole task in the 'task' field", ErrEmptyTask)
		}

		b := governor.FromContext(ctx)
		if !b.EnterSubagent() {
			return "", tooDeep()
		}
		defer b.ExitSubagent()

		depth := b.Progress().Depth
		out, err := runChild(ctx, child, cfg.name, depth, task)
		if err != nil {
			return "", err
		}
		return out, nil
	})
}

// ===== parallel_delegate =====

const parallelDelegateDescription = "Hand SEVERAL independent sub-tasks to sub-agents that run at the " +
	"same time, and get all their answers back in one result.\n\n" +
	"Each sub-agent starts from an EMPTY conversation and sees NONE of this one — not the user's " +
	"request, not your instructions, not your earlier tool results — and none of the other tasks. " +
	"Every entry in `tasks` must therefore stand completely on its own: full paths, exact names, " +
	"the precise question, the shape of the answer.\n\n" +
	"Use this when you have work that decomposes into pieces with NO ordering between them: " +
	"researching unrelated questions, auditing several files, trying three approaches to the same " +
	"problem to compare. If task 2 needs task 1's answer, this is the wrong tool — delegate them " +
	"one at a time, or do the work yourself.\n\n" +
	"Results come back in the order you listed the tasks, each labelled with its task. A branch " +
	"that fails reports its error in place; the others still run, so you get a partial result set " +
	"rather than nothing. Sub-agents have no user to ask, so a task that needs a human decision " +
	"will fail.\n\n" +
	"Two or three focused tasks beat ten vague ones: every branch spends from the same run budget."

const parallelDelegateSchema = `{
  "type": "object",
  "properties": {
    "tasks": {
      "type": "array",
      "description": "Independent, self-contained task descriptions, one per sub-agent. No shared context between them and no ordering; assume each reader knows nothing about this conversation.",
      "items": { "type": "string" },
      "minItems": 1
    }
  },
  "required": ["tasks"]
}`

type parallelDelegateIn struct {
	Tasks []string `json:"tasks"`
}

// ParallelDelegateTool fans a list of independent tasks out across copies of
// child, running up to [WithMaxBranches] of them at once.
//
// What this replaced is worth recording. The OCaml original spawned a
// [Domain] per branch and then had to rebuild the world inside it: effect
// handlers do not cross domains, so each branch re-installed the whole LLM /
// tool / log / time stack from a forked child config, and reached for Obj.magic
// to make the types line up. Here a branch is a goroutine that closes over the
// same context, so the model client, the tool set, the budget and the emitter
// are simply still there. The shared [governor.Budget] is also what keeps the
// fan-out honest: every branch charges the same tally, so four children cannot
// spend four budgets, and the first one to cross a cap cancels the context that
// all of them are running on.
//
// Panics on a nil child, for the reason [DelegateTool] does.
func ParallelDelegateTool(child *Agent, opts ...DelegateOption) tool.Def {
	if child == nil {
		panic("wombat: ParallelDelegateTool requires a non-nil child agent")
	}
	cfg := delegateConfig{
		name:        DefaultParallelDelegateName,
		description: parallelDelegateDescription,
		maxBranches: DefaultMaxBranches,
	}
	for _, o := range opts {
		o(&cfg)
	}

	return tool.Typed(tool.Def{
		Name:        cfg.name,
		Description: cfg.description,
		InputSchema: json.RawMessage(parallelDelegateSchema),
		Caps:        tool.CapMeta,
		Idempotent:  false,
		Timeout:     0,
		Category:    "meta",
	}, func(ctx context.Context, in parallelDelegateIn) (string, error) {
		tasks := make([]string, len(in.Tasks))
		for i, t := range in.Tasks {
			tasks[i] = strings.TrimSpace(t)
			if tasks[i] == "" {
				return "", fmt.Errorf("%w: tasks[%d] is empty", ErrEmptyTask, i)
			}
		}
		if len(tasks) == 0 {
			return "", fmt.Errorf("%w: 'tasks' must list at least one task", ErrEmptyTask)
		}

		// One tick for the whole fan-out. The cap is RECURSION depth, not
		// sibling count: N children at the same nesting level are depth+1, not
		// depth+N. Counting them individually would make a wide fan-out look
		// like a runaway recursion and abort a perfectly legal run.
		b := governor.FromContext(ctx)
		if !b.EnterSubagent() {
			return "", tooDeep()
		}
		defer b.ExitSubagent()
		depth := b.Progress().Depth

		type branch struct {
			out string
			err error
		}
		results := make([]branch, len(tasks))

		// A plain semaphore rather than an errgroup, because the failure
		// policy is the opposite of errgroup's: a branch that fails must not
		// cancel its siblings. Three good answers and one error is a result
		// the parent model can work with; four cancellations is not.
		sem := make(chan struct{}, cfg.maxBranches)
		var wg sync.WaitGroup
		for i, task := range tasks {
			wg.Add(1)
			go func() {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				// Branches emit into the parent's stream concurrently, so
				// their events WILL interleave. That is what the name on
				// SubagentEvent is for — it is the only thing a consumer can
				// demultiplex on.
				label := fmt.Sprintf("%s[%d]", cfg.name, i)
				out, err := runChild(ctx, child, label, depth, task)
				results[i] = branch{out: out, err: err}
			}()
		}
		wg.Wait()

		// Returned as a normal result even when branches failed, never as an
		// error. A tool error replaces the output with the error string, and
		// that would throw away the branches that did work — the parent model
		// would see "it failed" instead of three usable answers and one
		// diagnosis.
		var sb strings.Builder
		for i, r := range results {
			if i > 0 {
				sb.WriteString("\n\n")
			}
			fmt.Fprintf(&sb, "=== task %d of %d: %s ===\n", i+1, len(tasks), preview(tasks[i], 160))
			if r.err != nil {
				sb.WriteString("[failed] ")
				sb.WriteString(r.err.Error())
				continue
			}
			sb.WriteString(r.out)
		}
		return sb.String(), nil
	})
}

// ===== shared machinery =====

// runChild runs one child agent to completion on the PARENT's context and
// forwards its events to the parent's stream.
//
// Running on the parent's context is the payoff of the whole context-scoped
// design, and the reason delegation is a tool rather than a subsystem. The
// budget, the skill activation set, the transcript tape, the tracer and the
// cancellation that a governor abort produces are all values on ctx, so the
// child inherits every one of them by inheriting the context — no runtime to
// fork, no handler stack to reinstall, no child config to thread through. The
// one thing that does NOT carry over is any state the child agent's own
// [WithRunContext] decorators re-create; that is the intended escape hatch for
// a child that must start with, say, a clean skill activation set.
//
// [Agent.Start] installs the CHILD's emitter inside the child's context, so
// events raised inside the child land in the child's channel and never in the
// parent's. Draining that channel here and re-emitting on the parent's ctx is
// therefore a clean hand-off rather than a race between two sinks.
func runChild(ctx context.Context, child *Agent, label string, depth int, task string) (string, error) {
	ctx, span := trace.FromContext(ctx).Start(ctx, trace.KindSubagent, label)
	span.Set("wombat.subagent.depth", depth)
	var spanErr error
	defer func() { span.End(spanErr) }()

	out, err := runChildInner(ctx, child, label, depth, task)
	spanErr = err
	return out, err
}

func runChildInner(ctx context.Context, child *Agent, label string, depth int, task string) (string, error) {
	Emit(ctx, SubagentStart{Name: label, Task: task, Depth: depth})
	started := time.Now()

	out, err := func() (string, error) {
		run := child.Start(ctx, Ask(task))
		defer run.Close()

		for run.Next() {
			Emit(ctx, SubagentEvent{Name: label, Depth: depth, Inner: run.Event()})
		}
		if err := run.Err(); err != nil {
			// Wrapped, not swallowed: the parent's model reads this string and
			// decides whether to retry with a different decomposition, so it
			// has to say which child failed and how.
			return "", fmt.Errorf("sub-agent %q failed: %w", label, err)
		}

		switch o := run.Outcome().(type) {
		case Answer:
			return o.Text, nil

		case Submitted:
			// A child with a terminal tool answers in structured JSON. Handing
			// the payload back verbatim keeps it machine-readable for a parent
			// that parses it, and readable enough for one that does not.
			return string(o.Payload), nil

		case Paused:
			return "", fmt.Errorf("%w: sub-agent %q asked %q, but a sub-agent has no user. "+
				"Answer it yourself by re-delegating with that information written into the task, "+
				"or ask the user before delegating",
				ErrSubagentPaused, label, o.Schema.Summary())

		default:
			return "", fmt.Errorf("sub-agent %q ended in an unknown state (%T)", label, o)
		}
	}()

	ev := SubagentEnd{Name: label, OK: err == nil, Millis: millis(time.Since(started))}
	if err != nil {
		ev.Error = err.Error()
	}
	Emit(ctx, ev)
	return out, err
}

// tooDeep phrases the depth cap as an instruction rather than a failure.
//
// The model is the one that can act on it: it asked for a sub-agent, it cannot
// have one, and the useful reply is what to do instead. This works because
// [governor.Budget.EnterSubagent] refuses WITHOUT aborting the run — depth is a
// shape, not a resource, and a parent told "not deeper" can still do the job
// itself. Every other governor cap ends the run, and rightly so.
func tooDeep() error {
	return fmt.Errorf("%w: you are already at the maximum nesting level, so this task cannot be "+
		"delegated further. Do it yourself with your own tools, or break it into steps you can run "+
		"here", ErrDelegateTooDeep)
}

// preview shortens a task for a label, cutting on a rune boundary so the
// result is still valid UTF-8 in the transcript.
func preview(s string, limit int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= limit {
		return s
	}
	cut := limit
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}
