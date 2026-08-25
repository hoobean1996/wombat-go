package eval

import (
	"encoding/json"
	"time"

	"github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/permission"
)

// Obs is one sample, folded out of the event stream.
//
// From the stream and not from the transcript, for the reason rl/rollout.go
// gives about its own Steps: a tool_result does not say whether the permission
// gate refused the call or the tool itself failed, and it does not say how many
// tools were on offer when the model chose. Both are mechanism facts and both
// exist only while the run is happening.
type Obs struct {
	Arm    string
	Sample int

	// Iters is the last iteration the loop opened.
	Iters int

	// Offered is how many tools were visible at each model call, in order.
	// This is the direct observable for progressive disclosure: a gated run
	// reads 12, 12, 14 where an ungated one reads 14, 14, 14.
	Offered []int

	LLMCalls int
	Calls    []Call
	Perms    []Perm
	Subs     []Sub

	Usage Usage
	Cost  float64
	Wall  time.Duration

	Answer  string
	Outcome string
	Err     string

	Messages []llm.Message
}

// Usage is the run's token total.
type Usage struct{ In, Out, CacheRead, CacheWrite int }

// Call is one dispatched tool call.
type Call struct {
	Turn  int
	Name  string
	Input json.RawMessage
	OK    bool
	Out   string
	Err   string
	Ms    int64
}

// Perm is one permission verdict. Source distinguishes a rule that decided on
// its own from a standing grant and from nobody being there to ask, which is
// what makes "the allowlist actually fired" a checkable statement.
type Perm struct {
	Turn    int
	Tool    string
	Allowed bool
	Source  string
	Reason  string
}

// Sub is one delegated child.
type Sub struct {
	Name   string
	Depth  int
	Events int
	OK     bool
	Err    string
}

// FirstIndex returns the position of the first call to name, or -1.
func (o Obs) FirstIndex(name string) int {
	for i, c := range o.Calls {
		if c.Name == name {
			return i
		}
	}
	return -1
}

// CallsTo returns every call to name.
func (o Obs) CallsTo(names ...string) []Call {
	var out []Call
	for _, c := range o.Calls {
		for _, n := range names {
			if c.Name == n {
				out = append(out, c)
				break
			}
		}
	}
	return out
}

// ArgString reads a top-level string argument out of a call's input.
func (c Call) ArgString(key string) string {
	if len(c.Input) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(c.Input, &m) != nil {
		return ""
	}
	raw, ok := m[key]
	if !ok {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return ""
	}
	return s
}

// observe drains a run and folds it.
//
// It drains to completion in every case, including a failing one: an
// abandoned Run leaks its goroutine, and a probe that measured only the
// successful samples would be measuring the mechanism where it was needed
// least.
func observe(run *wombat.Run) Obs {
	var o Obs
	turn := 0
	byUse := map[llm.ToolUseID]int{}
	subs := map[string]int{}

	for run.Next() {
		switch e := run.Event().(type) {
		case wombat.IterStart:
			turn = e.N
			if e.N > o.Iters {
				o.Iters = e.N
			}

		case wombat.LLMStart:
			o.Offered = append(o.Offered, e.Tools)

		case wombat.LLMDone:
			o.LLMCalls++
			o.Usage.In += e.Usage.InputTokens
			o.Usage.Out += e.Usage.OutputTokens
			o.Usage.CacheRead += e.Usage.CacheReadTokens
			o.Usage.CacheWrite += e.Usage.CacheWriteTokens

		case wombat.Spend:
			// The running tally; the last one seen is the total. Read from the
			// event rather than summed here so a probe reports the same number
			// the budget enforced.
			o.Cost = e.CostUSD

		case wombat.ToolStart:
			byUse[e.UseID] = len(o.Calls)
			o.Calls = append(o.Calls, Call{Turn: turn, Name: e.Name, Input: e.Input})

		case wombat.ToolDone:
			i, ok := byUse[e.UseID]
			if !ok {
				// A done with no start: record it rather than dropping it, so
				// a hole in the stream shows up as a call with no arguments
				// instead of as nothing.
				o.Calls = append(o.Calls, Call{Turn: turn, Name: e.Name})
				i = len(o.Calls) - 1
			}
			o.Calls[i].OK = e.OK
			o.Calls[i].Out = e.Output
			o.Calls[i].Err = e.Error
			o.Calls[i].Ms = e.Millis

		case permission.Decided:
			o.Perms = append(o.Perms, Perm{
				Turn: turn, Tool: e.Tool, Allowed: e.Allowed,
				Source: e.Source, Reason: e.Reason,
			})

		case wombat.SubagentStart:
			subs[e.Name] = len(o.Subs)
			o.Subs = append(o.Subs, Sub{Name: e.Name, Depth: e.Depth})

		case wombat.SubagentEnd:
			if i, ok := subs[e.Name]; ok {
				o.Subs[i].OK = e.OK
				o.Subs[i].Err = e.Error
			}

		case wombat.SubagentEvent:
			// Counted, not stored. The point of the wrapper is that the
			// child's stream REACHES the parent; how many of its deltas did is
			// the evidence, and keeping them would make an Obs the size of a
			// transcript.
			if i, ok := subs[e.Name]; ok {
				o.Subs[i].Events++
			}
		}
	}

	o.Messages = run.Messages()
	if err := run.Err(); err != nil {
		o.Err = err.Error()
	}
	switch out := run.Outcome().(type) {
	case wombat.Answer:
		o.Outcome, o.Answer = "answer", out.Text
	case wombat.Paused:
		o.Outcome = "paused"
	case wombat.Submitted:
		o.Outcome = "submitted"
	default:
		o.Outcome = "none"
	}
	return o
}
