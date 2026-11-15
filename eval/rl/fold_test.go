package rl

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/permission"
)

func denialErr() error {
	return fmt.Errorf("%w: the write tool was refused: read-only policy", permission.ErrDenied)
}

func TestFold(t *testing.T) {
	tests := []struct {
		name   string
		events []core.Event
		want   []Step
	}{
		{
			name: "one turn with a model call",
			events: []core.Event{
				core.IterStart{N: 1, Max: 30},
				core.LLMDone{Usage: llm.Usage{InputTokens: 100, OutputTokens: 20}, Millis: 250},
			},
			want: []Step{{
				Iteration: 1, Millis: 250,
				Usage: llm.Usage{InputTokens: 100, OutputTokens: 20},
			}},
		},
		{
			name: "tools land in call order",
			events: []core.Event{
				core.IterStart{N: 1},
				core.LLMDone{Millis: 10},
				core.ToolStart{UseID: "a", Name: "read"},
				core.ToolDone{UseID: "a", Name: "read", OK: true},
				core.ToolStart{UseID: "b", Name: "write"},
				core.ToolDone{UseID: "b", Name: "write", OK: true},
			},
			want: []Step{{Iteration: 1, Millis: 10, Tools: []string{"read", "write"}}},
		},
		{
			name: "a failed call is Failed but not Denied",
			events: []core.Event{
				core.IterStart{N: 1},
				core.ToolStart{UseID: "a", Name: "bash"},
				core.ToolDone{UseID: "a", Name: "bash", OK: false,
					Error: "exit 1", Err: errors.New("exit 1")},
			},
			want: []Step{{Iteration: 1, Tools: []string{"bash"}, Failed: []string{"bash"}}},
		},
		{
			name: "a refusal is matched on the sentinel and is both Failed and Denied",
			events: []core.Event{
				core.IterStart{N: 1},
				core.ToolStart{UseID: "a", Name: "write"},
				core.ToolDone{UseID: "a", Name: "write", OK: false,
					Error: denialErr().Error(), Err: denialErr()},
			},
			want: []Step{{
				Iteration: 1,
				Tools:     []string{"write"},
				Failed:    []string{"write"},
				Denied:    []string{"write"},
			}},
		},
		{
			name: "an error that only READS like a refusal is not one",
			events: []core.Event{
				core.IterStart{N: 1},
				core.ToolStart{UseID: "a", Name: "bash"},
				// The tool's own output happens to contain the sentinel's text.
				core.ToolDone{UseID: "a", Name: "bash", OK: false,
					Error: "permission: denied", Err: errors.New("permission: denied")},
			},
			want: []Step{{Iteration: 1, Tools: []string{"bash"}, Failed: []string{"bash"}}},
		},
		{
			name: "a stream with no error values falls back to permission.Decided",
			events: []core.Event{
				core.IterStart{N: 1},
				core.ToolStart{UseID: "a", Name: "write"},
				permission.Decided{UseID: "a", Tool: "write", Allowed: false, Source: "policy"},
				// Err nil, as it is for events replayed from JSON.
				core.ToolDone{UseID: "a", Name: "write", OK: false, Error: "refused"},
			},
			want: []Step{{
				Iteration: 1,
				Tools:     []string{"write"},
				Failed:    []string{"write"},
				Denied:    []string{"write"},
			}},
		},
		{
			name: "an allowed Decided does not mark anything denied",
			events: []core.Event{
				core.IterStart{N: 1},
				core.ToolStart{UseID: "a", Name: "read"},
				permission.Decided{UseID: "a", Tool: "read", Allowed: true},
				core.ToolDone{UseID: "a", Name: "read", OK: false, Error: "disk error"},
			},
			want: []Step{{Iteration: 1, Tools: []string{"read"}, Failed: []string{"read"}}},
		},
		{
			name: "iterations open new steps",
			events: []core.Event{
				core.IterStart{N: 1},
				core.LLMDone{Millis: 5, Usage: llm.Usage{InputTokens: 1}},
				core.ToolStart{UseID: "a", Name: "read"},
				core.ToolDone{UseID: "a", Name: "read", OK: true},
				core.IterStart{N: 2},
				core.LLMDone{Millis: 7, Usage: llm.Usage{InputTokens: 2}},
			},
			want: []Step{
				{Iteration: 1, Millis: 5, Tools: []string{"read"}, Usage: llm.Usage{InputTokens: 1}},
				{Iteration: 2, Millis: 7, Usage: llm.Usage{InputTokens: 2}},
			},
		},
		{
			name: "a sub-agent folds into the CURRENT step, prefixed",
			events: []core.Event{
				core.IterStart{N: 1},
				core.LLMDone{Millis: 30, Usage: llm.Usage{InputTokens: 10}},
				core.ToolStart{UseID: "d", Name: "delegate"},
				core.SubagentStart{Name: "researcher", Depth: 1},
				// The child's own loop must not become the parent's turn.
				core.SubagentEvent{Name: "researcher", Inner: core.IterStart{N: 1}},
				core.SubagentEvent{Name: "researcher", Inner: core.LLMDone{
					Millis: 900, Usage: llm.Usage{InputTokens: 500, OutputTokens: 40}}},
				core.SubagentEvent{Name: "researcher", Inner: core.ToolStart{UseID: "s1", Name: "grep"}},
				core.SubagentEvent{Name: "researcher", Inner: core.ToolDone{UseID: "s1", Name: "grep", OK: true}},
				core.SubagentEvent{Name: "researcher", Inner: core.IterStart{N: 2}},
				core.SubagentEvent{Name: "researcher", Inner: core.ToolStart{UseID: "s2", Name: "write"}},
				core.SubagentEvent{Name: "researcher", Inner: core.ToolDone{
					UseID: "s2", Name: "write", OK: false, Err: denialErr()}},
				core.SubagentEnd{Name: "researcher", OK: true},
				core.ToolDone{UseID: "d", Name: "delegate", OK: true},
			},
			want: []Step{{
				Iteration: 1,
				// The child's latency is deliberately NOT added: the parent's
				// delegate call already spans it.
				Millis: 30,
				Usage:  llm.Usage{InputTokens: 510, OutputTokens: 40},
				Tools:  []string{"delegate", "researcher/grep", "researcher/write"},
				Failed: []string{"researcher/write"},
				Denied: []string{"researcher/write"},
			}},
		},
		{
			name: "nesting composes the prefix",
			events: []core.Event{
				core.IterStart{N: 1},
				core.SubagentEvent{Name: "a", Inner: core.SubagentEvent{
					Name: "b", Inner: core.ToolStart{UseID: "x", Name: "ls"}}},
			},
			want: []Step{{Iteration: 1, Tools: []string{"a/b/ls"}}},
		},
		{
			name: "work before any IterStart still lands somewhere",
			events: []core.Event{
				core.ToolStart{UseID: "a", Name: "read"},
			},
			want: []Step{{Iteration: 1, Tools: []string{"read"}}},
		},
		{
			name:   "events this fold does not care about change nothing",
			events: []core.Event{core.TextDelta{Text: "hi"}, core.LLMStart{Tools: 3}},
			want:   nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var f folder
			for _, ev := range tc.events {
				f.fold(ev, "")
			}
			if !reflect.DeepEqual(f.steps, tc.want) {
				t.Fatalf("steps =\n  %+v\nwant\n  %+v", f.steps, tc.want)
			}
		})
	}
}
