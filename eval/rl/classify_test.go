package rl

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/permission"
)

func TestClassify(t *testing.T) {
	deniedStep := []Step{{Iteration: 1, Tools: []string{"w"}, Failed: []string{"w"}, Denied: []string{"w"}}}

	tests := []struct {
		name      string
		ep        Episode
		threshold float64
		want      FailureKind
	}{
		{"clean and fully scored", Episode{Reward: 1}, 1, Success},
		{"clean and exactly at the threshold", Episode{Reward: 0.7}, 0.7, Success},
		{"clean and short of the threshold", Episode{Reward: 0.9}, 1, VerifierFailed},
		{"clean and scored nothing", Episode{}, 1, VerifierFailed},

		{"the loop's iteration cap", Episode{Err: fmt.Errorf("%w (30)", core.ErrMaxIterations)}, 1, MaxIterations},
		{"the governor's step cap is the same failure", Episode{Err: governor.ErrStepLimit}, 1, MaxIterations},
		{"cost", Episode{Err: governor.ErrBudgetExhausted}, 1, BudgetExceeded},
		{"wall clock", Episode{Err: governor.ErrWallClock}, 1, WallClock},
		{"a deadline is a wall clock", Episode{Err: context.DeadlineExceeded}, 1, WallClock},
		{"a repeated call", Episode{Err: governor.ErrToolLoop}, 1, ToolLoop},
		{"the tool-call cap", Episode{Err: governor.ErrToolCallLimit}, 1, ToolLoop},
		{"a truncated reply", Episode{Err: core.ErrMaxTokens}, 1, MaxTokens},
		{"a refusal", Episode{Err: &core.RefusalError{Reason: "no"}}, 1, Refused},
		{"a contained panic", Episode{Err: fmt.Errorf("%w: boom", core.ErrPanic)}, 1, Panicked},
		{"an overflowing transcript", Episode{Err: &llm.APIError{Class: llm.ErrContextWindow}}, 1, ContextWindow},
		{"a provider outage", Episode{Err: &llm.APIError{Class: llm.ErrOverloaded, Status: 529}}, 1, ProviderError},
		{"bad credentials", Episode{Err: &llm.APIError{Class: llm.ErrAuth, Status: 401}}, 1, ProviderError},
		{"a refused run", Episode{Err: fmt.Errorf("%w: nope", permission.ErrDenied)}, 1, Denied},
		{"cancellation", Episode{Err: context.Canceled}, 1, Cancelled},
		{"something nobody named", Episode{Err: errors.New("disk on fire")}, 1, Other},
		{
			"an unrecognised error after a run of refusals is a permission story",
			Episode{Err: errors.New("gave up"), Steps: deniedStep}, 1, Denied,
		},
		{
			"a named cause still wins over the refusals underneath it",
			Episode{Err: governor.ErrBudgetExhausted, Steps: deniedStep}, 1, BudgetExceeded,
		},
		{
			"the reward is irrelevant once there is an error",
			Episode{Err: core.ErrMaxTokens, Reward: 1}, 1, MaxTokens,
		},
		{
			"wrapping several layers deep still matches",
			Episode{Err: fmt.Errorf("driving sample 3: %w",
				fmt.Errorf("run: %w", core.ErrMaxIterations))}, 1, MaxIterations,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ep := tc.ep
			if got := classify(&ep, tc.threshold); got != tc.want {
				t.Fatalf("classify = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestClassifyCoversEveryKind fails when a kind is added to the taxonomy and
// nothing in the table above can produce it. A FailureKind nobody can reach is
// a row of the histogram that will read zero forever.
func TestClassifyCoversEveryKind(t *testing.T) {
	reachable := map[FailureKind]error{
		Success:        nil,
		VerifierFailed: nil,
		MaxIterations:  core.ErrMaxIterations,
		BudgetExceeded: governor.ErrBudgetExhausted,
		WallClock:      governor.ErrWallClock,
		ToolLoop:       governor.ErrToolLoop,
		ContextWindow:  llm.ErrContextWindow,
		Refused:        core.ErrRefused,
		MaxTokens:      core.ErrMaxTokens,
		Denied:         permission.ErrDenied,
		Panicked:       core.ErrPanic,
		Cancelled:      context.Canceled,
		ProviderError:  llm.ErrServer,
		Other:          errors.New("unnamed"),
	}

	for _, k := range Kinds() {
		err, ok := reachable[k]
		if !ok {
			t.Errorf("FailureKind %q has no error that produces it", k)
			continue
		}
		ep := Episode{Err: err}
		switch k {
		case Success:
			ep.Reward = 1
		case VerifierFailed:
			ep.Reward = 0
		}
		if got := classify(&ep, 1); got != k {
			t.Errorf("classify for %q produced %q", k, got)
		}
	}
}
