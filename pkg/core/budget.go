package core

import (
	"context"

	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// budgetAttempts feeds the run budget's loop detection with every call the
// model asks for, and refuses to start one after the budget has tripped.
//
// It sits OUTSIDE the caller's middleware — the permission gate — because a
// call the policy refused is still the model asking, and asking for the same
// refused call again and again is a stuck agent. The key is what makes that
// detection possible: same tool, same arguments.
//
// It lives here rather than in package tool because it is policy: the tool
// package should not know what a budget is.
func budgetAttempts(next tool.Handler) tool.Handler {
	return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
		if err := ctx.Err(); err != nil {
			return "", context.Cause(ctx)
		}
		governor.FromContext(ctx).Attempt(d.Name + "\x00" + string(use.Input))
		if err := ctx.Err(); err != nil {
			return "", context.Cause(ctx)
		}
		return next(ctx, d, use)
	}
}

// budgetToolCalls counts a call against the run's ToolCalls limit. It sits
// INSIDE every gate, the ones a parent passed down included, so what it counts
// is calls that are about to run: one a policy refused never used the budget.
func budgetToolCalls(next tool.Handler) tool.Handler {
	return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
		governor.FromContext(ctx).CountToolCall()
		if err := ctx.Err(); err != nil {
			return "", context.Cause(ctx)
		}
		return next(ctx, d, use)
	}
}

// TrackCost prices each completed call, charges it to the run budget and
// emits a [Spend] event.
//
// This is the only place that sees every response, which is why it is also
// where cost enforcement lives: charging the budget may cancel the context,
// and every blocking operation downstream unwinds on its own from there.
//
// Install it as the innermost cost-aware layer of an [llm.Client] chain:
//
//	client := llm.Chain(leaf, core.TrackCost(llm.DefaultPricing), llm.WithRetry(p))
//
// Inside retry, so a retried call is charged once per attempt — attempts cost
// real money whether or not they succeed.
func TrackCost(p llm.Pricing) llm.Middleware {
	if p == nil {
		p = llm.DefaultPricing
	}
	return func(next llm.Client) llm.Client {
		// Calls that arrive with no budget on their context — a client used
		// on its own rather than by a run — are tallied on one budget of this
		// client's, so the Spend events they produce add up instead of each
		// reporting a single call. A run never gets here: Agent.Start gives
		// every run a budget of its own, which is what keeps two runs' spend
		// apart. A caller who wants its own scope passes governor.WithBudget
		// or governor.Ensure.
		own := governor.FromContext(governor.Ensure(context.Background()))
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			resp, err := next.Complete(ctx, req)
			if err != nil {
				return resp, err
			}

			model := resp.Model
			if model == "" {
				model = req.Model
			}

			b, ok := governor.Lookup(ctx)
			if !ok {
				b = own
			}
			b.AddCall(p.CostUSD(model, resp.Usage), governor.Tokens{
				In:         resp.Usage.InputTokens,
				Out:        resp.Usage.OutputTokens,
				CacheWrite: resp.Usage.CacheWriteTokens,
				CacheRead:  resp.Usage.CacheReadTokens,
			})

			pr := b.Progress()
			Emit(ctx, Spend{
				CostUSD:      pr.CostUSD,
				Calls:        pr.Calls,
				InputTokens:  pr.Tokens.In,
				OutputTokens: pr.Tokens.Out,
				CacheWrite:   pr.Tokens.CacheWrite,
				CacheRead:    pr.Tokens.CacheRead,
				ElapsedSec:   pr.Elapsed.Seconds(),
			})
			return resp, nil
		})
	}
}
