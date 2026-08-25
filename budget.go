package wombat

import (
	"context"

	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// budgetToolCalls counts dispatched calls against the run budget and refuses
// to start one after the budget has already tripped.
//
// It lives here rather than in package tool because it is policy: the tool
// package should not know what a budget is.
func budgetToolCalls(next tool.Handler) tool.Handler {
	return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
		if err := ctx.Err(); err != nil {
			return "", context.Cause(ctx)
		}
		// The key is what makes loop detection possible: same tool, same
		// arguments, again and again is a stuck agent, and the governor is the
		// backstop for when the advisory dedup annotation fails to shake it
		// loose.
		governor.FromContext(ctx).AddToolCall(d.Name + "\x00" + string(use.Input))
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
//	client := llm.Chain(leaf, wombat.TrackCost(llm.DefaultPricing), llm.WithRetry(p))
//
// Inside retry, so a retried call is charged once per attempt — attempts cost
// real money whether or not they succeed.
func TrackCost(p llm.Pricing) llm.Middleware {
	if p == nil {
		p = llm.DefaultPricing
	}
	return func(next llm.Client) llm.Client {
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			resp, err := next.Complete(ctx, req)
			if err != nil {
				return resp, err
			}

			model := resp.Model
			if model == "" {
				model = req.Model
			}

			b := governor.FromContext(ctx)
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
