package trace

import (
	"context"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// Attribute keys.
//
// The gen_ai.* names are OpenTelemetry's gen-ai semantic conventions, used
// verbatim so a [Sink] bridging to a real TracerProvider can pass them
// straight through instead of translating. The wombat.* names are this
// harness's own vocabulary, which no convention covers.
const (
	AttrRequestModel  = "gen_ai.request.model"
	AttrResponseModel = "gen_ai.response.model"
	AttrFinishReason  = "gen_ai.response.finish_reason"
	AttrInputTokens   = "gen_ai.usage.input_tokens"
	AttrOutputTokens  = "gen_ai.usage.output_tokens"
	AttrCacheRead     = "gen_ai.usage.cache_read_tokens"
	AttrCacheWrite    = "gen_ai.usage.cache_write_tokens"
	AttrToolName      = "gen_ai.tool.name"
	AttrToolCallID    = "gen_ai.tool.call.id"

	AttrPurpose      = "wombat.purpose"
	AttrMessageCount = "wombat.messages"
	AttrToolCount    = "wombat.tools"
	AttrForcedTool   = "wombat.forced_tool"
	AttrToolCategory = "wombat.tool.category"
	AttrToolOK       = "wombat.tool.ok"
	AttrOutputBytes  = "wombat.tool.output_bytes"
)

// LLMMiddleware records one span per model call.
//
// Install it outermost, so retries collapse into the single logical call the
// caller actually experienced:
//
//	llm.Chain(client, llm.WithRetry(p), trace.LLMMiddleware())
//
// It needs no wiring beyond that: the tracer and the parent span both come off
// the context the agent already threads through Complete.
//
// The span is named for the call's [llm.Purpose] — "planner", "executor" — not
// its model, because the purpose is what you scan a report for and the model
// is one attribute away. Recorded: requested and resolved model, purpose, stop
// reason, token counts, message and tool counts, duration. Not recorded: the
// system prompt, the messages, or anything else the request carries.
func LLMMiddleware() llm.Middleware {
	return func(next llm.Client) llm.Client {
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			name := string(req.Purpose)
			if name == "" {
				name = "llm"
			}

			ctx, span := FromContext(ctx).Start(ctx, KindLLM, name)
			span.Set(AttrPurpose, string(req.Purpose))
			span.Set(AttrRequestModel, req.Model)
			span.Set(AttrMessageCount, len(req.Messages))
			span.Set(AttrToolCount, len(req.Tools))
			if req.Choice.Mode == llm.ChoiceTool {
				span.Set(AttrForcedTool, req.Choice.Name)
			}

			resp, err := next.Complete(ctx, req)
			if err != nil {
				span.End(err)
				return resp, err
			}

			// The resolved model is worth its own attribute: a gateway or
			// llm.WithModelRouting can answer with something other than what
			// was asked for, and that is exactly the kind of surprise a trace
			// is opened to find.
			model := resp.Model
			if model == "" {
				model = req.Model
			}
			span.Set(AttrResponseModel, model)
			span.Set(AttrFinishReason, string(resp.StopReason))
			span.Set(AttrInputTokens, resp.Usage.InputTokens)
			span.Set(AttrOutputTokens, resp.Usage.OutputTokens)
			if resp.Usage.CacheReadTokens != 0 {
				span.Set(AttrCacheRead, resp.Usage.CacheReadTokens)
			}
			if resp.Usage.CacheWriteTokens != 0 {
				span.Set(AttrCacheWrite, resp.Usage.CacheWriteTokens)
			}
			span.End(nil)
			return resp, nil
		})
	}
}

// ToolMiddleware records one span per tool dispatch.
//
//	core.WithToolMiddleware(trace.ToolMiddleware())
//
// That position puts it outside retry, the circuit breaker and dedup, so a
// call that was attempted three times is still one span — the same "logical
// call" boundary [tool.WithObserver] reports on.
//
// Recorded: tool name, category, the tool_use id, whether it succeeded, how
// many bytes came back, duration, and the error text on failure. Not recorded:
// the arguments or the observation. A tool that wants more in its span can
// call trace.FromContext(ctx) itself — the context it receives is this span's.
func ToolMiddleware() tool.Middleware {
	return func(next tool.Handler) tool.Handler {
		return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
			ctx, span := FromContext(ctx).Start(ctx, KindTool, d.Name)
			span.Set(AttrToolName, d.Name)
			if use.ID != "" {
				span.Set(AttrToolCallID, string(use.ID))
			}
			if d.Category != "" {
				span.Set(AttrToolCategory, d.Category)
			}

			out, err := next(ctx, d, use)

			// ok is redundant with Error being empty, and it is here anyway:
			// filtering a report on a boolean is one comparison, and every
			// dashboard that consumes traces wants a success rate.
			span.Set(AttrToolOK, err == nil)
			span.Set(AttrOutputBytes, len(out))
			span.End(err)
			return out, err
		}
	}
}
