package tape

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// LLM returns the middleware that records and replays model calls.
//
// This is the half that pays for itself. A model call costs money, takes
// seconds, and returns something different every time — it is simultaneously
// the expensive part of a run and the only part that is not reproducible, so
// both jobs the package exists for are really about this middleware. The tool
// half (see [Tape.Tools]) is a correctness measure rather than an economy: it
// keeps a replayed run from re-running side effects.
//
// Chain it OUTSIDE retry, so a hit skips the retry layer entirely and a live
// call is recorded once per logical call rather than once per attempt:
//
//	llm.Chain(client, llm.WithRetry(p), tp.LLM())
func (t *Tape) LLM() llm.Middleware {
	return func(next llm.Client) llm.Client {
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			canon, key, err := canonLLM(req)
			if err != nil {
				return llm.Response{}, fmt.Errorf("tape: canonicalise llm request: %w", err)
			}
			k := recKey{kind: kindLLM, key: key}

			if t.mode == Record {
				t.countMiss()
			} else if raw, ok := t.take(k); ok {
				// Cancellation is honoured even on a hit. A replayed run reads
				// from memory and would otherwise sail through a whole tape
				// after the user pressed Ctrl-C, which is both surprising and
				// a good way to make a cancellation test pass for free.
				if err := ctx.Err(); err != nil {
					return llm.Response{}, err
				}
				resp, streamed, err := decodeLLMResponse(raw)
				if err != nil {
					return llm.Response{}, fmt.Errorf("tape: decode recorded llm response %s: %w", key, err)
				}
				if streamed {
					replayDeltas(req.OnDelta, resp.Content)
				}
				return resp, nil
			} else if t.mode == Replay {
				return llm.Response{}, fmt.Errorf("%w for llm request %s (model %q, %d messages)",
					ErrTapeMiss, key, req.Model, len(req.Messages))
			}

			// Whether the live call streamed is a property of the client and
			// its configuration, not of the request, so it cannot be derived
			// at replay time and has to be observed here and written down.
			// The caller's sink is forwarded unchanged; this only counts.
			var streamed atomic.Bool
			if sink := req.OnDelta; sink != nil {
				req.OnDelta = func(d llm.Delta) {
					streamed.Store(true)
					sink(d)
				}
			}

			resp, err := next.Complete(ctx, req)
			if err != nil {
				// Failures are not recorded. An error is usually transient —
				// a 429, a dropped connection — and taping it would make the
				// next run replay the outage instead of getting past it. The
				// consequence, which a differential run has to know, is that
				// the retries visible in a recording's logs are invisible in
				// its replay.
				return resp, err
			}

			enc, err := encodeLLMResponse(resp, streamed.Load())
			if err != nil {
				t.mu.Lock()
				t.setWriteErr(fmt.Errorf("tape: encode llm response: %w", err))
				t.mu.Unlock()
				return resp, nil
			}
			t.record(k, canon, enc)
			return resp, nil
		})
	}
}

// Tools returns the middleware that records and replays tool dispatch, keyed
// on the tool name and its input bytes.
//
// The reason this exists is not cost. Tool calls are cheap and mostly
// deterministic; what they are is EFFECTFUL. Replaying an LLM tape without
// this would resume a crashed run by re-sending the same email, re-applying
// the same patch and re-charging the same card, because the model's recorded
// reply asks for all of it again. Taping the tool side turns a resume into a
// read of what already happened.
//
// Install it with wombat.WithToolMiddleware, which places it outside retry,
// the circuit breaker and truncation — so what is recorded is the logical
// verdict the model was shown, not an intermediate attempt.
func (t *Tape) Tools() tool.Middleware {
	return func(next tool.Handler) tool.Handler {
		return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
			// d.Name, not use.Name: the dispatcher has already resolved the
			// model's chosen name against the set, and the resolved definition
			// is what will actually run.
			canon, key, cerr := canonTool(d.Name, use.Input)
			if cerr != nil {
				return "", fmt.Errorf("tape: canonicalise tool request: %w", cerr)
			}
			k := recKey{kind: kindTool, key: key}

			if t.mode == Record {
				t.countMiss()
			} else if raw, ok := t.take(k); ok {
				if err := ctx.Err(); err != nil {
					return "", err
				}
				return decodeToolResponse(raw, key)
			} else if t.mode == Replay {
				return "", fmt.Errorf("%w for tool %q %s", ErrTapeMiss, d.Name, key)
			}

			out, err := next(ctx, d, use)

			// Unlike the LLM half, a tool failure IS recorded. A tool that
			// returns an error is doing something normal — the message becomes
			// an is_error tool_result the model reads and reacts to — so
			// dropping it would replay a different conversation. The
			// concession is that the sentinel is lost: see the package doc.
			enc, eerr := encodeToolResponse(out, err)
			if eerr != nil {
				t.mu.Lock()
				t.setWriteErr(fmt.Errorf("tape: encode tool response: %w", eerr))
				t.mu.Unlock()
				return out, err
			}
			t.record(k, canon, enc)
			return out, err
		}
	}
}

func decodeToolResponse(raw json.RawMessage, key string) (string, error) {
	var tr toolResponse
	if err := json.Unmarshal(raw, &tr); err != nil {
		return "", fmt.Errorf("tape: decode recorded tool response %s: %w", key, err)
	}
	if tr.OK {
		return tr.Output, nil
	}
	return tr.Output, errors.New(tr.Error)
}

// replayDeltas feeds a recorded reply to the caller's streaming sink.
//
// Called only when the recording actually streamed. Without it a replayed run
// emits no wombat.TextDelta at all and a diff against a streamed recording is
// nothing but missing text, which would make the tape useless for the job it
// was built for. What it cannot restore is CHUNKING: one delta per block here
// against hundreds live. Concatenated text matches, event boundaries do not.
//
// Thinking maps to Delta.Reasoning to keep the model's scratchpad out of the
// user-visible answer, exactly as a live provider does.
func replayDeltas(sink func(llm.Delta), blocks []llm.ContentBlock) {
	if sink == nil {
		return
	}
	for _, b := range blocks {
		switch v := b.(type) {
		case llm.Text:
			if v.Text != "" {
				sink(llm.Delta{Text: v.Text})
			}
		case llm.Thinking:
			if v.Text != "" {
				sink(llm.Delta{Reasoning: v.Text})
			}
		}
	}
}
