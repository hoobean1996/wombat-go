package shape

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// FillerTool is the tool a shape run calls to make the transcript grow.
const FillerTool = "filler"

// Recorder is an [llm.Client] that answers from a script and keeps a [Shot] of
// every request the harness assembled.
//
// Concurrency-safe because a delegating probe fans children across goroutines
// that share one client.
type Recorder struct {
	mu    sync.Mutex
	turn  int
	shots []Shot
	reply func(turn int) llm.Response
}

// NewRecorder builds a recorder whose replies come from reply, called with the
// 1-based turn number.
func NewRecorder(reply func(turn int) llm.Response) *Recorder {
	return &Recorder{reply: reply}
}

// Complete implements llm.Client.
func (r *Recorder) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	r.mu.Lock()
	r.turn++
	turn := r.turn
	r.shots = append(r.shots, Of(turn, req))
	r.mu.Unlock()
	return r.reply(turn), nil
}

// Shots returns a snapshot of what was recorded.
func (r *Recorder) Shots() []Shot {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Shot(nil), r.shots...)
}

// ToolLoop is the standard script: call [FillerTool] every turn until the last
// one, then answer.
//
// A tool call per turn rather than plain text, because a strategy only has
// something to trim once the transcript is carrying tool_use/tool_result
// PAIRS — and pairs are also the thing a position-based cut is not allowed to
// split, so this is the shape that exercises [core.SafeCut] rather than
// stepping around it.
func ToolLoop(lastTurn int) func(int) llm.Response {
	return func(turn int) llm.Response {
		if turn >= lastTurn {
			return llm.Response{
				StopReason: llm.StopEndTurn,
				Content:    []llm.ContentBlock{llm.Text{Text: "done"}},
				Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
			}
		}
		return llm.Response{
			StopReason: llm.StopToolUse,
			Content: []llm.ContentBlock{
				llm.Text{Text: fmt.Sprintf("step %d", turn)},
				llm.ToolUse{
					ID:    llm.ToolUseID(fmt.Sprintf("call-%03d", turn)),
					Name:  FillerTool,
					Input: json.RawMessage(fmt.Sprintf(`{"n":%d}`, turn)),
				},
			},
			Usage: llm.Usage{InputTokens: 10, OutputTokens: 5},
		}
	}
}

// Filler returns a tool whose observation is payload bytes of stable text.
//
// Stable, not random: two runs of the same configuration must produce the same
// [Shot], or a golden file is noise and a prefix hash means nothing.
func Filler(payload int) tool.Def {
	return tool.Def{
		Name:        FillerTool,
		Description: "Returns a fixed block of text. Exists to make a transcript grow.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer"}}}`),
		Caps:        tool.CapReadOnly,
		Fn: func(_ context.Context, in json.RawMessage) (string, error) {
			var arg struct {
				N int `json:"n"`
			}
			_ = json.Unmarshal(in, &arg)
			return fmt.Sprintf("observation %d: %s", arg.N, strings.Repeat("x", payload)), nil
		},
	}
}

// Config drives one shape run.
type Config struct {
	// Turns is how many model calls to make. The last one answers.
	Turns int

	// Payload is how many bytes each tool observation carries.
	Payload int

	// Prompt is the opening user turn.
	Prompt string

	// Options are applied on top of the defaults, so a caller supplies only
	// the mechanism under test:
	//
	//	shape.Run(ctx, shape.Config{Turns: 12, Options: []core.Option{
	//	    core.WithStrategy(core.SlidingWindowAt(20, 8)),
	//	}})
	Options []core.Option

	// Tools replaces the default single filler tool.
	Tools []tool.Def

	// Seed is a prior transcript to resume from, modelling a multi-turn
	// session rather than a fresh question.
	//
	// It is not a convenience. A position-based window can only cut
	// immediately before a user turn that carries no tool_result, and in a
	// single ReAct run there is exactly one such turn — the opening
	// instruction at index 0, which [core.SafeCut] deliberately refuses to
	// cut at. So a pure tool loop has NO legal cut point and every window
	// degrades to [core.Flat]. Windowing is a multi-turn mechanism, and
	// measuring it needs a multi-turn transcript.
	Seed []llm.Message
}

// Conversation builds a plain user/assistant transcript of rounds exchanges.
//
// Every user turn in it is a legal cut point, which is what gives a window
// something to do.
func Conversation(rounds, payload int) []llm.Message {
	out := make([]llm.Message, 0, rounds*2)
	for i := 1; i <= rounds; i++ {
		out = append(out,
			llm.UserText(fmt.Sprintf("earlier request %d: %s", i, strings.Repeat("u", payload))),
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.Text{Text: fmt.Sprintf("earlier reply %d: %s", i, strings.Repeat("a", payload))},
			}},
		)
	}
	return out
}

// Run drives an agent through cfg.Turns model calls and returns one [Shot] per
// call.
func Run(ctx context.Context, cfg Config) ([]Shot, error) {
	if cfg.Turns < 1 {
		return nil, fmt.Errorf("shape: Turns must be positive, got %d", cfg.Turns)
	}
	if cfg.Payload <= 0 {
		cfg.Payload = 256
	}
	if cfg.Prompt == "" {
		cfg.Prompt = "begin"
	}
	tools := cfg.Tools
	if tools == nil {
		tools = []tool.Def{Filler(cfg.Payload)}
	}

	rec := NewRecorder(ToolLoop(cfg.Turns))

	// The caller's options come last so a mechanism under test overrides the
	// defaults rather than fighting them. MaxIters is generous on purpose: a
	// run that stopped early would silently shorten the transcript and make a
	// window look stabler than it is.
	// Silent by default. This is a measuring instrument: its diagnostics are
	// the caller's tool logs, not its own, and a suite that prints a line per
	// filler call buries the one number the run exists to produce.
	opts := append([]core.Option{
		core.WithLogger(slog.New(slog.NewTextHandler(io.Discard, nil))),
		core.WithClient(rec),
		core.WithTools(tools...),
		core.WithModel("shape-recorder"),
		core.WithMaxIters(cfg.Turns + 2),
	}, cfg.Options...)

	a, err := core.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("shape: building the agent: %w", err)
	}
	in := core.Ask(cfg.Prompt)
	if len(cfg.Seed) > 0 {
		in = core.Continue(append(append([]llm.Message(nil), cfg.Seed...), llm.UserText(cfg.Prompt)))
	}
	if _, err := a.Run(ctx, in); err != nil {
		return nil, fmt.Errorf("shape: run: %w", err)
	}

	shots := rec.Shots()
	if len(shots) != cfg.Turns {
		return shots, fmt.Errorf("shape: wanted %d model calls, the loop made %d", cfg.Turns, len(shots))
	}
	return shots, nil
}

// RunSession drives a MULTI-TURN session: turns sequential runs, each
// continuing the transcript the last one left behind.
//
// This is the shape a position-based window is actually for, and [Run] cannot
// produce it. Inside one run the only legal cut point is the opening
// instruction, so the anchor can move at most as many times as the seed has
// plain user turns — which is why a single-run measurement makes every window
// look stable. A session adds a fresh user turn per turn, so cut points arrive
// as fast as the transcript grows, and that is when a window that recomputes
// its cut every turn starts missing the cache every turn.
//
// Each turn makes callsPerTurn tool calls before answering.
func RunSession(ctx context.Context, cfg Config, turns, callsPerTurn int) ([]Shot, error) {
	if turns < 1 {
		return nil, fmt.Errorf("shape: turns must be positive, got %d", turns)
	}
	if callsPerTurn < 0 {
		callsPerTurn = 0
	}
	if cfg.Payload <= 0 {
		cfg.Payload = 256
	}
	tools := cfg.Tools
	if tools == nil {
		tools = []tool.Def{Filler(cfg.Payload)}
	}

	// One recorder across the whole session: its turn counter is the session's
	// model-call ordinal, so the Shots line up end to end and a prefix
	// comparison spans the turn boundary — which is exactly where a resumed
	// session either hits the cache or does not.
	perTurn := callsPerTurn + 1
	rec := NewRecorder(func(call int) llm.Response {
		if (call-1)%perTurn == callsPerTurn {
			return llm.Response{
				StopReason: llm.StopEndTurn,
				Content:    []llm.ContentBlock{llm.Text{Text: "ok"}},
				Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
			}
		}
		return llm.Response{
			StopReason: llm.StopToolUse,
			Content: []llm.ContentBlock{
				llm.ToolUse{
					ID:    llm.ToolUseID(fmt.Sprintf("call-%04d", call)),
					Name:  FillerTool,
					Input: json.RawMessage(fmt.Sprintf(`{"n":%d}`, call)),
				},
			},
			Usage: llm.Usage{InputTokens: 10, OutputTokens: 5},
		}
	})

	opts := append([]core.Option{
		core.WithLogger(slog.New(slog.NewTextHandler(io.Discard, nil))),
		core.WithClient(rec),
		core.WithTools(tools...),
		core.WithModel("shape-recorder"),
		core.WithMaxIters(perTurn + 2),
	}, cfg.Options...)

	a, err := core.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("shape: building the agent: %w", err)
	}

	prior := append([]llm.Message(nil), cfg.Seed...)
	for t := 1; t <= turns; t++ {
		instruction := fmt.Sprintf("session turn %d: %s", t, strings.Repeat("q", cfg.Payload/4))
		var in core.Input
		if len(prior) == 0 {
			in = core.Ask(instruction)
		} else {
			in = core.Then(prior, instruction)
		}

		run := a.Start(ctx, in)
		for run.Next() { //nolint:revive // the shots are collected by the recorder
		}
		if err := run.Err(); err != nil {
			run.Close()
			return rec.Shots(), fmt.Errorf("shape: session turn %d: %w", t, err)
		}
		prior = run.Messages()
		run.Close()
	}
	return rec.Shots(), nil
}
