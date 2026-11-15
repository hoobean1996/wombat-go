package tape

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

type replies struct{ turns []llm.Response }

func (r *replies) Complete(context.Context, llm.Request) (llm.Response, error) {
	if len(r.turns) == 0 {
		return llm.Response{}, errors.New("no more replies")
	}
	out := r.turns[0]
	r.turns = r.turns[1:]
	return out, nil
}

type dead struct{}

func (dead) Complete(context.Context, llm.Request) (llm.Response, error) {
	return llm.Response{}, errors.New("the provider is down; a replay must not need it")
}

// Install records both halves, so a replay needs neither the provider nor the
// tool's side effect.
func TestInstallRecordsAndReplaysBothHalves(t *testing.T) {
	path := tapePath(t)
	var sent atomic.Int64
	email := tool.Def{Name: "send_email", Description: "sends", InputSchema: json.RawMessage(`{"type":"object"}`),
		Caps: tool.CapMutating,
		Fn:   func(context.Context, json.RawMessage) (string, error) { sent.Add(1); return "sent", nil }}

	script := &replies{turns: []llm.Response{
		{Content: []llm.ContentBlock{llm.ToolUse{ID: "e1", Name: "send_email", Input: json.RawMessage(`{"to":"x"}`)}}, StopReason: llm.StopToolUse},
		{Content: []llm.ContentBlock{llm.Text{Text: "emailed"}}, StopReason: llm.StopEndTurn},
	}}
	run := func(cl llm.Client, tp *Tape) core.Outcome {
		a, err := core.New(core.WithClient(cl), core.WithModel("m"), core.WithTools(email), tp.Install())
		if err != nil {
			t.Fatal(err)
		}
		out, err := a.Run(context.Background(), core.Ask("email x"))
		if err != nil {
			t.Fatal(err)
		}
		return out
	}

	rec := openTape(t, path, Record)
	run(script, rec)
	if err := rec.Close(); err != nil {
		t.Fatal(err)
	}
	if sent.Load() != 1 {
		t.Fatalf("recording sent %d emails, want 1", sent.Load())
	}

	out := run(dead{}, openTape(t, path, Replay))
	if a, ok := out.(core.Answer); !ok || a.Text != "emailed" {
		t.Errorf("replay outcome = %#v", out)
	}
	if sent.Load() != 1 {
		t.Errorf("the replay sent the email again (%d sent)", sent.Load())
	}
}
