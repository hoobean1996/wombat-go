package skill_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/skill"
	"github.com/hoobean1996/wombat-go/tool"
)

// script replays canned replies and records the requests.
type script struct {
	mu    sync.Mutex
	turns []llm.Response
	seen  []llm.Request
}

func (s *script) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seen = append(s.seen, req)
	if len(s.turns) == 0 {
		return llm.Response{}, errors.New("script: out of turns")
	}
	r := s.turns[0]
	s.turns = s.turns[1:]
	return r, nil
}

func call(id, name, input string) llm.Response {
	return llm.Response{Content: []llm.ContentBlock{llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(input)}},
		StopReason: llm.StopToolUse}
}

func say(s string) llm.Response {
	return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: s}}, StopReason: llm.StopEndTurn}
}

func demoSkill() skill.Skill {
	return skill.Skill{Name: "demo", Description: "A demo skill gating the secret tool.", Body: "the secret is 42"}
}

func secretTool() tool.Def {
	return tool.Def{Name: "secret", Description: "gated", InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn: func(context.Context, json.RawMessage) (string, error) { return "42", nil }}
}

func offered(req llm.Request) map[string]bool {
	out := map[string]bool{}
	for _, t := range req.Tools {
		out[t.Name] = true
	}
	return out
}

// Install wires everything a gated surface needs: the gated tool is hidden,
// loading the skill reveals it, and the catalogue is in the prompt.
func TestInstallWiresTheWholeSurface(t *testing.T) {
	reg := skill.New(demoSkill())
	reg.Gate("demo", "secret")
	g := reg.Bind([]tool.Def{secretTool()})

	cl := &script{turns: []llm.Response{call("l1", skill.LoadToolName, `{"name":"demo"}`), call("s1", "secret", `{}`), say("done")}}
	a, err := wombat.New(wombat.WithClient(cl), wombat.WithModel("m"), g.Install())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Run(context.Background(), wombat.Ask("go")); err != nil {
		t.Fatal(err)
	}
	if offered(cl.seen[0])["secret"] {
		t.Error("the gated tool was offered before its skill was loaded")
	}
	if !offered(cl.seen[1])["secret"] {
		t.Error("the gated tool was not offered after its skill was loaded")
	}
	if !strings.Contains(cl.seen[0].System, "<available_skills>") {
		t.Error("the catalogue is not in the system prompt")
	}
}

// The reviewer's reproduction: a pause tool gated behind a skill that has not
// been loaded. Calling it from memory used to pause the run anyway.
func TestGatedPauseToolIsRefusedUntilItsSkillIsLoaded(t *testing.T) {
	reg := skill.New(skill.Skill{Name: "intake", Description: "how to interview the user", Body: "ask one thing at a time"})
	reg.Gate("intake", "ask_user")
	ask := tool.Def{Name: "ask_user", Description: "asks", InputSchema: json.RawMessage(`{"type":"object"}`), Caps: tool.CapPause,
		Fn: func(context.Context, json.RawMessage) (string, error) { return "", errors.New("must never run") }}
	g := reg.Bind([]tool.Def{ask})

	cl := &script{turns: []llm.Response{call("c1", "ask_user", `{"question":"which env?"}`), say("did it myself")}}
	a, err := wombat.New(wombat.WithClient(cl), wombat.WithModel("m"), g.Install())
	if err != nil {
		t.Fatal(err)
	}
	out, err := a.Run(context.Background(), wombat.Ask("deploy it"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if _, paused := out.(wombat.Paused); paused {
		t.Fatal("the run paused on a tool whose skill was never loaded")
	}
	msgs := cl.seen[1].Messages
	tr, ok := msgs[len(msgs)-1].Content[0].(llm.ToolResult)
	if !ok || !tr.IsError || !strings.Contains(tr.Content, `needs the "intake" skill`) {
		t.Errorf("tool_result = %+v, want the skill gate's refusal", tr)
	}
}

// Wiring the set without a State is the mistake that used to fail silently.
// The agent says so when it is built.
func TestAnAgentWithoutSkillStateIsWarnedAboutAtNew(t *testing.T) {
	reg := skill.New(demoSkill())
	g := reg.Bind(nil)
	var logs bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logs, nil))

	if _, err := wombat.New(wombat.WithClient(&script{}), wombat.WithLogger(logger),
		wombat.WithToolSet(g.Set), wombat.WithToolMiddleware(g.Middleware)); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(logs.String(), "per-run state") {
		t.Errorf("no warning about the missing State:\n%s", logs.String())
	}

	logs.Reset()
	if _, err := wombat.New(wombat.WithClient(&script{}), wombat.WithLogger(logger), g.Install()); err != nil {
		t.Fatal(err)
	}
	if logs.Len() != 0 {
		t.Errorf("Install still produced a warning:\n%s", logs.String())
	}
}
