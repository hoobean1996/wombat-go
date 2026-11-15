package permission

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

type scriptedModel struct {
	turns []llm.Response
	seen  []llm.Request
}

func (s *scriptedModel) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	s.seen = append(s.seen, req)
	r := s.turns[0]
	s.turns = s.turns[1:]
	return r, nil
}

// With nobody to ask, the gate defers the call instead of refusing it; the
// decision a person makes later is what the gate answers with when the run
// resumes, and the tool's real output is what the model reads.
func TestDeferredApprovalRunsTheCallOnceDecided(t *testing.T) {
	for _, tc := range []struct {
		name    string
		allow   bool
		ran     int64
		inReply string
	}{
		{"approved", true, 1, "ran write_file"},
		{"refused", false, 0, "refused the write_file tool"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &calls{}
			use := llm.ToolUse{ID: "w1", Name: "write_file", Input: json.RawMessage(`{"path":"/etc/x"}`)}
			model := &scriptedModel{turns: []llm.Response{
				{Content: []llm.ContentBlock{use}, StopReason: llm.StopToolUse},
				{Content: []llm.ContentBlock{llm.Text{Text: "ok"}}, StopReason: llm.StopEndTurn},
			}}
			p := Policy{Rules: []Rule{func(_ context.Context, r Request) (Decision, string) {
				return Ask, "it writes outside the workspace"
			}}}
			a, err := core.New(core.WithClient(model), core.WithTools(writeFile(c)),
				core.WithToolMiddleware(Gate(p, Defer())))
			if err != nil {
				t.Fatal(err)
			}

			var kinds []string
			collect := func(r *core.Run) {
				for r.Next() {
					kinds = append(kinds, r.Event().Kind())
				}
			}
			r := a.Start(context.Background(), core.Ask("write it"))
			collect(r)
			d, ok := r.Outcome().(core.Deferred)
			if !ok || len(d.Calls) != 1 || d.Calls[0].Reason != "it writes outside the workspace" {
				t.Fatalf("outcome = %#v (err %v), want the write deferred with the policy's reason", r.Outcome(), r.Err())
			}
			if strings.Contains(strings.Join(kinds, " "), "permission_decided") {
				t.Errorf("a deferred call was reported decided: %v", kinds)
			}

			kinds = nil
			r = a.Start(context.Background(), core.Decide(r.Messages(), d, map[llm.ToolUseID]bool{"w1": tc.allow}))
			collect(r)
			if _, ok := r.Outcome().(core.Answer); !ok {
				t.Fatalf("outcome = %#v (err %v), want the run to carry on to an answer", r.Outcome(), r.Err())
			}
			if n := c.n.Load(); n != tc.ran {
				t.Errorf("the tool ran %d times, want %d", n, tc.ran)
			}
			last := model.seen[1].Messages
			res := last[len(last)-1].Content[0].(llm.ToolResult)
			if !strings.Contains(res.Content, tc.inReply) {
				t.Errorf("the model read %q, want %q in it", res.Content, tc.inReply)
			}
			if !strings.Contains(strings.Join(kinds, " "), "permission_decided") {
				t.Errorf("the decision was not reported: %v", kinds)
			}
		})
	}
}

// Defer's error is a deferral and nothing else: not a refusal the model is
// told about, and not an abandoned wait.
func TestDeferIsNotADenial(t *testing.T) {
	_, err := gated(Policy{Default: Ask}, Defer())(context.Background(), writeFile(nil), useOf("u1", args("path", "/x")))
	if _, deferred := tool.DeferralReason(err); !deferred {
		t.Fatalf("err = %v, want a deferral", err)
	}
	if strings.Contains(err.Error(), ErrDenied.Error()) {
		t.Errorf("a deferral reads as a denial: %v", err)
	}
}
