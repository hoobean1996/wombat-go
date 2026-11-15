package permission

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/llm"
)

// ===== a scripted model, just enough for a parent and a child =====

type turns struct {
	mu   sync.Mutex
	next []llm.Response
	seen []llm.Request
}

func (s *turns) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seen = append(s.seen, req)
	if len(s.next) == 0 {
		return llm.Response{}, fmt.Errorf("script exhausted after %d calls", len(s.seen))
	}
	r := s.next[0]
	s.next = s.next[1:]
	return r, nil
}

func callTurn(id, name, input string) llm.Response {
	return llm.Response{
		Content:    []llm.ContentBlock{llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(input)}},
		StopReason: llm.StopToolUse,
	}
}

func answerTurn(s string) llm.Response {
	return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: s}}, StopReason: llm.StopEndTurn}
}

func agentFor(t *testing.T, cl llm.Client, opts ...wombat.Option) *wombat.Agent {
	t.Helper()
	base := []wombat.Option{
		wombat.WithClient(cl),
		wombat.WithModel("m"),
		wombat.WithLogger(slog.New(slog.NewTextHandler(io.Discard, nil))),
	}
	a, err := wombat.New(append(base, opts...)...)
	if err != nil {
		t.Fatalf("wombat.New: %v", err)
	}
	return a
}

// countingRule counts how often a policy is consulted and never decides.
func countingRule(n *atomic.Int64) Rule {
	return func(context.Context, Request) (Decision, string) {
		n.Add(1)
		return Undecided, ""
	}
}

// delegation runs a parent whose model delegates once, to a child whose model
// calls bash once. It reports whether bash ran, and the tool_result the child's
// model got back for its call.
func delegation(t *testing.T, parentOpts, childOpts []wombat.Option) (ran bool, childResult llm.ToolResult) {
	t.Helper()
	var calls calls
	childCl := &turns{next: []llm.Response{callTurn("b1", "bash", `{"command":"rm -rf /"}`), answerTurn("child done")}}
	child := agentFor(t, childCl, append([]wombat.Option{wombat.WithTools(bash(&calls))}, childOpts...)...)

	parentCl := &turns{next: []llm.Response{callTurn("d1", "delegate", `{"task":"clean up"}`), answerTurn("parent done")}}
	parent := agentFor(t, parentCl,
		append([]wombat.Option{wombat.WithTools(wombat.DelegateTool(child))}, parentOpts...)...)

	if _, err := parent.Run(context.Background(), wombat.Ask("go")); err != nil {
		t.Fatalf("parent Run: %v", err)
	}

	childCl.mu.Lock()
	defer childCl.mu.Unlock()
	if len(childCl.seen) < 2 {
		t.Fatalf("the child's model was called %d times, want 2", len(childCl.seen))
	}
	for _, m := range childCl.seen[1].Messages {
		for _, b := range m.Content {
			if tr, ok := b.(llm.ToolResult); ok && tr.ToolUseID == "b1" {
				childResult = tr
			}
		}
	}
	return calls.n.Load() > 0, childResult
}

// TestGateFollowsTheDelegateCallIntoTheSubagent is a regression test.
//
// Workspace allows a delegate call without asking, on the understanding that
// the sub-agent's calls are gated too. Nothing made that true: a child built
// without the gate ran whatever its parent would have had to ask about. Here
// the parent's policy refuses bash outright and the child has no gate at all;
// the child's bash call must still be refused, by the parent's gate.
func TestGateFollowsTheDelegateCallIntoTheSubagent(t *testing.T) {
	policy := Policy{Rules: []Rule{DenyTools("bash")}, Default: Allow}

	ran, res := delegation(t,
		[]wombat.Option{wombat.WithToolMiddleware(Gate(policy, nil))},
		nil, // the child is built with no gate whatsoever
	)
	if ran {
		t.Fatal("the child ran bash: the parent's policy did not follow the delegate call")
	}
	if !res.IsError || !strings.Contains(res.Content, "deny list") {
		t.Errorf("the child's tool_result = %+v, want the parent policy's refusal", res)
	}
}

// The configuration the CLI uses — the same gate installed on parent and child
// — must decide each of the child's calls once, not once per copy: a second
// copy deciding again would ask a person the same question twice.
func TestTheSameGateOnParentAndChildDecidesOnce(t *testing.T) {
	var consulted atomic.Int64
	policy := Policy{Rules: []Rule{countingRule(&consulted), AllowTools("bash", "delegate")}}
	gate := Gate(policy, nil)

	ran, _ := delegation(t,
		[]wombat.Option{wombat.WithToolMiddleware(gate)},
		[]wombat.Option{wombat.WithToolMiddleware(gate)},
	)
	if !ran {
		t.Fatal("bash did not run, want it allowed")
	}
	// One verdict for the parent's delegate call, one for the child's bash.
	if got := consulted.Load(); got != 2 {
		t.Errorf("policy consulted %d times, want 2 — a call was decided by more than one copy of the gate", got)
	}
}

// A child with a gate of its own answers to both policies, its own first: a
// call the child's narrower policy refuses never reaches the parent's.
func TestAChildIsGovernedByItsOwnGateAndItsParents(t *testing.T) {
	var parentConsulted atomic.Int64
	parentPolicy := Policy{Rules: []Rule{countingRule(&parentConsulted)}, Default: Allow}

	t.Run("the child's own refusal wins", func(t *testing.T) {
		parentConsulted.Store(0)
		ran, res := delegation(t,
			[]wombat.Option{wombat.WithToolMiddleware(Gate(parentPolicy, nil))},
			[]wombat.Option{wombat.WithToolMiddleware(Gate(ReadOnly(), nil))},
		)
		if ran {
			t.Fatal("bash ran under a read-only child")
		}
		if !strings.Contains(res.Content, "refused") {
			t.Errorf("the child's tool_result = %+v, want its own policy's refusal", res)
		}
		// The parent was asked about the delegate call only.
		if got := parentConsulted.Load(); got != 1 {
			t.Errorf("parent policy consulted %d times, want 1 — it was asked about a call the child had already refused", got)
		}
	})

	t.Run("the parent's refusal still applies when the child allows", func(t *testing.T) {
		ran, _ := delegation(t,
			[]wombat.Option{wombat.WithToolMiddleware(Gate(Policy{Rules: []Rule{DenyTools("bash")}, Default: Allow}, nil))},
			[]wombat.Option{wombat.WithToolMiddleware(Gate(Policy{Default: Allow}, nil))},
		)
		if ran {
			t.Fatal("bash ran: the child's permissive gate overrode the parent's refusal")
		}
	})
}

// A grant given on the parent's side answers the child's identical question:
// grants and the inherited gate both travel on the context, and asking twice
// for the same call would be exactly the noise grants exist to remove.
func TestAnApprovalIsNotAskedForAgainInsideTheSubagent(t *testing.T) {
	var asked atomic.Int64
	approver := ApproverFunc(func(context.Context, Request) (bool, error) {
		asked.Add(1)
		return true, nil
	})
	gate := Gate(Policy{Rules: []Rule{AllowTools("delegate")}, Default: Ask}, approver)

	childCl := &turns{next: []llm.Response{
		callTurn("b1", "bash", `{"command":"ls"}`),
		callTurn("b2", "bash", `{"command":"ls"}`),
		answerTurn("child done"),
	}}
	child := agentFor(t, childCl, wombat.WithTools(bash(nil)))
	parentCl := &turns{next: []llm.Response{callTurn("d1", "delegate", `{"task":"look around"}`), answerTurn("done")}}
	parent := agentFor(t, parentCl,
		wombat.WithTools(wombat.DelegateTool(child)),
		wombat.WithToolMiddleware(gate),
		wombat.WithRunContext(func(ctx context.Context) context.Context { return WithGrants(ctx, NewGrants()) }),
	)
	if _, err := parent.Run(context.Background(), wombat.Ask("go")); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := asked.Load(); got != 1 {
		t.Errorf("the person was asked %d times, want 1 — the identical second call should be granted", got)
	}
}
