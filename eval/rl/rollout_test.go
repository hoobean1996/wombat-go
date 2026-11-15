package rl

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/tool"
)

// mkOf adapts a fixed agent-building closure to AgentFunc.
func mkOf(f func(t Task) (*core.Agent, error)) AgentFunc { return f }

// answerAfterOneTool is the shape most of these tests want: call a tool, then
// answer.
func answerAfterOneTool(toolName string) llm.Client {
	return turnClient(func(turn int, _ llm.Request) llm.Response {
		if turn == 0 {
			return toolTurn("u1", toolName, `{}`)
		}
		return textTurn("done")
	})
}

func TestRolloutReconstructsStepsFromTheEventStream(t *testing.T) {
	env := newMemEnv("steps", "do the thing")

	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, answerAfterOneTool("read"), []tool.Def{
			fnTool("read", func(context.Context, json.RawMessage) (string, error) {
				return "contents", nil
			}),
		}), nil
	})

	g, err := Rollout(t.Context(), mk, env, 1)
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	ep := g.Episodes[0]
	if ep.Err != nil {
		t.Fatalf("episode failed: %v", ep.Err)
	}
	if got, want := len(ep.Steps), 2; got != want {
		t.Fatalf("steps = %d, want %d: %+v", got, want, ep.Steps)
	}
	if got, want := ep.Steps[0].Tools, []string{"read"}; !reflect.DeepEqual(got, want) {
		t.Errorf("step 1 tools = %v, want %v", got, want)
	}
	if ep.Steps[0].Iteration != 1 || ep.Steps[1].Iteration != 2 {
		t.Errorf("iterations = %d,%d want 1,2", ep.Steps[0].Iteration, ep.Steps[1].Iteration)
	}
	// Usage exists only on the event stream; Agent.Run would have discarded it.
	if u := ep.Usage(); u.InputTokens == 0 || u.OutputTokens == 0 {
		t.Errorf("usage was not folded: %+v", u)
	}
	if ep.Failure != Success {
		t.Errorf("failure = %q, want success", ep.Failure)
	}
	if _, ok := ep.Outcome.(core.Answer); !ok {
		t.Errorf("outcome = %T, want core.Answer", ep.Outcome)
	}
	if len(ep.Messages) == 0 {
		t.Error("transcript was not captured")
	}
}

func TestRolloutRecordsDeniedCalls(t *testing.T) {
	env := newMemEnv("denied", "write a file")
	// Deny everything, through the real gate, so the error is the real one.
	gate := permission.Gate(permission.Policy{Default: permission.Deny}, nil)

	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, answerAfterOneTool("write"),
			[]tool.Def{fnTool("write", func(context.Context, json.RawMessage) (string, error) {
				return "wrote", nil
			})},
			core.WithToolMiddleware(gate),
		), nil
	})

	g, err := Rollout(t.Context(), mk, env, 1)
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	s := g.Episodes[0].Steps[0]
	if want := []string{"write"}; !reflect.DeepEqual(s.Failed, want) {
		t.Errorf("Failed = %v, want %v", s.Failed, want)
	}
	if want := []string{"write"}; !reflect.DeepEqual(s.Denied, want) {
		t.Fatalf("Denied = %v, want %v — the refusal was not matched on permission.ErrDenied", s.Denied, want)
	}
}

// TestRolloutIsolation is the property the whole package rests on: eight
// samples running at once must not see each other's workspace.
func TestRolloutIsolation(t *testing.T) {
	root := t.TempDir()
	const n = 8

	// Each sample writes its own marker and reads it back. A shared workspace
	// makes the readback return somebody else's number.
	mk := mkOf(func(task Task) (*core.Agent, error) {
		ws := task.Workspace // captured at CONSTRUCTION, which is why AgentFunc exists
		mark := fmt.Sprintf("sample-%d", task.Sample)

		write := fnTool("write", func(context.Context, json.RawMessage) (string, error) {
			// Slow enough that all eight overlap.
			time.Sleep(5 * time.Millisecond)
			return "", os.WriteFile(filepath.Join(ws, "mark.txt"), []byte(mark), 0o600)
		})
		read := fnTool("read", func(context.Context, json.RawMessage) (string, error) {
			b, err := os.ReadFile(filepath.Join(ws, "mark.txt"))
			return string(b), err
		})

		cl := turnClient(func(turn int, _ llm.Request) llm.Response {
			switch turn {
			case 0:
				return toolTurn("u1", "write", `{}`)
			case 1:
				return toolTurn("u2", "read", `{}`)
			default:
				return textTurn("done")
			}
		})
		return newAgent(t, cl, []tool.Def{write, read}), nil
	})

	env := Dir(root, "isolation", "write then read your marker",
		Score(Verifier(func(_ context.Context, ep *Episode) (string, float64) {
			want := fmt.Sprintf("sample-%d", ep.Task.Sample)
			b, err := os.ReadFile(filepath.Join(ep.Task.Workspace, "mark.txt"))
			if err != nil || string(b) != want {
				return "marker", 0
			}
			return "marker", 1
		})))

	g, err := Rollout(t.Context(), mk, env, n, WithConcurrency(n), WithKeepWorkspaces())
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}

	seen := map[string]int{}
	for i, ep := range g.Episodes {
		if ep.Err != nil {
			t.Fatalf("sample %d failed: %v", i, ep.Err)
		}
		if ep.Task.Sample != i {
			t.Fatalf("Episodes[%d] holds sample %d", i, ep.Task.Sample)
		}
		if seen[ep.Task.Workspace]++; seen[ep.Task.Workspace] > 1 {
			t.Fatalf("workspace %s was handed to two samples", ep.Task.Workspace)
		}
		if ep.Reward != 1 {
			t.Errorf("sample %d read back the wrong marker (reward %v)", i, ep.Reward)
		}
	}
	if got := g.PassAt(1); got != 1 {
		t.Errorf("pass@1 = %v, want 1", got)
	}
}

// TestRolloutGivesEachEpisodeItsOwnBudget pins the other half of isolation: a
// shared budget would let the first sample starve the rest.
func TestRolloutGivesEachEpisodeItsOwnBudget(t *testing.T) {
	const n = 4

	// Charges a full dollar per model call, so a $1 cap allows exactly one.
	charge := func(next llm.Client) llm.Client {
		return llm.ClientFunc(func(ctx context.Context, req llm.Request) (llm.Response, error) {
			resp, err := next.Complete(ctx, req)
			governor.FromContext(ctx).AddCall(1.00, governor.Tokens{In: 1})
			return resp, err
		})
	}

	env := newMemEnv("budget", "answer")
	mk := mkOf(func(Task) (*core.Agent, error) {
		cl := llm.Chain(turnClient(func(int, llm.Request) llm.Response {
			return textTurn("hi")
		}), charge)
		return newAgent(t, cl, nil), nil
	})

	g, err := Rollout(t.Context(), mk, env, n,
		WithConcurrency(n), WithBudget(governor.Limits{CostUSD: 1.00}))
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	for i, ep := range g.Episodes {
		if ep.Err != nil {
			t.Fatalf("sample %d was starved by another sample's spend: %v", i, ep.Err)
		}
		if got, want := ep.Spend.CostUSD, 1.00; got != want {
			t.Errorf("sample %d spend = %v, want %v (its own tally, not the rollout's)", i, got, want)
		}
	}
}

func TestRolloutFailureKindsEndToEnd(t *testing.T) {
	tests := []struct {
		name  string
		agent func(t *testing.T) (*core.Agent, error)
		limit governor.Limits
		want  FailureKind
	}{
		{
			name: "answers cleanly",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
					return textTurn("done")
				}), nil), nil
			},
			want: Success,
		},
		{
			name: "never stops calling tools",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, turnClient(func(turn int, _ llm.Request) llm.Response {
					return toolTurn(fmt.Sprint("u", turn), "noop", `{}`)
				}), []tool.Def{fnTool("noop", func(context.Context, json.RawMessage) (string, error) {
					return "ok", nil
				})}, core.WithMaxIters(3)), nil
			},
			want: MaxIterations,
		},
		{
			name: "runs out of reply",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
					return llm.Response{
						Content:    []llm.ContentBlock{llm.Text{Text: "half an ans"}},
						StopReason: llm.StopMaxTokens,
					}
				}), nil), nil
			},
			want: MaxTokens,
		},
		{
			name: "declines",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
					return llm.Response{
						Content:    []llm.ContentBlock{llm.Text{Text: "I will not"}},
						StopReason: llm.StopRefusal,
					}
				}), nil), nil
			},
			want: Refused,
		},
		{
			name: "outgrows the context window",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, failingClient(&llm.APIError{
					Class: llm.ErrContextWindow, Status: 400,
				}), nil), nil
			},
			want: ContextWindow,
		},
		{
			name: "hits a provider outage",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, failingClient(&llm.APIError{
					Class: llm.ErrOverloaded, Status: 529,
				}), nil), nil
			},
			want: ProviderError,
		},
		{
			name: "panics inside the loop",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, llm.ClientFunc(
					func(context.Context, llm.Request) (llm.Response, error) {
						panic("client exploded")
					}), nil), nil
			},
			want: Panicked,
		},
		{
			name: "exhausts its step budget",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, turnClient(func(turn int, _ llm.Request) llm.Response {
					return toolTurn(fmt.Sprint("u", turn), "noop", `{}`)
				}), []tool.Def{fnTool("noop", func(context.Context, json.RawMessage) (string, error) {
					return "ok", nil
				})}), nil
			},
			limit: governor.Limits{Steps: 2},
			want:  MaxIterations,
		},
		{
			name: "loops on one identical call",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, turnClient(func(turn int, _ llm.Request) llm.Response {
					return toolTurn(fmt.Sprint("u", turn), "noop", `{"same":1}`)
				}), []tool.Def{fnTool("noop", func(context.Context, json.RawMessage) (string, error) {
					return "ok", nil
				})}), nil
			},
			limit: governor.Limits{RepeatedToolCalls: 2},
			want:  ToolLoop,
		},
		{
			name: "runs out of wall clock",
			agent: func(t *testing.T) (*core.Agent, error) {
				return newAgent(t, llm.ClientFunc(
					func(ctx context.Context, _ llm.Request) (llm.Response, error) {
						<-ctx.Done()
						return llm.Response{}, ctx.Err()
					}), nil), nil
			},
			limit: governor.Limits{Wall: 20 * time.Millisecond},
			want:  WallClock,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			env := newMemEnv("kinds", "go")
			g, err := Rollout(t.Context(),
				mkOf(func(Task) (*core.Agent, error) { return tc.agent(t) }),
				env, 1, WithBudget(tc.limit))
			if err != nil {
				t.Fatalf("Rollout: %v", err)
			}
			if got := g.Episodes[0].Failure; got != tc.want {
				t.Fatalf("failure = %q (err %v), want %q", got, g.Episodes[0].Err, tc.want)
			}
		})
	}
}

func TestRolloutVerifierFailedIsSeparateFromSuccess(t *testing.T) {
	env := newMemEnv("scored", "go")
	env.scoreFn = func(*Episode) (float64, map[string]float64, error) {
		return 0.6, map[string]float64{"build": 0.6, "test": 0}, nil
	}
	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
			return textTurn("done")
		}), nil), nil
	})

	g, err := Rollout(t.Context(), mk, env, 1)
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	if got := g.Episodes[0].Failure; got != VerifierFailed {
		t.Fatalf("failure = %q, want %q — a clean run with a low score is the interesting row", got, VerifierFailed)
	}

	// The same episode passes once the threshold matches the task's scale.
	g, err = Rollout(t.Context(), mk, env, 1, WithSuccessThreshold(0.5))
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	if got := g.Episodes[0].Failure; got != Success {
		t.Fatalf("failure = %q, want success under a 0.5 threshold", got)
	}
}

func TestRolloutCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	env := newMemEnv("cancelled", "go")

	started := make(chan struct{})
	var once sync.Once
	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, llm.ClientFunc(func(c context.Context, _ llm.Request) (llm.Response, error) {
			once.Do(func() { close(started) })
			<-c.Done()
			return llm.Response{}, c.Err()
		}), nil), nil
	})

	done := make(chan *Group, 1)
	go func() {
		g, _ := Rollout(ctx, mk, env, 2, WithConcurrency(2))
		done <- g
	}()

	<-started
	cancel()

	g := <-done
	for i, ep := range g.Episodes {
		if ep.Failure != Cancelled {
			t.Errorf("sample %d failure = %q (err %v), want cancelled", i, ep.Failure, ep.Err)
		}
	}
}

func TestRolloutSurvivesABrokenEnvironment(t *testing.T) {
	env := newMemEnv("broken", "go")
	boom := errors.New("no disk space")
	env.reset = func(sample int) (Task, error) {
		if sample == 2 {
			return Task{}, boom
		}
		return Task{ID: "broken", Sample: sample, Prompt: "go"}, nil
	}
	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
			return textTurn("done")
		}), nil), nil
	})

	g, err := Rollout(t.Context(), mk, env, 4)
	if err != nil {
		t.Fatalf("Rollout returned an error for ONE broken sample: %v", err)
	}
	if got, want := g.Successes(), 3; got != want {
		t.Errorf("successes = %d, want %d", got, want)
	}
	bad := g.Episodes[2]
	if bad.Failure != Other || !errors.Is(bad.Err, boom) {
		t.Errorf("sample 2 = %q / %v, want other wrapping the reset error", bad.Failure, bad.Err)
	}
	if g.TaskID != "broken" {
		t.Errorf("TaskID = %q, want %q", g.TaskID, "broken")
	}
	// The broken sample never got a workspace, so there was nothing to clean.
	if got, want := env.cleanups(), 3; got != want {
		t.Errorf("cleanups = %d, want %d", got, want)
	}
}

func TestRolloutSurvivesAPanickingEnvironment(t *testing.T) {
	env := newMemEnv("panicky", "go")
	env.scoreFn = func(ep *Episode) (float64, map[string]float64, error) {
		if ep.Task.Sample == 1 {
			panic("scorer is broken")
		}
		return 1, nil, nil
	}
	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
			return textTurn("done")
		}), nil), nil
	})

	g, err := Rollout(t.Context(), mk, env, 3)
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	if got := g.Episodes[1].Failure; got != Panicked {
		t.Errorf("panicking sample = %q, want panic", got)
	}
	if got, want := g.Successes(), 2; got != want {
		t.Errorf("successes = %d, want %d — one bad sample must not lose the others", got, want)
	}
}

func TestRolloutCleanupKeepsFailures(t *testing.T) {
	env := newMemEnv("keep", "go")
	env.scoreFn = func(ep *Episode) (float64, map[string]float64, error) {
		if ep.Task.Sample%2 == 0 {
			return 1, nil, nil
		}
		return 0, nil, nil
	}
	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
			return textTurn("done")
		}), nil), nil
	})

	if _, err := Rollout(t.Context(), mk, env, 4); err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	for sample := range 4 {
		want := sample%2 == 1 // odd samples scored 0 and therefore failed
		if got := env.kept(sample); got != want {
			t.Errorf("sample %d kept = %v, want %v", sample, got, want)
		}
	}

	// WithKeepWorkspaces keeps the passing ones too.
	env2 := newMemEnv("keepall", "go")
	if _, err := Rollout(t.Context(), mk, env2, 2, WithKeepWorkspaces()); err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	for sample := range 2 {
		if !env2.kept(sample) {
			t.Errorf("sample %d was not kept despite WithKeepWorkspaces", sample)
		}
	}
}

func TestRolloutProgressCallback(t *testing.T) {
	env := newMemEnv("progress", "go")
	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
			return textTurn("done")
		}), nil), nil
	})

	var mu sync.Mutex
	seen := map[int]bool{}
	_, err := Rollout(t.Context(), mk, env, 5, WithConcurrency(5),
		WithProgress(func(task Task, ep *Episode) {
			mu.Lock()
			defer mu.Unlock()
			if ep == nil {
				t.Error("progress got a nil episode")
				return
			}
			seen[task.Sample] = true
		}))
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	if len(seen) != 5 {
		t.Fatalf("progress fired for %d of 5 samples", len(seen))
	}
}

func TestRolloutRejectsUnrunnableArguments(t *testing.T) {
	env := newMemEnv("x", "go")
	mk := mkOf(func(Task) (*core.Agent, error) { return nil, nil })

	if _, err := Rollout(t.Context(), nil, env, 1); err == nil {
		t.Error("a nil AgentFunc should be rejected")
	}
	if _, err := Rollout(t.Context(), mk, nil, 1); err == nil {
		t.Error("a nil Env should be rejected")
	}
	if _, err := Rollout(t.Context(), mk, env, 0); !errors.Is(err, ErrNoSamples) {
		t.Errorf("n=0 error = %v, want ErrNoSamples", err)
	}
	// A factory that hands back nil must be an episode failure, not a crash.
	g, err := Rollout(t.Context(), mk, env, 1)
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	if g.Episodes[0].Failure != Other || !strings.Contains(g.Episodes[0].Err.Error(), "nil agent") {
		t.Errorf("nil agent = %q / %v", g.Episodes[0].Failure, g.Episodes[0].Err)
	}
}

func TestDirEnv(t *testing.T) {
	root := t.TempDir()
	env := Dir(root, "todo", "write main.go", Score(FileExists("main", "main.go", 1)))

	task, err := env.Reset(t.Context(), 3)
	if err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if want := filepath.Join(root, "todo", "sample-3"); task.Workspace != want {
		t.Fatalf("workspace = %q, want %q", task.Workspace, want)
	}
	if !filepath.IsAbs(task.Workspace) {
		t.Errorf("workspace %q is not absolute", task.Workspace)
	}

	// A rerun is a rerun: last run's leftovers must not survive.
	stale := filepath.Join(task.Workspace, "stale.txt")
	if err := os.WriteFile(stale, []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := env.Reset(t.Context(), 3); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Error("Reset left the previous run's files in place")
	}

	// Cleanup honours Keep.
	if err := env.Cleanup(WithKeep(t.Context(), true), task); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if _, err := os.Stat(task.Workspace); err != nil {
		t.Error("Cleanup removed a workspace it was asked to keep")
	}
	if err := env.Cleanup(t.Context(), task); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if _, err := os.Stat(task.Workspace); !os.IsNotExist(err) {
		t.Error("Cleanup left the workspace behind")
	}
}
