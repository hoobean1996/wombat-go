package rl

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// ScoreFunc grades a finished episode. It is the shape [Score] produces and
// the shape [Dir] consumes.
type ScoreFunc func(ctx context.Context, ep *Episode) (float64, map[string]float64, error)

// Dir is the Env most tasks want: a scratch directory per sample.
//
// Reset makes root/id/sample-N, removing it first so a rerun is a rerun and
// not an accumulation — a task that passes only because the previous run left
// its build artifacts behind is the classic way a benchmark stops measuring
// anything.
//
// Cleanup removes the directory only when the episode SUCCEEDED. A failed
// episode's directory is the first thing anyone will want to look at: the
// half-written file, the test output, the thing the agent did instead of the
// thing it was asked to do. Deleting it is deleting the evidence, and re-running
// to reproduce costs another sample's worth of money and may not fail the same
// way. Successful ones are removed because they are just disk; keep them too
// with [WithKeepWorkspaces].
//
// root is created if it does not exist. A relative root is resolved against
// the process working directory, because [Task.Workspace] is documented as
// absolute and tools resolve paths against it.
func Dir(root, id, prompt string, score func(context.Context, *Episode) (float64, map[string]float64, error)) Env {
	return &dirEnv{root: root, id: id, prompt: prompt, score: score}
}

type dirEnv struct {
	root   string
	id     string
	prompt string
	score  ScoreFunc
}

// Name implements Env.
func (e *dirEnv) Name() string { return "dir:" + e.id }

// Reset implements Env.
func (e *dirEnv) Reset(_ context.Context, sample int) (Task, error) {
	root, err := filepath.Abs(e.root)
	if err != nil {
		return Task{}, fmt.Errorf("rl: resolving workspace root %q: %w", e.root, err)
	}
	ws := filepath.Join(root, e.id, "sample-"+strconv.Itoa(sample))

	// Removed then recreated, in that order, so a rerun starts from nothing.
	if err := os.RemoveAll(ws); err != nil {
		return Task{}, fmt.Errorf("rl: clearing workspace %s: %w", ws, err)
	}
	if err := os.MkdirAll(ws, 0o755); err != nil {
		return Task{}, fmt.Errorf("rl: creating workspace %s: %w", ws, err)
	}

	return Task{ID: e.id, Sample: sample, Prompt: e.prompt, Workspace: ws}, nil
}

// Score implements Env.
func (e *dirEnv) Score(ctx context.Context, ep *Episode) (float64, map[string]float64, error) {
	if e.score == nil {
		return 0, nil, errors.New("rl: Dir was given no score function")
	}
	return e.score(ctx, ep)
}

// Cleanup implements Env, honouring [Keep].
func (e *dirEnv) Cleanup(ctx context.Context, t Task) error {
	if Keep(ctx) || t.Workspace == "" {
		return nil
	}
	if err := os.RemoveAll(t.Workspace); err != nil {
		return fmt.Errorf("rl: removing workspace %s: %w", t.Workspace, err)
	}
	return nil
}
