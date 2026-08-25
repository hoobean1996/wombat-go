package rl

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
	"time"
)

// DefaultShellTimeout caps a [Shell] verifier that would otherwise hang.
//
// A verifier is not the agent: it runs a build or a test suite that either
// finishes or is broken, and a benchmark whose scoring phase blocks forever
// has failed in the least debuggable way available.
const DefaultShellTimeout = 2 * time.Minute

// VerifierOutputLimit is how much of a failing command's output is logged.
const VerifierOutputLimit = 8 << 10

// Verifier turns one aspect of "did it work" into a number.
//
// The name it returns is its key in the breakdown, so it must be stable across
// runs and across episodes — a key that encodes the failure would make every
// episode's breakdown a different shape and nothing could be averaged.
//
// A Verifier reports no error. Either the thing it checks is true and it
// scores, or it is not and it scores zero; "the check itself broke" is
// indistinguishable to a benchmark from "the check failed", and giving it a
// third answer only moves the decision to a caller that has less information.
type Verifier func(ctx context.Context, ep *Episode) (name string, score float64)

// Shell scores weight when cmd exits 0 in the episode's workspace, 0
// otherwise. It runs through /bin/sh -c, so a pipeline or a && chain works.
//
//	rl.Shell("test", "go test ./... 2>&1", 1.0)
//
// The command inherits the process environment with the workspace as its
// working directory, and is capped at [DefaultShellTimeout]. A non-zero exit
// logs the command's combined output at warn level, truncated to
// [VerifierOutputLimit] — the breakdown can only hold a number, and "test: 0"
// on its own is a result nobody can act on. The workspace is kept when the
// episode fails, so the same command can be rerun by hand.
func Shell(name, cmd string, weight float64) Verifier {
	return func(ctx context.Context, ep *Episode) (string, float64) {
		if ep.Task.Workspace == "" {
			slog.WarnContext(ctx, "rl: shell verifier has no workspace",
				slog.String("verifier", name), slog.String("cmd", cmd))
			return name, 0
		}

		ctx, cancel := context.WithTimeout(ctx, DefaultShellTimeout)
		defer cancel()

		c := exec.CommandContext(ctx, "/bin/sh", "-c", cmd)
		c.Dir = ep.Task.Workspace
		out, err := c.CombinedOutput()
		if err == nil {
			return name, weight
		}

		slog.WarnContext(ctx, "rl: shell verifier failed",
			slog.String("verifier", name),
			slog.String("cmd", cmd),
			slog.String("workspace", ep.Task.Workspace),
			slog.Any("err", err),
			slog.String("output", truncate(string(out), VerifierOutputLimit)))
		return name, 0
	}
}

// FileExists scores weight when rel exists in the workspace.
//
// rel is joined to the workspace and must stay inside it; a path that escapes
// scores 0 rather than reaching out of the sandbox to find a file the agent
// never wrote.
func FileExists(name, rel string, weight float64) Verifier {
	return func(ctx context.Context, ep *Episode) (string, float64) {
		p, ok := within(ep.Task.Workspace, rel)
		if !ok {
			slog.WarnContext(ctx, "rl: verifier path escapes the workspace",
				slog.String("verifier", name), slog.String("path", rel))
			return name, 0
		}
		if _, err := os.Stat(p); err != nil {
			return name, 0
		}
		return name, weight
	}
}

// FileContains scores weight when rel exists in the workspace and contains
// substr. Both conditions in one verifier because "the file is missing" and
// "the file is wrong" are the same finding at scoring time: the content is not
// there.
func FileContains(name, rel, substr string, weight float64) Verifier {
	return func(ctx context.Context, ep *Episode) (string, float64) {
		p, ok := within(ep.Task.Workspace, rel)
		if !ok {
			slog.WarnContext(ctx, "rl: verifier path escapes the workspace",
				slog.String("verifier", name), slog.String("path", rel))
			return name, 0
		}
		b, err := os.ReadFile(p)
		if err != nil || !strings.Contains(string(b), substr) {
			return name, 0
		}
		return name, weight
	}
}

// TurnPenalty docks perTurn for every ReAct iteration the episode took.
//
// A negative term, and the sign is the point: two agents that both finish the
// task are not equally good if one took thirty turns, and without a penalty
// the reward has no gradient toward doing less. Keep it small relative to the
// positive verifiers — a penalty that can outweigh the task teaches the agent
// to give up early, which scores better than trying.
func TurnPenalty(perTurn float64) Verifier {
	return func(_ context.Context, ep *Episode) (string, float64) {
		return "turn_penalty", -perTurn * float64(len(ep.Steps))
	}
}

// CostPenalty docks perUSD for every dollar the episode spent.
//
// Reads [Episode.Spend], which is the episode's own governor tally, so it
// counts sub-agent spend as well.
//
// Inert against a model nobody has a rate for: an unpriced episode spends
// $0.00 as far as the tally is concerned, so this term is 0 and the ranking it
// exists to produce has no cost gradient at all. That is the honest outcome —
// a made-up rate would put a fabricated number straight into the reward — but
// it does mean a -penalize run against such a model is ranking on turns and
// tool errors only. [Episode.Priced] says when that is happening.
func CostPenalty(perUSD float64) Verifier {
	return func(_ context.Context, ep *Episode) (string, float64) {
		return "cost_penalty", -perUSD * ep.Spend.CostUSD
	}
}

// ToolErrorPenalty docks perError for every tool call that failed, refusals
// included.
//
// Refusals included on purpose: an agent that repeatedly tries what it is not
// allowed to do is wasting the run whether or not the policy is right, and the
// [Denied] column already records that permission is what stopped it.
func ToolErrorPenalty(perError float64) Verifier {
	return func(_ context.Context, ep *Episode) (string, float64) {
		return "tool_error_penalty", -perError * float64(ep.ToolErrors())
	}
}

// Score combines verifiers into the function [Env.Score] wants: the sum, and
// the breakdown that explains it.
//
// A verifier that panics scores 0 and is logged; it does not take the rollout
// with it. Scoring runs after the expensive part is over, and losing eight
// episodes of real model spend to a typo in a check is not a trade anyone
// would make on purpose.
//
// Two verifiers with the same name would collide in one breakdown key and
// silently lose a term from a total that still included it. The later one is
// suffixed — "build#2" — so the breakdown always adds up to the reward.
func Score(vs ...Verifier) func(context.Context, *Episode) (float64, map[string]float64, error) {
	return func(ctx context.Context, ep *Episode) (float64, map[string]float64, error) {
		total := 0.0
		breakdown := make(map[string]float64, len(vs))

		for i, v := range vs {
			name, score := safeVerify(ctx, v, i, ep)
			if name == "" {
				name = "verifier#" + strconv.Itoa(i)
			}
			if _, dup := breakdown[name]; dup {
				name += "#" + strconv.Itoa(i)
			}
			breakdown[name] = score
			total += score
		}
		return total, breakdown, nil
	}
}

// safeVerify runs one verifier with a panic guard.
func safeVerify(ctx context.Context, v Verifier, i int, ep *Episode) (name string, score float64) {
	defer func() {
		if p := recover(); p != nil {
			// Named results, so the zero score survives the recover.
			if name == "" {
				name = "verifier#" + strconv.Itoa(i)
			}
			score = 0
			slog.ErrorContext(ctx, "rl: verifier panicked",
				slog.String("verifier", name),
				slog.String("task", ep.Task.ID),
				slog.Int("sample", ep.Task.Sample),
				slog.Any("panic", p),
				slog.String("stack", truncate(string(debug.Stack()), VerifierOutputLimit)))
		}
	}()
	if v == nil {
		return "", 0
	}
	return v(ctx, ep)
}

// within joins rel to root and reports whether the result is still inside it.
func within(root, rel string) (string, bool) {
	if root == "" {
		return "", false
	}
	p := filepath.Join(root, rel)
	r, err := filepath.Rel(root, p)
	if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
		return "", false
	}
	return p, true
}

// truncate bounds a diagnostic string, saying how much it dropped so nobody
// mistakes the tail of a log for the end of the failure.
func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return s[:limit] + fmt.Sprintf("\n… %d more bytes", len(s)-limit)
}
