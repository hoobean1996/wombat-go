package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/tool"
)

const (
	gitLogDefaultN = 20
	gitLogMaxN     = 100
)

// requireSafeRef rejects a ref that would be read as an option.
//
// git has no `--` escape that works for revisions — `git show -- x` means the
// PATH x — so a ref beginning with `-` would be parsed as a flag, and git has
// flags that execute things (--upload-pack, --exec). Refusing the whole class
// is cheap; no real ref starts with a dash.
func requireSafeRef(field, ref string) error {
	if ref == "" {
		return fmt.Errorf("field '%s' must not be empty", field)
	}
	if strings.HasPrefix(ref, "-") {
		return fmt.Errorf("field '%s' must not start with '-' (it would be read as a git option), got: %q", field, ref)
	}
	return nil
}

// ===== git_log =====

type gitLogIn struct {
	Cwd  string `json:"cwd"`
	Path string `json:"path"`
	N    int    `json:"n"`
}

// GitLog lists recent commits as one-line summaries.
func GitLog(r sandbox.Runner) tool.Def {
	mustNotBeNil(r != nil, "GitLog requires a non-nil Runner")

	return tool.Typed(tool.Def{
		Name: "git_log",
		Description: "Show recent commits in a git repo as one-line summaries (sha, " +
			"date, author, subject). Optionally restrict to a single path or " +
			"subtree. CWD MUST BE ABSOLUTE and point at the repo root (or " +
			"any subdirectory inside it). [n] caps results (default 20, hard " +
			"cap 100). Read-only.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "cwd": {
      "type": "string",
      "description": "Absolute path inside the git repo"
    },
    "path": {
      "type": "string",
      "description": "Optional path filter (relative to repo root or absolute)"
    },
    "n": {
      "type": "integer",
      "description": "Max commits to return (1-100)"
    }
  },
  "required": ["cwd"]
}`),
		Caps:       tool.CapReadOnly,
		Needs:      tool.NeedExec | tool.NeedFSRead,
		Idempotent: true,
		Timeout:    5 * time.Second,
		Category:   "vcs",
		// Spawning is the only part of this that can fail transiently, so the
		// classifier is the exec one. Replaying is free: `git log` is read-only
		// with an argv this tool builds, not one the model wrote.
		Retryable: retryExec,
	}, func(ctx context.Context, in gitLogIn) (string, error) {
		if err := requireAbs("cwd", in.Cwd); err != nil {
			return "", err
		}

		n := in.N
		if n <= 0 {
			n = gitLogDefaultN
		}
		n = min(n, gitLogMaxN)

		// -C rather than the Runner's directory: the Runner is constructed
		// once for the whole process, and the model names the repo per call.
		args := []string{
			"-C", in.Cwd,
			"log",
			"--pretty=format:%h %ad %an: %s",
			"--date=short",
			"-n", fmt.Sprint(n),
		}
		if in.Path != "" {
			// After `--` git reads everything as a pathspec, so a path
			// beginning with `-` is harmless here.
			args = append(args, "--", in.Path)
		}

		out, err := r.Run(ctx, "git", args...)
		if err != nil {
			return "", err
		}
		trimmed := strings.TrimSpace(out)
		if trimmed == "" {
			return "(no commits)", nil
		}
		return truncate(trimmed, maxToolOutput), nil
	})
}

// ===== git_show =====

type gitShowIn struct {
	Cwd string `json:"cwd"`
	Ref string `json:"ref"`
}

// GitShow prints one commit's metadata and full diff.
func GitShow(r sandbox.Runner) tool.Def {
	mustNotBeNil(r != nil, "GitShow requires a non-nil Runner")

	return tool.Typed(tool.Def{
		Name: "git_show",
		Description: "Show a single commit's metadata + full diff. CWD MUST BE " +
			"ABSOLUTE. [ref] can be a sha, tag, branch, or relative ref " +
			"(e.g., HEAD~3). Output is the unified diff prefixed by the " +
			"commit message. Read-only.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "cwd": {
      "type": "string",
      "description": "Absolute path inside the git repo"
    },
    "ref": {
      "type": "string",
      "description": "Commit ref (sha, branch, HEAD~N, ...)"
    }
  },
  "required": ["cwd", "ref"]
}`),
		Caps:       tool.CapReadOnly,
		Needs:      tool.NeedExec | tool.NeedFSRead,
		Idempotent: true,
		Timeout:    5 * time.Second,
		Category:   "vcs",
		// As git_log: fixed argv, read-only, so a transient spawn failure is
		// worth one more attempt and nothing else here is.
		Retryable: retryExec,
	}, func(ctx context.Context, in gitShowIn) (string, error) {
		if err := requireAbs("cwd", in.Cwd); err != nil {
			return "", err
		}
		if err := requireSafeRef("ref", in.Ref); err != nil {
			return "", err
		}

		out, err := r.Run(ctx, "git", "-C", in.Cwd, "show", "--no-color", in.Ref)
		if err != nil {
			return "", err
		}
		trimmed := strings.TrimSpace(out)
		if trimmed == "" {
			return "(empty output)", nil
		}
		return truncate(trimmed, maxToolOutput), nil
	})
}
