package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// ===== view_file =====

type viewFileIn struct {
	Path string `json:"path"`

	// ViewRange is [start, end], 1-indexed and inclusive. A slice rather than
	// a struct because that is the shape the schema advertises and the model
	// emits; anything that is not exactly two elements is ignored, matching
	// the OCaml's tolerant decode.
	ViewRange []int `json:"view_range"`
}

// ViewFile reads a file with line numbers, or lists a directory.
func ViewFile(fsys FS) tool.Def {
	mustNotBeNil(fsys != nil, "ViewFile requires a non-nil FS")

	return tool.Typed(tool.Def{
		Name: "view_file",
		Description: "Read a file and return its contents (with line numbers). Optional " +
			"view_range [start, end] (1-indexed, inclusive). For directories, " +
			"lists entries instead. PATH MUST BE ABSOLUTE (starts with /).",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute file or directory path"
    },
    "view_range": {
      "type": "array",
      "description": "Optional [start, end] line numbers, 1-indexed",
      "items": { "type": "integer" }
    }
  },
  "required": ["path"]
}`),
		Caps:       tool.CapReadOnly,
		Needs:      tool.NeedFSRead,
		Idempotent: true,
		Timeout:    5 * time.Second,
		Category:   "file_io",
		// Reads go through the FS interface rather than a subprocess, so the
		// transient set is retryFS, not retryExec. Replaying a read is free:
		// fixed path, no effect, and the second attempt costs one syscall.
		Retryable: retryFS,
	}, func(ctx context.Context, in viewFileIn) (string, error) {
		if err := requireAbs("path", in.Path); err != nil {
			return "", err
		}

		kind, err := fsys.Stat(ctx, in.Path)
		if err != nil {
			return "", err
		}
		switch kind {
		case KindMissing:
			return "", tool.CallerError(fmt.Errorf("no such file or directory: %s", in.Path))

		case KindDir:
			entries, err := fsys.ListDir(ctx, in.Path)
			if err != nil {
				return "", err
			}
			var b strings.Builder
			fmt.Fprintf(&b, "Directory: %s", in.Path)
			for _, e := range entries {
				b.WriteString("\n  ")
				b.WriteString(e)
			}
			return truncate(b.String(), maxToolOutput), nil

		default:
			data, err := fsys.ReadFile(ctx, in.Path)
			if err != nil {
				return "", err
			}
			return numberLines(string(data), in.ViewRange), nil
		}
	})
}

// numberLines renders content as "%5d\t%s" rows, restricted to the requested
// range.
//
// A single trailing newline is dropped before splitting. The OCaml did not do
// this, so "foo\n" was reported as two lines with an empty second one, and the
// model would routinely ask to edit a line that does not exist. One trailing
// newline is a line terminator, not a line.
func numberLines(content string, viewRange []int) string {
	content = strings.TrimSuffix(content, "\n")
	lines := strings.Split(content, "\n")
	total := len(lines)

	start, end := 1, total
	if len(viewRange) == 2 {
		start, end = viewRange[0], viewRange[1]
	}
	if start < 1 {
		start = 1
	}
	if end > total {
		end = total
	}

	var b strings.Builder
	for i := start; i <= end; i++ {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		fmt.Fprintf(&b, "%5d\t%s", i, lines[i-1])
	}
	return truncate(b.String(), maxToolOutput)
}

// ===== write_file =====

type writeFileIn struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// WriteFile creates or overwrites a file.
func WriteFile(fsys FS) tool.Def {
	mustNotBeNil(fsys != nil, "WriteFile requires a non-nil FS")

	return tool.Typed(tool.Def{
		Name: "write_file",
		Description: "Create or overwrite a file with the given content. Returns success " +
			"message with byte count. PATH MUST BE ABSOLUTE (starts with /).",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute path to file"
    },
    "content": {
      "type": "string",
      "description": "Full file content"
    }
  },
  "required": ["path", "content"]
}`),
		Caps:  tool.CapMutating,
		Needs: tool.NeedFSWrite,
		// Not idempotent: overwrites file content. A retry after a timeout
		// could clobber a file some other step has since edited.
		Idempotent: false,
		Timeout:    10 * time.Second,
		Category:   "file_io",
	}, func(ctx context.Context, in writeFileIn) (string, error) {
		if err := requireAbs("path", in.Path); err != nil {
			return "", err
		}
		if err := fsys.WriteFile(ctx, in.Path, []byte(in.Content)); err != nil {
			return "", err
		}
		return fmt.Sprintf("wrote %d bytes to %s", len(in.Content), in.Path), nil
	})
}

// ===== str_replace =====

type strReplaceIn struct {
	Path   string `json:"path"`
	OldStr string `json:"old_str"`
	NewStr string `json:"new_str"`
}

// StrReplace edits a file by unique exact-string substitution.
func StrReplace(fsys FS) tool.Def {
	mustNotBeNil(fsys != nil, "StrReplace requires a non-nil FS")

	return tool.Typed(tool.Def{
		Name: "str_replace",
		Description: "Edit a file by replacing an exact string with a new one. The old_str " +
			"must match exactly once in the file (whitespace-sensitive). " +
			"PATH MUST BE ABSOLUTE (starts with /).",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute path to file"
    },
    "old_str": {
      "type": "string",
      "description": "Exact string to replace (must occur once)"
    },
    "new_str": {
      "type": "string",
      "description": "Replacement string"
    }
  },
  "required": ["path", "old_str", "new_str"]
}`),
		Caps:  tool.CapMutating,
		Needs: tool.NeedFSRead | tool.NeedFSWrite,
		// Not idempotent: a replayed edit finds no match and reports failure,
		// which would read to the model as "my edit was lost".
		Idempotent: false,
		Timeout:    10 * time.Second,
		Category:   "file_io",
	}, func(ctx context.Context, in strReplaceIn) (string, error) {
		if err := requireAbs("path", in.Path); err != nil {
			return "", err
		}

		data, err := fsys.ReadFile(ctx, in.Path)
		if err != nil {
			return "", err
		}
		content := string(data)

		// The uniqueness requirement is the whole safety story of this tool:
		// an ambiguous old_str means the model does not actually know which
		// site it is editing, and guessing is worse than refusing. An empty
		// old_str would match everywhere, so it counts as no match at all.
		n := 0
		if in.OldStr != "" {
			n = strings.Count(content, in.OldStr)
		}
		switch {
		case n == 0:
			return "", tool.CallerError(fmt.Errorf("old_str not found in %s", in.Path))
		case n > 1:
			return "", tool.CallerError(fmt.Errorf("old_str matches %d times in %s — must be unique", n, in.Path))
		}

		updated := strings.Replace(content, in.OldStr, in.NewStr, 1)
		if err := fsys.WriteFile(ctx, in.Path, []byte(updated)); err != nil {
			return "", err
		}
		return fmt.Sprintf("replaced 1 occurrence in %s (%d → %d bytes)",
			in.Path, len(content), len(updated)), nil
	})
}

// ===== save_tool_result =====

// ErrNoTranscript is returned by save_tool_result when it runs without an
// agent loop to read the transcript from.
var ErrNoTranscript = errors.New("builtin: no transcript lookup in this context")

type saveToolResultIn struct {
	ToolUseID string `json:"tool_use_id"`
	Path      string `json:"path"`
}

// SaveToolResult writes a previous tool's output to disk by id.
//
// It exists to dodge an output-token bill. Saving a 5 KB JSON blob with
// write_file forces the model to re-emit all 5 KB inside the content
// argument; four such calls in one assistant message routinely blow
// max_tokens. Naming the tool_use_id costs a dozen tokens instead, and the
// bytes never leave the harness.
//
// The transcript arrives through tool.LookupFrom(ctx) rather than through a
// constructor argument because it is per-run state, not a dependency: the
// same Def serves every run of the agent. Outside an agent loop the lookup is
// nil and the tool reports [ErrNoTranscript].
func SaveToolResult(fsys FS) tool.Def {
	mustNotBeNil(fsys != nil, "SaveToolResult requires a non-nil FS")

	return tool.Typed(tool.Def{
		Name: "save_tool_result",
		Description: "Save a previous tool's full result to disk WITHOUT re-emitting " +
			"the content in your output. Pass `tool_use_id` (the id of a " +
			"previous successful tool call, e.g. \"toolu_01ABC...\" — " +
			"visible as `toolu_…` in tool_use blocks in the conversation) " +
			"and `path` (absolute). Use this INSTEAD of `write_file` when " +
			"saving large tool outputs (finance data, search results, " +
			"fetched files) — `write_file` makes you regenerate the data " +
			"in your output, which costs max_tokens budget and frequently " +
			"blows the limit on parallel saves. PATH MUST BE ABSOLUTE.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "tool_use_id": {
      "type": "string",
      "description": "ID of a prior tool_use whose result you want saved (e.g. \"toolu_01ABC...\")."
    },
    "path": {
      "type": "string",
      "description": "Absolute path to write the result to."
    }
  },
  "required": ["tool_use_id", "path"]
}`),
		Caps:  tool.CapMutating,
		Needs: tool.NeedFSWrite,
		// Not idempotent for write_file's reason: it ends in the same
		// overwriting fsys.WriteFile, and a replay after a timeout can clobber
		// a file a later step has since edited.
		Idempotent: false,
		// Same budget as write_file, because it is the same durable write —
		// temp file, fsync, rename, fsync the parent. The old 5s meant an
		// identical payload to an identical path could time out here and
		// succeed there, purely by which tool the model happened to pick.
		Timeout:  10 * time.Second,
		Category: "file_io",
	}, func(ctx context.Context, in saveToolResultIn) (string, error) {
		if in.ToolUseID == "" {
			return "", tool.CallerError(errors.New("field 'tool_use_id' must not be empty"))
		}
		if err := requireAbs("path", in.Path); err != nil {
			return "", err
		}

		lookup := tool.LookupFrom(ctx)
		if lookup == nil {
			return "", fmt.Errorf("%w: save_tool_result only works inside an agent loop, use write_file instead", ErrNoTranscript)
		}

		content, err := lookup(llm.ToolUseID(in.ToolUseID))
		if err != nil {
			return "", fmt.Errorf("tool_result %s: %w", in.ToolUseID, err)
		}
		if err := fsys.WriteFile(ctx, in.Path, []byte(content)); err != nil {
			return "", err
		}
		return fmt.Sprintf("saved %d bytes from tool_result %s to %s (no LLM regen)",
			len(content), in.ToolUseID, in.Path), nil
	})
}
