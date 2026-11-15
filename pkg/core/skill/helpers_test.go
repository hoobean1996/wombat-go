// Tests for package skill are white-box (package skill) rather than
// skill_test: most of what matters here is the exported surface, but the
// frontmatter subset is implemented in three unexported helpers
// (splitFrontmatter, readBlock, unquote) whose edge cases — a folded block, a
// tab-indented continuation, a value that is only quotes — are far cheaper to
// pin directly than to reconstruct through Parse. Being in the package also
// lets the load/unload tests reach r.loadTool() without standing up a Bind.
package skill

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// mkSkill writes <root>/<dir>/SKILL.md.
func mkSkill(t *testing.T, root, dir, content string) string {
	t.Helper()
	full := filepath.Join(root, dir)
	if err := os.MkdirAll(full, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", full, err)
	}
	path := filepath.Join(full, FileName)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(%s): %v", path, err)
	}
	return path
}

// frontmatter assembles a SKILL.md from a frontmatter block and a body.
func frontmatter(meta, body string) string {
	return delim + "\n" + meta + "\n" + delim + "\n" + body
}

const demoBody = "MARKER-SKILL-BODY-9f3a\n\nUse the secret tool."

func demoSkill() Skill {
	return Skill{
		Name:        "demo",
		Description: "A demo skill gating the secret tool.",
		Body:        demoBody,
	}
}

// echoTool is an ordinary, ungated tool.
func echoTool(name string) tool.Def {
	return tool.Def{
		Name:        name,
		Description: name,
		InputSchema: rawObject,
		Fn:          func(context.Context, json.RawMessage) (string, error) { return name + "-ran", nil },
	}
}

var rawObject = []byte(`{"type":"object"}`)

// secretTool is the tool the demo skill gates.
func secretTool() tool.Def {
	d := echoTool("secret")
	d.Description = "gated"
	return d
}

// names extracts and sorts tool names.
func names(defs []tool.Def) []string {
	out := make([]string, 0, len(defs))
	for _, d := range defs {
		out = append(out, d.Name)
	}
	sort.Strings(out)
	return out
}

// orderedNames extracts tool names in the order given.
func orderedNames(defs []tool.Def) []string {
	out := make([]string, 0, len(defs))
	for _, d := range defs {
		out = append(out, d.Name)
	}
	return out
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

// dispatch1 runs one tool_use through h and returns the single result.
func dispatch1(t *testing.T, ctx context.Context, h tool.Handler, set tool.Set, use llm.ToolUse) tool.Result {
	t.Helper()
	rs := tool.NewDispatcher(h).Dispatch(ctx, set, []llm.ToolUse{use})
	if len(rs) != 1 {
		t.Fatalf("Dispatch returned %d results, want 1", len(rs))
	}
	return rs[0]
}

// loadUse builds a load_skill call.
func loadUse(id, name string) llm.ToolUse {
	return llm.ToolUse{
		ID:    llm.ToolUseID(id),
		Name:  LoadToolName,
		Input: []byte(fmt.Sprintf(`{"name":%q}`, name)),
	}
}

// unloadUse builds an unload_skill call.
func unloadUse(id, name string) llm.ToolUse {
	return llm.ToolUse{
		ID:    llm.ToolUseID(id),
		Name:  UnloadToolName,
		Input: []byte(fmt.Sprintf(`{"name":%q}`, name)),
	}
}

// mustPanic asserts fn panics with a message containing want.
func mustPanic(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("no panic, want one containing %q", want)
			return
		}
		if msg := fmt.Sprint(r); !strings.Contains(msg, want) {
			t.Errorf("panic = %q, want it to contain %q", msg, want)
		}
	}()
	fn()
}
