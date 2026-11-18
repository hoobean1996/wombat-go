package postgres

import (
	"strings"
	"testing"

	"wombat/pkg/core/tool"
)

// A state the harness gains is a migration too: tool_calls takes only the
// states its CHECK names.
func TestTheSchemaKnowsEveryCallState(t *testing.T) {
	all := strings.Join(migrations, "\n")
	for _, st := range tool.States() {
		if !strings.Contains(all, "'"+string(st)+"'") {
			t.Errorf("tool_calls.state does not take %q", st)
		}
	}
}
