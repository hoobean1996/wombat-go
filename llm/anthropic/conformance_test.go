package anthropic

import (
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/internal/providertest"
)

// TestConformance runs the behaviour every provider client owes the harness
// at the HTTP boundary — stalls, cancellations, error bodies, oversized
// replies — against this client.
func TestConformance(t *testing.T) {
	frame := func(typ, data string) string { return "event: " + typ + "\ndata: " + data + "\n\n" }
	providertest.Run(t, providertest.Wire{
		New: func(t *testing.T, baseURL string, l providertest.Limits) llm.Client {
			clearEnv(t)
			c, err := New(Config{APIKey: "k", BaseURL: baseURL, Model: "m"})
			if err != nil {
				t.Fatal(err)
			}
			c.idle, c.total, c.maxBody = l.Idle, l.Total, l.MaxBody
			return c
		},
		StreamStart: strings.Join([]string{
			frame("message_start", `{"type":"message_start","message":{"model":"m","usage":{"input_tokens":1}}}`),
			frame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
			frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}`),
		}, ""),
		StreamFinish: strings.Join([]string{
			frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`),
			frame("content_block_stop", `{"type":"content_block_stop","index":0}`),
			frame("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`),
			frame("message_stop", `{"type":"message_stop"}`),
		}, ""),
		Reply:   `{"model":"m","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
		Partial: `{"model":"m",`,
	})
}
