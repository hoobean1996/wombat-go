package openai

import (
	"testing"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/llm/internal/providertest"
)

// TestConformance runs the behaviour every provider client owes the harness
// at the HTTP boundary — stalls, cancellations, error bodies, oversized
// replies — against this client.
func TestConformance(t *testing.T) {
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
		StreamStart: "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
		StreamFinish: "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n",
		Reply:   minimalReply,
		Partial: `{"model":"m",`,
	})
}
