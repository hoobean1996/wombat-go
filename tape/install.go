package tape

import wombat "github.com/hoobean1996/wombat-go"

// Install records and replays both halves of an agent's run — its model calls
// and its tool calls — as one option:
//
//	tp, _ := tape.Open("run.jsonl", tape.Auto)
//	defer tp.Close()
//	a, _ := wombat.New(wombat.WithClient(client), tp.Install())
//
// The two halves are installed where each has to be. The LLM half wraps the
// client the agent was given, so it sits outside that client's retry layer: a
// replayed call skips retry entirely, and a recorded one is recorded once per
// logical call rather than once per attempt. The tool half joins the tool
// middleware, outside retry, the circuit breaker and truncation, so what it
// records is the verdict the model was shown.
//
// Installing only the LLM half — the easy mistake when the two had to be
// wired separately — replays the model asking for every side effect again,
// and the tools performing them again: a resumed run re-sends the email.
func (t *Tape) Install() wombat.Option {
	return wombat.Options(
		wombat.WithLLMMiddleware(t.LLM()),
		wombat.WithToolMiddleware(t.Tools()),
	)
}
