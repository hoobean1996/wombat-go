// Package shape measures the mechanisms that decide what a request LOOKS like,
// without calling a model.
//
// A transcript strategy, a gated tool set and the system-prompt builder all do
// their work before the request leaves the process, so their whole effect is
// visible in the [llm.Request] the harness assembles. That makes them testable
// the cheap way: script the replies, run the loop, and look at what it built.
//
// This lane exists because the expensive lane cannot see these mechanisms at
// all. A window that keeps the prompt prefix byte-stable does not move a task
// score — its only symptom is the bill — so a benchmark measuring pass@k is
// blind to it by construction, and would stay blind through a regression that
// doubled the cost of every long run.
//
// # The prefix hash
//
// Both major providers cache on a PREFIX of the request: the leading bytes must
// be identical to the previous call or the whole prompt is re-read at full
// price. [Shot.Prefix] is the running hash of the materialized messages, so the
// length of the longest common prefix between consecutive turns is exactly how
// far a cache could have hit — computed here from the request itself, with no
// provider, no key and no reliance on a usage field the server may not send.
package shape

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/hoobean1996/wombat-go/llm"
)

// Shot is one assembled request, reduced to the parts a mechanism controls.
//
// Deliberately not the request itself: raw content makes a golden file that
// churns on every prompt edit, and the question here is never "what did it
// say" but "what shape did the harness build".
type Shot struct {
	// Turn is the model call's 1-based ordinal.
	Turn int

	// SystemSHA fingerprints the rendered system prompt. It must not move
	// between turns of one run: the system block is the head of the cache
	// prefix, so a single byte of drift there costs every later hit.
	SystemSHA string
	SystemLen int

	// Tools are the visible tool names, in the order they were offered. Both
	// the set AND the order matter — the list is serialized into every
	// request, so a set that permutes itself invalidates the prefix just as
	// surely as one that changes.
	Tools []string

	// Messages is one entry per materialized message.
	Messages []MsgShape

	// Prefix[i] is the hash of messages[0..i], so Prefix is a running
	// fingerprint of every proper prefix of the transcript.
	Prefix []string

	Forced string
}

// MsgShape is one message, reduced to role, block kinds and size.
type MsgShape struct {
	Role   string   `json:"role"`
	Blocks []string `json:"blocks"`
	Bytes  int      `json:"bytes"`
	SHA    string   `json:"sha"`
}

func sha(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:8])
}

// Of reduces one request to a Shot.
func Of(turn int, req llm.Request) Shot {
	s := Shot{
		Turn:      turn,
		SystemSHA: sha([]byte(req.System)),
		SystemLen: len(req.System),
		Forced:    req.Choice.Name,
	}
	for _, t := range req.Tools {
		s.Tools = append(s.Tools, t.Name)
	}

	// The running hash folds the previous hash in, so Prefix[i] depends on
	// every message up to i and on their ORDER — which is what a provider's
	// prefix cache is keyed on too.
	running := sha256.New()
	for _, m := range req.Messages {
		raw, err := json.Marshal(m)
		if err != nil {
			// A message that will not marshal cannot be sent either; record
			// the failure in the shape rather than panicking in a measurement.
			raw = []byte(fmt.Sprintf("<unmarshalable: %v>", err))
		}
		ms := MsgShape{Role: string(m.Role), Bytes: len(raw), SHA: sha(raw)}
		for _, b := range m.Content {
			ms.Blocks = append(ms.Blocks, blockKind(b))
		}
		s.Messages = append(s.Messages, ms)

		running.Write(raw)
		s.Prefix = append(s.Prefix, hex.EncodeToString(running.Sum(nil)[:8]))
	}
	return s
}

// blockKind names a content block's type.
//
// A type switch rather than llm's own blockKind(), which is unexported on
// purpose — the interface is closed to implementations outside that package,
// and this measurement has no business widening it. The default arm is not
// dead code: a block type added to llm would land there, and "unknown" in a
// golden file is a visible prompt to come back here.
func blockKind(b llm.ContentBlock) string {
	switch b.(type) {
	case llm.Text:
		return "text"
	case llm.ToolUse:
		return "tool_use"
	case llm.ToolResult:
		return "tool_result"
	case llm.Thinking:
		return "thinking"
	default:
		return "unknown"
	}
}

// TotalBytes is the size of the materialized transcript.
func (s Shot) TotalBytes() int {
	n := 0
	for _, m := range s.Messages {
		n += m.Bytes
	}
	return n
}

// SharedPrefix reports how many leading messages a and b have in common.
//
// This is the cache boundary: a provider that caches on a prefix can reuse
// exactly this many messages and must re-read everything after them.
func SharedPrefix(a, b Shot) int {
	n := min(len(a.Prefix), len(b.Prefix))
	for i := range n {
		if a.Prefix[i] != b.Prefix[i] {
			return i
		}
	}
	return n
}
