package wombat

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/hoobean1996/wombat-go/llm"
)

// View is what a [Strategy] gets to reason about.
//
// Not just a message slice, because a message slice is not enough to make a
// good decision. Two tool_result blocks are the same type whether one is nine
// kilobytes of loaded skill and the other is a line of grep output, so a
// strategy handed only messages can trim by position and nothing else. Results
// carries what the harness learned while producing them — which tool ran, what
// it labelled itself with, how big the observation was — and that is what lets
// eviction be a policy rather than an arithmetic accident.
type View struct {
	Messages []llm.Message

	// Results is keyed by tool_use id. Entries exist only for calls this run
	// dispatched: a transcript resumed from disk has messages with no entries,
	// and a strategy must degrade to position-based trimming for those rather
	// than assume.
	Results map[llm.ToolUseID]ResultInfo

	// Tokens estimates the input tokens of a request carrying msgs — the
	// system prompt and tool definitions included — calibrated against what
	// the provider counted for this run's previous calls. Use
	// [View.EstimateTokens], which also works on a View built by hand with
	// Tokens left nil.
	Tokens func(msgs []llm.Message) int

	// Window is the model's context window in tokens, as configured with
	// [WithContextWindow]; 0 when nobody said.
	Window int
}

// EstimateTokens is [View.Tokens], or a plain four-bytes-a-token estimate
// when the View carries no estimator.
func (v View) EstimateTokens(msgs []llm.Message) int {
	if v.Tokens != nil {
		return v.Tokens(msgs)
	}
	return (*tokenEstimator)(nil).estimate(msgs)
}

// ResultInfo is what the harness knows about one dispatched tool call.
type ResultInfo struct {
	Tool  string
	Tags  []string
	Bytes int
}

// HasTag reports whether the call that produced id carried tag.
func (v View) HasTag(id llm.ToolUseID, tag string) bool {
	info, ok := v.Results[id]
	return ok && slices.Contains(info.Tags, tag)
}

// Tagged lists the tool_use ids carrying any of tags, in transcript order.
func (v View) Tagged(tags ...string) []llm.ToolUseID {
	var out []llm.ToolUseID
	for _, m := range v.Messages {
		for _, b := range m.Content {
			tu, ok := b.(llm.ToolUse)
			if !ok {
				continue
			}
			for _, t := range tags {
				if v.HasTag(tu.ID, t) {
					out = append(out, tu.ID)
					break
				}
			}
		}
	}
	return out
}

// Strategy decides which slice of a transcript is materialized for one model
// call. It is the seam between "the conversation the agent has had" and "the
// messages that fit in a context window".
//
// Every implementation must return a transcript that still passes
// [Convo.Validate] — never cutting between a tool_use and its tool_result.
// [SafeCut] does the arithmetic for position-based trimming; [DropPairs] does
// it for surgical removal.
type Strategy interface {
	Apply(View) []llm.Message
	String() string
}

// Flat sends the whole transcript. The default, and correct until a run is
// long enough to matter.
var Flat Strategy = flat{}

type flat struct{}

func (flat) Apply(v View) []llm.Message { return v.Messages }
func (flat) String() string             { return "flat" }

// Sequence applies strategies left to right, each seeing the previous one's
// output. Result metadata is carried through unchanged; entries whose ids have
// been evicted simply stop matching.
func Sequence(ss ...Strategy) Strategy { return seq(ss) }

type seq []Strategy

// Apply hands each strategy the previous one's output in an otherwise
// unchanged View. Rebuilding the View from Messages and Results — which this
// once did — dropped the estimator and the window, and a token strategy inside
// a sequence quietly did nothing.
func (s seq) Apply(v View) []llm.Message {
	for _, st := range s {
		v.Messages = st.Apply(v)
	}
	return v.Messages
}

func (s seq) String() string {
	parts := make([]string, len(s))
	for i, st := range s {
		parts[i] = st.String()
	}
	return "sequence(" + strings.Join(parts, " → ") + ")"
}

// SafeCut returns the largest index <= want at which the transcript can be
// truncated without orphaning a tool_result. It returns 0 when no cut is
// possible, which means "send everything".
//
// A cut is safe only immediately before a user turn carrying no tool_result
// blocks: such a turn is a fresh instruction, not the answer to a tool_use
// that would be left behind.
//
// The search runs BACKWARD from want, so keepRecent is a lower bound on what
// survives — a window may keep more than asked, never less. Searching forward
// would be the other bias, and it is the wrong one twice over: it can drop
// context the caller asked to keep, and when the transcript ends in a long
// tool loop it finds nothing at all and degrades to no truncation, precisely
// when truncation is needed most.
func SafeCut(msgs []llm.Message, want int) int {
	if want <= 0 || len(msgs) == 0 {
		return 0
	}
	if want >= len(msgs) {
		want = len(msgs) - 1
	}
	for i := want; i > 0; i-- {
		m := msgs[i]
		if m.Role != llm.RoleUser {
			continue
		}
		answersATool := false
		for _, b := range m.Content {
			if _, ok := b.(llm.ToolResult); ok {
				answersATool = true
				break
			}
		}
		if !answersATool {
			return i
		}
	}
	return 0
}

// SlidingWindow keeps at least the most recent keepRecent messages, cutting at
// the nearest safe boundary at or before that point. When no safe boundary
// exists the whole transcript is sent: an over-long request is recoverable,
// an orphaned tool_result is a hard provider error.
//
// The cut is recomputed from scratch on every call, so the surviving prefix
// moves forward as the transcript grows. That is the honest reading of "keep
// the most recent N", and it is also why this is the wrong strategy for a long
// run against a provider that caches prompt prefixes — see [SlidingWindowAt],
// which exists to hold the prefix still.
func SlidingWindow(keepRecent int) Strategy {
	return window{trigger: 0, keep: keepRecent}
}

// SlidingWindowAt leaves the transcript alone until it reaches triggerAt
// messages, then trims — but to an ANCHOR that only moves in steps, so the
// surviving prefix stays byte-identical across consecutive calls.
//
// # The anchor, and why it is not just len(msgs)-keepRecent
//
// Both Anthropic and OpenAI cache on a prefix of the request. A cache hit
// requires the leading bytes to be identical to the previous call; one message
// different at the front and the whole prompt is re-read at full price. A
// window that recomputes its cut every turn moves the first surviving message
// forward by one message-pair per turn, which means every single turn of a long
// run is a cache miss. Nothing reports this. It is not an error, the answers are
// correct, and the only symptom is the bill.
//
// So the cut is quantized. With
//
//	delta  = triggerAt - keepRecent
//	anchor = floor((n - keepRecent) / delta) * delta
//
// the anchor is constant for delta consecutive values of n and then jumps by
// delta.
//
// # triggerAt is the ceiling, keepRecent is the floor
//
// Read that off the arithmetic before choosing the numbers, because it is the
// one thing about this function that will surprise you. The kept suffix
// breathes between keepRecent and triggerAt-1 messages rather than sitting at
// exactly keepRecent: the transcript is allowed to grow to triggerAt, gets cut
// back to keepRecent, and grows again. That slack is not a defect, it is the
// mechanism — it is what buys delta cache hits per re-anchor — but it means
// SlidingWindowAt(200, 20) can materialize 199 messages, not 20.
//
// So size triggerAt against the context window and keepRecent against how much
// history the task needs. If you want a hard ceiling near keepRecent, set
// triggerAt just above it and accept fewer cache hits; if you want a stable
// prefix, widen the gap. A plain [SlidingWindow] holds at exactly keepRecent
// and re-anchors every turn, which is the other end of the same trade.
//
// The earlier version of this function had triggerAt mean nothing at all: past
// the first crossing it trimmed on every call, so it was [SlidingWindow] with a
// delayed start, and the ceiling was keepRecent.
//
// [SafeCut] is then applied to the anchor rather than to a fresh count, and it
// composes without argument: it is a pure function of msgs[0:anchor], messages
// are only ever appended, so a stable anchor gives a stable cut. A cut moved
// backward to avoid orphaning a tool_result stays moved to the same place.
//
// This is the closed form the OCaml original uses for the same window,
// with delta one smaller because that implementation trims at n > triggerAt and
// this one trims at n >= triggerAt.
//
// keepRecent >= triggerAt is degenerate — "trim once we reach N, but always keep
// more than N" — and never trims, which is what the contradiction asks for.
func SlidingWindowAt(triggerAt, keepRecent int) Strategy {
	return window{trigger: triggerAt, keep: keepRecent}
}

type window struct {
	trigger int
	keep    int
}

func (w window) Apply(v View) []llm.Message {
	msgs := v.Messages
	if w.keep <= 0 || len(msgs) <= w.keep {
		return msgs
	}

	want := len(msgs) - w.keep
	if w.trigger > 0 {
		if len(msgs) < w.trigger {
			return msgs
		}
		// Quantized so the prefix holds still; see SlidingWindowAt.
		delta := w.trigger - w.keep
		if delta <= 0 {
			return msgs
		}
		want = want / delta * delta
	}

	// want is never above len(msgs)-keep, quantized or not, so the floor on
	// what survives is still keepRecent.
	cut := SafeCut(msgs, want)
	if cut == 0 {
		return msgs
	}
	return msgs[cut:]
}

func (w window) String() string {
	if w.trigger > 0 {
		return fmt.Sprintf("sliding_window_at(trigger=%d, keep=%d)", w.trigger, w.keep)
	}
	return fmt.Sprintf("sliding_window(keep=%d)", w.keep)
}

// DropPairs removes whole tool_use / tool_result pairs for the given ids.
//
// This is the eviction that a flat message list cannot express. Trimming by
// position throws away everything before a line, most of which is probably
// worth keeping; dropping a pair reclaims exactly the nine kilobytes of loaded
// skill and leaves the reasoning around it intact. Both halves go together, so
// the conversation stays valid — which is only possible because content is a
// typed block list and not opaque text.
//
// Messages left with no blocks are removed, and the result is re-checked for
// role alternation.
func DropPairs(msgs []llm.Message, ids map[llm.ToolUseID]bool) []llm.Message {
	if len(ids) == 0 {
		return msgs
	}

	out := make([]llm.Message, 0, len(msgs))
	for _, m := range msgs {
		kept := make([]llm.ContentBlock, 0, len(m.Content))
		for _, b := range m.Content {
			switch v := b.(type) {
			case llm.ToolUse:
				if ids[v.ID] {
					continue
				}
			case llm.ToolResult:
				if ids[v.ToolUseID] {
					continue
				}
			}
			kept = append(kept, b)
		}
		if len(kept) == 0 {
			continue
		}
		out = append(out, llm.Message{Role: m.Role, Content: kept})
	}
	return Convo(out).normalize()
}

// DropTagged evicts every tool call carrying one of tags, except those inside
// the final keepRecent messages, once the transcript reaches triggerAt.
//
// The obvious use is a loaded skill body: it is large, it is identifiable, and
// once the model has acted on it the tokens are dead weight. Pair it with a
// [tool.Reconciler] set so the skill's activation is retired along with its
// body, instead of leaving the model holding tools whose knowledge is gone.
func DropTagged(triggerAt, keepRecent int, tags ...string) Strategy {
	return dropTagged{trigger: triggerAt, keep: keepRecent, tags: tags}
}

type dropTagged struct {
	trigger int
	keep    int
	tags    []string
}

func (d dropTagged) Apply(v View) []llm.Message {
	if len(d.tags) == 0 || len(v.Results) == 0 {
		return v.Messages
	}
	if d.trigger > 0 && len(v.Messages) < d.trigger {
		return v.Messages
	}

	protected := max(len(v.Messages)-d.keep, 0)

	victims := map[llm.ToolUseID]bool{}
	for i, m := range v.Messages {
		if i >= protected {
			break
		}
		for _, b := range m.Content {
			tu, ok := b.(llm.ToolUse)
			if !ok {
				continue
			}
			for _, t := range d.tags {
				if v.HasTag(tu.ID, t) {
					victims[tu.ID] = true
					break
				}
			}
		}
	}
	return DropPairs(v.Messages, victims)
}

func (d dropTagged) String() string {
	return fmt.Sprintf("drop_tagged(trigger=%d, keep=%d, tags=%s)",
		d.trigger, d.keep, strings.Join(d.tags, ","))
}

// Compacted replaces the dropped prefix with a one-turn summary once the
// transcript reaches compactAt messages.
//
// Deprecated: use [WithCompaction]. A Strategy cannot do this job well: it has
// no context, so a model-backed compactor cannot be cancelled or budgeted; it
// cannot fail; and it runs on every call without being able to keep what it
// produced, because the stored transcript is not a strategy's to change. It
// used to show: the compactor ran on every model call over a prefix one turn
// longer each time, and since every summary came out different, the cache
// prefix — which starts with the summary — missed on every call.
//
// Both of those are bounded now. The cut moves in steps of compactAt-keepRecent
// messages, the way [SlidingWindowAt]'s anchor does, and the summary of a given
// prefix is remembered, so the compactor runs once per step and the bytes it
// produced are reused until the next one. What remains is the missing context
// and the missing error, which is why this is deprecated rather than fixed.
func Compacted(compactAt, keepRecent int, compactor func([]llm.Message) string) Strategy {
	return &compacted{at: compactAt, keep: keepRecent, fn: compactor}
}

type compacted struct {
	at   int
	keep int
	fn   func([]llm.Message) string

	mu   sync.Mutex
	memo map[[32]byte]string
}

// compactedMemo bounds the summaries a Compacted strategy remembers. A run
// needs one per step; the bound only matters to a strategy shared by many runs.
const compactedMemo = 64

func (c *compacted) Apply(v View) []llm.Message {
	msgs := v.Messages
	if c.fn == nil || c.keep <= 0 || c.at <= 0 || len(msgs) < c.at || len(msgs) <= c.keep {
		return msgs
	}
	want := len(msgs) - c.keep
	if delta := c.at - c.keep; delta > 0 {
		want = want / delta * delta
	}
	cut := SafeCut(msgs, want)
	if cut == 0 {
		return msgs
	}

	summary := c.summarize(msgs[:cut])
	if summary == "" {
		return msgs[cut:]
	}

	head := llm.UserText("<conversation_summary>\n" + summary + "\n</conversation_summary>")
	return Convo(msgs[cut:]).prepend(head)
}

// summarize returns the remembered summary of prefix, or makes one.
func (c *compacted) summarize(prefix []llm.Message) string {
	raw, err := json.Marshal(prefix)
	if err != nil {
		return c.fn(prefix)
	}
	key := sha256.Sum256(raw)
	c.mu.Lock()
	s, ok := c.memo[key]
	c.mu.Unlock()
	if ok {
		return s
	}
	s = c.fn(prefix)
	c.mu.Lock()
	if c.memo == nil || len(c.memo) >= compactedMemo {
		c.memo = make(map[[32]byte]string)
	}
	c.memo[key] = s
	c.mu.Unlock()
	return s
}

func (c *compacted) String() string {
	return fmt.Sprintf("compacted(at=%d, keep=%d)", c.at, c.keep)
}

// prepend puts m at the front, merging if the first turn shares its role.
func (c Convo) prepend(m llm.Message) []llm.Message {
	if len(c) == 0 {
		return []llm.Message{m}
	}
	if c[0].Role != m.Role {
		return append([]llm.Message{m}, c...)
	}
	merged := make([]llm.ContentBlock, 0, len(m.Content)+len(c[0].Content))
	merged = append(merged, m.Content...)
	merged = append(merged, c[0].Content...)

	out := make([]llm.Message, len(c))
	copy(out, c)
	out[0].Content = merged
	return out
}

// normalize merges adjacent same-role turns and drops empty ones, restoring
// the alternation that surgical block removal can break.
func (c Convo) normalize() []llm.Message {
	out := make([]llm.Message, 0, len(c))
	for _, m := range c {
		if len(m.Content) == 0 {
			continue
		}
		if n := len(out); n > 0 && out[n-1].Role == m.Role {
			merged := make([]llm.ContentBlock, 0, len(out[n-1].Content)+len(m.Content))
			merged = append(merged, out[n-1].Content...)
			merged = append(merged, m.Content...)
			out[n-1].Content = merged
			continue
		}
		out = append(out, m)
	}
	// A transcript must open with a user turn; surgical removal can strip the
	// first one entirely.
	for len(out) > 0 && out[0].Role != llm.RoleUser {
		out = out[1:]
	}
	return out
}
