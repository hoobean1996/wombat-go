package daemon

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// ===== The ledger =====

// ToolCall is one tool call a session's turn made, as the ledger keeps it:
// what the model asked for, where it is, and what came of it. Its states are
// [tool.State]'s.
type ToolCall struct {
	Session string `json:"session_id"`

	// Step and Index say which call it is: the Index-th call of the session's
	// Step-th model reply. Steps are numbered across the session, so a pair
	// names one call for the session's life — a tool_use id is the
	// provider's, and some gateways reuse them.
	Step  int `json:"step"`
	Index int `json:"index"`

	// Turn is the turn that made it.
	Turn int `json:"turn"`

	// Parent is the key of the call whose sub-agent made this one, and Run
	// which of that call's sub-agents; empty for one of the session's own.
	Parent string `json:"parent,omitempty"`
	Run    int    `json:"run,omitempty"`

	UseID llm.ToolUseID   `json:"use_id"`
	Tool  string          `json:"tool"`
	Input json.RawMessage `json:"input"`

	State tool.State `json:"state"`

	// Attempts is how many times it started to run.
	Attempts int `json:"attempts"`

	// Output is what the model was shown for it, once there was something,
	// and IsError whether that was an error; Media and Tags went with it.
	Output  string      `json:"output,omitempty"`
	IsError bool        `json:"is_error,omitempty"`
	Media   []llm.Media `json:"media,omitempty"`
	Tags    []string    `json:"tags,omitempty"`

	// Reason says why it waits — for a decision, for a result — or why it
	// did not run; Handle names what a waiting call waits for.
	Reason string `json:"reason,omitempty"`
	Handle string `json:"handle,omitempty"`

	// Decision is what a person decided about it, when one did.
	Decision *Decision `json:"decision,omitempty"`

	// Owner is the daemon that last moved it.
	Owner string `json:"owner,omitempty"`

	Created  time.Time `json:"created"`
	Started  time.Time `json:"started,omitzero"`
	Finished time.Time `json:"finished,omitzero"`
}

// Key names the call for its tool — see [tool.CallKey] — and for a sub-agent
// it starts.
func (c ToolCall) Key() string { return CallKey(c.Session, c.Step, c.Index) }

// CallKey names the index-th call of a session's step: unique for the
// session's life, and the same however often the call runs.
func CallKey(session string, step, index int) string {
	return fmt.Sprintf("%s/%d.%d", session, step, index)
}

// Decision is what a person decided about a call: to run it or not, and who.
type Decision struct {
	Allow bool   `json:"allow"`
	By    Origin `json:"by"`
}

// StepOpen is a model reply written down before anything it asks for
// happens; see [Store.OpenStep].
type StepOpen struct {
	Session string

	// Owner and Epoch fence the write: it is made only while that daemon
	// holds the session, under that lease.
	Owner string
	Epoch int64

	Turn int

	// Parent and Run are, for a step a sub-agent took, the call that
	// started it and which of that call's sub-agents it is.
	Parent string
	Run    int

	// Calls are the reply's calls, pending; the store numbers their Step.
	Calls []ToolCall

	// Inflight is where the session's own turn had got to, for one of its
	// steps, and Sub where the sub-agent had got to, for one of a
	// sub-agent's. The store numbers their Step too.
	Inflight *Inflight
	Sub      *SubRun

	At time.Time
}

// CallMoves is calls of one step changing state; see [Store.MoveCalls].
type CallMoves struct {
	Session string
	Owner   string
	Epoch   int64
	Step    int
	Moves   []CallMove
	At      time.Time
}

// CallMove is one call of a step changing state: what it is now, and what it
// came to, when anything did.
type CallMove struct {
	Index int
	State tool.State

	// Attempt counts one more start: the call is about to run.
	Attempt bool

	Output   string
	IsError  bool
	Media    []llm.Media
	Tags     []string
	Reason   string
	Handle   string
	Decision *Decision
}

// CallFilter selects calls from the ledger. A zero field selects everything.
type CallFilter struct {
	Session string

	// Step selects one step's calls, Turn one turn's.
	Step int
	Turn int

	// Parent selects a sub-agent's calls by the call that started it; "-"
	// selects only the session's own.
	Parent string

	States []tool.State
	Limit  int
}

// Inflight is a turn that has not settled, as its journal last wrote it
// down: what a daemon that takes the session over carries the turn on from,
// instead of starting it again.
type Inflight struct {
	Turn int `json:"turn"`

	// Step is the step the transcript ends on: the model's reply, whose
	// calls the ledger has.
	Step int `json:"step"`

	// Base is how many messages of the session's transcript Messages carries
	// on from; -1 when Messages is the whole of it, because the turn
	// compacted what came before.
	Base     int           `json:"base"`
	Messages []llm.Message `json:"messages"`

	// Iterations is how many iterations the turn had made.
	Iterations int `json:"iterations"`

	// Events are the inbox events the turn is for, and ReplyTo the messages
	// its answer replies to.
	Events  []int64 `json:"events"`
	ReplyTo []int64 `json:"reply_to,omitempty"`

	// At, Origins and Folds are the turn's provenance: the message its
	// events made and where they came from, and the compactions it made.
	At      int      `json:"at"`
	Origins []Origin `json:"origins,omitempty"`
	Folds   []int    `json:"folds,omitempty"`

	// Used is what the turn had used of its budget, and Counted how much of
	// that the session's Spend already holds — from a run of the turn that
	// stopped and was counted, rather than died.
	Used    governor.Used `json:"used"`
	Counted Spend         `json:"counted"`

	// State is what the run keeps besides its transcript (core.Checkpoint.
	// State), and Suspended says the run stopped on purpose at the end of a
	// slice, to carry on in the next, rather than died.
	State     json.RawMessage `json:"state,omitempty"`
	Suspended bool            `json:"suspended,omitempty"`
}

// Transcript is the turn's transcript, given the session's: what it carries
// on from, and what the turn added.
func (f *Inflight) Transcript(session []llm.Message) []llm.Message {
	if f.Base < 0 || f.Base > len(session) {
		return slices.Clone(f.Messages)
	}
	return slices.Concat(session[:f.Base], f.Messages)
}

// SubRun is where a sub-agent a tool call started had got to, as its
// journal last wrote it down: what the call, running again after its turn
// was resumed, carries the sub-agent on from.
type SubRun struct {
	Session string `json:"session_id"`
	Parent  string `json:"parent"`
	Run     int    `json:"run"`

	// Step is the step its transcript ends on.
	Step       int           `json:"step"`
	Messages   []llm.Message `json:"messages"`
	Iterations int           `json:"iterations"`
	Updated    time.Time     `json:"updated"`

	// State is what the sub-agent's run keeps besides its transcript.
	State json.RawMessage `json:"state,omitempty"`
}

// Progress is where a run had got to when it stopped at the end of its
// slice; see [Store.SaveProgress].
type Progress struct {
	Session string
	Owner   string
	Epoch   int64

	// Inflight is the session's own turn; or Sub, with Parent and Run, a
	// sub-agent's.
	Inflight *Inflight
	Parent   string
	Run      int
	Sub      *SubRun

	At time.Time
}

// EncodeInflight and DecodeInflight are an [Inflight] as a store keeps it:
// nil for none. For a [Store] implementation.
func EncodeInflight(f *Inflight) ([]byte, error) {
	if f == nil {
		return nil, nil
	}
	return json.Marshal(f)
}

// DecodeInflight is the other half of [EncodeInflight].
func DecodeInflight(b []byte) (*Inflight, error) {
	if len(b) == 0 || string(b) == "null" {
		return nil, nil
	}
	var f Inflight
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("daemon: decoding a turn in flight: %w", err)
	}
	return &f, nil
}

// ===== Between the ledger and the harness =====

// callOf is a ledger call as a journal hands it back to a run it resumes.
func callOf(c ToolCall) core.Call {
	out := core.Call{
		Index:  c.Index,
		Use:    llm.ToolUse{ID: c.UseID, Name: c.Tool, Input: c.Input},
		State:  c.State,
		Tags:   c.Tags,
		Reason: c.Reason,
		Handle: c.Handle,
	}
	if c.State.Final() || (c.State == tool.StateInterrupted && (c.Output != "" || c.IsError)) {
		out.Result = &llm.ToolResult{ToolUseID: c.UseID, Content: c.Output, IsError: c.IsError, Media: c.Media}
	}
	return out
}

// callsOf is a step's calls as a run resumed on it is given them.
func callsOf(calls []ToolCall) []core.Call {
	out := make([]core.Call, 0, len(calls))
	for _, c := range calls {
		out = append(out, callOf(c))
	}
	slices.SortFunc(out, func(a, b core.Call) int { return a.Index - b.Index })
	return out
}

// moveOf is a call a run reports as a ledger move.
func moveOf(c core.Call) CallMove {
	m := CallMove{Index: c.Index, State: c.State, Reason: clean(c.Reason), Handle: clean(c.Handle)}
	for _, t := range c.Tags {
		m.Tags = append(m.Tags, clean(t))
	}
	if c.Result != nil {
		m.Output, m.IsError, m.Media = clean(c.Result.Content), c.Result.IsError, c.Result.Media
	}
	return m
}

// clean is text as the ledger keeps it: valid UTF-8, without NULs. A tool's
// output need be neither — a binary file read is not — and Postgres's text
// takes neither.
func clean(s string) string {
	if utf8.ValidString(s) && !strings.ContainsRune(s, 0) {
		return s
	}
	return strings.ReplaceAll(strings.ToValidUTF8(s, "\uFFFD"), "\x00", "\uFFFD")
}

// moveable reports whether a call in state from may be moved to to: a call
// that has ended only to another ending.
func moveable(from, to tool.State) bool {
	return !from.Final() || to.Final()
}

// ApplyMove is m applied to c at at, as both stores apply it. For a [Store]
// implementation.
func ApplyMove(c *ToolCall, m CallMove, owner string, at time.Time) {
	c.State, c.Owner = m.State, owner
	c.Output, c.IsError, c.Media, c.Tags = m.Output, m.IsError, m.Media, m.Tags
	c.Reason, c.Handle = m.Reason, m.Handle
	if m.Decision != nil {
		c.Decision = m.Decision
	}
	if m.Attempt {
		c.Attempts++
		c.Started = at
		c.Finished = time.Time{}
	}
	if m.State.Final() && c.Finished.IsZero() {
		c.Finished = at
	}
}

// CheckMove reports whether m may be applied to c. For a [Store]
// implementation.
func CheckMove(c ToolCall, m CallMove) error {
	if !m.State.Valid() {
		return fmt.Errorf("daemon: no such call state %q", m.State)
	}
	if !moveable(c.State, m.State) {
		return fmt.Errorf("daemon: call %s is %s, and cannot become %s", c.Key(), c.State, m.State)
	}
	return nil
}
