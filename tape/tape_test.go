// White-box (package tape): several of these assertions are about the on-disk
// format and about state the package deliberately does not export — the
// latched write error, the per-key seq counter, the "loaded is the only replay
// source" rule. Reaching for them through the public API only would test less
// and say less about why.
package tape

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// ===== helpers =====

// liveClient answers from a script and counts how many times it was reached.
// A replayed run must never reach it.
type liveClient struct {
	mu    sync.Mutex
	turns []llm.Response
	calls int
	seen  []llm.Request

	// stream, when set, is pushed to req.OnDelta before answering, so the
	// recorded entry is marked as having streamed.
	stream string
}

func (c *liveClient) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	c.mu.Lock()
	i := c.calls
	c.calls++
	c.seen = append(c.seen, req)
	c.mu.Unlock()

	if c.stream != "" && req.OnDelta != nil {
		req.OnDelta(llm.Delta{Text: c.stream})
	}
	if i >= len(c.turns) {
		return llm.Response{}, fmt.Errorf("liveClient: out of turns (call %d)", i+1)
	}
	return c.turns[i], nil
}

func (c *liveClient) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

func textResp(s string) llm.Response {
	return llm.Response{
		Content:    []llm.ContentBlock{llm.Text{Text: s}},
		StopReason: llm.StopEndTurn,
		Usage:      llm.Usage{InputTokens: 11, OutputTokens: 3},
		Model:      "test-model",
	}
}

func askReq(text string) llm.Request {
	return llm.Request{
		Model:     "test-model",
		System:    "be terse",
		Messages:  []llm.Message{llm.UserText(text)},
		MaxTokens: 256,
	}
}

// openTape opens a tape and closes it at the end of the test.
func openTape(t *testing.T, path string, mode Mode) *Tape {
	t.Helper()
	tp, err := Open(path, mode)
	if err != nil {
		t.Fatalf("Open(%q, %s): got error %v, want nil", path, mode, err)
	}
	t.Cleanup(func() { _ = tp.Close() })
	return tp
}

// tapePath returns a fresh path inside the test's temp dir.
func tapePath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "run.jsonl")
}

// wireEntry mirrors the documented line shape WITHOUT reusing the package's
// own struct, so a rename or a reordering in codec.go shows up here as a
// failure instead of being papered over.
type wireEntry struct {
	Kind     string          `json:"kind"`
	Key      string          `json:"key"`
	Seq      int             `json:"seq"`
	Request  json.RawMessage `json:"request"`
	Response json.RawMessage `json:"response"`
}

func readLines(t *testing.T, path string) []wireEntry {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%q): got error %v, want nil", path, err)
	}
	var out []wireEntry
	for i, ln := range strings.Split(strings.TrimRight(string(raw), "\n"), "\n") {
		if ln == "" {
			continue
		}
		var e wireEntry
		if err := json.Unmarshal([]byte(ln), &e); err != nil {
			t.Fatalf("line %d is not a tape entry: %v\nline: %s", i+1, err, ln)
		}
		out = append(out, e)
	}
	return out
}

// bumpTool is a tool with a side effect the replay must not repeat.
func bumpTool(hits *int) tool.Def {
	return tool.Def{
		Name:        "bump",
		Description: "increments a counter",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn: func(context.Context, json.RawMessage) (string, error) {
			*hits++
			return "bumped", nil
		},
	}
}

// ===== record then replay =====

func TestRecordThenReplay(t *testing.T) {
	path := tapePath(t)

	// --- record ---
	rec := openTape(t, path, Record)
	live := &liveClient{turns: []llm.Response{textResp("first"), textResp("second")}}
	cl := llm.Chain(live, rec.LLM())

	sideEffects := 0
	th := rec.Tools()(tool.Direct)
	def := bumpTool(&sideEffects)

	for _, q := range []string{"a", "b"} {
		if _, err := cl.Complete(context.Background(), askReq(q)); err != nil {
			t.Fatalf("record Complete(%q): got error %v, want nil", q, err)
		}
	}
	out, err := th(context.Background(), def, llm.ToolUse{ID: "u1", Name: "bump", Input: json.RawMessage(`{"n":1}`)})
	if err != nil || out != "bumped" {
		t.Fatalf("record tool: got (%q, %v), want (\"bumped\", nil)", out, err)
	}
	if live.count() != 2 {
		t.Errorf("record llm calls: got %d, want 2", live.count())
	}
	if sideEffects != 1 {
		t.Errorf("record side effects: got %d, want 1", sideEffects)
	}
	if got := rec.Stats(); got.Recorded != 3 {
		t.Errorf("record Stats().Recorded: got %d, want 3", got.Recorded)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("record Close: got error %v, want nil", err)
	}

	// --- replay ---
	rp := openTape(t, path, Replay)
	dead := &liveClient{} // no turns: any live call is an error
	rcl := llm.Chain(dead, rp.LLM())
	rth := rp.Tools()(tool.Direct)

	for i, q := range []string{"a", "b"} {
		resp, err := rcl.Complete(context.Background(), askReq(q))
		if err != nil {
			t.Fatalf("replay Complete(%q): got error %v, want nil", q, err)
		}
		want := []string{"first", "second"}[i]
		if got := llm.TextOf(resp.Content); got != want {
			t.Errorf("replay text for %q: got %q, want %q", q, got, want)
		}
		if resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 3 {
			t.Errorf("replay usage: got %+v, want {11 3}", resp.Usage)
		}
		if resp.StopReason != llm.StopEndTurn {
			t.Errorf("replay stop reason: got %q, want %q", resp.StopReason, llm.StopEndTurn)
		}
		if resp.Model != "test-model" {
			t.Errorf("replay model: got %q, want %q", resp.Model, "test-model")
		}
	}
	rout, err := rth(context.Background(), def, llm.ToolUse{ID: "u1", Name: "bump", Input: json.RawMessage(`{"n":1}`)})
	if err != nil || rout != "bumped" {
		t.Fatalf("replay tool: got (%q, %v), want (\"bumped\", nil)", rout, err)
	}

	if dead.count() != 0 {
		t.Errorf("replay made live llm calls: got %d, want 0", dead.count())
	}
	if sideEffects != 1 {
		t.Errorf("replay re-ran the side effect: got %d bumps, want 1", sideEffects)
	}
	st := rp.Stats()
	if st.Hits != 3 || st.Misses != 0 || st.Recorded != 0 {
		t.Errorf("replay Stats: got %+v, want {Hits:3 Misses:0 Recorded:0}", st)
	}

	// A replay tape holds no writable handle at all, so it cannot corrupt the
	// artefact it is being judged against.
	if rp.f != nil {
		t.Error("replay tape holds a write handle, want none")
	}
}

func TestAutoReplaysHitAndCallsLiveOnMiss(t *testing.T) {
	path := tapePath(t)

	rec := openTape(t, path, Record)
	live := &liveClient{turns: []llm.Response{textResp("recorded")}}
	if _, err := llm.Chain(live, rec.LLM()).Complete(context.Background(), askReq("known")); err != nil {
		t.Fatalf("record: got error %v, want nil", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	auto := openTape(t, path, Auto)
	live2 := &liveClient{turns: []llm.Response{textResp("fresh")}}
	cl := llm.Chain(live2, auto.LLM())

	hit, err := cl.Complete(context.Background(), askReq("known"))
	if err != nil {
		t.Fatalf("auto hit: got error %v, want nil", err)
	}
	if got := llm.TextOf(hit.Content); got != "recorded" {
		t.Errorf("auto hit text: got %q, want %q", got, "recorded")
	}
	if live2.count() != 0 {
		t.Errorf("auto hit reached the live client %d times, want 0", live2.count())
	}

	miss, err := cl.Complete(context.Background(), askReq("novel"))
	if err != nil {
		t.Fatalf("auto miss: got error %v, want nil", err)
	}
	if got := llm.TextOf(miss.Content); got != "fresh" {
		t.Errorf("auto miss text: got %q, want %q", got, "fresh")
	}
	if live2.count() != 1 {
		t.Errorf("auto miss live calls: got %d, want 1", live2.count())
	}
	st := auto.Stats()
	if st.Hits != 1 || st.Misses != 1 || st.Recorded != 1 {
		t.Errorf("auto Stats: got %+v, want {Hits:1 Misses:1 Recorded:1}", st)
	}
	if err := auto.Close(); err != nil {
		t.Fatalf("auto Close: got error %v, want nil", err)
	}

	// The miss was appended next to the original, not instead of it.
	lines := readLines(t, path)
	if len(lines) != 2 {
		t.Fatalf("tape lines after auto: got %d, want 2", len(lines))
	}
}

func TestReplayMissIsErrTapeMiss(t *testing.T) {
	path := tapePath(t)
	rec := openTape(t, path, Record)
	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("x")}}, rec.LLM()).
		Complete(context.Background(), askReq("known")); err != nil {
		t.Fatalf("record: got error %v, want nil", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	rp := openTape(t, path, Replay)

	t.Run("llm", func(t *testing.T) {
		live := &liveClient{turns: []llm.Response{textResp("must not be used")}}
		_, err := llm.Chain(live, rp.LLM()).Complete(context.Background(), askReq("unknown"))
		if !errors.Is(err, ErrTapeMiss) {
			t.Fatalf("got error %v, want one matching ErrTapeMiss", err)
		}
		if live.count() != 0 {
			t.Errorf("a replay miss fell through to the live client %d times, want 0", live.count())
		}
		// The message has to be enough to grep the tape with.
		_, key, kerr := canonLLM(askReq("unknown"))
		if kerr != nil {
			t.Fatalf("canonLLM: %v", kerr)
		}
		if !strings.Contains(err.Error(), key) {
			t.Errorf("error %q does not name the key %s", err, key)
		}
	})

	t.Run("tool", func(t *testing.T) {
		ran := 0
		def := bumpTool(&ran)
		_, err := rp.Tools()(tool.Direct)(context.Background(), def,
			llm.ToolUse{ID: "u9", Name: "bump", Input: json.RawMessage(`{}`)})
		if !errors.Is(err, ErrTapeMiss) {
			t.Fatalf("got error %v, want one matching ErrTapeMiss", err)
		}
		if ran != 0 {
			t.Errorf("a replay miss ran the tool %d times, want 0", ran)
		}
		if !strings.Contains(err.Error(), `"bump"`) {
			t.Errorf("error %q does not name the tool", err)
		}
	})

	t.Run("no file is an error, not an empty tape", func(t *testing.T) {
		_, err := Open(filepath.Join(t.TempDir(), "absent.jsonl"), Replay)
		if err == nil {
			t.Fatal("Open(missing, Replay): got nil error, want one")
		}
	})
}

// ===== keying =====

// TestKeyIsSHA256OfRequestFieldOnTheLine is the package's stated wire
// contract: "the key field is the SHA-256 of the request field exactly as
// written on the line, so a tape can be verified with sha256 and jq and
// nothing else". It is computed here from the file's own bytes, never from
// canonLLM, so a change to the hash input that also changes what is written
// cannot hide.
func TestKeyIsSHA256OfRequestFieldOnTheLine(t *testing.T) {
	path := tapePath(t)
	rec := openTape(t, path, Record)

	// Angle brackets and an ampersand: the encoder must not HTML-escape, or
	// the hashed bytes and the written bytes drift apart.
	req := askReq("compare <a> & <b>")
	req.Tools = []llm.ToolSpec{{
		Name:        "calc",
		Description: "does <math>",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"e":{"type":"string"}}}`),
	}}
	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("ok <b>")}}, rec.LLM()).
		Complete(context.Background(), req); err != nil {
		t.Fatalf("record: got error %v, want nil", err)
	}
	ran := 0
	if _, err := rec.Tools()(tool.Direct)(context.Background(), bumpTool(&ran),
		llm.ToolUse{ID: "u1", Name: "bump", Input: json.RawMessage(`{"q":"a<b"}`)}); err != nil {
		t.Fatalf("record tool: got error %v, want nil", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	lines := readLines(t, path)
	if len(lines) != 2 {
		t.Fatalf("tape lines: got %d, want 2", len(lines))
	}
	for i, e := range lines {
		sum := sha256.Sum256(e.Request)
		want := hex.EncodeToString(sum[:])
		if e.Key != want {
			t.Errorf("line %d key: got %s, want sha256(request)=%s\nrequest: %s", i+1, e.Key, want, e.Request)
		}
	}

	// The fields tape encodes itself do go out unescaped, which is what
	// canonical()'s SetEscapeHTML(false) buys.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	for _, want := range []string{`"description":"does <math>"`, `"input":{"q":"a<b"}`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("tape does not contain %s unescaped:\n%s", want, raw)
		}
	}
}

// TestTapeDoesNotHTMLEscapeMessageContent is a FAILING test for a real defect,
// kept and skipped because fixing it means changing a non-test file.
//
// codec.go's canonical() turns HTML escaping off, and says why: "model traffic
// is full of <, > and &, and < in a file a human is expected to read a
// diff of helps nobody". It works for every field tape encodes itself — see
// the tool description and tool input in the test above. It does NOT work for
// the messages, which are the field that actually carries model traffic:
// llm.Message.MarshalJSON (llm/llm.go) calls plain json.Marshal, which escapes,
// and encoding/json compacts a Marshaler's output without ever un-escaping it.
// So "compare <a> & <b>" lands on the tape as
// "compare <a> & <b>".
//
// Not a correctness bug — the key still equals sha256 of the request field, and
// the entry round-trips — but the stated readability goal is defeated in
// exactly the place it was written for. The fix belongs in llm.Message
// (marshal through an encoder with SetEscapeHTML(false)), not here.
func TestTapeDoesNotHTMLEscapeMessageContent(t *testing.T) {

	path := tapePath(t)
	rec := openTape(t, path, Record)
	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("ok")}}, rec.LLM()).
		Complete(context.Background(), askReq("compare <a> & <b>")); err != nil {
		t.Fatalf("record: got error %v, want nil", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !strings.Contains(string(raw), "compare <a> & <b>") {
		t.Errorf("message content was HTML-escaped on the tape:\n%s", raw)
	}
}

func TestKeyExcludesOnDeltaAndPurpose(t *testing.T) {
	base := askReq("same question")

	withPurpose := base
	withPurpose.Purpose = llm.PurposePlanner
	withDelta := base
	withDelta.OnDelta = func(llm.Delta) {}
	withBoth := base
	withBoth.Purpose = llm.PurposeOther
	withBoth.OnDelta = func(llm.Delta) {}

	_, want, err := canonLLM(base)
	if err != nil {
		t.Fatalf("canonLLM: %v", err)
	}
	for name, req := range map[string]llm.Request{
		"purpose": withPurpose,
		"ondelta": withDelta,
		"both":    withBoth,
	} {
		_, got, err := canonLLM(req)
		if err != nil {
			t.Fatalf("canonLLM(%s): %v", name, err)
		}
		if got != want {
			t.Errorf("key with %s differs: got %s, want %s", name, got, want)
		}
	}

	// And end to end: a run tagged "planner" replays what a run tagged
	// "executor" recorded.
	path := tapePath(t)
	rec := openTape(t, path, Record)
	recReq := base
	recReq.Purpose = llm.PurposeOther
	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("shared")}}, rec.LLM()).
		Complete(context.Background(), recReq); err != nil {
		t.Fatalf("record: got error %v, want nil", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	rp := openTape(t, path, Replay)
	playReq := base
	playReq.Purpose = llm.PurposePlanner
	playReq.OnDelta = func(llm.Delta) {}
	resp, err := llm.Chain(&liveClient{}, rp.LLM()).Complete(context.Background(), playReq)
	if err != nil {
		t.Fatalf("replay with a different Purpose: got error %v, want a hit", err)
	}
	if got := llm.TextOf(resp.Content); got != "shared" {
		t.Errorf("replay text: got %q, want %q", got, "shared")
	}
}

func TestKeyDistinguishesRequests(t *testing.T) {
	base := askReq("q")
	_, baseKey, err := canonLLM(base)
	if err != nil {
		t.Fatalf("canonLLM: %v", err)
	}

	mut := map[string]func(*llm.Request){
		"model":       func(r *llm.Request) { r.Model = "other-model" },
		"system":      func(r *llm.Request) { r.System = "be verbose" },
		"messages":    func(r *llm.Request) { r.Messages = []llm.Message{llm.UserText("different")} },
		"max_tokens":  func(r *llm.Request) { r.MaxTokens = 512 },
		"tools":       func(r *llm.Request) { r.Tools = []llm.ToolSpec{{Name: "t"}} },
		"tool_choice": func(r *llm.Request) { r.Choice = llm.ForceTool("t") },
		"schema_order": func(r *llm.Request) {
			r.Tools = []llm.ToolSpec{{Name: "t", InputSchema: json.RawMessage(`{"b":1,"a":2}`)}}
		},
		// Sampling changes what the provider returns. These were once left
		// out, and a recording at one temperature replayed for another.
		"temperature":      func(r *llm.Request) { r.Temperature = ptr(0.7) },
		"temperature zero": func(r *llm.Request) { r.Temperature = ptr(0) },
		"top_p":            func(r *llm.Request) { r.TopP = ptr(0.9) },
	}
	for name, f := range mut {
		t.Run(name, func(t *testing.T) {
			r := base
			f(&r)
			_, got, err := canonLLM(r)
			if err != nil {
				t.Fatalf("canonLLM: %v", err)
			}
			if got == baseKey {
				t.Errorf("changing %s did not change the key: got %s for both", name, got)
			}
		})
	}

	// Two different temperatures are two different questions.
	hot, cold := base, base
	hot.Temperature, cold.Temperature = ptr(1), ptr(0)
	_, hotKey, _ := canonLLM(hot)
	_, coldKey, _ := canonLLM(cold)
	if hotKey == coldKey {
		t.Error("temperature 1 and temperature 0 share a key")
	}

	// A zero ToolChoice and an explicit ChoiceAuto are the same request on the
	// wire, so they must be the same key.
	explicitAuto := base
	explicitAuto.Choice = llm.ToolChoice{Mode: llm.ChoiceAuto}
	if _, got, _ := canonLLM(explicitAuto); got != baseKey {
		t.Errorf("explicit ChoiceAuto key: got %s, want the zero-value key %s", got, baseKey)
	}

	// nil and empty Messages are the same conversation.
	nilMsgs := base
	nilMsgs.Messages = nil
	emptyMsgs := base
	emptyMsgs.Messages = []llm.Message{}
	_, k1, _ := canonLLM(nilMsgs)
	_, k2, _ := canonLLM(emptyMsgs)
	if k1 != k2 {
		t.Errorf("nil vs empty Messages: got %s and %s, want the same key", k1, k2)
	}
}

func TestToolKeyIsNameAndInput(t *testing.T) {
	_, a, err := canonTool("bump", json.RawMessage(`{"a":1,"b":2}`))
	if err != nil {
		t.Fatalf("canonTool: %v", err)
	}
	_, b, _ := canonTool("bump", json.RawMessage(`{"b":2,"a":1}`))
	if a == b {
		t.Error("reordered input keys shared a tool key; input is documented as hashed verbatim")
	}
	_, c, _ := canonTool("other", json.RawMessage(`{"a":1,"b":2}`))
	if a == c {
		t.Error("two different tool names shared a key")
	}
}

// ===== sequences =====

// TestIdenticalRequestsFormASequence pins the behaviour take() is written for:
// a key maps to a LIST, replayed in recorded order, and once the list is
// exhausted the last entry repeats instead of erroring or falling through to
// a live call.
func TestIdenticalRequestsFormASequence(t *testing.T) {
	path := tapePath(t)

	rec := openTape(t, path, Record)
	live := &liveClient{turns: []llm.Response{textResp("one"), textResp("two"), textResp("three")}}
	cl := llm.Chain(live, rec.LLM())
	for i := 0; i < 3; i++ {
		if _, err := cl.Complete(context.Background(), askReq("same")); err != nil {
			t.Fatalf("record call %d: got error %v, want nil", i+1, err)
		}
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	lines := readLines(t, path)
	if len(lines) != 3 {
		t.Fatalf("lines: got %d, want 3", len(lines))
	}
	for i, e := range lines {
		if e.Seq != i {
			t.Errorf("line %d seq: got %d, want %d", i+1, e.Seq, i)
		}
		if e.Key != lines[0].Key {
			t.Errorf("line %d key: got %s, want the same key as line 1 (%s)", i+1, e.Key, lines[0].Key)
		}
	}

	rp := openTape(t, path, Replay)
	rcl := llm.Chain(&liveClient{}, rp.LLM())
	var got []string
	for i := 0; i < 5; i++ {
		resp, err := rcl.Complete(context.Background(), askReq("same"))
		if err != nil {
			t.Fatalf("replay call %d: got error %v, want nil", i+1, err)
		}
		got = append(got, llm.TextOf(resp.Content))
	}
	want := []string{"one", "two", "three", "three", "three"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("replay sequence: got %v, want %v", got, want)
	}
}

// TestEntriesRecordedThisSessionAreNotReplayable defends the rule in the
// comment on Tape.loaded: an Auto run that makes the same request twice must
// call live twice, or a retry loop would spin on an answer that can never
// change. The entry becomes replayable only on the next Open.
func TestEntriesRecordedThisSessionAreNotReplayable(t *testing.T) {
	path := tapePath(t)

	auto := openTape(t, path, Auto)
	live := &liveClient{turns: []llm.Response{textResp("first"), textResp("second")}}
	cl := llm.Chain(live, auto.LLM())
	for i := 0; i < 2; i++ {
		if _, err := cl.Complete(context.Background(), askReq("same")); err != nil {
			t.Fatalf("call %d: got error %v, want nil", i+1, err)
		}
	}
	if live.count() != 2 {
		t.Errorf("live calls in one Auto session: got %d, want 2 (this session's entries are not replayable)", live.count())
	}
	if st := auto.Stats(); st.Hits != 0 {
		t.Errorf("Stats().Hits: got %d, want 0", st.Hits)
	}
	if err := auto.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	// Reopened, the same pair replays with no live call at all.
	reopened := openTape(t, path, Auto)
	live2 := &liveClient{}
	cl2 := llm.Chain(live2, reopened.LLM())
	var got []string
	for i := 0; i < 2; i++ {
		resp, err := cl2.Complete(context.Background(), askReq("same"))
		if err != nil {
			t.Fatalf("reopened call %d: got error %v, want nil", i+1, err)
		}
		got = append(got, llm.TextOf(resp.Content))
	}
	if live2.count() != 0 {
		t.Errorf("reopened live calls: got %d, want 0", live2.count())
	}
	if fmt.Sprint(got) != fmt.Sprint([]string{"first", "second"}) {
		t.Errorf("reopened replay: got %v, want [first second]", got)
	}
}

// TestRecordAppendsAcrossSessions: a second Record session must add to the
// tape, not truncate it, and seq must keep counting from where the file left
// off rather than restarting at 0 (which would make the sequence replay in an
// arbitrary order after a sort).
func TestRecordAppendsAcrossSessions(t *testing.T) {
	path := tapePath(t)

	first := openTape(t, path, Record)
	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("s1")}}, first.LLM()).
		Complete(context.Background(), askReq("same")); err != nil {
		t.Fatalf("session 1: got error %v, want nil", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("session 1 Close: got error %v, want nil", err)
	}

	second := openTape(t, path, Record)
	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("s2")}}, second.LLM()).
		Complete(context.Background(), askReq("same")); err != nil {
		t.Fatalf("session 2: got error %v, want nil", err)
	}
	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("s3")}}, second.LLM()).
		Complete(context.Background(), askReq("other")); err != nil {
		t.Fatalf("session 2 second call: got error %v, want nil", err)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("session 2 Close: got error %v, want nil", err)
	}

	lines := readLines(t, path)
	if len(lines) != 3 {
		t.Fatalf("lines after two Record sessions: got %d, want 3 (truncated?)", len(lines))
	}
	if lines[0].Seq != 0 || lines[1].Seq != 1 {
		t.Errorf("seq for the repeated key: got %d then %d, want 0 then 1", lines[0].Seq, lines[1].Seq)
	}
	if lines[2].Seq != 0 {
		t.Errorf("seq for a fresh key: got %d, want 0", lines[2].Seq)
	}

	rp := openTape(t, path, Replay)
	cl := llm.Chain(&liveClient{}, rp.LLM())
	var got []string
	for i := 0; i < 2; i++ {
		resp, err := cl.Complete(context.Background(), askReq("same"))
		if err != nil {
			t.Fatalf("replay %d: got error %v, want nil", i+1, err)
		}
		got = append(got, llm.TextOf(resp.Content))
	}
	if fmt.Sprint(got) != fmt.Sprint([]string{"s1", "s2"}) {
		t.Errorf("cross-session sequence: got %v, want [s1 s2]", got)
	}
}

// ===== corruption =====

func TestLoadRejectsCorruptLines(t *testing.T) {
	good := func(t *testing.T) string {
		t.Helper()
		path := tapePath(t)
		rec := openTape(t, path, Record)
		if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("ok")}}, rec.LLM()).
			Complete(context.Background(), askReq("q")); err != nil {
			t.Fatalf("record: %v", err)
		}
		if err := rec.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		return path
	}

	// A crash mid-append leaves a half-written last line. Dropping it would
	// make a shortened tape indistinguishable from a shorter run, so it is a
	// loud error naming the line.
	t.Run("torn final line", func(t *testing.T) {
		path := good(t)
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		torn := append(raw, []byte(`{"kind":"llm","key":"abc","seq":0,"request":{"model":"m"`)...)
		if err := os.WriteFile(path, torn, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, err = Open(path, Replay)
		if err == nil {
			t.Fatal("Open on a torn tape: got nil error, want a loud one")
		}
		if !strings.Contains(err.Error(), "line 2") {
			t.Errorf("error %q does not name the offending line", err)
		}
		if !strings.Contains(err.Error(), path) {
			t.Errorf("error %q does not name the file", err)
		}
	})

	t.Run("interior garbage", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "bad.jsonl")
		if err := os.WriteFile(path, []byte("not json at all\n{\"kind\":\"llm\",\"key\":\"a\",\"seq\":0,\"request\":{},\"response\":{}}\n"), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, err := Open(path, Replay)
		if err == nil || !strings.Contains(err.Error(), "line 1") {
			t.Fatalf("got error %v, want one naming line 1", err)
		}
	})

	t.Run("unknown kind", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "kind.jsonl")
		if err := os.WriteFile(path, []byte(`{"kind":"weather","key":"a","seq":0,"request":{},"response":{}}`+"\n"), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, err := Open(path, Replay)
		if err == nil || !strings.Contains(err.Error(), "weather") {
			t.Fatalf("got error %v, want one naming the unknown kind", err)
		}
	})

	t.Run("missing key", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "nokey.jsonl")
		if err := os.WriteFile(path, []byte(`{"kind":"llm","seq":0,"request":{},"response":{}}`+"\n"), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, err := Open(path, Replay)
		if err == nil || !strings.Contains(err.Error(), "no key") {
			t.Fatalf("got error %v, want one about a missing key", err)
		}
	})

	// A line from a newer layout is refused by name: replayed, it could be
	// read as something it is not; skipped, it is a miss nobody can explain.
	t.Run("newer format", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "newer.jsonl")
		line := fmt.Sprintf(`{"kind":"llm","key":"a","seq":0,"request":{},"response":{},"format":%d}`, formatRevision+1)
		if err := os.WriteFile(path, []byte(line+"\n"), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, err := Open(path, Replay)
		if err == nil || !strings.Contains(err.Error(), "newer version") || !strings.Contains(err.Error(), "line 1") {
			t.Fatalf("got error %v, want one saying line 1 is from a newer version", err)
		}
	})

	// The revision this package writes is the one every existing tape is in,
	// so it costs them nothing: no new key on the line, no re-record.
	t.Run("current lines carry no revision", func(t *testing.T) {
		raw, err := os.ReadFile(good(t))
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if formatRevision == 0 && strings.Contains(string(raw), `"format"`) {
			t.Errorf("a revision-0 line spells its revision out, which changes every line of every re-recorded tape:\n%s", raw)
		}
	})

	t.Run("blank lines are tolerated", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "blank.jsonl")
		body := "\n" + `{"kind":"llm","key":"a","seq":0,"request":{},"response":{}}` + "\n\n"
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		if _, err := Open(path, Replay); err != nil {
			t.Fatalf("Open: got error %v, want nil", err)
		}
	})
}

// ===== write failures =====

// TestWriteFailureIsLatchedAndSurfacedByClose: an append that fails must not
// fail the call in flight — the model response is already paid for and the
// tool's side effect has already happened — so the failure is held and
// reported by Close. Discarding Close's error is how a tape quietly stops
// recording.
func TestWriteFailureIsLatchedAndSurfacedByClose(t *testing.T) {
	path := tapePath(t)
	tp, err := Open(path, Record)
	if err != nil {
		t.Fatalf("Open: got error %v, want nil", err)
	}

	// Simulate the disk going away mid-run.
	if err := tp.f.Close(); err != nil {
		t.Fatalf("pre-closing the handle: %v", err)
	}

	live := &liveClient{turns: []llm.Response{textResp("paid for")}}
	resp, err := llm.Chain(live, tp.LLM()).Complete(context.Background(), askReq("q"))
	if err != nil {
		t.Fatalf("the call in flight failed because the tape could not be written: got %v, want nil", err)
	}
	if got := llm.TextOf(resp.Content); got != "paid for" {
		t.Errorf("response text: got %q, want %q", got, "paid for")
	}

	ran := 0
	out, terr := tp.Tools()(tool.Direct)(context.Background(), bumpTool(&ran),
		llm.ToolUse{ID: "u1", Name: "bump", Input: json.RawMessage(`{}`)})
	if terr != nil || out != "bumped" {
		t.Errorf("tool call: got (%q, %v), want (\"bumped\", nil)", out, terr)
	}

	cerr := tp.Close()
	if cerr == nil {
		t.Fatal("Close: got nil error, want the latched append failure")
	}
	if !strings.Contains(cerr.Error(), path) {
		t.Errorf("Close error %q does not name the tape", cerr)
	}
	if st := tp.Stats(); st.Recorded != 0 {
		t.Errorf("Stats().Recorded after a failed append: got %d, want 0", st.Recorded)
	}
	if err := tp.Close(); err == nil {
		t.Error("second Close: got nil, want the same latched error (Close is idempotent, not amnesiac)")
	}
}

func TestRecordAfterCloseLatchesErrClosed(t *testing.T) {
	path := tapePath(t)
	tp := openTape(t, path, Record)
	if err := tp.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("late")}}, tp.LLM()).
		Complete(context.Background(), askReq("q")); err != nil {
		t.Fatalf("late call: got error %v, want nil", err)
	}
	err := tp.Close()
	if !errors.Is(err, ErrClosed) {
		t.Fatalf("Close after a post-close record: got %v, want an error matching ErrClosed", err)
	}
}

// ===== wire shape =====

// TestKeyWithoutSamplingIsUnchanged pins the key of a request that sets no
// sampling control to the value it had before Temperature and TopP joined the
// key. They are omitted when nil precisely so this holds: every tape recorded
// before the change must still replay.
func TestKeyWithoutSamplingIsUnchanged(t *testing.T) {
	canon, key, err := canonLLM(askReq("q"))
	if err != nil {
		t.Fatalf("canonLLM: %v", err)
	}
	const want = "1e5631e6f8c109df9e461eb63e3d1f90e134f411298698dea18aae469798c0fa"
	if key != want {
		t.Errorf("key = %s, want %s — the canonical form of an unsampled request changed:\n%s", key, want, canon)
	}
	if strings.Contains(string(canon), "temperature") || strings.Contains(string(canon), "top_p") {
		t.Errorf("an unset sampling control reached the canonical bytes: %s", canon)
	}
}

func ptr(f float64) *float64 { return &f }

func TestWireFieldOrder(t *testing.T) {
	path := tapePath(t)
	rec := openTape(t, path, Record)

	live := &liveClient{turns: []llm.Response{textResp("hello")}, stream: "hello"}
	req := askReq("q")
	req.OnDelta = func(llm.Delta) {}
	if _, err := llm.Chain(live, rec.LLM()).Complete(context.Background(), req); err != nil {
		t.Fatalf("record: got error %v, want nil", err)
	}
	ran := 0
	if _, err := rec.Tools()(tool.Direct)(context.Background(), bumpTool(&ran),
		llm.ToolUse{ID: "u1", Name: "bump", Input: json.RawMessage(`{"n":1}`)}); err != nil {
		t.Fatalf("record tool: got error %v, want nil", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("lines: got %d, want 2", len(lines))
	}

	// Field order is the wire contract: this file gets diffed, and a reordered
	// key set turns a one-line change into a whole-file change.
	for i, ln := range lines {
		if !strings.HasPrefix(ln, `{"kind":"`) {
			t.Errorf("line %d does not start with kind: %s", i+1, ln)
		}
		for _, want := range []string{`"kind":`, `"key":`, `"seq":`, `"request":`, `"response":`} {
			if !strings.Contains(ln, want) {
				t.Errorf("line %d is missing %s: %s", i+1, want, ln)
			}
		}
		if got := keyOrder(ln); fmt.Sprint(got) != `[kind key seq request response]` {
			t.Errorf("line %d top-level key order: got %v, want [kind key seq request response]", i+1, got)
		}
	}

	if got := keyOrder(string(lines[0])); len(got) == 0 {
		t.Fatal("could not read the key order")
	}
	// The llm response object: content, stop_reason, usage, model, streamed —
	// streamed last, because an additive tail is the cheapest place for a
	// diffed format to grow.
	var e0 wireEntry
	if err := json.Unmarshal([]byte(lines[0]), &e0); err != nil {
		t.Fatalf("line 1: %v", err)
	}
	if got := keyOrder(string(e0.Response)); fmt.Sprint(got) != `[content stop_reason usage model streamed]` {
		t.Errorf("llm response key order: got %v, want [content stop_reason usage model streamed]", got)
	}
	if got := keyOrder(string(e0.Request)); fmt.Sprint(got) != `[model system messages max_tokens]` {
		t.Errorf("llm request key order: got %v, want [model system messages max_tokens]", got)
	}

	var e1 wireEntry
	if err := json.Unmarshal([]byte(lines[1]), &e1); err != nil {
		t.Fatalf("line 2: %v", err)
	}
	if e1.Kind != "tool" {
		t.Errorf("second line kind: got %q, want %q", e1.Kind, "tool")
	}
	if got := keyOrder(string(e1.Request)); fmt.Sprint(got) != `[name input]` {
		t.Errorf("tool request key order: got %v, want [name input]", got)
	}
	if got := keyOrder(string(e1.Response)); fmt.Sprint(got) != `[ok output]` {
		t.Errorf("tool response key order: got %v, want [ok output]", got)
	}
}

// keyOrder returns the top-level object keys of a JSON object in the order
// they appear in the bytes, which is what a diff sees and what encoding/json
// throws away.
func keyOrder(obj string) []string {
	dec := json.NewDecoder(strings.NewReader(obj))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil
	}
	var keys []string
	depth := 0
	for dec.More() || depth > 0 {
		tok, err := dec.Token()
		if err != nil {
			return keys
		}
		switch v := tok.(type) {
		case json.Delim:
			switch v {
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth < 0 {
					return keys
				}
			}
		case string:
			if depth == 0 {
				keys = append(keys, v)
				// Skip the value.
				var discard json.RawMessage
				if err := dec.Decode(&discard); err != nil {
					return keys
				}
			}
		}
	}
	return keys
}

// ===== streaming and errors =====

func TestStreamedFlagControlsReplayDeltas(t *testing.T) {
	t.Run("recorded live call streamed", func(t *testing.T) {
		path := tapePath(t)
		rec := openTape(t, path, Record)
		live := &liveClient{turns: []llm.Response{textResp("streamy")}, stream: "streamy"}
		req := askReq("q")
		req.OnDelta = func(llm.Delta) {}
		if _, err := llm.Chain(live, rec.LLM()).Complete(context.Background(), req); err != nil {
			t.Fatalf("record: got error %v, want nil", err)
		}
		if err := rec.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}

		rp := openTape(t, path, Replay)
		var seen []string
		rreq := askReq("q")
		rreq.OnDelta = func(d llm.Delta) { seen = append(seen, d.Text) }
		if _, err := llm.Chain(&liveClient{}, rp.LLM()).Complete(context.Background(), rreq); err != nil {
			t.Fatalf("replay: got error %v, want nil", err)
		}
		if fmt.Sprint(seen) != fmt.Sprint([]string{"streamy"}) {
			t.Errorf("replayed deltas: got %v, want [streamy]", seen)
		}
	})

	t.Run("recorded live call did not stream", func(t *testing.T) {
		path := tapePath(t)
		rec := openTape(t, path, Record)
		live := &liveClient{turns: []llm.Response{textResp("buffered")}} // never calls OnDelta
		req := askReq("q")
		req.OnDelta = func(llm.Delta) {}
		if _, err := llm.Chain(live, rec.LLM()).Complete(context.Background(), req); err != nil {
			t.Fatalf("record: got error %v, want nil", err)
		}
		if err := rec.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}

		rp := openTape(t, path, Replay)
		var seen []llm.Delta
		rreq := askReq("q")
		rreq.OnDelta = func(d llm.Delta) { seen = append(seen, d) }
		if _, err := llm.Chain(&liveClient{}, rp.LLM()).Complete(context.Background(), rreq); err != nil {
			t.Fatalf("replay: got error %v, want nil", err)
		}
		if len(seen) != 0 {
			t.Errorf("replay invented %d deltas for a buffered recording, want 0", len(seen))
		}
	})
}

func TestLLMErrorsAreNotRecorded(t *testing.T) {
	path := tapePath(t)
	rec := openTape(t, path, Record)
	boom := llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		return llm.Response{}, errors.New("429 slow down")
	})
	if _, err := llm.Chain(boom, rec.LLM()).Complete(context.Background(), askReq("q")); err == nil {
		t.Fatal("got nil error, want the upstream failure")
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}
	if info, err := os.Stat(path); err != nil {
		t.Fatalf("Stat: %v", err)
	} else if info.Size() != 0 {
		t.Errorf("tape size after a failed call: got %d bytes, want 0 (an outage must not be taped)", info.Size())
	}
}

func TestToolErrorsAreRecordedAndReplayed(t *testing.T) {
	path := tapePath(t)
	failing := tool.Def{
		Name:        "flaky",
		Description: "fails",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn: func(context.Context, json.RawMessage) (string, error) {
			return "partial output", errors.New("disk on fire")
		},
	}

	rec := openTape(t, path, Record)
	out, err := rec.Tools()(tool.Direct)(context.Background(), failing,
		llm.ToolUse{ID: "u1", Name: "flaky", Input: json.RawMessage(`{}`)})
	if err == nil || out != "partial output" {
		t.Fatalf("record: got (%q, %v), want (\"partial output\", non-nil)", out, err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	rp := openTape(t, path, Replay)
	neverRuns := failing
	neverRuns.Fn = func(context.Context, json.RawMessage) (string, error) {
		t.Error("the tool ran during replay, want the recorded verdict")
		return "", nil
	}
	rout, rerr := rp.Tools()(tool.Direct)(context.Background(), neverRuns,
		llm.ToolUse{ID: "u1", Name: "flaky", Input: json.RawMessage(`{}`)})
	if rerr == nil {
		t.Fatal("replay: got nil error, want the recorded failure")
	}
	if rerr.Error() != "disk on fire" {
		t.Errorf("replayed error: got %q, want %q", rerr, "disk on fire")
	}
	if rout != "partial output" {
		t.Errorf("replayed output: got %q, want %q", rout, "partial output")
	}
}

// A tool that fails with an empty message must still replay as a failure: OK
// is a separate field precisely so that an empty error is not read as success.
func TestToolFailureWithEmptyMessageStaysAFailure(t *testing.T) {
	path := tapePath(t)
	silent := tool.Def{
		Name:        "silent",
		Description: "fails quietly",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn:          func(context.Context, json.RawMessage) (string, error) { return "", errors.New("") },
	}
	rec := openTape(t, path, Record)
	if _, err := rec.Tools()(tool.Direct)(context.Background(), silent,
		llm.ToolUse{ID: "u1", Name: "silent", Input: json.RawMessage(`{}`)}); err == nil {
		t.Fatal("record: got nil error, want one")
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	rp := openTape(t, path, Replay)
	_, err := rp.Tools()(tool.Direct)(context.Background(), silent,
		llm.ToolUse{ID: "u1", Name: "silent", Input: json.RawMessage(`{}`)})
	if err == nil {
		t.Error("replay of an empty-message failure: got nil error, want a failure")
	}
}

func TestReplayHonoursContextCancellation(t *testing.T) {
	path := tapePath(t)
	rec := openTape(t, path, Record)
	if _, err := llm.Chain(&liveClient{turns: []llm.Response{textResp("x")}}, rec.LLM()).
		Complete(context.Background(), askReq("q")); err != nil {
		t.Fatalf("record: %v", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	rp := openTape(t, path, Replay)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := llm.Chain(&liveClient{}, rp.LLM()).Complete(ctx, askReq("q")); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled replay: got %v, want context.Canceled", err)
	}
}

// ===== misc surface =====

func TestOpenRejectsBadArguments(t *testing.T) {
	if _, err := Open("", Record); err == nil {
		t.Error("Open with an empty path: got nil error, want one")
	}
	if _, err := Open(filepath.Join(t.TempDir(), "t.jsonl"), Mode(42)); err == nil {
		t.Error("Open with an unknown mode: got nil error, want one")
	}
}

func TestModeString(t *testing.T) {
	for mode, want := range map[Mode]string{Auto: "auto", Record: "record", Replay: "replay", Mode(9): "Mode(9)"} {
		if got := mode.String(); got != want {
			t.Errorf("Mode(%d).String(): got %q, want %q", int(mode), got, want)
		}
	}
}

// ===== concurrency =====

// One agent fans sub-agents across goroutines and they share the tape, so both
// middlewares and Stats have to hold up under -race.
func TestConcurrentUse(t *testing.T) {
	path := tapePath(t)

	const workers = 8
	const perWorker = 6

	rec := openTape(t, path, Record)
	live := llm.ClientFunc(func(_ context.Context, req llm.Request) (llm.Response, error) {
		return textResp("re: " + llm.TextOf(req.Messages[0].Content)), nil
	})
	cl := llm.Chain(live, rec.LLM())
	th := rec.Tools()(tool.Direct)
	echo := tool.Def{
		Name:        "echo",
		Description: "echoes",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn:          func(_ context.Context, in json.RawMessage) (string, error) { return string(in), nil },
	}

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				q := fmt.Sprintf("w%d-i%d", w, i)
				if _, err := cl.Complete(context.Background(), askReq(q)); err != nil {
					t.Errorf("worker %d llm: got error %v, want nil", w, err)
					return
				}
				if _, err := th(context.Background(), echo,
					llm.ToolUse{ID: llm.ToolUseID(q), Name: "echo", Input: json.RawMessage(`{"q":"` + q + `"}`)}); err != nil {
					t.Errorf("worker %d tool: got error %v, want nil", w, err)
					return
				}
				_ = rec.Stats()
			}
		}(w)
	}
	wg.Wait()
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}

	// Concurrent single Writes on an O_APPEND descriptor must not interleave:
	// every line still parses, and none is lost.
	lines := readLines(t, path)
	if want := workers * perWorker * 2; len(lines) != want {
		t.Fatalf("lines: got %d, want %d", len(lines), want)
	}

	rp := openTape(t, path, Replay)
	rcl := llm.Chain(llm.ClientFunc(func(context.Context, llm.Request) (llm.Response, error) {
		t.Error("replay reached the live client")
		return llm.Response{}, errors.New("live")
	}), rp.LLM())
	rth := rp.Tools()(tool.Direct)

	var wg2 sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg2.Add(1)
		go func(w int) {
			defer wg2.Done()
			for i := 0; i < perWorker; i++ {
				q := fmt.Sprintf("w%d-i%d", w, i)
				resp, err := rcl.Complete(context.Background(), askReq(q))
				if err != nil {
					t.Errorf("replay worker %d: got error %v, want nil", w, err)
					return
				}
				if got, want := llm.TextOf(resp.Content), "re: "+q; got != want {
					t.Errorf("replay worker %d: got %q, want %q", w, got, want)
				}
				if _, err := rth(context.Background(), echo,
					llm.ToolUse{ID: llm.ToolUseID(q), Name: "echo", Input: json.RawMessage(`{"q":"` + q + `"}`)}); err != nil {
					t.Errorf("replay worker %d tool: got error %v, want nil", w, err)
				}
			}
		}(w)
	}
	wg2.Wait()

	if st := rp.Stats(); st.Misses != 0 || st.Hits != workers*perWorker*2 {
		t.Errorf("replay Stats: got %+v, want %d hits and 0 misses", st, workers*perWorker*2)
	}
}

// Reasoning survives a replay only because both provider clients materialise
// the scratchpad into an llm.Thinking block; the tape replays that block as a
// Reasoning delta, exactly as a live provider does, so it never leaks into the
// user-visible answer.
func TestReplayEmitsReasoningForThinkingBlocks(t *testing.T) {
	path := tapePath(t)
	rec := openTape(t, path, Record)
	live := &liveClient{
		turns: []llm.Response{{
			Content: []llm.ContentBlock{
				llm.Thinking{Text: "let me think", Signature: "sig"},
				llm.Text{Text: "the answer"},
			},
			StopReason: llm.StopEndTurn,
		}},
		stream: "the answer",
	}
	req := askReq("q")
	req.OnDelta = func(llm.Delta) {}
	if _, err := llm.Chain(live, rec.LLM()).Complete(context.Background(), req); err != nil {
		t.Fatalf("record: got error %v, want nil", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	rp := openTape(t, path, Replay)
	var text, reasoning strings.Builder
	rreq := askReq("q")
	rreq.OnDelta = func(d llm.Delta) {
		text.WriteString(d.Text)
		reasoning.WriteString(d.Reasoning)
	}
	if _, err := llm.Chain(&liveClient{}, rp.LLM()).Complete(context.Background(), rreq); err != nil {
		t.Fatalf("replay: got error %v, want nil", err)
	}
	if got := reasoning.String(); got != "let me think" {
		t.Errorf("replayed reasoning: got %q, want %q", got, "let me think")
	}
	if got := text.String(); got != "the answer" {
		t.Errorf("replayed text: got %q, want %q", got, "the answer")
	}
}

// A hand-edited or half-migrated tape must fail loudly at the call site, with
// the key in the message so the offending line can be found.
func TestCorruptRecordedResponseIsAnError(t *testing.T) {
	t.Run("llm", func(t *testing.T) {
		_, key, err := canonLLM(askReq("q"))
		if err != nil {
			t.Fatalf("canonLLM: %v", err)
		}
		reqBytes, _, _ := canonLLM(askReq("q"))
		path := filepath.Join(t.TempDir(), "bad.jsonl")
		line, err := canonical(entry{Kind: kindLLM, Key: key, Seq: 0,
			Request: reqBytes, Response: json.RawMessage(`{"content":"not a message"}`)})
		if err != nil {
			t.Fatalf("canonical: %v", err)
		}
		if err := os.WriteFile(path, append(line, '\n'), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		rp := openTape(t, path, Replay)
		_, cerr := llm.Chain(&liveClient{}, rp.LLM()).Complete(context.Background(), askReq("q"))
		if cerr == nil {
			t.Fatal("got nil error, want a decode failure")
		}
		if !strings.Contains(cerr.Error(), key) {
			t.Errorf("error %q does not name the key %s", cerr, key)
		}
	})

	t.Run("tool", func(t *testing.T) {
		reqBytes, key, err := canonTool("bump", json.RawMessage(`{}`))
		if err != nil {
			t.Fatalf("canonTool: %v", err)
		}
		path := filepath.Join(t.TempDir(), "bad.jsonl")
		line, err := canonical(entry{Kind: kindTool, Key: key, Seq: 0,
			Request: reqBytes, Response: json.RawMessage(`"not an object"`)})
		if err != nil {
			t.Fatalf("canonical: %v", err)
		}
		if err := os.WriteFile(path, append(line, '\n'), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		rp := openTape(t, path, Replay)
		ran := 0
		_, terr := rp.Tools()(tool.Direct)(context.Background(), bumpTool(&ran),
			llm.ToolUse{ID: "u1", Name: "bump", Input: json.RawMessage(`{}`)})
		if terr == nil {
			t.Fatal("got nil error, want a decode failure")
		}
		if !strings.Contains(terr.Error(), key) {
			t.Errorf("error %q does not name the key %s", terr, key)
		}
		if ran != 0 {
			t.Errorf("the tool ran %d times after a decode failure, want 0", ran)
		}
	})
}

// Auto is the resume mode: a tape that covers half the run pays for the other
// half and nothing else — on the tool side too, where the currency is side
// effects rather than tokens.
func TestAutoResumesToolSideEffects(t *testing.T) {
	path := tapePath(t)
	hits := 0
	def := bumpTool(&hits)
	useA := llm.ToolUse{ID: "u1", Name: "bump", Input: json.RawMessage(`{"n":1}`)}
	useB := llm.ToolUse{ID: "u2", Name: "bump", Input: json.RawMessage(`{"n":2}`)}

	rec := openTape(t, path, Record)
	if _, err := rec.Tools()(tool.Direct)(context.Background(), def, useA); err != nil {
		t.Fatalf("record: got error %v, want nil", err)
	}
	if err := rec.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if hits != 1 {
		t.Fatalf("side effects after recording: got %d, want 1", hits)
	}

	auto := openTape(t, path, Auto)
	th := auto.Tools()(tool.Direct)
	if _, err := th(context.Background(), def, useA); err != nil {
		t.Fatalf("auto replay: got error %v, want nil", err)
	}
	if hits != 1 {
		t.Errorf("a replayed tool call re-ran the side effect: got %d bumps, want 1", hits)
	}
	if _, err := th(context.Background(), def, useB); err != nil {
		t.Fatalf("auto live: got error %v, want nil", err)
	}
	if hits != 2 {
		t.Errorf("a new tool call did not run: got %d bumps, want 2", hits)
	}
	if err := auto.Close(); err != nil {
		t.Fatalf("Close: got error %v, want nil", err)
	}
	if lines := readLines(t, path); len(lines) != 2 {
		t.Errorf("lines: got %d, want 2", len(lines))
	}
}

// TestControlToolsAreNotTaped: the agent loop sends pause and terminal calls
// through the tool middleware for a verdict, and their handler never runs, so
// there is nothing to protect by taping them. Taping them would also turn every
// pause in a tape recorded before they went through the chain into a Replay
// miss.
func TestControlToolsAreNotTaped(t *testing.T) {
	// An empty file in Replay mode: anything the middleware looked up would miss.
	path := tapePath(t)
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	rp := openTape(t, path, Replay)

	for _, caps := range []tool.Cap{tool.CapPause, tool.CapTerminal} {
		d := tool.Def{Name: "control", Caps: caps}
		reached := false
		out, err := rp.Tools()(func(context.Context, tool.Def, llm.ToolUse) (string, error) {
			reached = true
			return "accepted", nil
		})(context.Background(), d, llm.ToolUse{ID: "u1", Name: "control", Input: json.RawMessage(`{}`)})
		if err != nil || out != "accepted" || !reached {
			t.Errorf("caps %v: got (%q, %v, reached=%v), want the call passed straight through", caps, out, err, reached)
		}
	}
	if s := rp.Stats(); s.Hits+s.Misses+s.Recorded != 0 {
		t.Errorf("Stats = %+v, want the tape untouched by control calls", s)
	}
}

// A tool's attached media is recorded with its output and attached again on
// replay; a replayed run that lost the screenshot would show the model a
// different conversation.
func TestToolMediaIsRecordedAndReplayed(t *testing.T) {
	path := tapePath(t)
	img := llm.Media{MediaType: "image/png", Data: []byte{9, 8, 7}, Name: "s.png"}
	d := tool.Def{Name: "shot"}
	u := llm.ToolUse{ID: "u1", Name: "shot", Input: json.RawMessage(`{"page":1}`)}

	set := func(tp *Tape, fn tool.Fn) []tool.Result {
		d := d
		d.Fn = fn
		return tool.NewDispatcher(tool.Chain(tool.Direct, tp.Tools())).Dispatch(context.Background(), tool.NewSet(d), []llm.ToolUse{u})
	}

	rec := openTape(t, path, Record)
	got := set(rec, func(ctx context.Context, _ json.RawMessage) (string, error) {
		tool.Attach(ctx, img)
		return "Attached s.png", nil
	})
	if len(got[0].Media) != 1 {
		t.Fatalf("recording: Media = %v", got[0].Media)
	}
	if err := rec.Close(); err != nil {
		t.Fatal(err)
	}

	rp := openTape(t, path, Replay)
	got = set(rp, func(context.Context, json.RawMessage) (string, error) {
		t.Error("the tool ran on replay")
		return "", nil
	})
	if got[0].Output != "Attached s.png" || len(got[0].Media) != 1 || !reflect.DeepEqual(got[0].Media[0], img) {
		t.Errorf("replay = %q %v, want the recorded output and image", got[0].Output, got[0].Media)
	}
}
