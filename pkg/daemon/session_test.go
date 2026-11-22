package daemon

// White-box tests (package daemon, not daemon_test).
//
// The lifecycle's interesting behaviour is unexported — the ring's eviction,
// the sweeper's clock, the inbox's batching — and the point of putting the
// lifecycle in a type instead of a handler is that it can be driven directly.
// There is no network here and no http.Handler; the API layer is somebody
// else's file and somebody else's test.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/tool"
)

// ===== fixtures =====

var quietLogger = slog.New(slog.NewTextHandler(io.Discard, nil))

// TestMain silences the package default logger: permission.Gate audits every
// verdict through slog.Default, and a passing test should print nothing.
func TestMain(m *testing.M) {
	slog.SetDefault(quietLogger)
	os.Exit(m.Run())
}

// scripted replays canned responses in order and records what it was asked.
// Safe for concurrent use; these tests run under -race.
type scripted struct {
	mu    sync.Mutex
	turns []llm.Response
	i     int
	seen  []llm.Request
}

func script(turns ...llm.Response) *scripted { return &scripted{turns: turns} }

// Complete implements llm.Client.
func (s *scripted) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	s.mu.Lock()
	s.seen = append(s.seen, req)
	if s.i >= len(s.turns) {
		n := s.i
		s.mu.Unlock()
		return llm.Response{}, fmt.Errorf("scripted: out of turns after %d calls", n)
	}
	r := s.turns[s.i]
	s.i++
	s.mu.Unlock()

	if req.OnDelta != nil {
		if t := llm.TextOf(r.Content); t != "" {
			req.OnDelta(llm.Delta{Text: t})
		}
	}
	return r, nil
}

func (s *scripted) request(t *testing.T, n int) llm.Request {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if n >= len(s.seen) {
		t.Fatalf("wanted request %d, the client only saw %d", n, len(s.seen))
	}
	return s.seen[n]
}

func textTurn(s string) llm.Response {
	return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: s}}, StopReason: llm.StopEndTurn}
}

func toolTurn(id, name, input string) llm.Response {
	return llm.Response{
		Content: []llm.ContentBlock{
			llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(input)},
		},
		StopReason: llm.StopToolUse,
	}
}

const objSchema = `{"type":"object"}`

// countingTool records how many times it ran, which is how "the denied call
// did not execute" is asserted.
type countingTool struct {
	mu  sync.Mutex
	ran int
}

func (c *countingTool) def(name string) tool.Def {
	return tool.Def{
		Name: name, Description: "echoes", InputSchema: json.RawMessage(objSchema),
		Fn: func(context.Context, json.RawMessage) (string, error) {
			c.mu.Lock()
			c.ran++
			c.mu.Unlock()
			return "echoed", nil
		},
	}
}

func (c *countingTool) calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ran
}

func pauseTool(name string) tool.Def {
	return tool.Def{
		Name: name, Description: "asks the user", InputSchema: json.RawMessage(objSchema),
		Caps: tool.CapPause,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			return "", errors.New("pause tool handler must never run")
		},
	}
}

func terminalTool(name string) tool.Def {
	return tool.Def{
		Name: name, Description: "submits", InputSchema: json.RawMessage(objSchema),
		Caps: tool.CapTerminal,
		Fn: func(context.Context, json.RawMessage) (string, error) {
			return "", errors.New("terminal tool handler must never run")
		},
	}
}

// builder returns a ManagerConfig.Build that serves every session from one
// client and tool set.
func builder(cl llm.Client, opts ...core.Option) func(SessionOptions) (*core.Agent, error) {
	return func(SessionOptions) (*core.Agent, error) {
		base := []core.Option{
			core.WithClient(cl),
			core.WithModel("test-model"),
			core.WithLogger(quietLogger),
		}
		return core.New(append(base, opts...)...)
	}
}

// newManager makes a Manager that polls fast and retries at once, so a test
// waits milliseconds for what a deployment waits seconds for.
func newManager(t *testing.T, cfg ManagerConfig) *Manager {
	t.Helper()
	if cfg.Logger == nil {
		cfg.Logger = quietLogger
	}
	if cfg.Poll == 0 {
		cfg.Poll = 5 * time.Millisecond
	}
	if cfg.RetryDelay == 0 {
		cfg.RetryDelay = time.Millisecond
	}
	m, err := NewManager(cfg)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// session creates one session on a manager serving cl.
func session(t *testing.T, cl llm.Client, opts ...core.Option) (*Manager, *Session) {
	t.Helper()
	m := newManager(t, ManagerConfig{Build: builder(cl, opts...)})
	s, err := m.Create(testCtx(t), SessionOptions{Title: "t"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	return m, s
}

// ===== waiting, without sleeping =====

func testCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// waitEvent blocks until the turn that handles inbox event id has ended for
// good — not to be tried again — and returns its TurnEnded. It follows the
// log rather than polling, which is also a small proof that a follower sees
// turn boundaries at all.
func waitEvent(t *testing.T, s *Session, id int64) TurnEnded {
	t.Helper()
	for _, f := range s.Follow(testCtx(t), 0) {
		if te, ok := f.Event.(TurnEnded); ok && slices.Contains(te.Events, id) && te.RetryAt.IsZero() {
			return te
		}
	}
	t.Fatalf("the turn for event %d never ended (state %s)", id, s.Info().State)
	return TurnEnded{}
}

// send sends a message and waits for the turn it starts to end.
func send(t *testing.T, s *Session, prompt string) TurnEnded {
	t.Helper()
	e, err := s.Send(testCtx(t), prompt)
	if err != nil {
		t.Fatalf("Send(%q): %v", prompt, err)
	}
	return waitEvent(t, s, e.ID)
}

// until polls cond until it holds or the test times out.
func until(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// collect drains the log from `from` until pred is satisfied by a frame.
func collect(t *testing.T, s *Session, from int, stop func(Frame) bool) []Frame {
	t.Helper()
	var out []Frame
	for _, f := range s.Follow(testCtx(t), from) {
		out = append(out, f)
		if stop(f) {
			return out
		}
	}
	return out
}

func endOfTurn(turn int) func(Frame) bool {
	return func(f Frame) bool {
		te, ok := f.Event.(TurnEnded)
		return ok && te.Turn == turn
	}
}

// ===== multi-turn =====

// The central claim of the log: two turns, one log, one unbroken sequence. A
// UI that reconnects across a turn boundary must not lose turn 1.
func TestMultiTurnSequenceIsContinuous(t *testing.T) {
	cl := script(textTurn("first"), textTurn("second"))
	_, s := session(t, cl)

	if got := send(t, s, "hello").Outcome; got != "answer" {
		t.Fatalf("turn 1 outcome = %q, want answer", got)
	}
	if got := send(t, s, "again").Outcome; got != "answer" {
		t.Fatalf("turn 2 outcome = %q, want answer", got)
	}

	frames := collect(t, s, 0, endOfTurn(2))

	for i, f := range frames {
		if f.Seq != i {
			t.Fatalf("frame %d has seq %d; the sequence must be dense and start at 0", i, f.Seq)
		}
	}
	if first, ok := frames[0].Event.(TurnStarted); !ok || first.Turn != 1 || first.Prompt != "hello" {
		t.Fatalf("first frame = %#v, want TurnStarted{1, hello}", frames[0].Event)
	}

	// The turn-2 frames must come AFTER every turn-1 frame in one numbering,
	// which is the property a Last-Event-ID resume depends on.
	lastOfOne, firstOfTwo := -1, -1
	for _, f := range frames {
		switch f.Turn {
		case 1:
			lastOfOne = max(lastOfOne, f.Seq)
		case 2:
			if firstOfTwo < 0 {
				firstOfTwo = f.Seq
			}
		default:
			t.Fatalf("frame %d belongs to turn %d", f.Seq, f.Turn)
		}
	}
	if firstOfTwo != lastOfOne+1 {
		t.Fatalf("turn 2 starts at %d but turn 1 ends at %d; the log is not continuous",
			firstOfTwo, lastOfOne)
	}

	info := s.Info()
	if info.Turns != 2 || info.State != Idle || info.Answer != "second" {
		t.Fatalf("info = %+v, want 2 turns, idle, answer \"second\"", info)
	}
	if info.Events != len(frames) {
		t.Fatalf("Events = %d, saw %d frames", info.Events, len(frames))
	}

	// Turn 2 continued the transcript rather than starting a new one.
	if got := len(cl.request(t, 1).Messages); got != 3 {
		t.Fatalf("turn 2 sent %d messages, want 3 (user, assistant, user)", got)
	}
	if got := len(s.Messages()); got != 4 {
		t.Fatalf("transcript has %d messages, want 4", got)
	}
}

// An answer leaves the session Idle, not Done: a conversation is not over
// because the model stopped talking.
func TestAnswerLeavesTheSessionOpen(t *testing.T) {
	_, s := session(t, script(textTurn("hi")))
	send(t, s, "hello")

	if got := s.Info().State; got != Idle {
		t.Fatalf("state after an answer = %q, want idle", got)
	}
	if _, err := s.Send(testCtx(t), "more"); err != nil {
		t.Fatalf("a second Send after an answer: %v", err)
	}
}

// A turn that ended in Paused leaves a dangling tool_use. The next turn has to
// close it, or the provider rejects the whole request.
func TestPausedTurnIsResumable(t *testing.T) {
	cl := script(toolTurn("u1", "ask_user", `{"question":"which one?"}`), textTurn("ok"))
	_, s := session(t, cl, core.WithTools(pauseTool("ask_user")))

	ended := send(t, s, "start")
	if ended.Outcome != "paused" || ended.Question != "which one?" {
		t.Fatalf("turn 1 = %+v, want paused with the question", ended)
	}
	// The question is what SessionInfo reports as the answer, because it is
	// the last thing the conversation said.
	if got := s.Info().Answer; got != "which one?" {
		t.Fatalf("info answer = %q, want the pending question", got)
	}
	if got := s.Info().State; got != Waiting {
		t.Fatalf("state = %q, want waiting", got)
	}

	send(t, s, "the second one")

	msgs := cl.request(t, 1).Messages
	if err := core.Convo(msgs).Validate(); err != nil {
		t.Fatalf("turn 2 sent an invalid transcript: %v", err)
	}
	var answer string
	for _, m := range msgs {
		for _, b := range m.Content {
			if tr, ok := b.(llm.ToolResult); ok && tr.ToolUseID == "u1" {
				answer = tr.Content
			}
		}
	}
	if answer != "the second one" {
		t.Fatalf("the pause was answered with %q, want the message", answer)
	}
}

// A terminal tool ends the CONVERSATION, and that is the only thing that does.
func TestSubmittedIsTerminal(t *testing.T) {
	cl := script(toolTurn("u1", "submit", `{"answer":42}`))
	_, s := session(t, cl, core.WithTools(terminalTool("submit")),
		core.WithTerminalTool("submit"))

	if got := send(t, s, "go").Outcome; got != "submitted" {
		t.Fatalf("outcome = %q, want submitted", got)
	}
	if got := s.Info().State; got != Done {
		t.Fatalf("state = %q, want done", got)
	}
	if _, err := s.Send(testCtx(t), "more"); !errors.Is(err, ErrDone) {
		t.Fatalf("Send after done = %v, want ErrDone", err)
	}

	// Done is terminal, so a follower is released rather than left hanging on
	// a conversation that will never speak again.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range s.Follow(testCtx(t), 0) { //nolint:revive // draining is the point
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Follow did not return on a finished session")
	}
}

// ===== following =====

// One follower, attached before the first turn, must see both turns without
// ever being re-attached.
func TestFollowerSurvivesATurnBoundary(t *testing.T) {
	cl := script(textTurn("first"), textTurn("second"))
	_, s := session(t, cl)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	frames := make(chan Frame, 256)
	returned := make(chan struct{})
	go func() {
		defer close(returned)
		for _, f := range s.Follow(ctx, 0) {
			frames <- f
		}
	}()

	// The follower is attached and blocked before anything has happened.
	select {
	case <-returned:
		t.Fatal("Follow returned on a session that has not started a turn")
	case <-time.After(20 * time.Millisecond):
	}

	send(t, s, "one")
	send(t, s, "two")

	var seen []Frame
	deadline := time.After(5 * time.Second)
	for {
		select {
		case f := <-frames:
			seen = append(seen, f)
			if te, ok := f.Event.(TurnEnded); ok && te.Turn == 2 {
				goto sawBoth
			}
		case <-returned:
			t.Fatal("Follow returned between turns; the session is still alive")
		case <-deadline:
			t.Fatalf("follower only saw %d frames", len(seen))
		}
	}

sawBoth:
	if len(seen) < 4 {
		t.Fatalf("follower saw %d frames across two turns", len(seen))
	}
	for i, f := range seen {
		if f.Seq != i {
			t.Fatalf("follower saw seq %d at position %d; it missed something", f.Seq, i)
		}
	}

	// Still attached, still waiting for a third turn.
	select {
	case <-returned:
		t.Fatal("Follow returned after turn 2 ended")
	case <-time.After(20 * time.Millisecond):
	}

	cancel()
	select {
	case <-returned:
	case <-time.After(5 * time.Second):
		t.Fatal("Follow did not return when its context was cancelled")
	}
}

func TestFollowResumesFromAnArbitrarySequence(t *testing.T) {
	cl := script(textTurn("first"), textTurn("second"))
	_, s := session(t, cl)
	send(t, s, "one")
	send(t, s, "two")

	all := collect(t, s, 0, endOfTurn(2))
	for from := range len(all) {
		got := collect(t, s, from, endOfTurn(2))
		if got[0].Seq != from {
			t.Fatalf("resume from %d yielded %d first", from, got[0].Seq)
		}
		if len(got) != len(all)-from {
			t.Fatalf("resume from %d yielded %d frames, want %d", from, len(got), len(all)-from)
		}
	}
}

func TestFollowReturnsPromptlyOnContext(t *testing.T) {
	_, s := session(t, script(textTurn("hi")))
	send(t, s, "hello")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	started := time.Now()
	var n int
	for range s.Follow(ctx, 0) {
		n++
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("Follow took %v to notice its context", elapsed)
	}
	if n == 0 {
		t.Fatal("Follow yielded nothing before its context expired")
	}

	// An already-dead context yields the backlog and returns, rather than
	// blocking.
	dead, cancelDead := context.WithCancel(context.Background())
	cancelDead()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range s.Follow(dead, 0) { //nolint:revive // draining is the point
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Follow blocked on an already-cancelled context")
	}
}

// Eviction is observable exactly one way: the first sequence you are yielded
// is higher than the one you asked for. Being lied to instead — served a dense
// stream with a hole in the middle — is what this pins.
func TestEvictionClampsFrom(t *testing.T) {
	const emitted = 40
	cl := llm.ClientFunc(func(_ context.Context, req llm.Request) (llm.Response, error) {
		for i := range emitted {
			if req.OnDelta != nil {
				req.OnDelta(llm.Delta{Text: strconv.Itoa(i) + " "})
			}
		}
		return llm.Response{
			Content:    []llm.ContentBlock{llm.Text{Text: "done"}},
			StopReason: llm.StopEndTurn,
		}, nil
	})

	const ring = 8
	m := newManager(t, ManagerConfig{Build: builder(cl), BufferSize: ring})
	s, err := m.Create(testCtx(t), SessionOptions{})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	e, err := s.Send(testCtx(t), "go")
	if err != nil {
		t.Fatal(err)
	}
	until(t, "the turn to end", func() bool { return s.Info().Turns == 1 && s.Info().State == Idle })
	_ = e

	total := s.Info().Events
	if total <= ring {
		t.Fatalf("only %d frames; the ring of %d never overflowed", total, ring)
	}

	s.mu.Lock()
	inMemory := len(s.frames)
	s.mu.Unlock()
	if inMemory != ring {
		t.Fatalf("memory holds %d frames, want the ring's %d", inMemory, ring)
	}

	// What the ring let go of was kept: a follower from 0 reads it all, in
	// order, with no gap.
	got := collect(t, s, 0, endOfTurn(1))
	if len(got) != total {
		t.Fatalf("resumed with %d frames, want all %d", len(got), total)
	}
	for i, f := range got {
		if f.Seq != i {
			t.Fatalf("frame %d has seq %d: a gap, or out of order", i, f.Seq)
		}
	}

	// Once the store has let go of them too, a follower starts at the oldest
	// frame retained: the only way eviction is observable.
	if _, err := m.store.PruneFrames(testCtx(t), time.Now().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	got = collect(t, s, 0, endOfTurn(1))
	if len(got) != ring {
		t.Fatalf("retained %d frames, want %d", len(got), ring)
	}
	if want := total - ring; got[0].Seq != want {
		t.Fatalf("resume from 0 yielded seq %d, want the oldest retained %d", got[0].Seq, want)
	}
	if last := got[len(got)-1]; last.Seq != total-1 {
		t.Fatalf("last retained seq %d, want %d: eviction dropped the NEWEST", last.Seq, total-1)
	}
}

// ===== concurrency =====

// blockingClient parks inside Complete until it is released, which is how a
// turn is held mid-flight. It records what it was asked.
type blockingClient struct {
	entered chan struct{}
	release chan struct{}

	mu   sync.Mutex
	seen []llm.Request
}

func newBlockingClient() *blockingClient {
	return &blockingClient{entered: make(chan struct{}, 16), release: make(chan struct{})}
}

// Complete implements llm.Client.
func (b *blockingClient) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	b.mu.Lock()
	b.seen = append(b.seen, req)
	b.mu.Unlock()
	b.entered <- struct{}{}
	select {
	case <-b.release:
		return textTurn("released"), nil
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
}

func (b *blockingClient) requests() []llm.Request {
	b.mu.Lock()
	defer b.mu.Unlock()
	return slices.Clone(b.seen)
}

// What arrives during a turn waits for it, and is handed to the next turn
// together: two messages sent while the agent works are one turn, not two.
func TestMessagesDuringATurnAreTheNextTurnTogether(t *testing.T) {
	cl := newBlockingClient()
	_, s := session(t, cl)

	first, err := s.Send(testCtx(t), "one")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	<-cl.entered
	if got := s.Info().State; got != Running {
		t.Fatalf("state = %q, want running", got)
	}
	second, err := s.Send(testCtx(t), "two")
	if err != nil {
		t.Fatalf("a Send while a turn runs = %v, want it queued", err)
	}
	third, _ := s.Send(testCtx(t), "three")

	cl.release <- struct{}{}
	waitEvent(t, s, first.ID)
	<-cl.entered
	cl.release <- struct{}{}
	ended := waitEvent(t, s, second.ID)

	if !slices.Equal(ended.Events, []int64{second.ID, third.ID}) {
		t.Fatalf("turn 2 handled %v, want both queued messages %v", ended.Events, []int64{second.ID, third.ID})
	}
	if got := len(cl.requests()); got != 2 {
		t.Fatalf("the model was called %d times, want twice: once per turn", got)
	}
	prompt := llm.TextOf(cl.requests()[1].Messages[2].Content)
	if !strings.Contains(prompt, "two") || !strings.Contains(prompt, "three") || !strings.Contains(prompt, `kind="message"`) {
		t.Fatalf("turn 2's prompt = %q, want both messages, each marked", prompt)
	}
}

// Cancel must leave nothing behind — not the run, not the goroutine draining
// it.
func TestCancelReleasesGoroutines(t *testing.T) {
	cl := newBlockingClient()
	const sessions = 8
	m := newManager(t, ManagerConfig{Build: builder(cl), MaxRuns: sessions})

	settle(t)
	before := runtime.NumGoroutine()

	var all []*Session
	for range sessions {
		s, err := m.Create(testCtx(t), SessionOptions{})
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		if _, err := s.Send(testCtx(t), "go"); err != nil {
			t.Fatalf("Send: %v", err)
		}
		<-cl.entered
		all = append(all, s)
	}
	if got := runtime.NumGoroutine(); got <= before {
		t.Fatalf("%d live turns added no goroutines (%d -> %d); the fixture is not testing anything",
			sessions, before, got)
	}

	for _, s := range all {
		s.Cancel()
		until(t, "the cancelled turn to end", func() bool { return s.Info().State == Failed })
		if kind := s.Info().ErrorKind; kind != "cancelled" {
			t.Fatalf("error kind after Cancel = %q, want cancelled", kind)
		}
	}

	settle(t)
	if after := runtime.NumGoroutine(); after > before {
		t.Fatalf("goroutines %d -> %d after cancelling %d turns", before, after, sessions)
	}
}

// settle waits for the scheduler to retire whatever has already been
// signalled, so a goroutine count means something.
func settle(t *testing.T) {
	t.Helper()
	base := runtime.NumGoroutine()
	for range 100 {
		runtime.Gosched()
		time.Sleep(2 * time.Millisecond)
		n := runtime.NumGoroutine()
		if n == base {
			return
		}
		base = n
	}
}

// MaxRuns is a cap on turns at once, across sessions: what is over it waits
// its turn in the inbox.
func TestMaxRunsCapsTurnsAtOnce(t *testing.T) {
	cl := newBlockingClient()
	m := newManager(t, ManagerConfig{Build: builder(cl), MaxRuns: 1})
	a, _ := m.Create(testCtx(t), SessionOptions{})
	b, _ := m.Create(testCtx(t), SessionOptions{})
	ea, _ := a.Send(testCtx(t), "a")
	<-cl.entered
	eb, _ := b.Send(testCtx(t), "b")

	select {
	case <-cl.entered:
		t.Fatal("a second turn started with MaxRuns at 1")
	case <-time.After(50 * time.Millisecond):
	}
	if got := b.Info().State; got != Idle {
		t.Fatalf("b = %q while a runs, want it still waiting", got)
	}
	cl.release <- struct{}{}
	waitEvent(t, a, ea.ID)
	<-cl.entered
	cl.release <- struct{}{}
	waitEvent(t, b, eb.ID)
}

// ===== decisions =====

// gated builds a session whose every tool call needs a person's decision.
func gated(t *testing.T, cl llm.Client, tools ...tool.Def) (*Manager, *Session) {
	t.Helper()
	return session(t, cl,
		core.WithTools(tools...),
		core.WithToolMiddleware(permission.Gate(permission.AskEverything(), permission.Defer())),
	)
}

// A call that needs a decision ends the turn rather than holding it open; the
// decision comes back as an event, and the turn resumes where it stopped: the
// call runs, its real output is what the model reads, and the answer is the
// reply to the message that started it all.
func TestADecisionResumesTheTurn(t *testing.T) {
	echo := &countingTool{}
	cl := script(toolTurn("u1", "echo", `{"x":1}`), textTurn("all done"))
	m, s := gated(t, cl, echo.def("echo"))

	msg, err := s.Send(testCtx(t), "run the tool")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	stopped := waitEvent(t, s, msg.ID)
	if stopped.State != Approving || stopped.Outcome != "deferred" || len(stopped.Approvals) != 1 {
		t.Fatalf("turn 1 = %+v, want it stopped on the call", stopped)
	}
	if got := s.Info().State; got != Approving {
		t.Fatalf("state = %q, want approving", got)
	}
	if echo.calls() != 0 {
		t.Fatal("the call ran before anyone decided")
	}

	p := s.Pending()
	if len(p) != 1 || p[0].UseID != "u1" || p[0].Tool != "echo" || string(p[0].Input) != `{"x":1}` || p[0].Since.IsZero() {
		t.Fatalf("pending = %+v, want the tool, the id and the arguments", p)
	}
	asked := outboxOf(t, m, s.ID(), MessageApproval)
	if !strings.Contains(asked.Text, "echo") || !strings.Contains(asked.Text, "u1") || !strings.Contains(string(asked.Data), `"use_id":"u1"`) {
		t.Fatalf("the approval message = %+v, want the call in it", asked)
	}

	if _, err := s.Resolve(testCtx(t), "nope", true); !errors.Is(err, ErrNoSuchApproval) {
		t.Fatalf("Resolve(unknown) = %v, want ErrNoSuchApproval", err)
	}
	decision, err := s.Resolve(testCtx(t), "u1", true)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if _, err := s.Resolve(testCtx(t), "u1", true); !errors.Is(err, ErrAlreadyAnswered) {
		t.Fatalf("second Resolve = %v, want ErrAlreadyAnswered", err)
	}

	if ended := waitEvent(t, s, decision.ID); ended.Outcome != "answer" || ended.Answer != "all done" {
		t.Fatalf("the resumed turn = %+v, want the answer", ended)
	}
	if echo.calls() != 1 {
		t.Fatalf("the approved tool ran %d times, want 1", echo.calls())
	}
	if got := len(s.Pending()); got != 0 {
		t.Fatalf("%d approvals still pending after the turn", got)
	}
	if !strings.Contains(transcript(s), "echoed") {
		t.Fatal("the tool's real output never reached the transcript")
	}
	reply := outboxOf(t, m, s.ID(), MessageReply)
	if reply.Text != "all done" || !slices.Equal(reply.InReplyTo, []int64{msg.ID}) {
		t.Fatalf("the reply = %+v, want the answer, in reply to the message %d", reply, msg.ID)
	}
}

func TestARefusalIsWhatTheModelReads(t *testing.T) {
	echo := &countingTool{}
	cl := script(toolTurn("u1", "echo", `{}`), textTurn("fine, I stopped"))
	_, s := gated(t, cl, echo.def("echo"))

	waitEvent(t, s, mustSend(t, s, "run the tool").ID)
	decision, err := s.Resolve(testCtx(t), "u1", false)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	waitEvent(t, s, decision.ID)

	if echo.calls() != 0 {
		t.Fatalf("the refused tool ran %d times", echo.calls())
	}
	if got := s.Info().State; got != Idle {
		t.Fatalf("state = %q; a refusal is a tool result, not a failed turn", got)
	}
	last := cl.request(t, 1).Messages
	if res := last[len(last)-1].Content[0].(llm.ToolResult); !res.IsError || !strings.Contains(res.Content, "refused") {
		t.Fatalf("the model read %+v, want the refusal", res)
	}
}

// Two calls stopped on at once resume once both are decided, not before.
func TestATurnResumesOnceEveryCallIsDecided(t *testing.T) {
	echo := &countingTool{}
	two := llm.Response{StopReason: llm.StopToolUse, Content: []llm.ContentBlock{
		llm.ToolUse{ID: "a", Name: "echo", Input: json.RawMessage(`{}`)},
		llm.ToolUse{ID: "b", Name: "echo", Input: json.RawMessage(`{}`)},
	}}
	cl := script(two, textTurn("both done"))
	_, s := gated(t, cl, echo.def("echo"))
	waitEvent(t, s, mustSend(t, s, "run both").ID)

	first, _ := s.Resolve(testCtx(t), "a", true)
	until(t, "the first decision to be recorded", func() bool { return len(s.Pending()) == 1 })
	if s.Info().State != Approving || echo.calls() != 0 {
		t.Fatalf("after one of two decisions: state %s, %d calls ran; want still waiting", s.Info().State, echo.calls())
	}
	if _, err := s.Resolve(testCtx(t), "a", false); !errors.Is(err, ErrAlreadyAnswered) {
		t.Fatalf("deciding a decided call again = %v, want ErrAlreadyAnswered", err)
	}
	second, _ := s.Resolve(testCtx(t), "b", true)
	if ended := waitEvent(t, s, second.ID); ended.Answer != "both done" {
		t.Fatalf("the resumed turn = %+v", ended)
	}
	if echo.calls() != 2 {
		t.Fatalf("%d calls ran, want both", echo.calls())
	}
	_ = first
}

// A session waiting on a decision takes nothing else: a message sent
// meanwhile waits in the inbox, and is the turn after the decision's.
func TestAMessageWaitsForTheDecision(t *testing.T) {
	echo := &countingTool{}
	cl := script(toolTurn("u1", "echo", `{}`), textTurn("ran it"), textTurn("and the rest"))
	m, s := gated(t, cl, echo.def("echo"))
	waitEvent(t, s, mustSend(t, s, "run the tool").ID)

	later := mustSend(t, s, "also, this")
	time.Sleep(30 * time.Millisecond)
	if pending, _ := m.Inbox(testCtx(t), EventFilter{Session: s.ID(), Status: EventPending}); len(pending) != 1 || pending[0].ID != later.ID {
		t.Fatalf("pending = %+v, want the message still waiting", pending)
	}
	if s.Info().Turns != 1 {
		t.Fatal("a message started a turn on a session waiting for a decision")
	}

	decision, _ := s.Resolve(testCtx(t), "u1", true)
	waitEvent(t, s, decision.ID)
	if ended := waitEvent(t, s, later.ID); ended.Answer != "and the rest" {
		t.Fatalf("the message's turn = %+v", ended)
	}
}

// ===== deleting, unloading, stopping =====

func TestDeleteForgetsEverything(t *testing.T) {
	echo := &countingTool{}
	cl := script(toolTurn("u1", "echo", `{}`))
	m, s := gated(t, cl, echo.def("echo"))
	waitEvent(t, s, mustSend(t, s, "run the tool").ID)
	if _, err := m.AddTimer(testCtx(t), Timer{Session: s.ID(), Note: "n", Due: time.Now().Add(time.Hour)}); err != nil {
		t.Fatal(err)
	}

	if err := m.Delete(testCtx(t), s.ID()); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := m.Get(testCtx(t), s.ID()); !errors.Is(err, ErrNoSuchSession) {
		t.Fatalf("Get after Delete = %v, want ErrNoSuchSession", err)
	}
	if _, err := s.Send(testCtx(t), "more"); !errors.Is(err, ErrNoSuchSession) {
		t.Fatalf("Send to a deleted session = %v, want ErrNoSuchSession", err)
	}
	if err := m.Delete(testCtx(t), s.ID()); !errors.Is(err, ErrNoSuchSession) {
		t.Fatalf("Delete twice = %v, want ErrNoSuchSession", err)
	}
	timers, _ := m.Timers(testCtx(t), TimerFilter{Session: s.ID()})
	msgs, _ := m.Outbox(testCtx(t), OutboxFilter{Session: s.ID()}, 0)
	events, _ := m.Inbox(testCtx(t), EventFilter{Session: s.ID()})
	if len(timers)+len(msgs)+len(events) != 0 {
		t.Fatalf("left behind %d timers, %d messages, %d events", len(timers), len(msgs), len(events))
	}

	// A follower on a deleted session is released rather than stranded.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range s.Follow(testCtx(t), 0) { //nolint:revive // draining is the point
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Follow did not return on a deleted session")
	}
}

// A session with nothing to do leaves memory, and nothing else: it is in the
// store, and comes back — transcript and all — when something wants it.
func TestAnIdleSessionLeavesMemoryNotTheStore(t *testing.T) {
	cl := script(textTurn("hi"), textTurn("still here"))
	m := newManager(t, ManagerConfig{Build: builder(cl), Idle: 40 * time.Millisecond})
	s, _ := m.Create(testCtx(t), SessionOptions{Title: "kept"})
	send(t, s, "hello")

	until(t, "the session to leave memory", func() bool {
		m.mu.Lock()
		defer m.mu.Unlock()
		return m.sessions[s.ID()] == nil
	})
	list, _ := m.List(testCtx(t))
	if len(list) != 1 || list[0].Title != "kept" || list[0].Turns != 1 {
		t.Fatalf("List = %+v, want the session, from the store", list)
	}
	back, err := m.Get(testCtx(t), s.ID())
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if back == s || len(back.Messages()) != 2 {
		t.Fatalf("loaded back %d messages, want the transcript", len(back.Messages()))
	}
	if ended := send(t, back, "again"); ended.Answer != "still here" || ended.Turn != 2 {
		t.Fatalf("the loaded session's turn = %+v", ended)
	}
	// The old one is not a dead end: a message sent to it reaches the store.
	if _, err := s.Send(testCtx(t), "through the old handle"); err != nil {
		t.Errorf("Send through an unloaded session = %v", err)
	}
}

// A running turn and a stream being followed both keep a session in memory.
func TestIdleLeavesBusySessionsAlone(t *testing.T) {
	cl := newBlockingClient()
	m := newManager(t, ManagerConfig{Build: builder(cl), Idle: 20 * time.Millisecond})
	s, _ := m.Create(testCtx(t), SessionOptions{})
	e, _ := s.Send(testCtx(t), "go")
	<-cl.entered
	time.Sleep(100 * time.Millisecond)
	if got, _ := m.Get(testCtx(t), s.ID()); got != s {
		t.Fatal("a session with a turn in flight left memory")
	}
	cl.release <- struct{}{}
	waitEvent(t, s, e.ID)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		for range s.Follow(ctx, 0) { //nolint:revive // following is the point
		}
	}()
	time.Sleep(100 * time.Millisecond)
	if got, _ := m.Get(testCtx(t), s.ID()); got != s {
		t.Fatal("a session being followed left memory")
	}
}

// Close stops what runs, and what it stopped is not lost: its events are back
// in the inbox, and the next daemon on the store runs them.
func TestCloseHandsTheTurnToTheNextDaemon(t *testing.T) {
	st := MemoryStore()
	cl := newBlockingClient()
	m, err := NewManager(ManagerConfig{Build: builder(cl), Store: st, Logger: quietLogger, Poll: 5 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}

	settle(t)
	s, _ := m.Create(testCtx(t), SessionOptions{})
	e, _ := s.Send(testCtx(t), "go")
	<-cl.entered

	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := m.Create(testCtx(t), SessionOptions{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Create after Close = %v, want ErrClosed", err)
	}
	events, _ := st.Events(testCtx(t), EventFilter{})
	if len(events) != 1 || events[0].Status != EventPending {
		t.Fatalf("after Close the event is %+v, want it pending again", events)
	}

	next := newManager(t, ManagerConfig{Build: builder(script(textTurn("picked up"))), Store: st})
	again, err := next.Get(testCtx(t), s.ID())
	if err != nil {
		t.Fatal(err)
	}
	if ended := waitEvent(t, again, e.ID); ended.Answer != "picked up" {
		t.Fatalf("the next daemon's turn = %+v", ended)
	}
	if got := strings.Count(transcript(again), "go"); got != 1 {
		t.Fatalf("the prompt is in the transcript %d times, want once", got)
	}
}

// A daemon that died mid-turn left its events running and its session saved
// as running, under a lease that has run out. The next one takes the session
// over, puts the events back, runs them, and numbers the log past anything
// the dead one might have handed out.
func TestACrashedTurnRunsAgain(t *testing.T) {
	st := MemoryStore()
	ctx := testCtx(t)
	rec := SessionRecord{Version: RecordVersion, ID: newID(), State: Running, Turns: 1, Events: 7,
		Created: time.Now(), Updated: time.Now(), Lease: Lease{Owner: "dead", Until: time.Now().Add(time.Second)}}
	if err := st.CreateSession(ctx, rec); err != nil {
		t.Fatal(err)
	}
	payload, _ := json.Marshal(MessagePayload{Text: "do it"})
	e, _, _ := st.Enqueue(ctx, Event{Session: rec.ID, Kind: KindMessage, Source: SourceAPI, Payload: payload})
	if got, _ := st.Claim(ctx, rec.ID, "dead", time.Now(), 10); len(got) != 1 {
		t.Fatal("the fixture did not claim the event")
	}
	if _, err := st.Acquire(ctx, rec.ID, Lease{Owner: "early"}, time.Now(), 3); !errors.Is(err, ErrOwned) {
		t.Fatalf("Acquire while the dead daemon's lease runs = %v, want ErrOwned", err)
	}
	time.Sleep(time.Second) // the lease runs out

	m := newManager(t, ManagerConfig{Build: builder(script(textTurn("done now"))), Store: st})
	s, err := m.Get(ctx, rec.ID)
	if err != nil {
		t.Fatal(err)
	}
	ended := waitEvent(t, s, e.ID)
	if ended.Answer != "done now" {
		t.Fatalf("the turn run again = %+v", ended)
	}
	if first := collect(t, s, 0, endOfTurn(2))[0]; first.Seq < 7+crashGap {
		t.Fatalf("the log restarted at %d, inside what the dead daemon may have issued", first.Seq)
	}
}

func TestManagerRequiresABuilder(t *testing.T) {
	if _, err := NewManager(ManagerConfig{}); err == nil {
		t.Fatal("NewManager accepted a config with no Build")
	}
}

func TestListIsNewestFirst(t *testing.T) {
	m := newManager(t, ManagerConfig{Build: builder(script())})

	var ids []string
	for i := range 3 {
		s, err := m.Create(testCtx(t), SessionOptions{Title: strconv.Itoa(i)})
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		ids = append(ids, s.ID())
		time.Sleep(time.Millisecond) // distinct creation instants
	}

	got, err := m.List(testCtx(t))
	if err != nil || len(got) != 3 {
		t.Fatalf("List = %d, %v", len(got), err)
	}
	for i, info := range got {
		if want := ids[len(ids)-1-i]; info.ID != want {
			t.Fatalf("List[%d] = %s, want %s (newest first)", i, info.ID, want)
		}
	}
}

// ===== spend and wire shapes =====

func TestSpendAccumulatesAcrossTurns(t *testing.T) {
	usage := llm.Usage{InputTokens: 100, OutputTokens: 10}
	priced := func(text string) llm.Response {
		r := textTurn(text)
		r.Usage = usage
		r.Model = "test-model"
		return r
	}
	pricing := llm.Table{"test-model": {In: 1000, Out: 1000}}
	cl := llm.Chain(script(priced("one"), priced("two")), core.TrackCost(pricing))

	_, s := session(t, cl)
	first := send(t, s, "a")
	second := send(t, s, "b")

	if first.Spend.InputTokens != 100 || second.Spend.InputTokens != 100 {
		t.Fatalf("per-turn spend = %+v / %+v, want 100 input tokens each", first.Spend, second.Spend)
	}
	total := s.Info().Spend
	if total.InputTokens != 200 || total.OutputTokens != 20 || total.Calls != 2 {
		t.Fatalf("session spend = %+v, want the sum of both turns", total)
	}
	if total.CostUSD <= 0 || total.ElapsedSec < 0 {
		t.Fatalf("session spend = %+v, want a positive cost", total)
	}
}

func TestTurnEventsUseTheStreamWireFormat(t *testing.T) {
	// The discriminator is spliced in first, and < > & survive — the same
	// contract as the harness's own events.
	got := string(mustMarshalNoEscape(t, TurnStarted{Turn: 1, Prompt: "hi <there>", Events: []int64{4}}))
	if want := `{"type":"turn_started","turn":1,"prompt":"hi <there>","events":[4]}`; got != want {
		t.Fatalf("TurnStarted = %s, want %s", got, want)
	}

	got = string(mustMarshalNoEscape(t, TurnEnded{Turn: 2, State: Idle, Outcome: "answer", Answer: "ok"}))
	if !strings.HasPrefix(got, `{"type":"turn_ended",`) {
		t.Fatalf("TurnEnded = %s, want the discriminator first", got)
	}
	for _, want := range []string{`"turn":2`, `"state":"idle"`, `"outcome":"answer"`} {
		if !strings.Contains(got, want) {
			t.Fatalf("TurnEnded = %s, missing %s", got, want)
		}
	}
	if strings.Contains(got, "retry_at") {
		t.Fatalf("TurnEnded = %s; a turn not to be retried says nothing about it", got)
	}

	for _, ev := range EventTypes() {
		if ev.Kind() == "" {
			t.Fatalf("%T has an empty Kind", ev)
		}
	}
}

// mustMarshalNoEscape encodes through an encoder with HTML escaping off, which
// is what a stream writer does.
func mustMarshalNoEscape(t *testing.T, v any) []byte {
	t.Helper()
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		t.Fatalf("encode %T: %v", v, err)
	}
	return bytes.TrimRight(buf.Bytes(), "\n")
}

func TestErrorKindIsBounded(t *testing.T) {
	cases := []struct {
		err  error
		want string
	}{
		{nil, ""},
		{governor.ErrBudgetExhausted, "budget_exhausted"},
		{fmt.Errorf("wrapped: %w", governor.ErrToolLoop), "tool_loop"},
		{core.ErrMaxIterations, "max_iterations"},
		{governor.ErrStepLimit, "max_iterations"},
		{permission.ErrDenied, "denied"},
		{llm.ErrContextWindow, "context_window"},
		{context.Canceled, "cancelled"},
		{errors.New("something else"), "error"},
	}
	for _, c := range cases {
		if got := ErrorKind(c.err); got != c.want {
			t.Errorf("ErrorKind(%v) = %q, want %q", c.err, got, c.want)
		}
	}
}

// ===== helpers =====

func mustSend(t *testing.T, s *Session, prompt string) Event {
	t.Helper()
	e, err := s.Send(testCtx(t), prompt)
	if err != nil {
		t.Fatalf("Send(%q): %v", prompt, err)
	}
	return e
}

// outboxOf waits for the session's first message of a kind.
func outboxOf(t *testing.T, m *Manager, session, kind string) Message {
	t.Helper()
	var found Message
	until(t, "a "+kind+" in the outbox", func() bool {
		msgs, _ := m.Outbox(testCtx(t), OutboxFilter{Session: session}, 0)
		for _, msg := range msgs {
			if msg.Kind == kind {
				found = msg
				return true
			}
		}
		return false
	})
	return found
}

// transcript flattens a session's messages, which is what "did the model see
// X" reduces to.
func transcript(s *Session) string {
	var b strings.Builder
	for _, m := range s.Messages() {
		for _, blk := range m.Content {
			switch v := blk.(type) {
			case llm.Text:
				b.WriteString(v.Text)
			case llm.ToolResult:
				b.WriteString(v.Content)
			}
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// A decision about a call the session is not waiting on — one that reached
// the inbox without Resolve's checks — is not recorded, and decides nothing.
func TestAStrayDecisionDecidesNothing(t *testing.T) {
	echo := &countingTool{}
	m, s := gated(t, script(toolTurn("u1", "echo", `{}`), textTurn("done")), echo.def("echo"))
	waitEvent(t, s, mustSend(t, s, "run the tool").ID)

	payload, _ := json.Marshal(ApprovalPayload{UseID: "not-u1", Allow: true})
	stray, _, err := m.enqueue(testCtx(t), Event{Session: s.ID(), Kind: KindApproval, Source: SourceAPI, Payload: payload})
	if err != nil {
		t.Fatal(err)
	}
	until(t, "the stray decision to be handled", func() bool {
		events, _ := m.Inbox(testCtx(t), EventFilter{Session: s.ID(), Status: EventDone})
		return slices.ContainsFunc(events, func(e Event) bool { return e.ID == stray.ID })
	})
	if got := s.Info().State; got != Approving || len(s.Pending()) != 1 || echo.calls() != 0 {
		t.Fatalf("after a stray decision: state %s, pending %+v, %d calls; want still waiting on u1", got, s.Pending(), echo.calls())
	}
	rec, _ := m.store.LoadSession(testCtx(t), s.ID())
	if len(rec.Verdicts) != 0 {
		t.Fatalf("the record kept %v, want no verdict for a call nobody waits on", rec.Verdicts)
	}
}
