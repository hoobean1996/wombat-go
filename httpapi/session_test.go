package httpapi

// White-box tests (package httpapi, not httpapi_test).
//
// The lifecycle's interesting behaviour is unexported — the ring's eviction,
// the sweeper's clock, the parked-approval bookkeeping — and the point of
// putting the lifecycle in a type instead of a handler is that it can be
// driven directly. There is no network here and no http.Handler; the API
// layer is somebody else's file and somebody else's test.

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
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/permission"
	"github.com/hoobean1996/wombat-go/tool"
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
func builder(cl llm.Client, opts ...wombat.Option) func(SessionOptions) (*wombat.Agent, error) {
	return func(SessionOptions) (*wombat.Agent, error) {
		base := []wombat.Option{
			wombat.WithClient(cl),
			wombat.WithModel("test-model"),
			wombat.WithLogger(quietLogger),
		}
		return wombat.New(append(base, opts...)...)
	}
}

func newManager(t *testing.T, cfg ManagerConfig) *Manager {
	t.Helper()
	if cfg.Logger == nil {
		cfg.Logger = quietLogger
	}
	m, err := NewManager(cfg)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// session creates one session on a manager serving cl.
func session(t *testing.T, cl llm.Client, opts ...wombat.Option) (*Manager, *Session) {
	t.Helper()
	m := newManager(t, ManagerConfig{Build: builder(cl, opts...)})
	s, err := m.Create(SessionOptions{Title: "t"})
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

// waitTurn blocks until the given turn ends and returns its TurnEnded frame.
// It follows the log rather than polling, which is also a small proof that a
// follower sees turn boundaries at all.
func waitTurn(t *testing.T, s *Session, turn int) TurnEnded {
	t.Helper()
	for _, f := range s.Follow(testCtx(t), 0) {
		if te, ok := f.Event.(TurnEnded); ok && te.Turn == turn {
			return te
		}
	}
	t.Fatalf("turn %d never ended (state %s)", turn, s.Info().State)
	return TurnEnded{}
}

// send starts a turn and waits for it to end.
func send(t *testing.T, s *Session, prompt string) TurnEnded {
	t.Helper()
	turn, err := s.Send(prompt)
	if err != nil {
		t.Fatalf("Send(%q): %v", prompt, err)
	}
	return waitTurn(t, s, turn)
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

// The central claim of this package: two turns, one log, one unbroken
// sequence. A UI that reconnects across a turn boundary must not lose turn 1.
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

// An answer leaves the session Idle, not Done: a chat is not over because the
// model stopped talking.
func TestAnswerLeavesTheSessionOpen(t *testing.T) {
	_, s := session(t, script(textTurn("hi")))
	send(t, s, "hello")

	if got := s.Info().State; got != Idle {
		t.Fatalf("state after an answer = %q, want idle", got)
	}
	if _, err := s.Send("more"); err != nil {
		t.Fatalf("a second Send after an answer: %v", err)
	}
}

// A turn that ended in Paused leaves a dangling tool_use. The next turn has to
// close it, or the provider rejects the whole request.
func TestPausedTurnIsResumable(t *testing.T) {
	cl := script(toolTurn("u1", "ask_user", `{"question":"which one?"}`), textTurn("ok"))
	_, s := session(t, cl, wombat.WithTools(pauseTool("ask_user")))

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
	if err := wombat.Convo(msgs).Validate(); err != nil {
		t.Fatalf("turn 2 sent an invalid transcript: %v", err)
	}
	var closed bool
	for _, m := range msgs {
		for _, b := range m.Content {
			if tr, ok := b.(llm.ToolResult); ok && tr.ToolUseID == "u1" {
				closed = true
			}
		}
	}
	if !closed {
		t.Fatal("turn 2 did not answer the dangling tool_use left by the pause")
	}
}

// A terminal tool ends the CONVERSATION, and that is the only thing that does.
func TestSubmittedIsTerminal(t *testing.T) {
	cl := script(toolTurn("u1", "submit", `{"answer":42}`))
	_, s := session(t, cl, wombat.WithTools(terminalTool("submit")),
		wombat.WithTerminalTool("submit"))

	if got := send(t, s, "go").Outcome; got != "submitted" {
		t.Fatalf("outcome = %q, want submitted", got)
	}
	if got := s.Info().State; got != Done {
		t.Fatalf("state = %q, want done", got)
	}
	if _, err := s.Send("more"); !errors.Is(err, ErrDone) {
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

// A failed turn is NOT terminal: a provider hiccup must not cost the user the
// transcript.
func TestFailedTurnIsRetryable(t *testing.T) {
	cl := script() // out of turns on the first call
	_, s := session(t, cl)

	ended := send(t, s, "go")
	if ended.State != Failed || ended.Outcome != "" || ended.ErrorKind != "error" {
		t.Fatalf("turn 1 = %+v, want a failed turn with no outcome", ended)
	}
	info := s.Info()
	if info.State != Failed || info.Error == "" {
		t.Fatalf("info = %+v, want failed with a message", info)
	}
	if _, err := s.Send("try again"); err != nil {
		t.Fatalf("Send after a failure: %v", err)
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
	// blocking: the frames are already in hand, and a caller that asked for
	// them with a dead context is not harmed by receiving them.
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
	s, err := m.Create(SessionOptions{})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	send(t, s, "go")

	total := s.Info().Events
	if total <= ring {
		t.Fatalf("only %d frames; the ring of %d never overflowed", total, ring)
	}

	got := collect(t, s, 0, endOfTurn(1))
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
// turn is held mid-flight.
type blockingClient struct {
	entered chan struct{}
	release chan struct{}
}

func newBlockingClient() *blockingClient {
	return &blockingClient{entered: make(chan struct{}, 16), release: make(chan struct{})}
}

// Complete implements llm.Client.
func (b *blockingClient) Complete(ctx context.Context, _ llm.Request) (llm.Response, error) {
	b.entered <- struct{}{}
	select {
	case <-b.release:
		return textTurn("released"), nil
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
}

func TestSendIsBusyWhileATurnRuns(t *testing.T) {
	cl := newBlockingClient()
	_, s := session(t, cl)

	turn, err := s.Send("one")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	<-cl.entered

	if got := s.Info().State; got != Running {
		t.Fatalf("state = %q, want running", got)
	}
	if _, err := s.Send("two"); !errors.Is(err, ErrBusy) {
		t.Fatalf("second Send = %v, want ErrBusy", err)
	}

	close(cl.release)
	waitTurn(t, s, turn)

	if _, err := s.Send("two"); err != nil {
		t.Fatalf("Send once the turn is over: %v", err)
	}
}

// Cancel must leave nothing behind — not the run, not the goroutine draining
// it, not the tool call parked on a person.
func TestCancelReleasesGoroutines(t *testing.T) {
	cl := newBlockingClient()
	m := newManager(t, ManagerConfig{Build: builder(cl)})

	settle(t)
	before := runtime.NumGoroutine()

	const sessions = 8
	for range sessions {
		s, err := m.Create(SessionOptions{})
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		if _, err := s.Send("go"); err != nil {
			t.Fatalf("Send: %v", err)
		}
		<-cl.entered
	}
	if got := runtime.NumGoroutine(); got <= before {
		t.Fatalf("%d live turns added no goroutines (%d -> %d); the fixture is not testing anything",
			sessions, before, got)
	}

	for _, info := range m.List() {
		s, ok := m.Get(info.ID)
		if !ok {
			t.Fatalf("session %s vanished", info.ID)
		}
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

// ===== approvals =====

// gated builds a session whose every tool call is parked on a human.
func gated(t *testing.T, cl llm.Client, tools ...tool.Def) (*Manager, *Session) {
	t.Helper()
	return session(t, cl,
		wombat.WithTools(tools...),
		wombat.WithToolMiddleware(permission.Gate(permission.AskEverything(), Approver())),
	)
}

func TestApprovalRoundTrip(t *testing.T) {
	echo := &countingTool{}
	cl := script(toolTurn("u1", "echo", `{"x":1}`), textTurn("all done"))
	_, s := gated(t, cl, echo.def("echo"))

	turn, err := s.Send("run the tool")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}

	until(t, "the call to park on a human", func() bool { return len(s.Pending()) == 1 })

	p := s.Pending()[0]
	if p.UseID != "u1" || p.Tool != "echo" || string(p.Input) != `{"x":1}` {
		t.Fatalf("pending = %+v, want the tool, the id and the arguments", p)
	}
	if got := s.Info().State; got != Approving {
		t.Fatalf("state = %q, want approving", got)
	}
	if err := s.Resolve("nope", true); !errors.Is(err, ErrNoSuchApproval) {
		t.Fatalf("Resolve(unknown) = %v, want ErrNoSuchApproval", err)
	}

	if err := s.Resolve("u1", true); err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if err := s.Resolve("u1", true); !errors.Is(err, ErrAlreadyAnswered) {
		t.Fatalf("second Resolve = %v, want ErrAlreadyAnswered", err)
	}

	if ended := waitTurn(t, s, turn); ended.Outcome != "answer" {
		t.Fatalf("turn = %+v, want an answer", ended)
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
}

func TestApprovalDeniedDoesNotRunTheTool(t *testing.T) {
	echo := &countingTool{}
	cl := script(toolTurn("u1", "echo", `{}`), textTurn("fine, I stopped"))
	_, s := gated(t, cl, echo.def("echo"))

	turn, err := s.Send("run the tool")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	until(t, "the call to park", func() bool { return len(s.Pending()) == 1 })

	if err := s.Resolve("u1", false); err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	waitTurn(t, s, turn)

	if echo.calls() != 0 {
		t.Fatalf("the refused tool ran %d times", echo.calls())
	}
	if got := s.Info().State; got != Idle {
		t.Fatalf("state = %q; a refusal is a tool result, not a failed turn", got)
	}
}

// A parked call whose session is cancelled must be denied, not abandoned: a
// gate blocked on a channel nobody will ever write to is a leaked goroutine
// and a session stuck in Approving for ever.
func TestCancelDeniesPendingApprovals(t *testing.T) {
	echo := &countingTool{}
	cl := script(toolTurn("u1", "echo", `{}`), textTurn("unreachable"))
	_, s := gated(t, cl, echo.def("echo"))

	if _, err := s.Send("run the tool"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	until(t, "the call to park", func() bool { return len(s.Pending()) == 1 })

	settle(t)
	before := runtime.NumGoroutine()

	s.Cancel()
	until(t, "the cancelled turn to end", func() bool { return s.Info().State == Failed })

	if got := s.Pending(); len(got) != 0 {
		t.Fatalf("%d approvals still pending after Cancel", len(got))
	}
	if echo.calls() != 0 {
		t.Fatalf("the tool ran %d times despite the cancel", echo.calls())
	}
	if err := s.Resolve("u1", true); !errors.Is(err, ErrAlreadyAnswered) {
		t.Fatalf("answering a denied approval = %v, want ErrAlreadyAnswered", err)
	}

	settle(t)
	if after := runtime.NumGoroutine(); after > before {
		t.Fatalf("goroutines %d -> %d: the parked call leaked", before, after)
	}
}

// Deleting a session releases its parked calls too, through the same path.
func TestDeleteReleasesAParkedCall(t *testing.T) {
	echo := &countingTool{}
	cl := script(toolTurn("u1", "echo", `{}`), textTurn("unreachable"))
	m, s := gated(t, cl, echo.def("echo"))

	if _, err := s.Send("run the tool"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	until(t, "the call to park", func() bool { return len(s.Pending()) == 1 })

	if !m.Delete(s.ID()) {
		t.Fatal("Delete reported no such session")
	}
	if _, ok := m.Get(s.ID()); ok {
		t.Fatal("the session is still registered after Delete")
	}
	if _, err := s.Send("more"); !errors.Is(err, ErrNoSuchSession) {
		t.Fatalf("Send to a deleted session = %v, want ErrNoSuchSession", err)
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

// ===== the manager =====

func TestMaxSessions(t *testing.T) {
	m := newManager(t, ManagerConfig{Build: builder(script()), Max: 2})

	first, err := m.Create(SessionOptions{})
	if err != nil {
		t.Fatalf("Create 1: %v", err)
	}
	if _, err := m.Create(SessionOptions{}); err != nil {
		t.Fatalf("Create 2: %v", err)
	}
	if _, err := m.Create(SessionOptions{}); !errors.Is(err, ErrTooManySessions) {
		t.Fatalf("Create 3 = %v, want ErrTooManySessions", err)
	}

	m.Delete(first.ID())
	if _, err := m.Create(SessionOptions{}); err != nil {
		t.Fatalf("Create after a delete: %v", err)
	}
}

func TestTTLSweepsAQuietSession(t *testing.T) {
	m := newManager(t, ManagerConfig{Build: builder(script(textTurn("hi"))), TTL: 60 * time.Millisecond})

	s, err := m.Create(SessionOptions{})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	send(t, s, "hello")

	until(t, "the sweeper to reap a quiet session", func() bool {
		_, ok := m.Get(s.ID())
		return !ok
	})
	if got := len(m.List()); got != 0 {
		t.Fatalf("%d sessions still listed", got)
	}
}

// A running turn is never swept, however long it takes: it is bounded by its
// own budget, and reaping it would bill the user for work nobody receives.
func TestTTLDoesNotSweepARunningTurn(t *testing.T) {
	cl := newBlockingClient()
	m := newManager(t, ManagerConfig{Build: builder(cl), TTL: 30 * time.Millisecond})

	s, err := m.Create(SessionOptions{})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	turn, err := s.Send("go")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	<-cl.entered

	time.Sleep(150 * time.Millisecond)
	if _, ok := m.Get(s.ID()); !ok {
		t.Fatal("the sweeper reaped a session with a turn in flight")
	}

	close(cl.release)
	waitTurn(t, s, turn)
}

func TestManagerCloseStopsEverything(t *testing.T) {
	cl := newBlockingClient()
	m, err := NewManager(ManagerConfig{Build: builder(cl), Logger: quietLogger})
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}

	settle(t)
	before := runtime.NumGoroutine()

	s, err := m.Create(SessionOptions{})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := s.Send("go"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	<-cl.entered

	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := m.Create(SessionOptions{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Create after Close = %v, want ErrClosed", err)
	}

	settle(t)
	if after := runtime.NumGoroutine(); after > before {
		t.Fatalf("goroutines %d -> %d after Close; the sweeper or a turn survived", before, after)
	}
}

func TestManagerRequiresABuilder(t *testing.T) {
	if _, err := NewManager(ManagerConfig{}); err == nil {
		t.Fatal("NewManager accepted a config with no Build")
	}
}

func TestListIsOrderedAndDeleteReportsMisses(t *testing.T) {
	m := newManager(t, ManagerConfig{Build: builder(script())})

	var ids []string
	for i := range 3 {
		s, err := m.Create(SessionOptions{Title: strconv.Itoa(i)})
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		ids = append(ids, s.ID())
		time.Sleep(time.Millisecond) // distinct creation instants
	}

	got := m.List()
	if len(got) != 3 {
		t.Fatalf("List returned %d", len(got))
	}
	// Newest first: the session you are working in is the one you just made.
	for i, info := range got {
		want := ids[len(ids)-1-i]
		if info.ID != want {
			t.Fatalf("List[%d] = %s, want %s (newest first)", i, info.ID, want)
		}
	}
	if m.Delete("nope") {
		t.Fatal("Delete reported success for an unknown id")
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
	cl := llm.Chain(script(priced("one"), priced("two")), wombat.TrackCost(pricing))

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
	// contract as the harness's own events, asserted on the encoder path a
	// consumer uses. Plain json.Marshal re-escapes in its compact pass; that
	// is documented in event.go and is the caller's choice, not this type's.
	got := string(mustMarshalNoEscape(t, TurnStarted{Turn: 1, Prompt: "hi <there>"}))
	if want := `{"type":"turn_started","turn":1,"prompt":"hi <there>"}`; got != want {
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
		{wombat.ErrMaxIterations, "max_iterations"},
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
