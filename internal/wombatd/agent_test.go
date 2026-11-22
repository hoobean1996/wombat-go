package wombatd

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/mcp"
	"wombat/pkg/core/review"
	"wombat/pkg/core/tape"
	"wombat/pkg/core/tool"
	"wombat/pkg/core/trace"
	"wombat/pkg/daemon"
)

// TestMain lets this test binary stand in for an MCP server, so the tools.mcp
// wiring is tested against a real process speaking the real protocol.
func TestMain(m *testing.M) {
	if os.Getenv("WOMBATD_AS") == "mcp" {
		serveMCP(os.Stdin, os.Stdout)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// ===== a model that follows a script =====

type scripted struct {
	mu   sync.Mutex
	turn func(n int, req llm.Request) llm.Response
	seen []llm.Request
}

func (s *scripted) Complete(_ context.Context, req llm.Request) (llm.Response, error) {
	s.mu.Lock()
	n := len(s.seen)
	s.seen = append(s.seen, req)
	s.mu.Unlock()
	return s.turn(n, req), nil
}

func (s *scripted) requests() []llm.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.seen)
}

func say(text string) llm.Response {
	return llm.Response{Content: []llm.ContentBlock{llm.Text{Text: text}}, StopReason: llm.StopEndTurn}
}

func call(id, name string, input any) llm.Response {
	in, _ := json.Marshal(input)
	return llm.Response{Content: []llm.ContentBlock{llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: in}}, StopReason: llm.StopToolUse}
}

// testFactory is a factory over a scripted model, rooted at a fresh
// directory and configured with wombatd.yaml's defaults — the workspace
// permission among them, which lets an agent write in its workspace — on a
// shorter leash, and then as tweak says.
func testFactory(t *testing.T, cl llm.Client, tweak func(*config)) (*factory, string) {
	t.Helper()
	root := resolved(t, t.TempDir())
	cfg := defaults()
	cfg.Turn.MaxIters = 8
	if tweak != nil {
		tweak(&cfg)
	}
	return &factory{cfg: cfg, client: cl, root: root, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}, root
}

type turnResult struct {
	events  []core.Event
	outcome core.Outcome
	err     error
}

func (r turnResult) answer() string {
	if a, ok := r.outcome.(core.Answer); ok {
		return a.Text
	}
	return ""
}

func ofType[E core.Event](r turnResult) []E {
	var out []E
	for _, ev := range r.events {
		if e, ok := ev.(E); ok {
			out = append(out, e)
		}
	}
	return out
}

func runTurn(t *testing.T, f *factory, prompt string) turnResult {
	t.Helper()
	a, err := f.build(daemon.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	r := a.Start(ctx, core.Ask(prompt))
	defer r.Close()
	var res turnResult
	for r.Next() {
		res.events = append(res.events, r.Event())
	}
	res.outcome, res.err = r.Outcome(), r.Err()
	return res
}

// ===== the tools =====

// A command the model runs does not see the server's credentials: the runner
// every session gets filters the environment.
func TestToolsRunWithoutTheServersCredentials(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "sk-test-not-a-key")
	f, root := testFactory(t, &scripted{turn: func(int, llm.Request) llm.Response { return say("ok") }}, nil)
	a, err := f.build(daemon.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var ran bool
	for _, d := range a.Tools(context.Background()) {
		if d.Name != "bash" {
			continue
		}
		in, _ := json.Marshal(map[string]string{"command": "env > env.txt", "exec_dir": root})
		if _, err := d.Fn(context.Background(), in); err != nil {
			t.Fatal(err)
		}
		ran = true
	}
	if !ran {
		t.Fatal("a session under the workspace permission was offered no bash")
	}
	env, err := os.ReadFile(filepath.Join(root, "env.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(env), "sk-test-not-a-key") {
		t.Error("the model's command could read the provider key")
	}
	if !strings.Contains(string(env), "PATH=") {
		t.Error("the model's command lost PATH along with the credentials")
	}
}

// tools.skills_dir and tools.delegate both reach every session's tool list, and the
// skills' catalogue its system prompt.
func TestSkillsAndDelegationReachTheSession(t *testing.T) {
	cl := &scripted{turn: func(int, llm.Request) llm.Response { return say("ok") }}
	skills, err := loadSkills("../../skills", slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil || skills == nil {
		t.Fatalf("loading the shipped skills: %v", err)
	}
	f, _ := testFactory(t, cl, func(c *config) { c.Tools.Delegate = true })
	f.skills = skills
	if r := runTurn(t, f, "hello"); r.err != nil {
		t.Fatal(r.err)
	}
	req := cl.requests()[0]
	var names []string
	for _, s := range req.Tools {
		names = append(names, s.Name)
	}
	for _, want := range []string{"load_skill", "delegate", "bash"} {
		if !slices.Contains(names, want) {
			t.Errorf("tools %v are missing %s", names, want)
		}
	}
	if !strings.Contains(req.System, "git-history") {
		t.Error("the skills' catalogue is not in the system prompt")
	}
}

// ===== when a turn is done =====

// done.check runs whenever the model ends its turn, and a failing check sends it
// back with what the check said.
func TestCheckSendsTheModelBackUntilTheWorkIsDone(t *testing.T) {
	var root string
	cl := &scripted{turn: func(n int, req llm.Request) llm.Response {
		switch n {
		case 0:
			return say("finished") // it is not: done.txt does not exist
		case 1:
			return call("w1", "write_file", map[string]string{"path": filepath.Join(root, "done.txt"), "content": "yes"})
		default:
			return say("finished now")
		}
	}}
	f, r0 := testFactory(t, cl, func(c *config) { c.Done.Check = "test -f done.txt" })
	root = r0
	r := runTurn(t, f, "make done.txt")
	if r.err != nil || r.answer() != "finished now" {
		t.Fatalf("err %v, answer %q", r.err, r.answer())
	}
	if feedback := llm.TextOf(cl.requests()[1].Messages[len(cl.requests()[1].Messages)-1].Content); !strings.Contains(feedback, "not finished") || !strings.Contains(feedback, "test -f done.txt") {
		t.Errorf("the model was sent back with %q, want the check's verdict", feedback)
	}
	checks := ofType[core.CompletionChecked](r)
	if len(checks) != 2 || checks[0].Accepted || !checks[1].Accepted {
		t.Errorf("completion checks = %+v, want a rejection and then an acceptance", checks)
	}
}

// done.accept judges the answer on the workspace: an unticked TODO item is work,
// and the model is handed it.
func TestAcceptHandsAnUntickedTodoBack(t *testing.T) {
	var root string
	cl := &scripted{turn: func(n int, req llm.Request) llm.Response {
		switch n {
		case 0:
			return say("all done")
		case 1:
			return call("w1", "write_file", map[string]string{"path": filepath.Join(root, "TODO.md"), "content": "- [x] write the thing\n"})
		default:
			return say("all done now")
		}
	}}
	f, r0 := testFactory(t, cl, func(c *config) { c.Done.Accept = true })
	root = r0
	if err := os.WriteFile(filepath.Join(root, "TODO.md"), []byte("- [ ] write the thing\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	r := runTurn(t, f, "do the list")
	if r.err != nil || r.answer() != "all done now" {
		t.Fatalf("err %v, answer %q", r.err, r.answer())
	}
	checks := ofType[core.CompletionChecked](r)
	if len(checks) != 2 || checks[0].Accepted || !strings.Contains(checks[0].Feedback, "write the thing") || !checks[1].Accepted {
		t.Errorf("completion checks = %+v, want the unticked item handed back, then an acceptance", checks)
	}
}

// review.enabled hands an answered turn to a fresh reviewer, and what it found
// reaches the stream after the builder's work.
func TestReviewReportsOnTheStream(t *testing.T) {
	cl := &scripted{turn: func(n int, req llm.Request) llm.Response {
		if n == 0 {
			return say("built it")
		}
		return say("I read the code and found nothing I could demonstrate.")
	}}
	f, _ := testFactory(t, cl, func(c *config) { c.Review.Enabled = true })
	r := runTurn(t, f, "build the thing")
	if r.err != nil || r.answer() != "built it" {
		t.Fatalf("err %v, answer %q", r.err, r.answer())
	}
	reviews := ofType[review.Reviewed](r)
	if len(reviews) != 1 || !reviews[0].Ran || reviews[0].Sweeps != 1 {
		t.Fatalf("reviewed events = %+v, want one review that ran one sweep", reviews)
	}
	reqs := cl.requests()
	if len(reqs) < 2 {
		t.Fatalf("%d model calls: the reviewer never ran", len(reqs))
	}
	if task := llm.TextOf(reqs[1].Messages[0].Content); !strings.Contains(task, "build the thing") {
		t.Errorf("the reviewer was not told the task: %q", task)
	}
}

// Review mode builds nothing: the session's agent is offered no tools, and
// the review runs on the code as it is.
func TestReviewOnlyBuildsNothing(t *testing.T) {
	cl := &scripted{turn: func(n int, req llm.Request) llm.Response { return say("noted") }}
	f, _ := testFactory(t, cl, func(c *config) { c.Mode, c.Done.Check, c.Done.Accept = modeReview, "false", true })
	r := runTurn(t, f, "a todo CLI with persistence")
	if r.err != nil {
		t.Fatal(r.err)
	}
	first := cl.requests()[0]
	if len(first.Tools) != 0 {
		t.Errorf("the session's own agent was offered %d tools", len(first.Tools))
	}
	if !strings.Contains(first.System, "This session reviews code") {
		t.Error("the agent was not told what the session is for")
	}
	if len(ofType[review.Reviewed](r)) != 1 {
		t.Error("no review ran")
	}
	if checks := ofType[core.CompletionChecked](r); len(checks) != 1 || !checks[0].Accepted {
		t.Errorf("completion checks = %+v: done.check and done.accept judge a build, and there was none", checks)
	}
}

// Plan mode asks for a plan, and does not hold the plan to the gate a build
// is held to.
func TestPlanOnlyAsksForAPlanAndSkipsTheGate(t *testing.T) {
	cl := &scripted{turn: func(int, llm.Request) llm.Response { return say("planned") }}
	f, root := testFactory(t, cl, func(c *config) { c.Mode, c.Done.Accept = modePlan, true })
	if err := os.WriteFile(filepath.Join(root, "TODO.md"), []byte("- [ ] everything\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	r := runTurn(t, f, "a todo CLI")
	if r.err != nil || r.answer() != "planned" {
		t.Fatalf("err %v, answer %q", r.err, r.answer())
	}
	if !strings.Contains(cl.requests()[0].System, "Plan the task the user gives you") {
		t.Error("the planning instructions are not in the system prompt")
	}
	if checks := ofType[core.CompletionChecked](r); len(checks) != 0 {
		t.Errorf("a plan was put through the acceptance gate: %+v", checks)
	}
}

// ===== what is kept =====

// record.tape and record.trace see every session: the tool calls land on the tape and
// the spans in the trace.
func TestTapeAndTraceSeeTheSession(t *testing.T) {
	dir := t.TempDir()
	tp, err := tape.Open(filepath.Join(dir, "run.jsonl"), tape.Record)
	if err != nil {
		t.Fatal(err)
	}
	sink, closer, err := trace.FileSink(filepath.Join(dir, "trace.ndjson"))
	if err != nil {
		t.Fatal(err)
	}
	var root string
	cl := &scripted{turn: func(n int, req llm.Request) llm.Response {
		if n == 0 {
			return call("v1", "view_file", map[string]string{"path": filepath.Join(root, "notes.txt")})
		}
		return say("read it")
	}}
	f, r0 := testFactory(t, cl, nil)
	root = r0
	f.tape, f.tracer = tp, trace.New(sink)
	if err := os.WriteFile(filepath.Join(root, "notes.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	if r := runTurn(t, f, "read notes.txt"); r.err != nil {
		t.Fatal(r.err)
	}
	if err := tp.Close(); err != nil {
		t.Fatal(err)
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "run.jsonl")); !strings.Contains(string(b), `"kind":"tool"`) {
		t.Errorf("the tape recorded no tool call:\n%s", b)
	}
	if spans, err := trace.ReadFile(filepath.Join(dir, "trace.ndjson")); err != nil || len(spans) == 0 {
		t.Errorf("the trace has %d spans, %v", len(spans), err)
	}
	if err := renderTrace(filepath.Join(dir, "trace.ndjson"), filepath.Join(dir, "trace.html")); err != nil {
		t.Errorf("rendering the trace: %v", err)
	}
}

// ===== the MCP servers =====

func TestMCPServersReachTheSession(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	conns, err := mcp.ConnectAll(ctx, []mcp.Server{{
		Name: "greeter", Command: os.Args[0],
		Env: map[string]string{"WOMBATD_AS": "mcp", "GORACE": "atexit_sleep_ms=0"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer conns.Close()

	cl := &scripted{turn: func(n int, req llm.Request) llm.Response {
		if n == 0 {
			return call("m1", "greeter__greet", map[string]string{"name": "wombat"})
		}
		last := req.Messages[len(req.Messages)-1]
		for _, b := range last.Content {
			if tr, ok := b.(llm.ToolResult); ok {
				return say("the server said: " + strings.TrimSpace(tr.Content))
			}
		}
		return say("no result")
	}}
	f, _ := testFactory(t, cl, nil)
	f.mcp = conns
	r := runTurn(t, f, "greet")
	if r.err != nil {
		t.Fatal(r.err)
	}
	first := cl.requests()[0]
	if !slices.ContainsFunc(first.Tools, func(s llm.ToolSpec) bool { return s.Name == "greeter__greet" }) {
		t.Error("the server's tool was not offered")
	}
	if !strings.Contains(first.System, "Greet people by name.") {
		t.Error("the server's instructions are not in the system prompt")
	}
	if got := r.answer(); got != "the server said: hello, wombat" {
		t.Errorf("answer = %q", got)
	}
}

// serveMCP is the smallest stdio MCP server the MCP test needs: one tool,
// and instructions for the system prompt.
func serveMCP(in io.Reader, out io.Writer) {
	sc := bufio.NewScanner(in)
	enc := json.NewEncoder(out)
	for sc.Scan() {
		var m struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params struct {
				Arguments struct {
					Name string `json:"name"`
				} `json:"arguments"`
			} `json:"params"`
		}
		if json.Unmarshal(sc.Bytes(), &m) != nil || len(m.ID) == 0 {
			continue // notifications
		}
		var result any
		switch m.Method {
		case "initialize":
			result = map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{"tools": map[string]any{}},
				"serverInfo": map[string]any{"name": "greeter", "version": "0"}, "instructions": "Greet people by name."}
		case "tools/list":
			result = map[string]any{"tools": []any{map[string]any{
				"name": "greet", "description": "Greet someone.",
				"inputSchema": map[string]any{"type": "object", "properties": map[string]any{"name": map[string]string{"type": "string"}}},
				"annotations": map[string]any{"readOnlyHint": true, "openWorldHint": false},
			}}}
		case "tools/call":
			result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "hello, " + m.Params.Arguments.Name}}}
		default:
			_ = enc.Encode(map[string]any{"jsonrpc": "2.0", "id": m.ID, "error": map[string]any{"code": -32601, "message": "no"}})
			continue
		}
		_ = enc.Encode(map[string]any{"jsonrpc": "2.0", "id": m.ID, "result": result})
	}
}

// The notice has to arrive while there is still room to act on it. At two
// iterations left the only way to obey is to stop, which is the failure it is
// meant to prevent.
func TestIterationNoticeArrivesWhileThereIsRoomToAct(t *testing.T) {
	n := iterationNotice(80)
	ctx := context.Background()

	if got := n(ctx, 40); got != "" {
		t.Errorf("warned at half the budget: %q", got)
	}
	if got := n(ctx, 53); got != "" {
		t.Errorf("warned just under two thirds: %q", got)
	}
	got := n(ctx, 54)
	if got == "" {
		t.Fatal("silent at two thirds of the budget")
	}
	if !strings.Contains(got, "26 left") {
		t.Errorf("the remaining count is wrong or missing: %q", got)
	}
	if !strings.Contains(got, "does not exist yet") {
		t.Error("the notice says how much is left but not to check for missing work")
	}
	if !strings.Contains(got, "Do not shorten a checklist") {
		t.Error("nothing forbids the shortcut both measured runs took")
	}
}

func TestIterationNoticeDoesNotGoNegativeOrPanic(t *testing.T) {
	n := iterationNotice(10)
	for _, iter := range []int{9, 10, 11, 50} {
		got := n(context.Background(), iter)
		if got == "" {
			t.Fatalf("iteration %d produced no notice", iter)
		}
		var left int
		if _, err := fmt.Sscanf(got[strings.Index(got, "; "):], "; %d left", &left); err != nil {
			t.Fatalf("iteration %d: could not read the remainder from %q", iter, got)
		}
		if left < 0 {
			t.Errorf("iteration %d reported %d left", iter, left)
		}
	}
	if iterationNotice(0) != nil {
		t.Error("an unbounded run should get no iteration notice")
	}
}

// ===== full access =====

// runAs is runTurn for a session that asked for a permission.
func runAs(t *testing.T, f *factory, permission, prompt string) turnResult {
	t.Helper()
	a, err := f.build(daemon.SessionOptions{Permission: permission})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	r := a.Start(ctx, core.Ask(prompt))
	defer r.Close()
	var res turnResult
	for r.Next() {
		res.events = append(res.events, r.Event())
	}
	res.outcome, res.err = r.Outcome(), r.Err()
	return res
}

// Full access is the operator's to offer. Offered, a session that asks for it
// keeps the whole tool set and is asked about none of it: the command runs.
// Not offered, the request gets the operator's policy, under which the same
// command waits for a person. "off" stays out of a client's reach either way.
// tools.disable takes a tool away from every session, whatever its policy:
// Full Access does not bring it back.
func TestADisabledToolIsOfferedToNoSession(t *testing.T) {
	f, _ := testFactory(t, &scripted{turn: func(int, llm.Request) llm.Response { return say("ok") }}, func(c *config) {
		c.Tools.FullAccess = true
		c.Tools.Disable = []string{"write_file", "str_replace"}
	})
	for _, permission := range []string{"", "full", "readonly"} {
		a, err := f.build(daemon.SessionOptions{Permission: permission})
		if err != nil {
			t.Fatal(err)
		}
		var names []string
		for _, d := range a.Tools(context.Background()) {
			names = append(names, d.Name)
		}
		if slices.Contains(names, "write_file") || slices.Contains(names, "str_replace") {
			t.Errorf("permission %q: %v still offers a disabled tool", permission, names)
		}
		if !slices.Contains(names, "view_file") || !slices.Contains(names, "bash") {
			t.Errorf("permission %q: %v lost a tool nobody disabled", permission, names)
		}
	}
}

func TestFullAccessIsTheOperatorsToOffer(t *testing.T) {
	touch := func(root string) *scripted {
		return &scripted{turn: func(i int, _ llm.Request) llm.Response {
			if i == 0 {
				return call("c1", "bash", map[string]string{"command": "touch made-it", "exec_dir": root})
			}
			return say("done")
		}}
	}

	offered, root := testFactory(t, nil, func(c *config) { c.Tools.FullAccess = true })
	offered.client = touch(root)
	if got := offered.policy("full"); got != "full" {
		t.Fatalf("policy(full) with full_access = %q", got)
	}
	if got := offered.policy("off"); got != "workspace" {
		t.Fatalf("policy(off) = %q, want the operator's: off is never a client's to ask for", got)
	}
	res := runAs(t, offered, "full", "make the file")
	if res.err != nil || res.answer() != "done" {
		t.Fatalf("under full access: outcome %#v, err %v; want the command run and the answer", res.outcome, res.err)
	}
	if _, err := os.Stat(filepath.Join(root, "made-it")); err != nil {
		t.Fatalf("the command did not run under full access: %v", err)
	}

	refused, root := testFactory(t, nil, nil)
	refused.client = touch(root)
	if got := refused.policy("full"); got != "workspace" {
		t.Fatalf("policy(full) without full_access = %q, want the operator's", got)
	}
	res = runAs(t, refused, "full", "make the file")
	if _, ok := res.outcome.(core.Deferred); !ok {
		t.Fatalf("without full_access: outcome %#v, err %v; want the command waiting for a person", res.outcome, res.err)
	}
	if _, err := os.Stat(filepath.Join(root, "made-it")); err == nil {
		t.Fatal("the command ran on a server that does not offer full access")
	}
}

// tools.recovery: ask turns the report of an interrupted call into a
// question, for the tools whose calls would be reported, and only those.
func TestAskOnRecoveryAsksOnlyWhatWouldBeReported(t *testing.T) {
	in := []tool.Def{
		{Name: "write", Caps: tool.CapMutating},
		{Name: "read", Caps: tool.CapReadOnly},
		{Name: "send", Caps: tool.CapNetwork, Idempotent: true},
		{Name: "pay", Caps: tool.CapNetwork, Recovery: tool.RecoverReport},
	}
	got := askOnRecovery(in)
	want := map[string]tool.Recovery{"write": tool.RecoverAsk, "read": tool.RecoverRerun, "send": tool.RecoverRerun, "pay": tool.RecoverReport}
	for _, d := range got {
		if r := tool.RecoveryOf(d); r != want[d.Name] {
			t.Errorf("%s recovers by %v, want %v", d.Name, r, want[d.Name])
		}
	}
	if in[0].Recovery != tool.RecoverAuto {
		t.Error("the list it was given was changed")
	}
}
