package wombat

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// gatedTool blocks until release is closed, reporting on started when it runs,
// so a test can act while a tool call is genuinely in flight.
func gatedTool(name string) (tool.Def, chan struct{}, chan struct{}) {
	started, release := make(chan struct{}, 4), make(chan struct{})
	return tool.Def{
		Name: name, Description: "waits", InputSchema: json.RawMessage(objSchema),
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			started <- struct{}{}
			select {
			case <-release:
			case <-ctx.Done():
			}
			return "slow result", nil
		},
	}, started, release
}

func TestSteerIsDeliveredWithTheToolResults(t *testing.T) {
	slow, started, release := gatedTool("slow")
	cl := scripted(toolTurn("u1", "slow", `{}`), textTurn("switched to v2"))
	run := newAgent(t, cl, []tool.Def{slow}).Start(context.Background(), Ask("migrate the client"))

	var kinds []string
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		kinds, _ = drain(t, run)
	}()
	<-started
	if !run.Steer("use the v2 API instead") {
		t.Fatal("a running run refused a steering message")
	}
	close(release)
	wg.Wait()

	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	second := cl.request(t, 1)
	last := second.Messages[len(second.Messages)-1]
	if last.Role != llm.RoleUser || !strings.Contains(allText([]llm.Message{last}), "use the v2 API instead") {
		t.Errorf("the next request's last turn = %+v, want the results with the message", last)
	}
	if !strings.Contains(strings.Join(kinds, ","), "steered") {
		t.Errorf("events = %v, want a steered event", kinds)
	}
	if err := Convo(run.Messages()).Validate(); err != nil {
		t.Errorf("transcript invalid after steering: %v", err)
	}
}

// heldClient answers its first call only when told to, so a test can steer
// while the model is writing what would have been its final answer.
type heldClient struct {
	entered chan struct{}
	release chan struct{}
	next    *scriptedClient
	once    sync.Once
}

func (h *heldClient) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	first := false
	h.once.Do(func() { first = true })
	if first {
		h.entered <- struct{}{}
		<-h.release
	}
	return h.next.Complete(ctx, req)
}

func TestSteerDuringTheFinalAnswerIsAnsweredToo(t *testing.T) {
	next := scripted(textTurn("here is the v1 plan"), textTurn("ok, v2 then"))
	cl := &heldClient{entered: make(chan struct{}, 1), release: make(chan struct{}), next: next}
	run := newAgent(t, cl, nil).Start(context.Background(), Ask("plan the migration"))

	done := make(chan struct{})
	go func() { defer close(done); drain(t, run) }()
	<-cl.entered
	run.Steer("actually, target v2")
	close(cl.release)
	<-done

	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	if a, ok := run.Outcome().(Answer); !ok || a.Text != "ok, v2 then" {
		t.Errorf("outcome = %#v, want the answer to the steering message", run.Outcome())
	}
	msgs := run.Messages()
	if len(msgs) != 4 || !strings.Contains(allText(msgs[2:3]), "actually, target v2") {
		t.Errorf("transcript = %d messages, want prompt, first answer, steering, second answer", len(msgs))
	}
	if err := Convo(msgs).Validate(); err != nil {
		t.Errorf("transcript invalid: %v", err)
	}
}

func TestSteerAfterTheRunIsRefusedAndAPausedRunHandsItBack(t *testing.T) {
	run := newAgent(t, scripted(textTurn("done")), nil).Start(context.Background(), Ask("hi"))
	drain(t, run)
	if run.Steer("too late") {
		t.Error("a finished run accepted a steering message")
	}

	// A message queued while the call that pauses is in flight never reaches
	// the model: it is handed back.
	next := scripted(toolTurn("p1", "ask", `{"question":"which?"}`))
	cl := &heldClient{entered: make(chan struct{}, 1), release: make(chan struct{}), next: next}
	paused := newAgent(t, cl, []tool.Def{pauseTool("ask")}).Start(context.Background(), Ask("go"))
	done := make(chan struct{})
	go func() { defer close(done); drain(t, paused) }()
	<-cl.entered
	if !paused.Steer("and hurry") {
		t.Fatal("a running run refused a steering message")
	}
	close(cl.release)
	<-done
	if _, ok := paused.Outcome().(Paused); !ok {
		t.Fatalf("outcome = %#v, want Paused", paused.Outcome())
	}
	if got := paused.Undelivered(); len(got) != 1 || got[0] != "and hurry" {
		t.Errorf("Undelivered = %q, want the message handed back", got)
	}
}
