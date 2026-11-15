package wombat

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// verdicts is a check that returns its answers in order, then accepts.
func verdicts(answers ...string) (CompletionCheck, *[]Completion) {
	var seen []Completion
	return func(_ context.Context, c Completion) (string, error) {
		seen = append(seen, c)
		if len(answers) == 0 {
			return "", nil
		}
		a := answers[0]
		answers = answers[1:]
		return a, nil
	}, &seen
}

func completionEvents(evs []Event) []CompletionChecked {
	var out []CompletionChecked
	for _, ev := range evs {
		if c, ok := ev.(CompletionChecked); ok {
			out = append(out, c)
		}
	}
	return out
}

func TestCompletionCheckSendsAnUnfinishedAnswerBack(t *testing.T) {
	check, seen := verdicts("TestParse fails: got 3, want 4")
	cl := scripted(textTurn("all done"), textTurn("fixed TestParse too"))
	run := newAgent(t, cl, nil, WithCompletionCheck(3, check)).Start(context.Background(), Ask("fix the parser"))
	_, evs := drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	if a, ok := run.Outcome().(Answer); !ok || a.Text != "fixed TestParse too" {
		t.Fatalf("outcome = %#v, want the answer given after the feedback", run.Outcome())
	}
	if len(*seen) != 2 || (*seen)[0].Answer != "all done" {
		t.Errorf("the check saw %+v", *seen)
	}
	if !strings.Contains(allText(cl.request(t, 1).Messages), "TestParse fails: got 3, want 4") {
		t.Error("the model was not told what the check found")
	}
	got := completionEvents(evs)
	if len(got) != 2 || got[0].Accepted || !got[1].Accepted {
		t.Errorf("events = %+v, want a rejection then an acceptance", got)
	}
	if err := Convo(run.Messages()).Validate(); err != nil {
		t.Errorf("transcript invalid: %v", err)
	}
}

func TestCompletionCheckGivesUpAfterItsAttempts(t *testing.T) {
	check := func(context.Context, Completion) (string, error) { return "still failing", nil }
	cl := scripted(textTurn("done"), textTurn("done, again"))
	run := newAgent(t, cl, nil, WithCompletionCheck(1, check)).Start(context.Background(), Ask("go"))
	_, evs := drain(t, run)
	if a, ok := run.Outcome().(Answer); !ok || a.Text != "done, again" {
		t.Fatalf("outcome = %#v, want the run to end after one rejection", run.Outcome())
	}
	got := completionEvents(evs)
	if len(got) != 2 || !got[1].GaveUp || got[1].Accepted {
		t.Errorf("events = %+v, want the last one to record giving up", got)
	}
}

func TestCompletionCheckRejectsASubmission(t *testing.T) {
	check, _ := verdicts("the report is missing the totals")
	cl := scripted(
		toolTurn("s1", "submit", `{"report":"v1"}`),
		toolTurn("s2", "submit", `{"report":"v2"}`),
	)
	run := newAgent(t, cl, []tool.Def{terminalTool("submit")}, WithCompletionCheck(2, check)).
		Start(context.Background(), Ask("write the report"))
	drain(t, run)
	if err := run.Err(); err != nil {
		t.Fatal(err)
	}
	sub, ok := run.Outcome().(Submitted)
	if !ok || string(sub.Payload) != `{"report":"v2"}` {
		t.Fatalf("outcome = %#v, want the second submission", run.Outcome())
	}
	second := cl.request(t, 1)
	last := second.Messages[len(second.Messages)-1].Content[0].(llm.ToolResult)
	if !last.IsError || !strings.Contains(last.Content, "missing the totals") {
		t.Errorf("the rejected submission was answered with %+v", last)
	}
}

func TestACompletionCheckThatCannotRunDoesNotHoldTheRun(t *testing.T) {
	check := func(context.Context, Completion) (string, error) { return "", errors.New("runner is gone") }
	run := newAgent(t, scripted(textTurn("done")), nil, WithCompletionCheck(3, check)).Start(context.Background(), Ask("go"))
	_, evs := drain(t, run)
	if _, ok := run.Outcome().(Answer); !ok {
		t.Fatalf("outcome = %#v, want the answer", run.Outcome())
	}
	if got := completionEvents(evs); len(got) != 1 || got[0].Error == "" {
		t.Errorf("events = %+v, want the check's failure reported", got)
	}
}

// sh is a Sheller over /bin/sh, standing in for builtin.OSRunner.
type sh struct{}

func (sh) Shell(ctx context.Context, script string) (string, error) {
	out, err := exec.CommandContext(ctx, "/bin/sh", "-c", script).CombinedOutput()
	return string(out), err
}

func TestCommandCheck(t *testing.T) {
	ctx := context.Background()
	if fb, err := CommandCheck(sh{}, "exit 0", 0)(ctx, Completion{}); fb != "" || err != nil {
		t.Errorf("passing command: %q, %v", fb, err)
	}
	fb, err := CommandCheck(sh{}, "echo FAIL: TestX; exit 3", 0)(ctx, Completion{})
	if err != nil || !strings.Contains(fb, "status 3") || !strings.Contains(fb, "FAIL: TestX") {
		t.Errorf("failing command: %q, %v", fb, err)
	}
	fb, err = CommandCheck(sh{}, "sleep 5", 50*time.Millisecond)(ctx, Completion{})
	if err != nil || !strings.Contains(fb, "did not finish") {
		t.Errorf("hanging command: %q, %v", fb, err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := CommandCheck(sh{}, "sleep 5", 0)(cancelled, Completion{}); err == nil {
		t.Error("a cancelled run's check returned a verdict instead of the cancellation")
	}
}
