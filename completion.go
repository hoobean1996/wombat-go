package wombat

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

// Completion is what a [CompletionCheck] is shown: how the model wants to end
// the run, and the transcript that got it there.
type Completion struct {
	// Answer is the model's final text, when it ended its turn.
	Answer string
	// Submitted is the terminal call, when it called the terminal tool.
	Submitted *Submitted
	Messages  []llm.Message
}

// A CompletionCheck decides whether a run the model is about to end is done.
// It returns "" to let it end, or what is still wrong — a failing test's
// output, a file that was not written — which the model is told before it is
// sent back to work.
//
// An error means the check itself could not run, and is not a verdict on the
// work: the run ends as the model wanted, and the failure is logged and
// reported in the [CompletionChecked] event.
type CompletionCheck func(ctx context.Context, c Completion) (string, error)

// WithCompletionCheck runs check whenever the model ends the run — a final
// answer, or a call to the terminal tool — and sends it back to work when check
// says it is not done, at most attempts times.
//
// The motivating failure is the benchmark report's "solved but did not finish"
// and its mirror image, "finished but did not solve": a model is a poor judge
// of when it is done, and a run that ends because the model SAID so is only as
// good as that judgement. A check that runs the tests is a better one, and it
// is cheap to consult at the one moment it matters.
//
// For a final answer the feedback becomes a new user turn and the run carries
// on. For the terminal tool it becomes the call's tool_result, as an error, so
// the model reads why its submission was not accepted — exactly as it would a
// refusal from a permission policy, which is also a reason not to end.
//
// After attempts rejections the run ends as the model wanted, and the last
// [CompletionChecked] event says the check still disagreed. Ending is the
// safer failure: a check that can never be satisfied — a test the task does not
// cover, an environment that is broken — would otherwise hold the run until
// its iteration cap, at full price, for nothing.
func WithCompletionCheck(attempts int, check CompletionCheck) Option {
	return func(c *config) {
		if check == nil || attempts <= 0 {
			c.check = nil
			return
		}
		c.check = &completionCheck{attempts: attempts, fn: check}
	}
}

type completionCheck struct {
	attempts int
	fn       CompletionCheck
}

// CompletionChecked reports one consultation of a [CompletionCheck]. Accepted
// is the verdict; GaveUp is set on a rejection that no longer sends the model
// back, because it has used its attempts.
type CompletionChecked struct {
	Attempt  int    `json:"attempt"`
	Accepted bool   `json:"accepted"`
	Feedback string `json:"feedback,omitempty"`
	GaveUp   bool   `json:"gave_up,omitempty"`
	Error    string `json:"error,omitempty"`
	Millis   int64  `json:"ms"`
}

// Kind implements Event.
func (CompletionChecked) Kind() string { return "completion_checked" }

// MarshalJSON implements json.Marshaler.
func (e CompletionChecked) MarshalJSON() ([]byte, error) {
	type r CompletionChecked
	return eventJSON(e.Kind(), r(e))
}

// checkCompletion consults the agent's check, if it has one, and returns the
// feedback to send the model back with — "" when the run may end.
func (r *Run) checkCompletion(ctx context.Context, c Completion) string {
	cc := r.a.cfg.check
	if cc == nil {
		return ""
	}
	started := time.Now()
	feedback, err := cc.fn(ctx, c)
	feedback = strings.TrimSpace(feedback)
	ev := CompletionChecked{Attempt: r.rejections + 1, Millis: millis(time.Since(started))}
	switch {
	case err != nil:
		ev.Error = err.Error()
		if ctx.Err() == nil {
			slog.WarnContext(ctx, "wombat: the completion check failed to run; ending as the model asked",
				slog.String("agent", r.a.cfg.name), slog.Any("err", err))
		}
		r.emit(ev)
		return ""
	case feedback == "":
		ev.Accepted = true
		r.emit(ev)
		return ""
	}
	ev.Feedback = feedback
	if r.rejections >= cc.attempts {
		ev.GaveUp = true
		r.emit(ev)
		return ""
	}
	r.rejections++
	r.emit(ev)
	return feedback
}

// notDoneYet is the user turn a rejected final answer is followed by.
func notDoneYet(feedback string) string {
	return "The task is not finished yet. A check of your work says:\n\n" + feedback +
		"\n\nCarry on, and end your turn again once this is resolved."
}

// errNotAccepted is how a rejected submission reads to the model.
func errNotAccepted(feedback string) error {
	return fmt.Errorf("submission not accepted — a check of your work says:\n\n%s\n\nFix this and submit again", feedback)
}

// Sheller runs a shell script and returns its combined output. builtin.Runner
// is one — OSRunner, or SandboxRunner when the work must stay confined while
// it is checked.
type Sheller interface {
	Shell(ctx context.Context, script string) (string, error)
}

// CommandCheck is a [CompletionCheck] that runs script — `go test ./...`,
// `make check` — and passes when it exits 0. When it fails, the feedback is its
// output, cut to the last few kilobytes, which is where a test run keeps the
// failure. timeout bounds one run of it; 0 means two minutes.
//
// It runs code the agent wrote, so run it through the same Runner the agent's
// own commands go through: a check that escapes the sandbox the agent was
// kept in is a way out of it.
func CommandCheck(sh Sheller, script string, timeout time.Duration) CompletionCheck {
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}
	return func(parent context.Context, _ Completion) (string, error) {
		ctx, cancel := context.WithTimeout(parent, timeout)
		defer cancel()
		out, err := sh.Shell(ctx, script)
		if err == nil {
			return "", nil
		}
		if parent.Err() != nil {
			return "", context.Cause(parent) // the run is ending, not the check failing
		}
		var ec interface{ ExitCode() int }
		switch {
		case ctx.Err() != nil:
			// A check that hangs is a verdict too: the tests do not finish.
			return fmt.Sprintf("`%s` did not finish within %s:\n\n%s", script, timeout, tail(out)), nil
		case errors.As(err, &ec):
			return fmt.Sprintf("`%s` exited with status %d:\n\n%s", script, ec.ExitCode(), tail(out)), nil
		default:
			return "", fmt.Errorf("wombat: completion check %q: %w", script, err)
		}
	}
}

// tail keeps the end of a command's output, where a test run puts its failure.
func tail(out string) string {
	const keep = 4 << 10
	out = strings.TrimSpace(out)
	if len(out) > keep {
		out = "…" + strings.ToValidUTF8(out[len(out)-keep:], "")
	}
	return out
}
