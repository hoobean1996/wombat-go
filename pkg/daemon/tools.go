package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"wombat/pkg/core/tool"
)

// scope is the session a turn runs for, as the daemon's tools find it.
type scope struct {
	m       *Manager
	session string
	thread  string
}

type scopeKey struct{}

// withScope attaches the turn's session to its context. On the context, not
// in a closure, because [ManagerConfig.Build] makes the agent and its tools
// without ever seeing the session it is for.
func withScope(ctx context.Context, sc *scope) context.Context {
	return context.WithValue(ctx, scopeKey{}, sc)
}

var errNoScope = errors.New("this tool works only in a turn the daemon runs")

func scopeFrom(ctx context.Context) (*scope, error) {
	sc, _ := ctx.Value(scopeKey{}).(*scope)
	if sc == nil {
		return nil, errNoScope
	}
	return sc, nil
}

// The names of the daemon's tools.
const (
	ToolSendMessage   = "send_message"
	ToolSetTimer      = "set_timer"
	ToolCancelTimer   = "cancel_timer"
	ToolListTimers    = "list_timers"
	ToolWaitForResult = "wait_for_result"
)

// ToolNames lists the names of [Tools], for a permission policy to let
// through: they touch nothing on the host, and an agent that has to ask
// before it may say anything, or before it may remind itself of something,
// is not one anybody can leave running.
func ToolNames() []string {
	return []string{ToolSendMessage, ToolSetTimer, ToolCancelTimer, ToolListTimers, ToolWaitForResult}
}

// Tools are how the agent of a daemon session speaks, and how it wakes
// itself up: send_message posts to the outbox; set_timer, cancel_timer and
// list_timers manage the session's timers; and wait_for_result ends the turn
// until a result from outside comes in for the call. Install them in the
// agent [ManagerConfig.Build] makes; they work only in a turn the daemon
// runs, which is where they find their session.
//
// All of them are safe to run again — the same call sends once, sets one
// timer and cancels it once, however often it runs — so a turn carried on
// after its daemon stopped simply runs them again.
func Tools() []tool.Def {
	return []tool.Def{sendMessage(), setTimer(), cancelTimer(), listTimers(), waitForResult()}
}

func sendMessage() tool.Def {
	return tool.Def{
		Name: ToolSendMessage,
		Description: "Send a message to the people this session works for. Your final answer to a message " +
			"they sent reaches them on its own; use this for what nobody asked for yet — what a timer's check " +
			"found, that a long job has finished, that you have started on something. A turn woken by a timer " +
			"or an event says nothing to anyone unless it sends a message.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{
			"text":{"type":"string","description":"What to say: plain text or Markdown."}},
			"required":["text"],"additionalProperties":false}`),
		Caps:       tool.CapNetwork,
		Idempotent: true,
		Fn: func(ctx context.Context, in json.RawMessage) (string, error) {
			sc, err := scopeFrom(ctx)
			if err != nil {
				return "", err
			}
			var args struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal(in, &args); err != nil {
				return "", fmt.Errorf("bad arguments: %w", err)
			}
			if strings.TrimSpace(args.Text) == "" {
				return "", errors.New("text is empty")
			}
			msg := Message{Session: sc.session, Thread: sc.thread, Kind: MessageSent, Text: args.Text}
			if key := tool.CallKey(ctx); key != "" {
				// The same call, retried or run again by a turn carried on,
				// sends once.
				msg.DedupKey = "send:" + sc.session + ":" + key
			}
			got, _, err := sc.m.post(ctx, msg)
			if err != nil {
				return "", err
			}
			return fmt.Sprintf("sent (message %d)", got.ID), nil
		},
	}
}

func setTimer() tool.Def {
	return tool.Def{
		Name: ToolSetTimer,
		Description: "Wake yourself up later: once, to follow up on something — check whether a deploy " +
			"finished, see whether anyone answered — or every so often, for a recurring task. When it fires " +
			"you get a turn, with its note. Give exactly one of in or at.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{
			"note":{"type":"string","description":"What to do or check when it fires. You will be shown it then, with nothing else to remind you, so make it say everything."},
			"in":{"type":"string","description":"How long from now, as a Go duration: \"90s\", \"10m\", \"2h30m\"."},
			"at":{"type":"string","description":"When, as an RFC 3339 time: \"2026-11-16T09:00:00Z\"."},
			"every":{"type":"string","description":"Fire again at this interval after the first time, as a Go duration of at least 1m: \"24h\". Leave it out for a one-off follow-up."}},
			"required":["note"],"additionalProperties":false}`),
		Caps:       tool.CapMeta,
		Idempotent: true,
		Fn: func(ctx context.Context, in json.RawMessage) (string, error) {
			sc, err := scopeFrom(ctx)
			if err != nil {
				return "", err
			}
			var args struct {
				Note  string `json:"note"`
				In    string `json:"in"`
				At    string `json:"at"`
				Every string `json:"every"`
			}
			if err := json.Unmarshal(in, &args); err != nil {
				return "", fmt.Errorf("bad arguments: %w", err)
			}
			t := Timer{Session: sc.session, Note: args.Note, CreatedBy: "agent"}
			if key := tool.CallKey(ctx); key != "" {
				// The same call, run again, sets one timer.
				t.DedupKey = "timer:" + sc.session + ":" + key
			}
			if t.Due, err = DueAt(args.In, args.At, time.Now()); err != nil {
				return "", err
			}
			if args.Every != "" {
				if t.Every, err = time.ParseDuration(args.Every); err != nil {
					return "", fmt.Errorf("every: %w", err)
				}
			}
			got, err := sc.m.AddTimer(ctx, t)
			if err != nil {
				return "", err
			}
			return "set " + describeTimer(got), nil
		},
	}
}

// cancelled reports whether the session's timer id is cancelled.
func cancelled(ctx context.Context, sc *scope, id int64) bool {
	timers, err := sc.m.Timers(ctx, TimerFilter{Session: sc.session, Status: TimerCancelled})
	return err == nil && slices.ContainsFunc(timers, func(t Timer) bool { return t.ID == id })
}

func cancelTimer() tool.Def {
	return tool.Def{
		Name:        ToolCancelTimer,
		Description: "Cancel a timer this session set, by the id set_timer or list_timers gave.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{
			"id":{"type":"integer","description":"The timer's id."}},
			"required":["id"],"additionalProperties":false}`),
		Caps:       tool.CapMeta,
		Idempotent: true,
		Fn: func(ctx context.Context, in json.RawMessage) (string, error) {
			sc, err := scopeFrom(ctx)
			if err != nil {
				return "", err
			}
			var args struct {
				ID int64 `json:"id"`
			}
			if err := json.Unmarshal(in, &args); err != nil {
				return "", fmt.Errorf("bad arguments: %w", err)
			}
			if err := sc.m.CancelTimer(ctx, args.ID, sc.session); err != nil {
				// Cancelled already — by this same call, run again by a turn
				// carried on — is cancelled.
				if !errors.Is(err, ErrNoSuchTimer) || !cancelled(ctx, sc, args.ID) {
					return "", err
				}
			}
			return fmt.Sprintf("cancelled timer %d", args.ID), nil
		},
	}
}

func listTimers() tool.Def {
	return tool.Def{
		Name:        ToolListTimers,
		Description: "List this session's timers that have not fired for the last time, the soonest first.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
		Caps:        tool.CapReadOnly,
		Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			sc, err := scopeFrom(ctx)
			if err != nil {
				return "", err
			}
			timers, err := sc.m.Timers(ctx, TimerFilter{Session: sc.session, Status: TimerActive})
			if err != nil {
				return "", err
			}
			if len(timers) == 0 {
				return "no timers set", nil
			}
			lines := make([]string, len(timers))
			for i, t := range timers {
				lines[i] = describeTimer(t)
			}
			return strings.Join(lines, "\n"), nil
		},
	}
}

func waitForResult() tool.Def {
	return tool.Def{
		Name: ToolWaitForResult,
		Description: "Wait for a result that arrives from outside this session: a build or a job you started, " +
			"a check someone runs, an answer that comes back later. Your turn ends here, and nothing runs " +
			"meanwhile; when the result is posted for this call, the turn carries on with it as this call's " +
			"output. Give the handle of what you wait for — a job id, a URL — and a note saying what the result " +
			"should be, for whoever posts it.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{
			"handle":{"type":"string","description":"What you wait for: a job id, a build URL, a ticket."},
			"note":{"type":"string","description":"What result you need, for whoever or whatever posts it."}},
			"required":["note"],"additionalProperties":false}`),
		Caps:       tool.CapMeta,
		Idempotent: true,
		Fn: func(ctx context.Context, in json.RawMessage) (string, error) {
			if _, err := scopeFrom(ctx); err != nil {
				return "", err
			}
			var args struct {
				Handle string `json:"handle"`
				Note   string `json:"note"`
			}
			if err := json.Unmarshal(in, &args); err != nil {
				return "", fmt.Errorf("bad arguments: %w", err)
			}
			if strings.TrimSpace(args.Note) == "" {
				return "", errors.New("note is empty")
			}
			return "", tool.Await(args.Handle, args.Note)
		},
	}
}

func describeTimer(t Timer) string {
	s := fmt.Sprintf("timer %d, due %s", t.ID, t.Due.Format(time.RFC3339))
	if t.Every > 0 {
		s += ", every " + t.Every.String()
	}
	return s + ": " + t.Note
}

// DueAt is when a timer set "in" from now, or "at" a time, is due: exactly
// one of the two, in a Go duration or an RFC 3339 time.
func DueAt(in, at string, now time.Time) (time.Time, error) {
	switch {
	case (in == "") == (at == ""):
		return time.Time{}, fmt.Errorf("%w: give exactly one of in and at", ErrBadRequest)
	case in != "":
		d, err := time.ParseDuration(in)
		if err != nil || d <= 0 {
			return time.Time{}, fmt.Errorf("%w: in %q is not a duration from now", ErrBadRequest, in)
		}
		return now.Add(d), nil
	default:
		t, err := time.Parse(time.RFC3339, at)
		if err != nil {
			return time.Time{}, fmt.Errorf("%w: at %q is not an RFC 3339 time", ErrBadRequest, at)
		}
		return t, nil
	}
}
