package daemon

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// render is what a turn's model is told about the events it was given.
//
// One direct message on its own is just its text: a conversation reads as
// one. More than that, or anything else, is a list of events, each in a tag
// that says what it is, where it came from and who sent it — a webhook's
// payload is not the user's instruction, a message relayed from a chat is not
// necessarily from the user either, and the model should be able to see which
// is which.
func render(events []Event, paused bool) string {
	if len(events) == 1 && direct(events[0]) {
		if text, ok := messageText(events[0]); ok {
			return text
		}
	}
	var b strings.Builder
	switch {
	case paused && !slices.ContainsFunc(events, func(e Event) bool { return e.Kind == KindMessage }):
		b.WriteString("The user has not answered your question yet. Meanwhile, this arrived:\n\n")
	case len(events) > 1:
		fmt.Fprintf(&b, "%d events arrived since your last turn, oldest first:\n\n", len(events))
	}
	for i, e := range events {
		if i > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(eventBlock(e))
	}
	return b.String()
}

func eventBlock(e Event) string {
	var body string
	switch e.Kind {
	case KindMessage:
		body, _ = messageText(e)
	case KindTimer:
		var p TimerPayload
		_ = json.Unmarshal(e.Payload, &p)
		repeat := ""
		if p.Every != "" {
			repeat = ", and fires again every " + p.Every
		}
		body = fmt.Sprintf("A timer you set has fired (timer %d, due %s%s). Its note:\n%s",
			p.Timer, p.Due.Format(time.RFC3339), repeat, p.Note)
	default:
		body = string(e.Payload)
		if body == "" {
			body = "(no payload)"
		}
	}
	// A payload is whatever its sender put in it, and the closing tag is the
	// one thing it must not be able to say.
	body = strings.ReplaceAll(body, "</event", `<\/event`)
	from := ""
	if e.From != "" {
		from = fmt.Sprintf(" from=%q", e.From)
	}
	return fmt.Sprintf("<event id=\"%d\" kind=%q source=%q%s at=%q>\n%s\n</event>",
		e.ID, e.Kind, e.Source, from, e.Created.Format(time.RFC3339), body)
}

// direct is a person talking to the session through its own API without
// naming themselves: the conversation the session was made for. A message
// relayed from somewhere else, or from someone named, says so instead, so
// the model can tell a chat from a webhook and one speaker from another.
func direct(e Event) bool {
	return (e.Kind == KindMessage || e.Kind == "") && (e.Source == SourceAPI || e.Source == "") && e.From == ""
}

// maxFrom bounds Event.From: a name, not a message.
const maxFrom = 128

// CheckFrom refuses a From that is not a short line of text: it is shown to
// the model inside a tag, and to people in a list of who said what.
func CheckFrom(from string) error {
	switch {
	case from == "":
		return nil
	case len(from) > maxFrom:
		return fmt.Errorf("%w: from is longer than %d bytes", ErrBadRequest, maxFrom)
	case !utf8.ValidString(from):
		return fmt.Errorf("%w: from is not UTF-8", ErrBadRequest)
	case strings.TrimSpace(from) != from:
		return fmt.Errorf("%w: from %q starts or ends with a space", ErrBadRequest, from)
	case strings.ContainsFunc(from, unicode.IsControl):
		return fmt.Errorf("%w: from %q has a control character in it", ErrBadRequest, from)
	}
	return nil
}

// messageText is a message event's text.
func messageText(e Event) (string, bool) {
	if e.Kind != KindMessage && e.Kind != "" {
		return "", false
	}
	var p MessagePayload
	if err := json.Unmarshal(e.Payload, &p); err != nil {
		return "", false
	}
	return p.Text, true
}
