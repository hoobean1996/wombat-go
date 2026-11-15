package daemon

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"
)

// render is what a turn's model is told about the events it was given.
//
// One message on its own is just its text: a conversation reads as one. More
// than that, or anything that is not a person talking, is a list of events,
// each in a tag that says what it is and where it came from — a webhook's
// payload is not the user's instruction, and the model should be able to see
// which is which.
func render(events []Event, paused bool) string {
	if len(events) == 1 {
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
	return fmt.Sprintf("<event id=\"%d\" kind=%q source=%q at=%q>\n%s\n</event>",
		e.ID, e.Kind, e.Source, e.Created.Format(time.RFC3339), body)
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
