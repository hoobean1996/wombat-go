package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

// Webhook returns a [ManagerConfig.Deliver] that POSTs each message as JSON —
// the shape GET /api/outbox lists it in — to url, with headers set on every
// request: an Authorization the receiver checks, say. Any 2xx is delivered;
// anything else, or no answer within the delivery timeout, is tried again
// later.
//
// The receiver may see a message more than once — a daemon that dies between
// sending one and recording that it did sends it again — and its id is how it
// tells.
func Webhook(url string, headers map[string]string, client *http.Client) func(context.Context, Message) error {
	if client == nil {
		client = &http.Client{Timeout: deliverTimeout}
	}
	return func(ctx context.Context, m Message) error {
		body, err := json.Marshal(m)
		if err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Idempotency-Key", fmt.Sprintf("wombat-outbox-%d", m.ID))
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		if resp.StatusCode < 200 || resp.StatusCode > 299 {
			return fmt.Errorf("the webhook answered %s", resp.Status)
		}
		return nil
	}
}

// WakeRequest is what [WakeWebhook] posts: the session to advance, and when
// — now, or a time to come, for a queue that can hold a message until then.
// It is the body POST /api/advance takes, so a queue that forwards it as it
// is needs nothing in between.
type WakeRequest struct {
	SessionID string    `json:"session_id"`
	NotBefore time.Time `json:"not_before,omitzero"`
}

// WakeWebhook returns a [ManagerConfig.Wake] that POSTs a [WakeRequest] to
// url, with headers on every request: the publish endpoint of a queue that
// calls POST /api/advance with it — or that endpoint itself, for a host that
// has no queue. For a wake-up at a time to come, delayHeader, when set, is
// sent with the time as Unix seconds — "Upstash-Not-Before", for QStash —
// so the queue holds the call until then; without it the call is made now,
// and finds nothing to do yet but ask again.
//
// The call is made quickly or not at all: a lost wake-up is found by the next
// Tick.
func WakeWebhook(url string, headers map[string]string, delayHeader string, client *http.Client) func(context.Context, string, time.Time) error {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	return func(ctx context.Context, session string, at time.Time) error {
		w := WakeRequest{SessionID: session}
		if at.After(time.Now()) {
			w.NotBefore = at.UTC()
		}
		body, err := json.Marshal(w)
		if err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		if delayHeader != "" && !w.NotBefore.IsZero() {
			req.Header.Set(delayHeader, strconv.FormatInt(w.NotBefore.Unix(), 10))
		}
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		if resp.StatusCode < 200 || resp.StatusCode > 299 {
			return fmt.Errorf("the wake-up endpoint answered %s", resp.Status)
		}
		return nil
	}
}
