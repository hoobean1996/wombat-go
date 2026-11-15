package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
