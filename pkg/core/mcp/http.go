package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"wombat/internal/sse"
)

// errSessionExpired is the server answering 404 to a request that carried its
// session id: it has forgotten the session, and the client starts another.
var errSessionExpired = errors.New("the server ended the session")

// streamable is a server reached over Streamable HTTP: every message is a
// POST, answered with JSON or with an event stream that carries the response
// and anything the server sends before it.
type streamable struct {
	c       *Client
	url     string
	headers map[string]string
	client  *http.Client

	mu         sync.Mutex
	session    string
	version    string
	restarting bool // between reset and the new session's handshake
	closed     bool
}

func newHTTP(s Server, c *Client) (*streamable, error) {
	u, err := url.Parse(s.URL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("%q is not an http or https URL", s.URL)
	}
	// No client-wide timeout: a tool call can rightly take minutes, and the
	// call's own context is what bounds it.
	return &streamable{c: c, url: s.URL, headers: s.Headers, client: &http.Client{}}, nil
}

func (h *streamable) negotiated(version string) {
	h.mu.Lock()
	h.version, h.restarting = version, false
	h.mu.Unlock()
}

func (h *streamable) reset() {
	h.mu.Lock()
	h.session, h.version, h.restarting = "", "", true
	h.mu.Unlock()
}

// prepare sets the headers every request carries and returns the session id
// it was sent with.
func (h *streamable) prepare(req *http.Request, method string) (session string, err error) {
	for k, v := range h.headers {
		req.Header.Set(k, v)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return "", errClosed
	}
	if h.restarting && method != "initialize" {
		// Sent now it would carry no session at all, which a server refuses
		// with a 400 rather than the 404 that would start a session. Report
		// the expiry instead: the call then waits for the restart under way,
		// and is sent again inside the new session.
		return "", errSessionExpired
	}
	if h.session != "" {
		req.Header.Set("Mcp-Session-Id", h.session)
	}
	if h.version != "" {
		req.Header.Set("MCP-Protocol-Version", h.version)
	}
	return h.session, nil
}

func (h *streamable) send(ctx context.Context, m message) error {
	body, err := json.Marshal(m)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	session, err := h.prepare(req, m.Method)
	if err != nil {
		return err
	}
	resp, err := h.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if m.Method == "initialize" && resp.StatusCode/100 == 2 {
		if id := resp.Header.Get("Mcp-Session-Id"); id != "" {
			h.mu.Lock()
			h.session = id
			h.mu.Unlock()
		}
	}
	switch {
	case resp.StatusCode == http.StatusNotFound && session != "":
		return errSessionExpired
	case resp.StatusCode/100 != 2:
		return statusError(resp)
	}
	if !m.isRequest() {
		return nil // a notification or a response: 202 is all there is to hear
	}
	if resp.StatusCode == http.StatusAccepted {
		return errors.New("the server accepted the request and sent no response")
	}

	id, _ := parseID(m.ID)
	mediaType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	switch mediaType {
	case "text/event-stream":
		r := sse.NewReaderSize(resp.Body, maxMessage)
		for r.Next() {
			if h.answers(h.c.receive(r.Event().Data), id) {
				return nil // the server is done with this stream; so are we
			}
		}
		if err := r.Err(); err != nil {
			return fmt.Errorf("reading the server's event stream: %w", err)
		}
		return errors.New("the server's event stream ended without a response")
	case "application/json":
		data, err := io.ReadAll(io.LimitReader(resp.Body, maxMessage+1))
		if err != nil {
			return err
		}
		if len(data) > maxMessage {
			return fmt.Errorf("the response is larger than %d bytes", maxMessage)
		}
		if h.answers(h.c.receive(data), id) {
			return nil
		}
		return errors.New("the server's response did not answer the request")
	default:
		return fmt.Errorf("the server answered with Content-Type %q, which is neither JSON nor an event stream",
			resp.Header.Get("Content-Type"))
	}
}

// answers reports whether msgs hold the response to request id.
func (h *streamable) answers(msgs []message, id int64) bool {
	for _, m := range msgs {
		if got, ok := parseID(m.ID); ok && m.isResponse() && got == id {
			return true
		}
	}
	return false
}

func statusError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	msg := strings.TrimSpace(strings.ToValidUTF8(string(body), ""))
	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("the server refused the request (%s): check the credentials in the server's headers", resp.Status)
	}
	if msg == "" {
		return fmt.Errorf("the server answered %s", resp.Status)
	}
	return fmt.Errorf("the server answered %s: %s", resp.Status, msg)
}

// close deletes the session, as the specification asks a client that is done
// with one to. Best effort: a server that ignores it expires the session on
// its own.
func (h *streamable) close() error {
	h.mu.Lock()
	session := h.session
	version := h.version
	h.closed = true
	h.mu.Unlock()
	if session == "" {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, h.url, nil)
	if err != nil {
		return nil
	}
	for k, v := range h.headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Mcp-Session-Id", session)
	if version != "" {
		req.Header.Set("MCP-Protocol-Version", version)
	}
	if resp, err := h.client.Do(req); err == nil {
		resp.Body.Close()
	}
	return nil
}
