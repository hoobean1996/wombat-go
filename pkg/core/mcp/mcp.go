// Package mcp connects an agent to Model Context Protocol servers: it starts
// or reaches a server, lists the tools it offers, and turns each one into a
// tool.Def the agent dispatches like any other.
//
//	servers, _ := mcp.LoadConfig("mcp.json") // the usual {"mcpServers": {...}} file
//	conns, err := mcp.ConnectAll(ctx, servers)
//	defer conns.Close()
//	tools := append(builtin.Default(deps), conns.Tools()...)
//
// Both transports in the specification are here: stdio, for a server the
// harness starts as a child process, and Streamable HTTP, for one that is
// already running somewhere. Tools are the only capability used — no
// resources, prompts or sampling — because tools are what an agent loop
// consumes; a request from the server for anything else is answered "method
// not found", which the protocol allows. Authorization is a header in the
// configuration (a bearer token, say); the OAuth flow is not implemented.
//
// # Capabilities come from the server's annotations
//
// A tool's readOnlyHint and openWorldHint become tool.CapReadOnly (or
// tool.CapMutating) and tool.CapNetwork, so the permission policies, the
// read-only filters and the parallel scheduler treat an MCP tool exactly as
// they would a built-in one, and a tool that says nothing is taken to mutate
// and to reach the network, as the specification's defaults have it. The
// hints are, however, the SERVER's claims about its own tools, and a policy
// that trusts them trusts the server. For a server you did not write, set
// [Server.IgnoreHints]: every tool is then taken to mutate and reach the
// network whatever it says, and a policy that asks about such calls asks about
// every one.
//
// # The environment of a stdio server
//
// A stdio server gets the same filtered environment as the agent's own
// commands — the process basics and toolchain settings, none of the harness's
// credentials — plus exactly what its configuration names in env. A server
// that needs a token is given that token, not every token the harness holds;
// the configuration file can pass one through from the harness's environment
// as "${NAME}".
package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"wombat/pkg/core/tool"
)

// ProtocolVersion is the revision of the specification this client speaks.
const ProtocolVersion = "2025-06-18"

// Server describes one MCP server: a command to start for stdio, or a URL to
// reach over Streamable HTTP. Exactly one of Command and URL is set.
type Server struct {
	// Name identifies the server, and prefixes its tools' names — "github"
	// makes create_issue into github__create_issue — so two servers can offer
	// tools of the same name.
	Name string

	// Command, Args, Env and Dir start a stdio server. Env is added to the
	// filtered environment described in the package doc.
	Command string
	Args    []string
	Env     map[string]string
	Dir     string

	// URL and Headers reach a Streamable HTTP server.
	URL     string
	Headers map[string]string

	// IgnoreHints treats every tool as mutating and networked whatever its
	// annotations say; see the package doc.
	IgnoreHints bool

	// Timeout bounds one tool call, and the handshake. 0 leaves calls to the
	// agent's per-call fallback and gives the handshake thirty seconds.
	Timeout time.Duration
}

// Client is a connection to one server. It is safe for concurrent use: calls
// are multiplexed over the one connection by their JSON-RPC ids.
type Client struct {
	server Server
	t      transport
	log    *slog.Logger

	nextID  atomic.Int64
	mu      sync.Mutex
	pending map[int64]chan message
	err     error // why the connection ended; nil while it is up
	info    serverInfo

	initMu       sync.Mutex
	sessions     atomic.Uint64 // handshakes completed, so a restart happens once
	toolsChanged atomic.Bool
}

// transport carries messages to the server. Messages from it are handed to
// the Client's receive.
type transport interface {
	// send delivers m. For a request over HTTP it returns once the response
	// has been received and dispatched; over stdio, once m is written.
	send(ctx context.Context, m message) error
	close() error
}

// sessioned is a transport whose server keeps a session it can expire.
type sessioned interface {
	negotiated(version string)
	reset()
}

type serverInfo struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    json.RawMessage `json:"capabilities"`
	ServerInfo      struct {
		Name    string `json:"name"`
		Version string `json:"version"`
	} `json:"serverInfo"`
	Instructions string `json:"instructions,omitempty"`
}

// Connect starts or reaches s and completes the handshake. The connection
// lives until [Client.Close], independent of ctx, which bounds only the
// handshake.
func Connect(ctx context.Context, s Server) (*Client, error) {
	if s.Name == "" {
		return nil, errors.New("mcp: a server needs a name")
	}
	c := &Client{
		server:  s,
		log:     slog.Default().With(slog.String("mcp_server", s.Name)),
		pending: map[int64]chan message{},
	}
	var err error
	switch {
	case s.Command != "" && s.URL == "":
		var st *stdio
		if st, err = startStdio(s); err == nil {
			c.t = st
			st.listen(c)
		}
	case s.URL != "" && s.Command == "":
		c.t, err = newHTTP(s, c)
	default:
		err = errors.New("needs exactly one of a command and a url")
	}
	if err != nil {
		return nil, fmt.Errorf("mcp: server %q: %w", s.Name, err)
	}

	hctx, cancel := context.WithTimeout(ctx, handshakeTimeout(s))
	defer cancel()
	c.initMu.Lock()
	err = c.initialize(hctx)
	c.initMu.Unlock()
	if err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("mcp: server %q: %w", s.Name, err)
	}
	return c, nil
}

func handshakeTimeout(s Server) time.Duration {
	if s.Timeout > 0 {
		return s.Timeout
	}
	return 30 * time.Second
}

// initialize runs the handshake. The caller holds initMu.
func (c *Client) initialize(ctx context.Context) error {
	params := map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]string{"name": "wombat", "version": "1"},
	}
	var info serverInfo
	if err := c.roundTrip(ctx, "initialize", params, &info); err != nil {
		return fmt.Errorf("initialize: %w", err)
	}
	if info.ProtocolVersion == "" {
		return errors.New("initialize: the server did not say which protocol version it speaks")
	}
	c.mu.Lock()
	c.info = info
	c.mu.Unlock()
	if s, ok := c.t.(sessioned); ok {
		s.negotiated(info.ProtocolVersion)
	}
	if err := c.notify(ctx, "notifications/initialized", nil); err != nil {
		return fmt.Errorf("initialized: %w", err)
	}
	c.sessions.Add(1)
	return nil
}

// Name is the server's configured name.
func (c *Client) Name() string { return c.server.Name }

// Instructions is what the server asked to have put in front of the model, if
// anything — the "instructions" of its initialize result.
func (c *Client) Instructions() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.info.Instructions
}

// Close ends the connection: calls in flight fail, a stdio server is asked to
// exit and then made to, and an HTTP session is deleted.
func (c *Client) Close() error {
	c.fail(errClosed)
	return c.t.close()
}

var errClosed = errors.New("the connection was closed")

// fail ends the connection with err, releasing every call waiting on it. The
// first reason sticks.
func (c *Client) fail(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return
	}
	c.err = err
	for id, ch := range c.pending {
		close(ch)
		delete(c.pending, id)
	}
}

// ===== JSON-RPC =====

type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

func (m message) isResponse() bool { return m.Method == "" && len(m.ID) > 0 }
func (m message) isRequest() bool  { return m.Method != "" && len(m.ID) > 0 }

type rpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *rpcError) Error() string { return fmt.Sprintf("%s (JSON-RPC error %d)", e.Message, e.Code) }

const (
	errInvalidParams  = -32602
	errMethodNotFound = -32601
)

// parseID reads a JSON-RPC id. This client only issues integers, but a server
// that echoes one back as a string is still answering it.
func parseID(raw json.RawMessage) (int64, bool) {
	var n int64
	if json.Unmarshal(raw, &n) == nil {
		return n, true
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return 0, false
	}
	n, err := strconv.ParseInt(s, 10, 64)
	return n, err == nil
}

// call sends a request and waits for its response, starting a new session
// once if the server says it has forgotten the old one.
func (c *Client) call(ctx context.Context, method string, params, result any) error {
	session := c.sessions.Load()
	err := c.roundTrip(ctx, method, params, result)
	if !errors.Is(err, errSessionExpired) {
		return err
	}
	if rerr := c.restart(ctx, session); rerr != nil {
		return fmt.Errorf("%w, and starting a new one failed: %w", err, rerr)
	}
	return c.roundTrip(ctx, method, params, result)
}

// restart starts a new session, unless another call already has since
// session was the current one.
func (c *Client) restart(ctx context.Context, session uint64) error {
	c.initMu.Lock()
	defer c.initMu.Unlock()
	if c.sessions.Load() != session {
		return nil
	}
	if s, ok := c.t.(sessioned); ok {
		s.reset()
	}
	c.log.Info("mcp: the server ended the session; starting a new one")
	return c.initialize(ctx)
}

func (c *Client) roundTrip(ctx context.Context, method string, params, result any) error {
	id := c.nextID.Add(1)
	m := message{JSONRPC: "2.0", ID: json.RawMessage(strconv.FormatInt(id, 10)), Method: method}
	if params != nil {
		raw, err := json.Marshal(params)
		if err != nil {
			return err
		}
		m.Params = raw
	}

	ch := make(chan message, 1)
	c.mu.Lock()
	if c.err != nil {
		err := c.err
		c.mu.Unlock()
		return err
	}
	c.pending[id] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}()

	if err := c.t.send(ctx, m); err != nil {
		if ctx.Err() != nil {
			c.cancelled(ctx, method, id)
			return context.Cause(ctx)
		}
		return err
	}
	select {
	case resp, ok := <-ch:
		if !ok {
			c.mu.Lock()
			defer c.mu.Unlock()
			return c.err
		}
		if resp.Error != nil {
			return resp.Error
		}
		if result == nil || len(resp.Result) == 0 {
			return nil
		}
		if err := json.Unmarshal(resp.Result, result); err != nil {
			return fmt.Errorf("%s: the result does not decode: %w", method, err)
		}
		return nil
	case <-ctx.Done():
		c.cancelled(ctx, method, id)
		return context.Cause(ctx)
	}
}

// cancelled tells the server a request was abandoned, so it can stop the
// work. Best effort, and never for initialize, which the specification
// forbids cancelling.
func (c *Client) cancelled(ctx context.Context, method string, id int64) {
	if method == "initialize" {
		return
	}
	reason := context.Cause(ctx).Error()
	go func() {
		nctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = c.notify(nctx, "notifications/cancelled", map[string]any{"requestId": id, "reason": reason})
	}()
}

func (c *Client) notify(ctx context.Context, method string, params any) error {
	m := message{JSONRPC: "2.0", Method: method}
	if params != nil {
		raw, err := json.Marshal(params)
		if err != nil {
			return err
		}
		m.Params = raw
	}
	return c.t.send(ctx, m)
}

// decode reads one message, or a batch of them, from data. A batch is from an
// older revision of the protocol, and is still understood.
func decode(data []byte) ([]message, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil, nil
	}
	if data[0] == '[' {
		var batch []message
		err := json.Unmarshal(data, &batch)
		return batch, err
	}
	var m message
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return []message{m}, nil
}

// receive handles what the server sent — responses to this client's calls,
// requests the server makes of it, notifications — and returns it.
func (c *Client) receive(data []byte) []message {
	msgs, err := decode(data)
	if err != nil {
		c.log.Warn("mcp: ignoring output that is not a JSON-RPC message",
			slog.String("output", clipForLog(data)), slog.Any("err", err))
		return nil
	}
	for _, m := range msgs {
		c.dispatch(m)
	}
	return msgs
}

func (c *Client) dispatch(m message) {
	switch {
	case m.isResponse():
		id, ok := parseID(m.ID)
		if !ok {
			return
		}
		// Delivered and forgotten under the lock: each channel is sent to at
		// most once, so the send never blocks, and fail cannot close a channel
		// a response is on its way into.
		c.mu.Lock()
		if ch, ok := c.pending[id]; ok {
			delete(c.pending, id)
			ch <- m
		}
		c.mu.Unlock()

	case m.isRequest():
		reply := message{JSONRPC: "2.0", ID: m.ID}
		if m.Method == "ping" {
			reply.Result = json.RawMessage(`{}`)
		} else {
			reply.Error = &rpcError{Code: errMethodNotFound, Message: "this client does not provide " + m.Method}
		}
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = c.t.send(ctx, reply)
		}()

	case m.Method == "notifications/tools/list_changed":
		c.toolsChanged.Store(true)

	case m.Method == "notifications/message":
		var p struct {
			Level string          `json:"level"`
			Data  json.RawMessage `json:"data"`
		}
		_ = json.Unmarshal(m.Params, &p)
		c.log.Debug("mcp: server log", slog.String("level", p.Level), slog.String("data", clipForLog(p.Data)))
	}
}

func clipForLog(b []byte) string {
	const max = 200
	if len(b) > max {
		return strings.ToValidUTF8(string(b[:max]), "") + "…"
	}
	return string(b)
}

// ===== tools =====

// Tool is one tool as the server describes it.
type Tool struct {
	Name        string          `json:"name"`
	Title       string          `json:"title,omitempty"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema"`
	Annotations *Annotations    `json:"annotations,omitempty"`
}

// Annotations are the server's hints about a tool's behaviour. Pointers,
// because the specification gives each a default — a tool that says nothing is
// assumed to write and to reach the open world — and an omitted hint must not
// read as false.
type Annotations struct {
	Title           string `json:"title,omitempty"`
	ReadOnlyHint    *bool  `json:"readOnlyHint,omitempty"`
	DestructiveHint *bool  `json:"destructiveHint,omitempty"`
	IdempotentHint  *bool  `json:"idempotentHint,omitempty"`
	OpenWorldHint   *bool  `json:"openWorldHint,omitempty"`
}

// ListTools fetches every tool the server offers, following pagination.
func (c *Client) ListTools(ctx context.Context) ([]Tool, error) {
	var all []Tool
	cursor := ""
	for range 1000 {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		var page struct {
			Tools      []Tool `json:"tools"`
			NextCursor string `json:"nextCursor,omitempty"`
		}
		if err := c.call(ctx, "tools/list", params, &page); err != nil {
			return nil, fmt.Errorf("mcp: server %q: tools/list: %w", c.server.Name, err)
		}
		all = append(all, page.Tools...)
		if page.NextCursor == "" {
			c.toolsChanged.Store(false)
			return all, nil
		}
		cursor = page.NextCursor
	}
	return nil, fmt.Errorf("mcp: server %q: tools/list did not end after 1000 pages", c.server.Name)
}

// ToolsChanged reports whether the server has said its tool list changed
// since it was last listed. An agent keeps the tools it was built with; a
// long-lived host that cares lists them again and builds a new one.
//
// Over HTTP the notification only arrives on a response stream, because this
// client does not hold a standing stream open to the server.
func (c *Client) ToolsChanged() bool { return c.toolsChanged.Load() }

// Tools lists the server's tools as tool.Defs named <server>__<tool>.
func (c *Client) Tools(ctx context.Context) ([]tool.Def, error) {
	return c.tools(ctx, map[string]bool{})
}

func (c *Client) tools(ctx context.Context, taken map[string]bool) ([]tool.Def, error) {
	ts, err := c.ListTools(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]tool.Def, 0, len(ts))
	for _, t := range ts {
		d := c.Def(t)
		d.Name = uniqueName(taken, c.server.Name, t.Name)
		out = append(out, d)
	}
	return out, nil
}

// ===== many servers =====

// Connections is a set of connected servers.
type Connections struct {
	clients []*Client
	tools   []tool.Def
}

// ConnectAll connects to every server and lists its tools. A server that
// fails stops the whole thing: an agent quietly missing half its tools is a
// worse start than an error naming the server.
func ConnectAll(ctx context.Context, servers []Server) (*Connections, error) {
	cs := &Connections{}
	taken := map[string]bool{}
	for _, s := range servers {
		c, err := Connect(ctx, s)
		if err != nil {
			_ = cs.Close()
			return nil, err
		}
		cs.clients = append(cs.clients, c)
		defs, err := c.tools(ctx, taken)
		if err != nil {
			_ = cs.Close()
			return nil, err
		}
		cs.tools = append(cs.tools, defs...)
	}
	return cs, nil
}

// Tools is every connected server's tools, in server order. Nil-safe.
func (cs *Connections) Tools() []tool.Def {
	if cs == nil {
		return nil
	}
	return cs.tools
}

// Clients is the connections themselves. Nil-safe.
func (cs *Connections) Clients() []*Client {
	if cs == nil {
		return nil
	}
	return cs.clients
}

// Instructions joins what the servers asked to have put in front of the
// model, each under its server's name, for a system block. Nil-safe.
func (cs *Connections) Instructions() string {
	var b strings.Builder
	for _, c := range cs.Clients() {
		if in := strings.TrimSpace(c.Instructions()); in != "" {
			fmt.Fprintf(&b, "## %s\n%s\n\n", c.Name(), in)
		}
	}
	return strings.TrimSpace(b.String())
}

// Close closes every connection. Nil-safe.
func (cs *Connections) Close() error {
	var errs []error
	for _, c := range cs.Clients() {
		errs = append(errs, c.Close())
	}
	return errors.Join(errs...)
}
