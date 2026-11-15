// This file is the HTTP half of the package: the routes, the options, the
// error envelope and the SSE stream. Everything about a session's lifecycle
// lives next door in manager.go and session.go, which know nothing about HTTP
// — see the package doc in httpapi.go for why the split is worth having.
//
// Two rules hold across every handler here, and both are load-bearing:
//
//   - the method and the path are in the ServeMux pattern, never in an if;
//   - a Go error becomes a status and a token in exactly one function,
//     classify, and every handler returns through it. A status decided at the
//     call site drifts within a week, and the drift is invisible until some
//     client special-cases it.

package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

// Errors this handler raises on its own behalf. The lifecycle sentinels
// (ErrBusy and friends) belong to the Manager; these are about the request.
var (
	// ErrBadRequest marks anything the client got wrong about the request
	// itself: malformed JSON, an unknown field, a missing prompt, a resume
	// point that is not a number.
	//
	// Exported because a [ManagerConfig.Build] is the one place a HOST judges
	// client input — a workspace outside the directory the operator allowed,
	// a model this deployment does not offer — and without a sentinel it can
	// say so with, every such refusal reaches the client as a 500 that reads
	// like the server broke. Wrap it:
	//
	//	return nil, fmt.Errorf("%w: workspace %q is outside %s",
	//	    httpapi.ErrBadRequest, opts.Workspace, root)
	ErrBadRequest = errors.New("httpapi: malformed request")

	// errBodyTooLarge marks a request body over maxBodyBytes.
	errBodyTooLarge = errors.New("httpapi: request body too large")

	// errNotFound marks a path this handler serves nothing for. Distinct from
	// ErrNoSuchSession, which means the route was right and the id was not.
	errNotFound = errors.New("httpapi: no such endpoint")
)

// The bounded vocabulary of the "kind" field. A UI branches on these; they are
// therefore part of the wire contract and only ever added to, never reworded.
const (
	kindBadRequest      = "bad_request"
	kindBodyTooLarge    = "body_too_large"
	kindNotFound        = "not_found"
	kindNoSuchSession   = "no_such_session"
	kindNoSuchApproval  = "no_such_approval"
	kindBusy            = "busy"
	kindNotRunning      = "not_running"
	kindDone            = "done"
	kindAlreadyAnswered = "already_answered"
	kindTooManySessions = "too_many_sessions"
	kindClosed          = "closed"
	kindInternal        = "internal"
)

const (
	// maxBodyBytes caps every JSON request body. Generous, because a prompt
	// can legitimately carry a pasted stack trace or a file, and bounded,
	// because an unauthenticated socket must not let one client allocate
	// arbitrary memory.
	maxBodyBytes = 1 << 20 // 1 MiB

	// heartbeat is the SSE keep-alive period. A turn can spend a minute inside
	// one tool call, and proxies close a stream that has been quiet for less.
	heartbeat = 15 * time.Second

	// retryHint is the reconnect backoff suggested to EventSource. Short: the
	// conversation keeps going while the browser waits, so every second of
	// backoff is a second of output nobody is seeing.
	retryHint = 1000 * time.Millisecond

	// corsMaxAge is how long a browser may cache a preflight. Ten minutes
	// keeps the extra round trip off the hot path without pinning a stale
	// policy for a whole session.
	corsMaxAge = 10 * time.Minute
)

// ===== Options =====

// Option configures the handler returned by [New].
type Option func(*apiConfig)

type apiConfig struct {
	version string
	origins []string
	metrics http.Handler
	ui      fs.FS
	caps    Capabilities
}

// WithCORS allows the named browser origins to call this API.
//
// Without it there is no CORS handling at all, which is the right default for
// a handler mounted next to the UI that uses it: same origin needs no headers,
// and a preflight to an API that does not want cross-origin callers should
// fail.
//
// Preflight (OPTIONS with Access-Control-Request-Method) is answered here,
// before routing, because the resource patterns name their methods and would
// otherwise reject the probe. Last-Event-ID is in the allowed request headers
// on purpose — without it a cross-origin EventSource cannot resume.
//
// Passing "*" allows EVERY origin: any page on the internet the operator's
// browser visits can then drive this API with the operator's network access.
// There is no authentication in this package and no credentialed CORS mode
// (Access-Control-Allow-Credentials is never sent, so cookies never ride
// along), but the socket itself is the boundary — "*" is defensible for a
// loopback development server and is a hole anywhere else.
func WithCORS(origins ...string) Option {
	return func(c *apiConfig) { c.origins = append(c.origins, origins...) }
}

// WithMetrics mounts h at GET /metrics, for the scrape every long-lived
// process in a deployment already answers. Typically metric.Registry.Handler.
func WithMetrics(h http.Handler) Option {
	return func(c *apiConfig) { c.metrics = h }
}

// WithUI serves fsys at /, under the API rather than in front of it, so a
// single-page client and the API it calls share an origin and need no CORS.
//
// The /api/ namespace is matched more specifically and is never shadowed by a
// file in fsys, so an unknown /api/ path still answers with the JSON error
// envelope instead of the file server's HTML.
func WithUI(fsys fs.FS) Option {
	return func(c *apiConfig) { c.ui = fsys }
}

// WithVersion sets the build identifier reported by GET /api/health and GET
// /api/config.
func WithVersion(v string) Option {
	return func(c *apiConfig) { c.version = v }
}

// WithCapabilities sets the body of GET /api/config.
//
// It exists because the tool surface is a property of the agent the Manager's
// Build function produces, and the Manager deliberately exposes lifecycle
// rather than agent internals. Rather than have the config endpoint guess, or
// have a UI hardcode a list that rots the first time a tool is filtered out,
// the operator hands over the description it already has:
//
//	agent, err := build(...)
//	h := httpapi.New(m,
//	    httpapi.WithCapabilities(httpapi.Capabilities{
//	        DefaultModel: *model,
//	        Approvals:    gate != nil,
//	        Tools:        httpapi.ToolsOf(agent),
//	    }))
//
// Unset fields fall back to what the handler can know on its own: the version
// from [WithVersion] and the full list of [PermissionModes].
func WithCapabilities(caps Capabilities) Option {
	return func(c *apiConfig) { c.caps = caps }
}

// ===== Capability reporting =====

// Capabilities is the body of GET /api/config: everything a client would
// otherwise have to hardcode about this particular server.
type Capabilities struct {
	Version string `json:"version,omitempty"`

	// DefaultModel is what a session gets when SessionOptions.Model is empty.
	DefaultModel string `json:"default_model,omitempty"`

	// PermissionModes are the accepted values of SessionOptions.Permission.
	PermissionModes []string `json:"permission_modes"`

	// Approvals reports whether a tool call can park on a human here. False
	// means the approvals endpoints exist but will never have anything in
	// them, and a UI can leave the card out of its layout entirely.
	Approvals bool `json:"approvals"`

	// Tools is the surface the agent builder actually installed.
	Tools []ToolInfo `json:"tools"`
}

// ToolInfo is one tool, as a client needs to see it: enough to label a row in
// a timeline, not the schema the model is given.
type ToolInfo struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Category    string `json:"category,omitempty"`

	// ReadOnly is the one capability a UI renders differently, because it is
	// the one a user judges risk by.
	ReadOnly bool `json:"read_only"`
}

// PermissionModes returns the accepted values of SessionOptions.Permission, in
// increasing order of what they allow.
//
// A function rather than a package-level slice so a caller cannot alias and
// mutate the handler's answer.
func PermissionModes() []string {
	return []string{"off", "readonly", "workspace", "ask"}
}

// ToolsOf describes the tool surface an agent was built with.
//
// It reads the UNGATED set — the visibility a run's context might further
// narrow is a per-run fact, and GET /api/config is answering "what does this
// server offer" rather than "what can this turn do right now".
func ToolsOf(a *wombat.Agent) []ToolInfo {
	if a == nil {
		return nil
	}
	defs := a.Tools(context.Background())
	out := make([]ToolInfo, 0, len(defs))
	for _, d := range defs {
		out = append(out, ToolInfo{
			Name:        d.Name,
			Description: d.Description,
			Category:    d.Category,
			ReadOnly:    d.Has(tool.CapReadOnly),
		})
	}
	return out
}

// ===== The handler =====

type api struct {
	m       *Manager
	cfg     apiConfig
	started time.Time
}

// New returns the HTTP surface for m.
//
// It is an [http.Handler] and not a server, so a host with its own listener,
// its own middleware and its own authentication can mount it wherever it
// likes; cmd/wombat-serve is thirty lines on top of it.
//
// # There is no authentication here
//
// None at all. Any client that can reach the socket can start a session, spend
// the operator's money, read every transcript and — if it learns a session id
// — answer another client's approval prompts. Binding to anything other than
// loopback is therefore a decision the operator makes knowingly, behind auth
// they supply themselves. [WithCORS] governs which browser ORIGINS may call
// in; it is a browser's rule about pages, not this server's rule about
// clients, and it stops nothing that is not a browser.
//
// # No WriteTimeout
//
// The events endpoint is meant to stay open for the length of a conversation,
// and [http.Server.WriteTimeout] bounds a whole response. Leave it unset, or
// accept that every stream dies on the clock; the handler clears any deadline
// it can reach, but it cannot un-set a policy it never sees.
//
// Panics if m is nil: a handler with nothing to serve is a wiring mistake at
// construction time, and discovering it as a nil dereference on the first
// request is strictly worse.
func New(m *Manager, opts ...Option) http.Handler {
	if m == nil {
		panic("httpapi: New(nil)")
	}
	a := &api{m: m, started: time.Now()}
	for _, o := range opts {
		o(&a.cfg)
	}
	if a.cfg.caps.Version == "" {
		a.cfg.caps.Version = a.cfg.version
	}
	if a.cfg.caps.PermissionModes == nil {
		a.cfg.caps.PermissionModes = PermissionModes()
	}

	mux := http.NewServeMux()

	// Method and path both live in the pattern. A handler that starts with a
	// switch on r.Method is a router that forgot it had one, and it is the
	// reason 405s go missing.
	mux.HandleFunc("GET /api/health", a.health)
	mux.HandleFunc("GET /api/config", a.config)
	mux.HandleFunc("POST /api/sessions", a.createSession)
	mux.HandleFunc("GET /api/sessions", a.listSessions)
	mux.HandleFunc("GET /api/sessions/{id}", a.getSession)
	mux.HandleFunc("DELETE /api/sessions/{id}", a.deleteSession)
	mux.HandleFunc("GET /api/sessions/{id}/events", a.events)
	mux.HandleFunc("GET /api/sessions/{id}/messages", a.getMessages)
	mux.HandleFunc("POST /api/sessions/{id}/messages", a.postMessage)
	mux.HandleFunc("POST /api/sessions/{id}/steer", a.steer)
	mux.HandleFunc("GET /api/sessions/{id}/approvals", a.listApprovals)
	mux.HandleFunc("POST /api/sessions/{id}/approvals/{use_id}", a.resolveApproval)

	// Registered without a method, so every unrouted verb on every unrouted
	// /api/ path still answers with the error envelope. The cost is that
	// ServeMux's automatic 405 no longer fires inside /api/ — this pattern
	// matches, so the method-mismatch path is never reached — and a wrong verb
	// reads as 404 not_found. Worth it: a client that gets HTML, or bare text,
	// from an endpoint documented to speak JSON has to special-case parsing
	// before it can even report the failure.
	mux.HandleFunc("/api/", a.notFound)

	if a.cfg.metrics != nil {
		mux.Handle("GET /metrics", a.cfg.metrics)
	}
	if a.cfg.ui != nil {
		// Methodless, and the method checked by hand, because ServeMux refuses
		// to register "GET /" alongside "/api/": neither is more specific than
		// the other — one narrows the method, the other the path — so the mux
		// cannot say which should win, and says so at construction time.
		files := http.FileServerFS(a.cfg.ui)
		mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet && r.Method != http.MethodHead {
				a.notFound(w, r)
				return
			}
			files.ServeHTTP(w, r)
		})
	} else {
		mux.HandleFunc("/", a.notFound)
	}

	var h http.Handler = mux
	if len(a.cfg.origins) > 0 {
		h = a.withCORS(h)
	}
	return h
}

func (a *api) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, Health{
		Status:    "ok",
		Version:   a.cfg.version,
		UptimeSec: time.Since(a.started).Seconds(),
	})
}

func (a *api) config(w http.ResponseWriter, _ *http.Request) {
	caps := a.cfg.caps
	if caps.Tools == nil {
		// An explicit empty array, not null: a client iterating the field
		// should not have to nil-check a list.
		caps.Tools = []ToolInfo{}
	}
	writeJSON(w, http.StatusOK, caps)
}

func (a *api) notFound(w http.ResponseWriter, r *http.Request) {
	a.fail(w, r, fmt.Errorf("%w: %s %s", errNotFound, r.Method, r.URL.Path))
}

// createSession creates the conversation AND runs its first turn.
//
// One call rather than two, because a session with no turn in it is a state
// with no use: a client that crashed between the two would leave a paid-for
// slot occupied by nothing.
func (a *api) createSession(w http.ResponseWriter, r *http.Request) {
	var req CreateRequest
	if err := decodeBody(w, r, &req); err != nil {
		a.fail(w, r, err)
		return
	}
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		a.fail(w, r, fmt.Errorf("%w: prompt is required", ErrBadRequest))
		return
	}

	s, err := a.m.Create(req.SessionOptions)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if _, err := s.Send(prompt); err != nil {
		// The session never carried a turn, so it is dropped rather than left
		// idle: the caller has no id to clean it up with.
		a.m.Delete(s.ID())
		a.fail(w, r, err)
		return
	}

	// Relative to the collection that was just posted to, so the header stays
	// correct under any mount prefix and behind any proxy that rewrites one.
	w.Header().Set("Location", r.URL.EscapedPath()+"/"+s.ID())
	writeJSON(w, http.StatusCreated, s.Info())
}

func (a *api) listSessions(w http.ResponseWriter, _ *http.Request) {
	list := a.m.List()
	if list == nil {
		list = []SessionInfo{}
	}
	writeJSON(w, http.StatusOK, SessionList{Sessions: list})
}

func (a *api) getSession(w http.ResponseWriter, r *http.Request) {
	s, err := a.session(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, s.Info())
}

func (a *api) deleteSession(w http.ResponseWriter, r *http.Request) {
	if !a.m.Delete(r.PathValue("id")) {
		a.fail(w, r, ErrNoSuchSession)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (a *api) getMessages(w http.ResponseWriter, r *http.Request) {
	s, err := a.session(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	msgs := s.Messages()
	if msgs == nil {
		msgs = []llm.Message{}
	}
	// Not a named type in [APITypes]: the transcript is the harness's own
	// persistence shape, and declaring a second copy of it here to feed a code
	// generator would be a fork of a format this package does not own.
	writeJSON(w, http.StatusOK, struct {
		Messages []llm.Message `json:"messages"`
	}{msgs})
}

// postMessage continues the conversation with another turn.
//
// 202 rather than 200: Send starts the turn and returns, and everything worth
// watching arrives on the event stream. Answering 200 would imply the work is
// done, which is exactly the misunderstanding the SSE endpoint exists to fix.
func (a *api) postMessage(w http.ResponseWriter, r *http.Request) {
	s, err := a.session(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	var req SendRequest
	if err := decodeBody(w, r, &req); err != nil {
		a.fail(w, r, err)
		return
	}
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		a.fail(w, r, fmt.Errorf("%w: prompt is required", ErrBadRequest))
		return
	}
	if _, err := s.Send(prompt); err != nil {
		a.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusAccepted, s.Info())
}

// steer passes a message to the turn in progress; see wombat.Run.Steer. 202
// when the run took it, 409 with not_running when there is no turn to take it.
func (a *api) steer(w http.ResponseWriter, r *http.Request) {
	s, err := a.session(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	var req SendRequest
	if err := decodeBody(w, r, &req); err != nil {
		a.fail(w, r, err)
		return
	}
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		a.fail(w, r, fmt.Errorf("%w: prompt is required", ErrBadRequest))
		return
	}
	if err := s.Steer(prompt); err != nil {
		a.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusAccepted, s.Info())
}

func (a *api) listApprovals(w http.ResponseWriter, r *http.Request) {
	s, err := a.session(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	pending := s.Pending()
	if pending == nil {
		pending = []Approval{}
	}
	writeJSON(w, http.StatusOK, ApprovalList{Approvals: pending})
}

// resolveApproval answers one parked tool call.
//
// allow is a *bool so that a body omitting it is a 400 rather than a silent
// denial. Deny-by-default would be the safe reading, but a UI whose field name
// drifted would then refuse every call while looking like it worked.
func (a *api) resolveApproval(w http.ResponseWriter, r *http.Request) {
	s, err := a.session(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	var req ResolveRequest
	if err := decodeBody(w, r, &req); err != nil {
		a.fail(w, r, err)
		return
	}
	if req.Allow == nil {
		a.fail(w, r, fmt.Errorf("%w: allow must be true or false", ErrBadRequest))
		return
	}
	if err := s.Resolve(r.PathValue("use_id"), *req.Allow); err != nil {
		a.fail(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ===== SSE =====

// events streams the session's log as server-sent events.
//
// The stream is the session's, not a turn's: sequence numbers are global to
// the conversation, so a client that reconnects after a turn boundary resumes
// where it was rather than at the start of whatever turn is running now.
//
// Resumption has two spellings because a browser only has one of them.
// EventSource replays the last id it saw in Last-Event-ID and cannot be given
// a header by script, so ?from= is the fallback a deliberate reconnect uses.
// They mean subtly different things and both are exact: Last-Event-ID is the
// last sequence the client HAS, so the stream restarts one past it; ?from= is
// the first sequence the client WANTS.
//
// A request context that dies ends this handler and nothing else. The session
// keeps running — that is the entire reason a session exists rather than a
// request-scoped run — and the next connection resumes it.
func (a *api) events(w http.ResponseWriter, r *http.Request) {
	s, err := a.session(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	from, err := resumeFrom(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}

	rc := http.NewResponseController(w)

	// An SSE response must never be bounded by a write deadline: it is meant
	// to stay open for the length of a conversation, and http.Server's
	// WriteTimeout bounds the whole response. Clearing it here means this
	// handler is safe to mount inside a host server that set one for its own
	// endpoints. Not every ResponseWriter supports it, and the ones that do
	// not have no deadline to clear, so the error is deliberately ignored.
	_ = rc.SetWriteDeadline(time.Time{})

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no") // tell an nginx in front not to buffer
	w.WriteHeader(http.StatusOK)

	fmt.Fprintf(w, "retry: %d\n\n", retryHint.Milliseconds())
	if rc.Flush() != nil {
		return
	}

	// Cancelled on every return path, including a write failure, so the
	// goroutine below cannot outlive this handler even for the instant before
	// net/http cancels the request context itself.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Follow runs on its own goroutine so this one can interleave heartbeats
	// with frames: a range-over-func cannot be selected on, and the
	// ResponseWriter must only ever be written from one goroutine.
	frames := make(chan Frame)
	go func() {
		defer close(frames)
		for _, f := range s.Follow(ctx, from) {
			select {
			case frames <- f:
			case <-ctx.Done():
				return
			}
		}
	}()

	ticker := time.NewTicker(heartbeat)
	defer ticker.Stop()

	for {
		select {
		case f, ok := <-frames:
			if !ok {
				// Follow returned on its own, so the conversation is over.
				a.sendEnd(w, rc, s)
				return
			}
			if !sendFrame(w, rc, f) {
				return // client gone; the session runs on, ready to be resumed
			}
		case <-ticker.C:
			if _, err := fmt.Fprint(w, ": ping\n\n"); err != nil || rc.Flush() != nil {
				return
			}
		case <-ctx.Done():
			return
		}
	}
}

// sendFrame writes one log entry. Reports whether the connection is still
// usable.
func sendFrame(w http.ResponseWriter, rc *http.ResponseController, f Frame) bool {
	if f.Event == nil {
		return true // nothing to render; not a reason to drop the connection
	}
	data, err := marshal(f.Event)
	if err != nil {
		// One unrenderable event must not end a stream that is otherwise fine;
		// the gap is visible in the sequence numbers.
		return true
	}
	if _, err := fmt.Fprintf(w, "id: %d\nevent: %s\ndata: %s\n\n", f.Seq, f.Event.Kind(), data); err != nil {
		return false
	}
	return rc.Flush() == nil
}

// sendEnd closes the stream with the final SessionInfo.
//
// It carries no id, deliberately, so Last-Event-ID stays on the last real
// frame: a client that drops here and reconnects replays nothing and is simply
// told again how the conversation ended, which is idempotent. Giving it an id
// would resume past the end and leave the client waiting for a summary that
// never comes.
//
// The event exists at all because EventSource reconnects automatically. With
// no terminal marker a finished session becomes a reconnect loop — connect,
// resume past the end, close, wait retryHint, repeat — and the client has no
// way to know it should stop.
func (a *api) sendEnd(w http.ResponseWriter, rc *http.ResponseController, s *Session) {
	data, err := marshal(s.Info())
	if err != nil {
		return
	}
	if _, err := fmt.Fprintf(w, "event: end\ndata: %s\n\n", data); err != nil {
		return
	}
	_ = rc.Flush()
}

// resumeFrom is the first sequence number the client wants.
//
// Strict about garbage in either spelling. Being lenient — treating an
// unparseable resume point as 0 — replays the whole conversation into a client
// that thought it was catching up, and the duplicate render looks like an
// agent that repeated itself rather than like the bug it is.
func resumeFrom(r *http.Request) (int, error) {
	if raw := strings.TrimSpace(r.Header.Get("Last-Event-ID")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			return 0, fmt.Errorf("%w: Last-Event-ID %q is not a sequence number", ErrBadRequest, raw)
		}
		return n + 1, nil
	}
	if raw := strings.TrimSpace(r.URL.Query().Get("from")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			return 0, fmt.Errorf("%w: from=%q is not a sequence number", ErrBadRequest, raw)
		}
		return n, nil
	}
	return 0, nil
}

// ===== CORS =====

// withCORS answers preflights and stamps allowed responses.
func (a *api) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		allowed := a.allowOrigin(origin)
		if allowed != "" {
			h := w.Header()
			h.Set("Access-Control-Allow-Origin", allowed)
			// Not a Set: a cache keyed on this response must know the body
			// varies by origin even when something upstream added its own Vary.
			h.Add("Vary", "Origin")
			// Location is how a client learns the new session's URL, and a
			// cross-origin fetch cannot read a header that is not exposed.
			h.Set("Access-Control-Expose-Headers", "Location")
		}

		if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
			h := w.Header()
			h.Add("Vary", "Access-Control-Request-Method")
			h.Add("Vary", "Access-Control-Request-Headers")
			if allowed != "" {
				h.Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
				h.Set("Access-Control-Allow-Headers", "Content-Type, Last-Event-ID")
				h.Set("Access-Control-Max-Age", strconv.Itoa(int(corsMaxAge.Seconds())))
			}
			// 204 either way. A preflight from a disallowed origin that lacks
			// the Allow-* headers is already refused by the browser, and
			// answering it with an error status only muddies the console
			// message the developer has to debug from.
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// allowOrigin reports what to echo in Access-Control-Allow-Origin, or "" for
// a request that gets no CORS headers at all.
func (a *api) allowOrigin(origin string) string {
	if origin == "" {
		return ""
	}
	for _, o := range a.cfg.origins {
		if o == "*" {
			return "*"
		}
		if strings.EqualFold(o, origin) {
			// Echoed rather than answered with "*", so a cache cannot serve
			// one origin's response to another.
			return origin
		}
	}
	return ""
}

// ===== Plumbing =====

// session resolves the {id} wildcard, or ErrNoSuchSession.
func (a *api) session(r *http.Request) (*Session, error) {
	id := r.PathValue("id")
	s, ok := a.m.Get(id)
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrNoSuchSession, id)
	}
	return s, nil
}

// classify is the single place an error becomes a status and a token.
//
// Every handler returns through it. The alternative — each call site choosing
// its own http.StatusX — drifts the moment two handlers can produce the same
// error, and a client that special-cased one of them then breaks on the other.
func classify(err error) (status int, kind string) {
	switch {
	case errors.Is(err, ErrNoSuchSession):
		return http.StatusNotFound, kindNoSuchSession
	case errors.Is(err, ErrNoSuchApproval):
		return http.StatusNotFound, kindNoSuchApproval
	case errors.Is(err, errNotFound):
		return http.StatusNotFound, kindNotFound
	case errors.Is(err, ErrBusy):
		return http.StatusConflict, kindBusy
	case errors.Is(err, ErrNotRunning):
		return http.StatusConflict, kindNotRunning
	case errors.Is(err, ErrDone):
		// A conflict with the resource's state, exactly like ErrBusy: the
		// session exists and the request is well formed, but a conversation
		// the model ended with a terminal tool has nothing to continue from.
		return http.StatusConflict, kindDone
	case errors.Is(err, ErrAlreadyAnswered):
		return http.StatusConflict, kindAlreadyAnswered
	case errors.Is(err, ErrTooManySessions):
		return http.StatusTooManyRequests, kindTooManySessions
	case errors.Is(err, ErrClosed):
		// Shutting down, not broken. 503 is the one status a client should
		// retry against another instance rather than report as a bug.
		return http.StatusServiceUnavailable, kindClosed
	case errors.Is(err, errBodyTooLarge):
		// A body over the cap is still a bad body, so it keeps the 400 the
		// pinned table gives one; the distinct kind is what tells a UI to say
		// "too long" rather than "malformed".
		return http.StatusBadRequest, kindBodyTooLarge
	case errors.Is(err, ErrBadRequest):
		return http.StatusBadRequest, kindBadRequest
	default:
		return http.StatusInternalServerError, kindInternal
	}
}

// fail writes the error envelope.
//
// 500s are logged because they are the only class nobody else is watching: a
// 404 is the client's problem and shows up in its console, whereas a 500 means
// this process produced an error no handler expected.
func (a *api) fail(w http.ResponseWriter, r *http.Request, err error) {
	status, kind := classify(err)
	if status >= http.StatusInternalServerError {
		slog.Default().ErrorContext(r.Context(), "httpapi request failed",
			"method", r.Method, "path", r.URL.Path, "kind", kind, "err", err)
	}
	writeJSON(w, status, ErrorBody{ErrorDetail{Kind: kind, Message: err.Error()}})
}

// decodeBody reads one JSON object out of the request.
//
// Unknown fields are rejected. A UI that misspells "max_iters" and silently
// gets the default is a bug that surfaces days later as "the limit setting
// does nothing"; a 400 surfaces it on the first request. The body is capped
// for the same reason the session count is: nothing here is authenticated.
func decodeBody(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()

	if err := dec.Decode(dst); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return fmt.Errorf("%w: %d bytes is the limit: %w", errBodyTooLarge, maxBodyBytes, err)
		}
		return fmt.Errorf("%w: %w", ErrBadRequest, err)
	}
	// Trailing content means the client sent something other than the single
	// object this endpoint documents, and accepting the first half of it would
	// hide whatever the second half was trying to say.
	if dec.More() {
		return fmt.Errorf("%w: unexpected trailing content after the JSON object", ErrBadRequest)
	}
	return nil
}

// writeJSON encodes into a buffer first so a marshalling failure is still
// recoverable: once a status line is out, a half-written body is all the
// client gets and it cannot be told why.
func writeJSON(w http.ResponseWriter, status int, v any) {
	body, err := marshal(v)
	if err != nil {
		body = []byte(`{"error":{"kind":"internal","message":"response encoding failed"}}`)
		status = http.StatusInternalServerError
	}
	h := w.Header()
	h.Set("Content-Type", "application/json; charset=utf-8")
	h.Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

// marshal encodes without HTML escaping.
//
// Model output is full of <, > and &, and wombat's events already take care
// not to escape them; a plain json.Marshal here would put them back, because
// the outer encoder re-runs its compact pass over whatever a custom
// MarshalJSON returned. See wombat's marshalNoEscape for the full story.
func marshal(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
