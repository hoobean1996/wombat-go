// This file is the HTTP half of the package: the routes, the options, the
// error envelope and the SSE stream. Everything about a session's lifecycle
// lives in wombat/pkg/daemon, which knows nothing about HTTP — see the package
// doc in httpapi.go for why the split is worth having.
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
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"mime"
	"net"
	"net/http"
	"path"
	"slices"
	"strconv"
	"strings"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
	"wombat/pkg/daemon"
)

// Errors this handler raises on its own behalf. The lifecycle sentinels
// (ErrNoSuchSession and friends) belong to the daemon; these are about the
// request.
var (
	// errBodyTooLarge marks a request body over maxBodyBytes.
	errBodyTooLarge = errors.New("httpapi: request body too large")

	// errNotFound marks a path this handler serves nothing for. Distinct from
	// ErrNoSuchSession, which means the route was right and the id was not.
	errNotFound = errors.New("httpapi: no such endpoint")

	// errNotJSON marks a request body that does not say it is JSON.
	errNotJSON = errors.New("httpapi: request body must be sent as application/json")

	// errUnauthorized marks a request without the token [WithBearerToken]
	// asks for.
	errUnauthorized = errors.New("httpapi: missing or wrong bearer token")

	// errWrongHost marks a request for a host [WithAllowedHosts] does not
	// list.
	errWrongHost = errors.New("httpapi: this server does not answer for that host")
)

// The bounded vocabulary of the "kind" field. A UI branches on these; they are
// therefore part of the wire contract and only ever added to, never reworded.
const (
	kindBadRequest      = "bad_request"
	kindBodyTooLarge    = "body_too_large"
	kindNotFound        = "not_found"
	kindNoSuchSession   = "no_such_session"
	kindNoSuchApproval  = "no_such_approval"
	kindNoSuchTimer     = "no_such_timer"
	kindNoSuchMessage   = "no_such_message"
	kindNotRunning      = "not_running"
	kindDone            = "done"
	kindAlreadyAnswered = "already_answered"
	kindClosed          = "closed"
	kindInternal        = "internal"
	kindNotJSON         = "unsupported_media_type"
	kindUnauthorized    = "unauthorized"
	kindWrongHost       = "misdirected"
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

	// maxWait bounds a long poll of the outbox, under what a proxy in between
	// would cut it off at.
	maxWait = 60 * time.Second

	// defaultLimit and maxLimit bound a listing of the inbox or the outbox.
	defaultLimit = 100
	maxLimit     = 1000
)

// ===== Options =====

// Option configures the handler returned by [New].
type Option func(*apiConfig)

type apiConfig struct {
	version string
	origins []string
	metrics http.Handler
	caps    Capabilities
	token   string
	hosts   []string
	ui      fs.FS
}

// WithBearerToken requires token on every request to the API and to
// /metrics, as "Authorization: Bearer <token>". The events stream takes it as
// an access_token query parameter too, because a browser's EventSource cannot
// set a header — and only the events stream, so the token is in no other URL.
//
// GET /api/health stays open, for the load balancer that probes it. A CORS
// preflight is answered without it, as preflights carry no credentials. An
// empty token turns the check off.
func WithBearerToken(token string) Option {
	return func(c *apiConfig) { c.token = token }
}

// WithAllowedHosts answers only requests whose Host header names one of
// hosts, with or without a port. Anything else is refused with 421.
//
// It is the defense against DNS rebinding, which a server on loopback needs
// and does not get from being on loopback: a page on any site the operator
// visits can point a name it controls at 127.0.0.1, and from then on its
// requests reach this server as same-origin ones, responses included. The
// Host header still carries the attacker's name, and that is not a name the
// operator listed. A bearer token defends against the same thing, for a page
// cannot send what it does not know; this is the defense that needs nothing
// from the client.
func WithAllowedHosts(hosts ...string) Option {
	return func(c *apiConfig) {
		for _, h := range hosts {
			c.hosts = append(c.hosts, normalizeHost(h))
		}
	}
}

// normalizeHost lowercases a host and drops its port and IPv6 brackets.
func normalizeHost(h string) string {
	if host, _, err := net.SplitHostPort(h); err == nil {
		h = host
	}
	return strings.ToLower(strings.TrimSuffix(strings.TrimPrefix(h, "["), "]"))
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

// WithUI serves a browser front end from fsys at every path outside /api/ and
// /metrics. See [uiHandler] for what it serves where.
//
// The UI's files are served without the bearer token — they are the same for
// everyone and hold no data — and the page asks for the token before it calls
// the API. [WithAllowedHosts] still applies to them.
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

	// Approvals reports whether a tool call can wait for a person's decision
	// here. False means the approvals endpoints exist but will never have
	// anything in them, and a UI can leave the card out of its layout.
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
func ToolsOf(a *core.Agent) []ToolInfo {
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
	m       *daemon.Manager
	cfg     apiConfig
	started time.Time
}

// New returns the HTTP surface for m.
//
// It is an [http.Handler] and not a server, so a host with its own listener,
// its own middleware and its own authentication can mount it wherever it
// likes; cmd/wombatd is one such host.
//
// # Authentication is opt-in
//
// Without [WithBearerToken] there is none. Any client that can reach the
// socket can start a session, spend the operator's money, read every
// transcript and — if it learns a session id — answer another client's
// approval prompts. Binding to anything other than loopback without a token is
// therefore a decision the operator makes knowingly, behind auth they supply
// themselves. [WithCORS] governs which browser ORIGINS may call in; it is a
// browser's rule about pages, not this server's rule about clients, and it
// stops nothing that is not a browser.
//
// Loopback is not a boundary against the browser on the same machine either.
// A page on another site can send a request here, and the one thing that
// keeps it from creating a session is that a request body must be declared
// application/json: that is not a type a cross-site page may send without
// asking first, and the asking — a CORS preflight — is refused. A page that
// rebinds its own name to 127.0.0.1 needs no preflight at all; that takes
// [WithAllowedHosts], or a token.
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
func New(m *daemon.Manager, opts ...Option) http.Handler {
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
	mux.HandleFunc("POST /api/events", a.postEvent)
	mux.HandleFunc("GET /api/inbox", a.listInbox)
	mux.HandleFunc("GET /api/outbox", a.listOutbox)
	mux.HandleFunc("POST /api/outbox/{id}/ack", a.ackMessage)
	mux.HandleFunc("GET /api/timers", a.listTimers)
	mux.HandleFunc("POST /api/timers", a.addTimer)
	mux.HandleFunc("DELETE /api/timers/{id}", a.cancelTimer)

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
		mux.Handle("/", a.uiHandler(a.cfg.ui))
	} else {
		mux.HandleFunc("/", a.notFound)
	}

	// Outermost last: a misdirected request is refused before anything,
	// a preflight is answered before the token is asked for.
	var h http.Handler = mux
	if a.cfg.token != "" {
		h = a.withToken(h)
	}
	if len(a.cfg.origins) > 0 {
		h = a.withCORS(h)
	}
	if len(a.cfg.hosts) > 0 {
		h = a.withHosts(h)
	}
	return h
}

func (a *api) withHosts(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !slices.Contains(a.cfg.hosts, normalizeHost(r.Host)) {
			a.fail(w, r, fmt.Errorf("%w: %q", errWrongHost, r.Host))
			return
		}
		next.ServeHTTP(w, r)
	})
}

// bearerToken reads "Bearer <token>"; the scheme is case-insensitive, as
// every HTTP authentication scheme is.
func bearerToken(h string) (string, bool) {
	const scheme = "bearer "
	if len(h) < len(scheme) || !strings.EqualFold(h[:len(scheme)], scheme) {
		return "", false
	}
	return h[len(scheme):], true
}

func (a *api) withToken(next http.Handler) http.Handler {
	want := []byte(a.cfg.token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := path.Clean("/" + r.URL.Path)
		inAPI := p == "/api" || strings.HasPrefix(p, "/api/")
		if (!inAPI && p != "/metrics") || p == "/api/health" {
			next.ServeHTTP(w, r)
			return
		}
		got, ok := bearerToken(r.Header.Get("Authorization"))
		if !ok && r.Method == http.MethodGet && strings.HasPrefix(p, "/api/sessions/") && strings.HasSuffix(p, "/events") {
			got, ok = r.URL.Query().Get("access_token"), true
		}
		if !ok || subtle.ConstantTimeCompare([]byte(strings.TrimSpace(got)), want) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="wombat"`)
			a.fail(w, r, errUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
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

// createSession creates the conversation AND sends its first message.
//
// One call rather than two, because a session with nothing in it is a state
// with no use: a client that crashed between the two would leave a session
// behind that nobody will ever talk to.
func (a *api) createSession(w http.ResponseWriter, r *http.Request) {
	var req CreateRequest
	if err := decodeBody(w, r, &req); err != nil {
		a.fail(w, r, err)
		return
	}
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		a.fail(w, r, fmt.Errorf("%w: prompt is required", daemon.ErrBadRequest))
		return
	}

	s, err := a.m.Create(r.Context(), req.SessionOptions)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if _, err := s.Send(r.Context(), prompt); err != nil {
		// The session never got its message, so it is deleted rather than
		// left empty: the caller has no id to clean it up with.
		_ = a.m.Delete(context.WithoutCancel(r.Context()), s.ID())
		a.fail(w, r, err)
		return
	}

	// Relative to the collection that was just posted to, so the header stays
	// correct under any mount prefix and behind any proxy that rewrites one.
	w.Header().Set("Location", r.URL.EscapedPath()+"/"+s.ID())
	writeJSON(w, http.StatusCreated, s.Info())
}

func (a *api) listSessions(w http.ResponseWriter, r *http.Request) {
	list, err := a.m.List(r.Context())
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if list == nil {
		list = []daemon.SessionInfo{}
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
	if err := a.m.Delete(r.Context(), r.PathValue("id")); err != nil {
		a.fail(w, r, err)
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
	// An anonymous body rather than a named wire type: the transcript is the
	// harness's own persistence shape, and a second declaration of it here
	// would be a fork of a format this package does not own.
	writeJSON(w, http.StatusOK, struct {
		Messages []llm.Message `json:"messages"`
	}{msgs})
}

// postMessage puts a message in the session's inbox.
//
// 202 rather than 200: the message is queued for the session's next turn —
// at once if it is at rest, after the one running if it is not — and what it
// leads to arrives on the event stream and in the outbox.
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
		a.fail(w, r, fmt.Errorf("%w: prompt is required", daemon.ErrBadRequest))
		return
	}
	e, err := s.Send(r.Context(), prompt)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusAccepted, Accepted{Event: e, Session: s.Info()})
}

// steer passes a message to the turn in progress; see core.Run.Steer. 202
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
		a.fail(w, r, fmt.Errorf("%w: prompt is required", daemon.ErrBadRequest))
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
		pending = []daemon.Approval{}
	}
	writeJSON(w, http.StatusOK, ApprovalList{Approvals: pending})
}

// resolveApproval decides one call the session is waiting on. The decision
// is an event: 202, and the turn resumes once every waiting call has one.
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
		a.fail(w, r, fmt.Errorf("%w: allow must be true or false", daemon.ErrBadRequest))
		return
	}
	e, err := s.Resolve(r.Context(), r.PathValue("use_id"), *req.Allow)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusAccepted, Accepted{Event: e, Session: s.Info()})
}

// ===== The inbox, the outbox, the timers =====

// postEvent puts an event from outside in a session's inbox: 202 when it is
// new, 200 with the one received before when its dedup key says it is not.
func (a *api) postEvent(w http.ResponseWriter, r *http.Request) {
	var req EventRequest
	if err := decodeBody(w, r, &req); err != nil {
		a.fail(w, r, err)
		return
	}
	in := daemon.Inbound{
		Session: req.SessionID, Thread: req.Thread,
		Kind: req.Kind, Source: req.Source, DedupKey: req.DedupKey, Payload: req.Payload,
	}
	if req.Options != nil {
		in.Options = *req.Options
	}
	e, created, err := a.m.Receive(r.Context(), in)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	body := Accepted{Event: e, Duplicate: !created}
	if s, err := a.m.Get(r.Context(), e.Session); err == nil {
		body.Session = s.Info()
	}
	status := http.StatusAccepted
	if !created {
		status = http.StatusOK
	}
	writeJSON(w, status, body)
}

func (a *api) listInbox(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit, err := limitOf(q.Get("limit"))
	if err != nil {
		a.fail(w, r, err)
		return
	}
	events, err := a.m.Inbox(r.Context(), daemon.EventFilter{
		Session: q.Get("session"), Status: daemon.EventStatus(q.Get("status")), Limit: limit,
	})
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if events == nil {
		events = []daemon.Event{}
	}
	writeJSON(w, http.StatusOK, EventList{Events: events})
}

// listOutbox lists messages, oldest first. ?after= is the last id the client
// has, and ?wait= a duration to wait for one when there is none yet: a client
// that long-polls with both reads every message once, in order, as it is
// posted.
func (a *api) listOutbox(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit, err := limitOf(q.Get("limit"))
	if err != nil {
		a.fail(w, r, err)
		return
	}
	var after int64
	if raw := q.Get("after"); raw != "" {
		if after, err = strconv.ParseInt(raw, 10, 64); err != nil || after < 0 {
			a.fail(w, r, fmt.Errorf("%w: after=%q is not a message id", daemon.ErrBadRequest, raw))
			return
		}
	}
	var wait time.Duration
	if raw := q.Get("wait"); raw != "" {
		if wait, err = time.ParseDuration(raw); err != nil || wait < 0 {
			a.fail(w, r, fmt.Errorf("%w: wait=%q is not a duration", daemon.ErrBadRequest, raw))
			return
		}
		wait = min(wait, maxWait)
		_ = http.NewResponseController(w).SetWriteDeadline(time.Now().Add(wait + 10*time.Second))
	}
	msgs, err := a.m.Outbox(r.Context(), daemon.OutboxFilter{
		Session: q.Get("session"), Status: daemon.MessageStatus(q.Get("status")), After: after, Limit: limit,
	}, wait)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if msgs == nil {
		msgs = []daemon.Message{}
	}
	writeJSON(w, http.StatusOK, MessageList{Messages: msgs})
}

// ackMessage marks a message handled, so a dispatcher — or the next client —
// does not deliver it again.
func (a *api) ackMessage(w http.ResponseWriter, r *http.Request) {
	id, err := idOf(r, "message")
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if err := a.m.Ack(r.Context(), id); err != nil {
		a.fail(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (a *api) listTimers(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	timers, err := a.m.Timers(r.Context(), daemon.TimerFilter{
		Session: q.Get("session"), Status: daemon.TimerStatus(q.Get("status")),
	})
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if timers == nil {
		timers = []daemon.Timer{}
	}
	writeJSON(w, http.StatusOK, TimerList{Timers: timers})
}

func (a *api) addTimer(w http.ResponseWriter, r *http.Request) {
	var req TimerRequest
	if err := decodeBody(w, r, &req); err != nil {
		a.fail(w, r, err)
		return
	}
	due, err := daemon.DueAt(req.In, req.At, time.Now())
	if err != nil {
		a.fail(w, r, err)
		return
	}
	t := daemon.Timer{Session: req.SessionID, Note: req.Note, Due: due, CreatedBy: daemon.SourceAPI}
	if req.Every != "" {
		if t.Every, err = time.ParseDuration(req.Every); err != nil {
			a.fail(w, r, fmt.Errorf("%w: every=%q is not a duration", daemon.ErrBadRequest, req.Every))
			return
		}
	}
	got, err := a.m.AddTimer(r.Context(), t)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusCreated, got)
}

func (a *api) cancelTimer(w http.ResponseWriter, r *http.Request) {
	id, err := idOf(r, "timer")
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if err := a.m.CancelTimer(r.Context(), id, ""); err != nil {
		a.fail(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func idOf(r *http.Request, what string) (int64, error) {
	raw := r.PathValue("id")
	id, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || id <= 0 {
		return 0, fmt.Errorf("%w: %q is not a %s id", daemon.ErrBadRequest, raw, what)
	}
	return id, nil
}

func limitOf(raw string) (int, error) {
	if raw == "" {
		return defaultLimit, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("%w: limit=%q is not a positive number", daemon.ErrBadRequest, raw)
	}
	return min(n, maxLimit), nil
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
	frames := make(chan daemon.Frame)
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
func sendFrame(w http.ResponseWriter, rc *http.ResponseController, f daemon.Frame) bool {
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
func (a *api) sendEnd(w http.ResponseWriter, rc *http.ResponseController, s *daemon.Session) {
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
			return 0, fmt.Errorf("%w: Last-Event-ID %q is not a sequence number", daemon.ErrBadRequest, raw)
		}
		return n + 1, nil
	}
	if raw := strings.TrimSpace(r.URL.Query().Get("from")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			return 0, fmt.Errorf("%w: from=%q is not a sequence number", daemon.ErrBadRequest, raw)
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
				h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, Last-Event-ID")
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
func (a *api) session(r *http.Request) (*daemon.Session, error) {
	return a.m.Get(r.Context(), r.PathValue("id"))
}

// classify is the single place an error becomes a status and a token.
//
// Every handler returns through it. The alternative — each call site choosing
// its own http.StatusX — drifts the moment two handlers can produce the same
// error, and a client that special-cased one of them then breaks on the other.
func classify(err error) (status int, kind string) {
	switch {
	case errors.Is(err, daemon.ErrNoSuchSession):
		return http.StatusNotFound, kindNoSuchSession
	case errors.Is(err, daemon.ErrNoSuchApproval):
		return http.StatusNotFound, kindNoSuchApproval
	case errors.Is(err, daemon.ErrNoSuchTimer):
		return http.StatusNotFound, kindNoSuchTimer
	case errors.Is(err, daemon.ErrNoSuchMessage):
		return http.StatusNotFound, kindNoSuchMessage
	case errors.Is(err, errNotFound):
		return http.StatusNotFound, kindNotFound
	case errors.Is(err, daemon.ErrNotRunning):
		return http.StatusConflict, kindNotRunning
	case errors.Is(err, daemon.ErrDone):
		// A conflict with the resource's state: the session exists and the
		// request is well formed, but a conversation the model ended with a
		// terminal tool has nothing to continue from.
		return http.StatusConflict, kindDone
	case errors.Is(err, daemon.ErrAlreadyAnswered):
		return http.StatusConflict, kindAlreadyAnswered
	case errors.Is(err, daemon.ErrClosed):
		// Shutting down, not broken. 503 is the one status a client should
		// retry against another instance rather than report as a bug.
		return http.StatusServiceUnavailable, kindClosed
	case errors.Is(err, errBodyTooLarge):
		// A body over the cap is still a bad body, so it keeps the 400 the
		// pinned table gives one; the distinct kind is what tells a UI to say
		// "too long" rather than "malformed".
		return http.StatusBadRequest, kindBodyTooLarge
	case errors.Is(err, daemon.ErrBadRequest):
		return http.StatusBadRequest, kindBadRequest
	case errors.Is(err, errNotJSON):
		return http.StatusUnsupportedMediaType, kindNotJSON
	case errors.Is(err, errUnauthorized):
		return http.StatusUnauthorized, kindUnauthorized
	case errors.Is(err, errWrongHost):
		return http.StatusMisdirectedRequest, kindWrongHost
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
// for the same reason the session count is: nothing here need be
// authenticated.
//
// The body must be declared application/json, and that is a security check,
// not pedantry. text/plain is a type any web page may POST to any origin
// without a preflight, so a body decoded whatever its type let a page on any
// site the operator visited start a session here, prompt and all — and the
// session then ran tools on the operator's machine.
func decodeBody(w http.ResponseWriter, r *http.Request, dst any) error {
	if mt, _, err := mime.ParseMediaType(r.Header.Get("Content-Type")); err != nil || mt != "application/json" {
		return fmt.Errorf("%w (got %q)", errNotJSON, r.Header.Get("Content-Type"))
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()

	if err := dec.Decode(dst); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return fmt.Errorf("%w: %d bytes is the limit: %w", errBodyTooLarge, maxBodyBytes, err)
		}
		return fmt.Errorf("%w: %w", daemon.ErrBadRequest, err)
	}
	// Trailing content means the client sent something other than the single
	// object this endpoint documents, and accepting the first half of it would
	// hide whatever the second half was trying to say.
	if dec.More() {
		return fmt.Errorf("%w: unexpected trailing content after the JSON object", daemon.ErrBadRequest)
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
