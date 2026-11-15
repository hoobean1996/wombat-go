// Package anthropic is an [llm.Client] for the Anthropic Messages API.
//
//	cl, err := anthropic.New(anthropic.Config{Model: "claude-opus-5"})
//	resp, err := cl.Complete(ctx, llm.Request{Messages: []llm.Message{llm.UserText("hi")}})
//
// # Why there is no subprocess here
//
// The OCaml original spawned curl and re-implemented HTTP on top of it: a
// pipe pair per call, a hand-rolled Unix.select line reader to get first-byte
// and idle deadlines, a status-line parser that had to cope with proxy CONNECT
// responses interleaved in the same stdout, and a waitpid to recover the exit
// code. All of that is deleted. net/http already gives us, for free and
// correctly: connection reuse, proxy support ([Config.Proxy] or the standard
// proxy environment variables), per-phase timeouts on the Transport, and a body
// that unblocks the moment the request context is cancelled — which is also how
// [governor] enforces a budget. The one thing the standard library does not
// give us is a mid-body idle deadline, so exactly that one thing is rebuilt
// here, in six lines, with a time.AfterFunc that cancels the request context.
//
// # Prompt caching
//
// This client places three cache_control breakpoints — system, tool list, last
// message — and their placement is load-bearing. See request.go: getting it
// wrong does not fail, it silently doubles the bill.
//
// # Concurrency
//
// A Client is immutable after [New] and safe for concurrent use; sub-agents
// fanned across goroutines share one client and one connection pool.
package anthropic

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/internal/httpcall"
)

// Defaults applied by [New] when the corresponding Config field is zero.
const (
	DefaultBaseURL   = "https://api.anthropic.com"
	DefaultVersion   = "2023-06-01"
	DefaultMaxTokens = 8192
)

// BetaContextManagement enables server-side context management. Put it in
// [Config.Beta] to have the client emit the clear_tool_uses edit described in
// request.go; without it the block is omitted, because sending
// context_management without the beta header is a 400.
const BetaContextManagement = "context-management-2025-06-27"

// ErrNoAPIKey is returned by [New] when neither Config.APIKey nor
// $ANTHROPIC_API_KEY is set.
var ErrNoAPIKey = errors.New("anthropic: no API key (set Config.APIKey or $ANTHROPIC_API_KEY)")

// Transport deadlines.
//
// The values are the first layer of a budget whose other two layers — the
// stream idle watchdog and the total cap — live in llm/internal/httpcall,
// shared with every other provider client:
//
//   - responseHeaderTimeout catches a server that accepts the connection and
//     then never answers — the CLOSE_WAIT wedge seen in production;
//   - httpcall.StreamIdleTimeout catches a stream that starts and then stops
//     mid-body, which no net/http timeout covers;
//   - httpcall.RequestTimeout is the total cap, and applies ONLY when the
//     caller's context carries no deadline of its own.
const (
	dialTimeout           = 10 * time.Second
	tlsTimeout            = 10 * time.Second
	responseHeaderTimeout = 90 * time.Second
)

// Config configures a [Client]. The zero value is usable as long as
// $ANTHROPIC_API_KEY is set.
type Config struct {
	// APIKey authenticates the call. Falls back to $ANTHROPIC_API_KEY.
	APIKey string

	// BaseURL is the API root, without the /v1/messages path. Falls back to
	// $ANTHROPIC_BASE_URL, then [DefaultBaseURL]. Set it to route through a
	// corporate gateway; to reach that gateway through an HTTP proxy, set
	// [Config.Proxy].
	BaseURL string

	// Proxy routes requests through an HTTP proxy: "host:port" or a full
	// URL. Empty falls back to the standard proxy environment variables.
	//
	// The bare "host:port" form is accepted — and normalised to
	// http://host:port — because that is what curl -x takes and what the
	// deployed gateway config already stores. It is rejected outright, from
	// [New], if it cannot be turned into a usable proxy URL: a typo that
	// silently sends corporate traffic direct to the internet is the exact
	// failure this field exists to prevent.
	//
	// Ignored when [Config.HTTPClient] is set — that client brings its own
	// Transport, and its proxy policy wins outright — but still validated
	// there, so a bad value is never quietly accepted.
	Proxy string

	// Model is used when llm.Request.Model is empty.
	Model string

	// MaxTokens is used when llm.Request.MaxTokens is 0. Defaults to
	// [DefaultMaxTokens].
	MaxTokens int

	// Temperature and TopP are the default sampling controls, used when the
	// corresponding llm.Request field is nil. Nil here too means the client
	// says nothing about sampling and the provider's own default applies — a
	// default the provider chose is not the same as a default you chose, and a
	// plain float64 could not tell the two apart, because 0 is a real
	// temperature and the one that pins a run for a differential diff.
	//
	// [New] rejects a temperature outside [0, 2] or a top_p outside [0, 1].
	//
	// The Messages API accepts one of the two, never both, so [New] also
	// rejects a Config that sets both. The same rule at request time is a
	// precedence rule rather than an error: a Request that names either control
	// supplies both, so these defaults are ignored wholesale rather than
	// half-merged into an illegal body. See the sampling method in request.go.
	Temperature *float64
	TopP        *float64

	// Version is the anthropic-version header. Defaults to [DefaultVersion].
	Version string

	// Beta lists anthropic-beta values, joined into one header.
	// [BetaContextManagement] additionally switches on the context_management
	// request block.
	//
	// An "anthropic-beta" entry in [Config.ExtraHeaders] replaces this header
	// wholesale, and then it — not this field — is what decides whether the
	// context_management block is emitted. See ExtraHeaders. [ConfigFromEnv]
	// avoids the ambiguity entirely by folding such a header into this field.
	Beta []string

	// ExtraHeaders are applied last and may override any header set above,
	// including anthropic-beta. Gateway routing keys belong in the caller's
	// config, not in a process-global that rewrites every request.
	//
	// Overriding anthropic-beta also moves the context_management decision:
	// this package emits that block only when the anthropic-beta value it
	// actually sends contains [BetaContextManagement], because the API answers
	// a body it did not enable with "Extra inputs are not permitted" — a 400 on
	// every single request. So an override that omits the value silently turns
	// context management off, rather than breaking the client; an override that
	// carries it keeps the block, whatever [Config.Beta] says. Keys are matched
	// case-insensitively and sent in net/http's canonical spelling, as HTTP
	// header names are case-insensitive; two spellings of one name therefore
	// collapse into one header instead of being sent twice.
	//
	// [ConfigFromEnv] never returns an anthropic-beta entry here: it moves that
	// line into [Config.Beta] so the environment cannot produce a Config with
	// two disagreeing beta lists.
	ExtraHeaders map[string]string

	// HTTPClient overrides the default client. The default has no total
	// Timeout — that would cut a long stream off at the knees — and instead
	// bounds each phase on the Transport. Supply your own to add tracing,
	// custom headers or a different proxy policy; doing so also takes over
	// proxying, so [Config.Proxy] is ignored.
	HTTPClient *http.Client

	// Stream controls SSE. The zero value (llm.StreamAuto) streams when the
	// request carries an OnDelta sink.
	//
	// llm.StreamAlways is worth asking for even with no sink — it keeps a long,
	// high-max_tokens generation from tripping an idle proxy timeout — and
	// llm.StreamNever is not hypothetical either: some gateways drop the usage
	// record from a streamed reply, so a caller that needs token accounting
	// more than it needs deltas has to be able to force the buffered path.
	Stream llm.StreamMode
}

// Environment variables read by [ConfigFromEnv].
const (
	envAPIKey        = "ANTHROPIC_API_KEY"
	envBaseURL       = "ANTHROPIC_BASE_URL"
	envProxy         = "ANTHROPIC_PROXY"
	envModel         = "ANTHROPIC_MODEL"
	envCustomHeaders = "ANTHROPIC_CUSTOM_HEADERS"
	envBeta          = "ANTHROPIC_BETA"
	envStream        = "ANTHROPIC_STREAM"
	envTemperature   = "ANTHROPIC_TEMPERATURE"
	envTopP          = "ANTHROPIC_TOP_P"

	// The AGENT_LLM_* and WOMBAT_* names are not invented here: they are what
	// the existing deployment already exports, so honouring them as fallbacks
	// lets a working config file be reused verbatim instead of duplicated
	// under a second set of names that can then drift.
	envBaseURLAlt = "AGENT_LLM_BASE_URL"
	envProxyAlt   = "AGENT_LLM_PROXY"
	envModelAlt   = "WOMBAT_MODEL"
)

// ConfigFromEnv reads a Config from the environment. Missing variables
// leave zero values, so the result can be overridden field by field
// before it reaches New.
//
//	$ANTHROPIC_API_KEY        -> APIKey
//	$ANTHROPIC_BASE_URL       -> BaseURL, else $AGENT_LLM_BASE_URL
//	$ANTHROPIC_PROXY          -> Proxy,   else $AGENT_LLM_PROXY
//	$ANTHROPIC_MODEL          -> Model,   else $WOMBAT_MODEL
//	$ANTHROPIC_CUSTOM_HEADERS -> ExtraHeaders, newline-separated "Name: value"
//	$ANTHROPIC_BETA           -> Beta, comma-separated
//	$ANTHROPIC_STREAM         -> Stream, "auto" | "always" | "never"
//	$ANTHROPIC_TEMPERATURE    -> Temperature
//	$ANTHROPIC_TOP_P          -> TopP
//
// The AGENT_LLM_* and WOMBAT_MODEL fallbacks are the names the existing
// deployment already uses; reading them means an operator's current config
// file works here unchanged.
//
// One normalisation, so the two beta inputs cannot silently fight: an
// anthropic-beta line in $ANTHROPIC_CUSTOM_HEADERS wins and is moved into
// [Config.Beta], replacing whatever $ANTHROPIC_BETA said, and is NOT left in
// ExtraHeaders. The returned Config therefore names the beta list exactly once,
// which is what makes "override field by field" honest — editing Beta on a
// Config that also carried a beta header would otherwise do nothing.
//
// Nothing is validated here; an unparseable proxy or base URL is reported by
// [New], which is the call that can return an error. The sampling variables are
// the one place with a seam: a value that is not a number at all is dropped
// here with a warning, while a number out of range survives to be named by
// [New]. A typo must not stop a run that would otherwise work, but "0.9" where
// a temperature belongs is a decision, and one worth reporting.
func ConfigFromEnv() Config {
	cfg := Config{
		APIKey:       os.Getenv(envAPIKey),
		BaseURL:      firstEnv(envBaseURL, envBaseURLAlt),
		Proxy:        firstEnv(envProxy, envProxyAlt),
		Model:        firstEnv(envModel, envModelAlt),
		ExtraHeaders: parseHeaderLines(os.Getenv(envCustomHeaders)),
		Beta:         splitList(os.Getenv(envBeta)),
		Stream:       parseStreamMode(os.Getenv(envStream)),
		Temperature:  envFloat(envTemperature),
		TopP:         envFloat(envTopP),
	}
	if v, ok := cfg.ExtraHeaders[betaHeader]; ok {
		delete(cfg.ExtraHeaders, betaHeader)
		if len(cfg.ExtraHeaders) == 0 {
			cfg.ExtraHeaders = nil
		}
		cfg.Beta = splitList(v)
	}
	return cfg
}

// firstEnv returns the first variable that is set to a non-empty value. Empty
// counts as unset: an exported-but-blank variable is how a shell config says
// "I did not configure this", and treating it as a deliberate empty string
// would suppress the fallback for no reason.
func firstEnv(names ...string) string {
	for _, n := range names {
		if v := strings.TrimSpace(os.Getenv(n)); v != "" {
			return v
		}
	}
	return ""
}

// parseHeaderLines reads newline-separated "Name: value" pairs.
//
// Keys come back in http.Header's canonical spelling so that two lines naming
// the same header collapse — last one wins — instead of being applied twice in
// map order. A line with no colon is skipped rather than failing the whole
// parse: this value usually arrives from a shell config, and one fat-fingered
// line should not cost the caller its gateway routing keys.
func parseHeaderLines(s string) map[string]string {
	out := map[string]string{}
	for line := range strings.SplitSeq(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		name, value, ok := strings.Cut(line, ":")
		name = strings.TrimSpace(name)
		if !ok || name == "" {
			continue
		}
		out[http.CanonicalHeaderKey(name)] = strings.TrimSpace(value)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// envFloat reads one variable as an optional float64. Unset or blank yields
// nil, which is what keeps the field off the wire entirely.
//
// A value that will not parse is logged and ignored rather than returned as an
// error, and that is the whole point of doing it here: ConfigFromEnv has no
// error channel, and a fat-fingered ANTHROPIC_TEMPERATURE must not be able to
// stop a run that would otherwise work. It is warned about rather than dropped
// in silence because the run then samples at the provider's default, which
// looks like the setting being ignored — which it is.
func envFloat(name string) *float64 {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		slog.Warn("anthropic: ignoring unparseable sampling variable",
			slog.String("var", name), slog.String("value", raw))
		return nil
	}
	return &v
}

// splitList parses a comma-separated list, dropping blanks.
func splitList(s string) []string {
	var out []string
	for f := range strings.SplitSeq(s, ",") {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

// parseStreamMode maps the spelling to the mode. Anything unrecognised —
// including the empty string — is llm.StreamAuto, because ConfigFromEnv has no
// way to report an error and refusing to run over a typo'd preference would be
// a worse outcome than using the default.
func parseStreamMode(s string) llm.StreamMode {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "always":
		return llm.StreamAlways
	case "never":
		return llm.StreamNever
	default:
		return llm.StreamAuto
	}
}

// Client calls the Anthropic Messages API. Build one with [New].
type Client struct {
	apiKey    string
	endpoint  string
	model     string
	maxTokens int
	version   string
	beta      string // pre-joined header value
	ctxMgmt   bool
	stream    llm.StreamMode
	// temperature and topP are the per-client defaults. Nil means the client
	// has no opinion and the field never reaches the wire. New guarantees at
	// most one of them is non-nil.
	temperature *float64
	topP        *float64
	// extra is keyed by canonical header name and copied from the Config, so a
	// caller mutating its own map later cannot reach into a live client.
	extra map[string]string
	hc    *http.Client
	// idle, total and maxBody are the httpcall defaults; fields only so a test
	// can shorten them.
	idle, total time.Duration
	maxBody     int
}

// New validates cfg and builds a client. It performs no I/O.
func New(cfg Config) (*Client, error) {
	key := cfg.APIKey
	if key == "" {
		key = os.Getenv(envAPIKey)
	}
	if key == "" {
		return nil, ErrNoAPIKey
	}

	base := cfg.BaseURL
	if base == "" {
		// Only the primary name here. The deployment-specific fallbacks are
		// [ConfigFromEnv]'s job: New is also called with a Config assembled in
		// code, and that caller has not opted into anyone's env conventions.
		base = os.Getenv(envBaseURL)
	}
	if base == "" {
		base = DefaultBaseURL
	}
	base = strings.TrimRight(base, "/")
	u, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("anthropic: bad base URL %q: %w", base, err)
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("anthropic: base URL %q needs a scheme and host", base)
	}

	if cfg.MaxTokens < 0 {
		return nil, fmt.Errorf("anthropic: negative MaxTokens %d", cfg.MaxTokens)
	}

	if err := validateSampling(cfg.Temperature, cfg.TopP); err != nil {
		return nil, err
	}

	c := &Client{
		apiKey:      key,
		endpoint:    base + "/v1/messages",
		model:       cfg.Model,
		maxTokens:   cfg.MaxTokens,
		version:     cfg.Version,
		beta:        strings.Join(cfg.Beta, ","),
		stream:      cfg.Stream,
		temperature: cfg.Temperature,
		topP:        cfg.TopP,
		extra:       canonicalHeaders(cfg.ExtraHeaders),
		hc:          cfg.HTTPClient,
		idle:        httpcall.StreamIdleTimeout,
		total:       httpcall.RequestTimeout,
		maxBody:     httpcall.MaxResponseBody,
	}
	if c.maxTokens == 0 {
		c.maxTokens = DefaultMaxTokens
	}
	if c.version == "" {
		c.version = DefaultVersion
	}
	// Gate the body block on the header we will actually send, not on
	// cfg.Beta: an ExtraHeaders override replaces the header, and a
	// context_management block whose beta is not enabled is a 400 on every
	// request rather than a degraded one.
	if v, ok := c.extra[betaHeader]; ok {
		c.ctxMgmt = hasBeta(strings.Split(v, ","), BetaContextManagement)
	} else {
		c.ctxMgmt = hasBeta(cfg.Beta, BetaContextManagement)
	}

	// Validated even when HTTPClient is about to win, so a malformed proxy is
	// reported wherever it appears rather than only in the configurations that
	// happen to use it.
	var proxy *url.URL
	if cfg.Proxy != "" {
		if proxy, err = normalizeProxy(cfg.Proxy); err != nil {
			return nil, err
		}
	}
	switch {
	case c.hc != nil:
		// The caller's client, and therefore the caller's proxy policy.
	case proxy != nil:
		c.hc = proxyHTTPClient(proxy)
	default:
		c.hc = defaultHTTPClient()
	}
	return c, nil
}

// validateSampling checks the [Config] sampling defaults.
//
// The provider would reject these values anyway; catching them here names the
// field that is wrong instead of leaving a 400 to say "invalid request body",
// and it does so at construction, before a run has spent anything.
//
// Both bounds are written as a negated in-range test so that NaN — which
// compares false against everything, and which strconv.ParseFloat will happily
// produce from the string "NaN" — is rejected rather than sailing through.
func validateSampling(temperature, topP *float64) error {
	if temperature != nil && !(*temperature >= 0 && *temperature <= 2) {
		return fmt.Errorf("anthropic: Temperature %v is outside [0, 2]", *temperature)
	}
	if topP != nil && !(*topP >= 0 && *topP <= 1) {
		return fmt.Errorf("anthropic: TopP %v is outside [0, 1]", *topP)
	}
	// The Messages API takes one or the other. Refused at construction because
	// a Config that names both has no defensible reading — unlike a Request,
	// which overrides the pair (see Client.sampling) — and because a client
	// whose every call is a guaranteed 400 should not exist.
	if temperature != nil && topP != nil {
		return fmt.Errorf("anthropic: Config sets both Temperature (%v) and TopP (%v); the Messages API accepts one or the other",
			*temperature, *topP)
	}
	return nil
}

// betaHeader is the canonical spelling of the beta header, used as the
// ExtraHeaders lookup key. http.Header canonicalises on Set, so the map must
// be canonicalised too or an override would be applied twice under two names.
const betaHeader = "Anthropic-Beta"

// canonicalHeaders copies m with http.Header's key spelling. Returns nil for an
// empty input so the common case costs no allocation.
func canonicalHeaders(m map[string]string) map[string]string {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[http.CanonicalHeaderKey(strings.TrimSpace(k))] = v
	}
	return out
}

// hasBeta reports whether values contains want, ignoring the spacing a
// hand-written comma list picks up.
func hasBeta(values []string, want string) bool {
	for _, v := range values {
		if strings.TrimSpace(v) == want {
			return true
		}
	}
	return false
}

// normalizeProxy turns a Config.Proxy value into a fixed proxy URL.
//
// A bare "host:port" is prefixed with http:// rather than handed to url.Parse
// as-is: Parse reads "localhost:3128" as scheme "localhost" with opaque
// "3128", which http.Transport then ignores, so the traffic would leave the
// machine unproxied and the mistake would only show up as a connection refused
// somewhere far away. The scheme allowlist is there for the same reason —
// http.Transport only understands these four, and anything else would be a
// proxy setting that silently does nothing.
func normalizeProxy(raw string) (*url.URL, error) {
	s := strings.TrimSpace(raw)
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil {
		return nil, fmt.Errorf("anthropic: bad proxy %q: %w", raw, err)
	}
	switch u.Scheme {
	case "http", "https", "socks5", "socks5h":
	default:
		return nil, fmt.Errorf("anthropic: proxy %q: unsupported scheme %q", raw, u.Scheme)
	}
	if u.Hostname() == "" {
		return nil, fmt.Errorf("anthropic: proxy %q needs a host", raw)
	}
	return u, nil
}

// defaultHTTPClient bounds each phase of a call rather than the whole call.
//
// http.Client.Timeout is deliberately left at 0: it covers reading the body
// too, so any value large enough for a ten-minute stream is useless as a
// connect timeout, and any value small enough to be a useful connect timeout
// truncates the stream. Per-phase Transport timeouts have neither problem.
func defaultHTTPClient() *http.Client {
	return &http.Client{Transport: newTransport(http.ProxyFromEnvironment)}
}

// proxyHTTPClient is defaultHTTPClient pinned to one proxy.
//
// It builds a fresh Transport instead of setting Proxy on an existing one.
// Mutating a Transport that other clients already hold is a data race against
// their in-flight requests, and — worse than the race detector shouting — it
// would reroute traffic that never asked to be proxied. Every client gets its
// own Transport, so the only thing shared between two clients built from
// different Configs is the code below.
func proxyHTTPClient(u *url.URL) *http.Client {
	return &http.Client{Transport: newTransport(http.ProxyURL(u))}
}

// newTransport is the one place the per-phase timeouts are spelled out, so the
// proxied and unproxied clients cannot drift apart.
func newTransport(proxy func(*http.Request) (*url.URL, error)) *http.Transport {
	return &http.Transport{
		Proxy: proxy,
		DialContext: (&net.Dialer{
			Timeout:   dialTimeout,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		TLSHandshakeTimeout:   tlsTimeout,
		ResponseHeaderTimeout: responseHeaderTimeout,
		ExpectContinueTimeout: time.Second,
		IdleConnTimeout:       90 * time.Second,
		MaxIdleConns:          64,
		// One agent fans sub-agents across goroutines against a single
		// host; the stdlib default of 2 idle conns per host would force a
		// fresh TLS handshake for most of them.
		MaxIdleConnsPerHost: 32,
	}
}

// Complete implements [llm.Client].
func (c *Client) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	if model == "" {
		return llm.Response{}, fmt.Errorf("anthropic: no model on the request and no client default: %w", llm.ErrBadRequest)
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = c.maxTokens
	}

	stream := c.stream.Enabled(req.OnDelta != nil)

	body, err := c.encode(req, model, maxTokens, stream)
	if err != nil {
		return llm.Response{}, err
	}

	// The shared call budget: our own total cap only when the caller set no
	// deadline, and for a stream a watchdog that cancels on silence.
	ctx, cancel := httpcall.Bound(ctx, c.total)
	defer cancel()
	var watch *httpcall.Watchdog
	if stream {
		ctx, watch = httpcall.Watch(ctx, c.idle)
		defer watch.Stop()
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return llm.Response{}, fmt.Errorf("anthropic: build request: %w", err)
	}
	c.setHeaders(httpReq, stream)

	resp, err := c.hc.Do(httpReq)
	if err != nil {
		return llm.Response{}, httpcall.Abort(ctx, "anthropic", fmt.Errorf("anthropic: POST %s: %w", c.endpoint, err))
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return llm.Response{}, httpcall.StatusError(resp)
	}

	var out llm.Response
	if stream {
		out, err = readStream(httpcall.Cap(watch.Body(resp.Body), "anthropic", c.maxBody), req.OnDelta, model)
	} else {
		out, err = decodeMessage(resp.Body, model, c.maxBody)
	}
	// A body read cut off by the context is the context's doing, and only the
	// cause says whether that was a stall or the caller stopping the run.
	if err = httpcall.Settle(ctx, "anthropic", err); err != nil {
		return llm.Response{}, err
	}
	return out, nil
}

// setHeaders writes the protocol headers, then lets Config.ExtraHeaders have
// the last word — including over anthropic-beta, whose override New has already
// taken into account when deciding on the context_management block.
func (c *Client) setHeaders(r *http.Request, stream bool) {
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("X-Api-Key", c.apiKey)
	r.Header.Set("Anthropic-Version", c.version)
	if c.beta != "" {
		r.Header.Set(betaHeader, c.beta)
	}
	if stream {
		r.Header.Set("Accept", "text/event-stream")
	} else {
		r.Header.Set("Accept", "application/json")
	}
	for k, v := range c.extra {
		r.Header.Set(k, v)
	}
}

// scrub replaces invalid UTF-8 with U+FFFD.
//
// It runs on the way IN, on everything the provider says, and that direction
// is the point. Go strings happily hold invalid bytes, but json.Marshal
// silently rewrites them to U+FFFD on the way out — so an assistant turn that
// was cut off mid-codepoint by max_tokens, or a tool result carrying
// mis-encoded file bytes, becomes a message that we store one way and send
// another. Worse, the raw bytes of a tool_use input bypass json.Marshal
// entirely (json.RawMessage is copied verbatim) and poison every subsequent
// request with a 400 that no retry can clear. Scrubbing at ingress means the
// transcript we persist is byte-identical to the transcript we replay.
func scrub(s string) string { return strings.ToValidUTF8(s, "�") }
