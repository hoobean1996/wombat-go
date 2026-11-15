// Package openai is an [llm.Client] for OpenAI-compatible
// /v1/chat/completions endpoints: OpenAI itself, but equally vLLM, Ollama,
// OpenRouter, LiteLLM and the various corporate gateways that copy the shape.
//
// It is deliberately the *compat* dialect and not the Responses API: the
// chat-completions body is the one thing every one of those servers agrees on.
//
// The interesting work is not the transport — net/http does that — but the
// translation in request.go. [llm.Message] is a role plus a list of blocks,
// which is Anthropic's shape; OpenAI wants the system prompt as a message, a
// model's tool calls in a sibling tool_calls array, and every tool result as
// its own message. One llm.Message can therefore become several wire messages.
//
//	c, err := openai.New(openai.Config{Model: "gpt-4o"})   // key from $OPENAI_API_KEY
//	resp, err := c.Complete(ctx, llm.Request{System: "…", Messages: msgs})
//
// Corporate gateways usually need a proxy, routing headers and a non-default
// base URL, all of which [ConfigFromEnv] reads; the result is a plain Config,
// so anything the environment got wrong can be fixed before it reaches [New]:
//
//	cfg := openai.ConfigFromEnv()          // OPENAI_BASE_URL, OPENAI_PROXY, …
//	cfg.Stream = llm.StreamNever           // this gateway drops streamed usage
//	c, err := openai.New(cfg)
//
// A Client is immutable after New and safe for concurrent use by any number of
// goroutines, which is what lets one agent fan sub-agents across a shared
// client.
package openai

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/llm/internal/httpcall"
)

// Defaults applied by [New] when the corresponding [Config] field is zero.
const (
	// DefaultBaseURL already includes the /v1 prefix, because that is how
	// every gateway documents its own base URL ("point OPENAI_BASE_URL at
	// http://host:8000/v1"). The path appended to it is /chat/completions.
	DefaultBaseURL = "https://api.openai.com/v1"

	// DefaultModel matches the OCaml original. It exists so a zero Config is
	// usable in a scratch program; production callers set Model explicitly.
	DefaultModel = "gpt-4o"

	// DefaultMaxTokens caps one reply. Same value as the harness default, so
	// a request that does not override it produces the same body either way.
	DefaultMaxTokens = 8192
)

// Config constructs a Client. The zero value is usable if $OPENAI_API_KEY is
// set; every field is an override.
type Config struct {
	// APIKey falls back to $OPENAI_API_KEY. An empty key after that fallback
	// is NOT an error: local vLLM and Ollama servers reject a bogus
	// Authorization header, so the header is simply omitted when unset.
	APIKey string

	// BaseURL defaults to DefaultBaseURL. "/chat/completions" is appended to
	// it. A bare origin (no path) gets "/v1" filled in, because gateways
	// commonly publish the origin alone while OpenAI documents the /v1 form,
	// and the same environment variable carries both spellings in practice.
	// A URL that already has a path is used exactly as given.
	BaseURL string

	// Model is the default model id, used when llm.Request.Model is empty.
	Model string

	// MaxTokens is the default reply cap, used when llm.Request.MaxTokens
	// is zero.
	MaxTokens int

	// Temperature and TopP are the default sampling controls, used when the
	// corresponding llm.Request field is nil. Nil here too means this client
	// says nothing about sampling and the provider's own default stands — a
	// default the provider chose is not the same as a default you chose, and a
	// plain float64 could not tell those apart, because 0 is a real temperature
	// and the one that makes a run reproducible.
	//
	// They are independent: unlike llm/anthropic, this API accepts both at
	// once, so a configured TopP still applies to a request that overrides only
	// Temperature.
	//
	// [New] rejects a Temperature outside [0, 2] or a TopP outside [0, 1].
	Temperature *float64
	TopP        *float64

	// HTTPClient overrides the transport. The default has no overall Timeout
	// on purpose — that would kill a long stream mid-flight — and relies on
	// the caller's context plus a response-header deadline instead.
	//
	// It wins outright over Proxy: a caller who hands over a whole client has
	// already decided how its requests are routed, and quietly rewriting that
	// client's Transport would be both a surprise and a data race.
	HTTPClient *http.Client

	// Proxy routes requests through an HTTP proxy: "host:port" or a full
	// URL. Empty falls back to the standard proxy environment variables.
	//
	// The bare "host:port" form is accepted because that is what curl -x
	// takes and what the deployed gateway config carries; it is normalised to
	// http://host:port. A value that cannot be normalised is an error from
	// [New] rather than a silent fall back to a direct connection, which
	// would send corporate traffic straight at the internet.
	Proxy string

	// Stream controls SSE. The zero value (llm.StreamAuto) streams when the
	// request carries an OnDelta sink.
	//
	// llm.StreamAlways also streams the calls with no sink, which keeps a
	// gateway that buffers a whole non-streamed reply from tripping an idle
	// proxy timeout on a long generation.
	//
	// llm.StreamNever is the fallback when a gateway will not report usage
	// while streaming. This client always sends
	// stream_options:{"include_usage":true} and reads the usage record off
	// whichever chunk carries it — real OpenAI puts it on a trailing chunk
	// with no choices, other servers attach it to the finish_reason chunk,
	// and both work. A server that drops it regardless reports zero tokens,
	// which silently disables cost budgeting; forcing StreamNever trades
	// deltas for accounting on such an endpoint.
	Stream llm.StreamMode

	// ExtraHeaders are set last and may override the headers above. This is
	// the supported replacement for the OCaml client's $OPENAI_CUSTOM_HEADERS
	// env var: gateway routing keys belong in the caller's config, not in a
	// global that silently rewrites every request in the process.
	ExtraHeaders map[string]string
}

var _ llm.Client = (*Client)(nil)

// Client is an OpenAI-compatible chat-completions client.
type Client struct {
	endpoint  string
	apiKey    string
	model     string
	maxTokens int
	http      *http.Client
	stream    llm.StreamMode
	extra     map[string]string
	// temperature and topP are the per-client defaults. Nil means no opinion,
	// and the field never reaches the wire.
	temperature *float64
	topP        *float64
	// idle, total and maxBody are the httpcall defaults — the call budget
	// every provider client shares; fields only so a test can shorten them.
	idle, total time.Duration
	maxBody     int
}

// newTransport returns a fresh transport with this package's timeouts.
//
// It is a constructor and not a package-level value because Config.Proxy has
// to override Transport.Proxy on a transport nobody else holds. Mutating a
// shared one would be a data race against every in-flight request on it, and
// would reroute the traffic of unrelated clients that happen to share it.
//
// ResponseHeaderTimeout is the load-bearing setting: it bounds the wait for
// the first byte of the response (the OCaml client's 30s first-byte deadline)
// without bounding the body, which for a stream can legitimately run for
// minutes. There is deliberately no Client.Timeout for the same reason.
func newTransport() *http.Transport {
	return &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxIdleConnsPerHost:   16,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
	}
}

// defaultHTTPClient is shared by every Client that supplies neither its own
// client nor a proxy, so connection pooling survives across clients pointed at
// the same host.
var defaultHTTPClient = &http.Client{Transport: newTransport()}

// New validates cfg and returns a ready client. It only fails on programmer
// error — a base URL or proxy that will not parse, or a negative token cap —
// never on anything the network could cause.
func New(cfg Config) (*Client, error) {
	base := cfg.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	u, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("openai: bad base URL %q: %w", base, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("openai: base URL %q must be http or https", base)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("openai: base URL %q has no host", base)
	}

	// A bare origin gets /v1 appended.
	//
	// Deployments disagree about where the version prefix lives: OpenAI
	// documents a base of https://api.openai.com/v1, while gateways commonly
	// publish the origin alone and expect the client to know the path. Both
	// spellings are in the wild for the same OPENAI_BASE_URL variable, and
	// the failure mode of guessing wrong is a bare 404 that says nothing
	// useful.
	//
	// So: no path means the caller gave an origin, and /v1 is the only
	// sensible completion. A non-empty path is left exactly as given, which
	// keeps prefixed deployments (https://host/openai/v1, /api/llm/v1)
	// working — those are cases where guessing really would be wrong.
	if p := strings.Trim(u.Path, "/"); p == "" {
		u.Path = "/v1"
		base = u.String()
	}
	if cfg.MaxTokens < 0 {
		return nil, fmt.Errorf("openai: MaxTokens %d must not be negative", cfg.MaxTokens)
	}
	if err := validateSampling(cfg.Temperature, cfg.TopP); err != nil {
		return nil, err
	}

	key := cfg.APIKey
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	model := cfg.Model
	if model == "" {
		model = DefaultModel
	}
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = DefaultMaxTokens
	}
	// Validated even when HTTPClient makes it moot, so a typo in a deployment
	// config is reported where it was written instead of being ignored.
	proxy, err := parseProxy(cfg.Proxy)
	if err != nil {
		return nil, err
	}

	httpc := cfg.HTTPClient
	switch {
	case httpc != nil:
		// Caller-supplied client wins; see Config.HTTPClient.
	case proxy != nil:
		tr := newTransport()
		tr.Proxy = http.ProxyURL(proxy)
		httpc = &http.Client{Transport: tr}
	default:
		httpc = defaultHTTPClient
	}

	// Copied so a later mutation of the caller's map cannot race with an
	// in-flight request.
	var extra map[string]string
	if len(cfg.ExtraHeaders) > 0 {
		extra = make(map[string]string, len(cfg.ExtraHeaders))
		for k, v := range cfg.ExtraHeaders {
			extra[k] = v
		}
	}

	return &Client{
		endpoint:    strings.TrimSuffix(base, "/") + "/chat/completions",
		apiKey:      key,
		model:       model,
		maxTokens:   maxTokens,
		http:        httpc,
		stream:      cfg.Stream,
		extra:       extra,
		temperature: cfg.Temperature,
		topP:        cfg.TopP,
		idle:        httpcall.StreamIdleTimeout,
		total:       httpcall.RequestTimeout,
		maxBody:     httpcall.MaxResponseBody,
	}, nil
}

// validateSampling checks the [Config] sampling defaults.
//
// The provider rejects these values anyway; refusing them here names the field
// that is wrong rather than leaving a 400 body to say "invalid_request_error",
// and does it at construction, before the run has spent anything.
//
// Each bound is written as a negated in-range test so that NaN — false against
// every comparison, and exactly what strconv.ParseFloat returns for the string
// "NaN" in $OPENAI_TEMPERATURE — is rejected instead of sailing through.
func validateSampling(temperature, topP *float64) error {
	if temperature != nil && !(*temperature >= 0 && *temperature <= 2) {
		return fmt.Errorf("openai: Temperature %v is outside [0, 2]", *temperature)
	}
	if topP != nil && !(*topP >= 0 && *topP <= 1) {
		return fmt.Errorf("openai: TopP %v is outside [0, 1]", *topP)
	}
	return nil
}

// parseProxy normalises a [Config.Proxy] value. An empty value yields a nil
// URL, which leaves the transport on http.ProxyFromEnvironment.
//
// "host:port" is treated as an http proxy rather than rejected: url.Parse
// happily reads "localhost:3128" as scheme "localhost" with opaque "3128",
// so without the fix-up the common form would parse without error and then
// proxy nothing.
func parseProxy(raw string) (*url.URL, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return nil, nil
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil {
		return nil, fmt.Errorf("openai: bad proxy %q: %w", raw, err)
	}
	switch u.Scheme {
	case "http", "https", "socks5", "socks5h":
	default:
		return nil, fmt.Errorf("openai: proxy %q has unsupported scheme %q", raw, u.Scheme)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("openai: proxy %q has no host", raw)
	}
	return u, nil
}

// ConfigFromEnv reads a Config from the environment. Missing variables leave
// zero values, so the result can be overridden field by field before it
// reaches New.
//
// The two-name lookups exist because the deployed configuration predates this
// client: WOMBAT_MODEL and OPENAI_CUSTOM_HEADERS are the older spellings, and
// the OPENAI_-prefixed name wins so a per-provider override is possible in a
// process that also runs another provider.
//
//	$OPENAI_TEMPERATURE -> Temperature
//	$OPENAI_TOP_P       -> TopP
//
// Those two are the only ones with a parse step, and it has a seam worth
// naming: a value that is not a number is dropped here with a warning, while a
// number out of range survives to be reported by [New]. A typo must not stop a
// run that would otherwise work; "9" where a top_p belongs is a decision, and
// one worth failing over.
func ConfigFromEnv() Config {
	return Config{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		BaseURL:      os.Getenv("OPENAI_BASE_URL"),
		Proxy:        os.Getenv("OPENAI_PROXY"),
		Model:        firstEnv("OPENAI_MODEL", "WOMBAT_MODEL"),
		ExtraHeaders: ParseHeaders(firstEnv("OPENAI_EXTRA_HEADERS", "OPENAI_CUSTOM_HEADERS")),
		Stream:       parseStreamMode(os.Getenv("OPENAI_STREAM")),
		Temperature:  envFloat("OPENAI_TEMPERATURE"),
		TopP:         envFloat("OPENAI_TOP_P"),
	}
}

// envFloat reads one variable as an optional float64. Unset or blank yields
// nil, which is what keeps the field off the wire entirely.
//
// An unparseable value is logged and ignored rather than reported, because
// ConfigFromEnv has no error channel and a fat-fingered $OPENAI_TEMPERATURE
// must not be the reason a run does not start. It is warned about rather than
// dropped in silence: the run then samples at the provider's default, which is
// indistinguishable from the setting being ignored — because it was.
func envFloat(name string) *float64 {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		slog.Warn("openai: ignoring unparseable sampling variable",
			slog.String("var", name), slog.String("value", raw))
		return nil
	}
	return &v
}

// firstEnv returns the first of names that is set to a non-empty value.
func firstEnv(names ...string) string {
	for _, n := range names {
		if v := os.Getenv(n); v != "" {
			return v
		}
	}
	return ""
}

// ParseHeaders reads newline-separated "Name: value" lines into a header map,
// the format the deployed gateway configuration already uses for its routing
// keys. It returns nil for an empty or header-less input so it can be assigned
// straight to [Config.ExtraHeaders].
//
// It is deliberately forgiving — blank lines and lines with no colon are
// skipped rather than reported — because the alternative is a process that
// refuses to start over a stray line in an env var, and there is no error
// channel on the config path anyway.
func ParseHeaders(s string) map[string]string {
	var out map[string]string
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		name, value, ok := strings.Cut(line, ":")
		name = strings.TrimSpace(name)
		if !ok || name == "" {
			continue
		}
		if out == nil {
			out = map[string]string{}
		}
		out[name] = strings.TrimSpace(value)
	}
	return out
}

// parseStreamMode reads the OPENAI_STREAM spelling of [llm.StreamMode]. An
// unrecognised value falls back to llm.StreamAuto: the env var is a hint from
// an operator, and the auto default behaves sensibly either way.
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

// Complete implements [llm.Client].
func (c *Client) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = c.maxTokens
	}
	// See Config.Stream: llm.StreamAuto (the zero value) streams exactly when
	// somebody is listening.
	stream := c.stream.Enabled(req.OnDelta != nil)
	// Merged onto the local copy of the request — Complete takes it by value —
	// so encodeRequest has exactly one place to read sampling from and cannot
	// drift from the precedence rule. The caller's Request is untouched.
	req.Temperature, req.TopP = c.sampling(req)

	body, err := encodeRequest(req, model, maxTokens, stream)
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
		return llm.Response{}, fmt.Errorf("openai: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	} else {
		httpReq.Header.Set("Accept", "application/json")
	}
	for k, v := range c.extra {
		httpReq.Header.Set(k, v)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return llm.Response{}, httpcall.Abort(ctx, "openai", fmt.Errorf("openai: post %s: %w", c.endpoint, err))
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return llm.Response{}, httpcall.StatusError(resp)
	}

	var out llm.Response
	if stream {
		out, err = decodeStream(httpcall.Cap(watch.Body(resp.Body), "openai", c.maxBody), req.OnDelta, model)
	} else {
		out, err = decodeResponse(resp.Body, model, c.maxBody)
	}
	// A body read cut off by the context is the context's doing, and only the
	// cause says whether that was a stall or the caller stopping the run.
	if err = httpcall.Settle(ctx, "openai", err); err != nil {
		return llm.Response{}, err
	}
	return out, nil
}
