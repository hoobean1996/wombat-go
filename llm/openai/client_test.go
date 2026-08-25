package openai

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

func TestNewValidation(t *testing.T) {
	clearEnv(t)

	tests := []struct {
		name    string
		cfg     Config
		wantErr string
	}{
		{name: "base URL that will not parse", cfg: Config{BaseURL: "://nope"}, wantErr: "bad base URL"},
		{name: "base URL with no scheme", cfg: Config{BaseURL: "gw.internal/v1"}, wantErr: "must be http or https"},
		{name: "base URL with a non-http scheme", cfg: Config{BaseURL: "ftp://gw.internal/v1"}, wantErr: "must be http or https"},
		{name: "base URL with no host", cfg: Config{BaseURL: "https:///v1"}, wantErr: "has no host"},
		{name: "negative max tokens", cfg: Config{MaxTokens: -1}, wantErr: "must not be negative"},
		{name: "proxy with an unsupported scheme", cfg: Config{Proxy: "ftp://p:21"}, wantErr: "unsupported scheme"},
		{name: "proxy with no host", cfg: Config{Proxy: "http://"}, wantErr: "has no host"},
		{
			// Validated even where HTTPClient makes it moot, so a typo is
			// reported where it was written.
			name:    "bad proxy is rejected even when HTTPClient wins",
			cfg:     Config{Proxy: "ftp://p:21", HTTPClient: &http.Client{}},
			wantErr: "unsupported scheme",
		},
		{name: "an empty API key is fine (local vLLM / Ollama)", cfg: Config{}},
		{name: "zero MaxTokens is fine", cfg: Config{MaxTokens: 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := New(tt.cfg)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("New: got error %v, want nil", err)
				}
				if c == nil {
					t.Fatal("New: got nil client, want a client")
				}
				return
			}
			if err == nil {
				t.Fatalf("New: got nil error, want one containing %q", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("New error: got %q, want it to contain %q", err, tt.wantErr)
			}
		})
	}
}

// TestBaseURLV1Completion is the documented asymmetry: a bare ORIGIN gets /v1
// filled in, because gateways publish the origin alone while OpenAI documents
// the /v1 form and the same env var carries both. A URL that already has a
// path is used exactly as given — guessing there really would be wrong.
func TestBaseURLV1Completion(t *testing.T) {
	clearEnv(t)

	tests := []struct {
		name string
		base string
		want string
	}{
		{name: "default", base: "", want: "https://api.openai.com/v1/chat/completions"},
		{name: "bare origin gets /v1", base: "https://gw.internal", want: "https://gw.internal/v1/chat/completions"},
		{name: "origin with a lone slash counts as bare", base: "https://gw.internal/", want: "https://gw.internal/v1/chat/completions"},
		{name: "origin with a port", base: "http://localhost:8000", want: "http://localhost:8000/v1/chat/completions"},
		{name: "an explicit /v1 is left alone", base: "https://gw.internal/v1", want: "https://gw.internal/v1/chat/completions"},
		{name: "a prefixed deployment is left alone", base: "https://gw.internal/openai/v1", want: "https://gw.internal/openai/v1/chat/completions"},
		{name: "a prefixed deployment with a trailing slash", base: "https://gw.internal/api/llm/v1/", want: "https://gw.internal/api/llm/v1/chat/completions"},
		{name: "a path that is not a version prefix is still left alone", base: "https://gw.internal/proxy", want: "https://gw.internal/proxy/chat/completions"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := New(Config{BaseURL: tt.base})
			if err != nil {
				t.Fatalf("New(%q): got error %v, want nil", tt.base, err)
			}
			if c.endpoint != tt.want {
				t.Errorf("endpoint for base %q: got %q, want %q", tt.base, c.endpoint, tt.want)
			}
		})
	}
}

func TestNewDefaults(t *testing.T) {
	clearEnv(t)
	c, err := New(Config{})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	if c.model != DefaultModel {
		t.Errorf("model: got %q, want %q", c.model, DefaultModel)
	}
	if c.maxTokens != DefaultMaxTokens {
		t.Errorf("maxTokens: got %d, want %d", c.maxTokens, DefaultMaxTokens)
	}
	if c.http != defaultHTTPClient {
		t.Errorf("http client: got %p, want the shared default %p (pooling must survive across clients)", c.http, defaultHTTPClient)
	}
}

func TestNewAPIKeyFromEnv(t *testing.T) {
	clearEnv(t)
	t.Setenv("OPENAI_API_KEY", "from-env")
	c, err := New(Config{})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	if got, want := c.apiKey, "from-env"; got != want {
		t.Errorf("apiKey: got %q, want %q", got, want)
	}
}

// TestProxyNormalisation: url.Parse reads "localhost:3128" as scheme
// "localhost" with opaque "3128", which http.Transport silently ignores — the
// traffic would leave the machine unproxied.
func TestProxyNormalisation(t *testing.T) {
	tests := []struct {
		in      string
		want    string // "" means nil URL
		wantErr string
	}{
		{in: "", want: ""},
		{in: "   ", want: ""},
		{in: "localhost:3128", want: "http://localhost:3128"},
		{in: "  proxy.internal:3128 ", want: "http://proxy.internal:3128"},
		{in: "http://proxy.internal:3128", want: "http://proxy.internal:3128"},
		{in: "https://proxy.internal:3128", want: "https://proxy.internal:3128"},
		{in: "socks5://proxy.internal:1080", want: "socks5://proxy.internal:1080"},
		{in: "socks5h://proxy.internal:1080", want: "socks5h://proxy.internal:1080"},
		{in: "ftp://p:21", wantErr: "unsupported scheme"},
		{in: "http://", wantErr: "has no host"},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			u, err := parseProxy(tt.in)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("parseProxy(%q): got (%v, %v), want an error containing %q", tt.in, u, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseProxy(%q): got error %v, want nil", tt.in, err)
			}
			if tt.want == "" {
				if u != nil {
					t.Errorf("parseProxy(%q): got %q, want nil so the transport keeps ProxyFromEnvironment", tt.in, u)
				}
				return
			}
			if u == nil || u.String() != tt.want {
				t.Errorf("parseProxy(%q): got %v, want %q", tt.in, u, tt.want)
			}
		})
	}
}

func TestProxyReachesTheTransport(t *testing.T) {
	clearEnv(t)
	c, err := New(Config{Proxy: "proxy.internal:3128"})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	if c.http == defaultHTTPClient {
		t.Fatal("a proxied client must not share the default transport")
	}
	tr, ok := c.http.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport: got %T, want *http.Transport", c.http.Transport)
	}
	req, _ := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", nil)
	u, err := tr.Proxy(req)
	if err != nil {
		t.Fatalf("Transport.Proxy: got error %v, want nil", err)
	}
	if u == nil || u.String() != "http://proxy.internal:3128" {
		t.Errorf("proxy URL: got %v, want http://proxy.internal:3128", u)
	}
}

func TestHTTPClientWinsOverProxy(t *testing.T) {
	clearEnv(t)
	mine := &http.Client{}
	c, err := New(Config{Proxy: "proxy.internal:3128", HTTPClient: mine})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	if c.http != mine {
		t.Errorf("http client: got %p, want the caller's %p", c.http, mine)
	}
}

// TestExtraHeadersAndAuth checks the headers actually reach the wire, and that
// ExtraHeaders is applied last so it can override Authorization.
func TestExtraHeadersAndAuth(t *testing.T) {
	srv, cap := newServer(t, okJSON(minimalReply))
	c := newTestClient(t, srv, func(cfg *Config) {
		cfg.ExtraHeaders = map[string]string{
			"X-Gateway-Route": "team-a",
			"Authorization":   "Bearer override",
		}
	})
	if _, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	h := cap.header(t, 0)
	for k, want := range map[string]string{
		"Content-Type":    "application/json",
		"Accept":          "application/json",
		"X-Gateway-Route": "team-a",
		"Authorization":   "Bearer override",
	} {
		if got := h.Get(k); got != want {
			t.Errorf("header %s: got %q, want %q", k, got, want)
		}
	}
	if got, want := cap.path(t, 0), "/v1/chat/completions"; got != want {
		t.Errorf("path: got %q, want %q", got, want)
	}
}

// TestNoAuthorizationHeaderWithoutAKey: local vLLM and Ollama reject a bogus
// Authorization header, so an unset key must omit it entirely rather than
// sending "Bearer ".
func TestNoAuthorizationHeaderWithoutAKey(t *testing.T) {
	srv, cap := newServer(t, okJSON(minimalReply))
	c := newTestClient(t, srv, func(cfg *Config) { cfg.APIKey = "" })
	if _, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if got := cap.header(t, 0).Values("Authorization"); len(got) != 0 {
		t.Errorf("Authorization: got %v, want the header omitted", got)
	}
}

// TestExtraHeadersCaseCollapse documents a real divergence from
// llm/anthropic: this package does NOT canonicalise ExtraHeaders keys, while
// http.Header.Set does. Two spellings of one name therefore collapse into a
// single header whose VALUE depends on Go map iteration order. The test asserts
// only the deterministic half — exactly one header goes out — because the other
// half genuinely is not deterministic.
func TestExtraHeadersCaseCollapseIsNondeterministic(t *testing.T) {
	srv, cap := newServer(t, okJSON(minimalReply))
	c := newTestClient(t, srv, func(cfg *Config) {
		cfg.ExtraHeaders = map[string]string{"x-route": "a", "X-Route": "b"}
	})
	if _, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	got := cap.header(t, 0).Values("X-Route")
	if len(got) != 1 {
		t.Errorf("X-Route: got %v (%d values), want exactly 1", got, len(got))
	}
	if len(got) == 1 && got[0] != "a" && got[0] != "b" {
		t.Errorf("X-Route: got %q, want one of a/b", got[0])
	}
}

// ===== stream mode and stream_options =====

// TestStreamModeAndStreamOptions is two invariants in one table. The second is
// the load-bearing one: stream_options must NEVER ride on a non-streamed
// request, because at least one gateway answers that with
// "`stream_options` requires `stream` to be true" — a 400 on every call.
func TestStreamModeAndStreamOptions(t *testing.T) {
	tests := []struct {
		name       string
		mode       llm.StreamMode
		sink       bool
		wantStream bool
	}{
		{"auto with no sink buffers", llm.StreamAuto, false, false},
		{"auto with a sink streams", llm.StreamAuto, true, true},
		{"always streams with no sink", llm.StreamAlways, false, true},
		{"never buffers even with a sink", llm.StreamNever, true, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reply := okJSON(minimalReply)
			if tt.wantStream {
				reply = sseReply(
					`{"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
					`{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
					`{"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}`,
					doneSentinel,
				)
			}
			srv, cap := newServer(t, reply)
			c := newTestClient(t, srv, func(cfg *Config) { cfg.Stream = tt.mode })

			req := llm.Request{Messages: []llm.Message{llm.UserText("hi")}}
			var deltas []string
			if tt.sink {
				req.OnDelta = func(d llm.Delta) {
					if d.Text != "" {
						deltas = append(deltas, d.Text)
					}
				}
			}
			resp, err := c.Complete(context.Background(), req)
			if err != nil {
				t.Fatalf("Complete: got error %v, want nil", err)
			}
			if got, want := llm.TextOf(resp.Content), "hi"; got != want {
				t.Errorf("text: got %q, want %q", got, want)
			}

			raw := cap.body(t, 0)
			var body map[string]any
			decodeBody(t, raw, &body)

			if got, _ := body["stream"].(bool); got != tt.wantStream {
				t.Errorf(`request "stream": got %v, want %v`, body["stream"], tt.wantStream)
			}
			_, hasOpts := body["stream_options"]
			if hasOpts != tt.wantStream {
				t.Errorf("stream_options present: got %v, want %v — sending it on a non-streamed request is a 400\nbody: %s",
					hasOpts, tt.wantStream, raw)
			}
			if tt.wantStream {
				opts, _ := body["stream_options"].(map[string]any)
				if inc, _ := opts["include_usage"].(bool); !inc {
					t.Errorf("stream_options: got %v, want include_usage:true (otherwise the cost budget sees zero tokens)", opts)
				}
			}
			wantAccept := "application/json"
			if tt.wantStream {
				wantAccept = "text/event-stream"
			}
			if got := cap.header(t, 0).Get("Accept"); got != wantAccept {
				t.Errorf("Accept: got %q, want %q", got, wantAccept)
			}
			if tt.sink && !tt.wantStream && len(deltas) != 0 {
				t.Errorf("deltas with StreamNever: got %v, want none", deltas)
			}
		})
	}
}

// ===== status classification =====

func TestClassifyStatusMapping(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		body       string
		retryAfter string
		wantClass  error
		wantAfter  time.Duration
	}{
		{name: "400", status: 400, body: `{"error":{"message":"bad"}}`, wantClass: llm.ErrBadRequest},
		{
			name:      "400 naming the context window",
			status:    400,
			body:      `{"error":{"message":"This model's maximum context length is 8192 tokens; too many tokens supplied"}}`,
			wantClass: llm.ErrContextWindow,
		},
		{name: "401", status: 401, body: "unauthorized", wantClass: llm.ErrAuth},
		{name: "403", status: 403, body: "forbidden", wantClass: llm.ErrAuth},
		{name: "404", status: 404, body: "no such model", wantClass: llm.ErrNotFound},
		{name: "422", status: 422, body: "unprocessable", wantClass: llm.ErrBadRequest},
		{
			name: "429 with a hint", status: 429, body: "slow down",
			retryAfter: "20", wantClass: llm.ErrRateLimit, wantAfter: 20 * time.Second,
		},
		{name: "500", status: 500, body: "boom", wantClass: llm.ErrServer},
		{name: "502", status: 502, body: "<html>bad gateway</html>", wantClass: llm.ErrServer},
		// 529 is Anthropic's spelling, but a gateway in front of either
		// provider can emit it and llm.ClassifyStatus is shared.
		{name: "529", status: 529, body: "overloaded", wantClass: llm.ErrOverloaded},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := newServer(t, func(w http.ResponseWriter) {
				if tt.retryAfter != "" {
					w.Header().Set("Retry-After", tt.retryAfter)
				}
				w.WriteHeader(tt.status)
				w.Write([]byte(tt.body))
			})
			c := newTestClient(t, srv, nil)
			_, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.UserText("hi")}})
			if err == nil {
				t.Fatalf("Complete: got nil error, want %v", tt.wantClass)
			}
			if !errors.Is(err, tt.wantClass) {
				t.Errorf("errors.Is(err, %v): got false, want true (err=%v)", tt.wantClass, err)
			}
			var ae *llm.APIError
			if !errors.As(err, &ae) {
				t.Fatalf("errors.As(*llm.APIError): got false, want true (err=%T)", err)
			}
			if ae.Status != tt.status {
				t.Errorf("Status: got %d, want %d", ae.Status, tt.status)
			}
			if ae.RetryAfter != tt.wantAfter {
				t.Errorf("RetryAfter: got %v, want %v", ae.RetryAfter, tt.wantAfter)
			}
			if ae.Message != tt.body {
				t.Errorf("Message: got %q, want the body %q", ae.Message, tt.body)
			}
		})
	}
}

func TestRetryAfterParsing(t *testing.T) {
	tests := []struct {
		name string
		hdr  string
		want time.Duration
	}{
		{name: "absent", hdr: "", want: 0},
		{name: "integer seconds", hdr: "30", want: 30 * time.Second},
		{name: "zero", hdr: "0", want: 0},
		{name: "negative is ignored", hdr: "-1", want: 0},
		{name: "garbage is ignored", hdr: "later", want: 0},
		{name: "past HTTP date is ignored", hdr: "Mon, 02 Jan 2006 15:04:05 GMT", want: 0},
		// Unlike llm/anthropic, this parser uses Atoi, so a fractional value
		// falls through to the date parser and yields 0. RFC 9110 says
		// delay-seconds is an integer, so that is defensible — pinned so the
		// divergence between the two clients is deliberate, not accidental.
		{name: "fractional seconds are not understood", hdr: "1.5", want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := http.Header{}
			if tt.hdr != "" {
				h.Set("Retry-After", tt.hdr)
			}
			if got := retryAfter(h); got != tt.want {
				t.Errorf("retryAfter(%q): got %v, want %v", tt.hdr, got, tt.want)
			}
		})
	}
	t.Run("future HTTP date", func(t *testing.T) {
		h := http.Header{}
		h.Set("Retry-After", time.Now().Add(90*time.Second).UTC().Format(http.TimeFormat))
		got := retryAfter(h)
		if got < 80*time.Second || got > 91*time.Second {
			t.Errorf("retryAfter(future date): got %v, want roughly 90s", got)
		}
	})
}

// TestCancelledContextIsNotClassifiedAsTransport: a cancelled call is the
// caller's decision, and dressing it up as a transport blip would have the
// retry middleware hammer a run the operator just stopped.
func TestCancelledContextIsNotClassifiedAsTransport(t *testing.T) {
	srv, _ := newServer(t, okJSON(minimalReply))
	c := newTestClient(t, srv, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := c.Complete(ctx, llm.Request{Messages: []llm.Message{llm.UserText("hi")}})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("errors.Is(err, context.Canceled): got false, want true (err=%v)", err)
	}
	if llm.Retryable(err) {
		t.Errorf("llm.Retryable: got true, want false (err=%v)", err)
	}
}

func TestRequestOverridesModelAndMaxTokens(t *testing.T) {
	srv, cap := newServer(t, okJSON(minimalReply))
	c := newTestClient(t, srv, func(cfg *Config) { cfg.MaxTokens = 100 })
	_, err := c.Complete(context.Background(), llm.Request{
		Messages:  []llm.Message{llm.UserText("hi")},
		Model:     "override",
		MaxTokens: 512,
	})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	var body struct {
		Model     string `json:"model"`
		MaxTokens int    `json:"max_tokens"`
	}
	decodeBody(t, cap.body(t, 0), &body)
	if body.Model != "override" {
		t.Errorf("model: got %q, want %q", body.Model, "override")
	}
	if body.MaxTokens != 512 {
		t.Errorf("max_tokens: got %d, want %d", body.MaxTokens, 512)
	}
}

func TestCompleteWithNoMessagesNeverHitsTheWire(t *testing.T) {
	srv, cap := newServer(t, okJSON(minimalReply))
	c := newTestClient(t, srv, nil)
	_, err := c.Complete(context.Background(), llm.Request{})
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Errorf("errors.Is(err, llm.ErrBadRequest): got false, want true (err=%v)", err)
	}
	if cap.count() != 0 {
		t.Errorf("requests made: got %d, want 0", cap.count())
	}
}

// ===== env parsing =====

func TestConfigFromEnv(t *testing.T) {
	clearEnv(t)
	t.Setenv("OPENAI_API_KEY", "key")
	t.Setenv("OPENAI_BASE_URL", "https://gw.internal/v1")
	t.Setenv("OPENAI_PROXY", "proxy:3128")
	t.Setenv("WOMBAT_MODEL", "older-spelling")
	t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Route: team-a")
	t.Setenv("OPENAI_STREAM", "always")

	got := ConfigFromEnv()
	want := Config{
		APIKey:       "key",
		BaseURL:      "https://gw.internal/v1",
		Proxy:        "proxy:3128",
		Model:        "older-spelling",
		ExtraHeaders: map[string]string{"X-Route": "team-a"},
		Stream:       llm.StreamAlways,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ConfigFromEnv:\n got %+v\nwant %+v", got, want)
	}

	// The OPENAI_-prefixed names win, so a process running two providers can
	// override per provider.
	t.Setenv("OPENAI_MODEL", "newer-spelling")
	t.Setenv("OPENAI_EXTRA_HEADERS", "X-Route: team-b")
	got = ConfigFromEnv()
	if got.Model != "newer-spelling" {
		t.Errorf("Model: got %q, want %q", got.Model, "newer-spelling")
	}
	if w := map[string]string{"X-Route": "team-b"}; !reflect.DeepEqual(got.ExtraHeaders, w) {
		t.Errorf("ExtraHeaders: got %v, want %v", got.ExtraHeaders, w)
	}
}

func TestParseHeaders(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want map[string]string
	}{
		{name: "empty", in: "", want: nil},
		{name: "blank lines only", in: "\n\n  \n", want: nil},
		{name: "no colon is skipped", in: "garbage", want: nil},
		{name: "empty name is skipped", in: ": value", want: nil},
		{name: "one pair", in: "X-Route: team-a", want: map[string]string{"X-Route": "team-a"}},
		{
			name: "several, trimmed, forgiving of a bad line",
			in:   " A : 1 \noops\nB:2\n",
			want: map[string]string{"A": "1", "B": "2"},
		},
		{name: "value may contain a colon", in: "X: a:b", want: map[string]string{"X": "a:b"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ParseHeaders(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseHeaders(%q): got %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestParseStreamMode(t *testing.T) {
	tests := []struct {
		in   string
		want llm.StreamMode
	}{
		{"", llm.StreamAuto},
		{"auto", llm.StreamAuto},
		{"always", llm.StreamAlways},
		{" ALWAYS ", llm.StreamAlways},
		{"never", llm.StreamNever},
		{"NeVeR", llm.StreamNever},
		{"maybe", llm.StreamAuto},
	}
	for _, tt := range tests {
		if got := parseStreamMode(tt.in); got != tt.want {
			t.Errorf("parseStreamMode(%q): got %v, want %v", tt.in, got, tt.want)
		}
	}
}

func TestTruncateBoundsAQuotedBody(t *testing.T) {
	if got := truncate("short"); got != "short" {
		t.Errorf("truncate(short): got %q, want %q", got, "short")
	}
	long := strings.Repeat("x", (2<<10)+50)
	got := truncate(long)
	if !strings.HasSuffix(got, "…") {
		t.Errorf("truncate(long): got a %d-byte string with no ellipsis", len(got))
	}
	if len(got) > (2<<10)+len("…") {
		t.Errorf("truncate(long): got %d bytes, want at most %d", len(got), (2<<10)+len("…"))
	}
}

func TestParseProxyUnparseable(t *testing.T) {
	_, err := parseProxy("http://host with spaces:3128")
	if err == nil {
		t.Fatal("parseProxy: got nil error, want one")
	}
	if !strings.Contains(err.Error(), "bad proxy") {
		t.Errorf("error: got %q, want it to contain %q", err, "bad proxy")
	}
}

func TestFirstEnv(t *testing.T) {
	t.Setenv("WOMBAT_TEST_A", "")
	t.Setenv("WOMBAT_TEST_B", "second")
	if got, want := firstEnv("WOMBAT_TEST_A", "WOMBAT_TEST_B"), "second"; got != want {
		t.Errorf("firstEnv: got %q, want %q — an exported-but-blank variable means unset", got, want)
	}
	if got := firstEnv("WOMBAT_TEST_A", "WOMBAT_TEST_MISSING"); got != "" {
		t.Errorf("firstEnv with nothing set: got %q, want empty", got)
	}
}

// TestCompleteTransportFailure: a dead endpoint must be classified as
// transport so the retry middleware can act on it.
func TestCompleteTransportFailure(t *testing.T) {
	srv, _ := newServer(t, okJSON(minimalReply))
	c := newTestClient(t, srv, nil)
	srv.Close() // nothing is listening any more

	_, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.UserText("hi")}})
	if !errors.Is(err, llm.ErrTransport) {
		t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
	}
	if !llm.Retryable(err) {
		t.Errorf("llm.Retryable: got false, want true (err=%v)", err)
	}
}
