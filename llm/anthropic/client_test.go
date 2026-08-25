package anthropic

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
		wantErr string // substring; "" means success
		wantIs  error
	}{
		{
			name:    "no api key anywhere",
			cfg:     Config{},
			wantErr: "no API key",
			wantIs:  ErrNoAPIKey,
		},
		{
			name:    "base URL that will not parse",
			cfg:     Config{APIKey: "k", BaseURL: "://nope"},
			wantErr: "bad base URL",
		},
		{
			name:    "base URL with no scheme",
			cfg:     Config{APIKey: "k", BaseURL: "api.anthropic.com"},
			wantErr: "needs a scheme and host",
		},
		{
			name:    "base URL with no host",
			cfg:     Config{APIKey: "k", BaseURL: "https://"},
			wantErr: "needs a scheme and host",
		},
		{
			name:    "negative max tokens",
			cfg:     Config{APIKey: "k", MaxTokens: -1},
			wantErr: "negative MaxTokens",
		},
		{
			name:    "proxy with an unsupported scheme",
			cfg:     Config{APIKey: "k", Proxy: "ftp://proxy:3128"},
			wantErr: "unsupported scheme",
		},
		{
			name:    "proxy with no host",
			cfg:     Config{APIKey: "k", Proxy: "http://"},
			wantErr: "needs a host",
		},
		{
			// A caller-supplied HTTPClient makes Proxy moot, but a typo must
			// still be reported where it was written.
			name:    "bad proxy is rejected even when HTTPClient wins",
			cfg:     Config{APIKey: "k", Proxy: "ftp://proxy:3128", HTTPClient: &http.Client{}},
			wantErr: "unsupported scheme",
		},
		{
			name: "zero MaxTokens is fine",
			cfg:  Config{APIKey: "k"},
		},
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
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v): got false, want true (err=%v)", tt.wantIs, err)
			}
		})
	}
}

func TestNewDefaults(t *testing.T) {
	clearEnv(t)

	c, err := New(Config{APIKey: "k"})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	if got, want := c.endpoint, DefaultBaseURL+"/v1/messages"; got != want {
		t.Errorf("endpoint: got %q, want %q", got, want)
	}
	if got, want := c.maxTokens, DefaultMaxTokens; got != want {
		t.Errorf("maxTokens: got %d, want %d", got, want)
	}
	if got, want := c.version, DefaultVersion; got != want {
		t.Errorf("version: got %q, want %q", got, want)
	}
}

func TestNewAPIKeyFromEnv(t *testing.T) {
	clearEnv(t)
	t.Setenv(envAPIKey, "from-env")

	c, err := New(Config{})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	if got, want := c.apiKey, "from-env"; got != want {
		t.Errorf("apiKey: got %q, want %q", got, want)
	}
}

// TestNewBaseURLPath pins the endpoint construction. Unlike llm/openai, this
// client never rewrites the path: it appends /v1/messages to whatever it was
// given, because that is what the Anthropic base URL convention is (the
// documented base has no /v1). A gateway published WITH a /v1 suffix therefore
// double-prefixes — pinned here so the asymmetry with openai is visible rather
// than discovered as a 404.
func TestNewBaseURLPath(t *testing.T) {
	clearEnv(t)

	tests := []struct {
		base string
		want string
	}{
		{"https://api.anthropic.com", "https://api.anthropic.com/v1/messages"},
		{"https://api.anthropic.com/", "https://api.anthropic.com/v1/messages"},
		{"https://api.anthropic.com///", "https://api.anthropic.com/v1/messages"},
		{"https://gw.internal/anthropic", "https://gw.internal/anthropic/v1/messages"},
		{"http://localhost:8080", "http://localhost:8080/v1/messages"},
		// The foot-gun, documented rather than fixed.
		{"https://gw.internal/v1", "https://gw.internal/v1/v1/messages"},
	}
	for _, tt := range tests {
		t.Run(tt.base, func(t *testing.T) {
			c, err := New(Config{APIKey: "k", BaseURL: tt.base})
			if err != nil {
				t.Fatalf("New(%q): got error %v, want nil", tt.base, err)
			}
			if c.endpoint != tt.want {
				t.Errorf("endpoint for base %q: got %q, want %q", tt.base, c.endpoint, tt.want)
			}
		})
	}
}

func TestNewBaseURLFromEnv(t *testing.T) {
	clearEnv(t)
	t.Setenv(envBaseURL, "https://gw.internal")
	// New reads only the primary name; the AGENT_LLM_* fallback is
	// ConfigFromEnv's job.
	t.Setenv(envBaseURLAlt, "https://wrong.example")

	c, err := New(Config{APIKey: "k"})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	if got, want := c.endpoint, "https://gw.internal/v1/messages"; got != want {
		t.Errorf("endpoint: got %q, want %q", got, want)
	}
}

// TestProxyNormalisation is the one that matters: url.Parse reads
// "localhost:3128" as scheme "localhost", which http.Transport ignores, so an
// un-normalised value sends corporate traffic straight at the internet.
func TestProxyNormalisation(t *testing.T) {
	clearEnv(t)

	tests := []struct {
		in      string
		want    string
		wantErr string
	}{
		{in: "localhost:3128", want: "http://localhost:3128"},
		{in: "  proxy.internal:3128  ", want: "http://proxy.internal:3128"},
		{in: "http://proxy.internal:3128", want: "http://proxy.internal:3128"},
		{in: "https://proxy.internal:3128", want: "https://proxy.internal:3128"},
		{in: "socks5://proxy.internal:1080", want: "socks5://proxy.internal:1080"},
		{in: "socks5h://proxy.internal:1080", want: "socks5h://proxy.internal:1080"},
		{in: "user:pw@proxy.internal:3128", want: "http://user:pw@proxy.internal:3128"},
		{in: "ftp://proxy:21", wantErr: "unsupported scheme"},
		{in: "http://", wantErr: "needs a host"},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			u, err := normalizeProxy(tt.in)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("normalizeProxy(%q): got nil error, want one containing %q", tt.in, tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("normalizeProxy(%q) error: got %q, want it to contain %q", tt.in, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("normalizeProxy(%q): got error %v, want nil", tt.in, err)
			}
			if u.String() != tt.want {
				t.Errorf("normalizeProxy(%q): got %q, want %q", tt.in, u, tt.want)
			}
		})
	}
}

// TestProxyReachesTheTransport closes the loop: a normalised proxy must
// actually be installed, not merely parsed.
func TestProxyReachesTheTransport(t *testing.T) {
	clearEnv(t)

	c, err := New(Config{APIKey: "k", Proxy: "proxy.internal:3128"})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	tr, ok := c.hc.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport type: got %T, want *http.Transport", c.hc.Transport)
	}
	req, _ := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages", nil)
	u, err := tr.Proxy(req)
	if err != nil {
		t.Fatalf("Transport.Proxy: got error %v, want nil", err)
	}
	if u == nil {
		t.Fatal("Transport.Proxy: got nil URL, want http://proxy.internal:3128 (traffic would leave unproxied)")
	}
	if got, want := u.String(), "http://proxy.internal:3128"; got != want {
		t.Errorf("proxy URL: got %q, want %q", got, want)
	}
}

func TestHTTPClientWinsOverProxy(t *testing.T) {
	clearEnv(t)

	mine := &http.Client{}
	c, err := New(Config{APIKey: "k", Proxy: "proxy.internal:3128", HTTPClient: mine})
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	if c.hc != mine {
		t.Errorf("http client: got %p, want the caller's %p", c.hc, mine)
	}
}

// ===== headers on the wire =====

func TestHeadersOnTheWire(t *testing.T) {
	clearEnv(t)
	srv, cap := newServer(t, okJSON(minimalMessage))

	c := newTestClient(t, srv, func(cfg *Config) {
		cfg.Version = "2024-01-01"
		cfg.Beta = []string{"beta-one", BetaContextManagement}
		cfg.ExtraHeaders = map[string]string{
			"x-gateway-route": "team-a",
			"X-Trace-Id":      "abc",
		}
	})
	if _, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}

	h := cap.header(t, 0)
	want := map[string]string{
		"Content-Type":      "application/json",
		"X-Api-Key":         "k",
		"Anthropic-Version": "2024-01-01",
		"Anthropic-Beta":    "beta-one," + BetaContextManagement,
		"Accept":            "application/json",
		"X-Gateway-Route":   "team-a",
		"X-Trace-Id":        "abc",
	}
	for k, v := range want {
		if got := h.Get(k); got != v {
			t.Errorf("header %s: got %q, want %q", k, got, v)
		}
	}
	if got := cap.url(t, 0); got != "/v1/messages" {
		t.Errorf("path: got %q, want %q", got, "/v1/messages")
	}
}

// TestExtraHeadersOverrideAndCollapse pins the case-insensitive collapse.
// http.Header.Set canonicalises, so two spellings of one name in the config
// map would otherwise be applied twice in random map order.
func TestExtraHeadersOverrideAndCollapse(t *testing.T) {
	clearEnv(t)
	srv, cap := newServer(t, okJSON(minimalMessage))

	c := newTestClient(t, srv, func(cfg *Config) {
		cfg.Beta = []string{"from-config"}
		cfg.ExtraHeaders = map[string]string{
			"anthropic-beta":    "from-override",
			"anthropic-version": "9999-99-99",
			"x-api-key":         "override-key",
		}
	})
	if _, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}

	h := cap.header(t, 0)
	if got := h.Values("Anthropic-Beta"); len(got) != 1 || got[0] != "from-override" {
		t.Errorf("Anthropic-Beta: got %v, want exactly [from-override]", got)
	}
	if got := h.Get("Anthropic-Version"); got != "9999-99-99" {
		t.Errorf("Anthropic-Version: got %q, want the override %q", got, "9999-99-99")
	}
	if got := h.Get("X-Api-Key"); got != "override-key" {
		t.Errorf("X-Api-Key: got %q, want the override %q", got, "override-key")
	}
}

// ===== stream mode =====

func TestStreamModeDecidesStreaming(t *testing.T) {
	clearEnv(t)

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
			srv, cap := newServer(t, func(w http.ResponseWriter, n int) {
				if tt.wantStream {
					sseReply(
						ev("message_start", `{"type":"message_start","message":{"model":"m","usage":{"input_tokens":1}}}`),
						ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
						ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`),
						ev("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`),
					)(w, n)
					return
				}
				okJSON(minimalMessage)(w, n)
			})

			c := newTestClient(t, srv, func(cfg *Config) { cfg.Stream = tt.mode })
			req := llm.Request{Messages: []llm.Message{userTurn("hi")}}
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

			var body struct {
				Stream bool `json:"stream"`
			}
			decodeBody(t, cap.body(t, 0), &body)
			if body.Stream != tt.wantStream {
				t.Errorf(`request "stream": got %v, want %v`, body.Stream, tt.wantStream)
			}
			wantAccept := "application/json"
			if tt.wantStream {
				wantAccept = "text/event-stream"
			}
			if got := cap.header(t, 0).Get("Accept"); got != wantAccept {
				t.Errorf("Accept: got %q, want %q", got, wantAccept)
			}
			// StreamNever with a sink means no deltas at all: the documented
			// trade of pacing for accounting.
			if tt.sink && !tt.wantStream && len(deltas) != 0 {
				t.Errorf("deltas with StreamNever: got %v, want none", deltas)
			}
			if tt.sink && tt.wantStream && len(deltas) == 0 {
				t.Error("deltas while streaming: got none, want at least one")
			}
		})
	}
}

// "stream":false is omitted entirely (omitempty), which keeps the body
// byte-identical to one built before streaming existed — and therefore keeps
// the cache prefix stable.
func TestNonStreamingOmitsStreamKey(t *testing.T) {
	clearEnv(t)
	srv, cap := newServer(t, okJSON(minimalMessage))
	c := newTestClient(t, srv, nil)

	if _, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if got := string(cap.body(t, 0)); strings.Contains(got, `"stream"`) {
		t.Errorf(`body carries a "stream" key when not streaming: %s`, got)
	}
}

// ===== status classification =====

func TestClassifyStatusMapping(t *testing.T) {
	clearEnv(t)

	tests := []struct {
		name       string
		status     int
		body       string
		retryAfter string
		wantClass  error
		wantAfter  time.Duration
	}{
		{name: "400 is a bad request", status: 400, body: `{"error":{"message":"bad"}}`, wantClass: llm.ErrBadRequest},
		{
			name:      "400 naming the context window is reclassified",
			status:    400,
			body:      `{"error":{"message":"prompt is too long: 300000 tokens > 200000 maximum"}}`,
			wantClass: llm.ErrContextWindow,
		},
		{name: "401 is auth", status: 401, body: "nope", wantClass: llm.ErrAuth},
		{name: "403 is auth", status: 403, body: "nope", wantClass: llm.ErrAuth},
		{name: "404 is not found", status: 404, body: "no such model", wantClass: llm.ErrNotFound},
		{
			name: "429 is rate limit and carries the hint", status: 429, body: "slow down",
			retryAfter: "30", wantClass: llm.ErrRateLimit, wantAfter: 30 * time.Second,
		},
		{name: "529 is overloaded", status: 529, body: "overloaded", wantClass: llm.ErrOverloaded},
		{
			name: "500 is a server error", status: 500, body: "boom",
			retryAfter: "2", wantClass: llm.ErrServer, wantAfter: 2 * time.Second,
		},
		{name: "503 is a server error", status: 503, body: "unavailable", wantClass: llm.ErrServer},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := newServer(t, func(w http.ResponseWriter, _ int) {
				if tt.retryAfter != "" {
					w.Header().Set("Retry-After", tt.retryAfter)
				}
				w.WriteHeader(tt.status)
				w.Write([]byte(tt.body))
			})
			c := newTestClient(t, srv, nil)

			_, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
			if err == nil {
				t.Fatalf("Complete: got nil error, want %v", tt.wantClass)
			}
			if !errors.Is(err, tt.wantClass) {
				t.Errorf("errors.Is(err, %v): got false, want true (err=%v)", tt.wantClass, err)
			}
			var ae *llm.APIError
			if !errors.As(err, &ae) {
				t.Fatalf("errors.As(*llm.APIError): got false, want true (err=%T %v)", err, err)
			}
			if ae.Status != tt.status {
				t.Errorf("APIError.Status: got %d, want %d", ae.Status, tt.status)
			}
			if ae.RetryAfter != tt.wantAfter {
				t.Errorf("APIError.RetryAfter: got %v, want %v", ae.RetryAfter, tt.wantAfter)
			}
			if ae.Message != tt.body {
				t.Errorf("APIError.Message: got %q, want the provider body %q quoted back verbatim", ae.Message, tt.body)
			}
		})
	}
}

func TestRetryAfterParsing(t *testing.T) {
	tests := []struct {
		name string
		hdr  string
		want time.Duration
		// approx marks a value derived from wall clock, checked as a range.
		approx bool
	}{
		{name: "absent", hdr: "", want: 0},
		{name: "integer seconds", hdr: "12", want: 12 * time.Second},
		{name: "fractional seconds", hdr: "1.5", want: 1500 * time.Millisecond},
		{name: "zero", hdr: "0", want: 0},
		{name: "negative is ignored", hdr: "-5", want: 0},
		{name: "garbage is ignored", hdr: "soon", want: 0},
		{name: "past HTTP date is ignored", hdr: "Mon, 02 Jan 2006 15:04:05 GMT", want: 0},
		{name: "future HTTP date", hdr: "", approx: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := http.Header{}
			if tt.approx {
				h.Set("Retry-After", time.Now().Add(90*time.Second).UTC().Format(http.TimeFormat))
				got := retryAfter(h)
				if got < 80*time.Second || got > 91*time.Second {
					t.Errorf("retryAfter(future date): got %v, want roughly 90s", got)
				}
				return
			}
			if tt.hdr != "" {
				h.Set("Retry-After", tt.hdr)
			}
			if got := retryAfter(h); got != tt.want {
				t.Errorf("retryAfter(%q): got %v, want %v", tt.hdr, got, tt.want)
			}
		})
	}
}

func TestCompleteWithoutModel(t *testing.T) {
	clearEnv(t)
	srv, cap := newServer(t, okJSON(minimalMessage))
	c := newTestClient(t, srv, func(cfg *Config) { cfg.Model = "" })

	_, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}})
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Errorf("errors.Is(err, llm.ErrBadRequest): got false, want true (err=%v)", err)
	}
	if cap.count() != 0 {
		t.Errorf("requests made: got %d, want 0 — the call must fail before hitting the wire", cap.count())
	}
}

// TestRequestModelOverridesClient checks the per-request override reaches the
// body, not just the local variable.
func TestRequestModelAndMaxTokensOverride(t *testing.T) {
	clearEnv(t)
	srv, cap := newServer(t, okJSON(minimalMessage))
	c := newTestClient(t, srv, func(cfg *Config) { cfg.MaxTokens = 100 })

	_, err := c.Complete(context.Background(), llm.Request{
		Messages:  []llm.Message{userTurn("hi")},
		Model:     "override-model",
		MaxTokens: 4242,
	})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	var body struct {
		Model     string `json:"model"`
		MaxTokens int    `json:"max_tokens"`
	}
	decodeBody(t, cap.body(t, 0), &body)
	if body.Model != "override-model" {
		t.Errorf("model: got %q, want %q", body.Model, "override-model")
	}
	if body.MaxTokens != 4242 {
		t.Errorf("max_tokens: got %d, want %d", body.MaxTokens, 4242)
	}
}

// ===== abortError =====

// TestAbortErrorClassification pins the retry-vs-stop distinction: a stall is
// transport (retryable), a deliberate cancellation keeps its cause so
// errors.Is(err, governor.ErrBudgetExhausted) still answers upstream.
func TestAbortErrorClassification(t *testing.T) {
	sentinel := errors.New("budget exhausted")

	t.Run("live context is transport", func(t *testing.T) {
		err := abortError(context.Background(), errors.New("dial tcp: refused"))
		if !errors.Is(err, llm.ErrTransport) {
			t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
		}
	})
	t.Run("idle stream is transport", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(errStreamIdle)
		err := abortError(ctx, errors.New("body closed"))
		if !errors.Is(err, llm.ErrTransport) {
			t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
		}
	})
	t.Run("deadline is transport", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
		defer cancel()
		<-ctx.Done()
		err := abortError(ctx, errors.New("body closed"))
		if !errors.Is(err, llm.ErrTransport) {
			t.Errorf("errors.Is(err, llm.ErrTransport): got false, want true (err=%v)", err)
		}
	})
	t.Run("deliberate cancellation keeps its cause and is NOT transport", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(sentinel)
		err := abortError(ctx, errors.New("body closed"))
		if !errors.Is(err, sentinel) {
			t.Errorf("errors.Is(err, sentinel): got false, want true (err=%v)", err)
		}
		if errors.Is(err, llm.ErrTransport) {
			t.Errorf("errors.Is(err, llm.ErrTransport): got true, want false — a governor abort must not look retryable (err=%v)", err)
		}
	})
}

// ===== ConfigFromEnv =====

func TestConfigFromEnv(t *testing.T) {
	clearEnv(t)
	t.Setenv(envAPIKey, "key")
	t.Setenv(envBaseURLAlt, "https://alt.example")
	t.Setenv(envProxyAlt, "proxy:3128")
	t.Setenv(envModelAlt, "wombat-model")
	t.Setenv(envBeta, " a , b ,, ")
	t.Setenv(envStream, "NEVER")
	t.Setenv(envCustomHeaders, "X-Route: team-a\nnot a header line\n\nx-route: team-b\n")

	got := ConfigFromEnv()
	if got.APIKey != "key" {
		t.Errorf("APIKey: got %q, want %q", got.APIKey, "key")
	}
	if got.BaseURL != "https://alt.example" {
		t.Errorf("BaseURL: got %q, want the AGENT_LLM_BASE_URL fallback %q", got.BaseURL, "https://alt.example")
	}
	if got.Proxy != "proxy:3128" {
		t.Errorf("Proxy: got %q, want %q", got.Proxy, "proxy:3128")
	}
	if got.Model != "wombat-model" {
		t.Errorf("Model: got %q, want %q", got.Model, "wombat-model")
	}
	if !reflect.DeepEqual(got.Beta, []string{"a", "b"}) {
		t.Errorf("Beta: got %v, want [a b]", got.Beta)
	}
	if got.Stream != llm.StreamNever {
		t.Errorf("Stream: got %v, want llm.StreamNever", got.Stream)
	}
	// Two spellings of one header collapse, last line wins.
	want := map[string]string{"X-Route": "team-b"}
	if !reflect.DeepEqual(got.ExtraHeaders, want) {
		t.Errorf("ExtraHeaders: got %v, want %v", got.ExtraHeaders, want)
	}
}

// TestConfigFromEnvFoldsBetaHeader pins the normalisation that stops the two
// beta inputs from disagreeing: a beta line in the custom headers is MOVED
// into Config.Beta, so editing that field afterwards actually does something.
func TestConfigFromEnvFoldsBetaHeader(t *testing.T) {
	clearEnv(t)
	t.Setenv(envAPIKey, "key")
	t.Setenv(envBeta, "ignored-because-the-header-wins")
	t.Setenv(envCustomHeaders, "anthropic-beta: "+BetaContextManagement+", other")

	got := ConfigFromEnv()
	if _, ok := got.ExtraHeaders[betaHeader]; ok {
		t.Errorf("ExtraHeaders still carries %s: %v — the config would then name the beta list twice", betaHeader, got.ExtraHeaders)
	}
	if got.ExtraHeaders != nil {
		t.Errorf("ExtraHeaders: got %v, want nil once the only entry was folded away", got.ExtraHeaders)
	}
	if want := []string{BetaContextManagement, "other"}; !reflect.DeepEqual(got.Beta, want) {
		t.Errorf("Beta: got %v, want %v", got.Beta, want)
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
		{"Never", llm.StreamNever},
		{"yes please", llm.StreamAuto},
	}
	for _, tt := range tests {
		if got := parseStreamMode(tt.in); got != tt.want {
			t.Errorf("parseStreamMode(%q): got %v, want %v", tt.in, got, tt.want)
		}
	}
}

func TestTruncateStopsOnARuneBoundary(t *testing.T) {
	// "中文" is two 3-byte runes; cutting at 4 must fall back to 3.
	got := truncate("中文", 4)
	if !strings.HasPrefix(got, "中") || strings.Contains(got, "\ufffd") {
		t.Errorf("truncate: got %q, want a clean cut after 中", got)
	}
	if got := truncate("short", 100); got != "short" {
		t.Errorf("truncate(short): got %q, want %q", got, "short")
	}
}

func TestScrubReplacesInvalidUTF8(t *testing.T) {
	if got, want := scrub("a\xffb"), "a\ufffdb"; got != want {
		t.Errorf("scrub: got %q, want %q", got, want)
	}
}
