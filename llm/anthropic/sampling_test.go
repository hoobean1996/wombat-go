package anthropic

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"math"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
)

// f64 is the pointer literal these tests are made of. The pointer is the whole
// design — 0 is a real temperature, so "unset" cannot be spelled as a zero.
func f64(v float64) *float64 { return &v }

// wireSampling reads the two fields back loosely, so an absent key and an
// explicit 0 stay distinguishable.
type wireSampling struct {
	Temperature *float64 `json:"temperature"`
	TopP        *float64 `json:"top_p"`
}

// samplingClient builds a client with the environment neutralised, for the
// encode-level assertions that never touch a socket.
func samplingClient(t *testing.T, mutate func(*Config)) *Client {
	t.Helper()
	clearEnv(t)
	cfg := Config{APIKey: "k", Model: "test-model"}
	if mutate != nil {
		mutate(&cfg)
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	return c
}

func encodeSampling(t *testing.T, c *Client, req llm.Request) wireSampling {
	t.Helper()
	req.Messages = append([]llm.Message{userTurn("hi")}, req.Messages...)
	raw, err := c.encode(req, "m", 100, false)
	if err != nil {
		t.Fatalf("encode: got error %v, want nil", err)
	}
	var got wireSampling
	decodeBody(t, raw, &got)
	return got
}

// TestSamplingIsAbsentUnlessAsked is the point of the pointers: a request that
// says nothing about sampling must say nothing on the wire, because a default
// the provider chose is not the same as a default we chose.
func TestSamplingIsAbsentUnlessAsked(t *testing.T) {
	c := samplingClient(t, nil)
	raw, err := c.encode(llm.Request{Messages: []llm.Message{userTurn("hi")}}, "m", 100, false)
	if err != nil {
		t.Fatalf("encode: got error %v, want nil", err)
	}
	for _, key := range []string{`"temperature"`, `"top_p"`} {
		if strings.Contains(string(raw), key) {
			t.Errorf("body carries %s when nothing asked for it: %s", key, raw)
		}
	}
}

// TestSamplingPrecedence pins how a Request and the Config defaults combine.
//
// The pair rule is the interesting half: the Messages API refuses a body with
// both controls, so a request that names either one replaces both defaults
// rather than half-merging into a body that is a guaranteed 400.
func TestSamplingPrecedence(t *testing.T) {
	tests := []struct {
		name            string
		cfgTemp, cfgTop *float64
		reqTemp, reqTop *float64
		wantTemp        *float64
		wantTop         *float64
	}{
		{
			name: "nothing anywhere sends nothing",
		},
		{
			name:     "request temperature",
			reqTemp:  f64(1.5),
			wantTemp: f64(1.5),
		},
		{
			name:    "request top_p",
			reqTop:  f64(0.9),
			wantTop: f64(0.9),
		},
		{
			// The reproducible-run case. A float64 with omitempty would drop
			// this and silently sample at the provider's default instead.
			name:     "temperature zero is sent, not treated as unset",
			reqTemp:  f64(0),
			wantTemp: f64(0),
		},
		{
			name:     "config default applies when the request is silent",
			cfgTemp:  f64(0.3),
			wantTemp: f64(0.3),
		},
		{
			name:    "config top_p default applies when the request is silent",
			cfgTop:  f64(0.5),
			wantTop: f64(0.5),
		},
		{
			name:     "request overrides the config default",
			cfgTemp:  f64(0.3),
			reqTemp:  f64(1),
			wantTemp: f64(1),
		},
		{
			name:     "request zero overrides a non-zero config default",
			cfgTemp:  f64(0.7),
			reqTemp:  f64(0),
			wantTemp: f64(0),
		},
		{
			// Field-by-field fallback would send BOTH here and 400 every call,
			// with no way to unset the default from the request side. This is
			// the combination $ANTHROPIC_TOP_P plus an rl rollout produces.
			name:     "a request temperature displaces a configured top_p",
			cfgTop:   f64(0.9),
			reqTemp:  f64(1),
			wantTemp: f64(1),
		},
		{
			name:    "a request top_p displaces a configured temperature",
			cfgTemp: f64(0.3),
			reqTop:  f64(0.8),
			wantTop: f64(0.8),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := samplingClient(t, func(cfg *Config) {
				cfg.Temperature, cfg.TopP = tt.cfgTemp, tt.cfgTop
			})
			got := encodeSampling(t, c, llm.Request{Temperature: tt.reqTemp, TopP: tt.reqTop})
			checkFloatPtr(t, "temperature", got.Temperature, tt.wantTemp)
			checkFloatPtr(t, "top_p", got.TopP, tt.wantTop)
		})
	}
}

func checkFloatPtr(t *testing.T, name string, got, want *float64) {
	t.Helper()
	switch {
	case got == nil && want == nil:
	case got == nil:
		t.Errorf("%s: got the key omitted, want %v", name, *want)
	case want == nil:
		t.Errorf("%s: got %v, want the key omitted", name, *got)
	case *got != *want:
		t.Errorf("%s: got %v, want %v", name, *got, *want)
	}
}

// TestRequestWithBothSamplingControlsIsRejected: the API answers a body
// carrying both with a 400, so this client refuses it first, names both values,
// and never spends the round trip.
func TestRequestWithBothSamplingControlsIsRejected(t *testing.T) {
	clearEnv(t)
	srv, cap := newServer(t, okJSON(minimalMessage))
	c := newTestClient(t, srv, nil)

	_, err := c.Complete(context.Background(), llm.Request{
		Messages:    []llm.Message{userTurn("hi")},
		Temperature: f64(1),
		TopP:        f64(0.9),
	})
	if err == nil {
		t.Fatal("Complete: got nil error, want a rejection of temperature+top_p")
	}
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Errorf("errors.Is(err, llm.ErrBadRequest): got false, want true (err=%v)", err)
	}
	if !strings.Contains(err.Error(), "not both") {
		t.Errorf("error: got %q, want it to say the two are mutually exclusive", err)
	}
	if cap.count() != 0 {
		t.Errorf("requests made: got %d, want 0 — a guaranteed 400 must not reach the wire", cap.count())
	}
}

// TestSamplingReachesTheWire checks the whole path, not just the encoder: the
// per-request value and the client default both have to survive Complete.
func TestSamplingReachesTheWire(t *testing.T) {
	clearEnv(t)
	srv, cap := newServer(t, okJSON(minimalMessage))
	c := newTestClient(t, srv, func(cfg *Config) { cfg.TopP = f64(0.25) })

	if _, err := c.Complete(context.Background(), llm.Request{Messages: []llm.Message{userTurn("hi")}}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	// A fresh struct per body: json.Unmarshal leaves a pointer field untouched
	// when the key is absent, so a reused one would report the previous value.
	var first wireSampling
	decodeBody(t, cap.body(t, 0), &first)
	checkFloatPtr(t, "top_p", first.TopP, f64(0.25))
	checkFloatPtr(t, "temperature", first.Temperature, nil)

	if _, err := c.Complete(context.Background(), llm.Request{
		Messages:    []llm.Message{userTurn("hi")},
		Temperature: f64(1.2),
	}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	// The configured top_p is displaced rather than merged: sending both is a
	// 400.
	var second wireSampling
	decodeBody(t, cap.body(t, 1), &second)
	checkFloatPtr(t, "temperature", second.Temperature, f64(1.2))
	checkFloatPtr(t, "top_p", second.TopP, nil)
}

// TestNewValidatesSampling: the provider would reject these anyway, and doing
// it here names the field before the run has spent anything.
func TestNewValidatesSampling(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr string // "" means New must succeed
	}{
		{name: "temperature at the bottom of the range", cfg: Config{Temperature: f64(0)}},
		{name: "temperature at the top of the range", cfg: Config{Temperature: f64(2)}},
		{name: "top_p at the top of the range", cfg: Config{TopP: f64(1)}},
		{name: "temperature above 2", cfg: Config{Temperature: f64(2.1)}, wantErr: "Temperature 2.1 is outside [0, 2]"},
		{name: "negative temperature", cfg: Config{Temperature: f64(-0.5)}, wantErr: "Temperature -0.5 is outside [0, 2]"},
		{name: "top_p above 1", cfg: Config{TopP: f64(1.1)}, wantErr: "TopP 1.1 is outside [0, 1]"},
		{name: "negative top_p", cfg: Config{TopP: f64(-1)}, wantErr: "TopP -1 is outside [0, 1]"},
		{
			// NaN compares false against every bound, so a naive range test
			// would let it through and put "NaN" in the JSON — which is not
			// even valid JSON.
			name:    "NaN temperature",
			cfg:     Config{Temperature: f64(math.NaN())},
			wantErr: "Temperature NaN is outside [0, 2]",
		},
		{name: "NaN top_p", cfg: Config{TopP: f64(math.NaN())}, wantErr: "TopP NaN is outside [0, 1]"},
		{
			// A Config naming both has no defensible reading: every call it
			// makes would be a 400.
			name:    "both at once",
			cfg:     Config{Temperature: f64(1), TopP: f64(0.9)},
			wantErr: "accepts one or the other",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearEnv(t)
			cfg := tt.cfg
			cfg.APIKey = "k"
			_, err := New(cfg)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("New: got error %v, want nil", err)
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

func TestConfigFromEnvSampling(t *testing.T) {
	t.Run("parses both", func(t *testing.T) {
		clearEnv(t)
		t.Setenv(envTemperature, " 1.25 ")
		t.Setenv(envTopP, "0")

		got := ConfigFromEnv()
		checkFloatPtr(t, "Temperature", got.Temperature, f64(1.25))
		checkFloatPtr(t, "TopP", got.TopP, f64(0))
	})

	t.Run("unset leaves nil", func(t *testing.T) {
		clearEnv(t)
		got := ConfigFromEnv()
		checkFloatPtr(t, "Temperature", got.Temperature, nil)
		checkFloatPtr(t, "TopP", got.TopP, nil)
	})

	// A bad env var must not stop a run that would otherwise work, so the value
	// is dropped — but loudly, because the alternative is a run that silently
	// samples at the provider's default and looks like it took the setting.
	t.Run("an unparseable value is warned about and ignored", func(t *testing.T) {
		clearEnv(t)
		t.Setenv(envTemperature, "hot")
		t.Setenv(envTopP, "0.9")
		buf := captureDefaultLogger(t)

		got := ConfigFromEnv()
		checkFloatPtr(t, "Temperature", got.Temperature, nil)
		checkFloatPtr(t, "TopP", got.TopP, f64(0.9))

		log := buf.String()
		if !strings.Contains(log, envTemperature) || !strings.Contains(log, "hot") {
			t.Errorf("log: got %q, want it to name %s and the value it dropped", log, envTemperature)
		}
	})

	// The seam: unparseable is ignored here, but out of range is a decision and
	// is reported by New, which has an error to report it with.
	t.Run("an out-of-range value survives to be rejected by New", func(t *testing.T) {
		clearEnv(t)
		t.Setenv(envTemperature, "9")

		cfg := ConfigFromEnv()
		checkFloatPtr(t, "Temperature", cfg.Temperature, f64(9))
		cfg.APIKey = "k"
		if _, err := New(cfg); err == nil || !strings.Contains(err.Error(), "outside [0, 2]") {
			t.Errorf("New: got error %v, want one naming the range", err)
		}
	})
}

// captureDefaultLogger redirects slog.Default for the duration of the test.
func captureDefaultLogger(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &buf
}
