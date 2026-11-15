package openai

import (
	"bytes"
	"context"
	"log/slog"
	"math"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
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

// TestSamplingIsAbsentUnlessAsked is what the pointers buy: a request that says
// nothing about sampling says nothing on the wire, so the provider's own
// default still applies — a default it chose, not one we chose for it.
func TestSamplingIsAbsentUnlessAsked(t *testing.T) {
	raw, err := encodeRequest(llm.Request{Messages: []llm.Message{llm.UserText("hi")}}, "m", 100, false)
	if err != nil {
		t.Fatalf("encodeRequest: got error %v, want nil", err)
	}
	for _, key := range []string{`"temperature"`, `"top_p"`} {
		if strings.Contains(string(raw), key) {
			t.Errorf("body carries %s when nothing asked for it: %s", key, raw)
		}
	}
}

// TestSamplingIsEncodedWhenSet covers the encoder alone, including the zero
// that a float64 with omitempty would have thrown away.
func TestSamplingIsEncodedWhenSet(t *testing.T) {
	tests := []struct {
		name     string
		req      llm.Request
		wantTemp *float64
		wantTop  *float64
	}{
		{name: "temperature only", req: llm.Request{Temperature: f64(1.5)}, wantTemp: f64(1.5)},
		{name: "top_p only", req: llm.Request{TopP: f64(0.9)}, wantTop: f64(0.9)},
		{
			// Unlike llm/anthropic, this API takes both at once.
			name:     "both together, which this API allows",
			req:      llm.Request{Temperature: f64(0.7), TopP: f64(0.95)},
			wantTemp: f64(0.7),
			wantTop:  f64(0.95),
		},
		{
			// The reproducible-run case, and the reason for the pointer.
			name:     "temperature zero is sent, not treated as unset",
			req:      llm.Request{Temperature: f64(0)},
			wantTemp: f64(0),
		},
		{name: "top_p zero is sent too", req: llm.Request{TopP: f64(0)}, wantTop: f64(0)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := tt.req
			req.Messages = []llm.Message{llm.UserText("hi")}
			raw, err := encodeRequest(req, "m", 100, false)
			if err != nil {
				t.Fatalf("encodeRequest: got error %v, want nil", err)
			}
			var got wireSampling
			decodeBody(t, raw, &got)
			checkFloatPtr(t, "temperature", got.Temperature, tt.wantTemp)
			checkFloatPtr(t, "top_p", got.TopP, tt.wantTop)
		})
	}
}

// TestSamplingPrecedence pins how a Request and the Config defaults combine.
// Field by field, because this API accepts the two together: a configured
// top_p keeps applying to a request that overrides only the temperature.
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
			name:     "config default applies when the request is silent",
			cfgTemp:  f64(0.3),
			cfgTop:   f64(0.8),
			wantTemp: f64(0.3),
			wantTop:  f64(0.8),
		},
		{
			name:     "request overrides the config default",
			cfgTemp:  f64(0.3),
			reqTemp:  f64(1.4),
			wantTemp: f64(1.4),
		},
		{
			name:     "request zero overrides a non-zero config default",
			cfgTemp:  f64(0.7),
			reqTemp:  f64(0),
			wantTemp: f64(0),
		},
		{
			name:     "the two fields fall back independently",
			cfgTemp:  f64(0.3),
			cfgTop:   f64(0.8),
			reqTemp:  f64(1),
			wantTemp: f64(1),
			wantTop:  f64(0.8),
		},
		{
			name:    "request top_p over a configured one",
			cfgTop:  f64(0.8),
			reqTop:  f64(0.1),
			wantTop: f64(0.1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, cap := newServer(t, okJSON(minimalReply))
			c := newTestClient(t, srv, func(cfg *Config) {
				cfg.Temperature, cfg.TopP = tt.cfgTemp, tt.cfgTop
			})
			_, err := c.Complete(context.Background(), llm.Request{
				Messages:    []llm.Message{llm.UserText("hi")},
				Temperature: tt.reqTemp,
				TopP:        tt.reqTop,
			})
			if err != nil {
				t.Fatalf("Complete: got error %v, want nil", err)
			}
			var got wireSampling
			decodeBody(t, cap.body(t, 0), &got)
			checkFloatPtr(t, "temperature", got.Temperature, tt.wantTemp)
			checkFloatPtr(t, "top_p", got.TopP, tt.wantTop)
		})
	}
}

// TestCompleteDoesNotMutateTheCallersRequest: merging the defaults happens on
// Complete's own copy, so a Request reused across two differently configured
// clients does not pick up the first one's sampling.
func TestCompleteDoesNotMutateTheCallersRequest(t *testing.T) {
	srv, _ := newServer(t, okJSON(minimalReply))
	c := newTestClient(t, srv, func(cfg *Config) { cfg.Temperature = f64(1.1) })

	req := llm.Request{Messages: []llm.Message{llm.UserText("hi")}}
	if _, err := c.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if req.Temperature != nil {
		t.Errorf("caller's Request.Temperature: got %v, want nil — Complete must not write back its defaults", *req.Temperature)
	}
}

// TestNewValidatesSampling: the provider rejects these anyway, and doing it
// here names the field before the run has spent anything.
func TestNewValidatesSampling(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr string // "" means New must succeed
	}{
		{name: "temperature at the bottom of the range", cfg: Config{Temperature: f64(0)}},
		{name: "temperature at the top of the range", cfg: Config{Temperature: f64(2)}},
		{name: "top_p at the top of the range", cfg: Config{TopP: f64(1)}},
		{name: "both together are fine here", cfg: Config{Temperature: f64(1), TopP: f64(0.9)}},
		{name: "temperature above 2", cfg: Config{Temperature: f64(2.5)}, wantErr: "Temperature 2.5 is outside [0, 2]"},
		{name: "negative temperature", cfg: Config{Temperature: f64(-0.5)}, wantErr: "Temperature -0.5 is outside [0, 2]"},
		{name: "top_p above 1", cfg: Config{TopP: f64(1.5)}, wantErr: "TopP 1.5 is outside [0, 1]"},
		{name: "negative top_p", cfg: Config{TopP: f64(-1)}, wantErr: "TopP -1 is outside [0, 1]"},
		{
			// NaN compares false against every bound, so a naive range test
			// would let it through and put NaN in the body — which is not even
			// valid JSON.
			name:    "NaN temperature",
			cfg:     Config{Temperature: f64(math.NaN())},
			wantErr: "Temperature NaN is outside [0, 2]",
		},
		{name: "NaN top_p", cfg: Config{TopP: f64(math.NaN())}, wantErr: "TopP NaN is outside [0, 1]"},
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
		t.Setenv("OPENAI_TEMPERATURE", " 1.25 ")
		t.Setenv("OPENAI_TOP_P", "0")

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
	// is dropped — but loudly, because otherwise the run samples at the
	// provider's default and looks like it honoured the setting.
	t.Run("an unparseable value is warned about and ignored", func(t *testing.T) {
		clearEnv(t)
		t.Setenv("OPENAI_TEMPERATURE", "0.7,0.9")
		t.Setenv("OPENAI_TOP_P", "0.9")
		buf := captureDefaultLogger(t)

		got := ConfigFromEnv()
		checkFloatPtr(t, "Temperature", got.Temperature, nil)
		checkFloatPtr(t, "TopP", got.TopP, f64(0.9))

		log := buf.String()
		if !strings.Contains(log, "OPENAI_TEMPERATURE") || !strings.Contains(log, "0.7,0.9") {
			t.Errorf("log: got %q, want it to name OPENAI_TEMPERATURE and the value it dropped", log)
		}
	})

	// The seam: unparseable is ignored here, but out of range is a decision,
	// and New is the call that has an error to report it with.
	t.Run("an out-of-range value survives to be rejected by New", func(t *testing.T) {
		clearEnv(t)
		t.Setenv("OPENAI_TOP_P", "9")

		cfg := ConfigFromEnv()
		checkFloatPtr(t, "TopP", cfg.TopP, f64(9))
		if _, err := New(cfg); err == nil || !strings.Contains(err.Error(), "outside [0, 1]") {
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
