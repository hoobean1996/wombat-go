package llm

import (
	"math"
	"testing"
)

func closeEnough(got, want float64) bool {
	return math.Abs(got-want) < 1e-9
}

func TestTableCostUSD(t *testing.T) {
	table := Table{
		"claude-haiku-4-5": {In: 1, Out: 5, CacheWrite: 1.25, CacheRead: 0.10},
	}
	u := Usage{InputTokens: 1_000_000, OutputTokens: 2_000_000, CacheWriteTokens: 4_000_000, CacheReadTokens: 10_000_000}

	// 1*1 + 2*5 + 4*1.25 + 10*0.10 = 1 + 10 + 5 + 1 = 17
	if got := table.CostUSD("claude-haiku-4-5", u); !closeEnough(got, 17) {
		t.Errorf("CostUSD: got %v, want 17", got)
	}

	t.Run("scales linearly below a million", func(t *testing.T) {
		got := table.CostUSD("claude-haiku-4-5", Usage{InputTokens: 1000, OutputTokens: 500})
		want := 1000*1.0/1e6 + 500*5.0/1e6 // 0.001 + 0.0025
		if !closeEnough(got, want) {
			t.Errorf("CostUSD: got %v, want %v", got, want)
		}
	})

	t.Run("zero usage costs zero", func(t *testing.T) {
		if got := table.CostUSD("claude-haiku-4-5", Usage{}); got != 0 {
			t.Errorf("CostUSD: got %v, want 0", got)
		}
	})
}

func TestTableLongestPrefixMatch(t *testing.T) {
	// Overlapping prefixes: the longest match must win, otherwise a dated Opus
	// id could be priced with the generic "claude-" rate.
	table := Table{
		"claude":            {In: 1},
		"claude-sonnet":     {In: 2},
		"claude-sonnet-4-5": {In: 3},
		"gpt-4":             {In: 4},
	}
	tests := []struct {
		model string
		wantI float64
	}{
		{"claude-sonnet-4-5", 3},
		{"claude-sonnet-4-5-20250929", 3}, // a dated release is covered by its family
		{"claude-sonnet-4-20250514", 2},
		{"claude-sonnet", 2},
		{"claude-opus-4-1", 1},
		{"claude", 1},
		{"gpt-4o-mini", 4},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			// 1M input tokens makes the cost equal to the In rate.
			got := table.CostUSD(tt.model, Usage{InputTokens: 1_000_000})
			if !closeEnough(got, tt.wantI) {
				t.Errorf("CostUSD(%q): got %v, want %v (the longest matching prefix)", tt.model, got, tt.wantI)
			}
		})
	}
}

func TestTableExactMatchWinsOverPrefix(t *testing.T) {
	// An exact key must beat a prefix even when the prefix is longer than
	// nothing — the map lookup short-circuits before the scan.
	table := Table{
		"claude-sonnet-4-5":          {In: 3},
		"claude-sonnet-4-5-20250929": {In: 99},
	}
	if got := table.CostUSD("claude-sonnet-4-5-20250929", Usage{InputTokens: 1_000_000}); !closeEnough(got, 99) {
		t.Errorf("CostUSD: got %v, want 99 (the exact key)", got)
	}
}

func TestTableUnpricedModelIsZero(t *testing.T) {
	// A stale table must not kill a run. An unpriced model costs 0 and the zero
	// shows up plainly in the Spend event.
	table := Table{"claude-opus-5": {In: 15, Out: 75}}
	tests := []string{
		"llama-3.3-70b",            // self-hosted, nobody is billing per token
		"gpt-5",                    // just not in the table
		"claude",                   // a PREFIX of a key is not a match
		"claude-opus",              // ditto
		"",                         // no model at all
		"CLAUDE-OPUS-5",            // matching is case sensitive
		" claude-opus-5",           // a leading space is not the same model
		"my-gateway/claude-opus-5", // a gateway-qualified id does not prefix-match
	}
	for _, model := range tests {
		t.Run(model, func(t *testing.T) {
			got := table.CostUSD(model, Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
			if got != 0 {
				t.Errorf("CostUSD(%q): got %v, want 0", model, got)
			}
		})
	}
}

func TestEmptyAndFreeTables(t *testing.T) {
	u := Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000}
	if got := (Table{}).CostUSD("claude-opus-5", u); got != 0 {
		t.Errorf("empty Table: got %v, want 0", got)
	}
	if got := FreePricing.CostUSD("claude-opus-5", u); got != 0 {
		t.Errorf("FreePricing: got %v, want 0", got)
	}
	var nilTable Table
	if got := nilTable.CostUSD("claude-opus-5", u); got != 0 {
		t.Errorf("nil Table: got %v, want 0", got)
	}
}

func TestDefaultPricing(t *testing.T) {
	tests := []struct {
		name  string
		model string
		usage Usage
		want  float64
	}{
		{
			name:  "opus 5 input and output",
			model: "claude-opus-5",
			usage: Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000},
			want:  15 + 75,
		},
		{
			name:  "a dated haiku id is priced by its family prefix",
			model: "claude-haiku-4-5-20251001",
			usage: Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000},
			want:  1 + 5,
		},
		{
			name:  "cache write is 1.25x input, cache read 0.1x",
			model: "claude-sonnet-5",
			usage: Usage{CacheWriteTokens: 1_000_000, CacheReadTokens: 1_000_000},
			want:  3.75 + 0.30,
		},
		{
			name:  "claude-opus-4 does not get the opus-5 rate by accident",
			model: "claude-opus-4-1-20250805",
			usage: Usage{InputTokens: 1_000_000},
			want:  15,
		},
		{
			// "claude-sonnet-4" is a prefix of "claude-sonnet-4-5..." and there
			// is no sonnet-4-5 row, so it prices at the sonnet-4 rate rather
			// than falling through to zero.
			name:  "sonnet 4-5 falls back to the sonnet 4 row",
			model: "claude-sonnet-4-5-20250929",
			usage: Usage{InputTokens: 1_000_000},
			want:  3,
		},
		{
			name:  "an unknown model is free",
			model: "some-local-model",
			usage: Usage{InputTokens: 10_000_000},
			want:  0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DefaultPricing.CostUSD(tt.model, tt.usage)
			if !closeEnough(got, tt.want) {
				t.Errorf("CostUSD(%q): got %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestTableImplementsPricing(t *testing.T) {
	var _ Pricing = Table{}
	var _ Pricing = DefaultPricing
	var _ Pricing = FreePricing
}
