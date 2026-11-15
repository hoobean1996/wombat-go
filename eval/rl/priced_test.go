package rl

// The honesty of the cost column, from the bottom up: which models the event
// stream saw, what that plus a pricing says about the spend tally, and what a
// whole rollout ends up recording.

import (
	"errors"
	"reflect"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
)

// priceless is a Pricing that is not a Pricer: it makes no claim either way,
// so llm.Priced takes it at its word.
type priceless struct{}

func (priceless) CostUSD(string, llm.Usage) float64 { return 0 }

func TestFoldRecordsTheModelsThatAnswered(t *testing.T) {
	tests := []struct {
		name   string
		events []core.Event
		want   []string
	}{
		{
			name: "the resolved model, once per distinct id",
			events: []core.Event{
				core.IterStart{N: 1},
				core.LLMDone{Model: "some-gateway-model", Usage: llm.Usage{InputTokens: 10}},
				core.IterStart{N: 2},
				core.LLMDone{Model: "some-gateway-model", Usage: llm.Usage{InputTokens: 10}},
			},
			want: []string{"some-gateway-model"},
		},
		{
			name: "a response that names no model falls back to the request",
			events: []core.Event{
				core.IterStart{N: 1},
				core.LLMStart{Model: "asked-for"},
				core.LLMDone{Usage: llm.Usage{OutputTokens: 3}},
			},
			want: []string{"asked-for"},
		},
		{
			// A child's tokens are the parent's money, so a child on an
			// unpriced model makes the parent's cost wrong too.
			name: "a sub-agent's model counts",
			events: []core.Event{
				core.IterStart{N: 1},
				core.LLMDone{Model: "parent", Usage: llm.Usage{InputTokens: 1}},
				core.SubagentEvent{Name: "researcher", Inner: core.LLMDone{
					Model: "child", Usage: llm.Usage{InputTokens: 1}}},
			},
			want: []string{"parent", "child"},
		},
		{
			name: "a call that consumed nothing cannot have produced a misleading zero",
			events: []core.Event{
				core.IterStart{N: 1},
				core.LLMDone{Model: "free-lunch"},
			},
			want: nil,
		},
		{
			name: "a stream that never names a model names nothing",
			events: []core.Event{
				core.IterStart{N: 1},
				core.LLMDone{Usage: llm.Usage{InputTokens: 5}},
			},
			want: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var f folder
			for _, ev := range tc.events {
				f.fold(ev, "")
			}
			if !reflect.DeepEqual(f.models, tc.want) {
				t.Fatalf("models = %v, want %v", f.models, tc.want)
			}
		})
	}
}

func TestPricedSpend(t *testing.T) {
	table := llm.Table{"claude-sonnet-5": {In: 3, Out: 15}}

	tests := []struct {
		name         string
		pricing      llm.Pricing
		models       []string
		usage        llm.Usage
		cost         float64
		wantPriced   bool
		wantUnpriced []string
	}{
		{
			name:       "a priced model with a real tally is a measurement",
			pricing:    table,
			models:     []string{"claude-sonnet-5"},
			usage:      llm.Usage{InputTokens: 1000, OutputTokens: 100},
			cost:       0.0045,
			wantPriced: true,
		},
		{
			name:         "a model the table does not know is not",
			pricing:      table,
			models:       []string{"some-gateway-model"},
			usage:        llm.Usage{InputTokens: 1000, OutputTokens: 100},
			cost:         0,
			wantPriced:   false,
			wantUnpriced: []string{"some-gateway-model"},
		},
		{
			// The proxy, which is all a rollout has without WithPricing.
			name:         "tokens out and a zero tally is not a price, whatever the table says",
			pricing:      nil,
			models:       []string{"claude-sonnet-5"},
			usage:        llm.Usage{InputTokens: 1000},
			cost:         0,
			wantPriced:   false,
			wantUnpriced: []string{"claude-sonnet-5"},
		},
		{
			// A priced model with no cost recorded means the TrackCost
			// middleware is not in the chain. Same lie, different route.
			name:         "a priced model with a zero tally is still not a price",
			pricing:      table,
			models:       []string{"claude-sonnet-5"},
			usage:        llm.Usage{InputTokens: 1000},
			cost:         0,
			wantPriced:   false,
			wantUnpriced: []string{"claude-sonnet-5"},
		},
		{
			name:       "an episode that never spent anything reports an honest zero",
			pricing:    table,
			wantPriced: true,
		},
		{
			// One priced model and one that is not: the total under-reports,
			// so the whole episode is unpriced and only the culprit is named.
			name:         "a mixed episode names only the model it could not price",
			pricing:      table,
			models:       []string{"claude-sonnet-5", "some-gateway-model"},
			usage:        llm.Usage{InputTokens: 1000},
			cost:         0.003,
			wantPriced:   false,
			wantUnpriced: []string{"some-gateway-model"},
		},
		{
			// A Pricing that is not a Pricer makes no claim, so it cannot
			// contradict a tally that looks fine.
			name:       "a plain Pricing is taken at its word",
			pricing:    priceless{},
			models:     []string{"whatever"},
			usage:      llm.Usage{InputTokens: 1000},
			cost:       0.01,
			wantPriced: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ep := &Episode{
				Steps: []Step{{Iteration: 1, Usage: tc.usage}},
				Spend: governor.Progress{CostUSD: tc.cost},
			}
			priced, unpriced := pricedSpend(tc.pricing, tc.models, ep)
			if priced != tc.wantPriced {
				t.Errorf("priced = %v, want %v", priced, tc.wantPriced)
			}
			if !reflect.DeepEqual(unpriced, tc.wantUnpriced) {
				t.Errorf("unpriced = %v, want %v", unpriced, tc.wantUnpriced)
			}
		})
	}
}

// TestRolloutMarksAnUnpricedEpisode is the whole bug end to end: a gateway
// model nobody has a rate for, a run that really spent tokens, and a cost of
// zero that must not be reported as a cost.
func TestRolloutMarksAnUnpricedEpisode(t *testing.T) {
	env := newMemEnv("gateway", "do the thing")
	pricing := llm.Table{"claude-sonnet-5": {In: 3, Out: 15}}

	mk := mkOf(func(Task) (*core.Agent, error) {
		cl := llm.Chain(turnClient(func(int, llm.Request) llm.Response {
			return textTurn("done")
		}), core.TrackCost(pricing))
		return newAgent(t, cl, nil, core.WithModel("some-gateway-model")), nil
	})

	g, err := Rollout(t.Context(), mk, env, 1, WithPricing(pricing))
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	ep := g.Episodes[0]
	if ep.Err != nil {
		t.Fatalf("episode failed: %v", ep.Err)
	}
	if tokens(ep.Usage()) == 0 {
		t.Fatal("the episode spent no tokens, so it cannot demonstrate anything")
	}
	if ep.Spend.CostUSD != 0 {
		t.Fatalf("cost = %v, want the 0 an unpriced model produces", ep.Spend.CostUSD)
	}
	if ep.Priced {
		t.Error("an episode on a model with no rate reports its cost as priced")
	}
	if want := []string{"some-gateway-model"}; !reflect.DeepEqual(ep.Unpriced, want) {
		t.Errorf("Unpriced = %v, want %v", ep.Unpriced, want)
	}
}

// TestRolloutMarksAPricedEpisode is the control: the same wiring against a
// model in the table records a number and says it is one.
func TestRolloutMarksAPricedEpisode(t *testing.T) {
	env := newMemEnv("priced", "do the thing")
	pricing := llm.Table{"claude-sonnet-5": {In: 3, Out: 15}}

	mk := mkOf(func(Task) (*core.Agent, error) {
		cl := llm.Chain(turnClient(func(int, llm.Request) llm.Response {
			r := textTurn("done")
			r.Model = "claude-sonnet-5"
			return r
		}), core.TrackCost(pricing))
		return newAgent(t, cl, nil, core.WithModel("claude-sonnet-5")), nil
	})

	g, err := Rollout(t.Context(), mk, env, 1, WithPricing(pricing))
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	ep := g.Episodes[0]
	if ep.Spend.CostUSD <= 0 {
		t.Fatalf("cost = %v, want a positive number", ep.Spend.CostUSD)
	}
	if !ep.Priced {
		t.Errorf("a priced model was reported as unpriced: %v", ep.Unpriced)
	}
	if len(ep.Unpriced) != 0 {
		t.Errorf("Unpriced = %v, want empty", ep.Unpriced)
	}
	if !g.Priced() {
		t.Error("the group reports itself unpriced")
	}
}

// TestRolloutWithoutPricingUsesTheProxy: no WithPricing, no way to name the
// model from a table — but tokens went out and the tally is zero, and that is
// enough to know the zero is not a price. The name comes off the stream.
func TestRolloutWithoutPricingUsesTheProxy(t *testing.T) {
	env := newMemEnv("proxy", "do the thing")

	mk := mkOf(func(Task) (*core.Agent, error) {
		// No TrackCost anywhere in the chain, which is the other way a cost
		// column ends up reading zero after a real run.
		return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
			return textTurn("done")
		}), nil), nil
	})

	g, err := Rollout(t.Context(), mk, env, 1)
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	ep := g.Episodes[0]
	if ep.Priced {
		t.Error("a zero tally against real tokens was reported as priced")
	}
	// newAgent sets WithModel("test-model"), and the fake response names no
	// model, so the name has to come from the request side of the stream.
	if want := []string{"test-model"}; !reflect.DeepEqual(ep.Unpriced, want) {
		t.Errorf("Unpriced = %v, want %v", ep.Unpriced, want)
	}
}

// TestRolloutEpisodeThatNeverRanIsPriced: zero cost is the truth for an
// episode that never called a model, and marking it unpriced would put an n/a
// on a report for a run that was never going to have a number.
func TestRolloutEpisodeThatNeverRanIsPriced(t *testing.T) {
	env := newMemEnv("broken", "never happens")
	env.reset = func(int) (Task, error) { return Task{}, errors.New("no disk") }

	mk := mkOf(func(Task) (*core.Agent, error) {
		return newAgent(t, turnClient(func(int, llm.Request) llm.Response {
			return textTurn("unreachable")
		}), nil), nil
	})

	g, err := Rollout(t.Context(), mk, env, 1)
	if err != nil {
		t.Fatalf("Rollout: %v", err)
	}
	ep := g.Episodes[0]
	if ep.Err == nil {
		t.Fatal("the episode was supposed to fail its reset")
	}
	if !ep.Priced {
		t.Error("an episode that never spent a token reports its zero as unpriced")
	}
}
