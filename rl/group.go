package rl

import (
	"math"
	"slices"
)

// Group is n samples of ONE task, which is the unit a pass@k is computed over.
//
// Episodes is indexed by sample: Episodes[i].Task.Sample == i. [Rollout] fills
// it by index rather than by append, so the group is byte-identical whatever
// order the samples happened to finish in — a report that reshuffles itself
// between runs cannot be diffed.
type Group struct {
	Env      string
	TaskID   string
	Episodes []*Episode

	// Threshold is the reward at or above which a cleanly finished episode
	// counted as [Success] — the value [WithSuccessThreshold] set, or
	// [DefaultSuccessThreshold].
	//
	// Recorded on the group rather than left implicit because a reward is
	// unreadable without it. 0.94 is a pass under penalties and a failure
	// without them, and a report that has the number can say which; one that
	// has to guess says nothing and lets the reader guess instead.
	Threshold float64
}

// Solved reports whether ep did the work, whether or not it managed to stop.
//
// The distinction the success column cannot make. An episode that scored full
// marks and then ran out of iterations is not the same animal as one that ran
// out of iterations having produced nothing: the first is an agent that cannot
// tell it is finished, the second is an agent that cannot do the task, and
// they are different bugs with different fixes. Both read as a plain 0 in
// pass@k, which is correct — an agent that never terminates has not succeeded
// — and is also the whole reason this exists to be reported alongside.
// A zero Threshold means "not recorded", not "everything passes". Groups built
// by hand and groups read back from an older run both have one, and reading it
// literally would declare every failed episode solved — the loudest possible
// wrong answer from a line whose whole job is to be believed.
func (g *Group) Solved(ep *Episode) bool {
	return ep != nil && g.Threshold > 0 && ep.Failure != Success && ep.Reward >= g.Threshold
}

// Successes counts the episodes that succeeded.
func (g *Group) Successes() int {
	c := 0
	for _, ep := range g.Episodes {
		if ep != nil && ep.Failure == Success {
			c++
		}
	}
	return c
}

// PassAt estimates the probability that at least one of k independent samples
// would succeed, from the n samples actually drawn.
//
// The estimator is the standard unbiased one from Chen et al., "Evaluating
// Large Language Models Trained on Code" (2021), §2.1: with n samples of which
// c succeeded,
//
//	pass@k = 1 - C(n-c, k) / C(n, k)
//
// which is one minus the probability that a k-subset drawn without replacement
// from the n samples contains no successful one.
//
// The naive alternative — "did any of the FIRST k samples succeed" — is
// biased, and biased in the direction that flatters you. It is a single
// Bernoulli draw of the quantity being estimated, so it has the variance of
// one coin flip; averaged over tasks it is unbiased for pass@k only if you
// draw a fresh independent k for every task and throw the other n-k samples
// away. Reusing the same n samples to compute pass@1, pass@2 and pass@8 by
// prefix — which is what everybody actually does — correlates the estimates
// and systematically misreports the small-k end. The combinatorial form uses
// all n samples for every k and has no such tilt.
//
// Returns 0 when k > n: there is no honest estimate of a quantity you did not
// sample enough to see.
//
// Computed as a running product rather than through factorials. C(200, 8) is
// nowhere near overflowing a float64, but 200! is not representable at all,
// and the product form
//
//	prod_{i=0}^{k-1} (n-c-i) / (n-i)
//
// never leaves the unit interval.
func (g *Group) PassAt(k int) float64 {
	n := len(g.Episodes)
	c := g.Successes()

	if k <= 0 || k > n {
		return 0
	}
	// Fewer failures than k means every k-subset contains a success.
	if n-c < k {
		return 1
	}

	p := 1.0
	for i := range k {
		p *= float64(n-c-i) / float64(n-i)
	}
	return 1 - p
}

// Mean is the mean reward over the group's episodes, 0 for an empty group.
func (g *Group) Mean() float64 {
	if len(g.Episodes) == 0 {
		return 0
	}
	sum := 0.0
	for _, ep := range g.Episodes {
		if ep != nil {
			sum += ep.Reward
		}
	}
	return sum / float64(len(g.Episodes))
}

// Std is the population standard deviation of the group's rewards.
//
// Population and not sample: the n episodes are the whole group, not a sample
// drawn from it, and the sample form is undefined at n=1 — which is exactly
// the case a smoke test runs. Read it as spread, not as a confidence interval.
func (g *Group) Std() float64 {
	n := len(g.Episodes)
	if n == 0 {
		return 0
	}
	m := g.Mean()
	sum := 0.0
	for _, ep := range g.Episodes {
		r := 0.0
		if ep != nil {
			r = ep.Reward
		}
		d := r - m
		sum += d * d
	}
	return math.Sqrt(sum / float64(n))
}

// Best returns the highest-scoring episode, or nil for an empty group.
//
// Ties break toward the lower sample number, so the answer does not depend on
// iteration order — and the lowest sample is the cheapest one to reproduce.
func (g *Group) Best() *Episode {
	var best *Episode
	for _, ep := range g.Episodes {
		if ep == nil {
			continue
		}
		if best == nil || ep.Reward > best.Reward {
			best = ep
		}
	}
	return best
}

// Worst returns the lowest-scoring episode, or nil for an empty group. This is
// the one a human opens first.
func (g *Group) Worst() *Episode {
	var worst *Episode
	for _, ep := range g.Episodes {
		if ep == nil {
			continue
		}
		if worst == nil || ep.Reward < worst.Reward {
			worst = ep
		}
	}
	return worst
}

// MedianTurns is the median number of ReAct iterations across the group.
func (g *Group) MedianTurns() float64 {
	return median(g.collect(func(ep *Episode) float64 { return float64(len(ep.Steps)) }))
}

// MedianCost is the median episode cost in USD across the group. Meaningless
// unless [Group.Priced] — see [Episode.Priced].
func (g *Group) MedianCost() float64 {
	return median(g.collect(func(ep *Episode) float64 { return ep.Spend.CostUSD }))
}

// MedianPromptTokens is the median [Episode.PromptTokens] across the group.
func (g *Group) MedianPromptTokens() float64 {
	return median(g.collect(func(ep *Episode) float64 { return float64(ep.PromptTokens()) }))
}

// MedianOutputTokens is the median [Episode.OutputTokens] across the group.
func (g *Group) MedianOutputTokens() float64 {
	return median(g.collect(func(ep *Episode) float64 { return float64(ep.OutputTokens()) }))
}

// Priced reports whether every episode's cost is a real price.
//
// All of them, not most: a group's cost is only as trustworthy as its least
// trustworthy episode, and a median over three priced episodes and one silent
// zero is a number with no meaning. An empty group is priced — there is
// nothing to be wrong about.
func (g *Group) Priced() bool {
	for _, ep := range g.Episodes {
		if ep != nil && !ep.Priced {
			return false
		}
	}
	return true
}

// UnpricedModels names the models the group could not price, sorted and
// deduplicated. Empty when [Group.Priced], and possibly empty when not — see
// [Episode.Unpriced].
func (g *Group) UnpricedModels() []string {
	var out []string
	for _, ep := range g.Episodes {
		if ep != nil {
			out = append(out, ep.Unpriced...)
		}
	}
	// Sorted rather than first-seen: this ends up on a line of a report that
	// gets diffed between runs, and n samples of one task see the same models
	// in whatever order they happened to finish.
	slices.Sort(out)
	return slices.Compact(out)
}

// Failures counts each [FailureKind] in the group.
func (g *Group) Failures() map[FailureKind]int {
	out := make(map[FailureKind]int, len(g.Episodes))
	for _, ep := range g.Episodes {
		if ep != nil {
			out[ep.Failure]++
		}
	}
	return out
}

func (g *Group) collect(f func(*Episode) float64) []float64 {
	out := make([]float64, 0, len(g.Episodes))
	for _, ep := range g.Episodes {
		if ep != nil {
			out = append(out, f(ep))
		}
	}
	return out
}

// median sorts a copy and takes the middle, averaging the two middles for an
// even count. Copies because a caller's slice is not ours to reorder.
func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := slices.Clone(xs)
	slices.Sort(s)
	mid := len(s) / 2
	if len(s)%2 == 1 {
		return s[mid]
	}
	return (s[mid-1] + s[mid]) / 2
}
