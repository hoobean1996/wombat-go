package rl

import (
	"math"
	"testing"
)

// groupOf builds a group with c successes out of n, giving successes reward 1
// and failures reward 0.
func groupOf(n, c int) *Group {
	g := &Group{Env: "e", TaskID: "t", Episodes: make([]*Episode, n)}
	for i := range n {
		ep := &Episode{Task: Task{ID: "t", Sample: i}, Failure: VerifierFailed}
		if i < c {
			ep.Failure, ep.Reward = Success, 1
		}
		g.Episodes[i] = ep
	}
	return g
}

func TestPassAt(t *testing.T) {
	// Hand-computed from 1 - C(n-c,k)/C(n,k).
	tests := []struct {
		name    string
		n, c, k int
		want    float64
	}{
		{"n8 c3 k1 is just the success rate", 8, 3, 1, 3.0 / 8.0},
		// 1 - (5/8)(4/7) = 1 - 20/56
		{"n8 c3 k2", 8, 3, 2, 1 - 20.0/56.0},
		// 1 - C(5,3)/C(8,3) = 1 - 10/56
		{"n8 c3 k3", 8, 3, 3, 1 - 10.0/56.0},
		// 1 - C(5,5)/C(8,5) = 1 - 1/56
		{"n8 c3 k5", 8, 3, 5, 1 - 1.0/56.0},
		// only five failures, so every 6-subset contains a success
		{"n8 c3 k6 is certain", 8, 3, 6, 1},
		{"n8 c3 k8 is certain", 8, 3, 8, 1},
		{"no successes is never", 8, 0, 4, 0},
		{"all successes is always", 8, 8, 1, 1},
		// 1 - C(9,2)/C(10,2) = 1 - 36/45 = 0.2
		{"n10 c1 k2", 10, 1, 2, 0.2},
		{"k above n is unestimable", 4, 2, 5, 0},
		{"k zero is meaningless", 4, 2, 0, 0},
		{"k negative is meaningless", 4, 2, -1, 0},
		{"empty group", 0, 0, 1, 0},
		{"single sample that passed", 1, 1, 1, 1},
		{"single sample that failed", 1, 0, 1, 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := groupOf(tc.n, tc.c).PassAt(tc.k)
			if math.Abs(got-tc.want) > 1e-12 {
				t.Fatalf("PassAt(%d) with n=%d c=%d = %v, want %v", tc.k, tc.n, tc.c, got, tc.want)
			}
		})
	}
}

// TestPassAtIsUnbiased checks the estimator against the definition it claims
// to implement, by averaging over every possible k-subset of the n samples.
// This is the property the naive "first k" version does not have.
func TestPassAtIsUnbiased(t *testing.T) {
	const n, c, k = 7, 3, 3
	g := groupOf(n, c)

	// Enumerate every k-subset by bitmask and average the indicator.
	total, hits := 0, 0
	for mask := range 1 << n {
		if popcount(mask) != k {
			continue
		}
		total++
		for i := range n {
			if mask&(1<<i) != 0 && g.Episodes[i].Failure == Success {
				hits++
				break
			}
		}
	}
	want := float64(hits) / float64(total)
	if got := g.PassAt(k); math.Abs(got-want) > 1e-12 {
		t.Fatalf("PassAt(%d) = %v, exhaustive average over %d subsets = %v", k, got, total, want)
	}
}

func popcount(x int) int {
	n := 0
	for ; x != 0; x &= x - 1 {
		n++
	}
	return n
}

func TestGroupStatistics(t *testing.T) {
	g := &Group{Env: "e", TaskID: "t", Episodes: []*Episode{
		{Task: Task{ID: "t", Sample: 0}, Reward: 1, Failure: Success, Steps: make([]Step, 4)},
		{Task: Task{ID: "t", Sample: 1}, Reward: 0, Failure: VerifierFailed, Steps: make([]Step, 2)},
		{Task: Task{ID: "t", Sample: 2}, Reward: 0.5, Failure: VerifierFailed, Steps: make([]Step, 6)},
		{Task: Task{ID: "t", Sample: 3}, Reward: 1, Failure: Success, Steps: make([]Step, 8)},
	}}
	g.Episodes[0].Spend.CostUSD = 0.10
	g.Episodes[1].Spend.CostUSD = 0.20
	g.Episodes[2].Spend.CostUSD = 0.30
	g.Episodes[3].Spend.CostUSD = 0.40

	if got, want := g.Successes(), 2; got != want {
		t.Errorf("Successes = %d, want %d", got, want)
	}
	if got, want := g.Mean(), 0.625; math.Abs(got-want) > 1e-12 {
		t.Errorf("Mean = %v, want %v", got, want)
	}
	// population variance of {1,0,0.5,1} about 0.625
	wantStd := math.Sqrt((0.140625 + 0.390625 + 0.015625 + 0.140625) / 4)
	if got := g.Std(); math.Abs(got-wantStd) > 1e-12 {
		t.Errorf("Std = %v, want %v", got, wantStd)
	}
	if got, want := g.MedianTurns(), 5.0; got != want {
		t.Errorf("MedianTurns = %v, want %v", got, want)
	}
	if got, want := g.MedianCost(), 0.25; math.Abs(got-want) > 1e-12 {
		t.Errorf("MedianCost = %v, want %v", got, want)
	}
	// Ties break toward the lower sample so the answer is reproducible.
	if best := g.Best(); best == nil || best.Task.Sample != 0 {
		t.Errorf("Best = %v, want sample 0", best)
	}
	if worst := g.Worst(); worst == nil || worst.Task.Sample != 1 {
		t.Errorf("Worst = %v, want sample 1", worst)
	}
}

func TestGroupStatisticsEmpty(t *testing.T) {
	var g Group
	if g.Mean() != 0 || g.Std() != 0 || g.MedianTurns() != 0 || g.MedianCost() != 0 {
		t.Fatalf("empty group should be all zeroes, got %v %v %v %v",
			g.Mean(), g.Std(), g.MedianTurns(), g.MedianCost())
	}
	if g.Best() != nil || g.Worst() != nil {
		t.Fatal("empty group should have no best or worst episode")
	}
}
