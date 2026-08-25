package shape

import "fmt"

// Stability measures how much of each turn's transcript a prefix cache could
// have reused from the turn before.
//
// Read Held first. A strategy that never trims holds every pair trivially; the
// interesting number is what a strategy that DOES trim manages to hold, and
// the difference between two windows is the whole reason this type exists.
type Stability struct {
	// Pairs is the number of consecutive turn pairs examined.
	Pairs int

	// Held counts pairs where the entire previous transcript was still a
	// prefix of the new one — a full cache hit was available.
	Held int

	// Reanchored counts pairs where it was not: the leading messages changed,
	// so everything before the divergence had to be re-read at full price.
	Reanchored int

	// Shared[i] is how many leading messages pair i had in common.
	Shared []int

	// PrevLen[i] is how many the earlier turn of pair i had, which is the
	// most that could have been reused.
	PrevLen []int

	// Fraction is sum(Shared)/sum(PrevLen): the share of reusable prefix that
	// actually stayed reusable. 1.0 means every turn could hit in full.
	Fraction float64
}

// String renders the one line a report shows.
func (s Stability) String() string {
	return fmt.Sprintf("held %d/%d pairs, %d re-anchored, reusable fraction %.2f",
		s.Held, s.Pairs, s.Reanchored, s.Fraction)
}

// PrefixStability computes the cache boundary across a run.
func PrefixStability(shots []Shot) Stability {
	var st Stability
	if len(shots) < 2 {
		return st
	}
	sumShared, sumPrev := 0, 0
	for i := 1; i < len(shots); i++ {
		prev, cur := shots[i-1], shots[i]
		shared := SharedPrefix(prev, cur)
		prevLen := len(prev.Messages)

		st.Shared = append(st.Shared, shared)
		st.PrevLen = append(st.PrevLen, prevLen)
		st.Pairs++
		if shared >= prevLen {
			st.Held++
		} else {
			st.Reanchored++
		}
		sumShared += shared
		sumPrev += prevLen
	}
	if sumPrev > 0 {
		st.Fraction = float64(sumShared) / float64(sumPrev)
	}
	return st
}

// SystemDrift reports the turns at which the rendered system prompt changed.
//
// It must always be empty. The system block is the head of the cache prefix,
// so one byte of drift there costs every hit behind it — and nothing about a
// correct run should rebuild it, since [wombat.New] renders it once and freezes
// the string.
func SystemDrift(shots []Shot) []int {
	var out []int
	for i := 1; i < len(shots); i++ {
		if shots[i].SystemSHA != shots[i-1].SystemSHA {
			out = append(out, shots[i].Turn)
		}
	}
	return out
}

// ToolDrift reports the turns at which the offered tool list changed, as
// "turn: before -> after" descriptions.
//
// Unlike system drift this is not automatically a fault: a skill gate is
// SUPPOSED to widen the list mid-run. It is reported so a probe can assert the
// change happened where the mechanism says it should and nowhere else.
func ToolDrift(shots []Shot) []string {
	var out []string
	for i := 1; i < len(shots); i++ {
		if !sameStrings(shots[i-1].Tools, shots[i].Tools) {
			out = append(out, fmt.Sprintf("turn %d: %d -> %d tools",
				shots[i].Turn, len(shots[i-1].Tools), len(shots[i].Tools)))
		}
	}
	return out
}

// MaterializedBytes is the total transcript size sent across the run, which is
// what an eviction strategy is trying to reduce.
func MaterializedBytes(shots []Shot) int {
	n := 0
	for _, s := range shots {
		n += s.TotalBytes()
	}
	return n
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
