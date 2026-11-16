package daemon

import (
	"maps"
	"slices"

	"wombat/pkg/core/llm"
)

// provenance is what one turn adds to its session's origins: the message its
// events made and where they came from, and the compactions that renumbered
// the transcript while it ran.
type provenance struct {
	// at is the index of the message the turn's events made, in the
	// transcript before any compaction of the turn. -1 for none.
	at      int
	origins []Origin

	// folds are the turn's compactions, in order: each folded messages 1
	// through n into the first; see core.TranscriptCompacted.
	folds []int
}

// originsOf is where events come from, oldest first.
func originsOf(events []Event) []Origin {
	out := make([]Origin, 0, len(events))
	for _, e := range events {
		out = append(out, OriginOf(e))
	}
	return out
}

// apply is old with the turn's provenance applied, for the transcript msgs
// the turn left.
//
// The turn's message gets its origins. Each compaction keeps the first
// message where it is, folds messages 1 through n away — their origins go with
// them — and moves every later one up by n. Last, an origin left pointing at
// anything but a person's turn is dropped rather than shown on the wrong
// message: a turn that stopped before its message was added, say, or one
// whose compaction this session never heard about.
func (p provenance) apply(old map[int][]Origin, msgs []llm.Message) map[int][]Origin {
	out := maps.Clone(old)
	if p.at >= 0 && len(p.origins) > 0 {
		if out == nil {
			out = map[int][]Origin{}
		}
		// Concat, not append: the slice may be shared with the record the
		// session had, which a store or a reader may still hold.
		out[p.at] = slices.Concat(out[p.at], p.origins)
	}
	for _, n := range p.folds {
		moved := make(map[int][]Origin, len(out))
		for i, o := range out {
			switch {
			case i == 0:
				moved[0] = o
			case i > n:
				moved[i-n] = o
			}
		}
		out = moved
	}
	for i := range out {
		if i < 0 || i >= len(msgs) || msgs[i].Role != llm.RoleUser {
			delete(out, i)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
