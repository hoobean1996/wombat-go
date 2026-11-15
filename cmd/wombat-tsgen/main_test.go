package main

import (
	"os"
	"strings"
	"testing"
)

// TestEventsTSIsCurrent: web/events.ts is generated, and checked in so a
// front end builds without Go. A change to an event or an API body committed
// without `make ts` leaves it declaring a stream that no longer exists, and
// tsc vouches for every line of it.
func TestEventsTSIsCurrent(t *testing.T) {
	b, err := os.ReadFile("../../web/events.ts")
	if err != nil {
		t.Fatal(err)
	}
	want, got := strings.Split(string(b), "\n"), strings.Split(generate(), "\n")
	for i := range max(len(want), len(got)) {
		var w, g string
		if i < len(want) {
			w = want[i]
		}
		if i < len(got) {
			g = got[i]
		}
		if w != g {
			t.Fatalf("web/events.ts is stale — run `make ts`. First difference, line %d:\n  checked in: %q\n  generated:  %q", i+1, w, g)
		}
	}
}
