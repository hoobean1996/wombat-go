package benchmarks

import "wombat/eval/rl"

// addFeature is the regression task: a working program, one new requirement,
// and a grade that depends on the OLD behaviour surviving.
//
// Adding a feature to nothing is task 4. Adding one to code that already works
// is the job, and the failure mode it tests for is the common one — an agent
// that rewrites main.go around the new flag and quietly changes what the
// program prints without it. Two of the four verifiers exist only to catch
// that: `default_unchanged` pins the pre-existing output, and [Unchanged] pins
// the pre-existing test file so the agent cannot relax the specification it is
// supposed to keep satisfying.
func addFeature() Task {
	return Task{
		ID:      "add-feature",
		Summary: "add a -top flag to a working word counter without breaking it",
		Prompt: `This Go module is a working word-count program. Read it first.

Add one feature: a ` + "`-top N`" + ` flag.

- With ` + "`-top N`" + `, print the N most frequent words, one per line, most
  frequent first, ties broken alphabetically, and print ONLY the word — no
  count, no other text.
- Without the flag, the program must behave EXACTLY as it does today.
- Do not modify count_test.go, and do not add dependencies.
- ` + "`go build ./...`" + ` and ` + "`go test ./...`" + ` must pass.

For example, ` + "`printf 'a b a c b a' | go run . -top 2`" + ` must print:

    a
    b`,
		Files: map[string]string{
			"go.mod":        addFeatureGoMod,
			"main.go":       addFeatureMain,
			"count.go":      addFeatureCount,
			"count_test.go": addFeatureTest,
		},
		Verifiers: []rl.Verifier{
			// -run pins the ORIGINAL test names, so an agent that adds its own
			// tests neither helps nor hurts itself here; the checksum below is
			// what stops it editing these two.
			rl.Shell("existing_tests", `go test -run 'TestCount$|TestCountEmpty$' ./...`, 0.20),
			Unchanged("tests_untouched", "count_test.go", addFeatureTest, 0.10),
			rl.Shell("build", "go build ./...", 0.10),
			rl.Shell("top_flag",
				`test "$(printf 'a b a c b a' | go run . -top 2 | tr -d '[:space:]')" = "ab"`, 0.40),
			rl.Shell("default_unchanged",
				`test "$(printf 'b a a' | go run . | tr -d '[:space:]')" = "a2b1"`, 0.20),
		},
	}
}

const addFeatureGoMod = `module wordcount

go 1.25
`

const addFeatureCount = `package main

import "strings"

// Count tallies how many times each whitespace-separated word appears in s.
func Count(s string) map[string]int {
	out := make(map[string]int)
	for _, w := range strings.Fields(s) {
		out[w]++
	}
	return out
}
`

const addFeatureMain = `// Command wordcount reads text on stdin and prints each distinct word with
// how many times it occurred, one per line, in alphabetical order.
package main

import (
	"fmt"
	"io"
	"os"
	"slices"
)

func main() {
	src, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, "wordcount:", err)
		os.Exit(1)
	}

	counts := Count(string(src))
	words := make([]string, 0, len(counts))
	for w := range counts {
		words = append(words, w)
	}
	slices.Sort(words)

	for _, w := range words {
		fmt.Printf("%s %d\n", w, counts[w])
	}
}
`

const addFeatureTest = `package main

import "testing"

func TestCount(t *testing.T) {
	got := Count("a b a")
	if len(got) != 2 || got["a"] != 2 || got["b"] != 1 {
		t.Fatalf(` + "`Count(\"a b a\") = %v, want map[a:2 b:1]`" + `, got)
	}
}

func TestCountEmpty(t *testing.T) {
	if got := Count("  \n\t  "); len(got) != 0 {
		t.Fatalf("Count(blank) = %v, want an empty map", got)
	}
}
`
