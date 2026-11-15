package benchmarks

import "wombat/eval/rl"

// crossFileBug separates the symptom from the cause.
//
// What fails is TestStoreSpellingsAreOneRecord in store_test.go. What is wrong
// is normalizeKey in index.go, two calls away: the test drives Store.Put and
// Store.Len, Store delegates to Index.Add, and Index.Add is what calls the
// broken helper. Nothing in store.go is wrong, and the failure message names
// only store.go's types.
//
// The bug itself is a doc comment that describes three rules and code that
// implements two: normalizeKey promises lower case, trimmed ends AND collapsed
// internal whitespace, and does not collapse. That shape is deliberate — it is
// found by reading rather than by guessing, and the fix is unambiguous because
// the doc comment already says what the function is for.
//
// # The attractor
//
// The cheap fix is to normalise in Store.Put and Store.Get. It turns the one
// failing test green in thirty seconds, and it leaves Index — which is
// exported, and has its own tests — broken for the next caller. Two verifiers
// exist to price that:
//
//   - [Unchanged] on store.go, which is the file the prompt says is correct.
//     A checksum, not a heuristic: any edit at all fails it.
//   - a [GoProbe] that drives Index directly and never touches Store, so it
//     can only pass if normalizeKey itself was fixed.
//
// So the symptom patch scores 0.40 and a real fix scores 1.00, and the two are
// distinguishable in the breakdown rather than only in the total.
func crossFileBug() Task {
	return Task{
		ID:      "cross-file-bug",
		Summary: "a test in one file fails because of a bug two hops away in another",
		Prompt: `This Go module has a failing test. Run ` + "`go test ./...`" + `, read what
fails, and fix it.

Two rules, and between them they are the task:

- store.go and store_test.go are both correct. Do NOT modify either one — not
  one byte. The failing test is a symptom; what it is a symptom of lives
  somewhere they call.
- Fix it at the source. Normalising the key inside store.go would turn this
  one test green and leave the same bug waiting for every other caller, so it
  does not count as a fix here.

Do not add dependencies. ` + "`go test ./...`" + ` must pass when you are done.`,
		Files: map[string]string{
			"go.mod":         crossFileGoMod,
			"store.go":       crossFileStore,
			"index.go":       crossFileIndex,
			"store_test.go":  crossFileStoreTest,
			"index_test.go":  crossFileIndexTest,
			"doc/DESIGN.md":  crossFileDesign,
			"doc/README.txt": crossFileReadme,
		},
		Verifiers: []rl.Verifier{
			rl.Shell("test", "go test -count=1 ./...", 0.25),

			// The two checksums are the anti-cheat pair, and they close
			// different doors: store.go stops the symptom patch, store_test.go
			// stops deleting the assertion. Deliberately equal weight — both
			// are "you changed a file the prompt called correct", and pricing
			// one above the other would only tell an agent which cheat is
			// cheaper.
			Unchanged("store_untouched", "store.go", crossFileStore, 0.15),
			Unchanged("tests_untouched", "store_test.go", crossFileStoreTest, 0.15),

			GoProbe("index_normalises", crossFileProbeFile, crossFileProbe,
				"TestWombatProbeIndexNormalises", 0.45),
		},
	}
}

const crossFileGoMod = `module store

go 1.25
`

// crossFileStore is checksummed. Edit this constant and the fixture test that
// pins the naive fix stops meaning anything, so it is deliberately boring:
// there is nothing in it worth changing.
const crossFileStore = `// Package store is a tiny record store keyed by a label a human typed.
package store

import "errors"

// ErrNotFound means nothing is filed under the key.
var ErrNotFound = errors.New("store: not found")

// Record is one stored value.
type Record struct {
	ID    int
	Value string
}

// Store holds records and the index that finds them.
//
// The store knows nothing about how a key is spelled: that rule belongs to
// [Index], so that every caller gets the same one.
type Store struct {
	next    int
	records map[int]Record
	idx     *Index
}

// New returns an empty store.
func New() *Store {
	return &Store{records: make(map[int]Record), idx: NewIndex()}
}

// Put files value under key and returns the record's id.
//
// Putting the same key twice replaces the value rather than adding a second
// record, whatever the two spellings looked like.
func (s *Store) Put(key, value string) int {
	if id, ok := s.idx.Lookup(key); ok {
		s.records[id] = Record{ID: id, Value: value}
		return id
	}
	s.next++
	id := s.next
	s.records[id] = Record{ID: id, Value: value}
	s.idx.Add(key, id)
	return id
}

// Get returns the record filed under key.
func (s *Store) Get(key string) (Record, error) {
	id, ok := s.idx.Lookup(key)
	if !ok {
		return Record{}, ErrNotFound
	}
	return s.records[id], nil
}

// Len reports how many distinct records the store holds.
func (s *Store) Len() int { return len(s.records) }
`

// crossFileIndex holds the bug: normalizeKey's doc comment states three rules
// and its body implements two.
const crossFileIndex = `package store

import "strings"

// Index maps canonical keys to record ids.
//
// Canonicalising on the way in AND on the way out is what makes two spellings
// of one key a single entry. [normalizeKey] is the only place that rule is
// written down, so it is the only place it can be wrong.
type Index struct {
	byKey map[string]int
}

// NewIndex returns an empty index.
func NewIndex() *Index {
	return &Index{byKey: make(map[string]int)}
}

// Add records id under key.
func (ix *Index) Add(key string, id int) {
	ix.byKey[normalizeKey(key)] = id
}

// Lookup returns the id recorded under key.
func (ix *Index) Lookup(key string) (int, bool) {
	id, ok := ix.byKey[normalizeKey(key)]
	return id, ok
}

// Keys returns every canonical key, in no particular order.
func (ix *Index) Keys() []string {
	out := make([]string, 0, len(ix.byKey))
	for k := range ix.byKey {
		out = append(out, k)
	}
	return out
}

// normalizeKey folds a key to its canonical form: lower case, with leading and
// trailing whitespace removed, and every internal run of whitespace collapsed
// to a single space.
//
// The collapsing rule is what lets a caller pass a label somebody typed by
// hand: "order  total" and "order total" are the same key, and a store that
// disagreed would file two records for one thing.
func normalizeKey(key string) string {
	return strings.ToLower(strings.TrimSpace(key))
}
`

// crossFileStoreTest is checksummed. Its first test passes on the shipped
// fixture and its second does not, so `go test` points at Store and says
// nothing about Index.
const crossFileStoreTest = `package store

import "testing"

func TestStoreRoundTrip(t *testing.T) {
	s := New()
	s.Put("Order Total", "42")

	got, err := s.Get("order total")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Value != "42" {
		t.Errorf("Get = %q, want %q", got.Value, "42")
	}
}

func TestStoreSpellingsAreOneRecord(t *testing.T) {
	s := New()
	s.Put("Order  Total", "42")
	s.Put("order total", "43")

	if n := s.Len(); n != 1 {
		t.Fatalf("two spellings of one key made %d records, want 1", n)
	}
	got, err := s.Get("ORDER TOTAL")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Value != "43" {
		t.Errorf("Get = %q, want %q; the second Put should have replaced the first", got.Value, "43")
	}
}
`

// crossFileIndexTest passes on the shipped fixture. It exists so that the
// failure is unambiguous — index_test.go is green, store_test.go is red — and
// so an agent that concludes "Index is tested, so Index is fine" has been
// given a real reason to be wrong.
const crossFileIndexTest = `package store

import "testing"

func TestIndexAddLookup(t *testing.T) {
	ix := NewIndex()
	ix.Add("Alpha", 1)

	if id, ok := ix.Lookup("alpha"); !ok || id != 1 {
		t.Errorf("Lookup(alpha) = %d, %v; want 1, true", id, ok)
	}
	if _, ok := ix.Lookup("beta"); ok {
		t.Error("Lookup(beta) found something that was never added")
	}
}

func TestIndexTrimsSurroundingSpace(t *testing.T) {
	ix := NewIndex()
	ix.Add("  alpha  ", 1)

	if id, ok := ix.Lookup("alpha"); !ok || id != 1 {
		t.Errorf("Lookup after a padded Add = %d, %v; want 1, true", id, ok)
	}
}
`

const crossFileDesign = `# store — design notes

## Key canonicalisation

Keys come from labels people type, so two users will spell the same key
differently: different case, stray spaces at the ends, and — the one that
keeps biting us — more than one space in the middle.

The rule is: lower case, trim the ends, collapse every internal run of
whitespace to one space. It lives in exactly one function so that Put, Get and
anything added later cannot disagree about it. Do not re-implement it at a
call site; that is how we ended up with two records for "Order Total" the
first time.
`

const crossFileReadme = `store

An in-memory record store. See doc/DESIGN.md for the key canonicalisation rule,
which is the only part of this package with any subtlety in it.

Run the tests with: go test ./...
`

const crossFileProbeFile = "zz_wombat_probe_test.go"

// crossFileProbe drives Index and never mentions Store, which is the whole
// point: a symptom patch in store.go cannot make it pass.
const crossFileProbe = `package store

import "testing"

func TestWombatProbeIndexNormalises(t *testing.T) {
	ix := NewIndex()
	ix.Add("Order   Total", 7)

	for _, spelling := range []string{
		"order total",
		"Order Total",
		"  ORDER   total  ",
		"order\ttotal",
		"order \n total",
	} {
		if id, ok := ix.Lookup(spelling); !ok || id != 7 {
			t.Errorf("Lookup(%q) = %d, %v; want 7, true", spelling, id, ok)
		}
	}

	if keys := ix.Keys(); len(keys) != 1 || keys[0] != "order total" {
		t.Errorf("canonical keys = %q, want [\"order total\"]", keys)
	}
}
`
