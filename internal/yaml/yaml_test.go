package yaml

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

type server struct {
	Addr    string   `yaml:"addr"`
	Origins []string `yaml:"origins"`
	Metrics bool     `yaml:"metrics"`
}

type doc struct {
	Server  server            `yaml:"server"`
	Name    string            `yaml:"name"`
	Iters   int               `yaml:"iters"`
	Budget  float64           `yaml:"budget"`
	Wall    time.Duration     `yaml:"wall"`
	Headers map[string]string `yaml:"headers"`
	System  string            `yaml:"system"`
	Steps   []step            `yaml:"steps"`
	Untag   string
	Skipped string `yaml:"-"`
}

type step struct {
	Run  string   `yaml:"run"`
	Args []string `yaml:"args"`
}

func TestUnmarshal(t *testing.T) {
	const in = `---
# a comment at the top
server:
  addr: 127.0.0.1:8080   # host:port stays one string
  origins:
  - http://localhost:3000
  - 'https://example.com#frag'
  metrics: true

name: "tab\there \u00e9 \"quoted\" # not a comment"
iters: 12
budget: 1.5
wall: 1h30m
headers: {X-Client: wombat, "X-Two": 'it''s'}
system: |
  You are terse.

    Indented, and kept.
untag: by its name
steps:
  - run: go test
    args: [-race, ./...]
  - run: go vet
`
	var got doc
	if err := Unmarshal([]byte(in), &got); err != nil {
		t.Fatal(err)
	}
	want := doc{
		Server: server{
			Addr:    "127.0.0.1:8080",
			Origins: []string{"http://localhost:3000", "https://example.com#frag"},
			Metrics: true,
		},
		Name:    "tab\there é \"quoted\" # not a comment",
		Iters:   12,
		Budget:  1.5,
		Wall:    90 * time.Minute,
		Headers: map[string]string{"X-Client": "wombat", "X-Two": "it's"},
		System:  "You are terse.\n\n  Indented, and kept.\n",
		Untag:   "by its name",
		Steps: []step{
			{Run: "go test", Args: []string{"-race", "./..."}},
			{Run: "go vet"},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got  %#v\nwant %#v", got, want)
	}
}

// A key that is absent, or null, leaves its field as it was: that is what
// lets a struct filled with defaults keep every one the file does not name.
// A key that is present replaces its field, a list included.
func TestAbsentAndNullKeepWhatWasThere(t *testing.T) {
	got := doc{
		Name:    "default",
		Iters:   3,
		Server:  server{Addr: "default:1", Origins: []string{"a", "b"}},
		Headers: map[string]string{"Keep": "me"},
	}
	in := "name:\niters: ~\nserver:\n  addr: null\n  origins: [c]\n"
	if err := Unmarshal([]byte(in), &got); err != nil {
		t.Fatal(err)
	}
	if got.Name != "default" || got.Iters != 3 || got.Server.Addr != "default:1" || got.Headers["Keep"] != "me" {
		t.Errorf("a null or absent key changed a field: %+v", got)
	}
	if !reflect.DeepEqual(got.Server.Origins, []string{"c"}) {
		t.Errorf("origins = %q, want the file's list in place of the default", got.Server.Origins)
	}
	if err := Unmarshal([]byte("# nothing but a comment\n\n"), &got); err != nil || got.Name != "default" {
		t.Errorf("an empty document: err %v, name %q", err, got.Name)
	}
}

func TestBlockStrings(t *testing.T) {
	for _, tc := range []struct {
		name, in, want string
	}{
		{"literal keeps breaks and clips the end", "s: |\n  a\n  b\n\n\nnext: 1", "a\nb\n"},
		{"strip", "s: |-\n  a\n  b\n", "a\nb"},
		{"keep", "s: |+\n  a\n\n\nnext: 1", "a\n\n\n"},
		{"leading blank lines are kept", "s: |\n\n  a\n", "\na\n"},
		{"folded joins a paragraph", "s: >\n  one\n  two\n\n  three\n", "one two\nthree\n"},
		{"folded keeps a more-indented line", "s: >-\n  a\n    code\n  b\n", "a\n  code\nb"},
		{"a comment after the header", "s: | # why\n  a\n", "a\n"},
		{"ends at the parent's indentation", "s: |\n  a\nt: x\n", "a\n"},
		{"empty", "s: |\nt: x\n", ""},
		{"in a list", "l:\n  - |\n    a\n  - b\n", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got struct {
				S string   `yaml:"s"`
				T string   `yaml:"t"`
				L []string `yaml:"l"`
				N int      `yaml:"next"`
			}
			if err := Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatal(err)
			}
			if tc.name == "in a list" {
				if !reflect.DeepEqual(got.L, []string{"a\n", "b"}) {
					t.Errorf("l = %q", got.L)
				}
				return
			}
			if got.S != tc.want {
				t.Errorf("s = %q, want %q", got.S, tc.want)
			}
		})
	}
}

func TestScalarsMeanWhatTheySay(t *testing.T) {
	var got struct {
		Octal  int    `yaml:"octal"`
		Hex    int    `yaml:"hex"`
		Word   string `yaml:"word"`
		Number string `yaml:"number"`
		Colon  string `yaml:"colon"`
		Hash   string `yaml:"hash"`
		Empty  string `yaml:"empty"`
		On     string `yaml:"on"`
	}
	in := "octal: 012\nhex: 0x1F\nword: inf\nnumber: 0123\ncolon: a:b\nhash: a#b\nempty: \"\"\non: on\n"
	if err := Unmarshal([]byte(in), &got); err != nil {
		t.Fatal(err)
	}
	if got.Octal != 12 || got.Hex != 31 {
		t.Errorf("octal %d (want 12: a leading zero is not octal), hex %d", got.Octal, got.Hex)
	}
	if got.Word != "inf" || got.Number != "0123" || got.Colon != "a:b" || got.Hash != "a#b" || got.Empty != "" || got.On != "on" {
		t.Errorf("strings changed on the way in: %+v", got)
	}
}

func TestGeneric(t *testing.T) {
	var got map[string]any
	in := "a: 1\nb: 1.5\nc: true\nd: \"true\"\ne: [x, 2]\nf: {g: h}\ni: ~\nj: inf\n"
	if err := Unmarshal([]byte(in), &got); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"a": 1, "b": 1.5, "c": true, "d": "true",
		"e": []any{"x", 2}, "f": map[string]any{"g": "h"}, "i": nil, "j": "inf",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got  %#v\nwant %#v", got, want)
	}
}

// Every refusal names the line, and says what is wrong in words a person
// editing the file can act on.
func TestErrorsNameTheLine(t *testing.T) {
	for _, tc := range []struct {
		name, in string
		line     int
		want     string
	}{
		{"unknown key", "server:\n  adr: x\n", 2, `unknown key "adr" in server (known: addr, origins, metrics)`},
		{"unknown top-level key", "nmae: x\n", 1, "at the top level"},
		{"duplicate key", "name: a\nname: b\n", 2, "given twice"},
		{"tab", "server:\n\taddr: x\n", 2, "tab"},
		{"bad bool", "server:\n  metrics: yes\n", 2, `server.metrics: "yes" is not true or false`},
		{"quoted number", "iters: \"12\"\n", 1, "quoted, which makes it a string"},
		{"not a number", "iters: twelve\n", 1, "not a whole number"},
		{"bad duration", "wall: 90\n", 1, "a duration"},
		{"list for a string", "name:\n  - a\n", 2, "name wants a single value, not a list"},
		{"string for a list", "server:\n  origins: a\n", 2, "wants a list"},
		{"string for a struct", "server: x\n", 1, "wants a mapping"},
		{"unterminated quote", "name: \"abc\n", 1, "does not end on its line"},
		{"bad escape", `name: "\q"` + "\n", 1, "not an escape"},
		{"anchor", "name: &a x\n", 1, "anchors"},
		{"alias", "name: *a\n", 1, "anchors"},
		{"tag", "name: !str x\n", 1, "tags"},
		{"nested flow", "steps: [[a]]\n", 1, "inside another"},
		{"unclosed flow", "headers: {a: b\n", 1, "does not close"},
		{"run-on plain string", "name: one\n  two\n", 2, "continues a value"},
		{"a second document", "name: a\n---\nname: b\n", 2, "second document"},
		{"a list on the key's line", "server:\n  origins: - a\n", 2, "on the lines below"},
		{"a colon in a value", "name: Note: this\n", 1, "quote the value"},
		{"over-indented key", "server:\n  addr: x\n    metrics: true\n", 3, "continues a value"},
		{"not a key", "server:\n  just words\n", 2, "expected a key"},
		{"item among keys", "server:\n  addr: x\n  - y\n", 3, "list item among the keys"},
		{"block indicator", "system: |2\n  x\n", 1, "only | and >"},
		{"dedent past the root", "  name: a\nx: b\n", 2, "indented less"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var d doc
			err := Unmarshal([]byte(tc.in), &d)
			e, ok := err.(*Error)
			if !ok {
				t.Fatalf("err = %v (%T), want an *Error", err, err)
			}
			if e.Line != tc.line || !strings.Contains(e.Msg, tc.want) {
				t.Errorf("got line %d %q, want line %d and %q", e.Line, e.Msg, tc.line, tc.want)
			}
		})
	}
}

func TestUnmarshalWantsAPointer(t *testing.T) {
	var d doc
	if err := Unmarshal([]byte("name: x"), d); err == nil {
		t.Error("decoded into a copy")
	}
	if err := Unmarshal([]byte("\xff"), &d); err == nil {
		t.Error("accepted a document that is not UTF-8")
	}
}

// FuzzUnmarshal: whatever the document, Unmarshal returns — a value or an
// error — and never panics.
func FuzzUnmarshal(f *testing.F) {
	for _, seed := range []string{
		"a: 1\n",
		"a:\n  - b\n  - c: d\n    e: [f, 'g']\n",
		"s: |+\n  x\n\n",
		"k: {a: \"b\\u00e9\", c: 'd''e'}\n",
		"- - a\n  - b\n",
		"a: >-\n  x\n    y\n\n  z\n",
		"---\nserver:\n  addr: :8080 # c\n",
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var m map[string]any
		_ = Unmarshal(data, &m)
		var d doc
		_ = Unmarshal(data, &d)
	})
}
