package tool

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

// TestMain silences the Error-level slog line that recovered() emits for every
// recovered panic. Several tests here panic on purpose; without this the test
// output is unreadable.
func TestMain(m *testing.M) {
	slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	os.Exit(m.Run())
}

// def is a minimal valid Def whose handler returns out/err.
func def(t *testing.T, name string, fn Fn) Def {
	t.Helper()
	return Def{
		Name:        name,
		Description: name + " description",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Fn:          fn,
	}
}

func ok(s string) Fn {
	return func(context.Context, json.RawMessage) (string, error) { return s, nil }
}

func fails(err error) Fn {
	return func(context.Context, json.RawMessage) (string, error) { return "", err }
}

func use(id, name, input string) llm.ToolUse {
	return llm.ToolUse{ID: llm.ToolUseID(id), Name: name, Input: json.RawMessage(input)}
}

// ===== Def =====

func TestDefSpec(t *testing.T) {
	d := Def{
		Name:        "view_file",
		Description: "reads a file",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
		// Fields below must NOT leak into the wire projection.
		Caps:       CapReadOnly,
		Needs:      NeedFSRead,
		Idempotent: true,
		Timeout:    time.Second,
		Category:   "file_io",
		Retryable:  func(error) bool { return true },
		Fn:         ok("x"),
	}

	got := d.Spec()
	want := llm.ToolSpec{Name: d.Name, Description: d.Description, InputSchema: d.InputSchema}
	if got.Name != want.Name || got.Description != want.Description || string(got.InputSchema) != string(want.InputSchema) {
		t.Errorf("Spec() = %+v, want %+v", got, want)
	}

	// The schema is handed to the model byte for byte.
	if string(got.InputSchema) != string(d.InputSchema) {
		t.Errorf("InputSchema mutated: got %s, want %s", got.InputSchema, d.InputSchema)
	}
}

func TestSpecs(t *testing.T) {
	defs := []Def{
		{Name: "a", Description: "A", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{Name: "b", Description: "B", InputSchema: json.RawMessage(`{"type":"object"}`)},
	}
	got := Specs(defs)
	if len(got) != 2 {
		t.Fatalf("len(Specs) = %d, want 2", len(got))
	}
	if got[0].Name != "a" || got[1].Name != "b" {
		t.Errorf("Specs order = %q,%q, want a,b", got[0].Name, got[1].Name)
	}

	if n := len(Specs(nil)); n != 0 {
		t.Errorf("len(Specs(nil)) = %d, want 0", n)
	}
}

func TestDefHas(t *testing.T) {
	tests := []struct {
		name string
		caps Cap
		ask  Cap
		want bool
	}{
		{"exact single", CapReadOnly, CapReadOnly, true},
		{"absent single", CapReadOnly, CapExec, false},
		{"every bit present", CapExec | CapMutating | CapNetwork, CapExec | CapNetwork, true},
		{"only one bit present", CapExec | CapMutating, CapExec | CapNetwork, false},
		{"zero cap is always satisfied", 0, 0, true},
		{"zero ask against a set def", CapTerminal, 0, true},
		{"nothing declared", 0, CapPause, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := (Def{Caps: tt.caps}).Has(tt.ask); got != tt.want {
				t.Errorf("Def{Caps:%b}.Has(%b) = %v, want %v", tt.caps, tt.ask, got, tt.want)
			}
		})
	}
}

// ===== Typed =====

type typedIn struct {
	Path string `json:"path"`
	N    int    `json:"n"`
}

func TestTyped(t *testing.T) {
	var seen typedIn
	d := Typed(Def{Name: "t", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`)},
		func(_ context.Context, in typedIn) (string, error) {
			seen = in
			return "handled " + in.Path, nil
		})

	t.Run("decodes into the typed value", func(t *testing.T) {
		out, err := d.Fn(context.Background(), json.RawMessage(`{"path":"/tmp/x","n":3}`))
		if err != nil {
			t.Fatalf("Fn returned error %v, want nil", err)
		}
		if out != "handled /tmp/x" {
			t.Errorf("out = %q, want %q", out, "handled /tmp/x")
		}
		if seen != (typedIn{Path: "/tmp/x", N: 3}) {
			t.Errorf("decoded = %+v, want {Path:/tmp/x N:3}", seen)
		}
	})

	t.Run("empty input is the zero value, not an error", func(t *testing.T) {
		for _, raw := range []json.RawMessage{nil, {}, json.RawMessage("")} {
			seen = typedIn{Path: "sentinel"}
			out, err := d.Fn(context.Background(), raw)
			if err != nil {
				t.Fatalf("Fn(%q) returned error %v, want nil", raw, err)
			}
			if seen != (typedIn{}) {
				t.Errorf("Fn(%q) decoded = %+v, want zero value", raw, seen)
			}
			if out != "handled " {
				t.Errorf("Fn(%q) out = %q, want %q", raw, out, "handled ")
			}
		}
	})

	t.Run("decode failure is reported as invalid input", func(t *testing.T) {
		// Two properties, and the second is the one with teeth. The message
		// teaches the MODEL that it emitted a bad call and can correct it; the
		// ErrInvalidInput identity teaches the MIDDLEWARE the same thing, so a
		// model fumbling a schema three times does not get the tool taken away
		// by the circuit breaker.
		for _, raw := range []string{`{"path":`, `{"n":"three"}`, `[1,2]`} {
			out, err := d.Fn(context.Background(), json.RawMessage(raw))
			if err == nil {
				t.Fatalf("Fn(%s) returned nil error, want a decode failure", raw)
			}
			if !errors.Is(err, ErrInvalidInput) {
				t.Errorf("Fn(%s) error = %q, want errors.Is(err, ErrInvalidInput)", raw, err)
			}
			if !IsCallerFault(err) {
				t.Errorf("Fn(%s) error = %q, want it blamed on the caller", raw, err)
			}
			if !strings.Contains(err.Error(), d.Name) {
				t.Errorf("Fn(%s) error = %q, want it to name the tool %q", raw, err, d.Name)
			}
			if out != "" {
				t.Errorf("Fn(%s) out = %q, want %q", raw, out, "")
			}
		}
	})

	t.Run("carries the rest of the Def through unchanged", func(t *testing.T) {
		src := Def{
			Name: "n", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`),
			Caps: CapReadOnly, Needs: NeedFSRead, Idempotent: true,
			Timeout: 7 * time.Second, Category: "cat",
		}
		got := Typed(src, func(context.Context, typedIn) (string, error) { return "", nil })
		if got.Name != src.Name || got.Caps != src.Caps || got.Needs != src.Needs ||
			!got.Idempotent || got.Timeout != src.Timeout || got.Category != src.Category {
			t.Errorf("Typed dropped metadata: got %+v, want the fields of %+v", got, src)
		}
	})
}

// ===== Sets =====

func TestNewSet(t *testing.T) {
	a := def(t, "a", ok("a"))
	b := def(t, "b", ok("b"))
	s := NewSet(a, b)

	if got := s.Visible(context.Background()); len(got) != 2 || got[0].Name != "a" || got[1].Name != "b" {
		t.Errorf("Visible order = %v, want [a b]", names(got))
	}
	if d, found := s.Find("b"); !found || d.Name != "b" {
		t.Errorf("Find(b) = (%q, %v), want (b, true)", d.Name, found)
	}
	if _, found := s.Find("nope"); found {
		t.Error("Find(nope) = found, want not found")
	}
	if got := NewSet().Visible(context.Background()); len(got) != 0 {
		t.Errorf("empty set Visible = %v, want empty", names(got))
	}
}

// TestNewSetDuplicatePanics pins the deliberate panic: a shadowed tool means
// the model calls one thing and the harness runs another, silently.
func TestNewSetDuplicatePanics(t *testing.T) {
	defer func() {
		v := recover()
		if v == nil {
			t.Fatal("NewSet with a duplicate name did not panic, want a panic")
		}
		msg, isStr := v.(string)
		if !isStr || !strings.Contains(msg, "duplicate tool name dup") {
			t.Errorf("panic value = %v, want a string mentioning %q", v, "duplicate tool name dup")
		}
	}()
	NewSet(def(t, "dup", ok("1")), def(t, "other", ok("2")), def(t, "dup", ok("3")))
}

func names(defs []Def) []string {
	out := make([]string, len(defs))
	for i, d := range defs {
		out[i] = d.Name
	}
	return out
}

func TestFilter(t *testing.T) {
	defs := []Def{
		{Name: "keep1"}, {Name: "drop"}, {Name: "keep2"},
	}
	got := Filter(defs, func(d Def) bool { return strings.HasPrefix(d.Name, "keep") })
	if len(got) != 2 || got[0].Name != "keep1" || got[1].Name != "keep2" {
		t.Errorf("Filter = %v, want [keep1 keep2] (order preserved)", names(got))
	}
	if got := Filter(defs, func(Def) bool { return false }); len(got) != 0 {
		t.Errorf("Filter(none) = %v, want empty", names(got))
	}
	if got := Filter(nil, func(Def) bool { return true }); len(got) != 0 {
		t.Errorf("Filter(nil) = %v, want empty", names(got))
	}
}

func TestOnlyCaps(t *testing.T) {
	// OnlyCaps keeps a tool whose caps are a SUBSET of what is allowed: an
	// allow-list, so a tool declaring a capability nobody granted is dropped.
	tests := []struct {
		name    string
		allowed Cap
		caps    Cap
		want    bool
	}{
		{"read-only under read-only", CapReadOnly, CapReadOnly, true},
		{"exec under read-only", CapReadOnly, CapExec, false},
		{"bash under read-only|network", CapReadOnly | CapNetwork, CapExec | CapMutating | CapNetwork, false},
		{"read-only under read-only|network", CapReadOnly | CapNetwork, CapReadOnly, true},
		{"http_get under read-only|network", CapReadOnly | CapNetwork, CapReadOnly | CapNetwork, true},
		{"no caps declared passes anything", CapReadOnly, 0, true},
		{"no caps declared passes an empty allowance", 0, 0, true},
		{"anything fails an empty allowance", 0, CapReadOnly, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := OnlyCaps(tt.allowed)(Def{Caps: tt.caps}); got != tt.want {
				t.Errorf("OnlyCaps(%b)(Def{Caps:%b}) = %v, want %v", tt.allowed, tt.caps, got, tt.want)
			}
		})
	}
}

func TestProvided(t *testing.T) {
	tests := []struct {
		name  string
		have  Need
		needs Need
		want  bool
	}{
		{"fs read on a host with fs read", NeedFSRead, NeedFSRead, true},
		{"fs read on a host with nothing", 0, NeedFSRead, false},
		{"bash on a host without exec", NeedFSRead | NeedFSWrite, NeedExec | NeedFSRead | NeedFSWrite, false},
		{"bash on a host with exec", NeedExec | NeedFSRead | NeedFSWrite, NeedExec | NeedFSRead | NeedFSWrite, true},
		{"pure tool needs nothing", 0, 0, true},
		{"pure tool on a rich host", NeedExec | NeedNetwork, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Provided(tt.have)(Def{Needs: tt.needs}); got != tt.want {
				t.Errorf("Provided(%b)(Def{Needs:%b}) = %v, want %v", tt.have, tt.needs, got, tt.want)
			}
		})
	}
}

// ===== FindPause / FindTerminal =====

func TestFindPauseAndTerminal(t *testing.T) {
	pause := Def{Name: "ask_user", Caps: CapPause}
	term := Def{Name: "submit", Caps: CapTerminal}
	plain := Def{Name: "calculator", Caps: CapReadOnly}
	set := NewSet(plain, pause, term)

	t.Run("finds the first pause in the batch", func(t *testing.T) {
		u, found := FindPause(set, []llm.ToolUse{
			use("1", "calculator", `{}`),
			use("2", "ask_user", `{}`),
			use("3", "ask_user", `{}`),
		})
		if !found || u.ID != "2" {
			t.Errorf("FindPause = (%q, %v), want (2, true)", u.ID, found)
		}
	})

	t.Run("no pause in the set means never", func(t *testing.T) {
		only := NewSet(plain)
		if u, found := FindPause(only, []llm.ToolUse{use("1", "ask_user", `{}`)}); found {
			t.Errorf("FindPause = (%q, true), want not found: ask_user is not in this set", u.ID)
		}
	})

	t.Run("an unknown name is not a pause", func(t *testing.T) {
		if _, found := FindPause(set, []llm.ToolUse{use("1", "ghost", `{}`)}); found {
			t.Error("FindPause matched an unknown tool, want not found")
		}
	})

	t.Run("pause keys on the capability, not on the name", func(t *testing.T) {
		odd := NewSet(Def{Name: "confirm_deploy", Caps: CapPause})
		u, found := FindPause(odd, []llm.ToolUse{use("9", "confirm_deploy", `{}`)})
		if !found || u.ID != "9" {
			t.Errorf("FindPause = (%q, %v), want (9, true)", u.ID, found)
		}
	})

	t.Run("finds the first terminal in the batch", func(t *testing.T) {
		u, found := FindTerminal(set, []llm.ToolUse{
			use("1", "calculator", `{}`),
			use("2", "submit", `{}`),
		})
		if !found || u.ID != "2" {
			t.Errorf("FindTerminal = (%q, %v), want (2, true)", u.ID, found)
		}
	})

	t.Run("a pause tool is not a terminal tool", func(t *testing.T) {
		if _, found := FindTerminal(set, []llm.ToolUse{use("1", "ask_user", `{}`)}); found {
			t.Error("FindTerminal matched a CapPause tool, want not found")
		}
	})

	t.Run("empty batch", func(t *testing.T) {
		if _, found := FindPause(set, nil); found {
			t.Error("FindPause(nil) = found, want not found")
		}
		if _, found := FindTerminal(set, nil); found {
			t.Error("FindTerminal(nil) = found, want not found")
		}
	})
}

// ===== Annotate =====

// TestAnnotateOutsideDispatcherIsNoOp pins the documented promise that a tool
// which annotates is still an ordinary function: callable in a test with a
// bare context, no harness, no panic.
func TestAnnotateOutsideDispatcherIsNoOp(t *testing.T) {
	Annotate(context.Background(), "skill:demo", "bulk")
	Annotate(context.TODO())

	// Even a context carrying a nil *annotations must not panic.
	ctx := context.WithValue(context.Background(), annotateKey{}, (*annotations)(nil))
	Annotate(ctx, "x")
}

func TestAnnotateCollectedByDispatcher(t *testing.T) {
	tagger := def(t, "tagger", func(ctx context.Context, _ json.RawMessage) (string, error) {
		Annotate(ctx, "skill:demo")
		Annotate(ctx, "bulk", "evictable")
		return "body", nil
	})
	plain := def(t, "plain", ok("no tags"))

	d := NewDispatcher(Direct)
	got := d.Dispatch(context.Background(), NewSet(tagger, plain), []llm.ToolUse{
		use("1", "tagger", `{}`),
		use("2", "plain", `{}`),
	})

	if want := []string{"skill:demo", "bulk", "evictable"}; !equalStrings(got[0].Tags, want) {
		t.Errorf("tags = %v, want %v (in annotation order)", got[0].Tags, want)
	}
	if len(got[1].Tags) != 0 {
		t.Errorf("plain tool tags = %v, want none", got[1].Tags)
	}

	// Tags are the harness's notes, never the model's: Block carries only the
	// observation.
	if tr, isResult := got[0].Block().(llm.ToolResult); !isResult || tr.Content != "body" {
		t.Errorf("Block() = %#v, want a ToolResult with Content %q", got[0].Block(), "body")
	}
}

// TestAnnotateOnAFailedCallIsStillCollected: a tool that labelled its output
// and then failed still gets its label recorded, because the strategy that
// reads tags has to see the error result too.
func TestAnnotateOnAFailedCall(t *testing.T) {
	boom := errors.New("nope")
	d := def(t, "tagger", func(ctx context.Context, _ json.RawMessage) (string, error) {
		Annotate(ctx, "attempted")
		return "", boom
	})
	got := NewDispatcher(Direct).Dispatch(context.Background(), NewSet(d), []llm.ToolUse{use("1", "tagger", `{}`)})
	if !errors.Is(got[0].Err, boom) {
		t.Fatalf("Err = %v, want %v", got[0].Err, boom)
	}
	if !equalStrings(got[0].Tags, []string{"attempted"}) {
		t.Errorf("tags = %v, want [attempted]", got[0].Tags)
	}
}

func equalStrings(a, b []string) bool {
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

// ===== Info / Lookup on the context =====

func TestInfoFrom(t *testing.T) {
	t.Run("zero value outside a dispatcher", func(t *testing.T) {
		if got := InfoFrom(context.Background()); got != (Info{}) {
			t.Errorf("InfoFrom(background) = %+v, want the zero Info", got)
		}
	})

	t.Run("round trips", func(t *testing.T) {
		ctx := WithInfo(context.Background(), Info{UseID: "toolu_01"})
		if got := InfoFrom(ctx).UseID; got != "toolu_01" {
			t.Errorf("UseID = %q, want %q", got, "toolu_01")
		}
	})

	t.Run("the dispatcher sets the id of the call in flight", func(t *testing.T) {
		var seen []llm.ToolUseID
		d := def(t, "peek", func(ctx context.Context, _ json.RawMessage) (string, error) {
			seen = append(seen, InfoFrom(ctx).UseID)
			return "", nil
		})
		NewDispatcher(Direct).Dispatch(context.Background(), NewSet(d), []llm.ToolUse{
			use("toolu_a", "peek", `{}`),
			use("toolu_b", "peek", `{}`),
		})
		if len(seen) != 2 || seen[0] != "toolu_a" || seen[1] != "toolu_b" {
			t.Errorf("per-call UseIDs = %v, want [toolu_a toolu_b]", seen)
		}
	})
}

func TestLookupFrom(t *testing.T) {
	t.Run("nil outside an agent loop", func(t *testing.T) {
		if got := LookupFrom(context.Background()); got != nil {
			t.Error("LookupFrom(background) != nil, want nil so tools can detect it")
		}
	})

	t.Run("round trips", func(t *testing.T) {
		ctx := WithLookup(context.Background(), func(id llm.ToolUseID) (string, error) {
			if id == "known" {
				return "observation", nil
			}
			return "", errors.New("not found")
		})
		l := LookupFrom(ctx)
		if l == nil {
			t.Fatal("LookupFrom returned nil, want the installed resolver")
		}
		if got, err := l("known"); err != nil || got != "observation" {
			t.Errorf("lookup(known) = (%q, %v), want (observation, nil)", got, err)
		}
		if _, err := l("other"); err == nil {
			t.Error("lookup(other) returned nil error, want a failure")
		}
	})
}

// ===== WithObserver =====

func TestWithObserverBracketsOnePairPerLogicalCall(t *testing.T) {
	// The observer sits OUTSIDE retry, so three attempts still make one pair.
	attempts := 0
	d := Def{
		Name: "flaky", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`),
		Idempotent: true,
		Retryable:  func(error) bool { return true },
		Fn: func(context.Context, json.RawMessage) (string, error) {
			attempts++
			if attempts < 3 {
				return "", errors.New("transient")
			}
			return "recovered", nil
		},
	}

	var phases []Phase
	var doneOut string
	h := Chain(Direct,
		WithRetry(RetryPolicy{MaxAttempts: 3, Base: time.Millisecond, Jitter: -1}),
		WithObserver(func(_ context.Context, o Observation) {
			phases = append(phases, o.Phase)
			if o.Phase == PhaseDone {
				doneOut = o.Output
			}
		}),
	)

	out, err := h(context.Background(), d, use("1", "flaky", `{}`))
	if err != nil || out != "recovered" {
		t.Fatalf("handler = (%q, %v), want (recovered, nil)", out, err)
	}
	if attempts != 3 {
		t.Errorf("attempts = %d, want 3", attempts)
	}
	if len(phases) != 2 || phases[0] != PhaseStart || phases[1] != PhaseDone {
		t.Errorf("observed phases = %v, want exactly [PhaseStart PhaseDone]", phases)
	}
	if doneOut != "recovered" {
		t.Errorf("observed output = %q, want %q", doneOut, "recovered")
	}
}

func TestAttachReachesTheResultBlock(t *testing.T) {
	img := llm.Media{MediaType: "image/png", Data: []byte{1, 2, 3}}
	shot := def(t, "shot", func(ctx context.Context, _ json.RawMessage) (string, error) {
		Attach(ctx, img)
		if got := Attached(ctx); len(got) != 1 {
			t.Errorf("Attached = %v, want the one image", got)
		}
		return "Attached a screenshot.", nil
	})
	broken := def(t, "broken", func(ctx context.Context, _ json.RawMessage) (string, error) {
		Attach(ctx, img)
		return "", errors.New("render failed")
	})

	got := NewDispatcher(Direct).Dispatch(context.Background(), NewSet(shot, broken), []llm.ToolUse{
		use("1", "shot", `{}`), use("2", "broken", `{}`),
	})
	tr := got[0].Block().(llm.ToolResult)
	if len(tr.Media) != 1 || tr.Content != "Attached a screenshot." {
		t.Errorf("Block() = %#v, want the text and the image", tr)
	}
	if got[0].Size() != len("Attached a screenshot.")+3 {
		t.Errorf("Size() = %d, want the text and the image counted", got[0].Size())
	}
	// An error result is text: whatever the failing call attached is dropped.
	if tr := got[1].Block().(llm.ToolResult); !tr.IsError || len(tr.Media) != 0 {
		t.Errorf("failed call Block() = %#v, want an error with no media", tr)
	}

	// Outside a dispatcher both are no-ops.
	Attach(context.Background(), img)
	if Attached(context.Background()) != nil {
		t.Error("Attached outside a dispatcher returned something")
	}
}
