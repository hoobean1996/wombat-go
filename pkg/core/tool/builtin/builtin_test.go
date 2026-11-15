package builtin

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// TestDefaultToolSurface is the table-driven contract check on the whole
// registry: whatever a constructor does, its Def has to be something a
// provider will accept.
func TestDefaultToolSurface(t *testing.T) {
	defs := Default(Deps{})
	if len(defs) == 0 {
		t.Fatal("Default() returned no tools")
	}

	seen := make(map[string]bool, len(defs))
	for _, d := range defs {
		t.Run(d.Name, func(t *testing.T) {
			if d.Name == "" {
				t.Error("Name is empty")
			}
			if seen[d.Name] {
				t.Errorf("duplicate name %q: tool.NewSet would panic on this set", d.Name)
			}
			seen[d.Name] = true

			if strings.TrimSpace(d.Description) == "" {
				t.Error("Description is empty: the model has nothing to choose on")
			}
			if !utf8.ValidString(d.Description) {
				t.Error("Description is not valid UTF-8")
			}

			// InputSchema is handed to the provider byte for byte, so it has to
			// be a JSON object with "type":"object" — an array or a bare string
			// is a 400 from the API on the first call.
			if len(d.InputSchema) == 0 {
				t.Fatal("InputSchema is empty")
			}
			var schema map[string]any
			if err := json.Unmarshal(d.InputSchema, &schema); err != nil {
				t.Fatalf("InputSchema is not valid JSON: %v", err)
			}
			if schema == nil {
				t.Fatal("InputSchema decoded to null, want a JSON object")
			}
			if got := schema["type"]; got != "object" {
				t.Errorf(`InputSchema["type"] = %v, want "object"`, got)
			}
			if props, hasProps := schema["properties"]; hasProps {
				if _, isObject := props.(map[string]any); !isObject {
					t.Errorf(`InputSchema["properties"] is %T, want a JSON object`, props)
				}
			}
			// Anything listed as required must actually be described.
			if req, hasReq := schema["required"]; hasReq {
				list, isList := req.([]any)
				if !isList {
					t.Fatalf(`InputSchema["required"] is %T, want an array`, req)
				}
				props, _ := schema["properties"].(map[string]any)
				for _, name := range list {
					key, isStr := name.(string)
					if !isStr {
						t.Errorf("required entry %v is %T, want a string", name, name)
						continue
					}
					if _, described := props[key]; !described {
						t.Errorf("required field %q is not in properties", key)
					}
				}
			}

			if d.Fn == nil {
				t.Error("Fn is nil: Direct would refuse the call")
			}
			if d.Caps == 0 {
				t.Error("Caps = 0: a tool that declares nothing survives every OnlyCaps filter")
			}
			// WithRetry needs BOTH, so a classifier on a non-idempotent tool
			// never fires. That combination is allowed (bash keeps its
			// classifier as declarative metadata) but the reverse — idempotent
			// with no classifier — is just a missed retry, and the package doc
			// says exactly which tools are in which bucket.
			if !d.Idempotent && d.Retryable != nil && d.Name != "bash" {
				t.Errorf("Retryable is set on a non-idempotent tool; only bash documents that")
			}
		})
	}

	t.Run("registry order is stable", func(t *testing.T) {
		// The order is part of the prompt-cache prefix.
		want := []string{
			"calculator", "http_get", "current_time", "bash", "view_file",
			"grep_search", "git_log", "git_show", "write_file",
			"save_tool_result", "str_replace", "ask_user",
		}
		var got []string
		for _, d := range defs {
			got = append(got, d.Name)
		}
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("Default() order =\n  %v\nwant\n  %v", got, want)
		}
	})

	t.Run("the set is constructible", func(t *testing.T) {
		// NewSet panics on a duplicate; this is the end-to-end version of the
		// uniqueness check above.
		tool.NewSet(defs...)
	})

	t.Run("a read-only filter leaves the documented six", func(t *testing.T) {
		var got []string
		for _, d := range tool.Filter(defs, tool.OnlyCaps(tool.CapReadOnly)) {
			got = append(got, d.Name)
		}
		want := []string{"calculator", "current_time", "view_file", "grep_search", "git_log", "git_show"}
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("OnlyCaps(CapReadOnly) = %v, want %v (the package doc's list)", got, want)
		}
	})

	t.Run("a host with no subprocess hides the exec tools", func(t *testing.T) {
		var got []string
		for _, d := range tool.Filter(defs, tool.Provided(NeedFSReadWriteOnly())) {
			got = append(got, d.Name)
		}
		for _, hidden := range []string{"bash", "grep_search", "git_log", "git_show"} {
			if strings.Contains(strings.Join(got, ","), hidden) {
				t.Errorf("%q survived a host without NeedExec, want it hidden", hidden)
			}
		}
		if !strings.Contains(strings.Join(got, ","), "view_file") {
			t.Errorf("view_file was hidden on an fs-only host, want it kept: got %v", got)
		}
	})
}

// NeedFSReadWriteOnly is a host that offers a filesystem and nothing else.
func NeedFSReadWriteOnly() tool.Need { return tool.NeedFSRead | tool.NeedFSWrite }

func TestDefaultFillsInDependencies(t *testing.T) {
	// Every field nil: Default must substitute working implementations rather
	// than hand back Defs that nil-deref on the first call.
	defs := Default(Deps{})
	byName := map[string]tool.Def{}
	for _, d := range defs {
		byName[d.Name] = d
	}

	out, err := byName["calculator"].Fn(context.Background(), json.RawMessage(`{"expression":"1+1"}`))
	if err != nil || out != "2" {
		t.Errorf("calculator = (%q, %v), want (2, nil)", out, err)
	}

	// http.DefaultClient is deliberately NOT the default: it has no timeout,
	// and a tool that can block forever defeats the run budget. The
	// substituted client's timeout must match http_get's own Def.Timeout, so a
	// caller who forgets the per-call middleware still cannot hang a run.
	t.Run("the default HTTP client's timeout matches http_get's", func(t *testing.T) {
		if http.DefaultClient.Timeout != 0 {
			t.Fatal("http.DefaultClient has a timeout, so this assertion no longer means anything")
		}
		if defaultHTTPTimeout != byName["http_get"].Timeout {
			t.Errorf("defaultHTTPTimeout = %v, want it to match http_get's Def.Timeout %v",
				defaultHTTPTimeout, byName["http_get"].Timeout)
		}
		if defaultHTTPTimeout != httpGetTimeout {
			t.Errorf("defaultHTTPTimeout = %v, want %v", defaultHTTPTimeout, httpGetTimeout)
		}
	})

	t.Run("an injected clock is used", func(t *testing.T) {
		fixed := time.Date(2024, 3, 1, 12, 30, 45, 0, time.UTC)
		defs := Default(Deps{Now: func() time.Time { return fixed }})
		for _, d := range defs {
			if d.Name != "current_time" {
				continue
			}
			out, err := d.Fn(context.Background(), nil)
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if out != "2024-03-01T12:30:45" {
				t.Errorf("current_time = %q, want %q", out, "2024-03-01T12:30:45")
			}
			return
		}
		t.Fatal("current_time is missing from Default()")
	})
}

func TestCurrentTimeRejectsANilClock(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("CurrentTime(nil) did not panic, want a construction-time panic")
		}
	}()
	CurrentTime(nil)
}

// ===== ask_user =====

// TestAskUserFnIsLoud: reaching the handler means the agent loop failed to
// intercept a CapPause tool. Returning nil would strand the run waiting for an
// answer that is never coming.
func TestAskUser(t *testing.T) {
	d := AskUser()

	t.Run("its Fn returns an explanatory error", func(t *testing.T) {
		out, err := d.Fn(context.Background(), json.RawMessage(`{"question":"which branch?"}`))
		if err == nil {
			t.Fatalf("out = %q, want ErrAskUserNotIntercepted", out)
		}
		if err != ErrAskUserNotIntercepted {
			t.Errorf("err = %v, want ErrAskUserNotIntercepted", err)
		}
		if !strings.Contains(err.Error(), "must never be invoked") {
			t.Errorf("err = %q, want it to explain that this is a harness bug", err)
		}
		if out != "" {
			t.Errorf("out = %q, want %q", out, "")
		}
	})

	// Pause is a property of the tool, not of a hard-coded name: the loop finds
	// it with tool.FindPause, which keys on CapPause.
	t.Run("it is discoverable by capability", func(t *testing.T) {
		if !d.Has(tool.CapPause) {
			t.Errorf("Caps = %b, want CapPause", d.Caps)
		}
		set := tool.NewSet(Calculator(), d)
		u, found := tool.FindPause(set, []llm.ToolUse{
			{ID: "u1", Name: "calculator", Input: json.RawMessage(`{"expression":"1"}`)},
			{ID: "u2", Name: AskUserName, Input: json.RawMessage(`{"question":"?"}`)},
		})
		if !found || u.ID != "u2" {
			t.Errorf("FindPause = (%q, %v), want (u2, true)", u.ID, found)
		}
	})

	t.Run("it declares no timeout: the pause is bounded by the user", func(t *testing.T) {
		if d.Timeout != 0 {
			t.Errorf("Timeout = %v, want 0", d.Timeout)
		}
		if d.Name != AskUserName {
			t.Errorf("Name = %q, want %q", d.Name, AskUserName)
		}
	})

	// The description is the tool: it is what teaches the model to emit a real
	// JSON Schema object rather than a JSON-encoded string.
	t.Run("the description warns against a stringified schema", func(t *testing.T) {
		if !strings.Contains(d.Description, "NOT A STRING") {
			t.Error("the description no longer warns against a JSON-encoded schema")
		}
		if !strings.Contains(d.Description, "question") {
			t.Error("the description no longer documents the back-compat question field")
		}
	})
}

// ===== shared helpers =====

func TestTruncate(t *testing.T) {
	// Sizes are realistic on purpose. A clip that cannot shrink its input
	// returns the input — spending a ninety-byte marker to elide three bytes
	// would make the output longer than what it was asked to shorten — so a
	// three-byte limit tests nothing but that fallback.
	tests := []struct {
		name  string
		in    string
		max   int
		wantN int // exact output length, or -1 for "unchanged"
	}{
		{"under the limit", "short", 100, -1},
		{"exactly at the limit", "abcde", 5, -1},
		{"one byte over cannot be shrunk", strings.Repeat("a", 101), 100, -1},
		{"comfortably over", strings.Repeat("a", 1000), 100, 100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := truncate(tt.in, tt.max)
			if tt.wantN < 0 {
				if got != tt.in {
					t.Errorf("truncate(%d bytes, %d) changed it; want it left alone", len(tt.in), tt.max)
				}
				return
			}
			if len(got) > len(tt.in) {
				t.Errorf("truncate(%d bytes, %d) = %d bytes, longer than the input", len(tt.in), tt.max, len(got))
			}
			if !strings.HasPrefix(tt.in, got[:tt.wantN]) {
				t.Error("the kept text is not a prefix of the input")
			}
			if !strings.Contains(got, "truncated") {
				t.Error("no marker")
			}
			if !utf8.ValidString(got) {
				t.Error("not valid UTF-8")
			}
		})
	}

	t.Run("the cut backs off to a rune boundary", func(t *testing.T) {
		// Splitting a multi-byte rune produces invalid UTF-8, and an invalid
		// tool_result is rejected by the provider on the NEXT call — a failure
		// that surfaces nowhere near its cause. Sweep every offset so the check
		// does not depend on a lucky alignment.
		for _, r := range []string{"é", "中", "🙂"} {
			body := strings.Repeat(r, 2000)
			for limit := 100; limit < 100+len(r)*4; limit++ {
				got := truncate(body, limit)
				if !utf8.ValidString(got) {
					t.Errorf("truncate(2000 x %q, %d) is not valid UTF-8", r, limit)
				}
			}
		}
	})
}

func TestMustNotBeNil(t *testing.T) {
	// Discovered on the first line of main rather than three hours into a run.
	defer func() {
		v := recover()
		if v == nil {
			t.Fatal("mustNotBeNil(false, ...) did not panic")
		}
		if msg, isStr := v.(string); !isStr || !strings.Contains(msg, "builtin: needs a thing") {
			t.Errorf("panic = %v, want a string containing %q", v, "builtin: needs a thing")
		}
	}()
	mustNotBeNil(true, "not reached")
	mustNotBeNil(false, "needs a thing")
}

// clipMarkerMax bounds the gap marker tool.Clip inserts. The byte budget is on
// CONTENT, so a capped observation overshoots maxToolOutput by one marker; the
// tests allow for that rather than pretending the cap is exact.
const clipMarkerMax = 200

// TestTruncateKeepsTheTail is the property that head-only truncation loses, and
// it is the one that matters most for a coding agent.
//
// Command output is back-loaded. `go build` prints the compiler errors last, a
// test suite prints the failing assertion after every passing test name, a
// stack trace ends at the throw. A 200 KB failing test run clipped head-only
// hands the model eight kilobytes of test names that PASSED and a note that the
// rest is gone — every byte true, and none of it the answer.
func TestTruncateKeepsTheTail(t *testing.T) {
	const (
		head = "FIRST LINE: go test ./...\n"
		tail = "\n--- FAIL: TestTheOneThatMatters\nFAIL\texit status 1\n"
	)
	big := head + strings.Repeat("--- PASS: TestSomethingElse (0.00s)\n", 6000) + tail

	got := tool.Clip(big, maxToolOutput)

	if len(got) > maxToolOutput+clipMarkerMax {
		t.Errorf("len = %d, want at most %d", len(got), maxToolOutput+clipMarkerMax)
	}
	if !strings.HasPrefix(got, head) {
		t.Errorf("the head was lost; output starts %q", got[:min(len(got), 60)])
	}
	if !strings.HasSuffix(got, tail) {
		t.Errorf("the tail was lost; output ends %q — this is the whole point",
			got[max(0, len(got)-60):])
	}
	if !strings.Contains(got, "omitted from the middle") {
		t.Error("no gap marker; the model would read the two halves as contiguous")
	}
	if !utf8.ValidString(got) {
		t.Error("output is not valid UTF-8")
	}
}

// TestTruncateSplitsOnRuneBoundaries: two cuts now, not one, and the second is
// the easy one to get wrong. Backing the tail's start up would re-admit the
// trailing bytes of a rune whose leading byte is in the omitted middle.
func TestTruncateSplitsOnRuneBoundaries(t *testing.T) {
	for _, r := range []string{"é", "中", "🙂"} {
		body := strings.Repeat(r, 4000)
		for _, limit := range []int{maxToolOutput, 1000, 1001, 999, 128, 129} {
			got := tool.Clip(body, limit)
			if !utf8.ValidString(got) {
				t.Errorf("truncate(%d x %q, %d) is not valid UTF-8", 4000, r, limit)
			}
		}
	}
}
