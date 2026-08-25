// White-box (package llm) on purpose: the ContentBlock set is closed by an
// unexported method, so the "unknown block" branch of Message.MarshalJSON is
// unreachable from an external test package. Testing it needs a type declared
// in this package.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// ===== Message JSON =====

func TestMessageJSONRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		msg  Message
		want string // exact wire bytes
	}{
		{
			name: "text",
			msg:  Message{Role: RoleUser, Content: []ContentBlock{Text{Text: "hello"}}},
			want: `{"role":"user","content":[{"type":"text","text":"hello"}]}`,
		},
		{
			name: "empty text still carries its type",
			msg:  Message{Role: RoleUser, Content: []ContentBlock{Text{}}},
			want: `{"role":"user","content":[{"type":"text"}]}`,
		},
		{
			name: "tool_use",
			msg: Message{Role: RoleAssistant, Content: []ContentBlock{
				ToolUse{ID: "tu_1", Name: "calculator", Input: json.RawMessage(`{"expression":"2+2"}`)},
			}},
			want: `{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"calculator","input":{"expression":"2+2"}}]}`,
		},
		{
			name: "tool_use with no input",
			msg: Message{Role: RoleAssistant, Content: []ContentBlock{
				ToolUse{ID: "tu_2", Name: "noargs"},
			}},
			want: `{"role":"assistant","content":[{"type":"tool_use","id":"tu_2","name":"noargs"}]}`,
		},
		{
			name: "tool_result",
			msg: Message{Role: RoleUser, Content: []ContentBlock{
				ToolResult{ToolUseID: "tu_1", Content: "4"},
			}},
			want: `{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"4"}]}`,
		},
		{
			name: "tool_result error",
			msg: Message{Role: RoleUser, Content: []ContentBlock{
				ToolResult{ToolUseID: "tu_1", Content: "boom", IsError: true},
			}},
			want: `{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"boom","is_error":true}]}`,
		},
		{
			name: "thinking keeps its signature",
			msg: Message{Role: RoleAssistant, Content: []ContentBlock{
				Thinking{Text: "let me think", Signature: "sig-abc"},
			}},
			want: `{"role":"assistant","content":[{"type":"thinking","text":"let me think","signature":"sig-abc"}]}`,
		},
		{
			name: "every block type in one message",
			msg: Message{Role: RoleAssistant, Content: []ContentBlock{
				Thinking{Text: "hmm", Signature: "s"},
				Text{Text: "here goes"},
				ToolUse{ID: "u", Name: "t", Input: json.RawMessage(`{}`)},
				ToolResult{ToolUseID: "u", Content: "r", IsError: true},
			}},
			want: `{"role":"assistant","content":[` +
				`{"type":"thinking","text":"hmm","signature":"s"},` +
				`{"type":"text","text":"here goes"},` +
				`{"type":"tool_use","id":"u","name":"t","input":{}},` +
				`{"type":"tool_result","tool_use_id":"u","content":"r","is_error":true}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(tt.msg)
			if err != nil {
				t.Fatalf("Marshal: got error %v, want nil", err)
			}
			if string(got) != tt.want {
				t.Errorf("Marshal:\ngot  %s\nwant %s", got, tt.want)
			}

			var back Message
			if err := json.Unmarshal(got, &back); err != nil {
				t.Fatalf("Unmarshal: got error %v, want nil", err)
			}
			if back.Role != tt.msg.Role {
				t.Errorf("role: got %q, want %q", back.Role, tt.msg.Role)
			}
			if len(back.Content) != len(tt.msg.Content) {
				t.Fatalf("block count: got %d, want %d", len(back.Content), len(tt.msg.Content))
			}
			// Re-marshalling the decoded value must produce the same bytes; that
			// is the property a persisted transcript actually depends on.
			again, err := json.Marshal(back)
			if err != nil {
				t.Fatalf("re-Marshal: got error %v, want nil", err)
			}
			if string(again) != tt.want {
				t.Errorf("round trip is not byte-stable:\ngot  %s\nwant %s", again, tt.want)
			}
		})
	}
}

func TestMessageMarshalEmptyContentIsArrayNotNull(t *testing.T) {
	// Providers reject "content": null. A message with no blocks must still
	// encode as an empty array.
	got, err := json.Marshal(Message{Role: RoleUser})
	if err != nil {
		t.Fatalf("Marshal: got error %v, want nil", err)
	}
	if want := `{"role":"user","content":[]}`; string(got) != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestMessageUnmarshalContentIsNeverNil(t *testing.T) {
	var m Message
	if err := json.Unmarshal([]byte(`{"role":"user","content":null}`), &m); err != nil {
		t.Fatalf("Unmarshal: got error %v, want nil", err)
	}
	if m.Content == nil {
		t.Error("Content: got nil, want non-nil empty slice")
	}
	if len(m.Content) != 0 {
		t.Errorf("len(Content): got %d, want 0", len(m.Content))
	}
}

func TestMessageMarshalPreservesToolInputByteForByte(t *testing.T) {
	// ToolUse.Input is json.RawMessage precisely so that key order survives.
	// Round-tripping through a map would sort the keys and change the bytes the
	// model (and the prompt cache) sees.
	in := json.RawMessage(`{"zebra":1,"alpha":2,"middle":[3,{"b":1,"a":2}]}`)
	raw, err := json.Marshal(Message{Role: RoleAssistant, Content: []ContentBlock{
		ToolUse{ID: "u", Name: "t", Input: in},
	}})
	if err != nil {
		t.Fatalf("Marshal: got error %v, want nil", err)
	}
	if !strings.Contains(string(raw), `"input":`+string(in)) {
		t.Errorf("input was reordered:\ngot  %s\nwant substring %q", raw, `"input":`+string(in))
	}

	var back Message
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("Unmarshal: got error %v, want nil", err)
	}
	tu, ok := back.Content[0].(ToolUse)
	if !ok {
		t.Fatalf("block type: got %T, want ToolUse", back.Content[0])
	}
	if string(tu.Input) != string(in) {
		t.Errorf("Input: got %s, want %s", tu.Input, in)
	}
}

func TestMessageUnmarshalRejectsUnknownBlockType(t *testing.T) {
	tests := []struct {
		name string
		in   string
	}{
		{"provider-specific type we do not model", `{"role":"user","content":[{"type":"image","text":"x"}]}`},
		{"missing type", `{"role":"user","content":[{"text":"x"}]}`},
		{"empty type", `{"role":"user","content":[{"type":"","text":"x"}]}`},
		{"unknown type after a valid one", `{"role":"user","content":[{"type":"text","text":"a"},{"type":"redacted_thinking"}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var m Message
			err := json.Unmarshal([]byte(tt.in), &m)
			if err == nil {
				t.Fatalf("Unmarshal(%s): got nil error, want an unknown-block-type error", tt.in)
			}
			if !strings.Contains(err.Error(), "unknown content block type") {
				t.Errorf("error: got %q, want it to mention \"unknown content block type\"", err)
			}
		})
	}
}

// unknownBlock is a ContentBlock this package knows nothing about. It can only
// exist in a white-box test, which is why this file is `package llm`.
type unknownBlock struct{}

func (unknownBlock) blockKind() string { return "unknown" }

func TestMessageMarshalRejectsUnknownBlockType(t *testing.T) {
	_, err := json.Marshal(Message{Role: RoleUser, Content: []ContentBlock{unknownBlock{}}})
	if err == nil {
		t.Fatal("Marshal: got nil error, want an unknown-content-block error")
	}
	if !strings.Contains(err.Error(), "unknown content block") {
		t.Errorf("error: got %q, want it to mention \"unknown content block\"", err)
	}
}

func TestMessageUnmarshalRejectsMalformedJSON(t *testing.T) {
	tests := []struct {
		name string
		in   string
	}{
		{"truncated object", `{"role":`},
		{"role is not a string", `{"role":123,"content":[]}`},
		{"content is not an array", `{"role":"user","content":"hi"}`},
		{"a block is not an object", `{"role":"user","content":["hi"]}`},
		{"not an object at all", `[]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var m Message
			// Called directly rather than through json.Unmarshal: the encoding
			// layer rejects a syntax error before it ever reaches
			// Message.UnmarshalJSON, so the decode-failure branch is only
			// reachable this way.
			if err := m.UnmarshalJSON([]byte(tt.in)); err == nil {
				t.Errorf("UnmarshalJSON(%s): got nil error, want a decode error", tt.in)
			}
			// And it must also fail through the normal path.
			var m2 Message
			if err := json.Unmarshal([]byte(tt.in), &m2); err == nil {
				t.Errorf("json.Unmarshal(%s): got nil error, want a decode error", tt.in)
			}
		})
	}
}

// ===== TextOf / ToolUses =====

func TestTextOf(t *testing.T) {
	tests := []struct {
		name   string
		blocks []ContentBlock
		want   string
	}{
		{"nil", nil, ""},
		{"empty", []ContentBlock{}, ""},
		{"single", []ContentBlock{Text{Text: "hello"}}, "hello"},
		{
			name:   "two spans are newline separated",
			blocks: []ContentBlock{Text{Text: "a"}, Text{Text: "b"}},
			want:   "a\nb",
		},
		{
			name:   "thinking is excluded: it is not the answer",
			blocks: []ContentBlock{Thinking{Text: "secret scratchpad", Signature: "s"}, Text{Text: "the answer"}},
			want:   "the answer",
		},
		{
			name:   "tool blocks are excluded",
			blocks: []ContentBlock{ToolUse{ID: "u", Name: "t"}, Text{Text: "x"}, ToolResult{ToolUseID: "u", Content: "r"}},
			want:   "x",
		},
		{
			name:   "only non-text blocks yields empty",
			blocks: []ContentBlock{ToolUse{ID: "u", Name: "t"}, Thinking{Text: "t"}},
			want:   "",
		},
		{
			name:   "a leading empty span does not introduce a newline",
			blocks: []ContentBlock{Text{Text: ""}, Text{Text: "b"}},
			want:   "b",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := TextOf(tt.blocks); got != tt.want {
				t.Errorf("TextOf: got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestToolUses(t *testing.T) {
	t.Run("returns every use in order", func(t *testing.T) {
		blocks := []ContentBlock{
			Text{Text: "first"},
			ToolUse{ID: "a", Name: "one"},
			Thinking{Text: "t"},
			ToolUse{ID: "b", Name: "two"},
			ToolResult{ToolUseID: "a", Content: "r"},
		}
		got := ToolUses(blocks)
		if len(got) != 2 {
			t.Fatalf("count: got %d, want 2", len(got))
		}
		if got[0].ID != "a" || got[1].ID != "b" {
			t.Errorf("order: got [%s %s], want [a b]", got[0].ID, got[1].ID)
		}
	})

	t.Run("no uses yields nil", func(t *testing.T) {
		if got := ToolUses([]ContentBlock{Text{Text: "x"}}); got != nil {
			t.Errorf("got %v, want nil", got)
		}
		if got := ToolUses(nil); got != nil {
			t.Errorf("got %v, want nil", got)
		}
	})
}

func TestUserText(t *testing.T) {
	m := UserText("hi")
	if m.Role != RoleUser {
		t.Errorf("role: got %q, want %q", m.Role, RoleUser)
	}
	if got := TextOf(m.Content); got != "hi" {
		t.Errorf("text: got %q, want %q", got, "hi")
	}
}

func TestUsageAdd(t *testing.T) {
	u := Usage{InputTokens: 1, OutputTokens: 2, CacheWriteTokens: 3, CacheReadTokens: 4}
	u.Add(Usage{InputTokens: 10, OutputTokens: 20, CacheWriteTokens: 30, CacheReadTokens: 40})
	want := Usage{InputTokens: 11, OutputTokens: 22, CacheWriteTokens: 33, CacheReadTokens: 44}
	if u != want {
		t.Errorf("got %+v, want %+v", u, want)
	}
}

func TestToolUseIDString(t *testing.T) {
	if got := ToolUseID("tu_1").String(); got != "tu_1" {
		t.Errorf("got %q, want %q", got, "tu_1")
	}
}

func TestForceTool(t *testing.T) {
	got := ForceTool("calculator")
	want := ToolChoice{Mode: ChoiceTool, Name: "calculator"}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

// ===== StreamMode =====

func TestStreamModeEnabled(t *testing.T) {
	tests := []struct {
		mode    StreamMode
		hasSink bool
		want    bool
	}{
		{StreamAuto, false, false},
		{StreamAuto, true, true},
		{StreamAlways, false, true},
		{StreamAlways, true, true},
		{StreamNever, false, false},
		{StreamNever, true, false},
		// An out-of-range value must behave like the zero value rather than
		// silently forcing a mode.
		{StreamMode(99), true, true},
		{StreamMode(99), false, false},
	}
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			if got := tt.mode.Enabled(tt.hasSink); got != tt.want {
				t.Errorf("StreamMode(%d).Enabled(%v): got %v, want %v", tt.mode, tt.hasSink, got, tt.want)
			}
		})
	}
}

func TestStreamAutoIsTheZeroValue(t *testing.T) {
	var m StreamMode
	if m != StreamAuto {
		t.Errorf("zero StreamMode: got %d, want StreamAuto (%d)", m, StreamAuto)
	}
}

// ===== Chain =====

// recorder is a Middleware that appends to a shared slice on the way in and on
// the way out, so the nesting order is observable.
func recorder(log *[]string, name string) Middleware {
	return func(next Client) Client {
		return ClientFunc(func(ctx context.Context, req Request) (Response, error) {
			*log = append(*log, "enter:"+name)
			resp, err := next.Complete(ctx, req)
			*log = append(*log, "exit:"+name)
			return resp, err
		})
	}
}

func TestChainOrdering(t *testing.T) {
	// The documented contract: later entries end up further OUT, so
	// Chain(leaf, retry, logging) has logging see the post-retry verdict.
	var log []string
	leaf := ClientFunc(func(context.Context, Request) (Response, error) {
		log = append(log, "leaf")
		return Response{StopReason: StopEndTurn}, nil
	})

	c := Chain(leaf, recorder(&log, "inner"), recorder(&log, "middle"), recorder(&log, "outer"))
	if _, err := c.Complete(context.Background(), Request{}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}

	want := []string{
		"enter:outer", "enter:middle", "enter:inner",
		"leaf",
		"exit:inner", "exit:middle", "exit:outer",
	}
	if strings.Join(log, ",") != strings.Join(want, ",") {
		t.Errorf("call order:\ngot  %v\nwant %v", log, want)
	}
}

func TestChainWithNoMiddlewareReturnsBase(t *testing.T) {
	leaf := ClientFunc(func(context.Context, Request) (Response, error) { return Response{}, nil })
	got := Chain(leaf)
	if _, err := got.Complete(context.Background(), Request{}); err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
}

func TestChainPropagatesErrorOutward(t *testing.T) {
	sentinel := errors.New("leaf exploded")
	var log []string
	leaf := ClientFunc(func(context.Context, Request) (Response, error) { return Response{}, sentinel })
	c := Chain(leaf, recorder(&log, "inner"), recorder(&log, "outer"))

	_, err := c.Complete(context.Background(), Request{})
	if !errors.Is(err, sentinel) {
		t.Errorf("error: got %v, want %v", err, sentinel)
	}
	// Both layers must still run their deferred half.
	if len(log) != 4 {
		t.Errorf("both layers should have unwound: got %v, want 4 entries", log)
	}
}

func TestClientFuncImplementsClient(t *testing.T) {
	var c Client = ClientFunc(func(_ context.Context, req Request) (Response, error) {
		return Response{Model: req.Model}, nil
	})
	resp, err := c.Complete(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatalf("Complete: got error %v, want nil", err)
	}
	if resp.Model != "m" {
		t.Errorf("Model: got %q, want %q", resp.Model, "m")
	}
}
