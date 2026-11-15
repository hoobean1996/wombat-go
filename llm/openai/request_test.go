package openai

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
)

// TestToolResultsFanOutIntoSeveralToolMessages is the shape mismatch this
// package exists for: llm.Message is a role plus blocks (Anthropic's shape),
// but OpenAI pairs results to calls by id and REJECTS a turn that answers
// three tool calls with one message.
func TestToolResultsFanOutIntoSeveralToolMessages(t *testing.T) {
	msgs := []llm.Message{
		llm.UserText("do three things"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.ToolUse{ID: "call_1", Name: "a", Input: json.RawMessage(`{"x":1}`)},
			llm.ToolUse{ID: "call_2", Name: "b"},
			llm.ToolUse{ID: "call_3", Name: "c", Input: json.RawMessage(`{"z":3}`)},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "call_1", Content: "one"},
			llm.ToolResult{ToolUseID: "call_2", Content: "two"},
			llm.ToolResult{ToolUseID: "call_3", Content: "boom", IsError: true},
		}},
	}
	got := wireMessages(t, "be terse", msgs)

	wantRoles := []string{"system", "user", "assistant", "tool", "tool", "tool"}
	if !reflect.DeepEqual(roles(got), wantRoles) {
		t.Fatalf("roles: got %v, want %v", roles(got), wantRoles)
	}
	for i, want := range []struct{ id, content string }{
		{"call_1", "one"}, {"call_2", "two"}, {"call_3", "boom"},
	} {
		m := got[3+i]
		if m.ToolCallID != want.id {
			t.Errorf("tool message %d: tool_call_id got %q, want %q", i, m.ToolCallID, want.id)
		}
		if contentOf(t, m) != want.content {
			t.Errorf("tool message %d: content got %q, want %q", i, contentOf(t, m), want.content)
		}
	}

	// A tool-call-only assistant turn must send content:null, not omit the key
	// — some gateways reject the turn otherwise.
	asst := got[2]
	if asst.Content != nil {
		t.Errorf("assistant content: got %q, want null for a tool-call-only turn", *asst.Content)
	}
	if len(asst.ToolCalls) != 3 {
		t.Fatalf("tool_calls: got %d, want 3", len(asst.ToolCalls))
	}
	if asst.ToolCalls[0].Type != "function" {
		t.Errorf("tool_calls[0].type: got %q, want %q", asst.ToolCalls[0].Type, "function")
	}
	// Arguments are a JSON-encoded STRING, not an object. That double encoding
	// is the API's.
	if got, want := asst.ToolCalls[0].Function.Arguments, `{"x":1}`; got != want {
		t.Errorf("tool_calls[0].function.arguments: got %q, want %q", got, want)
	}
	// A call with no input must still send a legal empty object.
	if got, want := asst.ToolCalls[1].Function.Arguments, `{}`; got != want {
		t.Errorf("tool_calls[1].function.arguments: got %q, want %q", got, want)
	}

	// content:null literally on the wire.
	raw, err := encodeRequest(llm.Request{Messages: msgs}, "m", 100, false)
	if err != nil {
		t.Fatalf("encodeRequest: got error %v, want nil", err)
	}
	if !strings.Contains(string(raw), `"content":null`) {
		t.Errorf(`body has no "content":null for the tool-call-only turn: %s`, raw)
	}
}

// TestTextAfterToolResultsAppendsToTheLastToolMessage looks odd until you try
// the alternative. The API requires an assistant tool_calls turn to be
// followed IMMEDIATELY by its tool messages, so a user message wedged between
// them — which is exactly what the harness produces when it staples a turn
// notice onto a results turn — is a 400. Folding the text into the last result
// keeps the words and keeps the turn legal.
func TestTextAfterToolResultsAppendsToTheLastToolMessage(t *testing.T) {
	tests := []struct {
		name        string
		blocks      []llm.ContentBlock
		wantRoles   []string
		wantContent []string
	}{
		{
			name: "text before results leads as a user message",
			blocks: []llm.ContentBlock{
				llm.Text{Text: "here you go"},
				llm.ToolResult{ToolUseID: "c1", Content: "r1"},
			},
			wantRoles:   []string{"user", "tool"},
			wantContent: []string{"here you go", "r1"},
		},
		{
			name: "text after results is folded into the LAST tool message",
			blocks: []llm.ContentBlock{
				llm.ToolResult{ToolUseID: "c1", Content: "r1"},
				llm.ToolResult{ToolUseID: "c2", Content: "r2"},
				llm.Text{Text: "<budget_status>almost out</budget_status>"},
			},
			wantRoles:   []string{"tool", "tool"},
			wantContent: []string{"r1", "r2\n<budget_status>almost out</budget_status>"},
		},
		{
			name: "several trailing texts all land on the last tool message",
			blocks: []llm.ContentBlock{
				llm.ToolResult{ToolUseID: "c1", Content: "r1"},
				llm.Text{Text: "one"},
				llm.Text{Text: "two"},
			},
			wantRoles:   []string{"tool"},
			wantContent: []string{"r1\none\ntwo"},
		},
		{
			name: "text on both sides",
			blocks: []llm.ContentBlock{
				llm.Text{Text: "before"},
				llm.ToolResult{ToolUseID: "c1", Content: "r1"},
				llm.Text{Text: "after"},
			},
			wantRoles:   []string{"user", "tool"},
			wantContent: []string{"before", "r1\nafter"},
		},
		{
			name: "several leading texts join with a newline",
			blocks: []llm.ContentBlock{
				llm.Text{Text: "a"},
				llm.Text{Text: "b"},
			},
			wantRoles:   []string{"user"},
			wantContent: []string{"a\nb"},
		},
		{
			// A ToolUse in a user turn is a caller bug that Convo.Validate
			// catches upstream; forwarding it would be a 400.
			name: "a stray ToolUse in a user turn is dropped",
			blocks: []llm.ContentBlock{
				llm.Text{Text: "hi"},
				llm.ToolUse{ID: "nope", Name: "x"},
			},
			wantRoles:   []string{"user"},
			wantContent: []string{"hi"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := wireMessages(t, "", []llm.Message{{Role: llm.RoleUser, Content: tt.blocks}})
			if !reflect.DeepEqual(roles(got), tt.wantRoles) {
				t.Fatalf("roles: got %v, want %v", roles(got), tt.wantRoles)
			}
			for i, want := range tt.wantContent {
				if c := contentOf(t, got[i]); c != want {
					t.Errorf("message %d (%s) content: got %q, want %q", i, got[i].Role, c, want)
				}
			}
		})
	}
}

func TestEncodeAssistant(t *testing.T) {
	tests := []struct {
		name        string
		blocks      []llm.ContentBlock
		wantContent *string
		wantCalls   int
	}{
		{
			name:        "text only",
			blocks:      []llm.ContentBlock{llm.Text{Text: "hello"}},
			wantContent: strptr("hello"),
		},
		{
			name:        "two text blocks join with a newline",
			blocks:      []llm.ContentBlock{llm.Text{Text: "a"}, llm.Text{Text: "b"}},
			wantContent: strptr("a\nb"),
		},
		{
			name:      "tool calls only send content:null",
			blocks:    []llm.ContentBlock{llm.ToolUse{ID: "c1", Name: "t"}},
			wantCalls: 1,
		},
		{
			name:        "text plus a call",
			blocks:      []llm.ContentBlock{llm.Text{Text: "let me"}, llm.ToolUse{ID: "c1", Name: "t"}},
			wantContent: strptr("let me"),
			wantCalls:   1,
		},
		{
			// Thinking has nowhere to go: the compat API has no signed-reasoning
			// channel, and inventing a text block for it would feed the model
			// its own scratchpad as if it were an answer. This drop is what
			// makes the decoders' reasoning_content handling safe.
			name:        "thinking is dropped, never round-tripped upstream",
			blocks:      []llm.ContentBlock{llm.Thinking{Text: "secret plan", Signature: "sig"}, llm.Text{Text: "answer"}},
			wantContent: strptr("answer"),
		},
		{
			name:   "thinking alone leaves content null",
			blocks: []llm.ContentBlock{llm.Thinking{Text: "secret plan"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := wireMessages(t, "", []llm.Message{{Role: llm.RoleAssistant, Content: tt.blocks}})
			if len(got) != 1 {
				t.Fatalf("messages: got %d, want 1 — an assistant turn folds into exactly one wire message", len(got))
			}
			m := got[0]
			switch {
			case tt.wantContent == nil && m.Content != nil:
				t.Errorf("content: got %q, want null", *m.Content)
			case tt.wantContent != nil && m.Content == nil:
				t.Errorf("content: got null, want %q", *tt.wantContent)
			case tt.wantContent != nil && *m.Content != *tt.wantContent:
				t.Errorf("content: got %q, want %q", *m.Content, *tt.wantContent)
			}
			if len(m.ToolCalls) != tt.wantCalls {
				t.Errorf("tool_calls: got %d, want %d", len(m.ToolCalls), tt.wantCalls)
			}
			if raw, _ := json.Marshal(m); strings.Contains(string(raw), "secret plan") {
				t.Errorf("the scratchpad travelled back upstream: %s", raw)
			}
		})
	}
}

func TestSystemPromptLeads(t *testing.T) {
	got := wireMessages(t, "be terse", []llm.Message{llm.UserText("hi")})
	if !reflect.DeepEqual(roles(got), []string{"system", "user"}) {
		t.Fatalf("roles: got %v, want [system user]", roles(got))
	}
	if c := contentOf(t, got[0]); c != "be terse" {
		t.Errorf("system content: got %q, want %q", c, "be terse")
	}
	// No system prompt means no leading message at all.
	got = wireMessages(t, "", []llm.Message{llm.UserText("hi")})
	if !reflect.DeepEqual(roles(got), []string{"user"}) {
		t.Errorf("roles with no system prompt: got %v, want [user]", roles(got))
	}
}

func TestEncodeToolChoice(t *testing.T) {
	tests := []struct {
		name    string
		choice  llm.ToolChoice
		want    string // "" means the key is absent
		wantErr string
	}{
		{name: "auto is omitted", choice: llm.ToolChoice{}},
		{name: "any becomes the string required", choice: llm.ToolChoice{Mode: llm.ChoiceAny}, want: `"required"`},
		{name: "none becomes the string none", choice: llm.ToolChoice{Mode: llm.ChoiceNone}, want: `"none"`},
		{
			// OpenAI nests the name one level deeper than Anthropic does.
			name:   "tool nests the name under function",
			choice: llm.ForceTool("calculator"),
			want:   `{"type":"function","function":{"name":"calculator"}}`,
		},
		{name: "tool with no name is rejected", choice: llm.ToolChoice{Mode: llm.ChoiceTool}, wantErr: "needs a name"},
		{name: "an unknown mode is rejected", choice: llm.ToolChoice{Mode: "sometimes"}, wantErr: "unknown tool choice mode"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := encodeRequest(llm.Request{
				Messages: []llm.Message{llm.UserText("hi")},
				Choice:   tt.choice,
			}, "m", 100, false)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("encodeRequest: got nil error, want one containing %q", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("error: got %q, want it to contain %q", err, tt.wantErr)
				}
				if !errors.Is(err, llm.ErrBadRequest) {
					t.Errorf("errors.Is(err, llm.ErrBadRequest): got false, want true (err=%v)", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("encodeRequest: got error %v, want nil", err)
			}
			var doc struct {
				ToolChoice json.RawMessage `json:"tool_choice"`
			}
			decodeBody(t, raw, &doc)
			if tt.want == "" {
				if doc.ToolChoice != nil {
					t.Errorf("tool_choice: got %s, want the key omitted so the body stays byte-identical", doc.ToolChoice)
				}
				return
			}
			if string(doc.ToolChoice) != tt.want {
				t.Errorf("tool_choice: got %s, want %s", doc.ToolChoice, tt.want)
			}
		})
	}
}

func TestEncodeTools(t *testing.T) {
	// Deliberately non-alphabetical keys: a map round trip would sort them,
	// which changes what the model sees and breaks the cache prefix.
	schema := `{"type":"object","properties":{"zulu":{"type":"string"},"alpha":{"type":"number"}},"required":["zulu"]}`
	raw, err := encodeRequest(llm.Request{
		Messages: []llm.Message{llm.UserText("hi")},
		Tools: []llm.ToolSpec{
			{Name: "calculator", Description: "does sums", InputSchema: json.RawMessage(schema)},
			{Name: "bare"},
		},
	}, "m", 100, false)
	if err != nil {
		t.Fatalf("encodeRequest: got error %v, want nil", err)
	}
	if !strings.Contains(string(raw), schema) {
		t.Errorf("tool schema was rewritten.\n got: %s\nwant it to contain verbatim: %s", raw, schema)
	}

	var doc struct {
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	decodeBody(t, raw, &doc)
	if len(doc.Tools) != 2 {
		t.Fatalf("tools: got %d, want 2", len(doc.Tools))
	}
	if doc.Tools[0].Type != "function" {
		t.Errorf("tools[0].type: got %q, want %q", doc.Tools[0].Type, "function")
	}
	if doc.Tools[0].Function.Name != "calculator" {
		t.Errorf("tools[0].function.name: got %q, want %q", doc.Tools[0].Function.Name, "calculator")
	}
	// null or an absent parameters key is a 400.
	if got, want := string(doc.Tools[1].Function.Parameters), string(emptySchema); got != want {
		t.Errorf("tools[1].function.parameters: got %s, want %s", got, want)
	}
}

func TestNoToolsKeyWhenThereAreNoTools(t *testing.T) {
	raw, err := encodeRequest(llm.Request{Messages: []llm.Message{llm.UserText("hi")}}, "m", 100, false)
	if err != nil {
		t.Fatalf("encodeRequest: got error %v, want nil", err)
	}
	if strings.Contains(string(raw), `"tools"`) {
		t.Errorf(`body carries a "tools" key with no tools: %s`, raw)
	}
}

func TestEncodeRequestRejectsAnEmptyConversation(t *testing.T) {
	_, err := encodeRequest(llm.Request{}, "m", 100, false)
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("errors.Is(err, llm.ErrBadRequest): got false, want true (err=%v)", err)
	}
	// A system prompt alone is enough to make the conversation non-empty,
	// because it becomes a message here.
	if _, err := encodeRequest(llm.Request{System: "hi"}, "m", 100, false); err != nil {
		t.Errorf("encodeRequest with only a system prompt: got error %v, want nil", err)
	}
}

// TestEncodeIsDeterministic is the prompt-cache invariant: identical requests
// must produce identical bytes or the prefix match never hits.
func TestEncodeIsDeterministic(t *testing.T) {
	req := llm.Request{
		System:   "sys",
		Messages: []llm.Message{llm.UserText("q")},
		Tools: []llm.ToolSpec{
			{Name: "a", InputSchema: json.RawMessage(`{"type":"object","properties":{"z":{},"a":{}}}`)},
			{Name: "b"},
		},
		Choice: llm.ForceTool("a"),
	}
	first, err := encodeRequest(req, "m", 100, true)
	if err != nil {
		t.Fatalf("encodeRequest: got error %v, want nil", err)
	}
	for i := range 20 {
		again, err := encodeRequest(req, "m", 100, true)
		if err != nil {
			t.Fatalf("encodeRequest: got error %v, want nil", err)
		}
		if string(first) != string(again) {
			t.Fatalf("encode is not deterministic (run %d):\n got %s\nwant %s", i, again, first)
		}
	}
}

// TestInvalidUTF8SurvivesEncoding: encoding/json replaces invalid bytes with
// U+FFFD when it writes a string, so mis-encoded tool output cannot produce a
// body the provider rejects. Unlike llm/anthropic there is no scrubbing pass
// here, and this is why one is not needed.
//
// The assertions are about meaning, not spelling. Which bytes encoding/json
// uses for the replacement is its own business and has changed between Go
// releases: older ones wrote the six-byte escape \ufffd, current ones write
// the rune's raw UTF-8 (ef bf bd), and both are the same JSON string. What
// this package depends on is that the body is valid, that no raw invalid byte
// survives, and that the provider decodes U+FFFD where the bad bytes were.
func TestInvalidUTF8SurvivesEncoding(t *testing.T) {
	raw, err := encodeRequest(llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "c1", Content: "bytes: \xff\xfe"},
		}}},
	}, "m", 100, false)
	if err != nil {
		t.Fatalf("encodeRequest: got error %v, want nil", err)
	}
	if !json.Valid(raw) {
		t.Fatalf("body is not valid JSON: %s", raw)
	}
	// A byte search, deliberately. strings.ContainsRune(s, 0xff) looks for the
	// rune U+00FF (the two bytes c3 bf) and would pass with the raw 0xff
	// still sitting in the body.
	for _, b := range []byte{0xff, 0xfe} {
		if strings.IndexByte(string(raw), b) >= 0 {
			t.Errorf("body still carries the invalid byte %#x: %q", b, raw)
		}
	}

	var body struct {
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if len(body.Messages) != 1 {
		t.Fatalf("body has %d messages, want 1: %s", len(body.Messages), raw)
	}
	if got, want := body.Messages[0].Content, "bytes: \ufffd\ufffd"; got != want {
		t.Errorf("decoded content = %q, want %q (each invalid byte replaced with U+FFFD)", got, want)
	}
}

func TestEncodeMedia(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\n")
	raw, err := encodeRequest(llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.Text{Text: "look"},
			llm.Media{MediaType: "image/png", Data: png},
		}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.ToolUse{ID: "u1", Name: "view_media", Input: json.RawMessage(`{}`)},
			llm.ToolUse{ID: "u2", Name: "grep", Input: json.RawMessage(`{}`)},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "u1", Content: "Attached spec.pdf", Media: []llm.Media{{MediaType: "application/pdf", Data: []byte("%PDF"), Name: "spec.pdf"}}},
			llm.ToolResult{ToolUseID: "u2", Content: "no match"},
		}},
	}}, "m", 100, false)
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Messages []struct {
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			ToolCallID string          `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	var roles []string
	for _, m := range body.Messages {
		roles = append(roles, m.Role)
	}
	// The tool messages stay contiguous after the assistant turn; the media
	// that rode on a result follows all of them.
	if want := []string{"user", "assistant", "tool", "tool", "user"}; strings.Join(roles, ",") != strings.Join(want, ",") {
		t.Fatalf("roles = %v, want %v\n%s", roles, want, raw)
	}
	if got := string(body.Messages[0].Content); got != `[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]` {
		t.Errorf("user media message = %s", got)
	}
	if got := string(body.Messages[2].Content); got != `"Attached spec.pdf"` {
		t.Errorf("a tool message must stay text: %s", got)
	}
	trailing := string(body.Messages[4].Content)
	for _, want := range []string{`Attached to the result of tool call u1:`, `{"type":"file","file":{"filename":"spec.pdf","file_data":"data:application/pdf;base64,JVBERg=="}}`} {
		if !strings.Contains(trailing, want) {
			t.Errorf("trailing media message is missing %s:\n%s", want, trailing)
		}
	}
}

// A text-only conversation encodes exactly as before media existed: plain
// string content, which is what every text-only gateway accepts.
func TestEncodeTextOnlyKeepsStringContent(t *testing.T) {
	raw, err := encodeRequest(llm.Request{Messages: []llm.Message{llm.UserText("hi")}}, "m", 100, false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"messages":[{"role":"user","content":"hi"}]`) {
		t.Errorf("body = %s", raw)
	}
}
