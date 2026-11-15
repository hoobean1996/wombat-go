package anthropic

import (
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
)

// encodeFor builds a body with the given config, without any HTTP.
func encodeFor(t *testing.T, cfg Config, req llm.Request) []byte {
	t.Helper()
	clearEnv(t)
	if cfg.APIKey == "" {
		cfg.APIKey = "k"
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: got error %v, want nil", err)
	}
	body, err := c.encode(req, "m", 1024, false)
	if err != nil {
		t.Fatalf("encode: got error %v, want nil", err)
	}
	return body
}

func toolSpec(name string) llm.ToolSpec {
	return llm.ToolSpec{
		Name:        name,
		Description: "does " + name,
		InputSchema: json.RawMessage(`{"type":"object","properties":{"zebra":{"type":"string"},"alpha":{"type":"number"}}}`),
	}
}

// TestCacheBreakpointPlacement is the expensive-to-get-wrong one.
//
// Three breakpoints, in exactly three places: the system prompt, the LAST
// tool, and the LAST content block of the LAST message. Getting this wrong
// does not fail a call — it silently doubles the bill, because the cache is a
// prefix match and a breakpoint on an earlier tool leaves everything after it
// outside the cached prefix. That is why this test asserts POSITIONS and a
// total count, not merely "caching is on somewhere".
func TestCacheBreakpointPlacement(t *testing.T) {
	req := llm.Request{
		System: "you are terse",
		Tools:  []llm.ToolSpec{toolSpec("alpha"), toolSpec("beta"), toolSpec("gamma")},
		Messages: []llm.Message{
			llm.UserText("q1"),
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.Text{Text: "thinking out loud"},
				llm.ToolUse{ID: "tu_1", Name: "alpha", Input: json.RawMessage(`{"x":1}`)},
			}},
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				llm.ToolResult{ToolUseID: "tu_1", Content: "42"},
				llm.Text{Text: "and now?"},
			}},
		},
	}
	body := encodeFor(t, Config{}, req)

	want := []string{
		"system[0]",
		"tools[2]",               // the END of the tool list, never an earlier one
		"messages[2].content[1]", // the last block of the last message
	}
	got := cachePaths(t, body)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("cache_control breakpoints: got %v, want %v\nbody: %s", got, want, body)
	}
	if n := strings.Count(string(body), `"cache_control"`); n != 3 {
		t.Errorf("cache_control occurrences: got %d, want 3 (the four-breakpoint budget must not be burned)", n)
	}
}

func TestCacheBreakpointEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		req  llm.Request
		want []string
	}{
		{
			name: "no system prompt drops that breakpoint",
			req: llm.Request{
				Tools:    []llm.ToolSpec{toolSpec("only")},
				Messages: []llm.Message{llm.UserText("q")},
			},
			want: []string{"tools[0]", "messages[0].content[0]"},
		},
		{
			name: "no tools drops that breakpoint",
			req: llm.Request{
				System:   "sys",
				Messages: []llm.Message{llm.UserText("q")},
			},
			want: []string{"system[0]", "messages[0].content[0]"},
		},
		{
			// The API rejects cache_control on a thinking block, so the moving
			// breakpoint is dropped rather than risking a 400.
			name: "a trailing thinking block takes no breakpoint",
			req: llm.Request{
				System: "sys",
				Messages: []llm.Message{
					llm.UserText("q"),
					{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
						llm.Text{Text: "answer"},
						llm.Thinking{Text: "hmm", Signature: "sig"},
					}},
				},
			},
			want: []string{"system[0]"},
		},
		{
			name: "an empty trailing message takes no breakpoint",
			req: llm.Request{
				System: "sys",
				Messages: []llm.Message{
					llm.UserText("q"),
					{Role: llm.RoleAssistant, Content: nil},
				},
			},
			want: []string{"system[0]"},
		},
		{
			name: "no messages at all",
			req:  llm.Request{System: "sys"},
			want: []string{"system[0]"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := encodeFor(t, Config{}, tt.req)
			got := cachePaths(t, body)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("cache_control breakpoints: got %v, want %v\nbody: %s", got, tt.want, body)
			}
		})
	}
}

// TestContextManagementGating pins the block to the beta header that is
// ACTUALLY SENT. Sending context_management without the beta enabled is
// "Extra inputs are not permitted" — a 400 on every single request — so an
// ExtraHeaders override that drops the beta has to take the block with it.
func TestContextManagementGating(t *testing.T) {
	msgs := []llm.Message{llm.UserText("q")}
	tools := []llm.ToolSpec{toolSpec("alpha")}

	tests := []struct {
		name  string
		cfg   Config
		tools []llm.ToolSpec
		want  bool
	}{
		{
			name:  "beta on, tools present",
			cfg:   Config{Beta: []string{BetaContextManagement}},
			tools: tools,
			want:  true,
		},
		{
			name:  "beta with surrounding space still counts",
			cfg:   Config{Beta: []string{"other", "  " + BetaContextManagement + "  "}},
			tools: tools,
			want:  true,
		},
		{
			name:  "no beta, no block",
			cfg:   Config{},
			tools: tools,
			want:  false,
		},
		{
			name:  "some other beta, no block",
			cfg:   Config{Beta: []string{"prompt-caching-2024-07-31"}},
			tools: tools,
			want:  false,
		},
		{
			// The edit prunes tool_use blocks; with no tools there is nothing to
			// prune, and the block is omitted.
			name:  "beta on but no tools, no block",
			cfg:   Config{Beta: []string{BetaContextManagement}},
			tools: nil,
			want:  false,
		},
		{
			// THE regression case. Config.Beta asks for context management, but
			// an ExtraHeaders override replaces the header wholesale and drops
			// it — so the block must vanish, or every request 400s.
			name: "ExtraHeaders override that drops the beta drops the block",
			cfg: Config{
				Beta:         []string{BetaContextManagement},
				ExtraHeaders: map[string]string{"anthropic-beta": "something-else"},
			},
			tools: tools,
			want:  false,
		},
		{
			// And the mirror: an override that CARRIES the beta keeps the block
			// even though Config.Beta never mentioned it.
			name: "ExtraHeaders override that carries the beta keeps the block",
			cfg: Config{
				ExtraHeaders: map[string]string{"Anthropic-Beta": "other, " + BetaContextManagement},
			},
			tools: tools,
			want:  true,
		},
		{
			name: "an unrelated ExtraHeaders entry does not disturb the gate",
			cfg: Config{
				Beta:         []string{BetaContextManagement},
				ExtraHeaders: map[string]string{"X-Route": "team-a"},
			},
			tools: tools,
			want:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := encodeFor(t, tt.cfg, llm.Request{Messages: msgs, Tools: tt.tools})
			var doc struct {
				ContextManagement *contextManagement `json:"context_management"`
			}
			decodeBody(t, body, &doc)
			if got := doc.ContextManagement != nil; got != tt.want {
				t.Fatalf("context_management present: got %v, want %v\nbody: %s", got, tt.want, body)
			}
			if !tt.want {
				return
			}
			want := &contextManagement{Edits: []contextEdit{{
				Type:         clearToolUsesType,
				Trigger:      &cmThreshold{Type: inputTokensThreshold, Value: clearTriggerTokens},
				Keep:         &cmThreshold{Type: toolUsesThresholdKind, Value: clearKeepToolUses},
				ClearAtLeast: &cmThreshold{Type: inputTokensThreshold, Value: clearAtLeastTokens},
			}}}
			if !reflect.DeepEqual(doc.ContextManagement, want) {
				t.Errorf("context_management: got %+v, want %+v", doc.ContextManagement, want)
			}
		})
	}
}

// TestToolSchemaIsPassedThroughVerbatim guards the "structs, never maps" rule:
// a map round trip sorts keys, which changes the bytes the model sees AND
// invalidates the cached prefix.
func TestToolSchemaIsPassedThroughVerbatim(t *testing.T) {
	schema := `{"type":"object","properties":{"zulu":{"type":"string"},"alpha":{"type":"number"}},"required":["zulu"]}`
	body := encodeFor(t, Config{}, llm.Request{
		Messages: []llm.Message{llm.UserText("q")},
		Tools:    []llm.ToolSpec{{Name: "t", InputSchema: json.RawMessage(schema)}},
	})
	if !strings.Contains(string(body), schema) {
		t.Errorf("tool schema was rewritten.\n got body: %s\nwant it to contain verbatim: %s", body, schema)
	}
}

func TestEncodeToolDefaults(t *testing.T) {
	body := encodeFor(t, Config{}, llm.Request{
		Messages: []llm.Message{llm.UserText("q")},
		Tools:    []llm.ToolSpec{{Name: "bare"}},
	})
	var doc struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description *string         `json:"description"`
			InputSchema json.RawMessage `json:"input_schema"`
		} `json:"tools"`
	}
	decodeBody(t, body, &doc)
	if len(doc.Tools) != 1 {
		t.Fatalf("tools: got %d, want 1", len(doc.Tools))
	}
	if got, want := string(doc.Tools[0].InputSchema), string(emptySchema); got != want {
		t.Errorf("input_schema for a tool with none: got %s, want %s (null is a 400)", got, want)
	}
	if doc.Tools[0].Description != nil {
		t.Errorf("description: got %q, want it omitted when empty", *doc.Tools[0].Description)
	}
}

func TestEncodeBlocks(t *testing.T) {
	body := encodeFor(t, Config{}, llm.Request{
		Messages: []llm.Message{
			llm.UserText("q"),
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.Thinking{Text: "let me think", Signature: "sig-abc"},
				llm.Text{Text: "hello"},
				llm.ToolUse{ID: "tu_1", Name: "alpha", Input: json.RawMessage(`{"x":1}`)},
				llm.ToolUse{ID: "tu_2", Name: "beta"}, // no input at all
			}},
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				llm.ToolResult{ToolUseID: "tu_1", Content: "ok"},
				llm.ToolResult{ToolUseID: "tu_2", Content: "boom", IsError: true},
			}},
		},
	})

	var doc struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	decodeBody(t, body, &doc)

	if len(doc.Messages) != 3 {
		t.Fatalf("messages: got %d, want 3", len(doc.Messages))
	}
	if got, want := doc.Messages[1].Role, "assistant"; got != want {
		t.Errorf("role: got %q, want %q", got, want)
	}

	blocks := doc.Messages[1].Content
	if got, want := len(blocks), 4; got != want {
		t.Fatalf("assistant blocks: got %d, want %d", got, want)
	}
	if got, want := blocks[0]["type"], "thinking"; got != want {
		t.Errorf("block 0 type: got %v, want %v", got, want)
	}
	// The signature must survive the round trip or the next turn is rejected.
	if got, want := blocks[0]["signature"], "sig-abc"; got != want {
		t.Errorf("thinking signature: got %v, want %v", got, want)
	}
	if got, want := blocks[3]["type"], "tool_use"; got != want {
		t.Errorf("block 3 type: got %v, want %v", got, want)
	}
	// A nil input would marshal as null, which the API rejects.
	in, ok := blocks[3]["input"].(map[string]any)
	if !ok || len(in) != 0 {
		t.Errorf("tool_use input with no arguments: got %v, want an empty object", blocks[3]["input"])
	}

	results := doc.Messages[2].Content
	if got, want := results[0]["tool_use_id"], "tu_1"; got != want {
		t.Errorf("tool_use_id: got %v, want %v", got, want)
	}
	if _, present := results[0]["is_error"]; present {
		t.Errorf("is_error on a successful result: got present, want omitted")
	}
	if got, want := results[1]["is_error"], true; got != want {
		t.Errorf("is_error: got %v, want %v", got, want)
	}
}

func TestEncodeToolChoice(t *testing.T) {
	tests := []struct {
		name   string
		choice llm.ToolChoice
		want   string // "" means the key must be absent
	}{
		{name: "auto is omitted", choice: llm.ToolChoice{}, want: ""},
		{name: "any", choice: llm.ToolChoice{Mode: llm.ChoiceAny}, want: `{"type":"any"}`},
		{name: "none", choice: llm.ToolChoice{Mode: llm.ChoiceNone}, want: `{"type":"none"}`},
		{name: "tool", choice: llm.ForceTool("alpha"), want: `{"type":"tool","name":"alpha"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := encodeFor(t, Config{}, llm.Request{
				Messages: []llm.Message{llm.UserText("q")},
				Choice:   tt.choice,
			})
			var doc struct {
				ToolChoice json.RawMessage `json:"tool_choice"`
			}
			decodeBody(t, body, &doc)
			got := string(doc.ToolChoice)
			if tt.want == "" {
				if doc.ToolChoice != nil {
					t.Errorf("tool_choice: got %s, want the key omitted so the body stays byte-identical", got)
				}
				return
			}
			if got != tt.want {
				t.Errorf("tool_choice: got %s, want %s", got, tt.want)
			}
		})
	}
}

func TestEncodeOmitsEmptySystem(t *testing.T) {
	body := encodeFor(t, Config{}, llm.Request{Messages: []llm.Message{llm.UserText("q")}})
	if strings.Contains(string(body), `"system"`) {
		t.Errorf(`body carries a "system" key with no system prompt: %s`, body)
	}
}

// TestEncodeIsDeterministic is the prompt-cache invariant: the same request
// must produce the same bytes every time, or the prefix match never hits.
func TestEncodeIsDeterministic(t *testing.T) {
	req := llm.Request{
		System:   "sys",
		Tools:    []llm.ToolSpec{toolSpec("a"), toolSpec("b")},
		Messages: []llm.Message{llm.UserText("q")},
		Choice:   llm.ForceTool("a"),
	}
	first := encodeFor(t, Config{Beta: []string{BetaContextManagement}}, req)
	for i := range 20 {
		again := encodeFor(t, Config{Beta: []string{BetaContextManagement}}, req)
		if string(first) != string(again) {
			t.Fatalf("encode is not deterministic (run %d):\n got %s\nwant %s", i, again, first)
		}
	}
}

func TestEncodeUnknownBlockIsRejected(t *testing.T) {
	// llm.ContentBlock is closed by an unexported method, so no test can build
	// an unknown variant from here. The defensive branch in encodeBlock is
	// therefore unreachable today; this records that rather than pretending it
	// is covered.
	t.Skip("llm.ContentBlock is a closed interface (unexported blockKind); the default branch of encodeBlock cannot be reached from a test without adding a variant to package llm, which would be a production change")
}

func TestHasBeta(t *testing.T) {
	tests := []struct {
		vals []string
		want bool
	}{
		{nil, false},
		{[]string{"x"}, false},
		{[]string{BetaContextManagement}, true},
		{[]string{"a", " " + BetaContextManagement + "\t"}, true},
		{[]string{BetaContextManagement + "-nope"}, false},
	}
	for _, tt := range tests {
		if got := hasBeta(tt.vals, BetaContextManagement); got != tt.want {
			t.Errorf("hasBeta(%v): got %v, want %v", tt.vals, got, tt.want)
		}
	}
}

func TestCanonicalHeaders(t *testing.T) {
	got := canonicalHeaders(map[string]string{" x-route ": "a"})
	if want := map[string]string{"X-Route": "a"}; !reflect.DeepEqual(got, want) {
		t.Errorf("canonicalHeaders: got %v, want %v", got, want)
	}
	if got := canonicalHeaders(nil); got != nil {
		t.Errorf("canonicalHeaders(nil): got %v, want nil", got)
	}
}

func TestParseHeaderLines(t *testing.T) {
	got := parseHeaderLines("A: 1\n\nno-colon\n: novalue\nb:2\nB: 3\n")
	want := map[string]string{"A": "1", "B": "3"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseHeaderLines: got %v, want %v", got, want)
	}
	if got := parseHeaderLines(""); got != nil {
		t.Errorf("parseHeaderLines(\"\"): got %v, want nil", got)
	}
}

func TestSplitList(t *testing.T) {
	if got, want := splitList(" a , ,b, "), []string{"a", "b"}; !reflect.DeepEqual(got, want) {
		t.Errorf("splitList: got %v, want %v", got, want)
	}
	if got := splitList(""); got != nil {
		t.Errorf("splitList(\"\"): got %v, want nil", got)
	}
}

func TestStreamErrorClassification(t *testing.T) {
	tests := []struct {
		typ  string
		want error
	}{
		{"overloaded_error", llm.ErrOverloaded},
		{"rate_limit_error", llm.ErrRateLimit},
		{"invalid_request_error", llm.ErrBadRequest},
		{"authentication_error", llm.ErrAuth},
		{"permission_error", llm.ErrAuth},
		{"not_found_error", llm.ErrNotFound},
		{"api_error", llm.ErrServer},
		{"something_new", llm.ErrServer},
	}
	for _, tt := range tests {
		t.Run(tt.typ, func(t *testing.T) {
			err := streamError(&apiError{Type: tt.typ, Message: "detail"})
			if !errors.Is(err, tt.want) {
				t.Errorf("errors.Is(err, %v): got false, want true (err=%v)", tt.want, err)
			}
			if !strings.Contains(err.Error(), "detail") {
				t.Errorf("error text: got %q, want it to carry the provider detail", err)
			}
		})
	}
	if err := streamError(nil); !errors.Is(err, llm.ErrServer) {
		t.Errorf("streamError(nil): got %v, want an ErrServer", err)
	}
}

func TestEncodeMedia(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\n")
	body := encodeFor(t, Config{}, llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.Text{Text: "look"},
			llm.Media{MediaType: "image/png", Data: png},
		}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.ToolUse{ID: "u1", Name: "view_media", Input: json.RawMessage(`{}`)}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "u1", Content: "Attached spec.pdf", Media: []llm.Media{{MediaType: "application/pdf", Data: []byte("%PDF"), Name: "spec.pdf"}}},
		}},
	}})
	s := string(body)
	for _, want := range []string{
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}`,
		`"content":[{"type":"text","text":"Attached spec.pdf"},{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERg=="},"title":"spec.pdf"`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("body is missing %s:\n%s", want, s)
		}
	}
	// The moving cache breakpoint lands on the last block, which here is a
	// tool_result carrying media.
	if !strings.Contains(s, `"title":"spec.pdf"}],"cache_control":{"type":"ephemeral"}`) {
		t.Errorf("the last block carries no cache breakpoint:\n%s", s)
	}
}

func TestEncodeMediaRefusesAnUnsupportedType(t *testing.T) {
	clearEnv(t)
	c, err := New(Config{APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.encode(llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{
		llm.Media{MediaType: "video/mp4", Data: []byte{0}},
	}}}}, "m", 10, false)
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Errorf("err = %v, want ErrBadRequest naming the type", err)
	}
}

// TestServerClearingNeverTakesASkillBody: a skill's activation lives on the
// client, which retires a skill whose body it removed. A body the server
// cleared would leave the skill's tools on offer with their instructions gone.
func TestServerClearingNeverTakesASkillBody(t *testing.T) {
	cfg := Config{Beta: []string{BetaContextManagement}}
	for _, tc := range []struct {
		tools []llm.ToolSpec
		want  []string
	}{
		{[]llm.ToolSpec{toolSpec("alpha"), toolSpec("load_skill")}, []string{"load_skill"}},
		{[]llm.ToolSpec{toolSpec("alpha")}, nil}, // never name a tool the request does not carry
	} {
		body := encodeFor(t, cfg, llm.Request{Messages: []llm.Message{llm.UserText("q")}, Tools: tc.tools})
		var doc struct {
			ContextManagement *contextManagement `json:"context_management"`
		}
		decodeBody(t, body, &doc)
		if got := doc.ContextManagement.Edits[0].ExcludeTools; !slices.Equal(got, tc.want) {
			t.Errorf("exclude_tools = %v, want %v", got, tc.want)
		}
	}
}

// TestClearedTokensAreReported: what the server cleared is reported, so the
// harness can still compare its estimate with the request it actually sent.
func TestClearedTokensAreReported(t *testing.T) {
	body := `{"model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
		"usage":{"input_tokens":100,"output_tokens":5},
		"context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919","cleared_tool_uses":3,"cleared_input_tokens":4000}]}}`
	resp, err := decodeMessage(strings.NewReader(body), "m", 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.ClearedInputTokens != 4000 || resp.Usage.RequestTokens() != 4100 || resp.Usage.PromptTokens() != 100 {
		t.Errorf("usage = %+v, want 4000 cleared on top of the 100 read", resp.Usage)
	}

	stream := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"m\",\"usage\":{\"input_tokens\":100}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"ok\"}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}," +
		"\"context_management\":{\"applied_edits\":[{\"type\":\"clear_tool_uses_20250919\",\"cleared_input_tokens\":4000}]}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	sresp, err := readStream(strings.NewReader(stream), nil, "m")
	if err != nil {
		t.Fatal(err)
	}
	if sresp.Usage.ClearedInputTokens != 4000 || sresp.Usage.InputTokens != 100 {
		t.Errorf("streamed usage = %+v, want the cleared tokens alongside the input", sresp.Usage)
	}
}
