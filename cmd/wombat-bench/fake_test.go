package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// fakeProvider is an Anthropic Messages endpoint that answers from a script.
//
// It exists so the command can be exercised end to end — flags, agent
// construction, tool dispatch, scoring, pass@k, both output files — with no API
// key and no cost. That matters more than it sounds: everything interesting
// about this command is the wiring between five packages, and the wiring is
// exactly what a unit test of any one of them cannot reach.
//
// It is stateless across requests on purpose. The harness resends the whole
// transcript every turn, so the fake can read the turn number off the
// conversation instead of keeping a counter per episode — which is what lets
// several episodes hit one server concurrently without a session id.
type fakeProvider struct {
	// script chooses a reply from the workspace path and the turn number.
	script func(workspace string, turn int) reply

	mu    sync.Mutex
	calls int
	temps []float64

	// model is the id every reply claims to have come from, empty for
	// [fakeModel]. Guarded by mu: a test sets it with [fakeProvider.answerAs]
	// while the server is already up.
	model string
}

// answerAs makes every subsequent reply claim to come from model. A test uses
// it to answer as something no pricing table knows.
func (f *fakeProvider) answerAs(model string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.model = model
}

// Temperatures returns the temperature seen on every request, in arrival
// order. A nil entry means the request carried none.
func (f *fakeProvider) Temperatures() []float64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]float64(nil), f.temps...)
}

// fakeModel is a real model id from llm.DefaultPricing, so the cost column in
// the report is exercised rather than being zero everywhere. Nothing reaches
// Anthropic — the base URL points at the httptest server — but the pricing
// table is looked up by id and an invented one prices at nothing.
const fakeModel = "claude-haiku-4-5"

// reply is one canned model response.
type reply struct {
	// text is the assistant's visible answer; set for a final turn.
	text string

	// tool, when non-empty, makes this a tool_use reply.
	tool  string
	input map[string]any
}

// answer builds a terminal reply.
func answer(s string) reply { return reply{text: s} }

// call builds a tool_use reply.
func call(name string, input map[string]any) reply { return reply{tool: name, input: input} }

// workspaceRE pulls the sample's directory out of the system prompt.
//
// The fake has to know where the episode is running, and the system prompt is
// where the command puts it — which makes this both the simplest way to find
// out and an assertion that the working_directory env block is actually
// rendered. If the command stopped emitting it, every scripted trajectory here
// would target the wrong path and the tests would fail loudly.
var workspaceRE = regexp.MustCompile(`(?s)<working_directory>\n(\S+)`)

// sampleRE reads the sample number back out of the workspace path, which is
// how the script makes different samples behave differently — the whole point
// of a pass@k being interesting.
var sampleRE = regexp.MustCompile(`sample-(\d+)$`)

func (f *fakeProvider) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/messages" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	var req struct {
		System      []struct{ Text string } `json:"system"`
		Temperature *float64                `json:"temperature"`
		Messages    []struct {
			Role    string `json:"role"`
			Content []any  `json:"content"`
		} `json:"messages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var system strings.Builder
	for _, b := range req.System {
		system.WriteString(b.Text)
	}
	ws := ""
	if m := workspaceRE.FindStringSubmatch(system.String()); m != nil {
		ws = m[1]
	}

	// The turn number is how many times the model has already spoken.
	turn := 0
	for _, m := range req.Messages {
		if m.Role == "assistant" {
			turn++
		}
	}

	f.mu.Lock()
	f.calls++
	if req.Temperature != nil {
		f.temps = append(f.temps, *req.Temperature)
	}
	// Under the same lock as the counters: a test sets it from its own
	// goroutine and -c requests are in flight on others.
	model := f.model
	f.mu.Unlock()

	writeReply(w, f.script(ws, turn), model)
}

// sampleOf reads the sample index off a workspace path, -1 when it has none.
func sampleOf(workspace string) int {
	m := sampleRE.FindStringSubmatch(workspace)
	if m == nil {
		return -1
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		return -1
	}
	return n
}

// writeReply encodes one response in the Messages API's shape. An empty model
// answers as [fakeModel].
func writeReply(w http.ResponseWriter, rep reply, model string) {
	if model == "" {
		model = fakeModel
	}
	type block struct {
		Type  string         `json:"type"`
		Text  string         `json:"text,omitempty"`
		ID    string         `json:"id,omitempty"`
		Name  string         `json:"name,omitempty"`
		Input map[string]any `json:"input,omitempty"`
	}
	body := struct {
		Model      string  `json:"model"`
		Content    []block `json:"content"`
		StopReason string  `json:"stop_reason"`
		Usage      struct {
			In  int `json:"input_tokens"`
			Out int `json:"output_tokens"`
		} `json:"usage"`
	}{Model: model}

	// Numbers a cost table can price, so the report's cost column is exercised
	// rather than being zero everywhere.
	body.Usage.In, body.Usage.Out = 1200, 180

	if rep.tool != "" {
		body.Content = []block{{
			Type:  "tool_use",
			ID:    "toolu_" + rep.tool + "_" + strconv.Itoa(len(rep.input)),
			Name:  rep.tool,
			Input: rep.input,
		}}
		body.StopReason = "tool_use"
	} else {
		body.Content = []block{{Type: "text", Text: rep.text}}
		body.StopReason = "end_turn"
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(body); err != nil {
		panic(fmt.Sprintf("fake provider: %v", err))
	}
}

// newFake starts a fake provider and points the environment at it.
//
// Every ANTHROPIC_* and fallback variable the client reads is cleared, so the
// test cannot be changed by whatever is in the developer's shell — including
// the case that matters most, a real API key that would turn a unit test into
// a bill.
func newFake(t *testing.T, script func(workspace string, turn int) reply) *fakeProvider {
	t.Helper()

	f := &fakeProvider{script: script}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)

	for k, v := range map[string]string{
		"ANTHROPIC_API_KEY":  "test-key-not-a-real-one",
		"ANTHROPIC_BASE_URL": srv.URL,
		"ANTHROPIC_MODEL":    fakeModel,

		// Buffered, not streamed: the fake would otherwise have to speak SSE
		// to say four sentences, and none of what this test covers is in the
		// transport.
		"ANTHROPIC_STREAM": "never",

		"ANTHROPIC_PROXY":          "",
		"ANTHROPIC_BETA":           "",
		"ANTHROPIC_CUSTOM_HEADERS": "",
		"AGENT_LLM_BASE_URL":       "",
		"AGENT_LLM_PROXY":          "",
		"WOMBAT_MODEL":             "",
		"WOMBAT_PROVIDER":          "anthropic",
	} {
		t.Setenv(k, v)
	}
	return f
}
