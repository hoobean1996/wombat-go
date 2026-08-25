package skill

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/tool"
)

func hasTag(tags []string, want string) bool { return contains(tags, want) }

// TestLoadSkillReturnsTheBodyThenAStub.
//
// The second call is a memo, not a repeat: the body is already in the
// transcript verbatim, and in a plan/act pipeline where planner, executor and
// recovery each call load_skill for the same domain, returning it again pays
// for nine kilobytes three more times. That is how a context window dies.
func TestLoadSkillReturnsTheBodyThenAStub(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	first := dispatch1(t, ctx, h, g.Set, loadUse("u1", "demo"))
	if first.Err != nil {
		t.Fatalf("first load error = %v, want nil", first.Err)
	}
	if !strings.Contains(first.Output, demoBody) {
		t.Errorf("first load output = %q, want it to contain the body", first.Output)
	}
	if !strings.Contains(first.Output, `[skill "demo" activated`) {
		t.Errorf("first load output = %q, want the activation footer", first.Output)
	}
	// Footer, not header: a model reading a long markdown blob should hit the
	// content first.
	if !strings.HasPrefix(first.Output, demoBody) {
		t.Errorf("first load output starts with %q, want the body first", first.Output[:min(40, len(first.Output))])
	}
	if !StateFrom(ctx).IsActive("demo") {
		t.Error("demo is not active after load_skill")
	}

	second := dispatch1(t, ctx, h, g.Set, loadUse("u2", "demo"))
	if second.Err != nil {
		t.Fatalf("second load error = %v, want nil", second.Err)
	}
	if strings.Contains(second.Output, demoBody) {
		t.Errorf("second load output = %q, want a stub without the body", second.Output)
	}
	if !strings.Contains(second.Output, "already loaded earlier in this run") {
		t.Errorf("second load output = %q, want it to point at the copy already present", second.Output)
	}
	// The stub keeps the activation truthful: the gated tools stay exposed.
	if !StateFrom(ctx).IsActive("demo") {
		t.Error("demo went inactive after the stub, want it still loaded")
	}
	if got := names(g.Set.Visible(ctx)); !contains(got, "secret") {
		t.Errorf("Visible = %v after the stub, want secret still offered", got)
	}
	// The first body's id is the one that sticks, so Reconcile still matches.
	if id, _ := StateFrom(ctx).BodyOf("demo"); id != "u1" {
		t.Errorf("BodyOf(demo) = %q after a second load, want u1", id)
	}
}

// TestLoadSkillAnnotatesOnEveryPath. The tag is the contract with
// wombat.DropTagged: the harness cannot tell nine kilobytes of skill body from
// nine kilobytes of grep output by looking at the tool_result. A stub or an
// error result that went untagged would survive eviction forever.
func TestLoadSkillAnnotatesOnEveryPath(t *testing.T) {
	_, g, h := gatedFixture(t)

	tests := []struct {
		name    string
		setup   func(context.Context)
		use     llm.ToolUse
		wantTag string
		wantErr bool
	}{
		{
			name:    "the body path",
			use:     loadUse("u1", "demo"),
			wantTag: TagPrefix + "demo",
		},
		{
			name:    "the stub path",
			setup:   func(ctx context.Context) { StateFrom(ctx).Activate("demo", "earlier") },
			use:     loadUse("u2", "demo"),
			wantTag: TagPrefix + "demo",
		},
		{
			name:    "the unknown-skill error path",
			use:     loadUse("u3", "nope"),
			wantTag: TagPrefix + "nope",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := WithState(t.Context(), NewState())
			if tt.setup != nil {
				tt.setup(ctx)
			}
			res := dispatch1(t, ctx, h, g.Set, tt.use)
			if gotErr := res.Err != nil; gotErr != tt.wantErr {
				t.Fatalf("error = %v, wantErr = %v", res.Err, tt.wantErr)
			}
			if !hasTag(res.Tags, Tag) {
				t.Errorf("tags = %v, want them to include %q", res.Tags, Tag)
			}
			if !hasTag(res.Tags, tt.wantTag) {
				t.Errorf("tags = %v, want them to include %q", res.Tags, tt.wantTag)
			}
		})
	}
}

// TestLoadSkillAnnotatesADecodeFailure is the one path that is NOT annotated.
//
// tool.Typed unmarshals the arguments before the handler body runs, so a
// malformed `{"name": 123}` returns "invalid input: ..." without ever reaching
// the tool.Annotate call at the top of the handler. The resulting is_error
// tool_result carries no skill tag, so wombat.DropTagged can never evict it —
// exactly the "an error result that went untagged would [survive eviction
// forever]" case the handler's own comment says the unconditional Annotate is
// there to prevent.
//
// Skipped rather than deleted: fixing it needs a production change (annotate
// outside Typed, or take the raw json.RawMessage in load_skill), and this
// package's tests are not allowed to make one.
func TestLoadSkillAnnotatesADecodeFailure(t *testing.T) {

	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	res := dispatch1(t, ctx, h, g.Set, llm.ToolUse{
		ID: "u1", Name: LoadToolName, Input: json.RawMessage(`{"name": 123}`),
	})
	if res.Err == nil {
		t.Fatalf("output = %q, want a decode error", res.Output)
	}
	if !hasTag(res.Tags, Tag) {
		t.Errorf("tags = %v, want them to include %q so the error result can be evicted", res.Tags, Tag)
	}
}

func TestLoadSkillUnknownName(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	res := dispatch1(t, ctx, h, g.Set, loadUse("u1", "finance-tools"))
	if !errors.Is(res.Err, ErrUnknownSkill) {
		t.Fatalf("error = %v, want one wrapping %v", res.Err, ErrUnknownSkill)
	}
	// The message lists the real names, which is usually enough for the model
	// to correct itself.
	if !strings.Contains(res.Err.Error(), "demo") {
		t.Errorf("error = %q, want it to list the available skills", res.Err)
	}
	if len(StateFrom(ctx).Active()) != 0 {
		t.Errorf("Active() = %v after a failed load, want none", StateFrom(ctx).Active())
	}
}

func TestLoadSkillTrimsTheName(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	res := dispatch1(t, ctx, h, g.Set, loadUse("u1", "  demo\n"))
	if res.Err != nil {
		t.Fatalf("error = %v, want nil: surrounding whitespace must be tolerated", res.Err)
	}
	if !StateFrom(ctx).IsActive("demo") {
		t.Error("demo is not active, want the trimmed name to have been used")
	}
	if !hasTag(res.Tags, TagPrefix+"demo") {
		t.Errorf("tags = %v, want %q", res.Tags, TagPrefix+"demo")
	}
}

func TestLoadSkillListsActiveSkills(t *testing.T) {
	r := New(
		Skill{Name: "alpha", Description: "A.", Body: "alpha body"},
		Skill{Name: "beta", Description: "B.", Body: "beta body"},
	)
	g := r.Bind(nil)
	h := tool.Chain(tool.Direct, g.Middleware)
	ctx := WithState(t.Context(), NewState())

	first := dispatch1(t, ctx, h, g.Set, loadUse("u1", "beta"))
	if !strings.Contains(first.Output, "Active skills: beta") {
		t.Errorf("output = %q, want it to list beta as active", first.Output)
	}
	second := dispatch1(t, ctx, h, g.Set, loadUse("u2", "alpha"))
	// Sorted, because an observation that reorders itself between otherwise
	// identical turns is noise the model has to reconcile.
	if !strings.Contains(second.Output, "Active skills: alpha, beta") {
		t.Errorf("output = %q, want the active list sorted", second.Output)
	}
}

// ===== unload =====

func TestUnloadSkill(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	if res := dispatch1(t, ctx, h, g.Set, loadUse("u1", "demo")); res.Err != nil {
		t.Fatalf("load_skill error = %v", res.Err)
	}

	res := dispatch1(t, ctx, h, g.Set, unloadUse("u2", "demo"))
	if res.Err != nil {
		t.Fatalf("unload error = %v, want nil", res.Err)
	}
	if !strings.Contains(res.Output, `[skill "demo" unloaded`) {
		t.Errorf("output = %q, want it to confirm the unload", res.Output)
	}
	if !strings.Contains(res.Output, "Active skills: (none)") {
		t.Errorf("output = %q, want it to report an empty active list", res.Output)
	}
	if StateFrom(ctx).IsActive("demo") {
		t.Error("demo is still active after unload_skill")
	}
	if got := names(g.Set.Visible(ctx)); contains(got, "secret") {
		t.Errorf("Visible = %v after unload, want secret retired", got)
	}
}

// TestUnloadSkillIsIdempotent: unloading a skill that is not active is a no-op
// reported as an ordinary observation, not an is_error. Nothing is wrong — the
// desired end state already holds — and an error card here would push the
// model into pointless recovery.
func TestUnloadSkillIsIdempotent(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	if res := dispatch1(t, ctx, h, g.Set, loadUse("u1", "demo")); res.Err != nil {
		t.Fatalf("load_skill error = %v", res.Err)
	}
	first := dispatch1(t, ctx, h, g.Set, unloadUse("u2", "demo"))
	if first.Err != nil {
		t.Fatalf("first unload error = %v, want nil", first.Err)
	}

	for i, use := range []llm.ToolUse{unloadUse("u3", "demo"), unloadUse("u4", "demo")} {
		res := dispatch1(t, ctx, h, g.Set, use)
		if res.Err != nil {
			t.Fatalf("repeat unload %d error = %v, want nil: unloading twice is a no-op", i, res.Err)
		}
		if !strings.Contains(res.Output, "was not active") {
			t.Errorf("repeat unload %d output = %q, want it to say there was nothing to do", i, res.Output)
		}
		if StateFrom(ctx).IsActive("demo") {
			t.Error("demo became active again after a repeat unload")
		}
	}
}

// TestUnloadSkillOnANeverLoadedSkill: still a no-op, still not an error.
func TestUnloadSkillOnANeverLoadedSkill(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	res := dispatch1(t, ctx, h, g.Set, unloadUse("u1", "demo"))
	if res.Err != nil {
		t.Fatalf("error = %v, want nil", res.Err)
	}
	if !strings.Contains(res.Output, "was not active") {
		t.Errorf("output = %q, want it to say there was nothing to do", res.Output)
	}
}

// TestUnloadSkillUnknownName: a typo the model should see, not a no-op it will
// believe worked.
func TestUnloadSkillUnknownName(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	res := dispatch1(t, ctx, h, g.Set, unloadUse("u1", "finance-tools"))
	if !errors.Is(res.Err, ErrUnknownSkill) {
		t.Fatalf("error = %v, want one wrapping %v", res.Err, ErrUnknownSkill)
	}
	if !strings.Contains(res.Err.Error(), "demo") {
		t.Errorf("error = %q, want it to list the known skills", res.Err)
	}
}

func TestUnloadSkillTrimsTheName(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	if res := dispatch1(t, ctx, h, g.Set, loadUse("u1", "demo")); res.Err != nil {
		t.Fatalf("load_skill error = %v", res.Err)
	}
	if res := dispatch1(t, ctx, h, g.Set, unloadUse("u2", " demo ")); res.Err != nil {
		t.Fatalf("unload error = %v, want nil", res.Err)
	}
	if StateFrom(ctx).IsActive("demo") {
		t.Error("demo is still active, want the trimmed name to have been used")
	}
}

// TestLoadAfterUnloadReturnsTheBodyAgain: once the skill has been retired the
// body is no longer guaranteed to be in context, so the memo must not fire.
func TestLoadAfterUnloadReturnsTheBodyAgain(t *testing.T) {
	_, g, h := gatedFixture(t)
	ctx := WithState(t.Context(), NewState())

	if res := dispatch1(t, ctx, h, g.Set, loadUse("u1", "demo")); res.Err != nil {
		t.Fatalf("load_skill error = %v", res.Err)
	}
	if res := dispatch1(t, ctx, h, g.Set, unloadUse("u2", "demo")); res.Err != nil {
		t.Fatalf("unload_skill error = %v", res.Err)
	}

	again := dispatch1(t, ctx, h, g.Set, loadUse("u3", "demo"))
	if again.Err != nil {
		t.Fatalf("reload error = %v, want nil", again.Err)
	}
	if !strings.Contains(again.Output, demoBody) {
		t.Errorf("reload output = %q, want the body back", again.Output)
	}
	if id, _ := StateFrom(ctx).BodyOf("demo"); id != "u3" {
		t.Errorf("BodyOf(demo) = %q, want the new body id u3", id)
	}
}

// TestMetaToolsRunOutsideARun: a tool is an ordinary function, so the handlers
// must work with no State on the context. The activation goes to a throwaway,
// which is why nothing leaks.
func TestMetaToolsRunOutsideARun(t *testing.T) {
	g := New(demoSkill()).Bind(nil)

	load, _ := g.Set.Find(LoadToolName)
	out, err := load.Fn(context.Background(), json.RawMessage(`{"name":"demo"}`))
	if err != nil {
		t.Fatalf("load_skill error = %v, want nil", err)
	}
	if !strings.Contains(out, demoBody) {
		t.Errorf("output = %q, want the body", out)
	}

	unload, _ := g.Set.Find(UnloadToolName)
	out, err = unload.Fn(context.Background(), json.RawMessage(`{"name":"demo"}`))
	if err != nil {
		t.Fatalf("unload_skill error = %v, want nil", err)
	}
	if !strings.Contains(out, "was not active") {
		t.Errorf("output = %q, want the throwaway state to have kept nothing", out)
	}
}

func TestJoin(t *testing.T) {
	tests := []struct {
		in   []string
		want string
	}{
		{nil, "(none)"},
		{[]string{}, "(none)"},
		{[]string{"a"}, "a"},
		{[]string{"a", "b"}, "a, b"},
	}
	for _, tt := range tests {
		if got := join(tt.in); got != tt.want {
			t.Errorf("join(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
