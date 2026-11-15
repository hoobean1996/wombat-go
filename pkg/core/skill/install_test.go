package skill_test

import (
	"context"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/skill"
	"wombat/pkg/core/tool"
)

const installFrontmatter = `---
name: method
install: true
description: How to open a task.
---
Name the invariants before writing code.`

const loadableFrontmatter = `---
name: domain
description: Everything about widgets.
---
A widget has three parts.`

func parse(t *testing.T, text string) skill.Skill {
	t.Helper()
	s, err := skill.Parse(text, "")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return s
}

// An installed skill is standing configuration, not something to route to.
// Everything about how it reaches the model differs from a loadable one.
func TestInstalledSkillIsPresentRatherThanOffered(t *testing.T) {
	method := parse(t, installFrontmatter)
	domain := parse(t, loadableFrontmatter)
	if !method.Install {
		t.Fatal("install: true was not parsed")
	}
	if domain.Install {
		t.Fatal("a skill with no install key was treated as installed")
	}

	reg := skill.New(method, domain)

	// The catalogue offers what can be loaded. Listing an installed skill
	// there invites a call that can only answer "you already have this".
	idx := reg.Index()
	if strings.Contains(idx, "method") {
		t.Errorf("the installed skill is listed as loadable:\n%s", idx)
	}
	if !strings.Contains(idx, "domain") {
		t.Errorf("the loadable skill is missing from the catalogue:\n%s", idx)
	}

	// Its body goes to the prompt instead.
	body := reg.InstalledBody()
	if !strings.Contains(body, "Name the invariants") {
		t.Errorf("the installed body is not rendered for the prompt: %q", body)
	}
	if strings.Contains(body, "A widget has three parts") {
		t.Error("a loadable skill's body leaked into the installed block")
	}
}

// The gate exists so a model does not get a tool before the knowledge that
// tool needs. An installed skill satisfies that condition before the run
// starts, so its tools must be there from the first call — and the visibility
// side and the enforcement side must agree, or a tool is offered and refused.
func TestInstalledSkillOpensItsGateFromTheFirstCall(t *testing.T) {
	reg := skill.New(parse(t, installFrontmatter), parse(t, loadableFrontmatter))
	reg.Gate("method", "measure")
	reg.Gate("domain", "polish")

	base := []tool.Def{
		{Name: "measure", Caps: tool.CapReadOnly},
		{Name: "polish", Caps: tool.CapReadOnly},
		{Name: "ungated", Caps: tool.CapReadOnly},
	}
	g := reg.Bind(base)

	ctx := skill.WithState(context.Background(), skill.NewState())
	visible := map[string]bool{}
	for _, d := range g.Set.Visible(ctx) {
		visible[d.Name] = true
	}
	if !visible["measure"] {
		t.Error("a tool gated behind an INSTALLED skill was hidden on the first call")
	}
	if visible["polish"] {
		t.Error("a tool gated behind a loadable skill was visible before it was loaded")
	}
	if !visible["ungated"] {
		t.Error("an ungated tool went missing")
	}

	// Enforcement must agree with visibility, or the gate produces the one
	// outcome it exists to prevent: a tool offered and then refused.
	ran := func(context.Context, tool.Def, llm.ToolUse) (string, error) { return "ran", nil }
	h := g.Middleware(ran)

	if out, err := h(ctx, base[0], llm.ToolUse{ID: "1", Name: "measure"}); err != nil {
		t.Errorf("a tool gated behind an installed skill was refused: %v", err)
	} else if out != "ran" {
		t.Errorf("measure returned %q", out)
	}
	if _, err := h(ctx, base[1], llm.ToolUse{ID: "2", Name: "polish"}); err == nil {
		t.Error("a tool gated behind an unloaded skill was allowed to run")
	}
}

// Asking to load something already present is a mistake worth one line, not a
// wall. An error here reads as a dead end and invites a retry.
func TestLoadingAnInstalledSkillReturnsAStubNotAnError(t *testing.T) {
	reg := skill.New(parse(t, installFrontmatter))
	g := reg.Bind(nil)

	load, ok := g.Set.Find(skill.LoadToolName)
	if !ok {
		t.Fatal("load_skill is not in the set")
	}
	ctx := skill.WithState(context.Background(), skill.NewState())
	out, err := load.Fn(ctx, []byte(`{"name":"method"}`))
	if err != nil {
		t.Fatalf("loading an installed skill errored: %v", err)
	}
	if !strings.Contains(out, "installed") {
		t.Errorf("the stub does not say the skill is already present: %q", out)
	}
	if strings.Contains(out, "Name the invariants") {
		t.Error("the stub repeated the whole body, which is already in the prompt")
	}
}
