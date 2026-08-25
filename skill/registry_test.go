package skill

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/tool"
)

func threeSkills() []Skill {
	return []Skill{
		{Name: "alpha", Description: "Alpha's first line.\nAnd a second line nobody indexes."},
		{Name: "beta", Description: "Beta's one line."},
		{Name: "gamma", Description: "Gamma's one line."},
	}
}

func TestNewSortsAndCopies(t *testing.T) {
	in := []Skill{{Name: "gamma", Description: "g"}, {Name: "alpha", Description: "a"}, {Name: "beta", Description: "b"}}
	r := New(in...)

	got := r.Skills()
	if len(got) != 3 || got[0].Name != "alpha" || got[1].Name != "beta" || got[2].Name != "gamma" {
		t.Fatalf("Skills() = %v, want them sorted alpha, beta, gamma", got)
	}

	// Skills() hands back a copy: the registry's own order is load-bearing and
	// not the caller's to shuffle.
	got[0] = Skill{Name: "zzz", Description: "z"}
	if again := r.Skills(); again[0].Name != "alpha" {
		t.Errorf("after mutating the returned slice, Skills()[0] = %q, want alpha", again[0].Name)
	}
}

// TestNewPanicsOnDuplicate: unlike LoadDir, which gets its input from disk and
// must survive bad content, a duplicate here means the program constructed one.
func TestNewPanicsOnDuplicate(t *testing.T) {
	mustPanic(t, "duplicate skill name demo", func() {
		New(Skill{Name: "demo", Description: "one"}, Skill{Name: "demo", Description: "two"})
	})
}

func TestIndex(t *testing.T) {
	r := New(threeSkills()...)
	want := "- alpha: Alpha's first line.\n- beta: Beta's one line.\n- gamma: Gamma's one line."
	if got := r.Index(); got != want {
		t.Errorf("Index() =\n%q\nwant\n%q", got, want)
	}
}

func TestIndexEmptyRegistry(t *testing.T) {
	// "" so a caller can append it unconditionally: WithSystemBlock drops empty
	// bodies.
	if got := New().Index(); got != "" {
		t.Errorf("Index() = %q, want \"\" for an empty registry", got)
	}
}

// TestIndexIsDeterministic is the prompt-cache regression test.
//
// Index is spliced into the system prompt, which IS the prompt-cache prefix,
// and it is rendered once when the agent is constructed. If the registry
// iterated a map, two processes holding exactly the same skills would emit
// different bytes — same menu, different order — and every cold start would
// silently miss the cache for the entire system prompt. The cost shows up as a
// bill, never as a bug, which is why it needs a test rather than a reviewer.
//
// Same for the load_skill/unload_skill input schemas: the enum is handed to
// the provider byte for byte and is part of the same cached prefix.
func TestIndexIsDeterministicAcrossInputOrder(t *testing.T) {
	s := threeSkills()
	forward := New(s[0], s[1], s[2])
	reverse := New(s[2], s[1], s[0])
	shuffled := New(s[1], s[2], s[0])

	if a, b := forward.Index(), reverse.Index(); a != b {
		t.Errorf("Index() differs with input order:\n forward: %q\n reverse: %q", a, b)
	}
	if a, b := forward.Index(), shuffled.Index(); a != b {
		t.Errorf("Index() differs with input order:\n forward:  %q\n shuffled: %q", a, b)
	}

	// The same bytes have to come out of Bind, which is what actually reaches
	// the agent.
	gf := forward.Bind([]tool.Def{echoTool("one"), echoTool("two")})
	gr := reverse.Bind([]tool.Def{echoTool("one"), echoTool("two")})
	if gf.Index != gr.Index {
		t.Errorf("Gated.Index differs with input order:\n %q\n %q", gf.Index, gr.Index)
	}

	for _, name := range []string{LoadToolName, UnloadToolName} {
		df, _ := gf.Set.Find(name)
		dr, _ := gr.Set.Find(name)
		if string(df.InputSchema) != string(dr.InputSchema) {
			t.Errorf("%s InputSchema differs with input order:\n %s\n %s",
				name, df.InputSchema, dr.InputSchema)
		}
		if df.Description != dr.Description {
			t.Errorf("%s Description differs with input order:\n %q\n %q",
				name, df.Description, dr.Description)
		}
	}

	// And a second Bind of the same registry is byte-identical too, so
	// rebuilding an agent in-process does not move the prefix either.
	if again := forward.Bind(nil); again.Index != gf.Index {
		t.Errorf("a second Bind produced a different index:\n %q\n %q", again.Index, gf.Index)
	}
}

// TestIndexUsesOnlyTheFirstDescriptionLine: this is a menu, not the content.
// Every extra line is paid for on every request forever.
func TestIndexUsesOnlyTheFirstDescriptionLine(t *testing.T) {
	r := New(Skill{
		Name:        "pdf-forms",
		Description: "Filling and flattening AcroForm PDFs.\nUse when the task mentions a fillable PDF.",
		Body:        "nine kilobytes of procedure",
	})
	got := r.Index()
	if want := "- pdf-forms: Filling and flattening AcroForm PDFs."; got != want {
		t.Errorf("Index() = %q, want %q", got, want)
	}
	if strings.Contains(got, "fillable") {
		t.Errorf("Index() = %q, want the second description line left out", got)
	}
	if strings.Contains(got, "nine kilobytes") {
		t.Errorf("Index() = %q, want the body left out", got)
	}
}

// TestIndexCarriesNoTags: WithSystemBlock wraps whatever body it is given, so
// emitting the tags here too would nest the tag inside itself.
func TestIndexCarriesNoTags(t *testing.T) {
	if got := New(threeSkills()...).Index(); strings.Contains(got, "<available_skills>") {
		t.Errorf("Index() = %q, want no <available_skills> wrapper", got)
	}
}

// ===== gates =====

func TestGateAndGateFor(t *testing.T) {
	r := New(threeSkills()...)
	r.Gate("alpha", "tool_a")
	r.Gate("beta", "tool_b")

	if got, ok := r.GateFor("tool_a"); !ok || got != "alpha" {
		t.Errorf("GateFor(tool_a) = %q, %v; want alpha, true", got, ok)
	}
	if got, ok := r.GateFor("tool_b"); !ok || got != "beta" {
		t.Errorf("GateFor(tool_b) = %q, %v; want beta, true", got, ok)
	}
	if got, ok := r.GateFor("ungated"); ok {
		t.Errorf("GateFor(ungated) = %q, %v; want \"\", false", got, ok)
	}

	// Re-declaring the same gate is a harmless no-op.
	r.Gate("alpha", "tool_a")
	if got, _ := r.GateFor("tool_a"); got != "alpha" {
		t.Errorf("GateFor(tool_a) = %q after a repeat Gate, want alpha", got)
	}
}

// TestGateConflictPanics: two declarations disagreeing about who owns a tool
// is a wiring bug, and picking one silently means a tool that appears or
// vanishes depending on link order.
func TestGateConflictPanics(t *testing.T) {
	r := New(threeSkills()...)
	r.Gate("alpha", "shared_tool")
	mustPanic(t, `tool "shared_tool" is already gated by "alpha", cannot also gate it by "beta"`, func() {
		r.Gate("beta", "shared_tool")
	})
}

// TestGateOnAnUnknownSkillHidesForever is the conservative reading: the skill
// file is missing, so its knowledge is missing, so its tools are not offered.
func TestGateOnAnUnknownSkillHidesForever(t *testing.T) {
	r := New(demoSkill())
	r.Gate("skill-that-does-not-exist", "orphan")
	g := r.Bind([]tool.Def{echoTool("orphan"), echoTool("plain")})

	ctx := WithState(t.Context(), NewState())
	if got := names(g.Set.Visible(ctx)); contains(got, "orphan") {
		t.Errorf("Visible = %v, want no orphan: its gating skill does not exist", got)
	}
	// And there is no way to load it, because the name is not in the registry.
	h := tool.Chain(tool.Direct, g.Middleware)
	r2 := dispatch1(t, ctx, h, g.Set, loadUse("u1", "skill-that-does-not-exist"))
	if r2.Err == nil {
		t.Fatalf("load_skill error = nil, want %v", ErrUnknownSkill)
	}
	if got := names(g.Set.Visible(ctx)); contains(got, "orphan") {
		t.Errorf("Visible = %v after a failed load, want no orphan", got)
	}
}

// ===== input schema =====

type parsedSchema struct {
	Type       string `json:"type"`
	Properties struct {
		Name struct {
			Type        string   `json:"type"`
			Enum        []string `json:"enum"`
			Description string   `json:"description"`
		} `json:"name"`
	} `json:"properties"`
	Required []string `json:"required"`
}

func TestNameSchemaEnumeratesSkills(t *testing.T) {
	g := New(threeSkills()...).Bind(nil)

	for _, name := range []string{LoadToolName, UnloadToolName} {
		t.Run(name, func(t *testing.T) {
			d, ok := g.Set.Find(name)
			if !ok {
				t.Fatalf("Find(%q) = false, want the meta tool", name)
			}
			var s parsedSchema
			if err := json.Unmarshal(d.InputSchema, &s); err != nil {
				t.Fatalf("InputSchema is not valid JSON: %v (%s)", err, d.InputSchema)
			}
			if s.Type != "object" {
				t.Errorf("schema type = %q, want object", s.Type)
			}
			want := []string{"alpha", "beta", "gamma"}
			if strings.Join(s.Properties.Name.Enum, ",") != strings.Join(want, ",") {
				t.Errorf("enum = %v, want %v in sorted order", s.Properties.Name.Enum, want)
			}
			if strings.Join(s.Required, ",") != "name" {
				t.Errorf("required = %v, want [name]", s.Required)
			}
			if !strings.Contains(s.Properties.Name.Description, "alpha, beta, gamma") {
				t.Errorf("description = %q, want it to list the skills", s.Properties.Name.Description)
			}
		})
	}
}

// TestNameSchemaOmitsAnEmptyEnum: an empty enum is not valid JSON Schema and
// providers reject the whole tool list over it.
func TestNameSchemaOmitsAnEmptyEnum(t *testing.T) {
	g := New().Bind(nil)

	for _, name := range []string{LoadToolName, UnloadToolName} {
		d, _ := g.Set.Find(name)
		if strings.Contains(string(d.InputSchema), `"enum"`) {
			t.Errorf("%s InputSchema = %s, want no enum key with zero skills", name, d.InputSchema)
		}
		var s parsedSchema
		if err := json.Unmarshal(d.InputSchema, &s); err != nil {
			t.Fatalf("%s InputSchema is not valid JSON: %v", name, err)
		}
		if !strings.Contains(s.Properties.Name.Description, "no skills are registered") {
			t.Errorf("%s description = %q, want it to say there are no skills",
				name, s.Properties.Name.Description)
		}
	}
}

func TestNameList(t *testing.T) {
	if got := New().nameList(); got != "(none)" {
		t.Errorf("nameList() = %q, want (none)", got)
	}
	if got := New(threeSkills()...).nameList(); got != "alpha, beta, gamma" {
		t.Errorf("nameList() = %q, want %q", got, "alpha, beta, gamma")
	}
}

func TestMetaToolMetadata(t *testing.T) {
	g := New(demoSkill()).Bind(nil)
	for _, name := range []string{LoadToolName, UnloadToolName} {
		d, _ := g.Set.Find(name)
		// CapReadOnly|CapMeta so a planner filtered to those caps can still
		// pull domain knowledge before deciding how to decompose the work.
		if !d.Has(tool.CapReadOnly) || !d.Has(tool.CapMeta) {
			t.Errorf("%s caps = %v, want CapReadOnly|CapMeta", name, d.Caps)
		}
		if d.Has(tool.CapMutating) || d.Has(tool.CapExec) || d.Has(tool.CapNetwork) {
			t.Errorf("%s caps = %v, want nothing beyond CapReadOnly|CapMeta", name, d.Caps)
		}
		// Idempotent: the second call is a stub, so a retry cannot double-load.
		if !d.Idempotent {
			t.Errorf("%s Idempotent = false, want true", name)
		}
		if d.Timeout <= 0 {
			t.Errorf("%s Timeout = %v, want a positive cap", name, d.Timeout)
		}
	}
}
