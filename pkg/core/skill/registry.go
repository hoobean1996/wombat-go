package skill

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// ErrUnknownSkill is returned when the model names a skill that is not in the
// registry. It reaches the model as an is_error tool_result listing the real
// names, which is usually enough for it to correct itself — though the enum in
// the input schema is what should have prevented the call in the first place.
var ErrUnknownSkill = errors.New("skill: unknown skill")

// ErrNoState reports that load_skill was dispatched inside an agent run whose
// context carries no [State] — the run was wired without
// core.WithRunContext(... skill.WithState(ctx, skill.NewState()) ...).
//
// It is an error rather than a quiet fallback because the fallback lies. With
// no State to record it in, the activation lands on a throwaway: load_skill
// reports the skill activated and its tools available, the gated tools never
// appear, a second load pastes the whole body again, and nothing anywhere says
// why. Failing the first load names the missing wiring where a person will see
// it. Outside a dispatch — a test calling the handler directly — the throwaway
// is still used, so the handler stays callable anywhere.
var ErrNoState = errors.New("skill: this run has no skill.State, so loaded skills cannot be tracked; " +
	"install one with core.WithRunContext(func(ctx context.Context) context.Context { " +
	"return skill.WithState(ctx, skill.NewState()) })")

// Registry is the static catalogue: every skill the process knows about, plus
// the tool gates declared over them.
//
// Built once at startup and read from every turn of every run. It holds no
// per-run state — that is [State] — which is what makes a single Registry safe
// to share across concurrent runs of the same agent.
type Registry struct {
	// skills is kept sorted by name, and every derived artifact (Index, the
	// schema enum, error messages) is produced from it in that order.
	skills []Skill
	byName map[string]Skill

	// gates maps tool name -> gating skill name. Written only by [Gate], which
	// is documented startup-only; see there for why there is no mutex.
	gates map[string]string
}

// New builds a registry over skills.
//
// Duplicate names panic. Unlike [LoadDir], which gets its input from disk and
// must survive bad content, a duplicate here means the program constructed one:
// load_skill(name) would be ambiguous, the schema enum would list the name
// twice, and which body the model receives would depend on map order.
func New(skills ...Skill) *Registry {
	r := &Registry{
		byName: make(map[string]Skill, len(skills)),
		gates:  make(map[string]string),
	}
	for _, s := range skills {
		if _, dup := r.byName[s.Name]; dup {
			panic("skill: duplicate skill name " + s.Name)
		}
		r.byName[s.Name] = s
	}
	r.skills = append(r.skills, skills...)
	sortSkills(r.skills)
	return r
}

// Skills returns the catalogue, sorted by name. The slice is a copy; the
// Registry's own order is load-bearing and not the caller's to shuffle.
func (r *Registry) Skills() []Skill {
	return append([]Skill(nil), r.skills...)
}

// Index renders the catalogue the model chooses from. Passed to
// core.WithSystemBlock("available_skills", …), it reaches the model as:
//
//	<available_skills>
//	- pdf-forms: Filling and flattening AcroForm PDFs.
//	- sql-tuning: Reading EXPLAIN output and fixing slow queries.
//	</available_skills>
//
// The <available_skills> tags belong to the caller, not to this string.
// WithSystemBlock wraps whatever body it is given in a tag of the name it is
// given, so emitting them here too would nest the tag inside itself — one
// wrapper per layer that thinks it owns the block. Exactly one layer does.
//
// One line per skill, and only the description's FIRST line: this is a menu,
// not the content. The model needs enough to decide whether to call
// load_skill, and every extra line is paid for on every request forever.
//
// Sorted by name, and that is not cosmetic. This string is spliced into the
// system prompt, which is the prompt-cache prefix and is rendered ONCE when
// the agent is constructed. Iterating a map here would emit different bytes on
// every process start — same skills, different order — and every cold start
// would silently miss the cache for the whole system prompt. The cost shows up
// as a bill, never as a bug.
//
// Returns "" when the registry is empty, so a caller can append it
// unconditionally: WithSystemBlock drops empty bodies.
func (r *Registry) Index() string {
	var lines []string
	for _, s := range r.skills {
		if s.Install {
			// Its body is already in the prompt. Listing it as loadable
			// invites a call that can only return a stub.
			continue
		}
		lines = append(lines, fmt.Sprintf("- %s: %s", s.Name, firstLine(s.Description)))
	}
	return strings.Join(lines, "\n")
}

// Installed returns the skills that are part of the standing configuration,
// in catalogue order.
func (r *Registry) Installed() []Skill {
	var out []Skill
	for _, s := range r.skills {
		if s.Install {
			out = append(out, s)
		}
	}
	return out
}

// InstalledBody renders every installed skill's body for a system block.
//
// Order follows the sorted catalogue, and the rendering is fixed, because this
// text lands in the system prompt and therefore in the prompt-cache prefix: a
// block that reordered itself between processes would invalidate the cache on
// every start for no reason anyone could see.
func (r *Registry) InstalledBody() string {
	installed := r.Installed()
	if len(installed) == 0 {
		return ""
	}
	// No name heading. An installed skill is not something to fetch, and
	// printing "## wide-research" above its body advertises a handle that
	// looks loadable — measured, twice: the model's first action was
	// load_skill on a skill whose content it was already reading, spending a
	// turn to be told it already had it. The body stands on its own.
	var b strings.Builder
	for i, s := range installed {
		if i > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(s.Body)
	}
	return b.String()
}

// IsInstalled reports whether name is part of the standing configuration.
func (r *Registry) IsInstalled(name string) bool {
	s, ok := r.byName[name]
	return ok && s.Install
}

// Gate declares that toolName is hidden until skillName is loaded.
//
// STARTUP ONLY: call every Gate before the first run begins. A Registry is
// otherwise immutable, and this method is the single exception; it is
// deliberately unsynchronized, because a mutex here would buy nothing (the
// read side is every turn of every run, the write side is a handful of calls
// during construction) while implying the map may be mutated later, which it
// may not.
//
// Gating a tool behind a skill that is not in the registry is allowed and
// leaves the tool permanently hidden. That is the conservative reading: the
// skill file is missing, so its knowledge is missing, so its tools should not
// be offered.
//
// A second gate on the same tool from a DIFFERENT skill panics — two
// declarations disagreeing about who owns a tool is a wiring bug, and picking
// one silently means a tool that appears or vanishes depending on link order.
func (r *Registry) Gate(skillName, toolName string) {
	if prev, ok := r.gates[toolName]; ok && prev != skillName {
		panic(fmt.Sprintf("skill: tool %q is already gated by %q, cannot also gate it by %q", toolName, prev, skillName))
	}
	r.gates[toolName] = skillName
}

// GateFor reports the skill gating toolName, if any. A tool with no gate is
// always visible.
func (r *Registry) GateFor(toolName string) (string, bool) {
	s, ok := r.gates[toolName]
	return s, ok
}

// names returns the catalogue's names in index order.
func (r *Registry) names() []string { return sortedNames(r.skills) }

// nameList renders the names for a prompt or an error message.
func (r *Registry) nameList() string {
	if len(r.skills) == 0 {
		return "(none)"
	}
	return strings.Join(r.names(), ", ")
}

func sortSkills(s []Skill) {
	slices.SortFunc(s, func(a, b Skill) int { return strings.Compare(a.Name, b.Name) })
}

// ===== Input schema =====

// nameInput is the argument shape of both meta tools.
type nameInput struct {
	Name string `json:"name"`
}

type propSchema struct {
	Type        string   `json:"type"`
	Enum        []string `json:"enum,omitempty"`
	Description string   `json:"description"`
}

type objectSchema struct {
	Type       string                `json:"type"`
	Properties map[string]propSchema `json:"properties"`
	Required   []string              `json:"required"`
}

// nameSchema builds the input schema for a tool taking one skill name.
//
// The enum is the point. Without it the model invents plausible names —
// "finance-tools" when the registry has "finance" — and every miss costs a
// full iteration: the call goes out, comes back as an is_error tool_result the
// model has to read and recover from, and leaves a red card in the transcript.
// With the enum, the provider's own schema validation rejects the call before
// it is ever emitted, so the mistake costs nothing.
//
// An empty enum is not valid JSON Schema and providers reject the whole tool
// list over it, so the zero-skill case omits it and says so in the description
// instead. Structs rather than map[string]any so field ORDER is fixed: the
// schema is handed to the provider byte for byte and is part of the cached
// prefix, exactly like [Registry.Index].
func nameSchema(names []string, desc string) json.RawMessage {
	p := propSchema{Type: "string", Description: desc}
	if len(names) > 0 {
		p.Enum = names
	}
	raw, err := json.Marshal(objectSchema{
		Type:       "object",
		Properties: map[string]propSchema{"name": p},
		Required:   []string{"name"},
	})
	if err != nil {
		// Unreachable: the value is a struct of strings. Panic rather than
		// return an invalid schema, which would fail obscurely at the provider.
		panic("skill: building input schema: " + err.Error())
	}
	return raw
}
