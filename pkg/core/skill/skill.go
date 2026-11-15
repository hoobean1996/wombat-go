// Package skill turns a directory of markdown files into domain knowledge the
// model pulls on demand, plus the tool gating that goes with it.
//
// A skill is a markdown file with YAML frontmatter at <root>/<name>/SKILL.md.
// Only its name and one-line description sit in the system prompt; the body —
// routinely nine kilobytes of procedure — enters the transcript exactly when
// the model calls load_skill, and can leave again when a transcript strategy
// evicts it. Tools may be gated behind a skill so that the model's tool list
// stays small until the relevant domain is actually in play.
//
// # Wiring
//
// Three options plus a strategy, and they have to agree with each other:
//
//	skills, _ := skill.LoadDir("./skills", func(err error) { log.Print(err) })
//	reg := skill.New(skills...)
//	reg.Gate("pdf-forms", "fill_pdf")
//	g := reg.Bind(builtin.Default(deps))
//
//	a, err := core.New(
//	    core.WithClient(client),
//	    core.WithToolSet(g.Set),
//	    core.WithToolMiddleware(g.Middleware),
//	    core.WithSystemBlock("available_skills", g.Index),
//	    core.WithRunContext(func(ctx context.Context) context.Context {
//	        return skill.WithState(ctx, skill.NewState())
//	    }),
//	    core.WithStrategy(core.DropTagged(40, 12, skill.Tag)),
//	)
//
// Each line earns its place:
//
//   - WithToolSet installs the gated surface AND the reconciler: [Gated.Set]
//     implements [tool.Reconciler], which the agent loop calls after the
//     strategy has run.
//   - WithToolMiddleware is the enforcement half. Hiding a tool from the list
//     is not the same as refusing to run it; see [Gated.Middleware].
//   - WithRunContext is where the activation set is born. It must be per-run:
//     a core.Agent is immutable and shared, so activations kept anywhere
//     else would leak between concurrent runs.
//   - WithSystemBlock supplies the <available_skills> tags; [Registry.Index]
//     returns only the lines that go inside them.
//   - DropTagged is optional, but it is why every loaded body is annotated
//     with [Tag]. Without a strategy that reads the tag, bodies accumulate.
//
// # File format
//
//	---
//	name: pdf-forms
//	description: |
//	  Filling and flattening AcroForm PDFs.
//	  Use when the task mentions a fillable PDF.
//	---
//	# PDF forms
//	...
package skill

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
)

// FileName is the file [LoadDir] looks for inside each subdirectory of a
// skills root.
const FileName = "SKILL.md"

// Skill is one lazily-loaded chunk of domain expertise.
//
// Name and Description are cheap and always in the system prompt; Body is the
// expensive part and only ever reaches the model as a tool_result.
type Skill struct {
	// Name is the identifier the model passes to load_skill. It is also the
	// suffix of the annotation tag, so it must be stable across processes.
	Name string

	// Description is what the model judges relevance by. Its first line goes
	// into the index.
	Description string

	// Body is the full markdown after the frontmatter, whitespace-trimmed.
	Body string

	// Source is the path this was read from, or "" for [Parse] with no path.
	// Diagnostics only — nothing branches on it.
	Source string

	// Install makes this skill part of the agent's standing configuration
	// rather than something the model chooses to load.
	//
	// The distinction is between a skill that carries DOMAIN knowledge and one
	// that describes a METHOD. Domain knowledge is dead weight until the model
	// is in that domain, so it is loaded on demand and the model decides when.
	// A method — how to open a task, how to review, how to test — is worthless
	// once the work is finished, so leaving the decision to the model is
	// leaving it too late.
	//
	// Measured: a methodology skill whose description said to load it before
	// implementing anything with concurrency went unloaded through a
	// 33-iteration run of exactly that task. Rewording the tool got it loaded
	// first — and the model then loaded a DIFFERENT skill and stopped. Routing
	// is the wrong mechanism for something that must simply be present.
	//
	// An installed skill's body goes into the system prompt at construction,
	// so it costs prompt-cache prefix and no turn, and cannot be missed. Its
	// gated tools are visible from the first call.
	Install bool
}

// Parse errors. Wrapped with the offending path or name, so callers match with
// [errors.Is] and log the wrapper.
var (
	// ErrNoFrontmatter means the file did not open with a --- delimited block.
	ErrNoFrontmatter = errors.New("skill: missing YAML frontmatter")

	// ErrBadFrontmatter means a frontmatter line was not key: value.
	ErrBadFrontmatter = errors.New("skill: malformed frontmatter")

	// ErrNoName means neither the frontmatter nor the path supplied a name.
	ErrNoName = errors.New("skill: no name")

	// ErrNoDescription means the frontmatter had no description. This is fatal
	// rather than defaulted: the description is the ONLY thing the model sees
	// before deciding to load, so a skill without one is either never loaded
	// or loaded at random. An empty index line is worse than a loud startup
	// error.
	ErrNoDescription = errors.New("skill: no description")
)

const delim = "---"

// Parse reads skill markdown. sourcePath may be "" — it is used for error
// messages and as the fallback for a missing name.
//
// The frontmatter grammar is a fixed subset of YAML: `key: value` and block
// scalars (`key: |`, `key: >`). That subset is hand-rolled on purpose. A YAML
// dependency here would be this module's first, imposed on every consumer, to
// read two string fields out of a file format we also define.
//
// Unknown keys are ignored rather than rejected: SKILL.md files are shared
// with other tools (editors, other harnesses) that add their own keys, and
// failing on a key we simply do not need would make this package the reason a
// perfectly good skill cannot be used.
func Parse(text, sourcePath string) (Skill, error) {
	where := sourcePath
	if where == "" {
		where = "<memory>"
	}

	meta, body, err := splitFrontmatter(text)
	if err != nil {
		return Skill{}, fmt.Errorf("%s: %w", where, err)
	}

	name := strings.TrimSpace(meta["name"])
	if name == "" {
		// Fall back to the containing directory. The on-disk layout already
		// names the skill (<root>/pdf-forms/SKILL.md), so requiring the author
		// to repeat it in the frontmatter buys nothing but a chance to disagree
		// with itself.
		if sourcePath != "" {
			name = filepath.Base(filepath.Dir(sourcePath))
		}
		if name == "" || name == "." || name == string(filepath.Separator) {
			return Skill{}, fmt.Errorf("%s: %w", where, ErrNoName)
		}
	}

	desc := strings.TrimSpace(meta["description"])
	if desc == "" {
		return Skill{}, fmt.Errorf("%s: %w (skill %q)", where, ErrNoDescription, name)
	}

	// Anything other than an explicit true leaves the skill loadable, which is
	// the conservative default: a typo in the frontmatter must not silently
	// move a large body into every request's cache prefix.
	install := strings.EqualFold(strings.TrimSpace(meta["install"]), "true")

	return Skill{
		Name:        name,
		Description: desc,
		Body:        strings.TrimSpace(body),
		Source:      sourcePath,
		Install:     install,
	}, nil
}

// ParseFile reads and parses one SKILL.md.
func ParseFile(path string) (Skill, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Skill{}, fmt.Errorf("skill: read %s: %w", path, err)
	}
	return Parse(string(b), path)
}

// LoadDir scans root for <name>/SKILL.md and returns the skills it could
// parse, sorted by name.
//
// A file that fails to parse is reported through onError and skipped. One
// broken skill must not stall startup: skills are user-authored content
// dropped into a directory, and refusing to start the agent because the
// seventeenth one has a typo punishes the wrong person. onError may be nil.
//
// A missing root is NOT an error — it means "no skills configured", which is
// the common case for an agent that ships without any. A root that exists but
// cannot be read IS an error, because that is a misconfiguration the operator
// wants to hear about.
func LoadDir(root string, onError func(error)) ([]Skill, error) {
	report := func(err error) {
		if onError != nil {
			onError(err)
		}
	}

	entries, err := os.ReadDir(root)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("skill: scan %s: %w", root, err)
	}

	out := make([]Skill, 0, len(entries))
	seen := make(map[string]string, len(entries))

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		path := filepath.Join(root, e.Name(), FileName)
		if _, statErr := os.Stat(path); statErr != nil {
			// A directory without a SKILL.md is not a broken skill, it is not
			// a skill at all (README dirs, .git, scratch space). Silent.
			continue
		}
		s, perr := ParseFile(path)
		if perr != nil {
			report(fmt.Errorf("skill: skipping %s: %w", path, perr))
			continue
		}
		// Two directories can declare the same frontmatter name. Report and
		// skip rather than panic: unlike [New], this input is user content,
		// and the same "do not stall startup" rule applies.
		if prev, dup := seen[s.Name]; dup {
			report(fmt.Errorf("skill: skipping %s: duplicate skill name %q (already loaded from %s)", path, s.Name, prev))
			continue
		}
		seen[s.Name] = path
		out = append(out, s)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// splitFrontmatter separates the leading --- block from the body.
func splitFrontmatter(text string) (map[string]string, string, error) {
	// Normalize CRLF once, here, so every downstream comparison ("---", "|")
	// can be exact rather than tolerant.
	text = strings.ReplaceAll(text, "\r\n", "\n")
	// An editor-written BOM would make the first line "<BOM>---" and fail the
	// delimiter check for a reason nobody can see in a diff.
	text = strings.TrimPrefix(text, "\ufeff")

	lines := strings.Split(text, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != delim {
		return nil, "", ErrNoFrontmatter
	}

	end := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == delim {
			end = i
			break
		}
	}
	if end < 0 {
		return nil, "", fmt.Errorf("%w: no closing %q", ErrNoFrontmatter, delim)
	}

	meta, err := parseKeys(lines[1:end])
	if err != nil {
		return nil, "", err
	}
	return meta, strings.Join(lines[end+1:], "\n"), nil
}

// parseKeys reads the fixed YAML subset: scalars and block scalars.
//
// Last value wins for a repeated key, matching every YAML implementation that
// does not error on duplicates. We do not error: see [Parse] on unknown keys.
func parseKeys(lines []string) (map[string]string, error) {
	out := make(map[string]string, 4)

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		if strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}

		k, v, found := strings.Cut(line, ":")
		if !found {
			return nil, fmt.Errorf("%w: %q", ErrBadFrontmatter, line)
		}
		key := strings.TrimSpace(k)
		if key == "" {
			return nil, fmt.Errorf("%w: empty key in %q", ErrBadFrontmatter, line)
		}
		rest := strings.TrimSpace(v)

		if rest != "|" && rest != ">" {
			out[key] = unquote(rest)
			continue
		}

		// Block scalar. Every real SKILL.md writes its description this way,
		// so the subset has to cover it even though the task only needs two
		// scalars. Continuation lines are the indented run that follows;
		// consuming them here is also what keeps them from tripping the
		// "no colon" check above.
		block, next := readBlock(lines, i+1)
		if rest == ">" {
			block = strings.ReplaceAll(block, "\n", " ")
		}
		out[key] = strings.TrimRight(block, " \n")
		i = next - 1
	}
	return out, nil
}

// readBlock collects the indented continuation of a block scalar starting at
// start, returning the dedented text and the index of the first line after it.
func readBlock(lines []string, start int) (string, int) {
	indent := -1
	var b strings.Builder
	wrote := false

	i := start
	for ; i < len(lines); i++ {
		ln := lines[i]
		if strings.TrimSpace(ln) == "" {
			// A blank line belongs to the block only if the block has started;
			// leading blanks are separators.
			if wrote {
				b.WriteByte('\n')
			}
			continue
		}
		lead := len(ln) - len(strings.TrimLeft(ln, " \t"))
		if indent < 0 {
			indent = max(lead, 1)
		}
		if lead < indent {
			break
		}
		if wrote {
			b.WriteByte('\n')
		}
		b.WriteString(ln[indent:])
		wrote = true
	}
	return b.String(), i
}

// unquote strips one matched pair of surrounding quotes. YAML treats 'x' and
// "x" as the string x, and authors write both; leaving the quotes in would put
// them in the system prompt.
func unquote(s string) string {
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

// firstLine returns the first non-empty line, trimmed. The index gets one line
// per skill; a multi-line description pasted in whole would break the format
// and inflate the cached prefix for no gain, since the model only needs enough
// to decide whether to load.
func firstLine(s string) string {
	for ln := range strings.SplitSeq(s, "\n") {
		if t := strings.TrimSpace(ln); t != "" {
			return t
		}
	}
	return ""
}

// sortedNames returns the skill names in a stable order.
func sortedNames(skills []Skill) []string {
	names := make([]string, len(skills))
	for i, s := range skills {
		names[i] = s.Name
	}
	slices.Sort(names)
	return names
}
