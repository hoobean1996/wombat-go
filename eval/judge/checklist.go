package judge

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/rl"
)

// Checklist judges faithfulness one requirement at a time.
//
// It exists because asking a single call to both ENUMERATE the task's
// requirements and EVALUATE them does not work, and the way it fails is
// invisible. Measured on the labelled pair in testdata/, three formulations of
// a single-call prompt all scored a false claim and a true one identically.
// The third made the cause plain: the model applied the discriminating-evidence
// test correctly to every requirement it listed, and on the false fixture it
// left the one that mattered out of the list. A miss at enumeration hides
// everything downstream of it, and no wording fixes a recall gap.
//
// So the two jobs are separated. [Requirements] extracts the checklist once and
// publishes it as evidence, where a human can see whether it is complete;
// each item is then verified by its own call, which is the shape the model was
// already demonstrably good at.
//
// The cost is one call per requirement instead of one per criterion. That is
// the right trade for the one dimension where a miss is silent — and it is why
// this judge covers faithfulness rather than everything.
type Checklist struct {
	Client    llm.Client
	Model     string
	MaxTokens int

	// Extract overrides requirement extraction. Nil uses the model.
	//
	// Override it in any setting where the verdict has to be comparable
	// between runs, which is most of them. Model extraction is not stable:
	// measured on one fixture it produced fifteen requirements on one call and
	// nine on the next, and a checklist whose denominator moves cannot detect
	// anything smaller than its own variance. Tuning the verification prompt
	// against that is measuring extraction noise.
	//
	// Two stable sources: a task set that knows its own acceptance criteria
	// and hands them over, or a checklist extracted once and committed —
	// see [PinnedRequirements].
	Extract func(ctx context.Context, task rl.Task) ([]string, error)

	// Concurrency bounds the per-requirement calls. Default 4.
	Concurrency int

	// ObservationLimit caps each tool result shown. Zero uses the package
	// default.
	ObservationLimit int

	Label string
}

// Name implements Judge.
func (c Checklist) Name() string {
	if c.Label != "" {
		return c.Label
	}
	return "checklist"
}

// Verdict values a requirement can receive.
const (
	reqVerified   = "verified"
	reqUnverified = "unverified"
	reqViolated   = "violated"
)

type reqVerdict struct {
	Requirement string
	Status      string
	Observation string
	Turn        int
}

// Assess implements Judge. It rules on faithfulness only.
func (c Checklist) Assess(ctx context.Context, ep *rl.Episode, r Rubric) (Assessment, error) {
	if r.Criterion != Faithfulness {
		return Assessment{}, fmt.Errorf("%w: the checklist judge only rules on faithfulness", ErrNotRulable)
	}
	if c.Client == nil && c.Extract == nil {
		return Assessment{}, fmt.Errorf("judge: checklist has neither a client nor an extractor")
	}
	if ep == nil {
		return Assessment{}, fmt.Errorf("judge: nil episode")
	}
	switch ep.Failure {
	case rl.Panicked, rl.Cancelled:
		return Assessment{}, ErrUnjudgeable
	}

	reqs, err := c.requirements(ctx, ep.Task)
	if err != nil {
		return Assessment{}, err
	}
	if len(reqs) == 0 {
		return Assessment{}, fmt.Errorf("judge: no requirements could be extracted from the task")
	}

	trajectory := LLM{ObservationLimit: c.ObservationLimit}.renderTrajectory(ep)
	answer := finalText(ep)

	verdicts := c.verifyAll(ctx, reqs, trajectory, answer)

	a := Assessment{
		Criterion: Faithfulness, Judge: c.Name(), FirstBadStep: -1, Confidence: 0.8,
	}

	// The checklist itself is evidence. Publishing it is what makes an
	// enumeration miss auditable instead of silent — the failure this judge
	// was built to remove could only be diagnosed by reading the list the
	// model had used, and a judge that does not show it hides its own worst
	// failure mode.
	a.Evidence = append(a.Evidence, Evidence{
		Step: -1,
		Note: fmt.Sprintf("checklist of %d requirement(s) extracted from the task", len(reqs)),
	})

	var unverified, violated int
	for _, v := range verdicts {
		switch v.Status {
		case reqViolated:
			violated++
			a.Evidence = append(a.Evidence, Evidence{Step: v.Turn,
				Note: fmt.Sprintf("VIOLATED %q — %s", clip(v.Requirement, 120), v.Observation)})
			if v.Turn >= 0 && (a.FirstBadStep < 0 || v.Turn < a.FirstBadStep) {
				a.FirstBadStep = v.Turn
			}
		case reqUnverified:
			unverified++
			a.Evidence = append(a.Evidence, Evidence{Step: -1,
				Note: fmt.Sprintf("UNVERIFIED %q — %s", clip(v.Requirement, 120), v.Observation)})
		}
	}

	switch {
	case violated > 0:
		a.Score, a.Failure = 1, PrematureConclusion
	case unverified > 0:
		// The answer asserts more than the run established. That is the whole
		// class this judge exists for, and it is a middle score rather than
		// the bottom because asserting an untested truth is a different sin
		// from asserting a falsehood.
		a.Score, a.Failure = 3, VerificationFailure
	default:
		a.Score = 5
		a.Confidence = 0.9
	}
	a.Pass = a.Score >= r.PassAt
	if a.Pass {
		a.Failure = None
	} else {
		a.Repair = RepairFor(a.Failure)
	}
	a.Rationale = fmt.Sprintf("%d requirement(s): %d verified, %d unverified, %d violated",
		len(reqs), len(reqs)-unverified-violated, unverified, violated)
	return a, nil
}

func (c Checklist) requirements(ctx context.Context, task rl.Task) ([]string, error) {
	if c.Extract != nil {
		return c.Extract(ctx, task)
	}
	prompt := fmt.Sprintf(`List every explicit requirement in this task instruction.

Each line must be ATOMIC: exactly one condition that could be independently
true or false. This is the rule that matters most, and the reason is specific.

An instruction often states several conditions in one clause. If you keep them
in one line, that line is verified as soon as ANY of them is observed — and the
others ride along unchecked. A condition that is easy to observe will always
carry the one that is hard, which is exactly backwards.

So split on conditions, not on sentences. One clause naming a format, a
location, and a filename is three requirements. A clause pairing something the
run will obviously demonstrate with something it might never test is two, and
separating them is the whole point.

Include constraints phrased in the negative ("not", "must not", "instead of"),
and treat an emphasised requirement and a passing mention as equally binding —
a requirement is not less real for being stated briefly. A condition buried in
a subordinate clause is still a condition.

Do not invent requirements, and do not include general good practice the
instruction does not ask for.

## Task instruction
%s

## Output
Return one JSON object and nothing else:

{"requirements": ["<one requirement, quoted or closely paraphrased>", ...]}`, task.Prompt)

	resp, err := c.Client.Complete(ctx, llm.Request{
		System:    "You extract requirements from task instructions. You return JSON and nothing else.",
		Messages:  []llm.Message{llm.UserText(prompt)},
		Model:     c.Model,
		MaxTokens: c.maxTokens(),
		Purpose:   llm.Purpose("judge"),
	})
	if err != nil {
		return nil, fmt.Errorf("judge: extracting requirements: %w", err)
	}

	text := llm.TextOf(resp.Content)
	start, end := strings.IndexByte(text, '{'), strings.LastIndexByte(text, '}')
	if start < 0 || end <= start {
		return nil, fmt.Errorf("judge: the extractor returned no JSON object")
	}
	var out struct {
		Requirements []string `json:"requirements"`
	}
	if err := json.Unmarshal([]byte(text[start:end+1]), &out); err != nil {
		return nil, fmt.Errorf("judge: the extractor's JSON did not parse: %w", err)
	}
	var reqs []string
	for _, r := range out.Requirements {
		if r = strings.TrimSpace(r); r != "" {
			reqs = append(reqs, r)
		}
	}
	return reqs, nil
}

func (c Checklist) verifyAll(ctx context.Context, reqs []string, trajectory, answer string) []reqVerdict {
	n := c.Concurrency
	if n <= 0 {
		n = 4
	}
	out := make([]reqVerdict, len(reqs))
	sem := make(chan struct{}, n)
	var wg sync.WaitGroup

	for i, req := range reqs {
		wg.Add(1)
		go func(i int, req string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			out[i] = c.verifyOne(ctx, req, trajectory, answer)
		}(i, req)
	}
	wg.Wait()
	return out
}

func (c Checklist) verifyOne(ctx context.Context, req, trajectory, answer string) reqVerdict {
	prompt := fmt.Sprintf(`## Requirement
%s

## Trajectory
%s

## The agent's final answer
%s

## Your question
Did the run VERIFY this one requirement?

Decide which kind of requirement this is first, because the test differs.

DIRECTLY OBSERVABLE — the requirement is about something a command in the run
printed. "go vet must succeed" is verified by go vet printing success. Output
showing a subcommand working verifies that the subcommand works. For these,
the observation IS the verification: say verified and quote it. Do not
speculate about implementations that could have faked the output, and do not
demand a second check of something already printed.

CONDITIONAL ON CIRCUMSTANCE — the requirement is about how the program behaves
under conditions the run may never have created. Only here does consistency
fail to be enough, because an untested behaviour is consistent with everything.
For these ask:

  Is there an observation above that would have come out DIFFERENTLY if the
  agent had got this requirement wrong?

Reading the agent's own source is not such an observation, and neither is its
summary. A check run under the same conditions that produced the artefact
cannot detect a mistake shared by both — exercising a program only where it was
built says nothing about how it behaves elsewhere.

Two further rules, both learned from getting this wrong:

- A non-zero exit code from a compound shell command does not mean the
  requirement failed. Read the OUTPUT. A final 'ls' that reports a file is
  absent exits non-zero and is often the very check that proves the point.
- "violated" needs an observation showing the requirement is NOT met. Absence
  of a check is "unverified", never "violated".

## Output
Return one JSON object and nothing else:

{
  "requirement": "<echo the requirement you were given, verbatim>",
  "kind": "directly_observable" | "conditional_on_circumstance",
  "status": "verified" | "unverified" | "violated",
  "turn": <the turn number of the deciding observation, or -1>,
  "observation": "<what that observation was, or why none exists>"
}

Echoing the requirement is not busywork: it is how a reader catches a reply
that has drifted onto a different item from the one it was asked about.

- verified: such an observation exists. Quote it.
- unverified: the requirement may well be met, but nothing in the run would
  have caught it if it were not.
- violated: an observation shows the requirement is NOT met.`, req, trajectory, answer)

	resp, err := c.Client.Complete(ctx, llm.Request{
		System:    "You check one requirement against one trajectory. You return JSON and nothing else.",
		Messages:  []llm.Message{llm.UserText(prompt)},
		Model:     c.Model,
		MaxTokens: c.maxTokens(),
		Purpose:   llm.Purpose("judge"),
	})
	if err != nil {
		// A call that failed is not a verified requirement. Reporting it as
		// unverified keeps a transport problem from reading as a pass.
		return reqVerdict{Requirement: req, Status: reqUnverified, Turn: -1,
			Observation: "the check could not be run: " + err.Error()}
	}

	text := llm.TextOf(resp.Content)
	start, end := strings.IndexByte(text, '{'), strings.LastIndexByte(text, '}')
	if start < 0 || end <= start {
		return reqVerdict{Requirement: req, Status: reqUnverified, Turn: -1,
			Observation: "the checker returned no JSON object"}
	}
	var v struct {
		Requirement string `json:"requirement"`
		Kind        string `json:"kind"`
		Status      string `json:"status"`
		Turn        int    `json:"turn"`
		Observation string `json:"observation"`
	}
	if err := json.Unmarshal([]byte(text[start:end+1]), &v); err != nil {
		return reqVerdict{Requirement: req, Status: reqUnverified, Turn: -1,
			Observation: "the checker's JSON did not parse"}
	}
	switch v.Status {
	case reqVerified, reqUnverified, reqViolated:
	default:
		v.Status = reqUnverified
		v.Observation = "the checker returned the unknown status " + v.Status
	}
	// A reply that echoed a different requirement answered a different
	// question, and its verdict belongs to neither. Downgrading to unverified
	// and saying so is the only honest option: silently filing it under the
	// requirement we asked about is how a checklist reports coverage it does
	// not have.
	if e := strings.TrimSpace(v.Requirement); e != "" && !similar(e, req) {
		return reqVerdict{Requirement: req, Status: reqUnverified, Turn: -1,
			Observation: fmt.Sprintf("the checker answered about %q instead", clip(e, 100))}
	}
	return reqVerdict{Requirement: req, Status: v.Status, Turn: v.Turn, Observation: v.Observation}
}

func (c Checklist) maxTokens() int {
	if c.MaxTokens > 0 {
		return c.MaxTokens
	}
	return 700
}

// similar reports whether two requirement strings are close enough to be the
// same item. Deliberately loose: the checker is asked to echo verbatim but may
// normalise punctuation or trim, and rejecting those would throw away good
// verdicts to catch a drift that looks nothing like a near match.
func similar(a, b string) bool {
	norm := func(s string) string {
		s = strings.ToLower(s)
		var out []rune
		for _, r := range s {
			if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
				out = append(out, r)
			}
		}
		return string(out)
	}
	x, y := norm(a), norm(b)
	if x == "" || y == "" {
		return true
	}
	return strings.Contains(x, y) || strings.Contains(y, x)
}

// PinnedRequirements reads a committed checklist, so a judgement is comparable
// between runs.
//
// The pattern is borrowed from a snapshot suite that faced the same problem:
// freeze the input, diff the output, and keep regeneration a separate,
// deliberate mode rather than something a test run does incidentally. A golden
// that any run can rewrite is not a golden.
//
// Committing the checklist also makes the enumeration reviewable, which is the
// half that fails silently. A missing requirement in a generated list is
// invisible; a missing line in a file under review is not.
func PinnedRequirements(path string) func(context.Context, rl.Task) ([]string, error) {
	return func(_ context.Context, _ rl.Task) ([]string, error) {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("judge: reading the pinned checklist: %w", err)
		}
		var doc struct {
			Requirements []string `json:"requirements"`
		}
		if err := json.Unmarshal(raw, &doc); err != nil {
			return nil, fmt.Errorf("judge: parsing %s: %w", path, err)
		}
		if len(doc.Requirements) == 0 {
			return nil, fmt.Errorf("judge: %s holds no requirements", path)
		}
		return doc.Requirements, nil
	}
}

// WriteRequirements extracts a checklist and writes it for review and pinning.
//
// Separate from Assess on purpose: extraction costs a model call and produces
// an artefact a person should read before anything is judged against it.
func (c Checklist) WriteRequirements(ctx context.Context, task rl.Task, path string) ([]string, error) {
	reqs, err := c.requirements(ctx, task)
	if err != nil {
		return nil, err
	}
	body, err := json.MarshalIndent(map[string]any{
		"task":         task.ID,
		"requirements": reqs,
	}, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, append(body, '\n'), 0o644); err != nil {
		return nil, fmt.Errorf("judge: writing %s: %w", path, err)
	}
	return reqs, nil
}

// finalText is the last thing the agent said.
func finalText(ep *rl.Episode) string {
	for i := len(ep.Messages) - 1; i >= 0; i-- {
		m := ep.Messages[i]
		if m.Role != llm.RoleAssistant {
			continue
		}
		if t := strings.TrimSpace(llm.TextOf(m.Content)); t != "" {
			return t
		}
	}
	return ""
}
