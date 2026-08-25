package judge

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/rl"
)

// LLM is a model-based judge.
//
// It answers the same question [Rules] answers, over the same anchors, and
// returns the same [Assessment]. That is the entire design constraint and it
// is not decoration: a model judge nobody can check is a scalar with extra
// steps, and the only cheap way to check one is to run it against a
// deterministic judge on the criteria both can reach and count where they
// disagree. See [Agreement].
//
// It exists for the three criteria a rule cannot settle — correctness,
// reasoning quality, user alignment — and is worth running on the other six
// only to calibrate it.
type LLM struct {
	Client    llm.Client
	Model     string
	MaxTokens int

	// Retries is how many times a malformed or invalid verdict is asked for
	// again. Default 1.
	//
	// A retry rather than a repair: a verdict that failed validation is not
	// patched up locally, because patching it means the pipeline inventing
	// the part the judge would not supply — usually the evidence, which is
	// the part that matters.
	Retries int

	// ObservationLimit caps how much of each tool result is shown. Zero uses
	// DefaultObservationLimit.
	ObservationLimit int

	// Label distinguishes this judge in a report when several models are on
	// the panel.
	Label string

	// JudgeCorrectness lets this judge rule on [Correctness]. Off by default,
	// and the default is a calibration result rather than caution.
	//
	// Against a trajectory with a verified spec deviation, a model judge
	// scored correctness 5 at confidence 1.00 and cited the agent's own
	// functional test as its evidence — a test that ran inside the workspace
	// and so could not have detected the deviation either way. A transcript
	// reader inherits the agent's blind spots, because the only verification
	// in the transcript is the agent's.
	//
	// Correctness belongs to [Oracle], which reads verifiers that ran outside
	// the trajectory. Turn this on only to MEASURE the gap between the two,
	// never to fill it.
	JudgeCorrectness bool
}

// DefaultObservationLimit is how much of one observation reaches the judge.
const DefaultObservationLimit = 1200

// Name implements Judge.
func (j LLM) Name() string {
	if j.Label != "" {
		return j.Label
	}
	if j.Model != "" {
		return "llm:" + j.Model
	}
	return "llm"
}

// Assess implements Judge.
func (j LLM) Assess(ctx context.Context, ep *rl.Episode, r Rubric) (Assessment, error) {
	// The refusal comes FIRST, ahead of even the nil-client check. A policy
	// that only takes effect once a client happens to be configured is not a
	// policy, it is a filter — and it would let a panel that forgot to set one
	// report "no client" for a criterion this judge must decline on principle.
	if r.Criterion == Correctness && !j.JudgeCorrectness {
		return Assessment{}, fmt.Errorf("%w: correctness comes from the oracle, not from reading the transcript "+
			"(set JudgeCorrectness to measure the gap on purpose)", ErrNotRulable)
	}
	if j.Client == nil {
		return Assessment{}, fmt.Errorf("judge: LLM judge has no client")
	}
	if ep == nil {
		return Assessment{}, fmt.Errorf("judge: nil episode")
	}
	switch ep.Failure {
	case rl.Panicked, rl.Cancelled:
		return Assessment{}, ErrUnjudgeable
	}

	tries := j.Retries
	if tries < 0 {
		tries = 0
	}
	if j.Retries == 0 {
		tries = 1
	}

	prompt := j.render(ep, r)
	maxTokens := j.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}

	var last error
	for attempt := 0; attempt <= tries; attempt++ {
		msg := prompt
		if last != nil {
			// Tell it what was wrong. A judge that is simply asked again
			// produces the same malformed answer again.
			msg += fmt.Sprintf("\n\nYour previous reply was rejected: %s\nReturn only the JSON object.", last)
		}
		resp, err := j.Client.Complete(ctx, llm.Request{
			System:    judgeSystem,
			Messages:  []llm.Message{llm.UserText(msg)},
			Model:     j.Model,
			MaxTokens: maxTokens,
			Purpose:   llm.Purpose("judge"),
		})
		if err != nil {
			return Assessment{}, fmt.Errorf("judge: %s: %w", j.Name(), err)
		}

		a, perr := parseVerdict(llm.TextOf(resp.Content), r)
		if perr != nil {
			last = perr
			continue
		}
		a.Judge = j.Name()
		if verr := a.Validate(r); verr != nil {
			last = verr
			continue
		}
		if a.Repair == NoRepair && a.Failure != None {
			a.Repair = RepairFor(a.Failure)
		}
		return a, nil
	}
	return Assessment{}, fmt.Errorf("judge: %s produced no valid verdict for %s: %w", j.Name(), r.Criterion, last)
}

// criterionGuidance is the extra instruction a dimension needs beyond its
// anchors.
//
// Kept per criterion rather than in the system prompt because each of these is
// a correction to one specific way judges go wrong, and a general prompt that
// carried all of them would dilute every one of them.
var criterionGuidance = map[Criterion]string{
	Faithfulness: faithfulnessGuidance,
}

// faithfulnessGuidance exists because of a measured failure.
//
// On a labelled pair of trajectories whose answers made the SAME claim and
// differed only in whether it was true, a model judge scored both 5. Reading
// its evidence showed why: it was checking each claim for CONSISTENCY with
// what was observed, and an untested claim is consistent with everything. In
// the true case it even cited the discriminating observation; it simply did
// not treat that observation's ABSENCE in the other case as meaning anything.
//
// So the instruction reframes the question from "is this claim consistent"
// to "could this claim have been caught if it were wrong", which is the
// question that separates the pair.
const faithfulnessGuidance = `Work from the TASK REQUIREMENTS, not from the answer's emphasis.

The answer decides what to highlight, and a claim stated in passing is exactly
as much a claim as one under its own heading. Judging by what the answer chose
to stress means an agent can lower the bar by mentioning its weakest point
briefly.

So:

1. List every explicit requirement in the task instruction above. Include the
   ones phrased as constraints ("must", "not", "instead of").
2. For each one, find the observation in the trajectory that CHECKS it.

   The test is not whether the requirement is consistent with what was
   observed — an untested requirement is consistent with everything. The test
   is:

     Would that observation have come out DIFFERENTLY if the agent had got
     this requirement wrong?

   - If yes, the requirement is verified. Cite that observation.
   - If no, it is UNVERIFIED, however plausible the implementation looks and
     however confidently the answer asserts it. Reading the agent's own source,
     or its own summary, is not an observation about behaviour.

3. Score on the requirements, not on the prose. If ANY explicit requirement
   went unverified, the answer is asserting more than the run established and
   cannot score above the middle level — even when every other requirement is
   thoroughly checked.

Watch for a check performed under the same conditions that produced the
artefact: it cannot detect a mistake shared by both. Exercising a program only
where it was built says nothing about how it behaves elsewhere; a passing build
says nothing about runtime.`

const judgeSystem = "You are an evaluation judge. You grade one dimension of one agent " +
	"trajectory against a fixed rubric. You quote only what the trajectory literally contains, " +
	"you never infer an observation that is not shown to you, and you return one JSON object and " +
	"nothing else."

func (j LLM) render(ep *rl.Episode, r Rubric) string {
	var b strings.Builder

	fmt.Fprintf(&b, "## Dimension\n%s — %s\n\n", r.Criterion, r.Question)

	b.WriteString("## Scale\nChoose exactly one of these scores.\n")
	for _, l := range r.Levels {
		fmt.Fprintf(&b, "%d: %s\n", l.Score, l.Anchor)
	}
	fmt.Fprintf(&b, "\nA score of %d or above passes.\n\n", r.PassAt)

	fmt.Fprintf(&b, "## Task\nid: %s\ninstruction: %s\n\n", ep.Task.ID, ep.Task.Prompt)

	b.WriteString("## Trajectory\n")
	b.WriteString(j.renderTrajectory(ep))

	if g := criterionGuidance[r.Criterion]; g != "" {
		b.WriteString("\n## How to test this dimension\n")
		b.WriteString(g)
		b.WriteString("\n")
	}

	b.WriteString("\n## Failure categories\nIf the score does not pass, pick exactly one:\n")
	for _, f := range All {
		fmt.Fprintf(&b, "- %s\n", f)
	}

	b.WriteString(`
## Output
Return one JSON object, nothing before or after:

{
  "score": <one of the scores above>,
  "evidence": [{"step": <turn number, or -1 for the run as a whole>, "note": "<what you saw>"}],
  "failure": "<a category above, or \"\" when the score passes>",
  "rationale": "<one or two sentences>",
  "confidence": <0.0 to 1.0>
}

Rules:
- Evidence is required for any score below the top of the scale. If you cannot
  point at something in the trajectory above, the score is the top one.
- Quote only what is printed above. Do not assume a command did what its name
  suggests; look at what came back.
- confidence is how sure you are given ONLY what you were shown. Low confidence
  is a legitimate answer and is more useful than a confident guess.
- Judge this dimension alone. Other judges are scoring the others.`)

	return b.String()
}

func (j LLM) renderTrajectory(ep *rl.Episode) string {
	limit := j.ObservationLimit
	if limit <= 0 {
		limit = DefaultObservationLimit
	}

	var b strings.Builder
	results := map[llm.ToolUseID]llm.ToolResult{}
	for _, m := range ep.Messages {
		for _, blk := range m.Content {
			if tr, ok := blk.(llm.ToolResult); ok {
				results[tr.ToolUseID] = tr
			}
		}
	}

	turn := 0
	for _, m := range ep.Messages {
		switch m.Role {
		case llm.RoleUser:
			if txt := strings.TrimSpace(llm.TextOf(m.Content)); txt != "" {
				fmt.Fprintf(&b, "\n[user] %s\n", txt)
			}
			continue
		case llm.RoleAssistant:
			turn++
		}
		if txt := strings.TrimSpace(llm.TextOf(m.Content)); txt != "" {
			fmt.Fprintf(&b, "\nturn %d [agent] %s\n", turn, txt)
		}
		for _, blk := range m.Content {
			tu, ok := blk.(llm.ToolUse)
			if !ok {
				continue
			}
			fmt.Fprintf(&b, "turn %d [call] %s %s\n", turn, tu.Name, clip(string(tu.Input), 400))
			if tr, ok := results[tu.ID]; ok {
				tag := "result"
				if tr.IsError {
					tag = "ERROR"
				}
				fmt.Fprintf(&b, "turn %d [%s] %s\n", turn, tag, clip(tr.Content, limit))
			} else {
				fmt.Fprintf(&b, "turn %d [result] (this call was never answered)\n", turn)
			}
		}
	}

	fmt.Fprintf(&b, "\n[run ended] %s", ep.Failure)
	if ep.Err != nil {
		fmt.Fprintf(&b, " (%s)", ep.Err)
	}
	b.WriteString("\n")
	return b.String()
}

// clip keeps the head and the tail, because command output is back-loaded:
// the compiler errors and the failing assertion are at the end, and a judge
// handed only the front is grading the part that always looks fine.
func clip(s string, limit int) string {
	if limit <= 0 || len(s) <= limit {
		return s
	}
	half := limit / 2
	return s[:half] + fmt.Sprintf("\n… %d bytes omitted …\n", len(s)-limit) + s[len(s)-half:]
}

type verdictJSON struct {
	Score      int        `json:"score"`
	Evidence   []Evidence `json:"evidence"`
	Failure    string     `json:"failure"`
	Rationale  string     `json:"rationale"`
	Confidence float64    `json:"confidence"`
}

// parseVerdict pulls the JSON object out of a reply that may have prose around
// it, and refuses anything the rubric does not recognise.
func parseVerdict(text string, r Rubric) (Assessment, error) {
	start := strings.IndexByte(text, '{')
	end := strings.LastIndexByte(text, '}')
	if start < 0 || end <= start {
		return Assessment{}, fmt.Errorf("no JSON object in the reply")
	}
	var v verdictJSON
	if err := json.Unmarshal([]byte(text[start:end+1]), &v); err != nil {
		return Assessment{}, fmt.Errorf("the JSON did not parse: %w", err)
	}
	if r.Anchor(v.Score) == "" {
		return Assessment{}, fmt.Errorf("score %d is not a level on this scale", v.Score)
	}

	f := Failure(strings.TrimSpace(v.Failure))
	if !Valid(f) {
		return Assessment{}, fmt.Errorf("%q is not one of the failure categories", v.Failure)
	}

	a := Assessment{
		Criterion:    r.Criterion,
		Score:        v.Score,
		Pass:         v.Score >= r.PassAt,
		Evidence:     v.Evidence,
		Failure:      f,
		Rationale:    strings.TrimSpace(v.Rationale),
		Confidence:   v.Confidence,
		FirstBadStep: -1,
	}
	if a.Pass {
		a.Failure = None
	}
	for _, e := range a.Evidence {
		if e.Step >= 0 && (a.FirstBadStep < 0 || e.Step < a.FirstBadStep) {
			a.FirstBadStep = e.Step
		}
	}
	if a.Confidence == 0 {
		// An omitted confidence is not zero confidence. Zero would mean the
		// judge disowns its own verdict, and treating a missing field that way
		// would let one lazy reply drag an aggregate to nothing.
		a.Confidence = 0.5
	}
	return a, nil
}

// ===== calibration =====

// Agreement compares two judges over the same episodes.
//
// This is what makes a model judge admissible. Run it alongside [Rules] on the
// criteria rules can reach: exact agreement says the model has learned the
// anchors, and the disagreements are the sample worth reading by hand. A model
// judge shipped without this number is an opinion with a JSON schema.
type Agreement struct {
	Criterion Criterion

	N        int
	Exact    int // same score
	SamePass int // same pass/fail verdict, whatever the score

	// MeanAbsDelta is the average distance between the two scores, which
	// separates "off by one level" from "opposite verdict".
	MeanAbsDelta float64

	// Disagreements are the episodes to read.
	Disagreements []Disagreement
}

// Disagreement is one episode two judges scored differently.
type Disagreement struct {
	Task   string
	Sample int
	A, B   Assessment
}

// ExactRate is the share of episodes the two judges scored identically.
func (a Agreement) ExactRate() float64 {
	if a.N == 0 {
		return 0
	}
	return float64(a.Exact) / float64(a.N)
}

// PassRate is the share where they agreed on pass/fail, which is the weaker
// and more forgiving measure — and the one that actually gates a decision.
func (a Agreement) PassAgreement() float64 {
	if a.N == 0 {
		return 0
	}
	return float64(a.SamePass) / float64(a.N)
}

// Compare measures agreement between two corpora over one criterion.
//
// The corpora must be over the same episodes in the same order; anything that
// does not line up by (task, sample) is skipped rather than compared, because
// a silent misalignment would report agreement between unrelated runs.
func Compare(c Criterion, a, b Corpus) Agreement {
	index := map[string]*Report{}
	for _, r := range b {
		index[fmt.Sprintf("%s#%d", r.Task, r.Sample)] = r
	}

	out := Agreement{Criterion: c}
	var sumDelta float64
	for _, ra := range a {
		rb, ok := index[fmt.Sprintf("%s#%d", ra.Task, ra.Sample)]
		if !ok {
			continue
		}
		aa, ok1 := ra.Of(c)
		ab, ok2 := rb.Of(c)
		if !ok1 || !ok2 {
			continue
		}
		out.N++
		d := aa.Score - ab.Score
		if d < 0 {
			d = -d
		}
		sumDelta += float64(d)
		if aa.Score == ab.Score {
			out.Exact++
		} else {
			out.Disagreements = append(out.Disagreements, Disagreement{
				Task: ra.Task, Sample: ra.Sample, A: aa, B: ab,
			})
		}
		if aa.Pass == ab.Pass {
			out.SamePass++
		}
	}
	if out.N > 0 {
		out.MeanAbsDelta = sumDelta / float64(out.N)
	}
	return out
}
