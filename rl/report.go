package rl

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/llm"
)

// Report aggregates groups.
//
// Both renderings are deterministic: groups in the order they were added,
// failure kinds in [Kinds] order, episodes by sample. A report that reshuffles
// itself between runs cannot be diffed, and diffing two runs is the main thing
// anyone does with one.
type Report struct{ Groups []*Group }

// Add appends a group, ignoring nil so a caller can pass the result of a
// rollout that failed without branching first.
func (r *Report) Add(gs ...*Group) {
	for _, g := range gs {
		if g != nil {
			r.Groups = append(r.Groups, g)
		}
	}
}

// Episodes returns every episode across every group, in group then sample
// order.
func (r *Report) Episodes() []*Episode {
	var out []*Episode
	for _, g := range r.Groups {
		for _, ep := range g.Episodes {
			if ep != nil {
				out = append(out, ep)
			}
		}
	}
	return out
}

// Failures counts each [FailureKind] across every episode.
func (r *Report) Failures() map[FailureKind]int {
	out := make(map[FailureKind]int)
	for _, ep := range r.Episodes() {
		out[ep.Failure]++
	}
	return out
}

// Priced reports whether every episode in the report has a real cost. See
// [Episode.Priced]: when this is false the cost columns are not numbers to
// compare, and [Report.UnpricedModels] says who is responsible.
func (r *Report) Priced() bool {
	for _, g := range r.Groups {
		if !g.Priced() {
			return false
		}
	}
	return true
}

// UnpricedModels names every model the report could not price, sorted and
// deduplicated across groups.
func (r *Report) UnpricedModels() []string {
	var out []string
	for _, g := range r.Groups {
		out = append(out, g.UnpricedModels()...)
	}
	slices.Sort(out)
	return slices.Compact(out)
}

// Worst returns the lowest-scoring episode in the report, or nil when there
// are none. Ties break toward the earliest group and lowest sample, so the
// answer is stable.
func (r *Report) Worst() *Episode {
	var worst *Episode
	for _, ep := range r.Episodes() {
		if worst == nil || ep.Reward < worst.Reward {
			worst = ep
		}
	}
	return worst
}

// WriteText renders the table a human reads: one row per task, then the
// failure histogram across every episode, then the worst episode so it can be
// opened.
//
// pass@k is reported at k=n, n being the group's own sample count, because
// that is the largest k the samples support — see [Group.PassAt], which
// returns 0 above it rather than guessing.
//
// The COST column reads "n/a" for a group whose spend was not priced, and a
// line under the table names the models responsible. Tokens sit beside it and
// are always real, so a run against a model nobody has a rate for is still
// comparable against the run before it — which is the job the cost column was
// silently failing to do when it read $0.0000.
func (r *Report) WriteText(w io.Writer) error {
	bw := bufio.NewWriter(w)

	if len(r.Groups) == 0 {
		bw.WriteString("no groups\n")
		return bw.Flush()
	}

	// Two passes: measure, then print. Alignment is not decoration here — a
	// column of rewards that does not line up cannot be scanned for the
	// outlier, which is the only reason to print a table rather than JSON.
	const (
		hTask  = "TASK"
		hEnv   = "ENV"
		hN     = "N"
		hP1    = "PASS@1"
		hPK    = "PASS@K"
		hMean  = "MEAN"
		hStd   = "STD"
		hTurns = "TURNS"
		hTokIn = "TOK/IN"
		hTokOu = "TOK/OUT"
		hCost  = "COST"
	)
	rows := make([][]string, 0, len(r.Groups))
	for _, g := range r.Groups {
		n := len(g.Episodes)
		rows = append(rows, []string{
			g.TaskID,
			g.Env,
			fmt.Sprintf("%d", n),
			fmt.Sprintf("%.3f", g.PassAt(1)),
			fmt.Sprintf("%.3f", g.PassAt(n)),
			fmt.Sprintf("%.3f", g.Mean()),
			fmt.Sprintf("%.3f", g.Std()),
			fmt.Sprintf("%.1f", g.MedianTurns()),
			compactTokens(g.MedianPromptTokens()),
			compactTokens(g.MedianOutputTokens()),
			cost(g),
		})
	}
	head := []string{hTask, hEnv, hN, hP1, hPK, hMean, hStd, hTurns, hTokIn, hTokOu, hCost}
	// Task and env read left, every number reads right.
	leftAligned := []bool{true, true, false, false, false, false, false, false, false, false, false}

	width := make([]int, len(head))
	for i, h := range head {
		width[i] = len(h)
	}
	for _, row := range rows {
		for i, cell := range row {
			width[i] = max(width[i], len(cell))
		}
	}

	writeRow := func(cells []string) {
		for i, cell := range cells {
			if i > 0 {
				bw.WriteString("  ")
			}
			pad := width[i] - len(cell)
			if leftAligned[i] {
				bw.WriteString(cell)
				bw.WriteString(strings.Repeat(" ", pad))
			} else {
				bw.WriteString(strings.Repeat(" ", pad))
				bw.WriteString(cell)
			}
		}
		bw.WriteString("\n")
	}

	writeRow(head)
	for _, row := range rows {
		writeRow(row)
	}

	r.writeUnpriced(bw)
	r.writeHistogram(bw)
	r.writeSolvedButUnfinished(bw)

	if worst := r.Worst(); worst != nil {
		fmt.Fprintf(bw, "\nworst: %s  reward %.3f  %s\n", worst.Label(), worst.Reward, worst.Failure)
		if worst.Task.Workspace != "" {
			fmt.Fprintf(bw, "  %s\n", worst.Task.Workspace)
		}
		if worst.Err != nil {
			fmt.Fprintf(bw, "  %s\n", firstLine(worst.Err.Error()))
		}
	}

	return bw.Flush()
}

// writeUnpriced prints the one line that says why a COST column reads n/a.
//
// One line and always present when anything is unpriced, because the whole
// point is that the reader must not have to notice an absence. "n/a" in a
// column can be read past; a line naming some-gateway-model cannot.
func (r *Report) writeUnpriced(bw *bufio.Writer) {
	if r.Priced() {
		return
	}
	names := r.UnpricedModels()
	if len(names) == 0 {
		// Unpriced with nothing to name: the run spent tokens and reported no
		// cost, and its stream never said what answered.
		names = []string{"unnamed model"}
	}
	fmt.Fprintf(bw, "\nunpriced: %s — no rate for the above, so COST reads n/a; compare TOK/IN and TOK/OUT\n",
		strings.Join(names, ", "))
}

// writeHistogram prints the failure-kind counts across every episode.
//
// In [Kinds] order rather than by count: the shape of the histogram is what
// you compare between two runs, and a chart whose rows move when the numbers
// move cannot be compared at a glance. Kinds with no episodes are omitted,
// since a screen of zeroes hides the four rows that matter.
func (r *Report) writeHistogram(bw *bufio.Writer) {
	counts := r.Failures()
	total, widest, tallest := 0, 0, 0
	for _, k := range Kinds() {
		c := counts[k]
		total += c
		if c > 0 {
			widest = max(widest, len(k))
			tallest = max(tallest, c)
		}
	}
	if total == 0 {
		return
	}

	fmt.Fprintf(bw, "\nfailures (%d episodes)\n", total)
	const barWidth = 32
	for _, k := range Kinds() {
		c := counts[k]
		if c == 0 {
			continue
		}
		bars := c * barWidth / tallest
		fmt.Fprintf(bw, "  %-*s  %4d  %s\n", widest, string(k), c, strings.Repeat("#", bars))
	}
}

// ===== JSONL =====

// stepJSON is one turn on the wire.
type stepJSON struct {
	Iteration int      `json:"iteration"`
	Tools     []string `json:"tools"`
	Failed    []string `json:"failed"`
	Denied    []string `json:"denied"`
	Millis    int64    `json:"ms"`
	Usage     usageacc `json:"usage"`
}

// usageacc is llm.Usage with keys a trainer can read without knowing a
// provider's vocabulary.
type usageacc struct {
	Input      int `json:"input_tokens"`
	Output     int `json:"output_tokens"`
	CacheWrite int `json:"cache_write_tokens"`
	CacheRead  int `json:"cache_read_tokens"`
}

func toUsage(u llm.Usage) usageacc {
	return usageacc{u.InputTokens, u.OutputTokens, u.CacheWriteTokens, u.CacheReadTokens}
}

// episodeJSON is one line of the JSONL: one episode, whole.
//
// A struct rather than a map, because struct fields marshal in declaration
// order and map keys marshal sorted — and a golden test of a training file is
// only useful if the bytes are stable.
type episodeJSON struct {
	Env       string             `json:"env"`
	Task      string             `json:"task"`
	Sample    int                `json:"sample"`
	Prompt    string             `json:"prompt"`
	Workspace string             `json:"workspace,omitempty"`
	Reward    float64            `json:"reward"`
	Breakdown map[string]float64 `json:"breakdown,omitempty"`
	Failure   FailureKind        `json:"failure"`
	Success   bool               `json:"success"`

	Outcome   string          `json:"outcome"`
	Answer    string          `json:"answer,omitempty"`
	Tool      string          `json:"submitted_tool,omitempty"`
	Payload   json.RawMessage `json:"submitted_payload,omitempty"`
	Error     string          `json:"error,omitempty"`
	Turns     int             `json:"turns"`
	ToolCalls int             `json:"tool_calls"`
	ToolErrs  int             `json:"tool_errors"`
	Denials   int             `json:"denied"`

	WallMillis int64   `json:"wall_ms"`
	CostUSD    float64 `json:"cost_usd"`

	// Priced and Unpriced ride NEXT TO cost_usd rather than replacing it. A
	// downstream consumer that sums cost_usd over a corpus has to be able to
	// see that some of those zeroes are not zeroes — dropping the field would
	// hide that, and rounding it to null would break every reader that already
	// expects a number.
	Priced   bool     `json:"priced"`
	Unpriced []string `json:"unpriced_models"`

	Usage usageacc `json:"usage"`

	Steps    []stepJSON    `json:"steps"`
	Messages []llm.Message `json:"messages"`
}

// WriteJSONL writes one line per episode: the whole trajectory, for later use
// as training data.
//
// Self-describing on purpose. A trainer reads "reward", "success", "messages"
// and "steps" without a schema file, and nothing here is an index into a table
// it would have to be given separately — the cost of a few repeated strings
// per line against a format that is still readable in a year.
//
// HTML escaping is off, matching the rest of the project: a transcript is full
// of angle brackets and ampersands, and < in a training corpus is a
// tokenizer's problem nobody asked for.
func (r *Report) WriteJSONL(w io.Writer) error {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	enc.SetEscapeHTML(false)

	for _, g := range r.Groups {
		for _, ep := range g.Episodes {
			if ep == nil {
				continue
			}
			if err := enc.Encode(jsonOf(g, ep)); err != nil {
				return fmt.Errorf("rl: encoding episode %s: %w", ep.Label(), err)
			}
		}
	}
	return bw.Flush()
}

func jsonOf(g *Group, ep *Episode) episodeJSON {
	out := episodeJSON{
		Env:        g.Env,
		Task:       ep.Task.ID,
		Sample:     ep.Task.Sample,
		Prompt:     ep.Task.Prompt,
		Workspace:  ep.Task.Workspace,
		Reward:     ep.Reward,
		Breakdown:  ep.Breakdown,
		Failure:    ep.Failure,
		Success:    ep.Failure == Success,
		Outcome:    "none",
		Turns:      len(ep.Steps),
		ToolCalls:  ep.ToolCalls(),
		ToolErrs:   ep.ToolErrors(),
		Denials:    denials(ep),
		WallMillis: ep.Wall.Milliseconds(),
		CostUSD:    ep.Spend.CostUSD,
		Priced:     ep.Priced,
		Unpriced:   orEmpty(ep.Unpriced),
		Usage:      toUsage(ep.Usage()),
		Steps:      make([]stepJSON, 0, len(ep.Steps)),
		Messages:   ep.Messages,
	}

	switch o := ep.Outcome.(type) {
	case wombat.Answer:
		out.Outcome, out.Answer = "answer", o.Text
	case wombat.Submitted:
		out.Outcome, out.Tool, out.Payload = "submitted", o.Tool, o.Payload
	case wombat.Paused:
		out.Outcome = "paused"
	}
	if ep.Err != nil {
		out.Error = ep.Err.Error()
	}

	for _, s := range ep.Steps {
		out.Steps = append(out.Steps, stepJSON{
			Iteration: s.Iteration,
			// Never nil: a trainer that has to distinguish null from [] in
			// every field is reading a format that was not designed for it.
			Tools:  orEmpty(s.Tools),
			Failed: orEmpty(s.Failed),
			Denied: orEmpty(s.Denied),
			Millis: s.Millis,
			Usage:  toUsage(s.Usage),
		})
	}
	if out.Messages == nil {
		out.Messages = []llm.Message{}
	}
	return out
}

func orEmpty(xs []string) []string {
	if xs == nil {
		return []string{}
	}
	return xs
}

// cost renders a group's median cost, or "n/a" when that median is not a
// price.
//
// "n/a" rather than "$0.0000", and this is the whole fix: a zero in a money
// column is a claim that the run was free, it is the claim a reader most wants
// to be true, and it is the one they will not check. An absence has to look
// like an absence.
func cost(g *Group) string {
	if !g.Priced() {
		return "n/a"
	}
	return fmt.Sprintf("$%.4f", g.MedianCost())
}

// compactTokens renders a token count in a column's worth of characters:
// "8420", "31.4k", "2.10M".
//
// Rounded on purpose. Three significant figures is enough to see that one
// agent used twice the context of another, which is what this column is for,
// and the exact counts are in the JSONL for anyone doing arithmetic.
func compactTokens(v float64) string {
	switch {
	case v >= 1e6:
		return fmt.Sprintf("%.2fM", v/1e6)
	case v >= 1e4:
		return fmt.Sprintf("%.1fk", v/1e3)
	default:
		return fmt.Sprintf("%.0f", v)
	}
}

// firstLine keeps a multi-line error — a contained panic carries a stack —
// from taking over the summary.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i] + " …"
	}
	return s
}

// writeSolvedButUnfinished names the episodes that did the work and could not
// stop.
//
// Without this line the report says pass@1 0.000 for a task one sample scored
// full marks on, and the only trace of it is a standard deviation the reader
// has to do arithmetic on. Two very different conclusions read identically:
// "this agent cannot do the task" and "this agent can do the task but does not
// know when it is finished". The second is a much cheaper thing to fix, and it
// is invisible in every column of the table.
//
// The pass@k numbers are deliberately NOT adjusted. An agent that runs until
// the iteration cap has not succeeded, whatever the workspace ends up
// containing; a benchmark that scored it as a pass would be measuring the
// verifier rather than the agent.
func (r *Report) writeSolvedButUnfinished(bw *bufio.Writer) {
	type row struct {
		label  string
		reward float64
		kind   FailureKind
	}
	var rows []row
	for _, g := range r.Groups {
		for _, ep := range g.Episodes {
			if g.Solved(ep) {
				rows = append(rows, row{ep.Label(), ep.Reward, ep.Failure})
			}
		}
	}
	if len(rows) == 0 {
		return
	}

	fmt.Fprintf(bw, "\nsolved but did not finish (%d): full reward, and the run still failed —\n"+
		"the agent did the work and could not tell it was done. Not counted as a pass.\n", len(rows))
	for _, x := range rows {
		fmt.Fprintf(bw, "  %s  reward %.3f  %s\n", x.label, x.reward, x.kind)
	}
}
