// This file scores the episodes a benchmark run already produced, and prints
// the report and the repair queue. It runs only when given something to score:
//
//	make bench TASKS=easy                         # writes runs/episodes.jsonl
//	go test ./eval/judge -run '^TestJudgeEpisodes$' -v -args -- -in "$PWD/runs/episodes.jsonl"
//	make judge IN=runs/episodes.jsonl LLM=1       # the same, adding the model judge
//
// Everything after `--` is this runner's own flags. The test runs in this
// directory, so -in wants an absolute path; make passes one.
//
// Separate from the benchmark run on purpose. Judging inside the rollout would
// make every rubric change cost another set of live agent runs, and nobody
// iterates on a rubric at that price. The episodes file is the seam: rollouts
// are expensive and rare, judgements are cheap and should be re-run whenever
// the rubric moves.
//
// The output that matters is the last block. Everything above it — scores,
// pass rates, the failure histogram — is working out; the repair queue is the
// answer, and a run that stops before it has not finished.
package judge_test

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"sort"
	"testing"

	"wombat/eval/judge"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/llm/anthropic"
	"wombat/pkg/core/llm/openai"
)

// TestJudgeEpisodes is the entry point, and is skipped unless it has been
// given episodes to score.
func TestJudgeEpisodes(t *testing.T) {
	args := flag.Args()
	if len(args) == 0 {
		t.Skip("give it episodes: -args -- -in /abs/path/episodes.jsonl [-llm]")
	}
	if code := judgeEpisodes(args, os.Stdout, os.Stderr); code != 0 {
		t.Fatalf("exit %d; see above", code)
	}
}

func judgeEpisodes(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("judge", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		in        = fs.String("in", "", "episodes.jsonl from a rollout (required)")
		provider  = fs.String("provider", envOr("WOMBAT_PROVIDER", "openai"), "anthropic | openai")
		model     = fs.String("model", "", "model id (empty uses the provider default)")
		useLLM    = fs.Bool("llm", false, "add the model judge for reasoning quality and user alignment")
		checklist = fs.Bool("checklist", false,
			"judge faithfulness one requirement at a time.\n"+
				"Costs one model call per requirement per episode, and is the only mode that\n"+
				"reliably catches a claim the run never tested — a single call asked to both\n"+
				"enumerate requirements and evaluate them hides its enumeration misses.")
		asJSON = fs.Bool("json", false, "emit the reports as JSON")
		attrs  = fs.String("attributions", "",
			"append every attribution to this JSONL ledger, for later review.\n"+
				"Each record starts as \"unreviewed\"; a person edits the status to confirmed,\n"+
				"false_positive or unclear. Until some are reviewed, the repair queue is a list\n"+
				"of guesses whose precision nobody has measured.")
	)
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *in == "" {
		fmt.Fprintln(stderr, "judge: -in is required")
		return 2
	}

	episodes, err := judge.LoadJSONL(*in)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if len(episodes) == 0 {
		fmt.Fprintf(stderr, "judge: %s holds no episodes\n", *in)
		return 1
	}

	panel := judge.Panel{judge.Oracle{}, judge.Rules{}}
	if *useLLM || *checklist {
		client, err := buildClient(*provider, *model)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if *checklist {
			panel = append(panel, judge.Checklist{Client: client, Model: *model, Concurrency: 6})
		}
		if *useLLM {
			panel = append(panel, judge.LLM{Client: client, Model: *model, Retries: 2})
		}
	}

	cfg := judge.Config{Panel: panel, Escalate: 0.6, Authority: 0.7}
	ctx := context.Background()

	var corpus judge.Corpus
	var unjudged int
	for _, ep := range episodes {
		rep, err := judge.Evaluate(ctx, ep, cfg)
		if err != nil {
			unjudged++
			slog.Warn("no verdict", "episode", ep.Label(), "err", err)
			continue
		}
		corpus = append(corpus, rep)
	}
	if len(corpus) == 0 {
		fmt.Fprintln(stderr, "judge: nothing could be judged")
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		for _, r := range corpus {
			if err := enc.Encode(r); err != nil {
				fmt.Fprintln(stderr, err)
				return 1
			}
		}
		return 0
	}

	if *attrs != "" {
		if err := appendLedger(*attrs, corpus.Attributions(0.7, oracleBar)); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}

	render(stdout, corpus, unjudged, judge.Panel(panel))

	if *attrs != "" {
		reportLedger(stdout, *attrs)
	}
	return 0
}

// oracleBar is the reward at or above which the oracle is taken to have
// passed an episode, matching judge.DefaultOraclePass.
const oracleBar = 0.999

func render(w io.Writer, c judge.Corpus, unjudged int, panel judge.Panel) {
	names := make([]string, 0, len(panel))
	for _, j := range panel {
		names = append(names, j.Name())
	}
	fmt.Fprintf(w, "\n%d episode(s), panel: %v\n", len(c), names)
	if unjudged > 0 {
		// Named rather than folded into the total: a corpus smaller than its
		// rollout is measuring a subset, and which subset matters.
		fmt.Fprintf(w, "%d episode(s) could not be judged and are NOT in any number below\n", unjudged)
	}

	fmt.Fprintf(w, "\n%-22s %-8s %6s  %s\n", "EPISODE", "VERDICT", "SCORE", "DIAGNOSIS")
	for _, r := range c {
		verdict := "pass"
		if !r.Pass {
			verdict = "FAIL"
		}
		fmt.Fprintf(w, "%-22s %-8s %6.2f  %s\n",
			trunc(fmt.Sprintf("%s#%d", r.Task, r.Sample), 22), verdict, r.Overall, trunc(r.Diagnosis.Summary, 70))
	}

	rate, ci := c.PassRate()
	fmt.Fprintf(w, "\noverall pass rate %.2f %s over %d episodes\n", rate, ci, len(c))

	fmt.Fprintf(w, "\n%-22s %5s %8s %-14s %6s %s\n", "CRITERION", "N", "PASS", "95% CI", "MEAN", "UNJUDGED")
	for _, s := range c.Aggregate() {
		if s.N == 0 {
			fmt.Fprintf(w, "%-22s %5s %8s %-14s %6s %d\n", s.Criterion, "-", "-", "-", "-", s.Unjudged)
			continue
		}
		fmt.Fprintf(w, "%-22s %5d %8.2f %-14s %6.2f %d\n",
			s.Criterion, s.N, s.PassRate, s.CI.String(), s.MeanScore, s.Unjudged)
	}

	if con := c.Concerns(); len(con) > 0 {
		fmt.Fprintf(w, "\nconcerns — flagged, but the judge was not confident; not acted on\n")
		for _, a := range con {
			fmt.Fprintf(w, "  %-20s score %d conf %.2f  %s\n",
				a.Criterion, a.Score, a.Confidence, trunc(a.Rationale, 60))
		}
	}

	if hist := c.FailureHistogram(); len(hist) > 0 {
		fmt.Fprintf(w, "\nroot causes\n")
		type kv struct {
			f judge.Failure
			n int
		}
		var rows []kv
		for f, n := range hist {
			rows = append(rows, kv{f, n})
		}
		sort.Slice(rows, func(i, j int) bool {
			if rows[i].n != rows[j].n {
				return rows[i].n > rows[j].n
			}
			return rows[i].f < rows[j].f
		})
		for _, r := range rows {
			fmt.Fprintf(w, "  %-32s %d\n", r.f, r.n)
		}
	}

	// The calibration block, and it goes ABOVE the repair queue because it
	// decides whether the queue is worth reading.
	//
	// Two of these four buckets are the only cheap evidence there is about the
	// judges themselves. A repair queue built on unvalidated judgements sends
	// you to fix the wrong component: measured on a four-episode corpus, one
	// of the two entries in the queue came from a faithfulness flag on a patch
	// the verifiers had passed, where all three cited "ungrounded" tokens were
	// an agent explaining a real observation in its own notation.
	keys, buckets := c.Slice(judge.ByOracle(oracleBar))
	fmt.Fprintf(w, "\njudge against oracle (reward >= %.2f)\n", oracleBar)
	for _, k := range keys {
		mark := "  "
		if k == "rewarded_but_judged_failing" || k == "judged_passing_but_unrewarded" {
			mark = "! "
		}
		fmt.Fprintf(w, "%s%-32s %d\n", mark, k, len(buckets[k]))
	}
	if n := len(buckets["rewarded_but_judged_failing"]); n > 0 {
		fmt.Fprintf(w, "  ! %d episode(s) the oracle passed and the judges failed: read these before\n"+
			"    acting on anything below. They are either a real defect the verifiers cannot\n"+
			"    see, or a defect in the rubric.\n", n)
	}
	if n := len(buckets["judged_passing_but_unrewarded"]); n > 0 {
		fmt.Fprintf(w, "  ! %d episode(s) the judges passed and the oracle failed: the rubric is\n"+
			"    missing whatever the verifiers are checking.\n", n)
	}

	// Undiagnosed failures come FIRST and are never folded into the queue.
	// They are the episodes nobody knows what to do about, and a report that
	// ranks the known problems above the unknown one has its priorities
	// inverted.
	if un := c.Undiagnosed(); len(un) > 0 {
		fmt.Fprintf(w, "\nfailed with no component named — read these first\n")
		for _, r := range un {
			fmt.Fprintf(w, "  %-22s %s\n", fmt.Sprintf("%s#%d", r.Task, r.Sample), r.Diagnosis.Summary)
		}
	}

	q := c.RepairQueue()
	fmt.Fprintf(w, "\nrepair queue — what to change, most episodes first\n")
	if len(q) == 0 {
		failing := 0
		for _, r := range c {
			if !r.Pass {
				failing++
			}
		}
		if failing == 0 {
			fmt.Fprintf(w, "  nothing: every judged episode passed\n\n")
		} else {
			// The distinction matters: one of these means ship it and the
			// other means the pipeline could not tell you anything.
			fmt.Fprintf(w, "  empty, but %d episode(s) failed — every failure named no component.\n"+
				"  That is a gap in the judgement, not a clean run.\n\n", failing)
		}
		return
	}
	for i, r := range q {
		fmt.Fprintf(w, "  %d. %-24s %d episode(s), %.0f%% of the corpus\n",
			i+1, r.Target, r.Episodes, r.Share*100)
		var causes []judge.Failure
		for f := range r.Causes {
			causes = append(causes, f)
		}
		sort.Slice(causes, func(i, j int) bool { return causes[i] < causes[j] })
		for _, f := range causes {
			fmt.Fprintf(w, "       %s x%d\n", f, r.Causes[f])
		}
	}
	fmt.Fprintln(w)
}

// appendLedger adds this run's attributions to the ledger.
//
// Append rather than replace: the value of the file is the history. A run that
// overwrote it would erase every status a reviewer had filled in, which is the
// only part of the record that took work.
func appendLedger(path string, attrs []judge.Attribution) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("judge: opening the ledger: %w", err)
	}
	defer f.Close() //nolint:errcheck // append-only, error surfaced by Write
	return judge.WriteAttributions(f, attrs)
}

// reportLedger prints what the whole ledger says, not just this run.
func reportLedger(w io.Writer, path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close() //nolint:errcheck // read-only
	all, err := judge.ReadAttributions(f)
	if err != nil || len(all) == 0 {
		return
	}
	st := judge.Summarise(all)

	fmt.Fprintf(w, "attribution ledger: %s\n", path)
	fmt.Fprintf(w, "  %d attribution(s), %d authoritative, %d disagreeing with the oracle\n",
		st.Total, st.Authoritative, st.Disagreeing)

	var statuses []string
	for k := range st.ByStatus {
		statuses = append(statuses, k)
	}
	sort.Strings(statuses)
	for _, k := range statuses {
		fmt.Fprintf(w, "  %-16s %d\n", k, st.ByStatus[k])
	}
	if p := st.Precision(); p < 0 {
		fmt.Fprintf(w, "  precision       unknown — nothing reviewed yet, so every repair above is a guess\n")
	} else {
		fmt.Fprintf(w, "  precision       %.2f over %d reviewed\n",
			p, st.ByStatus["confirmed"]+st.ByStatus["false_positive"])
	}
	fmt.Fprintln(w)
}

func buildClient(provider, model string) (llm.Client, error) {
	switch provider {
	case "anthropic", "claude":
		cfg := anthropic.ConfigFromEnv()
		if model != "" {
			cfg.Model = model
		}
		return anthropic.New(cfg)
	case "openai":
		cfg := openai.ConfigFromEnv()
		if model != "" {
			cfg.Model = model
		}
		return openai.New(cfg)
	default:
		return nil, fmt.Errorf("judge: unknown provider %q", provider)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
