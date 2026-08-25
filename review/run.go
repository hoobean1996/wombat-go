package review

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/hoobean1996/wombat-go"
)

// Run reviews the code in cfg.Workspace and, when a Fixer is supplied, repairs
// what it finds.
//
// The findings are established by RUNNING the tests the reviewer wrote, never
// by reading what it said about them. That is the whole design: a reviewer
// that overstates cannot inflate the count, and one that understates cannot
// hide a test that fails.
func Run(ctx context.Context, cfg Config) *Report {
	started := time.Now()
	rep := &Report{Workspace: cfg.Workspace}
	defer func() { rep.Wall = time.Since(started) }()

	if cfg.Reviewer == nil && cfg.NewReviewer == nil {
		rep.Err = fmt.Errorf("review: no reviewer agent")
		return rep
	}
	if cfg.Workspace == "" {
		rep.Err = fmt.Errorf("review: no workspace")
		return rep
	}

	// Anything left over from an earlier review would be counted as this
	// review's work, and a stale failing test would be reported as a fresh
	// defect.
	if err := removeReviewFiles(cfg.Workspace); err != nil {
		rep.Err = err
		return rep
	}

	planFile, todoFile := cfg.PlanFile, cfg.TodoFile
	if planFile == "" {
		planFile = DefaultPlanFile
	}
	if todoFile == "" {
		todoFile = DefaultTodoFile
	}
	rep.Unfinished = unfinishedTodos(cfg.Workspace, todoFile)
	for _, t := range rep.Unfinished {
		cfg.log("review: planned but unfinished: %s", t)
	}

	rep.Discarded = discardedErrors(cfg.Workspace)
	if n := len(rep.Discarded); n > 0 {
		cfg.log("review: %d test site(s) discard a call's last return value", n)
	}

	planBefore := planSnapshot(cfg.Workspace, planFile, todoFile)

	plan := readPlan(cfg.Workspace, planFile)
	if plan != "" {
		cfg.log("review: reviewing against %s (%d bytes)", planFile, len(plan))
	}

	// Sweep until the reviewers stop finding anything new.
	//
	// One pass is not a clean bill of health. The measured case: a reviewer
	// reported one defect and stopped, and a subsequent read of the same code
	// by hand found five more. It had not finished looking; it had run out of
	// ideas. A fixed round count treats those as the same thing, so the stop
	// condition here is convergence — DryStreak consecutive sweeps that turn
	// up no finding the earlier sweeps had not already recorded.
	sweeps, dryNeeded := cfg.Sweeps, cfg.DryStreak
	if sweeps <= 0 {
		sweeps = 1
	}
	if dryNeeded <= 0 {
		dryNeeded = defaultDryStreak
	}

	// Convergence needs every lens tried, then DryStreak dry sweeps after that.
	// Below this a run cannot converge however clean the code is, and saying so
	// up front is better than reporting "never converged" as if it were a
	// finding about the code.
	if minToConverge := len(lenses) + dryNeeded - 1; sweeps < minToConverge {
		cfg.log("review: %d sweep(s) cannot converge: %d lenses plus a %d-sweep dry streak needs %d",
			sweeps, len(lenses), dryNeeded, minToConverge)
	}

	seen := map[string]bool{}
	dry := 0

	for sweep := 1; sweep <= sweeps && dry < dryNeeded; sweep++ {
		reviewer, err := cfg.reviewerFor(sweep)
		if err != nil {
			rep.Err = fmt.Errorf("review: sweep %d: %w", sweep, err)
			break
		}
		rep.Sweeps = sweep

		lens := lensFor(sweep)
		cfg.log("review: sweep %d/%d reading %s [%s]", sweep, sweeps, cfg.Workspace, lens.name)

		prompt := reviewPrompt(cfg.Task, plan, rep.Discarded, lens, rep.Findings)
		if _, err := reviewer.Run(ctx, wombat.Ask(prompt)); err != nil {
			// A reviewer that died still leaves whatever tests it had written,
			// and those are worth collecting: measured runs were cut short by
			// an output cap after finding real defects. The error is kept so
			// the report cannot be read as a clean review.
			rep.Err = fmt.Errorf("review: reviewer did not finish: %w", err)
		}
		rep.Ran = true

		// Only findings no earlier sweep recorded advance the search. A sweep
		// that re-reports an outstanding defect has told us nothing new, and
		// counting it would reset the streak and prevent convergence.
		var fresh []Finding
		for _, f := range collect(ctx, cfg) {
			if seen[f.Test] {
				continue
			}
			seen[f.Test] = true
			fresh = append(fresh, f)
		}
		if len(fresh) == 0 {
			// A dry sweep only counts once every lens has had a turn.
			//
			// Without this the streak is a claim about the lenses that ran,
			// stated as if it were about the code. A measured run stopped after
			// two dry sweeps — "open" and "error paths" — and reported that it
			// found nothing, while the defect that the "shared state" lens had
			// found in the previous run was still present and still silently
			// returning another task's value. That lens never ran.
			if sweep < len(lenses) {
				cfg.log("review: sweep %d [%s] found nothing new; %d lens(es) still untried, not counting it as dry",
					sweep, lens.name, len(lenses)-sweep)
				continue
			}
			dry++
			cfg.log("review: sweep %d found nothing new (%d/%d dry)", sweep, dry, dryNeeded)
			continue
		}
		dry = 0
		rep.Findings = append(rep.Findings, fresh...)
		cfg.log("review: sweep %d found %d new defect(s), %d total", sweep, len(fresh), len(rep.Findings))

		if cfg.Fixer == nil || cfg.Rounds <= 0 {
			continue
		}
		for round := 1; round <= cfg.Rounds; round++ {
			outstanding := rep.Outstanding()
			if len(outstanding) == 0 {
				break
			}
			rep.Rounds++
			cfg.log("review: repair round %d, %d outstanding", rep.Rounds, len(outstanding))

			if _, err := cfg.Fixer.Run(ctx, wombat.Ask(fixPrompt(cfg.Task, outstanding))); err != nil {
				cfg.log("review: fixer did not finish: %v", err)
			}
			// Re-run the SAME tests. A repair is credited only when the test
			// that demonstrated the defect passes; the fixer's own account of
			// what it changed is not evidence.
			markFixed(ctx, cfg, rep)
		}
	}
	rep.Converged = dry >= dryNeeded

	// Checked last so it covers the reviewer and every repair round.
	rep.PlanEdited = planEdits(cfg.Workspace, planBefore)
	for _, name := range rep.PlanEdited {
		cfg.log("review: %s was rewritten during the run", name)
	}
	return rep
}

// collect runs the reviewer's tests and keeps the ones that fail.
//
// A test the reviewer wrote and that PASSES is not a finding — it is a
// hypothesis it checked and discarded, and counting it would reward writing
// tests rather than finding defects.
func collect(ctx context.Context, cfg Config) []Finding {
	files := reviewFiles(cfg.Workspace)
	if len(files) == 0 {
		return nil
	}
	failures := runTests(ctx, cfg)

	byTest := map[string]string{}
	for name, msg := range failures {
		byTest[name] = msg
	}

	var out []Finding
	for _, file := range files {
		for _, name := range testNamesIn(filepath.Join(cfg.Workspace, file)) {
			msg, failed := byTest[name]
			if !failed {
				continue
			}
			out = append(out, Finding{Test: name, File: file, Message: msg})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Test < out[j].Test })
	return out
}

// markFixed re-runs the tests and credits the ones that now pass.
//
// A finding is credited only if its test STILL EXISTS. Deleting the failing
// test is the cheapest way to turn it green and the one repair that leaves the
// defect exactly where it was, so "no longer failing" is not sufficient
// evidence on its own — the demonstration has to survive the repair.
func markFixed(ctx context.Context, cfg Config, rep *Report) {
	present := map[string]bool{}
	for _, file := range reviewFiles(cfg.Workspace) {
		for _, name := range testNamesIn(filepath.Join(cfg.Workspace, file)) {
			present[name] = true
		}
	}

	failures := runTests(ctx, cfg)
	for i := range rep.Findings {
		f := &rep.Findings[i]
		if !present[f.Test] {
			f.Fixed = false
			f.Message = "the test demonstrating this was removed during repair; the defect is unaddressed"
			continue
		}
		_, stillFailing := failures[f.Test]
		f.Fixed = !stillFailing
	}
}

var (
	reTestFail = regexp.MustCompile(`(?m)^\s*--- FAIL: (Test\w+)`)
	reTestFunc = regexp.MustCompile(`(?m)^func (Test\w+)\(`)
)

// runTests returns the reviewer tests that failed, mapped to their output.
func runTests(ctx context.Context, cfg Config) map[string]string {
	tctx, cancel := context.WithTimeout(ctx, cfg.testTimeout())
	defer cancel()

	cmd := exec.CommandContext(tctx, "go", "test", "-run", TestPrefix, "-v", "./...")
	cmd.Dir = cfg.Workspace
	out, _ := cmd.CombinedOutput() // a non-zero exit is the normal case here
	text := string(out)

	failed := map[string]string{}
	for _, m := range reTestFail.FindAllStringSubmatch(text, -1) {
		failed[m[1]] = firstFailureLine(text, m[1])
	}
	return failed
}

// firstFailureLine pulls the message a failing test printed, which is the
// reviewer's actual claim about what is wrong.
func firstFailureLine(output, test string) string {
	lines := strings.Split(output, "\n")
	start := -1
	for i, l := range lines {
		if strings.Contains(l, "=== RUN") && strings.Contains(l, test) {
			start = i
			break
		}
	}
	if start < 0 {
		return ""
	}
	for i := start + 1; i < len(lines); i++ {
		l := strings.TrimSpace(lines[i])
		if strings.HasPrefix(l, "--- ") {
			break
		}
		// A t.Errorf line looks like "  file_test.go:42: message".
		if idx := strings.Index(l, ".go:"); idx >= 0 {
			if colon := strings.Index(l[idx+4:], ": "); colon >= 0 {
				return strings.TrimSpace(l[idx+4+colon+2:])
			}
		}
	}
	return ""
}

// reviewFiles lists the test files this review produced, relative to the
// workspace.
func reviewFiles(dir string) []string {
	var out []string
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if name := d.Name(); isReviewFile(name) {
			if rel, rerr := filepath.Rel(dir, path); rerr == nil {
				out = append(out, rel)
			}
		}
		return nil
	})
	sort.Strings(out)
	return out
}

// isReviewFile reports whether a base name is one the reviewer wrote.
func isReviewFile(name string) bool {
	return strings.HasPrefix(name, "review") && strings.HasSuffix(name, "_test.go")
}

func removeReviewFiles(dir string) error {
	for _, f := range reviewFiles(dir) {
		if err := os.Remove(filepath.Join(dir, f)); err != nil {
			return fmt.Errorf("review: clearing %s: %w", f, err)
		}
	}
	return nil
}

func testNamesIn(path string) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []string
	for _, m := range reTestFunc.FindAllStringSubmatch(string(b), -1) {
		if strings.HasPrefix(m[1], TestPrefix) {
			out = append(out, m[1])
		}
	}
	return out
}
