package seed_test

import (
	"context"
	"testing"

	"github.com/hoobean1996/wombat-go/eval/judge"
	"github.com/hoobean1996/wombat-go/eval/seed"
)

// The first real measurement of whether this eval detects anything.
//
// Keyless on purpose: the corpus is mutations of committed real episodes and
// the panel is oracle + rules, so it runs in CI, in milliseconds, with no key.
// A detection suite that needs a model and a network is one nobody runs.
func TestSeededCorpus(t *testing.T) {
	bases, err := judge.LoadJSONL("testdata/bases.episodes.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if len(bases) == 0 {
		t.Fatal("no base episodes")
	}

	cases, err := seed.Build(bases, seed.Catalogue)
	if err != nil {
		t.Fatal(err)
	}

	cfg := judge.Config{
		Panel:     judge.Panel{judge.Oracle{}, judge.Rules{}},
		Escalate:  0.6,
		Authority: 0.7,
	}
	rep, err := seed.Run(context.Background(), cases, cfg)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("%d base episode(s) x %d defect(s) + controls = %d cases",
		len(bases), len(seed.Catalogue), len(cases))
	t.Logf("recall      %.2f  (%d/%d detectable mutants attributed correctly)",
		rep.Recall(), rep.DetectableFound, rep.DetectableTotal)
	t.Logf("noticed     %.2f  (%d/%d failed at all — the gap is misattribution)",
		rep.NoticeRate(), rep.DetectableNoticed, rep.DetectableTotal)
	t.Logf("blind spots %d/%d found (defects nothing on this panel claims to catch)",
		rep.BlindFound, rep.BlindTotal)
	t.Logf("false alarm %.2f  (%d/%d untouched controls failed)",
		rep.FalseAlarmRate(), rep.FalseAlarms, rep.Controls)

	t.Log("\nper defect:")
	byDefect := map[string]struct{ found, noticed, total int }{}
	for _, o := range rep.Outcomes {
		if o.Defect == "" {
			continue
		}
		e := byDefect[o.Defect]
		e.total++
		if o.Attributed {
			e.found++
		}
		if o.Failed {
			e.noticed++
		}
		byDefect[o.Defect] = e
	}
	for _, d := range seed.Catalogue {
		e := byDefect[d.Name]
		mark := "   "
		if !d.Detectable {
			mark = "(-)"
		}
		t.Logf("  %s %-28s attributed %d/%d  noticed %d/%d  expect %s",
			mark, d.Name, e.found, e.total, e.noticed, e.total, d.Expect)
	}

	// Controls are the assertion that matters most here. A panel that fails
	// untouched episodes is not detecting defects, it is failing everything,
	// and every recall number above would be meaningless.
	if rep.FalseAlarms > 0 {
		for _, o := range rep.Outcomes {
			if o.Defect == "" && o.Failed {
				t.Errorf("control %s failed with %v; an untouched real episode must pass",
					o.Episode.Task.ID, o.Got)
			}
		}
	}
}
