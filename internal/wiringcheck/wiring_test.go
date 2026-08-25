package wiringcheck

import (
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/skill"
)

// The shipped skills directory, wired the way the CLI wires it.
//
// The package unit tests cover the mechanism with synthetic skills; this one
// covers the thing that actually ships, because a skill whose frontmatter
// stopped saying install: true would still pass every test in skill/ while
// silently going back to being something the model has to find.
func TestRealSkillsDirProducesAnInstalledBlock(t *testing.T) {
	skills, err := skill.LoadDir("../../skills", func(e error) { t.Errorf("load: %v", e) })
	if err != nil {
		t.Fatal(err)
	}
	reg := skill.New(skills...)
	g := reg.Bind(nil)

	t.Logf("installed block: %d bytes", len(g.Installed))
	t.Logf("loadable catalogue:\n%s", g.Index)

	if !strings.Contains(g.Installed, "Name the invariants") {
		t.Error("the wide-research methodology is not in the installed block")
	}
	if strings.Contains(g.Index, "wide-research") {
		t.Error("an installed skill is still offered as loadable")
	}
	for _, want := range []string{"git-history", "go-concurrency"} {
		if !strings.Contains(g.Index, want) {
			t.Errorf("loadable skill %q vanished from the catalogue", want)
		}
	}
}
