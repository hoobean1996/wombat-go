package judge

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"wombat/eval/rl"
	"wombat/pkg/core"
	"wombat/pkg/core/llm"
)

// episodeLine mirrors what rl.Report.WriteJSONL emits.
//
// A separate struct rather than reusing rl's unexported one: the file is the
// contract between the two halves, and duplicating the shape here means a
// field rl stops writing shows up as a compile-time hole in exactly one place
// instead of as a silently zero value spread through every judgement.
type episodeLine struct {
	Env       string             `json:"env"`
	Task      string             `json:"task"`
	Sample    int                `json:"sample"`
	Prompt    string             `json:"prompt"`
	Workspace string             `json:"workspace"`
	Reward    float64            `json:"reward"`
	Breakdown map[string]float64 `json:"breakdown"`
	Failure   rl.FailureKind     `json:"failure"`
	Outcome   string             `json:"outcome"`
	Answer    string             `json:"answer"`
	Error     string             `json:"error"`
	Messages  []llm.Message      `json:"messages"`
	Steps     []struct {
		Iteration int      `json:"iteration"`
		Tools     []string `json:"tools"`
		Failed    []string `json:"failed"`
		Denied    []string `json:"denied"`
		Millis    int64    `json:"ms"`
	} `json:"steps"`
}

// LoadJSONL reads the episodes a rollout wrote.
//
// This is the seam that lets the judging half run offline, hours after the
// rollout and on a different machine. It matters more than it looks: judging
// inside the rollout means every rubric change costs another set of live agent
// runs, and nobody iterates on a rubric at that price.
func LoadJSONL(path string) ([]*rl.Episode, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("judge: opening %s: %w", path, err)
	}
	defer f.Close() //nolint:errcheck // read-only
	return ReadJSONL(f)
}

// ReadJSONL decodes episodes from a reader.
func ReadJSONL(r io.Reader) ([]*rl.Episode, error) {
	sc := bufio.NewScanner(r)
	// Episode lines carry whole transcripts and routinely exceed the default
	// 64 KB token limit; a scanner that silently stops at the first long line
	// would report a corpus that is merely the short episodes.
	sc.Buffer(make([]byte, 0, 1<<20), 32<<20)

	var out []*rl.Episode
	for line := 1; sc.Scan(); line++ {
		raw := sc.Bytes()
		if len(raw) == 0 {
			continue
		}
		var e episodeLine
		if err := json.Unmarshal(raw, &e); err != nil {
			return nil, fmt.Errorf("judge: line %d: %w", line, err)
		}
		ep := &rl.Episode{
			Task: rl.Task{
				ID: e.Task, Sample: e.Sample, Prompt: e.Prompt, Workspace: e.Workspace,
			},
			Messages:  e.Messages,
			Reward:    e.Reward,
			Breakdown: e.Breakdown,
			Failure:   e.Failure,
		}
		for _, s := range e.Steps {
			ep.Steps = append(ep.Steps, rl.Step{
				Iteration: s.Iteration, Tools: s.Tools,
				Failed: s.Failed, Denied: s.Denied, Millis: s.Millis,
			})
		}
		if e.Answer != "" {
			ep.Outcome = core.Answer{Text: e.Answer}
		}
		if e.Error != "" {
			ep.Err = fmt.Errorf("%s", e.Error)
		}
		out = append(out, ep)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("judge: reading %s: %w", "episodes", err)
	}
	return out, nil
}
