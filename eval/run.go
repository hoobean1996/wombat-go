package eval

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
)

// Config drives a suite run.
type Config struct {
	Client  llm.Client
	Model   string
	Samples int
	Workers int

	// Dir is where sample workspaces are made. Empty uses the OS temp dir.
	Dir string

	// Keep leaves workspaces behind. Default is to keep only the samples of a
	// probe that failed, since those are the ones anybody will want to open.
	Keep bool

	// Expensive includes probes that need a long run to observe anything.
	Expensive bool

	Progress func(probe string, arm string, sample int, o Obs)
	Log      *slog.Logger
}

func (c Config) samples() int {
	if c.Samples > 0 {
		return c.Samples
	}
	return DefaultSamples
}

func (c Config) workers() int {
	if c.Workers > 0 {
		return c.Workers
	}
	return DefaultWorkers
}

func (c Config) log() *slog.Logger {
	if c.Log != nil {
		return c.Log
	}
	return slog.Default()
}

// Defaults for a suite run.
//
// Three samples rather than one because an invariant that held once held once,
// and rather than eight because every sample is a live model call and this
// suite is meant to be run on a change, not overnight.
const (
	DefaultSamples = 3
	DefaultWorkers = 3
)

// Run evaluates probes and returns one result each, in input order.
func Run(ctx context.Context, cfg Config, probes []*Probe) []*Result {
	out := make([]*Result, 0, len(probes))
	for _, p := range probes {
		if p.Expensive && !cfg.Expensive {
			continue
		}
		out = append(out, runProbe(ctx, cfg, p))
	}
	return out
}

func runProbe(ctx context.Context, cfg Config, p *Probe) *Result {
	started := time.Now()
	res := &Result{Probe: p, Runs: Runs{}}

	if len(p.Arms) == 0 {
		res.Err = fmt.Errorf("eval: probe %s has no arms", p.ID)
		return res
	}

	type job struct {
		arm    Arm
		sample int
	}
	var jobs []job
	for _, a := range p.Arms {
		for s := range cfg.samples() {
			jobs = append(jobs, job{arm: a, sample: s})
		}
	}

	var (
		mu   sync.Mutex
		sem  = make(chan struct{}, cfg.workers())
		wg   sync.WaitGroup
		fail error
	)
	for _, j := range jobs {
		wg.Add(1)
		go func(j job) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			o, err := runSample(ctx, cfg, p, j.arm, j.sample)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if fail == nil {
					fail = err
				}
				return
			}
			res.Runs[j.arm.Name] = append(res.Runs[j.arm.Name], o)
			if cfg.Progress != nil {
				cfg.Progress(p.ID, j.arm.Name, j.sample, o)
			}
		}(j)
	}
	wg.Wait()

	if fail != nil {
		res.Err = fail
		res.Wall = time.Since(started)
		return res
	}

	// Samples finish out of order under concurrency; sort so a report and a
	// re-run of the same probe list the same sample in the same place.
	for name := range res.Runs {
		s := res.Runs[name]
		for i := 1; i < len(s); i++ {
			for j := i; j > 0 && s[j].Sample < s[j-1].Sample; j-- {
				s[j], s[j-1] = s[j-1], s[j]
			}
		}
	}

	for _, e := range p.Expect {
		v := e.Check(res.Runs)
		res.Verdicts = append(res.Verdicts, NamedVerdict{Name: e.Name, Kind: e.Kind, Verdict: v})
	}
	res.Wall = time.Since(started)
	return res
}

func runSample(ctx context.Context, cfg Config, p *Probe, arm Arm, sample int) (Obs, error) {
	dir, err := sampleDir(cfg, p, arm, sample)
	if err != nil {
		return Obs{}, err
	}
	keep := cfg.Keep
	defer func() {
		if !keep {
			os.RemoveAll(dir)
		}
	}()

	if p.Setup != nil {
		if err := p.Setup(dir); err != nil {
			keep = true
			return Obs{}, fmt.Errorf("eval: %s/%s sample %d: setup: %w", p.ID, arm.Name, sample, err)
		}
	}

	agent, err := arm.Build(Deps{
		Client:    cfg.Client,
		Model:     cfg.Model,
		Workspace: dir,
		Skills:    p.Skills,
	})
	if err != nil {
		return Obs{}, fmt.Errorf("eval: %s/%s: building the agent: %w", p.ID, arm.Name, err)
	}
	if p.MaxIters > 0 {
		if agent, err = agent.With(wombat.WithMaxIters(p.MaxIters)); err != nil {
			return Obs{}, fmt.Errorf("eval: %s/%s: capping the agent at %d iterations: %w", p.ID, arm.Name, p.MaxIters, err)
		}
	}

	limits := p.Limits
	runCtx, cancel := governor.WithBudget(ctx, limits)
	defer cancel()

	started := time.Now()
	run := agent.Start(runCtx, wombat.Ask(p.Prompt))
	defer run.Close()

	o := observe(run)
	o.Arm, o.Sample, o.Wall = arm.Name, sample, time.Since(started)

	// A failed sample is data, not an error: "the mechanism broke the run" is
	// exactly the kind of answer this suite exists to produce, and turning it
	// into a Go error would abort the probe and lose the other arm.
	if o.Err != "" {
		cfg.log().WarnContext(ctx, "eval: sample failed",
			slog.String("probe", p.ID), slog.String("arm", arm.Name),
			slog.Int("sample", sample), slog.String("err", o.Err))
		keep = true
	}
	return o, nil
}

func sampleDir(cfg Config, p *Probe, arm Arm, sample int) (string, error) {
	base := cfg.Dir
	if base == "" {
		base = os.TempDir()
	}
	if err := os.MkdirAll(base, 0o755); err != nil {
		return "", fmt.Errorf("eval: preparing %s: %w", base, err)
	}
	dir, err := os.MkdirTemp(base, fmt.Sprintf("wombat-eval-%s-%s-%d-", p.ID, arm.Name, sample))
	if err != nil {
		return "", fmt.Errorf("eval: workspace for %s/%s: %w", p.ID, arm.Name, err)
	}
	return filepath.Clean(dir), nil
}
