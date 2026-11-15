package judge

import (
	"context"
	"errors"
	"log/slog"
	"sync"

	"wombat/eval/rl"
)

// Verifier adapts the pipeline into an [rl.Verifier], so a judged score lands
// in the same reward the oracle contributes to.
//
// Lossy on purpose, and worth naming: rl.Verifier returns one float, so the
// diagnosis — the root cause, the first bad step, the repair target — is
// discarded at this boundary. Everything upstream exists to produce those, and
// a pipeline that only ever runs through here has thrown away its own output.
// Use [Collector] when the reports are wanted, which is most of the time.
//
// weight scales the contribution; the score is the normalized overall in 0..1.
func Verifier(name string, cfg Config, weight float64) rl.Verifier {
	return func(ctx context.Context, ep *rl.Episode) (string, float64) {
		rep, err := Evaluate(ctx, ep, cfg)
		if err != nil {
			// An unjudgeable episode scores nothing rather than zero: a
			// harness panic is not evidence about the agent, and rl has no
			// third answer, so the honest move is to contribute no reward and
			// say why in the log.
			slog.WarnContext(ctx, "judge: no verdict", "episode", ep.Label(), "err", err)
			return name, 0
		}
		return name, rep.Overall * weight
	}
}

// Collector is a Verifier that keeps every report it produced.
//
// This is how a rollout gets a [Corpus] without a second pass over the
// episodes: rl calls the verifier once per episode anyway, so the reports are
// already being computed and the only question is whether anything keeps them.
//
// Safe for concurrent use — Rollout scores samples on their own goroutines.
type Collector struct {
	Name   string
	Config Config

	// Weight scales the reward contribution. Zero means this collector
	// observes without influencing the reward at all, which is the right
	// setting while a rubric is still being calibrated: a judge that is also
	// steering the number it is being calibrated against cannot be checked.
	Weight float64

	mu       sync.Mutex
	reports  Corpus
	unjudged []string
}

// Verifier returns the rl.Verifier that feeds this collector.
func (c *Collector) Verifier() rl.Verifier {
	return func(ctx context.Context, ep *rl.Episode) (string, float64) {
		name := c.Name
		if name == "" {
			name = "judge"
		}
		rep, err := Evaluate(ctx, ep, c.Config)
		if err != nil {
			c.mu.Lock()
			c.unjudged = append(c.unjudged, ep.Label())
			c.mu.Unlock()
			if !errors.Is(err, ErrUnjudgeable) {
				slog.WarnContext(ctx, "judge: no verdict", "episode", ep.Label(), "err", err)
			}
			return name, 0
		}
		c.mu.Lock()
		c.reports = append(c.reports, rep)
		c.mu.Unlock()
		return name, rep.Overall * c.Weight
	}
}

// Corpus returns the reports gathered so far.
func (c *Collector) Corpus() Corpus {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append(Corpus(nil), c.reports...)
}

// Unjudged returns the labels of episodes no verdict could be reached on.
//
// Reported rather than silently dropped: an eval whose corpus is smaller than
// its rollout is measuring a subset, and which subset matters — if the ones
// that panicked were all the hard tasks, every number above is flattering.
func (c *Collector) Unjudged() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.unjudged...)
}
