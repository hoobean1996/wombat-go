package sandbox

import (
	"os"
	"slices"
	"strings"

	"wombat/internal/safeenv"
)

// Option configures the environment of [Unconfined] and [Confined].
type Option func(*envConfig)

// WithEnvAllow lets more of the harness's environment through to commands, by
// name; an entry ending in "*" is a prefix.
//
// The default list carries the process basics, the Go and C toolchains'
// settings and the proxy variables — see the doc on [Unconfined] for what it
// leaves out and why. A project whose build needs something else, a
// NODE_OPTIONS or a JAVA_HOME, names it here.
func WithEnvAllow(names ...string) Option {
	return func(c *envConfig) { c.allow = append(c.allow, names...) }
}

// WithEnv sets variables for every command, as KEY=VALUE entries, over
// whatever the filter passes through. An entry replaces a variable of the same
// name and is set even when the filter would have dropped the name.
//
// It is the per-runner counterpart of setting a variable on the whole process,
// and the better one: what the agent's commands need to see is often not what
// the harness itself should run with.
func WithEnv(kv ...string) Option {
	return func(c *envConfig) { c.set = append(c.set, kv...) }
}

// WithInheritedEnv passes the harness's entire environment through, unfiltered
// — what [Unconfined] did before it filtered anything.
//
// Every credential the harness was started with is then one `env` away from
// the model. That is a reasonable thing to accept on a throwaway machine whose
// environment holds nothing worth stealing, and nowhere else.
func WithInheritedEnv() Option {
	return func(c *envConfig) { c.inherit = true }
}

// envConfig is the environment a runner hands its commands: either the
// harness's whole environment or the allowlisted part of it, then the
// explicit entries on top.
type envConfig struct {
	inherit bool
	allow   []string
	set     []string
	// sandbox holds entries a [Confined] sets for itself, applied between
	// the filtered base and the caller's own entries, so a caller's WithEnv
	// still wins.
	sandbox []string
}

func newEnvConfig(opts []Option) envConfig {
	c := envConfig{allow: slices.Clone(safeenv.DefaultAllow)}
	for _, o := range opts {
		o(&c)
	}
	return c
}

// environ is the environment for a command starting now. Always non-nil:
// exec.Cmd reads a nil Env as "inherit everything", which is exactly the
// behaviour this exists to stop.
func (c envConfig) environ() []string {
	var base []string
	if c.inherit {
		base = os.Environ()
	} else {
		base = safeenv.Filter(os.Environ(), c.allow)
	}
	env := safeenv.Overlay(safeenv.Overlay(base, c.sandbox), c.set)
	if env == nil {
		env = []string{}
	}
	return env
}

// explicit reports the value the caller set for name with [WithEnv], if any.
// The last entry wins, as it does in the environment the command receives.
func (c envConfig) explicit(name string) (string, bool) {
	for i := len(c.set) - 1; i >= 0; i-- {
		if k, v, _ := strings.Cut(c.set[i], "="); k == name {
			return v, true
		}
	}
	return "", false
}
