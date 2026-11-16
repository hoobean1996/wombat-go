package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/accept"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/llm/anthropic"
	"wombat/pkg/core/llm/openai"
	"wombat/pkg/core/mcp"
	"wombat/pkg/core/metric"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/skill"
	"wombat/pkg/core/tape"
	"wombat/pkg/core/tool"
	"wombat/pkg/core/tool/builtin"
	"wombat/pkg/core/trace"
)

// factory turns [daemon.SessionOptions] into an agent.
//
// The client, the metric registry, the tracer, the tape, the skills and the
// MCP servers are shared — they are process resources, and one provider
// connection pool serving every conversation is the point of having one.
// Everything a session can influence is decided per session, in agent.go, and
// CLAMPED: a field a browser can set is a field a hostile browser can set, and
// httpapi deliberately passes the options through without looking at them
// because only the host knows what its deployment allows.
type factory struct {
	cfg      config
	client   llm.Client
	root     string // tools.working_dir, resolved
	logger   *slog.Logger
	registry *metric.Registry // nil without server.metrics
	metrics  *metric.Metrics  // nil without server.metrics
	// sandbox is the network mode of a confined runner; nil runs commands
	// unconfined, with the environment still filtered.
	sandbox *sandbox.Network
	// deny is what a confined command may neither read nor write, beyond the
	// credential directories: tools.deny_read, and the configuration file.
	deny   []string
	mcp    *mcp.Connections // nil without tools.mcp
	skills *skill.Registry  // nil without tools.skills_dir
	tape   *tape.Tape       // nil without record.tape
	tracer *trace.Tracer    // nil without record.trace
}

// newFactory builds what every session shares, and fails now on anything
// that would otherwise fail every session.
func newFactory(cfg config, rec *recorders, logger *slog.Logger) (*factory, error) {
	root, err := filepath.Abs(cfg.Tools.WorkingDir)
	if err == nil {
		// Resolved once, here, so that every workspace is compared with the
		// directory itself rather than with one of the names it goes by.
		root, err = filepath.EvalSymlinks(root)
	}
	if err != nil {
		return nil, fmt.Errorf("tools.working_dir: %w", err)
	}
	client, err := newClient(cfg.Model, logger)
	if err != nil {
		return nil, err
	}
	f := &factory{
		cfg:    cfg,
		client: rec.wrap(client),
		root:   root,
		logger: logger,
		tape:   rec.tape,
		tracer: rec.tracer,
	}

	if cfg.Server.Metrics {
		// One registry per process, and none when it is off: that keeps the
		// instruments out of the middleware chains entirely, rather than
		// installing ones that measure into nowhere.
		f.registry = metric.NewRegistry()
		f.metrics = metric.New(f.registry)
	}
	if cfg.Tools.Confine {
		n := networks[cfg.Tools.ConfineNetwork]
		f.sandbox = &n
		f.deny = cfg.Tools.DenyRead
	}
	if f.skills, err = loadSkills(cfg.Tools.SkillsDir, logger); err != nil {
		return nil, err
	}
	if cfg.Done.Regressions != "" {
		// A ledger that cannot be read would otherwise fail every turn,
		// quietly, in the log.
		if _, err := accept.LoadLedger(cfg.Done.Regressions); err != nil {
			return nil, fmt.Errorf("done.regressions: %w", err)
		}
	}
	// Last, because it starts processes: nothing after it can fail and leave
	// them running, except the check below, which stops them.
	if cfg.Tools.MCP != "" {
		if f.mcp, err = connectMCP(cfg.Tools.MCP, logger); err != nil {
			return nil, err
		}
	}
	if err := f.checkDisabled(); err != nil {
		f.close()
		return nil, err
	}
	return f, nil
}

// checkDisabled refuses a tools.disable entry that names no tool: a
// misspelt name would leave on offer the very tool it meant to take away.
// The names are the built-in tools' and the MCP servers'.
func (f *factory) checkDisabled() error {
	if len(f.cfg.Tools.Disable) == 0 {
		return nil
	}
	known := map[string]bool{}
	for _, d := range builtin.Default(builtin.Deps{}) {
		known[d.Name] = true
	}
	known[builtin.ViewMedia(builtin.OSFS("")).Name] = true
	for _, d := range f.mcp.Tools() {
		known[d.Name] = true
	}
	var unknown []string
	for _, name := range f.cfg.Tools.Disable {
		if !known[name] {
			unknown = append(unknown, fmt.Sprintf("%q", name))
		}
	}
	if len(unknown) > 0 {
		return fmt.Errorf("tools.disable: %s names no tool; the tools are %s", strings.Join(unknown, ", "), strings.Join(slices.Sorted(maps.Keys(known)), ", "))
	}
	return nil
}

// close stops what newFactory started. run defers it before the manager's
// Close, so it runs after: a restored session's first turn may call an MCP
// server.
func (f *factory) close() { _ = f.mcp.Close() }

// newClient builds the provider client every session shares, from the model
// section and nothing else.
func newClient(m modelConfig, logger *slog.Logger) (llm.Client, error) {
	var client llm.Client
	var err error
	switch m.Provider {
	case "anthropic":
		client, err = anthropic.New(anthropic.Config{
			APIKey:       m.APIKey,
			BaseURL:      m.BaseURL,
			Proxy:        m.Proxy,
			Model:        m.Name,
			ExtraHeaders: m.Headers,
			Stream:       streamModes[m.Stream],
		})
	default: // openai: validate admits nothing else
		client, err = openai.New(openai.Config{
			APIKey:       m.APIKey,
			BaseURL:      m.BaseURL,
			Proxy:        m.Proxy,
			Model:        m.Name,
			ExtraHeaders: m.Headers,
			Stream:       streamModes[m.Stream],
		})
	}
	if err != nil {
		return nil, fmt.Errorf("building the %s client: %w", m.Provider, err)
	}

	// Order is semantic. Cost is charged per attempt, so it sits inside retry.
	// Overflow recovery sits outside retry, so a transient blip is retried at
	// the current transcript size instead of escalating.
	return llm.Chain(client,
		llm.WithValidation,
		core.TrackCost(llm.DefaultPricing),
		llm.WithRetry(llm.DefaultRetryPolicy),
		core.WithOverflowRecovery(),
		llm.WithLogging(logger),
	), nil
}

// loadSkills discovers the tools.skills_dir skills, once. The catalogue goes
// into every session's system prompt, and is therefore part of the
// prompt-cache prefix, so it is fixed before the first call; loading and
// unloading are what happen during a run.
func loadSkills(dir string, logger *slog.Logger) (*skill.Registry, error) {
	if dir == "" {
		return nil, nil
	}
	skills, err := skill.LoadDir(dir, func(err error) { logger.Warn("skipping skill", "err", err) })
	if err != nil {
		return nil, fmt.Errorf("loading skills: %w", err)
	}
	if len(skills) == 0 {
		return nil, nil
	}
	reg := skill.New(skills...)
	// History archaeology is a distinct mode, and its two tools are noise in
	// the surface until the model says it is doing that. A skill that claims
	// them in its own manifest keeps its claim.
	for _, t := range []string{"git_log", "git_show"} {
		if _, ok := reg.GateFor(t); !ok {
			reg.Gate("git-history", t)
		}
	}
	logger.Info("skills loaded", "count", len(skills), "dir", dir)
	return reg, nil
}

// connectMCP starts the servers tools.mcp names, once, for every session.
func connectMCP(path string, logger *slog.Logger) (*mcp.Connections, error) {
	servers, err := mcp.LoadConfig(path)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	conns, err := mcp.ConnectAll(ctx, servers)
	if err != nil {
		return nil, err
	}
	logger.Info("mcp servers connected", "servers", len(servers), "tools", len(conns.Tools()))
	return conns, nil
}

// outcomeClass is the vocabulary of the "outcome" label on this server's tool
// counters, and the reason [metric.WithErrorClass] exists.
//
// The metric package cannot write this function: saying "denied" requires
// knowing about permission.ErrDenied, and metric importing permission would
// make it un-importable FROM permission. So the host supplies the words, and
// the words are chosen for the dashboard rather than for the error hierarchy.
//
// The distinction that earns its keep is denied versus error. A refused tool
// call is the sandbox working — a session under the workspace permission
// produces them all day — and a panel that counts it as a failure pages
// somebody at 3am for a policy doing its job. Likewise cancelled: a browser
// that closed the tab is not an incident.
//
// Timeout is checked BEFORE cancellation on purpose. A tool that outran
// tool.WithTimeout is a cancellation at the context layer and would match the
// broader arm, and "timeout" is both more specific and the only one of the two
// that means something is actually wrong.
//
// Deferred is its own word for the same reason denied is: a call waiting for
// a person is the policy doing its job, not a tool failing.
//
// Every arm returns one of six fixed words, so this can add at most six
// series per tool — see the cardinality section of the metric package doc for
// what happens to a classifier that does not promise that.
func outcomeClass(err error) string {
	switch {
	case err == nil:
		return "ok"
	case errors.Is(err, tool.ErrDeferred):
		return "deferred"
	case errors.Is(err, permission.ErrDenied):
		return "denied"
	case errors.Is(err, tool.ErrTimeout):
		return "timeout"
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return "cancelled"
	default:
		return "error"
	}
}
