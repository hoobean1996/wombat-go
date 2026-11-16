package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"wombat/internal/fspath"
	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/metric"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/tool"
	"wombat/pkg/core/tool/builtin"
	"wombat/pkg/daemon"
)

// session is what build decided about one session. Every agent built for it
// — the one the browser talks to, and the reviewers and fixers a review runs
// — shares it: one workspace, one policy, one model.
type session struct {
	workspace string
	// gate is the session's permission policy, or nil when there is none and
	// the tool list alone keeps the agent away from the shell.
	gate  tool.Middleware
	model string
	iters int
}

// role is what an agent built for a session is for.
type role int

const (
	builder  role = iota // the session's own agent: the one the browser talks to
	reviewer             // a fresh reader for one review sweep
	fixer                // repairs what a review found
)

func (r role) String() string {
	switch r {
	case reviewer:
		return "reviewer"
	case fixer:
		return "fixer"
	}
	return "web"
}

// reviewMaxIters caps a reviewer when review.iters does not say otherwise.
// It writes tests and runs them; it builds nothing, and a reviewer given the
// builder's budget spends it widening its search long after the useful
// findings are in.
const reviewMaxIters = 200

func (f *factory) build(opts daemon.SessionOptions) (*core.Agent, error) {
	s, err := f.session(opts)
	if err != nil {
		return nil, err
	}
	return f.agent(s, builder)
}

func (f *factory) session(opts daemon.SessionOptions) (session, error) {
	workspace, err := f.workspace(opts.Workspace)
	if err != nil {
		return session{}, err
	}
	gate, err := defaultPolicy(f.policy(opts.Permission), workspace)
	if err != nil {
		return session{}, err
	}
	model := f.cfg.Model.Name
	if opts.Model != "" {
		model = opts.Model
	}
	iters := f.cfg.Turn.MaxIters
	if opts.MaxIters > 0 && opts.MaxIters < iters {
		// Only downward. A session may ask for a shorter leash and never a
		// longer one, because turn.max_iters is the operator's budget and the
		// option is the client's preference.
		iters = opts.MaxIters
	}
	return session{workspace: workspace, gate: gate, model: model, iters: iters}, nil
}

// runner is what every command an agent of this session spawns goes through
// — the tools', and the ones done.check, done.accept and the review run:
// confined under tools.confine, and otherwise the plain one, which still
// filters the environment so the provider key this process holds stays here.
func (f *factory) runner(workspace string) (sandbox.Runner, error) {
	if f.sandbox == nil {
		return sandbox.Unconfined(workspace), nil
	}
	r, err := sandbox.Confined(workspace, sandbox.Config{Network: *f.sandbox})
	if err != nil {
		return nil, fmt.Errorf("tools.confine: %w", err)
	}
	return r, nil
}

// agent assembles one agent of a session. The ORDER below is the point, and
// is the order a decision has to be made in: the tool list first, then the
// gates around it, then the set is installed — once, because WithTools
// resets the set, and a tool registered after the skill-gated set was built
// would silently discard it.
func (f *factory) agent(s session, r role) (*core.Agent, error) {
	runner, err := f.runner(s.workspace)
	if err != nil {
		return nil, err
	}
	// With a policy, reads outside the workspace are a question for a person;
	// without one, there is nobody to ask, and the file tools are rooted there.
	gate, fsRoot := s.gate, ""
	if gate == nil {
		fsRoot = s.workspace
	}
	fsys := builtin.OSFS(fsRoot)
	tools := builtin.Default(builtin.Deps{
		FS:   fsys,
		Exec: runner,
		HTTP: &http.Client{Timeout: 30 * time.Second},
		Now:  time.Now,
	})
	if f.cfg.Tools.Media {
		tools = append(tools, builtin.ViewMedia(fsys))
	}
	// Before the filter below, so a server's tools are held to the same
	// capability rules as the built-in ones.
	tools = append(tools, f.mcp.Tools()...)

	// ask_user suspends the run, and this front end has somewhere to put the
	// answer — a paused turn leaves the session Waiting and the next prompt is
	// routed back through the pause with core.AnswerPause. So the pause
	// tools stay.
	//
	// With no policy chosen, the tool list is what keeps a browser-reachable
	// agent away from the shell, and confineReads is what keeps its reads in
	// the workspace. With a policy, the full surface is on offer and the
	// policy decides — which is the point of having one.
	if gate == nil {
		tools = tool.Filter(tools, tool.OnlyCaps(tool.CapReadOnly|tool.CapNetwork))
		gate = confineReads(s.workspace)
	}
	iters := s.iters
	if r != builder {
		iters = min(s.iters, reviewMaxIters)
		if f.cfg.Review.Iters > 0 {
			iters = f.cfg.Review.Iters
		}
	}

	opts := []core.Option{
		core.WithName(r.String()),
		core.WithClient(f.client),
		core.WithModel(s.model),
		core.WithMaxIters(iters),
		core.WithMaxTokens(f.cfg.Model.MaxTokens),
		core.WithToolParallelism(f.cfg.Turn.ToolParallelism),
		core.WithTurnNotice(iterationNotice(iters)),
		core.WithLogger(f.logger),
		// Say what the directory is FOR, not just what it is. A bare path in
		// the system prompt does not connect to the bash tool's exec_dir
		// parameter in the model's head; naming the parameter does.
		core.WithEnvBlock("working_directory", s.workspace+
			"\n\nThis is the project you are working on. Pass it as exec_dir to shell tools "+
			"and use it as the base for any relative path, unless the user names a different location."),
	}
	if f.cfg.Model.System != "" {
		opts = append(opts, core.WithSystemPrompt(f.cfg.Model.System))
	}
	if notes := f.mcp.Instructions(); notes != "" {
		opts = append(opts, core.WithEnvBlock("mcp_servers",
			"What the MCP servers say about their tools, which are named <server>__<tool>:\n\n"+notes))
	}

	// The tool middleware, inside out: a later entry wraps an earlier one. The
	// tape records only what the policy let through; the skill gate, which
	// Install adds below, sits outside the policy; and WithMetrics and
	// WithTracing are applied by New outside everything, so the denied/error
	// split outcomeClass exists for counts the calls either gate refused.
	//
	// Every caller-supplied middleware sits OUTSIDE tool.WithTimeout and
	// tool.WithRetry: a call the policy defers is not retried as if it had
	// failed, and one that is approved is not asked about again by a retry.
	opts = append(opts, f.shared(gate)...)

	if r == builder && f.cfg.Tools.Delegate && f.cfg.Mode != modeReview {
		child, err := f.researcher(s, tools, gate)
		if err != nil {
			return nil, err
		}
		tools = append(tools, core.DelegateTool(child))
		// A delegated task runs a whole child agent, and 120 seconds of real
		// work is nothing. The child's own iteration cap and the turn's budget
		// bound it instead.
		opts = append(opts, core.WithToolTimeoutFallback(0))
	}
	// The daemon's own tools — how the agent speaks to the outside and wakes
	// itself up later — are the session's agent's alone, added after the
	// delegate's child was made from the list: a sub-agent or a reviewer has
	// no one to speak to.
	if r == builder {
		tools = append(tools, daemon.Tools()...)
	}
	// Now the list is final: install it, gated by the skills if there are any
	// — except in review mode, where the session's own agent builds nothing:
	// it hands the task over, and the review that follows reads the code as
	// it is.
	if r == builder && f.cfg.Mode == modeReview {
		opts = append(opts, core.WithTools())
	} else {
		opts = append(opts, f.toolSet(tools)...)
	}
	if f.cfg.Turn.ContextWindow > 0 {
		// Elide below the compaction threshold, so a compaction is the rare
		// event and elision the everyday one.
		opts = append(opts, core.WithContextWindow(f.cfg.Turn.ContextWindow), core.WithStrategy(core.TokenWindow(0.7, 0.5)))
		if f.cfg.Turn.CompactAt > 0 {
			opts = append(opts, core.WithCompaction(int(f.cfg.Turn.CompactAt*float64(f.cfg.Turn.ContextWindow)), f.cfg.Turn.ContextWindow/4,
				core.Summarize(f.client, s.model)))
		}
	}
	if r == builder {
		opts = append(opts, f.turn(s, runner)...)
	}
	if f.metrics != nil {
		opts = append(opts, core.WithMetrics(f.metrics, metric.WithErrorClass(outcomeClass)))
	}
	return core.New(opts...)
}

// shared is the middleware every agent of a session carries, a sub-agent's
// included: a child that could run what its parent has to ask about would make
// the policy decorative.
func (f *factory) shared(gate tool.Middleware) []core.Option {
	var opts []core.Option
	if f.tape != nil {
		opts = append(opts, core.WithToolMiddleware(f.tape.Tools()))
	}
	opts = append(opts, core.WithToolMiddleware(gate))
	if f.tracer != nil {
		opts = append(opts, core.WithTracing(f.tracer))
	}
	return opts
}

// researcher is the sub-agent tools.delegate offers: the same client and the same
// policy, read-only tools, a short leash, and no delegate tool of its own, so
// recursion stops here. It is gated by the same skills: a child that could
// reach tools its parent has to unlock would make the gate decorative.
func (f *factory) researcher(s session, tools []tool.Def, gate tool.Middleware) (*core.Agent, error) {
	opts := []core.Option{
		core.WithName("researcher"),
		core.WithClient(f.client),
		core.WithModel(s.model),
		core.WithPurpose(llm.PurposeSubagent),
		core.WithMaxIters(12),
		core.WithLogger(f.logger),
	}
	opts = append(opts, f.shared(gate)...)
	opts = append(opts, f.toolSet(tool.Filter(tools, tool.OnlyCaps(tool.CapReadOnly|tool.CapNetwork)))...)
	if f.metrics != nil {
		opts = append(opts, core.WithMetrics(f.metrics, metric.WithErrorClass(outcomeClass)))
	}
	child, err := core.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("building the sub-agent: %w", err)
	}
	return child, nil
}

// toolSet installs a tool list, gated by the skills when there are any.
// Install is the set, the middleware that enforces it, the catalogue and the
// per-run activation state, as one option, so none of them can be left out.
func (f *factory) toolSet(tools []tool.Def) []core.Option {
	if f.skills == nil {
		return []core.Option{core.WithTools(tools...)}
	}
	return []core.Option{f.skills.Bind(tools).Install()}
}

// policy clamps a session's requested permission against the operator's.
//
// Only in the restrictive direction, with one exception the operator makes:
// "full" — everything, asking nobody — is reachable from a request when
// tools.full_access says so. "off" never is: a client asking for no gate at
// all is a client asking for an ungated shell, and the answer is the
// operator's configuration, not theirs. A session that asked for full access
// on a server that no longer offers it gets the operator's policy, like any
// other request this server would not grant.
func (f *factory) policy(requested string) string {
	want := strings.ToLower(strings.TrimSpace(requested))
	if want == "" {
		return f.cfg.Tools.Permission
	}
	if want == "full" && f.fullAccess() {
		return want
	}
	if strictness(want) > strictness(f.cfg.Tools.Permission) {
		return want
	}
	return f.cfg.Tools.Permission
}

// fullAccess reports whether a session may ask for the "full" policy.
func (f *factory) fullAccess() bool {
	return f.cfg.Tools.FullAccess || f.cfg.Tools.Permission == "full"
}

// strictness orders the policies so policy can pick the tighter of two.
func strictness(mode string) int {
	switch mode {
	case "full":
		return 0
	case "off":
		return 1
	case "workspace":
		return 2
	case "ask":
		return 3
	case "readonly", "read-only":
		return 4
	default:
		return -1 // unknown: never wins, and defaultPolicy reports it
	}
}

// workspace resolves a session's requested directory INSIDE the process root.
//
// The check is on the resolved path — symlinks followed, as far as the path
// exists — not on the string, so "../.." and a symlink that points out both
// fail, and what is returned is the directory itself: the file tools, the
// permission rules and the sandbox all go on to use this path, and a name
// that merely looks inside the root would take all three outside it. The
// comparison used to be lexical, and a link inside the root to anywhere at
// all passed. Without the check, tools.working_dir would be a suggestion:
// SessionOptions.Workspace arrives from a JSON body, and a browser that asked
// for "/" would get an agent rooted at the filesystem.
func (f *factory) workspace(requested string) (string, error) {
	if strings.TrimSpace(requested) == "" {
		return f.root, nil
	}
	abs := requested
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(f.root, abs)
	}
	abs, err := filepath.Abs(abs)
	if err == nil {
		abs, err = realPath(abs)
	}
	if err != nil {
		return "", fmt.Errorf("resolving the requested workspace: %w", err)
	}
	if !fspath.Within(f.root, abs) {
		// daemon.ErrBadRequest, so this reaches the client as a 400 naming
		// what it asked for. Without it a refusal the CLIENT caused arrives as
		// a 500, which reads as "the server is broken" and sends whoever is
		// debugging it to the wrong logs.
		return "", fmt.Errorf("%w: workspace %q is outside %s",
			daemon.ErrBadRequest, requested, f.root)
	}
	return abs, nil
}

// realPath resolves the symlinks in the longest prefix of p that exists — a
// session may name a workspace before it is created — and appends the rest.
func realPath(p string) (string, error) {
	p = filepath.Clean(p)
	rest := ""
	for {
		real, err := filepath.EvalSymlinks(p)
		if err == nil {
			return filepath.Join(real, rest), nil
		}
		parent := filepath.Dir(p)
		if !errors.Is(err, fs.ErrNotExist) || parent == p {
			return "", err
		}
		rest = filepath.Join(filepath.Base(p), rest)
		p = parent
	}
}

// confineReads is the gate a session without a policy still gets: every path
// a tool is given must be inside the workspace, and nobody is asked anything.
//
// The tool list keeps such a session away from the shell, and that is not
// enough on its own: view_file, grep_search and the git tools only read, and
// read-only is not harmless when the path is ~/.ssh/id_ed25519 and the agent
// answers to a browser.
func confineReads(workspace string) tool.Middleware {
	nobody := permission.ApproverFunc(func(context.Context, permission.Request) (bool, error) { return false, nil })
	return permission.Gate(permission.Policy{
		Rules:   []permission.Rule{permission.FSRoot(workspace)},
		Default: permission.Allow,
	}, nobody)
}

// defaultPolicy turns a permission mode into tool middleware, or nil for
// "off". "full" is a gate that allows everything: the whole tool surface
// stays on offer — no gate at all would cut it to the read-only tools — and
// every verdict is still audited.
//
// What the policy wants a person to decide is deferred, with
// [permission.Defer]: the turn ends with the call waiting, the session is
// saved mid-call, and the decision comes back over the API as an event — a
// turn never sits open waiting for someone who may be asleep. The daemon's
// own tools are let through first, whatever the mode: they touch nothing on
// this machine, and an agent that has to ask before it may say anything, or
// remind itself of anything, is one nobody can leave running.
func defaultPolicy(mode, root string) (tool.Middleware, error) {
	var p permission.Policy
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "off":
		return nil, nil
	case "readonly", "read-only":
		p = permission.ReadOnly()
	case "workspace", "":
		p = permission.Workspace(root)
	case "ask":
		p = permission.AskEverything()
	case "full":
		p = permission.Policy{Default: permission.Allow}
	default:
		return nil, fmt.Errorf("bad permission %q (want off, readonly, workspace, ask or full)", mode)
	}
	p.Rules = append([]permission.Rule{permission.AllowTools(daemon.ToolNames()...)}, p.Rules...)
	return permission.Gate(p, permission.Defer()), nil
}
