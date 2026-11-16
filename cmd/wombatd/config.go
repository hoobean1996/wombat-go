package main

import (
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"maps"
	"net/url"
	"os"
	"regexp"
	"slices"
	"strings"
	"time"

	"wombat/internal/yaml"
	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/sandbox"
	"wombat/pkg/core/tape"
	"wombat/pkg/daemon"
)

// config is wombatd.yaml.
//
// Every key has a default, set in defaults, so a file names only what it
// changes. wombatd.example.yaml, at the root of the repository, shows every
// key with its default and says what it is for — the documentation lives
// there, where the person editing the file reads it, and
// TestTheExampleConfigShowsEveryKeyAndItsDefault keeps it honest.
type config struct {
	Server serverConfig `yaml:"server"` // who may call
	Store  storeConfig  `yaml:"store"`  // where the sessions, the inbox, the timers and the outbox are kept
	Daemon daemonConfig `yaml:"daemon"` // how the inbox is worked
	Outbox outboxConfig `yaml:"outbox"` // where what the agent says is sent
	Model  modelConfig  `yaml:"model"`  // the provider every session talks to
	Tools  toolsConfig  `yaml:"tools"`  // what a session may touch
	Turn   turnConfig   `yaml:"turn"`   // how far one turn may go
	Mode   string       `yaml:"mode"`   // what every session does: build, plan or review
	Done   doneConfig   `yaml:"done"`   // when a build is done
	Review reviewConfig `yaml:"review"` // the adversarial review after an answer
	Record recordConfig `yaml:"record"` // what is written down, to replay or to read
}

type serverConfig struct {
	Addr      string   `yaml:"addr"`
	Advertise string   `yaml:"advertise"`
	Token     string   `yaml:"token"`
	CORS      []string `yaml:"cors"`
	Metrics   bool     `yaml:"metrics"`
	Log       string   `yaml:"log"`
}

type storeConfig struct {
	Driver string `yaml:"driver"`
	DSN    string `yaml:"dsn"`
}

type daemonConfig struct {
	Name     string        `yaml:"name"`
	Poll     time.Duration `yaml:"poll"`
	MaxRuns  int           `yaml:"max_runs"`
	Attempts int           `yaml:"attempts"`
	Idle     time.Duration `yaml:"idle"`
}

type outboxConfig struct {
	Webhook string            `yaml:"webhook"`
	Headers map[string]string `yaml:"headers"`
}

type modelConfig struct {
	Provider  string            `yaml:"provider"`
	Name      string            `yaml:"name"`
	BaseURL   string            `yaml:"base_url"`
	APIKey    string            `yaml:"api_key"`
	Proxy     string            `yaml:"proxy"`
	Headers   map[string]string `yaml:"headers"`
	Stream    string            `yaml:"stream"`
	MaxTokens int               `yaml:"max_tokens"`
	System    string            `yaml:"system"`
}

type toolsConfig struct {
	WorkingDir     string `yaml:"working_dir"`
	Permission     string `yaml:"permission"`
	FullAccess     bool   `yaml:"full_access"`
	Confine        bool   `yaml:"confine"`
	ConfineNetwork string `yaml:"confine_network"`
	MCP            string `yaml:"mcp"`
	SkillsDir      string `yaml:"skills_dir"`
	Media          bool   `yaml:"media"`
	Delegate       bool   `yaml:"delegate"`
}

type turnConfig struct {
	BudgetUSD       float64       `yaml:"budget_usd"`
	Wall            time.Duration `yaml:"wall"`
	MaxIters        int           `yaml:"max_iters"`
	ToolParallelism int           `yaml:"tool_parallelism"`
	ContextWindow   int           `yaml:"context_window"`
	CompactAt       float64       `yaml:"compact_at"`
}

type doneConfig struct {
	Check       string `yaml:"check"`
	Attempts    int    `yaml:"attempts"`
	Accept      bool   `yaml:"accept"`
	Regressions string `yaml:"regressions"`
}

type reviewConfig struct {
	Enabled bool `yaml:"enabled"`
	Sweeps  int  `yaml:"sweeps"`
	Dry     int  `yaml:"dry"`
	Iters   int  `yaml:"iters"`
	Fix     int  `yaml:"fix"`
}

type recordConfig struct {
	Tape      string `yaml:"tape"`
	TapeMode  string `yaml:"tape_mode"`
	Trace     string `yaml:"trace"`
	TraceHTML string `yaml:"trace_html"`
}

// The stores a session can be kept in.
const (
	driverMemory   = "memory"   // this process, and gone with it
	driverPostgres = "postgres" // a database, and there when the daemon comes back
)

// The modes, which decide what every session's turns are for.
const (
	modeBuild  = "build"  // do the task
	modePlan   = "plan"   // write the plan for it, and no code
	modeReview = "review" // review the code already in the workspace
)

func defaults() config {
	return config{
		Server: serverConfig{Addr: "127.0.0.1:8080", Metrics: true, Log: "info"},
		Store:  storeConfig{Driver: driverMemory},
		Daemon: daemonConfig{
			Poll: daemon.DefaultPoll, MaxRuns: daemon.DefaultMaxRuns,
			Attempts: daemon.DefaultAttempts, Idle: daemon.DefaultIdle,
		},
		Model:  modelConfig{Provider: "openai", Stream: "auto", MaxTokens: core.DefaultMaxTokens},
		Tools:  toolsConfig{WorkingDir: ".", Permission: "workspace", ConfineNetwork: "loopback"},
		Turn:   turnConfig{BudgetUSD: 1, MaxIters: 12, ToolParallelism: core.DefaultToolParallelism},
		Mode:   modeBuild,
		Done:   doneConfig{Attempts: 3},
		Review: reviewConfig{Sweeps: 1, Dry: 2},
		Record: recordConfig{TapeMode: "auto"},
	}
}

// daemonName is what daemon.name may look like.
var daemonName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,127}$`)

// The words a key may hold, and what each one means.
var (
	drivers     = map[string]bool{driverMemory: true, driverPostgres: true}
	providers   = map[string]bool{"openai": true, "anthropic": true}
	permissions = map[string]bool{"off": true, "readonly": true, "workspace": true, "ask": true, "full": true}
	modes       = map[string]bool{modeBuild: true, modePlan: true, modeReview: true}
	logLevels   = map[string]slog.Level{"debug": slog.LevelDebug, "info": slog.LevelInfo, "warn": slog.LevelWarn, "error": slog.LevelError}
	streamModes = map[string]llm.StreamMode{"auto": llm.StreamAuto, "always": llm.StreamAlways, "never": llm.StreamNever}
	tapeModes   = map[string]tape.Mode{"auto": tape.Auto, "record": tape.Record, "replay": tape.Replay}
	networks    = map[string]sandbox.Network{"loopback": sandbox.NetLoopback, "none": sandbox.NetNone, "full": sandbox.NetFull}
)

// loadConfig reads the file at path over the defaults, and refuses one that
// cannot mean what it says — at startup, rather than on the first session.
func loadConfig(path string) (config, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return config{}, fmt.Errorf("%s does not exist; copy wombatd.example.yaml to start from", path)
	}
	if err != nil {
		return config{}, err
	}
	cfg := defaults()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return config{}, fmt.Errorf("%s: %w", path, err)
	}
	if errs := cfg.validate(); len(errs) > 0 {
		for i, err := range errs {
			errs[i] = fmt.Errorf("%s: %w", path, err)
		}
		return config{}, errors.Join(errs...)
	}
	return cfg, nil
}

// validate reports every key that holds what it may not, by its path in the
// file.
func (c config) validate() []error {
	var errs []error
	bad := func(key, format string, args ...any) {
		errs = append(errs, fmt.Errorf("%s: %s", key, fmt.Sprintf(format, args...)))
	}
	oneOf := func(key, v string, words []string) {
		if !slices.Contains(words, v) {
			bad(key, "%q is not one of %s", v, strings.Join(words, ", "))
		}
	}
	atLeast := func(key string, v, least int) {
		if v < least {
			bad(key, "%d is less than %d", v, least)
		}
	}

	if c.Server.Addr == "" {
		bad("server.addr", "is empty")
	}
	if a := c.Server.Advertise; a != "" {
		if u, err := url.Parse(a); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || (u.Path != "" && u.Path != "/") {
			bad("server.advertise", "%q is not an http(s)://host:port address", a)
		}
	}
	if n := c.Daemon.Name; n != "" && !daemonName.MatchString(n) {
		bad("daemon.name", "%q is not a name: letters, digits and . _ : / @ -, at most 128", n)
	}
	oneOf("server.log", c.Server.Log, slices.Sorted(maps.Keys(logLevels)))

	oneOf("store.driver", c.Store.Driver, slices.Sorted(maps.Keys(drivers)))
	switch {
	case c.Store.Driver == driverPostgres && c.Store.DSN == "":
		bad("store.dsn", "is empty, and store.driver postgres needs one: postgres://user:password@host:5432/db")
	case c.Store.Driver == driverMemory && c.Store.DSN != "":
		bad("store.dsn", "is set, and store.driver is memory: say postgres to keep sessions there")
	}
	positive := func(key string, d time.Duration) {
		if d <= 0 {
			bad(key, "%v is not a positive duration", d)
		}
	}
	positive("daemon.poll", c.Daemon.Poll)
	positive("daemon.idle", c.Daemon.Idle)
	atLeast("daemon.max_runs", c.Daemon.MaxRuns, 1)
	atLeast("daemon.attempts", c.Daemon.Attempts, 1)
	if c.Outbox.Webhook != "" {
		if u, err := url.Parse(c.Outbox.Webhook); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			bad("outbox.webhook", "%q is not an http or https URL", c.Outbox.Webhook)
		}
	} else if len(c.Outbox.Headers) > 0 {
		bad("outbox.headers", "are sent to outbox.webhook, which is not set")
	}

	oneOf("model.provider", c.Model.Provider, slices.Sorted(maps.Keys(providers)))
	oneOf("model.stream", c.Model.Stream, slices.Sorted(maps.Keys(streamModes)))
	atLeast("model.max_tokens", c.Model.MaxTokens, 1)

	oneOf("tools.permission", c.Tools.Permission, slices.Sorted(maps.Keys(permissions)))
	oneOf("tools.confine_network", c.Tools.ConfineNetwork, slices.Sorted(maps.Keys(networks)))

	if c.Turn.BudgetUSD < 0 {
		bad("turn.budget_usd", "is negative; 0 is no cap")
	}
	if c.Turn.Wall < 0 {
		bad("turn.wall", "is negative; 0 is no cap")
	}
	atLeast("turn.max_iters", c.Turn.MaxIters, 1)
	atLeast("turn.tool_parallelism", c.Turn.ToolParallelism, 1)
	atLeast("turn.context_window", c.Turn.ContextWindow, 0)
	switch {
	case c.Turn.CompactAt == 0:
	case c.Turn.CompactAt < 0 || c.Turn.CompactAt >= 1:
		bad("turn.compact_at", "%v is not a fraction between 0 and 1", c.Turn.CompactAt)
	case c.Turn.ContextWindow == 0:
		bad("turn.compact_at", "is a fraction of turn.context_window, which is not set")
	}

	oneOf("mode", c.Mode, slices.Sorted(maps.Keys(modes)))
	atLeast("done.attempts", c.Done.Attempts, 1)
	atLeast("review.sweeps", c.Review.Sweeps, 1)
	atLeast("review.dry", c.Review.Dry, 1)
	atLeast("review.iters", c.Review.Iters, 0)
	atLeast("review.fix", c.Review.Fix, 0)

	oneOf("record.tape_mode", c.Record.TapeMode, slices.Sorted(maps.Keys(tapeModes)))
	if c.Record.TraceHTML != "" && c.Record.Trace == "" {
		bad("record.trace_html", "renders the record.trace file, which is not set")
	}
	return errs
}

// warnIfReadable complains, as ssh does about a private key, when the file
// holds a secret and anyone but its owner can read it: a token, a key, a
// database's password, a webhook's credentials.
func warnIfReadable(path string, cfg config, logger *slog.Logger) {
	if cfg.Server.Token == "" && cfg.Model.APIKey == "" && cfg.Store.DSN == "" && len(cfg.Outbox.Headers) == 0 {
		return
	}
	if fi, err := os.Stat(path); err == nil && fi.Mode().Perm()&0o077 != 0 {
		logger.Warn("the configuration holds a secret and others can read it; chmod 600 it",
			"file", path, "mode", fi.Mode().Perm())
	}
}

func newLogger(level string) *slog.Logger {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevels[level]}))
	// permission.Gate audits every verdict through slog.Default, so the
	// deployment's handler has to be installed there too, or the audit trail
	// goes to a different stream from everything else.
	slog.SetDefault(logger)
	return logger
}
