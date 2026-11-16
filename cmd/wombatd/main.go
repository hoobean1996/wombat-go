// Command wombatd is the agent, always on: sessions that run on events — a
// person's message, a webhook, a timer the agent set itself — and answer
// through an outbox, over an HTTP API (see wombat/pkg/httpapi).
//
//	cp wombatd.example.yaml wombatd.yaml   # the model, and what sessions may do
//	go run ./cmd/wombatd                   # or: wombatd -config path/to/it.yaml
//	curl -s localhost:8080/api/sessions -H 'Content-Type: application/json' -d '{"prompt":"hi"}'
//
// Everything about how a session runs lives in [wombat/pkg/daemon], and the
// routes in [wombat/pkg/httpapi]; both are tested without a socket. What is
// left here is what belongs to a command: reading the configuration, opening
// the store, building one provider client, and deciding what an agent may do
// — per session, because the directory the file tools are rooted at and the
// policy that gates them are exactly the choices a client makes when it
// starts a conversation.
//
// All of it is configured by one YAML file, and nothing else: no flag but
// -config, no environment variable. The files, from the outside in:
//
//   - main.go: start everything in order, and stop it in reverse.
//   - config.go: the file — its keys, their defaults, what they may not say.
//   - server.go: who may call, and the listener.
//   - store.go: where the sessions are kept, and where the outbox goes.
//   - record.go: the tape and the trace.
//   - factory.go: what every session shares — the client, the MCP servers,
//     the skills, the metrics.
//   - agent.go: what one session's agents are made of.
//   - done.go: when a turn is done — the gates, and the review.
package main

import (
	"cmp"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strings"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/daemon"
	"wombat/pkg/httpapi"
)

func main() {
	path := flag.String("config", "wombatd.yaml", "the configuration file; wombatd.example.yaml shows every key")
	flag.Parse()
	if flag.NArg() > 0 {
		fmt.Fprintf(os.Stderr, "wombatd: unexpected arguments %q; everything is configured in %s\n", flag.Args(), *path)
		os.Exit(2)
	}

	cfg, err := loadConfig(*path)
	if err == nil {
		cfg, err = cfg.hidingFile(*path)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "wombatd:", err)
		os.Exit(2)
	}
	logger := newLogger(cfg.Server.Log)
	warnIfReadable(*path, cfg, logger)

	if err := run(cfg, logger); err != nil {
		logger.Error("wombatd", "err", err)
		os.Exit(1)
	}
}

// run starts everything in the order it depends on, and the defers stop it in
// reverse: the recorders outlive the sessions that write to them, the store
// and the MCP servers outlive the sessions that use them, and the listener
// drains before the sessions it serves are stopped — which puts what they
// were running back in the inbox, for the next start.
func run(cfg config, logger *slog.Logger) error {
	rec, err := openRecorders(cfg.Record, logger)
	if err != nil {
		return err
	}
	defer rec.close()

	name, addr := identity(cfg.Server, cfg.Daemon)
	st, err := openStore(cfg.Store, name, logger)
	if err != nil {
		return err
	}
	defer st.Close()

	f, err := newFactory(cfg, rec, logger)
	if err != nil {
		return err
	}
	defer f.close()

	// Built once now, for two reasons: a misconfiguration fails here rather
	// than on a client's first request, and GET /api/config describes the
	// tools of an agent that was actually built, not a list typed twice.
	probe, err := f.build(daemon.SessionOptions{})
	if err != nil {
		return fmt.Errorf("building the reference agent: %w", err)
	}

	mgr, err := newManager(cfg, st, name, addr, f.build, logger)
	if err != nil {
		return err
	}
	defer mgr.Close()

	logger.Info("listening", "addr", cfg.Server.Addr, "daemon", name, "advertise", addr, "provider", cfg.Model.Provider,
		"mode", cfg.Mode, "permission", cfg.Tools.Permission, "full_access", f.fullAccess(), "working_dir", f.root,
		"store", cfg.Store.Driver, "outbox_webhook", cfg.Outbox.Webhook != "")
	if cfg.Store.Driver == driverPostgres && addr == "" {
		logger.Warn("server.advertise is empty and cannot be worked out from server.addr: other daemons on this store cannot forward this one the requests for the sessions it runs")
	}
	return serve(cfg.Server.Addr, httpapi.New(mgr, handlerOptions(cfg, f, probe, logger)...), logger)
}

// identity is who this daemon is among those sharing its store: its name —
// daemon.name, or this host and port, the same each time it starts so that it
// takes its sessions back at once — and the address another daemon forwards
// it requests at: server.advertise, or server.addr when that names a host.
func identity(srv serverConfig, d daemonConfig) (name, addr string) {
	host, port, err := net.SplitHostPort(srv.Addr)
	name = d.Name
	if name == "" {
		h, _ := os.Hostname()
		name = cmp.Or(h, "wombatd") + ":" + cmp.Or(port, "0")
	}
	addr = srv.Advertise
	if addr == "" && err == nil && host != "" && host != "0.0.0.0" && host != "::" {
		addr = "http://" + net.JoinHostPort(host, port)
	}
	return name, strings.TrimSuffix(addr, "/")
}

// newManager runs the sessions: where they are kept, how each one's agent is
// built, how the inbox is worked, and what one turn may spend.
func newManager(cfg config, st daemon.Store, name, addr string, build func(daemon.SessionOptions) (*core.Agent, error), logger *slog.Logger) (*daemon.Manager, error) {
	mc := daemon.ManagerConfig{
		Store: st,
		Build: build,
		Name:  name,
		Addr:  addr,
		// No Steps limit. Each agent's own iteration cap bounds it, and Steps
		// counts the whole tree: a delegated task's iterations and every
		// review sweep's are charged to the turn. Set to max_iters+1 it never
		// fired before the cap did for a lone agent, and cut a review off in
		// its first sweep. The tree is bounded by what it costs and how long
		// it takes instead.
		Limits: governor.Limits{
			CostUSD:           cfg.Turn.BudgetUSD,
			Wall:              cfg.Turn.Wall,
			RepeatedToolCalls: 5,
		},
		Poll:     cfg.Daemon.Poll,
		MaxRuns:  cfg.Daemon.MaxRuns,
		Attempts: cfg.Daemon.Attempts,
		Idle:     cfg.Daemon.Idle,
		Logger:   logger,
	}
	if cfg.Outbox.Webhook != "" {
		mc.Deliver = daemon.Webhook(cfg.Outbox.Webhook, cfg.Outbox.Headers, nil)
	}
	return daemon.NewManager(mc)
}
