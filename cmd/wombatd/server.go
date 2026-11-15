package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/httpapi"
)

// shutdownGrace is how long an in-flight request has to finish once a signal
// arrives. Short, because the long-lived responses — the event stream, a long
// poll of the outbox — lose nothing by being dropped: a client reconnects to
// the next daemon and picks up where it was.
const shutdownGrace = 5 * time.Second

// handlerOptions is everything the HTTP surface needs that is not a session:
// what /api/config says, /metrics, and who may call.
func handlerOptions(cfg config, f *factory, probe *core.Agent, logger *slog.Logger) []httpapi.Option {
	opts := []httpapi.Option{
		httpapi.WithVersion(version()),
		httpapi.WithCapabilities(httpapi.Capabilities{
			DefaultModel: cfg.Model.Name,
			Approvals:    cfg.Tools.Permission != "off",
			Tools:        httpapi.ToolsOf(probe),
		}),
	}
	if f.registry != nil {
		opts = append(opts, httpapi.WithMetrics(f.registry.Handler()))
	}
	if len(cfg.Server.CORS) > 0 {
		opts = append(opts, httpapi.WithCORS(cfg.Server.CORS...))
	}
	return append(opts, access(cfg.Server.Addr, cfg.Server.Token, logger)...)
}

// access decides who may call. The token, when there is one, from anywhere;
// and on a loopback address, only by the names a browser on this machine
// knows it by — which a page that rebinds a name of its own to 127.0.0.1 is
// not using. A server anyone can reach and no one has to authenticate to says
// so in its log.
func access(addr, token string, logger *slog.Logger) []httpapi.Option {
	var opts []httpapi.Option
	if token != "" {
		opts = append(opts, httpapi.WithBearerToken(token))
		logger.Info("API requests must carry server.token as a bearer token")
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}
	if loopback(host) {
		opts = append(opts, httpapi.WithAllowedHosts("localhost", "127.0.0.1", "::1", host))
	} else if token == "" {
		logger.Warn("listening beyond loopback with no server.token: anyone who can reach this address can run the agent, spend its budget and read every transcript",
			"addr", addr)
	}
	return opts
}

func loopback(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(strings.TrimSuffix(strings.TrimPrefix(host, "["), "]"))
	return ip != nil && ip.IsLoopback()
}

// serve listens until a signal, then lets the requests in flight finish. The
// sessions are stopped after that, by run's deferred Manager.Close, which puts
// the events of a turn it stops back in the inbox: a client that is mid-poll
// gets an answer rather than a reset connection, and nothing it sent is lost.
func serve(addr string, h http.Handler, logger *slog.Logger) error {
	srv := &http.Server{
		Addr:    addr,
		Handler: h,
		// No WriteTimeout: it bounds a whole response, and the event stream is
		// meant to stay open for the length of a conversation. What ends a turn
		// is the per-turn budget.
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errs := make(chan error, 1)
	go func() { errs <- srv.ListenAndServe() }()

	select {
	case err := <-errs:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("serving: %w", err)
	case <-ctx.Done():
		logger.Info("shutting down")
		shutdown, cancel := context.WithTimeout(context.Background(), shutdownGrace)
		defer cancel()
		return srv.Shutdown(shutdown)
	}
}

// version is what GET /api/health and GET /api/config report. Read from the
// build info rather than a -ldflags variable, so `go install` produces a
// binary that knows what it is.
func version() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "dev"
	}
	for _, s := range info.Settings {
		if s.Key == "vcs.revision" && len(s.Value) >= 12 {
			return s.Value[:12]
		}
	}
	if info.Main.Version != "" {
		return info.Main.Version
	}
	return "dev"
}
