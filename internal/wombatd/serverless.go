package wombatd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"time"

	"wombat/pkg/daemon"
	"wombat/pkg/httpapi"
	"wombat/pkg/store/postgres"
)

// ConfigEnv is the environment variable a serverless wombatd reads its
// configuration from: the YAML a wombatd.yaml would hold, with ${VARS} in it
// expanded from the environment, so a secret can live in a variable of its
// own.
const ConfigEnv = "WOMBATD_CONFIG_YAML"

// serverlessSlice is daemon.slice, and serverless.stream_limit, on a
// serverless host that sets neither: well inside the five minutes such a host
// commonly lets a request run.
const serverlessSlice = 4 * time.Minute

// Handler is wombatd as a serverless function runs it: one request at a
// time, with nothing kept between them that the store does not keep. It is
// built on the first request, from [ConfigEnv], and kept for the requests the
// same instance serves after; one that fails to build is tried again on the
// next. A Vercel function is
//
//	func Handler(w http.ResponseWriter, r *http.Request) { wombatd.Handler().ServeHTTP(w, r) }
//
// No loop runs in it. A message, a decision, a timer set — anything that
// gives a session work — POSTs to serverless.wake.url, a queue that calls
// POST /api/advance, which runs the session's next slice (daemon.slice) and
// asks again while there is more; a cron calling GET /api/tick?advance=true
// fires timers and runs what a lost wake-up left.
func Handler() http.Handler { return &lazy }

var lazy lazyHandler

// lazyHandler builds the server on its first request, and again after one
// that failed to.
type lazyHandler struct {
	mu sync.Mutex
	h  http.Handler
}

func (l *lazyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	l.mu.Lock()
	if l.h == nil {
		h, err := newServerless(os.Getenv)
		if err != nil {
			l.mu.Unlock()
			slog.Error("wombatd: not started", "err", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(httpapi.ErrorBody{Error: httpapi.ErrorDetail{Kind: "not_started", Message: err.Error()}})
			return
		}
		l.h = h
	}
	h := l.h
	l.mu.Unlock()
	h.ServeHTTP(w, r)
}

// newServerless builds a serverless wombatd from the environment getenv
// reads.
func newServerless(getenv func(string) string) (http.Handler, error) {
	cfg, err := configFromEnv(getenv)
	if err != nil {
		return nil, err
	}
	logger := newLogger(cfg.Server.Log)
	rec, err := openRecorders(cfg.Record, logger)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), openTimeout)
	defer cancel()
	// Opened, and never locked: every instance is a daemon of its own name.
	st, err := postgres.Open(ctx, cfg.Store.DSN)
	if err != nil {
		return nil, fmt.Errorf("store: %w", err)
	}
	f, err := newFactory(cfg, rec, logger)
	if err != nil {
		st.Close()
		return nil, err
	}
	probe, err := f.build(daemon.SessionOptions{})
	if err != nil {
		st.Close()
		return nil, fmt.Errorf("building the reference agent: %w", err)
	}
	mc := managerConfig(cfg, st, "", "", f.build, logger)
	mc.Stateless = true
	if w := cfg.Serverless.Wake; w.URL != "" {
		mc.Wake = daemon.WakeWebhook(w.URL, w.Headers, w.DelayHeader, nil)
	}
	mgr, err := daemon.NewManager(mc)
	if err != nil {
		st.Close()
		return nil, err
	}
	opts := handlerOptions(cfg, f, probe, logger)
	opts = append(opts, httpapi.WithStreamLimit(cfg.Serverless.StreamLimit))
	logger.Info("wombatd ready, serverless", "provider", cfg.Model.Provider, "slice", cfg.Daemon.Slice,
		"wake", cfg.Serverless.Wake.URL != "", "mode", cfg.Mode, "permission", cfg.Tools.Permission)
	return httpapi.New(mgr, opts...), nil
}

// configFromEnv is the configuration in the environment, as a serverless
// host needs it: a store that outlives the instance, a slice for each
// request, and a server that answers whatever name the host gives it.
func configFromEnv(getenv func(string) string) (config, error) {
	raw := getenv(ConfigEnv)
	if raw == "" {
		return config{}, fmt.Errorf("%s is not set: it holds the configuration, as wombatd.yaml would", ConfigEnv)
	}
	cfg, err := parseConfig([]byte(os.Expand(raw, getenv)), ConfigEnv)
	if err != nil {
		return config{}, err
	}
	if cfg.Store.Driver != driverPostgres {
		return config{}, errors.New(ConfigEnv + ": store.driver must be postgres on a serverless host: nothing else outlives a request")
	}
	if cfg.Daemon.Slice == 0 {
		cfg.Daemon.Slice = serverlessSlice
	}
	if cfg.Serverless.StreamLimit == 0 {
		cfg.Serverless.StreamLimit = serverlessSlice
	}
	// The host owns the listener and the name it is reached at; what keeps
	// a loopback server to loopback names has nothing to do here.
	cfg.Server.Addr = "0.0.0.0:0"
	return cfg, nil
}
