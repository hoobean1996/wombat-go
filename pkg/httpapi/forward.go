package httpapi

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"

	"wombat/pkg/daemon"
)

/*
 * Several daemons can share one store, each running the sessions it holds
 * (daemon.Store.Acquire). What a session has in memory — the turn in flight,
 * the event log a stream follows, a steer, a cancel — is on the daemon that
 * runs it, so a request about a session is served there: by this daemon when
 * it holds the session, or takes it because nobody does; and forwarded,
 * whole, to the one that holds it otherwise. A load balancer can then send
 * any request to any daemon.
 *
 * Forwarded once at most: a daemon that receives a forwarded request for a
 * session it does not hold — it moved meanwhile — says so with 503 elsewhere
 * rather than forwarding it on, and the client tries again.
 */

// forwardedBy marks a request one daemon forwarded to another.
const forwardedBy = "X-Wombat-Forwarded-By"

// placed serves a request about the session named by the path's param where
// that session runs.
func (a *api) placed(param string, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Taken here if nobody holds it; any other error the handler meets
		// again, and reports as it always has.
		_, err := a.m.Get(r.Context(), r.PathValue(param))
		var owned *daemon.OwnedError
		if errors.As(err, &owned) {
			a.forward(w, r, owned)
			return
		}
		h(w, r)
	}
}

// forward hands r to the daemon that holds the session, and its answer —
// a stream included — back to the client.
func (a *api) forward(w http.ResponseWriter, r *http.Request, owned *daemon.OwnedError) {
	target, err := url.Parse(owned.Lease.Addr)
	if owned.Lease.Addr == "" || err != nil || target.Host == "" || r.Header.Get(forwardedBy) != "" {
		a.fail(w, r, owned)
		return
	}
	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.SetXForwarded()
			pr.Out.Header.Set(forwardedBy, a.m.Name())
			// CORS is answered here, by the daemon the browser asked; the
			// other's own headers would only double them.
			pr.Out.Header.Del("Origin")
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			a.fail(w, r, fmt.Errorf("%w (forwarding to %s: %v)", owned, owned.Lease.Addr, err))
		},
	}
	proxy.ServeHTTP(w, r)
}
