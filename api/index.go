// Package handler is wombatd on Vercel: every request to /api, /files and
// /metrics comes here (vercel.json), and is served by the same server
// cmd/wombatd runs, one request at a time — no loop, nothing kept between
// requests but the Postgres store. See wombat/internal/wombatd.Handler for
// what it needs: WOMBATD_CONFIG_YAML, a postgres store, and a queue — or a
// cron — to run the next slice of a session's turn.
package handler

import (
	"net/http"

	"wombat/internal/wombatd"
)

// Handler serves one request.
func Handler(w http.ResponseWriter, r *http.Request) {
	wombatd.Handler().ServeHTTP(w, r)
}
