// Command wombatd is the agent, always on: sessions that run on events — a
// person's message, a webhook, a timer the agent set itself — and answer
// through an outbox, over an HTTP API (see wombat/pkg/httpapi).
//
//	cp wombatd.example.yaml wombatd.yaml   # the model, and what sessions may do
//	go run ./cmd/wombatd                   # or: wombatd -config path/to/it.yaml
//
// The server is wombat/internal/wombatd; api/index.go runs the same server
// on a serverless host, one request at a time.
package main

import "wombat/internal/wombatd"

func main() { wombatd.Main() }
