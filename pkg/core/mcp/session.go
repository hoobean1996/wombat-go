package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
)

// MetaSession is the _meta key a tool call carries the calling session's id
// under, for a server configured with Server.Session.
const MetaSession = "wombat/session"

// MetaCall is the _meta key every tool call the harness dispatches carries a
// key for itself under, for a server that makes its effects idempotent: the
// same whenever the call runs — retried, or run again by a run resumed after
// its process died — and no other call's. It is a digest of the call's name
// (tool.CallKey), so it says nothing about the session to a server not told
// that.
const MetaCall = "wombat/call"

// callDigest is what MetaCall carries for the call named key.
func callDigest(key string) string {
	sum := sha256.Sum256([]byte("wombat/call:" + key))
	return hex.EncodeToString(sum[:16])
}

type sessionKey struct{}

// WithSession attaches the session a turn runs for to the turn's context.
// Only the servers configured with Server.Session are told it.
func WithSession(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, sessionKey{}, id)
}

func sessionFrom(ctx context.Context) string {
	id, _ := ctx.Value(sessionKey{}).(string)
	return id
}
