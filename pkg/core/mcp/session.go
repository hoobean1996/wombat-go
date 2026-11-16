package mcp

import "context"

// MetaSession is the _meta key a tool call carries the calling session's id
// under, for a server configured with Server.Session.
const MetaSession = "wombat/session"

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
