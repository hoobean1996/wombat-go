// Package daemon keeps agent sessions going with nobody watching: the
// always-on half of wombat, and HTTP-free, so all of it is tested without a
// socket — wombat/pkg/httpapi is a thin projection of it onto routes.
//
// # Everything is an event
//
// Nothing starts a turn but an event in a session's inbox: a person's
// message, a decision on a call that was waiting for one, a timer the agent
// set for itself, anything a client posts — a webhook, a CI result. The
// inbox is durable, so an event is safe once [Manager.Receive] or
// [Session.Send] returns, whatever happens to this process after; and an
// event can carry a dedup key, so a webhook delivered twice is handled once.
//
// A worker takes a session's pending events — all of them, at once — and
// runs one turn on them. A session never has two turns at once: what arrives
// during one waits for the next, and is handed over together, so a burst of
// events is one turn and not ten. Turns of different sessions run side by
// side, up to [ManagerConfig.MaxRuns].
//
// A turn that fails for a reason worth another try — the provider fell over,
// the daemon was stopped mid-turn — puts its events back, to be tried again
// after a backoff, up to [ManagerConfig.Attempts] times; it runs again from
// the transcript as it was before, so nothing it did is in the conversation
// twice. Anything else a turn did on the way — a file written, a message sent
// — it may do again: delivery is at least once.
//
// # Timers
//
// A timer puts an event in its session's inbox when it is due: once — a
// follow-up — or every so often — a schedule. The agent sets its own with the
// set_timer tool; a client, with [Manager.AddTimer]. A daemon that was down
// when timers were due fires each once when it comes back, not once per
// period it missed.
//
// # Decisions
//
// A call the permission policy wants a person to approve does not hold the
// turn open: the gate defers it (see [permission.Defer]), the turn ends in
// [core.Deferred], the session is saved mid-call and is [Approving], and the
// outbox says what is waiting. The decision comes back as an event — see
// [Session.Resolve] — and once every call is decided the turn resumes where
// it stopped, the approved calls run, and the model reads their real output.
// Until then the session takes nothing but decisions.
//
// # The outbox
//
// What the agent has to say goes to the outbox: the reply to a person's
// messages, a question it asks, the calls it needs approved, and what it
// chooses to say with the send_message tool — a timer's turn says nothing
// unless it does. A client reads the outbox and acks what it has handled; with
// [ManagerConfig.Deliver] set — see [Webhook] — the daemon sends each message
// itself.
//
// # What is kept
//
// Sessions and the three tables live in a [Store]: [MemoryStore], or Postgres
// with wombat/pkg/store/postgres. A session in memory is a cache of its
// record — it is loaded when it is needed and dropped after
// [ManagerConfig.Idle] without anything to do — and so is its event log, which
// a stream resumes from and a restart does not keep.
package daemon
