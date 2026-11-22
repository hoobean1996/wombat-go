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
// the daemon was stopped mid-turn, or died — puts its events back, to be
// tried again after a backoff, up to [ManagerConfig.Attempts] times; and it
// carries on from where it got to, rather than starting over. See "The
// ledger".
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
// Until then the session takes nothing but decisions. An approved call runs
// once: the ledger says whether it ran before a resumed turn was cut short.
//
// # Slices, and a daemon that keeps nothing
//
// A turn need not run in one go. With [ManagerConfig.Slice] it stops between
// two steps once the next might not fit, written down where it got to — its
// transcript, and everything its run keeps (core.Checkpoint.State) — and its
// events go back to the inbox at once, for the next slice: here, or on any
// daemon sharing the store. The turn carries on under the same number, with
// its budget, spend and grants; its log says turn_suspended where each slice
// ended. A message steering it, or a cancellation, sent while it runs
// elsewhere or between two slices, goes through the inbox ([KindSteer],
// [KindCancel]).
//
// With [ManagerConfig.Stateless] nothing is kept between calls and no loop
// runs, as on a serverless host: [Manager.Advance] runs a session's next
// slice, [Manager.Tick] does once what the loops do, and
// [ManagerConfig.Wake] — a queue, with [WakeWebhook] — is told whenever a
// session has work, now or later. A request reads a session as a view,
// without its lease, and follows its log from the store, where every frame
// is kept.
//
// # Waiting for a result
//
// A call that starts something finishing outside the session — a build, a
// job — can end the turn instead of holding it open, with [tool.Await]; the
// agent does so with the wait_for_result tool. The session is [Awaiting], and
// the result posted for the call ([Session.Complete]) is what the call
// returns: once every call the session waits on has its result, or its
// decision, the turn resumes.
//
// # The ledger
//
// A turn writes itself down as it goes (see [core.Journal]): every reply of
// the model before anything the reply asks for happens — its tool calls,
// pending, and where the turn has got to, in one write — and every call that
// is not safe to run again as it starts and as it ends. That is the ledger
// ([ToolCall], [Manager.ToolCalls]), kept beside the session.
//
// A turn tried again — here after the provider fell over, or by the daemon
// that takes the session over after this one died — and a turn resumed on a
// decision start from what the ledger has: a call that ended is answered with
// what it returned and does not run again; one that never started runs now;
// one that was running, and so may or may not have taken effect, is run again
// if its tool is safe to run again, and otherwise reported to the model as
// interrupted — or put to a person, as its [tool.Recovery] says. The daemon's
// own tools are safe to run again: a message is sent once, and a timer set
// once, however often the call runs.
//
// A turn carried on keeps its number, its budget — what its earlier runs used
// counts against it — and its spend, counted once. Every write fences on the
// lease the turn began under and its epoch ([Lease.Epoch]), so a daemon that
// lost the session writes none of it. The ledger keeps a call for
// [ManagerConfig.LedgerRetention] after it ended.
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
// Sessions, the three tables and the ledger live in a [Store]: [MemoryStore],
// or Postgres with wombat/pkg/store/postgres. A session in memory is a cache of its
// record — it is loaded when it is needed and dropped after
// [ManagerConfig.Idle] without anything to do — and so is its event log, which
// a stream resumes from and a restart does not keep.
package daemon
