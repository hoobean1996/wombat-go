# Wombat · Web UI

The browser front end for `wombatd`: React 19 + Vite + TypeScript, with no
UI library and no state library. The production build (`dist/`) is embedded
into the `wombatd` binary by `embed.go` and served at `/`, on the same port as
the API, so there is no CORS to configure and one binary to deploy.

It started as a copy of Tsubaki's web app (`tsubaki/apps/web`) and keeps its
look — `src/styles/ui.css` and `extra.css` are carried over — but everything
that talked to Tsubaki's backend was replaced: it speaks only wombatd's HTTP
API.

## Running it

```bash
# from the repository root
make ui        # npm ci + build into ui/dist (needed before `make bin` embeds it)
make serve     # builds the UI (when npm is installed) and wombatd, then runs it
               # → open the "web UI" URL wombatd logs, http://127.0.0.1:8080/ by default
```

With hot reload, against a wombatd that is already running:

```bash
make ui-dev                                  # http://localhost:5173, /api proxied to 127.0.0.1:8080
WOMBATD_URL=http://127.0.0.1:8082 make ui-dev # a wombatd on another port
make ui-check                                # tsc + oxlint
```

If `server.token` is set in `wombatd.yaml`, the page asks for it once and keeps
it in this browser's localStorage.

## Pages

| Path | What it shows | API |
|---|---|---|
| `/` | New conversation | `POST /api/sessions` |
| `/c/:id` | A session: the transcript, the turn in progress as it streams, approval cards, the question it is waiting on | `GET /api/sessions/{id}`, `…/messages`, `…/approvals`, the SSE stream `…/events`; `POST …/messages`, `…/steer`, `…/approvals/{use_id}`; `PATCH /api/sessions/{id}` for the Normal / Full Access switch |
| `/inbox` | The events that wake sessions — webhooks, timers, relayed chats, decisions — but not what was typed into a chat, which is on its page; send one by hand | `GET /api/inbox`, `POST /api/events` |
| `/outbox` | What the agents said, less their replies to a chat; mark it handled | `GET /api/outbox` (long-polled for the badge and notifications), `POST /api/outbox/{id}/ack` |
| `/timers` | Timers, set by an agent or by hand; set and cancel them | `GET/POST /api/timers`, `DELETE /api/timers/{id}` |
| `/settings` | The token, what the server reports about itself, its tools | `GET /api/health`, `GET /api/config` |

## What the Inbox and Outbox leave out

A message typed into a conversation is an inbox event too (kind `message`,
source `api`), and the agent's answer to it is an outbox message. Both are
already on the conversation's page, so the Inbox and Outbox pages, the outbox
badge and its notifications leave them out (`src/lib/chat.ts`). An outbox
message is left out when every event in its `in_reply_to` is such a message.
Events from anywhere else are listed, relayed chats with a `from` among them.
Only the UI filters: the outbox still holds every message, pending until acked,
for any other client reading it.

## How a session is drawn

`src/lib/session.ts` is the one subtle part. wombatd commits a session's
transcript only when a turn ends, so `GET …/messages` never contains half a
turn. The page draws:

1. the committed transcript, grouped into exchanges (`src/lib/transcript.ts`):
   what was said, then the agent's tool calls (collapsed) and its reply;
2. the turn in progress, built from the event stream — `turn_started`,
   `text_delta`, `tool_start`/`tool_done`, `permission_requested`, … — in the
   order things happened;

and when `turn_ended` arrives it drops (2) and fetches (1) again, which now
contains it. Nothing is shown twice. A session that is running when the page
opens is streamed from sequence 0, so its turn so far is rebuilt; otherwise the
stream starts at the session's next sequence number. EventSource reconnects by
itself and resumes with `Last-Event-ID`.

## Layout

```
src/
  App.tsx              connecting, the token gate, the shell, routing
  types.ts             wombatd's wire types, by hand from the Go structs
  lib/api.ts           every endpoint; the token; the error envelope
  lib/store.ts         shared state: connection, config, sessions (polled), outbox (long-polled)
  lib/session.ts       one session: transcript + live turn from the event stream
  lib/transcript.ts    transcript → exchanges
  lib/chat.ts          what the Inbox and Outbox pages leave out: the chat itself
  lib/router.ts        a 30-line history router
  components/          Sidebar, Thread, Composer, Markdown, ui primitives
  pages/               Home, Chat, Inbox, Outbox, Timers, Settings
  styles/              ui.css + extra.css (from Tsubaki), wombat.css (added here)
embed.go               package ui: go:embed all:dist
```

`src/types.ts` is written by hand from `pkg/daemon`, `pkg/httpapi` and
`pkg/core/llm`; a field added in Go has to be added here too.

Replies are rendered from Markdown without HTML, and images in them are links
rather than `<img>` (`components/Markdown.tsx`): a reply is model output and
must not make the browser fetch anything. wombatd also serves the page with a
Content-Security-Policy that keeps it to its own origin, and refuses framing.
