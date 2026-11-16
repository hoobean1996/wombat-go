<div align="center">

<img src="ui/public/favicon.svg" width="88" alt="">

# Wombat

**Always-on AI agents, in one Go binary.**

Most agents only work while you watch. A Wombat agent wakes up when something
happens (a webhook, a chat message, a timer it set for itself), does the work,
tells you what it found, and goes back to sleep. A restart loses nothing.

**English** · [简体中文](README.zh-CN.md)

![Go](https://img.shields.io/badge/Go-1.25%2B-00ADD8?logo=go&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-pgx%20only-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[Why Wombat](#why-wombat) · [How it works](#how-it-works) · [Get started](#get-started) · [What you can build](#what-you-can-build) · [Go library](#use-it-as-a-go-library) · [Docs](#documentation)

</div>

---

## Why Wombat

A chat assistant stops when you close the tab. The agents worth having keep
going. They triage the issue that came in overnight, check whether
yesterday's fix actually shipped, write the morning digest, and ask before
they touch anything risky.

Anthropic's article [Building always-on agents](https://claude.com/resources/articles/building-always-on-agents)
makes the point that such an agent runs **the same agent loop** as any other.
What changes is the plumbing around the loop: what wakes the agent, what
carries over from one run to the next, and how its work reaches you. Wombat
is that plumbing, ready to run:

| | |
|---|---|
| 📥 **Event inbox** | Webhooks, alerts, chat messages and timers all land in one durable inbox. A redelivered event with the same `dedup_key` is dropped. Events about the same `thread` reach the same session |
| ⏰ **Timers** | The agent schedules its own follow-ups ("check again in an hour") and recurring jobs ("every morning") |
| 🛡️ **Approvals** | A risky tool call waits for a person's decision without holding a turn open, however long that takes. Trust the task? Switch the session to **Full Access** and nothing asks. Commands run with a scrubbed environment, and can be confined to an OS sandbox |
| 📤 **Outbox** | Replies, questions and approval requests go to an outbox. Your app long-polls it, or wombatd POSTs each message to your webhook |
| 💾 **Durable** | Keep state in memory to try it out, or in Postgres for real use. Kill wombatd mid-turn and the next one finishes the work |
| 🗣️ **Who said what** | Every event records its `source` and its sender (`from`), so the model can tell a webhook payload from your instruction, and one person from another |
| 💸 **Budgets** | Each turn is capped in dollars, wall-clock time and steps |
| 🖥️ **Web UI** | A browser UI built into the binary: chat, approve calls, and watch the inbox, outbox and timers |

It is plain Go. The harness uses only the standard library, and the one
dependency is `pgx`, for the Postgres store. The model, the tools and the
filesystem are all interfaces you can replace.

## How it works

```mermaid
flowchart LR
    subgraph events [Things that happen]
        chat[Chat message]
        hook[Webhook or alert]
        timer[Timer]
        approve[Your approval]
    end
    events --> inbox[(Inbox)]
    inbox --> turn[One turn per session]
    turn --> agent[Agent loop:<br/>model + tools]
    agent -- set_timer --> timer
    agent --> outbox[(Outbox)]
    outbox --> you[Web UI, your app,<br/>or a webhook]
```

Every turn starts with an event. A worker hands a session **all** of its
pending events at once, runs one turn of the agent loop, and records the
result in a single transaction. If anything happens while the turn is
running, it waits for the next turn. Between turns the agent is asleep and
costs nothing.

## Get started

**You need** Go 1.25+ and access to a model: Anthropic, OpenAI, DeepSeek, any
OpenAI-compatible endpoint, or a local Ollama, which needs no key. Building
the web UI also needs Node.js and npm. Without them you still get the full
API.

### 1. Clone and configure

```bash
git clone https://github.com/hoobean1996/wombat-go.git
cd wombat-go
cp wombatd.example.yaml wombatd.yaml && chmod 600 wombatd.yaml
```

Fill in the `model` section of `wombatd.yaml`. The file is git-ignored, so
your key stays on your machine.

```yaml
model:
  provider: openai              # anthropic, or openai for any OpenAI-compatible API
  base_url: https://api.deepseek.com/v1
  api_key: sk-...
  name: deepseek-flash
```

| Provider | `provider` | `base_url` | `name`, for example |
|---|---|---|---|
| Anthropic | `anthropic` | `""` | `claude-sonnet-5-5` |
| OpenAI | `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `openai` | `https://api.deepseek.com/v1` | `deepseek-flash` |
| Ollama, local with no key | `openai` | `http://localhost:11434/v1` | `qwen2.5-coder` |

### 2. Start it

```bash
make serve
```

### 3. Open the web UI

Go to **http://127.0.0.1:8080**, describe a task, and watch it work. The reply
streams in and each tool call shows up as it happens. Anything outside the
working directory, and any command that is not obviously harmless, appears as
a card for you to allow or deny.

### 4. Wake it from the outside

Chatting is one way to wake an agent. What makes it always-on is that events
keep arriving while nobody is looking:

```bash
# A webhook: someone comments on an issue. The same thread always reaches the same session.
curl -s localhost:8080/api/events -H 'Content-Type: application/json' -d '{
  "thread": "github:you/repo#12", "kind": "issue_comment", "source": "github",
  "from": "octocat", "payload": {"body": "the build is red again"}}'

# What the agent has to say. Your app long-polls this, or set outbox.webhook to have it pushed.
curl -s 'localhost:8080/api/outbox?after=0&wait=30s'
```

The agent can also wake itself. Try asking in the chat: *"Check the build
again in an hour, and tell me if it is still red."* It sets a timer, which
you can see on the **Timers** page. An hour later the timer puts an event in
the inbox, and the agent reports through the outbox.

### 5. Make it durable

```yaml
store:
  driver: postgres
  dsn: postgres://wombat:wombat@127.0.0.1:55432/wombat   # `make pg-up` starts one in podman
server:
  token: a-long-random-string                            # set this before listening beyond localhost
```

wombatd creates its tables on first start. After a restart, including one
in the middle of a turn, it picks up where it stopped.

> [!WARNING]
> Without `server.token`, anything that can reach the port can run the agent
> and read every conversation. wombatd listens only on loopback by default.
> Set a token before you expose it anywhere else.

## What you can build

- **Issue and PR triage.** Point GitHub webhooks at `/api/events` with
  `"thread": "github:owner/repo#N"`. Each issue gets its own long-lived
  session that remembers the whole discussion.
- **An on-call companion.** Alerts arrive as events. The agent investigates
  with read-only tools and asks before it restarts anything.
- **Daily digests.** A repeating timer wakes the agent each morning. It reads
  what changed and sends a summary with `send_message`.
- **Follow-ups.** Ask it to "make sure the fix is deployed by tonight". It
  sets its own timer and comes back to check.
- **Chat bridges.** Relay Slack or Discord messages as events with `from` set
  to the speaker, and post the outbox back. One session per thread, with
  several people talking to it.

## Use it as a Go library

The agent loop is a library you can call directly:

```go
client, _ := anthropic.New(anthropic.ConfigFromEnv())
agent, _ := core.New(
	core.WithClient(llm.Chain(client, core.TrackCost(llm.DefaultPricing))),
	core.WithModel("claude-sonnet-5-5"),
	core.WithTools(builtin.Default(builtin.Deps{})...),
)

ctx, cancel := governor.WithBudget(context.Background(),
	governor.Limits{CostUSD: 1, Wall: 10 * time.Minute}) // over budget = cancelled
defer cancel()

run := agent.Start(ctx, core.Ask("How many Go files are in this directory?"))
defer run.Close()
for run.Next() { // a run is a stream of typed events
	if ev, ok := run.Event().(core.TextDelta); ok {
		fmt.Print(ev.Text)
	}
}
```

The always-on part is a library too. `daemon.NewManager` runs sessions over
a store (in memory or `postgres.Open`), and `httpapi.New` serves it as an
ordinary `http.Handler`, so you can embed all of wombatd in your own program.
For your own tools, outcomes, multi-turn runs and the `replace` directive the
`wombat` module path needs, see the **[guide](docs/GUIDE.md#en-quickstart-lib)**.

## Documentation

| | |
|---|---|
| **[Guide](docs/GUIDE.md)** | The Go library, every configuration key, the HTTP API, permissions, budgets, skills, sub-agents, MCP, tracing, and troubleshooting |
| **[Design notes](docs/DESIGN.md)** | Why each part is built the way it is, and what was learned along the way |
| **[Web UI](ui/README.md)** | How the front end works, and how to develop it |
| **[Evaluation](eval/README.md)** | The benchmark suite, and pass@k scoring |

## Development

```bash
make test       # go test -race ./...; runs offline, no API key needed
make ui-dev     # the web UI with hot reload, against a running wombatd
make help       # every target
```

## License

[MIT](LICENSE)
