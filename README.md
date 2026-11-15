<div align="center">

# wombat-go

**A ReAct agent harness for Go, and an always-on agent service built on it.**

**English** · [简体中文](README.zh-CN.md)

![Go](https://img.shields.io/badge/Go-1.25%2B-00ADD8?logo=go&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-pgx%20only-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

---

- [What is wombat-go?](#en-what)
- [Features](#en-features)
- [Requirements](#en-requirements)
- [Quick start: the service](#en-quickstart)
- [Quick start: Go library](#en-quickstart-lib)
- [Core concepts](#en-concepts)
- [Built-in tools](#en-tools)
- [Configuration](#en-config)
- [Always on](#en-always)
- [HTTP API](#en-http)
- [Commands and make targets](#en-cli)
- [Feature guide](#en-guide)
- [Project layout](#en-layout)
- [Development](#en-dev)
- [Troubleshooting](#en-faq)
- [Design notes](#en-design)
- [License](#en-license)

<a id="en-what"></a>

## What is wombat-go?

wombat-go is a Go library for building **LLM agents**, and `wombatd`, a service
that keeps them running. The library runs the classic **ReAct** loop: ask the
model, run the tools it asks for, feed the results back, and repeat until the
model is done.

```
              ┌───────────────────────────────────────────────┐
              ▼                                               │
  task ──▶  model  ──▶ wants a tool? ── yes ──▶ run tools ──▶ results
                            │
                            no
                            ▼
                       final answer
```

You supply three things:

1. **A model client**: Anthropic, or any OpenAI-compatible endpoint (OpenAI,
   DeepSeek, vLLM, Ollama, a corporate gateway).
2. **Tools** the model may call: a shell, file editing, grep, git, HTTP, MCP
   servers, or functions you write.
3. **A task**, and optionally a **budget** (cost, time, steps) it must stay within.

`wombatd` puts that loop behind an HTTP API and keeps it going with nobody
watching: a session wakes on **events** — a person's message, a webhook, a
timer the agent set for itself — runs one turn on them, and says what it has
to say through an **outbox**. Everything it needs to survive a restart is kept
in memory or in Postgres.

It is written for people who want to see and control every part of the loop.
The harness is plain Go using only the standard library — the Postgres store
is the one package with a dependency — and every dependency is injected, so
the model, a tool or the filesystem can each be swapped for a fake in a test.

<a id="en-features"></a>

## Features

| | |
|---|---|
| **Two providers, one interface** | Anthropic Messages API and any OpenAI-compatible `/chat/completions` endpoint, including gateways that need proxies and custom headers |
| **Streaming events** | A run is an iterator of typed events (`TextDelta`, `ReasoningDelta`, `ToolStart`, …) that you can render in a terminal, stream over SSE, or assert on in a test |
| **12 built-in tools** | `bash`, `view_file`, `write_file`, `str_replace`, `grep_search`, `git_log`, `http_get`, …, plus MCP tools, your own, and the daemon's `send_message` and timer tools |
| **Budgets** | Cap cost in USD, wall-clock time, steps, tool calls and sub-agent depth. Going over a limit cancels the run cleanly |
| **Safety** | Per-call permission policies, with human approval that never holds a turn open; a scrubbed environment for child processes; and an optional OS sandbox (`sandbox-exec` / `bwrap`) |
| **Long tasks** | Token-aware context trimming, automatic summarization (compaction), and recovery from context-overflow errors |
| **Skills and sub-agents** | Tools that appear only after a skill is loaded, and child agents that share the parent's budget |
| **Always on** | Sessions run on events from a durable inbox, one turn at a time; timers wake them later; failed turns are tried again; what they say goes to an outbox, which a client polls or a webhook receives |
| **Memory or Postgres** | One store interface, two implementations that pass the same conformance suite; a restart carries on where it stopped |
| **HTTP API** | Sessions, a resumable SSE stream per session, the inbox, the outbox and timers, with bearer-token auth |
| **Observability** | OpenTelemetry-shaped traces with an HTML viewer, Prometheus metrics, and record/replay tapes |
| **Evaluation** | A benchmark suite of coding tasks with pass@k scoring and anti-cheat checks |

<a id="en-requirements"></a>

## Requirements

| | |
|---|---|
| **Go** | 1.25.6 or newer |
| **An LLM API** | An Anthropic API key, **or** any OpenAI-compatible endpoint. A local Ollama or vLLM server needs no key |
| *Optional* | Postgres, for `store.driver: postgres` — or podman, for `make pg-up` |
| *Optional* | `sandbox-exec` (built into macOS) or `bwrap` (Linux), for `tools.confine` |

<a id="en-quickstart"></a>

## Quick start: the service

**1. Clone and configure**

```bash
git clone https://github.com/hoobean1996/wombat-go.git
cd wombat-go
cp wombatd.example.yaml wombatd.yaml
```

`wombatd` is configured by this one file — no flags but `-config`. Fill in
the `model` section for **one** provider; `wombatd.yaml` is git-ignored, so
your key stays on your machine. (Leave `api_key` empty and the provider looks
for `$OPENAI_API_KEY` or `$ANTHROPIC_API_KEY` before giving up; that is the one
thing wombatd takes from the environment.)

```yaml
# Option A: OpenAI, or any OpenAI-compatible API (DeepSeek, a gateway, ...)
model:
  provider: openai
  base_url: https://api.openai.com/v1
  api_key: sk-...
  name: gpt-4o-mini
```

```yaml
# Option B: Anthropic
model:
  provider: anthropic
  base_url: ""          # empty is the public endpoint
  api_key: sk-ant-...
  name: claude-sonnet-5-5
```

```yaml
# Option C: a local model with Ollama (no key needed)
model:
  provider: openai
  base_url: http://localhost:11434/v1
  api_key: ""           # empty sends no key
  name: qwen2.5-coder
```

The file holds a secret now, so `chmod 600 wombatd.yaml`; wombatd warns when
others can read it.

**2. Start it**

```bash
make serve        # builds bin/wombatd and runs it with wombatd.yaml
```

It listens on `127.0.0.1:8080` and keeps everything in memory. To keep
sessions across restarts, see [Always on](#en-always).

**3. Ask the agent something, and watch it work**

```bash
curl -s localhost:8080/api/sessions -H 'Content-Type: application/json' \
     -d '{"prompt": "how many Go files are in this repo? use bash"}'   # → {"id": "...", ...}
curl -sN localhost:8080/api/sessions/<id>/events                    # the run, as server-sent events
```

Under the default policy, a command that is not obviously harmless — and
anything outside `tools.working_dir` — waits for you. The turn stops on the
call, and the session is `approving`:

```bash
curl -s localhost:8080/api/sessions/<id>/approvals                  # what is waiting, and why
curl -s localhost:8080/api/sessions/<id>/approvals/<use_id> \
     -H 'Content-Type: application/json' -d '{"allow": true}'       # the turn carries on
```

**4. Keep the conversation going, and read what it says**

```bash
curl -s localhost:8080/api/sessions/<id>/messages \
     -H 'Content-Type: application/json' -d '{"prompt": "and how many are tests?"}'
curl -s 'localhost:8080/api/outbox?after=0&wait=30s'                # replies, questions, approvals
```

A message sent while a turn is running waits for the next turn, and several
that arrive meanwhile are handed over together. If the agent ended a turn by
asking you a question, your next message is the answer to it.

> [!WARNING]
> Without `server.token` the API has no authentication: anything that can
> reach the port can run the agent and read every transcript. wombatd listens
> on loopback by default and, there, answers only to loopback host names; set
> a token before listening anywhere else. See [Safety](#en-safety).

<a id="en-quickstart-lib"></a>

## Quick start: Go library

The module path is `wombat`, so another module uses it through a `replace`
directive pointing at a checkout:

```bash
git clone https://github.com/hoobean1996/wombat-go.git
cd your-module
go mod edit -require=wombat@v0.0.0 -replace=wombat=../wombat-go
```

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"wombat/pkg/core"
	"wombat/pkg/core/governor"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/llm/anthropic"
	"wombat/pkg/core/tool/builtin"
)

func main() {
	// 1. A model client. ConfigFromEnv reads ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
	//    ANTHROPIC_BASE_URL, ... For OpenAI-compatible endpoints use
	//    openai.New(openai.ConfigFromEnv()) instead.
	leaf, err := anthropic.New(anthropic.ConfigFromEnv())
	if err != nil {
		log.Fatal(err)
	}
	// TrackCost charges each response to the run's budget.
	// Without it, a CostUSD limit never trips.
	client := llm.Chain(leaf, core.TrackCost(llm.DefaultPricing))

	// 2. An agent: the client plus the tools it may use.
	agent, err := core.New(
		core.WithClient(client),
		core.WithModel("claude-sonnet-5-5"),
		core.WithTools(builtin.Default(builtin.Deps{})...),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 3. A budget: the run is cancelled at $1 or 10 minutes, whichever comes first.
	ctx, cancel := governor.WithBudget(context.Background(),
		governor.Limits{CostUSD: 1.00, Wall: 10 * time.Minute})
	defer cancel()

	// 4. Start a run and consume its event stream.
	run := agent.Start(ctx, core.Ask("How many Go files are in this directory?"))
	defer run.Close()

	for run.Next() {
		switch ev := run.Event().(type) {
		case core.TextDelta:
			fmt.Print(ev.Text)
		case core.ToolStart:
			fmt.Fprintln(os.Stderr, "\n→", ev.Name)
		}
	}
	if err := run.Err(); err != nil {
		log.Fatal(err)
	}
}
```

```bash
ANTHROPIC_API_KEY=sk-ant-... go run .
```

**How did the run end?** After the loop, `run.Outcome()` tells you:

```go
switch run.Outcome().(type) {
case core.Answer:    // the model finished its turn
case core.Paused:    // it asked the user a question (ask_user); answer it and resume
case core.Submitted: // it called a terminal tool; the tool's arguments are the result
case core.Deferred:  // a call waits for a person's decision; core.Decide resumes it
}
```

**Multi-turn:** pass the previous transcript to the next run.

```go
run = agent.Start(ctx, core.Then(run.Messages(), "What was it times three?"))
```

Use `core.AnswerPause` or `core.Resume` instead when the previous turn ended
in `Paused`, and `core.Decide` when it ended in `Deferred`.

**Add your own tool.** A tool is a name, a JSON schema, its capabilities and a
plain Go function:

```go
type weatherIn struct {
	City string `json:"city"`
}

weather := tool.Typed(tool.Def{
	Name:        "get_weather",
	Description: "Current weather for a city.",
	InputSchema: json.RawMessage(`{
	    "type": "object",
	    "properties": {"city": {"type": "string"}},
	    "required": ["city"]
	}`),
	Caps: tool.CapReadOnly, // tells the scheduler and the permission gate what it does
}, func(ctx context.Context, in weatherIn) (string, error) {
	return "Sunny, 22°C in " + in.City, nil
})

agent, err := core.New(
	core.WithClient(client),
	core.WithTools(append(builtin.Default(builtin.Deps{}), weather)...),
)
```

Always set `Caps`. A tool that declares nothing is treated as one that writes
and reaches the network, so it never runs in parallel and the permission gate
treats it with suspicion.

<a id="en-concepts"></a>

## Core concepts

| Concept | What it is |
|---|---|
| `Agent` | Built once with `core.New(options...)`. Immutable and safe to share across goroutines |
| `Run` | One execution of an agent on an input. Iterate with `Next()` / `Event()`, then check `Err()` and `Outcome()` |
| `Event` | Something a UI can render: `TextDelta`, `ReasoningDelta`, `ToolStart`, `Spend`, `ContextTrimmed`, … |
| `Outcome` | How a successful run ended: `Answer`, `Paused`, `Submitted` or `Deferred`. Failures come from `run.Err()` and are matched with `errors.Is` |
| `tool.Def` | A tool: name, description, JSON schema, capabilities (`Caps`) and a `func(ctx, args) (string, error)` |
| Middleware | Wrappers around the model client (`llm.Chain`) or around tools: retry, logging, cost tracking, permission, timeouts |
| Budget | `governor.WithBudget(ctx, limits)` returns a `context.Context` that is cancelled when a limit is exceeded |
| Strategy | Decides which part of a long transcript is sent to the model (`TokenWindow`, `DropTagged`, …) |
| `daemon.Manager` | Runs sessions: takes their events from the inbox, runs one turn per session at a time, fires timers, delivers the outbox |
| `daemon.Store` | Where the sessions, the inbox, the timers and the outbox are kept: `daemon.MemoryStore()` or `pkg/store/postgres` |

<a id="en-tools"></a>

## Built-in tools

`builtin.Default(deps)` returns these 12 tools:

| Tool | What it does |
|---|---|
| `bash` | Runs a shell command (60 s timeout). Long output keeps the start and the end, and the full output is saved to a file the model can grep |
| `view_file` | Reads a file with line numbers, optionally a line range, or lists a directory |
| `write_file` | Creates or overwrites a file |
| `str_replace` | Edits a file by replacing one exact string |
| `grep_search` | Searches files with an extended regex |
| `git_log` / `git_show` | Lists recent commits / shows one commit with its diff |
| `http_get` | Fetches a URL |
| `calculator` | Evaluates an arithmetic expression |
| `current_time` | Returns the current date and time |
| `save_tool_result` | Saves an earlier tool result to disk without repeating it |
| `ask_user` | Pauses the run to ask the user a question |

Optional extras, each a key in `wombatd.yaml`: `view_media` (lets the model
look at images and PDFs, `tools.media`), `delegate` (a sub-agent,
`tools.delegate`), `load_skill` / `unload_skill` (`tools.skills_dir`), and MCP
tools named `<server>__<tool>` (`tools.mcp`).

wombatd also gives every session's agent the daemon's own tools, which no
permission policy asks about — they touch nothing on the machine:

| Tool | What it does |
|---|---|
| `send_message` | Says something to the people the session works for, through the outbox |
| `set_timer` | Wakes the session later: once (`in` / `at`), or repeatedly (`every`) |
| `cancel_timer` / `list_timers` | Manage the session's timers |

<a id="en-config"></a>

## Configuration

`wombatd` reads one YAML file, `wombatd.yaml` by default (`-config path`
names another). `wombatd.example.yaml` shows every key with its default and
what it is for. Your file names only what it changes, and a key wombatd does
not know is an error naming its line, not a setting that silently did not
take effect.

| Section | What it sets |
|---|---|
| `server` | `addr`, `token` (bearer auth), `cors`, `metrics`, `log` |
| `store` | `driver` (`memory` or `postgres`) and `dsn` |
| `daemon` | `poll`, `max_runs` (turns at once), `attempts` (tries per event), `idle` |
| `outbox` | `webhook` (where each message is POSTed) and its `headers` |
| `model` | `provider`, `name`, `base_url`, `api_key`, `proxy`, `headers`, `stream`, `max_tokens`, `system` |
| `tools` | `working_dir`, `permission`, `confine`, `confine_network`, `mcp`, `skills_dir`, `media`, `delegate` |
| `turn` | `budget_usd`, `wall`, `max_iters`, `tool_parallelism`, `context_window`, `compact_at` |
| `mode` | `build`, `plan` or `review` |
| `done` | `check` (a command that must pass), `attempts`, `accept`, `regressions` |
| `review` | `enabled`, `sweeps`, `dry`, `iters`, `fix` |
| `record` | `tape`, `tape_mode`, `trace`, `trace_html` |

A build that has to pass its tests, an acceptance gate and an adversarial
review before it counts as done, kept in Postgres:

```yaml
store:
  driver: postgres
  dsn: postgres://wombat:wombat@127.0.0.1:55432/wombat
model:
  provider: anthropic
  base_url: ""
  name: claude-sonnet-5-5
  api_key: sk-ant-…
turn:
  budget_usd: 5
done:
  check: go test ./...
  accept: true
review:
  enabled: true
  sweeps: 6
```

**Provider variables, for the library and the benchmarks.** In library code,
`openai.ConfigFromEnv()` and `anthropic.ConfigFromEnv()` read the variables
below into a plain `Config` you can edit before calling `New`; the library
reads nothing else from the environment behind your back, apart from falling
back to the API key (and Anthropic to `ANTHROPIC_BASE_URL`) when the `Config`
leaves it empty. `make bench`, `make judge` and `make models` load them from
`.env` — copy `.env.example`, which documents each one. wombatd reads none of
them but that fallback.

| Variable | Description |
|---|---|
| `WOMBAT_PROVIDER` | `openai` or `anthropic`, for the benchmark suite |
| `WOMBAT_MODEL` | Default model for whichever provider is selected |
| `OPENAI_BASE_URL` | Origin plus `/v1`, for example `https://api.openai.com/v1`. `/chat/completions` is appended |
| `OPENAI_API_KEY` | Sent as `Authorization: Bearer …`. Leave it empty for local servers that need no auth |
| `OPENAI_MODEL`, `OPENAI_PROXY`, `OPENAI_EXTRA_HEADERS`, `OPENAI_STREAM`, `OPENAI_TEMPERATURE`, `OPENAI_TOP_P` | Model, proxy, extra headers (one `Name: value` per line), `auto` / `always` / `never`, sampling |
| `ANTHROPIC_API_KEY` | Required |
| `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_PROXY`, `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_BETA`, `ANTHROPIC_STREAM`, `ANTHROPIC_TEMPERATURE`, `ANTHROPIC_TOP_P` | The same for Anthropic; `ANTHROPIC_BETA` is comma-separated beta flags |

<a id="en-always"></a>

## Always on

`pkg/daemon` keeps sessions going with nobody watching, after Anthropic's
"Building always-on agents". Nothing starts a turn but an **event** in a
session's **inbox**, and what the agent has to say goes to an **outbox**.

- **The inbox.** A person's message, a decision on a call that was waiting
  for one, a timer that fired, anything a client posts to `POST /api/events` —
  a webhook, an alert — is a row, safe once the request returns. An event
  with a `dedup_key` is received once however often it is delivered, and an
  event about a `thread` — an issue, a chat thread — finds the session for
  that thread, or starts one.
- **One turn at a time.** A worker takes a session's pending events, all of
  them, and runs one turn on them; what arrives meanwhile is the next turn's.
  `daemon.max_runs` caps turns across sessions.
- **Trying again.** A turn the daemon stopped, or that the provider failed in a
  way it says is passing (a rate limit, a 5xx, the network), runs again — from
  the transcript it started from, so nothing is in the conversation twice — up
  to `daemon.attempts`. A daemon that died mid-turn has its events put back
  when the next one starts.
- **Timers.** The agent sets its own with `set_timer` — once, to follow up, or
  every so often — and a client with `POST /api/timers`. A timer that is due
  puts an event in the inbox; a daemon that was down fires each one once when
  it comes back, not once for every period it missed.
- **Decisions.** A call the policy wants a person to approve ends the turn
  instead of holding it open: the session is saved mid-call, takes nothing but
  decisions, and resumes once they are all in — the approved calls run then,
  and the model reads their real output.
- **The outbox.** The reply to a message, a question the model asks, the calls
  waiting for a decision, and what the agent chooses to say with
  `send_message` — a timer's turn says nothing unless it does. A client
  long-polls `GET /api/outbox?after=<last id>&wait=30s` and acks with
  `POST /api/outbox/{id}/ack`; with `outbox.webhook` set, wombatd POSTs each
  message itself, in order, trying one up to 10 times, with a growing pause,
  before it moves on to the ones behind it.

An event from outside, about a thread, and a reminder for tomorrow morning:

```bash
curl -s localhost:8080/api/events -H 'Content-Type: application/json' -d '{
  "thread": "github:you/repo#12", "kind": "issue_comment", "source": "github",
  "dedup_key": "delivery-8f2c", "payload": {"body": "the build is red again"}}'
curl -s localhost:8080/api/timers -H 'Content-Type: application/json' -d '{
  "session_id": "<id>", "note": "check whether the fix went out", "at": "2026-11-16T09:00:00Z"}'
```

**Keeping it.** With the default `store.driver: memory`, everything is gone
when wombatd stops. With `postgres`, it is kept in four tables — `sessions`,
`inbox`, `timers` and `outbox` — which wombatd creates on its first start, and
a restart carries on where it stopped. One wombatd to a schema: a second one
on the same schema is refused at the start (add `?search_path=name` to the DSN
to give each its own). For a local database:

```bash
make pg-up        # Postgres 17 in podman, on 127.0.0.1:55432
```

```yaml
store:
  driver: postgres
  dsn: postgres://wombat:wombat@127.0.0.1:55432/wombat
```

<a id="en-http"></a>

## HTTP API

`wombatd` serves the `httpapi` package, a plain `http.Handler` over a
`daemon.Manager`:

| Method and path | Purpose |
|---|---|
| `POST /api/sessions` | Create a session with its first message: `{"prompt": "..."}`, and optionally `title`, `permission`, `workspace`, `model`, `max_iters` |
| `GET /api/sessions` / `GET /api/sessions/{id}` | List sessions, newest first / one session |
| `DELETE /api/sessions/{id}` | Cancel and forget a session, its timers and messages included |
| `GET /api/sessions/{id}/events` | Server-Sent Events across every turn. Reconnect with `Last-Event-ID` to resume |
| `GET /api/sessions/{id}/messages` | The transcript |
| `POST /api/sessions/{id}/messages` | A message, for the next turn: `{"prompt": "..."}` |
| `POST /api/sessions/{id}/steer` | A message for the turn that is still running |
| `GET /api/sessions/{id}/approvals` | Calls waiting for a decision |
| `POST /api/sessions/{id}/approvals/{use_id}` | Decide one: `{"allow": true}` |
| `POST /api/events` | An event from outside: `{"session_id" or "thread", "kind", "source", "dedup_key", "payload"}` |
| `GET /api/inbox` | Events, newest first: `?session=`, `?status=`, `?limit=` |
| `GET /api/outbox` | Messages, oldest first: `?after=`, `?wait=` (at most 60s), `?session=`, `?status=`, `?limit=` |
| `POST /api/outbox/{id}/ack` | Mark a message handled |
| `GET /api/timers` / `POST /api/timers` / `DELETE /api/timers/{id}` | List (`?session=`, `?status=`), set (`{"session_id", "note", "in" or "at", "every"}`) and cancel timers |
| `GET /api/health`, `GET /api/config`, `GET /metrics` | Liveness, what this server offers, Prometheus |

A message, a decision and an event are answered with `202 Accepted`, because
they are queued: what they lead to arrives on the event stream and in the
outbox. A new session is `201 Created`, and an event whose `dedup_key` was
seen before is `200`, with the event received the first time. Every error has
the same shape, `{"error": {"kind": "...", "message": "..."}}`, with a bounded
`kind` a client can branch on. The package documentation has the details.

<a id="en-cli"></a>

## Commands and make targets

There is one command, `wombatd -config wombatd.yaml`. `make bin` builds it
into `bin/`.

| Target | What it does |
|---|---|
| `make` | gofmt check, `go vet`, `go build` (the default target) |
| `make test` | `go test -race ./...` |
| `make bin` | Builds `bin/wombatd` |
| `make serve` | Builds and runs wombatd with `wombatd.yaml`; `CONFIG=other.yaml` picks another file |
| `make pg-up` / `make pg-down` | Starts / removes a Postgres in podman, on port 55432 |
| `make test-pg` | The Postgres store's tests against it, a daemon restart among them |
| `make bench` | Runs the benchmarks, for example `make bench N=8 C=4 TASKS=todo-app,fix-bug` |
| `make judge` | Scores a benchmark run's episodes: `make judge IN=runs/episodes.jsonl` |
| `make models` | Lists the model IDs your OpenAI-compatible endpoint offers |
| `make help` | Lists the documented targets |

<a id="en-guide"></a>

## Feature guide

Each section below is a short summary. [docs/DESIGN.md](docs/DESIGN.md)
explains why each mechanism works the way it does.

<a id="en-safety"></a>

### Safety

Three independent layers:

| Layer | What it does | How to enable |
|---|---|---|
| **Permission policy** | Allows, denies or asks a human about each tool call, looking at the actual command, path and arguments | `tools.permission` (wombatd), `permission.Gate` (library) |
| **Environment scrubbing** | Commands the agent runs inherit only an allowlist of environment variables, so your API keys are not visible to them | On by default |
| **OS sandbox** | Commands run under `sandbox-exec` (macOS) or `bwrap` (Linux): writes only inside the workspace and the temp directory, credential directories unreadable, network loopback-only by default | `tools.confine: true` |

Permission levels in wombatd, from the loosest to the strictest.
`tools.permission` sets one for every session; a session may ask for a
stricter one when it is created (`"permission": "readonly"`), never a looser
one:

| Level | Behavior |
|---|---|
| `off` | No gate at all: every call runs. Only the operator can choose it |
| `workspace` | Reads and writes inside the working directory are allowed, and so are common harmless commands (`go test`, `ls`, `cat`, …). Anything else — another command, the network, a path outside — waits for a decision (the default) |
| `ask` | Every call waits for a decision |
| `readonly` | Only read-only tools run; everything else is refused |

The daemon's own tools (`send_message`, the timer tools) run under every level.

A call that waits for a decision does not hold the turn open: the turn ends
on it, the outbox says what is waiting, and the decision comes back over the
API, however long it takes. In library code, an approver either answers at
once or defers:

```go
approver := permission.ApproverFunc(func(ctx context.Context, r permission.Request) (bool, error) {
	fmt.Printf("allow %s? (%s) [y/N] ", r.Tool.Name, r.Reason)
	var answer string
	fmt.Scanln(&answer)
	return answer == "y", nil
})

agent, err := core.New(
	core.WithClient(client),
	core.WithTools(builtin.Default(builtin.Deps{})...),
	core.WithToolMiddleware(permission.Gate(permission.Workspace("."), approver)),
)
```

With `permission.Defer()` as the approver, the run ends in `core.Deferred`
instead, and `core.Decide(run.Messages(), deferred, verdicts)` resumes it
once someone has decided. A `nil` approver turns every "ask" into "deny".

### Budgets

```go
ctx, cancel := governor.WithBudget(ctx, governor.Limits{
	CostUSD:       2.00,
	Wall:          15 * time.Minute,
	Steps:         50,
	ToolCalls:     200,
	SubagentDepth: 2,
})
defer cancel()
```

Going over any limit cancels the context, and everything in flight (HTTP
requests, child processes) stops. `context.Cause(ctx)` tells you which limit
tripped. Cost is computed from `llm.DefaultPricing`. A model that is not in that
table costs `0`, so for such models rely on `Wall` and `Steps`, or pass your
own rates with `core.WithPricing`. In wombatd the budget is per turn:
`turn.budget_usd`, `turn.wall`, `turn.max_iters`.

### Long conversations

```go
core.WithContextWindow(200_000),
core.WithStrategy(core.TokenWindow(0.7, 0.5)),                  // trim at 70%, back to ~50%
core.WithCompaction(160_000, 50_000, core.Summarize(client, "")), // summarize old turns
```

Old tool results are shortened first, and only then are whole old exchanges
dropped. Compaction replaces the oldest part of the transcript with a summary.
wombatd: `turn.context_window: 200000` and `turn.compact_at: 0.85`.

### Checking that the work is done, and steering

```go
core.WithCompletionCheck(3, core.CommandCheck(runner, "go test ./...", 0))
```

When the model says it is finished, the check runs. If it fails, the model gets
the output and keeps working, up to 3 times. wombatd: `done.check: go test ./...`.

To redirect a run while it is going, call `run.Steer("use the v2 API")`. Over
HTTP, use `POST /api/sessions/{id}/steer`.

### Skills

A skill is a folder with a `SKILL.md` file. The front matter gives it a name and
a description; the body is instructions the model loads only when it needs them.

```markdown
---
name: git-history
description: Reading a repository's history. Load before any question about the past.
---

# Reading git history
...
```

In wombatd, `tools.skills_dir: skills`. The model sees a catalogue of skills
and calls `load_skill` to load one. A skill can also unlock tools that stay
hidden until it is loaded:

```go
skills, _ := skill.LoadDir("./skills", func(err error) { log.Print(err) })
reg := skill.New(skills...)
reg.Gate("git-history", "git_log") // git_log appears only after git-history is loaded
g := reg.Bind(builtin.Default(deps))

agent, _ := core.New(core.WithClient(client), g.Install())
```

This repository ships four example skills in `skills/`.

### Sub-agents

A `delegate` tool runs a whole child agent. The child shares the parent's
budget, permission gate and tracing, because they all travel on the `context`.

```go
child, _ := core.New(core.WithClient(client), core.WithName("researcher"),
	core.WithTools(tool.Filter(all, tool.OnlyCaps(tool.CapReadOnly))...))

parent, _ := core.New(core.WithClient(client),
	core.WithTools(append(all, core.DelegateTool(child))...),
	core.WithToolTimeoutFallback(0), // the child is bounded by its own iteration cap
)
```

wombatd: `tools.delegate: true` adds a read-only researcher sub-agent.

### MCP (Model Context Protocol)

Connect to MCP servers over stdio or Streamable HTTP. The config file uses the
same format as desktop MCP clients:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}
    },
    "docs": {"type": "http", "url": "https://example.com/mcp"}
  }
}
```

In wombatd, `tools.mcp: mcp.json`; the servers are started once and shared by
every session. Tools appear as `<server>__<tool>`. Their MCP hints
(`readOnlyHint`, …) become capabilities, so permission policies apply to them
exactly as to built-in tools. In code, use `mcp.LoadConfig`, `mcp.ConnectAll`
and `conns.Tools()`.

### Record and replay

```go
tp, _ := tape.Open("run.jsonl", tape.Auto)
defer tp.Close()
agent, _ := core.New(core.WithClient(client), tp.Install())
```

Model responses and tool results are recorded, keyed by a hash of the request.
A replay is fast, deterministic and makes no network calls. wombatd:
`record.tape: run.jsonl` and `record.tape_mode: replay`.

### Tracing and metrics

```yaml
record:
  trace: run.ndjson         # every session's spans
  trace_html: run.html      # rendered as a self-contained report on shutdown
```

Traces use OpenTelemetry `gen_ai.*` attribute names, nest sub-agents
automatically, and never record message content. wombatd exposes Prometheus
metrics on `GET /metrics` (`server.metrics`, on by default).

### Benchmarks and evaluation

```bash
make bench TASKS=easy N=4                 # the suite, as a test that runs only when asked
make judge IN=runs/episodes.jsonl         # score the episodes afterwards
```

`eval/benchmarks/` holds eight coding tasks in two tiers. The easy tier should
always score 100%; if it does not, the harness is broken. The hard tier is the
one that tells agents apart. Results are reported as pass@k. The rest of
`eval/` measures whether the harness mechanisms themselves work; see
[eval/README.md](eval/README.md).

<a id="en-layout"></a>

## Project layout

```
pkg/core/             Agent, Run, Event, Outcome, Convo, Strategy   ← start here
pkg/core/llm/         provider-neutral types, Client interface, middleware, pricing
pkg/core/llm/anthropic/   Anthropic Messages API client
pkg/core/llm/openai/      OpenAI-compatible client
pkg/core/tool/        tool definitions, dispatch, middleware (retry, timeout, breaker, …)
pkg/core/tool/builtin/    the 12 built-in tools, and view_media
pkg/core/sandbox/     the runners every command goes through: filtered, and confined
pkg/core/permission/  per-call authorization, with a human in the loop
pkg/core/governor/    budgets as context cancellation
pkg/core/skill/       skills: instructions and tools that load on demand
pkg/core/mcp/         Model Context Protocol client
pkg/core/tape/        record and replay
pkg/core/trace/       spans, NDJSON output, HTML report
pkg/core/metric/      Prometheus metrics
pkg/core/accept/      deterministic "is it really done?" gates (build, vet, test, TODO.md)
pkg/core/review/      adversarial code review that reports defects as failing tests
pkg/daemon/           always on: sessions, the inbox, timers, deferred decisions, the outbox
pkg/daemon/storetest/ what a store has to do, as tests both stores pass
pkg/store/postgres/   the sessions and the three tables in Postgres
pkg/httpapi/          the daemon over HTTP: events in, SSE and the outbox out
cmd/wombatd/          the service; everything configured by wombatd.yaml
eval/                 evaluation of the harness itself
eval/rl/              episodes, verifiers, pass@k
eval/benchmarks/      eight coding tasks, and the suite that runs them
internal/             fspath, safeenv, shellquote, sse, yaml, and the wiring checks
skills/               example skills
docs/                 design notes
```

<a id="en-dev"></a>

## Development

```bash
make              # gofmt check + go vet + go build
make test         # go test -race ./...
make pg-up && make test-pg   # the Postgres store, against a real database
```

The test suite runs offline against scripted model clients and needs no API
key; the Postgres tests skip unless they are given a database. The only
module it downloads is `pgx`, for `pkg/store/postgres`.

<a id="en-faq"></a>

## Troubleshooting

| Symptom | Fix |
|---|---|
| `wombatd.yaml does not exist` | Run `cp wombatd.example.yaml wombatd.yaml` and fill in the model section, or pass `-config path` |
| `unknown key "…" in …` | A misspelt key, with its line number. `wombatd.example.yaml` shows every key there is |
| `no API key` error | Set `model.api_key`, and check that `model.provider` matches the key |
| `no model on the request and no client default` | Set `model.name` |
| Not sure which model IDs your endpoint accepts | `make models` (reads `.env`) |
| Token counts and cost stay at 0, so the budget never trips | Some gateways drop usage data when streaming. Set `model.stream: never`. Also check that the model is priced in `pkg/core/llm/pricing.go` |
| HTTP 400 with `tools.media: true` | The model does not accept images. Turn `tools.media` off |
| A session sits in `approving` | It is waiting for a decision: `GET /api/sessions/{id}/approvals`, then POST one. It takes nothing else until then |
| `another daemon is running on this schema` | Another wombatd uses that Postgres schema. Stop it, or give this one its own with `?search_path=name` in `store.dsn` |
| Messages pile up in the inbox | Look at `GET /api/inbox?session=…` and the log: a session waiting on a decision runs nothing else, and `daemon.max_runs` caps turns at once |

<a id="en-design"></a>

## Design notes

wombat-go is a Go rewrite of a private OCaml 5 project built on **algebraic
effects**, where every side effect (LLM calls, tools, file I/O, time, human
input) is an effect, and handlers decide what actually happens. In Go, most of
those effects turned out to be plain **dependency injection**. Three decisions
shape the API:

1. **A tool takes a context and its arguments, nothing else.** Whatever it
   needs (a filesystem, a runner, an HTTP client) is passed in when it is
   constructed, so a tool is an ordinary function you can test directly.
2. **A run is a stream of events**, not a return value plus callbacks. A
   terminal, an SSE stream and a test all consume the same stream.
3. **A budget is a `context.Context`.** Going over a limit cancels it, and every
   blocking call in the process already knows how to unwind from that.

The full write-up covers context structure, skills, fault tolerance, tracing,
permissions, the always-on daemon and evaluation, and what was gained and lost
compared with the OCaml original: **[docs/DESIGN.md](docs/DESIGN.md)**.

<a id="en-license"></a>

## License

wombat-go is released under the [MIT License](LICENSE).
