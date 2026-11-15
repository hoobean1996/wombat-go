<div align="center">

# wombat-go

**A ReAct agent harness for Go, with zero third-party dependencies.**

**English** · [简体中文](README.zh-CN.md)

![Go](https://img.shields.io/badge/Go-1.25%2B-00ADD8?logo=go&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

</div>

---

- [What is wombat-go?](#en-what)
- [Features](#en-features)
- [Requirements](#en-requirements)
- [Quick start: command line](#en-quickstart-cli)
- [Quick start: Go library](#en-quickstart-lib)
- [Core concepts](#en-concepts)
- [Built-in tools](#en-tools)
- [Configuration](#en-config)
- [Command-line tools](#en-cli)
- [Feature guide](#en-guide)
- [Project layout](#en-layout)
- [Development](#en-dev)
- [Troubleshooting](#en-faq)
- [Design notes](#en-design)

<a id="en-what"></a>

## What is wombat-go?

wombat-go is a Go library, plus a few command-line tools, for building **LLM
agents**. It runs the classic **ReAct** loop: ask the model, run the tools it
asks for, feed the results back, and repeat until the model is done.

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

It is written for people who want to see and control every part of the loop.
Everything is plain Go using only the standard library, and every dependency is
injected, so the model, a tool or the filesystem can each be swapped for a fake
in a test.

<a id="en-features"></a>

## Features

| | |
|---|---|
| **Two providers, one interface** | Anthropic Messages API and any OpenAI-compatible `/chat/completions` endpoint, including gateways that need proxies and custom headers |
| **Streaming events** | A run is an iterator of typed events (`TextDelta`, `ReasoningDelta`, `ToolStart`, …) that you can render in a terminal, as JSONL, or in a web UI |
| **12 built-in tools** | `bash`, `view_file`, `write_file`, `str_replace`, `grep_search`, `git_log`, `http_get`, …, plus MCP tools and your own |
| **Budgets** | Cap cost in USD, wall-clock time, steps, tool calls and sub-agent depth. Going over a limit cancels the run cleanly |
| **Safety** | Per-call permission policies with human approval, a scrubbed environment for child processes, and an optional OS sandbox (`sandbox-exec` / `bwrap`) |
| **Long tasks** | Token-aware context trimming, automatic summarization (compaction), and recovery from context-overflow errors |
| **Skills and sub-agents** | Tools that appear only after a skill is loaded, and child agents that share the parent's budget |
| **Observability** | OpenTelemetry-shaped traces with an HTML viewer, Prometheus metrics, and record/replay tapes |
| **HTTP server** | Resumable multi-turn sessions over Server-Sent Events, with a bundled web UI |
| **Evaluation** | A benchmark suite of coding tasks with pass@k scoring and anti-cheat checks |

<a id="en-requirements"></a>

## Requirements

| | |
|---|---|
| **Go** | 1.25.6 or newer |
| **An LLM API** | An Anthropic API key, **or** any OpenAI-compatible endpoint. A local Ollama or vLLM server needs no key |
| **Python 3** | Only for `scripts/wombat`, the terminal renderer |
| *Optional* | `sandbox-exec` (built into macOS) or `bwrap` (Linux), for `-confine` |

<a id="en-quickstart-cli"></a>

## Quick start: command line

**1. Clone and configure**

```bash
git clone https://github.com/hoobean1996/wombat-go.git
cd wombat-go
cp .env.example .env
```

Open `.env` and fill in **one** provider. `.env` is git-ignored, so your key
stays on your machine.

```bash
# Option A: OpenAI, or any OpenAI-compatible API (DeepSeek, a gateway, ...)
export WOMBAT_PROVIDER="openai"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
export WOMBAT_MODEL="gpt-4o-mini"
```

```bash
# Option B: Anthropic
export WOMBAT_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
export WOMBAT_MODEL="claude-sonnet-5-5"
```

```bash
# Option C: a local model with Ollama (no key needed)
export WOMBAT_PROVIDER="openai"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY=""            # empty means no auth header is sent
export WOMBAT_MODEL="qwen2.5-coder"
```

**2. Ask the agent something**

```bash
scripts/wombat "how many Go files are in this repo? use bash"
```

`scripts/wombat` builds the binaries into `bin/`, loads `.env`, runs one agent
turn and renders it as readable text. You see the answer stream in, and each
tool call as it happens. Add `-r` to also see the model's reasoning.

> [!WARNING]
> By default the agent has an **unrestricted shell in the current directory**.
> For anything beyond a quick try, add `-permission workspace`, `-confine`, or
> both. See [Safety](#en-safety).

**3. Keep the conversation going**

```bash
scripts/wombat -session /tmp/s.json "remember the number 8317"
scripts/wombat -session /tmp/s.json "what was it times three?"
```

The transcript is saved to the session file after every turn. If the agent
ended a turn by asking you a question, your next message is treated as the
answer to it.

**4. Or chat in the browser**

```bash
make serve        # then open http://localhost:8080
```

This starts a web UI with live streaming. By default, shell commands and
anything outside the working directory wait for you on an **Allow / Deny** card.

<a id="en-quickstart-lib"></a>

## Quick start: Go library

```bash
go get github.com/hoobean1996/wombat-go
```

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	wombat "github.com/hoobean1996/wombat-go"
	"github.com/hoobean1996/wombat-go/governor"
	"github.com/hoobean1996/wombat-go/llm"
	"github.com/hoobean1996/wombat-go/llm/anthropic"
	"github.com/hoobean1996/wombat-go/tool/builtin"
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
	client := llm.Chain(leaf, wombat.TrackCost(llm.DefaultPricing))

	// 2. An agent: the client plus the tools it may use.
	agent, err := wombat.New(
		wombat.WithClient(client),
		wombat.WithModel("claude-sonnet-5-5"),
		wombat.WithTools(builtin.Default(builtin.Deps{})...),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 3. A budget: the run is cancelled at $1 or 10 minutes, whichever comes first.
	ctx, cancel := governor.WithBudget(context.Background(),
		governor.Limits{CostUSD: 1.00, Wall: 10 * time.Minute})
	defer cancel()

	// 4. Start a run and consume its event stream.
	run := agent.Start(ctx, wombat.Ask("How many Go files are in this directory?"))
	defer run.Close()

	for run.Next() {
		switch ev := run.Event().(type) {
		case wombat.TextDelta:
			fmt.Print(ev.Text)
		case wombat.ToolStart:
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
case wombat.Answer:    // the model finished its turn
case wombat.Paused:    // it asked the user a question (ask_user); answer it and resume
case wombat.Submitted: // it called a terminal tool; the tool's arguments are the result
}
```

**Multi-turn:** pass the previous transcript to the next run.

```go
run = agent.Start(ctx, wombat.Then(run.Messages(), "What was it times three?"))
```

Use `wombat.AnswerPause` or `wombat.Resume` instead when the previous turn ended
in `Paused`.

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

agent, err := wombat.New(
	wombat.WithClient(client),
	wombat.WithTools(append(builtin.Default(builtin.Deps{}), weather)...),
)
```

Always set `Caps`. A tool that declares nothing is treated as one that writes
and reaches the network, so it never runs in parallel and the permission gate
treats it with suspicion.

<a id="en-concepts"></a>

## Core concepts

| Concept | What it is |
|---|---|
| `Agent` | Built once with `wombat.New(options...)`. Immutable and safe to share across goroutines |
| `Run` | One execution of an agent on an input. Iterate with `Next()` / `Event()`, then check `Err()` and `Outcome()` |
| `Event` | Something a UI can render: `TextDelta`, `ReasoningDelta`, `ToolStart`, `Spend`, `ContextTrimmed`, … |
| `Outcome` | How a successful run ended: `Answer`, `Paused` or `Submitted`. Failures come from `run.Err()` and are matched with `errors.Is` |
| `tool.Def` | A tool: name, description, JSON schema, capabilities (`Caps`) and a `func(ctx, args) (string, error)` |
| Middleware | Wrappers around the model client (`llm.Chain`) or around tools: retry, logging, cost tracking, permission, timeouts |
| Budget | `governor.WithBudget(ctx, limits)` returns a `context.Context` that is cancelled when a limit is exceeded |
| Strategy | Decides which part of a long transcript is sent to the model (`TokenWindow`, `DropTagged`, …) |

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

Optional extras: `view_media` (lets the model look at images and PDFs, CLI flag
`-media`), `delegate` (a sub-agent, `-delegate`), `load_skill` / `unload_skill`
(`-skills-dir`), and MCP tools named `<server>__<tool>` (`-mcp`).

<a id="en-config"></a>

## Configuration

The CLIs read their settings from environment variables, which `scripts/wombat`
and `make` load from `.env`. In library code, `openai.ConfigFromEnv()` and
`anthropic.ConfigFromEnv()` read the same variables and return a plain `Config`
struct you can edit before calling `New`. The library reads nothing else from
the environment behind your back, apart from falling back to the API key.

| Variable | Description |
|---|---|
| `WOMBAT_PROVIDER` | `openai` or `anthropic`. Set it explicitly: the CLIs have different defaults |
| `WOMBAT_MODEL` | Default model for whichever provider is selected |
| **OpenAI-compatible** | |
| `OPENAI_BASE_URL` | Origin plus `/v1`, for example `https://api.openai.com/v1`. `/chat/completions` is appended |
| `OPENAI_API_KEY` | Sent as `Authorization: Bearer …`. Leave it empty for local servers that need no auth |
| `OPENAI_MODEL` | Takes priority over `WOMBAT_MODEL` |
| `OPENAI_PROXY` | HTTP proxy, as `host:port` or a full URL |
| `OPENAI_EXTRA_HEADERS` | Extra headers, one `Name: value` per line |
| `OPENAI_STREAM` | `auto` (default), `always` or `never` |
| `OPENAI_TEMPERATURE`, `OPENAI_TOP_P` | Sampling parameters |
| **Anthropic** | |
| `ANTHROPIC_API_KEY` | Required |
| `ANTHROPIC_MODEL` | Takes priority over `WOMBAT_MODEL`. There is no built-in default model |
| `ANTHROPIC_BASE_URL` | Default `https://api.anthropic.com` (falls back to `AGENT_LLM_BASE_URL`) |
| `ANTHROPIC_PROXY` | HTTP proxy (falls back to `AGENT_LLM_PROXY`) |
| `ANTHROPIC_CUSTOM_HEADERS` | Extra headers, one `Name: value` per line |
| `ANTHROPIC_BETA` | Comma-separated beta flags |
| `ANTHROPIC_STREAM` | `auto` (default), `always` or `never` |
| `ANTHROPIC_TEMPERATURE`, `ANTHROPIC_TOP_P` | Sampling parameters |
| **Scripts** | |
| `WOMBAT_ENV` | Env file that `scripts/wombat` loads (default `./.env`). For `make`, pass `ENV=path` |

`.env.example` documents each of these in more detail.

<a id="en-cli"></a>

## Command-line tools

`make bin` builds all of them into `bin/`.

| Binary | Purpose |
|---|---|
| `wombat-jsonl` | Runs one agent turn and writes a JSON-Lines event stream to stdout. `scripts/wombat` wraps it with a renderer |
| `wombat-serve` | HTTP + SSE server with a built-in web UI |
| `wombat-bench` | Runs the benchmark suite and reports pass@k |
| `wombat-judge` | Scores the episodes that `wombat-bench` produced |
| `wombat-trace-html` | Turns a trace file into a self-contained HTML report |
| `wombat-tsgen` | Generates TypeScript types for the event stream and the HTTP API |

**Common `wombat-jsonl` / `scripts/wombat` flags**

| Area | Flags |
|---|---|
| Model | `-provider openai\|anthropic`, `-model ID`, `-system "prompt"`, `-max-tokens N` |
| Workspace | `-working-dir DIR` (where shell and git tools run), `-read-only` (hide tools that write or execute) |
| Safety | `-permission off\|readonly\|workspace\|ask`, `-confine`, `-confine-network loopback\|none\|full` |
| Limits | `-budget USD`, `-wall 10m`, `-max-iters N` (default 30) |
| Conversation | `-session FILE` (multi-turn), `-resume FILE` (continue a transcript, read-only) |
| Long tasks | `-context-window TOKENS`, `-compact-at 0.85`, `-check 'go test ./...'`, `-check-attempts N` |
| Extras | `-skills-dir skills`, `-delegate`, `-mcp mcp.json`, `-media` |
| Recording | `-tape FILE`, `-tape-mode auto\|record\|replay`, `-trace FILE`, `-trace-html FILE`, `-metrics-file FILE` |
| Logging | `-log debug\|info\|warn\|error`, `-v`, and `-r` (show reasoning, `scripts/wombat` only) |

Run `bin/wombat-jsonl -h` for the full list, including the build, review and
acceptance workflow flags (`-plan-only`, `-accept`, `-review`, …).

**Common `wombat-serve` flags:** `-addr :8080`, `-working-dir DIR`,
`-permission workspace` (default), `-budget 1.0` (USD per turn),
`-max-iters 12`, `-sessions-dir DIR` (keep sessions across restarts), `-cors`,
`-confine`, `-mcp`.

**Make targets**

| Target | What it does |
|---|---|
| `make` | gofmt check, `go vet`, `go build` (the default target) |
| `make test` | `go test -race ./...` |
| `make bin` | Builds every command into `bin/` |
| `make serve` | Starts `wombat-serve` with `.env` loaded. Pass extra flags with `FLAGS="..."` |
| `make bench` | Runs the benchmarks, for example `make bench N=8 C=4 TASKS=todo-app,fix-bug` |
| `make models` | Lists the model IDs your OpenAI-compatible endpoint offers |
| `make ts` | Regenerates `web/events.ts` |
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
| **Permission policy** | Allows, denies or asks a human about each tool call, looking at the actual command, path and arguments | `-permission …` (CLI), `permission.Gate` (library) |
| **Environment scrubbing** | Commands the agent runs inherit only an allowlist of environment variables, so your API keys are not visible to them | On by default |
| **OS sandbox** | Commands run under `sandbox-exec` (macOS) or `bwrap` (Linux): writes only inside the workspace and the temp directory, credential directories unreadable, network loopback-only by default | `-confine` |

Permission levels:

| Level | Behavior |
|---|---|
| `off` | No checks (the `wombat-jsonl` default) |
| `readonly` | Only read-only tools run |
| `workspace` | Reads and writes inside the working directory are allowed, and so are common harmless commands (`go test`, `ls`, `cat`, …). Anything else needs approval |
| `ask` | Every call needs approval |

`wombat-jsonl` has no way to ask you, so "needs approval" means "denied", and
the model is told why. Use `wombat-serve` if you want to approve calls in the
browser. In library code:

```go
approver := permission.ApproverFunc(func(ctx context.Context, r permission.Request) (bool, error) {
	fmt.Printf("allow %s? (%s) [y/N] ", r.Tool.Name, r.Reason)
	var answer string
	fmt.Scanln(&answer)
	return answer == "y", nil
})

agent, err := wombat.New(
	wombat.WithClient(client),
	wombat.WithTools(builtin.Default(builtin.Deps{})...),
	wombat.WithToolMiddleware(permission.Gate(permission.Workspace("."), approver)),
)
```

A `nil` approver turns every "ask" into "deny".

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
own rates with `wombat.WithPricing`. CLI: `-budget`, `-wall`, `-max-iters`.

### Long conversations

```go
wombat.WithContextWindow(200_000),
wombat.WithStrategy(wombat.TokenWindow(0.7, 0.5)),                  // trim at 70%, back to ~50%
wombat.WithCompaction(160_000, 50_000, wombat.Summarize(client, "")), // summarize old turns
```

Old tool results are shortened first, and only then are whole old exchanges
dropped. Compaction replaces the oldest part of the transcript with a summary.
CLI: `-context-window 200000 -compact-at 0.85`.

### Checking that the work is done, and steering

```go
wombat.WithCompletionCheck(3, wombat.CommandCheck(runner, "go test ./...", 0))
```

When the model says it is finished, the check runs. If it fails, the model gets
the output and keeps working, up to 3 times. CLI: `-check 'go test ./...'`.

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

```bash
scripts/wombat -skills-dir skills "who last changed run.go, and why?"
```

The model sees a catalogue of skills and calls `load_skill` to load one. A skill
can also unlock tools that stay hidden until it is loaded:

```go
skills, _ := skill.LoadDir("./skills", func(err error) { log.Print(err) })
reg := skill.New(skills...)
reg.Gate("git-history", "git_log") // git_log appears only after git-history is loaded
g := reg.Bind(builtin.Default(deps))

agent, _ := wombat.New(wombat.WithClient(client), g.Install())
```

This repository ships four example skills in `skills/`.

### Sub-agents

A `delegate` tool runs a whole child agent. The child shares the parent's
budget, permission gate and tracing, because they all travel on the `context`.

```go
child, _ := wombat.New(wombat.WithClient(client), wombat.WithName("researcher"),
	wombat.WithTools(tool.Filter(all, tool.OnlyCaps(tool.CapReadOnly))...))

parent, _ := wombat.New(wombat.WithClient(client),
	wombat.WithTools(append(all, wombat.DelegateTool(child))...),
	wombat.WithToolTimeoutFallback(0), // the child is bounded by its own iteration cap
)
```

CLI: `-delegate` adds a read-only researcher sub-agent.

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

```bash
scripts/wombat -mcp mcp.json "list my open pull requests"
```

Tools appear as `<server>__<tool>`. Their MCP hints (`readOnlyHint`, …) become
capabilities, so permission policies apply to them exactly as to built-in tools.
In code, use `mcp.LoadConfig`, `mcp.ConnectAll` and `conns.Tools()`.

### Record and replay

```go
tp, _ := tape.Open("run.jsonl", tape.Auto)
defer tp.Close()
agent, _ := wombat.New(wombat.WithClient(client), tp.Install())
```

Model responses and tool results are recorded, keyed by a hash of the request.
A replay is fast, deterministic and makes no network calls. CLI:
`-tape run.jsonl -tape-mode replay`.

### Tracing and metrics

```bash
scripts/wombat -trace run.ndjson -trace-html run.html "..."
bin/wombat-trace-html run.ndjson        # or render an existing trace later
```

Traces use OpenTelemetry `gen_ai.*` attribute names, nest sub-agents
automatically, and never record message content. `wombat-serve` exposes
Prometheus metrics on `GET /metrics`, and `wombat-jsonl -metrics-file` writes a
one-off dump.

### HTTP API

`wombat-serve` (or the `httpapi` package, a plain `http.Handler`) serves
multi-turn sessions:

| Method and path | Purpose |
|---|---|
| `POST /api/sessions` | Create a session and start its first turn: `{"prompt": "..."}` |
| `GET /api/sessions/{id}/events` | Server-Sent Events stream. Reconnect with `Last-Event-ID` to resume |
| `POST /api/sessions/{id}/messages` | Send the next turn: `{"prompt": "..."}` |
| `POST /api/sessions/{id}/steer` | Send a message to the turn that is still running |
| `GET /api/sessions/{id}/approvals` | List calls waiting for a human |
| `POST /api/sessions/{id}/approvals/{use_id}` | Answer one: `{"allow": true}` |
| `DELETE /api/sessions/{id}` | Cancel and delete a session |

```bash
curl -s localhost:8080/api/sessions -H 'Content-Type: application/json' \
     -d '{"prompt": "what does this repo do?"}'      # returns {"id": "...", ...}
curl -N localhost:8080/api/sessions/<id>/events
```

The full list (health, config, listing sessions, the transcript) is in the
`httpapi` package documentation. TypeScript types for every event and request
are in `web/events.ts`, generated from the Go types.

### Benchmarks and evaluation

```bash
make bench TASKS=easy N=4                 # or: bin/wombat-bench -tasks hard -n 4
bin/wombat-judge -in runs/episodes.jsonl  # score the episodes afterwards
```

`benchmarks/` holds eight coding tasks in two tiers. The easy tier should always
score 100%; if it does not, the harness is broken. The hard tier is the one that
tells agents apart. Results are reported as pass@k. `eval/` measures whether the
harness mechanisms themselves work; see [eval/README.md](eval/README.md).

<a id="en-layout"></a>

## Project layout

```
.                 Agent, Run, Event, Outcome, Convo, Strategy   ← start here
llm/              provider-neutral types, Client interface, middleware, pricing
llm/anthropic/    Anthropic Messages API client
llm/openai/       OpenAI-compatible client
tool/             tool definitions, dispatch, middleware (retry, timeout, breaker, …)
tool/builtin/     the 12 built-in tools, view_media, plain and sandboxed runners
permission/       per-call authorization, with a human in the loop
governor/         budgets as context cancellation
skill/            skills: instructions and tools that load on demand
skills/           example skills
mcp/              Model Context Protocol client
tape/             record and replay
trace/            spans, NDJSON output, HTML report
metric/           Prometheus metrics
httpapi/          multi-turn sessions over SSE
accept/           deterministic "is it really done?" gates (build, vet, test, TODO.md)
review/           adversarial code review that reports defects as failing tests
rl/               episodes, verifiers, pass@k
benchmarks/       eight coding tasks
eval/             evaluation of the harness itself
cmd/              the command-line tools
scripts/          wombat (run and render), pretty.py (the renderer)
web/              TypeScript event types, generated
docs/             design notes
```

<a id="en-dev"></a>

## Development

```bash
make              # gofmt check + go vet + go build
make test         # go test -race ./...
make ts           # regenerate web/events.ts after changing event types
```

The test suite runs offline against scripted model clients and needs no API key.
The only thing you need installed is the Go toolchain: there are no third-party
modules to download.

<a id="en-faq"></a>

## Troubleshooting

| Symptom | Fix |
|---|---|
| `scripts/wombat: no env file` | Run `cp .env.example .env` and fill it in, or point `WOMBAT_ENV` at your env file |
| `no API key` error | Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env`, and check that `WOMBAT_PROVIDER` matches |
| `no model on the request and no client default` | Set `WOMBAT_MODEL` (or `ANTHROPIC_MODEL`), or pass `-model` |
| Not sure which model IDs your endpoint accepts | `make models` |
| Token counts and cost stay at 0, so the budget never trips | Some gateways drop usage data when streaming. Set `OPENAI_STREAM="never"` (or `ANTHROPIC_STREAM`). Also check that the model is priced in `llm/pricing.go` |
| Output seems frozen with a reasoning model | It is thinking. Use `scripts/wombat -r` to see the reasoning stream |
| HTTP 400 when using `-media` | The model does not accept images. Drop `-media` |
| Every shell command is refused | You are using `-permission workspace` or `ask` with `wombat-jsonl`, which cannot ask you. Use `-permission off` with `-confine`, or use `wombat-serve` |
| Prompt with quotes or `&&` breaks under `make` | Use `scripts/wombat "..."` rather than a make target |

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
   terminal, a JSONL pipe and a web UI all consume the same stream.
3. **A budget is a `context.Context`.** Going over a limit cancels it, and every
   blocking call in the process already knows how to unwind from that.

The full write-up covers context structure, skills, fault tolerance, tracing,
permissions and evaluation, and what was gained and lost compared with the
OCaml original: **[docs/DESIGN.md](docs/DESIGN.md)**.
