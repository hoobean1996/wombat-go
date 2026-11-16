# wombat-go — design notes

> This is the long-form design rationale: why the harness is shaped the way it
> is, and what each mechanism learned from being wrong first. For installation,
> configuration and usage, start with the [README](../README.md).

A ReAct agent harness in Go, and an always-on agent service built on it.

It is a rewrite of a private OCaml 5 project of mine that is built on algebraic
effects.

The OCaml version's thesis is that every side effect — LLM calls, tool
dispatch, file I/O, logging, time, human input — should be an algebraic effect,
so that the handler stack installed around the agent decides what actually
happens. This version does not translate that. It asks what the same
properties look like when Go writes them.

The answer is: mostly, they were dependency injection.

```go
client, err := anthropic.New(anthropic.Config{Model: "claude-opus-5"})
if err != nil { return err }

a, err := core.New(
    core.WithClient(client),
    core.WithTools(builtin.Default(builtin.Deps{})...),
)

ctx, cancel := governor.WithBudget(ctx, governor.Limits{CostUSD: 1.00})
defer cancel()

run := a.Start(ctx, core.Ask("what does this repo do?"))
defer run.Close()

for run.Next() {
    switch ev := run.Event().(type) {
    case core.TextDelta: os.Stdout.WriteString(ev.Text)
    case core.ToolStart: log.Println("→", ev.Name)
    }
}
if err := run.Err(); err != nil { return err }
```

## Try it

```
git clone https://github.com/hoobean1996/wombat-go && cd wombat-go
cp wombatd.example.yaml wombatd.yaml   # fill in the model section
make serve                             # builds bin/wombatd and runs it

# in another terminal: start a session, watch it run, and read what it says
curl -s localhost:8080/api/sessions -H 'Content-Type: application/json' \
     -d '{"prompt":"what does this repo do?"}'
curl -sN localhost:8080/api/sessions/<id>/events
curl -s 'localhost:8080/api/outbox?after=0&wait=30s'
```

There is one command, and its clients talk to it over HTTP. `wombatd` runs
sessions that wake on events — a person's message, a webhook, a timer the
agent set for itself — and answer through an outbox (see [Always on](#always-on)
and the `httpapi` package doc for every route); and every capability
below is a key in `wombatd.yaml`, its one source of configuration:
`wombatd.example.yaml` shows them all, with their defaults and what each is
for, and a file names only what it changes. A build that has to pass its
tests, an acceptance gate and an adversarial review before it counts as done:

```yaml
model:
  provider: anthropic
  name: claude-sonnet-4-5
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

A key wombatd does not know is an error naming its line, not a setting that
silently did not take effect.

If a turn stopped on `ask_user`, the next message in that session is the
*answer* to the question rather than a new instruction.

## Three decisions

**A tool takes a context and its arguments. Nothing else.**

```go
type Fn func(ctx context.Context, in json.RawMessage) (string, error)
```

Whatever a tool needs is captured when it is constructed — `ViewFile(fsys)`,
`Bash(runner)`, `HTTPGet(client)`. Eight of the OCaml's thirteen effects exist
only because an OCaml tool handler was `Yojson.t -> result` and had no other
route to the outside world. Give the closure the dependency and they are not
needed. A tool is then an ordinary function: testing one requires no harness.

**A run is a stream of events, not a value plus four callbacks.**

The OCaml harness returns an outcome and emits through `on_log`,
`on_text_delta`, `on_event` and `on_tool_event`; a 473-line sidecar exists to
sew those back into one JSONL stream. Here the stream is the primary API, and
putting it on the wire is a loop and an encoder: `wombatd`'s event stream
is `for run.Next()` and an SSE frame. Four callbacks were never a design —
they were the shape of not having a stream type.

Diagnostics are separate: they go to `log/slog`. The event stream carries
semantic events a front end renders.

**A budget is a context.**

```go
ctx, cancel := governor.WithBudget(ctx, governor.Limits{CostUSD: 1.00, Wall: 10*time.Minute})
```

Exceeding a limit cancels the context. Every blocking operation in the process
— the HTTP request, `exec.CommandContext`, a channel select — already unwinds
on its own, so aborting from arbitrary depth costs zero new return values.
`context.Cause(ctx)` reports which limit tripped. The OCaml equivalent raises
through the whole stack from inside an effect handler and needs care at every
`try` that might swallow it.

A cap of N lets N happen: `Steps: 3` is three model calls, sub-agents'
included, and `ToolCalls: 1` runs the one call it allows. Only a call that
reaches its tool counts toward `ToolCalls` — one a permission gate turned down
used nothing — but every request counts toward `RepeatedToolCalls`, refused or
not, because asking again and again for a call that keeps being refused is
exactly the loop it is there to catch.

## Talking to a gateway

Corporate deployments put the model behind a proxy, a routing header and a
non-default base URL. `ConfigFromEnv` reads all of it and hands back a plain
`Config`, so anything the environment got wrong is fixable before `New` sees
it:

```go
cfg := openai.ConfigFromEnv()   // OPENAI_BASE_URL, OPENAI_PROXY, OPENAI_EXTRA_HEADERS, …
cfg.Stream = llm.StreamNever    // if this gateway drops usage while streaming
client, err := openai.New(cfg)
```

wombatd takes the same things from the model section of its file instead:

```yaml
model:
  base_url: https://gateway.example.com/v1
  proxy: proxy.example.com:3128
  headers: {X-Route: my-team}
  stream: never
```

Two things a real gateway teaches you that a mock does not:

- **Reasoning models stream a second channel.** `reasoning_content` (OpenAI
  compat) and `thinking_delta` (Anthropic) both arrive as
  `llm.Delta{Reasoning: …}` and surface as `ReasoningDelta` events, kept
  separate from the answer. On some models it is most of the generated tokens,
  so a UI that only renders `TextDelta` looks frozen.
- **Streaming and accounting can conflict.** Usage rides on a trailing chunk;
  some servers omit it, and then a streamed call reports zero tokens and cost
  budgeting quietly stops working. `llm.StreamNever` trades deltas for
  accounting when that happens.
- **A stream is a response too.** It is held to the same size cap as a
  buffered body, and one that runs past it fails as a server error instead of
  growing for as long as the gateway keeps sending.

## Context has structure, and the structure is the point

Three layers, each structured differently on purpose:

| | shape | why |
|---|---|---|
| conversation | typed blocks: `Text` / `ToolUse` / `ToolResult` / `Thinking` | lets eviction be surgical |
| system prompt | `[]block` at construction, one frozen string afterwards | the cache prefix must be byte-stable, so nothing may rebuild it |
| what the harness knows | `View.Results`, keyed by tool_use id | a strategy that sees only messages can trim by position and nothing else |

That third row is what makes the rest work. Two `ToolResult` blocks have the
same type whether one is nine kilobytes of loaded skill and the other is a
line of grep output. A tool labels its own observation:

```go
tool.Annotate(ctx, skill.Tag, skill.TagPrefix+name)
```

and a strategy acts on the label:

```go
core.DropTagged(40, 12, skill.Tag)   // evict skill bodies, keep everything else
```

`DropTagged` removes whole `tool_use`/`tool_result` **pairs**, not a prefix of
the transcript. It reclaims exactly the nine kilobytes and leaves the reasoning
around them intact — only possible because content is a typed block list.

A window counted in messages cannot help the run that needs it most: one
instruction followed by a hundred tool calls has exactly one place such a
window may cut, at the very start, and one message can be a line of grep or
sixty kilobytes of log. `TokenWindow` cuts in tokens, and inside the exchanges:

```go
core.WithContextWindow(200_000),
core.WithStrategy(core.TokenWindow(0.7, 0.5)),       // trim at 70%, back to about 50%
core.WithCompaction(160_000, 50_000, core.Summarize(client, "")),
```

Old tool results are elided first — each keeps its place and its `tool_use`,
with a note saying what was there — and only then are old exchanges dropped
whole. The estimate is calibrated against the prompt tokens the provider
reports after every call, and the boundary moves in quantized steps of token
mass, so the prefix the provider caches is not edited every turn. Compaction
is the heavier tool: past its trigger the old part of the transcript is
replaced, for the rest of the run, by a summary of it. `ContextTrimmed`,
`TranscriptCompacted` and `ContextOverflow` events say when each happened.

When the provider refuses a request anyway, `WithOverflowRecovery` shrinks it
a rung at a time and sends it again. It starts in tokens — eliding old tool
results — because the run that overflows most, one instruction and a long run
of tool calls, has nowhere a window counted in messages may cut. A rung counts
only if the request came out smaller in tokens; each refusal is taken as an
upper bound on the real window, so the next attempt aims under the smallest
one refused; and a compaction, having made room, starts the ladder again from
the bottom. With Anthropic's server-side `clear_tool_uses` on, `load_skill`
results are excluded from it, so the server cannot evict a skill's body behind
the back of the bookkeeping that retires its tools.

## What a tool hands back

- **Long output is kept, not lost.** `bash` returns the start and the end of
  what a command printed, because a failing test run's assertion is at the end.
  The whole of it is saved to a file the clipped result names, so a model that
  needs the one stack frame in the middle greps that file instead of paying for
  the run again.
- **Pictures are observations too.** `tool.Attach(ctx, llm.Media{...})` adds
  an image or a PDF to a call's result, and both providers carry it to the
  model. `builtin.ViewMedia` (`tools.media` in wombatd) lets the model look at a
  screenshot or a rendered chart rather than at its bytes.
- **Reads are bounded.** `view_file` and `str_replace` read regular files
  only, at most `builtin.MaxReadBytes`, and give up when the call is
  cancelled, so `/dev/zero`, a FIFO or a multi-gigabyte log cannot take the
  process down with them. A tool's result is capped as well;
  `WithTruncationExempt` names the ones that must arrive whole, and
  `skill.Install` exempts `load_skill`, because a skill body is shown once and
  a truncated one could never be read in full.
- **Reads run together; anything else is a barrier.** A turn's calls run up to
  four at a time, but only read-only tools overlap: a write, a command, or a
  tool that declares nothing waits for every call before it and holds back
  every call after it. A turn of five greps runs at once; a turn that writes a
  file and then tests it runs exactly as written.

## Skills

Discovery is static, load and unload are dynamic:

```go
skills, _ := skill.LoadDir("./skills", func(err error) { log.Print(err) })
reg := skill.New(skills...)
reg.Gate("git-history", "git_log")     // hidden until the skill is loaded
g := reg.Bind(builtin.Default(deps))

a, _ := core.New(core.WithClient(client), g.Install())
```

`Install` is the gated set, the middleware that enforces it, the catalogue and
a per-run activation state, as one option. They used to be four options to pass
together, and leaving one out failed quietly — a hidden tool that still ran
when named, or a `load_skill` that reported success while its tools never
appeared — so `New` also asks a set that needs per-run state whether a run
would have it, and logs a warning when it would not.

Live, against a real model: turn 1 offers 12 tools, the model reads the
catalogue and calls `load_skill("git-history")`, turn 2 offers 14.

Four things that are easy to get wrong and are handled:

- **Activation is per run, not per agent.** An `Agent` is immutable and shared
  across goroutines; activation state is born in `WithRunContext` so two
  concurrent runs cannot see each other's skills.
- **Visibility hides; middleware enforces.** Dropping a tool from the list does
  not stop a model that saw the name earlier in the transcript from calling it.
  `Set.Find` therefore stays unfiltered, so the refusal can say *call
  `load_skill("git-history")` first* instead of *unknown tool*.
- **Eviction retires the skill.** A skill's body enters the conversation as a
  tool_result. When a strategy drops it, `tool.Reconciler` tells the set, and
  the gated tools go with it — otherwise the model keeps tools whose knowledge
  left the context.
- **The catalogue is sorted.** It lands in the system prompt, so map iteration
  order would change the bytes between processes and silently kill the cache.

Unloading retires tools, not tokens: the body stays in the transcript until a
strategy evicts it. Those are two mechanisms and it is worth knowing which one
you are reaching for.

## Sub-agents

A delegate tool runs a whole child agent, on the parent's context:

```go
child, _ := core.New(core.WithClient(client), core.WithName("researcher"),
    core.WithTools(tool.Filter(all, tool.OnlyCaps(tool.CapReadOnly))...))

parent, _ := core.New(...,
    core.WithTools(append(all, core.DelegateTool(child))...),
    core.WithToolTimeoutFallback(0),   // a child's bound is its own iteration cap
)
```

Running on the parent's context is the payoff of the whole design: the budget,
the skill activation set, the tape and the tracer all reach the child by being
values on `ctx`. Nothing forks, nothing is reinstalled. The child's events reach
the parent's stream wrapped in `SubagentEvent`, so a UI can nest them.

`ParallelDelegateTool` runs a batch of children at once. A batch is capped at
`core.DefaultMaxTasks` (16, `WithMaxTasks` to change it) and refused whole
beyond that, and a task waiting for a free slot gives up when the run does.

The depth cap is the one governor limit that **refuses instead of aborting**.
Every other cap is a resource the run has exhausted; nesting depth is a shape,
and a parent told "not deeper" can still do the job itself — but only if it is
still running.

## Fault tolerance

| layer | what it does |
|---|---|
| `WithRecovery` | innermost: a panicking tool becomes an error the model reads, and trips the breaker exactly like any other failure |
| dispatcher backstop | catches a panic in a *middleware*, which under `WithParallelism` would otherwise die on a worker goroutine |
| loop recover | a panic in a strategy or a client becomes `ErrPanic`, not a dead process |
| `WithTimeout` | a context deadline, so a tool that honours `ctx` is actually stopped — not abandoned |
| `WithRetry` | needs `Idempotent` **and** `Retryable`; a write is never replayed because an error looked transient |
| `WithCircuitBreaker` | per tool, **process-wide** — a dependency that is down is down for every run |
| `WithDedupRepeats` | per **run** — being stuck in a loop is a property of one conversation |
| `WithOverflowRecovery` | the one error the harness can actually fix: shrink the transcript and retry, sticky until a compaction makes room again |

The model sees a short message; the stack goes to `slog`.

## Recording and replay

```go
tp, _ := tape.Open("run.jsonl", tape.Auto)
defer tp.Close()
a, _ := core.New(core.WithClient(client), tp.Install())
```

`Install` puts the model half outside the client's retry layer and the tool
half in the tool chain. Recording only the model half was the easy mistake, and
it replays the model asking for every side effect again — and the tools
performing them again.

Entries are keyed by a **content hash of the request**, not by position. The
OCaml original keyed by position and died with `tape misalignment` the moment a
run diverged — which, for a differential-testing tool, is exactly when you need
it. A miss in `Auto` is a live call. A line may carry a format revision; every
line so far is revision 0 and spells nothing, and a tape with a line from a
revision newer than the reader's is refused by name rather than replayed as
something it is not.

Measured on a real run: record took minutes against a gateway, replay took
**0.107 s with the endpoint pointed at a dead address**, and the answer was
byte-identical. Deltas coalesce to one per block on replay, and a replayed call
charges no budget — both documented, neither hidden.

## Tracing

```go
sink, closer, _ := trace.FileSink("run.ndjson")
defer closer.Close()
a, _ := core.New(..., core.WithTracing(trace.New(sink)))
```

```
run        wombat                 26632ms
  iteration  wombat iteration 1   21375ms
    llm        other               2819ms
    tool       delegate           18540ms
      subagent   delegate         18539ms
        run        researcher     18531ms
          iteration  …               900ms
            llm        subagent      814ms
            tool       git_log        77ms
```

That nesting is free: a goroutine inherits `ctx` and therefore its parent span.
The OCaml needed `Domain.DLS` plus an explicit `Trace.fork`, which does not
exist here.

Attribute keys are OpenTelemetry `gen_ai.*` semconv verbatim, so bridging to a
real `TracerProvider` is about thirty lines in your own module — the dependency
is yours to add, not this library's. `trace.WriteHTML` renders a self-contained
report with no external assets, and neither middleware records message content:
traces get shipped to places transcripts should not go.

## Permission

A tool call can be allowed, denied, or put in front of a person. The gate is a
`tool.Middleware`, so it sees the same call the tool would have seen — the
command string, the path, the arguments — and decides with all of it in hand.

```go
gate := permission.Gate(permission.Workspace("/work/proj"), approver)
```

Three things this got right only after being wrong first.

**A rooted filesystem is not a sandbox.** An early version confined the file
tools to a prefix and left `bash` alone; a model refused `view_file /etc/hosts`
and read the same file with `cat` on the next turn. The flag was renamed to say
what it does, and the honest boundary is `permission`, which has an opinion
about the shell too.

**An allowlist that never fires is not a safety feature.** `SafeCommands` skips
the approval for `go build`, `go test`, `ls`, `cat` and about thirty others so
that nobody is asked twenty times in an afternoon. It originally refused any
command containing a shell metacharacter, which is the safe reading and also
every command a model writes — in a live run an agent could not test the code
it had just written, because it wrote `go test ./... 2>&1 | head -n 300`. A
pipeline is now admitted, with *every stage checked independently*: that is
strictly stronger than a prefix match over the whole string, which never
examined the dangerous half of `go test; rm -rf /` at all.

**A denial has to teach.** When the same agent was refused, the message said
"try an approach the policy allows outright" and it hit the same wall three
times. It now names the commands and the constraint.

**A policy is not a sandbox either.** Every process the tools started used to
inherit the harness's environment, provider key included, so one `go test` a
policy waved through was enough to print it. `sandbox.Unconfined` now passes an
allowlist — the process basics, the toolchain's own settings, proxies — and
nothing else; `WithEnvAllow`, `WithEnv` and `WithInheritedEnv` change that, and
the verifiers that run the agent's code get the same filtered environment. For
containment rather than hygiene, `sandbox.Confined` runs every command under
`sandbox-exec` on macOS or `bwrap` on Linux: writes only inside the workspace
and the temp directory, the usual credential directories unreadable, the
network loopback-only unless asked otherwise. It refuses to run rather than
running unconfined, and `tools.confine` turns it on in wombatd — for the
tools' commands and for the ones `done.check`, `done.accept` and the review
run, which are the agent's code too.

**Nobody has to be waiting.** An approver either answers at once or defers:
`permission.Defer()` ends the turn on the call instead of holding it open, as
`core.Deferred` — the calls waiting, and the results of the ones that ran
beside them. `core.Decide` resumes it with what people decided: the approved
calls run then, through the same gate, and the model reads their real output
rather than "approved". wombatd defers everything its policy asks about, so a
session can wait a week for a decision without a goroutine waiting with it.

Leaving `SSH_AUTH_SOCK` out of the environment turned out to hide the agent's
path and nothing else: its socket sat in the shared temp directory, which is
writable and therefore open to sockets, and a child could list it as well as
the harness could. The sandbox now closes, in every network mode, the sockets
through which a process acts with the user's authority rather than its own —
ssh and gpg agents, the tmux or editor the harness runs inside, the user's
session bus and runtime directory, container daemons. It is a list, so a socket
it does not know about needs a `DenyRead` entry; `ReadCredentials` opens it
along with the credential directories.

## Knowing when it is done

A run that ends because the model said so is only as good as that judgement,
and the benchmark report below has a line for each way it goes wrong.
`WithCompletionCheck` asks something better at the moment it matters:

```go
core.WithCompletionCheck(3, core.CommandCheck(runner, "go test ./...", 0))
```

A rejected answer is followed by a user turn carrying the check's output; a
rejected terminal call gets it as its `tool_result`, as an error. After the
given number of rejections the run ends as the model wanted, and the last
`CompletionChecked` event says the check still disagreed — a check that can
never pass must not hold a run until its iteration cap, at full price. Run it
through the same `Runner` as the agent's own commands: a check that escapes
the sandbox is a way out of it.

The other direction is a person who can see the run going wrong.
`run.Steer("use the v2 API")` puts a message in front of the model at its next
step without stopping the turn; one that arrives too late for it comes back
from `run.Undelivered()` rather than being dropped. Over HTTP it is
`POST /api/sessions/{id}/steer`.

## MCP

`pkg/core/mcp` connects to Model Context Protocol servers, over stdio or Streamable
HTTP, and offers their tools like any others:

```go
servers, _ := mcp.LoadConfig("mcp.json")    // {"mcpServers": {...}}, as desktop clients write it
conns, err := mcp.ConnectAll(ctx, servers)
defer conns.Close()
tools := append(builtin.Default(deps), conns.Tools()...)
```

A tool is offered as `<server>__<tool>`, and its annotations become
capabilities — `readOnlyHint` is `CapReadOnly`, `openWorldHint` is
`CapNetwork`, and a tool that says nothing is taken to write and to reach the
network — so the read-only filters, the permission policies and the parallel
scheduler treat it as they would a built-in one. Those hints are the server's
claims about itself; `"ignoreHints": true` takes every tool of that server to
mutate. A stdio server gets the same filtered environment as the agent's
commands plus exactly the `env` its entry names (`"${GITHUB_TOKEN}"` passes one
through), the images a tool returns reach the model as images, and a server
that dies fails its calls with the end of its stderr. `tools.mcp: mcp.json`
offers its tools to every wombatd session.

## Evaluation

`eval/rl` turns a run into an `Episode`: a trajectory with a reward. `Rollout` runs
n samples of an `Env` and reports **pass@k** with the unbiased estimator from
Chen et al. rather than "did any of n succeed", which overstates every number
it produces.

`eval/benchmarks` is eight coding tasks defined as data, in two tiers. The easy
tier is the floor — a live model scores 4/4, and when it does not the harness
is broken rather than the agent. The hard tier discriminates: files the prompt
never names, a bug two hops from its symptom, a genuinely underspecified
request, and a haystack that does not fit in the window.

The interesting part is the anti-cheat. `GoProbe` writes a test into a *private
copy* of the workspace after the agent has stopped, from bytes it never had
access to, and it closes two freebies a naive scorer hands out: `go test ./...`
exits 0 in a module whose packages were all deleted, and a `-run` pattern
matching no test also exits 0.

```
$ make bench TASKS=hard N=4
TASK                N  PASS@1  PASS@K   MEAN    STD  TURNS  TOK/IN  TOK/OUT  COST
refactor-interface  4   0.000   0.000  0.250  0.433   30.0  423.1k    67.3k   n/a

unpriced: some-gateway-model — no rate for the above, so COST reads n/a

solved but did not finish (1): full reward, and the run still failed —
the agent did the work and could not tell it was done. Not counted as a pass.
  refactor-interface#2  reward 1.000  max_iterations
```

That last block exists because the first version of this report said pass@1
0.000 for a task one sample had solved outright, and the only trace of it was a
standard deviation. "This agent cannot do the task" and "this agent cannot tell
when it is done" are different bugs with different fixes, and they rendered
identically.

## Always on

`pkg/daemon` keeps sessions going with nobody watching, after Anthropic's
"Building always-on agents": nothing starts a turn but an event in a session's
inbox, and what the agent has to say goes to an outbox.

- **The inbox.** A person's message, a decision on a call that was waiting for
  one, a timer that fired, anything a client posts to `POST /api/events` — a
  webhook, an alert — is a row, safe once the request returns. An event with a
  dedup key is received once however often it is delivered, and an event about
  a thread — an issue, a chat thread — finds the session for that thread, or
  starts one.
- **Who said what.** An event carries what it is, where it came from and, when
  the sender knows, who sent it: `kind`, `source`, `from`. The model reads a
  direct message as plain text and anything else in a tag naming all three —
  a webhook's payload is not the user's instruction, and a chat relayed from
  elsewhere has more than one speaker. The transcript stays byte for byte what
  the model was sent, for the prompt cache, so where its messages came from is
  kept beside it rather than in it: the session's origins, keyed by message
  index, renumbered when a compaction folds the past away.
- **One turn at a time.** A worker takes a session's pending events, all of
  them, and runs one turn on them; what arrives meanwhile is the next turn's.
  A turn the provider fell over in, or that a restart cut short, runs again,
  from the transcript it started from.
- **Timers.** The agent sets its own with `set_timer` — once, to follow up, or
  every so often — and a client with `POST /api/timers`. A timer that is due
  puts an event in the inbox; a daemon that was down fires each one once when
  it comes back, not once for every period it missed.
- **Decisions.** A call the policy puts in front of a person defers, as above:
  the session is saved mid-call, waits for decisions and takes nothing else,
  and resumes once they are all in.
- **The outbox.** The reply to a message, a question the model asks, the calls
  waiting for a decision, and what the agent chooses to say with
  `send_message` — a timer's turn says nothing unless it does. A client
  long-polls `GET /api/outbox?after=<id>&wait=30s` and acks; with
  `outbox.webhook` set, wombatd POSTs each message itself, in order.

```
POST   /api/sessions                          {"prompt": …, "from"?} → a session, and its first message
POST   /api/sessions/{id}/messages            {"prompt": …, "from"?} → a message, for the next turn
GET    /api/sessions/{id}/messages            the transcript, and its messages' origins
GET    /api/sessions/{id}/events              SSE across every turn, honours Last-Event-ID
POST   /api/sessions/{id}/steer               {"prompt": …} → a message for the turn still running
POST   /api/sessions/{id}/approvals/{use_id}  {"allow": true|false, "from"?} → decide a waiting call
POST   /api/events                            {"thread"?, "kind", "source", "from"?, "dedup_key"?, "payload"}
GET    /api/outbox                            what the agent said; ?after= ?wait=
POST   /api/timers                            {"session_id", "note", "in"|"at", "every"?}
```

The full table — the inbox, listing, acking, `/metrics` — is in the `httpapi`
package doc. Sequence numbers on a session's stream are **session-global, not
per turn**, so a client that reconnects across a turn boundary resumes at the
event it dropped instead of at the start of the current turn.

Everything lives in a `daemon.Store`: in memory, or — with `store.driver:
postgres` — in four tables, `sessions`, `inbox`, `timers` and `outbox`,
which `pkg/store/postgres` creates on the first start. Both pass one
conformance suite, `pkg/daemon/storetest`, so the daemon's tests against the
memory store stand for Postgres too; `make pg-up` starts one in podman and
`make test-pg` runs the suite and a restart test against it. One wombatd to a
schema: a second is refused at the start, because the first thing a daemon does
is put back the events a dead one left running. Transcripts are `json`, not
`jsonb`, so a tool call's input comes back with its keys in the order the
model wrote them — a prompt cache that matches a transcript byte for byte
would otherwise miss on every turn after a reload.

Being on loopback is not being private from the browser on the same machine.
A request body must be sent as `application/json`, which is not a type a page
on another site may POST without asking first — before that check, any page
the operator visited could start a session here, prompt and all, and the
session ran tools. A page can also rebind a name of its own to 127.0.0.1 and
become same-origin; `wombatd` listens on loopback by default and, there,
answers only to loopback names (`httpapi.WithAllowedHosts`).
`server.token` adds a bearer token (`httpapi.WithBearerToken`); beyond
loopback, a server without one logs that anyone who can reach it can drive
it.

## What is kept

The middleware chains, which are the best part of the original and translate
exactly:

```go
client := llm.Chain(leaf,
    llm.WithValidation,
    core.TrackCost(llm.DefaultPricing),
    llm.WithRetry(llm.DefaultRetryPolicy),   // cost is charged per attempt
    llm.WithLogging(logger),
)
```

Order is semantic — chaos outside retry, tracing outside chaos, observation
outermost so one logical call yields one event pair. Unlike a stack of effect
handlers, position here cannot cause an inner layer to be silently shadowed.

`Pause` and `Terminal` were effects that abandoned their continuation. They are
outcomes:

```go
switch o := run.Outcome().(type) {
case core.Answer:    // the model ended its turn
case core.Paused:    // it called a CapPause tool; answer and resume
case core.Submitted: // it called a CapTerminal tool; the args are the result
case core.Deferred:  // a call waits for a person; core.Decide resumes it
}
```

Failure is `run.Err()`, matched with `errors.Is`. One failure channel, not two.

## What is lost

- Exhaustive matching over `Event` and `Outcome`. A linter, not a compiler.
- Illegal transcripts being unrepresentable. `Convo` is an append-only slice
  validated at the boundary, because Go cannot express the OCaml version's
  abstract type without a package per invariant.
- Installing a handler from outside to change an inner leaf's behavior. The
  OCaml documents this and never uses it.
- The effect system itself, which is the original project's reason to exist.

## Layout

The module is `wombat`. Everything a program builds an agent from is under
`pkg/core`; what keeps agents running is `pkg/daemon`, and its HTTP surface
`pkg/httpapi`; the one command is `cmd/wombatd`.

```
pkg/core/             Agent, Run, Event, Outcome, Convo, Strategy   ← the front door
pkg/core/llm/         wire vocabulary, Client, middleware, pricing
pkg/core/llm/anthropic/   Messages API client (net/http, no curl subprocess)
pkg/core/llm/openai/      OpenAI-compatible client
pkg/core/tool/        Def, Set, Typed, dispatch, middleware, fault attribution
pkg/core/tool/builtin/    the twelve built-in tools, and view_media
pkg/core/sandbox/     the runners every command goes through: filtered, and confined
pkg/core/permission/  per-call authorization, with a human in the loop
pkg/core/skill/       progressive disclosure: tools that appear when a skill loads
pkg/core/mcp/         Model Context Protocol client: a server's tools as tool.Defs
pkg/core/governor/    budgets as context cancellation
pkg/core/accept/      the acceptance gate: build, vet, test, a finished plan
pkg/core/review/      the adversarial review: defects shown by failing tests
pkg/core/tape/        content-hash record and replay
pkg/core/trace/       OTel-shaped spans, NDJSON and a self-contained HTML report
pkg/core/metric/      Prometheus text exposition, hand-written
pkg/daemon/           always on: sessions, the inbox, timers, deferred decisions, the outbox
pkg/daemon/storetest/ what a daemon.Store has to do, as tests both stores pass
pkg/store/postgres/   the sessions and the three tables in Postgres
pkg/httpapi/          the daemon over HTTP: events in, SSE and the outbox out
cmd/wombatd/          the agent as a service; everything configured by wombatd.yaml
eval/                 does the harness work: request shape, wire conformance, judges (make judge)
eval/rl/              episodes, verifiers, pass@k
eval/benchmarks/      eight coding tasks as data, and the suite that runs them (make bench)
internal/             fspath, safeenv, shellquote, sse, yaml, and the wiring checks
skills/               the skills shipped with the harness
```

One dependency, `pgx`, and only `pkg/store/postgres` imports it; the harness
itself is the standard library.
