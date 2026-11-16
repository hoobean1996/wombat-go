# wombat 使用指南

[English](GUIDE.md) · **简体中文** · [← README](../README.zh-CN.md)

[README](../README.zh-CN.md) 没有展开的内容都在这里：Go 库的用法、每一个配置项、常驻运行的机制、HTTP API，以及各项功能的说明。每个部分为什么这样设计，见 [DESIGN.md](DESIGN.md)。

- [作为 Go 库使用](#zh-quickstart-lib)
- [核心概念](#zh-concepts)
- [内置工具](#zh-tools)
- [配置](#zh-config)
- [常驻运行](#zh-always)
- [HTTP API](#zh-http)
- [命令与 make 目标](#zh-cli)
- [功能指南](#zh-guide)
- [目录结构](#zh-layout)
- [开发](#zh-dev)
- [常见问题](#zh-faq)

<a id="zh-quickstart-lib"></a>

## 作为 Go 库使用

模块路径是 `wombat`，所以其他模块需要通过一条指向本地检出目录的 `replace` 指令来使用它：

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
	// 1. 模型客户端。ConfigFromEnv 读取 ANTHROPIC_API_KEY、ANTHROPIC_MODEL、
	//    ANTHROPIC_BASE_URL 等环境变量。OpenAI 兼容接口改用
	//    openai.New(openai.ConfigFromEnv())。
	leaf, err := anthropic.New(anthropic.ConfigFromEnv())
	if err != nil {
		log.Fatal(err)
	}
	// TrackCost 把每次响应的费用计入本次运行的预算。
	// 不加它，CostUSD 限额永远不会触发。
	client := llm.Chain(leaf, core.TrackCost(llm.DefaultPricing))

	// 2. 创建 Agent：模型客户端 + 它可以使用的工具。
	agent, err := core.New(
		core.WithClient(client),
		core.WithModel("claude-sonnet-5-5"),
		core.WithTools(builtin.Default(builtin.Deps{})...),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 3. 设置预算：花费达到 1 美元或运行满 10 分钟，以先到者为准，运行就会被取消。
	ctx, cancel := governor.WithBudget(context.Background(),
		governor.Limits{CostUSD: 1.00, Wall: 10 * time.Minute})
	defer cancel()

	// 4. 启动运行，并消费事件流。
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

**运行是怎么结束的？** 循环结束后，用 `run.Outcome()` 判断：

```go
switch run.Outcome().(type) {
case core.Answer:    // 模型完成了这一轮
case core.Paused:    // 模型通过 ask_user 向用户提问；回答后可以继续
case core.Submitted: // 模型调用了终止型工具，工具参数就是结果
case core.Deferred:  // 有调用在等人做决定；用 core.Decide 恢复运行
}
```

**多轮对话：** 把上一次的对话记录传给下一次运行。

```go
run = agent.Start(ctx, core.Then(run.Messages(), "What was it times three?"))
```

如果上一轮以 `Paused` 结束，改用 `core.AnswerPause` 或 `core.Resume`；如果以 `Deferred` 结束，改用 `core.Decide`。

**添加自定义工具。** 一个工具由名称、JSON Schema、能力声明和一个普通 Go 函数组成：

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
	Caps: tool.CapReadOnly, // 告诉调度器和权限检查这个工具会做什么
}, func(ctx context.Context, in weatherIn) (string, error) {
	return "Sunny, 22°C in " + in.City, nil
})

agent, err := core.New(
	core.WithClient(client),
	core.WithTools(append(builtin.Default(builtin.Deps{}), weather)...),
)
```

一定要设置 `Caps`。没有声明任何能力的工具会被当作“会写入、会联网”的工具处理：它永远不会并行执行，权限检查也会对它从严处理。

<a id="zh-concepts"></a>

## 核心概念

| 概念 | 说明 |
|---|---|
| `Agent` | 用 `core.New(options...)` 创建一次即可。创建后不可变，可以在多个 goroutine 间安全共享 |
| `Run` | Agent 针对某个输入的一次执行。用 `Next()` / `Event()` 遍历事件，最后检查 `Err()` 和 `Outcome()` |
| `Event` | 可以由界面展示的事件：`TextDelta`、`ReasoningDelta`、`ToolStart`、`Spend`、`ContextTrimmed` 等 |
| `Outcome` | 成功运行的结束方式：`Answer`、`Paused`、`Submitted` 或 `Deferred`。失败通过 `run.Err()` 返回，用 `errors.Is` 判断类型 |
| `tool.Def` | 工具定义：名称、描述、JSON Schema、能力声明（`Caps`），以及 `func(ctx, args) (string, error)` |
| Middleware（中间件） | 包裹模型客户端（`llm.Chain`）或工具的一层逻辑：重试、日志、计费、权限、超时 |
| Budget（预算） | `governor.WithBudget(ctx, limits)` 返回一个 `context.Context`，超出限额时自动取消 |
| Strategy（策略） | 决定长对话中哪些内容发给模型（`TokenWindow`、`DropTagged` 等） |
| `daemon.Manager` | 运行会话：从收件箱取出事件，每个会话同一时间只执行一轮，触发计时器，投递发件箱中的消息 |
| `daemon.Store` | 会话、收件箱、计时器和发件箱的存放位置：`daemon.MemoryStore()` 或 `pkg/store/postgres` |

<a id="zh-tools"></a>

## 内置工具

`builtin.Default(deps)` 返回以下 12 个工具：

| 工具 | 作用 |
|---|---|
| `bash` | 执行 shell 命令（超时 60 秒）。输出过长时保留开头和结尾，完整输出存到文件里，模型可以再去 grep |
| `view_file` | 读取文件（带行号，可指定行范围），或列出目录 |
| `write_file` | 创建或覆盖文件 |
| `str_replace` | 通过精确替换一段字符串来编辑文件 |
| `grep_search` | 用扩展正则表达式搜索文件 |
| `git_log` / `git_show` | 列出最近的提交 / 查看某次提交及其 diff |
| `http_get` | 获取一个 URL 的内容 |
| `calculator` | 计算算术表达式 |
| `current_time` | 返回当前日期和时间 |
| `save_tool_result` | 把之前某次工具结果保存到磁盘，而不必重新输出一遍 |
| `ask_user` | 暂停运行，向用户提问 |

可选的额外工具，每个对应 `wombatd.yaml` 中的一个键：`view_media`（让模型查看图片和 PDF，`tools.media`）、`delegate`（子 Agent，`tools.delegate`）、`load_skill` / `unload_skill`（`tools.skills_dir`），以及以 `<server>__<tool>` 命名的 MCP 工具（`tools.mcp`）。

wombatd 还会给每个会话的 Agent 提供守护进程自己的工具。任何权限策略都不会拦下它们询问——它们不会触碰机器上的任何东西：

| 工具 | 作用 |
|---|---|
| `send_message` | 通过发件箱，对这个会话所服务的人说些什么 |
| `set_timer` | 在之后唤醒会话：一次性（`in` / `at`），或周期性（`every`） |
| `cancel_timer` / `list_timers` | 管理会话的计时器 |

<a id="zh-config"></a>

## 配置

`wombatd` 读取一个 YAML 文件，默认是 `wombatd.yaml`（用 `-config path` 指定其他文件）。`wombatd.example.yaml` 列出了每个键、它的默认值和用途。你的文件只需写出要修改的键；wombatd 不认识的键会报错并指出所在行，而不是被悄悄忽略。

| 配置段 | 设置内容 |
|---|---|
| `server` | `addr`、`token`（bearer 认证）、`cors`、`metrics`、`log` |
| `store` | `driver`（`memory` 或 `postgres`）和 `dsn` |
| `daemon` | `poll`、`max_runs`（同时执行的轮数）、`attempts`（每个事件的尝试次数）、`idle` |
| `outbox` | `webhook`（每条消息 POST 到的地址）及其 `headers` |
| `model` | `provider`、`name`、`base_url`、`api_key`、`proxy`、`headers`、`stream`、`max_tokens`、`system` |
| `tools` | `working_dir`、`permission`、`full_access`、`confine`、`confine_network`、`mcp`、`skills_dir`、`media`、`delegate` |
| `turn` | `budget_usd`、`wall`、`max_iters`、`tool_parallelism`、`context_window`、`compact_at` |
| `mode` | `build`、`plan` 或 `review` |
| `done` | `check`（必须通过的命令）、`attempts`、`accept`、`regressions` |
| `review` | `enabled`、`sweeps`、`dry`、`iters`、`fix` |
| `record` | `tape`、`tape_mode`、`trace`、`trace_html` |

下面是一个构建任务的配置：必须通过测试、验收检查和对抗式审查才算完成，数据保存在 Postgres 中：

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

**模型服务相关的环境变量（供库和基准测试使用）。** 在库代码中，`openai.ConfigFromEnv()` 和 `anthropic.ConfigFromEnv()` 会把下列变量读入一个普通的 `Config` 结构体，你可以先修改它再调用 `New`；除了在 `Config` 没有提供 API Key 时回退读取环境变量（Anthropic 还会回退读取 `ANTHROPIC_BASE_URL`）之外，库本身不会偷偷读取任何环境变量。`make bench`、`make judge` 和 `make models` 从 `.env` 加载这些变量——复制一份 `.env.example` 即可，里面有每个变量的说明。除了上述回退之外，wombatd 不读取其中任何变量。

| 变量 | 说明 |
|---|---|
| `WOMBAT_PROVIDER` | `openai` 或 `anthropic`，供基准测试使用 |
| `WOMBAT_MODEL` | 当前所选服务的默认模型 |
| `OPENAI_BASE_URL` | 服务根地址加 `/v1`，例如 `https://api.openai.com/v1`，程序会自动拼接 `/chat/completions` |
| `OPENAI_API_KEY` | 以 `Authorization: Bearer …` 发送。不需要认证的本地服务请留空 |
| `OPENAI_MODEL`、`OPENAI_PROXY`、`OPENAI_EXTRA_HEADERS`、`OPENAI_STREAM`、`OPENAI_TEMPERATURE`、`OPENAI_TOP_P` | 模型、代理、额外请求头（每行一个 `Name: value`）、`auto` / `always` / `never`、采样参数 |
| `ANTHROPIC_API_KEY` | 必填 |
| `ANTHROPIC_MODEL`、`ANTHROPIC_BASE_URL`、`ANTHROPIC_PROXY`、`ANTHROPIC_CUSTOM_HEADERS`、`ANTHROPIC_BETA`、`ANTHROPIC_STREAM`、`ANTHROPIC_TEMPERATURE`、`ANTHROPIC_TOP_P` | Anthropic 的对应变量；`ANTHROPIC_BETA` 是逗号分隔的 beta 功能标记 |

<a id="zh-always"></a>

## 常驻运行

`pkg/daemon` 参照 Anthropic 的“Building always-on agents”，让会话在无人值守时也能持续运转。只有会话**收件箱（inbox）** 中的**事件**才能启动一轮，而 Agent 要说的话都进入**发件箱（outbox）**。

- **收件箱。** 用户的消息、对某个等待决定的调用做出的决定、一个触发的计时器，以及客户端发到 `POST /api/events` 的任何东西——webhook、告警——都是一行记录，请求一返回就已安全保存。带 `dedup_key` 的事件无论投递多少次都只接收一次；关于某个 `thread`（一个 issue、一个聊天线程）的事件会找到该线程对应的会话，没有就新建一个。
- **谁说了什么。** 每个事件都写明它是什么（`kind`）、从哪里来（`source`），以及在发送方知道时，是谁发的（`from`）。由桥接程序转发的聊天是 `"kind": "message"`，payload 为 `{"text": "..."}`，`source` 是桥接程序自己的名字，`from` 是说话的人。直接消息——`source` 为 `api`、没有 `from`——模型读到的就是纯文本；其他事件都包在一个写明这三项的标签里，所以 webhook 的 payload 不会被当成你的指令，一个说话人也不会被当成另一个。对话记录保持模型收到时的原样；`GET /api/sessions/{id}/messages` 在它旁边返回 `origins`，说明哪条消息是由哪些事件产生的。
- **每次一轮。** worker 取出一个会话的全部待处理事件，针对它们执行一轮；期间新到的事件留给下一轮。`daemon.max_runs` 限制所有会话同时执行的轮数。
- **重试。** 被守护进程中止的一轮，或者因模型服务表明是暂时性的错误（限流、5xx、网络问题）而失败的一轮，会重新执行——从这一轮开始时的对话记录重来，所以对话中不会出现重复内容——最多执行 `daemon.attempts` 次。守护进程如果在一轮中途退出，下一个守护进程启动时会把这些事件放回收件箱。
- **计时器。** Agent 用 `set_timer` 给自己设置计时器——一次性的，用于跟进；或者每隔一段时间触发一次——客户端则用 `POST /api/timers` 设置。到期的计时器会往收件箱里放入一个事件；守护进程停机后恢复时，每个计时器只触发一次，而不是把错过的每个周期都补触发一遍。
- **审批决定。** 权限策略需要人工审批的调用会结束当前这一轮，而不是让它一直挂起：会话在调用中途被保存，之后只接收决定，等所有决定都到齐后才恢复——被批准的调用这时才执行，模型读到的是它们的真实输出。
- **发件箱。** 对消息的回复、模型提出的问题、等待决定的调用，以及 Agent 选择用 `send_message` 说的话——由计时器触发的一轮，除非调用了它，否则什么也不说。客户端用 `GET /api/outbox?after=<last id>&wait=30s` 长轮询，并用 `POST /api/outbox/{id}/ack` 确认；设置了 `outbox.webhook` 时，wombatd 会按顺序自己 POST 每条消息，每条最多尝试 10 次、间隔逐次加长，之后才继续投递排在后面的消息。

一个来自外部、关于某个线程的事件，以及一个明天早上的提醒：

```bash
curl -s localhost:8080/api/events -H 'Content-Type: application/json' -d '{
  "thread": "github:you/repo#12", "kind": "issue_comment", "source": "github", "from": "octocat",
  "dedup_key": "delivery-8f2c", "payload": {"body": "the build is red again"}}'
curl -s localhost:8080/api/timers -H 'Content-Type: application/json' -d '{
  "session_id": "<id>", "note": "check whether the fix went out", "at": "2026-11-16T09:00:00Z"}'
```

**持久保存。** 使用默认的 `store.driver: memory` 时，wombatd 一停止，所有数据都会丢失。使用 `postgres` 时，数据保存在四张表中——`sessions`、`inbox`、`timers` 和 `outbox`——wombatd 首次启动时会创建它们，重启后从中断处继续。一个 schema 只能对应一个 wombatd：第二个使用同一 schema 的实例会在启动时被拒绝（在 DSN 后加上 `?search_path=name`，可以让每个实例使用各自的 schema）。本地数据库可以这样启动：

```bash
make pg-up        # 在 podman 中运行 Postgres 17，监听 127.0.0.1:55432
```

```yaml
store:
  driver: postgres
  dsn: postgres://wombat:wombat@127.0.0.1:55432/wombat
```

<a id="zh-http"></a>

## HTTP API

`wombatd` 提供的是 `httpapi` 包，它是构建在 `daemon.Manager` 之上的一个普通 `http.Handler`：

| 方法与路径 | 用途 |
|---|---|
| `POST /api/sessions` | 创建会话并附带第一条消息：`{"prompt": "..."}`，可选 `from`、`title`、`permission`、`workspace`、`model`、`max_iters` |
| `GET /api/sessions` / `GET /api/sessions/{id}` | 列出会话（最新的在前）/ 查看单个会话 |
| `PATCH /api/sessions/{id}` | 修改会话的权限，从下一轮开始生效：`{"permission": "full"}`（Full Access，同时允许正在等待的调用）或 `""`（服务器默认值） |
| `DELETE /api/sessions/{id}` | 取消并删除会话，连同它的计时器和消息 |
| `GET /api/sessions/{id}/events` | 跨越所有轮次的 Server-Sent Events 事件流。断线后带 `Last-Event-ID` 重连即可续传 |
| `GET /api/sessions/{id}/messages` | 对话记录，以及旁边的 `origins`：对每条由收件箱事件产生的消息，按其下标给出这些事件的 `kind`、`source` 和 `from` |
| `POST /api/sessions/{id}/messages` | 发送一条消息，留给下一轮处理：`{"prompt": "...", "from": "..."}`——`from` 可选，写明是谁说的 |
| `POST /api/sessions/{id}/steer` | 给仍在运行的这一轮发送消息 |
| `GET /api/sessions/{id}/approvals` | 列出等待决定的调用 |
| `POST /api/sessions/{id}/approvals/{use_id}` | 对其中一个做出决定：`{"allow": true, "from": "..."}` |
| `POST /api/events` | 来自外部的事件：`{"session_id" or "thread", "kind", "source", "from", "dedup_key", "payload"}` |
| `GET /api/inbox` | 事件，最新的在前：`?session=`、`?status=`、`?limit=` |
| `GET /api/outbox` | 消息，最早的在前：`?after=`、`?wait=`（最长 60 秒）、`?session=`、`?status=`、`?limit=` |
| `POST /api/outbox/{id}/ack` | 把一条消息标记为已处理 |
| `GET /api/timers` / `POST /api/timers` / `DELETE /api/timers/{id}` | 列出（`?session=`、`?status=`）、设置（`{"session_id", "note", "in" or "at", "every"}`）和取消计时器 |
| `GET /api/health`、`GET /api/config`、`GET /metrics` | 存活检查、本服务提供的功能、Prometheus 指标 |

消息、决定和事件都以 `202 Accepted` 响应，因为它们是排队处理的：由它们引发的结果会出现在事件流和发件箱中。新建会话返回 `201 Created`；如果事件的 `dedup_key` 之前出现过，则返回 `200`，并附上第一次接收的那个事件。所有错误的格式都相同：`{"error": {"kind": "...", "message": "..."}}`，其中 `kind` 的取值是有限的，客户端可以据此分支处理。详情见包文档。

<a id="zh-cli"></a>

## 命令与 make 目标

只有一个命令：`wombatd -config wombatd.yaml`。`make bin` 会把它编译到 `bin/`。

| 目标 | 作用 |
|---|---|
| `make` | gofmt 检查、`go vet`、`go build`（默认目标） |
| `make test` | `go test -race ./...` |
| `make bin` | 构建 `bin/wombatd`；如果 `make ui` 已经构建过网页界面，就把它嵌入进去 |
| `make serve` | 构建网页界面（如果装了 npm）和 wombatd，并用 `wombatd.yaml` 运行；用 `CONFIG=other.yaml` 指定其他文件 |
| `make ui` | 把网页界面构建到 `ui/dist`，供 `make bin` 嵌入（需要 npm） |
| `make ui-dev` | 在 `:5173` 上以热重载方式运行网页界面，连接 `:8080` 上的 wombatd（用 `WOMBATD_URL=...` 指定其他地址） |
| `make ui-check` | 对网页界面做类型检查和 lint |
| `make pg-up` / `make pg-down` | 在 podman 中启动 / 删除一个 Postgres，端口 55432 |
| `make test-pg` | 针对这个数据库运行 Postgres 存储的测试，其中包括守护进程重启的测试 |
| `make bench` | 运行基准测试，例如 `make bench N=8 C=4 TASKS=todo-app,fix-bug` |
| `make judge` | 为一次基准测试运行产出的 episode 打分：`make judge IN=runs/episodes.jsonl` |
| `make models` | 列出 OpenAI 兼容接口提供的模型 ID |
| `make help` | 列出有说明的 make 目标 |

<a id="zh-guide"></a>

## 功能指南

下面每一节都是简要说明。各个机制为什么这样设计，详见 [docs/DESIGN.md](DESIGN.md)（英文）。

<a id="zh-safety"></a>

### 安全

三层相互独立的防护：

| 层 | 作用 | 启用方式 |
|---|---|---|
| **权限策略** | 根据实际的命令、路径和参数，对每次工具调用做出允许、拒绝或请人审批的决定 | `tools.permission`（wombatd），`permission.Gate`（库） |
| **环境变量过滤** | Agent 执行的命令只继承白名单内的环境变量，看不到你的 API Key | 默认开启 |
| **操作系统沙箱** | 命令在 `sandbox-exec`（macOS）或 `bwrap`（Linux）中运行：只能写工作区和临时目录，凭据目录不可读，默认只能访问本机回环网络 | `tools.confine: true` |

wombatd 中的权限级别，从最宽松到最严格排列。`tools.permission` 为所有会话设置一个级别；会话在创建时可以要求更严格的级别（`"permission": "readonly"`），但不能要求更宽松的。唯一的例外是运维者开放了的 **Full Access**：

| 级别 | 行为 |
|---|---|
| `full` | **Full Access。** 提供全部工具，任何调用都不需要等人确认。只有设置了 `tools.full_access: true`（或 `tools.permission: full`）时，会话才能要求它，否则请求会被拒绝 |
| `off` | 没有任何检查，因此只提供只读工具，并且限制在工作区内；不会询问任何人。只有运维者可以选择它 |
| `workspace` | 允许在工作目录内读写，也允许常见的无害命令（`go test`、`ls`、`cat` 等）。其他任何操作——别的命令、网络访问、工作目录之外的路径——都要等待决定（默认值） |
| `ask` | 每次调用都要等待决定 |
| `readonly` | 只执行只读工具，其他一律拒绝 |

守护进程自己的工具（`send_message`、计时器工具）在所有级别下都能执行。

会话的权限级别可以随时修改：`PATCH /api/sessions/{id}`，传 `{"permission": "full"}`（也就是网页界面里的 **Normal / Full Access** 开关），或者传 `""` 回到服务器默认值，从下一轮开始生效。切换到 `full` 时，会话正在等待的调用会被自动允许，正在运行的这一轮如果停在需要确认的调用上，也会被自动允许。Full Access 下 `tools.confine` 依然生效。

等待决定的调用不会让这一轮一直挂起：这一轮在该调用处结束，发件箱会说明在等什么，决定之后再通过 API 传回来，不论要等多久。在库代码中，approver 要么立即作答，要么推迟决定：

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

如果用 `permission.Defer()` 作为 approver，运行会以 `core.Deferred` 结束，等有人做出决定后，再用 `core.Decide(run.Messages(), deferred, verdicts)` 恢复运行。approver 传 `nil` 时，所有“需要审批”都会变成“拒绝”。

### 预算

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

超出任何一项限额都会取消 context，所有进行中的操作（HTTP 请求、子进程）随之停止。`context.Cause(ctx)` 会告诉你是哪一项超限。费用按 `llm.DefaultPricing` 计算。不在价格表中的模型，费用记为 `0`，这类模型请改用 `Wall`、`Steps` 来限制，或者通过 `core.WithPricing` 传入自己的价格。在 wombatd 中，预算按轮计算：`turn.budget_usd`、`turn.wall`、`turn.max_iters`。

### 长对话

```go
core.WithContextWindow(200_000),
core.WithStrategy(core.TokenWindow(0.7, 0.5)),                  // 达到 70% 时裁剪，降回约 50%
core.WithCompaction(160_000, 50_000, core.Summarize(client, "")), // 把旧的轮次压缩成摘要
```

裁剪时先缩短旧的工具结果，不够时才整段丢弃旧的对话轮次。压缩（compaction）则是把最早的一部分对话替换成一段摘要。wombatd：`turn.context_window: 200000` 和 `turn.compact_at: 0.85`。

### 完成检查与中途引导

```go
core.WithCompletionCheck(3, core.CommandCheck(runner, "go test ./...", 0))
```

模型宣布完成时会运行这个检查。检查失败时，模型会拿到检查输出并继续工作，最多重试 3 次。wombatd：`done.check: go test ./...`。

想在运行过程中调整方向，调用 `run.Steer("use the v2 API")` 即可。通过 HTTP 则使用 `POST /api/sessions/{id}/steer`。

### Skills（技能）

一个 skill 是一个包含 `SKILL.md` 文件的文件夹。文件头部的 front matter 写名称和描述，正文是操作说明，模型只在需要时才加载正文。

```markdown
---
name: git-history
description: Reading a repository's history. Load before any question about the past.
---

# Reading git history
...
```

在 wombatd 中设置 `tools.skills_dir: skills`。模型会看到一个 skill 目录，需要时调用 `load_skill` 加载。skill 还可以“解锁”某些工具，在加载之前这些工具对模型是隐藏的：

```go
skills, _ := skill.LoadDir("./skills", func(err error) { log.Print(err) })
reg := skill.New(skills...)
reg.Gate("git-history", "git_log") // 加载 git-history 之后才会出现 git_log
g := reg.Bind(builtin.Default(deps))

agent, _ := core.New(core.WithClient(client), g.Install())
```

本仓库的 `skills/` 目录里有 4 个示例 skill。

### 子 Agent

`delegate` 工具会运行一个完整的子 Agent。子 Agent 与父 Agent 共享预算、权限检查和链路追踪，因为它们都放在 `context` 里传递。

```go
child, _ := core.New(core.WithClient(client), core.WithName("researcher"),
	core.WithTools(tool.Filter(all, tool.OnlyCaps(tool.CapReadOnly))...))

parent, _ := core.New(core.WithClient(client),
	core.WithTools(append(all, core.DelegateTool(child))...),
	core.WithToolTimeoutFallback(0), // 子 Agent 由它自己的迭代次数上限约束
)
```

wombatd：`tools.delegate: true` 会添加一个只读的 researcher 子 Agent。

### MCP（Model Context Protocol）

通过 stdio 或 Streamable HTTP 连接 MCP 服务器。配置文件格式与桌面端 MCP 客户端相同：

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

在 wombatd 中设置 `tools.mcp: mcp.json`；这些服务器只启动一次，由所有会话共享。MCP 工具以 `<server>__<tool>` 命名。它们的 MCP 提示（`readOnlyHint` 等）会转换成能力声明，所以权限策略对它们的处理与内置工具完全一致。在代码中使用 `mcp.LoadConfig`、`mcp.ConnectAll` 和 `conns.Tools()`。

### 录制与重放

```go
tp, _ := tape.Open("run.jsonl", tape.Auto)
defer tp.Close()
agent, _ := core.New(core.WithClient(client), tp.Install())
```

模型响应和工具结果都会被录制下来，以请求内容的哈希作为索引。重放速度快、结果确定，而且完全不发网络请求。wombatd：`record.tape: run.jsonl` 和 `record.tape_mode: replay`。

### 链路追踪与指标

```yaml
record:
  trace: run.ndjson         # 所有会话的 span
  trace_html: run.html      # 关闭时渲染成一份自包含的报告
```

追踪数据使用 OpenTelemetry 的 `gen_ai.*` 属性名，子 Agent 会自动嵌套显示，并且不会记录消息内容。wombatd 在 `GET /metrics` 提供 Prometheus 指标（`server.metrics`，默认开启）。

### 基准测试与评测

```bash
make bench TASKS=easy N=4                 # 运行基准测试套件（一个只在显式要求时才运行的测试）
make judge IN=runs/episodes.jsonl         # 之后再为这些 episode 打分
```

`eval/benchmarks/` 包含 8 个编程任务，分为两档。简单档应该始终拿满分，拿不到说明框架本身有问题；困难档才用来区分不同 Agent 的能力。结果用 pass@k 表示。`eval/` 的其余部分用来衡量框架的各个机制本身是否有效，详见 [eval/README.md](../eval/README.md)（英文）。

<a id="zh-layout"></a>

## 目录结构

```
pkg/core/             Agent、Run、Event、Outcome、Convo、Strategy   ← 从这里开始看
pkg/core/llm/         与具体服务无关的类型、Client 接口、中间件、价格表
pkg/core/llm/anthropic/   Anthropic Messages API 客户端
pkg/core/llm/openai/      OpenAI 兼容客户端
pkg/core/tool/        工具定义、调度、中间件（重试、超时、熔断等）
pkg/core/tool/builtin/    12 个内置工具，以及 view_media
pkg/core/sandbox/     所有命令都要经过的执行器：环境过滤版和沙箱隔离版
pkg/core/permission/  逐次调用的权限控制，支持人工审批
pkg/core/governor/    以 context 取消实现的预算控制
pkg/core/skill/       Skills：按需加载的说明和工具
pkg/core/mcp/         Model Context Protocol 客户端
pkg/core/tape/        录制与重放
pkg/core/trace/       span、NDJSON 输出、HTML 报告
pkg/core/metric/      Prometheus 指标
pkg/core/accept/      确定性的“是否真正完成”检查（build、vet、test、TODO.md）
pkg/core/review/      对抗式代码审查，以失败的测试用例报告缺陷
pkg/daemon/           常驻运行：会话、收件箱、计时器、延后的审批决定、发件箱
pkg/daemon/storetest/ 以测试形式规定存储必须满足的行为，两种存储都要通过
pkg/store/postgres/   保存在 Postgres 中的会话表和另外三张表
pkg/httpapi/          通过 HTTP 暴露守护进程：事件进，SSE 和发件箱出；同时提供网页界面
cmd/wombatd/          服务本体；一切都由 wombatd.yaml 配置
ui/                   网页界面（React + Vite），通过 ui/embed.go 嵌入 wombatd
eval/                 框架自身的评测
eval/rl/              episode、校验器、pass@k
eval/benchmarks/      8 个编程任务，以及运行它们的测试套件
internal/             fspath、safeenv、shellquote、sse、yaml，以及组件装配检查
skills/               示例 skill
docs/                 设计文档
```

<a id="zh-dev"></a>

## 开发

```bash
make              # gofmt 检查 + go vet + go build
make test         # go test -race ./...
make pg-up && make test-pg   # 针对真实数据库测试 Postgres 存储
make ui-check     # 网页界面：tsc + oxlint
make ui-dev       # 以热重载方式运行网页界面，连接正在运行的 wombatd
```

测试套件使用脚本化的假模型客户端，离线运行，不需要 API Key；没有提供数据库时，Postgres 测试会跳过。唯一需要下载的模块是 `pgx`，供 `pkg/store/postgres` 使用。网页界面的依赖是 `ui/node_modules` 下的 npm 包，只在构建时使用；二进制文件里带的是构建产物，运行时不需要这些包。

<a id="zh-faq"></a>

## 常见问题

| 现象 | 解决办法 |
|---|---|
| `wombatd.yaml does not exist` | 执行 `cp wombatd.example.yaml wombatd.yaml` 并填写 model 部分，或者传 `-config path` |
| `unknown key "…" in …` | 键名拼错了，报错里带有行号。`wombatd.example.yaml` 列出了所有可用的键 |
| 报错 `no API key` | 设置 `model.api_key`，并确认 `model.provider` 与 Key 对应 |
| `no model on the request and no client default` | 设置 `model.name` |
| 不知道接口支持哪些模型 ID | 执行 `make models`（会读取 `.env`） |
| token 数和费用一直是 0，预算不生效 | 有些网关在流式模式下不返回用量数据，设置 `model.stream: never`。同时确认模型在 `pkg/core/llm/pricing.go` 的价格表里 |
| 开启 `tools.media: true` 后返回 HTTP 400 | 该模型不支持图片输入，关闭 `tools.media` |
| 会话一直停在 `approving` | 它在等待决定：在网页界面里处理审批卡片，或者先 `GET /api/sessions/{id}/approvals`，再 POST 一个决定。在此之前它不会处理别的任何事情 |
| `/` 显示“built without its web UI” | 先执行 `make ui`，再执行 `make bin`（或者直接 `make serve`）。构建网页界面需要 npm，二进制文件运行时不需要 |
| 网页界面要求输入 token | `wombatd.yaml` 中设置了 `server.token`；输入一次即可，该浏览器会记住它 |
| `another daemon is running on this schema` | 另一个 wombatd 正在使用这个 Postgres schema。停掉它，或者在 `store.dsn` 中加上 `?search_path=name`，给这个实例单独的 schema |
| 消息在收件箱里越积越多 | 查看 `GET /api/inbox?session=…` 和日志：等待决定的会话不会执行别的任何事情，而且 `daemon.max_runs` 限制了同时执行的轮数 |
