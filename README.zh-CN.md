<div align="center">

# wombat-go

**一个用 Go 编写、零第三方依赖的 ReAct Agent 框架。**

[English](README.md) · **简体中文**

![Go](https://img.shields.io/badge/Go-1.25%2B-00ADD8?logo=go&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

</div>

---

- [wombat-go 是什么](#zh-what)
- [功能特性](#zh-features)
- [环境要求](#zh-requirements)
- [快速开始：命令行](#zh-quickstart-cli)
- [快速开始：作为 Go 库使用](#zh-quickstart-lib)
- [核心概念](#zh-concepts)
- [内置工具](#zh-tools)
- [配置](#zh-config)
- [命令行工具](#zh-cli)
- [功能指南](#zh-guide)
- [目录结构](#zh-layout)
- [开发](#zh-dev)
- [常见问题](#zh-faq)
- [设计说明](#zh-design)

<a id="zh-what"></a>

## wombat-go 是什么

wombat-go 是一个用于构建 **LLM Agent（智能体）** 的 Go 库，并附带几个命令行工具。
它实现了经典的 **ReAct** 循环：调用模型，执行模型请求的工具，把结果交回模型，
如此反复，直到模型完成任务。

```
              ┌───────────────────────────────────────────────┐
              ▼                                               │
  任务  ──▶  模型  ──▶  需要调用工具？ ── 是 ──▶ 执行工具 ──▶ 结果
                            │
                            否
                            ▼
                        最终回答
```

你只需要提供三样东西：

1. **模型客户端**：Anthropic，或任意 OpenAI 兼容接口（OpenAI、DeepSeek、vLLM、
   Ollama、公司内部网关等）。
2. **工具**：shell、文件编辑、grep、git、HTTP、MCP 服务器，或你自己写的函数。
3. **任务**，以及可选的**预算**（费用、时间、步数）。

这个项目面向希望看清并掌控循环中每个环节的开发者。所有代码都是只依赖标准库的
普通 Go 代码，所有依赖都通过注入传入，所以在测试中，模型、工具、文件系统都可以
替换成假实现。

<a id="zh-features"></a>

## 功能特性

| | |
|---|---|
| **两种模型接口，统一抽象** | 支持 Anthropic Messages API 和任意 OpenAI 兼容的 `/chat/completions` 接口，包括需要代理和自定义请求头的网关 |
| **流式事件** | 一次运行就是一个带类型的事件迭代器（`TextDelta`、`ReasoningDelta`、`ToolStart` 等），可以直接渲染到终端、JSONL 或网页 |
| **12 个内置工具** | `bash`、`view_file`、`write_file`、`str_replace`、`grep_search`、`git_log`、`http_get` 等，还可以接入 MCP 工具和自定义工具 |
| **预算控制** | 限制费用（美元）、运行时长、步数、工具调用次数和子 Agent 嵌套深度，超出任何一项都会干净地终止运行 |
| **安全** | 逐次调用的权限策略（支持人工审批），子进程环境变量过滤，可选的操作系统级沙箱（`sandbox-exec` / `bwrap`） |
| **长任务** | 按 token 计算的上下文裁剪、自动摘要压缩（compaction），以及上下文超长报错后的自动恢复 |
| **Skills 与子 Agent** | 只在加载某个 skill 后才出现的工具；与父 Agent 共享预算的子 Agent |
| **可观测性** | 兼容 OpenTelemetry 结构的链路追踪和 HTML 查看器、Prometheus 指标、可重放的录制（tape） |
| **HTTP 服务** | 基于 Server-Sent Events、支持断线续传的多轮会话，并自带网页界面 |
| **评测** | 一套编程任务基准测试，用 pass@k 计分，并带有防作弊校验 |

<a id="zh-requirements"></a>

## 环境要求

| | |
|---|---|
| **Go** | 1.25.6 或更高版本 |
| **大模型 API** | Anthropic API Key，**或**任意 OpenAI 兼容接口。本地运行的 Ollama / vLLM 不需要 Key |
| **Python 3** | 仅 `scripts/wombat`（终端渲染脚本）需要 |
| *可选* | `sandbox-exec`（macOS 自带）或 `bwrap`（Linux），用于 `-confine` 沙箱 |

<a id="zh-quickstart-cli"></a>

## 快速开始：命令行

**1. 克隆并配置**

```bash
git clone https://github.com/hoobean1996/wombat-go.git
cd wombat-go
cp .env.example .env
```

打开 `.env`，填写**一种**模型服务即可。`.env` 已加入 `.gitignore`，Key 只会留在本地。

```bash
# 方式 A：OpenAI，或任意 OpenAI 兼容接口（DeepSeek、公司网关等）
export WOMBAT_PROVIDER="openai"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
export WOMBAT_MODEL="gpt-4o-mini"
```

```bash
# 方式 B：Anthropic
export WOMBAT_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
export WOMBAT_MODEL="claude-sonnet-5-5"
```

```bash
# 方式 C：用 Ollama 运行本地模型（不需要 Key）
export WOMBAT_PROVIDER="openai"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY=""            # 留空表示不发送认证头
export WOMBAT_MODEL="qwen2.5-coder"
```

**2. 向 Agent 提问**

```bash
scripts/wombat "这个仓库里有多少个 Go 文件？请用 bash 统计"
```

`scripts/wombat` 会把所有命令编译到 `bin/`，加载 `.env`，执行一轮 Agent 对话，并把
结果渲染成易读的文本。你会看到回答逐字输出，每次工具调用也会实时显示。加上 `-r`
还能看到模型的推理过程。

> [!WARNING]
> 默认情况下，Agent 在当前目录拥有**不受限制的 shell 权限**。除了简单试用之外，
> 建议加上 `-permission workspace` 或 `-confine`，或两者都加。详见[安全](#zh-safety)。

**3. 多轮对话**

```bash
scripts/wombat -session /tmp/s.json "记住数字 8317"
scripts/wombat -session /tmp/s.json "它乘以 3 是多少？"
```

每轮结束后，完整对话都会保存到 session 文件。如果上一轮 Agent 是以向你提问结束的，
你下一次输入的内容会被当作对这个问题的回答。

**4. 或者在浏览器里聊天**

```bash
make serve        # 然后打开 http://localhost:8080
```

这会启动一个支持实时流式输出的网页界面。默认情况下，shell 命令以及工作目录之外的
操作，会在页面上弹出 **Allow / Deny（允许 / 拒绝）** 卡片，等你确认。

<a id="zh-quickstart-lib"></a>

## 快速开始：作为 Go 库使用

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
	// 1. 模型客户端。ConfigFromEnv 读取 ANTHROPIC_API_KEY、ANTHROPIC_MODEL、
	//    ANTHROPIC_BASE_URL 等环境变量。OpenAI 兼容接口改用
	//    openai.New(openai.ConfigFromEnv())。
	leaf, err := anthropic.New(anthropic.ConfigFromEnv())
	if err != nil {
		log.Fatal(err)
	}
	// TrackCost 把每次响应的费用计入本次运行的预算。
	// 不加它，CostUSD 限额永远不会触发。
	client := llm.Chain(leaf, wombat.TrackCost(llm.DefaultPricing))

	// 2. 创建 Agent：模型客户端 + 可用的工具。
	agent, err := wombat.New(
		wombat.WithClient(client),
		wombat.WithModel("claude-sonnet-5-5"),
		wombat.WithTools(builtin.Default(builtin.Deps{})...),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 3. 设置预算：花费达到 1 美元或运行满 10 分钟，以先到者为准，运行就会被取消。
	ctx, cancel := governor.WithBudget(context.Background(),
		governor.Limits{CostUSD: 1.00, Wall: 10 * time.Minute})
	defer cancel()

	// 4. 启动运行，并消费事件流。
	run := agent.Start(ctx, wombat.Ask("这个目录里有多少个 Go 文件？"))
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

**运行是怎么结束的？** 循环结束后，用 `run.Outcome()` 判断：

```go
switch run.Outcome().(type) {
case wombat.Answer:    // 模型正常给出了回答
case wombat.Paused:    // 模型通过 ask_user 向用户提问；回答后可以继续
case wombat.Submitted: // 模型调用了终止型工具，工具参数就是结果
}
```

**多轮对话：** 把上一次的对话记录传给下一次运行。

```go
run = agent.Start(ctx, wombat.Then(run.Messages(), "它乘以 3 是多少？"))
```

如果上一轮以 `Paused` 结束，改用 `wombat.AnswerPause` 或 `wombat.Resume`。

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

agent, err := wombat.New(
	wombat.WithClient(client),
	wombat.WithTools(append(builtin.Default(builtin.Deps{}), weather)...),
)
```

一定要设置 `Caps`。没有声明能力的工具会被当作“会写入、会联网”的工具处理：
它永远不会并行执行，权限检查也会对它从严处理。

<a id="zh-concepts"></a>

## 核心概念

| 概念 | 说明 |
|---|---|
| `Agent` | 用 `wombat.New(options...)` 创建一次即可。创建后不可变，可以在多个 goroutine 间共享 |
| `Run` | Agent 针对某个输入的一次执行。用 `Next()` / `Event()` 遍历事件，最后检查 `Err()` 和 `Outcome()` |
| `Event` | 可以展示给用户的事件：`TextDelta`、`ReasoningDelta`、`ToolStart`、`Spend`、`ContextTrimmed` 等 |
| `Outcome` | 成功运行的结束方式：`Answer`、`Paused` 或 `Submitted`。失败通过 `run.Err()` 返回，用 `errors.Is` 判断类型 |
| `tool.Def` | 工具定义：名称、描述、JSON Schema、能力声明（`Caps`），以及 `func(ctx, args) (string, error)` |
| Middleware（中间件） | 包裹模型客户端（`llm.Chain`）或工具的一层逻辑：重试、日志、计费、权限、超时等 |
| Budget（预算） | `governor.WithBudget(ctx, limits)` 返回一个 `context.Context`，超出限额时自动取消 |
| Strategy（策略） | 决定长对话中哪些内容发给模型（`TokenWindow`、`DropTagged` 等） |

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

可选的额外工具：`view_media`（让模型查看图片和 PDF，命令行参数 `-media`）、
`delegate`（子 Agent，`-delegate`）、`load_skill` / `unload_skill`（`-skills-dir`），
以及以 `<server>__<tool>` 命名的 MCP 工具（`-mcp`）。

<a id="zh-config"></a>

## 配置

命令行工具从环境变量读取配置，`scripts/wombat` 和 `make` 会从 `.env` 加载这些变量。
在库代码中，`openai.ConfigFromEnv()` 和 `anthropic.ConfigFromEnv()` 读取同样的变量，
返回一个普通的 `Config` 结构体，你可以先修改其中的字段，再传给 `New`。除了在未提供
API Key 时回退读取环境变量之外，库本身不会偷偷读取任何环境变量。

| 变量 | 说明 |
|---|---|
| `WOMBAT_PROVIDER` | `openai` 或 `anthropic`。建议显式设置，因为各命令行工具的默认值不同 |
| `WOMBAT_MODEL` | 当前所选服务的默认模型 |
| **OpenAI 兼容接口** | |
| `OPENAI_BASE_URL` | 域名加 `/v1`，例如 `https://api.openai.com/v1`，程序会自动拼接 `/chat/completions` |
| `OPENAI_API_KEY` | 以 `Authorization: Bearer …` 发送。不需要认证的本地服务请留空 |
| `OPENAI_MODEL` | 优先级高于 `WOMBAT_MODEL` |
| `OPENAI_PROXY` | HTTP 代理，格式为 `host:port` 或完整 URL |
| `OPENAI_EXTRA_HEADERS` | 额外请求头，每行一个 `Name: value` |
| `OPENAI_STREAM` | `auto`（默认）、`always` 或 `never` |
| `OPENAI_TEMPERATURE`、`OPENAI_TOP_P` | 采样参数 |
| **Anthropic** | |
| `ANTHROPIC_API_KEY` | 必填 |
| `ANTHROPIC_MODEL` | 优先级高于 `WOMBAT_MODEL`。没有内置默认模型 |
| `ANTHROPIC_BASE_URL` | 默认 `https://api.anthropic.com`（未设置时回退到 `AGENT_LLM_BASE_URL`） |
| `ANTHROPIC_PROXY` | HTTP 代理（未设置时回退到 `AGENT_LLM_PROXY`） |
| `ANTHROPIC_CUSTOM_HEADERS` | 额外请求头，每行一个 `Name: value` |
| `ANTHROPIC_BETA` | 逗号分隔的 beta 功能标记 |
| `ANTHROPIC_STREAM` | `auto`（默认）、`always` 或 `never` |
| `ANTHROPIC_TEMPERATURE`、`ANTHROPIC_TOP_P` | 采样参数 |
| **脚本** | |
| `WOMBAT_ENV` | `scripts/wombat` 加载的环境文件（默认 `./.env`）。使用 `make` 时传 `ENV=路径` |

每个变量的详细说明见 `.env.example`。

<a id="zh-cli"></a>

## 命令行工具

执行 `make bin` 会把所有命令编译到 `bin/` 目录。

| 命令 | 用途 |
|---|---|
| `wombat-jsonl` | 执行一轮 Agent 对话，向 stdout 输出 JSON Lines 事件流。`scripts/wombat` 在它外面加了一层渲染 |
| `wombat-serve` | HTTP + SSE 服务，自带网页界面 |
| `wombat-bench` | 运行基准测试，输出 pass@k |
| `wombat-judge` | 对 `wombat-bench` 产出的结果打分 |
| `wombat-trace-html` | 把追踪文件渲染成独立的 HTML 报告 |
| `wombat-tsgen` | 为事件流和 HTTP API 生成 TypeScript 类型 |

**`wombat-jsonl` / `scripts/wombat` 常用参数**

| 类别 | 参数 |
|---|---|
| 模型 | `-provider openai\|anthropic`、`-model ID`、`-system "提示词"`、`-max-tokens N` |
| 工作区 | `-working-dir DIR`（shell 和 git 工具的工作目录）、`-read-only`（隐藏会写入或执行命令的工具） |
| 安全 | `-permission off\|readonly\|workspace\|ask`、`-confine`、`-confine-network loopback\|none\|full` |
| 限额 | `-budget 美元`、`-wall 10m`、`-max-iters N`（默认 30） |
| 会话 | `-session FILE`（多轮对话）、`-resume FILE`（基于已有对话记录继续，只读不写） |
| 长任务 | `-context-window TOKENS`、`-compact-at 0.85`、`-check 'go test ./...'`、`-check-attempts N` |
| 扩展 | `-skills-dir skills`、`-delegate`、`-mcp mcp.json`、`-media` |
| 记录 | `-tape FILE`、`-tape-mode auto\|record\|replay`、`-trace FILE`、`-trace-html FILE`、`-metrics-file FILE` |
| 日志 | `-log debug\|info\|warn\|error`、`-v`，以及 `-r`（显示推理过程，仅 `scripts/wombat`） |

执行 `bin/wombat-jsonl -h` 查看全部参数，包括规划、验收、代码审查等工作流参数
（`-plan-only`、`-accept`、`-review` 等）。

**`wombat-serve` 常用参数：** `-addr :8080`、`-working-dir DIR`、
`-permission workspace`（默认）、`-budget 1.0`（每轮的美元上限）、`-max-iters 12`、
`-sessions-dir DIR`（重启后保留会话）、`-cors`、`-confine`、`-mcp`。

**Make 命令**

| 命令 | 作用 |
|---|---|
| `make` | gofmt 检查、`go vet`、`go build`（默认目标） |
| `make test` | `go test -race ./...` |
| `make bin` | 编译所有命令到 `bin/` |
| `make serve` | 加载 `.env` 并启动 `wombat-serve`，额外参数用 `FLAGS="..."` 传入 |
| `make bench` | 运行基准测试，例如 `make bench N=8 C=4 TASKS=todo-app,fix-bug` |
| `make models` | 列出 OpenAI 兼容接口提供的模型 ID |
| `make ts` | 重新生成 `web/events.ts` |
| `make help` | 列出有说明的 make 目标 |

<a id="zh-guide"></a>

## 功能指南

下面每一节都是简要说明。各个机制为什么这样设计，详见
[docs/DESIGN.md](docs/DESIGN.md)（英文）。

<a id="zh-safety"></a>

### 安全

三层相互独立的防护：

| 层 | 作用 | 启用方式 |
|---|---|---|
| **权限策略** | 根据实际的命令、路径和参数，对每次工具调用做出允许、拒绝或请人审批的决定 | `-permission …`（命令行），`permission.Gate`（库） |
| **环境变量过滤** | Agent 执行的命令只继承白名单内的环境变量，看不到你的 API Key | 默认开启 |
| **操作系统沙箱** | 命令在 `sandbox-exec`（macOS）或 `bwrap`（Linux）中运行：只能写工作区和临时目录，凭据目录不可读，默认只能访问本机回环网络 | `-confine` |

权限级别：

| 级别 | 行为 |
|---|---|
| `off` | 不做检查（`wombat-jsonl` 的默认值） |
| `readonly` | 只允许只读工具 |
| `workspace` | 允许在工作目录内读写，也允许常见的无害命令（`go test`、`ls`、`cat` 等），其他操作需要审批 |
| `ask` | 每次调用都需要审批 |

`wombat-jsonl` 无法向你询问，所以“需要审批”就等于“拒绝”，并且会告诉模型被拒绝的
原因。如果想在浏览器里逐个审批，请使用 `wombat-serve`。在库代码中：

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

approver 传 `nil` 时，所有“需要审批”都会变成“拒绝”。

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

超出任何一项限额都会取消 context，所有进行中的操作（HTTP 请求、子进程）随之停止。
`context.Cause(ctx)` 会告诉你是哪一项超限。费用按 `llm.DefaultPricing` 计算。不在价格表
中的模型，费用记为 `0`，这类模型请改用 `Wall`、`Steps` 来限制，或者通过
`wombat.WithPricing` 传入自己的价格。命令行参数：`-budget`、`-wall`、`-max-iters`。

### 长对话

```go
wombat.WithContextWindow(200_000),
wombat.WithStrategy(wombat.TokenWindow(0.7, 0.5)),                  // 达到 70% 时裁剪到约 50%
wombat.WithCompaction(160_000, 50_000, wombat.Summarize(client, "")), // 把旧对话压缩成摘要
```

裁剪时先缩短旧的工具结果，不够时才整段丢弃旧的对话轮次。压缩（compaction）则是把
最早的一部分对话替换成一段摘要。命令行参数：`-context-window 200000 -compact-at 0.85`。

### 完成检查与中途引导

```go
wombat.WithCompletionCheck(3, wombat.CommandCheck(runner, "go test ./...", 0))
```

模型宣布完成时会运行这个检查。检查失败时，模型会拿到检查输出并继续工作，最多重试
3 次。命令行参数：`-check 'go test ./...'`。

想在运行过程中调整方向，调用 `run.Steer("改用 v2 API")` 即可。通过 HTTP 则是
`POST /api/sessions/{id}/steer`。

### Skills（技能）

一个 skill 是一个包含 `SKILL.md` 的文件夹。文件头部的 front matter 写名称和描述，
正文是操作说明，模型只在需要时才加载正文。

```markdown
---
name: git-history
description: Reading a repository's history. Load before any question about the past.
---

# Reading git history
...
```

```bash
scripts/wombat -skills-dir skills "run.go 最近是谁改的？为什么改？"
```

模型会看到一个 skill 目录，需要时调用 `load_skill` 加载。skill 还可以“解锁”某些
工具，在加载之前这些工具对模型是隐藏的：

```go
skills, _ := skill.LoadDir("./skills", func(err error) { log.Print(err) })
reg := skill.New(skills...)
reg.Gate("git-history", "git_log") // 加载 git-history 之后才会出现 git_log
g := reg.Bind(builtin.Default(deps))

agent, _ := wombat.New(wombat.WithClient(client), g.Install())
```

仓库的 `skills/` 目录里有 4 个示例 skill。

### 子 Agent

`delegate` 工具会运行一个完整的子 Agent。预算、权限检查和链路追踪都放在 `context`
里传递，所以子 Agent 自动与父 Agent 共享它们。

```go
child, _ := wombat.New(wombat.WithClient(client), wombat.WithName("researcher"),
	wombat.WithTools(tool.Filter(all, tool.OnlyCaps(tool.CapReadOnly))...))

parent, _ := wombat.New(wombat.WithClient(client),
	wombat.WithTools(append(all, wombat.DelegateTool(child))...),
	wombat.WithToolTimeoutFallback(0), // 子 Agent 由它自己的迭代次数上限约束
)
```

命令行参数：`-delegate` 会提供一个只读的 researcher 子 Agent。

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

```bash
scripts/wombat -mcp mcp.json "列出我还没合并的 pull request"
```

MCP 工具以 `<server>__<tool>` 命名。服务器给出的提示（`readOnlyHint` 等）会转换成
能力声明，所以权限策略对它们的处理与内置工具完全一致。在代码中使用
`mcp.LoadConfig`、`mcp.ConnectAll` 和 `conns.Tools()`。

### 录制与重放

```go
tp, _ := tape.Open("run.jsonl", tape.Auto)
defer tp.Close()
agent, _ := wombat.New(wombat.WithClient(client), tp.Install())
```

模型响应和工具结果都会被录制下来，以请求内容的哈希作为索引。重放速度快、结果确定，
而且完全不发网络请求。命令行参数：`-tape run.jsonl -tape-mode replay`。

### 链路追踪与指标

```bash
scripts/wombat -trace run.ndjson -trace-html run.html "..."
bin/wombat-trace-html run.ndjson        # 也可以之后再渲染已有的追踪文件
```

追踪数据使用 OpenTelemetry 的 `gen_ai.*` 属性名，子 Agent 会自动嵌套显示，并且不会
记录消息内容。`wombat-serve` 在 `GET /metrics` 提供 Prometheus 指标，
`wombat-jsonl -metrics-file` 则在退出时写出一份指标快照。

### HTTP API

`wombat-serve`（或者 `httpapi` 包，它是一个普通的 `http.Handler`）提供多轮会话：

| 方法与路径 | 用途 |
|---|---|
| `POST /api/sessions` | 创建会话并开始第一轮：`{"prompt": "..."}` |
| `GET /api/sessions/{id}/events` | SSE 事件流，断线后带 `Last-Event-ID` 重连即可续传 |
| `POST /api/sessions/{id}/messages` | 发送下一轮消息：`{"prompt": "..."}` |
| `POST /api/sessions/{id}/steer` | 给正在运行的这一轮发送引导消息 |
| `GET /api/sessions/{id}/approvals` | 列出等待人工审批的调用 |
| `POST /api/sessions/{id}/approvals/{use_id}` | 审批一次调用：`{"allow": true}` |
| `DELETE /api/sessions/{id}` | 取消并删除会话 |

```bash
curl -s localhost:8080/api/sessions -H 'Content-Type: application/json' \
     -d '{"prompt": "这个仓库是做什么的？"}'          # 返回 {"id": "...", ...}
curl -N localhost:8080/api/sessions/<id>/events
```

完整接口列表（健康检查、配置、会话列表、对话记录）见 `httpapi` 包的文档。所有事件和
请求的 TypeScript 类型在 `web/events.ts` 中，由 Go 类型自动生成。

### 基准测试与评测

```bash
make bench TASKS=easy N=4                 # 或：bin/wombat-bench -tasks hard -n 4
bin/wombat-judge -in runs/episodes.jsonl  # 之后再为结果打分
```

`benchmarks/` 包含 8 个编程任务，分为两档。简单档应该始终拿满分，拿不到说明框架本身
有问题；困难档才用来区分不同 Agent 的能力。结果用 pass@k 表示。`eval/` 用来衡量框架
的各个机制本身是否有效，详见 [eval/README.md](eval/README.md)（英文）。

<a id="zh-layout"></a>

## 目录结构

```
.                 Agent、Run、Event、Outcome、Convo、Strategy   ← 从这里开始看
llm/              与具体服务无关的类型、Client 接口、中间件、价格表
llm/anthropic/    Anthropic Messages API 客户端
llm/openai/       OpenAI 兼容客户端
tool/             工具定义、调度、中间件（重试、超时、熔断等）
tool/builtin/     12 个内置工具、view_media、普通执行器和沙箱执行器
permission/       逐次调用的权限控制，支持人工审批
governor/         以 context 取消实现的预算控制
skill/            Skills：按需加载的说明和工具
skills/           示例 skill
mcp/              Model Context Protocol 客户端
tape/             录制与重放
trace/            链路追踪、NDJSON 输出、HTML 报告
metric/           Prometheus 指标
httpapi/          基于 SSE 的多轮会话
accept/           确定性的“是否真正完成”检查（build、vet、test、TODO.md）
review/           对抗式代码审查，以失败的测试用例报告缺陷
rl/               episode、校验器、pass@k
benchmarks/       8 个编程任务
eval/             框架自身的评测
cmd/              命令行工具
scripts/          wombat（运行并渲染）、pretty.py（渲染器）
web/              自动生成的 TypeScript 事件类型
docs/             设计文档
```

<a id="zh-dev"></a>

## 开发

```bash
make              # gofmt 检查 + go vet + go build
make test         # go test -race ./...
make ts           # 修改事件类型后重新生成 web/events.ts
```

测试使用脚本化的假模型客户端，离线运行，不需要 API Key。唯一需要安装的是 Go 工具链，
没有任何第三方模块需要下载。

<a id="zh-faq"></a>

## 常见问题

| 现象 | 解决办法 |
|---|---|
| `scripts/wombat: no env file` | 执行 `cp .env.example .env` 并填写，或者用 `WOMBAT_ENV` 指向你的环境文件 |
| 报错 `no API key` | 在 `.env` 中设置 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`，并确认 `WOMBAT_PROVIDER` 与之对应 |
| `no model on the request and no client default` | 设置 `WOMBAT_MODEL`（或 `ANTHROPIC_MODEL`），或者传 `-model` |
| 不知道接口支持哪些模型 ID | 执行 `make models` |
| token 数和费用一直是 0，预算不生效 | 有些网关在流式模式下不返回用量数据，设置 `OPENAI_STREAM="never"`（或 `ANTHROPIC_STREAM`）。同时确认模型在 `llm/pricing.go` 的价格表里 |
| 使用推理模型时输出像卡住了 | 模型正在思考，用 `scripts/wombat -r` 查看推理过程 |
| 加了 `-media` 后返回 HTTP 400 | 该模型不支持图片输入，去掉 `-media` |
| 所有 shell 命令都被拒绝 | 你在 `wombat-jsonl` 中用了 `-permission workspace` 或 `ask`，而它无法向你询问。改用 `-permission off` 配合 `-confine`，或者使用 `wombat-serve` |
| 提示词里有引号或 `&&`，用 `make` 执行时出错 | 改用 `scripts/wombat "..."`，不要通过 make 目标传提示词 |

<a id="zh-design"></a>

## 设计说明

wombat-go 是一个私有 OCaml 5 项目的 Go 重写版。原项目基于**代数效应（algebraic
effects）**：每一种副作用（LLM 调用、工具、文件 I/O、时间、用户输入）都是一个效应，
由外层的 handler 决定实际怎么执行。到了 Go 里，这些效应大多变成了普通的
**依赖注入**。整个 API 由三个决定塑造：

1. **工具只接收 context 和参数，别的什么都不要。** 工具需要的东西（文件系统、命令
   执行器、HTTP 客户端）在构造时传入，所以工具就是一个普通函数，可以直接测试。
2. **一次运行就是一条事件流**，而不是一个返回值加一堆回调。终端、JSONL 管道和网页
   界面消费的是同一条事件流。
3. **预算就是一个 `context.Context`。** 超出限额就取消它，进程中所有阻塞调用本来就
   知道如何响应取消。

完整的设计文档涵盖上下文结构、skills、容错、链路追踪、权限、评测，以及与 OCaml 原版
相比的得与失：**[docs/DESIGN.md](docs/DESIGN.md)**（英文）。
