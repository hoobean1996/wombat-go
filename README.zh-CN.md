<div align="center">

<img src="ui/public/favicon.svg" width="88" alt="">

# Wombat

**常驻的 AI Agent，一个 Go 二进制就能跑起来。**

大多数 Agent 只在你盯着它的时候干活。Wombat 的 Agent 平时在睡，有事发生时会被唤醒，比如一个 webhook、一条聊天消息，或者它自己定的计时器。它把活干完、把结果告诉你，然后接着睡。重启也不会丢东西。

[English](README.md) · **简体中文**

![Go](https://img.shields.io/badge/Go-1.25%2B-00ADD8?logo=go&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-pgx%20only-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[为什么是 Wombat](#zh-why) · [工作原理](#zh-how) · [快速开始](#zh-start) · [能做什么](#zh-build) · [Go 库](#zh-lib) · [文档](#zh-docs)

</div>

---

<a id="zh-why"></a>

## 为什么是 Wombat

聊天助手在你关掉页面时就停了，真正有用的 Agent 会继续干活：处理夜里新来的 issue，确认昨天的修复有没有真的上线，早上把日报写好；碰到有风险的操作，先来问你。

Anthropic 在 [Building always-on agents](https://claude.com/resources/articles/building-always-on-agents) 一文中指出，这种 Agent 跑的还是**同一个 Agent 循环**。变的是循环外面的那套基础设施：什么唤醒它，两次运行之间留下什么，结果怎么送到你手上。Wombat 把这一套做好了，开箱即用：

| | |
|---|---|
| 📥 **事件收件箱** | webhook、告警、聊天消息、计时器，统统进同一个持久化的 inbox。同一个 `dedup_key` 重复投递的事件会被丢弃，同一个 `thread` 的事件总是进同一个会话 |
| ⏰ **计时器** | Agent 自己安排跟进（"一小时后再看一眼"）和周期任务（"每天早上"） |
| 🛡️ **审批** | 有风险的工具调用会等人来决定，但不会让这一轮一直挂着，等多久都行。信得过当前任务，就把会话切到 **Full Access**，什么都不用再确认。命令在清理过的环境变量中运行，还可以关进操作系统沙箱 |
| 📤 **发件箱** | 回复、提问、待审批的请求都进 outbox。你的程序可以长轮询它，也可以让 wombatd 把每条消息 POST 到你的 webhook |
| 💾 **持久化** | 试用时状态放内存，正式用放 Postgres。一轮跑到一半杀掉 wombatd，下一个实例会接着做完 |
| 🧩 **可横向扩展** | 一个 Postgres 上可以跑任意多个 wombatd。每个会话（每个 inbox）同一时间只由其中一个运行，任何一个都能接任何请求，挂掉的实例的会话会被其他实例接管 |
| 🗣️ **谁说了什么** | 每个事件都记录来源（`source`）和发送者（`from`），模型分得清哪些是 webhook 的内容、哪些是你的指令，也分得清不同的人 |
| 💸 **预算** | 每一轮都有花费、时间和步数的上限 |
| 🖥️ **网页界面** | 内置在二进制里：可以聊天、审批调用，在对话右侧直接打开 Agent 写出的文件（HTML、Markdown、图片、PDF、CSV、代码），还能查看 inbox、outbox 和计时器 |

它是纯 Go 写的：框架本身只用标准库，唯一的依赖是 Postgres 存储用到的 `pgx`。模型、工具和文件系统都是接口，可以随意替换。

<a id="zh-how"></a>

## 工作原理

```mermaid
flowchart LR
    subgraph events [发生的事]
        chat[聊天消息]
        hook[webhook 或告警]
        timer[计时器]
        approve[你的审批]
    end
    events --> inbox[(Inbox)]
    inbox --> turn[每个会话一次跑一轮]
    turn --> agent[Agent 循环：<br/>模型 + 工具]
    agent -- set_timer --> timer
    agent --> outbox[(Outbox)]
    outbox --> you[网页界面、你的程序<br/>或 webhook]
```

每一轮都从一个事件开始。worker 把一个会话**所有**待处理的事件一次交给它，跑一轮 Agent 循环，再用一个事务记下结果。这一轮运行期间新到的事件，留到下一轮处理。两轮之间 Agent 在睡，一分钱都不花。

<a id="zh-start"></a>

## 快速开始

**你需要** Go 1.25+，以及一个能用的模型：Anthropic、OpenAI、DeepSeek、任何兼容 OpenAI 的接口，或者本地的 Ollama（不需要 key）。构建网页界面还需要 Node.js 和 npm；没有它们也能用完整的 API。

### 1. 克隆并配置

```bash
git clone https://github.com/hoobean1996/wombat-go.git
cd wombat-go
cp wombatd.example.yaml wombatd.yaml && chmod 600 wombatd.yaml
```

在 `wombatd.yaml` 里填好 `model` 一节。这个文件已经被 git 忽略，key 只留在你的机器上。

```yaml
model:
  provider: openai              # anthropic，或者 openai（任何兼容 OpenAI 的接口）
  base_url: https://api.deepseek.com/v1
  api_key: sk-...
  name: deepseek-flash
```

| 模型服务 | `provider` | `base_url` | `name` 示例 |
|---|---|---|---|
| Anthropic | `anthropic` | `""` | `claude-sonnet-5-5` |
| OpenAI | `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `openai` | `https://api.deepseek.com/v1` | `deepseek-flash` |
| Ollama（本地，不需要 key） | `openai` | `http://localhost:11434/v1` | `qwen2.5-coder` |

### 2. 启动

```bash
make serve
```

### 3. 打开网页界面

访问 **http://127.0.0.1:8080**，描述一个任务，看它干活：回复逐字出现，每个工具调用一发生就显示出来。凡是超出工作目录的操作，或者不明显无害的命令，都会弹出一张卡片，由你允许或拒绝。

### 4. 从外部唤醒它

聊天只是唤醒 Agent 的一种方式。让它"常驻"的，是那些没人盯着时照样会来的事件：

```bash
# 一个 webhook：有人在 issue 下评论了。同一个 thread 总是进同一个会话。
curl -s localhost:8080/api/events -H 'Content-Type: application/json' -d '{
  "thread": "github:you/repo#12", "kind": "issue_comment", "source": "github",
  "from": "octocat", "payload": {"body": "the build is red again"}}'

# Agent 要说的话。你的程序可以长轮询这里，也可以配置 outbox.webhook 让 wombatd 推给你。
curl -s 'localhost:8080/api/outbox?after=0&wait=30s'
```

Agent 也能自己唤醒自己。在聊天里试试：*"一小时后再看看构建，还是红的就告诉我。"* 它会设一个计时器，在**计时器**页面能看到。一小时后计时器往 inbox 里放一个事件，Agent 再通过 outbox 告诉你结果。

### 5. 持久化运行

```yaml
store:
  driver: postgres
  dsn: postgres://wombat:wombat@127.0.0.1:55432/wombat   # `make pg-up` 会用 podman 起一个
server:
  token: a-long-random-string                            # 监听 localhost 以外的地址之前必须设置
```

wombatd 第一次启动时会自己建表。之后重启，哪怕是在一轮的中途，也会从停下的地方继续。

> [!WARNING]
> 没有设置 `server.token` 时，任何能访问这个端口的人都能驱动 Agent、读到所有对话。wombatd 默认只监听本机回环地址；要对外开放，请先设置 token。

<a id="zh-build"></a>

## 能用它做什么

- **Issue 和 PR 分拣。** 把 GitHub webhook 指向 `/api/events`，并带上 `"thread": "github:owner/repo#N"`。每个 issue 都有自己的长期会话，记得完整的讨论过程。
- **值班助手。** 告警作为事件进来。Agent 先用只读工具排查，重启任何东西之前都会先问你。
- **每日摘要。** 一个周期计时器每天早上唤醒 Agent，它看完有哪些变化，再用 `send_message` 发一份摘要。
- **自动跟进。** 跟它说"今晚之前确认修复已经上线"，它会自己设计时器，到点回来检查。
- **聊天桥接。** 把 Slack 或 Discord 的消息作为事件转发进来，`from` 填说话的人，再把 outbox 发回去。一个话题一个会话，可以好几个人一起跟它聊。

<a id="zh-lib"></a>

## 作为 Go 库使用

Agent 循环本身就是一个可以直接调用的库：

```go
client, _ := anthropic.New(anthropic.ConfigFromEnv())
agent, _ := core.New(
	core.WithClient(llm.Chain(client, core.TrackCost(llm.DefaultPricing))),
	core.WithModel("claude-sonnet-5-5"),
	core.WithTools(builtin.Default(builtin.Deps{})...),
)

ctx, cancel := governor.WithBudget(context.Background(),
	governor.Limits{CostUSD: 1, Wall: 10 * time.Minute}) // 超出预算 = 取消
defer cancel()

run := agent.Start(ctx, core.Ask("How many Go files are in this directory?"))
defer run.Close()
for run.Next() { // 一次运行就是一串有类型的事件
	if ev, ok := run.Event().(core.TextDelta); ok {
		fmt.Print(ev.Text)
	}
}
```

常驻的那部分同样是库：`daemon.NewManager` 在一个存储（内存，或者 `postgres.Open`）之上运行会话，`httpapi.New` 把它变成一个普通的 `http.Handler`。所以整个 wombatd 都可以嵌进你自己的程序。自定义工具、运行结果的几种类型、多轮对话，以及 `wombat` 这个模块路径需要的 `replace` 指令，见[**使用指南**](docs/GUIDE.zh-CN.md#zh-quickstart-lib)。

<a id="zh-docs"></a>

## 文档

| | |
|---|---|
| **[使用指南](docs/GUIDE.zh-CN.md)** | Go 库、每一个配置项、HTTP API、权限、预算、Skills、子 Agent、MCP、链路追踪、常见问题 |
| **[设计说明](docs/DESIGN.md)**（英文） | 每个部分为什么这样设计，以及一路踩过的坑 |
| **[网页界面](ui/README.md)**（英文） | 前端的工作方式，以及如何开发 |
| **[评测](eval/README.md)**（英文） | 基准测试套件和 pass@k 评分 |

## 开发

```bash
make test       # go test -race ./...；离线运行，不需要 API key
make ui-dev     # 带热更新的网页界面，连接正在运行的 wombatd
make help       # 所有 make 目标
```

## 许可证

[MIT](LICENSE)
