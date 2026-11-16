import { useState, type ReactNode } from 'react'
import { looksLikeFile, useFileOpener } from '../lib/files'
import { fmtDuration, fmtTokens, oneLine } from '../lib/format'
import type { LiveTurn } from '../lib/session'
import type { Exchange, Step } from '../lib/transcript'
import type { Approval, InboxEvent, SessionInfo } from '../types'
import { Markdown } from './Markdown'
import { Icon, Logo, StateBadge } from './ui'

// ── tool calls ───────────────────────────────────────────────────────────

/** The one line that says what a call does: the command, the path, the URL. */
function summary(name: string, input: unknown): string {
  const i = (input && typeof input === 'object' ? input : {}) as Record<string, unknown>
  const s = (k: string) => (typeof i[k] === 'string' ? (i[k] as string) : '')
  switch (name) {
    case 'bash': return s('command')
    case 'view_file': case 'write_file': case 'str_replace': case 'view_media': return s('path')
    case 'grep_search': return `${s('pattern')}  ${s('path')}`.trim()
    case 'http_get': return s('url')
    case 'git_log': case 'git_show': return `${s('ref') || s('path')}`.trim() || s('cwd')
    case 'send_message': return s('text')
    case 'set_timer': return `${s('in') || s('at')}${s('every') ? ` · 每 ${s('every')}` : ''} · ${s('note')}`
    case 'calculator': return s('expression')
    case 'ask_user': return s('question')
    default: return input === undefined ? '' : oneLine(input)
  }
}

const STEP_ICON: Record<Step['status'], [string, string]> = {
  running: ['', ''], done: ['check', 'ok'], failed: ['alert', 'bad'], waiting: ['lock', 'warn'], denied: ['x', ''], pending: ['clock', ''],
}
const STEP_NOTE: Partial<Record<Step['status'], string>> = { waiting: '等待确认', denied: '已拒绝', pending: '未执行' }

/** The file a tool call wrote or read, when it names one the file panel can show. */
function fileOf(st: Step): string | null {
  if (!['write_file', 'str_replace', 'view_file', 'view_media'].includes(st.name)) return null
  const p = (st.input && typeof st.input === 'object' ? (st.input as Record<string, unknown>).path : undefined)
  return typeof p === 'string' && looksLikeFile(p) ? p : null
}

function StepRow({ st }: { st: Step }) {
  const [open, setOpen] = useState(false)
  const openFile = useFileOpener()
  const file = openFile && st.status !== 'denied' ? fileOf(st) : null
  const line = summary(st.name, st.input)
  const [icon, cls] = STEP_ICON[st.status]
  const detail = st.input !== undefined || st.output || st.error
  return (
    <div className="step-wrap" style={st.depth ? { paddingLeft: Math.min(st.depth, 4) * 20 } : undefined}>
      <div className={`step ${st.status === 'denied' ? 'denied' : ''} ${detail ? 'clickable' : ''}`} onClick={() => detail && setOpen(o => !o)}>
        {st.status === 'running' ? <span className="run-ind" /> : <Icon name={icon} cls={`sm ${cls}`} />}
        <span className="step-name">{st.name}</span>
        {line && <code title={line}>{line}</code>}
        <span className="m2">{STEP_NOTE[st.status] ?? (st.ms !== undefined ? fmtDuration(st.ms) : '')}</span>
        {file && openFile && (
          <button className="step-open" title="在右侧打开这个文件" onClick={e => { e.stopPropagation(); openFile(file) }}><Icon name="panel" cls="sm" /></button>
        )}
        {detail && <Icon name="chev" cls={`sm chev ${open ? 'open' : ''}`} />}
      </div>
      {open && (
        <div className="step-detail">
          {st.input !== undefined && <><div className="sd-h">输入</div><pre>{JSON.stringify(st.input, null, 2)}</pre></>}
          {st.error && <><div className="sd-h bad">错误</div><pre>{st.error}</pre></>}
          {st.output && <><div className="sd-h">输出</div><pre>{st.output}</pre></>}
        </div>
      )}
    </div>
  )
}

function Steps({ steps, thinking, open: initiallyOpen = false }: { steps: Step[]; thinking: number; open?: boolean }) {
  const [open, setOpen] = useState(initiallyOpen)
  if (!steps.length && !thinking) return null
  const failed = steps.filter(s => s.status === 'failed').length
  const title = steps.length
    ? `${steps.length} 个工具调用${failed ? ` · ${failed} 个失败` : ''}${thinking ? ` · 思考 ${thinking} 次` : ''}`
    : `思考了 ${thinking} 次`
  // While collapsed, a call that needs attention stays in view.
  const shown = open ? steps : steps.filter(s => s.status === 'running' || s.status === 'waiting')
  return (
    <div className={`steps ${open ? '' : 'collapsed'}`}>
      <div className="steps-h" onClick={() => setOpen(o => !o)}>
        <Icon name="terminal" cls="sm" />{title}<Icon name="chev" cls="sm chev" />
      </div>
      {shown.map(st => <StepRow key={st.id} st={st} />)}
    </div>
  )
}

// ── what a person sees ───────────────────────────────────────────────────

export function UserBubble({ text, queued = false }: { text: string; queued?: boolean }) {
  return <div className={`m-user ${queued ? 'queued' : ''}`}>{queued && <span className="q-tag">排队中 · 下一轮处理</span>}{text}</div>
}

function AiMeta({ model, children }: { model?: string; children?: ReactNode }) {
  return (
    <div className="ai-meta">
      <Logo /><b>Wombat</b>
      {model && <><span className="sepdot">·</span>{model}</>}
      {children}
    </div>
  )
}

/** A finished (or stopped) exchange from the transcript. */
export function AssistantBlock({ ex, model, status }: { ex: Exchange; model?: string; status?: ReactNode }) {
  return (
    <div className="m-ai">
      <AiMeta model={model}>{status && <><span className="sepdot">·</span>{status}</>}</AiMeta>
      <Steps steps={ex.steps} thinking={ex.thinking} />
      {ex.text && <Markdown text={ex.text.replace(/^\n+/, '')} />}
    </div>
  )
}

/** The turn in progress, drawn from the event stream in the order things happened. */
export function LiveBlock({ live, model }: { live: LiveTurn; model?: string }) {
  const last = live.parts.at(-1)
  // Nothing is streaming and no tool is running: the model is being asked.
  const waitingOnModel = !last || last.kind === 'note' || (last.kind === 'tool' && last.status !== 'running' && last.status !== 'waiting')
  return (
    <div className="m-ai">
      <AiMeta model={model}>
        <span className="sepdot">·</span><span className="st-run"><span className="run-ind" />正在运行{live.iter ? ` · 第 ${live.iter.n}/${live.iter.max} 步` : ''}</span>
      </AiMeta>
      {live.parts.map((p, i) => {
        switch (p.kind) {
          case 'text':
            return p.text.trim() ? <Markdown key={i} text={p.text.replace(/^\n+/, '')} cursor={p === last} /> : null
          case 'think':
            return <div key={i} className="think-line"><Icon name="sparkle" cls="sm" />{p === last ? '思考中…' : '思考'}</div>
          case 'tool':
            return <StepRow key={p.id} st={p} />
          case 'note':
            return <div key={i} className={`note-line ${p.bad ? 'bad' : ''}`}><Icon name={p.bad ? 'alert' : 'info'} cls="sm" />{p.text}</div>
        }
      })}
      {waitingOnModel && <p><span className="cursor" /></p>}
    </div>
  )
}

// ── what needs the person ────────────────────────────────────────────────

export function ApprovalCard({ ap, onDecide, busy }: { ap: Approval; onDecide: (allow: boolean) => void; busy: boolean }) {
  const line = summary(ap.tool, ap.input)
  return (
    <div className="approve">
      <div className="ap-h"><Icon name="alert" />需要你确认<span className="rule">{ap.tool}</span></div>
      {line && <div className="ap-cmd">{ap.tool === 'bash' && <span className="pr">$ </span>}{line}</div>}
      {ap.reason && <div className="ap-meta">{ap.reason}</div>}
      {!line && ap.input !== undefined && <pre className="ap-input">{JSON.stringify(ap.input, null, 2)}</pre>}
      <div className="ap-acts">
        <span className="hint">拒绝后，模型会读到“被拒绝”，可以在输入框里告诉它怎么改</span>
        <button className="btn sm" disabled={busy} onClick={() => onDecide(false)}>拒绝</button>
        <button className="btn sm dark" disabled={busy} onClick={() => onDecide(true)}>允许</button>
      </div>
    </div>
  )
}

/** The end of the thread: what the session is waiting for, or why it stopped. */
export function SessionStatus({ info }: { info: SessionInfo }) {
  if (info.state === 'waiting' && info.answer) {
    return (
      <div className="ask">
        <div className="ask-h"><Icon name="chat" />它在等你回答</div>
        <Markdown text={info.answer} />
        <div className="ask-desc">在下面的输入框里回复：下一条消息就是对这个问题的回答。</div>
      </div>
    )
  }
  if (info.state === 'failed') {
    return (
      <div className="fail-card">
        <div className="ask-h"><Icon name="alert" cls="bad" />这一轮出错了{info.error_kind && <span className="rule">{info.error_kind}</span>}</div>
        {info.error && <pre>{info.error}</pre>}
        <div className="ask-desc">可以直接再发一条消息，会话会接着之前的对话继续。</div>
      </div>
    )
  }
  return null
}

export function SpendLine({ info }: { info: SessionInfo }) {
  const s = info.spend
  const tokens = s.input_tokens + s.output_tokens + s.cache_read_tokens
  return (
    <span className="spend" title={`输入 ${s.input_tokens} · 输出 ${s.output_tokens} · 缓存命中 ${s.cache_read_tokens} tokens · ${s.calls} 次模型调用`}>
      <StateBadge state={info.state} />
      {tokens > 0 && <span>{fmtTokens(tokens)} tokens</span>}
      {s.cost_usd > 0 && <span>${s.cost_usd < 0.01 ? s.cost_usd.toFixed(4) : s.cost_usd.toFixed(2)}</span>}
    </span>
  )
}

export function QueuedList({ events }: { events: InboxEvent[] }) {
  return <>{events.map(e => <UserBubble key={e.id} text={String((e.payload as { text?: unknown } | undefined)?.text ?? '')} queued />)}</>
}
