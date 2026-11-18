import { useCallback, useEffect, useState } from 'react'
import * as api from '../lib/api'
import { oneLine, relTime, sessionLabel } from '../lib/format'
import { useNow } from '../lib/hooks'
import { setQuery, useLocation } from '../lib/router'
import { useApp } from '../lib/store'
import { toast } from '../lib/toast'
import type { CallState, ToolCall } from '../types'
import { Icon, Link } from '../components/ui'

const STATE_LABEL: Record<CallState, string> = {
  pending: '未开始', running: '运行中', succeeded: '成功', failed: '失败', denied: '被拒绝',
  deferred: '待确认', waiting: '等结果', interrupted: '被中断', cancelled: '未执行',
}

/** The filters on offer, and the states each one selects. */
const FILTERS: { key: string; label: string; states?: CallState[] }[] = [
  { key: 'all', label: '全部' },
  { key: 'open', label: '未结束', states: ['pending', 'running', 'deferred', 'waiting', 'interrupted'] },
  { key: 'failed', label: '失败', states: ['failed', 'interrupted'] },
  { key: 'succeeded', label: '成功', states: ['succeeded'] },
]

function CallOutput({ c }: { c: ToolCall }) {
  const text = c.output ?? ''
  if (!text) return <span className="muted">—</span>
  const first = text.split('\n', 1)[0]
  if (first.length === text.length && text.length <= 120) return <span className={c.is_error ? 'bad' : ''}>{text}</span>
  return (
    <details>
      <summary className={c.is_error ? 'bad' : ''}>{oneLine(first, 100)}</summary>
      <pre>{text}</pre>
    </details>
  )
}

export function ToolCallsPage() {
  const { query } = useLocation()
  const session = query.get('session') ?? ''
  const filter = FILTERS.find(f => f.key === query.get('state')) ?? FILTERS[0]
  const sessions = useApp(s => s.sessions)
  const [calls, setCalls] = useState<ToolCall[] | null>(null)
  const now = useNow(5_000)

  const load = useCallback(() => {
    api.listToolCalls({ session: session || undefined, state: filter.states?.join(','), limit: 200 })
      .then(setCalls, e => toast(api.errorText(e)))
  }, [session, filter])
  useEffect(() => {
    load()
    const t = setInterval(() => { if (!document.hidden) load() }, 5000)
    return () => clearInterval(t)
  }, [load])

  const name = (id: string) => { const s = sessions.find(x => x.id === id); return s ? sessionLabel(s) : id.slice(0, 8) }

  return (
    <div className="main">
      <div className="topbar"><div className="ttl">工具调用</div></div>
      <div className="page">
        <p className="page-intro">
          每个会话的每一次工具调用，从模型提出到拿到结果，都先记下来再执行。wombatd 中途停了，接手的 wombatd 从记下的地方接着跑：
          已经完成的调用不会再执行；正在运行、不知道有没有生效的调用，能安全重跑的会重跑，其余的告诉模型“可能已经生效”（或按 <code>tools.recovery: ask</code> 让人决定）。
        </p>
        <div className="filters">
          <div className="seg">
            {FILTERS.map(f => (
              <button key={f.key} className={filter.key === f.key ? 'on' : ''} onClick={() => setQuery('state', f.key === 'all' ? null : f.key, { replace: true })}>{f.label}</button>
            ))}
          </div>
          {session && <span className="chip">会话：{name(session)}<button onClick={() => setQuery('session', null, { replace: true })} title="清除"><Icon name="x" cls="sm" /></button></span>}
          <button className="ibtn" title="刷新" onClick={load}><Icon name="refresh" /></button>
        </div>
        {calls === null ? <div className="muted">正在加载……</div> : calls.length === 0 ? <div className="muted">没有调用记录</div> : (
          <table className="tbl">
            <thead><tr><th>调用</th><th>时间</th><th>会话</th><th>轮</th><th>工具</th><th>状态</th><th>结果</th></tr></thead>
            <tbody>
              {calls.map(c => (
                <tr key={`${c.session_id}/${c.step}.${c.index}`}>
                  <td className="mono" title={`tool_use ${c.use_id}`}>{c.step}.{c.index}{c.parent && <div className="sub" title={`由调用 ${c.parent} 的子 agent 发起`}>子 agent #{c.run}</div>}</td>
                  <td title={c.created}>{relTime(Date.parse(c.created), now)}</td>
                  <td>{session ? name(c.session_id) : <Link to={`/calls?session=${c.session_id}`}>{name(c.session_id)}</Link>}</td>
                  <td>{c.turn}</td>
                  <td className="cell-text" title={c.input === undefined ? '' : JSON.stringify(c.input)}>
                    <b>{c.tool}</b>{c.input !== undefined && <div className="sub">{oneLine(c.input, 80)}</div>}
                  </td>
                  <td>
                    <span className={`pill s-${c.state}`}>{STATE_LABEL[c.state] ?? c.state}</span>
                    {c.attempts > 1 && <div className="sub">执行了 {c.attempts} 次</div>}
                    {c.decision && <div className="sub">{c.decision.allow ? '允许' : '拒绝'}{c.decision.by.from ? `：${c.decision.by.from}` : ''}</div>}
                    {c.reason && c.state !== 'succeeded' && <div className="sub" title={c.reason}>{oneLine(c.reason, 60)}</div>}
                    {c.handle && <div className="sub">等：{c.handle}</div>}
                  </td>
                  <td className="calls-out"><CallOutput c={c} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
