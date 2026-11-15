import { useCallback, useEffect, useState } from 'react'
import * as api from '../lib/api'
import { oneLine, relTime, sessionLabel } from '../lib/format'
import { useNow } from '../lib/hooks'
import { setQuery, useLocation } from '../lib/router'
import { refreshSessions, useApp } from '../lib/store'
import { toast } from '../lib/toast'
import type { InboxEvent } from '../types'
import { Icon, Link, Modal } from '../components/ui'

const STATUSES = ['', 'pending', 'running', 'done', 'failed'] as const
const STATUS_LABEL: Record<string, string> = { '': '全部', pending: '等待中', running: '处理中', done: '已完成', failed: '失败' }

/** POST /api/events: an event from outside, the way a webhook would send it. */
function NewEventModal({ onClose, onSent }: { onClose: () => void; onSent: () => void }) {
  const sessions = useApp(s => s.sessions)
  const [target, setTarget] = useState<'session' | 'thread'>('thread')
  const [sessionId, setSessionId] = useState(sessions[0]?.id ?? '')
  const [thread, setThread] = useState('')
  const [kind, setKind] = useState('note')
  const [source, setSource] = useState('ui')
  const [dedup, setDedup] = useState('')
  const [payload, setPayload] = useState('{\n  "text": ""\n}')
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    let parsed: unknown
    try { parsed = payload.trim() ? JSON.parse(payload) : undefined } catch { return toast('payload 不是合法的 JSON') }
    setBusy(true)
    try {
      const r = await api.postEvent({
        kind: kind.trim(), source: source.trim() || undefined, dedup_key: dedup.trim() || undefined, payload: parsed,
        ...(target === 'session' ? { session_id: sessionId } : { thread: thread.trim() }),
      })
      toast(r.duplicate ? `重复的事件：dedup_key 已经收到过（事件 #${r.event.id}）` : `已放进 inbox：事件 #${r.event.id} → ${sessionLabel(r.session)}`)
      void refreshSessions()
      onSent()
      onClose()
    } catch (e) {
      toast(api.errorText(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <Modal title="发送事件" icon="download" onClose={onClose} footer={<>
      <button className="btn" onClick={onClose}>取消</button>
      <button className="btn dark" disabled={busy || !kind.trim() || (target === 'thread' ? !thread.trim() : !sessionId)} onClick={() => void submit()}>发送</button>
    </>}>
      <div className="form">
        <div className="form-row">
          <label>发给</label>
          <div className="seg">
            <button className={target === 'thread' ? 'on' : ''} onClick={() => setTarget('thread')}>线程</button>
            <button className={target === 'session' ? 'on' : ''} onClick={() => setTarget('session')}>指定会话</button>
          </div>
        </div>
        {target === 'thread' ? (
          <div className="form-row"><label>thread</label><input className="inp" value={thread} onChange={e => setThread(e.target.value)} placeholder="github:you/repo#12 —— 同一个线程的事件进同一个会话，没有就新建" /></div>
        ) : (
          <div className="form-row"><label>会话</label>
            <select className="inp" value={sessionId} onChange={e => setSessionId(e.target.value)}>
              {sessions.map(s => <option key={s.id} value={s.id}>{sessionLabel(s)}</option>)}
            </select>
          </div>
        )}
        <div className="form-row"><label>kind</label><input className="inp" value={kind} onChange={e => setKind(e.target.value)} placeholder="issue_comment" /></div>
        <div className="form-row"><label>source</label><input className="inp" value={source} onChange={e => setSource(e.target.value)} placeholder="github" /></div>
        <div className="form-row"><label>dedup_key</label><input className="inp" value={dedup} onChange={e => setDedup(e.target.value)} placeholder="可选：同一个 key 只会收一次" /></div>
        <div className="form-row top"><label>payload</label><textarea className="inp mono" rows={6} value={payload} onChange={e => setPayload(e.target.value)} /></div>
      </div>
    </Modal>
  )
}

export function InboxPage() {
  const { query } = useLocation()
  const session = query.get('session') ?? ''
  const status = query.get('status') ?? ''
  const sessions = useApp(s => s.sessions)
  const [events, setEvents] = useState<InboxEvent[] | null>(null)
  const [adding, setAdding] = useState(false)
  const now = useNow(10_000)

  const load = useCallback(() => {
    api.listInbox({ session: session || undefined, status: status || undefined, limit: 200 })
      .then(setEvents, e => toast(api.errorText(e)))
  }, [session, status])
  useEffect(() => {
    load()
    const t = setInterval(() => { if (!document.hidden) load() }, 4000)
    return () => clearInterval(t)
  }, [load])

  const name = (id: string) => { const s = sessions.find(x => x.id === id); return s ? sessionLabel(s) : id.slice(0, 8) }

  return (
    <div className="main">
      <div className="topbar">
        <div className="ttl">Inbox</div>
        <div className="right"><button className="btn sm dark" onClick={() => setAdding(true)}><Icon name="plus" cls="sm" />发送事件</button></div>
      </div>
      <div className="page">
        <p className="page-intro">唤醒会话的一切都是 inbox 里的一个事件：人发的消息、对调用的决定、到期的计时器、外部系统 POST 的 webhook。一个会话一次取走它所有等待中的事件，作为一轮来处理。</p>
        <div className="filters">
          <div className="seg">
            {STATUSES.map(s => <button key={s} className={status === s ? 'on' : ''} onClick={() => setQuery('status', s || null, { replace: true })}>{STATUS_LABEL[s]}</button>)}
          </div>
          {session && <span className="chip">会话：{name(session)}<button onClick={() => setQuery('session', null, { replace: true })} title="清除"><Icon name="x" cls="sm" /></button></span>}
          <button className="ibtn" title="刷新" onClick={load}><Icon name="refresh" /></button>
        </div>
        {events === null ? <div className="muted">正在加载……</div> : events.length === 0 ? <div className="muted">没有事件</div> : (
          <table className="tbl">
            <thead><tr><th>#</th><th>时间</th><th>会话</th><th>kind</th><th>来源</th><th>内容</th><th>状态</th></tr></thead>
            <tbody>
              {events.map(e => (
                <tr key={e.id}>
                  <td className="mono">{e.id}</td>
                  <td title={e.created}>{relTime(Date.parse(e.created), now)}</td>
                  <td><Link to={`/c/${e.session_id}`}>{name(e.session_id)}</Link></td>
                  <td><span className="tag">{e.kind}</span></td>
                  <td>{e.source}{e.dedup_key && <div className="sub" title="dedup_key">{e.dedup_key}</div>}</td>
                  <td className="cell-text" title={JSON.stringify(e.payload)}>{oneLine(e.payload)}</td>
                  <td>
                    <span className={`pill s-${e.status}`}>{STATUS_LABEL[e.status] ?? e.status}</span>
                    {e.attempts > 1 && <div className="sub">第 {e.attempts} 次</div>}
                    {e.error && <div className="sub bad" title={e.error}>{oneLine(e.error, 60)}</div>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {adding && <NewEventModal onClose={() => setAdding(false)} onSent={load} />}
    </div>
  )
}
