import { useCallback, useEffect, useState } from 'react'
import * as api from '../lib/api'
import { fmtGoDuration, relTime, sessionLabel } from '../lib/format'
import { useNow } from '../lib/hooks'
import { setQuery, useLocation } from '../lib/router'
import { useApp } from '../lib/store'
import { confirmDialog } from '../lib/confirm'
import { toast } from '../lib/toast'
import type { Timer } from '../types'
import { Icon, Link, Modal } from '../components/ui'

const STATUS_LABEL: Record<string, string> = { '': '全部', active: '等待中', fired: '已触发', cancelled: '已取消' }

function NewTimerModal({ session, onClose, onAdded }: { session: string; onClose: () => void; onAdded: () => void }) {
  const sessions = useApp(s => s.sessions)
  const [sessionId, setSessionId] = useState(session || sessions[0]?.id || '')
  const [note, setNote] = useState('')
  const [when, setWhen] = useState<'in' | 'at'>('in')
  const [inVal, setIn] = useState('10m')
  const [at, setAt] = useState('')
  const [every, setEvery] = useState('')
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    setBusy(true)
    try {
      const t = await api.addTimer({
        session_id: sessionId, note: note.trim(), every: every.trim() || undefined,
        ...(when === 'in' ? { in: inVal.trim() } : { at: new Date(at).toISOString() }),
      })
      toast(`计时器 #${t.id} 已设置：${relTime(Date.parse(t.due))}`)
      onAdded()
      onClose()
    } catch (e) {
      toast(api.errorText(e))
    } finally {
      setBusy(false)
    }
  }
  const ok = !!sessionId && !!note.trim() && (when === 'in' ? !!inVal.trim() : !!at)
  return (
    <Modal title="新建计时器" icon="clock" onClose={onClose} footer={<>
      <button className="btn" onClick={onClose}>取消</button>
      <button className="btn dark" disabled={busy || !ok} onClick={() => void submit()}>设置</button>
    </>}>
      <div className="form">
        <div className="form-row"><label>会话</label>
          <select className="inp" value={sessionId} onChange={e => setSessionId(e.target.value)}>
            {sessions.map(s => <option key={s.id} value={s.id}>{sessionLabel(s)}</option>)}
          </select>
        </div>
        <div className="form-row top"><label>note</label><textarea className="inp" rows={3} value={note} onChange={e => setNote(e.target.value)} placeholder="到时要做什么。触发时模型只会看到这段话，所以要写全。" /></div>
        <div className="form-row">
          <label>时间</label>
          <div className="seg">
            <button className={when === 'in' ? 'on' : ''} onClick={() => setWhen('in')}>多久之后</button>
            <button className={when === 'at' ? 'on' : ''} onClick={() => setWhen('at')}>指定时间</button>
          </div>
          {when === 'in'
            ? <input className="inp short" value={inVal} onChange={e => setIn(e.target.value)} placeholder="90s, 10m, 2h30m" />
            : <input className="inp short" type="datetime-local" value={at} onChange={e => setAt(e.target.value)} />}
        </div>
        <div className="form-row"><label>重复</label><input className="inp short" value={every} onChange={e => setEvery(e.target.value)} placeholder="可选：24h（至少 1m）" /></div>
      </div>
    </Modal>
  )
}

export function TimersPage() {
  const { query } = useLocation()
  const session = query.get('session') ?? ''
  const status = query.get('status') ?? 'active'
  const sessions = useApp(s => s.sessions)
  const [timers, setTimers] = useState<Timer[] | null>(null)
  const [adding, setAdding] = useState(false)
  const now = useNow(5_000)

  const load = useCallback(() => {
    api.listTimers({ session: session || undefined, status: status === 'all' ? undefined : status })
      .then(list => setTimers([...list].sort((a, b) => Date.parse(a.due) - Date.parse(b.due))), e => toast(api.errorText(e)))
  }, [session, status])
  useEffect(() => {
    load()
    const t = setInterval(() => { if (!document.hidden) load() }, 5000)
    return () => clearInterval(t)
  }, [load])

  const cancel = async (t: Timer) => {
    if (!(await confirmDialog('取消计时器', `取消计时器 #${t.id}？「${t.note}」`, '取消计时器'))) return
    try {
      await api.cancelTimer(t.id)
      toast('已取消')
      load()
    } catch (e) {
      toast(api.errorText(e))
    }
  }

  const name = (id: string) => { const s = sessions.find(x => x.id === id); return s ? sessionLabel(s) : id.slice(0, 8) }

  return (
    <div className="main">
      <div className="topbar">
        <div className="ttl">计时器</div>
        <div className="right"><button className="btn sm dark" disabled={!sessions.length} onClick={() => setAdding(true)}><Icon name="plus" cls="sm" />新建计时器</button></div>
      </div>
      <div className="page">
        <p className="page-intro">计时器到期时，会往它的会话的 inbox 里放一个事件，唤醒会话跑一轮。Agent 可以用 <code>set_timer</code> 给自己设（比如“明早再检查一次”），你也可以在这里设。wombatd 停机期间错过的计时器，回来后只补触发一次。</p>
        <div className="filters">
          <div className="seg">
            {(['active', 'fired', 'cancelled', 'all'] as const).map(s => (
              <button key={s} className={status === s ? 'on' : ''} onClick={() => setQuery('status', s === 'active' ? null : s, { replace: true })}>{STATUS_LABEL[s === 'all' ? '' : s]}</button>
            ))}
          </div>
          {session && <span className="chip">会话：{name(session)}<button onClick={() => setQuery('session', null, { replace: true })} title="清除"><Icon name="x" cls="sm" /></button></span>}
          <button className="ibtn" title="刷新" onClick={load}><Icon name="refresh" /></button>
        </div>
        {timers === null ? <div className="muted">正在加载……</div> : timers.length === 0 ? <div className="muted">没有计时器</div> : (
          <table className="tbl">
            <thead><tr><th>#</th><th>到期</th><th>会话</th><th>要做的事</th><th>重复</th><th>设置者</th><th>状态</th><th /></tr></thead>
            <tbody>
              {timers.map(t => (
                <tr key={t.id}>
                  <td className="mono">{t.id}</td>
                  <td title={t.due}>{relTime(Date.parse(t.due), now)}</td>
                  <td><Link to={`/c/${t.session_id}`}>{name(t.session_id)}</Link></td>
                  <td className="cell-text" title={t.note}>{t.note}</td>
                  <td>{t.every ? `每 ${fmtGoDuration(t.every)}` : '—'}</td>
                  <td>{t.created_by === 'agent' ? 'Agent' : t.created_by}</td>
                  <td><span className={`pill s-${t.status}`}>{STATUS_LABEL[t.status] ?? t.status}</span>{t.fired > 0 && <div className="sub">已触发 {t.fired} 次</div>}</td>
                  <td>{t.status === 'active' && <button className="btn sm" onClick={() => void cancel(t)}>取消</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {adding && <NewTimerModal session={session} onClose={() => setAdding(false)} onAdded={load} />}
    </div>
  )
}
