import { useCallback, useEffect, useState } from 'react'
import * as api from '../lib/api'
import { withoutChat } from '../lib/chat'
import { relTime, sessionLabel } from '../lib/format'
import { useNow } from '../lib/hooks'
import { setQuery, useLocation } from '../lib/router'
import { outboxChanged, useApp } from '../lib/store'
import { toast } from '../lib/toast'
import type { OutboxMessage } from '../types'
import { Icon, Link } from '../components/ui'
import { Markdown } from '../components/Markdown'

const KIND_LABEL: Record<string, string> = { reply: '回复', question: '提问', approval: '等待确认', awaiting: '等待结果', error: '出错', message: '主动消息' }
const STATUS_LABEL: Record<string, string> = { '': '全部', pending: '未处理', delivered: '已处理', failed: '投递失败' }

export function OutboxPage() {
  const { query } = useLocation()
  const status = query.get('status') ?? 'pending'
  const sessions = useApp(s => s.sessions)
  const tick = useApp(s => s.outboxTick)
  const [messages, setMessages] = useState<OutboxMessage[] | null>(null)
  const now = useNow(10_000)

  const load = useCallback(() => {
    api.listOutboxAll({ status: status === 'all' ? undefined : status })
      .then(list => withoutChat(list))
      .then(list => setMessages([...list].reverse()), e => toast(api.errorText(e)))
  }, [status])
  useEffect(load, [load, tick])

  const ackOne = async (m: OutboxMessage) => {
    try {
      await api.ack(m.id)
      outboxChanged()
      load()
    } catch (e) {
      toast(api.errorText(e))
    }
  }
  const ackAll = async () => {
    const list = (messages ?? []).filter(m => m.status === 'pending')
    for (const m of list) {
      try { await api.ack(m.id) } catch (e) { toast(api.errorText(e)); break }
    }
    outboxChanged()
    load()
  }

  const name = (id: string) => { const s = sessions.find(x => x.id === id); return s ? sessionLabel(s) : id.slice(0, 8) }
  const pending = (messages ?? []).filter(m => m.status === 'pending').length

  return (
    <div className="main">
      <div className="topbar">
        <div className="ttl">Outbox</div>
        <div className="right">{pending > 0 && <button className="btn sm" onClick={() => void ackAll()}><Icon name="check" cls="sm" />全部标为已处理</button>}</div>
      </div>
      <div className="page">
        <p className="page-intro">Agent 要对外说的话都进 outbox：对外部消息的回复、它提的问题、等待确认的调用，以及它用 <code>send_message</code> 主动发的消息（计时器触发的那一轮只有调用它才会出声）。客户端读到后 ack；配置了 <code>outbox.webhook</code> 时，wombatd 会自己把每条 POST 出去。对话里直接聊天引起的回复和确认请求不列在这里，它们就在各自的对话里。</p>
        <div className="filters">
          <div className="seg">
            {(['pending', 'delivered', 'failed', 'all'] as const).map(s => (
              <button key={s} className={status === s ? 'on' : ''} onClick={() => setQuery('status', s === 'pending' ? null : s, { replace: true })}>{STATUS_LABEL[s === 'all' ? '' : s]}</button>
            ))}
          </div>
          <button className="ibtn" title="刷新" onClick={load}><Icon name="refresh" /></button>
        </div>
        {messages === null ? <div className="muted">正在加载……</div> : messages.length === 0 ? <div className="muted">{status === 'pending' ? '没有未处理的消息' : '没有消息'}</div> : (
          <div className="ob-list">
            {messages.map(m => (
              <div key={m.id} className={`ob-item k-${m.kind} ${m.status === 'pending' ? 'unread' : ''}`}>
                <div className="ob-h">
                  <span className={`tag k-${m.kind}`}>{KIND_LABEL[m.kind] ?? m.kind}</span>
                  <Link to={`/c/${m.session_id}`} className="ob-s">{name(m.session_id)}</Link>
                  <span className="sub" title={m.created}>#{m.id} · {relTime(Date.parse(m.created), now)}</span>
                  <span className="grow" />
                  {m.status === 'pending'
                    ? <button className="btn sm" title="告诉 outbox 这条已经处理过了" onClick={() => void ackOne(m)}><Icon name="check" cls="sm" />标为已处理</button>
                    : <span className={`pill s-${m.status}`}>{STATUS_LABEL[m.status] ?? m.status}</span>}
                </div>
                <div className="ob-body"><Markdown text={m.text} /></div>
                {m.error && <div className="sub bad">{m.error}</div>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
