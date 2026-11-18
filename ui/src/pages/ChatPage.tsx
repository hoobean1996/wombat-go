import { Fragment, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as api from '../lib/api'
import { sessionLabel } from '../lib/format'
import { useClickOutside } from '../lib/hooks'
import { navigate, setQuery, useLocation } from '../lib/router'
import { FileOpenerContext } from '../lib/files'
import { FilePanel } from '../components/FilePanel'
import { useSessionView } from '../lib/session'
import { refreshSessions, useApp } from '../lib/store'
import { confirmDialog } from '../lib/confirm'
import { toast } from '../lib/toast'
import type { Message, SessionInfo } from '../types'
import { Composer } from '../components/Composer'
import { ApprovalCard, AssistantBlock, AwaitCard, LiveBlock, QueuedList, SessionStatus, SpendLine, UserBubble } from '../components/Thread'
import { exchanges } from '../lib/transcript'
import { Icon, Link, Modal } from '../components/ui'

function SessionMenu({ info, messages }: { info: SessionInfo; messages: Message[] }) {
  const [open, setOpen] = useState(false)
  const [raw, setRaw] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const close = useCallback(() => setOpen(false), [])
  useClickOutside(ref, close, open)
  const remove = async () => {
    setOpen(false)
    if (!(await confirmDialog('删除会话', `删除「${sessionLabel(info)}」？它的消息、计时器和 outbox 记录都会一起删除，不能恢复。`, '删除'))) return
    try {
      await api.deleteSession(info.id)
      void refreshSessions()
      navigate('/')
      toast('会话已删除')
    } catch (e) {
      toast(api.errorText(e))
    }
  }
  return (
    <div className="menu-wrap" ref={ref}>
      <button className="ibtn" title="更多" onClick={() => setOpen(o => !o)}><Icon name="more" /></button>
      {open && (
        <div className="menu">
          <button onClick={() => { setOpen(false); setRaw(true) }}><Icon name="code" cls="sm" />查看原始记录</button>
          <Link to={`/inbox?session=${info.id}`} className="menu-link"><Icon name="download" cls="sm" />这个会话的 Inbox</Link>
          <Link to={`/timers?session=${info.id}`} className="menu-link"><Icon name="clock" cls="sm" />这个会话的计时器</Link>
          <button onClick={() => { setOpen(false); void navigator.clipboard?.writeText(info.id).then(() => toast('已复制会话 ID')) }}><Icon name="copy" cls="sm" />复制会话 ID</button>
          <button className="danger" onClick={() => void remove()}><Icon name="trash" cls="sm" />删除会话</button>
        </div>
      )}
      {raw && (
        <Modal title="原始记录" icon="code" onClose={() => setRaw(false)}>
          <pre className="raw-json">{JSON.stringify({ session: info, messages }, null, 2)}</pre>
        </Modal>
      )}
    </div>
  )
}

export function ChatPage({ id }: { id: string }) {
  const view = useSessionView(id)
  const config = useApp(s => s.config)
  const { info, messages, approvals, awaiting, queued, live, refresh } = view
  // A call the transcript holds without a result is either waiting — for a
  // decision or a result, the approvals say which — or was never run.
  const list = useMemo(() => {
    const waiting = new Set([...approvals, ...awaiting].map(a => a.use_id))
    const all = exchanges(messages)
    if (!waiting.size) return all
    return all.map(ex => ({ ...ex, steps: ex.steps.map(s => (s.status === 'pending' && waiting.has(s.id) ? { ...s, status: 'waiting' as const } : s)) }))
  }, [messages, approvals, awaiting])
  const [deciding, setDeciding] = useState(false)
  const model = info?.options.model || config?.default_model
  // The file open beside the conversation, in the URL, so the back button closes it.
  const { query } = useLocation()
  const file = config?.files ? query.get('file') : null
  const openFile = useCallback((p: string) => setQuery('file', p), [])

  const threadRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  useLayoutEffect(() => {
    const el = threadRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [list, live, approvals, awaiting, queued, info?.state])

  if (view.missing) {
    return <div className="main"><div className="empty"><div><div className="big">找不到这个会话</div><div>它可能已经被删除，或者已经过期。<Link to="/">新建一个对话</Link></div></div></div></div>
  }
  if (view.error) {
    return <div className="main"><div className="empty"><div><div className="big">打不开这个会话</div><div>{view.error}</div></div></div></div>
  }
  if (!info) return <div className="main"><div className="empty"><div><span className="run-ind" /> 正在加载……</div></div></div>

  const decide = async (useId: string, allow: boolean) => {
    setDeciding(true)
    try {
      await api.decide(info.id, useId, allow)
      toast(allow ? '已允许' : '已拒绝')
      refresh()
    } catch (e) {
      toast(api.errorText(e))
    } finally {
      setDeciding(false)
    }
  }

  const post = async (useId: string, output: string, isError: boolean) => {
    setDeciding(true)
    try {
      await api.postResult(info.id, useId, output, isError)
      toast('结果已提交')
      refresh()
    } catch (e) {
      toast(api.errorText(e))
    } finally {
      setDeciding(false)
    }
  }

  const running = info.state === 'running' || !!live
  const stopped = info.state === 'approving' || info.state === 'awaiting'
  const lastIdx = list.length - 1

  return (
    <FileOpenerContext.Provider value={config?.files ? openFile : null}>
    <div className="main">
      <div className="topbar">
        <div className="ttl" title={sessionLabel(info)}>{sessionLabel(info)}</div>
        {info.thread && <span className="node-chip" title="这个会话处理的线程（POST /api/events 的 thread）">{info.thread}</span>}
        <div className="right">
          <SpendLine info={info} />
          <SessionMenu info={info} messages={messages} />
        </div>
      </div>
      <div
        className="thread" ref={threadRef}
        onScroll={e => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80 }}
      >
        <div className="msgs">
          {list.length === 0 && !live && <div className="thread-empty">还没有完成的对话：第一轮正在排队或运行。</div>}
          {list.map((ex, i) => (
            <Fragment key={i}>
              {ex.prompt !== null && <UserBubble text={ex.prompt} />}
              {(ex.steps.length > 0 || ex.text || ex.thinking > 0) && (
                <AssistantBlock ex={ex} model={model} status={i === lastIdx && !live && stopped ? <span className="st-wait">{info.state === 'approving' ? '等待你的确认' : '等待外部结果'}</span> : undefined} />
              )}
            </Fragment>
          ))}
          {live && (
            <>
              {live.prompt && <UserBubble text={live.prompt} />}
              <LiveBlock live={live} model={model} />
            </>
          )}
          {!live && stopped && approvals.map(ap => (
            <ApprovalCard key={ap.use_id} ap={ap} busy={deciding} onDecide={allow => void decide(ap.use_id, allow)} />
          ))}
          {!live && stopped && awaiting.map(ap => (
            <AwaitCard key={ap.use_id} ap={ap} busy={deciding} onPost={(out, isError) => void post(ap.use_id, out, isError)} />
          ))}
          {!live && <SessionStatus info={info} />}
          <QueuedList events={queued} />
        </div>
      </div>
      <div className="dock">
        {stopped && !live && (
          <div className="offline-bar"><Icon name={approvals.length ? 'lock' : 'clock'} /><span>
            {approvals.length > 0 && <b>有 {approvals.length} 个调用在等你确认。</b>}
            {awaiting.length > 0 && <b>有 {awaiting.length} 个调用在等外部结果。</b>}
            全部有了结果之后这一轮会继续；在那之前，新消息会排队。
          </span></div>
        )}
        <Composer key={info.id} sessionId={info.id} permission={info.options.permission ?? ''} running={running} up onSent={refresh} />
      </div>
    </div>
    {file && (
      // Read again whenever the transcript grows: a turn that ended may have rewritten it.
      <FilePanel key={file} sessionId={info.id} path={file} stamp={messages.length} onOpen={openFile} onClose={() => setQuery('file', null)} />
    )}
    </FileOpenerContext.Provider>
  )
}
