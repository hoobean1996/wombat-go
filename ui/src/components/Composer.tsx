import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import * as api from '../lib/api'
import { sessionLabel } from '../lib/format'
import { useClickOutside } from '../lib/hooks'
import { navigate } from '../lib/router'
import { refreshSessions, useApp } from '../lib/store'
import { toast } from '../lib/toast'
import { Icon } from './ui'

/** The permission wombatd calls "full": every tool, and nobody asked about any of it. */
const FULL = 'full'

/**
 * Normal or Full Access. Normal is the server's own policy (tools.permission:
 * commands, and anything outside the working directory, wait for you); Full
 * Access runs every call without asking, and is the operator's to offer
 * (tools.full_access).
 */
function PermMenu({ value, offered, inSession, onPick }: { value: string; offered: boolean; inSession: boolean; onPick: (p: string) => void }) {
  const row = (p: string, icon: string, title: string, desc: string, warn = false, disabled = false) => (
    <div className={`opt ${(p === FULL) === (value === FULL) ? 'sel' : ''} ${disabled ? 'disabled' : ''}`} onClick={() => { if (!disabled) onPick(p) }}>
      <div className={`o-ic ${warn ? 'warn' : ''}`}><Icon name={icon} /></div>
      <div className="o-main"><div className="o-t">{title}</div><div className="o-d">{desc}</div></div>
      <Icon name="check" cls="chk" />
    </div>
  )
  return (
    <>
      <div className="dd-h">权限模式{inSession ? ' · 从下一步起生效' : ''}</div>
      {row('', 'lock', 'Normal', '命令和工作目录以外的操作先问你（默认）')}
      {row(FULL, 'alert', 'Full Access', '跳过所有人工确认，所有工具调用直接执行', true, !offered)}
      <div className="dd-sep" />
      <div className="dd-note">{offered
        ? <>Full Access 只在你信任当前任务时使用。{inSession ? '切换后，正在等待的确认会自动允许，之后也不再询问。' : ''}开启了 <code>tools.confine</code> 时，命令仍在沙箱里执行。</>
        : <>这台 wombatd 没有开放 Full Access：在 <code>wombatd.yaml</code> 里设置 <code>tools.full_access: true</code> 后重启即可。</>}
      </div>
    </>
  )
}

interface Props {
  /** Absent on the home page: sending starts a session. */
  sessionId?: string
  /** The session's permission, as it asked for it: "full", or anything else for Normal. */
  permission?: string
  /** A turn is running: Enter queues the message for the next turn, and steering is offered. */
  running?: boolean
  draft?: { text: string; n: number }
  up?: boolean
  onSent?: () => void
}

export function Composer({ sessionId, permission = '', running = false, draft, up = false, onSent }: Props) {
  const config = useApp(s => s.config)
  const [text, setText] = useState(draft?.text ?? '')
  // On the home page, what the new session will ask for; in a session, what it has.
  const [homePerm, setHomePerm] = useState('')
  const perm = sessionId ? permission : homePerm
  const full = perm === FULL
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)
  const close = useCallback(() => setOpen(false), [])
  useClickOutside(rootRef, close, open)

  // A new draft (a clicked suggestion) replaces the text.
  const [prevDraft, setPrevDraft] = useState(draft)
  if (draft !== prevDraft) {
    setPrevDraft(draft)
    if (draft) setText(draft.text)
  }
  useEffect(() => { if (draft) taRef.current?.focus() }, [draft])
  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(220, Math.max(54, ta.scrollHeight))}px`
  }, [text])

  const body = text.trim()

  const submit = async (how: 'send' | 'steer') => {
    if (!body || busy) return
    setBusy(true)
    try {
      if (!sessionId) {
        const title = body.replace(/\s+/g, ' ').slice(0, 40)
        const s = await api.createSession(body, { title, ...(full ? { permission: FULL } : {}) })
        void refreshSessions()
        navigate(`/c/${s.id}`)
        toast(`已创建「${sessionLabel(s)}」`)
      } else if (how === 'steer') {
        await api.steer(sessionId, body)
        toast('已插话：模型会在下一步看到')
      } else {
        await api.sendMessage(sessionId, body)
        if (running) toast('已排队：这一轮结束后处理')
      }
      setText('')
      onSent?.()
    } catch (e) {
      toast(api.errorText(e)) // keep the draft so it can be sent again
    } finally {
      setBusy(false)
    }
  }

  const changePermission = async (p: string) => {
    setOpen(false)
    if ((p === FULL) === full) return
    if (!sessionId) {
      setHomePerm(p)
      if (p === FULL) toast('Full Access：这个会话的工具调用都不再需要确认')
      return
    }
    try {
      await api.setPermission(sessionId, p)
      toast(p === FULL ? 'Full Access：正在等待的确认已自动允许，之后不再询问' : 'Normal：命令和工作目录以外的操作会先问你')
      onSent?.()
    } catch (e) {
      toast(api.errorText(e))
    }
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== 'Enter' || e.shiftKey || e.nativeEvent.isComposing || e.keyCode === 229) return
    e.preventDefault()
    void submit('send')
  }

  const placeholder = busy ? '正在发送……'
    : !sessionId ? '描述一个任务，或者问点什么……（Enter 发送，Shift+Enter 换行）'
    : running ? '这一轮还在进行：Enter 排到下一轮，或者点「插话」告诉正在跑的这一轮'
    : '继续追问……'

  return (
    <div ref={rootRef} className="composer">
      {full && (
        <div className="c-warn">
          <Icon name="alert" />
          <span><b>Full Access</b> · 已跳过所有人工确认，工具调用会直接执行</span>
          <button className="c-warn-back" onClick={() => void changePermission('')}>改回 Normal</button>
        </div>
      )}
      <textarea
        ref={taRef} className="c-input" rows={2} value={text} placeholder={placeholder}
        onChange={e => setText(e.target.value)} onKeyDown={onKeyDown} autoFocus
      />
      <div className="c-tools">
        {config && (
          <div className="pick">
            <button
              className={`tbtn perm ${full ? 'full' : ''}`}
              title={full ? 'Full Access：跳过所有人工确认' : 'Normal：命令和工作目录以外的操作先问你'}
              onClick={() => setOpen(o => !o)}
            >
              <Icon name={full ? 'alert' : 'lock'} /><span>{full ? 'Full Access' : 'Normal'}</span><Icon name="chev" cls="sm" />
            </button>
            {open && (
              <div className={`dd perm open ${up ? 'up' : ''}`}>
                <PermMenu value={perm} offered={!!config.full_access} inSession={!!sessionId} onPick={p => void changePermission(p)} />
              </div>
            )}
          </div>
        )}
        {config?.default_model && <span className="tbtn model-auto" title="模型由 wombatd.yaml 的 model.name 决定"><Icon name="sparkle" /><span>{config.default_model}</span></span>}
        <div className="c-right">
          {sessionId && running && (
            <button className="btn sm" disabled={!body || busy} title="把这条消息交给正在运行的这一轮" onClick={() => void submit('steer')}>插话</button>
          )}
          <button className={`send ${body && !busy ? '' : 'disabled'}`} title={running ? '排到下一轮' : '发送'} onClick={() => void submit('send')}>
            <Icon name="up" />
          </button>
        </div>
      </div>
    </div>
  )
}
