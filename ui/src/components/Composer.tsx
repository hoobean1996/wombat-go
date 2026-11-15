import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import * as api from '../lib/api'
import { sessionLabel } from '../lib/format'
import { useClickOutside } from '../lib/hooks'
import { navigate } from '../lib/router'
import { refreshSessions, useApp } from '../lib/store'
import { toast } from '../lib/toast'
import { Icon } from './ui'

const PERM_DESC: Record<string, string> = {
  off: '不做检查：每个调用都直接执行',
  workspace: '工作目录内读写和常见无害命令直接执行，其余先问你',
  ask: '每个调用都先问你',
  readonly: '只允许只读工具，其余一律拒绝',
}

/** The permission a new session asks for: the server's own, or a stricter one. */
function PermMenu({ modes, value, onPick }: { modes: string[]; value: string; onPick: (m: string) => void }) {
  const row = (m: string, title: string, desc: string) => (
    <div key={m || 'default'} className={`opt ${m === value ? 'sel' : ''}`} onClick={() => onPick(m)}>
      <div className="o-ic"><Icon name={m === 'off' ? 'alert' : 'lock'} /></div>
      <div className="o-main"><div className="o-t">{title}</div><div className="o-d">{desc}</div></div>
      <Icon name="check" cls="chk" />
    </div>
  )
  return (
    <>
      <div className="dd-h">权限模式 · 只能比服务器设置的更严格</div>
      {row('', '服务器默认', 'wombatd.yaml 里 tools.permission 设置的模式')}
      {modes.map(m => row(m, m, PERM_DESC[m] ?? ''))}
    </>
  )
}

interface Props {
  /** Absent on the home page: sending starts a session. */
  sessionId?: string
  /** A turn is running: Enter queues the message for the next turn, and steering is offered. */
  running?: boolean
  draft?: { text: string; n: number }
  up?: boolean
  onSent?: () => void
}

export function Composer({ sessionId, running = false, draft, up = false, onSent }: Props) {
  const config = useApp(s => s.config)
  const [text, setText] = useState(draft?.text ?? '')
  const [perm, setPerm] = useState('')
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
        const s = await api.createSession(body, { title, ...(perm ? { permission: perm } : {}) })
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
      <textarea
        ref={taRef} className="c-input" rows={2} value={text} placeholder={placeholder}
        onChange={e => setText(e.target.value)} onKeyDown={onKeyDown} autoFocus
      />
      <div className="c-tools">
        {!sessionId && config && (
          <div className="pick">
            <button className="tbtn perm" title="这个会话的权限模式" onClick={() => setOpen(o => !o)}>
              <Icon name={perm === 'off' ? 'alert' : 'lock'} /><span>{perm || '默认权限'}</span><Icon name="chev" cls="sm" />
            </button>
            {open && <div className={`dd perm open ${up ? 'up' : ''}`}><PermMenu modes={config.permission_modes} value={perm} onPick={m => { setPerm(m); setOpen(false) }} /></div>}
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
