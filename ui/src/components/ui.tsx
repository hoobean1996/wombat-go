import { useEffect, type MouseEvent, type ReactNode } from 'react'
import { navigate } from '../lib/router'
import { useToasts } from '../lib/toast'
import type { SessionState } from '../types'

export function Icon({ name, cls = '' }: { name: string; cls?: string }) {
  return <svg className={`ic ${cls}`} aria-hidden="true"><use href={`#i-${name}`} /></svg>
}

export function Logo() {
  return <svg className="logo" aria-hidden="true"><use href="#logo" /></svg>
}

export function Link({ to, className, children, title }: { to: string; className?: string; children: ReactNode; title?: string }) {
  const onClick = (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    navigate(to)
  }
  return <a href={to} className={className} onClick={onClick} title={title}>{children}</a>
}

export function Modal({ title, icon, onClose, children, footer }: { title: string; icon?: string; onClose: () => void; children: ReactNode; footer?: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="modal-back" onPointerDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal" role="dialog" aria-label={title}>
        <div className="modal-h">{icon && <Icon name={icon} />}{title}<button className="ibtn" onClick={onClose} title="关闭"><Icon name="x" /></button></div>
        {children}
        {footer && <div className="modal-f">{footer}</div>}
      </div>
    </div>
  )
}

export function Toasts() {
  const toasts = useToasts()
  return <div className="toasts">{toasts.map(t => <div key={t.id} className="toast">{t.text}</div>)}</div>
}

const STATE_LABEL: Record<SessionState, string> = {
  idle: '空闲', running: '运行中', waiting: '等你回答', approving: '待确认', awaiting: '等结果', done: '已结束', failed: '出错',
}

/** A session's state as a small badge. */
export function StateBadge({ state }: { state: SessionState }) {
  return <span className={`badge st-${state}`}>{state === 'running' && <span className="run-ind" />}{STATE_LABEL[state] ?? state}</span>
}
