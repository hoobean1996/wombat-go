import { useMemo, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from 'react'
import * as api from '../lib/api'
import { confirmDialog } from '../lib/confirm'
import { dayGroup, sessionLabel } from '../lib/format'
import { navigate, type Route } from '../lib/router'
import { refreshSessions, useApp } from '../lib/store'
import { toast } from '../lib/toast'
import type { SessionInfo } from '../types'
import { Icon, Link, Logo } from './ui'

interface MenuState { session: SessionInfo; left: number; top: number }

const FLAG: Partial<Record<SessionInfo['state'], [string, string]>> = {
  approving: ['sb-approve', '待确认'],
  waiting: ['sb-approve', '待回答'],
  failed: ['sb-fail', '出错'],
}

export function Sidebar({ route, collapsed, onToggle, width = 252, onWidth, onResizeStart, onResizeEnd }: {
  route: Route; collapsed: boolean; onToggle: () => void; width?: number
  onWidth?: (n: number) => void; onResizeStart?: () => void; onResizeEnd?: () => void
}) {
  const sessions = useApp(s => s.sessions)
  const loaded = useApp(s => s.sessionsLoaded)
  const pending = useApp(s => s.outboxPending)
  const health = useApp(s => s.health)
  const [menu, setMenu] = useState<MenuState | null>(null)

  const groups = useMemo(() => {
    const sorted = [...sessions].sort((a, b) => Date.parse(b.updated) - Date.parse(a.updated))
    const out: { label: string; list: SessionInfo[] }[] = []
    for (const s of sorted) {
      const label = dayGroup(Date.parse(s.updated))
      if (out.at(-1)?.label !== label) out.push({ label, list: [] })
      out.at(-1)!.list.push(s)
    }
    return out
  }, [sessions])

  const openMenu = (e: ReactMouseEvent<HTMLButtonElement>, s: SessionInfo) => {
    e.preventDefault()
    e.stopPropagation()
    const r = e.currentTarget.getBoundingClientRect()
    setMenu({ session: s, left: Math.min(window.innerWidth - 220, r.right + 6), top: Math.min(window.innerHeight - 120, r.bottom + 4) })
  }

  const remove = async (s: SessionInfo) => {
    setMenu(null)
    const ok = await confirmDialog('删除会话', `删除「${sessionLabel(s)}」？正在运行的这一轮会被取消，它的消息、计时器和 outbox 记录都会一起删除，不能恢复。`, '删除')
    if (!ok) return
    try {
      await api.deleteSession(s.id)
      if (route.name === 'chat' && route.id === s.id) navigate('/')
      void refreshSessions()
      toast('会话已删除')
    } catch (e) {
      toast(api.errorText(e))
    }
  }

  const copyId = (s: SessionInfo) => {
    setMenu(null)
    void navigator.clipboard?.writeText(s.id).then(() => toast('已复制会话 ID'), () => toast(s.id))
  }

  const startResize = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (collapsed || !onWidth) return
    e.preventDefault()
    onResizeStart?.()
    const move = (ev: PointerEvent) => onWidth(Math.min(440, Math.max(220, ev.clientX)))
    const end = () => {
      window.removeEventListener('pointermove', move)
      onResizeEnd?.()
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
    }
    document.body.style.userSelect = 'none'
    document.body.style.cursor = 'col-resize'
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', end, { once: true })
  }

  const nav = (to: string, name: Route['name'], icon: string, label: string, meta?: number) => (
    <Link to={to} className={`sb-btn ${route.name === name ? 'active' : ''}`} title={label}>
      <Icon name={icon} /><span className="txt">{label}</span>{!!meta && <span className="meta badge-n">{meta}</span>}
    </Link>
  )

  return (
    <aside className="sb" style={collapsed ? undefined : { width }}>
      <div className="sb-top">
        <div className="brand" onClick={() => { if (collapsed) onToggle() }} title={collapsed ? '展开侧栏' : undefined}>
          <Logo /><span className="txt">Wombat</span>
        </div>
        <button className="ibtn" title="收起侧栏" onClick={onToggle}><Icon name="sidebar" /></button>
      </div>
      {nav('/', 'home', 'edit', '新对话')}
      {nav('/inbox', 'inbox', 'download', 'Inbox')}
      {nav('/outbox', 'outbox', 'upload', 'Outbox', pending)}
      {nav('/timers', 'timers', 'clock', '计时器')}

      <div className="sb-scroll">
        {loaded && sessions.length === 0 && <div className="sb-empty">还没有会话</div>}
        {groups.map(g => (
          <div key={g.label}>
            <div className="sb-label"><span>{g.label}</span></div>
            {g.list.map(s => {
              const active = route.name === 'chat' && route.id === s.id
              const flag = FLAG[s.state]
              return (
                <Link key={s.id} to={`/c/${s.id}`} className={`sb-item ${active ? 'active' : ''}`} title={sessionLabel(s)}>
                  <span className="txt">{sessionLabel(s)}</span>
                  {flag && <span className={flag[0]}>{flag[1]}</span>}
                  {s.state === 'running' && <span className="run-ind" title="正在运行" />}
                  <span className="sb-acts">
                    <button className="sb-act" title="更多" onClick={e => openMenu(e, s)}><Icon name="more" cls="sm" /></button>
                  </span>
                </Link>
              )
            })}
          </div>
        ))}
      </div>

      {menu && (
        <>
          <div className="sb-menu-back" onClick={() => setMenu(null)} />
          <div className="sb-menu" role="menu" style={{ left: menu.left, top: menu.top }}>
            <button role="menuitem" onClick={() => copyId(menu.session)}><Icon name="copy" /><span>复制会话 ID</span></button>
            <button role="menuitem" className="danger" onClick={() => void remove(menu.session)}><Icon name="trash" /><span>删除会话</span></button>
          </div>
        </>
      )}

      <div className="sb-foot">
        <div className="user">
          <div className="avatar"><Icon name="server" cls="sm" /></div>
          <div className="u-txt"><b>wombatd</b><span>{health ? `版本 ${health.version.slice(0, 12)}` : '未连接'}</span></div>
          <Link to="/settings" className={`ibtn u-set ${route.name === 'settings' ? 'on' : ''}`} title="设置"><Icon name="settings" /></Link>
        </div>
      </div>
      {!collapsed && (
        <div
          className="sb-resize" role="separator" aria-orientation="vertical" aria-label="调整侧栏宽度"
          aria-valuemin={220} aria-valuemax={440} aria-valuenow={width} tabIndex={0}
          title="拖动调整侧栏宽度，双击恢复默认"
          onPointerDown={startResize}
          onDoubleClick={() => onWidth?.(252)}
          onKeyDown={e => {
            if (e.key === 'ArrowLeft') { e.preventDefault(); onWidth?.(Math.max(220, width - 12)) }
            if (e.key === 'ArrowRight') { e.preventDefault(); onWidth?.(Math.min(440, width + 12)) }
          }}
        />
      )}
    </aside>
  )
}
