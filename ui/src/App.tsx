import { useEffect, useState } from 'react'
import './styles/ui.css'
import './styles/extra.css'
import './styles/wombat.css'
import { matchRoute, useLocation } from './lib/router'
import { sessionLabel } from './lib/format'
import { start, useApp } from './lib/store'
import { ConfirmDialog } from './components/ConfirmDialog'
import { Sidebar } from './components/Sidebar'
import { Icon, Link, Logo, Toasts } from './components/ui'
import { HomePage } from './pages/HomePage'
import { ChatPage } from './pages/ChatPage'
import { InboxPage } from './pages/InboxPage'
import { OutboxPage } from './pages/OutboxPage'
import { TimersPage } from './pages/TimersPage'
import { ToolCallsPage } from './pages/ToolCallsPage'
import { SettingsPage, TokenForm } from './pages/SettingsPage'

const TITLES = { home: '新对话', inbox: 'Inbox', outbox: 'Outbox', timers: '计时器', calls: '工具调用', settings: '设置', notfound: '找不到页面' } as const

/** Until wombatd answers. */
function Connecting({ down, error }: { down: boolean; error?: string }) {
  return (
    <div className="landing">
      <div className="landing-card">
        <div className="landing-logo"><Logo /></div>
        <div className="big">{down ? '连不上 wombatd' : '正在连接 wombatd……'}</div>
        {down && <p className="landing-sub">{error ? `${error}。` : ''}会自动重试；请确认 <code>wombatd</code> 在运行（<code>make serve</code>）。</p>}
      </div>
    </div>
  )
}

/** wombatd has server.token set and this browser does not have it (or has a wrong one). */
function TokenGate() {
  return (
    <div className="landing">
      <div className="landing-card">
        <div className="landing-logo"><Logo /></div>
        <div className="landing-kicker">Wombat</div>
        <h1>需要 API token</h1>
        <p className="landing-sub">这个 wombatd 设置了 <code>server.token</code>。输入它就能继续；它只保存在这个浏览器里。</p>
        <TokenForm />
      </div>
    </div>
  )
}

export default function App() {
  const conn = useApp(s => s.conn)
  const error = useApp(s => s.error)
  useEffect(() => { start() }, [])
  if (conn === 'unauthorized') return <TokenGate />
  if (conn !== 'ok') return <Connecting down={conn === 'down'} error={error} />
  return <Workbench />
}

function Workbench() {
  const { path } = useLocation()
  const route = matchRoute(path)
  const stale = useApp(s => s.stale)
  const chatTitle = useApp(s => {
    if (route.name !== 'chat') return undefined
    const x = s.sessions.find(y => y.id === route.id)
    return x && sessionLabel(x)
  })
  const [collapsed, setCollapsed] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    let saved = 0
    try { saved = Number(localStorage.getItem('wombat.sidebarWidth')) } catch { /* no storage */ }
    return saved >= 220 && saved <= 440 ? saved : 252
  })
  const [resizing, setResizing] = useState(false)
  const resizeSidebar = (next: number) => {
    const w = Math.min(440, Math.max(220, next))
    setSidebarWidth(w)
    try { localStorage.setItem('wombat.sidebarWidth', String(w)) } catch { /* no storage */ }
  }

  useEffect(() => {
    const page = route.name === 'chat' ? chatTitle ?? '会话' : TITLES[route.name]
    document.title = `${page} · Wombat`
  }, [route.name, chatTitle])

  return (
    <div className={`app ${collapsed ? 'rail' : ''} ${resizing ? 'resizing' : ''}`}>
      <Sidebar
        route={route} collapsed={collapsed} onToggle={() => setCollapsed(c => !c)}
        width={sidebarWidth} onWidth={resizeSidebar}
        onResizeStart={() => setResizing(true)} onResizeEnd={() => setResizing(false)}
      />
      {route.name === 'home' && <HomePage />}
      {route.name === 'chat' && <ChatPage key={route.id} id={route.id} />}
      {route.name === 'inbox' && <InboxPage />}
      {route.name === 'outbox' && <OutboxPage />}
      {route.name === 'timers' && <TimersPage />}
      {route.name === 'calls' && <ToolCallsPage />}
      {route.name === 'settings' && <SettingsPage />}
      {route.name === 'notfound' && (
        <div className="main"><div className="empty"><div><div className="big">找不到这个页面</div><Link to="/">回到新对话</Link></div></div></div>
      )}
      <ConfirmDialog />
      {stale && <div className="conn-bar"><Icon name="alert" cls="sm" />与 wombatd 的连接断开了，正在重连……</div>}
      <Toasts />
    </div>
  )
}
