import { useState } from 'react'
import { useApp } from '../lib/store'
import { Composer } from '../components/Composer'
import { Icon, Link } from '../components/ui'

const SUGGESTIONS = [
  { icon: 'terminal', label: '看看这个项目', text: '这个工作目录里是什么项目？用几句话说明它的结构和用途。' },
  { icon: 'clock', label: '10 分钟后提醒我', text: '用 set_timer 设一个 10 分钟后的计时器，到时用 send_message 提醒我休息一下。' },
  { icon: 'code', label: '跑一下测试', text: '在工作目录里跑一下测试，告诉我结果；如果有失败，说明原因。' },
]

export function HomePage() {
  const config = useApp(s => s.config)
  const sessions = useApp(s => s.sessions)
  const [draft, setDraft] = useState<{ text: string; n: number }>()
  const busy = sessions.filter(s => s.state === 'running').length
  const waiting = sessions.filter(s => s.state === 'approving' || s.state === 'waiting').length

  return (
    <div className="main">
      <div className="topbar">
        <div className="right">
          {waiting > 0 && <Link to="/outbox" className="chip-status"><Icon name="alert" cls="sm" /><span className="txt">{waiting} 个会话在等你</span></Link>}
          <span className="chip-status"><span className="txt">{busy} 个会话运行中</span></span>
        </div>
      </div>
      <div className="home">
        <div className="greet">今天让 Wombat 做什么？</div>
        <div className="greet-sub">
          {config ? <>模型 <b>{config.default_model || '默认'}</b> · {config.tools.length} 个工具 · 每个会话一次只跑一轮，结果也会进 Outbox</> : '正在读取服务器配置……'}
        </div>
        <Composer draft={draft} />
        <div className="sugs">
          {SUGGESTIONS.map(s => (
            <button key={s.label} className="sug" onClick={() => setDraft(d => ({ text: s.text, n: (d?.n ?? 0) + 1 }))}>
              <Icon name={s.icon} />{s.label}
            </button>
          ))}
        </div>
        <div className="home-foot">执行命令、访问工作目录之外的路径前，会先在会话里请你确认（取决于 wombatd.yaml 的 tools.permission）。</div>
      </div>
    </div>
  )
}
