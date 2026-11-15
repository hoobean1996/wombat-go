import { useState } from 'react'
import * as api from '../lib/api'
import { fmtDuration } from '../lib/format'
import { start, useApp } from '../lib/store'
import { toast } from '../lib/toast'
import { Icon } from '../components/ui'

export function TokenForm({ onSaved }: { onSaved?: () => void }) {
  const [token, setToken] = useState(api.getToken())
  const save = () => {
    api.setToken(token.trim())
    start()
    toast(token.trim() ? '已保存 token，正在重新连接' : '已清除 token')
    onSaved?.()
  }
  return (
    <div className="token-form">
      <input
        className="inp mono" type="password" value={token} onChange={e => setToken(e.target.value)}
        placeholder="wombatd.yaml 里的 server.token" autoComplete="off"
        onKeyDown={e => { if (e.key === 'Enter') save() }}
      />
      <button className="btn dark" onClick={save}>保存并连接</button>
    </div>
  )
}

export function SettingsPage() {
  const config = useApp(s => s.config)
  const health = useApp(s => s.health)
  return (
    <div className="main">
      <div className="topbar"><div className="ttl">设置</div></div>
      <div className="page">
        <section className="card">
          <h3><Icon name="lock" />API token</h3>
          <p className="page-intro">wombatd 配置了 <code>server.token</code> 时，每个请求都要带上它。它只保存在这个浏览器里（localStorage）。没有配置 token 时留空。</p>
          <TokenForm />
        </section>

        <section className="card">
          <h3><Icon name="server" />服务器</h3>
          <dl className="kv">
            <dt>版本</dt><dd className="mono">{health?.version ?? '—'}</dd>
            <dt>已运行</dt><dd>{health ? fmtDuration(health.uptime_sec * 1000) : '—'}</dd>
            <dt>默认模型</dt><dd className="mono">{config?.default_model || '—'}</dd>
            <dt>权限模式</dt><dd>{config?.permission_modes.join(' · ') || '—'}</dd>
            <dt>人工确认</dt><dd>{config ? (config.approvals ? '开启：需要确认的调用会停下来等你' : '关闭') : '—'}</dd>
          </dl>
          <p className="page-intro">这些都由 <code>wombatd.yaml</code> 决定，改完重启 wombatd 生效；界面不能修改服务器配置。</p>
        </section>

        <section className="card">
          <h3><Icon name="terminal" />工具 {config && <span className="meta">{config.tools.length}</span>}</h3>
          <table className="tbl">
            <thead><tr><th>名称</th><th>类别</th><th>只读</th><th>说明</th></tr></thead>
            <tbody>
              {(config?.tools ?? []).map(t => (
                <tr key={t.name}>
                  <td className="mono">{t.name}</td>
                  <td>{t.category || '—'}</td>
                  <td>{t.read_only ? <Icon name="check" cls="sm ok" /> : ''}</td>
                  <td className="cell-text" title={t.description}>{t.description?.split('\n')[0]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  )
}
