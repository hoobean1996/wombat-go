const MIN = 60_000
const HOUR = 60 * MIN
const DAY = 24 * HOUR

export function timeAgo(t: number, now = Date.now()) {
  const d = Math.max(0, now - t)
  if (d < 45_000) return '刚刚'
  if (d < HOUR) return `${Math.max(1, Math.round(d / MIN))} 分钟前`
  if (d < DAY) return `${Math.round(d / HOUR)} 小时前`
  const days = Math.floor(d / DAY)
  if (days === 1) return '昨天'
  if (days < 30) return `${days} 天前`
  return new Date(t).toLocaleDateString('zh-CN')
}

export function dayGroup(t: number, now = Date.now()) {
  const start = new Date(now)
  start.setHours(0, 0, 0, 0)
  const today = start.getTime()
  if (t >= today) return '今天'
  if (t >= today - DAY) return '昨天'
  if (t >= today - 7 * DAY) return '近 7 天'
  return '更早'
}

export function fmtSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${Math.round(bytes / 1024)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1).replace(/\.0$/, '')} MB`
  return `${(bytes / 1024 ** 3).toFixed(1).replace(/\.0$/, '')} GB`
}

/** Token counts the way people say them: 830, 5.2k, 24k, 1M. */
export function fmtTokens(n: number) {
  const short = (v: number, unit: string) => `${v < 10 ? +v.toFixed(1) : Math.round(v)}${unit}`
  return n < 1000 ? String(n) : n < 1_000_000 ? short(n / 1000, 'k') : short(n / 1_000_000, 'M')
}

export function fmtDuration(ms: number) {
  const s = Math.max(1, Math.round(ms / 1000))
  if (s < 60) return `${s} 秒`
  const m = Math.floor(s / 60)
  return s % 60 ? `${m} 分 ${s % 60} 秒` : `${m} 分钟`
}

export const clock = (t: number) => new Date(t).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })

let counter = 0
export const uid = (prefix = '') => `${prefix}${Date.now().toString(36).slice(-4)}${(++counter).toString(36)}${Math.random().toString(36).slice(2, 5)}`

export const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

/** A short label for a session: its title, its thread, or its id. */
export function sessionLabel(s: { id: string; title?: string; thread?: string }) {
  return s.title?.trim() || s.thread || `会话 ${s.id.slice(0, 8)}`
}

/** A Go time.Duration (nanoseconds) the way Go prints it: 1h30m, 10m, 45s. */
/** A Go duration as time.Duration.String writes it, without its zero units: "24h0m0s" → "24h", "1h30m0s" → "1h30m". */
export const fmtGoDuration = (d: string) => d.replace(/(\D)0s$/, '$1').replace(/(\D)0m$/, '$1')

/** "in 5 分钟" / "5 分钟前" for a time that may be in the future. */
export function relTime(t: number, now = Date.now()) {
  if (t <= now) return timeAgo(t, now)
  const d = t - now
  if (d < 60_000) return `${Math.max(1, Math.round(d / 1000))} 秒后`
  if (d < 3_600_000) return `${Math.round(d / 60_000)} 分钟后`
  if (d < 86_400_000) return `${Math.round(d / 3_600_000)} 小时后`
  return new Date(t).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

/** One line of a JSON value, for a table cell. */
export function oneLine(v: unknown, n = 120) {
  const s = (typeof v === 'string' ? v : JSON.stringify(v) ?? '').replace(/\s+/g, ' ').trim()
  return s.length > n ? s.slice(0, n - 1) + '…' : s
}
