import { useSyncExternalStore } from 'react'
import * as api from './api'
import { withoutChat } from './chat'
import { toast } from './toast'
import type { Health, ServerConfig, SessionInfo } from '../types'

/*
 * What every page shares: whether wombatd answers, what it offers, the list of
 * sessions, and how many outbox messages nobody has acknowledged yet.
 *
 * The session list is polled — wombatd has no stream across sessions, and a
 * list of a few dozen rows every few seconds costs nothing. The outbox is
 * long-polled, which is what GET /api/outbox?wait= is for, so a new reply or
 * question shows up at once wherever the user is — except a chat's own, which
 * its page shows (lib/chat.ts).
 */

export type Conn = 'connecting' | 'ok' | 'down' | 'unauthorized'

export interface AppState {
  conn: Conn
  error?: string
  /** Connected once, but the latest poll could not reach wombatd: it is probably restarting. */
  stale: boolean
  health: Health | null
  config: ServerConfig | null
  sessions: SessionInfo[]
  sessionsLoaded: boolean
  /** Outbox messages still pending — not acknowledged, not delivered by a webhook — leaving out chat replies (lib/chat.ts). */
  outboxPending: number
  /** Changes whenever new outbox messages arrive, so pages showing the outbox refetch. */
  outboxTick: number
}

let state: AppState = {
  conn: 'connecting', stale: false, health: null, config: null, sessions: [], sessionsLoaded: false, outboxPending: 0, outboxTick: 0,
}
const listeners = new Set<() => void>()
function set(patch: Partial<AppState>) {
  state = { ...state, ...patch }
  listeners.forEach(l => l())
}
const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } }

export const useApp = <T,>(pick: (s: AppState) => T): T => useSyncExternalStore(subscribe, () => pick(state))
export const getApp = () => state

// ── the loops ────────────────────────────────────────────────────────────

let generation = 0 // bumped on every (re)start, so a loop of an older one stops
let abort: AbortController | null = null
let pollTimer: ReturnType<typeof setInterval> | null = null
let retryTimer: ReturnType<typeof setTimeout> | null = null

function stop() {
  generation++
  abort?.abort()
  abort = null
  if (pollTimer) clearInterval(pollTimer)
  if (retryTimer) clearTimeout(retryTimer)
  pollTimer = retryTimer = null
}

/** Connects (again): after the first load, and after the token changes. */
export function start() {
  stop()
  const gen = generation
  set({ conn: 'connecting', error: undefined })
  void (async () => {
    try {
      const [health, config] = await Promise.all([api.health(), api.config()])
      if (gen !== generation) return
      set({ conn: 'ok', health, config })
    } catch (e) {
      if (gen !== generation) return
      if (e instanceof api.ApiError && e.status === 401) return set({ conn: 'unauthorized', error: undefined })
      set({ conn: 'down', error: api.errorText(e) })
      retryTimer = setTimeout(() => { if (gen === generation) start() }, 3000)
      return
    }
    await refreshSessions()
    pollTimer = setInterval(() => { if (!document.hidden) void refreshSessions() }, 3000)
    abort = new AbortController()
    void outboxLoop(gen, abort.signal)
  })()
}

window.addEventListener(api.UNAUTHORIZED, () => {
  if (state.conn === 'unauthorized') return
  stop()
  set({ conn: 'unauthorized' })
})

export async function refreshSessions() {
  try {
    const sessions = await api.listSessions()
    set({ sessions, sessionsLoaded: true, stale: false })
  } catch (e) {
    // The next poll tries again; a 401 is handled above.
    if (e instanceof api.ApiError && e.status === 0) set({ stale: true })
  }
}

async function refreshPending() {
  try {
    const pending = await withoutChat(await api.listOutboxAll({ status: 'pending' }))
    set({ outboxPending: pending.length })
  } catch { /* shown on the next tick */ }
}

const KIND_LABEL: Record<string, string> = { reply: '回复', question: '提问', approval: '等待确认', error: '出错', message: '消息' }

async function outboxLoop(gen: number, signal: AbortSignal) {
  const since = Date.now() - 5000 // older messages were there before this page; do not toast them
  let last = 0
  try {
    const all = await api.listOutbox({ after: 0, limit: 1000 }, signal)
    last = all.reduce((m, x) => Math.max(m, x.id), 0)
  } catch { /* start from 0; the first poll catches up */ }
  await refreshPending()
  while (gen === generation) {
    try {
      const fresh = await api.listOutbox({ after: last, wait: '30s', limit: 200 }, signal)
      if (gen !== generation) return
      if (!fresh.length) continue
      last = fresh.reduce((m, x) => Math.max(m, x.id), last)
      set({ outboxTick: state.outboxTick + 1 })
      void refreshPending()
      void refreshSessions()
      const viewing = location.pathname.match(/^\/c\/([^/]+)/)?.[1]
      // A chat's own replies are on its page, not news.
      for (const m of await withoutChat(fresh, signal)) {
        if (Date.parse(m.created) < since || m.session_id === viewing) continue
        toast(`${KIND_LABEL[m.kind] ?? m.kind}：${m.text.replace(/\s+/g, ' ').slice(0, 60)}`)
      }
    } catch {
      if (signal.aborted || gen !== generation) return
      await new Promise(r => setTimeout(r, 3000)) // wombatd went away; start() notices on its own via the poll
    }
  }
}

/** After acknowledging outbox messages: the count drops without waiting for the next arrival. */
export const outboxChanged = () => { set({ outboxTick: state.outboxTick + 1 }); void refreshPending() }
