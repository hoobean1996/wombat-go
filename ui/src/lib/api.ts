import type { Approval, Health, InboxEvent, Message, OutboxMessage, ServerConfig, SessionInfo, SessionOptions, Timer } from '../types'

/*
 * wombatd's HTTP API. The page is served by wombatd itself, so every path is
 * relative to it; in development Vite forwards /api to wombatd (vite.config.ts).
 *
 * With server.token set, every request carries it as a bearer token. A browser
 * keeps it in localStorage, which any script on this origin can read — fine for
 * a page that only ever runs wombatd's own code, and the reason this page
 * renders replies without HTML (components/Markdown.tsx).
 */

const TOKEN_KEY = 'wombat.token'

export function getToken(): string {
  try { return localStorage.getItem(TOKEN_KEY) ?? '' } catch { return '' }
}

export function setToken(token: string) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch { /* private window: the token lasts until the page reloads */ }
}

/** An error the server described: kind is the bounded token of its error envelope. */
export class ApiError extends Error {
  status: number
  kind: string
  constructor(status: number, kind: string, message: string) {
    super(message)
    this.status = status
    this.kind = kind
  }
}

/** Fired on any 401, so the app can ask for the token. */
export const UNAUTHORIZED = 'wombat:unauthorized'

async function call<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  let res: Response
  try {
    res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal, cache: 'no-store' })
  } catch (e) {
    if (signal?.aborted) throw e
    throw new ApiError(0, 'network', '连不上 wombatd')
  }
  if (res.status === 401) window.dispatchEvent(new Event(UNAUTHORIZED))
  if (res.status === 204) return undefined as T
  const text = await res.text()
  let data: unknown = undefined
  try { data = text ? JSON.parse(text) : undefined } catch { /* not JSON: a proxy's error page, say */ }
  if (!res.ok) {
    const env = (data as { error?: { kind?: string; message?: string } } | undefined)?.error
    throw new ApiError(res.status, env?.kind ?? 'http', env?.message ?? `${res.status} ${res.statusText}`)
  }
  return data as T
}

const enc = encodeURIComponent
const qs = (q: Record<string, string | number | undefined>) => {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(q)) if (v !== undefined && v !== '') p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ''
}

// ── the server ───────────────────────────────────────────────────────────
export const health = () => call<Health>('GET', '/api/health')
export const config = () => call<ServerConfig>('GET', '/api/config')

// ── sessions ─────────────────────────────────────────────────────────────
export const listSessions = () => call<{ sessions: SessionInfo[] }>('GET', '/api/sessions').then(r => r.sessions ?? [])
export const getSession = (id: string) => call<SessionInfo>('GET', `/api/sessions/${enc(id)}`)
export const createSession = (prompt: string, opts: SessionOptions = {}) => call<SessionInfo>('POST', '/api/sessions', { prompt, ...opts })
export const deleteSession = (id: string) => call<void>('DELETE', `/api/sessions/${enc(id)}`)
/** From the session's next turn on; "full" also allows what it is waiting on, "" is the server's default. */
export const setPermission = (id: string, permission: string) => call<SessionInfo>('PATCH', `/api/sessions/${enc(id)}`, { permission })
export const getMessages = (id: string) => call<{ messages: Message[] }>('GET', `/api/sessions/${enc(id)}/messages`).then(r => r.messages ?? [])
/** A message for the session's next turn (queued if a turn is running). */
export const sendMessage = (id: string, prompt: string) => call<unknown>('POST', `/api/sessions/${enc(id)}/messages`, { prompt })
/** A message for the turn that is running now. */
export const steer = (id: string, prompt: string) => call<unknown>('POST', `/api/sessions/${enc(id)}/steer`, { prompt })
export const listApprovals = (id: string) => call<{ approvals: Approval[] }>('GET', `/api/sessions/${enc(id)}/approvals`).then(r => r.approvals ?? [])
export const decide = (id: string, useId: string, allow: boolean) => call<unknown>('POST', `/api/sessions/${enc(id)}/approvals/${enc(useId)}`, { allow })

/** The URL of a session's event stream. EventSource cannot set a header, so the token rides in the query (only here). */
export function eventsURL(id: string, from: number): string {
  const token = getToken()
  return `/api/sessions/${enc(id)}/events${qs({ from, access_token: token || undefined })}`
}

// ── the inbox ────────────────────────────────────────────────────────────
export const listInbox = (q: { session?: string; status?: string; limit?: number } = {}, signal?: AbortSignal) =>
  call<{ events: InboxEvent[] }>('GET', `/api/inbox${qs(q)}`, undefined, signal).then(r => r.events ?? [])
export interface NewEvent { session_id?: string; thread?: string; kind: string; source?: string; from?: string; dedup_key?: string; payload?: unknown }
export const postEvent = (e: NewEvent) => call<{ event: InboxEvent; session: SessionInfo; duplicate?: boolean }>('POST', '/api/events', e)

// ── the outbox ───────────────────────────────────────────────────────────
export const listOutbox = (q: { after?: number; wait?: string; session?: string; status?: string; limit?: number } = {}, signal?: AbortSignal) =>
  call<{ messages: OutboxMessage[] }>('GET', `/api/outbox${qs(q)}`, undefined, signal).then(r => r.messages ?? [])
/** Every message a filter selects, oldest first. The endpoint pages by id, at most 1000 a page. */
export async function listOutboxAll(q: { session?: string; status?: string } = {}, signal?: AbortSignal) {
  const all: OutboxMessage[] = []
  for (let after = 0; ;) {
    const page = await listOutbox({ ...q, after, limit: 1000 }, signal)
    all.push(...page)
    if (page.length < 1000) return all
    after = page[page.length - 1].id
  }
}
export const ack = (id: number) => call<void>('POST', `/api/outbox/${id}/ack`)

// ── timers ───────────────────────────────────────────────────────────────
export const listTimers = (q: { session?: string; status?: string } = {}) =>
  call<{ timers: Timer[] }>('GET', `/api/timers${qs(q)}`).then(r => r.timers ?? [])
export interface NewTimer { session_id: string; note: string; in?: string; at?: string; every?: string }
export const addTimer = (t: NewTimer) => call<Timer>('POST', '/api/timers', t)
export const cancelTimer = (id: number) => call<void>('DELETE', `/api/timers/${id}`)

export const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e))
