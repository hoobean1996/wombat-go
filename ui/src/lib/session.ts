import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'
import { STREAM_EVENT_TYPES, type Approval, type InboxEvent, type Message, type SessionInfo, type StreamEvent } from '../types'

/*
 * One session as the chat page shows it.
 *
 * The transcript (GET .../messages) is what has happened: wombatd commits it
 * when a turn ends, so it never holds half a turn. The turn in progress is
 * drawn from the event stream instead — the prompt, the text as it streams,
 * each tool call as it starts and finishes — and dropped when the turn ends,
 * at which point the transcript is fetched again and has it. The two never
 * show the same thing twice.
 */

export type LivePart =
  | { kind: 'text'; text: string }
  | { kind: 'think'; text: string }
  | { kind: 'tool'; id: string; name: string; input?: unknown; status: 'running' | 'done' | 'failed' | 'waiting' | 'denied'; output?: string; error?: string; ms?: number; depth: number }
  | { kind: 'note'; text: string; bad?: boolean }

export interface LiveTurn {
  turn: number
  prompt: string
  parts: LivePart[]
  iter?: { n: number; max: number }
}

function appendText(parts: LivePart[], kind: 'text' | 'think', text: string): LivePart[] {
  const last = parts.at(-1)
  if (last && last.kind === kind) return [...parts.slice(0, -1), { ...last, text: last.text + text }]
  return [...parts, { kind, text }]
}

function patchTool(parts: LivePart[], id: string, patch: Partial<Extract<LivePart, { kind: 'tool' }>>): LivePart[] {
  return parts.map(p => (p.kind === 'tool' && p.id === id ? { ...p, ...patch } : p))
}

/** Folds one stream event into the turn in progress. */
export function apply(live: LiveTurn | null, e: StreamEvent, depth = 0): LiveTurn | null {
  if (e.type === 'turn_started') return { turn: e.turn, prompt: e.prompt ?? '', parts: [] }
  if (e.type === 'turn_ended') return null
  if (!live) return live // events of a turn that started before we were watching
  switch (e.type) {
    case 'text_delta':
      return depth ? live : { ...live, parts: appendText(live.parts, 'text', e.text) }
    case 'reasoning_delta':
      return depth ? live : { ...live, parts: appendText(live.parts, 'think', e.text) }
    case 'iter_start':
      return depth ? live : { ...live, iter: { n: e.n, max: e.max } }
    case 'tool_start':
      return { ...live, parts: [...live.parts, { kind: 'tool', id: e.use_id, name: e.name, input: e.input, status: 'running', depth }] }
    case 'tool_done':
      return { ...live, parts: patchTool(live.parts, e.use_id, { status: e.ok ? 'done' : 'failed', output: e.output, error: e.error, ms: e.ms }) }
    case 'permission_requested':
      return live.parts.some(p => p.kind === 'tool' && p.id === e.use_id)
        ? { ...live, parts: patchTool(live.parts, e.use_id, { status: 'waiting' }) }
        : { ...live, parts: [...live.parts, { kind: 'tool', id: e.use_id, name: e.tool, input: e.input, status: 'waiting', depth }] }
    case 'permission_decided':
      return { ...live, parts: patchTool(live.parts, e.use_id, { status: e.allowed ? 'running' : 'denied' }) }
    case 'subagent_event':
      return apply(live, e.inner, Math.max(1, e.depth))
    case 'subagent_start':
      return { ...live, parts: [...live.parts, { kind: 'note', text: `子 agent「${String(e.name ?? '')}」开始工作` }] }
    case 'steer_sent':
      return { ...live, parts: [...live.parts, { kind: 'note', text: `插话：${e.prompt}` }] }
    case 'completion_checked':
      return e.accepted ? live : { ...live, parts: [...live.parts, { kind: 'note', bad: true, text: e.gave_up ? '完成检查仍未通过，已停止重试' : `完成检查未通过（第 ${e.attempt} 次），继续工作` }] }
    default:
      return live
  }
}

export interface SessionView {
  info: SessionInfo | null
  messages: Message[]
  approvals: Approval[]
  /** Messages waiting in the inbox for this session's next turn. */
  queued: InboxEvent[]
  live: LiveTurn | null
  missing: boolean
  error?: string
  /** Fetches the session again (after sending, deciding, ...). */
  refresh: () => void
}

export function useSessionView(id: string): SessionView {
  const [info, setInfo] = useState<SessionInfo | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [queued, setQueued] = useState<InboxEvent[]>([])
  const [live, setLive] = useState<LiveTurn | null>(null)
  const [missing, setMissing] = useState(false)
  const [error, setError] = useState<string>()
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const loadRef = useRef<() => Promise<SessionInfo | undefined>>(async () => undefined)

  const refresh = useCallback(() => {
    clearTimeout(timer.current)
    timer.current = setTimeout(() => { void loadRef.current().catch(() => {}) }, 120)
  }, [])

  useEffect(() => {
    // The page is remounted for every session (App keys it by id), so there
    // is no previous session's state to clear here.
    const run = { alive: true }
    const stream: { es?: EventSource } = {}

    const load = async () => {
      const [i, m, q] = await Promise.all([api.getSession(id), api.getMessages(id), api.listInbox({ session: id, status: 'pending', limit: 50 })])
      const ap = i.state === 'approving' ? await api.listApprovals(id) : []
      if (!run.alive) return undefined
      setInfo(i); setMessages(m); setApprovals(ap); setQueued(q.filter(e => e.kind === 'message').reverse())
      return i
    }
    loadRef.current = load

    void (async () => {
      let i: SessionInfo | undefined
      try {
        i = await load()
      } catch (e) {
        if (!run.alive) return
        if (e instanceof api.ApiError && e.status === 404) setMissing(true)
        else setError(api.errorText(e))
        return
      }
      if (!i || !run.alive) return
      // A running turn is drawn from its first event, so start at the
      // beginning; otherwise only what happens from now on is new.
      const es = new EventSource(api.eventsURL(id, i.state === 'running' ? 0 : i.events))
      stream.es = es
      const on = (ev: MessageEvent<string>) => {
        let e: StreamEvent
        try { e = JSON.parse(ev.data) as StreamEvent } catch { return }
        setLive(l => apply(l, e))
        if (e.type === 'turn_started' || e.type === 'turn_ended' || e.type === 'permission_requested') refresh()
      }
      for (const t of STREAM_EVENT_TYPES) es.addEventListener(t, on as EventListener)
      // EventSource reconnects by itself, resuming with Last-Event-ID.
    })()

    return () => {
      run.alive = false
      stream.es?.close()
      clearTimeout(timer.current)
    }
  }, [id, refresh])

  return { info, messages, approvals, queued, live, missing, error, refresh }
}
