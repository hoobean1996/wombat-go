import type { Message } from '../types'

/*
 * From the transcript (pkg/core/llm messages) to what the thread draws.
 */

export interface Step {
  id: string
  name: string
  input?: unknown
  status: 'running' | 'done' | 'failed' | 'waiting' | 'denied' | 'pending'
  output?: string
  error?: string
  ms?: number
  depth?: number
}

/** One exchange: what a person (or an event) said, and everything the agent did about it. */
export interface Exchange {
  prompt: string | null
  steps: Step[]
  text: string
  thinking: number
}

/** Groups a transcript into exchanges. Tool results are folded into the call they answer. */
export function exchanges(messages: Message[]): Exchange[] {
  const out: Exchange[] = []
  const byId = new Map<string, Step>()
  let cur: Exchange | null = null
  const texts: string[][] = []
  for (const m of messages) {
    if (m.role === 'user') {
      const said = m.content.filter(b => b.type === 'text').map(b => (b as { text: string }).text).join('\n').trim()
      for (const b of m.content) {
        if (b.type !== 'tool_result') continue
        const st = byId.get(b.tool_use_id)
        if (st) Object.assign(st, { status: b.is_error ? 'failed' : 'done', output: b.content })
      }
      if (said) {
        cur = { prompt: said, steps: [], text: '', thinking: 0 }
        out.push(cur)
        texts.push([])
      }
      continue
    }
    if (!cur) {
      cur = { prompt: null, steps: [], text: '', thinking: 0 }
      out.push(cur)
      texts.push([])
    }
    for (const b of m.content) {
      if (b.type === 'text' && b.text.trim()) texts[texts.length - 1].push(b.text)
      else if (b.type === 'thinking') cur.thinking++
      else if (b.type === 'tool_use') {
        const st: Step = { id: b.id, name: b.name, input: b.input, status: 'pending' }
        byId.set(b.id, st)
        cur.steps.push(st)
      }
    }
  }
  out.forEach((x, i) => { x.text = texts[i].join('\n\n') })
  return out
}
