import * as api from './api'
import type { InboxEvent, OutboxMessage } from '../types'

/*
 * What the Inbox and Outbox pages leave out: the conversation itself.
 *
 * A message typed into a session's chat is kind "message" from source "api" —
 * what the composer sends, and anything else posted to
 * POST /api/sessions/{id}/messages — and a decision on one of its approval
 * cards is kind "approval", from "api" too. The agent's answers to those
 * messages, the requests for decisions among them, carry the messages'
 * events in in_reply_to. All of it is on the session's own page; listed
 * again in the inbox and the outbox, every exchange of every chat would bury
 * what those pages are for: webhooks, timers, chats relayed from elsewhere,
 * and what the agent says on its own.
 *
 * Only the pages filter. The outbox keeps every message, pending until acked,
 * for any other client reading it.
 */

/** A message typed into a session's chat, or a decision or a result posted there, rather than an event from outside. */
export const isChatEvent = (e: Pick<InboxEvent, 'kind' | 'source'>) =>
  (e.kind === 'message' || e.kind === 'approval' || e.kind === 'result') && (e.source === 'api' || e.source === '')

// Inbox event id → whether it is a chat message. What an event is and where
// it came from never change, so what is learnt here never goes stale.
const chatEvents = new Map<number, boolean>()

/**
 * An outbox message that answers chat messages and nothing else: the reply,
 * the question, the request for a decision or the error the session's own
 * page already shows. It needs the events it answers learnt first; see
 * withoutChat.
 */
export const isChatReply = (m: OutboxMessage) =>
  !!m.in_reply_to?.length && m.in_reply_to.every(id => chatEvents.get(id) === true)

/** The messages that are not part of a chat, once the events they answer are known. */
export async function withoutChat(messages: OutboxMessage[], signal?: AbortSignal): Promise<OutboxMessage[]> {
  const unknown = (m: OutboxMessage) => (m.in_reply_to ?? []).some(id => !chatEvents.has(id))
  // An outbox message names the events it answers but not what they were:
  // the inbox says, a session at a time.
  const sessions = new Set(messages.filter(unknown).map(m => m.session_id))
  await Promise.all([...sessions].map(async s => {
    for (const e of await api.listInbox({ session: s, limit: 1000 }, signal)) chatEvents.set(e.id, isChatEvent(e))
  }))
  // Still unknown: older than its session's newest 1000 events. Shown, rather
  // than looked up again on every refresh.
  for (const m of messages) for (const id of m.in_reply_to ?? []) if (!chatEvents.has(id)) chatEvents.set(id, false)
  return messages.filter(m => !isChatReply(m))
}
