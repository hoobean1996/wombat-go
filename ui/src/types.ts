/*
 * The shapes wombatd's HTTP API sends and takes, written by hand from the Go
 * types in pkg/daemon, pkg/httpapi and pkg/core/llm. A field added there has to
 * be added here; nothing checks that the two agree.
 */

// ── sessions ─────────────────────────────────────────────────────────────

/** pkg/daemon.State */
export type SessionState = 'idle' | 'running' | 'waiting' | 'approving' | 'done' | 'failed'

/** pkg/daemon.Spend */
export interface Spend {
  cost_usd: number
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  calls: number
  elapsed_sec: number
}

/** pkg/daemon.SessionOptions: what a client may ask for when it creates a session. */
export interface SessionOptions {
  model?: string
  permission?: string
  workspace?: string
  max_iters?: number
  title?: string
}

/** pkg/daemon.SessionInfo */
export interface SessionInfo {
  id: string
  title?: string
  thread?: string
  state: SessionState
  turns: number
  events: number // the next sequence number of the session's event stream
  created: string
  updated: string
  options: SessionOptions
  spend: Spend
  outcome?: 'answer' | 'paused' | 'submitted' | 'deferred'
  answer?: string // the last reply, or the question the session is waiting on
  error?: string
  error_kind?: string
}

/** pkg/daemon.Approval: a tool call waiting for a person's decision. */
export interface Approval {
  use_id: string
  tool: string
  reason: string
  input?: unknown
  since: string
}

// ── the transcript (pkg/core/llm's persistence shape) ────────────────────

export type ContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input?: unknown }
  | { type: 'tool_result'; tool_use_id: string; content?: string; is_error?: boolean }
  | { type: 'thinking'; text: string; signature?: string }
  | { type: 'media'; media_type: string; name?: string }

export interface Message {
  role: 'user' | 'assistant'
  content: ContentBlock[]
}

// ── the event stream: GET /api/sessions/{id}/events ──────────────────────

export interface TurnStarted { type: 'turn_started'; turn: number; prompt?: string; events: number[] }
export interface TurnEnded {
  type: 'turn_ended'
  turn: number
  state: SessionState
  events: number[]
  outcome?: string
  answer?: string
  question?: string
  tool?: string
  approvals?: Approval[]
  error?: string
  error_kind?: string
  retry_at?: string
  spend: Spend
  undelivered?: string[]
}
export interface TextDelta { type: 'text_delta'; text: string }
export interface ReasoningDelta { type: 'reasoning_delta'; text: string }
export interface IterStart { type: 'iter_start'; n: number; max: number }
export interface ToolStart { type: 'tool_start'; use_id: string; name: string; category?: string; input?: unknown }
export interface ToolDone { type: 'tool_done'; use_id: string; name: string; ok: boolean; output?: string; error?: string; ms: number }
export interface PermissionRequested { type: 'permission_requested'; use_id: string; tool: string; reason?: string; input?: unknown }
export interface PermissionDecided { type: 'permission_decided'; use_id: string; tool: string; allowed: boolean; reason?: string; source?: string }
export interface SubagentEvent { type: 'subagent_event'; name: string; depth: number; inner: StreamEvent }
export interface SteerSent { type: 'steer_sent'; turn: number; prompt: string }
export interface CompletionChecked { type: 'completion_checked'; attempt: number; accepted: boolean; feedback?: string; gave_up?: boolean; error?: string; ms: number }

/** Every event wombatd may send; the ones the UI draws are typed above. */
export type StreamEvent =
  | TurnStarted | TurnEnded | TextDelta | ReasoningDelta | IterStart | ToolStart | ToolDone
  | PermissionRequested | PermissionDecided | SubagentEvent | SteerSent | CompletionChecked
  | { type: 'subagent_start' | 'subagent_end' | 'llm_start' | 'llm_done' | 'llm_retry' | 'spend' | 'steered' | 'tool_args_delta' | 'tool_result'
      | 'empty_turn' | 'turn_truncated' | 'context_trimmed' | 'context_overflow' | 'transcript_compacted' | 'reviewed'; [k: string]: unknown }

/** The `event:` names to listen for on the stream: EventSource only hands named events to their own listeners. */
export const STREAM_EVENT_TYPES = [
  'turn_started', 'turn_ended', 'steer_sent', 'iter_start', 'turn_truncated', 'llm_start', 'llm_done', 'llm_retry',
  'text_delta', 'reasoning_delta', 'tool_args_delta', 'tool_start', 'tool_done', 'tool_result', 'spend', 'steered',
  'empty_turn', 'context_trimmed', 'context_overflow', 'transcript_compacted', 'completion_checked',
  'permission_requested', 'permission_decided', 'reviewed', 'subagent_start', 'subagent_event', 'subagent_end',
] as const

// ── the inbox, the outbox and timers ─────────────────────────────────────

export type InboxStatus = 'pending' | 'running' | 'done' | 'failed'

/** pkg/daemon.Event: something that wakes a session. */
export interface InboxEvent {
  id: number
  session_id: string
  kind: string
  source: string
  dedup_key?: string
  payload?: unknown
  status: InboxStatus
  attempts: number
  error?: string
  created: string
  not_before: string
}

export type OutboxStatus = 'pending' | 'delivered' | 'failed'
export type OutboxKind = 'reply' | 'question' | 'approval' | 'error' | 'message'

/** pkg/daemon.Message: something the agent said. */
export interface OutboxMessage {
  id: number
  session_id: string
  thread?: string
  kind: OutboxKind | string
  text: string
  data?: unknown
  in_reply_to?: number[]
  status: OutboxStatus
  attempts: number
  error?: string
  created: string
  delivered?: string
}

export type TimerStatus = 'active' | 'fired' | 'cancelled'

/** pkg/daemon.Timer. `every` is a Go time.Duration: nanoseconds. */
export interface Timer {
  id: number
  session_id: string
  note: string
  due: string
  every?: number
  status: TimerStatus
  created_by: string
  created: string
  fired: number
}

// ── the server ───────────────────────────────────────────────────────────

/** pkg/httpapi.Capabilities: GET /api/config */
export interface ServerConfig {
  version?: string
  default_model?: string
  permission_modes: string[]
  approvals: boolean
  tools: { name: string; description?: string; category?: string; read_only: boolean }[]
}

export interface Health {
  status: string
  version: string
  uptime_sec: number
}
