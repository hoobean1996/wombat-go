/*
 * The shapes wombatd's HTTP API sends and takes, written by hand from the Go
 * types in pkg/daemon, pkg/httpapi and pkg/core/llm. A field added there has to
 * be added here; nothing checks that the two agree.
 */

// ── sessions ─────────────────────────────────────────────────────────────

/** pkg/daemon.State */
export type SessionState = 'idle' | 'running' | 'waiting' | 'approving' | 'awaiting' | 'done' | 'failed'

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

/** pkg/daemon.Approval: a tool call waiting for a person's decision — or, with awaiting, for a result from outside. */
export interface Approval {
  use_id: string
  tool: string
  reason: string
  input?: unknown
  since: string
  /** It waits for a result (POST /api/sessions/{id}/results/{use_id}), not a decision; handle names what for. */
  awaiting?: boolean
  handle?: string
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

export interface TurnStarted { type: 'turn_started'; turn: number; prompt?: string; events: number[]; resumed?: boolean }
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
  awaiting?: Approval[]
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
/** A turn stopped between steps at the end of its slice (daemon.slice), and carries on in the next: still running. */
export interface TurnSuspended { type: 'turn_suspended'; turn: number; spend: Spend }
export interface CompletionChecked { type: 'completion_checked'; attempt: number; accepted: boolean; feedback?: string; gave_up?: boolean; error?: string; ms: number }

/** Every event wombatd may send; the ones the UI draws are typed above. */
export type StreamEvent =
  | TurnStarted | TurnEnded | TextDelta | ReasoningDelta | IterStart | ToolStart | ToolDone
  | PermissionRequested | PermissionDecided | SubagentEvent | SteerSent | CompletionChecked | TurnSuspended
  | { type: 'subagent_start' | 'subagent_end' | 'llm_start' | 'llm_done' | 'llm_retry' | 'spend' | 'steered' | 'tool_args_delta' | 'tool_result'
      | 'empty_turn' | 'turn_truncated' | 'context_trimmed' | 'context_overflow' | 'transcript_compacted' | 'reviewed' | 'tool_recovered'; [k: string]: unknown }

/** The `event:` names to listen for on the stream: EventSource only hands named events to their own listeners. */
export const STREAM_EVENT_TYPES = [
  'turn_started', 'turn_ended', 'turn_suspended', 'steer_sent', 'iter_start', 'turn_truncated', 'llm_start', 'llm_done', 'llm_retry',
  'text_delta', 'reasoning_delta', 'tool_args_delta', 'tool_start', 'tool_done', 'tool_result', 'spend', 'steered',
  'empty_turn', 'context_trimmed', 'context_overflow', 'transcript_compacted', 'completion_checked',
  'permission_requested', 'permission_decided', 'reviewed', 'subagent_start', 'subagent_event', 'subagent_end',
  'tool_recovered',
] as const

// ── the inbox, the outbox and timers ─────────────────────────────────────

export type InboxStatus = 'pending' | 'running' | 'done' | 'failed'

/** pkg/daemon.Event: something that wakes a session. */
export interface InboxEvent {
  id: number
  session_id: string
  kind: string
  source: string
  /** Who, at the source, sent it: a person in a relayed chat, whoever decided a call. */
  from?: string
  dedup_key?: string
  payload?: unknown
  status: InboxStatus
  attempts: number
  error?: string
  created: string
  not_before: string
}

export type OutboxStatus = 'pending' | 'delivered' | 'failed'
export type OutboxKind = 'reply' | 'question' | 'approval' | 'awaiting' | 'error' | 'message'

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

/** pkg/daemon.Timer. `every` is a Go duration as Timer.MarshalJSON writes it: "24h0m0s". */
export interface Timer {
  id: number
  session_id: string
  note: string
  due: string
  every?: string
  status: TimerStatus
  created_by: string
  created: string
  fired: number
}

// ── the ledger: GET /api/tool-calls ──────────────────────────────────────

/** pkg/core/tool.State: where a tool call is. succeeded, failed, denied and cancelled are final. */
export type CallState = 'pending' | 'running' | 'succeeded' | 'failed' | 'denied' | 'deferred' | 'waiting' | 'interrupted' | 'cancelled'

/** pkg/daemon.Origin: an inbox event — here, the decision on a call. */
export interface Origin { event: number; kind: string; source: string; from?: string; at: string }

/** pkg/daemon.ToolCall: one call a session's turn made, as the ledger keeps it. */
export interface ToolCall {
  session_id: string
  /** The call is the index-th of the session's step-th model reply: steps are numbered across the session. */
  step: number
  index: number
  turn: number
  /** For a call a sub-agent made: the key ("<session>/<step>.<index>") of the call that started it, and which of its sub-agents. */
  parent?: string
  run?: number
  use_id: string
  tool: string
  input?: unknown
  state: CallState
  /** How many times it started to run. */
  attempts: number
  output?: string
  is_error?: boolean
  tags?: string[]
  /** Why it waits — for a decision, a result — or why it did not run; handle names what a waiting call waits for. */
  reason?: string
  handle?: string
  decision?: { allow: boolean; by: Origin }
  owner?: string
  created: string
  started?: string
  finished?: string
}

// ── the server ───────────────────────────────────────────────────────────

/** pkg/httpapi.FileInfo: a file in a session's workspace, and a URL (no token needed) to read it. */
export interface FileInfo {
  path: string
  name: string
  size: number
  modified: string
  type: string
  url: string
}

/** pkg/httpapi.Capabilities: GET /api/config */
export interface ServerConfig {
  version?: string
  default_model?: string
  permission_modes: string[]
  approvals: boolean
  /** Whether a session may be given permission "full": every tool, nobody asked (tools.full_access). */
  full_access?: boolean
  /** Whether GET /api/sessions/{id}/file shows the files in a session's workspace. */
  files?: boolean
  tools: { name: string; description?: string; category?: string; read_only: boolean }[]
}

export interface Health {
  status: string
  version: string
  uptime_sec: number
}
