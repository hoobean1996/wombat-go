import { useSyncExternalStore } from 'react'

/*
 * An in-page confirmation, for actions that cannot be undone (deleting a
 * session, cancelling a timer). In-page rather than window.confirm, which
 * blocks the page — and its event stream — until it is answered.
 */

export interface ConfirmRequest {
  id: number
  title: string
  message: string
  confirmLabel: string
  resolve: (ok: boolean) => void
}

let current: ConfirmRequest | null = null
let queue: ConfirmRequest[] = []
let next = 0
const listeners = new Set<() => void>()
const notify = () => listeners.forEach(l => l())

export function confirmDialog(title: string, message: string, confirmLabel = '确认'): Promise<boolean> {
  return new Promise(resolve => {
    const req = { id: ++next, title, message, confirmLabel, resolve }
    if (current) queue = [...queue, req]
    else current = req
    notify()
  })
}

export function settleConfirm(ok: boolean) {
  if (!current) return
  current.resolve(ok)
  current = queue[0] ?? null
  queue = queue.slice(1)
  notify()
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}

export const useConfirm = () => useSyncExternalStore(subscribe, () => current)
