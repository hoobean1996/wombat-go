import { useSyncExternalStore } from 'react'

export interface Toast { id: number; text: string }

let toasts: Toast[] = []
let next = 0
const listeners = new Set<() => void>()
const notify = () => listeners.forEach(l => l())

export function toast(text: string) {
  const t = { id: ++next, text }
  toasts = [...toasts, t]
  notify()
  setTimeout(() => {
    toasts = toasts.filter(x => x.id !== t.id)
    notify()
  }, 2200)
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}

export const useToasts = () => useSyncExternalStore(subscribe, () => toasts)
