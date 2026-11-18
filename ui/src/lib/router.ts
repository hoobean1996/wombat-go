import { useSyncExternalStore } from 'react'

const listeners = new Set<() => void>()
const notify = () => listeners.forEach(l => l())
window.addEventListener('popstate', notify)

const href = () => location.pathname + location.search

export function navigate(to: string, opts: { replace?: boolean } = {}) {
  if (to === href()) return
  if (opts.replace) history.replaceState(null, '', to)
  else history.pushState(null, '', to)
  notify()
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}

export type Route =
  | { name: 'home' }
  | { name: 'chat'; id: string }
  | { name: 'inbox' }
  | { name: 'outbox' }
  | { name: 'timers' }
  | { name: 'calls' }
  | { name: 'settings' }
  | { name: 'notfound' }

export function matchRoute(path: string): Route {
  if (path === '/' || path === '') return { name: 'home' }
  const chat = path.match(/^\/c\/([^/]+)\/?$/)
  if (chat) return { name: 'chat', id: decodeURIComponent(chat[1]) }
  for (const name of ['inbox', 'outbox', 'timers', 'calls', 'settings'] as const) {
    if (new RegExp(`^/${name}/?$`).test(path)) return { name }
  }
  return { name: 'notfound' }
}

export function useLocation() {
  const current = useSyncExternalStore(subscribe, href)
  const url = new URL(current, location.origin)
  return { path: url.pathname, query: url.searchParams, href: current }
}

/** Sets or removes one query parameter on the current path. */
export function setQuery(key: string, value: string | null, opts: { replace?: boolean } = {}) {
  const url = new URL(href(), location.origin)
  if (value === null) url.searchParams.delete(key)
  else url.searchParams.set(key, value)
  const qs = url.searchParams.toString()
  navigate(url.pathname + (qs ? `?${qs}` : ''), opts)
}
