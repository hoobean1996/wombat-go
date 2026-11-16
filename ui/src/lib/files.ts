import { createContext, useContext } from 'react'

/*
 * Files the agent mentions — in a reply, in a tool call — open in the panel
 * beside the conversation (components/FilePanel.tsx). Inside a session page a
 * FileOpener is provided; anywhere else there is none, and a path is just text.
 */

export type FileKind = 'html' | 'image' | 'pdf' | 'video' | 'audio' | 'markdown' | 'csv' | 'json' | 'code' | 'text' | 'other'

const KINDS: Record<string, FileKind> = {
  html: 'html', htm: 'html',
  png: 'image', jpg: 'image', jpeg: 'image', gif: 'image', webp: 'image', svg: 'image', avif: 'image', bmp: 'image', ico: 'image',
  pdf: 'pdf',
  mp4: 'video', webm: 'video', mov: 'video', m4v: 'video',
  mp3: 'audio', wav: 'audio', ogg: 'audio', m4a: 'audio', flac: 'audio',
  md: 'markdown', markdown: 'markdown',
  csv: 'csv', tsv: 'csv',
  json: 'json',
  txt: 'text', log: 'text',
}
const CODE = 'go mod sum ts tsx js jsx mjs cjs py rs java kt c h cc cpp hpp rb php swift sh bash zsh sql yaml yml toml ini cfg conf proto graphql lua r scala vue svelte css scss less xml'.split(' ')
for (const ext of CODE) KINDS[ext] = 'code'

const extOf = (path: string) => path.split('/').pop()?.match(/\.([A-Za-z0-9]+)$/)?.[1]?.toLowerCase() ?? ''

export const kindOf = (path: string): FileKind => KINDS[extOf(path)] ?? 'other'

export const KIND_LABEL: Record<FileKind, string> = {
  html: 'HTML', image: '图片', pdf: 'PDF', video: '视频', audio: '音频', markdown: 'Markdown', csv: '表格', json: 'JSON', code: '代码', text: '文本', other: '文件',
}
export const KIND_ICON: Record<FileKind, string> = {
  html: 'globe', image: 'image', pdf: 'file', video: 'video', audio: 'music', markdown: 'file', csv: 'table', json: 'code', code: 'code', text: 'file', other: 'file',
}

/**
 * Whether text the agent wrote is a path worth opening: one word, no scheme,
 * ending in an extension the panel knows. "a.html", "./out/report.md" and
 * "/Users/me/x/chart.png" are; "go test", "v1.2" and "https://a.com/x.html"
 * are not.
 */
export function looksLikeFile(text: string): boolean {
  const t = text.trim()
  if (!t || t.length > 300 || /\s/.test(t) || /^[a-z][a-z0-9+.-]*:/i.test(t) || t.startsWith('~')) return false
  if (!/^[\w./\-@+%一-鿿]+$/.test(t)) return false
  const ext = extOf(t)
  return ext !== '' && ext in KINDS && !/^\d+$/.test(ext)
}

/** Joins a link written in a file to the folder that file is in: "../a.png" from "docs/x/r.md" is "docs/a.png". */
export function resolveFrom(base: string, href: string): string {
  if (href.startsWith('/')) return href
  const parts = base.split('/').slice(0, -1)
  for (const seg of href.split('/')) {
    if (seg === '..') parts.pop()
    else if (seg !== '.' && seg !== '') parts.push(seg)
  }
  return parts.join('/')
}

/** Opens a path in the panel beside the conversation. */
export type FileOpener = (path: string) => void

export const FileOpenerContext = createContext<FileOpener | null>(null)
export const useFileOpener = () => useContext(FileOpenerContext)
