import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'
import * as api from '../lib/api'
import { FileOpenerContext, KIND_ICON, KIND_LABEL, kindOf, resolveFrom, type FileKind } from '../lib/files'
import { fmtSize, relTime } from '../lib/format'
import type { FileInfo } from '../types'
import { Markdown } from './Markdown'
import { Icon } from './ui'

/*
 * The panel beside a conversation that shows a file from the session's
 * workspace: the page the agent wrote, its chart, its report.
 *
 * GET /api/sessions/{id}/file finds the file and hands back a URL under
 * /files/ that needs no token; that URL is what the iframe, the <img> and the
 * text fetch read. A page the agent wrote runs in an iframe without
 * allow-same-origin, and wombatd serves it with a sandboxing CSP as well: it
 * gets no origin, so it can reach neither this page's storage (the token) nor
 * the API.
 */

const WIDTH_KEY = 'wombat.filePanelWidth'
const MIN_WIDTH = 360
const TEXT_CAP = 2_000_000 // characters shown of a text file
const LINE_CAP = 5_000

function savedWidth(): number {
  try {
    const w = Number(localStorage.getItem(WIDTH_KEY))
    if (w >= MIN_WIDTH) return w
  } catch { /* storage may be unavailable */ }
  return Math.round(Math.min(900, Math.max(460, window.innerWidth * 0.42)))
}

// ── renderers ────────────────────────────────────────────────────────────

const TOKEN = /("[^"\n]*"|'[^'\n]*'|`[^`\n]*`|\b(?:func|package|import|from|return|if|else|elif|for|range|in|do|done|then|fi|set|class|while|const|let|var|function|def|type|struct|interface|echo|async|await|export|default|print|true|false|nil|null|None)\b)/g

function highlight(line: string): ReactNode {
  const trimmed = line.trimStart()
  if (trimmed.startsWith('#') || trimmed.startsWith('//') || trimmed.startsWith('--')) return <span className="t-c">{line}</span>
  const out: ReactNode[] = []
  let last = 0
  for (const m of line.matchAll(TOKEN)) {
    const at = m.index ?? 0
    if (at > last) out.push(line.slice(last, at))
    out.push(<span key={at} className={/^["'`]/.test(m[0]) ? 't-s' : 't-k'}>{m[0]}</span>)
    last = at + m[0].length
  }
  if (last < line.length) out.push(line.slice(last))
  return out
}

function CodeLines({ text }: { text: string }) {
  const lines = text.replace(/\n$/, '').split('\n')
  return (
    <div className="codev">
      {lines.slice(0, LINE_CAP).map((l, i) => (
        <div className="l" key={i}><span className="n">{i + 1}</span><span>{highlight(l)}</span></div>
      ))}
      {lines.length > LINE_CAP && <div className="l"><span className="n" /><span className="t-c">…… 只显示前 {LINE_CAP} 行，共 {lines.length} 行</span></div>}
    </div>
  )
}

/** CSV or TSV, with quoted fields. */
function parseTable(text: string, sep: string): string[][] {
  const rows: string[][] = []
  let row: string[] = [], cell = '', quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++ }
      else if (c === '"') quoted = false
      else cell += c
    } else if (c === '"' && cell === '') quoted = true
    else if (c === sep) { row.push(cell); cell = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(cell); rows.push(row); row = []; cell = ''
      if (rows.length > 1000) return rows
    } else cell += c
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row) }
  return rows
}

function TableView({ text, tsv }: { text: string; tsv: boolean }) {
  const all = parseTable(text, tsv ? '\t' : ',')
  const head = all[0] ?? []
  const rows = all.slice(1, 501)
  const numeric = head.map((_, c) => rows.length > 0 && rows.every(r => r[c] === undefined || r[c] === '' || !Number.isNaN(Number(r[c]))))
  return (
    <div className="tblv">
      <div className="tblv-h"><span>{Math.max(0, all.length - 1)}{all.length > 1000 ? '+' : ''} 行 × {head.length} 列</span><span>{all.length > 501 ? '显示前 500 行' : '全部'}</span></div>
      <div className="tblv-scroll">
        <table>
          <thead><tr>{head.map((h, i) => <th key={i} className={numeric[i] ? 'num' : ''}>{h}</th>)}</tr></thead>
          <tbody>{rows.map((r, i) => <tr key={i}>{head.map((_, c) => <td key={c} className={numeric[c] ? 'num' : ''}>{r[c]}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </div>
  )
}

function NoPreview({ info, why }: { info: FileInfo; why: string }) {
  return (
    <div className="nofile">
      <Icon name="file" />
      <div>{info.name}</div>
      <div className="nofile-why">{why}</div>
      <a className="btn sm" href={info.url} target="_blank" rel="noreferrer"><Icon name="external" cls="sm" />在新标签页打开</a>
    </div>
  )
}

type Text = { text: string; cut: boolean } | { error: string } | null

/** A text file's contents, read again whenever stamp changes. */
function useText(url: string | null, stamp: number): Text {
  const [state, setState] = useState<{ key: string; value: Text }>({ key: '', value: null })
  const key = `${url}#${stamp}`
  useEffect(() => {
    if (!url) return
    const ctl = new AbortController()
    fetch(url, { signal: ctl.signal, cache: 'no-store' })
      .then(async r => {
        if (!r.ok) throw new Error(`读取失败：HTTP ${r.status}`)
        const text = await r.text()
        setState({ key, value: { text: text.slice(0, TEXT_CAP), cut: text.length > TEXT_CAP } })
      })
      .catch(e => { if (!ctl.signal.aborted) setState({ key, value: { error: e instanceof Error ? e.message : String(e) } }) })
    return () => ctl.abort()
  }, [url, key])
  // A refresh keeps showing what was there until the new contents arrive.
  return state.value
}

function TextBody({ info, kind, stamp }: { info: FileInfo; kind: FileKind; stamp: number }) {
  const st = useText(info.url, stamp)
  if (st === null) return <div className="pv-loading">正在读取……</div>
  if ('error' in st) return <NoPreview info={info} why={st.error} />
  const cut = st.cut && <div className="pv-cut">文件太大，只显示了开头的一部分。</div>
  switch (kind) {
    case 'markdown':
      // Links in the document are relative to where it is, not to the workspace.
      return <>{cut}<div className="mdv"><Markdown text={st.text} /></div></>
    case 'csv':
      return <>{cut}<TableView text={st.text} tsv={info.name.toLowerCase().endsWith('.tsv')} /></>
    case 'json': {
      let pretty = st.text
      try { pretty = JSON.stringify(JSON.parse(st.text), null, 2) } catch { /* shown as it is */ }
      return <>{cut}<CodeLines text={pretty} /></>
    }
    default:
      return <>{cut}<CodeLines text={st.text} /></>
  }
}

function Body({ info, stamp }: { info: FileInfo; stamp: number }) {
  const kind = kindOf(info.path)
  switch (kind) {
    case 'html':
      return (
        <div className="sbx">
          {/* No allow-same-origin: the page gets no origin, and so neither this page's storage nor the API. */}
          <iframe key={stamp} title={info.name} src={info.url} sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms allow-modals allow-downloads" />
        </div>
      )
    case 'image':
      return <div className="imgv"><img key={stamp} src={`${info.url}?v=${stamp}`} alt={info.name} /></div>
    case 'pdf':
      return <iframe key={stamp} className="pdf-frame" title={info.name} src={info.url} />
    case 'video':
      return <div className="mediav"><video key={stamp} src={info.url} controls /></div>
    case 'audio':
      return <div className="mediav"><audio key={stamp} src={info.url} controls /></div>
    case 'other':
      if (!/^text\/|json|xml|javascript/.test(info.type)) return <NoPreview info={info} why="这种文件不能在这里预览。" />
      return <TextBody info={info} kind="text" stamp={stamp} />
    default:
      return <TextBody info={info} kind={kind} stamp={stamp} />
  }
}

// ── the panel ────────────────────────────────────────────────────────────

type Loaded = { info: FileInfo } | { error: string; missing: boolean } | null

/**
 * Shows path, from the session's workspace. stamp changes when the
 * conversation moves on — a turn ended, the agent may have rewritten the file —
 * and the file is read again.
 */
export function FilePanel({ sessionId, path, stamp, onClose, onOpen }: {
  sessionId: string
  path: string
  stamp: number
  onClose: () => void
  onOpen: (path: string) => void
}) {
  const [loaded, setLoaded] = useState<Loaded>(null)
  const [reload, setReload] = useState(0)
  const [width, setWidth] = useState(savedWidth)
  const [dragging, setDragging] = useState(false)
  const asideRef = useRef<HTMLElement>(null)
  const version = stamp * 1000 + reload

  useEffect(() => {
    let alive = true
    api.fileInfo(sessionId, path).then(
      info => { if (alive) setLoaded({ info }) },
      e => {
        if (!alive) return
        const missing = e instanceof api.ApiError && e.kind === 'no_such_file'
        setLoaded({ error: missing ? '它可能还没写好，或者已经被删除了。' : api.errorText(e), missing })
      },
    )
    return () => { alive = false }
  }, [sessionId, path, version])

  const startDrag = (e: ReactPointerEvent) => {
    e.preventDefault()
    setDragging(true)
    const move = (ev: PointerEvent) => {
      const max = Math.max(MIN_WIDTH, window.innerWidth - 420)
      setWidth(Math.round(Math.min(max, Math.max(MIN_WIDTH, window.innerWidth - ev.clientX))))
    }
    const up = () => {
      setDragging(false)
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      const w = asideRef.current?.getBoundingClientRect().width
      try { if (w) localStorage.setItem(WIDTH_KEY, String(Math.round(w))) } catch { /* storage may be unavailable */ }
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  const info = loaded && 'info' in loaded ? loaded.info : null
  const kind = kindOf(path)
  const name = path.split('/').pop() || path
  const dir = info ? info.path.split('/').slice(0, -1).join('/') : ''

  return (
    <aside ref={asideRef} className={`pv files ${dragging ? 'dragging' : ''}`} style={{ width }}>
      <div className="pv-resize" onPointerDown={startDrag} title="拖动调整宽度" />
      <div className="pv-inner">
        <div className="pv-h">
          <Icon name={KIND_ICON[kind]} />
          <div className="pv-title">
            <div className="pv-name" title={info?.path ?? path}>{info?.name ?? name}</div>
            <div className="pv-sub">
              {KIND_LABEL[kind]}
              {info && <> · {fmtSize(info.size)} · {relTime(Date.parse(info.modified))}修改{dir && <> · {dir}/</>}</>}
            </div>
          </div>
          <div className="pv-acts">
            <button className="ibtn" title="重新读取" onClick={() => setReload(r => r + 1)}><Icon name="refresh" /></button>
            {info && <a className="ibtn" title="在新标签页打开" href={info.url} target="_blank" rel="noreferrer"><Icon name="external" /></a>}
            <button className="ibtn" title="关闭" onClick={onClose}><Icon name="x" /></button>
          </div>
        </div>
        <div className="pv-body">
          {loaded === null && <div className="pv-loading">正在打开……</div>}
          {loaded && 'error' in loaded && (
            <div className="nofile">
              <Icon name="file" />
              <div>{loaded.missing ? '找不到这个文件' : '打不开这个文件'}</div>
              <div className="nofile-why">{path}</div>
              <div className="nofile-why">{loaded.error}</div>
            </div>
          )}
          {info && (
            <FileOpenerContext.Provider value={p => onOpen(resolveFrom(info.path, p))}>
              <Body info={info} stamp={version} />
            </FileOpenerContext.Provider>
          )}
        </div>
      </div>
    </aside>
  )
}
