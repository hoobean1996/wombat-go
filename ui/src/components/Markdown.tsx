import { Fragment, useMemo, type CSSProperties, type ReactNode } from 'react'
import { Lexer, type Token, type Tokens } from 'marked'
import { looksLikeFile, useFileOpener } from '../lib/files'
import { Icon } from './ui'

/*
 * Markdown as the agent writes it: GitHub's flavour, with tables, task lists,
 * strikethrough and autolinks. marked only parses; the elements are made here.
 * So HTML in a reply stays text, only web, mail and relative links are links,
 * and an image is a link rather than an <img>: a reply cannot make the browser
 * fetch an address of its choosing (and send data along).
 */

const OPTIONS = { gfm: true, breaks: true } // a single newline is a line break, as people mean it in chat

/*
 * A file the agent names — `diagrams/page.html` in backticks, or a link to a
 * path — opens in the panel beside the conversation, where there is one
 * (lib/files.ts). Components rather than plain elements, for the context.
 */

/** A path in backticks: clickable where a file panel can show it. */
function FileCode({ path }: { path: string }) {
  const open = useFileOpener()
  if (!open) return <code>{path}</code>
  return (
    <code
      className="file-link" role="button" tabIndex={0} title="在右侧打开"
      onClick={() => open(path)} onKeyDown={e => { if (e.key === 'Enter') open(path) }}
    >{path}</code>
  )
}

/** A link to a path rather than to the web: it opens in the file panel, or stays text. */
function FileAnchor({ path, children }: { path: string; children: ReactNode }) {
  const open = useFileOpener()
  if (!open) return <>{children}</>
  return <a href="#" className="file-a" title={`在右侧打开 ${path}`} onClick={e => { e.preventDefault(); open(path) }}>{children}</a>
}

/** A link that is a path in the workspace: not the web, not mail, not an anchor on the page. */
const isPathLink = (href: string) => !/^(https?:|mailto:|#)/i.test(href)

const unescapePath = (href: string) => { try { return decodeURI(href) } catch { return href } }

/** A link the page may follow: web, mail and relative ones, not javascript: or data:. */
function safeHref(href: string): string | null {
  const h = href.trim()
  if (/^(https?:|mailto:)/i.test(h)) return h
  return /^[a-z][a-z0-9+.-]*:/i.test(h) ? null : h
}

const decode = (s: string) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&')

function inlines(tokens: Token[] | undefined, key: string, tail?: ReactNode): ReactNode[] {
  const out = (tokens ?? []).map((t, i) => inline(t, `${key}.${i}`))
  if (tail) out.push(<Fragment key={`${key}.end`}>{tail}</Fragment>)
  return out
}

function inline(t: Token, k: string): ReactNode {
  switch (t.type) {
    case 'strong':
      return <b key={k}>{inlines(t.tokens, k)}</b>
    case 'em':
      return <em key={k}>{inlines(t.tokens, k)}</em>
    case 'del':
      return <del key={k}>{inlines(t.tokens, k)}</del>
    case 'codespan': {
      const text = decode(t.text)
      return looksLikeFile(text) ? <FileCode key={k} path={text} /> : <code key={k}>{text}</code>
    }
    case 'br':
      return <br key={k} />
    case 'link':
      return link(t as Tokens.Link, k)
    case 'image':
      return image(t as Tokens.Image, k)
    case 'html': // a tag in the text stays text, except a line break (common in table cells)
      return /^<br\s*\/?>$/i.test(t.text.trim()) ? <br key={k} /> : t.text
    case 'text':
      if (t.tokens?.length) return <Fragment key={k}>{inlines(t.tokens, k)}</Fragment>
      return (t as Tokens.Text).escaped ? decode(t.text) : t.text
    default:
      return 'text' in t && typeof t.text === 'string' ? t.text : t.raw
  }
}

function link(t: Tokens.Link, k: string): ReactNode {
  if (t.raw === t.text && !t.raw.startsWith('<')) return bareUrl(t, k)
  return anchor(t.href, inlines(t.tokens, k), k, t.title)
}

function anchor(url: string, label: ReactNode, k: string, title?: string | null): ReactNode {
  const href = safeHref(url)
  if (!href) return <Fragment key={k}>{label}</Fragment>
  if (isPathLink(href)) return <FileAnchor key={k} path={unescapePath(href)}>{label}</FileAnchor>
  return <a key={k} href={href} target="_blank" rel="noreferrer" title={title ?? undefined}>{label}</a>
}

/**
 * A URL written as is. GFM lets it run to the next space, which in Chinese is
 * usually the rest of the sentence ("https://a.com，然后"): it ends at the first
 * character no URL has, without trailing punctuation, and the rest is text again.
 */
function bareUrl(t: Tokens.Link, k: string): ReactNode {
  let end = t.raw.search(/[^A-Za-z0-9\-._~:/?#[\]@!$&'()*+,;=%]/)
  let url = end < 0 ? t.raw : t.raw.slice(0, end)
  for (;;) {
    const last = url.at(-1)
    if (last === undefined) break
    const unbalanced = last === ')' && url.split('(').length < url.split(')').length
    if (!".,:;!?'\"*_~".includes(last) && !unbalanced) break
    url = url.slice(0, -1)
  }
  end = url.length
  const href = t.href.startsWith('mailto:') ? t.href : t.raw.startsWith('www.') ? `http://${url}` : url
  const rest = t.raw.slice(end)
  return (
    <Fragment key={k}>
      {anchor(href, url, `${k}u`)}
      {rest && inlines(Lexer.lexInline(rest, OPTIONS), `${k}r`)}
    </Fragment>
  )
}

function image(t: Tokens.Image, k: string): ReactNode {
  // Never an <img>: a reply must not make the browser fetch an address of its
  // choosing. A link instead, which the browser only follows when clicked.
  const href = safeHref(t.href)
  if (!href) return t.text
  const label = <><Icon name="image" cls="sm" />{t.text || href}</>
  if (isPathLink(href)) return <FileAnchor key={k} path={unescapePath(href)}>{label}</FileAnchor>
  return <a key={k} href={href} target="_blank" rel="noreferrer">{label}</a>
}

/** Renders block tokens; tail (the streaming cursor) goes at the end of the last one. */
function blocks(tokens: Token[], key: string, tail?: ReactNode, tight = false): ReactNode[] {
  const shown = tokens.filter(t => t.type !== 'space' && t.type !== 'def')
  return shown.map((t, i) => block(t, `${key}.${i}`, i === shown.length - 1 ? tail : undefined, tight))
}

const alignOf = (a: Tokens.Table['align'][number]): CSSProperties | undefined => (a ? { textAlign: a } : undefined)

function block(t: Token, k: string, tail: ReactNode, tight: boolean): ReactNode {
  switch (t.type) {
    case 'heading': {
      const cls = t.depth === 1 ? 'md-h1' : t.depth === 2 ? 'md-h2' : 'md-h3'
      return <div key={k} className={cls} role="heading" aria-level={t.depth}>{inlines(t.tokens, k, tail)}</div>
    }
    case 'paragraph':
      return <p key={k}>{inlines(t.tokens, k, tail)}</p>
    case 'text': { // the text of a tight list item
      const body = t.tokens?.length ? inlines(t.tokens, k, tail) : [t.text, tail]
      return tight ? <Fragment key={k}>{body}</Fragment> : <p key={k}>{body}</p>
    }
    case 'code':
      return <pre key={k}><code>{t.text}</code>{tail}</pre>
    case 'blockquote':
      return <blockquote key={k}>{blocks(t.tokens ?? [], k, tail)}</blockquote>
    case 'hr':
      return <Fragment key={k}><hr />{tail}</Fragment>
    case 'list': {
      const l = t as Tokens.List
      const items = l.items.map((it, j) => (
        <li key={j} className={it.task ? 'md-task' : undefined}>
          {it.task && <span className="md-check">{it.checked ? '☑' : '☐'}</span>}
          {blocks(it.tokens.filter(x => x.type !== 'checkbox'), `${k}.${j}`, j === l.items.length - 1 ? tail : undefined, !l.loose)}
        </li>
      ))
      if (!l.ordered) return <ul key={k}>{items}</ul>
      return <ol key={k} start={typeof l.start === 'number' && l.start !== 1 ? l.start : undefined}>{items}</ol>
    }
    case 'table': {
      const tb = t as Tokens.Table
      const rows = tb.rows
      const lastCell = (r: number, c: number, n: number) => (r === rows.length - 1 && c === n - 1 ? tail : undefined)
      return (
        <div key={k} className="md-table">
          <table>
            <thead>
              <tr>{tb.header.map((c, j) => <th key={j} style={alignOf(tb.align[j])}>{inlines(c.tokens, `${k}.h${j}`, rows.length ? undefined : lastCell(-1, j, tb.header.length))}</th>)}</tr>
            </thead>
            {rows.length > 0 && (
              <tbody>
                {rows.map((r, ri) => (
                  <tr key={ri}>{r.map((c, j) => <td key={j} style={alignOf(tb.align[j])}>{inlines(c.tokens, `${k}.${ri}.${j}`, lastCell(ri, j, r.length))}</td>)}</tr>
                ))}
              </tbody>
            )}
          </table>
        </div>
      )
    }
    case 'html': // HTML written in a reply is shown, not run
      return <p key={k}>{t.text}{tail}</p>
    default:
      return <p key={k}>{t.raw}{tail}</p>
  }
}

export function Markdown({ text, cursor = false }: { text: string; cursor?: boolean }) {
  const tokens = useMemo(() => Lexer.lex(text, OPTIONS), [text])
  const tail = cursor ? <span className="cursor" /> : undefined
  const nodes = blocks(tokens, 'b', tail)
  return <div className="md">{nodes.length ? nodes : tail}</div>
}
