// Package yaml reads the part of YAML a configuration file uses — nested
// mappings, lists, scalars, comments and block strings — into Go values.
//
// It exists because this module depends on the standard library alone, and a
// configuration file needs very little of YAML. What it leaves out — anchors
// and aliases, tags, several documents in one file, flow collections inside
// flow collections, plain strings that run on to the next line — is an error
// naming the line, never a quiet misreading: a file that decodes means what
// it says.
//
// Decoding is strict for the same reason. A key the target struct does not
// have is an error, because a misspelt key in a configuration file is a
// setting that silently did not take effect. A key that is absent, or null,
// leaves its field as it was, so a struct filled with defaults before
// [Unmarshal] keeps every default the file does not mention. A key that is
// present replaces its field outright, lists and maps included.
//
// Fields are matched by their `yaml:"name"` tag — the tag gopkg.in/yaml.v3
// reads, so a struct written for this package decodes the same under that one.
package yaml

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Error is a document that cannot be read, or a value that does not fit the
// field it is decoded into.
type Error struct {
	Line int // 1-based; 0 when no one line is to blame
	Msg  string
}

func (e *Error) Error() string {
	if e.Line > 0 {
		return fmt.Sprintf("line %d: %s", e.Line, e.Msg)
	}
	return e.Msg
}

type kind int

const (
	nullKind kind = iota
	scalarKind
	mappingKind
	sequenceKind
)

// node is one value of the document, and the line it starts on.
type node struct {
	kind    kind
	line    int
	text    string  // a scalar's value
	plain   bool    // an unquoted scalar, which may also read as a bool or a number
	entries []entry // a mapping's, in document order
	items   []*node // a sequence's
}

type entry struct {
	key  string
	line int
	val  *node
}

// parser reads a document a line at a time. Indentation alone decides the
// block structure, so one line is all it ever looks at — except a block
// string, which reads the raw lines below its header.
type parser struct {
	lines []string
	pos   int // the next line to read
}

// line is a line that means something: neither blank nor only a comment.
type line struct {
	no     int    // 1-based
	indent int    // leading spaces
	text   string // the rest, trailing blanks trimmed
}

// fail stops the parse. Recovered in parse, and only there.
func (p *parser) fail(no int, format string, args ...any) {
	panic(&Error{Line: no, Msg: fmt.Sprintf(format, args...)})
}

func parse(data []byte) (root *node, err error) {
	if !utf8.Valid(data) {
		return nil, &Error{Msg: "the document is not UTF-8"}
	}
	defer func() {
		if r := recover(); r != nil {
			e, ok := r.(*Error)
			if !ok {
				panic(r)
			}
			root, err = nil, e
		}
	}()
	text := strings.TrimPrefix(string(data), "\ufeff")
	text = strings.TrimSuffix(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	p := &parser{lines: strings.Split(text, "\n")}

	l, ok := p.peek()
	if ok && l.indent == 0 && l.text == "---" {
		p.pos++
		l, ok = p.peek()
	}
	if !ok {
		return &node{kind: nullKind}, nil
	}
	root = p.block(l.indent)
	if l, ok := p.peek(); ok {
		if isMarker(l) {
			p.fail(l.no, "a second document; a file holds one")
		}
		p.fail(l.no, "this line is indented less than the block it follows")
	}
	return root, nil
}

// peek returns the next line that means something, without consuming it.
// Blank lines and comments before it are consumed: outside a block string,
// they are nothing.
func (p *parser) peek() (line, bool) {
	for ; p.pos < len(p.lines); p.pos++ {
		raw := p.lines[p.pos]
		text := strings.TrimLeft(raw, " ")
		if t := strings.TrimSpace(text); t == "" || t[0] == '#' {
			continue
		}
		if text[0] == '\t' {
			p.fail(p.pos+1, "a tab indents this line; YAML indents with spaces")
		}
		return line{no: p.pos + 1, indent: len(raw) - len(text), text: strings.TrimRight(text, " \t")}, true
	}
	return line{}, false
}

// block reads the mapping or the list whose entries start at column indent.
func (p *parser) block(indent int) *node {
	if l, _ := p.peek(); isItem(l.text) {
		return p.sequence(indent)
	}
	return p.mapping(indent)
}

// isMarker reports whether l starts a document, or ends one.
func isMarker(l line) bool {
	return l.indent == 0 && (l.text == "---" || l.text == "..." || strings.HasPrefix(l.text, "--- "))
}

// isItem reports whether text is a list item: a dash, then a space or nothing.
func isItem(text string) bool {
	return text == "-" || strings.HasPrefix(text, "- ")
}

func (p *parser) mapping(indent int) *node {
	m := &node{kind: mappingKind}
	for {
		l, ok := p.peek()
		if !ok || l.indent < indent || isMarker(l) {
			return m
		}
		if l.indent > indent {
			p.fail(l.no, "this line is indented more than the key above it")
		}
		if isItem(l.text) {
			p.fail(l.no, "a list item among the keys of a mapping")
		}
		name, rest, ok := splitKey(l.text)
		if !ok {
			p.fail(l.no, "expected a key, as in name: value")
		}
		if slices.ContainsFunc(m.entries, func(e entry) bool { return e.key == name }) {
			p.fail(l.no, "%q is given twice", name)
		}
		if m.line == 0 {
			m.line = l.no
		}
		p.pos++
		m.entries = append(m.entries, entry{key: name, line: l.no, val: p.value(rest, l.no, indent, true)})
	}
}

func (p *parser) sequence(indent int) *node {
	s := &node{kind: sequenceKind}
	for {
		l, ok := p.peek()
		if !ok || l.indent < indent || (l.indent == indent && !isItem(l.text)) {
			return s
		}
		if l.indent > indent {
			p.fail(l.no, "this line is indented more than the list item above it")
		}
		if s.line == 0 {
			s.line = l.no
		}
		rest := strings.TrimLeft(l.text[1:], " ")
		if rest != "" && rest[0] != '#' && (isItem(rest) || isKey(rest)) {
			// "- name: value", or a list in a list: a block that starts on the
			// dash's line. Read that line again with the dash blanked out, so
			// the block is read like any other, at the column its text is at.
			col := indent + len(l.text) - len(rest)
			p.lines[p.pos] = strings.Repeat(" ", col) + rest
			s.items = append(s.items, p.block(col))
			continue
		}
		p.pos++
		s.items = append(s.items, p.value(rest, l.no, indent, false))
	}
}

// value reads what follows "name:" or "- ": the rest of the line, and for a
// nested block or a block string the lines below it. indent is the column of
// the key or the dash; what belongs to them is indented further — except a
// list, which may sit at its key's own column.
func (p *parser) value(rest string, no, indent int, ofKey bool) *node {
	if rest == "" || rest[0] == '#' {
		next, ok := p.peek()
		switch {
		case ok && next.indent > indent:
			return p.block(next.indent)
		case ok && ofKey && next.indent == indent && isItem(next.text):
			return p.sequence(indent)
		}
		return &node{kind: nullKind, line: no}
	}
	if rest[0] == '|' || rest[0] == '>' {
		return p.blockString(rest, no, indent)
	}
	n := p.inline(rest, no)
	if next, ok := p.peek(); ok && next.indent > indent {
		p.fail(next.no, "this line continues a value that ended on the line above; "+
			"a string over several lines is written with | or >")
	}
	return n
}

// splitKey splits "name: value" at the colon that ends the key. The rest is
// returned as written, comment and all.
func splitKey(text string) (name, rest string, ok bool) {
	if text[0] == '"' || text[0] == '\'' {
		v, after, err := unquote(text)
		if err != nil {
			return "", "", false
		}
		after = strings.TrimLeft(after, " ")
		if after == ":" || strings.HasPrefix(after, ": ") {
			return v, strings.TrimSpace(after[1:]), true
		}
		return "", "", false
	}
	if strings.ContainsRune("[]{}&*!|>%@`#?,", rune(text[0])) || isItem(text) {
		return "", "", false
	}
	for i := 0; i < len(text); i++ {
		switch {
		case text[i] == '#' && i > 0 && isBlank(text[i-1]):
			return "", "", false // a comment, and no colon before it
		case text[i] == ':' && (i+1 == len(text) || isBlank(text[i+1])):
			return strings.TrimRight(text[:i], " \t"), strings.TrimSpace(text[i+1:]), true
		}
	}
	return "", "", false
}

func isKey(text string) bool {
	_, _, ok := splitKey(text)
	return ok
}

func isBlank(c byte) bool { return c == ' ' || c == '\t' }

// inline reads a value written on its key's line: a quoted or plain scalar,
// or a flow collection.
func (p *parser) inline(s string, no int) *node {
	switch s[0] {
	case '"', '\'':
		v, after, err := unquote(s)
		if err != nil {
			p.fail(no, "%v", err)
		}
		p.nothingAfter(after, no)
		return &node{kind: scalarKind, line: no, text: v}
	case '[', '{':
		n, after := p.flow(s, no)
		p.nothingAfter(after, no)
		return n
	case '&', '*', '!':
		p.fail(no, "anchors, aliases and tags are not supported")
	case '@', '`', '%':
		p.fail(no, "a plain value cannot start with %c; quote it", s[0])
	}
	if isItem(s) {
		p.fail(no, "a list cannot start on its key's line; put its items on the lines below, or write [a, b]")
	}
	v := stripComment(s)
	if isKey(v) {
		p.fail(no, "%q holds a colon and a space, which would make it a key; quote the value", v)
	}
	return plainScalar(v, no)
}

// nothingAfter fails unless only blanks or a comment follow a closed value.
func (p *parser) nothingAfter(after string, no int) {
	if t := strings.TrimLeft(after, " \t"); t != "" && t[0] != '#' {
		p.fail(no, "unexpected %q after the value", t)
	}
}

// stripComment removes a trailing comment: a # after a blank.
func stripComment(s string) string {
	for i := 1; i < len(s); i++ {
		if s[i] == '#' && isBlank(s[i-1]) {
			return strings.TrimRight(s[:i], " \t")
		}
	}
	return s
}

func plainScalar(v string, no int) *node {
	switch v {
	case "", "~", "null", "Null", "NULL":
		return &node{kind: nullKind, line: no}
	}
	return &node{kind: scalarKind, line: no, text: v, plain: true}
}

// unquote reads the quoted scalar at the start of s, which must end on the
// same line, and returns its value and what follows the closing quote.
func unquote(s string) (value, after string, err error) {
	q := s[0]
	var b strings.Builder
	for i := 1; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '\'' && q == '\'':
			if i+1 < len(s) && s[i+1] == '\'' {
				b.WriteByte('\'') // '' is how a single-quoted string holds a quote
				i++
				continue
			}
			return b.String(), s[i+1:], nil
		case c == '"' && q == '"':
			return b.String(), s[i+1:], nil
		case c == '\\' && q == '"':
			r, n, err := unescape(s[i+1:])
			if err != nil {
				return "", "", err
			}
			b.WriteString(r)
			i += n
		default:
			b.WriteByte(c)
		}
	}
	return "", "", errors.New("a quoted string does not end on its line; " +
		"a string over several lines is written with | or >")
}

// escapes are the one-character escapes of a double-quoted string.
var escapes = map[byte]string{
	'0': "\x00", 'a': "\a", 'b': "\b", 't': "\t", '\t': "\t", 'n': "\n", 'v': "\v",
	'f': "\f", 'r': "\r", 'e': "\x1b", ' ': " ", '"': `"`, '/': "/", '\\': `\`,
	'N': "\u0085", '_': "\u00a0", 'L': "\u2028", 'P': "\u2029",
}

// unescape decodes the escape that s, the text after a backslash, starts
// with, and reports how much of s it took.
func unescape(s string) (string, int, error) {
	if s == "" {
		return "", 0, errors.New("a backslash ends the line")
	}
	if r, ok := escapes[s[0]]; ok {
		return r, 1, nil
	}
	width := map[byte]int{'x': 2, 'u': 4, 'U': 8}[s[0]]
	if width == 0 || len(s) <= width {
		return "", 0, fmt.Errorf(`\%c is not an escape YAML has`, s[0])
	}
	v, err := strconv.ParseUint(s[1:1+width], 16, 32)
	if err != nil || !utf8.ValidRune(rune(v)) {
		return "", 0, fmt.Errorf(`\%s is not a character`, s[:1+width])
	}
	return string(rune(v)), 1 + width, nil
}

// flow reads a flow collection, [a, b] or {k: v}. It must close on the line
// it opens on, and holds scalars only.
func (p *parser) flow(s string, no int) (*node, string) {
	closing := byte(']')
	n := &node{kind: sequenceKind, line: no}
	if s[0] == '{' {
		closing, n.kind = '}', mappingKind
	}
	i := skipBlanks(s, 1)
	if i < len(s) && s[i] == closing {
		return n, s[i+1:]
	}
	for {
		if n.kind == mappingKind {
			k, j := p.flowScalar(s, i, no, true)
			j = skipBlanks(s, j)
			if j >= len(s) || s[j] != ':' {
				p.fail(no, "expected a colon after %q", k.text)
			}
			if slices.ContainsFunc(n.entries, func(e entry) bool { return e.key == k.text }) {
				p.fail(no, "%q is given twice", k.text)
			}
			var v *node
			v, i = p.flowScalar(s, skipBlanks(s, j+1), no, false)
			n.entries = append(n.entries, entry{key: k.text, line: no, val: v})
		} else {
			var v *node
			v, i = p.flowScalar(s, i, no, false)
			n.items = append(n.items, v)
		}
		i = skipBlanks(s, i)
		switch {
		case i < len(s) && s[i] == closing:
			return n, s[i+1:]
		case i < len(s) && s[i] == ',':
			if i = skipBlanks(s, i+1); i < len(s) && s[i] == closing {
				return n, s[i+1:] // a trailing comma
			}
		default:
			p.fail(no, "a %c that does not close on its line, or two entries with no comma between them", s[0])
		}
	}
}

// flowScalar reads one scalar of a flow collection, starting at s[i], and
// returns it and where it ends. A key ends at a colon; a value at a comma
// or the closing bracket.
func (p *parser) flowScalar(s string, i, no int, isKey bool) (*node, int) {
	if i >= len(s) {
		p.fail(no, "a flow collection that does not close on its line")
	}
	switch s[i] {
	case '"', '\'':
		v, after, err := unquote(s[i:])
		if err != nil {
			p.fail(no, "%v", err)
		}
		return &node{kind: scalarKind, line: no, text: v}, len(s) - len(after)
	case '[', '{':
		p.fail(no, "a flow collection inside another is not supported")
	case '&', '*', '!':
		p.fail(no, "anchors, aliases and tags are not supported")
	}
	j := i
	for ; j < len(s); j++ {
		c := s[j]
		if strings.IndexByte(",[]{}", c) >= 0 || c == '#' && isBlank(s[j-1]) ||
			isKey && c == ':' && (j+1 == len(s) || isBlank(s[j+1]) || strings.IndexByte(",}", s[j+1]) >= 0) {
			break
		}
	}
	v := strings.TrimSpace(s[i:j])
	if v == "" {
		p.fail(no, "an empty entry in a flow collection")
	}
	return plainScalar(v, no), j
}

func skipBlanks(s string, i int) int {
	for i < len(s) && isBlank(s[i]) {
		i++
	}
	return i
}

// blockString reads a | or > string: the lines below its header indented
// further than indent, less the indentation of the first of them.
//
// | keeps the line breaks. > folds each run of lines into one, the way
// paragraphs are: a blank line ends one, and a more-indented line keeps its
// breaks. The end is clipped to one line break, stripped of all with |- or
// >-, and kept as written with |+ or >+.
func (p *parser) blockString(header string, no, indent int) *node {
	h := stripComment(header)
	folded := h[0] == '>'
	chomp := byte(0)
	switch h[1:] {
	case "":
	case "-", "+":
		chomp = h[1]
	default:
		p.fail(no, "%q: only | and >, each optionally followed by - or +, are supported", h)
	}

	type textLine struct {
		blanks int // blank lines before this one
		text   string
	}
	var lines []textLine
	blanks, width := 0, -1
	for ; p.pos < len(p.lines); p.pos++ {
		raw := p.lines[p.pos]
		if strings.TrimSpace(raw) == "" {
			blanks++
			continue
		}
		n := len(raw) - len(strings.TrimLeft(raw, " "))
		if width < 0 {
			if n <= indent {
				break
			}
			width = n
		}
		if n < width {
			break
		}
		lines = append(lines, textLine{blanks: blanks, text: raw[width:]})
		blanks = 0
	}

	moreIndented := func(s string) bool { return s != "" && isBlank(s[0]) }
	var b strings.Builder
	for i, l := range lines {
		if i > 0 {
			prev := lines[i-1].text
			switch {
			case !folded || moreIndented(prev) || moreIndented(l.text):
				b.WriteByte('\n')
			case l.blanks == 0:
				b.WriteByte(' ')
			}
		}
		b.WriteString(strings.Repeat("\n", l.blanks))
		b.WriteString(l.text)
	}
	if len(lines) > 0 {
		switch chomp {
		case 0:
			b.WriteByte('\n')
		case '+':
			b.WriteString(strings.Repeat("\n", 1+blanks))
		}
	}
	return &node{kind: scalarKind, line: no, text: b.String()}
}
