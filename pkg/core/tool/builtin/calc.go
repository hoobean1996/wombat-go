package builtin

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"wombat/pkg/core/tool"
)

// A recursive-descent evaluator for the calculator tool.
//
// Grammar, tightest binding last:
//
//	expr    := term (('+' | '-') term)*
//	term    := unary (('*' | '/' | '%') unary)*
//	unary   := ('+' | '-') unary | power
//	power   := primary ('^' unary)?          // right associative: 2^3^2 = 512
//	primary := number | '(' expr ')' | func '(' expr ')'
//
// It is deliberately small. The tool exists so the model does not do
// arithmetic in its head, not so it has a computer algebra system; anything
// bigger belongs in bash with a real interpreter.

// ErrCalc is the class of every calculator failure, so a caller can tell a
// malformed expression from an infrastructure problem.
// ErrCalc is a [tool.CallerFault]: the expression was bad, not the evaluator.
var ErrCalc = tool.CallerError(errors.New("builtin: cannot evaluate expression"))

// calcFuncs are the named single-argument functions the parser accepts.
var calcFuncs = map[string]func(float64) float64{
	"sqrt":  math.Sqrt,
	"abs":   math.Abs,
	"floor": math.Floor,
	"ceil":  math.Ceil,
	"round": math.Round,
}

type calcParser struct {
	src string
	pos int
}

// evalExpr evaluates src and formats the result.
func evalExpr(src string) (string, error) {
	if strings.TrimSpace(src) == "" {
		return "", fmt.Errorf("%w: field 'expression' must not be empty", ErrCalc)
	}
	p := &calcParser{src: src}
	v, err := p.expr()
	if err != nil {
		return "", err
	}
	p.skipSpace()
	if p.pos < len(p.src) {
		return "", p.errorf("unexpected %q", p.src[p.pos])
	}
	switch {
	case math.IsNaN(v):
		return "", fmt.Errorf("%w: result is not a number", ErrCalc)
	case math.IsInf(v, 0):
		return "", fmt.Errorf("%w: result overflowed to infinity", ErrCalc)
	}
	return formatNumber(v), nil
}

func (p *calcParser) errorf(format string, args ...any) error {
	return fmt.Errorf("%w: %s at offset %d", ErrCalc, fmt.Sprintf(format, args...), p.pos)
}

func (p *calcParser) skipSpace() {
	for p.pos < len(p.src) && (p.src[p.pos] == ' ' || p.src[p.pos] == '\t' || p.src[p.pos] == '\n' || p.src[p.pos] == '\r') {
		p.pos++
	}
}

// accept consumes c if it is next.
func (p *calcParser) accept(c byte) bool {
	p.skipSpace()
	if p.pos < len(p.src) && p.src[p.pos] == c {
		p.pos++
		return true
	}
	return false
}

func (p *calcParser) expr() (float64, error) {
	v, err := p.term()
	if err != nil {
		return 0, err
	}
	for {
		switch {
		case p.accept('+'):
			r, err := p.term()
			if err != nil {
				return 0, err
			}
			v += r
		case p.accept('-'):
			r, err := p.term()
			if err != nil {
				return 0, err
			}
			v -= r
		default:
			return v, nil
		}
	}
}

func (p *calcParser) term() (float64, error) {
	v, err := p.unary()
	if err != nil {
		return 0, err
	}
	for {
		switch {
		case p.accept('*'):
			r, err := p.unary()
			if err != nil {
				return 0, err
			}
			v *= r
		case p.accept('/'):
			r, err := p.unary()
			if err != nil {
				return 0, err
			}
			if r == 0 {
				return 0, fmt.Errorf("%w: division by zero", ErrCalc)
			}
			v /= r
		case p.accept('%'):
			r, err := p.unary()
			if err != nil {
				return 0, err
			}
			if r == 0 {
				return 0, fmt.Errorf("%w: modulo by zero", ErrCalc)
			}
			v = math.Mod(v, r)
		default:
			return v, nil
		}
	}
}

func (p *calcParser) unary() (float64, error) {
	switch {
	case p.accept('-'):
		v, err := p.unary()
		return -v, err
	case p.accept('+'):
		return p.unary()
	default:
		return p.power()
	}
}

func (p *calcParser) power() (float64, error) {
	base, err := p.primary()
	if err != nil {
		return 0, err
	}
	if !p.accept('^') {
		return base, nil
	}
	// Right associative, and the exponent goes through unary so that
	// 2^-1 parses.
	exp, err := p.unary()
	if err != nil {
		return 0, err
	}
	return math.Pow(base, exp), nil
}

func (p *calcParser) primary() (float64, error) {
	p.skipSpace()
	if p.pos >= len(p.src) {
		return 0, p.errorf("unexpected end of expression")
	}

	if p.accept('(') {
		v, err := p.expr()
		if err != nil {
			return 0, err
		}
		if !p.accept(')') {
			return 0, p.errorf("missing closing parenthesis")
		}
		return v, nil
	}

	c := p.src[p.pos]
	switch {
	case c >= '0' && c <= '9', c == '.':
		return p.number()
	case (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'):
		return p.call()
	default:
		return 0, p.errorf("unexpected %q", c)
	}
}

func (p *calcParser) number() (float64, error) {
	start := p.pos
	for p.pos < len(p.src) {
		c := p.src[p.pos]
		if (c >= '0' && c <= '9') || c == '.' {
			p.pos++
			continue
		}
		break
	}
	v, err := strconv.ParseFloat(p.src[start:p.pos], 64)
	if err != nil {
		p.pos = start
		return 0, p.errorf("malformed number %q", p.src[start:])
	}
	return v, nil
}

func (p *calcParser) call() (float64, error) {
	start := p.pos
	for p.pos < len(p.src) {
		c := p.src[p.pos]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
			p.pos++
			continue
		}
		break
	}
	name := strings.ToLower(p.src[start:p.pos])

	if name == "pi" {
		return math.Pi, nil
	}
	if name == "e" {
		return math.E, nil
	}

	fn, ok := calcFuncs[name]
	if !ok {
		p.pos = start
		return 0, p.errorf("unknown function %q (supported: sqrt, abs, floor, ceil, round)", name)
	}
	if !p.accept('(') {
		return 0, p.errorf("%s expects a parenthesised argument", name)
	}
	arg, err := p.expr()
	if err != nil {
		return 0, err
	}
	if !p.accept(')') {
		return 0, p.errorf("missing closing parenthesis")
	}
	return fn(arg), nil
}

// formatNumber renders a result the way `bc -l` with scale=6 would: integers
// exactly, everything else to six decimals with trailing zeros trimmed. The
// scale is what keeps 1/3 from arriving as 0.3333333333333333 and inviting
// the model to treat float noise as significant.
func formatNumber(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatFloat(v, 'f', -1, 64)
	}
	if av := math.Abs(v); av >= 1e15 || av < 1e-6 {
		return strconv.FormatFloat(v, 'g', -1, 64)
	}
	s := strconv.FormatFloat(v, 'f', 6, 64)
	s = strings.TrimRight(s, "0")
	return strings.TrimSuffix(s, ".")
}
