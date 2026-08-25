package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/hoobean1996/wombat-go/tool"
)

func TestEvalExpr(t *testing.T) {
	tests := []struct {
		name string
		expr string
		want string
	}{
		// Arithmetic and precedence.
		{"addition", "2+3", "5"},
		{"precedence: * binds tighter than +", "2+3*4", "14"},
		{"precedence: parens override", "(2+3)*4", "20"},
		{"the documented example", "(15 * 7 + 3) / 2", "54"},
		{"subtraction is left associative", "10-3-2", "5"},
		{"division is left associative", "100/5/2", "10"},
		{"modulo", "10 % 3", "1"},
		{"modulo binds like *", "2+10%3", "3"},
		{"negative modulo keeps the sign of the dividend", "-10 % 3", "-1"},

		// Powers: right associative, and tighter than unary minus.
		{"power", "2^10", "1024"},
		{"power is right associative", "2^3^2", "512"},
		{"unary minus is looser than power", "-2^2", "-4"},
		{"a negative exponent parses", "2^-1", "0.5"},
		{"parenthesised negative base", "(-2)^2", "4"},

		// Unary.
		{"unary minus", "-5", "-5"},
		{"unary plus", "+5", "5"},
		{"stacked unary", "--5", "5"},
		{"unary applied to a paren", "-(2+3)", "-5"},

		// Functions and constants.
		{"sqrt", "sqrt(16)", "4"},
		{"abs", "abs(-3.5)", "3.5"},
		{"floor", "floor(2.7)", "2"},
		{"ceil", "ceil(2.1)", "3"},
		{"round", "round(2.5)", "3"},
		{"round half away from zero", "round(-2.5)", "-3"},
		{"nested functions", "sqrt(abs(-16))", "4"},
		{"function names are case-insensitive", "SQRT(9)", "3"},
		{"pi", "floor(pi*100)", "314"},
		{"e", "floor(e*100)", "271"},
		{"functions compose with arithmetic", "sqrt(16)*2+1", "9"},

		// Whitespace and formatting.
		{"whitespace is ignored", "  2   +\t3\n", "5"},
		{"decimals", "1.5+2.25", "3.75"},
		{"leading dot", ".5+.5", "1"},
		{"scale is six decimals", "1/3", "0.333333"},
		{"trailing zeros trimmed", "1/4", "0.25"},
		{"an integral result has no decimal point", "6/3", "2"},
		{"very small values fall back to exponent form", "1/10000000", "1e-07"},
		{"zero", "0", "0"},
		{"negative zero renders as -0", "-0", "-0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := evalExpr(tt.expr)
			if err != nil {
				t.Fatalf("evalExpr(%q) returned error %v, want %q", tt.expr, err, tt.want)
			}
			if got != tt.want {
				t.Errorf("evalExpr(%q) = %q, want %q", tt.expr, got, tt.want)
			}
		})
	}
}

func TestEvalExprErrors(t *testing.T) {
	tests := []struct {
		name    string
		expr    string
		wantMsg string
	}{
		{"empty", "", "must not be empty"},
		{"only whitespace", "   \t\n", "must not be empty"},
		{"division by zero", "1/0", "division by zero"},
		{"division by a zero expression", "1/(2-2)", "division by zero"},
		{"modulo by zero", "5 % 0", "modulo by zero"},
		{"dangling operator", "2+", "unexpected end of expression"},
		{"unclosed paren", "(1+2", "missing closing parenthesis"},
		{"unclosed function paren", "sqrt(4", "missing closing parenthesis"},
		{"unknown function", "foo(1)", `unknown function "foo"`},
		{"function without parens", "sqrt 4", "expects a parenthesised argument"},
		{"trailing garbage", "1 2", "unexpected"},
		{"stray closing paren", "1)", `unexpected ')'`},
		{"unexpected character", "1 $ 2", `unexpected '$'`},
		{"malformed number", "1.2.3", "malformed number"},
		{"NaN result", "sqrt(-1)", "result is not a number"},
		{"overflow to infinity", "9^9^9", "result overflowed to infinity"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := evalExpr(tt.expr)
			if err == nil {
				t.Fatalf("evalExpr(%q) = %q, want an error", tt.expr, got)
			}
			if !errors.Is(err, ErrCalc) {
				t.Errorf("evalExpr(%q) error = %v, want it to match ErrCalc", tt.expr, err)
			}
			if !strings.Contains(err.Error(), tt.wantMsg) {
				t.Errorf("evalExpr(%q) error = %q, want it to contain %q", tt.expr, err, tt.wantMsg)
			}
			if got != "" {
				t.Errorf("evalExpr(%q) = %q on error, want %q", tt.expr, got, "")
			}
		})
	}
}

// The offset in a parse error is what lets the model see WHERE it went wrong.
func TestEvalExprErrorReportsAnOffset(t *testing.T) {
	_, err := evalExpr("1 + $")
	if err == nil {
		t.Fatal("err = nil, want a parse error")
	}
	if !strings.Contains(err.Error(), "at offset 4") {
		t.Errorf("err = %q, want it to report offset 4", err)
	}
}

func TestCalculatorTool(t *testing.T) {
	d := Calculator()

	t.Run("declares no host requirements", func(t *testing.T) {
		// The whole point of parsing in process rather than shelling out to bc:
		// the "read-only, needs nothing" declaration is actually true.
		if d.Needs != 0 {
			t.Errorf("Needs = %b, want 0: the calculator spawns nothing", d.Needs)
		}
		if !d.Has(tool.CapReadOnly) {
			t.Errorf("Caps = %b, want CapReadOnly", d.Caps)
		}
		if !d.Idempotent {
			t.Error("Idempotent = false, want true")
		}
		if d.Retryable != nil {
			t.Error("Retryable != nil, want nil: a pure function fails identically on a retry")
		}
	})

	t.Run("evaluates through the JSON boundary", func(t *testing.T) {
		out, err := d.Fn(context.Background(), json.RawMessage(`{"expression":"2+3*4"}`))
		if err != nil || out != "14" {
			t.Errorf("Fn = (%q, %v), want (14, nil)", out, err)
		}
	})

	t.Run("a missing expression is an empty one", func(t *testing.T) {
		_, err := d.Fn(context.Background(), json.RawMessage(`{}`))
		if err == nil || !strings.Contains(err.Error(), "must not be empty") {
			t.Errorf("err = %v, want the empty-expression message", err)
		}
	})

	t.Run("a malformed call reports invalid input", func(t *testing.T) {
		_, err := d.Fn(context.Background(), json.RawMessage(`{"expression":42}`))
		if !errors.Is(err, tool.ErrInvalidInput) {
			t.Errorf("err = %v, want errors.Is(err, tool.ErrInvalidInput)", err)
		}
		if !tool.IsCallerFault(err) {
			t.Errorf("err = %v, want it blamed on the caller", err)
		}
	})
}

func TestFormatNumber(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{0, "0"},
		{42, "42"},
		{-42, "-42"},
		{1e14, "100000000000000"},
		{1e15, "1e+15"},    // past the exact-integer window
		{1e-6, "0.000001"}, // the last value the six-decimal scale can show
		{1e-7, "1e-07"},
		{0.5, "0.5"},
		{1.0 / 3.0, "0.333333"},
		{-1.0 / 3.0, "-0.333333"},
		{2.0 / 3.0, "0.666667"},
	}
	for _, tt := range tests {
		if got := formatNumber(tt.in); got != tt.want {
			t.Errorf("formatNumber(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
