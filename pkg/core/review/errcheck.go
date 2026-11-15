package review

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// discardedErrors reports every place a test file throws away the last return
// value of a call with `_`.
//
// By Go convention that position is the error, so this finds the construct that
// produced the one defect that survived a plan naming it, an implementation
// attempting it, a correct test driving it, and a review:
//
//	if _, ok, _ := q.Claim("w"); ok { t.Fatal("should not claim") }
//
// Claim returned (nil, false, err). ok was false, so the assertion held and the
// test went green while the call was returning an error that killed the CLI.
//
// It is deliberately mechanical. No model is needed to see a `_`, and the model
// that wrote the test is exactly the one that already looked past it.
//
// Not every hit is a defect — discarding the error from a helper is often fine.
// These are leads, not findings, and they are reported as such.
func discardedErrors(dir string) []string {
	if dir == "" {
		return nil
	}
	var out []string
	fset := token.NewFileSet()

	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // unreadable subtree: skip, never fail the review
		}
		if d.IsDir() {
			if name := d.Name(); name != "." && (strings.HasPrefix(name, ".") || name == "vendor" || name == "testdata") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, "_test.go") || isReviewFile(filepath.Base(path)) {
			return nil
		}
		f, perr := parser.ParseFile(fset, path, nil, 0)
		if perr != nil {
			return nil // unparseable: the compiler will say so, not the review
		}
		rel, rerr := filepath.Rel(dir, path)
		if rerr != nil {
			rel = path
		}
		ast.Inspect(f, func(n ast.Node) bool {
			var lhs []ast.Expr
			var rhs []ast.Expr
			switch s := n.(type) {
			case *ast.AssignStmt:
				lhs, rhs = s.Lhs, s.Rhs
			default:
				return true
			}
			// Only the multi-value form `a, b, _ := f()` discards a result of a
			// call; `_ = x` and `a, b := c, d` are something else.
			if len(rhs) != 1 || len(lhs) < 2 {
				return true
			}
			call, ok := rhs[0].(*ast.CallExpr)
			if !ok {
				return true
			}
			last, ok := lhs[len(lhs)-1].(*ast.Ident)
			if !ok || last.Name != "_" {
				return true
			}
			pos := fset.Position(last.Pos())
			out = append(out, fmt.Sprintf("%s:%d: %s(...) — last result discarded", rel, pos.Line, exprName(call.Fun)))
			return true
		})
		return nil
	})
	if err != nil {
		return nil
	}
	sort.Strings(out)
	return out
}

// exprName renders the callee for a lead line: `q.Claim`, `Open`, `pkg.New`.
func exprName(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.SelectorExpr:
		return exprName(x.X) + "." + x.Sel.Name
	case *ast.IndexExpr:
		return exprName(x.X)
	case *ast.CallExpr:
		return exprName(x.Fun)
	case *ast.ParenExpr:
		return exprName(x.X)
	}
	return "call"
}
