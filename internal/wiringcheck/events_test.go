package wiringcheck

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"wombat/pkg/core"
	"wombat/pkg/core/permission"
	"wombat/pkg/core/review"
	"wombat/pkg/daemon"
)

// TestEveryEventIsRegistered: each EventTypes() is a list kept by hand, and a
// type left out of it still marshals and still streams — but every check that
// walks the list, like the core's test that each event puts its "type" first,
// never sees it, and neither does a renderer that walks the list to learn what
// can arrive.
//
// An event is anything with a Kind() string method, which is all core.Event
// asks. The test reads the source for them, so it needs no list of its own.
func TestEveryEventIsRegistered(t *testing.T) {
	for _, pkg := range []struct {
		dir        string
		registered []core.Event
	}{
		{"../../pkg/core", core.EventTypes()},
		{"../../pkg/core/permission", permission.EventTypes()},
		{"../../pkg/daemon", daemon.EventTypes()},
		{"../../pkg/core/review", review.EventTypes()},
	} {
		var names []string
		for _, e := range pkg.registered {
			names = append(names, reflect.TypeOf(e).Name())
		}
		defined := kindTypes(t, pkg.dir)
		if len(defined) == 0 {
			t.Errorf("%s: found no event types at all; the source walk is broken", pkg.dir)
		}
		for _, name := range defined {
			if !slices.Contains(names, name) {
				t.Errorf("%s: %s has a Kind method but is not in EventTypes()", pkg.dir, name)
			}
		}
	}
}

// kindTypes names the types declared in dir, outside its tests, that have a
// method Kind() string.
func kindTypes(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var out []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Recv == nil || fn.Name.Name != "Kind" || fn.Type.Params.NumFields() != 0 || fn.Type.Results.NumFields() != 1 {
				continue
			}
			if res, ok := fn.Type.Results.List[0].Type.(*ast.Ident); !ok || res.Name != "string" {
				continue
			}
			recv := fn.Recv.List[0].Type
			if star, ok := recv.(*ast.StarExpr); ok {
				recv = star.X
			}
			if id, ok := recv.(*ast.Ident); ok {
				out = append(out, id.Name)
			}
		}
	}
	return out
}
