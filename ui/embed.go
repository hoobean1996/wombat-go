// Package ui is wombatd's browser front end: a Vite + React app whose built
// files, in dist/, are embedded here so the binary serves them itself.
//
// dist/ is not committed, apart from the dist/.gitkeep that lets this package
// compile on a checkout where nobody has built the front end; `make ui` builds
// it. A wombatd built without it serves a page saying so.
package ui

import (
	"embed"
	"io/fs"
)

//go:embed all:dist
var dist embed.FS

// FS returns the built front end, index.html at its root.
func FS() fs.FS {
	sub, err := fs.Sub(dist, "dist")
	if err != nil {
		panic(err) // "dist" is a valid path, and fs.Sub fails on nothing else
	}
	return sub
}

// Built reports whether the front end was built into this binary.
func Built() bool {
	_, err := fs.Stat(FS(), "index.html")
	return err == nil
}
