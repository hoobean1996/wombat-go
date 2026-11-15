//go:build !unix

package builtin

import "os"

// openForRead opens p for reading. There are no FIFOs to block on here.
func openForRead(p string) (*os.File, error) { return os.Open(p) }
