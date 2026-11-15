//go:build unix

package builtin

import (
	"os"
	"syscall"
)

// openForRead opens p without blocking: opening a FIFO for reading otherwise
// waits for a writer, which may never come. Reads of a regular file are not
// affected by the flag.
func openForRead(p string) (*os.File, error) {
	return os.OpenFile(p, os.O_RDONLY|syscall.O_NONBLOCK, 0)
}
