//go:build unix

package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/tool"
)

// TestReadFileRefusesWhatItCannotFinish: os.ReadFile read until EOF, so a
// FIFO blocked view_file past its deadline, /dev/zero would fill memory, and a
// 64 MiB file cost 256 MiB to return 8 KB — and cancelling the context could
// stop none of them.
//
// The device read here is /dev/null, not /dev/zero: a regression must fail
// this test, not take the machine's memory with it.
func TestReadFileRefusesWhatItCannotFinish(t *testing.T) {
	dir := t.TempDir()
	fifo := filepath.Join(dir, "pipe")
	if err := syscall.Mkfifo(fifo, 0o600); err != nil {
		t.Skip(err)
	}
	big := filepath.Join(dir, "big")
	f, err := os.Create(big)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(MaxReadBytes + 1); err != nil { // sparse: no disk used
		t.Fatal(err)
	}
	f.Close()

	fsys := OSFS("")
	for _, tc := range []struct {
		name, path string
		want       error
	}{
		{"a FIFO", fifo, ErrNotRegularFile},
		{"a character device", "/dev/null", ErrNotRegularFile},
		{"a file past the limit", big, ErrFileTooLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			err := within(t, fifo, func() error { _, err := fsys.ReadFile(context.Background(), tc.path); return err })
			if !errors.Is(err, tc.want) || !tool.IsCallerFault(err) {
				t.Errorf("err = %v, want %v, as a caller fault", err, tc.want)
			}
			runtime.ReadMemStats(&after)
			if grew := after.TotalAlloc - before.TotalAlloc; grew > 4<<20 {
				t.Errorf("refusing it allocated %d bytes, want it refused before reading", grew)
			}
		})
	}

	t.Run("a file at the limit is read", func(t *testing.T) {
		at := filepath.Join(dir, "at")
		f, _ := os.Create(at)
		_ = f.Truncate(MaxReadBytes)
		f.Close()
		data, err := fsys.ReadFile(context.Background(), at)
		if err != nil || len(data) != MaxReadBytes {
			t.Errorf("got %d bytes, %v; want the whole file", len(data), err)
		}
	})

	t.Run("a cancelled context reads nothing", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := fsys.ReadFile(ctx, big); !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want the cancellation", err)
		}
	})

	t.Run("view_file on a FIFO answers instead of hanging", func(t *testing.T) {
		err := within(t, fifo, func() error {
			_, err := ViewFile(fsys).Fn(context.Background(), json.RawMessage(`{"path":"`+fifo+`"}`))
			return err
		})
		if !errors.Is(err, ErrNotRegularFile) {
			t.Errorf("err = %v, want ErrNotRegularFile", err)
		}
	})
}

// within runs fn and fails the test if it has not returned in five seconds —
// first opening fifo for writing, which releases a reader blocked opening it,
// so that a regression fails here rather than hanging the whole run.
func within(t *testing.T, fifo string, fn func() error) error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- fn() }()
	select {
	case err := <-done:
		return err
	case <-time.After(5 * time.Second):
		if w, err := os.OpenFile(fifo, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil {
			w.Close()
		}
		select {
		case <-done:
		case <-time.After(5 * time.Second):
		}
		t.Fatal("the read did not return within five seconds")
		return nil
	}
}
