package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"time"

	"wombat/internal/safeenv"
)

// maxMessage bounds one message from a server: a base64 document of the
// largest size a model is shown, with room to spare.
const maxMessage = 32 << 20

// stopGrace is how long a stdio server is given to exit after its input is
// closed, and again after it is sent SIGTERM, before it is killed.
var stopGrace = 2 * time.Second

// stdio is a server running as a child process, one JSON-RPC message per
// line on its stdin and stdout.
type stdio struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *io.PipeReader
	wmu    sync.Mutex
	stderr *tailBuffer

	exited  chan struct{}
	waitErr error // set before exited is closed
	once    sync.Once
}

// startStdio starts the server. Nothing it writes is read until listen: the
// pipe holds it, and the Client has to have its transport before the first
// message can be answered — a server may ping before the handshake is done.
func startStdio(s Server) (*stdio, error) {
	cmd := exec.Command(s.Command, s.Args...)
	cmd.Dir = s.Dir
	cmd.Env = environ(s.Env)
	isolate(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	// Through an io.Pipe rather than StdoutPipe, so Wait can run as soon as
	// the process exits: StdoutPipe may not be waited on until every read
	// from it is done, and a grandchild that inherited the pipe can keep it
	// open long after the server is gone. WaitDelay then closes it.
	pr, pw := io.Pipe()
	cmd.Stdout = pw
	t := &stdio{cmd: cmd, stdin: stdin, stdout: pr, stderr: &tailBuffer{max: 4 << 10}, exited: make(chan struct{})}
	cmd.Stderr = t.stderr
	cmd.WaitDelay = stopGrace
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	go func() {
		t.waitErr = cmd.Wait()
		close(t.exited)
		_ = pw.Close()
	}()
	return t, nil
}

// listen hands everything the server writes to c, from now until it exits.
func (t *stdio) listen(c *Client) { go t.read(t.stdout, c) }

// environ is the filtered environment plus what the server's configuration
// sets, in a stable order.
func environ(set map[string]string) []string {
	kv := make([]string, 0, len(set))
	for k, v := range set {
		kv = append(kv, k+"="+v)
	}
	slices.Sort(kv)
	return safeenv.Overlay(safeenv.Environ(), kv)
}

func (t *stdio) read(r *io.PipeReader, c *Client) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), maxMessage)
	for sc.Scan() {
		c.receive(sc.Bytes())
	}
	readErr := sc.Err()
	if readErr != nil {
		// The stream cannot be resynchronised after a message too long to
		// read, so this connection is over. Stop the copy into the pipe and
		// the server, so Wait can return.
		_ = r.CloseWithError(readErr)
		_ = stopProcess(t.cmd.Process, false)
	}
	<-t.exited
	c.fail(t.exitError(readErr))
}

func (t *stdio) exitError(readErr error) error {
	var b strings.Builder
	if readErr != nil {
		fmt.Fprintf(&b, "reading from the server failed: %v", readErr)
	} else {
		b.WriteString("the server exited")
		if t.waitErr != nil {
			fmt.Fprintf(&b, " (%v)", t.waitErr)
		}
	}
	if tail := strings.TrimSpace(t.stderr.String()); tail != "" {
		fmt.Fprintf(&b, "; the end of its stderr:\n%s", tail)
	}
	return errors.New(b.String())
}

func (t *stdio) send(ctx context.Context, m message) error {
	raw, err := json.Marshal(m)
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	// A server that stops reading its stdin fills the pipe, and the write
	// blocks; the call it belongs to must still be able to give up. The write
	// carries on in the background, still holding the lock so messages are
	// never interleaved, until the server reads or Close breaks the pipe.
	done := make(chan error, 1)
	go func() {
		t.wmu.Lock()
		defer t.wmu.Unlock()
		_, err := t.stdin.Write(raw)
		done <- err
	}()
	select {
	case err := <-done:
		// A write fails because the server is gone or going, and "broken
		// pipe" says neither why nor that it is gone: wait for the exit, and
		// report that.
		if err != nil && t.waitExit(stopGrace) {
			return t.exitError(nil)
		}
		return err
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

// close stops the server the way the specification asks: close its input and
// let it exit, then SIGTERM, then SIGKILL — to its whole process group, since
// `npx some-server` is npm, which started node.
func (t *stdio) close() error {
	t.once.Do(func() {
		_ = t.stdin.Close() // not under wmu: a blocked write must not hold Close up
		if t.waitExit(stopGrace) {
			return
		}
		_ = stopProcess(t.cmd.Process, true)
		if t.waitExit(stopGrace) {
			return
		}
		_ = stopProcess(t.cmd.Process, false)
		<-t.exited
	})
	return nil
}

func (t *stdio) waitExit(d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-t.exited:
		return true
	case <-timer.C:
		return false
	}
}

// tailBuffer keeps the last max bytes written to it: the end of a server's
// stderr, which is where it says why it died.
type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
	max int
}

func (b *tailBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf = append(b.buf, p...)
	if over := len(b.buf) - b.max; over > 0 {
		b.buf = append(b.buf[:0], b.buf[over:]...)
	}
	return len(p), nil
}

func (b *tailBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return strings.ToValidUTF8(string(b.buf), "")
}
