package httpcall

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/hoobean1996/wombat-go/llm"
)

func TestRetryAfter(t *testing.T) {
	for _, tc := range []struct {
		hdr  string
		want time.Duration
	}{
		{"", 0},
		{"12", 12 * time.Second},
		{"1.5", 1500 * time.Millisecond},
		{"0", 0},
		{"-5", 0},
		{"NaN", 0},
		{"soon", 0},
		{"Mon, 02 Jan 2006 15:04:05 GMT", 0}, // a date in the past
	} {
		h := http.Header{}
		if tc.hdr != "" {
			h.Set("Retry-After", tc.hdr)
		}
		if got := RetryAfter(h); got != tc.want {
			t.Errorf("RetryAfter(%q) = %v, want %v", tc.hdr, got, tc.want)
		}
	}
	h := http.Header{}
	h.Set("Retry-After", time.Now().Add(90*time.Second).UTC().Format(http.TimeFormat))
	if got := RetryAfter(h); got < 80*time.Second || got > 91*time.Second {
		t.Errorf("RetryAfter(a date 90s out) = %v, want about 90s", got)
	}
}

// TestAbort pins the retry-versus-stop distinction.
func TestAbort(t *testing.T) {
	t.Run("a live context is transport", func(t *testing.T) {
		if err := Abort(context.Background(), "p", errors.New("refused")); !errors.Is(err, llm.ErrTransport) {
			t.Errorf("err = %v, want transport", err)
		}
	})
	t.Run("a silent stream is transport", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(ErrStreamIdle)
		err := Abort(ctx, "p", errors.New("body closed"))
		if !errors.Is(err, llm.ErrTransport) || !errors.Is(err, ErrStreamIdle) {
			t.Errorf("err = %v, want transport carrying ErrStreamIdle", err)
		}
	})
	t.Run("a deadline is transport", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
		defer cancel()
		<-ctx.Done()
		if err := Abort(ctx, "p", errors.New("body closed")); !errors.Is(err, llm.ErrTransport) {
			t.Errorf("err = %v, want transport", err)
		}
	})
	t.Run("a deliberate cancellation keeps its cause and is not transport", func(t *testing.T) {
		stop := errors.New("budget exhausted")
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(stop)
		err := Abort(ctx, "p", errors.New("body closed"))
		if !errors.Is(err, stop) || errors.Is(err, llm.ErrTransport) {
			t.Errorf("err = %v, want the cause and not transport", err)
		}
	})
}

func TestSettleOnlyReclassifiesFailuresOfADoneContext(t *testing.T) {
	if err := Settle(context.Background(), "p", nil); err != nil {
		t.Errorf("Settle(nil) = %v", err)
	}
	plain := errors.New("bad json")
	if err := Settle(context.Background(), "p", plain); err != plain {
		t.Errorf("Settle on a live context rewrote %v into %v", plain, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Settle(ctx, "p", plain); !errors.Is(err, context.Canceled) {
		t.Errorf("Settle on a cancelled context = %v, want the cancellation", err)
	}
}

func TestBoundOnlyWhenTheCallerHasNoDeadline(t *testing.T) {
	ctx, cancel := Bound(context.Background(), time.Hour)
	defer cancel()
	if _, ok := ctx.Deadline(); !ok {
		t.Error("no deadline applied to a context without one")
	}

	mine, cancelMine := context.WithTimeout(context.Background(), 10*time.Hour)
	defer cancelMine()
	got, cancel2 := Bound(mine, time.Hour)
	defer cancel2()
	if got != mine {
		t.Error("Bound replaced a caller's own deadline")
	}
}

func TestWatchdogResetsOnBytes(t *testing.T) {
	ctx, w := Watch(context.Background(), 40*time.Millisecond)
	defer w.Stop()
	pr, pw := io.Pipe()
	body := w.Body(pr)
	go func() {
		for range 6 {
			time.Sleep(15 * time.Millisecond)
			pw.Write([]byte(":\n"))
		}
		pw.Close()
	}()
	if _, err := io.ReadAll(body); err != nil {
		t.Fatal(err)
	}
	if ctx.Err() != nil {
		t.Errorf("the watchdog fired although bytes kept arriving: %v", context.Cause(ctx))
	}

	ctx2, w2 := Watch(context.Background(), 20*time.Millisecond)
	defer w2.Stop()
	_ = w2.Body(strings.NewReader(""))
	<-ctx2.Done()
	if !errors.Is(context.Cause(ctx2), ErrStreamIdle) {
		t.Errorf("cause = %v, want ErrStreamIdle", context.Cause(ctx2))
	}
}

func TestTruncateStopsOnARuneBoundary(t *testing.T) {
	got := Truncate("中文", 4) // two 3-byte runes; a cut at 4 backs off to 3
	if !strings.HasPrefix(got, "中") || strings.ContainsRune(got, '�') {
		t.Errorf("Truncate = %q, want a clean cut after 中", got)
	}
	if got := Truncate("short", 100); got != "short" {
		t.Errorf("Truncate(short) = %q", got)
	}
}

func TestReadBody(t *testing.T) {
	if _, err := ReadBody(strings.NewReader("12345"), "p", 4); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("err = %v, want the bound to refuse a 5-byte body at 4", err)
	}
	if raw, err := ReadBody(strings.NewReader("1234"), "p", 4); err != nil || string(raw) != "1234" {
		t.Errorf("ReadBody at exactly the limit = %q, %v", raw, err)
	}
}
