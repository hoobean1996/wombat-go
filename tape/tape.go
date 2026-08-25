// Package tape records and replays the model-facing side of an agent run.
//
// It exists for two jobs.
//
// Resume. A long run that crashed should not re-pay for the LLM calls it
// already made. Point the next process at the same tape and every call whose
// request is byte-identical to a recorded one is answered from the file; the
// first genuinely new call goes live and is appended.
//
// Differential testing. Run a binary against a recorded tape, diff the output,
// and you have checked that a change did not alter model-facing behavior —
// which is otherwise close to impossible, because the model is the one part of
// the system that will not give the same answer twice.
//
// # Wiring
//
//	tp, err := tape.Open("run.jsonl", tape.Auto)
//	if err != nil { return err }
//	defer func() { err = errors.Join(err, tp.Close()) }()
//
//	a, err := wombat.New(
//	    wombat.WithClient(llm.Chain(client, tp.LLM())),
//	    wombat.WithToolMiddleware(tp.Tools()),
//	    wombat.WithTools(defs...),
//	)
//
// Close's error is not decorative. A write that fails mid-run does not fail
// the call in flight — the model response has already been paid for and the
// tool side effect has already happened — so the failure is latched and
// surfaced by [Tape.Close]. Discard it and a tape can quietly stop recording.
//
// # Keying, and how this differs from its ancestor
//
// The OCaml original keyed entries by POSITION
// in the tape: the tape was a queue, each call popped the head, and a call
// that popped the wrong kind of entry died with "tape misalignment: agent path
// diverged from recording — delete tape and start over". That is a workable
// resume tool and a useless differential-testing tool, because divergence is
// precisely the thing you are trying to measure.
//
// Here an entry is keyed by a SHA-256 content hash of the canonicalised
// request. A run that reorders its calls, drops one, or adds one still hits on
// everything it has in common, and in [Auto] a genuine miss is a live call
// rather than a fatal error.
//
// The cost of that is real and worth naming: position keying gave the OCaml a
// guarantee this package does not have. Its LLM and tool entries shared one
// pending list, so they replayed in the correct INTERLEAVED order no matter
// which middleware asked first. Content keying has no global order at all.
// Each key is an independent sequence, and nothing detects that a replayed run
// visited its calls in a different order than the recording did.
//
// # What a replayed run still cannot reproduce
//
// A diff of two runs will show differences that are not behavior changes.
// Anyone using this for differential testing needs the list, so here it is
// concretely, with what to filter out:
//
//   - Durations. LLMDone.Millis and ToolDone.Millis are measured around this
//     middleware, so a replayed call reports single-digit milliseconds where
//     the recording reported seconds. Filter the "ms" field of every event.
//
//   - Timestamps. Nothing that reads a clock is taped. A wombat.WithEnvBlock
//     carrying the current date changes the system prompt, and therefore the
//     request hash, and therefore turns every LLM call into a miss the day
//     after the recording. Pin the clock or leave it out of the prompt.
//
//   - Retry jitter and backoff. llm.WithRetry draws from math/rand/v2 and
//     sleeps. A replayed hit never reaches the live client, so the retries
//     that happened during recording leave no trace at all: the recording's
//     log shows three 429s and a success, the replay shows one success.
//
//   - Trace and request ids minted by the provider. These are not fields of
//     llm.Response, so they are not on the tape and not in the replay.
//     llm.ToolUseID is the exception and it is the one that matters: it lives
//     inside the recorded content blocks, so tool_use ids replay verbatim and
//     a transcript diff stays aligned.
//
//   - Streaming granularity. Deltas are replayed only if the live call
//     produced any — that fact is recorded, because whether a client streams
//     is a property of the client and not of the request, and a replay that
//     invented deltas for a buffered recording would add events the recording
//     never had. When they are replayed, each text block arrives as ONE
//     llm.Delta, so a recording of 900 TextDelta events replays as a handful.
//     The concatenated text is identical; the event boundaries are not. Join
//     adjacent deltas before diffing. Reasoning survives only because both
//     llm/anthropic and llm/openai materialise the scratchpad into an
//     llm.Thinking block; a client that streamed Delta.Reasoning without
//     leaving one in Content would replay with the reasoning gone.
//
//   - Cache accounting. llm.Usage is recorded and replays verbatim, so
//     governor spend is reproducible — but the CacheRead/CacheWrite split in
//     the recording itself depends on server-side cache state at record time,
//     so two RECORDINGS of the same run differ here even though their replays
//     will not.
//
//   - Goroutine interleaving. With wombat.WithToolParallelism above 1, event
//     order between concurrent tools is nondeterministic live and stays
//     nondeterministic on replay. It also decides which of two concurrent
//     identical requests consumes seq 0 and which consumes seq 1.
//
//   - Error identity on the tool side. A recorded tool error is stored as its
//     message, so a replayed error is a plain error: errors.Is against
//     tool.ErrTimeout or tool.ErrCircuitOpen no longer matches. Middleware
//     that branches on a sentinel behaves differently under replay.
//
//   - Everything outside the two middlewares. A tool that reads the clock
//     directly, os.Getenv, filesystem state, a dispatcher installed with
//     wombat.WithDispatcher that bypasses the tool chain — none of it is
//     recorded and none of it is replayed.
//
// # File format
//
// JSONL, one entry per line, append-only, each line self-describing:
//
//	{"kind":"llm","key":"<sha256>","seq":0,"request":{…},"response":{…}}
//
// Every value is a struct, never a map, because Go sorts map keys and this
// file is meant to be diffed. The "key" field is the SHA-256 of the "request"
// field exactly as written on the line, so a tape can be verified with sha256
// and jq and nothing else.
package tape

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sync"
)

// Mode selects what happens on a hit and on a miss.
type Mode int

// Tape modes.
const (
	// Auto replays a hit, calls live on a miss, and records the result. This
	// is the resume mode: a tape that covers half the run pays for the other
	// half and nothing else.
	Auto Mode = iota

	// Record always calls live and always records. Use it to produce a tape.
	Record

	// Replay replays a hit and fails with [ErrTapeMiss] on a miss. Use it in
	// tests and in differential runs, where reaching the network at all means
	// the comparison is no longer hermetic.
	Replay
)

// String implements fmt.Stringer so a mode reads as itself in an error.
func (m Mode) String() string {
	switch m {
	case Auto:
		return "auto"
	case Record:
		return "record"
	case Replay:
		return "replay"
	default:
		return fmt.Sprintf("Mode(%d)", int(m))
	}
}

// ErrTapeMiss reports that [Replay] found no recorded entry for a request.
//
// The error text carries the key and, for tools, the tool name — enough to
// grep the tape for the near-miss and see what actually changed.
var ErrTapeMiss = errors.New("tape: no recorded entry")

// ErrClosed reports use of a tape after [Tape.Close].
var ErrClosed = errors.New("tape: closed")

// Stats counts what the tape did. Hits and Misses are lookups; Recorded is
// appends, so in [Auto] a miss that goes live increments both.
type Stats struct {
	Hits, Misses, Recorded int
}

// Tape is an open recording. Safe for concurrent use: one agent fans
// sub-agents across goroutines and they share the tape.
type Tape struct {
	path string
	mode Mode

	mu sync.Mutex

	// loaded is fixed at Open and is the ONLY source of replayable entries.
	//
	// Entries recorded during this session are deliberately not added. If they
	// were, a fresh Auto run that made the same request twice would replay its
	// own first answer instead of calling live — so the run would behave
	// differently from an untaped run, and a retry loop would spin on a
	// response that never changes. The recorded entry becomes replayable on
	// the NEXT open, which is exactly when resume wants it.
	loaded map[recKey][]json.RawMessage

	// cursor is how far into loaded[k] this session has consumed.
	cursor map[recKey]int

	// next is the seq to stamp on the next append under a key. Seeded from
	// len(loaded[k]) so seq stays the position of the entry within its key's
	// sequence across sessions, not within one session.
	next map[recKey]int

	stats Stats

	f *os.File

	// writeErr latches the first append failure. Reported by Close rather than
	// returned from the call that hit it: by then the response is paid for and
	// the tool's side effect has run, and failing that call would throw both
	// away to report a problem with the log.
	writeErr error
}

// recKey namespaces a hash by entry kind. The hashes are over different
// canonical shapes and would not collide, but a key that cannot be confused is
// cheaper than an argument about whether it can.
type recKey struct {
	kind string
	key  string
}

const (
	kindLLM  = "llm"
	kindTool = "tool"
)

// Open loads a tape and prepares it for the given mode.
//
// The whole file is read into memory. These files are small — a long agent run
// is a few hundred entries — and holding them lets a lookup be a map hit
// instead of a scan, which matters because every call makes one.
//
// In [Auto] and [Record] a missing file is created; in [Replay] it is an
// error, because replaying a tape that does not exist can only produce misses.
func Open(path string, mode Mode) (*Tape, error) {
	if path == "" {
		return nil, errors.New("tape: empty path")
	}
	switch mode {
	case Auto, Record, Replay:
	default:
		return nil, fmt.Errorf("tape: unknown mode %s", mode)
	}

	t := &Tape{
		path:   path,
		mode:   mode,
		loaded: make(map[recKey][]json.RawMessage),
		cursor: make(map[recKey]int),
		next:   make(map[recKey]int),
	}

	if err := t.load(); err != nil {
		return nil, err
	}
	if mode == Replay {
		// No file handle at all rather than a read-only one: a replay must not
		// be able to touch the tape it is being judged against, and the
		// clearest way to guarantee that is to have nothing to write to.
		return t, nil
	}

	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return nil, fmt.Errorf("tape: open %s for append: %w", path, err)
	}
	t.f = f
	return t, nil
}

// load reads every line into the replay pool.
//
// A line that will not parse is a hard error, including a torn final line from
// a crash mid-append. The alternative — dropping it — is the one thing the
// package must not do, because a silently shortened tape looks exactly like a
// tape whose run made fewer calls. The error names the line so that deleting
// it is a one-command fix.
func (t *Tape) load() error {
	f, err := os.Open(t.path)
	if err != nil {
		if os.IsNotExist(err) && t.mode != Replay {
			return nil
		}
		return fmt.Errorf("tape: open %s: %w", t.path, err)
	}
	defer f.Close()

	// bufio.Reader and not bufio.Scanner: a line holds a whole transcript and
	// routinely exceeds Scanner's 64 KiB ceiling, which it reports as a
	// generic "token too long" long after the cause is recoverable.
	br := bufio.NewReader(f)

	byKey := make(map[recKey][]seqEntry)

	for lineNo := 1; ; lineNo++ {
		line, rerr := br.ReadBytes('\n')
		if len(line) > 0 {
			if err := t.loadLine(byKey, line, lineNo); err != nil {
				return err
			}
		}
		if rerr != nil {
			if errors.Is(rerr, io.EOF) {
				break
			}
			return fmt.Errorf("tape: read %s: %w", t.path, rerr)
		}
	}

	for k, es := range byKey {
		// Sort by seq so a tape that was concatenated or hand-edited still
		// replays a key's responses in the order they were recorded in. Stable,
		// so duplicate seq numbers keep file order rather than shuffling.
		slices.SortStableFunc(es, func(a, b seqEntry) int {
			switch {
			case a.seq < b.seq:
				return -1
			case a.seq > b.seq:
				return 1
			}
			return 0
		})
		resps := make([]json.RawMessage, len(es))
		for i, e := range es {
			resps[i] = e.resp
		}
		t.loaded[k] = resps
		t.next[k] = len(resps)
	}
	return nil
}

// seqEntry is one loaded line, held with its seq until the whole file is read
// and each key's responses can be put back in recorded order.
type seqEntry struct {
	seq  int
	resp json.RawMessage
}

func (t *Tape) loadLine(byKey map[recKey][]seqEntry, line []byte, lineNo int) error {
	trimmed := trimSpace(line)
	if len(trimmed) == 0 {
		return nil
	}
	var e entry
	if err := json.Unmarshal(trimmed, &e); err != nil {
		return fmt.Errorf("tape: %s line %d is not a tape entry (delete the line to recover the rest): %w", t.path, lineNo, err)
	}
	switch e.Kind {
	case kindLLM, kindTool:
	default:
		return fmt.Errorf("tape: %s line %d has unknown kind %q", t.path, lineNo, e.Kind)
	}
	if e.Key == "" {
		return fmt.Errorf("tape: %s line %d has no key", t.path, lineNo)
	}
	k := recKey{kind: e.Kind, key: e.Key}
	byKey[k] = append(byKey[k], seqEntry{seq: e.Seq, resp: e.Response})
	return nil
}

// trimSpace drops the trailing newline and any stray whitespace without
// pulling in strings for one call on a byte slice.
func trimSpace(b []byte) []byte {
	i, j := 0, len(b)
	for i < j && isSpace(b[i]) {
		i++
	}
	for j > i && isSpace(b[j-1]) {
		j--
	}
	return b[i:j]
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

// Stats returns a snapshot of what this session has done.
func (t *Tape) Stats() Stats {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.stats
}

// Close flushes nothing — every entry was written as it was produced — and
// returns the first append failure, if there was one.
//
// Idempotent, so a deferred Close after an explicit one is harmless.
func (t *Tape) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	var cerr error
	if t.f != nil {
		cerr = t.f.Close()
		t.f = nil
	}
	if t.writeErr != nil {
		return t.writeErr
	}
	if cerr != nil {
		return fmt.Errorf("tape: close %s: %w", t.path, cerr)
	}
	return nil
}

// take returns the next recorded response for k, if the tape has one.
//
// A key maps to a SEQUENCE, because identical requests recur legitimately: a
// retry after a 429, two sub-agents handed the same sub-question, a loop that
// re-asks with an unchanged transcript. Collapsing them to one response would
// make every retry return the first answer, which for the "should I stop now?"
// class of prompt turns a two-turn run into an infinite one.
//
// When the sequence is exhausted the LAST response repeats. The alternatives
// were both worse:
//
//   - Fail. That is the OCaml's "tape misalignment" wearing a different hat.
//     A run that makes one more identical call than the recording — one extra
//     retry, one more sub-agent — would die, and the whole point of content
//     keying is that such a run should still be comparable.
//
//   - Fall through to live. In Auto that quietly reintroduces cost and
//     nondeterminism at the exact point a run diverges, which is the point you
//     most want held fixed; in Replay it is not available at all.
//
// Repeating is the choice that keeps a diverged run running and keeps Auto and
// Replay behaving the same way. The hazard it creates is worth stating: a run
// that has genuinely fallen into a loop repeating one request will replay
// forever rather than erroring. Nothing here guards that — wombat.WithMaxIters
// and governor.Limits do, and they are the right place for it.
func (t *Tape) take(k recKey) (json.RawMessage, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	seq := t.loaded[k]
	if len(seq) == 0 {
		t.stats.Misses++
		return nil, false
	}
	i := t.cursor[k]
	if i >= len(seq) {
		i = len(seq) - 1
	} else {
		t.cursor[k] = i + 1
	}
	t.stats.Hits++
	return seq[i], true
}

// record appends one entry and stamps it with the next seq for its key.
//
// The line is built in full and handed to a single Write on an O_APPEND
// descriptor, so a concurrent recorder cannot interleave bytes into the middle
// of it. There is no bufio in the path and no Sync either: the write reaches
// the page cache before the call returns, which is what makes a process crash
// survivable, and fsync per entry would cost more than the tape saves.
func (t *Tape) record(k recKey, req, resp json.RawMessage) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.f == nil {
		t.setWriteErr(fmt.Errorf("tape: record into %s: %w", t.path, ErrClosed))
		return
	}

	seq := t.next[k]
	t.next[k] = seq + 1

	line, err := canonical(entry{
		Kind:     k.kind,
		Key:      k.key,
		Seq:      seq,
		Request:  req,
		Response: resp,
	})
	if err != nil {
		t.setWriteErr(fmt.Errorf("tape: encode %s entry: %w", k.kind, err))
		return
	}
	if _, err := t.f.Write(append(line, '\n')); err != nil {
		t.setWriteErr(fmt.Errorf("tape: append to %s: %w", t.path, err))
		return
	}
	t.stats.Recorded++
}

// setWriteErr keeps the FIRST failure. A full disk produces one error per call
// for the rest of the run, and the first one is the one with the useful cause.
// Callers hold t.mu.
func (t *Tape) setWriteErr(err error) {
	if t.writeErr == nil {
		t.writeErr = err
	}
}

// countMiss records a lookup that was skipped rather than attempted, so Record
// mode's Stats still add up to one lookup per call.
func (t *Tape) countMiss() {
	t.mu.Lock()
	t.stats.Misses++
	t.mu.Unlock()
}
