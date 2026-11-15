package tool

import (
	"context"
	"errors"
	"sync"
	"time"

	"wombat/pkg/core/llm"
)

// Handler runs one tool call. Middleware wraps it.
//
// The signature takes the resolved Def rather than a name so that middleware
// can read policy metadata (Idempotent, Timeout, Retryable) without a second
// lookup.
type Handler func(ctx context.Context, d Def, use llm.ToolUse) (string, error)

// Middleware adds one behavior to a Handler.
type Middleware func(Handler) Handler

// Chain wraps h in mws. Later entries end up further OUT:
//
//	Chain(Direct, WithRetry(p), WithLogging(l))  // logging sees the post-retry verdict
func Chain(h Handler, mws ...Middleware) Handler {
	for _, mw := range mws {
		h = mw(h)
	}
	return h
}

// Direct is the leaf handler: it invokes the tool.
func Direct(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
	if d.Fn == nil {
		return "", errors.New("tool: " + d.Name + " has no handler")
	}
	return d.Fn(ctx, use.Input)
}

// ErrUnknownTool is returned when the model calls a name not in the set.
// It reaches the model as an is_error tool_result, which is usually enough
// for it to correct itself. A [CallerFault]: there is no tool here to be
// unhealthy.
var ErrUnknownTool = CallerError(errors.New("tool: unknown tool"))

// Result is the outcome of one dispatched call.
type Result struct {
	UseID  llm.ToolUseID
	Name   string
	Output string
	Err    error
	Dur    time.Duration

	// Tags are labels the tool attached with [Annotate]. They never reach the
	// model — they are the harness's own notes about what this observation
	// is, so a transcript strategy can act on meaning rather than position.
	Tags []string

	// Media is what the tool attached with [Attach]. Unlike Tags it does reach
	// the model, alongside Output.
	Media []llm.Media
}

// Block converts the result into the content block that closes the tool_use.
// Errors are surfaced to the model as text with is_error set, never dropped:
// the model needs to see what went wrong to pick a different approach.
func (r Result) Block() llm.ContentBlock {
	if r.Err != nil {
		return llm.ToolResult{ToolUseID: r.UseID, Content: r.Err.Error(), IsError: true}
	}
	return llm.ToolResult{ToolUseID: r.UseID, Content: r.Output, IsError: false, Media: r.Media}
}

// Size is how many bytes the result puts in front of the model: its output
// text and whatever media it carries.
func (r Result) Size() int {
	n := len(r.Output)
	for _, m := range r.Media {
		n += len(m.Data)
	}
	return n
}

// Blocks converts a batch, preserving order.
func Blocks(rs []Result) []llm.ContentBlock {
	out := make([]llm.ContentBlock, len(rs))
	for i, r := range rs {
		out[i] = r.Block()
	}
	return out
}

// Dispatcher executes a batch of tool_use blocks against a set and returns
// results in input order.
type Dispatcher interface {
	Dispatch(ctx context.Context, set Set, uses []llm.ToolUse) []Result
}

// DispatcherFunc adapts a function to Dispatcher.
type DispatcherFunc func(ctx context.Context, set Set, uses []llm.ToolUse) []Result

// Dispatch implements Dispatcher.
func (f DispatcherFunc) Dispatch(ctx context.Context, set Set, uses []llm.ToolUse) []Result {
	return f(ctx, set, uses)
}

type dispatcher struct {
	h        Handler
	parallel int
}

// DispatchOption configures a Dispatcher.
type DispatchOption func(*dispatcher)

// WithParallelism runs up to n calls of a batch concurrently — the calls that
// only read.
//
// Goroutines make concurrent dispatch trivial; what makes it unsafe is
// semantic. A batch may contain two writes to the same path, or a write and the
// `go test` that is meant to see it, and the model does not promise its batch
// is independent — it wrote the calls in an order and may well mean it.
//
// So the batch is scheduled by what each tool declares. A run of consecutive
// calls to [CapReadOnly] tools is fanned out, up to n at a time; any other
// call — mutating, exec, a tool that declares nothing, a name the set does not
// know — is a barrier: it starts only when everything before it has finished,
// and nothing after it starts until it has. Results come back in the model's
// order regardless. A turn of five greps and a view_file runs at once; a turn
// that writes a file and then tests it runs exactly as written.
//
// The default is 1, which runs everything in order. The agent's own default is
// higher; see core.WithToolParallelism.
func WithParallelism(n int) DispatchOption {
	return func(d *dispatcher) {
		if n > 0 {
			d.parallel = n
		}
	}
}

// NewDispatcher builds a Dispatcher over a handler chain.
func NewDispatcher(h Handler, opts ...DispatchOption) Dispatcher {
	d := &dispatcher{h: h, parallel: 1}
	for _, o := range opts {
		o(d)
	}
	return d
}

func (d *dispatcher) Dispatch(ctx context.Context, set Set, uses []llm.ToolUse) []Result {
	out := make([]Result, len(uses))

	run := func(i int) {
		u := uses[i]
		out[i] = Result{UseID: u.ID, Name: u.Name}

		// The backstop. [WithRecovery] covers only what is INSIDE it, and there
		// is a lot outside: a middleware that panics — a bad RetryPolicy, an
		// observer callback into UI code — or a panic in the bookkeeping right
		// here. Under WithParallelism that panic escapes on a worker goroutine,
		// where no caller can recover it and the process is simply gone, so the
		// recover has to live on the goroutine that runs the call.
		//
		// Recording it as Result.Err rather than re-panicking keeps the
		// invariant every caller of Dispatch relies on: one result per tool_use,
		// in input order. A dropped result would leave a dangling tool_use that
		// the provider rejects on the next turn.
		started := time.Now()
		defer func() {
			if v := recover(); v != nil {
				out[i].Output = ""
				out[i].Err = recovered(ctx, u.Name, v)
				out[i].Dur = time.Since(started)
			}
		}()

		def, ok := set.Find(u.Name)
		if !ok {
			out[i].Err = ErrUnknownTool
			return
		}

		ann := &annotations{}
		callCtx := context.WithValue(WithInfo(ctx, Info{UseID: u.ID}), annotateKey{}, ann)

		s, err := d.h(callCtx, def, u)
		out[i].Output, out[i].Err, out[i].Dur = s, err, time.Since(started)

		ann.mu.Lock()
		out[i].Tags = ann.tags
		if err == nil {
			out[i].Media = ann.media
		}
		ann.mu.Unlock()
	}

	if d.parallel <= 1 || len(uses) == 1 {
		for i := range uses {
			run(i)
		}
		return out
	}

	sem := make(chan struct{}, d.parallel)
	for start := 0; start < len(uses); {
		if !concurrent(set, uses[start]) {
			run(start) // a barrier: alone, in order
			start++
			continue
		}
		end := start + 1
		for end < len(uses) && concurrent(set, uses[end]) {
			end++
		}
		var wg sync.WaitGroup
		for i := start; i < end; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				run(i)
			}()
		}
		wg.Wait()
		start = end
	}
	return out
}

// concurrent reports whether a call may run alongside its neighbours: a tool
// that declares it only reads, and declares nothing that contradicts it.
func concurrent(set Set, u llm.ToolUse) bool {
	d, ok := set.Find(u.Name)
	return ok && d.Has(CapReadOnly) && d.Caps&(CapMutating|CapExec|CapPause|CapTerminal) == 0
}
