package wombat

import (
	"context"

	"github.com/hoobean1996/wombat-go/llm"
)

// Token estimation.
//
// A strategy deciding what fits in a context window has to talk in tokens,
// and counting them exactly needs the provider's tokenizer, which is neither
// available offline nor the same between providers. What IS available is the
// provider's own count of every request it has read: Usage.PromptTokens. So
// the estimate is a byte count, calibrated against that count.
//
// Each call teaches the estimator how many tokens one byte of this run's
// transcript costs — code, prose and CJK differ by a factor of three — and the
// next request is estimated from its bytes at that rate. Before the first call
// the rate is the familiar four bytes a token, which is the right order of
// magnitude for English and code and too optimistic for CJK; the first
// measurement corrects it.
//
// Media is the exception and is estimated on its own: an image costs tokens by
// its pixel area, not its file size, and pricing a five-megabyte PNG at four
// bytes a token would call it a million tokens when the provider charges
// about sixteen hundred.
const (
	defaultTokensPerByte = 0.25
	imageTokens          = 1600
	pdfTokensPerKiB      = 40
)

// tokenEstimator is one run's calibrated byte-to-token rate. Loop-owned: it is
// observed and read from the loop goroutine and from the LLM middleware that
// call runs through, never concurrently.
type tokenEstimator struct {
	perByte    float64
	calibrated bool
	// fixedBytes is the size of the request before any message: the system
	// prompt and the tool definitions.
	fixedBytes int
}

func newTokenEstimator() *tokenEstimator { return &tokenEstimator{perByte: defaultTokensPerByte} }

// estimate is the number of input tokens a request carrying msgs would cost.
func (e *tokenEstimator) estimate(msgs []llm.Message) int {
	if e == nil {
		return int(float64(transcriptBytes(msgs))*defaultTokensPerByte) + mediaTokens(msgs)
	}
	return int(float64(e.fixedBytes+transcriptBytes(msgs))*e.perByte) + mediaTokens(msgs)
}

// observe calibrates from one call: a request with msgs and fixedBytes of
// system prompt and tool definitions, which the provider counted as prompt
// tokens. A call that reported no usage — some streaming gateways never do —
// teaches nothing and changes nothing.
//
// The rate is sticky. It moves to a new measurement only when that differs from
// it by more than a tenth, because strategies quantize on token mass to hold
// the cache prefix still, and a rate that wandered by a percent every call
// would move their boundaries with it. An estimate within ten percent is as
// good as these decisions need.
func (e *tokenEstimator) observe(msgs []llm.Message, fixedBytes, prompt int) {
	e.fixedBytes = fixedBytes
	text := fixedBytes + transcriptBytes(msgs)
	prompt -= mediaTokens(msgs)
	if prompt <= 0 || text <= 0 {
		return
	}
	measured := float64(prompt) / float64(text)
	if !e.calibrated || measured > e.perByte*1.1 || measured < e.perByte*0.9 {
		e.perByte = measured
		e.calibrated = true
	}
}

// transcriptBytes is the size of everything in msgs a tokenizer reads as text.
func transcriptBytes(msgs []llm.Message) int {
	n := 0
	for _, m := range msgs {
		n += 4 // a turn's own framing
		for _, b := range m.Content {
			switch v := b.(type) {
			case llm.Text:
				n += len(v.Text)
			case llm.Thinking:
				n += len(v.Text)
			case llm.ToolUse:
				n += len(v.Name) + len(v.Input) + len(v.ID)
			case llm.ToolResult:
				n += len(v.Content) + len(v.ToolUseID)
			}
		}
	}
	return n
}

// mediaTokens is the estimated cost of the media in msgs: a flat price per
// image, which providers scale to a bounded size before reading, and a price
// per KiB for a document, whose pages are read both as text and as pictures.
func mediaTokens(msgs []llm.Message) int {
	n := 0
	add := func(m llm.Media) {
		if m.IsImage() {
			n += imageTokens
		} else {
			n += max(imageTokens, len(m.Data)/1024*pdfTokensPerKiB)
		}
	}
	for _, msg := range msgs {
		for _, b := range msg.Content {
			switch v := b.(type) {
			case llm.Media:
				add(v)
			case llm.ToolResult:
				for _, m := range v.Media {
					add(m)
				}
			}
		}
	}
	return n
}

// specBytes is the size of the tool definitions a request carries.
func specBytes(specs []llm.ToolSpec) int {
	n := 0
	for _, s := range specs {
		n += len(s.Name) + len(s.Description) + len(s.InputSchema)
	}
	return n
}

// ===== the estimator on the context =====
//
// The overflow middleware builds Views of its own, for requests it shrinks,
// and has to estimate them the way the loop does.

type estimatorKey struct{}

func withEstimator(ctx context.Context, e *tokenEstimator) context.Context {
	return context.WithValue(ctx, estimatorKey{}, e)
}

func estimatorFrom(ctx context.Context) *tokenEstimator {
	e, _ := ctx.Value(estimatorKey{}).(*tokenEstimator)
	return e
}

type windowKey struct{}

func withWindow(ctx context.Context, n int) context.Context {
	return context.WithValue(ctx, windowKey{}, n)
}

func windowFrom(ctx context.Context) int {
	n, _ := ctx.Value(windowKey{}).(int)
	return n
}

// viewFrom builds the View an llm middleware hands a strategy, from what the
// run published on ctx.
func viewFrom(ctx context.Context, msgs []llm.Message) View {
	e := estimatorFrom(ctx)
	return View{Messages: msgs, Results: resultsFrom(ctx), Tokens: e.estimate, Window: windowFrom(ctx)}
}

// sentRecorder is where the overflow middleware writes the messages it
// actually sent, when it shrank them; the loop calibrates on those.
type sentRecorder struct{ msgs []llm.Message }

func (s *sentRecorder) or(msgs []llm.Message) []llm.Message {
	if s != nil && s.msgs != nil {
		return s.msgs
	}
	return msgs
}

type sentKey struct{}

func withSent(ctx context.Context, s *sentRecorder) context.Context {
	return context.WithValue(ctx, sentKey{}, s)
}

func recordSent(ctx context.Context, msgs []llm.Message) {
	if s, ok := ctx.Value(sentKey{}).(*sentRecorder); ok && s != nil {
		s.msgs = msgs
	}
}
