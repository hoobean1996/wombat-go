package main

import (
	"fmt"
	"io"
	"log/slog"
	"os"

	"wombat/pkg/core/llm"
	"wombat/pkg/core/tape"
	"wombat/pkg/core/trace"
)

// recorders are what is written down across every session: the tape that
// records every model and tool call, or replays them, and the trace.
//
// Both outlive every session and report real failures only on close, so run
// opens them first and closes them last.
type recorders struct {
	cfg    recordConfig
	logger *slog.Logger
	tape   *tape.Tape    // nil without record.tape
	tracer *trace.Tracer // nil without record.trace
	trace  io.Closer
}

func openRecorders(cfg recordConfig, logger *slog.Logger) (*recorders, error) {
	r := &recorders{cfg: cfg, logger: logger}
	if cfg.Tape != "" {
		tp, err := tape.Open(cfg.Tape, tapeModes[cfg.TapeMode])
		if err != nil {
			return nil, err
		}
		r.tape = tp
	}
	if cfg.Trace != "" {
		sink, closer, err := trace.FileSink(cfg.Trace)
		if err != nil {
			r.close()
			return nil, fmt.Errorf("opening the trace: %w", err)
		}
		r.tracer, r.trace = trace.New(sink), closer
	}
	return r, nil
}

// wrap puts the tape around a client: outermost of the transport concerns,
// so a replayed call skips the network, the retries and the cost tracking
// alike.
func (r *recorders) wrap(c llm.Client) llm.Client {
	if r.tape == nil {
		return c
	}
	return llm.Chain(c, r.tape.LLM())
}

func (r *recorders) close() {
	if r.tape != nil {
		if err := r.tape.Close(); err != nil {
			r.logger.Error("closing the tape", "err", err)
		}
	}
	if r.trace == nil {
		return
	}
	if err := r.trace.Close(); err != nil {
		r.logger.Error("closing the trace", "err", err)
	}
	if r.cfg.TraceHTML != "" {
		if err := renderTrace(r.cfg.Trace, r.cfg.TraceHTML); err != nil {
			r.logger.Error("rendering the trace report", "err", err)
		}
	}
}

func renderTrace(in, out string) error {
	spans, err := trace.ReadFile(in)
	if err != nil {
		return err
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	if err := trace.WriteHTML(f, spans); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
