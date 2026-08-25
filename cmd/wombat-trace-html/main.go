// Command wombat-trace-html renders a trace NDJSON file as a self-contained
// HTML report.
//
//	wombat-trace-html run.ndjson            # writes run.html
//	wombat-trace-html -o report.html run.ndjson
//
// The output has no external assets, so it survives being emailed, attached to
// an issue, or opened from a directory with no network — which is where a
// trace usually ends up.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/hoobean1996/wombat-go/trace"
)

func main() {
	out := flag.String("o", "", "output path (default: input with .html)")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: wombat-trace-html [-o out.html] <trace.ndjson>")
		os.Exit(2)
	}
	in := flag.Arg(0)

	dst := *out
	if dst == "" {
		dst = strings.TrimSuffix(in, ".ndjson") + ".html"
	}

	spans, err := trace.ReadFile(in)
	if err != nil {
		fmt.Fprintf(os.Stderr, "wombat-trace-html: %v\n", err)
		os.Exit(1)
	}
	if len(spans) == 0 {
		fmt.Fprintf(os.Stderr, "wombat-trace-html: %s has no spans\n", in)
		os.Exit(1)
	}

	f, err := os.Create(dst)
	if err != nil {
		fmt.Fprintf(os.Stderr, "wombat-trace-html: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	if err := trace.WriteHTML(f, spans); err != nil {
		fmt.Fprintf(os.Stderr, "wombat-trace-html: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "%d spans → %s\n", len(spans), dst)
}
