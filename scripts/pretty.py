#!/usr/bin/env python3
"""Render the wombat JSONL event stream for a human.

    wombat-jsonl ... | scripts/pretty.py

Reads one JSON object per line from stdin and prints a readable transcript.
Unknown event types are shown rather than dropped, because an event nobody
renders is how a new feature stays invisible.
"""

import json
import sys

C = {
    "dim": "\033[2m", "off": "\033[0m", "b": "\033[1m",
    "red": "\033[31m", "grn": "\033[32m", "yel": "\033[33m",
    "blu": "\033[34m", "mag": "\033[35m", "cyn": "\033[36m",
}
if not sys.stdout.isatty():
    C = {k: "" for k in C}


def w(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def short(v, n=90):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    in_text = False          # are we mid-answer, so a newline is owed before the next line?
    reasoning_shown = 0      # how much scratchpad has been printed this turn
    args_open = {}           # streaming tool-call index -> accumulated argument text
    show_reasoning = "-r" in sys.argv or "--reasoning" in sys.argv

    def line(s=""):
        nonlocal in_text
        if in_text:
            w("\n")
            in_text = False
        w(s + "\n")

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            line(f"{C['dim']}· {short(raw, 120)}{C['off']}")
            continue

        t = d.get("type", "?")

        if t == "session_started":
            line(f"{C['b']}▸ {short(d.get('query',''), 100)}{C['off']}")
            bits = [f"{k}={v}" for k, v in d.items()
                    if k in ("provider", "model", "max_iters", "budget_usd") and v]
            line(f"{C['dim']}  {'  '.join(bits)}{C['off']}")

        elif t == "iter_start":
            line(f"{C['dim']}─── turn {d['n']}/{d['max']} {'─' * 40}{C['off']}")
            reasoning_shown = 0
            args_open.clear()

        elif t == "reasoning_delta":
            if show_reasoning:
                if reasoning_shown == 0:
                    w(f"{C['dim']}")
                w(d["text"])
                reasoning_shown += len(d["text"])
                in_text = True
            elif reasoning_shown == 0:
                reasoning_shown = 1
                line(f"{C['dim']}  (thinking… pass -r to see it){C['off']}")

        elif t == "text_delta":
            if reasoning_shown and show_reasoning:
                w(C["off"])
                reasoning_shown = 0
                line()
            w(d["text"])
            in_text = True

        elif t == "tool_args_delta":
            i = d.get("index", 0)
            if i not in args_open:
                args_open[i] = ""
                line(f"{C['cyn']}  ⟳ {d.get('name','?')}{C['off']}{C['dim']} writing arguments…{C['off']}")
            args_open[i] += d.get("text", "")

        elif t == "permission_requested":
            line(f"{C['yel']}  ⏸ approval needed: {d['tool']}{C['off']}")
            line(f"{C['dim']}    {short(d.get('input'), 100)}{C['off']}")
            line(f"{C['dim']}    why: {short(d.get('reason',''), 100)}{C['off']}")

        elif t == "permission_decided":
            mark = f"{C['grn']}✓ allowed" if d["allowed"] else f"{C['red']}✗ refused"
            line(f"{C['dim']}  {mark}{C['off']}{C['dim']} {d['tool']} ({d['source']}){C['off']}")

        elif t == "tool_start":
            line(f"{C['cyn']}  → {d['name']}{C['off']} {C['dim']}{short(d.get('input'), 100)}{C['off']}")

        elif t == "tool_done":
            body = d.get("output") if d["ok"] else d.get("error")
            mark = f"{C['grn']}←{C['off']}" if d["ok"] else f"{C['red']}✗{C['off']}"
            line(f"  {mark} {C['dim']}{d.get('ms',0)}ms  {short(body, 100)}{C['off']}")

        elif t == "subagent_start":
            line(f"{C['mag']}  ↳ {d['name']}{C['off']} {C['dim']}{short(d.get('task',''), 90)}{C['off']}")

        elif t == "subagent_event":
            inner = d.get("inner") or {}
            it = inner.get("type", "?")
            if it in ("tool_start", "tool_done", "permission_requested"):
                mark = short(inner.get("name") or inner.get("tool") or it, 40)
                line(f"{C['mag']}    │{C['off']} {C['dim']}{it}: {mark}{C['off']}")

        elif t == "subagent_end":
            mark = "ok" if d.get("ok") else "failed"
            line(f"{C['mag']}  ↳ {d['name']} {mark}{C['off']} {C['dim']}{d.get('ms',0)}ms{C['off']}")

        elif t == "spend":
            line(f"{C['dim']}  ${d['cost_usd']:.4f}  {d['input_tokens']}↓ {d['output_tokens']}↑"
                 f"  cache {d.get('cache_read_tokens',0)}  {d['elapsed_sec']:.0f}s{C['off']}")

        elif t == "done":
            line()
            line(f"{C['grn']}{C['b']}✔ {d['answer']}{C['off']}")

        elif t == "agent_waiting":
            line(f"{C['yel']}⏸ waiting on you: {d.get('question','')}{C['off']}")

        elif t == "submitted":
            line(f"{C['grn']}✔ {d['tool']} → {short(d.get('payload'), 120)}{C['off']}")

        elif t == "failed":
            line()
            line(f"{C['red']}{C['b']}✘ [{d.get('class','error')}] {d['reason']}{C['off']}")

        elif t in ("llm_start", "llm_done"):
            pass  # noise at this altitude; the trace has it

        else:
            line(f"{C['dim']}· {t} {short({k: v for k, v in d.items() if k != 'type'}, 90)}{C['off']}")

    if in_text:
        w("\n")


if __name__ == "__main__":
    main()
