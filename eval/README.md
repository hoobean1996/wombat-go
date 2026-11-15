# eval — does the *harness* work?

`benchmarks/` (here, in eval/) asks whether the **agent** can solve a task and answers with
pass@k. Every task there runs on a bare agent: the suite runner wires exactly one
option, `WithOverflowRecovery()`. So the mechanisms this project is actually
about — skill gating, delegation, the permission gate, transcript strategies,
the breaker/dedup split — are measured by **nothing**.

Their only coverage is unit tests against a scripted `llm.Client`. A scripted
client proves that *given a fake response the code takes a branch*. It cannot
prove that *the mechanism changed the outcome in the direction it claims*, and
it cannot see a mechanism whose only symptom is the bill.

This package is the missing half. It runs in three lanes, cheapest first.

---

## Lane 1 — request shape (`eval/shape`) · keyless · milliseconds · **built**

A transcript strategy, a gated tool set and the system-prompt builder all
finish their work *before* the request leaves the process. Their entire effect
is therefore visible in the `llm.Request` the harness assembles — no model
needed.

The load-bearing measurement is the **prefix hash**. Both major providers cache
on a prefix of the request: the leading bytes must be byte-identical to the
previous call or the whole prompt is re-read at full price. `Shot.Prefix` is the
running hash of the materialized messages, so the longest common prefix between
consecutive turns is exactly how far a cache could have hit — computed from the
request, with no provider, no key, and no reliance on a `cache_read` field the
server may not send.

```
go test ./eval/shape/
```

### What it already found

**1. Windowing does nothing inside a single run.**
`SafeCut` only cuts immediately before a user turn carrying no `tool_result`.
In a ReAct loop every user turn after the first is exactly that, and the
opening instruction at index 0 is excluded by the search itself — so a pure
tool loop has **no legal cut point** and every window degrades to `Flat`.

Not a bug (an over-long request is recoverable; an orphaned `tool_result` is a
hard provider error), but it decides where windowing can help at all: only in
multi-turn sessions, which is the daemon's case.

Pinned by `TestSingleRunToolLoopHasNoLegalCutPoint`.

**2. The quantized window's trade, measured.**
24-turn session, one tool call per turn, `keep=8`, `trigger=24`:

| strategy | pairs held | re-anchors | reusable | bytes sent |
|---|---|---|---|---|
| `Flat` | 47/47 | 0 | 1.00 | 338,762 |
| `SlidingWindow(8)` | 26/47 | 21 | 0.48 | 68,254 |
| `SlidingWindowAt(24,8)` | 42/47 | **5** | 0.83 | 104,582 |

Quantizing the cut drops re-anchors **21 → 5** for ~53% more bytes on the wire.
Whether that is a win depends on the cache discount: on a provider charging
~3.9× less for a cache read, it is.

Nothing in the repo measured this before, and no task score can: the answers are
identical either way.

Pinned by `TestSessionWindowCachePrefix`.

### Adding a lane-1 probe

```go
shots, err := shape.RunSession(ctx,
    shape.Config{Payload: 200, Options: []wombat.Option{
        wombat.WithStrategy(wombat.DropTagged(40, 12, skill.Tag)),
    }}, 24, 1)

st := shape.PrefixStability(shots)     // cache boundary
_  = shape.SystemDrift(shots)          // must always be empty
_  = shape.ToolDrift(shots)            // a gate SHOULD widen this; nothing else may
_  = shape.MaterializedBytes(shots)    // what eviction is trying to reduce
```

Use `shape.Run` for single-run mechanisms and `shape.RunSession` for anything
positional — see finding 1 for why the distinction is not cosmetic.

---

## Lane 2 — wire conformance (`eval/wire`) · keyless · milliseconds · **built**

Every fault test in this repo substitutes an `llm.Client`, a Go interface —
which bypasses the entire HTTP + SSE + JSON decode stack. The conditions the
harness's most interesting code claims to handle are *wire* conditions, and none
of them are reached that way.

A scriptable OpenAI-compatible fault server closes that gap. The behaviours
worth scripting map one-to-one onto code that currently has no wire-level test:

| behaviour | the code it would finally exercise |
|---|---|
| valid content-less stop, no usage chunk | the empty-turn retry in `run.go` |
| text deltas, then socket reset | keeping the half-answer on a stream failure |
| clean end with no `[DONE]` | the SSE decoder's boundary handling |
| invalid SSE JSON / bad chunk shape | decoder error paths |
| 429 / 500 / 503 | `llm.WithRetry` |
| a real context-overflow error body | the `WithOverflowRecovery` ladder |
| headers then idle | timeout and cancellation |

One design detail that turned out to matter: **an invalid request must not
consume a script entry**. Otherwise one adapter bug shifts the whole sequence
and produces a cascade whose real cause is several assertions back.
`TestRejectedRequestsDoNotConsumeTheScript` pins it.

```
go test ./eval/wire/
```

### What it establishes

Seven tests, all against the shipping `pkg/core/llm/openai` client — no substituted
`llm.Client` anywhere, so the HTTP, SSE and decode paths are the ones under
test:

| test | what it proves that a scripted client cannot |
|---|---|
| `TestEmptyTurnIsRetriedAndThenAnswers` | a content-less stop with **no usage chunk at all** survives the decoder and reaches the retry |
| `TestEmptyTurnGivesUpAfterThreeAttempts` | the retry is bounded |
| `TestReasoningBudgetExhaustedIsNotRetried` | the discrimination works on bytes that actually streamed — 4002 chars of `reasoning_content` against a 256-token cap fails on the FIRST call |
| `TestPartialAnswerSurvivesAMidStreamReset` | a real RST mid-stream still leaves the streamed prefix in the transcript |
| `TestContextOverflowEngagesTheLadder` | a 400 whose *prose* says overflow is classified as overflow, so the ladder engages: measured shrinking the request from 42 messages to 26 |
| `TestRateLimitIsRetried` | 429 classification survives the wire |
| `TestRejectedRequestsDoNotConsumeTheScript` | the script cannot silently shift |

### One finding, about the API rather than the wire

The first version of the reasoning test passed when it should have failed. The
cause was mine, not the harness's: `openai.Config.MaxTokens` and
`wombat.WithMaxTokens` are two different numbers, and the one the empty-turn
heuristic reasons about is the agent's — which is also the one that reaches the
request. A test that sets only the client's is asserting against a cap nobody
used. Worth knowing before writing the next one.

## Lane 3 — live contrast (`eval`) · needs a key · minutes · **core built, probes pending**

For claims that genuinely depend on a real model's behaviour: does it route to
the right skill, does a denial *teach*, does the parent recover when the depth
cap refuses.

The unit is a `Probe`: one task, two `Arm`s differing **only** in the mechanism,
n samples each, compared on a mechanism-specific observable — never pass@k. A
gate that halves the tool list does not move a task score.

Two kinds of expectation:

- **`Invariant`** must hold in *every* sample. "A gated tool never executes
  before its skill is loaded" is not a statistic; one violation is a bug.
- **`Contrast`** compares aggregates *across* arms, and only means something
  over enough samples.

A `Verdict` must always carry its numbers. A pass with no measurement in it is
not evidence — this suite exists because "the tests are green" was not evidence
either.

`Obs` is folded from the **event stream**, not the transcript, for the reason
`rl/rollout.go` (here, in eval/) gives about its own `Steps`: a `tool_result` does not say
whether the permission gate refused the call or the tool failed, and it does not
say how many tools were on offer when the model chose. `LLMStart.Tools` is the
direct observable for progressive disclosure — a gated run reads `12, 12, 14`
where an ungated one reads `14, 14, 14`.

### Probes to write

| probe | kind | observable |
|---|---|---|
| skill gating | invariant + contrast | no gated call succeeds before `load_skill`; offered-tool count rises after it; turn-1 count lower than ungated |
| skill eviction thrash | invariant | evict → gate refuses → reload → evict: a loop between the strategy and the gate, currently unobservable |
| permission allowlist | invariant | at least one `Decided{Allowed, Source:"policy"}` on an exec call — an allowlist that never fires is not a safety feature |
| denial teaches | invariant | no three consecutive identical denied calls |
| delegate on parent ctx | invariant | `SubagentStart/End` bracket the call; `SubagentEvent`s arrive; the run's spend includes the child's calls |
| depth cap refuses | invariant | after `ErrDelegateTooDeep` the parent still answers — the cap must refuse, not abort |

---

## Why the lanes are ordered this way

Lane 1 is the one that makes the suite CI-able: keyless, deterministic,
millisecond, and it covers the mechanisms whose regressions are otherwise
*completely* invisible. Lane 2 needs a server but still no key. Lane 3 costs
real money per run and should be reserved for the claims the other two cannot
reach.

The repo already ships `pkg/core/tape` — content-hash record and replay, measured at
0.107 s against a dead endpoint with byte-identical output — and nothing uses
it. Recording lane-3 probes once and replaying them is the cheapest way to give
the expensive lane a regression mode.
