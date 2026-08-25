package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/hoobean1996/wombat-go/tool"
)

// ===== calculator =====

type calculatorIn struct {
	Expression string `json:"expression"`
}

// Calculator evaluates an arithmetic expression.
//
// It takes no dependencies, and that is the point: the OCaml shelled out to
// `bc` while declaring itself read-only with no host requirements, which was
// a lie that only held because bc is usually installed. A parser in process
// makes the declaration true — no subprocess, no NeedExec, nothing to hide in
// a sandbox — and it removes the character allow-list that had to guard the
// shell string.
func Calculator() tool.Def {
	return tool.Typed(tool.Def{
		Name: "calculator",
		Description: "Evaluate a simple arithmetic expression. Supports +, -, *, /, parens, " +
			"powers (^). Example: '(15 * 7 + 3) / 2'",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "expression": {
      "type": "string",
      "description": "Arithmetic expression to evaluate"
    }
  },
  "required": ["expression"]
}`),
		Caps:       tool.CapReadOnly,
		Needs:      0,
		Idempotent: true,
		Timeout:    5 * time.Second,
		Category:   "compute",
		// Retryable stays nil on purpose: a pure function that failed once
		// fails identically, so retrying only burns wall clock.
	}, func(_ context.Context, in calculatorIn) (string, error) {
		return evalExpr(in.Expression)
	})
}

// ===== current_time =====

// CurrentTime reports the wall clock.
//
// The clock is a parameter because a replayed or recorded run has to be able
// to pin it: a tool that calls time.Now directly is a tool whose output can
// never be reproduced.
func CurrentTime(now func() time.Time) tool.Def {
	mustNotBeNil(now != nil, "CurrentTime requires a non-nil clock")

	return tool.Typed(tool.Def{
		Name:        "current_time",
		Description: "Get the current date and time in ISO 8601 format (local time).",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {}
}`),
		Caps:       tool.CapReadOnly,
		Needs:      0,
		Idempotent: true,
		Timeout:    time.Second,
		Category:   "compute",
		// Retryable stays nil on purpose: reading the injected clock cannot
		// fail transiently, so retrying only burns wall clock.
	}, func(_ context.Context, _ struct{}) (string, error) {
		return now().Format("2006-01-02T15:04:05"), nil
	})
}

// ===== ask_user =====

// AskUserName is the conventional name of the pause tool. Prefer detecting a
// pause with tool.FindPause, which keys on [tool.CapPause] rather than on a
// name; this exists for the cases that genuinely need the string.
const AskUserName = "ask_user"

// ErrAskUserNotIntercepted reports that the pause tool was actually dispatched.
//
// Reaching it means the agent loop failed to intercept a CapPause tool, which
// is a harness bug: returning nil here would strand the run waiting for an
// answer that is never coming, so it is loud instead.
var ErrAskUserNotIntercepted = errors.New("builtin: ask_user is a pause tool and its handler must never be invoked — the agent loop is expected to intercept it before dispatch")

// AskUser pauses the run and asks the user a structured question.
//
// It carries [tool.CapPause] and no Fn worth running: the loop finds it with
// tool.FindPause, suspends, and feeds the user's reply back as the
// tool_result. The description below is the tool — it is what teaches the
// model to emit a real JSON Schema object instead of a JSON-encoded string,
// which is the single most common way this tool is misused.
func AskUser() tool.Def {
	return tool.Def{
		Name: AskUserName,
		Description: `Pause the agent and ask the user via a structured form. The front-end renders the form from a JSON Schema you supply; the user's answer arrives back as a JSON object matching that schema. Use this when you genuinely need user input on a decision that materially changes downstream work — don't guess, don't ask trivial questions.

PREFERRED INPUT — pass a JSON Schema in [schema] AS A REAL JSON OBJECT, NOT A STRING:

CORRECT:
{
  "schema": {
    "type": "object",
    "title": "Project setup",
    "properties": {
      "db": {
        "type": "string",
        "enum": ["Postgres", "SQLite", "MySQL"],
        "description": "Backend database"
      },
      "single_user": {
        "type": "boolean",
        "default": true,
        "description": "Only one user at a time?"
      }
    },
    "required": ["db"]
  }
}

WRONG (the schema must NOT be a JSON-encoded string):
{"schema": "{\"type\":\"object\", ...}"}
— that gets you an empty form. Pass the object inline.

Supported field types: string (with optional enum / pattern / format / minLength / maxLength), integer, number (with min / max), boolean, array of enum-string (renders as multi-select). Use [title] for the form heading, per-property [description] for help text, [required] to gate submit. Keep schemas small — one form per pause, 1–5 fields max. The user's answer comes back as a JSON-stringified object matching the schema, so parse it accordingly.

BACK-COMPAT — passing just [question: "..."] still works (auto-wrapped into a single-string-answer schema), but prefer the structured form whenever the answer space is constrained.

From the planner: prefer asking BEFORE decomposing the goal when scope is ambiguous (DB choice, framework choice, single- vs multi-user). Don't ask if you could reasonably decide yourself.`,

		// Neither field is required: the runtime accepts a schema or falls
		// back to question, and listing both as optional lets the model pick
		// without fighting a validator.
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "schema": {
      "type": "object",
      "description": "A JSON Schema (subset) describing the form to render. Top-level type must be object; properties may be string (enum / pattern / format), integer, number, boolean, or array-of-enum-string."
    },
    "question": {
      "type": "string",
      "description": "Back-compat: a free-text question. Auto-wrapped into a single-string-answer schema. Prefer [schema] for structured input."
    }
  }
}`),
		Caps:  tool.CapPause,
		Needs: 0,
		// No timeout: the pause is bounded by the user, not by a clock, and
		// the handler never runs anyway.
		Idempotent: false,
		Category:   "meta",
		Fn: func(context.Context, json.RawMessage) (string, error) {
			return "", ErrAskUserNotIntercepted
		},
	}
}
