package tool

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"wombat/pkg/core/llm"
)

// tagger is middleware that records its name on the way in, so a test can read
// the order a chain ran in.
func tagger(name string, log *[]string) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, d Def, use llm.ToolUse) (string, error) {
			*log = append(*log, name)
			return next(ctx, d, use)
		}
	}
}

func TestWithInheritedAppliesAttachedMiddlewareInOrder(t *testing.T) {
	var log []string
	ctx := Inherit(context.Background(), "a", tagger("a", &log))
	ctx = Inherit(ctx, "b", tagger("b", &log))
	ctx = Inherit(ctx, "a", tagger("a-again", &log)) // same key: a no-op

	h := Chain(func(context.Context, Def, llm.ToolUse) (string, error) {
		log = append(log, "tool")
		return "ok", nil
	}, WithInherited)
	if _, err := h(ctx, Def{Name: "x"}, llm.ToolUse{}); err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(log, ","), "a,b,tool"; got != want {
		t.Errorf("ran %s, want %s — earliest attached outermost, and a key attached twice applied once", got, want)
	}
}

func TestWithInheritedIsInertWithNothingAttached(t *testing.T) {
	h := Chain(func(context.Context, Def, llm.ToolUse) (string, error) { return "ok", nil }, WithInherited)
	if out, err := h(context.Background(), Def{}, llm.ToolUse{}); out != "ok" || err != nil {
		t.Errorf("got (%q, %v), want (ok, nil)", out, err)
	}
}

func TestInheritDoesNotMutateTheParentContext(t *testing.T) {
	var log []string
	parent := Inherit(context.Background(), "a", tagger("a", &log))
	_ = Inherit(parent, "b", tagger("b", &log))

	h := Chain(func(context.Context, Def, llm.ToolUse) (string, error) { return "", nil }, WithInherited)
	_, _ = h(parent, Def{}, llm.ToolUse{})
	if got := strings.Join(log, ","); got != "a" {
		t.Errorf("the parent context ran %s, want only a — a child's attachment leaked upward", got)
	}
}

func TestClaimIsOncePerDispatchedCall(t *testing.T) {
	t.Run("outside a dispatcher every claim succeeds", func(t *testing.T) {
		if !Claim(context.Background(), "k") || !Claim(context.Background(), "k") {
			t.Error("Claim outside a dispatch reported false; there is no call to scope it to")
		}
	})

	t.Run("inside a call the first claim wins and other keys are independent", func(t *testing.T) {
		var got []bool
		def := Def{Name: "t", InputSchema: json.RawMessage(`{}`),
			Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
				got = append(got, Claim(ctx, "k"), Claim(ctx, "k"), Claim(ctx, "other"))
				return "", nil
			}}
		NewDispatcher(Direct).Dispatch(context.Background(), NewSet(def), []llm.ToolUse{{ID: "u1", Name: "t"}})
		if len(got) != 3 || !got[0] || got[1] || !got[2] {
			t.Errorf("claims = %v, want [true false true]", got)
		}
	})

	t.Run("each call has claims of its own", func(t *testing.T) {
		var got []bool
		def := Def{Name: "t", InputSchema: json.RawMessage(`{}`),
			Fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
				got = append(got, Claim(ctx, "k"))
				return "", nil
			}}
		NewDispatcher(Direct).Dispatch(context.Background(), NewSet(def),
			[]llm.ToolUse{{ID: "u1", Name: "t"}, {ID: "u2", Name: "t"}})
		if len(got) != 2 || !got[0] || !got[1] {
			t.Errorf("claims = %v, want [true true] — one call's claim leaked into the next", got)
		}
	})
}
