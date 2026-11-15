package permission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"

	"wombat/pkg/core"
	"wombat/pkg/core/llm"
	"wombat/pkg/core/tool"
)

// Sentinel errors, for callers that need to tell a refusal from a failure.
var (
	// ErrDenied wraps every refusal Gate produces, whether the policy refused
	// or a person did.
	ErrDenied = tool.CallerError(errors.New("permission: denied"))

	// ErrNoApprover reports that the policy wanted to ask and there was
	// nobody to ask. The resulting error wraps ErrDenied too — the call did
	// not run, which is all the model needs to know — so a caller can match
	// the general case and still distinguish this one.
	ErrNoApprover = errors.New("permission: approval required but no approver is installed")
)

// Approver answers an Ask, in one of two ways. It BLOCKS until it has an
// answer or ctx is done; or, when nobody is there to answer, it DEFERS — it
// returns an error made with [tool.Defer], the run ends at the call, and the
// call runs or is refused when the run is resumed with core.Decide. [Defer] is
// that approver. See [Gate] for why both keep the call pending rather than
// answering it with the approval.
type Approver interface {
	Approve(ctx context.Context, r Request) (bool, error)
}

// ApproverFunc adapts a function to [Approver].
type ApproverFunc func(context.Context, Request) (bool, error)

// Approve implements Approver.
func (f ApproverFunc) Approve(ctx context.Context, r Request) (bool, error) { return f(ctx, r) }

// Gate enforces a policy as tool middleware. A nil approver makes [Ask] mean
// [Deny].
//
// # The call stays pending while a person decides
//
// The obvious design is to answer the pending tool_use with the question,
// hand it to the caller and carry on with the answer — and it does not work.
// The approval itself becomes the tool's result: the model reads "yes" where
// it expected the contents of the file, and the tool never runs at all.
//
// So the call is never answered with the approval. A blocking approver keeps
// it PENDING inside this middleware while the human thinks; when the answer
// arrives the tool executes and its real output becomes the tool_result. A
// deferring one keeps it pending across a process boundary instead: the run
// ends in core.Deferred with the call unanswered, and core.Decide runs it —
// through this gate again, which finds the decision on the context — before
// the model is called. Either way the transcript contains the file, and no
// mention of the approval anywhere, which is correct: the approval is the
// operator's business, not part of the model's reasoning.
//
// What unblocks a wait that is never answered is cancellation, which the
// harness already has everywhere: a governor abort, a wall-clock cap, a
// disconnected client. Such a wait returns context.Cause(ctx), not a denial,
// because an abandoned question was never refused.
//
// # Where to install it
//
//	tool.Chain(inner,
//	    tool.WithRetry(...),           // inner
//	    tool.WithCircuitBreaker(...),
//	    permission.Gate(p, approver),  // <- here
//	    tool.WithLogging(l),
//	    tool.WithObserver(obs),        // outer
//	)
//
// OUTSIDE retry and the circuit breaker: a denial is a verdict, not a
// transient failure, and retrying it would ask a person the same question
// three times with exponential backoff between. It would also let a refused
// call count toward tripping the breaker, taking out a tool that never
// misbehaved. INSIDE observation, so a refused call still produces exactly one
// Start/Done pair and shows up in the trace as the thing that happened — a
// call that vanishes from the UI because it was refused is how an operator
// concludes the harness is broken.
//
// Where it sits relative to tool.WithTimeout is a real choice. Inside means
// the tool's own timeout also bounds the human's thinking time, which is
// usually wrong: a 30-second cap on view_file is about the read, not about
// lunch. Prefer to install Gate outside the timeout and bound the wait with
// the run's own budget.
//
// # A sub-agent is gated too, whether or not it was built with the gate
//
// A delegate call is one call to the parent and a whole agent's worth of calls
// to the child, and [Workspace] allows it without asking on the understanding
// that those calls are gated as well. That used to depend on whoever built the
// child remembering to install the same gate on it; a child built without one
// ran everything its parent would have had to ask about, silently.
//
// So the gate travels with the work it allowed. Every call it lets through
// carries it on the context ([tool.Inherit]); a sub-agent started by that call
// applies it to each of its own calls ([tool.WithInherited], which the agent's
// default dispatch chain includes), and so does every sub-agent below that. A
// child with a gate of its own is governed by both, its own ruling first. The
// same gate met twice in one chain — installed on the child and inherited from
// the parent, the usual configuration — decides once ([tool.Claim]).
//
// Grants travel the same way, on the context, so an approval given to the
// parent's call is not asked for again on the child's identical call.
func Gate(p Policy, a Approver) tool.Middleware {
	id := new(gateID)
	var gate tool.Middleware
	gate = func(next tool.Handler) tool.Handler {
		return func(ctx context.Context, d tool.Def, use llm.ToolUse) (string, error) {
			// One verdict per call, however many copies of this gate the call
			// passes through...
			if !tool.Claim(ctx, id) {
				return next(ctx, d, use)
			}
			// ...and the verdict follows the call into whatever it starts.
			ctx = tool.Inherit(ctx, id, gate)

			req := Request{Tool: d, Use: use}
			decision, reason := p.Decide(ctx, req)
			// The rule that decided speaks to the human through the request.
			req.Reason = reason

			switch decision {
			case Allow:
				settle(ctx, d, use, true, reason, sourcePolicy)
				return next(ctx, d, use)

			case Deny:
				settle(ctx, d, use, false, reason, sourcePolicy)
				return "", fmt.Errorf("%w: the %s tool was refused: %s. This is a standing "+
					"policy rather than a transient failure, so retrying the same call will "+
					"get the same answer. Use a tool you are permitted to use, or stop and "+
					"tell the user exactly what you need and why",
					ErrDenied, d.Name, reason)

			case Ask:
				return ask(ctx, next, a, req, d, use, reason)

			default:
				// Decide never returns Undecided; it resolves the fallthrough
				// itself. Refuse anyway rather than fall through to the tool:
				// the one thing this package must never do is run a call it
				// has no verdict for.
				settle(ctx, d, use, false, "the policy produced no verdict", sourcePolicy)
				return "", fmt.Errorf("%w: the %s tool was refused because the policy produced "+
					"no verdict for it. Do not retry; tell the user the permission policy is "+
					"misconfigured", ErrDenied, d.Name)
			}
		}
	}
	return gate
}

// gateID is one Gate's identity, the key it claims calls and inherits itself
// under. Not zero-sized: distinct zero-size allocations may share an address,
// and two gates must never be mistaken for one.
type gateID struct{ _ byte }

// ask handles the Ask branch: grant memory first, then a human.
func ask(
	ctx context.Context,
	next tool.Handler,
	a Approver,
	req Request,
	d tool.Def,
	use llm.ToolUse,
	reason string,
) (string, error) {
	// The memory is consulted BEFORE the event is emitted, so a call already
	// approved this run does not flash a prompt at a UI that would then have
	// to withdraw it.
	grants := GrantsFrom(ctx)
	key := grantKey(d, use)
	if grants.Has(key) {
		settle(ctx, d, use, true, reason, sourceGrant)
		return next(ctx, d, use)
	}

	// A person already decided this very call — the run was deferred on it
	// and resumed with core.Decide — so the decision is the answer, and
	// nobody is asked twice.
	if allow, decided := tool.Decision(ctx, use.ID); decided {
		if !allow {
			settle(ctx, d, use, false, reason, sourceApprover)
			return "", refused(d, reason)
		}
		grants.Remember(key)
		settle(ctx, d, use, true, reason, sourceApprover)
		return next(ctx, d, use)
	}

	if a == nil {
		settle(ctx, d, use, false, reason, sourceNoApprover)
		// Both sentinels: the call did not run (ErrDenied) and the cause is
		// configuration rather than judgement (ErrNoApprover).
		return "", fmt.Errorf("%w: %w: the %s tool needs a person to approve it (%s) and this "+
			"run has nobody to ask, so retrying will fail the same way. Use a tool that does "+
			"not need approval, or stop and tell the user what you need",
			ErrDenied, ErrNoApprover, d.Name, reason)
	}

	core.Emit(ctx, Requested{
		UseID:  use.ID,
		Tool:   d.Name,
		Reason: reason,
		Input:  jsonOrNil(use.Input),
	})

	ok, err := a.Approve(ctx, req)
	if _, deferred := tool.DeferralReason(err); deferred {
		// Not a verdict, so nothing is settled: the call is pending, and the
		// run ends here to wait for one.
		return "", err
	}
	if err != nil {
		settle(ctx, d, use, false, err.Error(), sourceApprover)

		// Two very different failures arrive here and they must not read alike.
		//
		// An ABANDONED wait — a governor abort, a wall-clock cap, a browser
		// that went away — is not a refusal, so it does not wrap ErrDenied.
		// The operator has to learn the run stopped, not that someone said no.
		if cause := context.Cause(ctx); cause != nil {
			return "", fmt.Errorf("permission: the approval of the %s tool was abandoned: %w", d.Name, cause)
		}

		// An approver that could not ask at all — a headless front end with no
		// console — never waited, so saying it did would be a lie to the one
		// reader who cannot check: the model. It DOES wrap ErrDenied, because
		// the fact every consumer needs is that the call did not run for
		// permission reasons; the approver's own error carries the detail.
		return "", fmt.Errorf("%w: the %s tool needs approval and none could be collected: %w",
			ErrDenied, d.Name, err)
	}
	if !ok {
		settle(ctx, d, use, false, reason, sourceApprover)
		return "", refused(d, reason)
	}

	// Remembered only on a yes. A no is not remembered on purpose: the second
	// occurrence of a refused call is usually the model trying again with a
	// human in the loop, and that person may well answer differently once they
	// see it twice.
	grants.Remember(key)
	settle(ctx, d, use, true, reason, sourceApprover)
	return next(ctx, d, use)
}

// refused is what the model reads when a person said no.
func refused(d tool.Def, reason string) error {
	return fmt.Errorf("%w: the person supervising this run refused the %s tool: %s. "+
		"Do not retry it and do not look for another way to achieve the same effect; "+
		"tell the user what you were trying to do and why, and let them decide",
		ErrDenied, d.Name, reason)
}

// Defer is the Approver of a run nobody is watching: it decides nothing, and
// defers the call instead. The run ends at the call in core.Deferred — the
// host shows the question to a person whenever they are next around — and
// core.Decide resumes it with their answer.
func Defer() Approver {
	return ApproverFunc(func(_ context.Context, r Request) (bool, error) {
		return false, tool.Defer(r.Reason)
	})
}

// settle records one verdict in both places it has to appear: the event stream
// a UI renders, and the log an operator reads afterwards.
//
// Logging goes through slog.Default rather than an injected logger because
// [Gate]'s signature is fixed by the API and a permission decision must be
// auditable whether or not anybody remembered to pass a logger. Install the
// deployment's handler with slog.SetDefault; the calls are *Context variants,
// so a handler that reads trace ids off the context gets them.
func settle(ctx context.Context, d tool.Def, use llm.ToolUse, allowed bool, reason, source string) {
	core.Emit(ctx, Decided{
		UseID:   use.ID,
		Tool:    d.Name,
		Allowed: allowed,
		Reason:  reason,
		Source:  source,
	})

	attrs := []any{
		slog.String("tool", d.Name),
		slog.String("use_id", string(use.ID)),
		slog.String("reason", reason),
		slog.String("rule", source),
	}
	if allowed {
		slog.Default().InfoContext(ctx, "permission allowed", attrs...)
		return
	}
	slog.Default().WarnContext(ctx, "permission denied", attrs...)
}

// jsonOrNil keeps an empty argument list out of the event. An empty non-nil
// json.RawMessage is not valid JSON and would fail the encode for the whole
// event, taking the prompt with it.
func jsonOrNil(in json.RawMessage) json.RawMessage {
	if len(in) == 0 {
		return nil
	}
	return in
}
