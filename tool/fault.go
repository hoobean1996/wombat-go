package tool

import "errors"

// Whose fault was it?
//
// Every tool failure has an author, and the middleware needs to know which one
// before it can react sensibly. Two failures that look identical at the type
// level — both are a non-nil error out of a Handler — mean opposite things:
//
//   - The TOOL failed. The search index refused the connection, the shell
//     could not be spawned, the HTTP call timed out. The dependency is sick,
//     the next call will probably fail too, and the useful response is to stop
//     calling it for a while.
//
//   - The CALL was bad. The model asked for a path outside the filesystem
//     root, passed a string where a number belongs, or ran a command that
//     exited non-zero. Here the tool worked perfectly: it took the request,
//     evaluated it, and returned the correct answer, which happens to be a
//     refusal. Nothing is sick. The next call, with better arguments, will
//     succeed.
//
// Collapsing the two is not a theoretical problem. [WithCircuitBreaker]
// counted every error alike, so a model that guessed five bad paths in a row
// silenced its own file reader for a minute — and a coding agent doing the
// only thing a coding agent does, edit-build-edit until the build is green,
// tripped the breaker on its own compiler errors and then could not run the
// build that would have proved it fixed them. The breaker exists to interrupt
// a doom loop; blaming the caller's mistakes on the tool is how it CAUSES one.
//
// The default is deliberately "the tool's fault". An unmarked error from an
// unknown tool could be anything, and a breaker that under-counts merely keeps
// calling a dependency that is down, while one that over-counts takes away a
// tool that works. Only the first is recoverable.
//
// Note what this does NOT change: a caller fault is still an error, still goes
// back to the model verbatim, still counts toward [WithDedupRepeats]. Repeating
// a bad path IS being stuck, and the model is the one who can unstick it. The
// only consumer of this distinction is health accounting.
type CallerFault interface {
	// CallerFault reports whether the failure blames the request rather than
	// the tool. Returning false is the same as not implementing the interface
	// at all, which lets a wrapper type answer either way at runtime.
	CallerFault() bool
}

// IsCallerFault reports whether err — or anything it wraps — blames the
// request rather than the tool.
//
// Nil is not a caller fault, because it is not a failure.
func IsCallerFault(err error) bool {
	var cf CallerFault
	return errors.As(err, &cf) && cf.CallerFault()
}

// CallerError marks err as the caller's fault, leaving its message and
// identity untouched.
//
// The wrapper is transparent in both directions that matter: Error() is the
// original text, so nothing the model reads changes, and Unwrap keeps
// errors.Is and errors.As working through it. That is what makes it safe to
// apply to a package-level sentinel —
//
//	var ErrOutsideRoot = tool.CallerError(errors.New("builtin: path escapes …"))
//
// — and still have errors.Is(err, ErrOutsideRoot) hold for a call that wrapped
// it with %w.
//
// Nil in, nil out, so it can wrap a call site that may not have failed.
func CallerError(err error) error {
	if err == nil {
		return nil
	}
	return callerError{err}
}

// callerError is a comparable struct rather than a pointer, so that a sentinel
// built by CallerError is matched by errors.Is through any number of %w
// wrappings — errors.Is compares with ==, and two copies of this value made
// from the same inner error are equal.
type callerError struct{ err error }

func (c callerError) Error() string     { return c.err.Error() }
func (c callerError) Unwrap() error     { return c.err }
func (c callerError) CallerFault() bool { return true }
