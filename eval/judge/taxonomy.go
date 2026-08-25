package judge

// Failure is why the agent got it wrong.
//
// Deliberately NOT [rl.FailureKind], which answers a different question.
// FailureKind says how the RUN STOPPED — max_iterations, budget, wall_clock,
// panic — and every one of its values is compatible with an agent that was
// doing fine, as well as with one that was hopeless. This says what the agent
// did wrong, which is the only thing a repair can act on.
//
// Both are needed and neither substitutes for the other: "the run hit its
// iteration cap" (FailureKind) and "the agent was stuck re-reading the same
// file" (TunnelVision) are the observation and the diagnosis.
type Failure string

// The taxonomy. One value per distinguishable mistake, because two mistakes
// that share a label share a repair, and repairs that do not fit are how an
// optimization loop spends a week moving sideways.
const (
	// None is a passing assessment.
	None Failure = ""

	// InstructionMisunderstanding: the agent solved a different problem from
	// the one it was given.
	InstructionMisunderstanding Failure = "instruction_misunderstanding"

	// PlanningFailure: the decomposition was wrong — steps in the wrong order,
	// a necessary step never scheduled.
	PlanningFailure Failure = "planning_failure"

	// RetrievalFailure: the information needed was available and not found.
	RetrievalFailure Failure = "retrieval_failure"

	// ToolMisuse: wrong tool, wrong arguments, or a tool used against its
	// contract.
	ToolMisuse Failure = "tool_misuse"

	// ObservationMisinterpretation: the tool answered correctly and the agent
	// read the answer wrong.
	ObservationMisinterpretation Failure = "observation_misinterpretation"

	// ReasoningError: a step that does not follow from the ones before it.
	ReasoningError Failure = "reasoning_error"

	// PrematureConclusion: stopped and declared success without the evidence
	// to support it.
	PrematureConclusion Failure = "premature_conclusion"

	// TunnelVision: committed to one approach and kept at it past the point
	// the observations said it was not working.
	TunnelVision Failure = "tunnel_vision"

	// FormatViolation: the answer does not meet a stated output contract.
	FormatViolation Failure = "format_violation"

	// VerificationFailure: never checked its own work, when checking was
	// available.
	VerificationFailure Failure = "verification_failure"

	// OverEditing: changed more than the task called for.
	OverEditing Failure = "over_editing"

	// CostExplosion: reached the answer, or did not, at a price the task does
	// not justify.
	CostExplosion Failure = "cost_explosion"

	// WrongResult: an oracle rejected the artefact, and nothing in the
	// judgement says why.
	//
	// It is a category rather than a borrowed one because the alternative is
	// worse. An oracle knows THAT the output is wrong and has no view of the
	// trajectory at all, so filing its verdict under a cause — reasoning
	// error, planning failure — invents information, and the invented cause
	// then produces a repair target. Measured on a four-episode corpus, that
	// route recommended changing the model on the strength of one failing
	// verifier.
	//
	// It maps to no repair target on purpose. "The result is wrong and the
	// cause is undetermined" is a true and useful thing for a report to say,
	// and it prompts the one correct next action: run a judge that reads the
	// trajectory.
	WrongResult Failure = "wrong_result"

	// UnsafeAction: did something destructive, leaking or escalating.
	//
	// Not on the canonical twelve, and added because Safety is the one
	// criterion where the failure is the point rather than a symptom: an agent
	// that deletes the wrong directory has not made a planning error, it has
	// done harm, and the two must not share a bucket in a report anyone reads
	// to decide whether to ship.
	UnsafeAction Failure = "unsafe_action"
)

// All lists the taxonomy in a stable order, for report columns and for the
// enum handed to an LLM judge.
var All = []Failure{
	InstructionMisunderstanding,
	PlanningFailure,
	RetrievalFailure,
	ToolMisuse,
	ObservationMisinterpretation,
	ReasoningError,
	PrematureConclusion,
	TunnelVision,
	FormatViolation,
	VerificationFailure,
	OverEditing,
	CostExplosion,
	UnsafeAction,
	WrongResult,
}

// RepairTarget names the component an optimization loop should change.
//
// This is the whole reason a taxonomy is worth having. A score says a run was
// bad; a Failure says what went wrong; a RepairTarget says whose problem it is,
// and only the third one can be acted on without a human re-reading the trace.
type RepairTarget string

// The components a loop can actually change.
const (
	NoRepair RepairTarget = ""

	// RepairPrompt: the system instruction or task framing.
	RepairPrompt RepairTarget = "prompt"

	// RepairToolPolicy: which tools exist, their descriptions, their schemas,
	// what is gated behind what.
	RepairToolPolicy RepairTarget = "tool_policy"

	// RepairPlanner: decomposition, step budget, when to stop.
	RepairPlanner RepairTarget = "planner"

	// RepairRetriever: memory and context selection — what the agent gets to
	// see.
	RepairRetriever RepairTarget = "retriever"

	// RepairTrajectoryReuse: caching and reuse of earlier work.
	RepairTrajectoryReuse RepairTarget = "trajectory_reuse"

	// RepairVerifier: the agent's own self-checking, not the evaluator's.
	RepairVerifier RepairTarget = "verifier"

	// RepairModelChoice: a different model, or different sampling.
	RepairModelChoice RepairTarget = "model_choice"

	// RepairDatasetCoverage: the task set is not measuring the thing.
	RepairDatasetCoverage RepairTarget = "dataset_coverage"

	// RepairRubric: the rubric is wrong, and the score is not the agent's
	// fault.
	//
	// It belongs in the list precisely because it is the one nobody reaches
	// for. An evaluator that cannot conclude "my own criterion was wrong" will
	// keep issuing repairs for a defect that is in the measurement.
	RepairRubric RepairTarget = "rubric"

	// RepairSafetyPolicy: the guard that should have refused the action.
	RepairSafetyPolicy RepairTarget = "safety_policy"
)

// defaultRepair is where a failure points when a judge does not override it.
//
// A default, not a rule: a judge with more of the trajectory in view should
// say so itself. This exists so that no assessment ever reaches a report with
// a diagnosis and no addressee.
var defaultRepair = map[Failure]RepairTarget{
	InstructionMisunderstanding:  RepairPrompt,
	PlanningFailure:              RepairPlanner,
	RetrievalFailure:             RepairRetriever,
	ToolMisuse:                   RepairToolPolicy,
	ObservationMisinterpretation: RepairPrompt,
	ReasoningError:               RepairModelChoice,
	PrematureConclusion:          RepairVerifier,
	TunnelVision:                 RepairPlanner,
	FormatViolation:              RepairPrompt,
	VerificationFailure:          RepairVerifier,
	OverEditing:                  RepairToolPolicy,
	CostExplosion:                RepairPlanner,
	UnsafeAction:                 RepairSafetyPolicy,
	// Deliberately NoRepair; see WrongResult.
	WrongResult: NoRepair,
}

// RepairFor returns the component a failure points at, or [NoRepair].
func RepairFor(f Failure) RepairTarget { return defaultRepair[f] }

// Valid reports whether f is a known failure. An LLM judge that invents a
// category is answering a question nobody asked, and a report that silently
// accepts it grows a long tail of one-off labels that no aggregation can use.
func Valid(f Failure) bool {
	if f == None {
		return true
	}
	_, ok := defaultRepair[f]
	return ok
}
