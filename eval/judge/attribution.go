package judge

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

// Attribution is one claim about what went wrong, recorded so it can be
// audited later.
//
// A score is forgotten by the next run; an attribution is acted on. "Repair the
// tool policy" sends somebody to change code, and if the claim behind it was a
// false positive that work is wasted and the real defect stays. Measured over
// one session, three of the attributions this pipeline produced were false
// positives from a single check with three distinct failure modes, and each was
// caught only because a human happened to look.
//
// So attributions are written out rather than rendered and discarded. The
// record is what makes "how often is this judge right" a question with an
// answer instead of an impression.
type Attribution struct {
	Task   string `json:"task"`
	Sample int    `json:"sample"`

	Criterion Criterion `json:"criterion"`
	Judge     string    `json:"judge"`

	Score      int     `json:"score"`
	Confidence float64 `json:"confidence"`

	// Authoritative records whether this attribution was allowed to fail the
	// episode, or was demoted to a concern by [Config.Authority].
	Authoritative bool `json:"authoritative"`

	Failure Failure      `json:"failure"`
	Repair  RepairTarget `json:"repair"`

	Evidence []Evidence `json:"evidence,omitempty"`

	// OracleReward is what the verifiers scored the episode, and OracleAgrees
	// is whether they and this attribution point the same way.
	//
	// The disagreements are the whole point of recording this. An attribution
	// that fails an episode the oracle passed is either a real defect the
	// verifiers cannot see or a defect in the rubric, and there is no way to
	// tell which without reading it — but there is also no way to know it
	// happened unless it was written down.
	OracleReward float64 `json:"oracle_reward"`
	OracleAgrees bool    `json:"oracle_agrees"`

	// Status is filled in by a person: "unreviewed", "confirmed",
	// "false_positive", or "unclear".
	//
	// It starts unreviewed and stays that way until somebody checks. A ledger
	// where everything reads "confirmed" because the field defaulted to it
	// would be worse than no ledger.
	Status string `json:"status"`
}

// StatusUnreviewed is what every attribution starts as.
const StatusUnreviewed = "unreviewed"

// Attributions extracts every non-passing assessment as an auditable claim.
//
// Non-passing rather than failing: an assessment demoted below [Config.Authority]
// did not gate its episode, but it still made a claim, and a claim nobody
// records is a claim nobody can score the judge on.
func (c Corpus) Attributions(authority, oracleBar float64) []Attribution {
	var out []Attribution
	for _, r := range c {
		oracleOK := r.Reward >= oracleBar
		for _, a := range r.Assessments {
			if a.Pass {
				continue
			}
			auth := a.Confidence >= authority
			out = append(out, Attribution{
				Task: r.Task, Sample: r.Sample,
				Criterion: a.Criterion, Judge: a.Judge,
				Score: a.Score, Confidence: a.Confidence, Authoritative: auth,
				Failure: a.Failure, Repair: a.Repair, Evidence: a.Evidence,
				OracleReward: r.Reward,
				// They agree when both say something is wrong. An attribution
				// against an episode the oracle passed is the interesting case
				// and is marked as disagreement whether or not it gated.
				OracleAgrees: !oracleOK,
				Status:       StatusUnreviewed,
			})
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Task != out[j].Task {
			return out[i].Task < out[j].Task
		}
		return out[i].Criterion < out[j].Criterion
	})
	return out
}

// WriteAttributions emits the ledger as JSONL.
//
// One object per line, appendable, so successive runs accumulate into a file a
// reviewer can grep and a script can score. Not JSON-array, which would have to
// be rewritten whole and cannot be appended to by two runs at once.
func WriteAttributions(w io.Writer, attrs []Attribution) error {
	enc := json.NewEncoder(w)
	for _, a := range attrs {
		if err := enc.Encode(a); err != nil {
			return fmt.Errorf("judge: writing attributions: %w", err)
		}
	}
	return nil
}

// AttributionStats summarises a reviewed ledger.
type AttributionStats struct {
	Total         int
	Authoritative int
	Disagreeing   int

	ByStatus    map[string]int
	ByCriterion map[Criterion]int
	ByRepair    map[RepairTarget]int
}

// Precision is confirmed over (confirmed + false positives), and -1 when
// nothing has been reviewed.
//
// This is the number the whole ledger exists to produce. Until it is known,
// every repair queue this pipeline emits is a list of guesses with a
// confidence nobody has measured.
func (s AttributionStats) Precision() float64 {
	good := s.ByStatus["confirmed"]
	bad := s.ByStatus["false_positive"]
	if good+bad == 0 {
		return -1
	}
	return float64(good) / float64(good+bad)
}

// Summarise counts a ledger.
func Summarise(attrs []Attribution) AttributionStats {
	s := AttributionStats{
		ByStatus:    map[string]int{},
		ByCriterion: map[Criterion]int{},
		ByRepair:    map[RepairTarget]int{},
	}
	for _, a := range attrs {
		s.Total++
		if a.Authoritative {
			s.Authoritative++
		}
		if !a.OracleAgrees {
			s.Disagreeing++
		}
		st := a.Status
		if st == "" {
			st = StatusUnreviewed
		}
		s.ByStatus[st]++
		s.ByCriterion[a.Criterion]++
		s.ByRepair[a.Repair]++
	}
	return s
}

// ReadAttributions loads a ledger, so a later run can report precision over
// everything reviewed so far rather than only over itself.
func ReadAttributions(r io.Reader) ([]Attribution, error) {
	dec := json.NewDecoder(r)
	var out []Attribution
	for {
		var a Attribution
		err := dec.Decode(&a)
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return out, fmt.Errorf("judge: reading attributions: %w", err)
		}
		out = append(out, a)
	}
}
