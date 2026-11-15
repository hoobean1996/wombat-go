package benchmarks

import (
	"fmt"
	"strings"

	"wombat/eval/rl"
)

// NeedleFileCount is how many daily logs the haystack holds.
//
// Sized against the transcript and not against a round number: 44 logs at
// roughly 700 bytes each is ~30KB of text, which is comfortably readable file
// by file and comfortably NOT readable inside a 24-turn episode that also has
// to think. An agent that opens them one at a time runs out of turns; one that
// cats the directory runs out of context. Either failure is the measurement.
const NeedleFileCount = 44

// NeedleAnswer is the deployment id the question is looking for. It appears in
// exactly one of the logs, and [TestNeedleAppearsExactlyOnce] keeps it that
// way.
const NeedleAnswer = "deploy-7f3a91c"

// needleInHaystack is the retrieval task.
//
// The fact is in one file out of 44 and the question does not quote it. What
// makes it more than a grep is the distractor structure: five logs mention a
// rollback and six mention settlement, and only one mentions both. A single
// grep for either word returns a handful of files that all look plausible, so
// the agent has to narrow rather than guess, and an agent that decides to just
// read everything will not finish.
//
// Nothing here needs the Go toolchain, which — like read-only-qa — makes it
// the hard-tier task to run when you want to know whether the harness works
// rather than whether the agent does.
func needleInHaystack() Task {
	return Task{
		ID:      "needle-in-haystack",
		Summary: "find one fact in 44 files without reading all 44",
		Prompt: fmt.Sprintf(`This directory holds %d daily engineering logs under logs/, one file per
day, plus a README.

Question: one deployment was reverted after it corrupted the settlement
ledger. Which deployment was it?

Find its deployment id and write ONLY that id — no punctuation, no sentence,
no explanation — into a file called ANSWER.txt at the root of this directory.
Then stop and tell me the id.

There are more logs than you can usefully read. Search them.`, NeedleFileCount),
		Files: needleFiles(),
		Verifiers: []rl.Verifier{
			rl.FileExists("answer_file", "ANSWER.txt", 0.10),
			rl.FileContains("answer", "ANSWER.txt", NeedleAnswer, 0.60),

			// Separate from `answer` so the breakdown distinguishes "found it
			// and wrote an essay" from "found it": the prompt asked for the id
			// alone, and following an output format is a thing worth
			// measuring on its own.
			rl.Shell("answer_exact",
				`test "$(tr -d '[:space:]' < ANSWER.txt)" = "`+NeedleAnswer+`"`, 0.30),
		},
	}
}

// needleFiles builds the haystack.
//
// Generated rather than written out, because 44 hand-written logs is 44 things
// to keep consistent, but generated DETERMINISTICALLY: every value is indexed
// out of a fixed table by the day number, there is no clock and no random
// source, and the distinctive lines are placed at fixed days. Two Resets
// produce identical bytes, which is what [TestFixtureWriteIsDeterministic]
// checks for every task in the suite.
func needleFiles() map[string]string {
	files := map[string]string{"README.md": needleREADME}
	for day := 1; day <= NeedleFileCount; day++ {
		files[fmt.Sprintf("logs/day-%02d.md", day)] = needleLog(day)
	}
	return files
}

// needleLog renders one day.
func needleLog(day int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Engineering log — day %02d\n\n", day)
	fmt.Fprintf(&b, "on call: %s\n\n", needleOncall[day%len(needleOncall)])
	fmt.Fprintf(&b, "- 09:15 standup: %s\n", needleStandup[day%len(needleStandup)])
	fmt.Fprintf(&b, "- 11:%02d %s\n", day%60, needleMorning[(day*3)%len(needleMorning)])

	// The middle slot is where the distinctive lines go. Keyed on the day
	// number, so the needle is always day 29 and never anywhere else.
	if line, ok := needleAfternoon[day]; ok {
		fmt.Fprintf(&b, "- 14:%02d %s\n", (day*7)%60, line)
	} else {
		fmt.Fprintf(&b, "- 14:%02d %s\n", (day*7)%60, needleFiller[(day*5)%len(needleFiller)])
	}

	fmt.Fprintf(&b, "- 15:%02d %s\n", (day*13)%60, needleFiller[(day*5+7)%len(needleFiller)])
	fmt.Fprintf(&b, "- 16:%02d %s\n", (day*11)%60, needleEvening[(day*2)%len(needleEvening)])
	fmt.Fprintf(&b, "- 17:30 wrap: %s\n", needleWrap[day%len(needleWrap)])
	fmt.Fprintf(&b, "\nnotes: %s\n", needleNotes[(day*3+1)%len(needleNotes)])
	return b.String()
}

const needleREADME = `# daily logs

One markdown file per working day under logs/, written by whoever was on call.
Format is loose — a heading, who was on call, and a bullet per notable event
with the time in front of it.

There is no index and no search tool. There has never been one; every time
somebody proposes building one it loses to something with a customer attached.
`

// needleAfternoon is the whole point of the fixture: the needle and its
// distractors, pinned to fixed days.
//
// Day 29 is the only line that mentions BOTH a rollback and the settlement
// ledger. Days 6, 13, 34 and 41 mention a rollback of something else, and days
// 3, 17, 22, 31 and 38 mention settlement without a rollback — so a grep for
// either word alone returns five or six candidates that all read plausibly,
// and the agent has to intersect them.
var needleAfternoon = map[int]string{
	3:  "settlement batch ran 40 minutes late again; the ledger caught up by 15:00, no data lost",
	6:  "rolled back deploy-2c19ee0 in search-index — the new tokenizer doubled p99 and nothing else",
	13: "rolled back deploy-b840f52 in web-frontend after the asset hashes came out wrong",
	17: "settlement reconciliation finished clean for the first time in three days",
	22: "walked the new hire through the settlement ledger schema, wrote it up in the wiki",
	29: "rolled back " + NeedleAnswer + " — it had been writing malformed rows into the settlement " +
		"ledger since the morning deploy, and payments spotted the corruption before we did",
	31: "settlement volume peaked at 4x normal; ledger held, no rollback needed",
	34: "rolled back deploy-91ab77d in metrics-scrape, the scrape interval change was wrong",
	38: "audited the settlement ledger for the quarter; three rows needed manual correction",
	41: "rolled back deploy-55c0de1 in notifications after the template cache went stale",
}

var needleOncall = []string{
	"priya", "marek", "dani", "sam", "ines", "tobias", "rae",
}

var needleStandup = []string{
	"three in flight, nothing blocked",
	"still chasing the flaky integration test",
	"capacity review moved to Thursday",
	"nobody blocked; two reviews outstanding",
	"discussed the on-call handover doc",
	"quiet week, catching up on cleanup",
	"planning the storage migration",
	"reviewing last month's incident actions",
	"one blocked on a review, otherwise fine",
}

var needleMorning = []string{
	"merged the retry-budget change into the gateway",
	"bumped the connection pool on the read replicas",
	"deleted 200 lines of dead feature-flag code",
	"fixed the dashboard that had been showing stale data since the clock change",
	"reviewed the disaster-recovery runbook, found two stale hostnames",
	"paired on the config loader refactor",
	"turned the debug logging back down after last week's investigation",
	"rotated the service credentials on schedule",
	"updated the build image to the current toolchain",
	"triaged the overnight alerts; all three were the same noisy check",
	"cleaned up the orphaned volumes in staging",
	"documented the queue-depth alert thresholds",
}

var needleFiller = []string{
	"load test at 3x baseline, everything held",
	"cache hit rate back to normal after the warmup change",
	"backfill job finished, 2.1M rows",
	"schema migration applied in staging, no drift",
	"reviewed the vendor's postmortem for last month's outage",
	"reduced the alert threshold on queue depth after two false pages",
	"spent the afternoon on the quarterly capacity model",
	"disk on the build host filled up again; pruned the cache",
	"rebuilt the flaky CI worker from the base image",
	"traced a latency spike to a noisy neighbour, moved the pod",
	"the new dashboards went live, mostly positive feedback",
	"upgraded the log shipper on half the fleet",
	"wrote the design doc for the retry budget",
	"cut the release branch, nothing unusual in it",
	"deprecated the old health-check endpoint, two callers left",
}

var needleEvening = []string{
	"deploy went out clean",
	"deploy held back until tomorrow, not enough of the team around",
	"paged once, and it resolved itself before anyone had finished logging in",
	"error rate flat all day",
	"no deploys today by choice, change freeze for the audit",
	"canary looked fine for an hour, promoted it",
	"rolled the canary forward to 50%",
	"latency graph has a step in it nobody can explain yet",
}

// needleNotes is the padding, and it is padding on purpose: a haystack made
// only of one-line bullets compresses in a reader's head, and the task is
// supposed to cost something to read.
var needleNotes = []string{
	"nothing outstanding from yesterday; the queue-depth alert has not fired since the threshold change, " +
		"and the two follow-ups from last week are both waiting on somebody else.",
	"the build host is still slower than it should be after the disk filled up. Somebody should work out " +
		"whether the cache pruning is running at all, but it is not urgent and nobody has picked it up.",
	"reminder for whoever is on next: the staging environment has been rebuilt, so any bookmarked host " +
		"names from before this week point at nothing. Use the service discovery entries instead.",
	"quiet enough that we got some of the backlog done. Two of the alerts we have been ignoring turned " +
		"out to have been misconfigured since the migration, and are now off.",
	"the vendor has still not replied about last month's incident. Chased once by email, once in the " +
		"shared channel. Escalating next week if there is still nothing.",
	"handover: nothing hot, one deploy waiting for review, and the change advisory board paperwork for " +
		"next week's migration is drafted but not submitted.",
	"spent longer than expected on the on-call handover doc. It is out of date in about six places, all " +
		"of them hostnames, and fixing them properly means fixing the generator that produced them.",
	"the latency step in the afternoon graph is still unexplained. It does not correlate with a deploy, " +
		"a config change, or traffic. Filed a follow-up and moved on.",
	"good day. Everything that was supposed to go out went out, the canary looked normal the whole way, " +
		"and nobody was paged.",
	"note for the retro: three of this week's alerts were the same underlying cause and we treated them " +
		"as three separate investigations. That is a process problem, not a systems one.",
	"the backfill is finally done. Row counts reconcile against the source within the tolerance we agreed, " +
		"and the temporary capacity has been released.",
}

var needleWrap = []string{
	"quiet day",
	"handover notes in the channel",
	"two follow-ups filed",
	"nothing outstanding",
	"one action for tomorrow: chase the vendor",
	"handed over to the next shift, nothing hot",
	"filed the paperwork for the change advisory board",
}
