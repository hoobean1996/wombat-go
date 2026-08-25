package benchmarks

import "github.com/hoobean1996/wombat-go/rl"

// readOnlyQA is the easiest task in the suite and the one that isolates
// retrieval.
//
// The answer is not in either file on its own: docs/ports.md maps the port to
// an internal codename and docs/services.md maps the codename to the service,
// so a single grep cannot produce it and a model that guesses from the port
// number has nothing to guess from. Two hops is the smallest number that
// distinguishes "found the file" from "read the file".
//
// Nothing here needs the Go toolchain, which makes it the task to run when you
// want to know whether the harness itself works.
func readOnlyQA() Task {
	return Task{
		ID:      "read-only-qa",
		Summary: "answer a question that needs two files cross-referenced",
		Prompt: `This directory contains a small documentation tree under docs/.

Question: which service listens on port 8081?

Read the documentation to find out, then write ONLY that service's name — no
punctuation, no sentence, no explanation — into a file called ANSWER.txt at the
root of this directory. Then stop and tell me the answer.`,
		Files: map[string]string{
			"README.md":         qaREADME,
			"docs/ports.md":     qaPorts,
			"docs/services.md":  qaServices,
			"docs/runbook.md":   qaRunbook,
			"docs/changelog.md": qaChangelog,
		},
		Verifiers: []rl.Verifier{
			rl.FileExists("answer_file", "ANSWER.txt", 0.2),
			rl.FileContains("answer", "ANSWER.txt", "ledger-api", 0.8),
		},
	}
}

const qaREADME = `# platform docs

Operational documentation for the platform team.

- docs/ports.md — which port belongs to which codename
- docs/services.md — which codename belongs to which service
- docs/runbook.md — what to do when one of them is down
- docs/changelog.md — recent changes
`

// Deliberately does NOT name a service. The port table is by codename, which
// is the first hop.
const qaPorts = `# Port allocations

Ports are allocated to internal CODENAMES, never to service names — services
get renamed and ports do not. Look the codename up in docs/services.md to find
out which service it currently belongs to.

| port | codename |
|------|----------|
| 8080 | quartz   |
| 8081 | basalt   |
| 8082 | gneiss   |
| 9090 | obsidian |
`

const qaServices = `# Services

| codename | service        | owning team |
|----------|----------------|-------------|
| quartz   | web-frontend   | platform    |
| basalt   | ledger-api     | payments    |
| gneiss   | search-index   | discovery   |
| obsidian | metrics-scrape | infra       |
`

const qaRunbook = `# Runbook

If a service stops answering its health check:

1. Find the port in docs/ports.md and note the codename.
2. Find the codename in docs/services.md and note the owning team.
3. Page the owning team. Do not restart anything owned by payments without
   telling them first — a restart mid-settlement leaves partial ledger rows.
`

const qaChangelog = `# Changelog

## 2026-02-11
- obsidian moved from 9091 to 9090.

## 2026-01-30
- gneiss took over the search workload from quartz. Quartz keeps 8080.

## 2026-01-04
- basalt was split out of quartz and given its own port.
`
