package benchmarks

import "github.com/hoobean1996/wombat-go/rl"

// todoApp is the greenfield task and the hardest of the four: an empty
// directory, a go.mod, and a program to write from nothing.
//
// The fixture is one file. go.mod is provided rather than left to the agent
// for a hermeticity reason and not a kindness one — `go mod init` is fine, but
// an agent that instead runs `go mod tidy` against GOPROXY=off spends turns on
// an error that has nothing to do with the task. Handing it a module with no
// requires makes "standard library only" the path of least resistance.
//
// Five verifiers, weighted so that vet/build/test/a-test-file together are
// worth less than the one check that the program actually does the job:
// `go build` passing on a CLI that stores nothing is not partial credit for a
// todo list, it is partial credit for compiling.
func todoApp() Task {
	return Task{
		ID:      "todo-app",
		Summary: "write a persistent todo CLI from scratch, with tests",
		Prompt: `Write a command-line todo application in Go, in this directory.

go.mod already exists (module todo). Use the standard library only — do not add
dependencies, and do not run ` + "`go get`" + `, there is no module proxy here.

Requirements:
- package main at the module root, so ` + "`go run .`" + ` works.
- Three subcommands:
  - ` + "`add <text>`" + ` — add a todo item.
  - ` + "`list`" + ` — print every item, one per line. Each line must contain
    that item's text.
  - ` + "`done <n>`" + ` — mark item n complete; ` + "`list`" + ` must show it
    as complete afterwards.
- Items persist between runs in a JSON file in the current working directory,
  so ` + "`add`" + ` in one process is visible to ` + "`list`" + ` in the next.
- At least one ` + "`*_test.go`" + ` file, testing something real.
- ` + "`go vet ./...`" + `, ` + "`go build ./...`" + ` and ` + "`go test ./...`" + `
  must all pass.`,
		Files: map[string]string{"go.mod": todoAppGoMod},
		Verifiers: []rl.Verifier{
			rl.Shell("vet", "go vet ./...", 0.15),

			// `go build ./...` alone is a freebie on this task: the fixture is
			// a go.mod and nothing else, ./... matches no packages, and the
			// command exits 0. The second half makes it mean what it looks
			// like it means — there is a main package at the module root.
			rl.Shell("build", "go build ./... && go build -o /dev/null .", 0.15),
			rl.Shell("test", "go test ./...", 0.20),
			rl.Shell("has_test_file", `ls *_test.go >/dev/null 2>&1`, 0.10),

			// The one that is actually the task: state has to survive the
			// process exiting, which is what separates a todo list from a
			// program that parses three subcommands.
			rl.Shell("add_then_list",
				`go run . add "buy milk" && go run . list | grep -q "buy milk"`, 0.40),
		},
	}
}

const todoAppGoMod = `module todo

go 1.25
`
