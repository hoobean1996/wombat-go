.DEFAULT_GOAL := check

# Where the environment comes from. Defaults to a local .env, but points at
# the OCaml wombat's config out of the box so an existing working setup can be
# reused without copying secrets around:
#
#     make ask ENV=.env Q='what is 6*7?'
ENV ?= .env

GO      ?= go
BIN     := bin
PKGS    := ./...

.PHONY: check
check: fmt vet build ## build, vet and gofmt

.PHONY: fmt
fmt:
	@out=$$(gofmt -l . ); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

.PHONY: vet
vet:
	$(GO) vet $(PKGS)

.PHONY: build
build:
	$(GO) build $(PKGS)

.PHONY: test
test:
	$(GO) test -race $(PKGS)

.PHONY: bin
bin:
	@mkdir -p $(BIN)
	$(GO) build -o $(BIN)/ ./cmd/...

.PHONY: ts
ts: ## regenerate the TypeScript event declarations
	@mkdir -p web
	$(GO) run ./cmd/wombat-tsgen > web/events.ts
	@echo "web/events.ts"

.PHONY: serve
serve: bin ## resumable SSE server + single-page client on :8080
	@set -a; . $(ENV); set +a; $(BIN)/wombat-serve $(FLAGS)

# ── live runs against a real endpoint ──────────────────────────────────
#
# Every target below sources $(ENV) first. Nothing in the library reads the
# environment on its own except the API key, so this is the only place
# configuration is implicit.

# For a prompt, use scripts/wombat rather than a make target: a make variable
# cannot carry prose safely, because a quote or an && in it is re-parsed by the
# shell make invokes and the pipeline silently breaks.
#
#   scripts/wombat "your prompt"
#   scripts/wombat -r -working-dir /tmp/x "with 'quotes' && ampersands"

.PHONY: ask-raw
ask-raw: bin ## raw JSONL, no prompt quoting help: make ask-raw Q=hello
	@set -a; . $(ENV); set +a; $(BIN)/wombat-jsonl $(FLAGS) "$(Q)"

N     ?= 8
C     ?= 4
OUT   ?= runs
TASKS ?=

.PHONY: bench
bench: bin ## run the benchmark suite: make bench N=8 C=4 TASKS=todo-app,fix-bug
	@set -a; . $(ENV); set +a; \
	$(BIN)/wombat-bench -n $(N) -c $(C) -out $(OUT) $(if $(TASKS),-tasks $(TASKS),) $(FLAGS)

.PHONY: models
models: ## list model ids the configured OpenAI-compatible endpoint offers
	@set -a; . $(ENV); set +a; \
	curl -s $${OPENAI_PROXY:+-x http://$$OPENAI_PROXY} "$$OPENAI_BASE_URL/models" \
	  -H "Authorization: Bearer $$OPENAI_API_KEY" | \
	  python3 -c 'import sys,json;[print(m["id"]) for m in json.load(sys.stdin)["data"]]'

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
