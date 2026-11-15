.DEFAULT_GOAL := check

# wombatd reads one YAML file and nothing else; copy wombatd.example.yaml to
# start. The live runs further down still take their endpoint from the
# environment, sourced from $(ENV).
CONFIG ?= wombatd.yaml
ENV    ?= .env

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

.PHONY: serve
serve: bin ## the agent, configured by $(CONFIG): make serve CONFIG=other.yaml
	$(BIN)/wombatd -config $(CONFIG)

# ── Postgres, in podman ─────────────────────────────────────────────────
#
# A throwaway database for store.driver: postgres and for the store's tests,
# on a port of its own so it cannot be mistaken for a real one:
#
#   make pg-up        # then, in wombatd.yaml: store: {driver: postgres, dsn: $(PG_DSN)}
#   make test-pg      # the Postgres store's tests, the daemon's restart test among them
#   make pg-down

PODMAN  ?= podman
PG_NAME ?= wombat-pg
PG_PORT ?= 55432
PG_DSN  ?= postgres://wombat:wombat@127.0.0.1:$(PG_PORT)/wombat

.PHONY: pg-up
pg-up: ## start a Postgres for wombatd in podman, on $(PG_PORT)
	@$(PODMAN) machine inspect --format '{{.State}}' 2>/dev/null | grep -qx running || \
	  $(PODMAN) machine start 2>/dev/null || true
	@if $(PODMAN) container exists $(PG_NAME); then $(PODMAN) start $(PG_NAME) >/dev/null; else \
	  $(PODMAN) run -d --name $(PG_NAME) -p 127.0.0.1:$(PG_PORT):5432 \
	    -e POSTGRES_USER=wombat -e POSTGRES_PASSWORD=wombat -e POSTGRES_DB=wombat \
	    docker.io/library/postgres:17 >/dev/null; fi
	@until $(PODMAN) exec $(PG_NAME) pg_isready -U wombat -d wombat >/dev/null 2>&1; do sleep 0.5; done
	@echo "postgres is up: $(PG_DSN)"

.PHONY: pg-down
pg-down: ## stop and remove the podman Postgres, and its data
	-$(PODMAN) rm -f $(PG_NAME)

.PHONY: test-pg
test-pg: ## the Postgres store's tests, against $(PG_DSN)
	$(GO) test -race -count=1 ./pkg/store/postgres/ -args -dsn '$(PG_DSN)'

# ── live runs against a real endpoint ──────────────────────────────────
#
# Every target below sources $(ENV) first. Nothing in the library reads the
# environment on its own except the API key, so this is the only place
# configuration is implicit.

N     ?= 8
C     ?= 4
OUT   ?= runs
TASKS ?=

# The suite and the judge are tests that run only when handed arguments, so
# `go test ./...` never spends money; everything after `--` is their own flags.
.PHONY: bench
bench: ## run the benchmark suite: make bench N=8 C=4 TASKS=todo-app,fix-bug
	@set -a; . $(ENV); set +a; \
	$(GO) test ./eval/benchmarks -run '^TestSuite$$' -count=1 -timeout 0 -v -args -- \
	  -n $(N) -c $(C) -out $(abspath $(OUT)) $(if $(TASKS),-tasks $(TASKS),) $(FLAGS)

IN ?= $(OUT)/episodes.jsonl

.PHONY: judge
judge: ## score a benchmark run's episodes: make judge IN=runs/episodes.jsonl [LLM=1]
	@set -a; . $(ENV); set +a; \
	$(GO) test ./eval/judge -run '^TestJudgeEpisodes$$' -count=1 -v -args -- \
	  -in $(abspath $(IN)) $(if $(LLM),-llm,) $(FLAGS)

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
