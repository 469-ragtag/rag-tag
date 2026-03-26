SHELL := /bin/bash

.DEFAULT_GOAL := help

CONFIG ?=
CONFIG_ARGS = $(if $(strip $(CONFIG)),--config $(CONFIG),)

DATASET ?= Building-Architecture
DB ?= output/$(DATASET).db
GRAPH_DATASET ?= $(DATASET)
APP_ARGS ?=

TARGET ?=
PRESET ?= smoke
BENCH_TARGET ?= $(TARGET)
BENCH_PRESET ?= $(PRESET)
BENCH_DB ?=
BENCH_GRAPH_DATASET ?=
BENCH_EXPERIMENT ?=
BENCH_QUESTIONS_FILE ?=
BENCH_SELECTOR_ARGS = $(if $(strip $(BENCH_QUESTIONS_FILE)),--questions-file $(BENCH_QUESTIONS_FILE),$(if $(strip $(BENCH_EXPERIMENT)),--experiment $(BENCH_EXPERIMENT),$(if $(strip $(BENCH_PRESET)),--preset $(BENCH_PRESET),))) $(if $(strip $(BENCH_TARGET)),--target $(BENCH_TARGET),)
BENCH_ANSWER_JUDGE_MODEL ?=
BENCH_TAGS ?=
BENCH_RESOURCE_ARGS = $(if $(strip $(BENCH_DB)),--db $(BENCH_DB),) $(if $(strip $(BENCH_GRAPH_DATASET)),--graph-dataset $(BENCH_GRAPH_DATASET),) $(if $(strip $(BENCH_ANSWER_JUDGE_MODEL)),--answer-judge-model $(BENCH_ANSWER_JUDGE_MODEL),) $(if $(strip $(BENCH_TAGS)),--tags $(BENCH_TAGS),)
BENCH_ARGS ?=

ROUTING_DB ?= output/Building-Architecture.db
ROUTING_MODE ?=
ROUTING_ARGS ?=

GRAPH_COMPARE_EXPERIMENT ?= graph-agent-compare
GRAPH_COMPARE_DB ?= output/Building-Architecture.db
GRAPH_COMPARE_DATASET ?= Building-Architecture
GRAPH_COMPARE_ARGS ?=

TEST_ARGS ?=
LINT_ARGS ?=

.PHONY: help setup install-hooks pre-commit ontology-map refresh-ifc43-rdf \
	ifc-to-jsonl jsonl-to-sql jsonl-to-graph build run run-trace tui tui-trace \
	tui-ready bench bench-trace benchmark benchmark-trace benchmark-sql benchmark-graph \
	routing-eval routing-eval-strict graph-agent-compare check-payload-parity \
	check-graph-relationships check-graph-contract check-batch4 check-batch5 \
	check-batch6 test lint lint-fix format check-format

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "\nAvailable targets:\n\n"} /^[a-zA-Z0-9_.-]+:.*## / { printf "  %-24s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Install project and dev dependencies
	uv sync --group dev

install-hooks: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit: ## Run pre-commit on all files
	uv run pre-commit run --all-files

ontology-map: ## Generate the IFC ontology map
	uv run rag-tag-generate-ontology-map

refresh-ifc43-rdf: ## Refresh the IFC4.3 RDF snapshot
	uv run rag-tag-refresh-ifc43-rdf

ifc-to-jsonl: ## Convert IFC files to JSONL artifacts
	uv run rag-tag-ifc-to-jsonl

jsonl-to-sql: ## Convert JSONL artifacts to SQLite databases
	uv run rag-tag-jsonl-to-sql

jsonl-to-graph: ## Build the graph artifacts from JSONL
	uv run rag-tag-jsonl-to-graph

build: ## Run the full IFC -> JSONL -> SQLite -> graph pipeline
	$(MAKE) ifc-to-jsonl
	$(MAKE) jsonl-to-sql
	$(MAKE) jsonl-to-graph

run: ## Run the CLI app for DATASET
	uv run rag-tag $(CONFIG_ARGS) --db $(DB) --graph-dataset $(GRAPH_DATASET) $(APP_ARGS)

run-trace: ## Run the CLI app with tracing enabled
	uv run rag-tag $(CONFIG_ARGS) --db $(DB) --graph-dataset $(GRAPH_DATASET) --trace $(APP_ARGS)

tui: ## Run the TUI for DATASET
	uv run rag-tag $(CONFIG_ARGS) --tui --db $(DB) --graph-dataset $(GRAPH_DATASET) $(APP_ARGS)

tui-trace: ## Run the TUI with tracing enabled
	uv run rag-tag $(CONFIG_ARGS) --tui --db $(DB) --graph-dataset $(GRAPH_DATASET) --trace $(APP_ARGS)

tui-ready: ## Rebuild artifacts, then launch the traced TUI
	$(MAKE) build
	$(MAKE) tui-trace

bench: ## Run benchmarks via TARGET/PRESET or compatibility selectors
	uv run python scripts/eval_benchmarks.py $(CONFIG_ARGS) $(BENCH_SELECTOR_ARGS) $(BENCH_RESOURCE_ARGS) $(BENCH_ARGS)

bench-trace: ## Run benchmarks with tracing enabled
	uv run python scripts/eval_benchmarks.py $(CONFIG_ARGS) $(BENCH_SELECTOR_ARGS) $(BENCH_RESOURCE_ARGS) --trace $(BENCH_ARGS)

benchmark: ## Compatibility alias for bench
	uv run python scripts/eval_benchmarks.py $(CONFIG_ARGS) $(BENCH_SELECTOR_ARGS) $(BENCH_RESOURCE_ARGS) $(BENCH_ARGS)

benchmark-trace: ## Compatibility alias for bench-trace
	uv run python scripts/eval_benchmarks.py $(CONFIG_ARGS) $(BENCH_SELECTOR_ARGS) $(BENCH_RESOURCE_ARGS) --trace $(BENCH_ARGS)

benchmark-sql: ## Run only benchmark cases tagged with sql
	$(MAKE) bench BENCH_TAGS="sql"

benchmark-graph: ## Run only benchmark cases tagged with graph
	$(MAKE) bench BENCH_TAGS="graph"

routing-eval: ## Run the router evaluation script
	uv run python scripts/eval_routing.py --db $(ROUTING_DB) $(if $(strip $(ROUTING_MODE)),--router-mode $(ROUTING_MODE),) $(ROUTING_ARGS)

routing-eval-strict: ## Run the router evaluation and fail on route mismatches
	uv run python scripts/eval_routing.py --db $(ROUTING_DB) --strict $(if $(strip $(ROUTING_MODE)),--router-mode $(ROUTING_MODE),) $(ROUTING_ARGS)

graph-agent-compare: ## Run the graph-agent comparison experiment
	uv run python scripts/eval_graph_models.py $(CONFIG_ARGS) --experiment $(GRAPH_COMPARE_EXPERIMENT) --db $(GRAPH_COMPARE_DB) --graph-dataset $(GRAPH_COMPARE_DATASET) $(GRAPH_COMPARE_ARGS)

check-payload-parity: ## Run payload-mode parity checks
	uv run python scripts/check_payload_mode_parity.py

check-graph-relationships: ## Run graph relationship checks
	uv run python scripts/check_graph_relationships.py

check-graph-contract: ## Run graph contract checks
	uv run python scripts/check_graph_contract.py

check-batch4: ## Run batch 4 verification script
	uv run python scripts/check_batch4.py

check-batch5: ## Run batch 5 verification script
	uv run python scripts/check_batch5.py

check-batch6: ## Run batch 6 verification script
	uv run python scripts/check_batch6.py

test: ## Run pytest
	uv run pytest $(TEST_ARGS)

lint: ## Run Ruff lint checks
	uv run ruff check . $(LINT_ARGS)

lint-fix: ## Run Ruff lint checks with auto-fix
	uv run ruff check --fix . $(LINT_ARGS)

format: ## Format the codebase with Ruff
	uv run ruff format .

check-format: ## Check formatting without modifying files
	uv run ruff format --check .
