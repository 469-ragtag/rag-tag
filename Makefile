SHELL := /bin/bash

.DEFAULT_GOAL := help

CONFIG ?=
CONFIG_ARGS = $(if $(strip $(CONFIG)),--config $(CONFIG),)

DATASET ?= BigBuildingBIMModel
DB ?= output/$(DATASET).db
GRAPH_DATASET ?= $(DATASET)
APP_ARGS ?=

BENCH_DATASET ?= Building-Architecture
BENCH_DB ?= output/$(BENCH_DATASET).db
BENCH_GRAPH_DATASET ?= $(BENCH_DATASET)
BENCH_EXPERIMENT ?= benchmark-e2e-v1
BENCH_QUESTIONS_FILE ?=
BENCH_SOURCE_ARGS = $(if $(strip $(BENCH_QUESTIONS_FILE)),--questions-file $(BENCH_QUESTIONS_FILE),--experiment $(BENCH_EXPERIMENT))
BENCH_ANSWER_JUDGE_MODEL ?= google-gla:gemini-2.5-flash
BENCH_TAGS ?=
BENCH_ARGS ?=

ROUTING_DB ?= $(BENCH_DB)
ROUTING_MODE ?=
ROUTING_ARGS ?=

GRAPH_COMPARE_EXPERIMENT ?= graph-agent-compare
GRAPH_COMPARE_DB ?= $(BENCH_DB)
GRAPH_COMPARE_DATASET ?= $(BENCH_GRAPH_DATASET)
GRAPH_COMPARE_ARGS ?=

TEST_ARGS ?=
LINT_ARGS ?=

.PHONY: help setup install-hooks pre-commit ontology-map refresh-ifc43-rdf \
	ifc-to-jsonl jsonl-to-sql jsonl-to-graph build run run-trace tui tui-trace \
	tui-ready benchmark benchmark-trace benchmark-sql benchmark-graph \
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

benchmark: ## Run the benchmark suite for BENCH_DATASET
	uv run python scripts/eval_benchmarks.py $(CONFIG_ARGS) $(BENCH_SOURCE_ARGS) --db $(BENCH_DB) --graph-dataset $(BENCH_GRAPH_DATASET) --answer-judge-model $(BENCH_ANSWER_JUDGE_MODEL) $(if $(strip $(BENCH_TAGS)),--tags $(BENCH_TAGS),) $(BENCH_ARGS)

benchmark-trace: ## Run the benchmark suite with tracing enabled
	uv run python scripts/eval_benchmarks.py $(CONFIG_ARGS) $(BENCH_SOURCE_ARGS) --db $(BENCH_DB) --graph-dataset $(BENCH_GRAPH_DATASET) --answer-judge-model $(BENCH_ANSWER_JUDGE_MODEL) --trace $(if $(strip $(BENCH_TAGS)),--tags $(BENCH_TAGS),) $(BENCH_ARGS)

benchmark-sql: ## Run only benchmark cases tagged with sql
	$(MAKE) benchmark BENCH_TAGS="sql"

benchmark-graph: ## Run only benchmark cases tagged with graph
	$(MAKE) benchmark BENCH_TAGS="graph"

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
