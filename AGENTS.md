# AGENTS.md

Operational instructions for coding agents working in this repository.

## 1) Instruction Precedence (MUST FOLLOW)

When instructions conflict, resolve in this order:

1. System/developer/runtime instructions from the active coding environment
2. This `AGENTS.md`
3. User request
4. Other repository docs (`README.md`, module docs, comments)

If conflict remains unclear, ask the user explicitly before editing.

## 2) Project Snapshot

- Domain: BIM / IFC digital twins
- Goal: Natural-language querying over IFC models using a hybrid retrieval pipeline
    - Deterministic counts/aggregations via SQL (no hallucinations)
    - Spatial/topological questions via graph traversal
- Core pipeline: IFC -> CSV -> SQLite + NetworkX graph + LLM tool interface

Current state:

- IFC parsing/export works via `src/rag_tag/parser/ifc_to_csv.py`
- Geometry extraction computes centroids and bounding boxes
- Graph is NetworkX with distance-based adjacency
- SQL schema exists and is used for counts/lists
- Router and graph agent are migrated to PydanticAI
- Router default model: `google-gla:gemini-2.5-flash`
- Graph agent default model: `cohere:command-a-03-2025`
- Optional observability via Logfire (`--trace`)
- Tool outputs normalized as `{status,data,error}` envelope
- PydanticAI providers: `google-gla` (AI Studio), `google-vertex` (Vertex)
- `COHERE_API_KEY` is auto-mapped to `CO_API_KEY`

## 3) Default Operating Mode for Feature Work

For any request to implement, modify, refactor, or build a feature/spec:

### 3.1 Plan-First Gate (MANDATORY)

1. Enter **plan mode first**.
2. Produce a thorough end-to-end plan **before writing code**.
3. Plan MUST include:
    - Scope and non-goals
    - Assumptions and dependencies
    - Affected files/modules
    - Validation strategy (lint/tests/manual checks)
    - Ordered implementation steps with dependency notes
4. Ask for **explicit user approval**.
5. Only after approval, begin implementation.

Do not skip plan approval unless user explicitly says to bypass planning.

### 3.2 Step-Batch Execution (MANDATORY)

Implement approved plans in batches, not all at once.

Batching policy:

- Group tightly coupled, low-risk steps together.
- Keep risky, cross-cutting, or ambiguous steps isolated.
- Re-evaluate batch size based on new findings.

After each batch, report:

- What was completed
- What remains
- Any blockers or changed assumptions

If scope changes materially, re-plan and re-confirm with the user.

## 4) Execution Protocol (MUST)

1. Understand request and constraints fully.
2. Gather only necessary context from code/docs.
3. Prefer minimal, root-cause fixes over broad rewrites.
4. Preserve existing architecture and style unless user requests redesign.
5. Use safe, parameterized operations (especially SQL).
6. Validate changed behavior with the most targeted checks first, then broader checks as needed.
7. Provide concise handoff with changed files, validation run, and remaining risks.

## 5) Scope and Safety Guardrails

- Do not invent requirements.
- Do not modify unrelated areas.
- Do not fabricate IDs, GUIDs, outputs, or test results.
- Do not claim completion without validation evidence.
- If blocked by missing info, ask focused questions.
- If a task is destructive or high-risk, confirm before executing.

## 6) Skills Usage

Skills are in `.agents/skills`.

Use these when applicable:

- `ifcopenshell-python` skill for changes involving `ifcopenshell`
- `pydantic-ai` skill for changes involving PydanticAI (most of this repo)

## 7) Environment and Tooling

- Python: `>=3.14` (see `pyproject.toml`)
- Package manager/runner: `uv`
- Lint/format: `ruff`
- Pre-commit available in dev dependencies

### Setup

- Sync env: `uv sync --group dev`
- Install hooks: `uv run pre-commit install`

### Main scripts

- IFC -> CSV: `uv run rag-tag-ifc-to-csv`
- CSV -> SQLite: `uv run rag-tag-csv-to-sql`
- CSV -> Graph: `uv run rag-tag-csv-to-graph`
- Run app: `uv run rag-tag`

Examples:

- `COHERE_API_KEY=... uv run rag-tag`
- `GEMINI_API_KEY=... uv run rag-tag`
- `COHERE_API_KEY=... uv run rag-tag --input`
- `uv run rag-tag --verbose`
- `LOGFIRE_TOKEN=... uv run rag-tag --trace`

Router/model overrides:

- `ROUTER_MODE=rule uv run rag-tag`
- `ROUTER_MODE=llm GEMINI_API_KEY=... uv run rag-tag`
- `AGENT_MODEL=cohere:command-a-03-2025 COHERE_API_KEY=... uv run rag-tag`
- `ROUTER_MODEL=google-gla:gemini-2.5-flash GEMINI_API_KEY=... uv run rag-tag`
- `ROUTER_MODEL=google-gla:gemini-3-flash-preview GEMINI_API_KEY=... uv run rag-tag`
- `ROUTER_MODEL=google-vertex:gemini-2.5-flash GEMINI_API_KEY=... uv run rag-tag`

Use specific DB:

- `COHERE_API_KEY=... uv run rag-tag --db ./output/Building-Architecture.db`

Evaluate routing:

- `uv run python scripts/eval_routing.py --db ./output/Building-Architecture.db`

### Lint/format

- Format: `uv run ruff format .`
- Check format: `uv run ruff format --check .`
- Lint: `uv run ruff check .`
- Lint + fix: `uv run ruff check --fix .`
- Lint one file: `uv run ruff check path/to/file.py`

### Tests

- No tests currently in repo.
- If tests are added:
    - `uv run pytest`
    - `uv run pytest tests/test_file.py`
    - `uv run pytest tests/test_file.py::test_name`

## 8) Code Style and Quality Rules

### Formatting and imports

- Use `ruff format`; avoid manual style-only churn.
- Keep line length `<= 88`.
- Import groups: stdlib, third-party, local.
- Avoid wildcard/unused imports.
- Prefer absolute imports from `rag_tag`.

### Types and naming

- Type public functions and core pipeline logic.
- Prefer `pathlib.Path` over `str` for file paths.
- Use `dataclass`/`TypedDict` for structured payloads where useful.
- Naming:
    - modules/files: `snake_case.py`
    - functions/variables: `snake_case`
    - classes: `PascalCase`
    - constants: `UPPER_SNAKE_CASE`

### Error handling and logging

- Validate IFC inputs before heavy processing.
- Raise specific, actionable exceptions.
- Catch broad exceptions only to add context.
- Include identifiers in error messages where possible.
- Prefer `logging` over `print` for pipeline stages.

## 9) Domain-Specific Invariants (MUST)

### Data contract

- Never fabricate GUIDs/IDs.
- For counts/aggregations, use SQL results only.
- Prefer both:
    - human-readable summary
    - structured JSON payload (entity type/count/ids/properties)

### Graph semantics

- `adjacent_to` should reflect spatial proximity, not label heuristics.
- `contains` and `typed_by` should come from IFC relationships.
- If geometry is missing, degrade gracefully and log fallback behavior.

### SQL safety

- Always use parameterized queries (`?` placeholders).
- Never interpolate untrusted user input into SQL.

## 10) Known Gaps (Do Not Ignore)

- Graph adjacency is centroid-distance based, not true topology.
- No Neo4j/Cypher backend yet; graph remains in NetworkX.

## 11) Response Contract for Agents

During active work:

- Provide concise progress updates at meaningful checkpoints.
- Be explicit about assumptions and blockers.

At completion:

- Summarize what changed.
- List files touched.
- List validation commands run and outcomes.
- Note unresolved risks or follow-ups.

## 12) Project Structure (Reference)

- `src/rag_tag/`
    - `__init__.py`
    - `__main__.py`
    - `paths.py`
    - `ifc_graph_tool.py`
    - `ifc_sql_tool.py`
    - `observability.py`
    - `run_agent.py`
    - `tui.py`
    - `textual_app.py`
    - `query_service.py`
- `src/rag_tag/agent/`
    - `graph_agent.py`
    - `graph_tools.py`
    - `models.py`
- `src/rag_tag/llm/`
    - `pydantic_ai.py`
- `src/rag_tag/router/`
    - `router.py`, `rules.py`, `llm.py`, `models.py`, `llm_models.py`
- `src/rag_tag/parser/`
    - `ifc_to_csv.py`, `csv_to_sql.py`, `csv_to_graph.py`, `ifc_geometry_parse.py`, `sql_schema.py`
- `scripts/` (e.g., `eval_routing.py`)
- `output/` runtime-generated artifacts
- `IFC-Files/` source IFC models

## 13) Future Rule Files

If `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` are added later, treat them as higher-priority and keep this file aligned.
