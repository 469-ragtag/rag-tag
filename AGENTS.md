# AGENTS.md

This file is for agentic coding assistants operating in this repository.

## Project Summary

- Domain: BIM / IFC digital twins
- Goal: Natural-language querying over IFC models using a hybrid retrieval pipeline
  - Deterministic aggregations and counts via SQL (no hallucinations)
  - Spatial/topological questions via graph traversal
- Primary pipeline: IFC -> CSV -> SQLite + NetworkX graph + LLM tool interface

## Repo Status Snapshot

- IFC parsing and export works via `parser/ifc_to_csv.py`.
- Geometry extraction computes centroids and bounding boxes.
- Graph is currently built in NetworkX with distance-based adjacency.
- SQL schema exists and is wired into the router for counts/lists.

## Cursor/Copilot Rules

- No Cursor rules detected (`.cursorrules` / `.cursor/rules/` not present).
- No Copilot rules detected (`.github/copilot-instructions.md` not present).
- If these files are added later, treat them as higher priority and mirror key rules here.

## Environment + Tooling

- Python: requires >=3.14 (see `pyproject.toml`)
- Package manager/runner: `uv` (use `uv run ...`)
- Lint/format: `ruff` (configured in `pyproject.toml`)
- Pre-commit: `pre-commit` is listed in dev deps

## Commands (Build/Lint/Test)

Use `uv` for all commands.

### Environment

- Sync environment:
  - `uv sync --group dev`
- Install pre-commit hooks:
  - `uv run pre-commit install`

### Run scripts

- IFC -> CSV:
  - `uv run python parser/ifc_to_csv.py`
- CSV -> SQLite:
  - `uv run python parser/csv_to_sql.py`
- CSV -> Graph (Plotly HTML):
  - `uv run python parser/csv_to_graph.py`
- Run the LLM agent (requires API key):
  - `COHERE_API_KEY=... uv run python run_agent.py`
  - Debug LLM I/O:
    - `COHERE_API_KEY=... uv run python run_agent.py --input`
  - Force router mode:
    - `ROUTER_MODE=rule uv run python run_agent.py`
    - `ROUTER_MODE=llm GEMINI_API_KEY=... uv run python run_agent.py`
  - Use a specific SQLite DB:
    - `COHERE_API_KEY=... uv run python run_agent.py --db ./output/Building-Architecture.db`
  - Output formatting:
    - Questions and answers are prefixed with `Q:` / `A:` and separated by divider lines.

### Linting + Formatting (ruff)

- Format all:
  - `uv run ruff format .`
- Check formatting (CI-safe):
  - `uv run ruff format --check .`
- Lint all:
  - `uv run ruff check .`
- Lint with autofix:
  - `uv run ruff check --fix .`
- Lint single file:
  - `uv run ruff check path/to/file.py`

### Tests

- No tests are currently present in the repo.
- If tests are added, standard commands should be:
  - `uv run pytest`
  - `uv run pytest tests/test_file.py`
  - `uv run pytest tests/test_file.py::test_name`

## Code Style Guidelines

### Formatting

- Use `ruff format` and avoid manual formatting tweaks.
- Keep line length <= 88 (see `pyproject.toml`).
- Prefer clarity over compact one-liners.

### Imports

- Group imports: stdlib, third-party, local.
- Avoid wildcard imports and unused imports (ruff will flag).
- Prefer absolute imports once a `src/` layout exists.

### Types

- Type all public functions and key pipeline steps.
- Prefer `pathlib.Path` over `str` for paths.
- Use `dataclass` or `TypedDict` for structured data.
- Avoid `# type: ignore` unless there is a strong justification.

### Naming

- Files/modules: `snake_case.py`.
- Functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Use IFC-consistent terms: `global_id`, `ifc_class`, `storey`, `space`, `element`.

### Error Handling

- Validate IFC inputs before heavy parsing.
- Raise specific exceptions with actionable messages.
- Catch exceptions only to add context; avoid blanket `except Exception`.
- Include identifiers in errors when available (IFC path, element id, class).

### Logging

- Prefer `logging` over `print` for pipeline stages.
- Log high-level stages: validate -> parse -> extract -> geometry -> SQL -> graph.
- Avoid logging full IFC objects; log IDs and counts.

### Data Contracts (Answers)

- Never fabricate GUIDs; only return IDs that exist in parsed data.
- For counts and aggregations, always use SQL results, not LLM guesses.
- Prefer returning both:
  - Human-readable summary
  - Structured JSON (entity type, count, IDs, and properties)

### Graph Semantics

- `adjacent_to` edges should reflect spatial proximity, not label heuristics.
- `contains` / `typed_by` edges should be derived from IFC relationships.
- If geometry is missing, degrade gracefully and log the fallback.

### SQL Safety

- Use parameterized queries only (`?` placeholders).
- Do not interpolate user input directly into SQL.

## Known Gaps (Do Not Ignore)

- Graph adjacency uses centroid distance, not true topology.
- No Neo4j/Cypher backend yet; graph lives in NetworkX only.
