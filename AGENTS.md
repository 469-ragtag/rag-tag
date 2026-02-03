# AGENTS.md

This file is for agentic coding assistants operating in this repository.

## Project (High-Level)

- Domain: BIM / Digital Twins (IFC)
- Goal: Natural-language queries over an IFC model using a hybrid approach:
    - Attribute + aggregation queries -> SQL/relational engine (deterministic counts/lists)
    - Spatial/topological queries -> graph engine (Neo4j/Cypher)

## Repo Status

This repo is a small Python prototype with a working IFC export script (see `parser/ifc-to-csv.py`).

## Policies (Non-Negotiable)

- No hallucinations on counts/aggregations: compute via SQL/relational query, not free-text.
- IFC handling: never assume an IFC is valid; validate before parsing (e.g., `ifcopenshell.validate()`).
- Graph edges: `CONTAINS` / `CONNECTED_TO` must be computed from geometry/topology, not guessed.
- Never fabricate IFC `global_id` / GUIDs; only return IDs that exist in the parsed model.

## Cursor/Copilot Rules

- No Cursor rules detected (`.cursorrules` / `.cursor/rules/` not present).
- No Copilot rules detected (`.github/copilot-instructions.md` not present).
- If these files are added later, treat them as higher priority than this document and mirror key rules here.

## Tooling (Recommended)

- Python: 3.14+ (repo sets `requires-python = ">=3.14"` and `.python-version` is `3.14`)
- Env + runner: `uv` (use `uv run ...`, do not rely on a system `python` executable)
- Lint + format: `ruff`
- Tests: `pytest`
- Type checking: `pyright`

## Commands (Build/Lint/Test)

This repo uses `uv.lock` + `pyproject.toml`, so these are the canonical commands.

### Environment

- Create/sync env from lockfile:
    - `uv sync`
- Add runtime deps:
    - `uv add <pkg>`
- Add dev deps:
    - `uv add --dev ruff pytest pyright`

Note: `.gitignore` currently ignores `AGENTS.md`.

### Run

- Run a module:
    - `uv run python -m <module>`
- Run a script:
    - `uv run python path/to/script.py`

### Repo Scripts

- IFC -> CSV export (recommended):
    - `uv run python parser/ifc-to-csv.py`
- With overrides:
    - `uv run python parser/ifc-to-csv.py --ifc-dir ./IFC-Files --out-dir ./output`

### Formatting (ruff)

- Format all:
    - `uv run ruff format .`
- Check formatting (CI-safe):
    - `uv run ruff format --check .`

### Linting (ruff)

- Lint all:
    - `uv run ruff check .`
- Lint with autofix:
    - `uv run ruff check --fix .`
- Lint a single file:
    - `uv run ruff check path/to/file.py`

### Tests (pytest)

- Run all tests:
    - `uv run pytest`
- Run a single test file:
    - `uv run pytest tests/test_parser.py`
- Run a single test (node id):
    - `uv run pytest tests/test_parser.py::test_validate_ifc`
- Run tests matching a substring:
    - `uv run pytest -k validate`
- Rerun last failures:
    - `uv run pytest --lf`

### Type Check (pyright)

- Run type checking:
    - `uv run pyright`

## Code Style Guidelines

### Formatting

- Use `ruff format` and do not hand-format.
- Prefer readability over clever one-liners.

### Imports

- Prefer absolute imports from the package namespace (once `src/` exists).
- Group imports: stdlib, third-party, local.
- Avoid wildcard imports; avoid redundant aliasing.

### Types

- Type all public functions (module-level APIs, pipeline steps, query interfaces).
- Prefer `pathlib.Path` over `str` for paths.
- Use `dataclass` / `TypedDict` for structured payloads; add `pydantic` only if validation is needed.
- Avoid `# type: ignore` unless justified; prefer precise types.

### Naming

- Files/modules: `snake_case.py`
- Functions/vars: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Use domain-consistent names: `global_id`, `storey`, `space`, `element`, `bbox`, `centroid`.

### Errors

- Fail fast on invalid inputs; raise specific exceptions with actionable messages.
- Catch exceptions only to add context; avoid blanket `except Exception` (especially around parsing).
- Include identifiers in errors when available (IFC path, `global_id`, element type).
- Prefer domain-specific exception types for pipeline steps (e.g., `InvalidIfcError`, `ExtractionError`).

### Logging

- Prefer `logging` over `print`.
- Log pipeline stages at INFO: validate -> parse -> extract -> geometry -> SQL -> graph.
- Avoid logging entire IFC objects; log IDs + concise stats.

### Data Contracts (Query Output)

- Prefer returning both a human-readable summary and structured JSON.
- JSON shape (when applicable): `entity_type: str`, `count: int | None`, `global_ids: list[str]`, `properties: dict[str, object]`.

### Testing

- Unit tests: geometry helpers and pure functions.
- Integration tests: parsing a small IFC fixture; deterministic row/node counts.
- For “how many” questions, tests must assert counts from SQL queries.
