# AGENTS.md

This file is for agentic coding assistants operating in this repository.

## Project Summary

- Domain: BIM / IFC digital twins
- Goal: Natural-language querying over IFC models using a hybrid retrieval pipeline
  - Deterministic aggregations and counts via SQL (no hallucinations)
  - Spatial/topological questions via graph traversal
- Primary pipeline: IFC -> CSV -> SQLite + NetworkX graph + LLM tool interface

## Repo Status Snapshot

- IFC parsing and export works via `src/rag_tag/parser/ifc_to_csv.py`.
- Geometry extraction computes centroids and bounding boxes.
- Graph is currently built in NetworkX with distance-based adjacency.
- SQL schema exists and is wired into the router for counts/lists.
- Project now uses a clean `src/` layout with `rag_tag` package.
- **Router and Graph Agent migrated to PydanticAI** (router complete, graph agent complete).
- Router uses structured output with `google-gla:gemini-2.5-flash` (default).
- Graph Agent uses tool calling with `cohere:command-a-03-2025` (default).
- Observability via Logfire (optional, enabled with `--trace` flag).
- Tool outputs are normalized with a `{status,data,error}` envelope.
- PydanticAI uses `google-gla` for AI Studio and `google-vertex` for Vertex AI (not `google`).
- `COHERE_API_KEY` is automatically mapped to `CO_API_KEY` for Cohere provider compatibility.

## Cursor/Copilot Rules

- No Cursor rules detected (`.cursorrules` / `.cursor/rules/` not present).
- No Copilot rules detected (`.github/copilot-instructions.md` not present).
- If these files are added later, treat them as higher priority and mirror key rules here.

## Skills

- Skills can be found in the `.agents/skills` folder.

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
  - `uv run rag-tag-ifc-to-csv`
- CSV -> SQLite:
  - `uv run rag-tag-csv-to-sql`
- CSV -> Graph (Plotly HTML):
  - `uv run rag-tag-csv-to-graph`
- Run the LLM agent (requires API key):
  - `COHERE_API_KEY=... uv run rag-tag`
  - `GEMINI_API_KEY=... uv run rag-tag`
  - Debug LLM I/O:
    - `COHERE_API_KEY=... uv run rag-tag --input`
  - Show full JSON details below each answer:
    - `uv run rag-tag --verbose`
  - Trace execution (Logfire, requires WRITE token):
    - `LOGFIRE_TOKEN=... uv run rag-tag --trace`
    - Note: LOGFIRE_TOKEN must be a write token (not read token)
    - Read tokens are for the query API only and will cause 401 errors if used for ingestion.
    - Alternative: Run `logfire auth` to authenticate
  - Force router mode:
    - `ROUTER_MODE=rule uv run rag-tag`
    - `ROUTER_MODE=llm GEMINI_API_KEY=... uv run rag-tag`
  - Model overrides (use PydanticAI format: provider:model-name):
    - `AGENT_MODEL=cohere:command-a-03-2025 COHERE_API_KEY=... uv run rag-tag`
    - `ROUTER_MODEL=google-gla:gemini-2.5-flash GEMINI_API_KEY=... uv run rag-tag`
    - `ROUTER_MODEL=google-gla:gemini-3-flash-preview GEMINI_API_KEY=... uv run rag-tag`
    - `ROUTER_MODEL=google-vertex:gemini-2.5-flash GEMINI_API_KEY=... uv run rag-tag` (Vertex AI)
  - Provider overrides (DEPRECATED - use ROUTER_MODEL/AGENT_MODEL instead):
    - `LLM_PROVIDER=gemini GEMINI_API_KEY=... uv run rag-tag`
    - `LLM_PROVIDER=cohere COHERE_API_KEY=... uv run rag-tag`
    - `AGENT_PROVIDER=cohere COHERE_API_KEY=... uv run rag-tag`
    - `ROUTER_PROVIDER=gemini GEMINI_API_KEY=... uv run rag-tag`
  - Use a specific SQLite DB:
    - `COHERE_API_KEY=... uv run rag-tag --db ./output/Building-Architecture.db`
  - Output formatting:
    - Questions and answers are prefixed with `Q:` / `A:` and separated by divider lines.
- Evaluate routing:
  - `uv run python scripts/eval_routing.py --db ./output/Building-Architecture.db`

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
- Use absolute imports from `rag_tag` package (e.g., `from rag_tag.router import ...`).

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

## Project Structure

- `src/rag_tag/`: Main package with all source code
  - `__init__.py`: Package initialization
  - `__main__.py`: Entry point for `python -m rag_tag`
  - `paths.py`: Centralized path discovery utilities
- `run_agent.py`: Main CLI application
- `trace.py`: JSONL tracing utilities (legacy, kept for reference)
- `observability.py`: Logfire integration for PydanticAI
- `ifc_graph_tool.py`: Graph query interface
- `ifc_sql_tool.py`: SQL query interface
- `tui.py`: Terminal UI formatting
- `agent/`: Graph agent implementation
  - `graph_agent.py`: PydanticAI-based graph agent
  - `graph_tools.py`: Typed tools for graph queries
  - `models.py`: Output schemas (GraphAnswer)
- `llm/`: LLM integration
  - `pydantic_ai.py`: Model resolver for PydanticAI
  - `router/`: Query routing logic (rule-based and LLM-based)
  - `parser/`: IFC parsing and conversion pipeline
- `scripts/`: Evaluation and utility scripts
- `output/`: Generated CSV, DB, and HTML files (created at runtime)
- `IFC-Files/`: Source IFC models (not in repo)
