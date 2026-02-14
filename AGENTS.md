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
- Graph is built in NetworkX with dynamic distance-based adjacency (`max(0.5, median_nn * 1.5)`).
- SQL schema exists and is wired into the router for counts/lists.
- Project uses a clean `src/` layout with `rag_tag` package.
- LLM integration uses PydanticAI (v1.58+) with typed agents and tools.
- Supported providers: Cohere, Gemini (Google GLA), OpenAI -- resolved via env vars.
- Observability via Logfire instrumentation (optional, enabled with `--trace`).
- Tool outputs are normalized with a `{status, data, error}` envelope.
- Router supports rule-based (default), LLM-based, and auto-detect modes.

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
    - `uv run rag-tag-ifc-to-csv`
- CSV -> SQLite:
    - `uv run rag-tag-csv-to-sql`
- CSV -> Graph (Plotly HTML):
    - `uv run rag-tag-csv-to-graph`
- Run the LLM agent (requires API key):
    - `COHERE_API_KEY=... uv run rag-tag`
    - `GEMINI_API_KEY=... uv run rag-tag`
    - `OPENAI_API_KEY=... uv run rag-tag`
    - Verbose output (full JSON details):
        - `uv run rag-tag --verbose`
    - Trace execution (Logfire):
        - `uv run rag-tag --trace`
        - `LOGFIRE_TOKEN=... uv run rag-tag --trace`
    - Force router mode:
        - `ROUTER_MODE=rule uv run rag-tag`
        - `ROUTER_MODE=llm GEMINI_API_KEY=... uv run rag-tag`
    - Provider overrides:
        - `LLM_PROVIDER=gemini GEMINI_API_KEY=... uv run rag-tag`
        - `LLM_PROVIDER=cohere COHERE_API_KEY=... uv run rag-tag`
        - `LLM_PROVIDER=openai OPENAI_API_KEY=... uv run rag-tag`
        - `AGENT_PROVIDER=cohere COHERE_API_KEY=... uv run rag-tag`
        - `ROUTER_PROVIDER=gemini GEMINI_API_KEY=... uv run rag-tag`
    - Model overrides:
        - `AGENT_MODEL=command-a-03-2025 COHERE_API_KEY=... uv run rag-tag`
        - `GEMINI_MODEL=gemini-2.5-flash GEMINI_API_KEY=... uv run rag-tag`
        - `COHERE_MODEL=command-r-08-2024 COHERE_API_KEY=... uv run rag-tag`
        - `OPENAI_MODEL=gpt-4o OPENAI_API_KEY=... uv run rag-tag`
    - Use a specific SQLite DB:
        - `uv run rag-tag --db ./output/Building-Architecture.db`
    - Deprecated flags (still accepted, print warnings):
        - `--input` (formerly debug LLM I/O; use `--trace` and `--verbose` instead)
        - `--trace-path` (Logfire handles trace storage)
- Evaluate routing:
    - `uv run python scripts/eval_routing.py --db ./output/Building-Architecture.db`
    - `uv run python scripts/eval_routing.py --router-mode rule`
    - `uv run python scripts/eval_routing.py --router-mode llm --strict`

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

## Environment Variables

| Variable          | Description                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| `COHERE_API_KEY`  | Cohere API key                                                                                      |
| `GEMINI_API_KEY`  | Gemini (Google GLA) API key                                                                         |
| `OPENAI_API_KEY`  | OpenAI API key                                                                                      |
| `LLM_PROVIDER`    | Force provider for both agent and router (`cohere`, `gemini`, `openai`)                             |
| `AGENT_PROVIDER`  | Override provider for the graph agent only                                                          |
| `ROUTER_PROVIDER` | Override provider for LLM routing only                                                              |
| `AGENT_MODEL`     | Override model name for the graph agent                                                             |
| `COHERE_MODEL`    | Override Cohere model name (default: `command-a-03-2025` for agent, `command-r-08-2024` for router) |
| `GEMINI_MODEL`    | Override Gemini model name (default: `gemini-2.5-flash` for router)                                 |
| `OPENAI_MODEL`    | Override OpenAI model name (default: `gpt-4o`)                                                      |
| `ROUTER_MODE`     | Router strategy: `rule`, `llm`, or unset for auto-detect                                            |
| `LOGFIRE_TOKEN`   | Logfire token for remote tracing (optional)                                                         |

## Architecture

### Query Routing

The router (`src/rag_tag/router/router.py`) dispatches questions to SQL or graph:

1. **Rule-based** (`ROUTER_MODE=rule`, default): Regex patterns classify questions.
    - Spatial keywords (adjacent, near, within, distance, connected, path, between, closest, etc.) -> graph
    - Count keywords (how many, count, number of, total) + IFC class -> SQL count
    - List keywords (list, find, show, which, what are) + IFC class -> SQL list
    - Recognizes 16+ IFC class aliases (wall, door, window, slab, column, beam, stair, space, room, roof, storey, pipe, duct, etc.)
    - Level detection: "ground floor", "basement", "level/storey/story/floor + N"
2. **LLM-based** (`ROUTER_MODE=llm`): PydanticAI agent classifies the question with automatic fallback to rules on failure.
3. **Auto-detect** (unset): Uses LLM routing if an API key is present, otherwise rules.

### Graph Agent

The graph agent (`src/rag_tag/agent/graph_agent.py`) is a PydanticAI agent with:

- `output_type=GraphAnswer` (structured Pydantic output)
- `instructions=SYSTEM_PROMPT` (graph schema and tool descriptions)
- `deps_type=nx.DiGraph` (NetworkX graph passed as dependency)
- `max_steps=6` (maximum reasoning steps)

### Graph Tool Actions

The graph query interface (`src/rag_tag/ifc_graph_tool.py`) exposes 6 actions:

| Action                   | Params                          | Description                                                           |
| ------------------------ | ------------------------------- | --------------------------------------------------------------------- |
| `find_nodes`             | `class`, `property_filters`     | Generic node search with optional class and property filters          |
| `traverse`               | `start`, `relation`, `depth`    | BFS traversal from a start node, optionally filtered by edge relation |
| `spatial_query`          | `near`, `max_distance`, `class` | Find elements within distance of a reference element                  |
| `get_elements_in_storey` | `storey`                        | Get all descendant elements of a storey (excludes containers)         |
| `find_elements_by_class` | `class`                         | Find all elements matching an IFC class                               |
| `get_adjacent_elements`  | `element_id`                    | Get neighbors connected via `adjacent_to` edges with distances        |

Element ID resolution: tries direct node ID, `Element::` prefix, and GlobalId property lookup.
Class normalization: auto-prepends `Ifc` if missing.

### SQL Tool

The SQL query interface (`src/rag_tag/ifc_sql_tool.py`) executes parameterized queries against the `elements` table:

| Intent  | Behavior                                                                                          |
| ------- | ------------------------------------------------------------------------------------------------- |
| `count` | `SELECT COUNT(*) FROM elements` with optional WHERE clauses                                       |
| `list`  | Count + `SELECT express_id, global_id, ifc_class, name, level, type_name` with LIMIT (default 50) |

Filters: `ifc_class = ?` and `LOWER(level) LIKE ?` (with `%` wrapping).

### Graph Model

- **Nodes**: `IfcProject`, `IfcBuilding`, `Storey::{name}`, `Type::{name}`, `Element::{GlobalId}`
- **Edges**: `aggregates`, `contained_in`, `typed_by`, `adjacent_to` (with `distance` attribute)
- **Adjacency threshold**: Dynamic -- `max(0.5, median_nearest_neighbor_distance * 1.5)`, fallback `1.0`
- **Adjacency**: Only between `Element::` nodes, bidirectional, Euclidean distance between centroids

### Output Formatting

The TUI (`src/rag_tag/tui.py`) formats responses:

- Questions printed as `Q: {question}` with route and reason
- Answers printed as `A: {answer}`
- SQL list results include a formatted table (Name, Class, Level, Type columns)
- ANSI colors auto-detected (disabled when piped)
- `--verbose` shows full JSON details

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

- `adjacent_to` edges reflect spatial proximity (centroid distance), not topology.
- `contained_in` edges derived from storey/level matching.
- `typed_by` edges derived from IFC type objects.
- `aggregates` edges represent project/building/storey hierarchy.
- If geometry is missing, degrade gracefully and log the fallback.

### SQL Safety

- Use parameterized queries only (`?` placeholders).
- Do not interpolate user input directly into SQL.

## Known Gaps (Do Not Ignore)

- Graph adjacency uses centroid distance, not true topology.
- No Neo4j/Cypher backend yet; graph lives in NetworkX only.
- No tests are present in the repo.

## Project Structure

- `src/rag_tag/`: Main package with all source code
    - `__init__.py`: Package initialization
    - `__main__.py`: Entry point for `python -m rag_tag`
    - `paths.py`: Centralized path discovery (`find_project_root`, `find_ifc_dir`)
    - `run_agent.py`: Main CLI application (reads stdin, routes, dispatches)
    - `observability.py`: Logfire tracing setup for PydanticAI
    - `ifc_graph_tool.py`: Graph query interface (6 actions, envelope responses)
    - `ifc_sql_tool.py`: SQL query interface (count/list with parameterized queries)
    - `tui.py`: Terminal UI formatting (colors, tables, Q/A output)
    - `agent/`: Graph agent workflow
        - `__init__.py`: Exports `GraphAgent`
        - `graph_agent.py`: PydanticAI graph agent with typed tools
        - `graph_tools.py`: PydanticAI async tool functions for graph queries
        - `models.py`: Pydantic models (`GraphAnswer`, `AnswerEnvelope`, `RouterDecision`)
    - `llm/`: LLM provider resolution
        - `__init__.py`: Exports `resolve_model`
        - `pydantic_ai.py`: Model factory (env vars -> PydanticAI models via provider classes)
    - `router/`: Query routing logic
        - `__init__.py`: Public API exports
        - `router.py`: Main routing dispatcher (rule-based / LLM / auto-detect with fallback)
        - `rules.py`: Rule-based router (regex patterns, IFC class aliases, level detection)
        - `llm.py`: LLM-based router using PydanticAI
        - `llm_models.py`: Pydantic models for LLM router responses (with field validators)
        - `models.py`: Data classes (`RouteDecision`, `SqlRequest`)
    - `parser/`: IFC parsing and conversion pipeline
        - `ifc_to_csv.py`: IFC to CSV export
        - `csv_to_sql.py`: CSV to SQLite conversion
        - `csv_to_graph.py`: CSV to NetworkX graph + Plotly HTML (dynamic adjacency threshold)
        - `ifc_geometry_parse.py`: Geometry extraction (centroids, bounding boxes)
        - `sql_schema.py`: SQL schema definitions
- `scripts/`: Evaluation and utility scripts
    - `eval_routing.py`: Router evaluation harness (15 test questions, `--strict` mode)
- `output/`: Generated CSV, DB, and HTML files (created at runtime)
- `IFC-Files/`: Source IFC models (not in repo)
