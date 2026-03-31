# rag-tag

`rag-tag` is a research toolkit for asking natural-language questions over IFC-based digital twins.

It turns IFC models into queryable runtime artifacts, routes each question to the right retrieval path, and exposes both a CLI agent and a local web viewer for inspecting the model, the graph, and the answer evidence.

## What it does

- Converts IFC models into JSONL element records, SQLite datasets, and a NetworkX graph
- Uses deterministic SQL for counts, lists, and aggregations
- Uses graph tools for spatial, containment, topology, and relationship questions
- Routes natural-language questions between SQL and graph workflows
- Supports local model/profile configuration across Gemini, Cohere, and Databricks-backed providers
- Includes a local web viewer for graph exploration, IFC upload/rebuild, query execution, and 3D model inspection
- Supports Logfire tracing for the main CLI and the viewer

## Architecture at a glance

```text
IFC -> JSONL -> SQLite + NetworkX -> Router -> SQL path or Graph agent
```

- `SQL path` handles deterministic count/list/group/aggregate queries
- `Graph path` handles spatial and topological reasoning over IFC hierarchy and derived graph relationships

## Quick start

### 1. Install dependencies

```bash
uv sync --group dev
```

### 2. Create local runtime config

```bash
cp config.example.yaml config.yaml
cp .env.sample .env
```

- Keep shared local defaults in `config.yaml`
- Keep secrets in `.env` or your shell environment
- Treat `config.example.yaml` as the checked-in reference template

Typical environment variables:

```bash
GEMINI_API_KEY=...
COHERE_API_KEY=...
DATABRICKS_HOST=...
DATABRICKS_TOKEN=...
LOGFIRE_TOKEN=...
```

### 3. Build artifacts from IFC

```bash
uv run rag-tag-generate-ontology-map
uv run rag-tag-ifc-to-jsonl --ifc-file IFC-Files/Building-Architecture.ifc --out-dir output
uv run rag-tag-jsonl-to-sql
uv run rag-tag-jsonl-to-graph
```

### 4. Run the agent

```bash
uv run rag-tag --db output/Building-Architecture.db --graph-dataset Building-Architecture
```

Useful variants:

```bash
uv run rag-tag
uv run rag-tag --tui
uv run rag-tag --input
uv run rag-tag --verbose
uv run rag-tag --trace
```

## Common workflows

### Ask questions in the CLI

Run the standard agent:

```bash
uv run rag-tag --db output/Building-Architecture.db --graph-dataset Building-Architecture
```

Use `--trace` to enable Logfire observability:

```bash
uv run rag-tag --db output/Building-Architecture.db --graph-dataset Building-Architecture --trace
```

### Open the local viewer

The viewer is the easiest way to inspect the graph, the source IFC model, and query results together.

Build from a raw IFC and launch the server:

```bash
uv run rag-tag-viewer --ifc ./IFC-Files/Building-Architecture.ifc
```

Or start from existing generated artifacts:

```bash
uv run rag-tag-viewer --graph-dataset Building-Architecture
uv run rag-tag-viewer --db ./output/Building-Architecture.db
```

Enable tracing in the viewer:

```bash
uv run rag-tag-viewer --ifc ./IFC-Files/Building-Architecture.ifc --trace
```

Viewer defaults:

- host: `127.0.0.1`
- port: `8000`
- graph payload mode: `minimal`

Main pages:

- `http://127.0.0.1:8000/` - graph view
- `http://127.0.0.1:8000/model` - 3D IFC model view
- `http://127.0.0.1:8000/how-to-use` - built-in viewer guide

### Build pipeline directly

Project scripts:

```bash
uv run rag-tag-ifc-to-jsonl
uv run rag-tag-jsonl-to-sql
uv run rag-tag-jsonl-to-graph
uv run rag-tag-refresh-ifc43-rdf
uv run rag-tag-generate-ontology-map
```

For parser-specific details, see `src/rag_tag/parser/README.md`.

## Current retrieval model

`rag-tag` uses a hybrid retrieval pipeline rather than a single "LLM answers everything" flow.

- `Router` decides whether a question should go to SQL or graph execution
- `SQL` is the source of truth for deterministic counts and aggregations
- `Graph agent` handles spatial reasoning, containment, topology, and multi-hop relationship queries
- `Tool outputs` are normalized into a stable `{status, data, error}` envelope for graph workflows

Examples:

- `How many parking spaces are there?` -> SQL
- `What is above room 216?` -> graph
- `Are there any trees outside the building?` -> graph

## Configuration model

Use `config.example.yaml` as the canonical template, then keep your working `config.yaml` local and untracked.

Put in `config.yaml`:

- default router and agent profiles
- router mode and graph orchestration defaults
- named provider wiring
- reusable profiles and experiment groups
- graph build defaults such as pruning and overlap controls

Put in `.env` or shell environment:

- API keys and tokens such as `GEMINI_API_KEY`, `COHERE_API_KEY`, `DATABRICKS_TOKEN`
- provider host or base URL values referenced by `*_env` config keys
- machine-local overrides such as `RAG_TAG_CONFIG`

Config discovery order when no override is provided:

- `config.yaml`
- `config.yml`
- `config.json`

Override config discovery with either:

```bash
uv run rag-tag --config ./config.yaml
RAG_TAG_CONFIG=./config.yaml uv run rag-tag
```

Useful runtime overrides:

```bash
uv run rag-tag --router-profile router-gemini-flash
uv run rag-tag --agent-profile cohere-command-a
uv run rag-tag --agent-profile dbx-claude-sonnet-4-6
ROUTER_MODEL=google-gla:gemini-2.5-flash uv run rag-tag
AGENT_MODEL=cohere:command-a-03-2025 uv run rag-tag
```

## Graph build controls

### Derived edge pruning

`config.example.yaml` now includes:

```yaml
graph_build:
    derived_edge_pruning:
        enabled: true
        exclude_classes:
            - IfcMember
            - IfcPlate
```

This pruning applies only to derived spatial/topology phases. It does not remove:

- nodes
- explicit IFC relationships
- hierarchy edges
- type edges

### Overlap edge emission

Raw `overlaps_xy` emission is configurable:

- `full`
- `threshold`
- `top_k`
- `none`

The checked-in default is `none` so dense models do not emit large raw overlap edge sets unless explicitly requested.

Examples:

```bash
uv run rag-tag-jsonl-to-graph --overlap-xy-mode none
uv run rag-tag-jsonl-to-graph --overlap-xy-mode threshold --overlap-xy-min-ratio 0.20
uv run rag-tag-jsonl-to-graph --overlap-xy-mode top_k --overlap-xy-top-k 5
uv run rag-tag-jsonl-to-graph --overlap-xy-mode full
```

Important note:

- `above` and `below` reasoning can still use internal XY overlap even when raw `overlaps_xy` edges are disabled

## Model providers and profiles

Current repo defaults and supported patterns:

- default router model: `google-gla:gemini-2.5-flash`
- default graph agent baseline: `cohere:command-a-03-2025`
- Databricks-backed profiles are supported through PydanticAI's OpenAI-compatible path

Databricks notes:

- keep `DATABRICKS_HOST` or `DATABRICKS_BASE_URL` in `.env` or shell env
- keep `DATABRICKS_TOKEN` in `.env` or shell env
- do not add `parallel_tool_calls` to Databricks profile settings; `rag-tag` strips it automatically for those profiles

Minimal example local config:

```yaml
defaults:
    router_profile: router-gemini-flash
    agent_profile: dbx-claude-sonnet-4-6
    router_mode: llm

providers:
    databricks:
        type: databricks
        host_env: DATABRICKS_HOST
        token_env: DATABRICKS_TOKEN

profiles:
    router-gemini-flash:
        model: google-gla:gemini-2.5-flash
        settings:
            temperature: 0.0

    cohere-command-a:
        model: cohere:command-a-03-2025
        settings:
            temperature: 0.1
            max_tokens: 1024

    dbx-claude-sonnet-4-6:
        provider: databricks
        model: databricks-claude-sonnet-4-6
        settings:
            temperature: 0.1
            max_tokens: 1024
```

## Tracing and observability

Logfire tracing is supported in both the main CLI and the viewer.

Examples:

```bash
uv run rag-tag --trace
uv run rag-tag-viewer --ifc ./IFC-Files/Building-Architecture.ifc --trace
```

If `LOGFIRE_TOKEN` is set, traces can be sent to Logfire. Without a token, local instrumentation can still be enabled for development.

## Evaluation workflows

Compare graph-agent behavior across configured profiles:

```bash
uv run python scripts/eval_graph_models.py \
  --config ./config.yaml \
  --experiment graph-agent-compare \
  --db ./output/Building-Architecture.db \
  --graph-dataset Building-Architecture \
  --output ./output/graph-model-report.json
```

Compare overlap modes on graph density and question behavior:

```bash
uv run python scripts/eval_overlap_modes.py \
  --jsonl output/BigBuildingBIMModel.jsonl \
  --db output/BigBuildingBIMModel.db \
  --graph-dataset BigBuildingBIMModel \
  --output output/eval_overlap_modes.json
```

## Repository layout

```text
src/rag_tag/
  run_agent.py                 CLI entrypoint
  query_service.py             routing and execution orchestration
  observability.py             Logfire setup
  ifc_sql_tool.py              SQL query helper
  ifc_graph_tool.py            graph query interface
  textual_app.py               Textual TUI
  viewer/                      local web viewer
  router/                      SQL vs graph routing
  agent/                       graph agent and graph tools
  parser/                      IFC -> JSONL/SQLite/graph pipeline

scripts/
  eval_graph_models.py         graph-agent profile comparison
  eval_overlap_modes.py        overlap mode comparison

output/                        generated JSONL, DB, graph, and viewer artifacts
IFC-Files/                     source IFC models
tests/                         regression and behavior coverage
```

## Development and validation

Core checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Useful targeted checks:

```bash
uv run pytest tests/test_viewer_api.py
uv run pytest tests/test_batch34_graph_semantics.py
uv run pytest tests/test_graph_batch1_bounded_outputs.py
```

## Reliability and design notes

- SQL counts and aggregations should come from SQL results, not model synthesis
- Graph outputs are bounded and include truncation metadata where relevant
- Graph adjacency is still heuristic and geometry-driven, not full BIM topology
- The graph backend is currently NetworkX, not Neo4j/Cypher
- If geometry is missing, graph reasoning degrades gracefully rather than inventing structure

## Known limitations

- `adjacent_to` is based on geometry heuristics, not authoritative topology
- raw IFC exports vary a lot by authoring tool, especially for site/exterior elements
- some outside/inside answers still rely on footprint or containment inference
- dense models can still produce expensive graph-agent runs if the query is broad or ambiguous

## Where to look next

- `config.example.yaml` for the full checked-in runtime template
- `src/rag_tag/parser/README.md` for parser-specific pipeline details
- `src/rag_tag/viewer/static/how-to-use.html` for the built-in viewer guide
- PR `#49` for the graph pruning and viewer feature set that shaped the current workflow
