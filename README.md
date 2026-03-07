# rag-tag

`rag-tag` is a research toolkit for IFC-based digital twins.

It converts IFC models into:

- JSONL element records (schema-aware, geometry-aware)
- flat SQLite tables for deterministic count/list/aggregation queries
- a NetworkX graph for hierarchy + spatial/topological traversal

Natural-language questions are routed to SQL or graph tools via PydanticAI.

## High-level architecture

```
IFC -> JSONL -> SQLite + NetworkX -> Router + Graph Agent
```

- SQL path: deterministic counts/lists/aggregations
- Graph path: spatial and topology reasoning over element relationships

## Active pipeline commands

```bash
uv run rag-tag-ifc-to-jsonl
uv run rag-tag-jsonl-to-sql
uv run rag-tag-jsonl-to-graph
```

Ontology support commands:

```bash
uv run rag-tag-refresh-ifc43-rdf
uv run rag-tag-generate-ontology-map
```

## Quick start

1. Install dependencies

```bash
uv sync --group dev
```

2. Optional: configure model keys (`.env` or shell env)

```bash
GEMINI_API_KEY=...
COHERE_API_KEY=...
```

3. Build artifacts

```bash
uv run rag-tag-generate-ontology-map
uv run rag-tag-ifc-to-jsonl
uv run rag-tag-jsonl-to-sql
uv run rag-tag-jsonl-to-graph
```

4. Run interactive agent

```bash
uv run rag-tag
```

## Query modes

- Default CLI: `uv run rag-tag`
- Textual TUI: `uv run rag-tag --tui`

Useful options:

```bash
uv run rag-tag --db ./output/Building-Architecture.db
uv run rag-tag --graph-dataset Building-Architecture
uv run rag-tag --input
uv run rag-tag --verbose
uv run rag-tag --trace
```

Dataset selection behavior:

- If `--graph-dataset` is set, that JSONL stem is used for graph routing.
- Else, if exactly one DB is selected, that DB stem is used.
- Else, SQL queries can still merge across DBs, but graph queries require an
  explicit dataset when multiple JSONL datasets are present.

## Key repository paths

```
src/rag_tag/
  run_agent.py                 # CLI entrypoint
  textual_app.py               # Textual TUI
  query_service.py             # shared routing/execution orchestration
  ifc_sql_tool.py              # SQL execution helper
  ifc_graph_tool.py            # graph query interface + filters
  router/                      # SQL vs graph routing logic
  agent/                       # PydanticAI graph agent + tools
  parser/
    ifc_to_jsonl.py            # IFC -> JSONL
    jsonl_to_sql.py            # JSONL -> SQLite
    jsonl_to_graph.py          # JSONL -> NetworkX graph
    parse_bsdd_to_map.py       # RDF/registry -> ontology map JSON
    ifc43_schema_registry.py   # IFC class/pset registry
    ifc_geometry_parse.py      # centroid/bbox extraction helpers
```

## JSONL ingestion notes

- Ingestion reads IFC schema and applies schema-aware behavior.
- Property sets are split into `PropertySets.Official` and `PropertySets.Custom`
  using class-specific `ValidPsets` from the ontology map.
- Unknown/unsupported schema families degrade gracefully (no crash):
  properties default to `Custom` and base-class expansion is empty.
- Geometry blocks may include centroid, bounding box, mesh vertices/faces, 2D
  footprint polygon, local placement matrix, and oriented bounding box when IFC
  geometry extraction succeeds.

## SQL and graph contracts

- SQLite schema stays flat for reliable LLM SQL generation.
- Graph nodes include both:
    - `properties` (flat compatibility view)
    - `payload` (full nested JSONL record)

### Canonical graph action contract

- All graph actions return the stable envelope: `{status,data,error}`.
- Canonical action names (allowlist):
  `get_elements_in_storey`, `find_elements_by_class`, `get_adjacent_elements`,
  `get_topology_neighbors`, `get_intersections_3d`, `find_nodes`, `traverse`,
  `spatial_query`, `find_elements_above`, `find_elements_below`,
  `get_element_properties`, `list_property_keys`.
- Required `data` payload fields are stable per action (e.g.,
  `find_nodes -> {class,elements}`, `traverse -> {start,relation,depth,results}`).

### Canonical relation taxonomy + source semantics

- Relation buckets:
  - hierarchy: `aggregates`, `contains`, `contained_in`
  - spatial: `adjacent_to`, `connected_to`
  - topology: `above`, `below`, `overlaps_xy`, `intersects_bbox`,
    `intersects_3d`, `touches_surface`
  - explicit IFC: `hosts`, `hosted_by`, `ifc_connected_to`, `typed_by`,
    `belongs_to_system`, `in_zone`, `classified_as`
- `source` semantics for relation-bearing outputs:
  - `ifc` = explicit IFC relation extracted from model relationships
  - `heuristic` = spatial proximity heuristic
  - `topology` = topology/geometry-derived relation
  - hierarchy edges may omit source (reported as null)

### Property filtering and key discovery

- Graph filtering supports both:
  - flat keys (e.g., `FireRating`)
  - dotted keys (e.g., `Pset_WallCommon.FireRating`)
- Key discovery is exposed via the canonical `list_property_keys` action/tool.
- In `GRAPH_PAYLOAD_MODE=minimal`, dotted key discovery falls back to SQLite
  when a DB path is wired into graph context.

## Linting and checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

## Reliability notes

- SQL merge mode reports partial database failures in `warning.failed_db_paths`
  and `warning.db_errors` rather than silently dropping them.
- `uv run rag-tag --strict-sql` makes SQL count/list queries fail closed when
  any selected database errors.
- Graph queries support explicit IFC `typed_by` relationships when type objects
  are present in the exported JSONL.
