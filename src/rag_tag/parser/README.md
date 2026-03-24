# parser/

Parser modules that transform IFC models into queryable JSONL, SQLite, and
NetworkX graph artifacts.

## Pipeline

```
IFC files  -->  ifc_to_jsonl.py  -->  JSONL  -->  jsonl_to_sql.py    --> SQLite (.db)
                                         \
                                          \-->  jsonl_to_graph.py  --> Graph (+ optional HTML)
```

## Modules

| Module | Input | Output | Purpose |
|---|---|---|---|
| `ifc_to_jsonl.py` | `.ifc` | `.jsonl` | Extract element/type identity, hierarchy, geometry, explicit relationships, psets, and quantities |
| `jsonl_to_sql.py` | `.jsonl` | `.db` | Build normalized SQLite tables for deterministic SQL |
| `jsonl_to_graph.py` | `.jsonl` | `networkx` (+ HTML viz) | Build hierarchy + spatial/topology graph with payload on nodes |
| `parse_bsdd_to_map.py` | optional `.ttl` RDF | `ifc_ontology_map.json` | Build offline ontology map (`BaseClasses`, `ValidPsets`) |
| `ifc43_schema_registry.py` | ifcopenshell schema + optional RDF | in-memory registry | Class normalization and known IFC pset definitions |
| `ifc_geometry_parse.py` | IFC geometry | centroid/bbox helpers | Geometry extraction utilities for parser/graph |
| `sql_schema.py` | none | schema DDL | Canonical SQLite schema and indexes |

## Recommended run order

```bash
# 1) Optional: refresh local bSDD RDF snapshot
uv run rag-tag-refresh-ifc43-rdf

# 2) Generate/update ontology map used by JSONL ingestion
uv run rag-tag-generate-ontology-map

# 3) IFC -> JSONL
uv run rag-tag-ifc-to-jsonl

# 4) JSONL -> SQLite
uv run rag-tag-jsonl-to-sql

# 5) JSONL -> Graph (+ optional visualization)
uv run rag-tag-jsonl-to-graph
```

Single-file conversion example:

```bash
uv run rag-tag-ifc-to-jsonl --ifc-file IFC-Files/Building-Architecture.ifc --out-dir output
```

## Overlap edge modes

`jsonl_to_graph.py` can compute XY footprint overlap internally for vertical
topology while controlling whether raw `overlaps_xy` edges are emitted. Configure
this in `graph_build.overlap_xy`:

- `full`: emit every positive-overlap pair
- `threshold`: emit only pairs whose overlap ratio passes `min_ratio`
- `top_k`: emit each node's strongest `top_k` overlaps using symmetric retention
- `none`: do not emit raw `overlaps_xy` edges

Default behavior:

- `mode: none`

Recommended first-pass values:

- `mode: threshold` with `min_ratio: 0.20`
- `mode: top_k` with `top_k: 5`

Important note:

- `above` / `below` still use positive XY overlap internally even when raw
  `overlaps_xy` edges are reduced or disabled.

Example config snippet:

```yaml
graph_build:
  overlap_xy:
    mode: none
    min_ratio: 0.20
    top_k: 5
```

You can also choose a mode directly from the normal graph-generation CLI:

```bash
uv run rag-tag-jsonl-to-graph --overlap-xy-mode none
uv run rag-tag-jsonl-to-graph --overlap-xy-mode threshold --overlap-xy-min-ratio 0.20
uv run rag-tag-jsonl-to-graph --overlap-xy-mode top_k --overlap-xy-top-k 5
uv run rag-tag-jsonl-to-graph --overlap-xy-mode full
```

To compare overlap modes on graph density and graph-question behavior:

```bash
uv run python scripts/eval_overlap_modes.py \
  --jsonl output/BigBuildingBIMModel.jsonl \
  --db output/BigBuildingBIMModel.db \
  --graph-dataset BigBuildingBIMModel \
  --output output/eval_overlap_modes.json
```

Optional focused comparisons:

```bash
uv run python scripts/eval_overlap_modes.py \
  --jsonl output/BigBuildingBIMModel.jsonl \
  --modes threshold top_k none \
  --threshold-min-ratio 0.20 \
  --top-k 5 \
  --skip-questions
```

## JSONL record shape (high level)

Each line is one element record. Key blocks:

- Top-level identity/class fields: `GlobalId`, `ExpressId`, `IfcType`, `ClassRaw`, `Name`
- `Hierarchy`: `ParentId`, `ParentType`, `Level`, `Path`
- `Geometry`: `Centroid`, `BoundingBox` (`min`/`max`), optional mesh arrays and
  derived geometry helpers (`FootprintPolygon2D`, `LocalPlacementMatrix`,
  `OrientedBoundingBox`)
- `PropertySets`: split into `Official` and `Custom`
- `Quantities`: quantity sets extracted from IFC
- `Relationships`: explicit IFC relations such as `hosts`, `ifc_connected_to`,
  `typed_by`, system/zone/classification assignments

Notes:

- Geometry output includes mesh arrays by default when extraction succeeds; the
  graph builder can later retain full payloads or use minimal payload mode.
- For unsupported schema families or missing ontology data, extraction degrades
  gracefully: properties default to `Custom` and base-class expansion is empty.

## Multi-dataset graph selection

- A single selected DB still implies the matching graph dataset stem.
- When multiple graph datasets are available and no dataset can be inferred,
  graph-query paths require an explicit dataset selection (`--graph-dataset` or
  `--db output/<stem>.db`). SQL queries may still merge across DBs.

## SQL reliability controls

- SQL merge warnings report partial DB failures via `failed_db_paths` and
  per-DB error details.
- `uv run rag-tag --strict-sql` makes merged SQL execution fail closed instead
  of returning a partial result.

## SQLite schema notes

`jsonl_to_sql.py` writes normalized tables:

- `elements`
- `properties` (includes `is_official` 0/1)
- `quantities` (includes `is_official` 0/1)

Design intent:

- Keep SQL flat and deterministic for LLM-generated count/list queries.
- Keep hierarchy/spatial reasoning in graph tools.
- Use parameterized SQL (`?` placeholders) only.

## Migration note

The legacy CSV parser modules have been removed. The active parser stack is
JSONL-only (`ifc_to_jsonl.py`, `jsonl_to_sql.py`, `jsonl_to_graph.py`).
