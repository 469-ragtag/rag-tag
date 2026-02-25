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
| `ifc_to_jsonl.py` | `.ifc` | `.jsonl` | Extract element identity, hierarchy, geometry, psets, and quantities |
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

## JSONL record shape (high level)

Each line is one element record. Key blocks:

- Top-level identity/class fields: `GlobalId`, `ExpressId`, `IfcType`, `ClassRaw`, `Name`
- `Hierarchy`: `ParentId`, `ParentType`, `Level`, `Path`
- `Geometry`: `Centroid`, `BoundingBox` (`min`/`max`)
- `PropertySets`: split into `Official` and `Custom`
- `Quantities`: quantity sets extracted from IFC

Notes:

- Geometry stores only derived centroid/bbox, never raw mesh arrays.
- For unsupported schema families or missing ontology data, extraction degrades
  gracefully: properties default to `Custom` and base-class expansion is empty.

## SQLite schema notes

`jsonl_to_sql.py` writes normalized tables:

- `elements`
- `properties` (includes `is_official` 0/1)
- `quantities` (includes `is_official` 0/1)

Design intent:

- Keep SQL flat and deterministic for LLM-generated count/list queries.
- Keep hierarchy/spatial reasoning in graph tools.
- Use parameterized SQL (`?` placeholders) only.

## Deprecated CSV pipeline

Legacy CSV modules are retained for reference only:

- `_deprecated_ifc_to_csv.py`
- `_deprecated_csv_to_sql.py`
- `_deprecated_csv_to_graph.py`

They are not the active pipeline and are not exposed via current CLI script
entry points.
