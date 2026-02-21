# parser/

Modules that transform IFC building models into queryable data stores.

## Pipeline

```
IFC files  ──►  ifc_to_csv.py  ──►  CSV  ──┬──►  csv_to_graph.py  ──►  Graph (HTML)
                                            └──►  csv_to_sql.py    ──►  SQLite (.db)
```

| Module | Input | Output | Purpose |
|---|---|---|---|
| `ifc_to_csv.py` | `.ifc` files | `.csv` | Extract elements + properties into flat CSV |
| `ifc43_schema_registry.py` | ifcopenshell schema + optional RDF | in-memory registry | Class hierarchy + known Pset definitions |
| `ifc_geometry_parse.py` | `.ifc` files | dict | Extract centroids and bounding boxes |
| `csv_to_graph.py` | `.csv` + geometry | `.html` | Build hierarchy graph + 3D visualisation |
| `csv_to_sql.py` | `.csv` | `.db` | Normalised SQLite for aggregation queries |

## Running

All commands use `uv`:

```bash
# IFC → CSV (schema-aware, uses embedded Pset definitions)
uv run rag-tag-ifc-to-csv

# IFC → CSV with a local bSDD RDF snapshot for richer hierarchy
uv run rag-tag-ifc-to-csv --bsdd-rdf-path output/metadata/bsdd/ifc43.ttl

# CSV → SQLite
uv run rag-tag-csv-to-sql

# CSV → 3D Graph
uv run rag-tag-csv-to-graph
```

Optional flags for `csv_to_sql.py`:

```bash
uv run rag-tag-csv-to-sql --csv-dir ./output --out-dir ./db
```

## Schema-aware extraction

`ifc_to_csv.py` now uses `ifc43_schema_registry.py` to expand property
coverage beyond what's literally attached in the IFC file.

**The problem it solves:** IFC models often skip attaching standard property
sets (Psets) to elements even though the spec says they should be there.
This meant walls with no `Pset_WallCommon`, beams with no fire rating column,
etc. — different models ended up with completely different CSV columns, making
comparison hard.

**What changes in the CSV:**
- Three class columns instead of one: `ClassRaw` (raw from IFC), `Class`
  (canonical, normalised via hierarchy), `ClassBase` (nearest standard class)
- All standard Pset/Qto columns for each element type are always present,
  even if empty — so the schema is consistent across models
- Unknown/custom Psets from the model are still kept as-is

**Column count comparison (our test models):**

| Model | Before | After |
|---|---|---|
| Building-Architecture | 12 property columns | 210 property columns |
| Building-Structural | 7 property columns | 210 property columns |

## Refreshing the bSDD RDF snapshot

The registry ships with a built-in Pset dictionary derived from the IFC 4.3
spec.  For an even richer class hierarchy you can download the official
IFC OWL ontology from buildingSMART and point the parser at it.

```bash
# download and pin the snapshot
uv run rag-tag-refresh-ifc43-rdf

# custom URL or output path
uv run rag-tag-refresh-ifc43-rdf --url <url> --out output/metadata/bsdd/ifc43.ttl
```

The snapshot is saved to `output/metadata/bsdd/ifc43.ttl` by default.
A sidecar `ifc43.json` is written alongside it with the download URL,
timestamp and SHA256 hash so you always know exactly which version you have.

You can also override the path at runtime:
```bash
# via CLI flag
uv run rag-tag-ifc-to-csv --bsdd-rdf-path /path/to/ifc43.ttl

# via environment variable
export BSDD_IFC43_RDF_PATH=/path/to/ifc43.ttl
uv run rag-tag-ifc-to-csv
```

If the snapshot is missing the parser falls back to the embedded schema
dictionary — nothing crashes, you just get slightly less hierarchy detail.

## SQL Schema

`csv_to_sql.py` creates a flat, non-hierarchical SQLite database with three tables:

```
┌─────────────┐       ┌──────────────┐       ┌──────────────┐
│  elements   │◄──FK──│  properties  │       │  quantities  │
│             │◄──FK──│              │       │              │
│ express_id  │       │ element_id   │       │ element_id ──┼──FK──► elements
│ global_id   │       │ pset_name    │       │ qto_name     │
│ ifc_class   │       │ property_name│       │ quantity_name│
│ name        │       │ value (TEXT) │       │ value (REAL) │
│ level       │       └──────────────┘       └──────────────┘
│ type_name   │
│ ...         │
└─────────────┘
```

### Design decisions

- **No hierarchy tables.** Project → Site → Building → Storey relationships
  are handled by the graph-RAG side.  SQL is for deterministic aggregations only.
- **`level` is plain text**, not a foreign key.  Keeps queries simple
  (`WHERE level = '00 groundfloor'`) and avoids coupling to graph structure.
- **Properties and quantities are normalised** from the wide-format CSV columns
  (`Pset_SlabCommon.FireRating` → row in `properties`).  This makes them
  filterable and joinable without knowing column names up front.

### Future considerations

- Add hierarchy tables (`ifc_project`, `ifc_building_storey`, etc.) if
  the SQL side needs structural queries.
- Add `source_file` column to `elements` for multi-model support.

### Example queries

```sql
-- Count walls
SELECT COUNT(*) FROM elements WHERE ifc_class = 'IfcWall';

-- Total slab volume
SELECT SUM(q.value) FROM quantities q
  JOIN elements e ON q.element_id = e.express_id
 WHERE e.ifc_class = 'IfcSlab' AND q.quantity_name = 'NetVolume';

-- Elements on the ground floor
SELECT ifc_class, name FROM elements WHERE level = '00 groundfloor';

-- Fire ratings
SELECT e.name, p.value FROM properties p
  JOIN elements e ON p.element_id = e.express_id
 WHERE p.property_name = 'FireRating';
```

## Security

All SQL in `csv_to_sql.py` uses parameterised queries (`?` placeholders).
No user data is ever interpolated into SQL strings.
