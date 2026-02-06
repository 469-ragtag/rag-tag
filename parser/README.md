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
| `ifc_geometry_parse.py` | `.ifc` files | dict | Extract centroids and bounding boxes |
| `csv_to_graph.py` | `.csv` + geometry | `.html` | Build hierarchy graph + 3D visualisation |
| `csv_to_sql.py` | `.csv` | `.db` | Normalised SQLite for aggregation queries |

## Running

All commands use `uv`:

```bash
# IFC → CSV
uv run python parser/ifc_to_csv.py

# CSV → SQLite
uv run python parser/csv_to_sql.py

# CSV → 3D Graph
uv run python parser/csv_to_graph.py
```

Optional flags for `csv_to_sql.py`:

```bash
uv run python parser/csv_to_sql.py --csv-dir ./output --out-dir ./db
```

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
