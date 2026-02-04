# rag-tag

Utilities for working with IFC (Industry Foundation Classes) files, with tools to extract tabular data, geometry, and hierarchical graph representations for analysis and visualization.

---

## Features

- **IFC → CSV**: Extract structured element data into CSV for analysis.
- **IFC → Graph (3D)**: Build a hierarchical, typed graph of IFC entities enriched with real geometry and visualize it interactively in 3D.
- **Geometry Extraction**: Compute centroids and axis-aligned bounding boxes for IFC elements using `ifcopenshell`.
- **Interactive Visualization**: Explore IFC structure using a Plotly-based 3D graph (HTML output).

---

## Project Structure

```
rag-tag/
├── IFC-Files/              # Input IFC files (auto-detected)
│   └── Building-Architecture.ifc
├── parser/
│   ├── ifc-to-csv.py       # IFC → CSV exporter
│   ├── ifc_to_graph.py     # IFC → Graph + 3D visualization
│   └── ifc_geometry_parse.py
├── output/
│   ├── Building-Architecture.csv
│   └── ifc_graph.html
├── pyproject.toml
└── README.md
```

---

## Requirements

Core dependencies:

- Python 3.10+
- `ifcopenshell`
- `pandas`
- `numpy`
- `networkx`
- `plotly`

Dev / tooling:

- `ruff`
- `pre-commit`
- `uv`

---

## Formatting & Linting (Ruff)

This repo uses **Ruff** for consistent formatting and basic linting across contributors.

### One-time setup

Install dev tools (Ruff + pre-commit):

```bash
uv sync --group dev
```

Install the git pre-commit hook:

```bash
uv run pre-commit install
```

### Run manually (recommended before pushing)

Format:

```bash
uv run ruff format .
```

Lint (auto-fix safe issues):

```bash
uv run ruff check --fix .
```

Run exactly what pre-commit runs:

```bash
uv run pre-commit run --all-files
```

---

## IFC → CSV

The script in `parser/ifc-to-csv.py` reads `.ifc` files and exports structured CSV output.

### Default behavior

- **Input**: Automatically detects an `IFC-Files/` directory by searching upward from the script location.
- **Output**: Writes CSV files to `output/` at the project root.
- **Schema**: Includes element identifiers, class names, levels, types, and other IFC properties.

Run:

```bash
cd parser
uv run ifc-to-csv.py
```

### Override paths

```bash
uv run ifc_to_csv.py --ifc-dir ./IFC-Files --out-dir ./output
```

---

## IFC → Graph (3D)

The graph pipeline builds a **hierarchical directed graph** from IFC data and enriches nodes with **real geometry** extracted from the model.

### What the graph represents

- **Nodes**:
  - `IfcProject`
  - `IfcBuilding`
  - `IfcBuildingStorey`
  - `IfcTypeObject`
  - Individual IFC elements (`IfcWall`, `IfcDoor`, etc.)

- **Edges**:
  - `aggregates` (Project → Building → Storey)
  - `contained_in` (Storey → Element)
  - `typed_by` (Type → Element)

- **Node properties**:
  - IFC class
  - Original CSV row (full property dict)
  - Geometry (centroid or bounding box)

### Geometry handling

Geometry is extracted using `ifcopenshell.geom`:

- **Centroid**: Mean of mesh vertices (used for node positioning)
- **Bounding box**: Axis-aligned min/max coordinates

If geometry extraction fails, nodes are placed with a small random offset to avoid overlap.

### Default behavior

- **Input**:
  - IFC file auto-detected from `IFC-Files/`
  - CSV loaded from `output/Building-Architecture.csv`
- **Output**:
  - Interactive HTML visualization written to `output/ifc_graph.html`

Run:

```bash
cd parser
uv run ifc_to_graph.py
```

### Output

- `output/ifc_graph.html`
  - Interactive 3D Plotly visualization
  - Hover on nodes to inspect IFC properties
  - Colored by IFC class
---

## Console Hierarchy Inspection (Optional)

A helper is included to print the raw IFC hierarchy directly to the console:

```python
print_ifc_hierarchy(ifc_file)
```

This prints:

```
IfcProject
 └── IfcBuilding
     └── IfcBuildingStorey
         └── IfcElement
```