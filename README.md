# rag-tag

A research-oriented toolkit for **IFC-based digital twins** that combines **geometry-aware graph construction** with **LLM-driven graph reasoning**. The project turns BIM data into a structured knowledge graph and enables natural-language querying via a controlled, multi-step agent (Graph RAG).

---

## What this project is about

Industry Foundation Classes (IFC) models contain rich information about buildings — geometry, hierarchy, types, and properties — but they are difficult to query without BIM expertise or specialized tools.

**rag-tag** bridges that gap by:

1. Converting IFC models into a **typed, hierarchical knowledge graph** enriched with real geometry
2. Exposing that graph through a **safe query interface**
3. Using an **LLM as a planning agent** that reasons over the graph via tool calls (Graph RAG)

The result is a foundation for **LLM-assisted digital twins** that can answer spatial, semantic, and structural questions about buildings.

---

## Key Features

* **IFC → CSV**
  Extract structured IFC element data into tabular form for inspection and downstream processing.

* **IFC → Geometry-Enriched Graph**
  Build a directed, hierarchical graph of IFC entities with centroids and spatial relationships.

* **Spatial Reasoning**
  Automatically infer `adjacent_to` relationships between elements based on 3D proximity.

* **Interactive 3D Visualization**
  Explore the IFC graph in a Plotly-based 3D viewer (HTML output).

* **Graph RAG Agent**
  Query the IFC graph using natural language via an LLM that plans multi-step graph queries instead of hallucinating answers.

---

## Project Structure

```
rag-tag/
├── IFC-Files/                 # Input IFC files (auto-detected)
│   └── Building-Architecture.ifc
├── src/
│   └── rag_tag/
│       ├── __init__.py
│       ├── __main__.py
│       ├── paths.py            # Project/IFC root discovery
│       ├── run_agent.py        # Interactive CLI for LLM-powered graph queries
│       ├── trace.py            # JSONL tracing utilities
│       ├── ifc_graph_tool.py   # Safe graph query interface for the LLM
│       ├── ifc_sql_tool.py     # SQL query helper
│       ├── tui.py              # Terminal formatting helpers
│       ├── agent/
│       │   └── graph_agent.py  # Provider-agnostic graph agent workflow
│       ├── llm/
│       │   ├── provider_registry.py
│       │   └── providers/      # LLM provider adapters (Cohere, Gemini)
│       ├── parser/
│       │   ├── ifc_to_csv.py   # IFC → CSV exporter
│       │   ├── csv_to_graph.py # CSV → IFC graph + geometry + visualization
│       │   ├── csv_to_sql.py   # CSV → SQLite exporter
│       │   ├── ifc_geometry_parse.py  # Geometry extraction utilities
│       │   └── sql_schema.py   # SQLite schema helpers
│       └── router/
│           ├── llm.py          # LLM router integration (provider-agnostic)
│           ├── llm_models.py   # Pydantic router schemas
│           ├── models.py       # Router data models
│           ├── rules.py        # Heuristic router
│           └── router.py       # Router entrypoint + fallback
├── scripts/
│   └── eval_routing.py         # Router evaluation harness
├── output/
│   ├── Building-Architecture.csv
│   └── ifc_graph.html
├── pyproject.toml
└── README.md
```

---

## CLI Commands

* `rag-tag` — interactive agent CLI
* `rag-tag-ifc-to-csv` — IFC to CSV export
* `rag-tag-csv-to-sql` — CSV to SQLite database
* `rag-tag-csv-to-graph` — CSV to graph + Plotly visualization

---

## Requirements

### Core dependencies

* Python 3.14+
* `ifcopenshell`
* `pandas`
* `numpy`
* `networkx`
* `plotly`
* `cohere`
* `google-genai`
* `pydantic`

### Dev / tooling

* `ruff`
* `pre-commit`
* `uv`

---

## Quick Start

```bash
uv sync --group dev
uv run rag-tag-ifc-to-csv
uv run rag-tag-csv-to-sql
uv run rag-tag-csv-to-graph
COHERE_API_KEY=your_key_here uv run rag-tag
```

## Formatting & Linting (Ruff)

This repository uses **Ruff** for formatting and linting.

### One-time setup

```bash
uv sync --group dev
uv run pre-commit install
```

### Manual checks

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pre-commit run --all-files
```

---

## IFC → CSV

The CLI `rag-tag-ifc-to-csv` extracts structured element data from IFC files.

### Default behavior

* **Input**: Auto-detects an `IFC-Files/` directory by searching upward
* **Output**: Writes CSV files to `output/` at the project root
* **Schema**: GlobalId, IFC class, level/storey, type, name, and selected properties

Run:

```bash
uv run rag-tag-ifc-to-csv
```

Override paths:

```bash
uv run rag-tag-ifc-to-csv --ifc-dir ./IFC-Files --out-dir ./output
```

---

## IFC → Graph (3D)

The graph pipeline converts IFC data into a **directed knowledge graph** enriched with geometry and spatial adjacency.

### Graph model

**Nodes**

* `IfcProject`
* `IfcBuilding`
* `IfcBuildingStorey`
* `IfcTypeObject`
* Individual IFC elements (`IfcWall`, `IfcDoor`, `IfcSlab`, ...)

**Edges**

* `aggregates` – project/building/storey hierarchy
* `contained_in` – storey → element
* `typed_by` – type → element
* `adjacent_to` – spatial proximity (distance-based)

**Node attributes**

* `label` – human-readable name
* `class_` – IFC class
* `properties` – full CSV row as a dictionary
* `geometry` – centroid or bounding box

---

### Geometry handling

Geometry is extracted using `ifcopenshell.geom`:

* **Centroid**: Mean of mesh vertices (used for node placement)
* **Bounding box**: Axis-aligned extents (optional)

If geometry is unavailable, nodes are positioned using the centroid of their children to avoid overlap.

---

### Run the graph pipeline

```bash
uv run rag-tag-csv-to-graph
```

Note: The graph builder creates storey nodes from actual `IfcBuildingStorey`
elements in the CSV (not arbitrary values in the `Level` column).

### Output

* `output/ifc_graph.html`

  * Interactive 3D Plotly visualization
  * Hover to inspect IFC properties
  * Color-coded by IFC class

---

## IFC → SQLite

Create a SQLite database for deterministic aggregations:

```bash
uv run rag-tag-csv-to-sql
```

### Output

* `output/*.db`

---

## LLM-Assisted Graph Querying (Graph RAG)

The project includes an **LLM-driven planning agent** that answers natural-language questions by reasoning over the IFC graph.

### Architecture overview

* **LLM provider (Cohere or Gemini)**: Acts as a planner
* **Graph tools**: Controlled Python functions (`src/rag_tag/ifc_graph_tool.py`)
* **Graph agent**: Explicit multi-step workflow (`src/rag_tag/agent/graph_agent.py`)
* **Router**: Rule-based by default, optional LLM routing (`src/rag_tag/router/`)

The LLM:

* does *not* see the graph directly
* must request graph operations via JSON tool calls
* can chain multiple steps before producing a final answer

---

### Example questions

* "What elements are on Level 2?"
* "Which walls are adjacent to this door?"
* "Find doors near the stair core"
* "What elements are close to this column within 2 meters?"

---

### Run the agent

Set your API key:

```bash
COHERE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

Optional: enable LLM routing (otherwise rule-based routing is used):

```bash
ROUTER_MODE=llm GEMINI_API_KEY=your_key_here
```

Then start the interactive agent:

```bash
uv run rag-tag
```

Force router mode (optional):

```bash
ROUTER_MODE=rule uv run rag-tag
ROUTER_MODE=llm GEMINI_API_KEY=your_key_here uv run rag-tag

Select a provider explicitly (optional):

```bash
LLM_PROVIDER=gemini GEMINI_API_KEY=your_key_here uv run rag-tag
LLM_PROVIDER=cohere COHERE_API_KEY=your_key_here uv run rag-tag
AGENT_PROVIDER=cohere COHERE_API_KEY=your_key_here uv run rag-tag
ROUTER_PROVIDER=gemini GEMINI_API_KEY=your_key_here uv run rag-tag
```

Use specific models (optional):

```bash
AGENT_MODEL=command-a-03-2025 COHERE_API_KEY=your_key_here uv run rag-tag
GEMINI_MODEL=gemini-3-flash-preview GEMINI_API_KEY=your_key_here uv run rag-tag
```
```

Use a specific SQLite DB for SQL queries:

```bash
uv run rag-tag --db ./output/Building-Architecture.db
```

To print LLM inputs/outputs (router + agent) to stderr:

```bash
uv run rag-tag --input
```

Trace agent execution (JSONL):

```bash
uv run rag-tag --trace
uv run rag-tag --trace --trace-path ./output/agent_trace.jsonl
```

Questions and answers are printed with `Q:` / `A:` headers and separated by divider lines for easier debugging.

The agent will:

1. Interpret the question
2. Plan one graph query at a time
3. Accumulate results in memory
4. Produce a final, grounded answer

---

## Console IFC Hierarchy Inspection (Optional)

A helper function prints the raw IFC hierarchy directly from the file:

```python
print_ifc_hierarchy(ifc_file)
```

Output structure:

```
IfcProject
 └── IfcBuilding
     └── IfcBuildingStorey
         └── IfcElement
```
