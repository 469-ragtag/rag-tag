# rag-tag

A research-oriented toolkit for **IFC-based digital twins** that combines **geometry-aware graph construction** with **LLM-driven graph reasoning**. The project turns BIM data into a structured knowledge graph and enables natural-language querying via a controlled, multi-step agent (Graph RAG).

---

## What this project is about

Industry Foundation Classes (IFC) models contain rich information about buildings -- geometry, hierarchy, types, and properties -- but they are difficult to query without BIM expertise or specialized tools.

**rag-tag** bridges that gap by:

1. Converting IFC models into a **typed, hierarchical knowledge graph** enriched with real geometry
2. Exposing that graph through a **safe query interface** (controlled tool calls, not raw graph access)
3. Routing simple count/list questions to **deterministic SQL** (no hallucinations)
4. Using an **LLM as a planning agent** that reasons over the graph via typed tool calls (Graph RAG)

The result is a foundation for **LLM-assisted digital twins** that can answer spatial, semantic, and structural questions about buildings.

---

## Key Features

- **IFC to CSV**
  Extract structured IFC element data into tabular form for inspection and downstream processing.

- **IFC to Geometry-Enriched Graph**
  Build a directed, hierarchical graph of IFC entities with centroids and spatial relationships.

- **Spatial Reasoning**
  Automatically infer `adjacent_to` relationships between elements based on 3D proximity (dynamic threshold: `max(0.5, median_nearest_neighbor * 1.5)`).

- **Interactive 3D Visualization**
  Explore the IFC graph in a Plotly-based 3D viewer (HTML output), color-coded by IFC class.

- **Hybrid Query Routing**
  Rule-based router sends count/list queries to SQL; spatial/topological queries go to the graph agent. Optional LLM-based routing available.

- **Graph RAG Agent**
  Query the IFC graph using natural language via a PydanticAI agent that plans multi-step graph queries instead of hallucinating answers.

- **Observability**
  Optional Logfire instrumentation traces agent execution end-to-end.

---

## Project Structure

```
rag-tag/
├── IFC-Files/                 # Input IFC files (auto-detected)
│   └── Building-Architecture.ifc
├── src/
│   └── rag_tag/
│       ├── __init__.py
│       ├── __main__.py        # Entry point for python -m rag_tag
│       ├── paths.py           # Project root and IFC directory discovery
│       ├── run_agent.py       # Interactive CLI for LLM-powered queries
│       ├── observability.py   # Logfire instrumentation setup
│       ├── ifc_graph_tool.py  # Graph query interface (6 actions)
│       ├── ifc_sql_tool.py    # SQL query interface (count/list)
│       ├── tui.py             # Terminal formatting (color, tables)
│       ├── agent/
│       │   ├── graph_agent.py # PydanticAI graph agent
│       │   ├── graph_tools.py # Typed async tools for graph queries
│       │   └── models.py      # Pydantic output models
│       ├── llm/
│       │   └── pydantic_ai.py # Model factory and provider resolution
│       ├── parser/
│       │   ├── ifc_to_csv.py  # IFC to CSV exporter
│       │   ├── csv_to_graph.py# CSV to graph + geometry + visualization
│       │   ├── csv_to_sql.py  # CSV to SQLite exporter
│       │   ├── ifc_geometry_parse.py  # Geometry extraction utilities
│       │   └── sql_schema.py  # SQLite schema helpers
│       └── router/
│           ├── router.py      # Router entrypoint + auto-detect + fallback
│           ├── rules.py       # Rule-based router (regex patterns)
│           ├── llm.py         # LLM-based router via PydanticAI
│           ├── llm_models.py  # Pydantic router response schemas
│           └── models.py      # RouteDecision and SqlRequest models
├── scripts/
│   └── eval_routing.py        # Router evaluation harness
├── output/                    # Generated CSV, DB, HTML (created at runtime)
├── pyproject.toml
├── AGENTS.md
└── README.md
```

---

## Requirements

### Core dependencies

- Python 3.14+
- `ifcopenshell` -- IFC parsing and geometry extraction
- `pandas` -- tabular data processing
- `numpy` -- numerical operations
- `networkx` -- knowledge graph construction
- `plotly` -- interactive 3D visualization
- `pydantic-ai` -- LLM agent framework (Cohere, Gemini, OpenAI)
- `pydantic` -- data validation and structured outputs
- `logfire` -- observability and agent tracing
- `nest-asyncio` -- async compatibility for sync wrappers
- `python-dotenv` -- environment variable loading

### Dev / tooling

- `uv` -- package manager and runner
- `ruff` -- formatting and linting
- `pre-commit` -- git hooks

---

## Quick Start

```bash
# 1. Install dependencies
uv sync --group dev

# 2. Parse IFC files into CSV
uv run rag-tag-ifc-to-csv

# 3. Build SQLite database (for count/list queries)
uv run rag-tag-csv-to-sql

# 4. Build knowledge graph + 3D visualization
uv run rag-tag-csv-to-graph

# 5. Start the interactive agent (choose one provider)
COHERE_API_KEY=your_key uv run rag-tag
GEMINI_API_KEY=your_key uv run rag-tag
OPENAI_API_KEY=your_key uv run rag-tag
```

---

## CLI Commands

| Command                | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `rag-tag`              | Interactive agent CLI (reads questions from stdin) |
| `rag-tag-ifc-to-csv`   | IFC to CSV export                                  |
| `rag-tag-csv-to-sql`   | CSV to SQLite database                             |
| `rag-tag-csv-to-graph` | CSV to graph + Plotly 3D visualization             |

### Agent CLI Flags

| Flag               | Description                                                           |
| ------------------ | --------------------------------------------------------------------- |
| `--verbose` / `-v` | Show full JSON details below each answer                              |
| `--db PATH`        | Path to SQLite database (default: newest `.db` in `output/` or `db/`) |
| `--trace`          | Enable Logfire tracing of agent execution                             |
| `--input`          | Deprecated. Use `--trace` and `--verbose` instead                     |
| `--trace-path`     | Deprecated. Logfire handles trace storage                             |

---

## Formatting and Linting

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

## IFC to CSV

The CLI `rag-tag-ifc-to-csv` extracts structured element data from IFC files.

### Default behavior

- **Input**: Auto-detects an `IFC-Files/` directory by searching upward from the working directory
- **Output**: Writes CSV files to `output/` at the project root
- **Schema**: GlobalId, IFC class, level/storey, type, name, and selected properties

```bash
uv run rag-tag-ifc-to-csv
```

Override paths:

```bash
uv run rag-tag-ifc-to-csv --ifc-dir ./IFC-Files --out-dir ./output
```

---

## IFC to Graph (3D)

The graph pipeline converts IFC data into a **directed knowledge graph** enriched with geometry and spatial adjacency.

### Graph model

**Nodes**

| Node Type           | ID Format             | Description                                         |
| ------------------- | --------------------- | --------------------------------------------------- |
| `IfcProject`        | `IfcProject`          | Root project node                                   |
| `IfcBuilding`       | `IfcBuilding`         | Building container                                  |
| `IfcBuildingStorey` | `Storey::{name}`      | Floor/storey (from actual `IfcBuildingStorey` rows) |
| `IfcTypeObject`     | `Type::{name}`        | Type definitions                                    |
| Elements            | `Element::{GlobalId}` | Individual IFC elements (walls, doors, slabs, etc.) |

**Edges**

| Relation       | Meaning                                                            |
| -------------- | ------------------------------------------------------------------ |
| `aggregates`   | Hierarchical: Project to Building, Building to Storey              |
| `contained_in` | Storey contains element                                            |
| `typed_by`     | Type object types element                                          |
| `adjacent_to`  | Spatial proximity within threshold (includes `distance` attribute) |

**Node attributes**

- `label` -- human-readable name
- `class_` -- IFC class
- `properties` -- full CSV row as a dictionary
- `geometry` -- centroid or bounding box

### Geometry handling

Geometry is extracted using `ifcopenshell.geom`:

- **Centroid**: Mean of mesh vertices (used for node placement)
- **Bounding box**: Axis-aligned extents (optional)

If geometry is unavailable, nodes are positioned using the centroid of their children to avoid overlap.

### Spatial adjacency

The adjacency threshold is computed dynamically:

1. Compute the nearest-neighbor distance for each element with geometry
2. Take the **median** of all nearest-neighbor distances
3. Multiply by **1.5**
4. Apply a floor of **0.5** (minimum threshold)
5. Fall back to **1.0** if fewer than 2 positions exist

Adjacency edges are only created between `Element::` nodes (excludes containers and types). Edges are bidirectional and include a `distance` attribute.

### Run the graph pipeline

```bash
uv run rag-tag-csv-to-graph
```

### Output

- `output/ifc_graph.html` -- Interactive 3D Plotly visualization
    - Hover to inspect IFC properties
    - Color-coded by IFC class (purple=Project, blue=Building, orange=Storey, red=Type, green=elements)

---

## IFC to SQLite

Create a SQLite database for deterministic aggregations:

```bash
uv run rag-tag-csv-to-sql
```

### SQL table schema

The `elements` table has columns: `express_id`, `global_id`, `ifc_class`, `name`, `level`, `type_name`.

### Output

- `output/*.db`

---

## LLM-Assisted Graph Querying (Graph RAG)

The project includes an **LLM-driven planning agent** that answers natural-language questions by reasoning over the IFC graph using **PydanticAI**.

### Architecture overview

```
User question
      |
      v
  [Router] ---> SQL route ---> SQLite (deterministic counts/lists)
      |
      v
  Graph route
      |
      v
  [PydanticAI Agent] ---> typed tool calls ---> [Graph Query Interface]
      |                                               |
      v                                               v
  Structured answer (GraphAnswer)              NetworkX graph
```

- **Router**: Rule-based by default (regex patterns), optional LLM routing via `ROUTER_MODE=llm`
- **PydanticAI agent**: LLM planner with typed tools (Cohere, Gemini, or OpenAI)
- **Graph tools**: 6 controlled async functions (`find_nodes`, `traverse`, `spatial_query`, `get_elements_in_storey`, `find_elements_by_class`, `get_adjacent_elements`)
- **Logfire observability**: Optional end-to-end tracing

The LLM:

- does _not_ see the graph directly
- must request graph operations via typed tool calls
- can chain multiple steps before producing a final answer (max 6 steps)
- produces structured outputs validated by Pydantic

### Query routing

| `ROUTER_MODE`    | Behavior                                                                 |
| ---------------- | ------------------------------------------------------------------------ |
| `rule` (default) | Regex-based: spatial keywords to graph, count/list with IFC class to SQL |
| `llm`            | LLM-based classification with automatic fallback to rules on failure     |
| Unset            | Auto-detects: uses LLM if API key is available, otherwise rules          |

### Example questions

- "How many doors are on Level 2?" (SQL route)
- "List all windows on the ground floor" (SQL route)
- "Which rooms are adjacent to the kitchen?" (graph route)
- "Find doors near the stair core" (graph route)
- "What elements are close to this column within 2 meters?" (graph route)

### Run the agent

Set your API key (choose one):

```bash
export COHERE_API_KEY=your_key
export GEMINI_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

Start the interactive agent:

```bash
uv run rag-tag
```

### Environment variables

| Variable          | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `COHERE_API_KEY`  | Cohere API key                                                          |
| `GEMINI_API_KEY`  | Gemini (Google GLA) API key                                             |
| `OPENAI_API_KEY`  | OpenAI API key                                                          |
| `LLM_PROVIDER`    | Force provider for both agent and router (`cohere`, `gemini`, `openai`) |
| `AGENT_PROVIDER`  | Override provider for the graph agent only                              |
| `ROUTER_PROVIDER` | Override provider for LLM routing only                                  |
| `AGENT_MODEL`     | Override model name for the graph agent                                 |
| `COHERE_MODEL`    | Override Cohere model name                                              |
| `GEMINI_MODEL`    | Override Gemini model name                                              |
| `OPENAI_MODEL`    | Override OpenAI model name                                              |
| `ROUTER_MODE`     | Router strategy: `rule`, `llm`, or auto-detect                          |
| `LOGFIRE_TOKEN`   | Logfire token for remote tracing (optional)                             |

### Provider and model examples

```bash
# Explicit provider selection
LLM_PROVIDER=gemini GEMINI_API_KEY=your_key uv run rag-tag
LLM_PROVIDER=cohere COHERE_API_KEY=your_key uv run rag-tag
LLM_PROVIDER=openai OPENAI_API_KEY=your_key uv run rag-tag

# Different providers for agent and router
AGENT_PROVIDER=cohere ROUTER_PROVIDER=gemini \
  COHERE_API_KEY=your_key GEMINI_API_KEY=your_key uv run rag-tag

# Model overrides
AGENT_MODEL=command-a-03-2025 COHERE_API_KEY=your_key uv run rag-tag
GEMINI_MODEL=gemini-2.0-flash-exp GEMINI_API_KEY=your_key uv run rag-tag
OPENAI_MODEL=gpt-4o OPENAI_API_KEY=your_key uv run rag-tag

# Use a specific SQLite DB
uv run rag-tag --db ./output/Building-Architecture.db

# Enable Logfire tracing
uv run rag-tag --trace
LOGFIRE_TOKEN=your_token uv run rag-tag --trace
```

### Output format

Questions and answers are printed with `Q:` / `A:` headers. SQL list results include a formatted table with columns: Name, Class, Level, Type. Use `--verbose` for full JSON details.

---

## Router Evaluation

Evaluate routing accuracy against a hardcoded test suite of 15 questions:

```bash
uv run python scripts/eval_routing.py --db ./output/Building-Architecture.db
uv run python scripts/eval_routing.py --router-mode rule
uv run python scripts/eval_routing.py --router-mode llm --strict
```

The `--strict` flag exits with a non-zero code if any route mismatches the expected route.

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
