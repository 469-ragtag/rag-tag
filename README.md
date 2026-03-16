# rag-tag

`rag-tag` is a research toolkit for natural-language querying over IFC/BIM digital twins.

It exists because BIM questions usually split into two different kinds of work:

- counts, lists, and aggregations need deterministic answers
- spatial and topological questions need graph traversal over model relationships

`rag-tag` handles those jobs differently on purpose. It routes count/list questions to SQLite, and routes spatial or topology questions to a graph runtime backed by either NetworkX or Neo4j.

## 1. Project overview and why it exists

The project turns IFC models into queryable runtime artifacts, then lets a router choose the right execution path for each question.

- SQL path: deterministic counts, lists, and aggregations
- Graph path: adjacency, containment, connectivity, and other spatial/topological reasoning

## 2. Architecture / pipeline at a glance

```text
IFC -> JSONL -> SQLite + graph runtime -> router -> SQL executor or graph agent
```

More concretely:

1. `rag-tag-ifc-to-jsonl` reads IFC files and exports one JSONL record per element.
2. `rag-tag-jsonl-to-sql` builds flat SQLite databases for reliable SQL generation.
3. Graph data is built from the same JSONL records:
    - in memory with NetworkX at app runtime, or
    - imported into Neo4j for a database-backed graph runtime.
4. `rag-tag` routes each question to SQL or graph execution.

The current graph backends are:

- `networkx`
- `neo4j`

## 3. Key features and backend choices

- Hybrid retrieval: SQL for exact counts/lists, graph for spatial reasoning
- IFC-aware export: identity, hierarchy, property sets, quantities, and geometry
- Two graph runtime backends: NetworkX and Neo4j
- Checked-in config system with `config.yaml`, `config.yml`, or `config.json`
- Secrets kept out of config and stored in `.env`
- CLI and Textual TUI entrypoints through the same `rag-tag` command
- Evaluation scripts for routing and graph-agent comparisons

Backend selection is controlled by:

- `defaults.graph_backend` in `config.yaml`, or
- a one-off shell override such as `GRAPH_BACKEND=neo4j`

## 4. Prerequisites

- Python `>=3.14`
- `uv`
- IFC files in `IFC-Files/`, or explicit file paths passed to the conversion CLI
- Optional: Docker Engine + Docker Compose for local Neo4j
- Optional: model/provider credentials in `.env` depending on the profiles you use

Install dev dependencies:

```bash
uv sync --group dev
```

## 5. Quick start

If you only need the shortest path to a working local run, choose one of these:

- NetworkX only: build JSONL + SQLite, then run `rag-tag`
- Neo4j manually: build JSONL + SQLite, import into Neo4j, then run `rag-tag`
- Neo4j one-command flow: use `./scripts/run_neo4j_workflow.sh`

### 5.1 Copy the example config

Use the checked-in example as your starting point:

```bash
cp config.example.yaml config.yaml
cp .env.sample .env
```

`config.example.yaml` is the recommended place for shared defaults. Keep secrets in `.env`.

For a simple local setup, make sure `config.yaml` uses a local graph-agent profile and the NetworkX backend:

```yaml
defaults:
    router_profile: router-gemini-flash
    agent_profile: cohere-command-a
    router_mode: llm
    graph_backend: networkx
```

Example `.env` values:

```bash
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
LOGFIRE_TOKEN=your_write_logfire_token
```

### 5.2 Build the main artifacts

Convert IFC files in `IFC-Files/` to JSONL:

```bash
uv run rag-tag-ifc-to-jsonl
```

Build SQLite databases from the generated JSONL files:

```bash
uv run rag-tag-jsonl-to-sql
```

Optional: build the NetworkX graph and HTML visualization for graph inspection:

```bash
uv run rag-tag-jsonl-to-graph
```

`rag-tag-jsonl-to-graph` is optional for normal app runtime. The app can build the in-memory NetworkX graph directly from JSONL when needed. This command is mainly useful for graph build checks and visualization.

### 5.3 Run the app

Launch the TUI against one dataset:

```bash
uv run rag-tag --tui --db ./output/Building-Architecture.db --graph-dataset Building-Architecture
```

Add `--trace` if you want Logfire observability for that session.

Run the basic CLI instead of the TUI:

```bash
uv run rag-tag --db ./output/Building-Architecture.db --graph-dataset Building-Architecture
```

### 5.4 Optional ontology helper commands

Refresh the IFC4.3 RDF snapshot:

```bash
uv run rag-tag-refresh-ifc43-rdf
```

Generate the ontology map used during IFC export:

```bash
uv run rag-tag-generate-ontology-map
```

## 6. Configuration

### `config.example.yaml` and `.env`

Use `config.example.yaml` as the starting point for:

- shared defaults
- provider settings
- named router and agent profiles
- experiment groups for evaluation scripts

Keep secrets and machine-local values in `.env`, for example:

- `GEMINI_API_KEY`
- `COHERE_API_KEY`
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `LOGFIRE_TOKEN`

### Config discovery order

`rag-tag` looks for config files in this order:

1. `config.yaml`
2. `config.yml`
3. `config.json`

Override that selection for one run:

```bash
uv run rag-tag --config ./config.yaml
```

Or via environment variable:

```bash
RAG_TAG_CONFIG=./config.yaml uv run rag-tag
```

### What belongs where

- `config.yaml`: shared defaults you want to keep with the project
- `.env`: secrets and machine-local values
- CLI flags: per-run knobs such as `--tui`, `--db`, `--graph-dataset`, and `--trace`

### Backend switching

Set the default backend in config:

```yaml
defaults:
    graph_backend: networkx
```

Or switch for one run without editing config:

```bash
GRAPH_BACKEND=neo4j uv run rag-tag --tui --db ./output/Building-Architecture.db --graph-dataset Building-Architecture
```

### Dataset selection and `--graph-dataset`

`--graph-dataset` selects the JSONL dataset stem used by graph queries.

Why it matters:

- in single-dataset runs, `rag-tag` can often infer the graph dataset from `--db`
- in multi-dataset setups, graph queries need a clear graph scope
- Neo4j and NetworkX both use that dataset selection to avoid querying the wrong model

Resolution order is:

1. explicit `--graph-dataset`
2. the stem of the selected `--db` file
3. no dataset selected

If multiple JSONL datasets are present and you do not set `--graph-dataset` or a matching `--db`, graph queries will ask for an explicit dataset.

## 7. Running with NetworkX

NetworkX is the default local graph runtime.

### Build artifacts

```bash
uv run rag-tag-ifc-to-jsonl
uv run rag-tag-jsonl-to-sql
```

Optional graph build and visualization check:

```bash
uv run rag-tag-jsonl-to-graph --no-viz
```

### Run with the NetworkX backend

Using config:

```yaml
defaults:
    graph_backend: networkx
```

Then run:

```bash
uv run rag-tag --tui --db ./output/Building-Architecture.db --graph-dataset Building-Architecture
```

Or force NetworkX for one run:

```bash
GRAPH_BACKEND=networkx uv run rag-tag --tui --db ./output/Building-Architecture.db --graph-dataset Building-Architecture
```

## 8. Running with Neo4j manually

Use this mode when you want Neo4j as the graph runtime instead of NetworkX.

If you already have Neo4j running somewhere else, you can skip the Docker Compose section and just set the connection credentials.

Important: the app still needs the SQLite database path when you use Neo4j.

- SQL-routed questions still execute against SQLite
- some graph property lookups are enriched from the selected SQLite DB path

So the normal Neo4j flow is:

1. build JSONL
2. build SQLite
3. import the chosen dataset into Neo4j
4. run `rag-tag` with `GRAPH_BACKEND=neo4j` or `defaults.graph_backend: neo4j`

### Set Neo4j credentials

Put these in `.env` or export them in your shell:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=ragtag-dev-password
NEO4J_DATABASE=neo4j
```

### Build JSONL and SQLite

```bash
uv run rag-tag-ifc-to-jsonl
uv run rag-tag-jsonl-to-sql
```

### Import one dataset into Neo4j

```bash
uv run rag-tag-jsonl-to-neo4j --jsonl-dir ./output --dataset Building-Architecture
```

Import all JSONL datasets in `output/` instead:

```bash
uv run rag-tag-jsonl-to-neo4j --jsonl-dir ./output
```

`rag-tag-jsonl-to-neo4j` rebuilds the canonical graph from JSONL, then truncates and re-inserts Neo4j data on import.

### Run the app with Neo4j

One-off shell override:

```bash
GRAPH_BACKEND=neo4j uv run rag-tag --tui --db ./output/Building-Architecture.db --graph-dataset Building-Architecture
```

Or set it in config:

```yaml
defaults:
    graph_backend: neo4j
```

Then run the usual command:

```bash
uv run rag-tag --tui --db ./output/Building-Architecture.db --graph-dataset Building-Architecture
```

## 9. Running Neo4j with Docker Compose

The repository includes `docker-compose.neo4j.yml` for a local Neo4j instance.

Start the service:

```bash
docker compose -f docker-compose.neo4j.yml up -d
```

The compose file defaults to:

- username: `neo4j`
- password: `ragtag-dev-password`

Neo4j Browser is available at `http://localhost:7474`.

You can override those values through environment variables such as `NEO4J_USERNAME` and `NEO4J_PASSWORD`.

After the container is running, use the manual Neo4j flow:

```bash
uv run rag-tag-ifc-to-jsonl
uv run rag-tag-jsonl-to-sql
uv run rag-tag-jsonl-to-neo4j --jsonl-dir ./output --dataset Building-Architecture
GRAPH_BACKEND=neo4j uv run rag-tag --tui --db ./output/Building-Architecture.db --graph-dataset Building-Architecture
```

Stop the local Neo4j service when you are done:

```bash
docker compose -f docker-compose.neo4j.yml down
```

Remove the persisted Neo4j volumes too:

```bash
docker compose -f docker-compose.neo4j.yml down -v
```

## 10. One-command Neo4j workflow script

The repo also includes `scripts/run_neo4j_workflow.sh`.

What it does:

1. starts local Neo4j with `docker-compose.neo4j.yml`
2. builds JSONL artifacts
3. builds SQLite artifacts
4. imports the selected dataset into Neo4j
5. launches `rag-tag` in TUI mode with `GRAPH_BACKEND=neo4j`

Run the default dataset (`Building-Architecture`):

```bash
./scripts/run_neo4j_workflow.sh
```

Run a specific dataset:

```bash
./scripts/run_neo4j_workflow.sh Building-Architecture
```

Pass extra `rag-tag` arguments after the dataset name:

```bash
./scripts/run_neo4j_workflow.sh Building-Architecture --agent-profile cohere-command-a
```

Notes:

- the script reads `.env` if present
- it defaults Neo4j to `bolt://localhost:7687`, `neo4j`, and `ragtag-dev-password`
- it launches the TUI with `--db ./output/<dataset>.db --graph-dataset <dataset>`
- it adds `--trace` by default unless you set `RAG_TAG_TRACE=0`
- it does not stop Neo4j when you exit the app
- it still expects your normal model/provider credentials and `config.yaml` defaults to be set correctly

Stop the local service after use:

```bash
docker compose -f docker-compose.neo4j.yml down
```

## 11. Query modes / common CLI examples

### Main app

Start the stdin-based CLI:

```bash
uv run rag-tag
```

Start the Textual TUI:

```bash
uv run rag-tag --tui
```

Run against a specific SQLite DB:

```bash
uv run rag-tag --db ./output/Building-Architecture.db
```

Pin the graph dataset explicitly:

```bash
uv run rag-tag --graph-dataset Building-Architecture
```

Enable Logfire tracing:

```bash
uv run rag-tag --trace
```

Show router and agent I/O on stderr:

```bash
uv run rag-tag --input
```

Show full JSON result details below answers:

```bash
uv run rag-tag --verbose
```

Fail closed on merged SQL errors:

```bash
uv run rag-tag --strict-sql
```

Use one-off profile overrides:

```bash
uv run rag-tag --router-profile router-gemini-flash --agent-profile cohere-command-a
```

### Data prep commands

Convert one IFC file explicitly:

```bash
uv run rag-tag-ifc-to-jsonl --ifc-file ./IFC-Files/Building-Architecture.ifc
```

Convert one JSONL file to SQLite explicitly:

```bash
uv run rag-tag-jsonl-to-sql --jsonl-file ./output/Building-Architecture.jsonl
```

Build graph visualization output:

```bash
uv run rag-tag-jsonl-to-graph
```

Import one dataset into Neo4j:

```bash
uv run rag-tag-jsonl-to-neo4j --jsonl-dir ./output --dataset Building-Architecture
```

Refresh IFC4.3 RDF support data:

```bash
uv run rag-tag-refresh-ifc43-rdf
```

Generate the ontology map:

```bash
uv run rag-tag-generate-ontology-map
```

## 12. Evaluation scripts

### Routing evaluation

Evaluate router decisions and optional SQL execution:

```bash
uv run python scripts/eval_routing.py --db ./output/Building-Architecture.db --router-mode llm --strict
```

### Graph-agent comparison

Compare graph-agent profiles from a config-defined experiment:

```bash
uv run python scripts/eval_graph_models.py \
  --config ./config.yaml \
  --experiment graph-agent-compare \
  --db ./output/Building-Architecture.db \
  --graph-dataset Building-Architecture \
  --output ./output/graph-model-report.json
```

Compare an explicit list of profiles instead:

```bash
uv run python scripts/eval_graph_models.py \
  --config ./config.yaml \
  --profiles cohere-command-a dbx-claude-sonnet-4-6 dbx-gpt-oss-20b \
  --db ./output/Building-Architecture.db \
  --graph-dataset Building-Architecture
```

Important notes:

- `eval_graph_models.py` forces `route="graph"`
- it compares graph-agent behavior, not router behavior
- if `--db` resolves to exactly one database, its stem can be reused as the graph dataset
- use `--graph-dataset` when you want to pin a specific dataset in a multi-dataset setup

## 13. Project structure

```text
src/rag_tag/
  run_agent.py                 # main CLI entrypoint
  textual_app.py               # Textual TUI
  query_service.py             # routing and execution orchestration
  ifc_sql_tool.py              # SQL execution helper
  ifc_graph_tool.py            # graph query interface
  graph/                       # graph runtime abstraction and backends
  router/                      # SQL vs graph routing
  agent/                       # graph agent and tools
  parser/
    ifc_to_jsonl.py            # IFC -> JSONL
    jsonl_to_sql.py            # JSONL -> SQLite
    jsonl_to_graph.py          # JSONL -> NetworkX graph + visualization
    jsonl_to_neo4j.py          # JSONL -> Neo4j import

scripts/
  eval_routing.py
  eval_graph_models.py
  run_neo4j_workflow.sh

IFC-Files/                     # source IFC models
output/                        # generated JSONL, SQLite, and reports
docker-compose.neo4j.yml       # local Neo4j service
config.example.yaml            # recommended config starting point
```

## 14. Reliability / contract notes

- Counts, lists, and aggregations should come from SQLite results, not graph guesses.
- Graph reasoning handles spatial, topological, and explicit IFC relationships.
- `adjacent_to` is proximity-based. It is not a full topology solver.
- `contains` and `typed_by` come from IFC relationship data.
- If geometry is missing, graph construction degrades gracefully instead of crashing.
- Graph actions return the stable envelope `{status,data,error}`.
- Graph nodes expose both:
    - `properties`: flat compatibility view
    - `payload`: full nested JSONL record
- SQL merge mode reports partial database failures in warnings instead of silently dropping them.
- `uv run rag-tag --strict-sql` makes merged SQL execution fail closed.
- Neo4j mirrors the canonical graph built from JSONL; the import path rebuilds from JSONL first, then projects into Neo4j.

Relation source semantics used in graph results:

- `ifc`: explicit IFC relationship
- `heuristic`: proximity-based spatial relation
- `topology`: geometry/topology-derived relation

## 15. Development checks

Format check:

```bash
uv run ruff format --check .
```

Lint:

```bash
uv run ruff check .
```

Tests:

```bash
uv run pytest
```

Optional pre-commit hook install:

```bash
uv run pre-commit install
```
