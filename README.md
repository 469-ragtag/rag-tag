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

2. Copy the checked-in config template and env template

```bash
cp config.example.yaml config.yaml
cp .env.sample .env
```

3. Put shared defaults in your local `config.yaml` and secrets in `.env`

```bash
DATABRICKS_TOKEN=...
GEMINI_API_KEY=...
COHERE_API_KEY=...
```

Start from `config.example.yaml`; it is the canonical checked-in reference for
provider settings, model profiles, default router/agent selections, and
experiment groupings. Keep your working `config.yaml` local (untracked), and
keep secrets and one-off shell overrides in `.env` or your shell environment.

4. Build artifacts

```bash
uv run rag-tag-generate-ontology-map
uv run rag-tag-ifc-to-jsonl
uv run rag-tag-jsonl-to-sql
uv run rag-tag-jsonl-to-graph
```

5. Run interactive agent

```bash
uv run rag-tag
```

Use `config.example.yaml` as the starting point for Databricks-backed profile
workflows and graph-model comparison runs.

If you already run the app like this:

```bash
uv run rag-tag --tui --db output/Building-Architecture.db --graph-dataset Building-Architecture --trace
```

that same command still works. The new config flow changes model/provider/profile
selection, not your normal CLI entrypoint. Keep using CLI flags for runtime session
options such as `--tui`, `--db`, `--graph-dataset`, and `--trace`.

## Configuration

Use `config.example.yaml` as the canonical checked-in reference, then copy it to
a local `config.yaml` for non-secret runtime defaults:

- `defaults` for shared router/agent profile selection and `router_mode`
- `providers` for named provider configuration such as Databricks hosts
- `profiles` for reusable router and graph-agent model selections
- `experiments` for repeatable graph comparison groups

Use `.env` or shell environment variables for secrets and one-off overrides:

- API keys and tokens such as `DATABRICKS_TOKEN`, `GEMINI_API_KEY`,
  `COHERE_API_KEY`
- machine-local overrides such as `RAG_TAG_CONFIG`
- temporary runtime model/profile overrides such as `ROUTER_MODEL`,
  `AGENT_MODEL`, `ROUTER_PROFILE`, `AGENT_PROFILE`

Config discovery order:

- repo-root `config.yaml`
- repo-root `config.yml`
- repo-root `config.json`

Override the discovered config file with either:

- `uv run rag-tag --config ./path/to/config.yaml`
- `RAG_TAG_CONFIG=./path/to/config.yaml uv run rag-tag`

The full checked-in template lives in `config.example.yaml`.

Practical split:

- local `config.yaml`: shared defaults you want to edit for your environment
- `.env`: secrets and machine-local overrides
- CLI flags: per-run session options such as TUI mode, selected DB, dataset, and tracing

`config.yaml` is intentionally gitignored so you can keep machine-local runtime
defaults without committing them.

Minimal local `config.yaml` setup for the same TUI command you use today:

`config.yaml`

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

`.env`

```bash
DATABRICKS_HOST=your-workspace-host.databricks.com
DATABRICKS_TOKEN=your_databricks_token
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
LOGFIRE_TOKEN=your_write_logfire_token
```

Then run the app exactly as before:

```bash
uv run rag-tag --tui --db output/Building-Architecture.db --graph-dataset Building-Architecture --trace
```

To switch models, either edit `defaults.agent_profile` in `config.yaml` or use a
one-off override like:

```bash
uv run rag-tag --tui --db output/Building-Architecture.db --graph-dataset Building-Architecture --trace --agent-profile dbx-gpt-oss-20b
```

To switch back to the current Cohere graph agent through config, set:

```yaml
defaults:
  agent_profile: cohere-command-a
```

or do it for one run only:

```bash
uv run rag-tag --tui --db output/Building-Architecture.db --graph-dataset Building-Architecture --trace --agent-profile cohere-command-a
```

## Databricks setup

Common Databricks environment variables:

```bash
DATABRICKS_HOST=dbc-00000000-0000.cloud.databricks.com
DATABRICKS_TOKEN=your_databricks_token
```

`DATABRICKS_TOKEN` is required for Databricks-backed profiles. `DATABRICKS_HOST`
should now live in `.env` or your shell environment, while `config.yaml`
references it via `host_env: DATABRICKS_HOST`.

If you prefer to store the full OpenAI-compatible serving URL instead of the host,
use `base_url_env` in config and define `DATABRICKS_BASE_URL` in `.env`.

For Databricks profiles, the `model` value is the serving endpoint name that is
resolved against the workspace's OpenAI-compatible `/serving-endpoints` base
URL. `config.example.yaml` includes example profiles for:

- Cohere Command A (current baseline)
- Claude Sonnet 4.6
- GPT OSS 20B
- Llama 4 Maverick
- Gemma 3 12B

These are example endpoint names only; update them to match your own Databricks
serving endpoints before use.

Databricks compatibility note:

- Databricks rejects the OpenAI-style `parallel_tool_calls` request field.
- Do not add `parallel_tool_calls` to Databricks profile settings in local config files.
- `rag-tag` strips that field automatically for Databricks-backed profiles.

## Query modes

- Default CLI: `uv run rag-tag`
- Textual TUI: `uv run rag-tag --tui`

Useful options:

```bash
uv run rag-tag --config ./config.yaml
uv run rag-tag --db ./output/Building-Architecture.db
uv run rag-tag --graph-dataset Building-Architecture
uv run rag-tag --router-profile router-gemini-flash
uv run rag-tag --agent-profile dbx-gpt-oss-20b
uv run rag-tag --input
uv run rag-tag --verbose
uv run rag-tag --trace
```

Runtime override notes:

- `--router-profile` and `--agent-profile` select named config profiles for one
  run without editing `config.yaml`.
- `ROUTER_PROFILE` and `AGENT_PROFILE` provide the same override via env vars.
- `ROUTER_MODEL` and `AGENT_MODEL` still bypass profile resolution for one-off
  runs and take precedence over config-selected profiles.
- `ROUTER_MODE` still overrides `defaults.router_mode` when set in the
  environment.

Dataset selection behavior:

- If `--graph-dataset` is set, that JSONL stem is used for graph routing.
- Else, if exactly one DB is selected, that DB stem is used.
- Else, SQL queries can still merge across DBs, but graph queries require an
  explicit dataset when multiple JSONL datasets are present.

## Graph model comparison

Use `scripts/eval_graph_models.py` to compare graph-agent behavior across
configured profiles.

Important behavior:

- the script intentionally forces `route="graph"`
- it compares graph-agent execution, not router behavior
- select profiles explicitly with `--profiles` or indirectly with
  `--experiment`
- write a JSON artifact with `--output` for later review or diffing

Practical example using a config-defined experiment and JSON report output:

```bash
uv run python scripts/eval_graph_models.py \
  --config ./config.yaml \
  --experiment graph-dbx-smoke \
  --db ./output/Building-Architecture.db \
  --graph-dataset Building-Architecture \
  --output ./output/graph-model-report.json
```

One-off comparison with explicit profiles:

```bash
uv run python scripts/eval_graph_models.py \
  --config ./config.yaml \
  --profiles cohere-command-a dbx-claude-sonnet-4-6 dbx-gpt-oss-20b dbx-llama-4-maverick \
  --questions-file ./graph-questions.json \
  --db ./output/Building-Architecture.db \
  --output ./output/graph-model-report.json
```

If `--db` resolves to exactly one SQLite database, its file stem is reused as
the graph dataset automatically. Pass `--graph-dataset` when you need to pin a
specific JSONL graph dataset instead.

## Benchmarking

Use `scripts/eval_benchmarks.py` to run end-to-end benchmarks over a checked-in
or local YAML case set. The benchmark runner:

- loads only the YAML cases you author
- expands the router x agent x prompt-strategy matrix
- runs the full routed query flow end to end
- writes Excel-friendly CSV artifacts plus a full JSON report
- uses an explicit answer judge model instead of Pydantic Evals' hidden default
- optionally enables Logfire tracing with `--trace`

### YAML case format

Put benchmark questions in a YAML file such as `evals/benchmark_cases_v1.yaml`.
V1 required fields are:

- `id`
- `question`
- `expected_route`

V1 recommended fields are:

- `reference_points`
- `tags`
- `max_duration_s`

Example:

```yaml
dataset_name: benchmark_cases_v1
cases:
  - id: q001
    question: Which rooms are adjacent to the kitchen?
    expected_route: graph
    reference_points:
      - identifies the kitchen correctly
      - avoids fabricated room names
    tags: [graph, spatial, adjacency]
    max_duration_s: 20

  - id: q002
    question: How many doors are on Level 2?
    expected_route: sql
    reference_points:
      - chooses sql routing
      - gives a deterministic count
    tags: [sql, count, level]
    max_duration_s: 10
```

### Run from config

Use the benchmark experiment defined in `config.example.yaml` as the starting
point for a one-command run:

```bash
uv run python scripts/eval_benchmarks.py \
  --config ./config.yaml \
  --experiment benchmark-e2e-v1 \
  --db ./output/Building-Architecture.db \
  --graph-dataset Building-Architecture \
  --answer-judge-model google-gla:gemini-2.5-flash \
  --trace
```

### Run with direct overrides

Use direct CLI overrides when you want to bypass experiment defaults:

```bash
uv run python scripts/eval_benchmarks.py \
  --questions-file ./evals/benchmark_cases_v1.yaml \
  --router-profiles router-gemini-flash dbx-gpt-oss-20b dbx-gemma-3-12b \
  --agent-profiles cohere-command-a dbx-claude-sonnet-4-6 dbx-gpt-oss-20b dbx-llama-4-maverick dbx-gemma-3-12b \
  --prompt-strategies baseline strict-grounded decompose \
  --answer-judge-model google-gla:gemini-2.5-flash \
  --repeat 1 \
  --max-concurrency 1 \
  --db ./output/Building-Architecture.db \
  --graph-dataset Building-Architecture
```

### Run a subset by tags

Use `--tags` to keep only YAML cases whose tag list intersects the requested
tags:

```bash
uv run python scripts/eval_benchmarks.py \
  --questions-file ./evals/benchmark_cases_v1.yaml \
  --tags sql graph \
  --db ./output/Building-Architecture.db
```

### Benchmark artifacts

Each benchmark run writes artifacts under `output/benchmarks/<run-id>/` unless
you override the output directory:

- `leaderboard.csv`
- `case_groups.csv`
- `runs.csv`
- `report.json`
- `run_manifest.json`

`leaderboard.csv` contains one row per router x agent x prompt-strategy
combination. `case_groups.csv` groups repeated runs by original case. `runs.csv`
contains one row per actual execution.

### Tracing and token fields

Use `--trace` to initialize Logfire for the benchmark run. If Logfire is
unavailable or `LOGFIRE_TOKEN` is missing, the benchmark still succeeds and
writes local artifacts.

Token usage fields are included when the provider returns usage metadata. When a
provider does not expose token counts, benchmark outputs leave those fields
blank rather than fabricating zeros. Use `token_coverage_rate` in the grouped
and leaderboard outputs to see how much of a run returned token data.

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
