#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DATASET="Building-Architecture"

dataset="$DEFAULT_DATASET"
if [[ $# -gt 0 && "$1" != --* ]]; then
  dataset="$1"
  shift
fi

extra_args=("$@")

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

: "${NEO4J_HOME:=$HOME/apps/neo4j}"
: "${NEO4J_URI:=bolt://localhost:7687}"
: "${NEO4J_USERNAME:=neo4j}"
: "${NEO4J_PASSWORD:=ragtag-dev-password}"
: "${NEO4J_DATABASE:=neo4j}"
: "${NEO4J_DATA_DIR:=$HOME/project-data/neo4j-data}"
: "${NEO4J_LOGS_DIR:=$HOME/project-data/neo4j-logs}"
: "${RAG_TAG_JSONL_DIR:=$ROOT_DIR/output}"
: "${RAG_TAG_DB_PATH:=$ROOT_DIR/output/${dataset}.db}"
: "${RAG_TAG_BUILD_JSONL:=1}"
: "${RAG_TAG_BUILD_SQL:=1}"
: "${RAG_TAG_TRACE:=0}"

export NEO4J_HOME
export NEO4J_URI
export NEO4J_USERNAME
export NEO4J_PASSWORD
export NEO4J_DATABASE
export NEO4J_DATA_DIR
export NEO4J_LOGS_DIR

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed." >&2
  exit 1
fi

if ! command -v java >/dev/null 2>&1; then
  echo "java is required to run Neo4j in a Eureka session." >&2
  exit 1
fi

neo4j_bin="$NEO4J_HOME/bin/neo4j"
cypher_shell_bin="$NEO4J_HOME/bin/cypher-shell"

if [[ ! -x "$neo4j_bin" ]]; then
  echo "Neo4j executable not found at $neo4j_bin" >&2
  echo "Set NEO4J_HOME to your user-space Neo4j install path." >&2
  exit 1
fi

if [[ ! -x "$cypher_shell_bin" ]]; then
  echo "cypher-shell not found at $cypher_shell_bin" >&2
  exit 1
fi

mkdir -p "$NEO4J_DATA_DIR" "$NEO4J_LOGS_DIR"

echo "Starting user-space Neo4j from $NEO4J_HOME"
if ! "$neo4j_bin" start >/dev/null 2>&1; then
  echo "Neo4j start returned a non-zero status. Continuing with readiness check." >&2
fi

echo "Waiting for Neo4j to become ready on $NEO4J_URI"
ready=0
for _ in $(seq 1 45); do
  if "$cypher_shell_bin" \
    -a "$NEO4J_URI" \
    -u "$NEO4J_USERNAME" \
    -p "$NEO4J_PASSWORD" \
    "RETURN 1;" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "$ready" -ne 1 ]]; then
  echo "Neo4j did not become ready in time." >&2
  echo "Check NEO4J_HOME and your neo4j.conf data/log paths." >&2
  exit 1
fi

if [[ "$RAG_TAG_BUILD_JSONL" == "1" ]]; then
  echo "Building JSONL artifacts"
  uv run rag-tag-ifc-to-jsonl
fi

if [[ "$RAG_TAG_BUILD_SQL" == "1" ]]; then
  echo "Building SQLite artifacts"
  uv run rag-tag-jsonl-to-sql
fi

echo "Importing dataset '$dataset' into Neo4j"
uv run rag-tag-jsonl-to-neo4j --jsonl-dir "$RAG_TAG_JSONL_DIR" --dataset "$dataset"

run_args=(
  uv run rag-tag
  --tui
  --db "$RAG_TAG_DB_PATH"
  --graph-dataset "$dataset"
)

if [[ "$RAG_TAG_TRACE" == "1" ]]; then
  run_args+=(--trace)
fi

if [[ ${#extra_args[@]} -gt 0 ]]; then
  run_args+=("${extra_args[@]}")
fi

echo "Launching rag-tag with the repo's Neo4j runtime integration"
GRAPH_BACKEND=neo4j "${run_args[@]}"
