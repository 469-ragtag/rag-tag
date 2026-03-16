#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.neo4j.yml"
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

: "${NEO4J_URI:=bolt://localhost:7687}"
: "${NEO4J_USERNAME:=neo4j}"
: "${NEO4J_PASSWORD:=ragtag-dev-password}"
: "${NEO4J_DATABASE:=neo4j}"

export NEO4J_URI
export NEO4J_USERNAME
export NEO4J_PASSWORD
export NEO4J_DATABASE

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed." >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  compose_cmd=(docker-compose)
else
  echo "Docker Compose is required but not installed." >&2
  exit 1
fi

echo "Starting Neo4j with $COMPOSE_FILE"
"${compose_cmd[@]}" -f "$COMPOSE_FILE" up -d

echo "Waiting for Neo4j to become ready..."
ready=0
for _ in $(seq 1 30); do
  if docker exec ragtag-neo4j \
    cypher-shell \
    -a bolt://localhost:7687 \
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
  exit 1
fi

jsonl_dir="${RAG_TAG_JSONL_DIR:-$ROOT_DIR/output}"
db_path="${RAG_TAG_DB_PATH:-$ROOT_DIR/output/${dataset}.db}"

echo "Building JSONL artifacts"
uv run rag-tag-ifc-to-jsonl

echo "Building SQLite artifacts"
uv run rag-tag-jsonl-to-sql

echo "Importing dataset '$dataset' into Neo4j"
uv run rag-tag-jsonl-to-neo4j --jsonl-dir "$jsonl_dir" --dataset "$dataset"

run_args=(
  uv run rag-tag
  --tui
  --db "$db_path"
  --graph-dataset "$dataset"
)

if [[ "${RAG_TAG_TRACE:-1}" == "1" ]]; then
  run_args+=(--trace)
fi

if [[ ${#extra_args[@]} -gt 0 ]]; then
  run_args+=("${extra_args[@]}")
fi

echo "Launching rag-tag with Neo4j backend"
GRAPH_BACKEND=neo4j "${run_args[@]}"
