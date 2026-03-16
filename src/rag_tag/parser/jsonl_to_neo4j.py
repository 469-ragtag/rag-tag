"""Import NetworkX graph into Neo4j by projection."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import networkx as nx

from rag_tag.graph.backends.neo4j_cypher import (
    CLEAR_DB,
    CREATE_CONSTRAINT,
    INSERT_NODES,
    INSERT_RELS,
)
from rag_tag.observability import _load_dotenv
from rag_tag.parser.jsonl_to_graph import build_graph

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None


def _encode_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except Exception:
        return None


def _geometry_payload(node_data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "geometry",
        "bbox",
        "mesh",
        "footprint_polygon",
        "obb",
        "local_placement_matrix",
        "z_min",
        "z_max",
        "height",
        "footprint_bbox_2d",
    )
    payload: dict[str, Any] = {}
    for key in keys:
        if key in node_data:
            payload[key] = node_data.get(key)
    return payload


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


def _extract_property_scalars(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_id": _scalar(properties.get("GlobalId")),
        "express_id": _scalar(properties.get("ExpressId")),
        "name": _scalar(properties.get("Name")),
        "class_raw": _scalar(properties.get("ClassRaw")),
        "ifc_type": _scalar(properties.get("IfcType")),
        "level": _scalar(properties.get("Level")),
        "type_name": _scalar(properties.get("TypeName")),
        "predefined_type": _scalar(properties.get("PredefinedType")),
        "object_type": _scalar(properties.get("ObjectType")),
        "tag": _scalar(properties.get("Tag")),
        "description": _scalar(properties.get("Description")),
    }


def import_networkx_graph(
    graph: nx.MultiDiGraph,
    *,
    batch_size: int = 2000,
    database: str | None = None,
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> None:
    _load_dotenv()
    if GraphDatabase is None:
        raise RuntimeError("neo4j package not installed")

    uri = uri or (os.environ.get("NEO4J_URI") or "").strip()
    username = username or (os.environ.get("NEO4J_USERNAME") or "").strip()
    password = password or (os.environ.get("NEO4J_PASSWORD") or "").strip()
    database = database or (os.environ.get("NEO4J_DATABASE") or "").strip() or None
    if not uri or not username or not password:
        raise RuntimeError("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not set")

    # Import is parity-first: build NetworkX graph, then project into Neo4j.
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            # Truncate first to keep imports deterministic and avoid collisions.
            session.run(CLEAR_DB)
            session.run(CREATE_CONSTRAINT)

            node_rows: list[dict[str, Any]] = []
            for node_id, data in graph.nodes(data=True):
                props = data.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                prop_scalars = _extract_property_scalars(props)
                node_rows.append(
                    {
                        "node_id": node_id,
                        "label": data.get("label"),
                        "class_": data.get("class_"),
                        "node_kind": data.get("node_kind"),
                        "dataset": data.get("dataset"),
                        "properties_json": _encode_json(props),
                        "payload_json": _encode_json(data.get("payload") or {}),
                        "geometry_json": _encode_json(_geometry_payload(data)),
                        **prop_scalars,
                    }
                )
                if len(node_rows) >= batch_size:
                    session.run(INSERT_NODES, rows=node_rows)
                    node_rows = []

            if node_rows:
                session.run(INSERT_NODES, rows=node_rows)

            rel_rows: list[dict[str, Any]] = []
            # Preserve original edge ordering for parallel relations.
            edge_index = 0
            if graph.is_multigraph():
                edge_iter = graph.edges(keys=True, data=True)
            else:
                edge_iter = ((u, v, None, d) for u, v, d in graph.edges(data=True))

            for u, v, _k, data in edge_iter:
                rel_rows.append(
                    {
                        "from": u,
                        "to": v,
                        "relation": data.get("relation"),
                        "source": data.get("source"),
                        "dataset": data.get("dataset")
                        or graph.nodes[u].get("dataset")
                        or graph.nodes[v].get("dataset"),
                        "distance": data.get("distance"),
                        "vertical_gap": data.get("vertical_gap"),
                        "overlap_area_xy": data.get("overlap_area_xy"),
                        "intersection_volume": data.get("intersection_volume"),
                        "contact_area": data.get("contact_area"),
                        "edge_index": edge_index,
                    }
                )
                edge_index += 1
                if len(rel_rows) >= batch_size:
                    session.run(INSERT_RELS, rows=rel_rows)
                    rel_rows = []

            if rel_rows:
                session.run(INSERT_RELS, rows=rel_rows)

    finally:
        driver.close()


def _resolve_jsonl_paths(jsonl_dir: Path | None, dataset: str | None) -> list[Path]:
    if jsonl_dir is None:
        jsonl_dir = Path("output")
    jsonl_dir = jsonl_dir.expanduser().resolve()
    if dataset:
        candidate = jsonl_dir / f"{dataset}.jsonl"
        if not candidate.is_file():
            raise FileNotFoundError(f"JSONL file not found: {candidate}")
        return [candidate]
    paths = sorted(jsonl_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found in {jsonl_dir}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Import IFC graph into Neo4j.")
    ap.add_argument(
        "--jsonl-dir",
        type=Path,
        default=None,
        help="Directory containing .jsonl files (default: ./output).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="JSONL stem to load (e.g. Building-Architecture).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Batch size for Neo4j UNWIND inserts.",
    )
    args = ap.parse_args()

    jsonl_paths = _resolve_jsonl_paths(args.jsonl_dir, args.dataset)
    LOG.info("Building NetworkX graph from %d JSONL file(s)", len(jsonl_paths))
    graph = build_graph(jsonl_paths, payload_mode="full")
    LOG.info(
        "Graph: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    LOG.info("Importing into Neo4j (truncate + insert)")
    import_networkx_graph(graph, batch_size=args.batch_size)
    LOG.info("Neo4j import complete")


if __name__ == "__main__":
    main()
