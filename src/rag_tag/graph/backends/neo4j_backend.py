"""Neo4j-backed graph runtime."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from rag_tag.graph.properties import cached_db_lookup, merge_db_element_data
from rag_tag.graph_contract import (
    CANONICAL_ACTIONS,
    CANONICAL_RELATION_SET,
    SPATIAL_RELATIONS,
    TOPOLOGY_RELATIONS,
    is_allowed_action,
    make_error_envelope,
    make_ok_envelope,
    normalize_action_name,
    normalize_relation_name,
    normalize_relation_source,
    relation_bucket,
)
from rag_tag.ifc_graph_tool import (
    INTERNAL_PAYLOAD_MODE,
    LLM_PAYLOAD_MODE,
    build_node_payload,
)
from rag_tag.observability import _load_dotenv

from .neo4j_cypher import (
    MATCH_ALL_NODES,
    MATCH_CLASS,
    MATCH_DESCENDANTS_CONTAINS,
    MATCH_NODE_BY_ID,
    MATCH_OUTGOING_RELATION_EXACT,
    MATCH_OUTGOING_RELATIONS,
    MATCH_SPATIAL_NEIGHBORS,
    MATCH_STOREY_BY_ID,
    MATCH_STOREY_BY_LABEL,
    MATCH_TOPOLOGY_NEIGHBORS,
)

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None


@dataclass(slots=True)
class Neo4jBackend:
    """Neo4j backend implementing canonical graph actions."""

    graph: nx.DiGraph | nx.MultiDiGraph | None = None
    db_path: Path | None = None
    selected_datasets: tuple[str, ...] = ()
    _catalog_graph: nx.DiGraph | nx.MultiDiGraph | None = None
    _driver: Any | None = None
    _database: str | None = None
    _conn_error: str | None = None

    def __post_init__(self) -> None:
        _load_dotenv()
        self.selected_datasets = tuple(
            dataset
            for dataset in self.selected_datasets
            if isinstance(dataset, str) and dataset
        )
        # Lazy in-memory catalog: used for fuzzy/class helpers that still rely on
        # NetworkX semantics without forcing a full agent refactor.
        self._catalog_graph = _scope_catalog_graph(self.graph, self.selected_datasets)
        self._driver = None
        self._database = (os.environ.get("NEO4J_DATABASE") or "").strip() or None
        self._conn_error = None
        self._connect()

    def _connect(self) -> None:
        if GraphDatabase is None:
            self._conn_error = "neo4j package not installed"
            return
        uri = (os.environ.get("NEO4J_URI") or "").strip()
        user = (os.environ.get("NEO4J_USERNAME") or "").strip()
        password = (os.environ.get("NEO4J_PASSWORD") or "").strip()
        if not uri or not user or not password:
            self._conn_error = "NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not set"
            return
        # Keep driver creation centralized so connection errors surface once.
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def _session(self):
        if self._driver is None:
            return nullcontext(None)
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def get_networkx_graph(self) -> nx.DiGraph | nx.MultiDiGraph:
        return self._ensure_catalog_graph()

    def set_context_db_path(self, db_path: Path | None) -> None:
        resolved = db_path.expanduser().resolve() if db_path is not None else None
        self.db_path = resolved
        if self._catalog_graph is None:
            return
        if resolved is None:
            self._catalog_graph.graph.pop("_db_path", None)
            return
        self._catalog_graph.graph["_db_path"] = resolved

    def _ensure_catalog_graph(self) -> nx.DiGraph | nx.MultiDiGraph:
        if self._catalog_graph is not None:
            return self._catalog_graph
        G = nx.DiGraph()
        with self._session() as session:
            if session is None:
                return G
            result = session.run(MATCH_ALL_NODES, **self._dataset_params())
            for record in result:
                node = record["n"]
                node_id = node.get("node_id")
                if not node_id:
                    continue
                # Neo4j properties cannot store nested maps; recover full maps
                # from JSON fields and reattach scalar fields for filtering.
                properties = _decode_json(node.get("properties_json")) or {}
                if not isinstance(properties, dict):
                    properties = {}
                _merge_scalar_properties(properties, node)
                payload = _decode_json(node.get("payload_json"))
                geometry = _decode_json(node.get("geometry_json")) or {}
                if not isinstance(geometry, dict):
                    geometry = {}
                G.add_node(
                    node_id,
                    label=node.get("label"),
                    class_=node.get("class_"),
                    node_kind=node.get("node_kind"),
                    properties=properties,
                    payload=payload if isinstance(payload, dict) else {},
                    dataset=node.get("dataset"),
                    **geometry,
                )
        G.graph["_payload_mode"] = "full"
        if self.selected_datasets:
            G.graph["datasets"] = list(self.selected_datasets)
        if self.db_path is not None:
            G.graph["_db_path"] = self.db_path
        self._catalog_graph = G
        return G

    def _dataset_params(self) -> dict[str, list[str]]:
        return {"datasets": list(self.selected_datasets)}

    def _err(
        self, message: str, code: str, details: dict | None = None
    ) -> dict[str, Any]:
        return make_error_envelope(message, code, details)

    def _ok(self, action: str, data: dict[str, Any] | None) -> dict[str, Any]:
        return make_ok_envelope(action, data)

    def _edge_source(self, relation: str | None, source: Any) -> str | None:
        bucket = relation_bucket(relation)
        if bucket is None or bucket == "hierarchy":
            return None
        if bucket == "explicit_ifc":
            return "ifc"
        if bucket == "topology":
            if relation in {"space_bounded_by", "bounds_space", "path_connected_to"}:
                return "ifc"
            return "topology"
        if bucket == "spatial":
            return "heuristic"
        cleaned = normalize_relation_source(source)
        return cleaned

    def _resolve_element_id(
        self, element_id: str
    ) -> tuple[str | None, dict[str, Any] | None]:
        if not isinstance(element_id, str):
            return None, {"error": "Invalid element_id: element_id must be a string"}
        with self._session() as session:
            if session is None:
                return None, {"error": "Neo4j not configured"}

            def _match_node_id(node_id: str) -> str | None:
                rec = session.run(
                    MATCH_NODE_BY_ID,
                    node_id=node_id,
                    **self._dataset_params(),
                ).single()
                if rec is None:
                    return None
                node = rec["n"]
                if node is None:
                    return None
                if str(node_id).startswith("Element::"):
                    return node_id
                return None

            if element_id.startswith("Element::"):
                resolved = _match_node_id(element_id)
                if resolved:
                    return resolved, None
            else:
                candidate = f"Element::{element_id}"
                resolved = _match_node_id(candidate)
                if resolved:
                    return resolved, None

            rows = list(
                session.run(
                    "MATCH (n:Node) WHERE n.node_id STARTS WITH 'Element::' "
                    "AND n.global_id = $gid "
                    "AND (size($datasets) = 0 OR n.dataset IN $datasets) "
                    "RETURN n.node_id AS id",
                    gid=element_id,
                    **self._dataset_params(),
                )
            )
            if len(rows) == 1:
                return rows[0]["id"], None
            if len(rows) > 1:
                return (
                    None,
                    {
                        "error": "Ambiguous element_id",
                        "candidates": [r["id"] for r in rows],
                    },
                )
            return None, {"error": f"Element not found: {element_id}"}

    def _resolve_storey_node(
        self, storey_query: str
    ) -> tuple[str | None, dict[str, Any] | None]:
        query = storey_query.strip()
        if not query:
            return None, {"error": "Missing param: storey"}
        with self._session() as session:
            if session is None:
                return None, {"error": "Neo4j not configured"}

            direct = query if query.startswith("Storey::") else f"Storey::{query}"
            rec = session.run(
                MATCH_STOREY_BY_ID,
                node_id=direct,
                **self._dataset_params(),
            ).single()
            if rec is not None:
                node = rec["n"]
                if node is not None:
                    return node.get("node_id"), None

            rows = list(
                session.run(
                    MATCH_STOREY_BY_LABEL,
                    label=query,
                    **self._dataset_params(),
                )
            )
            if len(rows) == 1:
                node = rows[0]["n"]
                return node.get("node_id"), None
            if len(rows) > 1:
                return (
                    None,
                    {
                        "error": "Ambiguous storey",
                        "candidates": [r["n"].get("node_id") for r in rows],
                    },
                )

            def _norm(text: str) -> str:
                return " ".join(
                    chunk
                    for chunk in "".join(
                        c.lower() if c.isalnum() else " " for c in text
                    ).split()
                )

            rows = list(
                session.run(
                    "MATCH (n:Node) WHERE toLower(n.class_) = 'ifcbuildingstorey' "
                    "AND (size($datasets) = 0 OR n.dataset IN $datasets) "
                    "RETURN n.node_id AS id, n.label AS label",
                    **self._dataset_params(),
                )
            )
            qn = _norm(query)
            matches = [r["id"] for r in rows if _norm(str(r["label"] or "")) == qn]
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                return None, {"error": "Ambiguous storey", "candidates": matches}
            return None, {"error": f"Storey not found: {storey_query}"}

    def query(
        self,
        action: str,
        params: dict[str, Any],
        *,
        payload_mode: str = LLM_PAYLOAD_MODE,
    ) -> dict[str, Any]:
        if not isinstance(action, str):
            return self._err("Invalid action: action must be a string", "invalid")
        action = normalize_action_name(action)
        if not is_allowed_action(action):
            return self._err(
                f"Unknown action: {action}",
                "unknown_action",
                {"allowed_actions": sorted(CANONICAL_ACTIONS)},
            )
        if not isinstance(params, dict):
            return self._err("Invalid params: params must be an object", "invalid")

        if action == "get_elements_in_storey":
            storey = params.get("storey")
            if not storey:
                return self._err("Missing param: storey", "missing_param")
            if not isinstance(storey, str):
                return self._err("Invalid param: storey must be a string", "invalid")
            node, err = self._resolve_storey_node(storey)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return self._err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                return self._err(str(error_msg), "not_found")
            if node is None:
                return self._err(f"Storey not found: {storey}", "not_found")

            container_classes = {
                "IfcProject",
                "IfcSite",
                "IfcBuilding",
                "IfcBuildingStorey",
                "IfcSpace",
                "IfcZone",
                "IfcSpatialZone",
                "IfcTypeObject",
            }

            elements: list[dict[str, Any]] = []
            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                result = session.run(
                    MATCH_DESCENDANTS_CONTAINS,
                    node_id=node,
                    **self._dataset_params(),
                )
                for record in result:
                    n = record["n"]
                    if n is None:
                        continue
                    cls = n.get("class_")
                    if cls in container_classes:
                        continue
                    elements.append(
                        {
                            "id": n.get("node_id"),
                            "label": n.get("label"),
                            "class_": cls,
                        }
                    )
            return self._ok(action, {"storey": storey, "elements": elements})

        if action == "find_elements_by_class":
            cls = params.get("class")
            if not cls:
                return self._err("Missing param: class", "missing_param")
            if not isinstance(cls, str):
                return self._err("Invalid param: class must be a string", "invalid")
            target = _normalize_class(cls)
            matches: list[dict[str, Any]] = []
            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                result = session.run(
                    MATCH_CLASS,
                    class_name=target,
                    **self._dataset_params(),
                )
                for record in result:
                    n = record["n"]
                    node_data = _node_data_from_neo4j(n)
                    matches.append(
                        build_node_payload(
                            node_data["id"], node_data, payload_mode=payload_mode
                        )
                    )
            return self._ok(action, {"class": target, "elements": matches})

        if action == "find_nodes":
            # Keep fuzzy/class helpers consistent with the NetworkX path by
            # delegating to an in-memory catalog graph.
            catalog = self._ensure_catalog_graph()
            return query_ifc_graph_catalog(catalog, action, params, payload_mode)

        if action == "list_property_keys":
            # Key discovery uses the same NetworkX-based helper logic.
            catalog = self._ensure_catalog_graph()
            return query_ifc_graph_catalog(catalog, action, params, payload_mode)

        if action in {"spatial_compare", "find_elements_within_clearance"}:
            catalog = self._ensure_catalog_graph()
            return query_ifc_graph_catalog(catalog, action, params, payload_mode)

        if self._conn_error:
            return self._err(self._conn_error, "neo4j_not_configured")

        if action == "get_element_properties":
            element_id = params.get("element_id")
            if element_id is None or element_id == "":
                return self._err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return self._err(
                    "Invalid param: element_id must be a string", "invalid"
                )
            resolved, err = self._resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return self._err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return self._err(str(error_msg), "invalid")
                return self._err(str(error_msg), "not_found")
            if not resolved:
                return self._err(f"Element not found: {element_id}", "not_found")

            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                rec = session.run(
                    MATCH_NODE_BY_ID,
                    node_id=resolved,
                    **self._dataset_params(),
                ).single()
                if rec is None:
                    return self._err(f"Element not found: {element_id}", "not_found")
                node = rec["n"]
                node_data = _node_data_from_neo4j(node)

            base_node_data: dict[str, Any] = dict(node_data)
            if self.db_path is not None:
                db_data = cached_db_lookup(
                    self._ensure_catalog_graph(),
                    resolved,
                    self.db_path,
                )
                if db_data is not None:
                    base_node_data = merge_db_element_data(base_node_data, db_data)

            return self._ok(
                action,
                build_node_payload(
                    resolved, base_node_data, payload_mode=INTERNAL_PAYLOAD_MODE
                ),
            )

        if action == "get_adjacent_elements":
            element_id = params.get("element_id")
            if not element_id:
                return self._err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return self._err(
                    "Invalid param: element_id must be a string", "invalid"
                )
            resolved, err = self._resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return self._err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return self._err(str(error_msg), "invalid")
                return self._err(str(error_msg), "not_found")
            if resolved is None:
                return self._err(f"Element not found: {element_id}", "not_found")

            neighbors: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                result = session.run(
                    MATCH_SPATIAL_NEIGHBORS,
                    node_id=resolved,
                    relations=list(SPATIAL_RELATIONS),
                    **self._dataset_params(),
                )
                for record in result:
                    m = record["m"]
                    r = record["r"]
                    dedupe_key = _spatial_neighbor_dedupe_key(m, r)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    relation = normalize_relation_name(r.get("relation"))
                    neighbors.append(
                        {
                            "id": m.get("node_id"),
                            "label": m.get("label"),
                            "class_": m.get("class_"),
                            "relation": relation,
                            "distance": r.get("distance"),
                            "source": self._edge_source(relation, r.get("source")),
                        }
                    )
            return self._ok(action, {"element_id": resolved, "adjacent": neighbors})

        if action == "get_topology_neighbors":
            element_id = params.get("element_id")
            relation = params.get("relation")
            if not element_id:
                return self._err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return self._err(
                    "Invalid param: element_id must be a string", "invalid"
                )
            if not relation:
                return self._err("Missing param: relation", "missing_param")
            if not isinstance(relation, str):
                return self._err("Invalid param: relation must be a string", "invalid")
            relation_value = normalize_relation_name(relation)
            allowed = set(TOPOLOGY_RELATIONS)
            if relation_value not in allowed:
                return self._err(
                    f"Unsupported topology relation: {relation}",
                    "invalid",
                    {"allowed_relations": sorted(allowed)},
                )
            resolved, err = self._resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return self._err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return self._err(str(error_msg), "invalid")
                return self._err(str(error_msg), "not_found")
            if resolved is None:
                return self._err(f"Element not found: {element_id}", "not_found")

            neighbors: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                result = session.run(
                    MATCH_TOPOLOGY_NEIGHBORS,
                    node_id=resolved,
                    relations=[relation_value],
                    **self._dataset_params(),
                )
                for record in result:
                    m = record["m"]
                    r = record["r"]
                    dedupe_key = _topology_neighbor_dedupe_key(m, r)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    rel = normalize_relation_name(r.get("relation"))
                    neighbors.append(
                        {
                            "id": m.get("node_id"),
                            "label": m.get("label"),
                            "class_": m.get("class_"),
                            "relation": rel,
                            "vertical_gap": r.get("vertical_gap"),
                            "overlap_area_xy": r.get("overlap_area_xy"),
                            "intersection_volume": r.get("intersection_volume"),
                            "contact_area": r.get("contact_area"),
                            "axis_angle_deg": r.get("axis_angle_deg"),
                            "parallel_score": r.get("parallel_score"),
                            "perpendicular_score": r.get("perpendicular_score"),
                            "facing_score": r.get("facing_score"),
                            "support_score": r.get("support_score"),
                            "containment_ratio": r.get("containment_ratio"),
                            "source": self._edge_source(rel, r.get("source")),
                        }
                    )
            return self._ok(
                action,
                {
                    "element_id": resolved,
                    "relation": relation_value,
                    "neighbors": neighbors,
                },
            )

        if action == "get_intersections_3d":
            element_id = params.get("element_id")
            if not element_id:
                return self._err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return self._err(
                    "Invalid param: element_id must be a string", "invalid"
                )
            resolved, err = self._resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return self._err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return self._err(str(error_msg), "invalid")
                return self._err(str(error_msg), "not_found")
            if resolved is None:
                return self._err(f"Element not found: {element_id}", "not_found")

            neighbors: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                result = session.run(
                    MATCH_TOPOLOGY_NEIGHBORS,
                    node_id=resolved,
                    relations=["intersects_3d"],
                    **self._dataset_params(),
                )
                for record in result:
                    m = record["m"]
                    r = record["r"]
                    dedupe_key = _topology_neighbor_dedupe_key(m, r)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    rel = normalize_relation_name(r.get("relation"))
                    neighbors.append(
                        {
                            "id": m.get("node_id"),
                            "label": m.get("label"),
                            "class_": m.get("class_"),
                            "relation": rel,
                            "intersection_volume": r.get("intersection_volume"),
                            "contact_area": r.get("contact_area"),
                            "source": self._edge_source(rel, r.get("source")),
                        }
                    )
            return self._ok(
                action, {"element_id": resolved, "intersections_3d": neighbors}
            )

        if action == "spatial_query":
            cls = params.get("class")
            if cls is not None and not isinstance(cls, str):
                return self._err("Invalid param: class must be a string", "invalid")
            class_filter = _normalize_class(cls) if cls else None
            near = params.get("near")
            max_distance = params.get("max_distance")
            if near is None:
                return self._err("Missing param: near", "missing_param")
            if not isinstance(near, (str, int, float)):
                return self._err(
                    "Invalid param: near must be a string or number", "invalid"
                )
            resolved, err = self._resolve_element_id(str(near))
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return self._err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return self._err(str(error_msg), "invalid")
                return self._err(str(error_msg), "not_found")
            if resolved is None:
                return self._err(f"Element not found: {near}", "not_found")
            if max_distance is None:
                return self._err("Missing param: max_distance", "missing_param")
            try:
                max_distance_value = float(max_distance)
            except (TypeError, ValueError):
                return self._err(
                    "Invalid param: max_distance must be a number", "invalid"
                )

            results: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                result = session.run(
                    MATCH_SPATIAL_NEIGHBORS,
                    node_id=resolved,
                    relations=list(SPATIAL_RELATIONS),
                    **self._dataset_params(),
                )
                for record in result:
                    m = record["m"]
                    r = record["r"]
                    dedupe_key = _spatial_neighbor_dedupe_key(m, r)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    dist = r.get("distance")
                    if dist is None or float(dist) > max_distance_value:
                        continue
                    if class_filter is not None:
                        if str(m.get("class_") or "").lower() != class_filter.lower():
                            continue
                    rel = normalize_relation_name(r.get("relation"))
                    results.append(
                        {
                            "id": m.get("node_id"),
                            "label": m.get("label"),
                            "class_": m.get("class_"),
                            "relation": rel,
                            "distance": dist,
                            "source": self._edge_source(rel, r.get("source")),
                        }
                    )
            return self._ok(
                action,
                {
                    "near": resolved,
                    "max_distance": max_distance_value,
                    "results": results,
                },
            )

        if action == "find_elements_above":
            element_id = params.get("element_id")
            max_gap = params.get("max_gap")
            if not element_id:
                return self._err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return self._err(
                    "Invalid param: element_id must be a string", "invalid"
                )
            max_gap_value: float | None = None
            if max_gap is not None:
                try:
                    max_gap_value = float(max_gap)
                except (TypeError, ValueError):
                    return self._err(
                        "Invalid param: max_gap must be a number", "invalid"
                    )
            resolved, err = self._resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return self._err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return self._err(str(error_msg), "invalid")
                return self._err(str(error_msg), "not_found")
            if resolved is None:
                return self._err(f"Element not found: {element_id}", "not_found")

            results: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                result = session.run(
                    MATCH_TOPOLOGY_NEIGHBORS,
                    node_id=resolved,
                    relations=["above"],
                    **self._dataset_params(),
                )
                for record in result:
                    m = record["m"]
                    r = record["r"]
                    dedupe_key = _topology_neighbor_dedupe_key(m, r)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    gap = r.get("vertical_gap")
                    if (
                        max_gap_value is not None
                        and gap is not None
                        and float(gap) > max_gap_value
                    ):
                        continue
                    rel = normalize_relation_name(r.get("relation"))
                    results.append(
                        {
                            "id": m.get("node_id"),
                            "label": m.get("label"),
                            "class_": m.get("class_"),
                            "relation": rel,
                            "vertical_gap": gap,
                            "source": self._edge_source(rel, r.get("source")),
                        }
                    )
            return self._ok(
                action,
                {"element_id": resolved, "max_gap": max_gap_value, "results": results},
            )

        if action == "find_elements_below":
            element_id = params.get("element_id")
            max_gap = params.get("max_gap")
            if not element_id:
                return self._err("Missing param: element_id", "missing_param")
            if not isinstance(element_id, str):
                return self._err(
                    "Invalid param: element_id must be a string", "invalid"
                )
            max_gap_value: float | None = None
            if max_gap is not None:
                try:
                    max_gap_value = float(max_gap)
                except (TypeError, ValueError):
                    return self._err(
                        "Invalid param: max_gap must be a number", "invalid"
                    )
            resolved, err = self._resolve_element_id(element_id)
            if err:
                error_msg = err.get("error", "Unknown error")
                if "Ambiguous" in str(error_msg):
                    return self._err(
                        str(error_msg),
                        "ambiguous",
                        {"candidates": err.get("candidates", [])},
                    )
                if "Invalid element_id" in str(error_msg):
                    return self._err(str(error_msg), "invalid")
                return self._err(str(error_msg), "not_found")
            if resolved is None:
                return self._err(f"Element not found: {element_id}", "not_found")

            results: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                result = session.run(
                    MATCH_TOPOLOGY_NEIGHBORS,
                    node_id=resolved,
                    relations=["below"],
                    **self._dataset_params(),
                )
                for record in result:
                    m = record["m"]
                    r = record["r"]
                    dedupe_key = _topology_neighbor_dedupe_key(m, r)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    gap = r.get("vertical_gap")
                    if (
                        max_gap_value is not None
                        and gap is not None
                        and float(gap) > max_gap_value
                    ):
                        continue
                    rel = normalize_relation_name(r.get("relation"))
                    results.append(
                        {
                            "id": m.get("node_id"),
                            "label": m.get("label"),
                            "class_": m.get("class_"),
                            "relation": rel,
                            "vertical_gap": gap,
                            "source": self._edge_source(rel, r.get("source")),
                        }
                    )
            return self._ok(
                action,
                {"element_id": resolved, "max_gap": max_gap_value, "results": results},
            )

        if action == "traverse":
            start = params.get("start")
            relation_param = params.get("relation")
            if not start:
                return self._err("Missing param: start", "missing_param")
            if not isinstance(start, str):
                return self._err("Invalid param: start must be a string", "invalid")
            if relation_param is not None and not isinstance(relation_param, str):
                return self._err("Invalid param: relation must be a string", "invalid")
            try:
                depth = int(params.get("depth", 1))
            except (TypeError, ValueError):
                return self._err("Invalid param: depth must be an integer", "invalid")
            if depth < 1:
                return self._err("Depth must be >= 1", "invalid")

            with self._session() as session:
                if session is None:
                    return self._err("Neo4j not configured", "neo4j_not_configured")
                exists = session.run(
                    MATCH_NODE_BY_ID,
                    node_id=start,
                    **self._dataset_params(),
                ).single()
                if exists is None:
                    return self._err(f"Start node not found: {start}", "not_found")

                visited = {start}
                frontier = {start}
                results: list[dict[str, Any]] = []

                relation_filter: set[str] | None = None
                relation_value: str | None = None
                if relation_param is not None:
                    relation_value = normalize_relation_name(relation_param)
                    if relation_value is None:
                        return self._err(
                            "Invalid param: relation must be non-empty", "invalid"
                        )
                    if relation_value not in CANONICAL_RELATION_SET:
                        return self._err(
                            f"Unsupported traverse relation: {relation_param}",
                            "invalid",
                            {"allowed_relations": sorted(CANONICAL_RELATION_SET)},
                        )
                    relation_filter = {relation_value}

                for _ in range(depth):
                    if not frontier:
                        break
                    if relation_filter is None:
                        query = MATCH_OUTGOING_RELATIONS
                        params_query = {
                            "frontier": list(frontier),
                            "relations": list(CANONICAL_RELATION_SET),
                        }
                    else:
                        query = MATCH_OUTGOING_RELATION_EXACT
                        params_query = {
                            "frontier": list(frontier),
                            "relation": relation_value,
                        }

                    rows = session.run(
                        query,
                        **params_query,
                        **self._dataset_params(),
                    )
                    next_frontier = set()
                    for record in rows:
                        to_node = record["m"]
                        to_id = to_node.get("node_id")
                        if to_id in visited:
                            continue
                        rel = record["r"]
                        relation = normalize_relation_name(rel.get("relation"))
                        if relation not in CANONICAL_RELATION_SET:
                            continue
                        visited.add(to_id)
                        next_frontier.add(to_id)
                        node_data = _node_data_from_neo4j(to_node)
                        results.append(
                            {
                                "from": record["from_id"],
                                "to": to_id,
                                "relation": relation,
                                "source": self._edge_source(
                                    relation, rel.get("source")
                                ),
                                "node": build_node_payload(
                                    to_id,
                                    node_data,
                                    payload_mode=payload_mode,
                                ),
                            }
                        )
                    frontier = next_frontier

            return self._ok(
                action,
                {
                    "start": start,
                    "relation": relation_value,
                    "depth": depth,
                    "results": results,
                },
            )

        return self._err(
            f"Unknown action: {action}",
            "unknown_action",
            {"allowed_actions": sorted(CANONICAL_ACTIONS)},
        )


def _normalize_class(value: str) -> str:
    v = value.strip()
    if not v:
        return v
    if not v.lower().startswith("ifc"):
        v = f"Ifc{v}"
    return v


def _decode_json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _node_data_from_neo4j(node: Any) -> dict[str, Any]:
    properties = _decode_json(node.get("properties_json")) or {}
    if not isinstance(properties, dict):
        properties = {}
    _merge_scalar_properties(properties, node)
    payload = _decode_json(node.get("payload_json"))
    if not isinstance(payload, dict):
        payload = {}
    geometry = _decode_json(node.get("geometry_json")) or {}
    if not isinstance(geometry, dict):
        geometry = {}
    return {
        "id": node.get("node_id"),
        "label": node.get("label"),
        "class_": node.get("class_"),
        "node_kind": node.get("node_kind"),
        "properties": properties,
        "payload": payload,
        **geometry,
    }


def _merge_scalar_properties(properties: dict[str, Any], node: Any) -> None:
    mapping = {
        "GlobalId": node.get("global_id"),
        "ExpressId": node.get("express_id"),
        "Name": node.get("name"),
        "ClassRaw": node.get("class_raw"),
        "IfcType": node.get("ifc_type"),
        "Level": node.get("level"),
        "TypeName": node.get("type_name"),
        "PredefinedType": node.get("predefined_type"),
        "ObjectType": node.get("object_type"),
        "Tag": node.get("tag"),
        "Description": node.get("description"),
    }
    for key, value in mapping.items():
        if key not in properties and value is not None:
            properties[key] = value


def _spatial_neighbor_dedupe_key(node: Any, edge: Any) -> tuple[str, str, str]:
    return (
        str(node.get("node_id") or ""),
        normalize_relation_name(edge.get("relation")) or "",
        str(edge.get("distance")),
    )


def _topology_neighbor_dedupe_key(node: Any, edge: Any) -> tuple[str, str, str]:
    return (
        str(node.get("node_id") or ""),
        normalize_relation_name(edge.get("relation")) or "",
        str(edge.get("vertical_gap") or edge.get("intersection_volume") or ""),
    )


def _scope_catalog_graph(
    graph: nx.DiGraph | nx.MultiDiGraph | None,
    selected_datasets: tuple[str, ...],
) -> nx.DiGraph | nx.MultiDiGraph | None:
    if graph is None or not selected_datasets:
        return graph

    selected = set(selected_datasets)
    node_ids = [
        node_id
        for node_id, data in graph.nodes(data=True)
        if data.get("dataset") is None or data.get("dataset") in selected
    ]
    scoped_graph = graph.subgraph(node_ids).copy()
    scoped_graph.graph["datasets"] = list(selected_datasets)
    return scoped_graph


def query_ifc_graph_catalog(
    graph: nx.DiGraph | nx.MultiDiGraph,
    action: str,
    params: dict[str, Any],
    payload_mode: str,
) -> dict[str, Any]:
    from rag_tag.ifc_graph_tool import query_ifc_graph  # noqa: PLC0415

    return query_ifc_graph(graph, action, params, payload_mode=payload_mode)
