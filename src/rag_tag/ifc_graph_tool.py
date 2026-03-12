from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable

import networkx as nx

from rag_tag.graph.properties import (
    apply_property_filters,
)
from rag_tag.graph.properties import (
    cached_db_lookup as _cached_db_lookup_shared,
)
from rag_tag.graph.properties import (
    collect_dotted_keys_from_sqlite as _collect_dotted_keys_from_sqlite_shared,
)
from rag_tag.graph.properties import (
    merge_db_element_data as _merge_db_element_data_shared,
)
from rag_tag.graph_contract import (
    CANONICAL_ACTIONS,
    CANONICAL_RELATION_SET,
    KNOWN_RELATION_SOURCE_SET,
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

_logger = logging.getLogger(__name__)

LLM_PAYLOAD_MODE = "llm"
INTERNAL_PAYLOAD_MODE = "internal"

LLM_PROPERTY_ALLOWLIST: tuple[str, ...] = (
    "GlobalId",
    "Name",
    "TypeName",
    "Level",
    "PredefinedType",
    "ObjectType",
    "Zone",
)
MAX_LLM_STRING_CHARS = 160
REDACTED_COMPLEX_VALUE = "[REDACTED_COMPLEX]"
TRUNCATED_SUFFIX = "...[truncated]"


def _resolve_payload_mode(payload_mode: str) -> str:
    """Return a supported payload mode, defaulting safely to llm."""
    if payload_mode == INTERNAL_PAYLOAD_MODE:
        return INTERNAL_PAYLOAD_MODE
    return LLM_PAYLOAD_MODE


def sanitize_llm_property_value(value: Any) -> Any:
    """Reduce property value exposure to scalar-safe content."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_LLM_STRING_CHARS:
            return value
        return f"{value[:MAX_LLM_STRING_CHARS]}{TRUNCATED_SUFFIX}"
    return REDACTED_COMPLEX_VALUE


def sanitize_properties_for_llm(properties: dict[str, Any] | None) -> dict[str, Any]:
    """Filter properties to an allowlisted, redacted view for LLM tools."""
    if not isinstance(properties, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in LLM_PROPERTY_ALLOWLIST:
        if key not in properties:
            continue
        safe[key] = sanitize_llm_property_value(properties.get(key))
    return safe


def build_node_payload(
    node_id: str, node_data: dict[str, Any], *, payload_mode: str = LLM_PAYLOAD_MODE
) -> dict[str, Any]:
    """Build node payload with mode-aware property exposure.

    The ``payload`` key is always present in the returned dict:
    - In ``INTERNAL_PAYLOAD_MODE``: the full raw payload dict (or None).
    - In ``LLM_PAYLOAD_MODE``: ``None`` — the LLM sees a null placeholder so
      the key is structurally stable, but raw payload data is not exposed.
    """
    mode = _resolve_payload_mode(payload_mode)
    raw_props = node_data.get("properties")
    if mode == INTERNAL_PAYLOAD_MODE:
        properties = raw_props if isinstance(raw_props, dict) else {}
    else:
        properties = sanitize_properties_for_llm(raw_props)

    return {
        "id": node_id,
        "label": node_data.get("label"),
        "class_": node_data.get("class_"),
        "properties": properties,
        # Always include the payload key; value is only exposed in internal mode.
        "payload": node_data.get("payload") if mode == INTERNAL_PAYLOAD_MODE else None,
    }


def _ok_action(action: str, data: dict[str, Any] | None) -> dict[str, Any]:
    """Wrap successful action result in canonical envelope."""
    return make_ok_envelope(action, data)


def _err(message: str, code: str, details: dict | None = None) -> dict[str, Any]:
    """Wrap error result in envelope."""
    return make_error_envelope(message, code, details)


def _merge_db_element_data(
    node_data: dict[str, Any],
    db_data: dict[str, Any],
) -> dict[str, Any]:
    """Merge DB-sourced element data into in-memory node data."""
    return _merge_db_element_data_shared(node_data, db_data)


def _cached_db_lookup(
    G: nx.DiGraph | nx.MultiDiGraph,
    node_id: str,
    db_path: Path,
    *,
    db_conn: Any | None = None,
) -> dict[str, Any] | None:
    """Look up element data from the SQLite DB with a graph-level result cache."""
    return _cached_db_lookup_shared(G, node_id, db_path, db_conn=db_conn)


def _collect_dotted_keys_from_sqlite(
    db_path: Path,
    class_filter: str | None,
) -> dict[str, list[Any]]:
    """Collect dotted PropertySet/Quantity keys from SQLite."""
    return _collect_dotted_keys_from_sqlite_shared(db_path, class_filter)


def query_ifc_graph(
    G: nx.DiGraph | nx.MultiDiGraph,
    action: str,
    params: Dict[str, Any],
    *,
    payload_mode: str = LLM_PAYLOAD_MODE,
) -> Dict[str, Any]:
    """Controlled interface between the LLM and NetworkX graph."""
    if not isinstance(action, str):
        return _err("Invalid action: action must be a string", "invalid")

    action = normalize_action_name(action)
    if not is_allowed_action(action):
        return _err(
            f"Unknown action: {action}",
            "unknown_action",
            {"allowed_actions": sorted(CANONICAL_ACTIONS)},
        )

    if not isinstance(params, dict):
        return _err("Invalid params: params must be an object", "invalid")

    resolved_payload_mode = _resolve_payload_mode(payload_mode)

    def _normalize_class(value: str) -> str:
        v = value.strip()
        if not v:
            return v
        if not v.lower().startswith("ifc"):
            v = f"Ifc{v}"
        return v

    def _find_nodes_by_label(label: str, class_filter: str | None = None) -> list[str]:
        target = label.strip().lower()
        if not target:
            return []
        matches = []
        for n, d in G.nodes(data=True):
            lbl = str(d.get("label", "")).strip().lower()
            if lbl != target:
                continue
            if class_filter is not None:
                if str(d.get("class_", "")).lower() != class_filter.lower():
                    continue
            matches.append(n)
        return matches

    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _edge_relation(edge: dict[str, Any]) -> str | None:
        relation = normalize_relation_name(edge.get("relation"))
        if relation in CANONICAL_RELATION_SET:
            return relation
        return None

    def _iter_edge_dicts(u: str, v: str) -> Iterable[dict[str, Any]]:
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            return ()
        if G.is_multigraph():
            if isinstance(edge_data, dict):
                return tuple(
                    attrs for attrs in edge_data.values() if isinstance(attrs, dict)
                )
            return ()
        if isinstance(edge_data, dict):
            return (edge_data,)
        return ()

    def _expected_source_for_relation(relation: str | None) -> str | None:
        bucket = relation_bucket(relation)
        if bucket == "explicit_ifc":
            return "ifc"
        if bucket == "topology":
            if relation in {"space_bounded_by", "bounds_space", "path_connected_to"}:
                return "ifc"
            return "topology"
        if bucket == "spatial":
            return "heuristic"
        return None

    def _edge_source(edge: dict[str, Any], relation: str | None = None) -> str | None:
        canonical_relation = relation if relation in CANONICAL_RELATION_SET else None
        if canonical_relation is None:
            canonical_relation = _edge_relation(edge)

        bucket = relation_bucket(canonical_relation)
        if bucket is None or bucket == "hierarchy":
            return None

        expected_source = _expected_source_for_relation(canonical_relation)
        if expected_source is not None:
            return expected_source

        source = normalize_relation_source(edge.get("source"))
        if source in KNOWN_RELATION_SOURCE_SET:
            return source
        return None

    def _resolve_element_id(
        element_id: str,
    ) -> tuple[str | None, Dict[str, Any] | None]:
        if not isinstance(element_id, str):
            return None, {"error": "Invalid element_id: element_id must be a string"}
        if element_id in G:
            if str(element_id).startswith("Element::"):
                return element_id, None
            return None, {
                "error": f"Invalid element_id (not an element): {element_id}"
            }
        if not element_id.startswith("Element::"):
            candidate = f"Element::{element_id}"
            if candidate in G:
                return candidate, None
        matches = []
        for n, d in G.nodes(data=True):
            if not str(n).startswith("Element::"):
                continue
            gid = d.get("properties", {}).get("GlobalId")
            if gid == element_id:
                matches.append(n)
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, {"error": "Ambiguous element_id", "candidates": matches}
        return None, {"error": f"Element not found: {element_id}"}

    def _resolve_storey_node(
        storey_query: str,
    ) -> tuple[str | None, Dict[str, Any] | None]:
        query = storey_query.strip()
        if not query:
            return None, {"error": "Missing param: storey"}

        # Direct id match supports both raw GlobalId and Storey::<GlobalId>.
        direct = query if query.startswith("Storey::") else f"Storey::{query}"
        if direct in G and (
            str(G.nodes[direct].get("class_", "")).lower() == "ifcbuildingstorey"
        ):
            return direct, None

        # Exact label match (legacy/user-friendly).
        exact = _find_nodes_by_label(query, class_filter="IfcBuildingStorey")
        if len(exact) == 1:
            return exact[0], None
        if len(exact) > 1:
            return None, {"error": "Ambiguous storey", "candidates": exact}

        # Normalized label fallback (spaces/punctuation/case).
        qn = _normalize_text(query)
        norm_matches: list[str] = []
        for n, d in G.nodes(data=True):
            if str(d.get("class_", "")).lower() != "ifcbuildingstorey":
                continue
            if _normalize_text(str(d.get("label", ""))) == qn:
                norm_matches.append(n)
        if len(norm_matches) == 1:
            return norm_matches[0], None
        if len(norm_matches) > 1:
            return None, {"error": "Ambiguous storey", "candidates": norm_matches}

        return None, {"error": f"Storey not found: {storey_query}"}

    def _storey_elements(start: str) -> Iterable[str]:
        """Traverse downward storey containment only (contains edges)."""
        visited = {start}
        q = deque([start])
        while q:
            node = q.popleft()
            for nbr in G.successors(node):
                has_contains = any(
                    normalize_relation_name(edge.get("relation")) == "contains"
                    for edge in _iter_edge_dicts(node, nbr)
                )
                if not has_contains:
                    continue
                if nbr in visited:
                    continue
                visited.add(nbr)
                q.append(nbr)
                yield nbr

    def _spatial_neighbors(node_id: str) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Yield unique spatial neighbors across both outgoing and incoming edges."""
        seen: set[tuple[str, str, str]] = set()
        spatial_relations = set(SPATIAL_RELATIONS)

        for nbr in G.successors(node_id):
            for edge in _iter_edge_dicts(node_id, nbr):
                relation = normalize_relation_name(edge.get("relation"))
                if relation not in spatial_relations:
                    continue
                dedupe_key = (nbr, relation or "", str(edge.get("distance")))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                yield nbr, edge

        for nbr in G.predecessors(node_id):
            for edge in _iter_edge_dicts(nbr, node_id):
                relation = normalize_relation_name(edge.get("relation"))
                if relation not in spatial_relations:
                    continue
                dedupe_key = (nbr, relation or "", str(edge.get("distance")))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                yield nbr, edge

    def _topology_neighbors(
        node_id: str,
        allowed_relations: set[str] | None = None,
    ) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Yield unique topology neighbors across both directions."""
        seen: set[tuple[str, str, str]] = set()
        topology_relations = set(TOPOLOGY_RELATIONS)
        if allowed_relations is None:
            allowed_relations = topology_relations

        for nbr in G.successors(node_id):
            for edge in _iter_edge_dicts(node_id, nbr):
                relation = normalize_relation_name(edge.get("relation"))
                if relation not in allowed_relations:
                    continue
                if relation is None:
                    continue
                key = (
                    nbr,
                    relation,
                    str(
                        edge.get("vertical_gap")
                        or edge.get("intersection_volume")
                        or ""
                    ),
                )
                if key in seen:
                    continue
                seen.add(key)
                yield nbr, edge

        for nbr in G.predecessors(node_id):
            for edge in _iter_edge_dicts(nbr, node_id):
                relation = normalize_relation_name(edge.get("relation"))
                if relation not in allowed_relations:
                    continue
                if relation is None:
                    continue
                key = (
                    nbr,
                    relation,
                    str(
                        edge.get("vertical_gap")
                        or edge.get("intersection_volume")
                        or ""
                    ),
                )
                if key in seen:
                    continue
                seen.add(key)
                yield nbr, edge

    def _apply_property_filters(
        node_id: str,
        filters: Dict[str, Any],
        *,
        db_conn: Any | None = None,
    ) -> bool:
        return apply_property_filters(G, node_id, filters, db_conn=db_conn)

    if action == "get_elements_in_storey":
        storey = params.get("storey")
        if not storey:
            return _err("Missing param: storey", "missing_param")
        if not isinstance(storey, str):
            return _err("Invalid param: storey must be a string", "invalid")
        node, err = _resolve_storey_node(storey)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            return _err(str(error_msg), "not_found")
        if node is None:
            return _err(f"Storey not found: {storey}", "not_found")

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

        elements = []
        for e in _storey_elements(node):
            cls = G.nodes[e].get("class_")
            if cls in container_classes:
                continue
            elements.append(
                {
                    "id": e,
                    "label": G.nodes[e].get("label"),
                    "class_": cls,
                }
            )
        return _ok_action(action, {"storey": storey, "elements": elements})

    if action == "find_elements_by_class":
        cls = params.get("class")
        if not cls:
            return _err("Missing param: class", "missing_param")
        if not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        target = _normalize_class(cls)

        matches = []
        for n, d in G.nodes(data=True):
            if str(d.get("class_", "")).lower() == target.lower():
                matches.append(
                    build_node_payload(n, d, payload_mode=resolved_payload_mode)
                )
        return _ok_action(action, {"class": target, "elements": matches})

    if action == "get_adjacent_elements":
        element_id = params.get("element_id")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")
        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        neighbors = []
        for nbr, edge in _spatial_neighbors(resolved):
            edge_relation = _edge_relation(edge)
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "distance": edge.get("distance"),
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(action, {"element_id": resolved, "adjacent": neighbors})

    if action == "get_topology_neighbors":
        element_id = params.get("element_id")
        relation = params.get("relation")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")
        if not relation:
            return _err("Missing param: relation", "missing_param")
        if not isinstance(relation, str):
            return _err("Invalid param: relation must be a string", "invalid")

        relation_value = normalize_relation_name(relation)
        allowed = set(TOPOLOGY_RELATIONS)
        if relation_value not in allowed:
            return _err(
                f"Unsupported topology relation: {relation}",
                "invalid",
                {"allowed_relations": sorted(allowed)},
            )

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        neighbors = []
        for nbr, edge in _topology_neighbors(resolved, {relation_value}):
            edge_relation = _edge_relation(edge)
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "vertical_gap": edge.get("vertical_gap"),
                    "overlap_area_xy": edge.get("overlap_area_xy"),
                    "intersection_volume": edge.get("intersection_volume"),
                    "contact_area": edge.get("contact_area"),
                    "source": _edge_source(edge, edge_relation),
                }
            )

        return _ok_action(
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
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        neighbors = []
        for nbr, edge in _topology_neighbors(resolved, {"intersects_3d"}):
            edge_relation = _edge_relation(edge)
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "intersection_volume": edge.get("intersection_volume"),
                    "contact_area": edge.get("contact_area"),
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(
            action, {"element_id": resolved, "intersections_3d": neighbors}
        )

    if action == "find_nodes":
        cls = params.get("class")
        if cls is not None and not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        class_filter = _normalize_class(cls) if cls else None
        property_filters = params.get("property_filters", {})
        if property_filters and not isinstance(property_filters, dict):
            return _err("Invalid param: property_filters must be an object", "invalid")

        db_lookup_conn: Any | None = None
        db_path_raw: Any = G.graph.get("_db_path")
        if property_filters and db_path_raw is not None:
            from rag_tag.sql_element_lookup import (  # noqa: PLC0415
                open_lookup_connection,
            )

            db_lookup_conn = open_lookup_connection(Path(db_path_raw))

        matches = []
        try:
            for n, d in G.nodes(data=True):
                if class_filter is not None:
                    if str(d.get("class_", "")).lower() != class_filter.lower():
                        continue
                if not _apply_property_filters(
                    n,
                    property_filters,
                    db_conn=db_lookup_conn,
                ):
                    continue
                matches.append(
                    build_node_payload(n, d, payload_mode=resolved_payload_mode)
                )
        finally:
            if db_lookup_conn is not None:
                db_lookup_conn.close()
        return _ok_action(action, {"class": class_filter, "elements": matches})

    if action == "traverse":
        start = params.get("start")
        relation_param = params.get("relation")
        if not start:
            return _err("Missing param: start", "missing_param")
        if not isinstance(start, str):
            return _err("Invalid param: start must be a string", "invalid")
        if relation_param is not None and not isinstance(relation_param, str):
            return _err("Invalid param: relation must be a string", "invalid")
        try:
            depth = int(params.get("depth", 1))
        except (TypeError, ValueError):
            return _err("Invalid param: depth must be an integer", "invalid")
        if start not in G:
            return _err(f"Start node not found: {start}", "not_found")
        if depth < 1:
            return _err("Depth must be >= 1", "invalid")

        visited = {start}
        frontier = {start}
        results = []
        relation_filter: set[str] | None = None
        relation_value: str | None = None
        if relation_param is not None:
            relation_value = normalize_relation_name(relation_param)
            if relation_value is None:
                return _err("Invalid param: relation must be non-empty", "invalid")
            if relation_value not in CANONICAL_RELATION_SET:
                return _err(
                    f"Unsupported traverse relation: {relation_param}",
                    "invalid",
                    {"allowed_relations": sorted(CANONICAL_RELATION_SET)},
                )
            relation_filter = {relation_value}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for nbr in G.successors(node):
                    matched_edges = []
                    for edge in _iter_edge_dicts(node, nbr):
                        edge_relation = _edge_relation(edge)
                        if edge_relation is None:
                            continue
                        if relation_filter and edge_relation not in relation_filter:
                            continue
                        matched_edges.append((edge, edge_relation))
                    if not matched_edges:
                        continue
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    edge, edge_relation = matched_edges[0]
                    results.append(
                        {
                            "from": node,
                            "to": nbr,
                            "relation": edge_relation,
                            "source": _edge_source(edge, edge_relation),
                            "node": build_node_payload(
                                nbr,
                                G.nodes[nbr],
                                payload_mode=resolved_payload_mode,
                            ),
                        }
                    )
            frontier = next_frontier

        return _ok_action(
            action,
            {
                "start": start,
                "relation": relation_value,
                "depth": depth,
                "results": results,
            },
        )

    if action == "spatial_query":
        cls = params.get("class")
        if cls is not None and not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        class_filter = _normalize_class(cls) if cls else None
        near = params.get("near")
        max_distance = params.get("max_distance")
        if near is None:
            return _err("Missing param: near", "missing_param")
        if not isinstance(near, (str, int, float)):
            return _err("Invalid param: near must be a string or number", "invalid")
        resolved, err = _resolve_element_id(str(near))
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {near}", "not_found")
        if max_distance is None:
            return _err("Missing param: max_distance", "missing_param")
        try:
            max_distance_value = float(max_distance)
        except (TypeError, ValueError):
            return _err("Invalid param: max_distance must be a number", "invalid")

        results = []
        for nbr, edge in _spatial_neighbors(resolved):
            dist = edge.get("distance")
            if dist is None or float(dist) > max_distance_value:
                continue
            if class_filter is not None:
                if str(G.nodes[nbr].get("class_", "")).lower() != class_filter.lower():
                    continue
            edge_relation = _edge_relation(edge)
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "distance": dist,
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(
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
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")
        max_gap_value: float | None = None
        if max_gap is not None:
            try:
                max_gap_value = float(max_gap)
            except (TypeError, ValueError):
                return _err("Invalid param: max_gap must be a number", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        results = []
        for nbr, edge in _topology_neighbors(resolved, {"above"}):
            gap = edge.get("vertical_gap")
            if (
                max_gap_value is not None
                and gap is not None
                and float(gap) > max_gap_value
            ):
                continue
            edge_relation = _edge_relation(edge)
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "vertical_gap": gap,
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(
            action,
            {
                "element_id": resolved,
                "max_gap": max_gap_value,
                "results": results,
            },
        )

    if action == "find_elements_below":
        element_id = params.get("element_id")
        max_gap = params.get("max_gap")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")
        max_gap_value: float | None = None
        if max_gap is not None:
            try:
                max_gap_value = float(max_gap)
            except (TypeError, ValueError):
                return _err("Invalid param: max_gap must be a number", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        results = []
        for nbr, edge in _topology_neighbors(resolved, {"below"}):
            gap = edge.get("vertical_gap")
            if (
                max_gap_value is not None
                and gap is not None
                and float(gap) > max_gap_value
            ):
                continue
            edge_relation = _edge_relation(edge)
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "vertical_gap": gap,
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(
            action,
            {
                "element_id": resolved,
                "max_gap": max_gap_value,
                "results": results,
            },
        )

    if action == "list_property_keys":
        cls = params.get("class")
        sample_values = params.get("sample_values", False)

        if cls is not None and not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        if not isinstance(sample_values, bool):
            return _err("Invalid param: sample_values must be a boolean", "invalid")

        class_filter = _normalize_class(cls) if cls else None
        key_samples: dict[str, list[Any]] = {}

        def _record_key(key: str, value: Any) -> None:
            if key not in key_samples:
                key_samples[key] = []
            if sample_values and len(key_samples[key]) < 3:
                key_samples[key].append(value)

        def _collect_pset_leaf_keys(
            pset_name: str,
            node: dict[str, Any],
            path_prefix: str = "",
        ) -> None:
            for raw_key, raw_value in node.items():
                key_part = str(raw_key)
                path = f"{path_prefix}.{key_part}" if path_prefix else key_part
                if isinstance(raw_value, dict):
                    _collect_pset_leaf_keys(pset_name, raw_value, path)
                else:
                    _record_key(f"{pset_name}.{path}", raw_value)

        for _, data in G.nodes(data=True):
            if class_filter is not None:
                if str(data.get("class_", "")).lower() != class_filter.lower():
                    continue

            props = data.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            for key, value in props.items():
                _record_key(str(key), value)

            payload = data.get("payload") or {}
            if not isinstance(payload, dict):
                continue

            pset_block = payload.get("PropertySets") or {}
            if isinstance(pset_block, dict):
                for section in ("Official", "Custom"):
                    section_block = pset_block.get(section) or {}
                    if not isinstance(section_block, dict):
                        continue
                    for pset_name, pset_props in section_block.items():
                        if not isinstance(pset_props, dict):
                            continue
                        _collect_pset_leaf_keys(str(pset_name), pset_props)

            quantities_block = payload.get("Quantities") or {}
            if not isinstance(quantities_block, dict):
                continue
            for qto_name, qto_data in quantities_block.items():
                if not isinstance(qto_data, dict):
                    continue
                _collect_pset_leaf_keys(str(qto_name), qto_data)

        payload_mode_value = str(G.graph.get("_payload_mode", "full")).lower()
        db_path_raw = G.graph.get("_db_path")
        if payload_mode_value == "minimal" and db_path_raw is not None:
            db_path = Path(db_path_raw)
            cache = G.graph.setdefault("_property_key_cache", {})
            cache_key = (str(db_path.resolve()), class_filter or "")
            db_key_samples = cache.get(cache_key)
            if db_key_samples is None:
                db_key_samples = _collect_dotted_keys_from_sqlite(
                    db_path, class_filter
                )
                cache[cache_key] = db_key_samples

            for key, samples in db_key_samples.items():
                if sample_values:
                    existing = key_samples.setdefault(key, [])
                    for sample in samples:
                        if len(existing) >= 3:
                            break
                        existing.append(sample)
                else:
                    _record_key(key, None)

        data: dict[str, Any] = {
            "keys": sorted(key_samples.keys()),
            "class_filter": class_filter,
            "class_filter_raw": cls,
        }
        if sample_values:
            data["samples"] = key_samples
        return _ok_action(action, data)

    if action == "get_element_properties":
        element_id = params.get("element_id")
        if element_id is None or element_id == "":
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if not resolved:
            return _err(f"Element not found: {element_id}", "not_found")

        base_node_data: dict[str, Any] = dict(G.nodes[resolved])

        # Attempt DB-backed property enrichment when a DB path is wired into
        # the graph context.  Uses the session-level cache to avoid reopening
        # the database for elements already fetched during filter evaluation.
        db_path_raw: Any = G.graph.get("_db_path")
        if db_path_raw is not None:
            db_path = Path(db_path_raw)
            db_data: dict[str, Any] | None = _cached_db_lookup(G, resolved, db_path)
            if db_data is not None:
                base_node_data = _merge_db_element_data(base_node_data, db_data)
            else:
                _logger.debug(
                    "get_element_properties: DB lookup found no row for %s"
                    " (db=%s) — using in-memory payload",
                    resolved,
                    db_path,
                )

        return _ok_action(
            action,
            build_node_payload(
                resolved, base_node_data, payload_mode=INTERNAL_PAYLOAD_MODE
            ),
        )

    return _err(
        f"Unknown action: {action}",
        "unknown_action",
        {"allowed_actions": sorted(CANONICAL_ACTIONS)},
    )
