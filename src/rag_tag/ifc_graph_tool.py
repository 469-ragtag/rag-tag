from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, Iterable

import networkx as nx


def _ok(data: dict) -> dict[str, Any]:
    """Wrap successful tool result in envelope."""
    return {"status": "ok", "data": data, "error": None}


def _err(message: str, code: str, details: dict | None = None) -> dict[str, Any]:
    """Wrap error result in envelope."""
    error_payload: dict[str, Any] = {"message": message, "code": code}
    if details:
        error_payload["details"] = details
    return {"status": "error", "data": None, "error": error_payload}


def query_ifc_graph(
    G: nx.DiGraph, action: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Controlled interface between the LLM and NetworkX graph."""

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

    def _resolve_element_id(
        element_id: str,
    ) -> tuple[str | None, Dict[str, Any] | None]:
        if element_id in G:
            return element_id, None
        if not element_id.startswith("Element::"):
            candidate = f"Element::{element_id}"
            if candidate in G:
                return candidate, None
        matches = []
        for n, d in G.nodes(data=True):
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

        # Direct id match supports new Storey::<GlobalId> identity.
        direct = f"Storey::{query}"
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
        """Traverse only containment edges to avoid leakage via spatial links."""
        visited = {start}
        q = deque([start])
        while q:
            node = q.popleft()
            for nbr in G.successors(node):
                edge = G[node][nbr]
                if edge.get("relation") != "contained_in":
                    continue
                if nbr in visited:
                    continue
                visited.add(nbr)
                q.append(nbr)
                yield nbr

    def _spatial_neighbors(node_id: str) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Yield unique spatial neighbors across both outgoing and incoming edges."""
        seen: set[str] = set()
        spatial_relations = {"adjacent_to", "connected_to"}

        for nbr in G.successors(node_id):
            if nbr in seen:
                continue
            edge = G[node_id][nbr]
            if edge.get("relation") not in spatial_relations:
                continue
            seen.add(nbr)
            yield nbr, edge

        for nbr in G.predecessors(node_id):
            if nbr in seen:
                continue
            edge = G[nbr][node_id]
            if edge.get("relation") not in spatial_relations:
                continue
            seen.add(nbr)
            yield nbr, edge

    def _topology_neighbors(
        node_id: str,
        allowed_relations: set[str] | None = None,
    ) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Yield unique topology neighbors across both directions."""
        seen: set[tuple[str, str]] = set()
        topology_relations = {"above", "below", "overlaps_xy", "intersects_bbox"}
        if allowed_relations is None:
            allowed_relations = topology_relations

        for nbr in G.successors(node_id):
            edge = G[node_id][nbr]
            relation = str(edge.get("relation"))
            if relation not in allowed_relations:
                continue
            key = (nbr, relation)
            if key in seen:
                continue
            seen.add(key)
            yield nbr, edge

        for nbr in G.predecessors(node_id):
            edge = G[nbr][node_id]
            relation = str(edge.get("relation"))
            if relation not in allowed_relations:
                continue
            key = (nbr, relation)
            if key in seen:
                continue
            seen.add(key)
            yield nbr, edge

    def _node_payload(node_id: str) -> Dict[str, Any]:
        data = G.nodes[node_id]
        return {
            "id": node_id,
            "label": data.get("label"),
            "class_": data.get("class_"),
            "properties": data.get("properties", {}),
        }

    def _apply_property_filters(node_id: str, filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        props = G.nodes[node_id].get("properties", {})
        for key, expected in filters.items():
            if props.get(key) != expected:
                return False
        return True

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
        return _ok({"storey": storey, "elements": elements})

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
                    {
                        "id": n,
                        "label": d.get("label"),
                        "class_": d.get("class_"),
                        "properties": d.get("properties", {}),
                    }
                )
        return _ok({"class": target, "elements": matches})

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
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        neighbors = []
        for nbr, edge in _spatial_neighbors(resolved):
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge.get("relation"),
                    "distance": edge.get("distance"),
                }
            )
        return _ok({"element_id": resolved, "adjacent": neighbors})

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

        relation_value = relation.strip().lower()
        allowed = {"above", "below", "overlaps_xy", "intersects_bbox"}
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
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        neighbors = []
        for nbr, edge in _topology_neighbors(resolved, {relation_value}):
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge.get("relation"),
                    "vertical_gap": edge.get("vertical_gap"),
                    "overlap_area_xy": edge.get("overlap_area_xy"),
                    "source": edge.get("source"),
                }
            )

        return _ok(
            {
                "element_id": resolved,
                "relation": relation_value,
                "neighbors": neighbors,
            }
        )

    if action == "find_nodes":
        cls = params.get("class")
        if cls is not None and not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        class_filter = _normalize_class(cls) if cls else None
        property_filters = params.get("property_filters", {})
        if property_filters and not isinstance(property_filters, dict):
            return _err("Invalid param: property_filters must be an object", "invalid")

        matches = []
        for n, d in G.nodes(data=True):
            if class_filter is not None:
                if str(d.get("class_", "")).lower() != class_filter.lower():
                    continue
            if not _apply_property_filters(n, property_filters):
                continue
            matches.append(_node_payload(n))
        return _ok({"class": class_filter, "elements": matches})

    if action == "traverse":
        start = params.get("start")
        relation = params.get("relation")
        if not start:
            return _err("Missing param: start", "missing_param")
        if not isinstance(start, str):
            return _err("Invalid param: start must be a string", "invalid")
        if relation is not None and not isinstance(relation, str):
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

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for nbr in G.successors(node):
                    edge = G[node][nbr]
                    if relation and edge.get("relation") != relation:
                        continue
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    results.append(
                        {
                            "from": node,
                            "to": nbr,
                            "relation": edge.get("relation"),
                            "node": _node_payload(nbr),
                        }
                    )
            frontier = next_frontier

        return _ok(
            {
                "start": start,
                "relation": relation,
                "depth": depth,
                "results": results,
            }
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
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge.get("relation"),
                    "distance": dist,
                }
            )
        return _ok(
            {
                "near": resolved,
                "max_distance": max_distance_value,
                "results": results,
            }
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
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": "above",
                    "vertical_gap": gap,
                }
            )
        return _ok(
            {
                "element_id": resolved,
                "max_gap": max_gap_value,
                "results": results,
            }
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
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": "below",
                    "vertical_gap": gap,
                }
            )
        return _ok(
            {
                "element_id": resolved,
                "max_gap": max_gap_value,
                "results": results,
            }
        )

    return _err(f"Unknown action: {action}", "unknown_action")
