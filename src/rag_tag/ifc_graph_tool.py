from __future__ import annotations

import re
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

    def _compact_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _extract_numeric_tokens(value: str) -> set[str]:
        return set(re.findall(r"\d+", value))

    def _extract_ordinal_tokens(value: str) -> set[int]:
        ordinals = set()
        for raw in re.findall(r"\b(\d+)(?:st|nd|rd|th)\b", value):
            ordinals.add(int(raw))
        return ordinals

    def _all_storey_nodes() -> list[str]:
        nodes = []
        for n, d in G.nodes(data=True):
            if not str(n).startswith("Storey::"):
                continue
            if str(d.get("class_", "")).lower() == "ifcbuildingstorey":
                nodes.append(n)
        return nodes

    def _resolve_storey_node(
        storey_query: str,
    ) -> tuple[str | None, Dict[str, Any] | None]:
        query = storey_query.strip()
        if not query:
            return None, {"error": "Missing param: storey"}

        direct_node = f"Storey::{query}"
        if direct_node in G:
            cls = str(G.nodes[direct_node].get("class_", ""))
            if cls.lower() == "ifcbuildingstorey":
                return direct_node, None

        exact_matches = _find_nodes_by_label(query, class_filter="IfcBuildingStorey")
        if len(exact_matches) == 1:
            return exact_matches[0], None
        if len(exact_matches) > 1:
            return None, {"error": "Ambiguous storey", "candidates": exact_matches}

        query_norm = _normalize_text(query)
        query_compact = _compact_text(query)
        query_tokens = set(query_norm.split())
        query_nums = _extract_numeric_tokens(query_norm)
        query_ords = _extract_ordinal_tokens(query_norm)

        norm_exact: list[str] = []
        token_matches: list[tuple[int, str]] = []
        for node in _all_storey_nodes():
            label = str(G.nodes[node].get("label", ""))
            label_norm = _normalize_text(label)
            label_compact = _compact_text(label)
            label_tokens = set(label_norm.split())
            label_nums = _extract_numeric_tokens(label_norm)
            label_ords = _extract_ordinal_tokens(label_norm)

            if label_norm == query_norm:
                norm_exact.append(node)
                continue

            score = 0
            if query_tokens and query_tokens.issubset(label_tokens):
                score += 2
            if query_nums and query_nums.intersection(label_nums):
                score += 2
            if query_ords and query_ords.intersection(label_ords):
                score += 2
            if "ground" in query_tokens and (
                "ground" in label_tokens or "00" in label_nums
            ):
                score += 2
            if query_norm and query_norm in label_norm:
                score += 1
            if query_compact and query_compact in label_compact:
                score += 1
            if score > 0:
                token_matches.append((score, node))

        if len(norm_exact) == 1:
            return norm_exact[0], None
        if len(norm_exact) > 1:
            return None, {"error": "Ambiguous storey", "candidates": norm_exact}

        if token_matches:
            token_matches.sort(key=lambda item: item[0], reverse=True)
            best_score = token_matches[0][0]
            best = [node for score, node in token_matches if score == best_score]
            if len(best) == 1:
                return best[0], None
            return None, {"error": "Ambiguous storey", "candidates": best}

        return None, {"error": f"Storey not found: {storey_query}"}

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

    def _descendants(start: str) -> Iterable[str]:
        return nx.descendants(G, start)

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
            return {"error": "Missing param: storey"}
        node, err = _resolve_storey_node(str(storey))
        if err:
            return err
        if node is None:
            return {"error": f"Storey not found: {storey}"}

        class_filter = params.get("class")
        normalized_class = _normalize_class(str(class_filter)) if class_filter else None

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
        for e in _descendants(node):
            cls = G.nodes[e].get("class_")
            if cls in container_classes:
                continue
            if normalized_class is not None:
                if str(cls or "").lower() != normalized_class.lower():
                    continue
            elements.append(
                {
                    "id": e,
                    "label": G.nodes[e].get("label"),
                    "class_": cls,
                }
            )
        return {
            "storey": storey,
            "storey_node": node,
            "storey_class": "IfcBuildingStorey",
            "class": normalized_class,
            "elements": elements,
        }

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
        for nbr in G.neighbors(resolved):
            edge = G[resolved][nbr]
            if edge.get("relation") == "adjacent_to":
                neighbors.append(
                    {
                        "id": nbr,
                        "label": G.nodes[nbr].get("label"),
                        "class_": G.nodes[nbr].get("class_"),
                        "distance": edge.get("distance"),
                    }
                )
        return _ok({"element_id": resolved, "adjacent": neighbors})

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
        for nbr in G.neighbors(resolved):
            edge = G[resolved][nbr]
            if edge.get("relation") != "adjacent_to":
                continue
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

    return _err(f"Unknown action: {action}", "unknown_action")
