from __future__ import annotations
from typing import Any, Dict, Iterable
import networkx as nx


def query_ifc_graph(G: nx.DiGraph, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
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

    def _resolve_element_id(element_id: str) -> tuple[str | None, Dict[str, Any] | None]:
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
        node = f"Storey::{storey}"
        if node not in G:
            storey_nodes = _find_nodes_by_label(storey, class_filter="IfcBuildingStorey")
            if len(storey_nodes) == 1:
                node = storey_nodes[0]
            elif len(storey_nodes) > 1:
                return {"error": "Ambiguous storey", "candidates": storey_nodes}
            else:
                return {"error": f"Storey not found: {storey}"}

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
            elements.append({
                "id": e,
                "label": G.nodes[e].get("label"),
                "class_": cls,
            })
        return {"storey": storey, "elements": elements}

    if action == "find_elements_by_class":
        cls = params.get("class")
        if not cls:
            return {"error": "Missing param: class"}
        target = _normalize_class(cls)

        matches = []
        for n, d in G.nodes(data=True):
            if str(d.get("class_", "")).lower() == target.lower():
                matches.append({
                    "id": n,
                    "label": d.get("label"),
                    "class_": d.get("class_"),
                    "properties": d.get("properties", {}),
                })
        return {"class": target, "elements": matches}

    if action == "get_adjacent_elements":
        element_id = params.get("element_id")
        if not element_id:
            return {"error": "Missing param: element_id"}
        resolved, err = _resolve_element_id(element_id)
        if err:
            return err
        if resolved is None:
            return {"error": f"Element not found: {element_id}"}

        neighbors = []
        for nbr in G.neighbors(resolved):
            edge = G[resolved][nbr]
            if edge.get("relation") == "adjacent_to":
                neighbors.append({
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "distance": edge.get("distance"),
                })
        return {"element_id": resolved, "adjacent": neighbors}

    if action == "find_nodes":
        cls = params.get("class")
        class_filter = _normalize_class(cls) if cls else None
        property_filters = params.get("property_filters", {})

        matches = []
        for n, d in G.nodes(data=True):
            if class_filter is not None:
                if str(d.get("class_", "")).lower() != class_filter.lower():
                    continue
            if not _apply_property_filters(n, property_filters):
                continue
            matches.append(_node_payload(n))
        return {"class": class_filter, "elements": matches}

    if action == "traverse":
        start = params.get("start")
        relation = params.get("relation")
        depth = int(params.get("depth", 1))
        if not start:
            return {"error": "Missing param: start"}
        if start not in G:
            return {"error": f"Start node not found: {start}"}
        if depth < 1:
            return {"error": "Depth must be >= 1"}

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
                    results.append({
                        "from": node,
                        "to": nbr,
                        "relation": edge.get("relation"),
                        "node": _node_payload(nbr),
                    })
            frontier = next_frontier

        return {"start": start, "relation": relation, "depth": depth, "results": results}

    if action == "spatial_query":
        cls = params.get("class")
        class_filter = _normalize_class(cls) if cls else None
        near = params.get("near")
        max_distance = params.get("max_distance")
        if near is None:
            return {"error": "Missing param: near"}
        resolved, err = _resolve_element_id(str(near))
        if err:
            return err
        if resolved is None:
            return {"error": f"Element not found: {near}"}
        if max_distance is None:
            return {"error": "Missing param: max_distance"}

        results = []
        for nbr in G.neighbors(resolved):
            edge = G[resolved][nbr]
            if edge.get("relation") != "adjacent_to":
                continue
            dist = edge.get("distance")
            if dist is None or float(dist) > float(max_distance):
                continue
            if class_filter is not None:
                if str(G.nodes[nbr].get("class_", "")).lower() != class_filter.lower():
                    continue
            results.append({
                "id": nbr,
                "label": G.nodes[nbr].get("label"),
                "class_": G.nodes[nbr].get("class_"),
                "distance": dist,
            })
        return {"near": resolved, "max_distance": max_distance, "results": results}

    return {"error": f"Unknown action: {action}"}
