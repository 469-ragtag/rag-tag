"""PydanticAI tools for graph queries.

Each tool wraps the existing graph query interface from ifc_graph_tool.py,
preserving the envelope structure (status/data/error) for compatibility.

Fuzzy normalisation is handled here (in the tool layer) so that ifc_graph_tool.py
remains untouched. The key improvements are:

- fuzzy_find_nodes: score-ranked text search across Name/ObjectType/Description.
- find_nodes: normalises class_ via rapidfuzz before querying; treats multi-word
  inputs as name searches; falls back to fuzzy_find_nodes when exact query is empty.
- list_property_keys: discovers available property keys to aid filter selection.
- traverse: docstring clarifies contains/contained_in semantics for location queries.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic_ai import RunContext
from rapidfuzz import fuzz, process

from rag_tag.ifc_graph_tool import query_ifc_graph

# Minimum rapidfuzz WRatio score (0-100) to accept a fuzzy class normalisation.
_CLASS_FUZZY_THRESHOLD = 72


# ---------------------------------------------------------------------------
# Module-level helpers (not registered as tools)
# ---------------------------------------------------------------------------


def _all_class_values(G: nx.DiGraph) -> list[str]:
    """Return sorted unique IFC class_ values present in the graph."""
    return sorted({str(d["class_"]) for _, d in G.nodes(data=True) if d.get("class_")})


def _normalize_class_fuzzy(class_input: str, G: nx.DiGraph) -> tuple[str | None, float]:
    """Match a user-supplied class name against actual class_ values in the graph.

    Returns:
        (best_match, score) where best_match is None if no match exceeds the
        threshold.
    """
    known = _all_class_values(G)
    if not known:
        return None, 0.0
    result = process.extractOne(class_input, known, scorer=fuzz.WRatio)
    if result is None:
        return None, 0.0
    best, score, _ = result
    return (
        (best, float(score))
        if score >= _CLASS_FUZZY_THRESHOLD
        else (None, float(score))
    )


def _fuzzy_find_nodes_impl(
    G: nx.DiGraph,
    query: str,
    class_filter: str | None = None,
    top_k: int = 10,
    min_score: float = 50.0,
) -> dict[str, Any]:
    """Score nodes by fuzzy-matching query against label, ObjectType, Description.

    Returns a standard envelope dict (status/data/error).
    """
    results: list[dict[str, Any]] = []
    for node_id, data in G.nodes(data=True):
        if class_filter is not None:
            if str(data.get("class_", "")).lower() != class_filter.lower():
                continue

        props: dict[str, Any] = data.get("properties", {}) or {}
        candidates = [
            str(data.get("label", "")),
            str(props.get("ObjectType", "")),
            str(props.get("Description", "")),
        ]

        best_score = max(
            (fuzz.WRatio(query, c) for c in candidates if c and c != "None"),
            default=0.0,
        )

        if best_score >= min_score:
            results.append(
                {
                    "id": node_id,
                    "label": data.get("label"),
                    "class_": data.get("class_"),
                    "score": round(best_score, 1),
                    "properties": props,
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "status": "ok",
        "data": {
            "query": query,
            "class_filter": class_filter,
            "matches": results[:top_k],
            "total": len(results),
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_graph_tools(agent: Any) -> None:
    """Register all graph query tools on the given PydanticAI agent.

    Args:
        agent: PydanticAI Agent instance to register tools on.
    """

    @agent.tool
    def fuzzy_find_nodes(
        ctx: RunContext[nx.DiGraph],
        query: str,
        class_filter: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Fuzzy-search for graph nodes by matching query against common text fields."""
        return _fuzzy_find_nodes_impl(
            ctx.deps, query, class_filter=class_filter, top_k=top_k
        )

    @agent.tool
    def find_nodes(
        ctx: RunContext[nx.DiGraph],
        class_: str | None = None,
        property_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Find nodes in the IFC graph by class and/or property filters.

        Multi-word class_ input is treated as a descriptive query and routed to
        fuzzy_find_nodes. class_ is fuzzy-normalised against classes present in
        the graph before querying.
        """
        G: nx.DiGraph = ctx.deps

        # Multi-word input means descriptive name, not IFC class.
        if class_ and " " in class_.strip():
            return _fuzzy_find_nodes_impl(G, class_, top_k=20)

        normalized_class = class_
        if class_:
            best, _score = _normalize_class_fuzzy(class_, G)
            if best is not None:
                normalized_class = best

        params: dict[str, Any] = {}
        if normalized_class:
            params["class"] = normalized_class
        if property_filters:
            params["property_filters"] = property_filters

        result = query_ifc_graph(G, "find_nodes", params)

        # Exact match empty -> fuzzy fallback over names/descriptions.
        if (
            result.get("status") == "ok"
            and len((result.get("data") or {}).get("elements", [])) == 0
            and class_
        ):
            fuzzy = _fuzzy_find_nodes_impl(G, class_, top_k=20)
            if (fuzzy.get("data") or {}).get("matches"):
                fuzzy["data"]["_fallback"] = (
                    "exact class match empty; fuzzy name results shown"
                )
            return fuzzy

        return result

    @agent.tool
    def traverse(
        ctx: RunContext[nx.DiGraph],
        start: str,
        relation: str | None = None,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Traverse the graph from a starting node following edges.

        For location/storey lookup:
        - relation='contains' to move from storey/space -> contained elements
        - relation='contained_in' to move from element -> containing structure
        """
        params: dict[str, Any] = {"start": start, "depth": depth}
        if relation:
            params["relation"] = relation
        return query_ifc_graph(ctx.deps, "traverse", params)

    @agent.tool
    def spatial_query(
        ctx: RunContext[nx.DiGraph],
        near: str,
        max_distance: float,
        class_: str | None = None,
    ) -> dict[str, Any]:
        """Find elements within a spatial distance of a reference element."""
        params: dict[str, Any] = {"near": near, "max_distance": max_distance}
        if class_:
            params["class"] = class_
        return query_ifc_graph(ctx.deps, "spatial_query", params)

    @agent.tool
    def get_elements_in_storey(
        ctx: RunContext[nx.DiGraph],
        storey: str,
    ) -> dict[str, Any]:
        """Get all non-container elements in a specific storey/level."""
        return query_ifc_graph(ctx.deps, "get_elements_in_storey", {"storey": storey})

    @agent.tool
    def find_elements_by_class(
        ctx: RunContext[nx.DiGraph],
        class_: str,
    ) -> dict[str, Any]:
        """Find all elements of a specific IFC class."""
        return query_ifc_graph(ctx.deps, "find_elements_by_class", {"class": class_})

    @agent.tool
    def get_adjacent_elements(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
    ) -> dict[str, Any]:
        """Get elements spatially adjacent to a given element."""
        return query_ifc_graph(
            ctx.deps, "get_adjacent_elements", {"element_id": element_id}
        )

    @agent.tool
    def get_topology_neighbors(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
        relation: str,
    ) -> dict[str, Any]:
        """Get topology neighbors for one relation.

        relation must be one of: above, below, overlaps_xy, intersects_bbox,
        intersects_3d, touches_surface.
        """
        return query_ifc_graph(
            ctx.deps,
            "get_topology_neighbors",
            {"element_id": element_id, "relation": relation},
        )

    @agent.tool
    def get_intersections_3d(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
    ) -> dict[str, Any]:
        """Get mesh-informed 3D intersection neighbors for an element."""
        return query_ifc_graph(
            ctx.deps, "get_intersections_3d", {"element_id": element_id}
        )

    @agent.tool
    def find_elements_above(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
        max_gap: float | None = None,
    ) -> dict[str, Any]:
        """Find elements above a reference element."""
        params: dict[str, Any] = {"element_id": element_id}
        if max_gap is not None:
            params["max_gap"] = max_gap
        return query_ifc_graph(ctx.deps, "find_elements_above", params)

    @agent.tool
    def find_elements_below(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
        max_gap: float | None = None,
    ) -> dict[str, Any]:
        """Find elements below a reference element."""
        params: dict[str, Any] = {"element_id": element_id}
        if max_gap is not None:
            params["max_gap"] = max_gap
        return query_ifc_graph(ctx.deps, "find_elements_below", params)

    @agent.tool
    def list_property_keys(
        ctx: RunContext[nx.DiGraph],
        class_: str | None = None,
        sample_values: bool = False,
    ) -> dict[str, Any]:
        """List property keys in the graph, optionally scoped to one IFC class."""
        G: nx.DiGraph = ctx.deps
        key_samples: dict[str, list[Any]] = {}

        for _, data in G.nodes(data=True):
            if class_ is not None:
                if str(data.get("class_", "")).lower() != class_.lower():
                    continue
            props: dict[str, Any] = data.get("properties", {}) or {}
            for key, value in props.items():
                if key not in key_samples:
                    key_samples[key] = []
                if sample_values and len(key_samples[key]) < 3:
                    key_samples[key].append(value)

        result: dict[str, Any] = {
            "status": "ok",
            "data": {"keys": sorted(key_samples.keys()), "class_filter": class_},
            "error": None,
        }
        if sample_values:
            result["data"]["samples"] = key_samples
        return result
