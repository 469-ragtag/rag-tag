"""PydanticAI tools for graph queries.

Each tool wraps the existing graph query interface from ifc_graph_tool.py,
preserving the envelope structure (status/data/error) for compatibility.

Fuzzy normalisation is handled here (in the tool layer) so that ifc_graph_tool.py
remains untouched.  The key improvements are:

- fuzzy_find_nodes: score-ranked text search across Name/ObjectType/Description.
- find_nodes: normalises class_ via rapidfuzz before querying; treats multi-word
  inputs as name searches; falls back to fuzzy_find_nodes when exact query is empty.
- list_property_keys: discovers available property keys to aid filter selection.
- traverse: docstring clarifies contained_in/contains semantics for location queries.
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
        """Fuzzy-search for graph nodes by matching *query* against Name,
        ObjectType, and Description fields.

        Use this tool when:
        - find_nodes returns zero elements.
        - The user supplies a descriptive phrase rather than an exact IFC class
          (e.g. 'plumbing wall', 'main entrance door', 'Level 2').
        - You need to locate a node by its human-readable label.

        Args:
            query: Free-text to match (e.g. 'plumbing wall', 'Level 2').
            class_filter: Optional exact IFC class to restrict results
                          (e.g. 'IfcWall').  Must be CamelCase with no spaces.
            top_k: Maximum number of scored matches to return (default 10).

        Returns:
            Envelope with status/data/error.  data.matches is a list sorted by
            score (descending) with fields: id, label, class_, score, properties.
        """
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

        **Class normalisation** — class_ is fuzzy-matched against class names
        that actually exist in the graph, so short forms ('Wall', 'door') and
        minor misspellings are resolved automatically.

        **Multi-word inputs** — if class_ contains a space (e.g. 'plumbing wall')
        it is treated as a Name/description search and routed to fuzzy_find_nodes
        automatically.

        **Empty-result fallback** — if the exact query returns zero elements,
        the tool re-runs as a fuzzy name search and marks the result with
        data._fallback so you know it is an approximate match.

        Args:
            class_: IFC class name (e.g. 'IfcDoor', 'Door', 'IfcWindow').
                    Do NOT pass multi-word phrases here.
            property_filters: Property key→value pairs to filter by.

        Returns:
            Envelope with status/data/error.  data.elements is the match list.
        """
        G: nx.DiGraph = ctx.deps

        # Multi-word input → descriptive name, not an IFC class.
        if class_ and " " in class_.strip():
            return _fuzzy_find_nodes_impl(G, class_, top_k=20)

        # Fuzzy-normalise class_ against the graph's actual vocabulary.
        normalized_class = class_
        if class_:
            best, _score = _normalize_class_fuzzy(class_, G)
            if best is not None:
                normalized_class = best
            # else keep original string; query_ifc_graph will prepend "Ifc" as
            # a last-resort attempt.

        params: dict[str, Any] = {}
        if normalized_class:
            params["class"] = normalized_class
        if property_filters:
            params["property_filters"] = property_filters

        result = query_ifc_graph(G, "find_nodes", params)

        # If exact query returned nothing, fall back to fuzzy name search.
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

        **Location / containment queries** — to find which storey/floor/space
        an element is in, use relation='contained_in' (child→parent direction).
        Look for IfcBuildingStorey nodes in the results.  Do NOT look for
        location in node properties; it is an edge.

        Common relation values:
        - 'contains'      — container → its children
        - 'contained_in'  — element → its spatial container (storey, space, …)
        - 'adjacent_to'   — spatially adjacent elements
        - 'typed_by'      — element → its IFC type object
        - 'type_of'       — type object → its instances

        Args:
            start: Starting node ID (e.g. 'Element::abc123' or 'Storey::Level 1').
            relation: Edge relation to follow (see above).  None = all edges.
            depth: Traversal depth (default 1).

        Returns:
            Envelope with status/data/error.  data.results is a list of
            {from, to, relation, node} dicts.
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
        """Find elements within a spatial distance of a reference element.

        Args:
            near: Reference element ID (e.g. 'Element::abc123' or GlobalId).
            max_distance: Maximum distance in model units (metres).
            class_: Optional IFC class filter (CamelCase, no spaces).

        Returns:
            Envelope with status/data/error.  data.results is the nearby list.
        """
        params: dict[str, Any] = {"near": near, "max_distance": max_distance}
        if class_:
            params["class"] = class_
        return query_ifc_graph(ctx.deps, "spatial_query", params)

    @agent.tool
    def get_elements_in_storey(
        ctx: RunContext[nx.DiGraph],
        storey: str,
    ) -> dict[str, Any]:
        """Get all non-container elements in a specific storey/level.

        Args:
            storey: Storey name (e.g. 'Level 2', 'Ground Floor').

        Returns:
            Envelope with status/data/error.  data.elements is the element list.
        """
        return query_ifc_graph(ctx.deps, "get_elements_in_storey", {"storey": storey})

    @agent.tool
    def find_elements_by_class(
        ctx: RunContext[nx.DiGraph],
        class_: str,
    ) -> dict[str, Any]:
        """Find all elements of a specific IFC class.

        Args:
            class_: IFC class name (e.g. 'IfcDoor', 'Door', 'IfcWindow').

        Returns:
            Envelope with status/data/error.  data.elements is the element list.
        """
        return query_ifc_graph(ctx.deps, "find_elements_by_class", {"class": class_})

    @agent.tool
    def get_adjacent_elements(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
    ) -> dict[str, Any]:
        """Get elements spatially adjacent to a given element.

        Args:
            element_id: Element ID (e.g. 'Element::abc123' or GlobalId).

        Returns:
            Envelope with status/data/error.  data.adjacent is the neighbour list.
        """
        return query_ifc_graph(
            ctx.deps, "get_adjacent_elements", {"element_id": element_id}
        )

    @agent.tool
    def list_property_keys(
        ctx: RunContext[nx.DiGraph],
        class_: str | None = None,
        sample_values: bool = False,
    ) -> dict[str, Any]:
        """List all property keys present in the graph, optionally scoped to an
        IFC class.  Use this before running find_nodes with property_filters to
        discover the correct key names and check which properties exist.

        Args:
            class_: Optional IFC class to scope the scan (e.g. 'IfcWall').
                    If None, all nodes are scanned.
            sample_values: If True, up to 3 sample values are returned per key.

        Returns:
            Envelope with status/data/error.  data.keys is the sorted list of
            property key names; data.samples (present only when sample_values=True)
            maps key → [value, …].
        """
        G: nx.DiGraph = ctx.deps
        key_samples: dict[str, list[Any]] = {}

        for _, data in G.nodes(data=True):
            if class_ is not None:
                if str(data.get("class_", "")).lower() != class_.lower():
                    continue
            props: dict[str, Any] = data.get("properties", {}) or {}
            for k, v in props.items():
                if k not in key_samples:
                    key_samples[k] = []
                if sample_values and len(key_samples[k]) < 3:
                    key_samples[k].append(v)

        result: dict[str, Any] = {
            "status": "ok",
            "data": {"keys": sorted(key_samples.keys()), "class_filter": class_},
            "error": None,
        }
        if sample_values:
            result["data"]["samples"] = key_samples
        return result
