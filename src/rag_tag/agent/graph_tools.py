"""PydanticAI tools for graph queries.

Tools in this module call the backend-agnostic action interface in
``rag_tag.ifc_graph_tool.query_ifc_graph`` and preserve the canonical
``{status,data,error}`` envelope.

The tool layer adds light UX helpers only (fuzzy matching / class normalization)
without changing public action names or response contracts.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from pydantic_ai import RunContext
from rapidfuzz import fuzz, process

from rag_tag.graph import GraphRuntime, get_networkx_graph
from rag_tag.graph.payloads import sanitize_properties_for_llm
from rag_tag.graph_contract import make_error_envelope, normalize_relation_name
from rag_tag.ifc_graph_tool import query_ifc_graph

# Minimum rapidfuzz WRatio score (0-100) to accept a fuzzy class normalisation.
_CLASS_FUZZY_THRESHOLD = 72
_TYPE_INTENT_TERMS = ("type", "family", "template", "style", "kind")
_CONTAINER_CLASSES = {
    "IfcProject",
    "IfcSite",
    "IfcBuilding",
    "IfcBuildingStorey",
    "IfcSpace",
    "IfcZone",
    "IfcSpatialZone",
    "IfcTypeObject",
}
_ZONE_CONTAINER_CLASSES = {"IfcZone", "IfcSpatialZone"}
_CONTAINER_EDGE_RELATIONS = {"contains", "aggregates"}


# ---------------------------------------------------------------------------
# Module-level helpers (not registered as tools)
# ---------------------------------------------------------------------------


def _all_class_values(G) -> list[str]:
    """Return sorted unique IFC class_ values present in the graph."""
    return sorted({str(d["class_"]) for _, d in G.nodes(data=True) if d.get("class_")})


def _normalize_class_fuzzy(class_input: str, G) -> tuple[str | None, float]:
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
    G,
    query: str,
    class_filter: str | None = None,
    top_k: int = 10,
    min_score: float = 50.0,
) -> dict[str, Any]:
    """Score nodes by fuzzy-matching query against label, ObjectType, Description.

    Returns a standard envelope dict (status/data/error).
    """
    # Detect if query mentions "type" to bias scoring toward/away from type nodes.
    query_lower = query.lower()
    mentions_type = "type" in query_lower

    results: list[dict[str, Any]] = []
    for node_id, data in G.nodes(data=True):
        if class_filter is not None:
            if str(data.get("class_", "")).lower() != class_filter.lower():
                continue

        props: dict[str, Any] = data.get("properties", {}) or {}
        payload: dict[str, Any] = data.get("payload") or {}
        candidates = [
            str(data.get("label", "")),
            str(props.get("ObjectType", "")),
            str(props.get("Description", "")),
            str(payload.get("Name", "")),
            str(payload.get("IfcType", "")),
            str(payload.get("ClassRaw", "")),
        ]

        # Include individual material names so queries like
        # "made of gypsum fiber-board" can surface the right element.
        for mat_name in props.get("Materials") or payload.get("Materials") or []:
            if mat_name:
                candidates.append(str(mat_name))

        best_score = max(
            (fuzz.WRatio(query, c) for c in candidates if c and c != "None"),
            default=0.0,
        )

        # Apply scoring bias to prefer occurrences or types based on query intent.
        is_type_node = str(data.get("class_", "")).endswith("Type")
        if mentions_type:
            # Query mentions "type": boost type nodes, penalize occurrences.
            if is_type_node:
                best_score += 5
            else:
                best_score -= 5
        else:
            # Query does NOT mention "type": boost occurrences, penalize types.
            if is_type_node:
                best_score -= 5
            else:
                best_score += 5

        if best_score >= min_score:
            results.append(
                {
                    "id": node_id,
                    "label": data.get("label"),
                    "class_": data.get("class_"),
                    "score": round(best_score, 1),
                    "properties": sanitize_properties_for_llm(props),
                    "payload": None,
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


def _find_container_elements_excluding_impl(
    runtime_or_graph: GraphRuntime | Any,
    container_or_query_fn: str | Any,
    container_id: str | None = None,
    *,
    exclude_container_ids: list[str] | None = None,
    depth: int = 4,
) -> dict[str, Any]:
    """Return non-container members of one container minus excluded containers.

    Supports both the current runtime-oriented call shape:
    ``(runtime, container_id, ...)`` and the older test helper shape:
    ``(graph, query_fn, container_id, ...)``.
    """
    graph = get_networkx_graph(runtime_or_graph)

    if callable(container_or_query_fn):
        _legacy_query_fn = container_or_query_fn
        del _legacy_query_fn
        resolved_container_id = container_id
    else:
        resolved_container_id = container_or_query_fn

    if not isinstance(resolved_container_id, str):
        return make_error_envelope(
            "Container not found: None",
            "not_found",
        )

    if resolved_container_id not in graph:
        return make_error_envelope(
            f"Container not found: {resolved_container_id}",
            "not_found",
        )

    exclude_container_ids = exclude_container_ids or []
    missing = [node_id for node_id in exclude_container_ids if node_id not in graph]
    if missing:
        return make_error_envelope(
            f"Excluded container not found: {missing[0]}",
            "not_found",
        )

    if depth < 1:
        return make_error_envelope(
            "Depth must be at least 1.",
            "invalid",
        )

    included = _collect_container_descendants(
        graph, [resolved_container_id], depth=depth
    )
    excluded = _collect_container_descendants(
        graph,
        exclude_container_ids,
        depth=depth,
    )
    element_ids = sorted(
        included - excluded,
        key=lambda node_id: (
            str(graph.nodes[node_id].get("label") or ""),
            str(node_id),
        ),
    )

    elements = [
        {
            "id": node_id,
            "label": graph.nodes[node_id].get("label"),
            "class_": graph.nodes[node_id].get("class_"),
            "properties": sanitize_properties_for_llm(
                graph.nodes[node_id].get("properties") or {}
            ),
            "payload": None,
        }
        for node_id in element_ids
    ]

    return {
        "status": "ok",
        "data": {
            "container_id": resolved_container_id,
            "exclude_container_ids": exclude_container_ids,
            "count": len(element_ids),
            "elements": elements,
            "results": elements,
        },
        "error": None,
    }


def _collect_container_descendants(
    graph: Any,
    start_ids: list[str],
    *,
    depth: int,
) -> set[str]:
    """Collect non-container descendants reachable via container membership edges."""
    if not start_ids:
        return set()

    seen = set(start_ids)
    descendants: set[str] = set()
    queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in start_ids)

    while queue:
        node_id, node_depth = queue.popleft()
        if node_depth >= depth:
            continue

        if graph.nodes[node_id].get("class_") in _ZONE_CONTAINER_CLASSES:
            for source_id, _target, edge_data in graph.in_edges(node_id, data=True):
                relation = normalize_relation_name(edge_data.get("relation"))
                if relation != "in_zone":
                    continue
                if source_id in seen:
                    continue

                seen.add(source_id)
                queue.append((source_id, node_depth + 1))

                if graph.nodes[source_id].get("class_") in _CONTAINER_CLASSES:
                    continue
                descendants.add(source_id)

        for _src, target_id, edge_data in graph.out_edges(node_id, data=True):
            relation = normalize_relation_name(edge_data.get("relation"))
            if relation not in _CONTAINER_EDGE_RELATIONS:
                continue
            if target_id in seen:
                continue

            seen.add(target_id)
            queue.append((target_id, node_depth + 1))

            if graph.nodes[target_id].get("class_") in _CONTAINER_CLASSES:
                continue
            descendants.add(target_id)

    return descendants


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
        ctx: RunContext[GraphRuntime],
        query: str,
        class_filter: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Fuzzy-search for graph nodes by matching query against common text fields."""
        graph = ctx.deps.get_networkx_graph()
        return _fuzzy_find_nodes_impl(
            graph, query, class_filter=class_filter, top_k=top_k
        )

    @agent.tool
    def find_nodes(
        ctx: RunContext[GraphRuntime],
        class_: str | None = None,
        property_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Find nodes in the IFC graph by class and/or property filters.

        Multi-word class_ input is treated as a descriptive query and routed to
        fuzzy_find_nodes. class_ is fuzzy-normalised against classes present in
        the graph before querying.

        ``property_filters`` supports two key formats:
        - Flat key (e.g. ``{"Name": "Wall A"}``): matched against the node's
          direct ``properties`` dict first, then searched across all psets.
        - Dotted key (e.g. ``{"Pset_WallCommon.FireRating": "EI 90"}``): targets
          a specific named PropertySet.  Use ``list_property_keys`` to discover
          valid keys.

        A missing key never matches an expected value of ``None``; filters only
        pass when the key is explicitly present with the expected value.
        """
        G = ctx.deps.get_networkx_graph()

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

        result = ctx.deps.query("find_nodes", params)

        # Exact match empty -> fuzzy fallback over names/descriptions.
        if (
            result.get("status") == "ok"
            and len((result.get("data") or {}).get("elements", [])) == 0
            and class_
        ):
            if property_filters:
                return make_error_envelope(
                    (
                        f"Exact match for properties {property_filters} failed. "
                        "The value might be formatted differently in the raw "
                        "data. Try using 'fuzzy_find_nodes' instead."
                    ),
                    "no_exact_property_match",
                )

        return result

    @agent.tool
    def traverse(
        ctx: RunContext[GraphRuntime],
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
        return ctx.deps.query("traverse", params)

    @agent.tool
    def spatial_query(
        ctx: RunContext[GraphRuntime],
        near: str,
        max_distance: float,
        class_: str | None = None,
    ) -> dict[str, Any]:
        """Find elements within a spatial distance of a reference element."""
        params: dict[str, Any] = {"near": near, "max_distance": max_distance}
        if class_:
            params["class"] = class_
        return ctx.deps.query("spatial_query", params)

    @agent.tool
    def spatial_compare(
        ctx: RunContext[GraphRuntime],
        element_a: str,
        element_b: str,
    ) -> dict[str, Any]:
        """Compare two elements using stored geometry and derived relation metrics."""
        return ctx.deps.query(
            "spatial_compare",
            {"element_a": element_a, "element_b": element_b},
        )

    @agent.tool
    def find_elements_within_clearance(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        max_distance: float,
        class_: str | None = None,
        measure: str = "surface",
    ) -> dict[str, Any]:
        """Find elements within a clearance threshold using geometry-aware distance."""
        params: dict[str, Any] = {
            "element_id": element_id,
            "max_distance": max_distance,
            "measure": measure,
        }
        if class_:
            params["class"] = class_
        return ctx.deps.query("find_elements_within_clearance", params)

    @agent.tool
    def get_elements_in_storey(
        ctx: RunContext[GraphRuntime],
        storey: str,
    ) -> dict[str, Any]:
        """Get all non-container elements in a specific storey/level."""
        return query_ifc_graph(ctx.deps, "get_elements_in_storey", {"storey": storey})

    @agent.tool
    def find_container_elements_excluding(
        ctx: RunContext[GraphRuntime],
        container_id: str,
        exclude_container_ids: list[str] | None = None,
        depth: int = 4,
    ) -> dict[str, Any]:
        """List elements inside one container while excluding other containers.

        Use this for questions like:
        - elements in a building but not in a given storey
        - elements in a space/zone/building excluding another container subset
        """
        return _find_container_elements_excluding_impl(
            ctx.deps,
            container_id,
            exclude_container_ids=exclude_container_ids,
            depth=depth,
        )

    @agent.tool
    def find_elements_by_class(
        ctx: RunContext[GraphRuntime],
        class_: str,
    ) -> dict[str, Any]:
        """Find all elements of a specific IFC class."""
        return ctx.deps.query("find_elements_by_class", {"class": class_})

    @agent.tool
    def get_adjacent_elements(
        ctx: RunContext[GraphRuntime],
        element_id: str,
    ) -> dict[str, Any]:
        """Get elements spatially adjacent to a given element."""
        return ctx.deps.query("get_adjacent_elements", {"element_id": element_id})

    @agent.tool
    def get_topology_neighbors(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        relation: str,
    ) -> dict[str, Any]:
        """Get topology neighbors for one relation.

        relation must be one of: above, below, overlaps_xy, intersects_bbox,
        intersects_3d, touches_surface, supports, supported_by, rests_on,
        parallel_to, perpendicular_to, facing, inside_3d, contains_3d,
        space_bounded_by, bounds_space, path_connected_to.
        """
        return ctx.deps.query(
            "get_topology_neighbors",
            {"element_id": element_id, "relation": relation},
        )

    @agent.tool
    def get_intersections_3d(
        ctx: RunContext[GraphRuntime],
        element_id: str,
    ) -> dict[str, Any]:
        """Get mesh-informed 3D intersection neighbors for an element."""
        return ctx.deps.query("get_intersections_3d", {"element_id": element_id})

    @agent.tool
    def find_elements_above(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        max_gap: float | None = None,
    ) -> dict[str, Any]:
        """Find elements above a reference element."""
        params: dict[str, Any] = {"element_id": element_id}
        if max_gap is not None:
            params["max_gap"] = max_gap
        return ctx.deps.query("find_elements_above", params)

    @agent.tool
    def find_elements_below(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        max_gap: float | None = None,
    ) -> dict[str, Any]:
        """Find elements below a reference element."""
        params: dict[str, Any] = {"element_id": element_id}
        if max_gap is not None:
            params["max_gap"] = max_gap
        return ctx.deps.query("find_elements_below", params)

    @agent.tool
    def list_property_keys(
        ctx: RunContext[GraphRuntime],
        class_: str | None = None,
        sample_values: bool = False,
    ) -> dict[str, Any]:
        """Discover filterable flat and dotted property keys.

        Delegates to the canonical ``list_property_keys`` graph action so key
        discovery uses the same backend-agnostic contract path as filtering.
        """
        params: dict[str, Any] = {"sample_values": sample_values}
        if class_ is not None:
            params["class"] = class_
        return ctx.deps.query("list_property_keys", params)

    @agent.tool
    def get_element_properties(
        ctx: RunContext[GraphRuntime],
        element_id: str,
    ) -> dict[str, Any]:
        """Fetch ALL properties for a specific element (DB-backed when available).

        Looks up *element_id* in the SQLite database wired into the graph
        context (when available), then merges with the in-memory graph
        payload.  Falls back to in-memory data only when no DB is configured.
        Returns the full, unredacted property envelope including PropertySets,
        Quantities, and flat properties.
        """
        return ctx.deps.query("get_element_properties", {"element_id": element_id})
