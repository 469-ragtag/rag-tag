"""PydanticAI tools for graph queries.

Tools in this module call the backend-agnostic action interface in
``rag_tag.ifc_graph_tool.query_ifc_graph`` and preserve the canonical
``{status,data,error}`` envelope.

The tool layer adds light UX helpers only (fuzzy matching / class normalization)
without changing public action names or response contracts.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any

from pydantic_ai import RunContext
from rapidfuzz import fuzz, process

from rag_tag.graph import GraphRuntime, get_networkx_graph
from rag_tag.graph.payloads import sanitize_properties_for_llm
from rag_tag.graph_contract import (
    collect_evidence,
    make_error_envelope,
    normalize_relation_name,
)
from rag_tag.ifc_graph_tool import query_ifc_graph

# Minimum rapidfuzz WRatio score (0-100) to accept a fuzzy class normalisation.
_CLASS_FUZZY_THRESHOLD = 72
_TYPE_INTENT_TERMS = ("type", "family", "template", "style", "kind")
_LEGACY_SCAN_MAX_RESULTS = 50
_LEGACY_NEIGHBOR_MAX_RESULTS = 25
_QUERY_TERM_PATTERN = re.compile(r"[a-z0-9]+")
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
_GENERIC_CONTAINER_STOPWORDS = {
    "a",
    "an",
    "any",
    "at",
    "by",
    "for",
    "in",
    "inside",
    "is",
    "near",
    "of",
    "on",
    "outside",
    "that",
    "the",
    "this",
    "to",
}
_GENERIC_CONTAINER_INTENTS = {
    "IfcBuilding": {"building"},
    "IfcSite": {"site"},
    "IfcBuildingStorey": {"floor", "level", "storey"},
    "IfcSpace": {"room", "space"},
    "IfcProject": {"project"},
}
_SECONDARY_CONTAINER_CLASSES = {
    "IfcProject",
    "IfcSite",
    "IfcBuilding",
    "IfcBuildingStorey",
    "IfcSpace",
}
_GENERIC_CONTAINER_CANONICAL_BOOST = 60.0
_GENERIC_CONTAINER_SECONDARY_BOOST = 4.0
_GENERIC_CONTAINER_PROXY_PENALTY = 18.0


# ---------------------------------------------------------------------------
# Module-level helpers (not registered as tools)
# ---------------------------------------------------------------------------


def _all_class_values(runtime: GraphRuntime) -> list[str]:
    """Return sorted unique IFC class_ values present in the graph."""
    graph = get_networkx_graph(runtime)
    return sorted(
        {str(d["class_"]) for _, d in graph.nodes(data=True) if d.get("class_")}
    )


def _normalize_class_fuzzy(
    class_input: str,
    runtime: GraphRuntime,
) -> tuple[str | None, float]:
    """Match a user-supplied class name against actual class_ values in the graph.

    Returns:
        (best_match, score) where best_match is None if no match exceeds the
        threshold.
    """
    known = _all_class_values(runtime)
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


def _class_is_type_like(class_name: str | None) -> bool:
    """Return True when *class_name* denotes an IFC type object."""
    if not class_name:
        return False
    return class_name.endswith("Type") or class_name == "IfcTypeObject"


def _query_has_type_intent(query: str) -> bool:
    """Return True when the user text explicitly asks about a type/family."""
    query_terms = {term for term in query.lower().replace("-", " ").split() if term}
    return any(term in query_terms for term in _TYPE_INTENT_TERMS)


def _query_terms(query: str) -> list[str]:
    """Return normalized alphanumeric query terms."""
    return _QUERY_TERM_PATTERN.findall(query.lower())


def _generic_container_target_class(query: str) -> str | None:
    """Return the canonical IFC container class for generic container queries.

    The bias is intentionally narrow: only apply when the query is essentially
    a generic container noun phrase like "building" or "the floor", not when it
    is a specific named phrase such as "server room" or "main building".
    """
    content_terms = {
        term for term in _query_terms(query) if term not in _GENERIC_CONTAINER_STOPWORDS
    }
    if not content_terms:
        return None

    matched_targets = [
        class_name
        for class_name, aliases in _GENERIC_CONTAINER_INTENTS.items()
        if content_terms.issubset(aliases)
    ]
    if len(matched_targets) != 1:
        return None
    return matched_targets[0]


def _fuzzy_find_nodes_impl(
    runtime: GraphRuntime,
    query: str,
    class_filter: str | None = None,
    top_k: int = 10,
    min_score: float = 50.0,
) -> dict[str, Any]:
    """Score nodes by fuzzy-matching query against label, ObjectType, Description.

    Returns a standard envelope dict (status/data/error).
    """
    G = get_networkx_graph(runtime)
    results: list[dict[str, Any]] = []
    query_has_type_intent = _query_has_type_intent(query)
    generic_container_target = (
        _generic_container_target_class(query) if class_filter is None else None
    )

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
        adjusted_score = best_score
        class_name = str(data.get("class_", ""))

        if class_filter is None and _class_is_type_like(class_name):
            if query_has_type_intent:
                adjusted_score += 8.0
            else:
                adjusted_score -= 8.0

        if generic_container_target is not None:
            if class_name == generic_container_target:
                adjusted_score += _GENERIC_CONTAINER_CANONICAL_BOOST
            elif class_name in _SECONDARY_CONTAINER_CLASSES:
                adjusted_score += _GENERIC_CONTAINER_SECONDARY_BOOST

            if class_name == "IfcBuildingElementProxy":
                adjusted_score -= _GENERIC_CONTAINER_PROXY_PENALTY

        if adjusted_score >= min_score:
            results.append(
                {
                    "id": node_id,
                    "label": data.get("label"),
                    "class_": data.get("class_"),
                    "score": round(adjusted_score, 1),
                    "properties": sanitize_properties_for_llm(props),
                    "payload": None,
                }
            )

    results.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item.get("label") or ""),
            str(item["id"]),
        )
    )
    visible_results = results[:top_k]
    evidence = collect_evidence(
        visible_results,
        source_tool="fuzzy_find_nodes",
        match_reason_builder=lambda item: f"fuzzy_score={item['score']}",
    )
    return {
        "status": "ok",
        "data": {
            "query": query,
            "class_filter": class_filter,
            "matches": visible_results,
            "total": len(results),
            "evidence": evidence,
        },
        "error": None,
    }


def _find_container_elements_excluding_impl(
    runtime: GraphRuntime,
    container_id: str,
    exclude_container_ids: list[str] | None = None,
    depth: int = 4,
) -> dict[str, Any]:
    """Return non-container members of one container minus excluded containers."""
    graph = get_networkx_graph(runtime)

    if container_id not in graph:
        return make_error_envelope(
            f"Container not found: {container_id}",
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

    included = _collect_container_descendants(graph, [container_id], depth=depth)
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
            "container_id": container_id,
            "exclude_container_ids": exclude_container_ids,
            "count": len(element_ids),
            "elements": elements,
            "evidence": collect_evidence(
                elements,
                source_tool="find_container_elements_excluding",
            ),
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
        return _fuzzy_find_nodes_impl(
            ctx.deps, query, class_filter=class_filter, top_k=top_k
        )

    @agent.tool
    def find_nodes(
        ctx: RunContext[GraphRuntime],
        class_: str | None = None,
        property_filters: dict[str, Any] | None = None,
        max_results: int = _LEGACY_SCAN_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Find nodes in the IFC graph by class and/or property filters.

        Multi-word class_ input is treated as a descriptive query and routed to
        fuzzy_find_nodes. class_ is fuzzy-normalised against classes present in
        the graph before querying. Returns a bounded candidate set and may mark
        `data.truncated=true` when more matches exist.

        ``property_filters`` supports two key formats:
        - Flat key (e.g. ``{"Name": "Wall A"}``): matched against the node's
          direct ``properties`` dict first, then searched across all psets.
        - Dotted key (e.g. ``{"Pset_WallCommon.FireRating": "EI 90"}``): targets
          a specific named PropertySet.  Use ``list_property_keys`` to discover
          valid keys.

        A missing key never matches an expected value of ``None``; filters only
        pass when the key is explicitly present with the expected value.
        """
        # Multi-word input means descriptive name, not IFC class.
        if class_ and " " in class_.strip():
            return _fuzzy_find_nodes_impl(ctx.deps, class_, top_k=max_results)

        normalized_class = class_
        if class_:
            best, _score = _normalize_class_fuzzy(class_, ctx.deps)
            if best is not None:
                normalized_class = best

        params: dict[str, Any] = {}
        if normalized_class:
            params["class"] = normalized_class
        if property_filters:
            params["property_filters"] = property_filters
        params["max_results"] = max_results

        result = query_ifc_graph(ctx.deps, "find_nodes", params)

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
        max_results: int = _LEGACY_NEIGHBOR_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Traverse the graph from a starting node following edges.

        For location/storey lookup:
        - relation='contains' to move from storey/space -> contained elements
        - relation='contained_in' to move from element -> containing structure
        Results are bounded and may be partial when `data.truncated=true`.
        """
        params: dict[str, Any] = {
            "start": start,
            "depth": depth,
            "max_results": max_results,
        }
        if relation:
            params["relation"] = relation
        return query_ifc_graph(ctx.deps, "traverse", params)

    @agent.tool
    def spatial_query(
        ctx: RunContext[GraphRuntime],
        near: str,
        max_distance: float,
        class_: str | None = None,
        max_results: int = _LEGACY_SCAN_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Find bounded nearby elements within a spatial distance."""
        params: dict[str, Any] = {
            "near": near,
            "max_distance": max_distance,
            "max_results": max_results,
        }
        if class_:
            params["class"] = class_
        return query_ifc_graph(ctx.deps, "spatial_query", params)

    @agent.tool
    def get_elements_in_storey(
        ctx: RunContext[GraphRuntime],
        storey: str,
        max_results: int = _LEGACY_SCAN_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Get a bounded list of non-container elements in a specific storey/level."""
        return query_ifc_graph(
            ctx.deps,
            "get_elements_in_storey",
            {"storey": storey, "max_results": max_results},
        )

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
        max_results: int = _LEGACY_SCAN_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Find a bounded set of elements of a specific IFC class."""
        return query_ifc_graph(
            ctx.deps,
            "find_elements_by_class",
            {"class": class_, "max_results": max_results},
        )

    @agent.tool
    def get_adjacent_elements(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        max_results: int = _LEGACY_NEIGHBOR_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Get a bounded adjacency set for a given element."""
        return query_ifc_graph(
            ctx.deps,
            "get_adjacent_elements",
            {"element_id": element_id, "max_results": max_results},
        )

    @agent.tool
    def get_topology_neighbors(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        relation: str,
        max_results: int = _LEGACY_NEIGHBOR_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Get topology neighbors for one relation.

        relation must be one of: above, below, overlaps_xy, intersects_bbox,
        intersects_3d, touches_surface, space_bounded_by, bounds_space,
        path_connected_to. Returns a bounded candidate list. Depending on the
        graph-build overlap policy, `overlaps_xy` may legitimately return no
        neighbors even when vertical `above` / `below` edges exist.
        """
        return query_ifc_graph(
            ctx.deps,
            "get_topology_neighbors",
            {
                "element_id": element_id,
                "relation": relation,
                "max_results": max_results,
            },
        )

    @agent.tool
    def get_intersections_3d(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        max_results: int = _LEGACY_NEIGHBOR_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Get a bounded mesh-informed 3D intersection set for an element."""
        return query_ifc_graph(
            ctx.deps,
            "get_intersections_3d",
            {"element_id": element_id, "max_results": max_results},
        )

    @agent.tool
    def find_elements_above(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        max_gap: float | None = None,
        max_results: int = _LEGACY_NEIGHBOR_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Find a bounded list of elements above a reference element."""
        params: dict[str, Any] = {
            "element_id": element_id,
            "max_results": max_results,
        }
        if max_gap is not None:
            params["max_gap"] = max_gap
        return query_ifc_graph(ctx.deps, "find_elements_above", params)

    @agent.tool
    def find_elements_below(
        ctx: RunContext[GraphRuntime],
        element_id: str,
        max_gap: float | None = None,
        max_results: int = _LEGACY_NEIGHBOR_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Find a bounded list of elements below a reference element."""
        params: dict[str, Any] = {
            "element_id": element_id,
            "max_results": max_results,
        }
        if max_gap is not None:
            params["max_gap"] = max_gap
        return query_ifc_graph(ctx.deps, "find_elements_below", params)

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
        return query_ifc_graph(ctx.deps, "list_property_keys", params)

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
        return query_ifc_graph(
            ctx.deps, "get_element_properties", {"element_id": element_id}
        )

    @agent.tool
    def trace_distribution_network(
        ctx: RunContext[GraphRuntime],
        start: str,
        max_depth: int = 3,
        relations: list[str] | None = None,
        max_results: int = 25,
    ) -> dict[str, Any]:
        """Trace a bounded distribution/network path set from a starting node.

        Prefer this over repeated `traverse` calls for network-style questions
        such as "what is connected downstream of this terminal?" or "trace this
        branch of the system".
        """
        params: dict[str, Any] = {
            "start": start,
            "max_depth": max_depth,
            "max_results": max_results,
        }
        if relations:
            params["relations"] = relations
        return query_ifc_graph(ctx.deps, "trace_distribution_network", params)

    @agent.tool
    def find_equipment_serving_space(
        ctx: RunContext[GraphRuntime],
        space: str,
        max_depth: int = 4,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Find likely equipment serving a space using bounded graph reasoning.

        Use this for HVAC/MEP-style questions about what serves a room/space.
        It combines space boundary clues, terminal discovery, and bounded network
        tracing instead of forcing the model to compose many low-level traversals.
        """
        return query_ifc_graph(
            ctx.deps,
            "find_equipment_serving_space",
            {
                "space": space,
                "max_depth": max_depth,
                "max_results": max_results,
            },
        )

    @agent.tool
    def find_shortest_path(
        ctx: RunContext[GraphRuntime],
        start: str,
        end: str,
        max_path_length: int = 8,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute a bounded shortest path between two graph anchors.

        Prefer this over manual hop-by-hop traversal when the user asks for the
        connection/path between two known elements or context nodes.
        """
        params: dict[str, Any] = {
            "start": start,
            "end": end,
            "max_path_length": max_path_length,
        }
        if relations:
            params["relations"] = relations
        return query_ifc_graph(ctx.deps, "find_shortest_path", params)

    @agent.tool
    def find_by_classification(
        ctx: RunContext[GraphRuntime],
        classification: str,
        max_results: int = 25,
    ) -> dict[str, Any]:
        """Find elements linked to matching classification context nodes.

        Use this when the question mentions a classification label, code, or
        reference rather than an IFC class name.
        """
        return query_ifc_graph(
            ctx.deps,
            "find_by_classification",
            {
                "classification": classification,
                "max_results": max_results,
            },
        )

    @agent.tool
    def aggregate_elements(
        ctx: RunContext[GraphRuntime],
        element_ids: list[str],
        metric: str,
        field: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate an exact graph-discovered element set via the SQLite context.

        Use this after graph tools return element IDs or GlobalIds and the user
        asks for a deterministic count, sum, average, minimum, or maximum over
        that exact set. Do not count or sum in the prompt when this tool fits.
        """
        params: dict[str, Any] = {
            "element_ids": element_ids,
            "metric": metric,
        }
        if field is not None:
            params["field"] = field
        return query_ifc_graph(ctx.deps, "aggregate_elements", params)

    @agent.tool
    def group_elements_by_property(
        ctx: RunContext[GraphRuntime],
        element_ids: list[str],
        property_key: str,
        max_groups: int = 20,
    ) -> dict[str, Any]:
        """Group an exact graph-discovered element set by one DB-backed field.

        Use this after graph discovery when the user asks for a deterministic
        breakdown by level, type, property, or quantity value.
        """
        return query_ifc_graph(
            ctx.deps,
            "group_elements_by_property",
            {
                "element_ids": element_ids,
                "property_key": property_key,
                "max_groups": max_groups,
            },
        )
