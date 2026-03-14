"""PydanticAI tools for graph queries.

Tools in this module call the backend-agnostic action interface in
``rag_tag.ifc_graph_tool.query_ifc_graph`` and preserve the canonical
``{status,data,error}`` envelope.

The tool layer adds light UX helpers only (fuzzy matching / class normalization)
without changing public action names or response contracts.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext
from rapidfuzz import fuzz, process

from rag_tag.graph import GraphRuntime
from rag_tag.graph_contract import make_error_envelope, make_ok_envelope
from rag_tag.ifc_graph_tool import build_node_payload, sanitize_properties_for_llm

# Minimum rapidfuzz WRatio score (0-100) to accept a fuzzy class normalisation.
_CLASS_FUZZY_THRESHOLD = 72


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


def _resolve_node_id_for_tool(
    G,
    node_query: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve a graph node id from an exact id or unique GlobalId."""
    if not isinstance(node_query, str):
        return None, {"error": "Invalid node id: expected string"}

    if node_query in G:
        return node_query, None

    prefixes = ("Element::", "Storey::", "System::", "Zone::", "Classification::")
    for prefix in prefixes:
        candidate = f"{prefix}{node_query}"
        if candidate in G:
            return candidate, None

    matches: list[str] = []
    for node_id, data in G.nodes(data=True):
        props = data.get("properties") or {}
        if isinstance(props, dict) and props.get("GlobalId") == node_query:
            matches.append(str(node_id))
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, {"error": "Ambiguous node id", "candidates": matches}
    return None, {"error": f"Node not found: {node_query}"}


def _is_container_node(node_id: str, node_data: dict[str, Any]) -> bool:
    """Return True when the node acts as a spatial/container context."""
    class_name = str(node_data.get("class_", ""))
    if node_id.startswith(("System::", "Zone::", "Classification::")):
        return True
    return class_name in {
        "IfcProject",
        "IfcSite",
        "IfcBuilding",
        "IfcBuildingStorey",
        "IfcSpace",
        "IfcZone",
        "IfcSpatialZone",
        "IfcTypeObject",
        "IfcGroup",
    }


def _collect_container_member_ids(
    G,
    query_fn,
    container_id: str,
    *,
    depth: int,
) -> tuple[set[str], dict[str, Any] | None]:
    """Collect non-container members under a container using existing graph tools."""
    members: set[str] = set()

    for relation in ("contains", "aggregates"):
        result = query_fn(
            "traverse",
            {"start": container_id, "relation": relation, "depth": depth},
        )
        if result.get("status") != "ok":
            continue
        payload = result.get("data") or {}
        for item in payload.get("results") or []:
            node = item.get("node") or {}
            node_id = node.get("id")
            if not isinstance(node_id, str) or node_id not in G:
                continue
            if _is_container_node(node_id, G.nodes[node_id]):
                continue
            members.add(node_id)

    container_class = str(G.nodes[container_id].get("class_", ""))
    if container_class in {"IfcZone", "IfcSpatialZone"} or container_id.startswith(
        "Zone::"
    ):
        for node_id, node_data in G.nodes(data=True):
            if not str(node_id).startswith("Element::"):
                continue
            if _is_container_node(str(node_id), node_data):
                continue
            zone_result = query_fn(
                "traverse",
                {"start": str(node_id), "relation": "in_zone", "depth": 1},
            )
            if zone_result.get("status") != "ok":
                continue
            payload = zone_result.get("data") or {}
            for item in payload.get("results") or []:
                zone_node = item.get("node") or {}
                if zone_node.get("id") == container_id:
                    members.add(str(node_id))
                    break

    return members, None


def _find_container_elements_excluding_impl(
    G,
    query_fn,
    container_id: str,
    *,
    exclude_container_ids: list[str] | None = None,
    depth: int = 3,
) -> dict[str, Any]:
    """Return non-container elements in a container minus excluded subcontainers."""
    if not isinstance(container_id, str) or not container_id.strip():
        return make_error_envelope(
            "Missing param: container_id",
            "missing_param",
        )
    if exclude_container_ids is None:
        exclude_container_ids = []
    if not isinstance(exclude_container_ids, list) or any(
        not isinstance(item, str) for item in exclude_container_ids
    ):
        return make_error_envelope(
            "Invalid param: exclude_container_ids must be a list of strings",
            "invalid",
        )
    try:
        depth_value = int(depth)
    except (TypeError, ValueError):
        return make_error_envelope("Invalid param: depth must be an integer", "invalid")
    if depth_value < 1:
        return make_error_envelope("Depth must be >= 1", "invalid")

    resolved_container_id, err = _resolve_node_id_for_tool(G, container_id)
    if err:
        error_msg = str(err.get("error", "Unknown error"))
        code = "ambiguous" if "Ambiguous" in error_msg else "not_found"
        return make_error_envelope(
            error_msg,
            code,
            {"candidates": err.get("candidates", [])} if code == "ambiguous" else None,
        )
    if resolved_container_id is None:
        return make_error_envelope(
            f"Node not found: {container_id}",
            "not_found",
        )

    resolved_exclude_ids: list[str] = []
    for exclude_id in exclude_container_ids:
        resolved_exclude_id, err = _resolve_node_id_for_tool(G, exclude_id)
        if err:
            error_msg = str(err.get("error", "Unknown error"))
            code = "ambiguous" if "Ambiguous" in error_msg else "not_found"
            return make_error_envelope(
                error_msg,
                code,
                {"candidates": err.get("candidates", [])}
                if code == "ambiguous"
                else None,
            )
        if resolved_exclude_id is not None:
            resolved_exclude_ids.append(resolved_exclude_id)

    included_ids, _err = _collect_container_member_ids(
        G,
        query_fn,
        resolved_container_id,
        depth=depth_value,
    )
    for exclude_id in resolved_exclude_ids:
        excluded_ids, _err = _collect_container_member_ids(
            G,
            query_fn,
            exclude_id,
            depth=depth_value,
        )
        included_ids.difference_update(excluded_ids)

    results = [
        build_node_payload(node_id, G.nodes[node_id])
        for node_id in sorted(
            included_ids,
            key=lambda current_id: (
                str(G.nodes[current_id].get("label", "")),
                current_id,
            ),
        )
    ]
    return make_ok_envelope(
        "find_container_elements_excluding",
        {
            "container_id": resolved_container_id,
            "exclude_container_ids": resolved_exclude_ids,
            "depth": depth_value,
            "results": results,
        },
    )


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
    def get_elements_in_storey(
        ctx: RunContext[GraphRuntime],
        storey: str,
    ) -> dict[str, Any]:
        """Get all non-container elements in a specific storey/level."""
        return ctx.deps.query("get_elements_in_storey", {"storey": storey})

    @agent.tool
    def find_container_elements_excluding(
        ctx: RunContext[GraphRuntime],
        container_id: str,
        exclude_container_ids: list[str] | None = None,
        depth: int = 3,
    ) -> dict[str, Any]:
        """Find non-container members in a container, excluding subcontainers."""
        graph = ctx.deps.get_networkx_graph()
        return _find_container_elements_excluding_impl(
            graph,
            ctx.deps.query,
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
        intersects_3d, touches_surface, space_bounded_by, bounds_space,
        path_connected_to.
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
