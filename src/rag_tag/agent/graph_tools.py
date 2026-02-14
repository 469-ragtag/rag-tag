"""PydanticAI tools for graph agent."""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic import Field
from pydantic_ai import RunContext

from rag_tag.ifc_graph_tool import query_ifc_graph

REFERENCE_CLASS_ALIASES: dict[str, set[str]] = {
    "house": {"IfcBuilding", "IfcProject", "IfcSite"},
    "building": {"IfcBuilding", "IfcProject", "IfcSite"},
    "site": {"IfcSite", "IfcProject"},
    "project": {"IfcProject"},
    "room": {"IfcSpace", "IfcRoom", "IfcBuildingStorey"},
    "floor": {"IfcBuildingStorey"},
    "storey": {"IfcBuildingStorey"},
}


async def find_nodes(
    ctx: RunContext[nx.DiGraph],
    class_name: str = "",
    property_filters: dict[str, Any] = Field(default_factory=dict),
) -> dict[str, Any]:
    """Find nodes by class and optional property filters.

    Args:
        ctx: PydanticAI context with graph
        class_name: IFC class name (e.g., 'IfcWall' or 'Wall'). Empty means no filter.
        property_filters: Property filters as dict (empty means no filters)

    Returns:
        Result envelope with status, data, and error
    """
    params: dict[str, Any] = {}
    if class_name:
        params["class"] = class_name
    if property_filters:
        params["property_filters"] = property_filters

    return query_ifc_graph(ctx.deps, "find_nodes", params)


async def traverse(
    ctx: RunContext[nx.DiGraph],
    start: str,
    relation: str = "",
    depth: int = 1,
) -> dict[str, Any]:
    """Traverse graph from a start node following edges.

    Args:
        ctx: PydanticAI context with graph
        start: Starting node ID
        relation: Relation filter (e.g., 'contains'). Empty means no filter.
        depth: Traversal depth (default 1)

    Returns:
        Result envelope with status, data, and error
    """
    params: dict[str, Any] = {"start": start, "depth": depth}
    if relation:
        params["relation"] = relation

    return query_ifc_graph(ctx.deps, "traverse", params)


async def spatial_query(
    ctx: RunContext[nx.DiGraph],
    near: str,
    max_distance: float,
    class_name: str = "",
) -> dict[str, Any]:
    """Find elements within distance of a reference element.

    Args:
        ctx: PydanticAI context with graph
        near: Reference element ID or GlobalId
        max_distance: Maximum distance threshold
        class_name: IFC class filter (empty means no filter)

    Returns:
        Result envelope with status, data, and error
    """
    params: dict[str, Any] = {"near": near, "max_distance": max_distance}
    if class_name:
        params["class"] = class_name

    return query_ifc_graph(ctx.deps, "spatial_query", params)


async def get_elements_in_storey(
    ctx: RunContext[nx.DiGraph],
    storey: str,
) -> dict[str, Any]:
    """Get all elements in a building storey.

    Args:
        ctx: PydanticAI context with graph
        storey: Storey name or ID

    Returns:
        Result envelope with status, data, and error
    """
    params = {"storey": storey}
    return query_ifc_graph(ctx.deps, "get_elements_in_storey", params)


async def find_elements_by_class(
    ctx: RunContext[nx.DiGraph],
    class_name: str = "",
) -> dict[str, Any]:
    """Find all elements of a specific IFC class.

    Args:
        ctx: PydanticAI context with graph
        class_name: IFC class name (e.g., 'IfcDoor' or 'Door'). Empty means no filter.

    Returns:
        Result envelope with status, data, and error
    """
    params: dict[str, Any] = {}
    if class_name:
        params["class"] = class_name
    return query_ifc_graph(ctx.deps, "find_elements_by_class", params)


async def get_adjacent_elements(
    ctx: RunContext[nx.DiGraph],
    element_id: str,
) -> dict[str, Any]:
    """Get elements adjacent to a given element.

    Args:
        ctx: PydanticAI context with graph
        element_id: Element ID or GlobalId

    Returns:
        Result envelope with status, data, and error
    """
    params = {"element_id": element_id}
    return query_ifc_graph(ctx.deps, "get_adjacent_elements", params)


async def resolve_entity_reference(
    ctx: RunContext[nx.DiGraph],
    reference: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Resolve a vague user reference (name/alias/id/guid) into graph candidates.

    Args:
        ctx: PydanticAI context with graph
        reference: User-provided entity text (e.g., "house", "kitchen", "3h9x...")
        limit: Maximum number of candidates to return

    Returns:
        Result envelope with status, data, and error
    """
    graph = ctx.deps
    normalized = reference.strip().lower()
    alias_classes = REFERENCE_CLASS_ALIASES.get(normalized, set())

    if not normalized:
        return {
            "status": "error",
            "data": None,
            "error": {
                "message": "Reference text cannot be empty.",
                "code": "INVALID_REFERENCE",
                "details": {"reference": reference},
            },
        }

    candidates: list[dict[str, Any]] = []

    for node_id, attrs in graph.nodes(data=True):
        node_id_str = str(node_id)
        node_id_lower = node_id_str.lower()

        class_name = str(attrs.get("class_") or "")
        class_lower = class_name.lower()
        label = str(attrs.get("label") or "")
        label_lower = label.lower()
        properties = attrs.get("properties") or {}
        properties = properties if isinstance(properties, dict) else {}

        global_id = str(
            properties.get("global_id")
            or properties.get("GlobalId")
            or properties.get("globalId")
            or ""
        )
        global_id_lower = global_id.lower()

        score = 0
        match_reasons: list[str] = []

        if normalized == node_id_lower:
            score += 120
            match_reasons.append("exact_node_id")
        if normalized == global_id_lower and global_id:
            score += 115
            match_reasons.append("exact_global_id")
        if normalized == label_lower and label:
            score += 95
            match_reasons.append("exact_label")

        if normalized in node_id_lower:
            score += 60
            match_reasons.append("node_id_contains")
        if label and normalized in label_lower:
            score += 55
            match_reasons.append("label_contains")
        if global_id and normalized in global_id_lower:
            score += 70
            match_reasons.append("global_id_contains")

        if normalized in class_lower and class_name:
            score += 50
            match_reasons.append("class_contains")

        if class_name in alias_classes:
            score += 65
            match_reasons.append("alias_class")

        if score <= 0:
            continue

        candidates.append(
            {
                "id": node_id_str,
                "label": label,
                "class": class_name,
                "global_id": global_id or None,
                "score": score,
                "match_reasons": match_reasons,
            }
        )

    candidates.sort(
        key=lambda item: (
            int(item.get("score", 0)),
            str(item.get("id", "")),
        ),
        reverse=True,
    )

    return {
        "status": "ok",
        "data": {
            "reference": reference,
            "normalized_reference": normalized,
            "alias_classes": sorted(alias_classes),
            "count": len(candidates),
            "candidates": candidates[: max(1, limit)],
        },
        "error": None,
    }
