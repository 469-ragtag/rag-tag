"""PydanticAI tools for graph agent."""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic_ai import RunContext

from rag_tag.ifc_graph_tool import query_ifc_graph


async def find_nodes(
    ctx: RunContext[nx.DiGraph],
    class_name: str | None = None,
    property_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find nodes by class and optional property filters.

    Args:
        ctx: PydanticAI context with graph
        class_name: IFC class name (e.g., 'IfcWall' or 'Wall')
        property_filters: Optional property filters as dict

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
    relation: str | None = None,
    depth: int = 1,
) -> dict[str, Any]:
    """Traverse graph from a start node following edges.

    Args:
        ctx: PydanticAI context with graph
        start: Starting node ID
        relation: Optional relation filter (e.g., 'contains')
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
    class_name: str | None = None,
) -> dict[str, Any]:
    """Find elements within distance of a reference element.

    Args:
        ctx: PydanticAI context with graph
        near: Reference element ID or GlobalId
        max_distance: Maximum distance threshold
        class_name: Optional IFC class filter

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
    class_name: str,
) -> dict[str, Any]:
    """Find all elements of a specific IFC class.

    Args:
        ctx: PydanticAI context with graph
        class_name: IFC class name (e.g., 'IfcDoor' or 'Door')

    Returns:
        Result envelope with status, data, and error
    """
    params = {"class": class_name}
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
