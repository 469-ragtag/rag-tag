"""PydanticAI tools for graph queries.

Each tool wraps the existing graph query interface from ifc_graph_tool.py,
preserving the envelope structure (status/data/error) for compatibility.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic_ai import RunContext

from rag_tag.ifc_graph_tool import query_ifc_graph


def register_graph_tools(agent):
    """Register all graph query tools on the given PydanticAI agent.

    Args:
        agent: PydanticAI Agent instance to register tools on
    """

    @agent.tool
    def find_nodes(
        ctx: RunContext[nx.DiGraph],
        class_: str | None = None,
        property_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Find nodes in the IFC graph by class and/or property filters.

        Args:
            class_: IFC class name (e.g., 'IfcDoor', 'Door', 'IfcWindow')
            property_filters: Dictionary of property key-value pairs to filter by

        Returns:
            Envelope with status/data/error. Data contains 'class' and 'elements' list.
        """
        params = {}
        if class_:
            params["class"] = class_
        if property_filters:
            params["property_filters"] = property_filters

        return query_ifc_graph(ctx.deps, "find_nodes", params)

    @agent.tool
    def traverse(
        ctx: RunContext[nx.DiGraph],
        start: str,
        relation: str | None = None,
        depth: int = 1,
    ) -> dict[str, Any]:
        """
        Traverse the graph from a starting node following edges.

        Args:
            start: Starting node ID
            relation: Optional edge relation to filter by (e.g., 'contains')
            depth: Traversal depth (default: 1)

        Returns:
            Envelope with status/data/error. Data contains 'results' list
            with traversal paths.
        """
        params = {"start": start, "depth": depth}
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
        """
        Find elements within a spatial distance of a reference element.

        Args:
            near: Reference element ID (e.g., 'Element::abc123' or GlobalId)
            max_distance: Maximum distance in meters
            class_: Optional IFC class filter

        Returns:
            Envelope with status/data/error. Data contains 'results' list
            with nearby elements.
        """
        params = {"near": near, "max_distance": max_distance}
        if class_:
            params["class"] = class_

        return query_ifc_graph(ctx.deps, "spatial_query", params)

    @agent.tool
    def get_elements_in_storey(
        ctx: RunContext[nx.DiGraph],
        storey: str,
    ) -> dict[str, Any]:
        """
        Get all non-container elements in a specific storey/level.

        Args:
            storey: Storey name (e.g., 'Level 2', 'Ground Floor')

        Returns:
            Envelope with status/data/error. Data contains 'storey' and 'elements' list.
        """
        return query_ifc_graph(ctx.deps, "get_elements_in_storey", {"storey": storey})

    @agent.tool
    def find_elements_by_class(
        ctx: RunContext[nx.DiGraph],
        class_: str,
    ) -> dict[str, Any]:
        """
        Find all elements of a specific IFC class.

        Args:
            class_: IFC class name (e.g., 'IfcDoor', 'Door', 'IfcWindow')

        Returns:
            Envelope with status/data/error. Data contains 'class' and 'elements' list.
        """
        return query_ifc_graph(ctx.deps, "find_elements_by_class", {"class": class_})

    @agent.tool
    def get_adjacent_elements(
        ctx: RunContext[nx.DiGraph],
        element_id: str,
    ) -> dict[str, Any]:
        """
        Get elements spatially adjacent to a given element.

        Args:
            element_id: Element ID (e.g., 'Element::abc123' or GlobalId)

        Returns:
            Envelope with status/data/error. Data contains 'element_id'
            and 'adjacent' list.
        """
        return query_ifc_graph(
            ctx.deps, "get_adjacent_elements", {"element_id": element_id}
        )
