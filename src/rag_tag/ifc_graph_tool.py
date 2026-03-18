from __future__ import annotations

from typing import Any

from rag_tag.graph import (
    INTERNAL_PAYLOAD_MODE,
    LLM_PAYLOAD_MODE,
    GraphRuntime,
    build_node_payload,
    query_graph_runtime,
    sanitize_llm_property_value,
    sanitize_properties_for_llm,
    wrap_networkx_graph,
)


def query_ifc_graph(
    graph: GraphRuntime | Any,
    action: str,
    params: dict[str, Any],
    *,
    payload_mode: str = LLM_PAYLOAD_MODE,
) -> dict[str, Any]:
    """Public graph action facade with compatibility for raw graph catalogs."""
    runtime = graph if isinstance(graph, GraphRuntime) else wrap_networkx_graph(graph)
    return query_graph_runtime(runtime, action, params, payload_mode=payload_mode)


__all__ = [
    "INTERNAL_PAYLOAD_MODE",
    "LLM_PAYLOAD_MODE",
    "build_node_payload",
    "query_ifc_graph",
    "sanitize_llm_property_value",
    "sanitize_properties_for_llm",
]
