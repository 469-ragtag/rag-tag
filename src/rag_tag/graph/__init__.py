"""Graph runtime exports with lazy loading to avoid import cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_tag.graph.runtime import GraphRuntime, register_backend

__all__ = [
    "GraphRuntime",
    "INTERNAL_PAYLOAD_MODE",
    "LLM_PAYLOAD_MODE",
    "build_node_payload",
    "register_backend",
    "wrap_networkx_graph",
    "close_runtime",
    "get_networkx_graph",
    "query_graph_runtime",
    "sanitize_llm_property_value",
    "sanitize_properties_for_llm",
]


def __getattr__(name: str):
    if name in __all__:
        from rag_tag.graph.payloads import (
            INTERNAL_PAYLOAD_MODE,
            LLM_PAYLOAD_MODE,
            build_node_payload,
            sanitize_llm_property_value,
            sanitize_properties_for_llm,
        )
        from rag_tag.graph.runtime import (
            GraphRuntime,
            close_runtime,
            get_networkx_graph,
            query_graph_runtime,
            register_backend,
            wrap_networkx_graph,
        )

        exports = {
            "GraphRuntime": GraphRuntime,
            "INTERNAL_PAYLOAD_MODE": INTERNAL_PAYLOAD_MODE,
            "LLM_PAYLOAD_MODE": LLM_PAYLOAD_MODE,
            "build_node_payload": build_node_payload,
            "register_backend": register_backend,
            "wrap_networkx_graph": wrap_networkx_graph,
            "close_runtime": close_runtime,
            "get_networkx_graph": get_networkx_graph,
            "query_graph_runtime": query_graph_runtime,
            "sanitize_llm_property_value": sanitize_llm_property_value,
            "sanitize_properties_for_llm": sanitize_properties_for_llm,
        }
        return exports[name]
    raise AttributeError(name)
