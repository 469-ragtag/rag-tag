"""Graph backend runtime exports (lazy to avoid import cycles)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_tag.graph.runtime import GraphRuntime, register_backend

__all__ = [
    "GraphRuntime",
    "register_backend",
    "wrap_networkx_graph",
    "close_runtime",
]


def __getattr__(name: str):
    if name in __all__:
        from rag_tag.graph.runtime import (
            GraphRuntime,
            register_backend,
            wrap_networkx_graph,
            close_runtime,
        )

        exports = {
            "GraphRuntime": GraphRuntime,
            "register_backend": register_backend,
            "wrap_networkx_graph": wrap_networkx_graph,
            "close_runtime": close_runtime,
        }
        return exports[name]
    raise AttributeError(name)
