"""Graph backend runtime exports (lazy to avoid import cycles)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_tag.graph.runtime import GraphRuntime, register_backend

__all__ = ["GraphRuntime", "register_backend"]


def __getattr__(name: str):
    if name in __all__:
        from rag_tag.graph.runtime import GraphRuntime, register_backend

        exports = {"GraphRuntime": GraphRuntime, "register_backend": register_backend}
        return exports[name]
    raise AttributeError(name)
