"""Graph runtime abstraction with pluggable backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import networkx as nx

from rag_tag.config import GRAPH_BACKEND_ENV_VAR, load_project_config


class GraphBackend(Protocol):
    """Backend protocol implemented by concrete graph backends."""

    def query(
        self,
        action: str,
        params: dict[str, Any],
        *,
        payload_mode: str = "llm",
    ) -> dict[str, Any]: ...

    def get_networkx_graph(self) -> nx.DiGraph | nx.MultiDiGraph: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class NetworkXBackend:
    """NetworkX-backed runtime adapter."""

    graph: nx.DiGraph | nx.MultiDiGraph

    def _delegate_runtime(self, payload_mode: str) -> Any:
        from pathlib import Path

        from rag_tag.graph.backends.networkx_backend import (  # noqa: PLC0415
            NetworkXGraphBackend,
        )
        from rag_tag.graph.types import (
            GraphRuntime as BackendGraphRuntime,  # noqa: PLC0415
        )

        backend = NetworkXGraphBackend()
        datasets = self.graph.graph.get("datasets")
        selected_datasets = (
            sorted(datasets)
            if isinstance(datasets, list)
            and all(isinstance(item, str) for item in datasets)
            else []
        )
        context_db_path = self.graph.graph.get("_db_path")
        resolved_db_path = (
            Path(context_db_path).expanduser().resolve()
            if context_db_path is not None
            else None
        )
        runtime_payload_mode = str(
            self.graph.graph.get("_payload_mode", payload_mode or "full")
        ).lower()
        return BackendGraphRuntime(
            backend_name=backend.name,
            backend=backend,
            selected_datasets=selected_datasets,
            payload_mode=runtime_payload_mode,
            context_db_path=resolved_db_path,
            backend_handle=self.graph,
        )

    def query(
        self,
        action: str,
        params: dict[str, Any],
        *,
        payload_mode: str = "llm",
    ) -> dict[str, Any]:
        runtime = self._delegate_runtime(payload_mode)
        return runtime.backend.query(runtime, action, params, payload_mode)

    def get_networkx_graph(self) -> nx.DiGraph | nx.MultiDiGraph:
        return self.graph

    def close(self) -> None:
        return None


BackendFactory = Callable[..., GraphBackend]


_BACKEND_REGISTRY: dict[str, BackendFactory] = {}
_MODULE_DIR = Path(__file__).resolve().parent


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a backend factory under *name* (lower-cased)."""
    _BACKEND_REGISTRY[name.strip().lower()] = factory


def _default_backend_name(start_dir: Path | None = None) -> str:
    loaded = load_project_config(start_dir or _MODULE_DIR)
    return _resolve_backend_name(loaded.config.defaults.graph_backend)


def _resolve_backend_name(config_default: str | None) -> str:
    env_name = os.environ.get(GRAPH_BACKEND_ENV_VAR)
    if env_name is not None and env_name.strip():
        return env_name.strip().lower()
    if config_default is not None and config_default.strip():
        return config_default.strip().lower()
    return "networkx"


class GraphRuntime:
    """Backend-agnostic graph runtime with a stable action interface."""

    def __init__(self, backend: GraphBackend, name: str) -> None:
        self._backend = backend
        self._name = name

    @classmethod
    def from_env(
        cls,
        *,
        graph: nx.DiGraph | nx.MultiDiGraph | None = None,
        db_path: Path | None = None,
        selected_datasets: list[str] | tuple[str, ...] | None = None,
        start_dir: Path | None = None,
    ) -> "GraphRuntime":
        name = _default_backend_name(start_dir)
        factory = _BACKEND_REGISTRY.get(name)
        if factory is None:
            raise ValueError(f"Unknown GRAPH_BACKEND: {name}")
        if selected_datasets is None and graph is not None:
            graph_datasets = graph.graph.get("datasets")
            if isinstance(graph_datasets, list) and all(
                isinstance(item, str) for item in graph_datasets
            ):
                selected_datasets = sorted(graph_datasets)
        return cls(
            factory(
                graph=graph,
                db_path=db_path,
                selected_datasets=selected_datasets,
            ),
            name,
        )

    def query(
        self,
        action: str,
        params: dict[str, Any],
        *,
        payload_mode: str = "llm",
    ) -> dict[str, Any]:
        return self._backend.query(action, params, payload_mode=payload_mode)

    def get_networkx_graph(self) -> nx.DiGraph | nx.MultiDiGraph:
        return self._backend.get_networkx_graph()

    def close(self) -> None:
        self._backend.close()

    @property
    def backend_name(self) -> str:
        return self._name


def close_runtime(runtime: GraphRuntime | None) -> None:
    """Close backend resources associated with a graph runtime."""
    if runtime is None:
        return
    runtime.close()


def get_networkx_graph(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph,
) -> nx.DiGraph | nx.MultiDiGraph:
    """Return the underlying NetworkX graph from a runtime or raw graph."""
    if isinstance(runtime, GraphRuntime):
        return runtime.get_networkx_graph()
    return runtime


def query_graph_runtime(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph,
    action: str,
    params: dict[str, Any],
    *,
    payload_mode: str = "llm",
) -> dict[str, Any]:
    """Query a graph runtime or raw NetworkX graph through the active backend."""
    resolved_runtime = (
        runtime if isinstance(runtime, GraphRuntime) else wrap_networkx_graph(runtime)
    )
    return resolved_runtime.query(action, params, payload_mode=payload_mode)


def wrap_networkx_graph(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    db_path: Path | None = None,
) -> GraphRuntime:
    """Wrap an existing NetworkX graph in a GraphRuntime with the networkx backend."""
    return GraphRuntime(NetworkXBackend(graph), "networkx")


# Register default backend.
register_backend(
    "networkx",
    lambda *, graph=None, db_path=None, selected_datasets=None: NetworkXBackend(
        graph or nx.MultiDiGraph()
    ),
)


def _register_neo4j_backend() -> None:
    """Register Neo4j backend lazily to avoid optional import at module import."""

    def _factory(
        *,
        graph: nx.DiGraph | nx.MultiDiGraph | None = None,
        db_path: Path | None = None,
        selected_datasets: list[str] | tuple[str, ...] | None = None,
    ) -> GraphBackend:
        from rag_tag.graph.backends.neo4j_backend import Neo4jBackend  # noqa: PLC0415

        return Neo4jBackend(
            graph=graph,
            db_path=db_path,
            selected_datasets=tuple(selected_datasets or ()),
        )

    register_backend("neo4j", _factory)


_register_neo4j_backend()
