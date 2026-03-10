from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from .payloads import LLM_PAYLOAD_MODE
from .properties import clear_runtime_db_caches
from .types import GraphBackend, GraphRuntime


def is_graph_runtime(value: object) -> bool:
    return isinstance(value, GraphRuntime)


def get_default_backend() -> GraphBackend:
    from .backends.networkx_backend import NetworkXGraphBackend

    return NetworkXGraphBackend()


def _extract_datasets(graph: nx.DiGraph | nx.MultiDiGraph) -> list[str]:
    raw = graph.graph.get("datasets")
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return sorted(raw)
    return []


def _extract_payload_mode(graph: nx.DiGraph | nx.MultiDiGraph) -> str:
    raw = graph.graph.get("_payload_mode")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return "full"


def wrap_networkx_graph(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    context_db_path: Path | None = None,
    payload_mode: str | None = None,
) -> GraphRuntime:
    """Wrap an existing NetworkX graph in a GraphRuntime."""
    backend = get_default_backend()

    resolved_db = context_db_path
    if resolved_db is None:
        raw_db = graph.graph.get("_db_path")
        if raw_db is not None:
            resolved_db = Path(raw_db).expanduser().resolve()

    runtime = GraphRuntime(
        backend_name=backend.name,
        backend=backend,
        selected_datasets=_extract_datasets(graph),
        payload_mode=payload_mode or _extract_payload_mode(graph),
        context_db_path=resolved_db,
        backend_handle=graph,
    )

    if "_property_cache" in graph.graph:
        runtime.caches["property_cache"] = graph.graph["_property_cache"]
    if "_property_key_cache" in graph.graph:
        runtime.caches["property_key_cache"] = graph.graph["_property_key_cache"]
    if "_db_lookup_conn" in graph.graph:
        runtime.caches["db_lookup_conn"] = graph.graph["_db_lookup_conn"]

    return runtime


def load_graph_runtime(
    dataset: str | None = None,
    payload_mode: str | None = None,
    *,
    backend: GraphBackend | None = None,
) -> GraphRuntime:
    runtime_backend = backend or get_default_backend()
    return runtime_backend.load(dataset=dataset, payload_mode=payload_mode)


def ensure_graph_runtime(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph | None,
    *,
    graph_dataset: str | None = None,
    context_db_path: Path | None = None,
    payload_mode: str | None = None,
    backend: GraphBackend | None = None,
) -> GraphRuntime:
    """Load or update a graph runtime without mutating backend-private graph attrs."""
    if runtime is None:
        runtime = load_graph_runtime(
            graph_dataset,
            payload_mode=payload_mode,
            backend=backend,
        )
    elif isinstance(runtime, GraphRuntime):
        pass
    else:
        runtime = wrap_networkx_graph(
            runtime,
            context_db_path=context_db_path,
            payload_mode=payload_mode,
        )

    if payload_mode is not None:
        runtime.payload_mode = payload_mode

    if graph_dataset and not runtime.selected_datasets:
        runtime.selected_datasets = [graph_dataset]

    if context_db_path is not None:
        resolved_db = context_db_path.expanduser().resolve()
        previous = (
            str(runtime.context_db_path.expanduser().resolve())
            if runtime.context_db_path is not None
            else None
        )
        current = str(resolved_db)
        if previous != current:
            clear_runtime_db_caches(runtime)
        runtime.context_db_path = resolved_db

    return runtime


def query_graph_runtime(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph,
    action: str,
    params: dict[str, Any],
    *,
    payload_mode: str = LLM_PAYLOAD_MODE,
) -> dict[str, Any]:
    resolved_runtime = (
        runtime if isinstance(runtime, GraphRuntime) else wrap_networkx_graph(runtime)
    )
    return resolved_runtime.backend.query(
        resolved_runtime,
        action,
        params,
        payload_mode,
    )


def close_runtime(runtime: GraphRuntime | None) -> None:
    if runtime is None:
        return
    runtime.backend.close(runtime)


def get_networkx_graph(
    runtime: GraphRuntime | nx.DiGraph | nx.MultiDiGraph,
) -> nx.DiGraph | nx.MultiDiGraph:
    if isinstance(runtime, (nx.DiGraph, nx.MultiDiGraph)):
        return runtime
    graph = runtime.backend_handle
    if isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
        return graph
    raise TypeError(
        f"Graph runtime backend handle is not a NetworkX graph: {type(graph)!r}"
    )
