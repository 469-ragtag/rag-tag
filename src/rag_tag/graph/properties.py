from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import networkx as nx

from .types import GraphRuntime

_logger = logging.getLogger(__name__)

_CACHE_MISS = object()
_PROPERTY_CACHE_MAX_ENTRIES = 2048


def _graph_handle(runtime: GraphRuntime) -> nx.DiGraph | nx.MultiDiGraph:
    graph = runtime.backend_handle
    if isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
        return graph
    raise TypeError(
        f"Graph runtime backend handle is not a NetworkX graph: {type(graph)!r}"
    )


def _runtime_cache(runtime: GraphRuntime, key: str, default: Any) -> Any:
    cache_obj = runtime.caches.get(key)
    if cache_obj is None:
        runtime.caches[key] = default
        return default
    return cache_obj


def get_property_cache(
    runtime: GraphRuntime,
) -> OrderedDict[tuple[str, str], Any]:
    """Return or create the session-level property cache stored on the runtime."""
    cache_obj = runtime.caches.get("property_cache")
    if isinstance(cache_obj, OrderedDict):
        return cache_obj
    if isinstance(cache_obj, dict):
        cache: OrderedDict[tuple[str, str], Any] = OrderedDict(cache_obj.items())
    else:
        cache = OrderedDict()
    runtime.caches["property_cache"] = cache
    return cache


def get_property_key_cache(
    runtime: GraphRuntime,
) -> dict[tuple[str, str], dict[str, list[Any]]]:
    """Return or create the dotted-key sample cache stored on the runtime."""
    cache_obj = runtime.caches.get("property_key_cache")
    if isinstance(cache_obj, dict):
        return cache_obj
    cache: dict[tuple[str, str], dict[str, list[Any]]] = {}
    runtime.caches["property_key_cache"] = cache
    return cache


def clear_runtime_db_caches(runtime: GraphRuntime) -> None:
    """Clear runtime-scoped DB caches after context DB changes."""
    runtime.caches.pop("property_cache", None)
    runtime.caches.pop("property_key_cache", None)

    cached_conn = runtime.caches.pop("db_lookup_conn", None)
    if cached_conn is not None:
        try:
            cached_conn.close()
        except Exception:  # noqa: BLE001
            pass


def merge_db_element_data(
    node_data: dict[str, Any],
    db_data: dict[str, Any],
) -> dict[str, Any]:
    """Merge DB-sourced element data into in-memory node data."""

    def merge_fill_missing(
        current: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        merged_dict: dict[str, Any] = dict(current)
        for key, incoming_value in incoming.items():
            if key not in merged_dict or merged_dict[key] is None:
                merged_dict[key] = incoming_value
                continue

            existing_value = merged_dict[key]
            if isinstance(existing_value, dict) and isinstance(incoming_value, dict):
                merged_dict[key] = merge_fill_missing(existing_value, incoming_value)

        return merged_dict

    merged: dict[str, Any] = dict(node_data)

    db_props = db_data.get("properties") or {}
    if isinstance(db_props, dict) and db_props:
        existing_props: dict[str, Any] = dict(node_data.get("properties") or {})
        merged["properties"] = merge_fill_missing(existing_props, db_props)

    db_payload = db_data.get("payload") or {}
    if isinstance(db_payload, dict) and db_payload:
        existing_payload: dict[str, Any] = dict(node_data.get("payload") or {})

        if "PropertySets" in db_payload:
            existing_psets: dict[str, Any] = dict(
                existing_payload.get("PropertySets") or {}
            )
            db_psets: dict[str, Any] = db_payload["PropertySets"]
            if isinstance(db_psets, dict):
                existing_psets = merge_fill_missing(existing_psets, db_psets)
            existing_payload["PropertySets"] = existing_psets

        if "Quantities" in db_payload:
            existing_qty: dict[str, Any] = dict(
                existing_payload.get("Quantities") or {}
            )
            db_qty: dict[str, Any] = db_payload["Quantities"]
            if isinstance(db_qty, dict):
                existing_qty = merge_fill_missing(existing_qty, db_qty)
            existing_payload["Quantities"] = existing_qty

        merged["payload"] = existing_payload

    return merged


def cached_db_lookup(
    runtime: GraphRuntime,
    node_id: str,
    db_path: Path,
    *,
    db_conn: Any | None = None,
) -> dict[str, Any] | None:
    """Look up element data from SQLite with a runtime-level result cache."""
    graph = _graph_handle(runtime)
    cache = get_property_cache(runtime)
    cache_key = (str(db_path.expanduser().resolve()), node_id)
    cached = cache.get(cache_key, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        cache.move_to_end(cache_key)
        return cached  # type: ignore[return-value]

    from rag_tag.sql_element_lookup import (  # noqa: PLC0415
        lookup_element_by_express_id,
        lookup_element_by_globalid,
    )

    node_props: dict[str, Any] = (graph.nodes.get(node_id) or {}).get(
        "properties"
    ) or {}
    db_data: dict[str, Any] | None = None

    global_id = node_props.get("GlobalId")
    if global_id:
        db_data = lookup_element_by_globalid(
            db_path,
            str(global_id),
            conn=db_conn,
        )

    if db_data is None:
        express_id_raw = node_props.get("ExpressId")
        if express_id_raw is not None:
            try:
                db_data = lookup_element_by_express_id(
                    db_path,
                    int(express_id_raw),
                    conn=db_conn,
                )
            except (TypeError, ValueError):
                _logger.debug(
                    "cached_db_lookup: invalid ExpressId %r for %s",
                    express_id_raw,
                    node_id,
                )

    cache[cache_key] = db_data
    cache.move_to_end(cache_key)
    while len(cache) > _PROPERTY_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)
    return db_data


def decode_typed_db_value(value: Any) -> Any:
    """Decode DB value using the shared sql_element_lookup decoder."""
    from rag_tag.sql_element_lookup import decode_db_value  # noqa: PLC0415

    return decode_db_value(value)


def collect_dotted_keys_from_sqlite(
    db_path: Path,
    class_filter: str | None,
) -> dict[str, list[Any]]:
    """Collect dotted PropertySet/Quantity keys from SQLite."""
    if not db_path.exists():
        return {}

    import sqlite3  # noqa: PLC0415

    where_sql = ""
    params: tuple[Any, ...] = ()
    if class_filter:
        where_sql = " WHERE LOWER(e.ifc_class) = LOWER(?)"
        params = (class_filter,)

    key_samples: dict[str, list[Any]] = {}

    def record_sample(key: str, value: Any) -> None:
        if key not in key_samples:
            key_samples[key] = []
        if len(key_samples[key]) < 3:
            key_samples[key].append(value)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            prop_rows = conn.execute(
                "SELECT p.pset_name, p.property_name, p.value "
                "FROM properties p "
                "JOIN elements e ON e.express_id = p.element_id"
                f"{where_sql}",
                params,
            ).fetchall()
            for row in prop_rows:
                pset_name = str(row["pset_name"])
                prop_name = str(row["property_name"])
                record_sample(
                    f"{pset_name}.{prop_name}",
                    decode_typed_db_value(row["value"]),
                )

            qty_rows = conn.execute(
                "SELECT q.qto_name, q.quantity_name, q.value "
                "FROM quantities q "
                "JOIN elements e ON e.express_id = q.element_id"
                f"{where_sql}",
                params,
            ).fetchall()
            for row in qty_rows:
                qto_name = str(row["qto_name"])
                qty_name = str(row["quantity_name"])
                record_sample(
                    f"{qto_name}.{qty_name}",
                    decode_typed_db_value(row["value"]),
                )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("list_property_keys DB fallback failed (%s): %s", db_path, exc)
        return {}

    return key_samples
