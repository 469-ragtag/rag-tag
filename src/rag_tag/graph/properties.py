"""Shared graph property helpers (DB enrichment + filtering)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable

import networkx as nx

_logger = logging.getLogger(__name__)

# Sentinel: distinguishes "not yet cached" from "cached, result was None".
_CACHE_MISS = object()
_PROPERTY_CACHE_MAX_ENTRIES = 2048


def get_property_cache(
    G: nx.DiGraph | nx.MultiDiGraph,
) -> OrderedDict[tuple[str, str], Any]:
    """Return or create the session-level property cache stored on the graph."""
    cache_obj = G.graph.get("_property_cache")
    if isinstance(cache_obj, OrderedDict):
        return cache_obj
    if isinstance(cache_obj, dict):
        cache: OrderedDict[tuple[str, str], Any] = OrderedDict(cache_obj.items())
    else:
        cache = OrderedDict()
    G.graph["_property_cache"] = cache
    return cache


def cached_db_lookup(
    G: nx.DiGraph | nx.MultiDiGraph,
    node_id: str,
    db_path: Path,
    *,
    db_conn: Any | None = None,
) -> dict[str, Any] | None:
    """Look up element data from the SQLite DB with a graph-level cache."""
    cache = get_property_cache(G)
    cache_key = (str(db_path.expanduser().resolve()), node_id)
    cached = cache.get(cache_key, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        cache.move_to_end(cache_key)
        return cached  # type: ignore[return-value]

    from rag_tag.sql_element_lookup import (  # noqa: PLC0415
        lookup_element_by_express_id,
        lookup_element_by_globalid,
    )

    node_props: dict[str, Any] = (G.nodes.get(node_id) or {}).get("properties") or {}
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


def merge_db_element_data(
    node_data: dict[str, Any],
    db_data: dict[str, Any],
) -> dict[str, Any]:
    """Merge DB-sourced element data into in-memory node data.

    DB data is used as a *fill-missing* enrichment source. Existing in-memory
    values are preserved to avoid degrading richer payload structures.
    """

    def _merge_fill_missing(
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
                merged_dict[key] = _merge_fill_missing(existing_value, incoming_value)

        return merged_dict

    merged: dict[str, Any] = dict(node_data)

    db_props = db_data.get("properties") or {}
    if isinstance(db_props, dict) and db_props:
        existing_props: dict[str, Any] = dict(node_data.get("properties") or {})
        merged["properties"] = _merge_fill_missing(existing_props, db_props)

    db_payload = db_data.get("payload") or {}
    if isinstance(db_payload, dict) and db_payload:
        existing_payload: dict[str, Any] = dict(node_data.get("payload") or {})

        if "PropertySets" in db_payload:
            existing_psets: dict[str, Any] = dict(
                existing_payload.get("PropertySets") or {}
            )
            db_psets: dict[str, Any] = db_payload["PropertySets"]
            if isinstance(db_psets, dict):
                existing_psets = _merge_fill_missing(existing_psets, db_psets)
            existing_payload["PropertySets"] = existing_psets

        if "Quantities" in db_payload:
            existing_qty: dict[str, Any] = dict(
                existing_payload.get("Quantities") or {}
            )
            db_qty: dict[str, Any] = db_payload["Quantities"]
            if isinstance(db_qty, dict):
                existing_qty = _merge_fill_missing(existing_qty, db_qty)
            existing_payload["Quantities"] = existing_qty

        merged["payload"] = existing_payload

    return merged


def collect_dotted_keys_from_sqlite(
    db_path: Path,
    class_filter: str | None,
) -> dict[str, list[Any]]:
    """Collect dotted PropertySet/Quantity keys from SQLite.

    Returns a mapping of ``dotted_key -> up to 3 sample values``.
    """
    if not db_path.exists():
        return {}

    import sqlite3  # noqa: PLC0415

    where_sql = ""
    params: tuple[Any, ...] = ()
    if class_filter:
        where_sql = " WHERE LOWER(e.ifc_class) = LOWER(?)"
        params = (class_filter,)

    key_samples: dict[str, list[Any]] = {}

    def _record_sample(key: str, value: Any) -> None:
        if key not in key_samples:
            key_samples[key] = []
        if len(key_samples[key]) < 3:
            key_samples[key].append(value)

    from rag_tag.sql_element_lookup import decode_db_value  # noqa: PLC0415

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
                _record_sample(
                    f"{pset_name}.{prop_name}",
                    decode_db_value(row["value"]),
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
                _record_sample(
                    f"{qto_name}.{qty_name}",
                    decode_db_value(row["value"]),
                )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("list_property_keys DB fallback failed (%s): %s", db_path, exc)
        return {}

    return key_samples


def apply_property_filters(
    G: nx.DiGraph | nx.MultiDiGraph,
    node_id: str,
    filters: Dict[str, Any],
    *,
    db_conn: Any | None = None,
) -> bool:
    """Return True when a node matches all property filters."""
    if not filters:
        return True
    data = G.nodes[node_id]
    props = data.get("properties") or {}
    if not isinstance(props, dict):
        props = {}

    payload = data.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    pset_block = payload.get("PropertySets") or {}
    if not isinstance(pset_block, dict):
        pset_block = {}

    _effective_quantities: dict[str, Any] = payload.get("Quantities") or {}
    if not isinstance(_effective_quantities, dict):
        _effective_quantities = {}

    def _match_flat_property_value(prop_val: Any, expected: Any) -> bool:
        if isinstance(prop_val, list):
            if isinstance(expected, str):
                return expected in prop_val
            if isinstance(expected, list):
                return all(v in prop_val for v in expected)
            return False
        return prop_val == expected

    _has_dotted_filter = any("." in k for k in filters)
    _needs_flat_pset_fallback = any(
        key not in props or not _match_flat_property_value(props[key], expected)
        for key, expected in filters.items()
        if "." not in key
    )

    if (
        (_has_dotted_filter or _needs_flat_pset_fallback)
        and not pset_block
        and not _effective_quantities
    ):
        _db_path_raw: Any = G.graph.get("_db_path")
        if _db_path_raw is not None:
            _db_data = cached_db_lookup(
                G,
                node_id,
                Path(_db_path_raw),
                db_conn=db_conn,
            )
            if _db_data is not None:
                _db_payload: dict[str, Any] = _db_data.get("payload") or {}
                if isinstance(_db_payload, dict):
                    _enriched_psets = _db_payload.get("PropertySets") or {}
                    if isinstance(_enriched_psets, dict):
                        pset_block = _enriched_psets
                    _enriched_qty = _db_payload.get("Quantities") or {}
                    if isinstance(_enriched_qty, dict):
                        _effective_quantities = _enriched_qty

    def _iter_psets() -> Iterable[tuple[str, dict[str, Any]]]:
        for section in ("Official", "Custom"):
            section_block = pset_block.get(section) or {}
            if not isinstance(section_block, dict):
                continue
            for raw_name, pset_props in section_block.items():
                if not isinstance(pset_props, dict):
                    continue
                yield str(raw_name), pset_props

        for qto_name, qto_data in _effective_quantities.items():
            if isinstance(qto_data, dict):
                yield str(qto_name), qto_data

    def _nested_lookup(mapping: dict[str, Any], dotted_path: str) -> tuple[bool, Any]:
        current: Any = mapping
        for part in dotted_path.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current

    def _match_dotted(pset_name: str, prop_name: str, expected: Any) -> bool:
        for current_pset, pset_props in _iter_psets():
            if current_pset != pset_name:
                continue
            exists, value = _nested_lookup(pset_props, prop_name)
            if exists and value == expected:
                return True
        return False

    def _match_flat_in_psets(key: str, expected: Any) -> bool:
        for _pset_name, pset_props in _iter_psets():
            if key in pset_props and pset_props[key] == expected:
                return True
        return False

    for key, expected in filters.items():
        if "." in key:
            pset_name, _, prop_name = key.partition(".")
            if not _match_dotted(pset_name, prop_name, expected):
                return False
            continue

        if key in props:
            prop_val = props[key]
            if _match_flat_property_value(prop_val, expected):
                continue

        if not _match_flat_in_psets(key, expected):
            return False

    return True
