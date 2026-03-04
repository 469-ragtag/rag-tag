from __future__ import annotations

import logging
import re
from collections import OrderedDict, deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable

import networkx as nx

from rag_tag.graph_contract import (
    CANONICAL_ACTIONS,
    CANONICAL_RELATION_SET,
    KNOWN_RELATION_SOURCE_SET,
    SPATIAL_RELATIONS,
    TOPOLOGY_RELATIONS,
    is_allowed_action,
    make_error_envelope,
    make_ok_envelope,
    normalize_action_name,
    normalize_relation_name,
    normalize_relation_source,
    relation_bucket,
)

_logger = logging.getLogger(__name__)

# Sentinel: distinguishes "not yet cached" from "cached, result was None".
_CACHE_MISS = object()
_PROPERTY_CACHE_MAX_ENTRIES = 2048

LLM_PAYLOAD_MODE = "llm"
INTERNAL_PAYLOAD_MODE = "internal"

LLM_PROPERTY_ALLOWLIST: tuple[str, ...] = (
    "GlobalId",
    "Name",
    "TypeName",
    "Level",
    "PredefinedType",
    "ObjectType",
    "Zone",
)
MAX_LLM_STRING_CHARS = 160
REDACTED_COMPLEX_VALUE = "[REDACTED_COMPLEX]"
TRUNCATED_SUFFIX = "...[truncated]"
TRACE_MEP_MAX_DEPTH_DEFAULT = 10
TRACE_MEP_MAX_RESULTS_DEFAULT = 200
TRACE_MEP_MAX_RESULTS_HARD_LIMIT = 1000
ACTION_LIST_MAX_RESULTS_DEFAULT = TRACE_MEP_MAX_RESULTS_DEFAULT
ACTION_LIST_MAX_RESULTS_HARD_LIMIT = TRACE_MEP_MAX_RESULTS_HARD_LIMIT
CONTEXT_FUZZY_MIN_SCORE = 86.0
CONTEXT_FUZZY_AMBIGUITY_DELTA = 2.0


def _resolve_payload_mode(payload_mode: str) -> str:
    """Return a supported payload mode, defaulting safely to llm."""
    if payload_mode == INTERNAL_PAYLOAD_MODE:
        return INTERNAL_PAYLOAD_MODE
    return LLM_PAYLOAD_MODE


def sanitize_llm_property_value(value: Any) -> Any:
    """Reduce property value exposure to scalar-safe content."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_LLM_STRING_CHARS:
            return value
        return f"{value[:MAX_LLM_STRING_CHARS]}{TRUNCATED_SUFFIX}"
    return REDACTED_COMPLEX_VALUE


def sanitize_properties_for_llm(properties: dict[str, Any] | None) -> dict[str, Any]:
    """Filter properties to an allowlisted, redacted view for LLM tools."""
    if not isinstance(properties, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in LLM_PROPERTY_ALLOWLIST:
        if key not in properties:
            continue
        safe[key] = sanitize_llm_property_value(properties.get(key))
    return safe


def build_node_payload(
    node_id: str, node_data: dict[str, Any], *, payload_mode: str = LLM_PAYLOAD_MODE
) -> dict[str, Any]:
    """Build node payload with mode-aware property exposure.

    The ``payload`` key is always present in the returned dict:
    - In ``INTERNAL_PAYLOAD_MODE``: the full raw payload dict (or None).
    - In ``LLM_PAYLOAD_MODE``: ``None`` — the LLM sees a null placeholder so
      the key is structurally stable, but raw payload data is not exposed.
    """
    mode = _resolve_payload_mode(payload_mode)
    raw_props = node_data.get("properties")
    if mode == INTERNAL_PAYLOAD_MODE:
        properties = raw_props if isinstance(raw_props, dict) else {}
    else:
        properties = sanitize_properties_for_llm(raw_props)

    return {
        "id": node_id,
        "label": node_data.get("label"),
        "class_": node_data.get("class_"),
        "properties": properties,
        # Always include the payload key; value is only exposed in internal mode.
        "payload": node_data.get("payload") if mode == INTERNAL_PAYLOAD_MODE else None,
    }


def _ok_action(action: str, data: dict[str, Any] | None) -> dict[str, Any]:
    """Wrap successful action result in canonical envelope."""
    return make_ok_envelope(action, data)


def _err(message: str, code: str, details: dict | None = None) -> dict[str, Any]:
    """Wrap error result in envelope."""
    return make_error_envelope(message, code, details)


def _merge_db_element_data(
    node_data: dict[str, Any],
    db_data: dict[str, Any],
) -> dict[str, Any]:
    """Merge DB-sourced element data into in-memory node data.

    DB data is used as a *fill-missing* enrichment source. Existing in-memory
    values are preserved to avoid degrading richer payload structures.

    Args:
        node_data: Raw ``G.nodes[node_id]`` dict from the NetworkX graph.
        db_data: Structured result from ``sql_element_lookup`` with
            ``properties`` and ``payload`` keys.

    Returns:
        New dict with merged data (the input dicts are not mutated).
    """

    def _merge_fill_missing(
        current: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge *incoming* into *current* without clobbering values."""
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

    # Merge flat properties without overriding existing non-null graph values.
    db_props = db_data.get("properties") or {}
    if isinstance(db_props, dict) and db_props:
        existing_props: dict[str, Any] = dict(node_data.get("properties") or {})
        merged["properties"] = _merge_fill_missing(existing_props, db_props)

    # Merge payload sections; DB fills holes but never overwrites richer in-memory
    # nested content.
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


def _get_property_cache(G: nx.DiGraph) -> OrderedDict[tuple[str, str], Any]:
    """Return or create the session-level property cache stored on the graph.

    The cache lives at ``G.graph["_property_cache"]`` and is keyed by
    ``(db_path, node_id)`` tuples so entries remain valid when the same graph
    object is reused with a different DB context.

    Values are DB element data dicts (or ``None`` when not found), with
    ``_CACHE_MISS`` used as the "not yet fetched" sentinel so that a ``None``
    result (element absent from DB) is not re-fetched on subsequent calls.
    """
    cache_obj = G.graph.get("_property_cache")
    if isinstance(cache_obj, OrderedDict):
        return cache_obj
    if isinstance(cache_obj, dict):
        cache: OrderedDict[tuple[str, str], Any] = OrderedDict(cache_obj.items())
    else:
        cache = OrderedDict()
    G.graph["_property_cache"] = cache
    return cache


def _cached_db_lookup(
    G: nx.DiGraph,
    node_id: str,
    db_path: Path,
    *,
    db_conn: Any | None = None,
) -> dict[str, Any] | None:
    """Look up element data from the SQLite DB with a graph-level result cache.

    Avoids reopening the database for the same element within a single agent
    session. Cache keys include both DB path and node id, so the same graph
    object can safely switch DB context without stale property leakage.

    Callers may optionally provide an open ``db_conn`` to reuse one SQLite
    connection across many lookups in a filter pass.

    Args:
        G: NetworkX graph with optional ``_property_cache`` graph attribute.
        node_id: Graph node identifier used as part of the cache key.
        db_path: Path to the SQLite database file.
        db_conn: Optional open sqlite connection for pass-level reuse.

    Returns:
        DB element data dict (``properties`` + ``payload`` keys) or ``None``
        when the element is not present in the database.
    """
    cache = _get_property_cache(G)
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
                    "_cached_db_lookup: invalid ExpressId %r for %s",
                    express_id_raw,
                    node_id,
                )

    cache[cache_key] = db_data
    cache.move_to_end(cache_key)
    while len(cache) > _PROPERTY_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)
    return db_data


def _decode_typed_db_value(value: Any) -> Any:
    """Decode DB value using the shared sql_element_lookup decoder."""
    from rag_tag.sql_element_lookup import decode_db_value  # noqa: PLC0415

    return decode_db_value(value)


def _collect_dotted_keys_from_sqlite(
    db_path: Path,
    class_filter: str | None,
) -> dict[str, list[Any]]:
    """Collect dotted PropertySet/Quantity keys from SQLite.

    Returns a mapping of ``dotted_key -> up to 3 sample values``.
    Any DB/schema/runtime error is handled gracefully by returning an empty map.
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
                    _decode_typed_db_value(row["value"]),
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
                    _decode_typed_db_value(row["value"]),
                )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("list_property_keys DB fallback failed (%s): %s", db_path, exc)
        return {}

    return key_samples


def query_ifc_graph(
    G: nx.DiGraph,
    action: str,
    params: Dict[str, Any],
    *,
    payload_mode: str = LLM_PAYLOAD_MODE,
) -> Dict[str, Any]:
    """Controlled interface between the LLM and NetworkX graph."""
    if not isinstance(action, str):
        return _err("Invalid action: action must be a string", "invalid")

    action = normalize_action_name(action)
    if not is_allowed_action(action):
        return _err(
            f"Unknown action: {action}",
            "unknown_action",
            {"allowed_actions": sorted(CANONICAL_ACTIONS)},
        )

    if not isinstance(params, dict):
        return _err("Invalid params: params must be an object", "invalid")

    resolved_payload_mode = _resolve_payload_mode(payload_mode)

    def _normalize_class(value: str) -> str:
        v = value.strip()
        if not v:
            return v
        if not v.lower().startswith("ifc"):
            v = f"Ifc{v}"
        return v

    def _find_nodes_by_label(label: str, class_filter: str | None = None) -> list[str]:
        target = label.strip().lower()
        if not target:
            return []
        matches = []
        for n, d in G.nodes(data=True):
            lbl = str(d.get("label", "")).strip().lower()
            if lbl != target:
                continue
            if class_filter is not None:
                if str(d.get("class_", "")).lower() != class_filter.lower():
                    continue
            matches.append(n)
        return matches

    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _fuzzy_context_matches(
        query: str,
        candidate_node_ids: Iterable[str],
    ) -> list[tuple[str, float]]:
        query_norm = _normalize_text(query)
        if not query_norm:
            return []

        scored: list[tuple[str, float]] = []
        for node_id in candidate_node_ids:
            label = str(G.nodes[node_id].get("label", ""))
            label_norm = _normalize_text(label)
            if not label_norm:
                continue

            score = SequenceMatcher(None, query_norm, label_norm).ratio() * 100.0
            if query_norm in label_norm:
                score = max(score, 96.0)
            elif label_norm in query_norm:
                score = max(score, 90.0)

            if score >= CONTEXT_FUZZY_MIN_SCORE:
                scored.append((str(node_id), score))

        scored.sort(key=lambda item: (-item[1], _node_sort_key(item[0])))
        return scored

    def _node_sort_key(node_id: str) -> tuple[str, str]:
        label = ""
        if node_id in G:
            label = _normalize_text(str(G.nodes[node_id].get("label", "")))
        return label, str(node_id)

    def _sorted_node_ids(node_ids: Iterable[str]) -> list[str]:
        return sorted((str(node_id) for node_id in node_ids), key=_node_sort_key)

    def _edge_relation(edge: dict[str, Any]) -> str | None:
        relation = normalize_relation_name(edge.get("relation"))
        if relation in CANONICAL_RELATION_SET:
            return relation
        return None

    def _expected_source_for_relation(relation: str | None) -> str | None:
        bucket = relation_bucket(relation)
        if bucket == "explicit_ifc":
            return "ifc"
        if bucket == "topology":
            return "topology"
        if bucket == "spatial":
            return "heuristic"
        return None

    def _edge_source(edge: dict[str, Any], relation: str | None = None) -> str | None:
        canonical_relation = relation if relation in CANONICAL_RELATION_SET else None
        if canonical_relation is None:
            canonical_relation = _edge_relation(edge)

        bucket = relation_bucket(canonical_relation)
        if bucket is None or bucket == "hierarchy":
            return None

        expected_source = _expected_source_for_relation(canonical_relation)
        if expected_source is not None:
            return expected_source

        source = normalize_relation_source(edge.get("source"))
        if source in KNOWN_RELATION_SOURCE_SET:
            return source
        return None

    def _resolve_element_id(
        element_id: str,
    ) -> tuple[str | None, Dict[str, Any] | None]:
        if not isinstance(element_id, str):
            return None, {"error": "Invalid element_id: element_id must be a string"}
        if element_id in G:
            if str(element_id).startswith("Element::"):
                return element_id, None
            return None, {"error": f"Invalid element_id (not an element): {element_id}"}
        if not element_id.startswith("Element::"):
            candidate = f"Element::{element_id}"
            if candidate in G:
                return candidate, None
        matches = []
        for n, d in G.nodes(data=True):
            if not str(n).startswith("Element::"):
                continue
            gid = d.get("properties", {}).get("GlobalId")
            if gid == element_id:
                matches.append(n)
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, {"error": "Ambiguous element_id", "candidates": matches}
        return None, {"error": f"Element not found: {element_id}"}

    def _resolve_storey_node(
        storey_query: str,
    ) -> tuple[str | None, Dict[str, Any] | None]:
        query = storey_query.strip()
        if not query:
            return None, {"error": "Missing param: storey"}

        # Direct id match supports both raw GlobalId and Storey::<GlobalId>.
        direct = query if query.startswith("Storey::") else f"Storey::{query}"
        if direct in G and (
            str(G.nodes[direct].get("class_", "")).lower() == "ifcbuildingstorey"
        ):
            return direct, None

        # Exact label match (legacy/user-friendly).
        exact = _find_nodes_by_label(query, class_filter="IfcBuildingStorey")
        if len(exact) == 1:
            return exact[0], None
        if len(exact) > 1:
            return None, {"error": "Ambiguous storey", "candidates": exact}

        # Normalized label fallback (spaces/punctuation/case).
        qn = _normalize_text(query)
        norm_matches: list[str] = []
        for n, d in G.nodes(data=True):
            if str(d.get("class_", "")).lower() != "ifcbuildingstorey":
                continue
            if _normalize_text(str(d.get("label", ""))) == qn:
                norm_matches.append(n)
        if len(norm_matches) == 1:
            return norm_matches[0], None
        if len(norm_matches) > 1:
            return None, {"error": "Ambiguous storey", "candidates": norm_matches}

        return None, {"error": f"Storey not found: {storey_query}"}

    def _resolver_error_envelope(err: Dict[str, Any]) -> dict[str, Any]:
        message = str(err.get("error", "Unknown error"))
        code = str(err.get("code", "")).strip().lower()

        if code == "ambiguous":
            candidates = _sorted_node_ids(err.get("candidates") or [])
            return _err(message, "ambiguous", {"candidates": candidates})
        if code in {"missing_param", "invalid", "not_found"}:
            return _err(message, code)

        if "Ambiguous" in message:
            candidates = _sorted_node_ids(err.get("candidates") or [])
            return _err(message, "ambiguous", {"candidates": candidates})
        if "Missing param" in message:
            return _err(message, "missing_param")
        if "Invalid" in message:
            return _err(message, "invalid")
        return _err(message, "not_found")

    def _resolve_element_error_envelope(err: Dict[str, Any]) -> dict[str, Any]:
        message = str(err.get("error", "Unknown error"))
        if "Ambiguous" in message:
            candidates = _sorted_node_ids(err.get("candidates") or [])
            return _err(message, "ambiguous", {"candidates": candidates})
        if "Invalid element_id" in message:
            return _err(message, "invalid")
        return _err(message, "not_found")

    def _parse_positive_int_param(
        param_name: str,
        *,
        default: int,
        hard_limit: int | None = None,
    ) -> tuple[int | None, dict[str, Any] | None]:
        raw_value = params.get(param_name, default)

        parsed: int
        if isinstance(raw_value, bool):
            return (
                None,
                _err(f"Invalid param: {param_name} must be an integer", "invalid"),
            )
        if isinstance(raw_value, int):
            parsed = raw_value
        elif isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not re.fullmatch(r"[+-]?\d+", candidate):
                return (
                    None,
                    _err(
                        f"Invalid param: {param_name} must be an integer",
                        "invalid",
                    ),
                )
            parsed = int(candidate)
        else:
            return (
                None,
                _err(f"Invalid param: {param_name} must be an integer", "invalid"),
            )

        if parsed < 1:
            return (
                None,
                _err(f"Invalid param: {param_name} must be >= 1", "invalid"),
            )

        if hard_limit is not None and parsed > hard_limit:
            parsed = hard_limit

        return parsed, None

    def _apply_result_limit(
        items: list[dict[str, Any]],
        max_results: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        if len(items) <= max_results:
            return items, False
        return items[:max_results], True

    def _resolve_context_node(
        query_value: str,
        *,
        param_name: str,
        prefix: str,
        class_name: str,
        display_name: str,
        enable_fuzzy: bool = False,
    ) -> tuple[str | None, Dict[str, Any] | None]:
        query = query_value.strip()
        if not query:
            return None, {
                "error": f"Missing param: {param_name}",
                "code": "missing_param",
            }

        class_lower = class_name.lower()
        class_nodes = [
            str(n)
            for n, d in G.nodes(data=True)
            if str(d.get("class_", "")).lower() == class_lower
        ]

        def _is_expected_class(node_id: str) -> bool:
            return node_id in G and (
                str(G.nodes[node_id].get("class_", "")).lower() == class_lower
            )

        prefix_token = f"{prefix}::"
        direct_ids: list[str] = []
        if query.startswith(prefix_token):
            direct_ids.append(query)
        else:
            direct_ids.append(f"{prefix_token}{query}")
            if query in G:
                direct_ids.append(query)

        for direct_id in direct_ids:
            if direct_id not in G:
                continue
            if _is_expected_class(direct_id):
                return direct_id, None
            return (
                None,
                {
                    "error": (
                        f"Invalid param: {param_name} is not a {display_name}: "
                        f"{direct_id}"
                    ),
                    "code": "invalid",
                },
            )

        lowered = query.lower()
        prefixed_lowered = (
            lowered
            if query.startswith(prefix_token)
            else f"{prefix_token}{query}".lower()
        )
        id_matches = [
            node_id
            for node_id in class_nodes
            if node_id.lower() in {lowered, prefixed_lowered}
        ]
        if len(id_matches) == 1:
            return id_matches[0], None
        if len(id_matches) > 1:
            return None, {
                "error": f"Ambiguous {param_name}",
                "code": "ambiguous",
                "candidates": _sorted_node_ids(id_matches),
            }

        gid_matches: list[str] = []
        for node_id in class_nodes:
            props = G.nodes[node_id].get("properties") or {}
            if not isinstance(props, dict):
                continue
            gid = props.get("GlobalId")
            if isinstance(gid, str) and gid.strip().lower() == lowered:
                gid_matches.append(node_id)
        if len(gid_matches) == 1:
            return gid_matches[0], None
        if len(gid_matches) > 1:
            return None, {
                "error": f"Ambiguous {param_name}",
                "code": "ambiguous",
                "candidates": _sorted_node_ids(gid_matches),
            }

        exact = _find_nodes_by_label(query, class_filter=class_name)
        if len(exact) == 1:
            return exact[0], None
        if len(exact) > 1:
            return None, {
                "error": f"Ambiguous {param_name}",
                "code": "ambiguous",
                "candidates": _sorted_node_ids(exact),
            }

        normalized_query = _normalize_text(query)
        normalized_matches = [
            node_id
            for node_id in class_nodes
            if _normalize_text(str(G.nodes[node_id].get("label", "")))
            == normalized_query
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0], None
        if len(normalized_matches) > 1:
            return None, {
                "error": f"Ambiguous {param_name}",
                "code": "ambiguous",
                "candidates": _sorted_node_ids(normalized_matches),
            }

        if enable_fuzzy:
            fuzzy_matches = _fuzzy_context_matches(query, class_nodes)
            if fuzzy_matches:
                top_score = fuzzy_matches[0][1]
                near_best = [
                    node_id
                    for node_id, score in fuzzy_matches
                    if (top_score - score) <= CONTEXT_FUZZY_AMBIGUITY_DELTA
                ]
                if len(near_best) == 1:
                    return near_best[0], None
                return None, {
                    "error": f"Ambiguous {param_name}",
                    "code": "ambiguous",
                    "candidates": _sorted_node_ids(near_best),
                }

        return None, {
            "error": f"{display_name} not found: {query_value}",
            "code": "not_found",
        }

    def _classification_label_matches(label: str, query: str) -> bool:
        """Return True when *query* matches a classification label robustly.

        Classification labels are stored as
        ``<SourceName>:<Identification>:<RefName>``. Users often query by code
        only (e.g. ``E-AAA``) or by the human name segment. This matcher checks
        exact/normalized segment equality first, then falls back to substring
        match against the full label.
        """
        label_text = str(label or "").strip()
        query_text = str(query or "").strip()
        if not label_text or not query_text:
            return False

        label_norm = _normalize_text(label_text)
        query_norm = _normalize_text(query_text)
        if not query_norm:
            return False

        if label_norm == query_norm:
            return True

        segments = [segment.strip() for segment in label_text.split(":") if segment]
        for segment in segments:
            if segment.lower() == query_text.lower():
                return True
            if _normalize_text(segment) == query_norm:
                return True

        return query_text.lower() in label_text.lower()

    def _resolve_classification_node(
        classification_query: str,
    ) -> tuple[str | None, Dict[str, Any] | None]:
        """Resolve an IfcClassificationReference by id/label/code fragment.

        Starts with strict context resolution, then applies classification-aware
        matching so queries like ``E-AAA`` resolve against labels such as
        ``CCI Construction:E-AAA:Single-family house``.
        """
        resolved, err = _resolve_context_node(
            classification_query,
            param_name="classification",
            prefix="Classification",
            class_name="IfcClassificationReference",
            display_name="Classification",
            enable_fuzzy=True,
        )
        if resolved is not None:
            return resolved, None
        if err is None or str(err.get("code", "")).lower() != "not_found":
            return resolved, err

        query = classification_query.strip()
        class_nodes = [
            str(node_id)
            for node_id, data in G.nodes(data=True)
            if str(data.get("class_", "")).lower() == "ifcclassificationreference"
        ]
        matches = [
            node_id
            for node_id in class_nodes
            if _classification_label_matches(
                str(G.nodes[node_id].get("label", "")), query
            )
        ]

        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, {
                "error": "Ambiguous classification",
                "code": "ambiguous",
                "candidates": _sorted_node_ids(matches),
            }

        return None, {
            "error": f"Classification not found: {classification_query}",
            "code": "not_found",
        }

    def _incoming_membership_nodes(
        context_node_id: str,
        relation_name: str,
    ) -> list[dict[str, Any]]:
        members: dict[str, dict[str, Any]] = {}
        for source_id in G.predecessors(context_node_id):
            source_node_id = str(source_id)
            if source_node_id == context_node_id:
                continue
            edge = G[source_node_id][context_node_id]
            edge_relation = _edge_relation(edge)
            if edge_relation != relation_name:
                continue

            source_data = G.nodes.get(source_node_id) or {}
            source_class = str(source_data.get("class_", "")).strip()
            if not source_class:
                continue
            if str(source_data.get("node_kind", "")).strip().lower() == "context":
                continue

            members[source_node_id] = {
                "id": source_node_id,
                "label": source_data.get("label"),
                "class_": source_data.get("class_"),
                "relation": edge_relation,
                "source": _edge_source(edge, edge_relation),
            }
        return [members[node_id] for node_id in _sorted_node_ids(members)]

    def _contains_descendants(
        start: str,
    ) -> list[tuple[str, int, dict[str, Any]]]:
        visited = {start}
        q: deque[tuple[str, int]] = deque([(start, 0)])
        descendants: list[tuple[str, int, dict[str, Any]]] = []

        while q:
            node, depth = q.popleft()
            candidates: list[tuple[str, dict[str, Any]]] = []
            for nbr in G.successors(node):
                edge = G[node][nbr]
                if _edge_relation(edge) != "contains":
                    continue
                nbr_id = str(nbr)
                if nbr_id in visited:
                    continue
                candidates.append((nbr_id, edge))

            candidates.sort(key=lambda item: _node_sort_key(item[0]))
            for nbr_id, edge in candidates:
                if nbr_id in visited:
                    continue
                visited.add(nbr_id)
                q.append((nbr_id, depth + 1))
                descendants.append((nbr_id, depth + 1, edge))

        return descendants

    def _storey_elements(start: str) -> Iterable[str]:
        """Traverse downward storey containment only (contains edges)."""
        visited = {start}
        q = deque([start])
        while q:
            node = q.popleft()
            for nbr in G.successors(node):
                edge = G[node][nbr]
                relation = normalize_relation_name(edge.get("relation"))
                if relation != "contains":
                    continue
                if nbr in visited:
                    continue
                visited.add(nbr)
                q.append(nbr)
                yield nbr

    def _spatial_neighbors(node_id: str) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Yield unique spatial neighbors across both outgoing and incoming edges."""
        seen: set[str] = set()
        spatial_relations = set(SPATIAL_RELATIONS)

        for nbr in G.successors(node_id):
            if nbr in seen:
                continue
            edge = G[node_id][nbr]
            relation = normalize_relation_name(edge.get("relation"))
            if relation not in spatial_relations:
                continue
            seen.add(nbr)
            yield nbr, edge

        for nbr in G.predecessors(node_id):
            if nbr in seen:
                continue
            edge = G[nbr][node_id]
            relation = normalize_relation_name(edge.get("relation"))
            if relation not in spatial_relations:
                continue
            seen.add(nbr)
            yield nbr, edge

    def _topology_neighbors(
        node_id: str,
        allowed_relations: set[str] | None = None,
    ) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Yield unique topology neighbors across both directions."""
        seen: set[tuple[str, str]] = set()
        topology_relations = set(TOPOLOGY_RELATIONS)
        if allowed_relations is None:
            allowed_relations = topology_relations

        for nbr in G.successors(node_id):
            edge = G[node_id][nbr]
            relation = normalize_relation_name(edge.get("relation"))
            if relation not in allowed_relations:
                continue
            if relation is None:
                continue
            key = (nbr, relation)
            if key in seen:
                continue
            seen.add(key)
            yield nbr, edge

        for nbr in G.predecessors(node_id):
            edge = G[nbr][node_id]
            relation = normalize_relation_name(edge.get("relation"))
            if relation not in allowed_relations:
                continue
            if relation is None:
                continue
            key = (nbr, relation)
            if key in seen:
                continue
            seen.add(key)
            yield nbr, edge

    def _apply_property_filters(
        node_id: str,
        filters: Dict[str, Any],
        *,
        db_conn: Any | None = None,
    ) -> bool:
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

        # Effective Quantities block — in-memory first; may be enriched below.
        _effective_quantities: dict[str, Any] = payload.get("Quantities") or {}
        if not isinstance(_effective_quantities, dict):
            _effective_quantities = {}

        def _match_flat_property_value(prop_val: Any, expected: Any) -> bool:
            """Return True when a direct flat property value matches *expected*."""
            if isinstance(prop_val, list):
                if isinstance(expected, str):
                    return expected in prop_val
                if isinstance(expected, list):
                    return all(v in prop_val for v in expected)
                return False
            return prop_val == expected

        # Minimal payload mode: pset_block will be empty because PropertySets
        # are not stored in-memory.  If filters require pset/quantity fallback
        # (dotted key OR flat key not directly matched), enrich from DB.
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
                _db_data = _cached_db_lookup(
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
            """Yield (pset_name, pset_dict) from Official/Custom psets and Quantities.

            Quantities (e.g. Qto_WallBaseQuantities) are stored at
            ``payload["Quantities"]`` — a sibling of ``PropertySets``, not
            nested inside it.  Including them here means dotted filters such
            as ``Qto_WallBaseQuantities.Length`` work identically to pset
            filters.
            """
            for section in ("Official", "Custom"):
                section_block = pset_block.get(section) or {}
                if not isinstance(section_block, dict):
                    continue
                for raw_name, pset_props in section_block.items():
                    if not isinstance(pset_props, dict):
                        continue
                    yield str(raw_name), pset_props

            # Also expose Quantities blocks so that dotted keys like
            # "Qto_WallBaseQuantities.Length" are matched by _match_dotted.
            # _effective_quantities was resolved above (in-memory or DB-enriched).
            for qto_name, qto_data in _effective_quantities.items():
                if isinstance(qto_data, dict):
                    yield str(qto_name), qto_data

        def _nested_lookup(
            mapping: dict[str, Any], dotted_path: str
        ) -> tuple[bool, Any]:
            """Return (exists, value) for a dotted key path within nested dicts."""
            current: Any = mapping
            for part in dotted_path.split("."):
                if not isinstance(current, dict) or part not in current:
                    return False, None
                current = current[part]
            return True, current

        def _match_dotted(pset_name: str, prop_name: str, expected: Any) -> bool:
            """Match a specific pset path like Pset.Property or Pset.A.B."""
            for current_pset, pset_props in _iter_psets():
                if current_pset != pset_name:
                    continue
                exists, value = _nested_lookup(pset_props, prop_name)
                if exists and value == expected:
                    return True
            return False

        def _match_flat_in_psets(key: str, expected: Any) -> bool:
            """Search all psets for a flat key; key must exist in pset for a match."""
            for _pset_name, pset_props in _iter_psets():
                if key in pset_props and pset_props[key] == expected:
                    return True
            return False

        for key, expected in filters.items():
            # Dotted key "PsetName.PropertyName": target specific named pset.
            # (Uses first dot only; deeper nesting not supported.)
            if "." in key:
                pset_name, _, prop_name = key.partition(".")
                if not _match_dotted(pset_name, prop_name, expected):
                    return False
                continue

            # Flat key: check direct properties first.  Require key existence so
            # that a missing key never accidentally matches an expected None.
            if key in props:
                prop_val = props[key]
                if _match_flat_property_value(prop_val, expected):
                    continue
                # Key exists in flat props but value mismatches; still fall
                # through to nested psets (same property name may appear there).

            # Search nested PropertySets (key must exist in pset to match).
            if not _match_flat_in_psets(key, expected):
                return False

        return True

    if action == "get_elements_in_storey":
        storey = params.get("storey")
        if not storey:
            return _err("Missing param: storey", "missing_param")
        if not isinstance(storey, str):
            return _err("Invalid param: storey must be a string", "invalid")
        node, err = _resolve_storey_node(storey)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            return _err(str(error_msg), "not_found")
        if node is None:
            return _err(f"Storey not found: {storey}", "not_found")

        container_classes = {
            "IfcProject",
            "IfcSite",
            "IfcBuilding",
            "IfcBuildingStorey",
            "IfcSpace",
            "IfcZone",
            "IfcSpatialZone",
            "IfcTypeObject",
        }

        elements = []
        for e in _storey_elements(node):
            cls = G.nodes[e].get("class_")
            if cls in container_classes:
                continue
            elements.append(
                {
                    "id": e,
                    "label": G.nodes[e].get("label"),
                    "class_": cls,
                }
            )
        return _ok_action(action, {"storey": storey, "elements": elements})

    if action == "get_elements_in_zone":
        zone = params.get("zone")
        if not zone:
            return _err("Missing param: zone", "missing_param")
        if not isinstance(zone, str):
            return _err("Invalid param: zone must be a string", "invalid")

        max_results, max_results_err = _parse_positive_int_param(
            "max_results",
            default=ACTION_LIST_MAX_RESULTS_DEFAULT,
            hard_limit=ACTION_LIST_MAX_RESULTS_HARD_LIMIT,
        )
        if max_results_err:
            return max_results_err
        if max_results is None:
            return _err("Invalid param: max_results must be an integer", "invalid")

        resolved_zone, err = _resolve_context_node(
            zone,
            param_name="zone",
            prefix="Zone",
            class_name="IfcZone",
            display_name="Zone",
            enable_fuzzy=True,
        )
        if err:
            return _resolver_error_envelope(err)
        if resolved_zone is None:
            return _err(f"Zone not found: {zone}", "not_found")

        all_elements = _incoming_membership_nodes(resolved_zone, "in_zone")
        elements, truncated = _apply_result_limit(all_elements, max_results)
        return _ok_action(
            action,
            {
                "zone": resolved_zone,
                "elements": elements,
                "max_results": max_results,
                "truncated": truncated,
            },
        )

    if action == "get_elements_in_system":
        system = params.get("system")
        if not system:
            return _err("Missing param: system", "missing_param")
        if not isinstance(system, str):
            return _err("Invalid param: system must be a string", "invalid")

        max_results, max_results_err = _parse_positive_int_param(
            "max_results",
            default=ACTION_LIST_MAX_RESULTS_DEFAULT,
            hard_limit=ACTION_LIST_MAX_RESULTS_HARD_LIMIT,
        )
        if max_results_err:
            return max_results_err
        if max_results is None:
            return _err("Invalid param: max_results must be an integer", "invalid")

        resolved_system, err = _resolve_context_node(
            system,
            param_name="system",
            prefix="System",
            class_name="IfcSystem",
            display_name="System",
            enable_fuzzy=True,
        )
        if err:
            return _resolver_error_envelope(err)
        if resolved_system is None:
            return _err(f"System not found: {system}", "not_found")

        all_elements = _incoming_membership_nodes(
            resolved_system,
            "belongs_to_system",
        )
        elements, truncated = _apply_result_limit(all_elements, max_results)
        return _ok_action(
            action,
            {
                "system": resolved_system,
                "elements": elements,
                "max_results": max_results,
                "truncated": truncated,
            },
        )

    if action == "get_elements_by_classification":
        classification = params.get("classification")
        if not classification:
            return _err("Missing param: classification", "missing_param")
        if not isinstance(classification, str):
            return _err(
                "Invalid param: classification must be a string",
                "invalid",
            )

        max_results, max_results_err = _parse_positive_int_param(
            "max_results",
            default=ACTION_LIST_MAX_RESULTS_DEFAULT,
            hard_limit=ACTION_LIST_MAX_RESULTS_HARD_LIMIT,
        )
        if max_results_err:
            return max_results_err
        if max_results is None:
            return _err("Invalid param: max_results must be an integer", "invalid")

        resolved_classification, err = _resolve_classification_node(classification)
        if err:
            return _resolver_error_envelope(err)
        if resolved_classification is None:
            return _err(
                f"Classification not found: {classification}",
                "not_found",
            )

        all_elements = _incoming_membership_nodes(
            resolved_classification,
            "classified_as",
        )
        elements, truncated = _apply_result_limit(all_elements, max_results)
        return _ok_action(
            action,
            {
                "classification": resolved_classification,
                "elements": elements,
                "max_results": max_results,
                "truncated": truncated,
            },
        )

    if action == "get_elements_in_space":
        space = params.get("space")
        if not space:
            return _err("Missing param: space", "missing_param")
        if not isinstance(space, str):
            return _err("Invalid param: space must be a string", "invalid")

        max_results, max_results_err = _parse_positive_int_param(
            "max_results",
            default=ACTION_LIST_MAX_RESULTS_DEFAULT,
            hard_limit=ACTION_LIST_MAX_RESULTS_HARD_LIMIT,
        )
        if max_results_err:
            return max_results_err
        if max_results is None:
            return _err("Invalid param: max_results must be an integer", "invalid")

        resolved_space, err = _resolve_context_node(
            space,
            param_name="space",
            prefix="Element",
            class_name="IfcSpace",
            display_name="IfcSpace",
            enable_fuzzy=True,
        )
        if err:
            return _resolver_error_envelope(err)
        if resolved_space is None:
            return _err(f"IfcSpace not found: {space}", "not_found")

        container_classes = {
            "IfcProject",
            "IfcSite",
            "IfcBuilding",
            "IfcBuildingStorey",
            "IfcSpace",
            "IfcZone",
            "IfcSpatialZone",
            "IfcTypeObject",
        }

        elements = []
        for node_id, depth, edge in _contains_descendants(resolved_space):
            if not node_id.startswith("Element::"):
                continue
            cls = G.nodes[node_id].get("class_")
            if cls in container_classes:
                continue
            edge_relation = _edge_relation(edge)
            elements.append(
                {
                    "id": node_id,
                    "label": G.nodes[node_id].get("label"),
                    "class_": cls,
                    "relation": edge_relation,
                    "source": _edge_source(edge, edge_relation),
                    "depth": depth,
                }
            )

        bounded_elements, truncated = _apply_result_limit(elements, max_results)
        return _ok_action(
            action,
            {
                "space": resolved_space,
                "elements": bounded_elements,
                "max_results": max_results,
                "truncated": truncated,
            },
        )

    if action == "get_hosted_elements":
        element_id = params.get("element_id")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")

        max_results, max_results_err = _parse_positive_int_param(
            "max_results",
            default=ACTION_LIST_MAX_RESULTS_DEFAULT,
            hard_limit=ACTION_LIST_MAX_RESULTS_HARD_LIMIT,
        )
        if max_results_err:
            return max_results_err
        if max_results is None:
            return _err("Invalid param: max_results must be an integer", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            return _resolve_element_error_envelope(err)
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        hosted: dict[str, dict[str, Any]] = {}

        for nbr in G.successors(resolved):
            edge = G[resolved][nbr]
            if _edge_relation(edge) != "hosts":
                continue
            hosted_id = str(nbr)
            if not hosted_id.startswith("Element::"):
                continue
            hosted[hosted_id] = {
                "id": hosted_id,
                "label": G.nodes[hosted_id].get("label"),
                "class_": G.nodes[hosted_id].get("class_"),
                "relation": "hosts",
                "source": _edge_source(edge, "hosts"),
            }

        for nbr in G.predecessors(resolved):
            edge = G[nbr][resolved]
            if _edge_relation(edge) != "hosted_by":
                continue
            hosted_id = str(nbr)
            if not hosted_id.startswith("Element::"):
                continue
            hosted.setdefault(
                hosted_id,
                {
                    "id": hosted_id,
                    "label": G.nodes[hosted_id].get("label"),
                    "class_": G.nodes[hosted_id].get("class_"),
                    "relation": "hosts",
                    "source": _edge_source(edge, "hosts"),
                },
            )

        all_hosted_elements = [hosted[node_id] for node_id in _sorted_node_ids(hosted)]
        hosted_elements, truncated = _apply_result_limit(
            all_hosted_elements, max_results
        )
        return _ok_action(
            action,
            {
                "element_id": resolved,
                "hosted_elements": hosted_elements,
                "max_results": max_results,
                "truncated": truncated,
            },
        )

    if action == "get_host":
        element_id = params.get("element_id")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            return _resolve_element_error_envelope(err)
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        hosts: dict[str, dict[str, Any]] = {}

        for nbr in G.successors(resolved):
            edge = G[resolved][nbr]
            if _edge_relation(edge) != "hosted_by":
                continue
            host_id = str(nbr)
            if not host_id.startswith("Element::"):
                continue
            hosts[host_id] = {
                "id": host_id,
                "label": G.nodes[host_id].get("label"),
                "class_": G.nodes[host_id].get("class_"),
                "relation": "hosted_by",
                "source": _edge_source(edge, "hosted_by"),
            }

        for nbr in G.predecessors(resolved):
            edge = G[nbr][resolved]
            if _edge_relation(edge) != "hosts":
                continue
            host_id = str(nbr)
            if not host_id.startswith("Element::"):
                continue
            hosts.setdefault(
                host_id,
                {
                    "id": host_id,
                    "label": G.nodes[host_id].get("label"),
                    "class_": G.nodes[host_id].get("class_"),
                    "relation": "hosted_by",
                    "source": _edge_source(edge, "hosted_by"),
                },
            )

        ordered_host_ids = _sorted_node_ids(hosts)
        if not ordered_host_ids:
            return _ok_action(action, {"element_id": resolved, "host": None})
        if len(ordered_host_ids) > 1:
            return _err(
                f"Ambiguous host for element_id: {resolved}",
                "ambiguous",
                {"candidates": ordered_host_ids},
            )

        host = hosts[ordered_host_ids[0]]
        return _ok_action(action, {"element_id": resolved, "host": host})

    if action == "trace_mep_network":
        element_id = params.get("element_id")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")

        max_depth, max_depth_err = _parse_positive_int_param(
            "max_depth",
            default=TRACE_MEP_MAX_DEPTH_DEFAULT,
        )
        if max_depth_err:
            return max_depth_err
        if max_depth is None:
            return _err("Invalid param: max_depth must be an integer", "invalid")

        max_results, max_results_err = _parse_positive_int_param(
            "max_results",
            default=TRACE_MEP_MAX_RESULTS_DEFAULT,
            hard_limit=TRACE_MEP_MAX_RESULTS_HARD_LIMIT,
        )
        if max_results_err:
            return max_results_err
        if max_results is None:
            return _err("Invalid param: max_results must be an integer", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            return _resolve_element_error_envelope(err)
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        network_frontier: deque[tuple[str, int]] = deque([(resolved, 0)])
        visited = {resolved}
        results = []
        truncated = False

        while network_frontier and not truncated:
            node_id, depth = network_frontier.popleft()
            if depth >= max_depth:
                continue

            candidates: dict[str, dict[str, Any]] = {}
            for nbr in G.successors(node_id):
                edge = G[node_id][nbr]
                if _edge_relation(edge) != "ifc_connected_to":
                    continue
                nbr_id = str(nbr)
                if not nbr_id.startswith("Element::"):
                    continue
                candidates[nbr_id] = edge

            for nbr in G.predecessors(node_id):
                edge = G[nbr][node_id]
                if _edge_relation(edge) != "ifc_connected_to":
                    continue
                nbr_id = str(nbr)
                if not nbr_id.startswith("Element::"):
                    continue
                candidates.setdefault(nbr_id, edge)

            for nbr_id in _sorted_node_ids(candidates):
                if nbr_id in visited:
                    continue

                if len(results) >= max_results:
                    truncated = True
                    break

                visited.add(nbr_id)
                network_frontier.append((nbr_id, depth + 1))
                edge = candidates[nbr_id]
                relation = "ifc_connected_to"
                results.append(
                    {
                        "from": node_id,
                        "to": nbr_id,
                        "relation": relation,
                        "source": _edge_source(edge, relation),
                        "hops": depth + 1,
                        "node": build_node_payload(
                            nbr_id,
                            G.nodes[nbr_id],
                            payload_mode=resolved_payload_mode,
                        ),
                    }
                )

        return _ok_action(
            action,
            {
                "element_id": resolved,
                "max_depth": max_depth,
                "max_results": max_results,
                "truncated": truncated,
                "results": results,
            },
        )

    if action == "find_elements_by_class":
        cls = params.get("class")
        if not cls:
            return _err("Missing param: class", "missing_param")
        if not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        target = _normalize_class(cls)

        matches = []
        for n, d in G.nodes(data=True):
            if str(d.get("class_", "")).lower() == target.lower():
                matches.append(
                    build_node_payload(n, d, payload_mode=resolved_payload_mode)
                )
        return _ok_action(action, {"class": target, "elements": matches})

    if action == "get_adjacent_elements":
        element_id = params.get("element_id")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")
        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        neighbors = []
        for nbr, edge in _spatial_neighbors(resolved):
            edge_relation = _edge_relation(edge)
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "distance": edge.get("distance"),
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(action, {"element_id": resolved, "adjacent": neighbors})

    if action == "get_topology_neighbors":
        element_id = params.get("element_id")
        relation = params.get("relation")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")
        if not relation:
            return _err("Missing param: relation", "missing_param")
        if not isinstance(relation, str):
            return _err("Invalid param: relation must be a string", "invalid")

        relation_value = normalize_relation_name(relation)
        allowed = set(TOPOLOGY_RELATIONS)
        if relation_value not in allowed:
            return _err(
                f"Unsupported topology relation: {relation}",
                "invalid",
                {"allowed_relations": sorted(allowed)},
            )

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        neighbors = []
        for nbr, edge in _topology_neighbors(resolved, {relation_value}):
            edge_relation = _edge_relation(edge)
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "vertical_gap": edge.get("vertical_gap"),
                    "overlap_area_xy": edge.get("overlap_area_xy"),
                    "intersection_volume": edge.get("intersection_volume"),
                    "contact_area": edge.get("contact_area"),
                    "source": _edge_source(edge, edge_relation),
                }
            )

        return _ok_action(
            action,
            {
                "element_id": resolved,
                "relation": relation_value,
                "neighbors": neighbors,
            },
        )

    if action == "get_intersections_3d":
        element_id = params.get("element_id")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        neighbors = []
        for nbr, edge in _topology_neighbors(resolved, {"intersects_3d"}):
            edge_relation = _edge_relation(edge)
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "intersection_volume": edge.get("intersection_volume"),
                    "contact_area": edge.get("contact_area"),
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(
            action, {"element_id": resolved, "intersections_3d": neighbors}
        )

    if action == "find_nodes":
        cls = params.get("class")
        if cls is not None and not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        class_filter = _normalize_class(cls) if cls else None
        property_filters = params.get("property_filters", {})
        if property_filters and not isinstance(property_filters, dict):
            return _err("Invalid param: property_filters must be an object", "invalid")

        db_lookup_conn: Any | None = None
        db_path_raw: Any = G.graph.get("_db_path")
        if property_filters and db_path_raw is not None:
            from rag_tag.sql_element_lookup import (  # noqa: PLC0415
                open_lookup_connection,
            )

            db_lookup_conn = open_lookup_connection(Path(db_path_raw))

        matches = []
        try:
            for n, d in G.nodes(data=True):
                if class_filter is not None:
                    if str(d.get("class_", "")).lower() != class_filter.lower():
                        continue
                if not _apply_property_filters(
                    n,
                    property_filters,
                    db_conn=db_lookup_conn,
                ):
                    continue
                matches.append(
                    build_node_payload(n, d, payload_mode=resolved_payload_mode)
                )
        finally:
            if db_lookup_conn is not None:
                db_lookup_conn.close()
        return _ok_action(action, {"class": class_filter, "elements": matches})

    if action == "traverse":
        start = params.get("start")
        relation_param = params.get("relation")
        if not start:
            return _err("Missing param: start", "missing_param")
        if not isinstance(start, str):
            return _err("Invalid param: start must be a string", "invalid")
        if relation_param is not None and not isinstance(relation_param, str):
            return _err("Invalid param: relation must be a string", "invalid")
        try:
            depth = int(params.get("depth", 1))
        except (TypeError, ValueError):
            return _err("Invalid param: depth must be an integer", "invalid")
        if start not in G:
            return _err(f"Start node not found: {start}", "not_found")
        if depth < 1:
            return _err("Depth must be >= 1", "invalid")

        visited = {start}
        frontier = {start}
        results = []
        relation_filter: set[str] | None = None
        relation_value: str | None = None
        if relation_param is not None:
            relation_value = normalize_relation_name(relation_param)
            if relation_value is None:
                return _err("Invalid param: relation must be non-empty", "invalid")
            if relation_value not in CANONICAL_RELATION_SET:
                return _err(
                    f"Unsupported traverse relation: {relation_param}",
                    "invalid",
                    {"allowed_relations": sorted(CANONICAL_RELATION_SET)},
                )
            relation_filter = {relation_value}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for nbr in G.successors(node):
                    edge = G[node][nbr]
                    edge_relation = _edge_relation(edge)
                    if edge_relation is None:
                        continue
                    if relation_filter and edge_relation not in relation_filter:
                        continue
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    results.append(
                        {
                            "from": node,
                            "to": nbr,
                            "relation": edge_relation,
                            "source": _edge_source(edge, edge_relation),
                            "node": build_node_payload(
                                nbr,
                                G.nodes[nbr],
                                payload_mode=resolved_payload_mode,
                            ),
                        }
                    )
            frontier = next_frontier

        return _ok_action(
            action,
            {
                "start": start,
                "relation": relation_value,
                "depth": depth,
                "results": results,
            },
        )

    if action == "spatial_query":
        cls = params.get("class")
        if cls is not None and not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        class_filter = _normalize_class(cls) if cls else None
        near = params.get("near")
        max_distance = params.get("max_distance")
        if near is None:
            return _err("Missing param: near", "missing_param")
        if not isinstance(near, (str, int, float)):
            return _err("Invalid param: near must be a string or number", "invalid")
        resolved, err = _resolve_element_id(str(near))
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {near}", "not_found")
        if max_distance is None:
            return _err("Missing param: max_distance", "missing_param")
        try:
            max_distance_value = float(max_distance)
        except (TypeError, ValueError):
            return _err("Invalid param: max_distance must be a number", "invalid")

        results = []
        for nbr, edge in _spatial_neighbors(resolved):
            dist = edge.get("distance")
            if dist is None or float(dist) > max_distance_value:
                continue
            if class_filter is not None:
                if str(G.nodes[nbr].get("class_", "")).lower() != class_filter.lower():
                    continue
            edge_relation = _edge_relation(edge)
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "distance": dist,
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(
            action,
            {
                "near": resolved,
                "max_distance": max_distance_value,
                "results": results,
            },
        )

    if action == "find_elements_above":
        element_id = params.get("element_id")
        max_gap = params.get("max_gap")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")
        max_gap_value: float | None = None
        if max_gap is not None:
            try:
                max_gap_value = float(max_gap)
            except (TypeError, ValueError):
                return _err("Invalid param: max_gap must be a number", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        results = []
        for nbr, edge in _topology_neighbors(resolved, {"above"}):
            gap = edge.get("vertical_gap")
            if (
                max_gap_value is not None
                and gap is not None
                and float(gap) > max_gap_value
            ):
                continue
            edge_relation = _edge_relation(edge)
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "vertical_gap": gap,
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(
            action,
            {
                "element_id": resolved,
                "max_gap": max_gap_value,
                "results": results,
            },
        )

    if action == "find_elements_below":
        element_id = params.get("element_id")
        max_gap = params.get("max_gap")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")
        max_gap_value: float | None = None
        if max_gap is not None:
            try:
                max_gap_value = float(max_gap)
            except (TypeError, ValueError):
                return _err("Invalid param: max_gap must be a number", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if resolved is None:
            return _err(f"Element not found: {element_id}", "not_found")

        results = []
        for nbr, edge in _topology_neighbors(resolved, {"below"}):
            gap = edge.get("vertical_gap")
            if (
                max_gap_value is not None
                and gap is not None
                and float(gap) > max_gap_value
            ):
                continue
            edge_relation = _edge_relation(edge)
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge_relation,
                    "vertical_gap": gap,
                    "source": _edge_source(edge, edge_relation),
                }
            )
        return _ok_action(
            action,
            {
                "element_id": resolved,
                "max_gap": max_gap_value,
                "results": results,
            },
        )

    if action == "list_property_keys":
        cls = params.get("class")
        sample_values = params.get("sample_values", False)

        if cls is not None and not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        if not isinstance(sample_values, bool):
            return _err("Invalid param: sample_values must be a boolean", "invalid")

        class_filter = _normalize_class(cls) if cls else None
        key_samples: dict[str, list[Any]] = {}

        def _record_key(key: str, value: Any) -> None:
            if key not in key_samples:
                key_samples[key] = []
            if sample_values and len(key_samples[key]) < 3:
                key_samples[key].append(value)

        def _collect_pset_leaf_keys(
            pset_name: str,
            node: dict[str, Any],
            path_prefix: str = "",
        ) -> None:
            for raw_key, raw_value in node.items():
                key_part = str(raw_key)
                path = f"{path_prefix}.{key_part}" if path_prefix else key_part
                if isinstance(raw_value, dict):
                    _collect_pset_leaf_keys(pset_name, raw_value, path)
                else:
                    _record_key(f"{pset_name}.{path}", raw_value)

        for _, data in G.nodes(data=True):
            if class_filter is not None:
                if str(data.get("class_", "")).lower() != class_filter.lower():
                    continue

            props = data.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            for key, value in props.items():
                _record_key(str(key), value)

            payload = data.get("payload") or {}
            if not isinstance(payload, dict):
                continue

            pset_block = payload.get("PropertySets") or {}
            if isinstance(pset_block, dict):
                for section in ("Official", "Custom"):
                    section_block = pset_block.get(section) or {}
                    if not isinstance(section_block, dict):
                        continue
                    for pset_name, pset_props in section_block.items():
                        if not isinstance(pset_props, dict):
                            continue
                        _collect_pset_leaf_keys(str(pset_name), pset_props)

            quantities_block = payload.get("Quantities") or {}
            if not isinstance(quantities_block, dict):
                continue
            for qto_name, qto_data in quantities_block.items():
                if not isinstance(qto_data, dict):
                    continue
                _collect_pset_leaf_keys(str(qto_name), qto_data)

        payload_mode_value = str(G.graph.get("_payload_mode", "full")).lower()
        db_path_raw = G.graph.get("_db_path")
        if payload_mode_value == "minimal" and db_path_raw is not None:
            db_path = Path(db_path_raw)
            cache = G.graph.setdefault("_property_key_cache", {})
            cache_key = (str(db_path.resolve()), class_filter or "")
            db_key_samples = cache.get(cache_key)
            if db_key_samples is None:
                db_key_samples = _collect_dotted_keys_from_sqlite(db_path, class_filter)
                cache[cache_key] = db_key_samples

            for key, samples in db_key_samples.items():
                if sample_values:
                    existing = key_samples.setdefault(key, [])
                    for sample in samples:
                        if len(existing) >= 3:
                            break
                        existing.append(sample)
                else:
                    _record_key(key, None)

        data: dict[str, Any] = {
            "keys": sorted(key_samples.keys()),
            "class_filter": class_filter,
            "class_filter_raw": cls,
        }
        if sample_values:
            data["samples"] = key_samples
        return _ok_action(action, data)

    if action == "get_element_properties":
        element_id = params.get("element_id")
        if element_id is None or element_id == "":
            return _err("Missing param: element_id", "missing_param")
        if not isinstance(element_id, str):
            return _err("Invalid param: element_id must be a string", "invalid")

        resolved, err = _resolve_element_id(element_id)
        if err:
            error_msg = err.get("error", "Unknown error")
            if "Ambiguous" in str(error_msg):
                return _err(
                    str(error_msg),
                    "ambiguous",
                    {"candidates": err.get("candidates", [])},
                )
            if "Invalid element_id" in str(error_msg):
                return _err(str(error_msg), "invalid")
            return _err(str(error_msg), "not_found")
        if not resolved:
            return _err(f"Element not found: {element_id}", "not_found")

        base_node_data: dict[str, Any] = dict(G.nodes[resolved])

        # Attempt DB-backed property enrichment when a DB path is wired into
        # the graph context.  Uses the session-level cache to avoid reopening
        # the database for elements already fetched during filter evaluation.
        db_path_raw: Any = G.graph.get("_db_path")
        if db_path_raw is not None:
            db_path = Path(db_path_raw)
            db_data: dict[str, Any] | None = _cached_db_lookup(G, resolved, db_path)
            if db_data is not None:
                base_node_data = _merge_db_element_data(base_node_data, db_data)
            else:
                _logger.debug(
                    "get_element_properties: DB lookup found no row for %s"
                    " (db=%s) — using in-memory payload",
                    resolved,
                    db_path,
                )

        return _ok_action(
            action,
            build_node_payload(
                resolved, base_node_data, payload_mode=INTERNAL_PAYLOAD_MODE
            ),
        )

    return _err(
        f"Unknown action: {action}",
        "unknown_action",
        {"allowed_actions": sorted(CANONICAL_ACTIONS)},
    )
