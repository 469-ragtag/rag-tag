from __future__ import annotations

import logging
import re
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, Iterable

import networkx as nx

_logger = logging.getLogger(__name__)

# Sentinel: distinguishes "not yet cached" from "cached, result was None".
_CACHE_MISS = object()
_PROPERTY_CACHE_MAX_ENTRIES = 1024

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


def _ok(data: dict) -> dict[str, Any]:
    """Wrap successful tool result in envelope."""
    return {"status": "ok", "data": data, "error": None}


def _err(message: str, code: str, details: dict | None = None) -> dict[str, Any]:
    """Wrap error result in envelope."""
    error_payload: dict[str, Any] = {"message": message, "code": code}
    if details:
        error_payload["details"] = details
    return {"status": "error", "data": None, "error": error_payload}


def _merge_db_element_data(
    node_data: dict[str, Any],
    db_data: dict[str, Any],
) -> dict[str, Any]:
    """Merge DB-sourced element data into in-memory node data.

    DB data enriches (fills in) flat properties and
    PropertySets/Quantities in the payload.  Other node attributes
    (geometry, label, class_, graph edges, etc.) are preserved unchanged
    from the in-memory node.

    Args:
        node_data: Raw ``G.nodes[node_id]`` dict from the NetworkX graph.
        db_data: Structured result from ``sql_element_lookup`` with
            ``properties`` and ``payload`` keys.

    Returns:
        New dict with merged data (the input dicts are not mutated).
    """
    merged: dict[str, Any] = dict(node_data)

    def _merge_missing(
        existing: dict[str, Any], incoming: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge nested mappings while preserving richer in-memory values."""
        merged_dict: dict[str, Any] = dict(existing)
        for key, incoming_value in incoming.items():
            if key not in merged_dict:
                merged_dict[key] = incoming_value
                continue

            existing_value = merged_dict[key]
            if isinstance(existing_value, dict) and isinstance(incoming_value, dict):
                merged_dict[key] = _merge_missing(existing_value, incoming_value)
                continue

            if existing_value is None or existing_value == "":
                merged_dict[key] = incoming_value
        return merged_dict

    # Merge flat properties: DB values only fill missing in-memory fields.
    db_props = db_data.get("properties") or {}
    if db_props:
        existing_props: dict[str, Any] = dict(node_data.get("properties") or {})
        merged["properties"] = _merge_missing(existing_props, db_props)

    # Merge payload sections: DB PropertySets and Quantities enrich in-memory data.
    db_payload = db_data.get("payload") or {}
    if db_payload:
        existing_payload: dict[str, Any] = dict(node_data.get("payload") or {})

        if "PropertySets" in db_payload:
            existing_psets: dict[str, Any] = dict(
                existing_payload.get("PropertySets") or {}
            )
            db_psets: dict[str, Any] = db_payload["PropertySets"]
            for section in ("Official", "Custom"):
                if section in db_psets:
                    existing_section: dict[str, Any] = dict(
                        existing_psets.get(section) or {}
                    )
                    db_section: dict[str, Any] = dict(db_psets.get(section) or {})
                    existing_psets[section] = _merge_missing(
                        existing_section, db_section
                    )
            existing_payload["PropertySets"] = existing_psets

        if "Quantities" in db_payload:
            existing_qty: dict[str, Any] = dict(
                existing_payload.get("Quantities") or {}
            )
            db_qty: dict[str, Any] = dict(db_payload.get("Quantities") or {})
            existing_payload["Quantities"] = _merge_missing(existing_qty, db_qty)

        merged["payload"] = existing_payload

    return merged


def _get_property_cache(G: nx.DiGraph) -> OrderedDict[tuple[str, str], Any]:
    """Return or create the session-level property cache stored on the graph.

    The cache lives at ``G.graph["_property_cache"]`` and is keyed by
    ``(<resolved-db-path>, <node-id>)``.
    Values are DB element data dicts (or ``None`` when not found), with
    ``_CACHE_MISS`` used as the "not yet fetched" sentinel so that a ``None``
    result (element absent from DB) is not re-fetched on subsequent calls.

    The cache is bounded (LRU-style) to avoid unbounded memory growth in
    long-lived sessions.
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
    db_conn: Any = None,
) -> dict[str, Any] | None:
    """Look up element data from the SQLite DB with a graph-level result cache.

    Avoids reopening the database for the same element within a single agent
    session. The first call for a given ``(db_path, node_id)`` pair performs
    the lookup and stores the result (or ``None``) in
    ``G.graph["_property_cache"]``.
    Subsequent calls return the cached value immediately.

    Args:
        G: NetworkX graph with optional ``_property_cache`` graph attribute.
        node_id: Graph node identifier used as part of the cache key.
        db_path: Path to the SQLite database file.
        db_conn: Optional open SQLite connection reused across many lookups.

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
        db_data = lookup_element_by_globalid(db_path, str(global_id), conn=db_conn)

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


def query_ifc_graph(
    G: nx.DiGraph,
    action: str,
    params: Dict[str, Any],
    *,
    payload_mode: str = LLM_PAYLOAD_MODE,
) -> Dict[str, Any]:
    """Controlled interface between the LLM and NetworkX graph."""
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

    def _resolve_element_id(
        element_id: str,
    ) -> tuple[str | None, Dict[str, Any] | None]:
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

        # Direct id match supports new Storey::<GlobalId> identity.
        direct = f"Storey::{query}"
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

    def _storey_elements(start: str) -> Iterable[str]:
        """Traverse only containment edges to avoid leakage via spatial links."""
        visited = {start}
        q = deque([start])
        while q:
            node = q.popleft()
            for nbr in G.successors(node):
                edge = G[node][nbr]
                if edge.get("relation") not in {"contains", "contained_in"}:
                    continue
                if nbr in visited:
                    continue
                visited.add(nbr)
                q.append(nbr)
                yield nbr

    def _spatial_neighbors(node_id: str) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Yield unique spatial neighbors across both outgoing and incoming edges."""
        seen: set[str] = set()
        spatial_relations = {"adjacent_to", "connected_to"}

        for nbr in G.successors(node_id):
            if nbr in seen:
                continue
            edge = G[node_id][nbr]
            if edge.get("relation") not in spatial_relations:
                continue
            seen.add(nbr)
            yield nbr, edge

        for nbr in G.predecessors(node_id):
            if nbr in seen:
                continue
            edge = G[nbr][node_id]
            if edge.get("relation") not in spatial_relations:
                continue
            seen.add(nbr)
            yield nbr, edge

    def _topology_neighbors(
        node_id: str,
        allowed_relations: set[str] | None = None,
    ) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Yield unique topology neighbors across both directions."""
        seen: set[tuple[str, str]] = set()
        topology_relations = {
            "above",
            "below",
            "overlaps_xy",
            "intersects_bbox",
            "intersects_3d",
            "touches_surface",
        }
        if allowed_relations is None:
            allowed_relations = topology_relations

        for nbr in G.successors(node_id):
            edge = G[node_id][nbr]
            relation = str(edge.get("relation"))
            if relation not in allowed_relations:
                continue
            key = (nbr, relation)
            if key in seen:
                continue
            seen.add(key)
            yield nbr, edge

        for nbr in G.predecessors(node_id):
            edge = G[nbr][node_id]
            relation = str(edge.get("relation"))
            if relation not in allowed_relations:
                continue
            key = (nbr, relation)
            if key in seen:
                continue
            seen.add(key)
            yield nbr, edge

    def _apply_property_filters(
        node_id: str,
        filters: Dict[str, Any],
        db_conn: Any = None,
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

        def _flat_prop_matches(prop_val: Any, expected: Any) -> bool:
            # List-valued property (e.g. Materials): support membership testing
            # so {"Materials": "gypsum"} matches ["gypsum", ...].
            if isinstance(prop_val, list):
                if isinstance(expected, str):
                    return expected in prop_val
                if isinstance(expected, list):
                    return all(v in prop_val for v in expected)
                return False
            return prop_val == expected

        # Minimal payload mode: pset_block will be empty because PropertySets
        # are not stored in-memory. When nested pset lookups are required and a
        # DB path is wired into the graph, fetch the full element data from the
        # DB (session-cached) so filters like "Pset_WallCommon.FireRating" or
        # flat-key fallbacks like {"ThermalTransmittance": 0.3} still work.
        _has_dotted_filter = any("." in k for k in filters)
        _needs_flat_pset_fallback = any(
            "." not in key
            and (key not in props or not _flat_prop_matches(props.get(key), expected))
            for key, expected in filters.items()
        )
        if (_has_dotted_filter or _needs_flat_pset_fallback) and not pset_block:
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
                if _flat_prop_matches(props[key], expected):
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
        return _ok({"storey": storey, "elements": elements})

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
        return _ok({"class": target, "elements": matches})

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
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge.get("relation"),
                    "distance": edge.get("distance"),
                }
            )
        return _ok({"element_id": resolved, "adjacent": neighbors})

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

        relation_value = relation.strip().lower()
        allowed = {
            "above",
            "below",
            "overlaps_xy",
            "intersects_bbox",
            "intersects_3d",
            "touches_surface",
        }
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
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge.get("relation"),
                    "vertical_gap": edge.get("vertical_gap"),
                    "overlap_area_xy": edge.get("overlap_area_xy"),
                    "intersection_volume": edge.get("intersection_volume"),
                    "contact_area": edge.get("contact_area"),
                    "source": edge.get("source"),
                }
            )

        return _ok(
            {
                "element_id": resolved,
                "relation": relation_value,
                "neighbors": neighbors,
            }
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
            neighbors.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge.get("relation"),
                    "intersection_volume": edge.get("intersection_volume"),
                    "contact_area": edge.get("contact_area"),
                    "source": edge.get("source"),
                }
            )
        return _ok({"element_id": resolved, "intersections_3d": neighbors})

    if action == "find_nodes":
        cls = params.get("class")
        if cls is not None and not isinstance(cls, str):
            return _err("Invalid param: class must be a string", "invalid")
        class_filter = _normalize_class(cls) if cls else None
        property_filters = params.get("property_filters", {})
        if property_filters and not isinstance(property_filters, dict):
            return _err("Invalid param: property_filters must be an object", "invalid")

        db_conn = None
        try:
            db_path_raw = G.graph.get("_db_path")
            if db_path_raw is not None and property_filters:
                from rag_tag.sql_element_lookup import (  # noqa: PLC0415
                    open_lookup_connection,
                )

                db_conn = open_lookup_connection(Path(db_path_raw))

            matches = []
            for n, d in G.nodes(data=True):
                if class_filter is not None:
                    if str(d.get("class_", "")).lower() != class_filter.lower():
                        continue
                if not _apply_property_filters(n, property_filters, db_conn=db_conn):
                    continue
                matches.append(
                    build_node_payload(n, d, payload_mode=resolved_payload_mode)
                )
        finally:
            if db_conn is not None:
                db_conn.close()
        return _ok({"class": class_filter, "elements": matches})

    if action == "traverse":
        start = params.get("start")
        relation = params.get("relation")
        if not start:
            return _err("Missing param: start", "missing_param")
        if not isinstance(start, str):
            return _err("Invalid param: start must be a string", "invalid")
        if relation is not None and not isinstance(relation, str):
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
        if relation:
            relation_filter = {relation}
            if relation in {"contains", "contained_in"}:
                relation_filter = {"contains", "contained_in"}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for nbr in G.successors(node):
                    edge = G[node][nbr]
                    if relation_filter and edge.get("relation") not in relation_filter:
                        continue
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    results.append(
                        {
                            "from": node,
                            "to": nbr,
                            "relation": edge.get("relation"),
                            "node": build_node_payload(
                                nbr,
                                G.nodes[nbr],
                                payload_mode=resolved_payload_mode,
                            ),
                        }
                    )
                # Backward-compatibility for legacy graphs that only encoded
                # container->child using relation="contained_in".
                if relation == "contained_in":
                    for pred in G.predecessors(node):
                        edge = G[pred][node]
                        if edge.get("relation") != "contained_in":
                            continue
                        if pred in visited:
                            continue
                        visited.add(pred)
                        next_frontier.add(pred)
                        results.append(
                            {
                                "from": node,
                                "to": pred,
                                "relation": edge.get("relation"),
                                "node": build_node_payload(
                                    pred,
                                    G.nodes[pred],
                                    payload_mode=resolved_payload_mode,
                                ),
                            }
                        )
            frontier = next_frontier

        return _ok(
            {
                "start": start,
                "relation": relation,
                "depth": depth,
                "results": results,
            }
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
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": edge.get("relation"),
                    "distance": dist,
                }
            )
        return _ok(
            {
                "near": resolved,
                "max_distance": max_distance_value,
                "results": results,
            }
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
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": "above",
                    "vertical_gap": gap,
                }
            )
        return _ok(
            {
                "element_id": resolved,
                "max_gap": max_gap_value,
                "results": results,
            }
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
            results.append(
                {
                    "id": nbr,
                    "label": G.nodes[nbr].get("label"),
                    "class_": G.nodes[nbr].get("class_"),
                    "relation": "below",
                    "vertical_gap": gap,
                }
            )
        return _ok(
            {
                "element_id": resolved,
                "max_gap": max_gap_value,
                "results": results,
            }
        )

    if action == "get_element_properties":
        element_id = params.get("element_id")
        if not element_id:
            return _err("Missing param: element_id", "missing_param")

        resolved, err = _resolve_element_id(element_id)
        if err or not resolved:
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

        return _ok(
            build_node_payload(
                resolved, base_node_data, payload_mode=INTERNAL_PAYLOAD_MODE
            )
        )

    return _err(f"Unknown action: {action}", "unknown_action")
