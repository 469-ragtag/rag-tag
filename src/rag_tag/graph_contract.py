"""Canonical graph action/relation contract shared across graph backends.

This module is the single source of truth for:
- allowed graph action names
- canonical relation taxonomy buckets
- known relation source semantics
- stable response envelope helpers
- required data fields for key action payloads
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

# ---------------------------------------------------------------------------
# Canonical actions
# ---------------------------------------------------------------------------

CANONICAL_ACTIONS: tuple[str, ...] = (
    "get_elements_in_storey",
    "find_elements_by_class",
    "find_elements_inside_footprint",
    "find_same_storey_elements",
    "resolve_element_set",
    "relate_element_set",
    "get_adjacent_elements",
    "get_topology_neighbors",
    "get_intersections_3d",
    "find_nodes",
    "traverse",
    "spatial_query",
    "find_elements_above",
    "find_elements_below",
    "get_element_properties",
    "list_property_keys",
    "trace_distribution_network",
    "find_equipment_serving_space",
    "find_shortest_path",
    "find_by_classification",
    "aggregate_elements",
    "group_elements_by_property",
)

CANONICAL_ACTION_SET = frozenset(CANONICAL_ACTIONS)

ROADMAP_ACTIONS: tuple[str, ...] = (
    "trace_distribution_network",
    "find_equipment_serving_space",
    "find_shortest_path",
    "find_by_classification",
    "aggregate_elements",
    "group_elements_by_property",
)

ROADMAP_ACTION_SET = frozenset(ROADMAP_ACTIONS)


# ---------------------------------------------------------------------------
# Canonical relation taxonomy
# ---------------------------------------------------------------------------

HIERARCHY_RELATIONS: tuple[str, ...] = (
    "aggregates",
    "contains",
    "contained_in",
)

SPATIAL_RELATIONS: tuple[str, ...] = (
    "adjacent_to",
    "connected_to",
)

TOPOLOGY_RELATIONS: tuple[str, ...] = (
    "above",
    "below",
    "aligned_with",
    "overlaps_xy",
    "intersects_bbox",
    "intersects_3d",
    "touches_surface",
    "space_bounded_by",
    "bounds_space",
    "shares_boundary_with",
    "path_connected_to",
)

EXPLICIT_IFC_RELATIONS: tuple[str, ...] = (
    "hosts",
    "hosted_by",
    "ifc_connected_to",
    "typed_by",
    "belongs_to_system",
    "in_zone",
    "classified_as",
)

SYMMETRIC_RELATIONS: tuple[str, ...] = (
    "adjacent_to",
    "connected_to",
    "aligned_with",
    "overlaps_xy",
    "intersects_bbox",
    "intersects_3d",
    "touches_surface",
    "shares_boundary_with",
    "ifc_connected_to",
    "path_connected_to",
)

RELATION_TAXONOMY: dict[str, tuple[str, ...]] = {
    "hierarchy": HIERARCHY_RELATIONS,
    "spatial": SPATIAL_RELATIONS,
    "topology": TOPOLOGY_RELATIONS,
    "explicit_ifc": EXPLICIT_IFC_RELATIONS,
}

CANONICAL_RELATION_SET = frozenset(
    relation for relations in RELATION_TAXONOMY.values() for relation in relations
)
SYMMETRIC_RELATION_SET = frozenset(SYMMETRIC_RELATIONS)


# ---------------------------------------------------------------------------
# Relation source semantics
# ---------------------------------------------------------------------------

KNOWN_RELATION_SOURCES: tuple[str, ...] = ("ifc", "heuristic", "topology")

RELATION_SOURCE_SEMANTICS: dict[str, str] = {
    "ifc": "Explicit relationship extracted from IFC relationship entities",
    "heuristic": "Derived by geometric/spatial heuristics",
    "topology": "Derived by topology analysis (bbox/mesh overlap ordering)",
}

KNOWN_RELATION_SOURCE_SET = frozenset(KNOWN_RELATION_SOURCES)


# ---------------------------------------------------------------------------
# Action payload invariants
# ---------------------------------------------------------------------------

EVIDENCE_LIMIT = 5
_BOUNDED_LIST_METADATA_DEFAULTS: dict[str, Any] = {
    "total_found": 0,
    "returned_count": 0,
    "truncated": False,
    "truncation_reason": None,
}

# Defaults also define required field presence for each action's data payload.
ACTION_DATA_DEFAULTS: dict[str, dict[str, Any]] = {
    "get_elements_in_storey": {
        "storey": None,
        "elements": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "find_elements_by_class": {
        "class": None,
        "elements": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "find_elements_inside_footprint": {
        "container": None,
        "class": None,
        "elements": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "find_same_storey_elements": {
        "anchor": None,
        "storey_id": None,
        "class": None,
        "elements": [],
    "resolve_element_set": {
        "query": None,
        "class_filter": None,
        "match_mode": None,
        "matches": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "relate_element_set": {
        "anchor_ids": [],
        "relation": None,
        "anchor_count": 0,
        "matched_anchor_count": 0,
        "unmatched_anchor_count": 0,
        "results": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "get_adjacent_elements": {
        "element_id": None,
        "adjacent": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "get_topology_neighbors": {
        "element_id": None,
        "relation": None,
        "neighbors": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "get_intersections_3d": {
        "element_id": None,
        "intersections_3d": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "find_nodes": {
        "class": None,
        "elements": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "traverse": {
        "start": None,
        "relation": None,
        "depth": 1,
        "results": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "spatial_query": {
        "near": None,
        "max_distance": None,
        "results": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "find_elements_above": {
        "element_id": None,
        "max_gap": None,
        "results": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "find_elements_below": {
        "element_id": None,
        "max_gap": None,
        "results": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "get_element_properties": {
        "id": None,
        "label": None,
        "class_": None,
        "properties": {},
        "payload": None,
        "evidence": [],
    },
    "list_property_keys": {"keys": [], "class_filter": None, "evidence": []},
    "trace_distribution_network": {
        "start": None,
        "relation": None,
        "max_depth": None,
        "results": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "find_equipment_serving_space": {
        "space": None,
        "equipment": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "find_shortest_path": {
        "start": None,
        "end": None,
        "relation": None,
        "path": [],
        "evidence": [],
    },
    "find_by_classification": {
        "classification": None,
        "elements": [],
        "evidence": [],
        **_BOUNDED_LIST_METADATA_DEFAULTS,
    },
    "aggregate_elements": {
        "metric": None,
        "field": None,
        "field_source": None,
        "aggregate_value": None,
        "matched_element_count": 0,
        "unmatched_element_count": 0,
        "missing_value_count": 0,
        "sample": [],
        "evidence": [],
    },
    "group_elements_by_property": {
        "property_key": None,
        "field_source": None,
        "groups": [],
        "matched_element_count": 0,
        "unmatched_element_count": 0,
        "missing_value_count": 0,
        "evidence": [],
    },
}
}

ACTION_REQUIRED_DATA_FIELDS: dict[str, tuple[str, ...]] = {
    action: tuple(defaults.keys()) for action, defaults in ACTION_DATA_DEFAULTS.items()
}

ACTION_EVIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "get_elements_in_storey": ("elements",),
    "find_elements_by_class": ("elements",),
    "find_elements_inside_footprint": ("elements",),
    "find_same_storey_elements": ("elements",),
    "resolve_element_set": ("matches",),
    "relate_element_set": ("results",),
    "get_adjacent_elements": ("adjacent",),
    "get_topology_neighbors": ("neighbors",),
    "get_intersections_3d": ("intersections_3d",),
    "find_nodes": ("elements",),
    "traverse": ("results",),
    "spatial_query": ("results",),
    "find_elements_above": ("results",),
    "find_elements_below": ("results",),
    "get_element_properties": (),
    "find_equipment_serving_space": ("equipment",),
    "find_shortest_path": ("path",),
    "find_by_classification": ("elements",),
    "aggregate_elements": (),
    "group_elements_by_property": (),
}


def _fresh_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(value.items())


def _iter_candidate_mappings(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = [record]
    for key in ("node", "element", "item"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    return candidates


def _iter_property_mappings(
    candidates: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    property_mappings: list[Mapping[str, Any]] = []
    for mapping in candidates:
        for key in ("properties", "payload"):
            nested = mapping.get(key)
            if isinstance(nested, Mapping):
                property_mappings.append(nested)
    return property_mappings


def _first_present(
    mappings: Iterable[Mapping[str, Any]],
    *keys: str,
) -> Any:
    for mapping in mappings:
        for key in keys:
            if key not in mapping:
                continue
            value = mapping.get(key)
            if value is None or value == "":
                continue
            return value
    return None


def build_evidence_item(
    record: Mapping[str, Any] | None,
    *,
    source_tool: str | None = None,
    relation: str | None = None,
    match_reason: str | None = None,
    ordinal: int | None = None,
) -> dict[str, Any] | None:
    """Build one compact evidence item from a graph or SQL result record."""
    if not isinstance(record, Mapping):
        return None

    candidates = _iter_candidate_mappings(record)
    all_mappings = [*candidates, *_iter_property_mappings(candidates)]

    global_id = _first_present(all_mappings, "global_id", "GlobalId")
    internal_id = _first_present(
        all_mappings,
        "id",
        "element_id",
        "node_id",
        "express_id",
        "to",
    )
    label = _first_present(all_mappings, "label", "name", "Name")
    class_ = _first_present(all_mappings, "class_", "ifc_class", "Class", "IfcType")
    relation_value = relation or _first_present(candidates, "relation")
    normalized_relation = normalize_relation_name(relation_value)

    evidence: dict[str, Any] = {}
    if global_id is not None:
        evidence["global_id"] = global_id
    if internal_id is not None:
        evidence["id"] = internal_id
    elif ordinal is not None:
        evidence["ordinal"] = ordinal
    if label is not None:
        evidence["label"] = label
    if class_ is not None:
        evidence["class_"] = class_
    if source_tool:
        evidence["source_tool"] = source_tool
    if normalized_relation is not None:
        evidence["relation"] = normalized_relation
    if match_reason:
        evidence["match_reason"] = match_reason

    return evidence or None


def collect_evidence(
    records: Iterable[Mapping[str, Any] | None],
    *,
    source_tool: str | None = None,
    limit: int = EVIDENCE_LIMIT,
    match_reason_builder: Callable[[Mapping[str, Any]], str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build a compact evidence list from result records."""
    evidence_items: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            continue
        match_reason = (
            match_reason_builder(record) if match_reason_builder is not None else None
        )
        evidence_item = build_evidence_item(
            record,
            source_tool=source_tool,
            match_reason=match_reason,
            ordinal=ordinal,
        )
        if evidence_item is None:
            continue
        evidence_items.append(evidence_item)
        if len(evidence_items) >= limit:
            break
    return evidence_items


def merge_evidence_items(
    *evidence_groups: Iterable[Mapping[str, Any]] | None,
    limit: int = EVIDENCE_LIMIT,
) -> list[dict[str, Any]]:
    """Merge compact evidence groups while preserving order and uniqueness."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for group in evidence_groups:
        if group is None:
            continue
        for item in group:
            if not isinstance(item, Mapping):
                continue
            key = (
                item.get("global_id"),
                item.get("id"),
                item.get("ordinal"),
                item.get("label"),
                item.get("class_"),
                item.get("relation"),
                item.get("source_tool"),
                item.get("match_reason"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(_fresh_mapping(item))
            if len(merged) >= limit:
                return merged

    return merged


def build_action_evidence(
    action: str, data: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Derive compact evidence for one canonical action payload."""
    if not isinstance(data, Mapping):
        return []
    if action == "get_element_properties":
        return collect_evidence([data], source_tool=action)

    for field_name in ACTION_EVIDENCE_FIELDS.get(action, ()):
        records = data.get(field_name)
        if isinstance(records, list) and records:
            return collect_evidence(records, source_tool=action)
    return []


def normalize_action_name(action: str) -> str:
    """Return a normalized action name suitable for allowlist checks."""
    return action.strip()


def is_allowed_action(action: str) -> bool:
    """Return True when *action* is in the canonical allowlist."""
    return action in CANONICAL_ACTION_SET


def normalize_relation_name(relation: Any) -> str | None:
    """Normalize a relation value to lower snake-case string when possible."""
    if relation is None:
        return None
    if not isinstance(relation, str):
        relation = str(relation)
    cleaned = relation.strip().lower()
    return cleaned or None


def is_symmetric_relation(relation: Any) -> bool:
    """Return True when a relation should be traversable from either endpoint."""
    normalized = normalize_relation_name(relation)
    return normalized in SYMMETRIC_RELATION_SET


def canonicalize_undirected_edge_endpoints(
    source_id: Any,
    target_id: Any,
) -> tuple[str, str]:
    """Return a deterministic endpoint ordering for one undirected edge record."""
    source_text = str(source_id)
    target_text = str(target_id)
    if source_text <= target_text:
        return source_text, target_text
    return target_text, source_text


def relation_bucket(relation: str | None) -> str | None:
    """Return the taxonomy bucket name for a relation, if known."""
    if relation is None:
        return None
    normalized = normalize_relation_name(relation)
    if normalized is None:
        return None
    for bucket, relations in RELATION_TAXONOMY.items():
        if normalized in relations:
            return bucket
    return None


def normalize_relation_source(source: Any) -> str | None:
    """Normalize relation source string; None remains None."""
    if source is None:
        return None
    if not isinstance(source, str):
        source = str(source)
    cleaned = source.strip().lower()
    return cleaned or None


def is_known_relation_source(source: str | None) -> bool:
    """Return True when *source* belongs to known source semantics."""
    if source is None:
        return False
    return normalize_relation_source(source) in KNOWN_RELATION_SOURCE_SET


def _fresh_default(value: Any) -> Any:
    """Return a fresh copy for mutable default payload values."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def ensure_action_data_fields(
    action: str, data: dict[str, Any] | None
) -> dict[str, Any]:
    """Ensure required fields exist for action payload invariants."""
    stable_data: dict[str, Any] = dict(data) if isinstance(data, dict) else {}
    defaults = ACTION_DATA_DEFAULTS.get(action)
    if not defaults:
        return stable_data
    for field_name, default_value in defaults.items():
        if field_name not in stable_data:
            stable_data[field_name] = _fresh_default(default_value)
    return stable_data


def missing_required_action_fields(
    action: str, data: dict[str, Any] | None
) -> tuple[str, ...]:
    """Return required action data fields missing from *data*."""
    required = ACTION_REQUIRED_DATA_FIELDS.get(action, ())
    if not required:
        return ()
    payload = data if isinstance(data, dict) else {}
    missing = [field_name for field_name in required if field_name not in payload]
    return tuple(missing)


def make_ok_envelope(action: str, data: dict[str, Any] | None) -> dict[str, Any]:
    """Build a successful contract envelope with action data invariants applied."""
    stable_data = ensure_action_data_fields(action, data)
    stable_data["evidence"] = merge_evidence_items(
        stable_data.get("evidence"),
        build_action_evidence(action, stable_data),
    )
    return {
        "status": "ok",
        "data": stable_data,
        "error": None,
    }


def make_error_envelope(
    message: str,
    code: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an error contract envelope preserving {status,data,error}."""
    error_payload: dict[str, Any] = {"message": message, "code": code}
    if details:
        error_payload["details"] = details
    return {
        "status": "error",
        "data": None,
        "error": error_payload,
    }


def has_valid_envelope_shape(payload: dict[str, Any]) -> bool:
    """Return True when payload uses the stable {status,data,error} envelope."""
    return set(payload.keys()) == {"status", "data", "error"}
