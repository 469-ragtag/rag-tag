"""Canonical graph action/relation contract shared across graph backends.

This module is the single source of truth for:
- allowed graph action names
- canonical relation taxonomy buckets
- known relation source semantics
- stable response envelope helpers
- required data fields for key action payloads
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Canonical actions
# ---------------------------------------------------------------------------

CANONICAL_ACTIONS: tuple[str, ...] = (
    "get_elements_in_storey",
    "find_elements_by_class",
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
)

CANONICAL_ACTION_SET = frozenset(CANONICAL_ACTIONS)


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
    "overlaps_xy",
    "intersects_bbox",
    "intersects_3d",
    "touches_surface",
    "space_bounded_by",
    "bounds_space",
    "path_connected_to",
)

EXPLICIT_IFC_RELATIONS: tuple[str, ...] = (
    "hosts",
    "hosted_by",
    "ifc_connected_to",
    "belongs_to_system",
    "in_zone",
    "classified_as",
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

# Defaults also define required field presence for each action's data payload.
ACTION_DATA_DEFAULTS: dict[str, dict[str, Any]] = {
    "get_elements_in_storey": {"storey": None, "elements": []},
    "find_elements_by_class": {"class": None, "elements": []},
    "get_adjacent_elements": {"element_id": None, "adjacent": []},
    "get_topology_neighbors": {
        "element_id": None,
        "relation": None,
        "neighbors": [],
    },
    "get_intersections_3d": {"element_id": None, "intersections_3d": []},
    "find_nodes": {"class": None, "elements": []},
    "traverse": {"start": None, "relation": None, "depth": 1, "results": []},
    "spatial_query": {"near": None, "max_distance": None, "results": []},
    "find_elements_above": {"element_id": None, "max_gap": None, "results": []},
    "find_elements_below": {"element_id": None, "max_gap": None, "results": []},
    "get_element_properties": {
        "id": None,
        "label": None,
        "class_": None,
        "properties": {},
        "payload": None,
    },
    "list_property_keys": {"keys": [], "class_filter": None},
}

ACTION_REQUIRED_DATA_FIELDS: dict[str, tuple[str, ...]] = {
    action: tuple(defaults.keys()) for action, defaults in ACTION_DATA_DEFAULTS.items()
}


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
    return {
        "status": "ok",
        "data": ensure_action_data_fields(action, data),
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
