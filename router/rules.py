from __future__ import annotations

import re

from .models import RouteDecision, SqlIntent, SqlRequest

_SPATIAL_CUES = (
    "adjacent",
    "near",
    "within",
    "distance",
    "connected",
    "connection",
    "route",
    "path",
    "between",
    "neighbor",
    "neighbour",
    "closest",
    "far",
    "touch",
    "intersect",
)

_COUNT_CUES = (
    "how many",
    "count",
    "number of",
    "total",
)

_LIST_CUES = (
    "list",
    "find",
    "show",
    "which",
    "what are",
    "display",
)

_IFC_CLASS_RE = re.compile(r"\bifc[a-z0-9_]+\b", re.IGNORECASE)

_CLASS_ALIASES: dict[str, str] = {
    "wall": "IfcWall",
    "walls": "IfcWall",
    "door": "IfcDoor",
    "doors": "IfcDoor",
    "window": "IfcWindow",
    "windows": "IfcWindow",
    "slab": "IfcSlab",
    "slabs": "IfcSlab",
    "column": "IfcColumn",
    "columns": "IfcColumn",
    "beam": "IfcBeam",
    "beams": "IfcBeam",
    "stair": "IfcStair",
    "stairs": "IfcStair",
    "stairwell": "IfcStair",
    "stairwells": "IfcStair",
    "space": "IfcSpace",
    "spaces": "IfcSpace",
    "room": "IfcSpace",
    "rooms": "IfcSpace",
    "roof": "IfcRoof",
    "roofs": "IfcRoof",
    "pipe": "IfcPipeSegment",
    "pipes": "IfcPipeSegment",
    "duct": "IfcDuctSegment",
    "ducts": "IfcDuctSegment",
}

_LEVEL_RE = re.compile(r"\b(level|storey|story|floor)\s+([a-z0-9 _.-]+)")
_LEVEL_STOP_WORDS = re.compile(
    r"\b(with|that|which|near|adjacent|connected|having|where|and|or)\b"
)


def route_question_rule(question: str) -> RouteDecision:
    q = question.strip()
    q_lower = q.lower()

    if _has_spatial_cues(q_lower):
        return RouteDecision("graph", "Spatial cue detected", None)

    sql_intent = _detect_sql_intent(q_lower)
    if sql_intent is None:
        return RouteDecision("graph", "No SQL intent detected", None)

    ifc_class = _detect_ifc_class(q)
    level_like = _detect_level_like(q_lower)

    if ifc_class is None and not _mentions_generic_elements(q_lower):
        return RouteDecision("graph", "SQL intent without class", None)

    limit = 50 if sql_intent == "list" else 0
    request = SqlRequest(
        intent=sql_intent,
        ifc_class=ifc_class,
        level_like=level_like,
        limit=limit,
    )
    return RouteDecision("sql", "Aggregation/list intent detected", request)


def _has_spatial_cues(question_lower: str) -> bool:
    return any(cue in question_lower for cue in _SPATIAL_CUES)


def _detect_sql_intent(question_lower: str) -> SqlIntent | None:
    if any(cue in question_lower for cue in _COUNT_CUES):
        return "count"
    if any(cue in question_lower for cue in _LIST_CUES):
        return "list"
    return None


def _detect_ifc_class(question: str) -> str | None:
    match = _IFC_CLASS_RE.search(question)
    if match:
        return _normalize_ifc_class(match.group(0))

    question_lower = question.lower()
    for term, ifc_class in _CLASS_ALIASES.items():
        if re.search(rf"\b{re.escape(term)}\b", question_lower):
            return ifc_class
    return None


def _normalize_ifc_class(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    if not cleaned.lower().startswith("ifc"):
        cleaned = f"Ifc{cleaned}"
    core = cleaned[3:]
    if not core:
        return "Ifc"
    return "Ifc" + core[0].upper() + core[1:]


def _detect_level_like(question_lower: str) -> str | None:
    if "ground floor" in question_lower:
        return "ground floor"
    if "ground level" in question_lower:
        return "ground level"
    if "basement" in question_lower:
        return "basement"

    match = _LEVEL_RE.search(question_lower)
    if not match:
        return None
    raw = match.group(2).strip()
    raw = _LEVEL_STOP_WORDS.split(raw)[0].strip()
    return raw or None


def _mentions_generic_elements(question_lower: str) -> bool:
    terms = ("element", "elements", "components")
    return any(term in question_lower for term in terms)
