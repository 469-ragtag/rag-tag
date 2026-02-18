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
    "next to",
    "beside",
    "inside",
    "contains",
    "contain",
    "contained",
    "overlap",
    "overlapping",
    "touching",
    "touches",
    "above",
    "below",
    "in front of",
    "behind",
)

_COUNT_CUES = (
    "how many",
    "count",
    "number of",
    "total",
)

_EXISTENCE_CUES = (
    "are there",
    "are there any",
    "is there",
    "is there any",
    "do we have",
    "does the building have",
    "does the house have",
    "exist",
    "exists",
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
_PROPERTY_CUE_RE = re.compile(
    r"\b(with|having|whose|where|without)\b|\bthat\s+(has|have)\b"
)

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
    "storey": "IfcBuildingStorey",
    "storeys": "IfcBuildingStorey",
    "story": "IfcBuildingStorey",
    "stories": "IfcBuildingStorey",
    "pipe": "IfcPipeSegment",
    "pipes": "IfcPipeSegment",
    "duct": "IfcDuctSegment",
    "ducts": "IfcDuctSegment",
}

_LEVEL_RE = re.compile(r"\b(level|storey|story|floor)\s+([a-z0-9 _.-]+)")
_LEVEL_STOP_WORDS = re.compile(
    r"\b(with|that|which|near|adjacent|connected|having|where|and|or"
    r"|on|in|of|the|a|an"
    r"|structure|building|house|model|project)\b"
)


def route_question_rule(question: str) -> RouteDecision:
    q = question.strip()
    q_lower = q.lower()

    if _has_spatial_cues(q_lower) or _has_relation_cues(q_lower):
        return RouteDecision("graph", "Spatial/relationship cue detected", None)

    sql_intent = _detect_sql_intent(q_lower)
    if sql_intent is None:
        return RouteDecision("graph", "No SQL intent detected", None)

    if _has_property_cues(q_lower):
        return RouteDecision("graph", "Property/constraint cue detected", None)

    ifc_classes = _detect_ifc_classes(q)
    if len(ifc_classes) > 1:
        return RouteDecision("graph", "Multiple IFC classes mentioned", None)

    ifc_class = ifc_classes[0] if ifc_classes else None
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


def _has_relation_cues(question_lower: str) -> bool:
    if "contains" in question_lower or "contained" in question_lower:
        return True
    return False


def _has_property_cues(question_lower: str) -> bool:
    return _PROPERTY_CUE_RE.search(question_lower) is not None


def _detect_sql_intent(question_lower: str) -> SqlIntent | None:
    if any(cue in question_lower for cue in _COUNT_CUES):
        return "count"
    if any(cue in question_lower for cue in _EXISTENCE_CUES):
        return "count"
    if any(cue in question_lower for cue in _LIST_CUES):
        return "list"
    return None


def _detect_ifc_class(question: str) -> str | None:
    classes = _detect_ifc_classes(question)
    return classes[0] if classes else None


def _detect_ifc_classes(question: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for match in _IFC_CLASS_RE.finditer(question):
        matches.append((match.start(), _normalize_ifc_class(match.group(0))))

    question_lower = question.lower()
    for term, ifc_class in _CLASS_ALIASES.items():
        for match in re.finditer(rf"\b{re.escape(term)}\b", question_lower):
            matches.append((match.start(), ifc_class))

    matches.sort(key=lambda item: item[0])
    ordered: list[str] = []
    for _, ifc_class in matches:
        if ifc_class not in ordered:
            ordered.append(ifc_class)
    return ordered


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
    if any(
        p in question_lower
        for p in (
            "on the structure",
            "in the structure",
            "in the building",
            "in the house",
            "of the building",
            "of the house",
        )
    ):
        return None
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
