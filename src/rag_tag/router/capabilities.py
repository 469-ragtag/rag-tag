from __future__ import annotations

import re
from typing import Literal, TypeAlias

GraphFirstCategory: TypeAlias = Literal[
    "room_space_containment",
    "adjacency_proximity_topology",
    "systems_serving_classification_zone",
    "fuzzy_named_object_lookup",
    "named_element_comparison",
    "materials_or_color",
]

GRAPH_FIRST_CATEGORY_LABELS: dict[GraphFirstCategory, str] = {
    "room_space_containment": "room/space containment membership",
    "adjacency_proximity_topology": "adjacency/proximity/topology",
    "systems_serving_classification_zone": (
        "systems/serving/classification/zone membership"
    ),
    "fuzzy_named_object_lookup": "fuzzy named-object lookup",
    "named_element_comparison": "comparison between specific named elements",
    "materials_or_color": "materials/color (unsupported by current SQLite schema)",
}

SQL_SAFE_CAPABILITY_LABELS: tuple[str, ...] = (
    "deterministic count/list/aggregate/group queries over IFC class, level, "
    "predefined_type, type_name, and element name",
    "structured property/quantity filters and aggregates over persisted "
    "properties/quantities",
)

IFC_SPACE_LEVEL_GRAPH_REASON = (
    "IfcSpace + level/storey questions are graph-first until SQLite space "
    "storey enrichment is available"
)

_SPATIAL_TOPOLOGY_CUES: tuple[str, ...] = (
    "adjacent",
    "near",
    "within distance",
    "distance",
    "connected",
    "connection",
    "route",
    "path",
    "between",
    "neighbor",
    "neighbour",
    "closest",
    "touch",
    "intersect",
    "next to",
    "beside",
    "overlap",
    "above",
    "below",
    "in front of",
    "behind",
)

_SYSTEM_RELATION_CUES: tuple[str, ...] = (
    "classification",
    "classified",
    "system",
    "systems",
    "serves",
    "serving",
    "served by",
    "zone",
    "zones",
)

_MATERIAL_COLOR_CUES: tuple[str, ...] = (
    "material",
    "materials",
    "color",
    "colour",
    "rgb",
)

_FUZZY_LOOKUP_CUES: tuple[str, ...] = (
    "geo-reference",
    "geo reference",
    "georeference",
    "fuzzy",
    "approximate name",
    "similar name",
    "closest match",
)

_COMPARISON_CUES: tuple[str, ...] = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "which is larger",
    "which is smaller",
    "which is higher",
    "which is lower",
    "larger than",
    "smaller than",
)

_ROOM_SPACE_CONTAINMENT_RE = re.compile(
    r"\b(in|inside|within)\s+(?:the\s+)?"
    r"(?:"
    r"(?:room|space)\s+[a-z0-9_.-]+"
    r"|kitchen|bathroom|bedroom|toilet|washroom|lobby|corridor|hallway|office"
    r")\b",
    re.IGNORECASE,
)


def detect_graph_first_categories(question: str) -> tuple[GraphFirstCategory, ...]:
    question_lower = question.lower()
    categories: list[GraphFirstCategory] = []

    if _ROOM_SPACE_CONTAINMENT_RE.search(question_lower):
        categories.append("room_space_containment")

    if any(cue in question_lower for cue in _SPATIAL_TOPOLOGY_CUES):
        categories.append("adjacency_proximity_topology")

    if any(cue in question_lower for cue in _SYSTEM_RELATION_CUES):
        categories.append("systems_serving_classification_zone")

    if any(cue in question_lower for cue in _FUZZY_LOOKUP_CUES):
        categories.append("fuzzy_named_object_lookup")

    if any(cue in question_lower for cue in _COMPARISON_CUES):
        categories.append("named_element_comparison")

    if any(cue in question_lower for cue in _MATERIAL_COLOR_CUES):
        categories.append("materials_or_color")

    deduped: list[GraphFirstCategory] = []
    for category in categories:
        if category not in deduped:
            deduped.append(category)
    return tuple(deduped)


def detect_graph_first_reason(question: str) -> str | None:
    categories = detect_graph_first_categories(question)
    if not categories:
        return None
    labels = [GRAPH_FIRST_CATEGORY_LABELS[category] for category in categories]
    return "Graph-first capability detected: " + ", ".join(labels)


def build_capability_matrix_prompt_block() -> str:
    graph_lines = "\n".join(
        f"- {GRAPH_FIRST_CATEGORY_LABELS[category]}"
        for category in GRAPH_FIRST_CATEGORY_LABELS
    )
    sql_lines = "\n".join(f"- {line}" for line in SQL_SAFE_CAPABILITY_LABELS)
    return (
        "Shared capability matrix:\n"
        "Graph-first categories (route='graph'):\n"
        f"{graph_lines}\n"
        "SQL-safe categories (route='sql' when deterministic and schema-supported):\n"
        f"{sql_lines}"
    )


def is_ifc_space_level_graph_first(
    ifc_class: str | None,
    level_like: str | None,
) -> bool:
    return ifc_class == "IfcSpace" and level_like is not None
