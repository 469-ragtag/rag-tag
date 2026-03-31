from __future__ import annotations

import re
from collections.abc import Iterable

_CANONICAL_CLASS_BY_LOWER: dict[str, str] = {
    "ifcwall": "IfcWall",
    "ifcwallstandardcase": "IfcWallStandardCase",
    "ifcwallelementedcase": "IfcWallElementedCase",
    "ifccurtainwall": "IfcCurtainWall",
    "ifcslab": "IfcSlab",
    "ifcslabelementedcase": "IfcSlabElementedCase",
    "ifcdoor": "IfcDoor",
    "ifcdoorstandardcase": "IfcDoorStandardCase",
    "ifcwindow": "IfcWindow",
    "ifcwindowstandardcase": "IfcWindowStandardCase",
    "ifcbeam": "IfcBeam",
    "ifcbeamstandardcase": "IfcBeamStandardCase",
    "ifccolumn": "IfcColumn",
    "ifccolumnstandardcase": "IfcColumnStandardCase",
    "ifcstair": "IfcStair",
    "ifcstairflight": "IfcStairFlight",
    "ifcroof": "IfcRoof",
    "ifcspace": "IfcSpace",
    "ifcbuildingstorey": "IfcBuildingStorey",
    "ifcpipesegment": "IfcPipeSegment",
    "ifcductsegment": "IfcDuctSegment",
    "ifcflowterminal": "IfcFlowTerminal",
    "ifcrailing": "IfcRailing",
    "ifcramp": "IfcRamp",
    "ifcchimney": "IfcChimney",
    "ifcsite": "IfcSite",
    "ifcproject": "IfcProject",
    "ifcbuilding": "IfcBuilding",
    "ifccovering": "IfcCovering",
    "ifcmember": "IfcMember",
    "ifcplate": "IfcPlate",
    "ifcfooting": "IfcFooting",
    "ifcfurniture": "IfcFurniture",
}


def normalize_ifc_class(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    canonical = _CANONICAL_CLASS_BY_LOWER.get(cleaned.lower())
    if canonical:
        return canonical
    if not cleaned.lower().startswith("ifc"):
        cleaned = f"Ifc{cleaned}"
    canonical = _CANONICAL_CLASS_BY_LOWER.get(cleaned.lower())
    if canonical:
        return canonical
    core = cleaned[3:]
    if not core:
        return "Ifc"
    return "Ifc" + core[0].upper() + core[1:]


_RAW_CLASS_ALIASES: dict[str, str] = {
    "wall": "IfcWall",
    "walls": "IfcWall",
    "curtain wall": "IfcCurtainWall",
    "curtain walls": "IfcCurtainWall",
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
    "furniture": "IfcFurniture",
    "furnishing": "IfcFurniture",
    "sink": "IfcFlowTerminal",
    "toilet": "IfcFlowTerminal",
    "lamp": "IfcFlowTerminal",
    "railing": "IfcRailing",
    "railings": "IfcRailing",
    "ramp": "IfcRamp",
    "ramps": "IfcRamp",
    "chimney": "IfcChimney",
    "chimneys": "IfcChimney",
    "site": "IfcSite",
    "project": "IfcProject",
    "building": "IfcBuilding",
    "covering": "IfcCovering",
    "coverings": "IfcCovering",
    "floor": "IfcSlab",
    "floors": "IfcSlab",
    "member": "IfcMember",
    "members": "IfcMember",
    "plate": "IfcPlate",
    "plates": "IfcPlate",
    "footing": "IfcFooting",
    "footings": "IfcFooting",
    "foundation": "IfcFooting",
}

CLASS_ALIASES: dict[str, str] = {
    term: normalize_ifc_class(ifc_class)
    for term, ifc_class in _RAW_CLASS_ALIASES.items()
}

_RAW_CLASS_HIERARCHY: dict[str, tuple[str, ...]] = {
    "IfcWall": ("IfcWall", "IfcWallStandardCase", "IfcWallElementedCase"),
    "IfcCurtainWall": ("IfcCurtainWall",),
    "IfcSlab": ("IfcSlab", "IfcSlabElementedCase"),
    "IfcDoor": ("IfcDoor", "IfcDoorStandardCase"),
    "IfcWindow": ("IfcWindow", "IfcWindowStandardCase"),
    "IfcBeam": ("IfcBeam", "IfcBeamStandardCase"),
    "IfcColumn": ("IfcColumn", "IfcColumnStandardCase"),
    "IfcStair": ("IfcStair", "IfcStairFlight"),
    "IfcRoof": ("IfcRoof",),
    "IfcSpace": ("IfcSpace",),
    "IfcBuildingStorey": ("IfcBuildingStorey",),
    "IfcPipeSegment": ("IfcPipeSegment",),
    "IfcDuctSegment": ("IfcDuctSegment",),
    "IfcFlowTerminal": ("IfcFlowTerminal",),
    "IfcRailing": ("IfcRailing",),
    "IfcRamp": ("IfcRamp",),
    "IfcChimney": ("IfcChimney",),
    "IfcSite": ("IfcSite",),
    "IfcProject": ("IfcProject",),
    "IfcBuilding": ("IfcBuilding",),
    "IfcCovering": ("IfcCovering",),
    "IfcMember": ("IfcMember",),
    "IfcPlate": ("IfcPlate",),
    "IfcFooting": ("IfcFooting",),
    "IfcFurniture": ("IfcFurniture",),
}


def _normalized_unique_classes(
    class_names: Iterable[str],
    *,
    fallback: str,
) -> list[str]:
    ordered: list[str] = []
    for raw in class_names:
        normalized = normalize_ifc_class(raw)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    fallback_normalized = normalize_ifc_class(fallback)
    if fallback_normalized and fallback_normalized not in ordered:
        ordered.insert(0, fallback_normalized)
    return ordered


CLASS_HIERARCHY: dict[str, tuple[str, ...]] = {
    normalize_ifc_class(base): tuple(
        _normalized_unique_classes(class_names, fallback=base)
    )
    for base, class_names in _RAW_CLASS_HIERARCHY.items()
}

_CLASS_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        CLASS_ALIASES.items(),
        key=lambda item: (-len(item[0]), item[0]),
    )
)


def expand_ifc_class_filter(base_class: str | None) -> tuple[str, ...]:
    if base_class is None:
        return ()
    normalized = normalize_ifc_class(base_class)
    if not normalized:
        return ()
    return CLASS_HIERARCHY.get(normalized, (normalized,))


def find_class_alias_matches(
    question_lower: str,
    *,
    ignored_spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    for term, ifc_class in _CLASS_ALIAS_PATTERNS:
        for match in re.finditer(rf"\b{re.escape(term)}\b", question_lower):
            span = match.span()
            if _span_is_contained(span, ignored_spans):
                continue
            candidates.append((span[0], span[1], ifc_class))

    selected: list[tuple[int, int, str]] = []
    for start, end, ifc_class in candidates:
        if any(
            _spans_overlap((start, end), (sel_start, sel_end))
            for sel_start, sel_end, _ in selected
        ):
            continue
        selected.append((start, end, ifc_class))

    selected.sort(key=lambda item: item[0])
    return selected


def _span_is_contained(
    span: tuple[int, int],
    ignored_spans: list[tuple[int, int]] | None,
) -> bool:
    if not ignored_spans:
        return False
    start, end = span
    return any(
        ignore_start <= start and end <= ignore_end
        for ignore_start, ignore_end in ignored_spans
    )


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
