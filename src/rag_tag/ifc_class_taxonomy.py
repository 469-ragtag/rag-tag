from __future__ import annotations

from collections.abc import Iterable

_CANONICAL_CLASS_BY_LOWER: dict[str, str] = {
    "ifcwall": "IfcWall",
    "ifcwallstandardcase": "IfcWallStandardCase",
    "ifcwallelementedcase": "IfcWallElementedCase",
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


def expand_ifc_class_filter(base_class: str | None) -> tuple[str, ...]:
    if base_class is None:
        return ()
    normalized = normalize_ifc_class(base_class)
    if not normalized:
        return ()
    return CLASS_HIERARCHY.get(normalized, (normalized,))
