# reads an IFC file and writes one JSON record per element to a .jsonl file
# each record has the element's identity, where it sits in the building hierarchy,
# its geometry (centroid + bbox), and its property sets split into official vs custom
#
# run with: uv run rag-tag-ifc-to-jsonl

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import ifcopenshell
import ifcopenshell.util.element as ifc_element

from rag_tag.parser.ifc43_schema_registry import STANDARD_PSETS, get_registry
from rag_tag.parser.ifc_geometry_parse import extract_geometry_data
from rag_tag.paths import find_ifc_dir, find_project_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _official_pset_names() -> set[str]:
    # build the set of pset names that are in the IFC standard
    # anything not in here goes into "Custom" instead of "Official"
    names: set[str] = set()
    for psets in STANDARD_PSETS.values():
        names.update(psets.keys())
    return names


def _to_python(value: Any) -> Any:
    # ifcopenshell doesn't always give back plain Python types —
    # sometimes you get enum objects or entity references, so we force cast
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return str(value)
    except Exception:
        return None


def _cast_pset(pset_dict: dict) -> dict:
    # ifcopenshell adds an "id" key to every pset dict that we don't want
    return {k: _to_python(v) for k, v in pset_dict.items() if k != "id"}


def _extract_hierarchy(element) -> dict:
    try:
        container = ifc_element.get_container(element)
    except Exception:
        container = None

    parent_id: str | None = None
    parent_type: str | None = None
    level: str | None = None

    if container is not None:
        try:
            gid = getattr(container, "GlobalId", None)
            parent_id = str(gid) if gid else None
        except Exception:
            pass
        try:
            parent_type = container.is_a()
        except Exception:
            pass
        try:
            name = getattr(container, "Name", None)
            level = str(name) if name else None
        except Exception:
            pass

    # walk up the tree to get the full spatial path (e.g. ["building", "floor 1"])
    path: list[str] = []
    try:
        node = container
        while node is not None:
            name = getattr(node, "Name", None)
            if name:
                path.append(str(name))
            node = ifc_element.get_container(node)
        path.reverse()  # we walked bottom-up so flip it
    except Exception:
        path = []

    return {
        "ParentId": parent_id,
        "ParentType": parent_type,
        "Level": level,
        "Path": path,
    }


def _extract_geometry(global_id: str | None, geom_by_id: dict) -> dict:
    empty = {"Centroid": None, "BoundingBox": None}
    if not global_id:
        return empty
    geom = geom_by_id.get(global_id)
    if geom is None:
        return empty

    centroid = geom.get("centroid")
    bbox = geom.get("bbox")

    return {
        "Centroid": list(centroid) if centroid is not None else None,
        "BoundingBox": (
            {"min": list(bbox[0]), "max": list(bbox[1])} if bbox is not None else None
        ),
    }


def extract_element(
    element,
    official_names: set[str],
    geom_by_id: dict,
    registry,
) -> dict:
    try:
        express_id: int | None = element.id()
    except Exception:
        express_id = None

    raw_class = element.is_a()
    norm = registry.normalize_class(raw_class)
    canonical_class = norm.get("canonical", raw_class)
    ancestors: list[str] = norm.get("ancestors", [])

    global_id = getattr(element, "GlobalId", None)
    name = getattr(element, "Name", None)
    description = getattr(element, "Description", None)
    object_type = getattr(element, "ObjectType", None)
    tag = getattr(element, "Tag", None)

    predefined_type: str | None = None
    try:
        if hasattr(element, "PredefinedType"):
            pt = element.PredefinedType
            predefined_type = str(pt) if pt else None
    except Exception:
        pass

    type_name: str | None = None
    try:
        obj_type = ifc_element.get_type(element)
        if obj_type and hasattr(obj_type, "Name"):
            n = obj_type.Name
            type_name = str(n) if n else None
    except Exception:
        pass

    hierarchy = _extract_hierarchy(element)
    geometry = _extract_geometry(str(global_id) if global_id else None, geom_by_id)

    official_psets: dict[str, dict] = {}
    custom_psets: dict[str, dict] = {}

    try:
        psets = ifc_element.get_psets(element, psets_only=True)
        for pset_name, pset_data in psets.items():
            if not isinstance(pset_data, dict):
                continue
            cast = _cast_pset(pset_data)
            if pset_name in official_names:
                official_psets[pset_name] = cast
            else:
                custom_psets[pset_name] = cast
    except Exception:
        pass

    quantities: dict[str, dict] = {}
    try:
        qtos = ifc_element.get_psets(element, qtos_only=True)
        for qto_name, qto_data in qtos.items():
            if not isinstance(qto_data, dict):
                continue
            quantities[qto_name] = _cast_pset(qto_data)
    except Exception:
        pass

    return {
        "GlobalId": str(global_id) if global_id else None,
        "ExpressId": express_id,
        "Name": str(name) if name else None,
        "IfcType": canonical_class,
        "ClassRaw": raw_class,
        "BaseClasses": ancestors,
        "PredefinedType": predefined_type,
        "Description": str(description) if description else None,
        "ObjectType": str(object_type) if object_type else None,
        "Tag": str(tag) if tag else None,
        "TypeName": type_name,
        "Hierarchy": hierarchy,
        "Geometry": geometry,
        "PropertySets": {
            "Official": official_psets,
            "Custom": custom_psets,
        },
        "Quantities": quantities,
    }


def convert_ifc_to_jsonl(ifc_path: Path, out_path: Path) -> int:
    logger.info("Opening %s", ifc_path)
    model = ifcopenshell.open(str(ifc_path))
    schema_version = getattr(model, "schema", "IFC4X3_ADD2")
    logger.info("Schema: %s", schema_version)

    registry = get_registry()
    official_names = _official_pset_names()

    logger.info("Extracting geometry (may take a moment)...")
    geom_list = extract_geometry_data(model)
    geom_by_id: dict = {g["GlobalId"]: g for g in geom_list}
    logger.info("Geometry extracted for %d elements", len(geom_by_id))

    # IfcProduct is the base class for physical building elements (walls, doors, etc.)
    # but IfcProject/IfcSite/IfcBuilding don't inherit from it so we add them manually
    elements: list = list(model.by_type("IfcProduct"))
    for extra in ("IfcProject", "IfcSite", "IfcBuilding"):
        try:
            elements.extend(model.by_type(extra))
        except Exception:
            pass

    # in IFC4+ some classes show up under both IfcProduct and the spatial types,
    # so deduplicate by express ID before processing
    seen: set[int] = set()
    unique: list = []
    for elem in elements:
        eid = elem.id()
        if eid not in seen:
            seen.add(eid)
            unique.append(elem)

    logger.info("Processing %d unique elements...", len(unique))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for elem in unique:
            try:
                record = extract_element(elem, official_names, geom_by_id, registry)
            except Exception as exc:
                logger.warning("Skipping #%s: %s", elem.id(), exc)
                continue
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
            count += 1

    logger.info("Wrote %d records to %s", count, out_path)
    return count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert IFC file(s) to JSONL (one JSON record per element)."
    )
    ap.add_argument(
        "--ifc-dir",
        type=Path,
        default=None,
        help="Directory containing .ifc files (default: auto-detected IFC-Files/).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for .jsonl files (default: <project_root>/output/).",
    )
    ap.add_argument(
        "--ifc-file",
        type=Path,
        default=None,
        help="Single IFC file to convert (overrides --ifc-dir).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    else:
        project_root = find_project_root(script_dir) or script_dir.parent.parent.parent
        out_dir = (project_root / "output").resolve()

    if args.ifc_file is not None:
        ifc_file = args.ifc_file.expanduser().resolve()
        if not ifc_file.is_file():
            print(f"IFC file not found: {ifc_file}")
            return
        out_path = out_dir / (ifc_file.stem + ".jsonl")
        n = convert_ifc_to_jsonl(ifc_file, out_path)
        print(f"\n{ifc_file.name} -> {out_path.name} ({n} elements)")
        return

    if args.ifc_dir is not None:
        ifc_dir = args.ifc_dir.expanduser().resolve()
    else:
        ifc_dir = find_ifc_dir(script_dir)
        if ifc_dir is None:
            print(
                "IFC directory not found. "
                "Use --ifc-dir or create an IFC-Files/ folder."
            )
            return

    if not ifc_dir.is_dir():
        print(f"IFC directory not found: {ifc_dir}")
        return

    ifc_files = sorted(f for f in ifc_dir.iterdir() if f.suffix.lower() == ".ifc")
    if not ifc_files:
        print(f"No .ifc files found in {ifc_dir}")
        return

    print(f"Converting {len(ifc_files)} IFC file(s) from {ifc_dir}")
    print(f"Writing JSONL to {out_dir}\n")

    for ifc_path in ifc_files:
        out_path = out_dir / (ifc_path.stem + ".jsonl")
        try:
            n = convert_ifc_to_jsonl(ifc_path, out_path)
            print(f"  {ifc_path.name} -> {out_path.name} ({n} elements)")
        except Exception as exc:
            print(f"  {ifc_path.name} -> ERROR: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
