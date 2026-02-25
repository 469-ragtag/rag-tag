from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.element as element
import pandas as pd

from rag_tag.parser.ifc43_schema_registry import IFC43SchemaRegistry, get_registry
from rag_tag.paths import find_bsdd_rdf_path, find_ifc_dir, find_project_root

logger = logging.getLogger(__name__)

# shorthand so function signatures stay under the 88-char line limit
_Registry = IFC43SchemaRegistry | None


# ---------------------------------------------------------------------------
# Core column order — kept consistent across every CSV we write.
# We added ClassRaw, Class, and ClassBase on top of the original 10 columns.
#
#   ClassRaw  = exactly what obj.is_a() returns (e.g. IfcWallStandardCase)
#   Class     = canonical form from the schema hierarchy (e.g. IfcWall)
#   ClassBase = nearest ancestor that has standard Psets — same as Class for now,
#               kept separate in case we want finer control later
#
# Downstream consumers (csv_to_sql, csv_to_graph) only look at "Class", so
# backward compatibility is preserved.  The two new columns are purely additive.
# ---------------------------------------------------------------------------

CORE_COLUMNS = [
    "ExpressId",
    "GlobalId",
    "ClassRaw",
    "Class",
    "ClassBase",
    "PredefinedType",
    "Name",
    "Description",
    "ObjectType",
    "Tag",
    "Level",
    "TypeName",
]


def _resolve_bsdd_path(cli_path: Path | None = None) -> Path:
    """
    Work out which bSDD RDF snapshot to use.

    We check three places in order:
      1. --bsdd-rdf-path flag passed on the command line
      2. BSDD_IFC43_RDF_PATH environment variable
      3. The default project location (output/metadata/bsdd/ifc43.ttl)

    If none of them point to a file that actually exists, we still return a
    path — the registry will handle the missing file gracefully.
    """
    # 1. explicit CLI argument wins
    if cli_path is not None:
        return cli_path.expanduser().resolve()

    # 2. environment variable
    env_val = os.environ.get("BSDD_IFC43_RDF_PATH")
    if env_val:
        return Path(env_val).expanduser().resolve()

    # 3. fall back to the project default
    return find_bsdd_rdf_path()


def get_objects_data_by_class(model, class_type, registry: _Registry = None):
    """
    Extract every object of class_type from the IFC model and return:
      - a list of dicts (one per object) with all the data we care about
      - a set of all "Pset.Property" column names seen across those objects

    The registry parameter is optional but important:
      - when provided, we pre-seed the column set with properties the spec
        *expects* for this class, even if the model doesn't have them
      - we also normalise ClassRaw into canonical Class / ClassBase values
    """
    objects = model.by_type(class_type)
    data_list = []
    pset_attributes: set[str] = set()

    # Pre-seed pset columns from the schema registry.
    # This is the core of the schema-aware improvement — we add columns for
    # properties that *should* be there according to IFC 4.3, so the CSV always
    # has consistent columns even when the model skips those Psets.
    if registry is not None:
        pset_attributes.update(registry.expected_properties_for_class(class_type))

    for obj in objects:
        # --- IDs ---
        try:
            express_id = obj.id()  # matches the #123 number in the IFC file
        except Exception:
            express_id = id(obj)

        # ClassRaw is always what the IFC file actually says
        raw_class = obj.is_a()

        # Class and ClassBase default to the raw value; we'll update them
        # below if we have a registry to normalise against
        canonical_class = raw_class
        base_class = raw_class
        if registry is not None:
            norm = registry.normalize_class(raw_class)
            canonical_class = norm["canonical"]
            base_class = norm["base"]

        obj_data: dict = {
            "ExpressId": express_id,
            "GlobalId": obj.GlobalId if hasattr(obj, "GlobalId") else "",
            "ClassRaw": raw_class,
            "Class": canonical_class,
            "ClassBase": base_class,
            "PredefinedType": "",
            "Name": obj.Name if hasattr(obj, "Name") else "",
            "Description": obj.Description if hasattr(obj, "Description") else "",
            "ObjectType": obj.ObjectType if hasattr(obj, "ObjectType") else "",
            "Tag": obj.Tag if hasattr(obj, "Tag") else "",
        }

        try:
            if hasattr(obj, "PredefinedType"):
                predefined = obj.PredefinedType
                obj_data["PredefinedType"] = str(predefined) if predefined else ""
        except Exception:
            pass

        # which floor/space is this element contained in?
        try:
            container = element.get_container(obj)
            obj_data["Level"] = (
                container.Name if container and hasattr(container, "Name") else ""
            )
        except Exception:
            obj_data["Level"] = ""

        # the type object (e.g. "Basic Wall: 200mm Concrete")
        try:
            obj_type = element.get_type(obj)
            obj_data["TypeName"] = (
                obj_type.Name if obj_type and hasattr(obj_type, "Name") else ""
            )
        except Exception:
            obj_data["TypeName"] = ""

        # --- Property sets (text / enum values) ---
        try:
            psets = element.get_psets(obj, psets_only=True)
            obj_data["PropertySets"] = psets
            add_pset_attributes(psets, pset_attributes)
        except Exception:
            obj_data["PropertySets"] = {}

        # --- Quantity sets (numeric measurements) ---
        try:
            qtos = element.get_psets(obj, qtos_only=True)
            obj_data["QuantitySets"] = qtos
            add_pset_attributes(qtos, pset_attributes)
        except Exception:
            obj_data["QuantitySets"] = {}

        data_list.append(obj_data)

    return data_list, pset_attributes


def add_pset_attributes(psets, attributes_set):
    """
    Add pset/qto attributes to the running set in "PsetName.PropertyName" format.
    We skip the internal "id" key that ifcopenshell adds to every pset dict.
    """
    if not psets:
        return

    for pset_name, pset_data in psets.items():
        if isinstance(pset_data, dict):
            for prop_name in pset_data.keys():
                if prop_name != "id":
                    attributes_set.add(f"{pset_name}.{prop_name}")


def get_attribute_value(obj_data, attribute):
    """
    Pull a single value out of obj_data by attribute name.

    Direct attributes (ExpressId, Name, etc.) are looked up straight away.
    Pset/Qto attributes use dot-notation like "Pset_WallCommon.FireRating"
    and we look in PropertySets first, then QuantitySets.
    """
    if "." not in attribute:
        return obj_data.get(attribute, None)

    pset_name, prop_name = attribute.split(".", 1)

    if "PropertySets" in obj_data and pset_name in obj_data["PropertySets"]:
        pset_data = obj_data["PropertySets"][pset_name]
        if isinstance(pset_data, dict) and prop_name in pset_data:
            return pset_data[prop_name]

    if "QuantitySets" in obj_data and pset_name in obj_data["QuantitySets"]:
        qto_data = obj_data["QuantitySets"][pset_name]
        if isinstance(qto_data, dict) and prop_name in qto_data:
            return qto_data[prop_name]

    return None


def export_to_csv(model, class_type, output_path, registry: _Registry = None):
    """
    Export all objects of class_type from the model to a single CSV file.

    Columns come in two groups:
      1. Core columns (always present, fixed order) — identity + class + names
      2. Pset/Qto columns (alphabetically sorted) — observed values from model
         PLUS schema-expected columns pre-seeded as empty strings when missing

    Returns the number of rows written, or 0 if there was nothing to export.
    """
    data, attributes = get_objects_data_by_class(model, class_type, registry=registry)

    if not data:
        return 0

    # sort pset/qto column names so the CSV is deterministic across runs
    # (important when comparing two models side-by-side)
    attributes_list = sorted(attributes)

    rows = []
    for obj_data in data:
        row = {}
        # fill in the pset/qto columns (empty string if not set on this object)
        for attr in attributes_list:
            value = get_attribute_value(obj_data, attr)
            row[attr] = value if value is not None else ""
        # fill in the core columns
        for key in CORE_COLUMNS:
            row[key] = obj_data.get(key, "")
        rows.append(row)

    all_columns = CORE_COLUMNS + attributes_list
    ordered_rows = [{col: row.get(col, "") for col in all_columns} for row in rows]

    df = pd.DataFrame(ordered_rows, columns=all_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return len(df)


def export_to_excel(model, class_types, output_path, registry: _Registry = None):
    """
    Export multiple IFC classes to an Excel file, one sheet per class.
    Useful for a quick overview — each sheet only shows non-empty columns.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for class_type in class_types:
            try:
                data, attributes = get_objects_data_by_class(
                    model, class_type, registry=registry
                )

                if not data:
                    continue

                attributes_list = sorted(attributes)
                rows = []

                for obj_data in data:
                    row = {}
                    for attr in attributes_list:
                        value = get_attribute_value(obj_data, attr)
                        row[attr] = value if value is not None else ""
                    for key in CORE_COLUMNS:
                        row[key] = obj_data.get(key, "")
                    rows.append(row)

                all_columns = CORE_COLUMNS + attributes_list
                ordered_rows = [
                    {col: row.get(col, "") for col in all_columns} for row in rows
                ]
                df = pd.DataFrame(ordered_rows, columns=all_columns)

                # filter to just this class and drop completely empty columns
                df_filtered = df[df["ClassRaw"] == class_type].copy()
                df_filtered = df_filtered.dropna(axis=1, how="all")

                if not df_filtered.empty:
                    sheet_name = class_type.replace("Ifc", "")[:31]
                    df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

            except Exception as e:
                print(f"  Warning: Could not export {class_type}: {e}")
                continue


def _log_skipped_summary(skipped: dict[str, str]) -> None:
    """Print a summary of any IFC classes that failed during processing."""
    if not skipped:
        return
    logger.warning("--- Skipped class summary ---")
    for cls, reason in skipped.items():
        logger.warning("  %s: %s", cls, reason)
    logger.warning("Total classes skipped: %d", len(skipped))


def parse_ifc_to_csv(
    ifc_path: Path,
    csv_path: Path,
    class_type: str = None,  # type: ignore
    registry: IFC43SchemaRegistry | None = None,
) -> int:
    """
    Parse one IFC file and write CSV(s).

    If class_type is given, export just that one class.
    Otherwise export all standard building element classes combined into one CSV.
    The registry is passed through to expand property coverage.
    """
    model = ifcopenshell.open(str(ifc_path))

    if class_type:
        return export_to_csv(model, class_type, csv_path, registry=registry)

    class_types = [
        "IfcWall", "IfcSlab", "IfcDoor", "IfcWindow", "IfcColumn", "IfcBeam",
        "IfcRoof", "IfcStair", "IfcRamp", "IfcFurniture", "IfcBuildingElementProxy",
        "IfcCovering", "IfcRailing", "IfcFlowSegment", "IfcFlowTerminal",
        "IfcFlowFitting", "IfcDuctSegment", "IfcPipeSegment", "IfcPlate",
        "IfcMember", "IfcFooting", "IfcPile", "IfcBuildingStorey", "IfcSpace",
        "IfcZone",
    ]

    total_rows = 0
    skipped: dict[str, str] = {}
    for ct in class_types:
        try:
            csv_name = f"{ifc_path.stem}_{ct.replace('Ifc', '')}.csv"
            n = export_to_csv(model, ct, csv_path.parent / csv_name, registry=registry)
            if n > 0:
                total_rows += n
        except Exception as exc:
            logger.warning("Class %s failed: %s", ct, exc)
            skipped[ct] = str(exc)

    _log_skipped_summary(skipped)
    return total_rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Parse IFC file(s) from IFC-Files/ and export to CSV."
    )
    parser.add_argument(
        "--ifc-dir",
        type=Path,
        default=None,
        help="Directory containing .ifc files (default: auto-detect IFC-Files/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write CSV outputs (default: <project-root>/output).",
    )
    parser.add_argument(
        "--bsdd-rdf-path",
        type=Path,
        default=None,
        help=(
            "Path to a local bSDD IFC 4.3 RDF/Turtle snapshot. "
            "Overrides BSDD_IFC43_RDF_PATH env var and the default project path. "
            "Run `uv run rag-tag-refresh-ifc43-rdf` to download the snapshot."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # --- resolve directories ---
    if args.ifc_dir is not None:
        ifc_dir = args.ifc_dir.expanduser().resolve()
    else:
        ifc_dir = find_ifc_dir(script_dir)
        if ifc_dir is None:
            print(
                "IFC directory not found. Use --ifc-dir or create an IFC-Files/ folder."
            )
            return

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    else:
        project_root = find_project_root(script_dir) or script_dir.parent
        out_dir = (project_root / "output").resolve()

    if not ifc_dir.is_dir():
        print(f"IFC directory not found: {ifc_dir}")
        return

    ifc_files = sorted(f for f in ifc_dir.iterdir() if f.suffix.lower() == ".ifc")
    if not ifc_files:
        print(f"No .ifc files found in {ifc_dir}")
        return

    # --- resolve bSDD snapshot path and create registry ---
    # Resolution order: --bsdd-rdf-path flag > BSDD_IFC43_RDF_PATH env var > default
    bsdd_path = _resolve_bsdd_path(args.bsdd_rdf_path)
    if bsdd_path.exists():
        logger.info("Using bSDD snapshot: %s", bsdd_path)
    else:
        logger.info(
            "No bSDD snapshot at %s — using embedded schema only "
            "(run `uv run rag-tag-refresh-ifc43-rdf` to download it)",
            bsdd_path,
        )

    print(f"Parsing {len(ifc_files)} IFC file(s) from {ifc_dir}")
    print(f"Writing CSVs to {out_dir}\n")

    print("Exporting to CSV (all classes combined)...")
    for p in ifc_files:
        csv_name = p.stem + ".csv"
        csv_path = out_dir / csv_name
        try:
            model = ifcopenshell.open(str(p))

            # build a registry matched to this model's schema version
            # (IFC4, IFC4X3, etc.) so hierarchy lookups are accurate
            registry = get_registry(
                snapshot_path=bsdd_path if bsdd_path.exists() else None,
                schema_name=model.schema,
            )

            all_data: list[dict] = []
            all_attributes: set[str] = set()

            class_types = [
                "IfcWall", "IfcSlab", "IfcDoor", "IfcWindow", "IfcColumn", "IfcBeam",
                "IfcRoof", "IfcStair", "IfcRamp", "IfcFurniture",
                "IfcBuildingElementProxy",
                "IfcCovering", "IfcRailing", "IfcFlowSegment", "IfcFlowTerminal",
                "IfcFlowFitting", "IfcDuctSegment", "IfcPipeSegment", "IfcPlate",
                "IfcMember", "IfcFooting", "IfcPile", "IfcBuildingStorey", "IfcSpace",
                "IfcZone", "IfcProject", "IfcSite", "IfcBuilding",
            ]

            skipped: dict[str, str] = {}
            for ct in class_types:
                try:
                    data, attributes = get_objects_data_by_class(
                        model, ct, registry=registry
                    )
                    all_data.extend(data)
                    all_attributes.update(attributes)
                except Exception as exc:
                    logger.warning("Class %s failed for %s: %s", ct, p.name, exc)
                    skipped[ct] = str(exc)

            _log_skipped_summary(skipped)

            if all_data:
                attributes_list = sorted(all_attributes)
                rows = []
                for obj_data in all_data:
                    row = {}
                    for attr in attributes_list:
                        value = get_attribute_value(obj_data, attr)
                        row[attr] = value if value is not None else ""
                    for key in CORE_COLUMNS:
                        row[key] = obj_data.get(key, "")
                    rows.append(row)

                all_columns = CORE_COLUMNS + attributes_list
                ordered_rows = [
                    {col: row.get(col, "") for col in all_columns} for row in rows
                ]
                df = pd.DataFrame(ordered_rows, columns=all_columns)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)

                # show a quick property coverage summary so we can see the improvement
                pset_cols = [c for c in all_columns if "." in c]
                filled = sum(
                    1 for col in pset_cols
                    for val in df[col] if val not in ("", None)
                )
                print(
                    f"  {p.name} -> {csv_name} "
                    f"({len(df)} rows, {len(pset_cols)} property columns, "
                    f"{filled} non-empty values)"
                )
            else:
                print(f"  {p.name} -> No data found")

        except Exception as e:
            print(f"  {p.name} -> ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
