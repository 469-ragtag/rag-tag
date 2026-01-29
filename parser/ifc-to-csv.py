from __future__ import annotations

import argparse
import json
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.element as element
import pandas as pd


def get_objects_data_by_class(model, class_type):
    """
    Get all objects of a specific class type and extract their data.
    Returns a list of dictionaries with object data and a set of all pset attributes.
    """
    objects = model.by_type(class_type)
    data_list = []
    pset_attributes = set()

    for obj in objects:
        try:
            express_id = (
                obj.id()
            )  # IfcOpenShell entity ExpressId (matches the #id in the IFC file)
        except Exception:
            express_id = id(obj)

        obj_data = {
            "ExpressId": express_id,
            "GlobalId": obj.GlobalId if hasattr(obj, "GlobalId") else "",
            "Class": obj.is_a(),
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

        # Get level/container name
        try:
            container = element.get_container(obj)
            obj_data["Level"] = (
                container.Name if container and hasattr(container, "Name") else ""
            )
        except Exception:
            obj_data["Level"] = ""

        # Get type name
        try:
            obj_type = element.get_type(obj)
            obj_data["TypeName"] = (
                obj_type.Name if obj_type and hasattr(obj_type, "Name") else ""
            )
        except Exception:
            obj_data["TypeName"] = ""

        # Get property sets
        try:
            psets = element.get_psets(obj, psets_only=True)
            obj_data["PropertySets"] = psets
            # Add pset attributes to our set
            add_pset_attributes(psets, pset_attributes)
        except Exception:
            obj_data["PropertySets"] = {}

        # Get quantity sets
        try:
            qtos = element.get_psets(obj, qtos_only=True)
            obj_data["QuantitySets"] = qtos
            # Add qto attributes to our set
            add_pset_attributes(qtos, pset_attributes)
        except Exception:
            obj_data["QuantitySets"] = {}

        data_list.append(obj_data)

    return data_list, pset_attributes


def add_pset_attributes(psets, attributes_set):
    """
    Add pset/qto attributes to the set in format "PsetName.PropertyName".
    """
    if not psets:
        return

    for pset_name, pset_data in psets.items():
        if isinstance(pset_data, dict):
            for prop_name in pset_data.keys():
                if prop_name != "id":  # Skip the 'id' key
                    attributes_set.add(f"{pset_name}.{prop_name}")


def get_attribute_value(obj_data, attribute):
    """
    Get the value of an attribute from object data.
    Handles both direct attributes and pset/qto attributes (with dot notation).
    """
    # If no dot, it's a direct attribute
    if "." not in attribute:
        return obj_data.get(attribute, None)

    # Split pset/qto name and property name
    parts = attribute.split(".", 1)
    if len(parts) != 2:
        return None

    pset_name, prop_name = parts

    # Check in PropertySets first
    if "PropertySets" in obj_data and pset_name in obj_data["PropertySets"]:
        pset_data = obj_data["PropertySets"][pset_name]
        if isinstance(pset_data, dict) and prop_name in pset_data:
            return pset_data[prop_name]

    # Check in QuantitySets
    if "QuantitySets" in obj_data and pset_name in obj_data["QuantitySets"]:
        qto_data = obj_data["QuantitySets"][pset_name]
        if isinstance(qto_data, dict) and prop_name in qto_data:
            return qto_data[prop_name]

    return None


def export_to_csv(model, class_type, output_path):
    """
    Export objects of a specific class to CSV using pandas.
    """
    data, attributes = get_objects_data_by_class(model, class_type)

    if not data:
        return 0

    # Convert attributes set to sorted list
    attributes_list = sorted(attributes)

    # Create rows for pandas DataFrame
    rows = []
    for obj_data in data:
        row = {}
        for attr in attributes_list:
            value = get_attribute_value(obj_data, attr)
            row[attr] = value if value is not None else ""
        # Add core attributes
        for key in [
            "ExpressId",
            "GlobalId",
            "Class",
            "PredefinedType",
            "Name",
            "Description",
            "ObjectType",
            "Tag",
            "Level",
            "TypeName",
        ]:
            row[key] = obj_data.get(key, "")
        rows.append(row)

    # Create DataFrame with core columns first, then pset/qto columns
    core_columns = [
        "ExpressId",
        "GlobalId",
        "Class",
        "PredefinedType",
        "Name",
        "Description",
        "ObjectType",
        "Tag",
        "Level",
        "TypeName",
    ]
    all_columns = core_columns + attributes_list

    # Reorder rows to match column order
    ordered_rows = []
    for row in rows:
        ordered_row = {col: row.get(col, "") for col in all_columns}
        ordered_rows.append(ordered_row)

    df = pd.DataFrame(ordered_rows, columns=all_columns)

    # Export to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return len(df)


def export_to_excel(model, class_types, output_path):
    """
    Export multiple object classes to Excel with separate sheets per class.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for class_type in class_types:
            try:
                data, attributes = get_objects_data_by_class(model, class_type)

                if not data:
                    continue

                attributes_list = sorted(attributes)
                rows = []

                for obj_data in data:
                    row = {}
                    for attr in attributes_list:
                        value = get_attribute_value(obj_data, attr)
                        row[attr] = value if value is not None else ""
                    # Add core attributes
                    for key in [
                        "ExpressId",
                        "GlobalId",
                        "Class",
                        "PredefinedType",
                        "Name",
                        "Description",
                        "ObjectType",
                        "Tag",
                        "Level",
                        "TypeName",
                    ]:
                        row[key] = obj_data.get(key, "")
                    rows.append(row)

                core_columns = [
                    "ExpressId",
                    "GlobalId",
                    "Class",
                    "PredefinedType",
                    "Name",
                    "Description",
                    "ObjectType",
                    "Tag",
                    "Level",
                    "TypeName",
                ]
                all_columns = core_columns + attributes_list

                ordered_rows = []
                for row in rows:
                    ordered_row = {col: row.get(col, "") for col in all_columns}
                    ordered_rows.append(ordered_row)

                df = pd.DataFrame(ordered_rows, columns=all_columns)

                # Filter to only this class and drop empty columns
                df_filtered = df[df["Class"] == class_type].copy()
                df_filtered = df_filtered.dropna(axis=1, how="all")

                if not df_filtered.empty:
                    # Use class name as sheet name (remove 'Ifc' prefix if present)
                    sheet_name = class_type.replace("Ifc", "")[
                        :31
                    ]  # Excel sheet name limit
                    df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

            except Exception as e:
                print(f"  Warning: Could not export {class_type}: {e}")
                continue


def parse_ifc_to_csv(ifc_path: Path, csv_path: Path, class_type: str = None) -> int:  # type: ignore
    """
    Parse one IFC file and write CSV(s).
    If class_type is specified, only export that class. Otherwise export all building elements.
    """
    model = ifcopenshell.open(str(ifc_path))

    if class_type:
        # Export specific class
        return export_to_csv(model, class_type, csv_path)
    else:
        # Export all building elements to separate CSVs
        class_types = [
            "IfcWall",
            "IfcSlab",
            "IfcDoor",
            "IfcWindow",
            "IfcColumn",
            "IfcBeam",
            "IfcRoof",
            "IfcStair",
            "IfcRamp",
            "IfcFurniture",
            "IfcBuildingElementProxy",
            "IfcCovering",
            "IfcRailing",
            "IfcFlowSegment",
            "IfcFlowTerminal",
            "IfcFlowFitting",
            "IfcDuctSegment",
            "IfcPipeSegment",
            "IfcPlate",
            "IfcMember",
            "IfcFooting",
            "IfcPile",
            "IfcBuildingStorey",
            "IfcSpace",
            "IfcZone",
        ]

        total_rows = 0
        for ct in class_types:
            try:
                csv_name = f"{ifc_path.stem}_{ct.replace('Ifc', '')}.csv"
                csv_path_class = csv_path.parent / csv_name
                n = export_to_csv(model, ct, csv_path_class)
                if n > 0:
                    total_rows += n
            except Exception:
                continue

        return total_rows


def _find_ifc_dir(start_dir: Path) -> Path | None:
    """Find an IFC-Files directory by searching upwards from start_dir."""
    for base in (start_dir, *start_dir.parents):
        candidate = base / "IFC-Files"
        if candidate.is_dir():
            return candidate
    return None


def _find_project_root(start_dir: Path) -> Path | None:
    """Find the project root by searching upwards for pyproject.toml."""
    for base in (start_dir, *start_dir.parents):
        if (base / "pyproject.toml").is_file():
            return base
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Parse IFC file(s) from IFC-Files/ and export to CSV."
    )
    parser.add_argument(
        "--ifc-dir",
        type=Path,
        default=None,
        help="Directory containing .ifc files (default: auto-detect IFC-Files/ up from this script).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write CSV outputs (default: <project-root>/output).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.ifc_dir is not None:
        ifc_dir = args.ifc_dir.expanduser().resolve()
    else:
        ifc_dir = _find_ifc_dir(script_dir)
        if ifc_dir is None:
            print(
                "IFC directory not found. Provide --ifc-dir or create an IFC-Files/ folder."
            )
            return

    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
    else:
        project_root = _find_project_root(script_dir) or script_dir.parent
        out_dir = (project_root / "output").resolve()

    if not ifc_dir.is_dir():
        print(f"IFC directory not found: {ifc_dir}")
        return

    ifc_files = sorted(f for f in ifc_dir.iterdir() if f.suffix.lower() == ".ifc")
    if not ifc_files:
        print(f"No .ifc files found in {ifc_dir}")
        return

    print(f"Parsing {len(ifc_files)} IFC file(s) from {ifc_dir}")
    print(f"Writing CSVs to {out_dir}\n")

    # Option 1: Export all to one CSV per IFC file (all classes combined)
    print("Exporting to CSV (all classes combined)...")
    for p in ifc_files:
        csv_name = p.stem + ".csv"
        csv_path = out_dir / csv_name
        try:
            # Get all building elements
            model = ifcopenshell.open(str(p))
            all_data = []
            all_attributes = set()

            class_types = [
                "IfcWall",
                "IfcSlab",
                "IfcDoor",
                "IfcWindow",
                "IfcColumn",
                "IfcBeam",
                "IfcRoof",
                "IfcStair",
                "IfcRamp",
                "IfcFurniture",
                "IfcBuildingElementProxy",
                "IfcCovering",
                "IfcRailing",
                "IfcFlowSegment",
                "IfcFlowTerminal",
                "IfcFlowFitting",
                "IfcDuctSegment",
                "IfcPipeSegment",
                "IfcPlate",
                "IfcMember",
                "IfcFooting",
                "IfcPile",
                "IfcBuildingStorey",
                "IfcSpace",
                "IfcZone",
                "IfcProject",
                "IfcSite",
                "IfcBuilding",
            ]

            for ct in class_types:
                try:
                    data, attributes = get_objects_data_by_class(model, ct)
                    all_data.extend(data)
                    all_attributes.update(attributes)
                except Exception:
                    continue

            if all_data:
                attributes_list = sorted(all_attributes)
                rows = []

                for obj_data in all_data:
                    row = {}
                    for attr in attributes_list:
                        value = get_attribute_value(obj_data, attr)
                        row[attr] = value if value is not None else ""
                    for key in [
                        "ExpressId",
                        "GlobalId",
                        "Class",
                        "PredefinedType",
                        "Name",
                        "Description",
                        "ObjectType",
                        "Tag",
                        "Level",
                        "TypeName",
                    ]:
                        row[key] = obj_data.get(key, "")
                    rows.append(row)

                core_columns = [
                    "ExpressId",
                    "GlobalId",
                    "Class",
                    "PredefinedType",
                    "Name",
                    "Description",
                    "ObjectType",
                    "Tag",
                    "Level",
                    "TypeName",
                ]
                all_columns = core_columns + attributes_list

                ordered_rows = []
                for row in rows:
                    ordered_row = {col: row.get(col, "") for col in all_columns}
                    ordered_rows.append(ordered_row)

                df = pd.DataFrame(ordered_rows, columns=all_columns)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                print(f"  {p.name} -> {csv_name} ({len(df)} rows)")
            else:
                print(f"  {p.name} -> No data found")

        except Exception as e:
            print(f"  {p.name} -> ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
