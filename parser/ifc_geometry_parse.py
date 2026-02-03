from pathlib import Path
import ifcopenshell
import ifcopenshell.geom
import numpy as np


def get_element_centroid(element) -> np.ndarray:
    """
    Returns the centroid of an IFC element's geometry as (x, y, z).
    If geometry cannot be extracted, returns (0.0, 0.0, 0.0).
    """
    try:
        settings = ifcopenshell.geom.settings()
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)
        centroid = verts.mean(axis=0)
        return centroid
    except Exception:
        return np.array([0.0, 0.0, 0.0])


def get_element_bounding_box(element) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the axis-aligned bounding box of an IFC element as (min_xyz, max_xyz).
    If geometry cannot be extracted, returns ((0,0,0), (0,0,0)).
    """
    try:
        settings = ifcopenshell.geom.settings()
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)
        min_xyz = verts.min(axis=0)
        max_xyz = verts.max(axis=0)
        return min_xyz, max_xyz
    except Exception:
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])


def get_ifc_model(ifc_path: Path):
    """
    Opens an IFC file and returns the model object.
    """
    return ifcopenshell.open(str(ifc_path))


def get_all_elements(model, class_types: list[str] = None):
    """
    Returns all elements of given class types.
    If class_types is None, returns all building elements.
    """
    if class_types is None:
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

    elements = []
    for ct in class_types:
        try:
            elements.extend(model.by_type(ct))
        except Exception:
            continue
    return elements


def extract_geometry_data(model, class_types: list[str] = None):
    """
    Returns a list of dicts containing element GlobalId, Class, and centroid coordinates.
    """
    elements = get_all_elements(model, class_types)
    data = []

    for elem in elements:
        centroid = get_element_centroid(elem)
        min_bb, max_bb = get_element_bounding_box(elem)

        elem_data = {
            "GlobalId": getattr(elem, "GlobalId", ""),
            "Class": elem.is_a(),
            "Name": getattr(elem, "Name", ""),
            "X": float(centroid[0]),
            "Y": float(centroid[1]),
            "Z": float(centroid[2]),
            "MinX": float(min_bb[0]),
            "MinY": float(min_bb[1]),
            "MinZ": float(min_bb[2]),
            "MaxX": float(max_bb[0]),
            "MaxY": float(max_bb[1]),
            "MaxZ": float(max_bb[2]),
        }

        data.append(elem_data)

    return data

