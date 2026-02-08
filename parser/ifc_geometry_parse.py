from __future__ import annotations

import logging
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import numpy as np

try:  # Optional dependency in some builds
    import ifcopenshell.validate as ifc_validate  # type: ignore
except Exception:  # pragma: no cover - best-effort validation
    ifc_validate = None


LOG = logging.getLogger(__name__)


class InvalidIfcError(ValueError):
    """Raised when an IFC file fails validation."""


def _build_geom_settings() -> ifcopenshell.geom.settings:
    settings = ifcopenshell.geom.settings()
    # Ensure coordinates are in model/world coordinates.
    settings.set(settings.USE_WORLD_COORDS, True)
    # Avoid boolean opening subtractions for speed/stability.
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, True)
    return settings


def get_element_centroid(
    element, settings: ifcopenshell.geom.settings
) -> np.ndarray | None:
    """
    Returns the centroid of an IFC element's geometry as (x, y, z).
    If geometry cannot be extracted, returns (0.0, 0.0, 0.0).
    """
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
        if verts.size == 0:
            return None
        return verts.mean(axis=0)
    except Exception as exc:
        LOG.debug("Centroid extraction failed for %s: %s", element.is_a(), exc)
        return None


def get_element_bounding_box(
    element, settings: ifcopenshell.geom.settings
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns the axis-aligned bounding box of an IFC element as (min_xyz, max_xyz).
    If geometry cannot be extracted, returns ((0,0,0), (0,0,0)).
    """
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
        if verts.size == 0:
            return None
        min_xyz = verts.min(axis=0)
        max_xyz = verts.max(axis=0)
        return min_xyz, max_xyz
    except Exception as exc:
        LOG.debug("BBox extraction failed for %s: %s", element.is_a(), exc)
        return None


def get_ifc_model(ifc_path: Path):
    """
    Opens an IFC file and returns the model object.
    """
    if ifc_validate is not None:
        try:
            try:
                # Some versions require logger as positional arg.
                errors = list(ifc_validate.validate(str(ifc_path), LOG))
            except TypeError:
                try:
                    errors = list(ifc_validate.validate(str(ifc_path), logger=LOG))
                except TypeError:
                    # Fallback for signatures without logger.
                    errors = list(ifc_validate.validate(str(ifc_path)))
            if errors:
                raise InvalidIfcError(
                    f"IFC validation reported {len(errors)} issues for {ifc_path}"
                )
        except Exception as exc:
            if isinstance(exc, InvalidIfcError):
                raise
            LOG.warning("IFC validation failed for %s: %s", ifc_path, exc)

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
    Returns a list of dicts containing element GlobalId, Class, and centroid
    coordinates.
    """
    elements = get_all_elements(model, class_types)
    data = []

    settings = _build_geom_settings()

    for elem in elements:
        centroid = get_element_centroid(elem, settings)
        bbox = get_element_bounding_box(elem, settings)

        elem_data = {
            "GlobalId": getattr(elem, "GlobalId", ""),
            "Class": elem.is_a(),
            "Name": getattr(elem, "Name", ""),
            "centroid": (
                tuple(float(v) for v in centroid) if centroid is not None else None
            ),
            "bbox": (
                (
                    tuple(float(v) for v in bbox[0]),
                    tuple(float(v) for v in bbox[1]),
                )
                if bbox is not None
                else None
            ),
        }

        data.append(elem_data)

    return data
