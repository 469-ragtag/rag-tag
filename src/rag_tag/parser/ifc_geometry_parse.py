"""Geometry extraction helpers built on top of ifcopenshell.

Provides centroid, bounding-box, and optional mesh data for IFC elements.
"""

from __future__ import annotations

import inspect
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
    """Return standard ifcopenshell geometry settings for this project."""
    settings = ifcopenshell.geom.settings()
    # Ensure coordinates are in model/world coordinates.
    settings.set(settings.USE_WORLD_COORDS, True)
    # Avoid boolean opening subtractions for speed/stability.
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, True)
    return settings


def _extract_shape_mesh(
    element, settings: ifcopenshell.geom.settings
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return mesh vertices and triangular faces for an element."""
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
    except Exception as exc:
        LOG.debug("Mesh extraction failed for %s: %s", element.is_a(), exc)
        return None, None

    try:
        verts = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
    except Exception:
        verts = None

    try:
        faces_raw = np.asarray(shape.geometry.faces, dtype=int)
        if faces_raw.size > 0 and faces_raw.size % 3 == 0:
            faces = faces_raw.reshape(-1, 3)
        else:
            faces = None
    except Exception:
        faces = None

    if verts is not None and verts.size == 0:
        verts = None
    if faces is not None and faces.size == 0:
        faces = None
    return verts, faces


def get_element_centroid(
    element, settings: ifcopenshell.geom.settings
) -> np.ndarray | None:
    """Return an element centroid array ``(x, y, z)`` or ``None``."""
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
    """Return ``(min_xyz, max_xyz)`` axis-aligned bounds or ``None``."""
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


def build_geom_settings() -> ifcopenshell.geom.settings:
    """Build reusable geometry settings for per-element extraction calls."""
    return _build_geom_settings()


def get_element_geometry(
    element,
    settings: ifcopenshell.geom.settings,
) -> dict:
    """Extract centroid and bounding box in a single shape pass."""
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
        if verts.size == 0:
            return {"centroid": None, "bbox": None}
        centroid = verts.mean(axis=0)
        min_xyz = verts.min(axis=0)
        max_xyz = verts.max(axis=0)
        return {"centroid": centroid, "bbox": (min_xyz, max_xyz)}
    except Exception as exc:
        LOG.debug("Geometry extraction failed for %s: %s", element.is_a(), exc)
        return {"centroid": None, "bbox": None}


def get_ifc_model(ifc_path: Path):
    """Open an IFC file after optional validation checks."""
    if ifc_validate is not None:
        try:
            errors = _run_ifc_validation(ifc_path)
            if errors:
                raise InvalidIfcError(
                    f"IFC validation reported {len(errors)} issues for {ifc_path}"
                )
        except Exception as exc:
            if isinstance(exc, InvalidIfcError):
                raise
            LOG.warning("IFC validation skipped for %s: %s", ifc_path, exc)

    return ifcopenshell.open(str(ifc_path))


def _run_ifc_validation(ifc_path: Path) -> list:
    """Run ``ifcopenshell.validate`` across signature variants."""
    validate_func = ifc_validate.validate  # type: ignore[union-attr]

    try:
        signature = inspect.signature(validate_func)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "logger" in signature.parameters:
        try:
            return _coerce_validation_errors(validate_func(str(ifc_path), logger=LOG))
        except TypeError:
            return _coerce_validation_errors(validate_func(str(ifc_path), LOG))

    try:
        return _coerce_validation_errors(validate_func(str(ifc_path)))
    except TypeError:
        return _coerce_validation_errors(validate_func(str(ifc_path), LOG))


def _coerce_validation_errors(result: object) -> list:
    """Normalize validator outputs to a concrete list of issues."""
    if result is None:
        return []
    return list(result)


def get_all_elements(model, class_types: list[str] = None):
    """Return model elements for requested classes or default class set."""
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
    """Return per-element geometry records with centroid, bbox, and mesh data."""
    elements = get_all_elements(model, class_types)
    data = []

    settings = _build_geom_settings()

    for elem in elements:
        verts, faces = _extract_shape_mesh(elem, settings)
        if verts is not None:
            centroid = verts.mean(axis=0)
            min_xyz = verts.min(axis=0)
            max_xyz = verts.max(axis=0)
            bbox = (min_xyz, max_xyz)
        else:
            centroid = None
            bbox = None

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
            "mesh_vertices": (
                [tuple(float(v) for v in p) for p in verts.tolist()]
                if verts is not None
                else None
            ),
            "mesh_faces": (
                [tuple(int(idx) for idx in f) for f in faces.tolist()]
                if faces is not None
                else None
            ),
        }

        data.append(elem_data)

    return data
