from __future__ import annotations

import inspect
import logging
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement as ifc_placement
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


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull_2d(points_xy: np.ndarray) -> np.ndarray | None:
    """Return convex hull points in CCW order for Nx2 point array."""
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        return None
    if len(points_xy) < 3:
        return None

    # Deduplicate after rounding to reduce numeric noise from tessellation.
    pts = np.unique(np.round(points_xy.astype(float), 6), axis=0)
    if len(pts) < 3:
        return None

    # Monotonic chain needs lexicographic order.
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    if len(hull) < 3:
        return None
    return hull


def compute_footprint_polygon_2d(vertices: np.ndarray | None) -> np.ndarray | None:
    """Return an XY convex-hull footprint polygon from world-space vertices."""
    if vertices is None or vertices.ndim != 2 or vertices.shape[1] != 3:
        return None
    return _convex_hull_2d(vertices[:, :2])


def compute_oriented_bbox(vertices: np.ndarray | None) -> dict[str, np.ndarray] | None:
    """Return PCA-based oriented XY bounding box plus vertical extent."""
    if vertices is None or vertices.ndim != 2 or vertices.shape[1] != 3:
        return None
    if len(vertices) < 3:
        return None

    points_xy = vertices[:, :2].astype(float)
    center_xy = points_xy.mean(axis=0)
    centered = points_xy - center_xy
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axis_u = eigvecs[:, order[0]]
    axis_v = eigvecs[:, order[1]]
    axis_u = axis_u / max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = axis_v / max(float(np.linalg.norm(axis_v)), 1e-12)

    proj_u = centered @ axis_u
    proj_v = centered @ axis_v
    min_u, max_u = float(np.min(proj_u)), float(np.max(proj_u))
    min_v, max_v = float(np.min(proj_v)), float(np.max(proj_v))
    extent_u = (max_u - min_u) * 0.5
    extent_v = (max_v - min_v) * 0.5

    center_offset = axis_u * ((min_u + max_u) * 0.5) + axis_v * ((min_v + max_v) * 0.5)
    obb_center_xy = center_xy + center_offset
    z_min = float(np.min(vertices[:, 2]))
    z_max = float(np.max(vertices[:, 2]))
    center_z = (z_min + z_max) * 0.5
    extent_z = (z_max - z_min) * 0.5

    corners_local = np.array(
        [
            [max_u, max_v],
            [max_u, min_v],
            [min_u, min_v],
            [min_u, max_v],
        ],
        dtype=float,
    )
    corners_xy = np.array(
        [center_xy + axis_u * cu + axis_v * cv for cu, cv in corners_local],
        dtype=float,
    )

    return {
        "center": np.array([obb_center_xy[0], obb_center_xy[1], center_z], dtype=float),
        "axes": np.array(
            [
                [axis_u[0], axis_u[1], 0.0],
                [axis_v[0], axis_v[1], 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "extents": np.array([extent_u, extent_v, extent_z], dtype=float),
        "corners_xy": corners_xy,
    }


def get_element_local_placement_matrix(element) -> np.ndarray | None:
    """Return element local placement as a 4x4 matrix when available."""
    try:
        placement = getattr(element, "ObjectPlacement", None)
        if placement is None:
            return None
        matrix = ifc_placement.get_local_placement(placement)
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (4, 4):
            return None
        return arr
    except Exception as exc:
        LOG.debug("Placement extraction failed for %s: %s", element.is_a(), exc)
        return None


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


def get_element_mesh(
    element, settings: ifcopenshell.geom.settings
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return element mesh as ``(vertices, faces)`` arrays.

    - ``vertices`` shape: ``(n, 3)`` float
    - ``faces`` shape: ``(m, 3)`` int triangle indices

    Returns ``(None, None)`` when mesh extraction fails.
    """
    return _extract_shape_mesh(element, settings)


def build_geom_settings() -> ifcopenshell.geom.settings:
    """
    Build the standard geometry settings object.

    Create this once per session and pass it to every per-element call —
    the settings object is thread-safe and cheap to share.  Recreating it
    inside a loop is wasteful (small but avoidable allocation).
    """
    return _build_geom_settings()


def get_element_geometry(
    element,
    settings: ifcopenshell.geom.settings,
) -> dict:
    """
    Extract centroid and axis-aligned bounding box in a single shape pass.

    Only derived summary values are retained; raw mesh vertices and faces are
    never stored so callers cannot accidentally serialise them to JSONL.

    Returns::

        {"centroid": np.ndarray | None, "bbox": (min_xyz, max_xyz) | None}

    ``centroid`` is shape ``(3,)`` and ``bbox`` is a 2-tuple of shape-``(3,)``
    arrays.  Both keys are present even on failure (set to ``None``).
    """
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
    """
    Opens an IFC file and returns the model object.
    """
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
    if result is None:
        return []
    return list(result)


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
