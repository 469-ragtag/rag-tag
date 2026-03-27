# builds a NetworkX graph from the .jsonl files produced by ifc_to_jsonl.py
# each node represents an IFC element with its geometry and properties attached
# edges represent containment (which floor/space something is in),
# spatial proximity (adjacent_to / connected_to), topology (above/below/overlaps),
# and explicit IFC semantics (hosts, hosted_by, ifc_connected_to, belongs_to_system,
# in_zone, classified_as) sourced from the Relationships block added in Batch 0.
#
# Edge provenance:
#   source="ifc"       -  explicit relationship extracted from IFC relations
#   source="heuristic" - spatial adjacency derived from centroid/bbox distance
#   source="topology"  - bbox intersection / vertical overlap (already present)
#
# Payload modes (GRAPH_PAYLOAD_MODE env var):
#   full    - store the full parsed JSONL record as each node's payload (default).
#   minimal - store only the fields needed for DB lookup references and basic
#             filtering (GlobalId, ExpressId, IfcType, ClassRaw, Name).
#             PropertySets / Quantities / Geometry blocks are omitted, yielding
#             a smaller in-memory graph.  Dotted pset filters still work when a
#             DB path is wired into the graph context (get_element_properties uses
#             a session-level cache to avoid repeated DB opens).
#
# run with: uv run rag-tag-jsonl-to-graph

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from rag_tag.config import (
    DEFAULT_OVERLAP_XY_MODE,
    load_project_config,
)
from rag_tag.graph_contract import (
    canonicalize_undirected_edge_endpoints,
    is_symmetric_relation,
)
from rag_tag.paths import find_project_root

LOG = logging.getLogger(__name__)
_MODULE_DIR = Path(__file__).resolve().parent

_VIZ_ALIGNED_WITH_ALLOWED_CLASSES = frozenset(
    {"IfcWall", "IfcBeam", "IfcMember", "IfcCurtainWall", "IfcRailing"}
)
_VIZ_ALIGNED_WITH_MAX_PER_NODE = 8
_VIZ_CONTAINER_CLASSES = frozenset(
    {
        "IfcProject",
        "IfcSite",
        "IfcBuilding",
        "IfcBuildingStorey",
        "IfcSpace",
        "IfcZone",
        "IfcSpatialZone",
        "IfcTypeObject",
    }
)

try:
    import ifcopenshell
    import ifcopenshell.geom

    IFC_GEOM_AVAILABLE = True
except Exception:
    IFC_GEOM_AVAILABLE = False

try:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    OCC_AVAILABLE = True
except Exception:
    OCC_AVAILABLE = False


# --- geometry math (no IFC dependency, just numpy) ---


def _distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.dot(diff))


def distance_between_bboxes(a: tuple, b: tuple) -> float:
    # minimum distance between two 3D bounding boxes â€” 0 if they overlap
    amin, amax = a
    bmin, bmax = b
    axis_gaps = [max(bmin[i] - amax[i], amin[i] - bmax[i], 0.0) for i in range(3)]
    return float(np.linalg.norm(np.array(axis_gaps, dtype=float)))


def distance_between_points(a: tuple, b: tuple) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _downsample_points(points: np.ndarray, max_points: int = 256) -> np.ndarray:
    """Uniformly downsample points to cap pairwise distance cost."""
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
    return points[idx]


def _mesh_min_vertex_distance(
    mesh_a: tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None,
    mesh_b: tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None,
    *,
    max_points_per_mesh: int = 256,
) -> float | None:
    """Approximate mesh distance using min vertex-vertex Euclidean distance."""
    if mesh_a is None or mesh_b is None:
        return None
    verts_a, _faces_a = mesh_a
    verts_b, _faces_b = mesh_b
    if not verts_a or not verts_b:
        return None

    try:
        arr_a = np.asarray(verts_a, dtype=float)
        arr_b = np.asarray(verts_b, dtype=float)
    except Exception:
        return None
    if arr_a.ndim != 2 or arr_a.shape[1] != 3:
        return None
    if arr_b.ndim != 2 or arr_b.shape[1] != 3:
        return None

    arr_a = _downsample_points(arr_a, max_points=max_points_per_mesh)
    arr_b = _downsample_points(arr_b, max_points=max_points_per_mesh)
    if arr_a.size == 0 or arr_b.size == 0:
        return None

    # Broadcast pairwise distances with bounded array size after downsampling.
    delta = arr_a[:, None, :] - arr_b[None, :, :]
    d2 = np.sum(delta * delta, axis=2)
    return float(np.sqrt(np.min(d2)))


def _bbox_xy_overlap_area(a: tuple, b: tuple) -> float:
    # how much the footprints of two elements overlap in the XY plane
    ax0, ay0, _ = a[0]
    ax1, ay1, _ = a[1]
    bx0, by0, _ = b[0]
    bx1, by1, _ = b[1]
    overlap_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_y = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(overlap_x * overlap_y)


def _polygon_area_xy(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _inside_half_plane(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> bool:
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-9


def _line_intersection_xy(
    s: tuple[float, float],
    e: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = s
    x2, y2 = e
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return e
    num_x = ((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4))
    num_y = ((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4))
    return (num_x / den, num_y / den)


def _convex_polygon_intersection_area(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> float:
    """Sutherland-Hodgman clip area for convex polygons (CCW expected)."""
    if len(subject) < 3 or len(clip) < 3:
        return 0.0
    output = subject[:]
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        if not output:
            return 0.0
        input_list = output
        output = []
        s = input_list[-1]
        for e in input_list:
            if _inside_half_plane(e, a, b):
                if not _inside_half_plane(s, a, b):
                    output.append(_line_intersection_xy(s, e, a, b))
                output.append(e)
            elif _inside_half_plane(s, a, b):
                output.append(_line_intersection_xy(s, e, a, b))
            s = e
    if len(output) < 3:
        return 0.0
    return _polygon_area_xy(output)


def _bboxes_intersect(a: tuple, b: tuple) -> bool:
    amin, amax = a
    bmin, bmax = b
    for i in range(3):
        if amax[i] < bmin[i] or bmax[i] < amin[i]:
            return False
    return True


def _bbox_volume(bbox: tuple) -> float:
    (min_xyz, max_xyz) = bbox
    dx = max(0.0, float(max_xyz[0]) - float(min_xyz[0]))
    dy = max(0.0, float(max_xyz[1]) - float(min_xyz[1]))
    dz = max(0.0, float(max_xyz[2]) - float(min_xyz[2]))
    return dx * dy * dz


def _compute_common_metrics(shape_a, shape_b) -> tuple[float, float] | None:
    """Compute exact 3D intersection/contact metrics via OCC booleans."""
    if not OCC_AVAILABLE:
        return None
    try:
        common = BRepAlgoAPI_Common(shape_a, shape_b)
        common.Build()
        if not common.IsDone():
            return None
        common_shape = common.Shape()
    except Exception as exc:
        LOG.debug("OCC boolean common failed: %s", exc)
        return None

    try:
        vol_props = GProp_GProps()
        brepgprop.VolumeProperties(common_shape, vol_props)
        intersection_volume = float(vol_props.Mass())
    except Exception:
        intersection_volume = 0.0

    try:
        surf_props = GProp_GProps()
        brepgprop.SurfaceProperties(common_shape, surf_props)
        contact_area = float(surf_props.Mass())
    except Exception:
        contact_area = 0.0

    return max(intersection_volume, 0.0), max(contact_area, 0.0)


def _compute_shape_min_distance(shape_a, shape_b) -> float | None:
    """Compute exact minimum distance between two OCC shapes."""
    if not OCC_AVAILABLE:
        return None
    try:
        dist_calc = BRepExtrema_DistShapeShape(shape_a, shape_b)
        dist_calc.Perform()
        if not dist_calc.IsDone():
            return None
        d = float(dist_calc.Value())
        if not math.isfinite(d):
            return None
        return max(0.0, d)
    except Exception as exc:
        LOG.debug("OCC min-distance failed: %s", exc)
        return None


def _build_occ_shape_index(
    element_refs: list[tuple[str, str, str]],
) -> dict[str, object]:
    """Build ``node_id -> OCC shape`` index from IFC files when available."""
    if not IFC_GEOM_AVAILABLE or not OCC_AVAILABLE:
        return {}

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir) or script_dir.parent.parent.parent
    ifc_dir = project_root / "IFC-Files"
    if not ifc_dir.is_dir():
        return {}

    ifc_files = sorted(ifc_dir.glob("*.ifc"))
    if not ifc_files:
        return {}

    try:
        occ_settings = ifcopenshell.geom.settings()
        occ_settings.set(occ_settings.USE_WORLD_COORDS, True)
        occ_settings.set(occ_settings.DISABLE_OPENING_SUBTRACTIONS, True)
        occ_settings.set(occ_settings.USE_PYTHON_OPENCASCADE, True)
    except Exception as exc:
        LOG.debug("OCC settings unavailable: %s", exc)
        return {}

    models_by_path: dict[Path, object | None] = {}
    ifc_file_by_stem = {ifc_file.stem: ifc_file for ifc_file in ifc_files}
    shape_by_node_id: dict[str, object] = {}

    def _open_model(ifc_file: Path) -> object | None:
        if ifc_file in models_by_path:
            return models_by_path[ifc_file]
        try:
            model = ifcopenshell.open(str(ifc_file))
        except Exception as exc:
            LOG.debug("Skipping IFC file %s: %s", ifc_file, exc)
            models_by_path[ifc_file] = None
            return None
        models_by_path[ifc_file] = model
        return model

    for node_id, dataset_key, gid in element_refs:
        preferred_file = ifc_file_by_stem.get(dataset_key)
        candidate_files: list[Path] = []
        if preferred_file is not None:
            candidate_files.append(preferred_file)
        candidate_files.extend(
            ifc_file for ifc_file in ifc_files if ifc_file != preferred_file
        )

        for ifc_file in candidate_files:
            model = _open_model(ifc_file)
            if model is None:
                continue
            try:
                elem = model.by_guid(gid)
                if elem is None:
                    continue
                shape = ifcopenshell.geom.create_shape(occ_settings, elem)
                shape_by_node_id[node_id] = shape.geometry
                break
            except Exception:
                continue

    if shape_by_node_id:
        LOG.info(
            "Loaded OCC shapes for %d/%d graph elements",
            len(shape_by_node_id),
            len(element_refs),
        )
    return shape_by_node_id


def _get_occ_shape_index(
    G: nx.DiGraph | nx.MultiDiGraph,
    element_refs: list[tuple[str, str, str]],
) -> dict[str, object]:
    """Return cached OCC shape index for the graph, building it if needed."""
    cached = G.graph.get("_occ_shape_index")
    cached_gids = G.graph.get("_occ_shape_index_gids")
    wanted = frozenset(node_id for node_id, _dataset_key, _gid in element_refs)
    if (
        isinstance(cached, dict)
        and isinstance(cached_gids, frozenset)
        and wanted.issubset(cached_gids)
    ):
        return cached

    built = _build_occ_shape_index(element_refs)
    G.graph["_occ_shape_index"] = built
    G.graph["_occ_shape_index_gids"] = frozenset(built.keys())
    return built


def _normalize_positions(positions: list) -> np.ndarray | None:
    if not positions:
        return None
    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def _estimate_cell_size(positions: np.ndarray) -> float:
    # heuristic: average spacing between elements based on model volume and count
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extent = maxs - mins
    volume = float(np.prod(np.maximum(extent, 1e-9)))
    avg_spacing = volume ** (1.0 / 3.0) / max(len(positions) ** (1.0 / 3.0), 1.0)
    return max(avg_spacing, 0.5)


def _cell_for_point(point: np.ndarray, cell_size: float) -> tuple:
    return tuple(np.floor(point / cell_size).astype(int))


def _neighbor_cell_keys(key: tuple, radius: int):
    # yields all grid cell keys at exactly Chebyshev radius from key
    # radius 0 = just the cell itself, radius 1 = the 26 surrounding cells, etc.
    if radius == 0:
        yield key
        return
    xr = range(key[0] - radius, key[0] + radius + 1)
    yr = range(key[1] - radius, key[1] + radius + 1)
    zr = range(key[2] - radius, key[2] + radius + 1)
    for cx, cy, cz in product(xr, yr, zr):
        if max(abs(cx - key[0]), abs(cy - key[1]), abs(cz - key[2])) == radius:
            yield (cx, cy, cz)


def _build_spatial_grid(positions: np.ndarray, cell_size: float) -> dict:
    grid: dict = defaultdict(list)
    for idx, point in enumerate(positions):
        grid[_cell_for_point(point, cell_size)].append(idx)
    return grid


@dataclass
class _KDNode:
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    idxs: np.ndarray
    left: "_KDNode | None" = None
    right: "_KDNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _build_kd_tree(
    pos: np.ndarray, idxs: np.ndarray, leaf_size: int = 24
) -> _KDNode | None:
    if idxs.size == 0:
        return None

    pts = pos[idxs]
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    if idxs.size <= leaf_size:
        return _KDNode(bbox_min=bbox_min, bbox_max=bbox_max, idxs=idxs)

    axis = int(np.argmax(bbox_max - bbox_min))
    order = np.argsort(pts[:, axis], kind="mergesort")
    sorted_idxs = idxs[order]
    mid = sorted_idxs.size // 2
    left_idxs = sorted_idxs[:mid]
    right_idxs = sorted_idxs[mid:]

    left = _build_kd_tree(pos, left_idxs, leaf_size=leaf_size)
    right = _build_kd_tree(pos, right_idxs, leaf_size=leaf_size)
    return _KDNode(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        idxs=np.empty(0, dtype=int),
        left=left,
        right=right,
    )


def _point_box_distance_sq(
    point: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> float:
    delta = np.maximum(0.0, np.maximum(bmin - point, point - bmax))
    return float(delta.dot(delta))


def _query_kd_radius(
    node: _KDNode | None,
    pos: np.ndarray,
    query_point: np.ndarray,
    radius_sq: float,
    out: list[int],
) -> None:
    if node is None:
        return
    if _point_box_distance_sq(query_point, node.bbox_min, node.bbox_max) > radius_sq:
        return

    if node.is_leaf:
        if node.idxs.size == 0:
            return
        pts = pos[node.idxs]
        d2 = np.sum((pts - query_point) ** 2, axis=1)
        for k in np.flatnonzero(d2 <= radius_sq):
            out.append(int(node.idxs[int(k)]))
        return

    _query_kd_radius(node.left, pos, query_point, radius_sq, out)
    _query_kd_radius(node.right, pos, query_point, radius_sq, out)


def _candidate_pairs_kdtree(pos: np.ndarray, radius: float) -> list[tuple[int, int]]:
    """Return unique index pairs within centroid radius using a KD-tree."""
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 2:
        return []
    root = _build_kd_tree(pos, np.arange(len(pos), dtype=int))
    if root is None:
        return []
    radius_sq = float(radius) * float(radius)
    pairs: set[tuple[int, int]] = set()
    for i in range(len(pos)):
        found: list[int] = []
        _query_kd_radius(root, pos, pos[i], radius_sq, found)
        for j in found:
            if j <= i:
                continue
            pairs.add((i, j))
    return sorted(pairs)


def compute_adjacency_threshold(positions: list) -> float:
    # figure out a reasonable distance threshold for "adjacent" elements
    # using the median nearest-neighbour distance across all centroids
    if len(positions) < 2:
        return 1.0
    pos = _normalize_positions(positions)
    if pos is None:
        return 1.0

    cell_size = _estimate_cell_size(pos)
    grid = _build_spatial_grid(pos, cell_size)
    cell_keys_arr = np.asarray(list(grid.keys()), dtype=int)
    min_key = cell_keys_arr.min(axis=0)
    max_key = cell_keys_arr.max(axis=0)

    nn_distances: list[float] = []
    for i, point in enumerate(pos):
        key = _cell_for_point(point, cell_size)
        max_radius = int(np.max(np.maximum(max_key - key, key - min_key)))
        best_sq = math.inf
        found_any = False
        for radius in range(max_radius + 1):
            for neighbor_key in _neighbor_cell_keys(key, radius):
                for j in grid.get(neighbor_key, []):
                    if i == j:
                        continue
                    d2 = _distance_sq(point, pos[j])
                    if d2 < best_sq:
                        best_sq = d2
                        found_any = True
            if found_any and math.sqrt(best_sq) <= radius * cell_size:
                break
        if found_any:
            nn_distances.append(math.sqrt(best_sq))

    if not nn_distances:
        return 1.0
    return max(0.5, float(np.median(nn_distances)) * 1.5)


# --- payload mode helpers ---


def _resolve_graph_payload_mode(mode: str) -> str:
    """Return a validated payload mode string; falls back to 'full' on bad input."""
    cleaned = (mode or "").strip().lower()
    return cleaned if cleaned in ("full", "minimal") else "full"


def _minimal_payload_stub(rec: dict) -> dict:
    """Return a compact payload containing only the fields needed for:

    - DB lookup references (GlobalId, ExpressId).
    - Fuzzy search candidates (Name, IfcType, ClassRaw).

    All other data (PropertySets, Quantities, Geometry, Materials,
    Relationships) is omitted to reduce memory usage in minimal mode.
    """
    stub: dict = {}
    for key in ("GlobalId", "ExpressId", "IfcType", "ClassRaw", "Name"):
        val = rec.get(key)
        if val is not None:
            stub[key] = val
    return stub


def _make_payload(rec: dict, payload_mode: str) -> dict:
    """Return the payload dict for a node according to the payload mode."""
    if payload_mode == "minimal":
        return _minimal_payload_stub(rec)
    return rec


# --- helpers for reading geometry and properties out of a JSONL record ---


def _geom_from_record(rec: dict) -> tuple[tuple | None, tuple | None]:
    geom_block = rec.get("Geometry") or {}
    centroid_raw = geom_block.get("Centroid")
    bbox_raw = geom_block.get("BoundingBox")

    centroid: tuple | None = None
    if isinstance(centroid_raw, (list, tuple)) and len(centroid_raw) == 3:
        centroid = tuple(float(v) for v in centroid_raw)

    bbox: tuple | None = None
    if isinstance(bbox_raw, dict):
        mn = bbox_raw.get("min")
        mx = bbox_raw.get("max")
        if mn and mx and len(mn) == 3 and len(mx) == 3:
            bbox = (
                tuple(float(v) for v in mn),
                tuple(float(v) for v in mx),
            )

    return centroid, bbox


def _oriented_geom_from_record(
    rec: dict,
) -> tuple[
    list[tuple[float, float]] | None,
    dict[str, object] | None,
    list[list[float]] | None,
]:
    geom_block = rec.get("Geometry") or {}

    footprint_raw = geom_block.get("FootprintPolygon2D")
    footprint: list[tuple[float, float]] | None = None
    if isinstance(footprint_raw, list):
        parsed: list[tuple[float, float]] = []
        for p in footprint_raw:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                parsed.append((float(p[0]), float(p[1])))
        if len(parsed) >= 3:
            footprint = parsed

    obb_raw = geom_block.get("OrientedBoundingBox")
    obb: dict[str, object] | None = None
    if isinstance(obb_raw, dict):
        center = obb_raw.get("center")
        axes = obb_raw.get("axes")
        extents = obb_raw.get("extents")
        corners_xy = obb_raw.get("corners_xy")
        if (
            isinstance(center, list)
            and len(center) == 3
            and isinstance(axes, list)
            and len(axes) == 3
            and isinstance(extents, list)
            and len(extents) == 3
        ):
            obb = {
                "center": tuple(float(v) for v in center),
                "axes": tuple(
                    tuple(float(v) for v in axis)
                    for axis in axes
                    if isinstance(axis, list) and len(axis) == 3
                ),
                "extents": tuple(float(v) for v in extents),
                "corners_xy": (
                    [
                        (float(p[0]), float(p[1]))
                        for p in corners_xy
                        if isinstance(p, list) and len(p) == 2
                    ]
                    if isinstance(corners_xy, list)
                    else None
                ),
            }

    placement_raw = geom_block.get("LocalPlacementMatrix")
    placement: list[list[float]] | None = None
    if (
        isinstance(placement_raw, list)
        and len(placement_raw) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in placement_raw)
    ):
        placement = [
            [float(v) for v in row] for row in placement_raw if isinstance(row, list)
        ]

    return footprint, obb, placement


def _mesh_from_record(
    rec: dict,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None:
    geom_block = rec.get("Geometry") or {}
    vertices = geom_block.get("MeshVertices")
    faces = geom_block.get("MeshFaces")
    if not isinstance(vertices, list) or not isinstance(faces, list):
        return None
    try:
        mesh_vertices = [tuple(float(v) for v in p) for p in vertices]
        mesh_faces = [tuple(int(i) for i in f) for f in faces]
    except (TypeError, ValueError):
        return None
    if not mesh_vertices or not mesh_faces:
        return None
    return mesh_vertices, mesh_faces


def _flat_properties(rec: dict) -> dict:
    # the graph tools still expect a flat "properties" dict on each node
    # so we pull the top-level fields here for backward compatibility
    return {
        "GlobalId": rec.get("GlobalId"),
        "ExpressId": rec.get("ExpressId"),
        "Class": rec.get("IfcType"),
        "ClassRaw": rec.get("ClassRaw"),
        "Name": rec.get("Name"),
        "Description": rec.get("Description"),
        "ObjectType": rec.get("ObjectType"),
        "Tag": rec.get("Tag"),
        "Level": (rec.get("Hierarchy") or {}).get("Level"),
        "TypeName": rec.get("TypeName"),
        "PredefinedType": rec.get("PredefinedType"),
        # Materials is a list[str]; preserved as-is for membership filtering.
        "Materials": rec.get("Materials") or [],
    }


# --- helpers for explicit IFC relationships ---


def _normalize_context_label(label: str) -> str:
    """Return a stable, whitespace-normalised version of a context label.

    Collapses interior runs of whitespace to a single space and strips
    leading/trailing whitespace so that equivalent labels produce the same
    deterministic node ID regardless of minor formatting differences.
    """
    return " ".join(label.split())


def _dataset_key_from_path(jsonl_path: Path) -> str:
    return jsonl_path.stem


def _project_node_id(dataset_key: str, *, namespaced: bool) -> str:
    if namespaced:
        return f"IfcProject::{dataset_key}"
    return "IfcProject"


def _building_node_id(dataset_key: str, *, namespaced: bool) -> str:
    if namespaced:
        return f"IfcBuilding::{dataset_key}"
    return "IfcBuilding"


def _storey_node_id(dataset_key: str, gid: str, *, namespaced: bool) -> str:
    if namespaced:
        return f"Storey::{dataset_key}::{gid}"
    return f"Storey::{gid}"


def _element_node_id(dataset_key: str, gid: str, *, namespaced: bool) -> str:
    if namespaced:
        return f"Element::{dataset_key}::{gid}"
    return f"Element::{gid}"


def _context_node_id(
    kind: str,
    dataset_key: str,
    label: str,
    *,
    namespaced: bool,
) -> str:
    if namespaced:
        return f"{kind}::{dataset_key}::{label}"
    return f"{kind}::{label}"


def _iter_relation_edge_attrs_between(
    G: nx.DiGraph | nx.MultiDiGraph,
    u: str,
    v: str,
):
    edge_data = G.get_edge_data(u, v)
    if edge_data is None:
        return
    if G.is_multigraph():
        for attrs in edge_data.values():
            if isinstance(attrs, dict):
                yield attrs
        return
    if isinstance(edge_data, dict):
        yield edge_data


def _has_relation(
    G: nx.DiGraph | nx.MultiDiGraph,
    u: str,
    v: str,
    relation: str,
    *,
    source: str | None = None,
) -> bool:
    normalized_relation = str(relation).strip().lower()
    normalized_source = str(source).strip().lower() if source is not None else None
    for attrs in _iter_relation_edge_attrs_between(G, u, v) or ():
        if str(attrs.get("relation", "")).strip().lower() != normalized_relation:
            continue
        if normalized_source is None:
            return True
        if str(attrs.get("source", "")).strip().lower() == normalized_source:
            return True
    return False


def _add_edge_once(
    G: nx.DiGraph | nx.MultiDiGraph,
    u: str,
    v: str,
    **edge_attrs: object,
) -> bool:
    relation_value = edge_attrs.get("relation")
    if relation_value is None:
        return False
    relation = str(relation_value).strip().lower()
    if not relation:
        return False
    source_value = edge_attrs.get("source")
    source = str(source_value).strip().lower() if source_value is not None else None
    if _has_relation(G, u, v, relation, source=source):
        return False
    G.add_edge(u, v, **edge_attrs)
    return True


def _add_symmetric_edge_once(
    G: nx.DiGraph | nx.MultiDiGraph,
    u: str,
    v: str,
    **edge_attrs: object,
) -> bool:
    canonical_u, canonical_v = canonicalize_undirected_edge_endpoints(u, v)
    return _add_edge_once(G, canonical_u, canonical_v, **edge_attrs)


def _add_explicit_relationships(
    G: nx.DiGraph | nx.MultiDiGraph,
    node_id: str,
    rels: dict,
    node_id_by_gid: dict[str, str],
    *,
    dataset_key: str | None = None,
    namespaced_ids: bool = False,
) -> None:
    """Materialise the explicit IFC relationships from a record's Relationships block.

    Edges added here carry ``source="ifc"`` to distinguish them from heuristic
    spatial edges (``source="heuristic"``) and topology edges (``source="topology"``).

    Relationship semantics
    ----------------------
    hosts / hosted_by  â€” element-to-element directed; target resolved via GlobalId.
    ifc_connected_to   â€” undirected IFC connectivity; stored once per pair.
    belongs_to_system  â€” element â†’ System context node (created on demand).
    in_zone            â€” element â†’ Zone context node (created on demand).
    classified_as      â€” element â†’ Classification context node (created on demand).
    """
    if node_id not in G:
        return
    if not isinstance(rels, dict):
        return
    dataset_label = dataset_key if isinstance(dataset_key, str) else ""

    def _add_ifc_edge_once(u: str, v: str, relation: str) -> None:
        edge_attrs = {"relation": relation, "source": "ifc"}
        if is_symmetric_relation(relation):
            _add_symmetric_edge_once(G, u, v, **edge_attrs)
            return
        _add_edge_once(G, u, v, **edge_attrs)

    # --- element-to-element: hosts ---
    for target_gid in rels.get("hosts") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            _add_ifc_edge_once(node_id, target, "hosts")

    # --- element-to-element: hosted_by ---
    for target_gid in rels.get("hosted_by") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            _add_ifc_edge_once(node_id, target, "hosted_by")

    # --- element-to-element: ifc_connected_to ---
    for target_gid in rels.get("ifc_connected_to") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            _add_ifc_edge_once(node_id, target, "ifc_connected_to")

    # --- element-to-type: typed_by ---
    for target_gid in rels.get("typed_by") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            _add_ifc_edge_once(node_id, target, "typed_by")

    # --- element-to-element: path_connected_to ---
    for target_gid in rels.get("path_connected_to") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            _add_ifc_edge_once(node_id, target, "path_connected_to")

    # --- space boundary relations ---
    for target_gid in rels.get("space_bounded_by") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            _add_ifc_edge_once(node_id, target, "space_bounded_by")
    for target_gid in rels.get("bounds_space") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            _add_ifc_edge_once(node_id, target, "bounds_space")

    # --- element â†’ System context node ---
    for raw_label in rels.get("belongs_to_system") or []:
        label = _normalize_context_label(raw_label)
        if not label:
            continue
        system_nid = _context_node_id(
            "System",
            dataset_label,
            label,
            namespaced=namespaced_ids,
        )
        if system_nid not in G:
            G.add_node(
                system_nid,
                label=label,
                class_="IfcSystem",
                node_kind="context",
                geometry=None,
                dataset=dataset_label,
            )
        _add_ifc_edge_once(node_id, system_nid, "belongs_to_system")

    # --- element â†’ Zone context node ---
    for raw_label in rels.get("in_zone") or []:
        label = _normalize_context_label(raw_label)
        if not label:
            continue
        zone_nid = _context_node_id(
            "Zone",
            dataset_label,
            label,
            namespaced=namespaced_ids,
        )
        if zone_nid not in G:
            G.add_node(
                zone_nid,
                label=label,
                class_="IfcZone",
                node_kind="context",
                geometry=None,
                dataset=dataset_label,
            )
        _add_ifc_edge_once(node_id, zone_nid, "in_zone")

    # --- element â†’ Classification context node ---
    for raw_label in rels.get("classified_as") or []:
        label = _normalize_context_label(raw_label)
        if not label:
            continue
        cls_nid = _context_node_id(
            "Classification",
            dataset_label,
            label,
            namespaced=namespaced_ids,
        )
        if cls_nid not in G:
            G.add_node(
                cls_nid,
                label=label,
                class_="IfcClassificationReference",
                node_kind="context",
                geometry=None,
                dataset=dataset_label,
            )
        _add_ifc_edge_once(node_id, cls_nid, "classified_as")


def _add_shared_space_boundary_edges(
    G: nx.DiGraph | nx.MultiDiGraph,
) -> int:
    """Connect spaces that share one or more explicit IFC boundary elements."""

    boundary_to_spaces: dict[str, set[str]] = defaultdict(set)
    for space_id, space_data in G.nodes(data=True):
        if _normalize_ifc_class_name(space_data.get("class_")) != "IfcSpace":
            continue
        for boundary_id in G.successors(space_id):
            if boundary_id not in G or boundary_id == space_id:
                continue
            if _normalize_ifc_class_name(G.nodes[boundary_id].get("class_")) == (
                "IfcOpeningElement"
            ):
                continue
            has_space_boundary = any(
                str(attrs.get("relation", "")).strip().lower() == "space_bounded_by"
                and str(attrs.get("source", "")).strip().lower() == "ifc"
                for attrs in _iter_relation_edge_attrs_between(G, space_id, boundary_id)
                or ()
            )
            if has_space_boundary:
                boundary_to_spaces[boundary_id].add(space_id)

    pair_to_boundaries: dict[tuple[str, str], set[str]] = defaultdict(set)
    for boundary_id, space_ids in boundary_to_spaces.items():
        ordered_spaces = sorted(space_ids)
        for index, source_id in enumerate(ordered_spaces):
            for target_id in ordered_spaces[index + 1 :]:
                pair_to_boundaries[(source_id, target_id)].add(boundary_id)

    edges_added = 0
    for (source_id, target_id), shared_boundary_elements in sorted(
        pair_to_boundaries.items()
    ):
        edge_attrs = {
            "relation": "shares_boundary_with",
            "source": "ifc",
            "derived_from": "space_bounded_by",
            "shared_boundary_elements": sorted(shared_boundary_elements),
        }
        if _add_symmetric_edge_once(G, source_id, target_id, **edge_attrs):
            edges_added += 1
    return edges_added


# --- graph construction ---


def _add_containment_edge(
    G: nx.DiGraph | nx.MultiDiGraph, parent_id: str, child_id: str
) -> None:
    if parent_id == child_id:
        return
    if parent_id not in G or child_id not in G:
        return
    G.add_edge(parent_id, child_id, relation="contains")
    G.add_edge(child_id, parent_id, relation="contained_in")


@dataclass(frozen=True)
class DerivedEdgePruningPolicy:
    """Resolved policy for pruning selected classes from derived edge phases."""

    enabled: bool
    exclude_classes: frozenset[str]


@dataclass(frozen=True)
class OverlapXYPolicy:
    """Resolved policy controlling raw ``overlaps_xy`` edge emission."""

    mode: str
    min_ratio: float
    top_k: int


@dataclass(frozen=True)
class _OverlapXYCandidate:
    """Positive-overlap candidate between two elements."""

    source_id: str
    target_id: str
    overlap_area_xy: float
    overlap_ratio: float


def _normalize_ifc_class_name(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not text.lower().startswith("ifc"):
        text = f"Ifc{text}"
    return text


def _normalize_excluded_classes(values: object) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, str):
        values = [values]
    normalized = {
        class_name
        for value in values
        if (class_name := _normalize_ifc_class_name(value)) is not None
    }
    return frozenset(sorted(normalized))


def _resolve_derived_edge_pruning_policy(
    *,
    enabled: bool | None = None,
    exclude_classes: object = None,
) -> DerivedEdgePruningPolicy:
    configured = load_project_config(
        _MODULE_DIR
    ).config.graph_build.derived_edge_pruning
    resolved_enabled = configured.enabled if enabled is None else enabled
    configured_classes = (
        configured.exclude_classes if exclude_classes is None else exclude_classes
    )
    resolved_classes = _normalize_excluded_classes(configured_classes)
    if not resolved_enabled:
        resolved_classes = frozenset()
    return DerivedEdgePruningPolicy(
        enabled=resolved_enabled,
        exclude_classes=resolved_classes,
    )


def _normalize_overlap_xy_mode(value: object | None) -> str:
    if value is None:
        return DEFAULT_OVERLAP_XY_MODE
    normalized = str(value).strip().lower()
    if normalized not in {"full", "threshold", "top_k", "none"}:
        raise ValueError("overlap_xy mode must be one of: full, threshold, top_k, none")
    return normalized


def _resolve_overlap_xy_policy(
    *,
    mode: object | None = None,
    min_ratio: object | None = None,
    top_k: object | None = None,
) -> OverlapXYPolicy:
    configured = load_project_config(_MODULE_DIR).config.graph_build.overlap_xy
    resolved_mode = _normalize_overlap_xy_mode(
        configured.mode if mode is None else mode
    )
    resolved_min_ratio = configured.min_ratio if min_ratio is None else float(min_ratio)
    resolved_top_k = configured.top_k if top_k is None else int(top_k)
    if not 0.0 <= resolved_min_ratio <= 1.0:
        raise ValueError("overlap_xy min_ratio must be between 0.0 and 1.0")
    if resolved_top_k < 1:
        raise ValueError("overlap_xy top_k must be >= 1")
    return OverlapXYPolicy(
        mode=resolved_mode,
        min_ratio=resolved_min_ratio,
        top_k=resolved_top_k,
    )


def _footprint_area_xy(
    footprint: list[tuple[float, float]] | None, bbox: tuple
) -> float:
    if footprint is not None:
        area = _polygon_area_xy(footprint)
        if area > 0.0:
            return area
    min_x, min_y, _min_z = bbox[0]
    max_x, max_y, _max_z = bbox[1]
    return max(0.0, float(max_x) - float(min_x)) * max(0.0, float(max_y) - float(min_y))


def _select_emitted_overlap_xy_candidates(
    candidates: list[_OverlapXYCandidate],
    *,
    policy: OverlapXYPolicy,
) -> tuple[set[tuple[str, str]], int, int]:
    if not candidates or policy.mode == "none":
        return set(), 0, 0
    if policy.mode == "full":
        emitted = {(item.source_id, item.target_id) for item in candidates}
        return emitted, 0, 0
    if policy.mode == "threshold":
        emitted = {
            (item.source_id, item.target_id)
            for item in candidates
            if item.overlap_ratio >= policy.min_ratio
        }
        return emitted, len(candidates) - len(emitted), 0

    ranked_by_node: dict[str, list[_OverlapXYCandidate]] = defaultdict(list)
    for item in candidates:
        ranked_by_node[item.source_id].append(item)
        ranked_by_node[item.target_id].append(item)

    kept_pairs: set[tuple[str, str]] = set()
    for node_id, node_candidates in ranked_by_node.items():
        ordered = sorted(
            node_candidates,
            key=lambda item: (
                -item.overlap_area_xy,
                item.source_id if item.source_id != node_id else item.target_id,
                item.target_id if item.source_id != node_id else item.source_id,
            ),
        )
        for item in ordered[: policy.top_k]:
            kept_pairs.add((item.source_id, item.target_id))

    emitted = {
        (item.source_id, item.target_id)
        for item in candidates
        if (item.source_id, item.target_id) in kept_pairs
    }
    return emitted, 0, len(candidates) - len(emitted)


def _record_derived_edge_phase_stats(
    G: nx.DiGraph | nx.MultiDiGraph,
    phase: str,
    *,
    excluded_classes: frozenset[str],
    eligible_nodes: int,
    skipped_by_class: dict[str, int],
    edges_added: int,
    threshold: float | None = None,
) -> None:
    graph_build = G.graph.setdefault("graph_build", {})
    pruning = graph_build.setdefault("derived_edge_pruning", {})
    phases = pruning.setdefault("phases", {})
    stats: dict[str, object] = {
        "excluded_classes": sorted(excluded_classes),
        "eligible_nodes": eligible_nodes,
        "skipped_nodes": sum(skipped_by_class.values()),
        "skipped_by_class": dict(sorted(skipped_by_class.items())),
        "edges_added": edges_added,
    }
    if threshold is not None:
        stats["threshold"] = threshold
    phases[phase] = stats

    LOG.info(
        "%s: eligible=%d skipped=%d edges_added=%d excluded=%s",
        phase,
        eligible_nodes,
        sum(skipped_by_class.values()),
        edges_added,
        ", ".join(sorted(excluded_classes)) if excluded_classes else "<none>",
    )


def _record_overlap_xy_stats(
    G: nx.DiGraph | nx.MultiDiGraph,
    *,
    policy: OverlapXYPolicy,
    candidate_pairs: int,
    emitted_pairs: int,
    rejected_by_threshold: int,
    rejected_by_top_k: int,
) -> None:
    graph_build = G.graph.setdefault("graph_build", {})
    graph_build["overlap_xy"] = {
        "mode": policy.mode,
        "min_ratio": policy.min_ratio,
        "top_k": policy.top_k,
        "candidate_positive_overlap_pairs": candidate_pairs,
        "emitted_overlap_pairs": emitted_pairs,
        "emitted_overlap_edges": emitted_pairs * 2,
        "rejected_by_threshold": rejected_by_threshold,
        "rejected_by_top_k": rejected_by_top_k,
    }
    LOG.info(
        "overlap_xy: mode=%s candidate_pairs=%d emitted_pairs=%d "
        "rejected_by_threshold=%d rejected_by_top_k=%d min_ratio=%.3f top_k=%d",
        policy.mode,
        candidate_pairs,
        emitted_pairs,
        rejected_by_threshold,
        rejected_by_top_k,
        policy.min_ratio,
        policy.top_k,
    )


def build_graph_from_jsonl(
    jsonl_paths: list[Path], payload_mode: str = "full"
) -> nx.MultiDiGraph:
    _resolved_mode = _resolve_graph_payload_mode(payload_mode)

    G = nx.MultiDiGraph()
    dataset_keys: list[str] = []
    namespaced_ids = len({_dataset_key_from_path(path) for path in jsonl_paths}) > 1

    G.graph["_payload_mode"] = _resolved_mode
    G.graph["edge_categories"] = {
        "hierarchy": ["aggregates", "contains", "contained_in"],
        "spatial": ["adjacent_to", "connected_to"],
        "topology": [
            "aligned_with",
            "intersects_bbox",
            "overlaps_xy",
            "above",
            "below",
            "intersects_3d",
            "touches_surface",
            "space_bounded_by",
            "bounds_space",
            "shares_boundary_with",
            "path_connected_to",
        ],
        # explicit IFC relationships extracted from the Relationships block (Batch 0+)
        "explicit": [
            "hosts",
            "hosted_by",
            "ifc_connected_to",
            "typed_by",
            "path_connected_to",
            "space_bounded_by",
            "bounds_space",
            "belongs_to_system",
            "in_zone",
            "classified_as",
        ],
    }
    G.graph["namespaced_ids"] = namespaced_ids

    node_id_by_gid_by_dataset: dict[str, dict[str, str]] = {}

    # we defer containment edges to a second pass because when we read element A,
    # its parent (element B) might not have been added to the graph yet
    deferred_containment: list[tuple[str, str, str]] = []

    # we also defer explicit relationships: all primary nodes must exist first so
    # that target GlobalIds can be resolved to their graph node IDs
    deferred_relationships: list[tuple[str, str, dict]] = []

    for jsonl_path in jsonl_paths:
        dataset_key = _dataset_key_from_path(jsonl_path)
        dataset_keys.append(dataset_key)
        node_id_by_gid = node_id_by_gid_by_dataset.setdefault(dataset_key, {})
        project_node_id = _project_node_id(dataset_key, namespaced=namespaced_ids)
        building_node_id = _building_node_id(dataset_key, namespaced=namespaced_ids)

        G.add_node(
            project_node_id,
            label="Project",
            class_="IfcProject",
            geometry=None,
            dataset=dataset_key,
        )
        G.add_node(
            building_node_id,
            label="Building",
            class_="IfcBuilding",
            geometry=None,
            dataset=dataset_key,
        )
        G.add_edge(project_node_id, building_node_id, relation="aggregates")

        LOG.info("Reading %s", jsonl_path)
        with jsonl_path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec: dict = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOG.warning(
                        "%s line %d: JSON error: %s", jsonl_path.name, line_no, exc
                    )
                    continue

                gid = rec.get("GlobalId")
                if not gid:
                    continue

                ifc_type = rec.get("IfcType") or rec.get("ClassRaw") or "IfcProduct"
                name = rec.get("Name") or gid
                centroid, bbox = _geom_from_record(rec)
                mesh = _mesh_from_record(rec)
                footprint_poly, obb, placement = _oriented_geom_from_record(rec)

                if ifc_type == "IfcProject":
                    node_id = project_node_id
                    G.nodes[project_node_id].update(
                        label=name,
                        class_=ifc_type,
                        payload=_make_payload(rec, _resolved_mode),
                        properties=_flat_properties(rec),
                        geometry=centroid,
                        bbox=bbox,
                        footprint_bbox_2d=(
                            (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
                            if bbox
                            else None
                        ),
                        footprint_polygon=footprint_poly,
                        obb=obb,
                        local_placement_matrix=placement,
                        dataset=dataset_key,
                    )
                elif ifc_type == "IfcBuilding":
                    node_id = building_node_id
                    G.nodes[building_node_id].update(
                        label=name,
                        class_=ifc_type,
                        payload=_make_payload(rec, _resolved_mode),
                        properties=_flat_properties(rec),
                        geometry=centroid,
                        bbox=bbox,
                        mesh=mesh,
                        footprint_bbox_2d=(
                            (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
                            if bbox
                            else None
                        ),
                        footprint_polygon=footprint_poly,
                        obb=obb,
                        local_placement_matrix=placement,
                        dataset=dataset_key,
                    )
                elif ifc_type == "IfcBuildingStorey":
                    node_id = _storey_node_id(
                        dataset_key,
                        gid,
                        namespaced=namespaced_ids,
                    )
                    G.add_node(
                        node_id,
                        label=name,
                        class_=ifc_type,
                        properties=_flat_properties(rec),
                        geometry=centroid,
                        bbox=bbox,
                        mesh=mesh,
                        footprint_bbox_2d=(
                            (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
                            if bbox
                            else None
                        ),
                        footprint_polygon=footprint_poly,
                        obb=obb,
                        local_placement_matrix=placement,
                        z_min=(bbox[0][2] if bbox else None),
                        z_max=(bbox[1][2] if bbox else None),
                        height=((bbox[1][2] - bbox[0][2]) if bbox else None),
                        payload=_make_payload(rec, _resolved_mode),
                        dataset=dataset_key,
                    )
                    G.add_edge(building_node_id, node_id, relation="aggregates")
                else:
                    node_id = _element_node_id(
                        dataset_key,
                        gid,
                        namespaced=namespaced_ids,
                    )
                    G.add_node(
                        node_id,
                        label=name,
                        class_=ifc_type,
                        properties=_flat_properties(rec),
                        geometry=centroid,
                        bbox=bbox,
                        mesh=mesh,
                        footprint_polygon=footprint_poly,
                        obb=obb,
                        local_placement_matrix=placement,
                        z_min=(bbox[0][2] if bbox else None),
                        z_max=(bbox[1][2] if bbox else None),
                        height=((bbox[1][2] - bbox[0][2]) if bbox else None),
                        footprint_bbox_2d=(
                            (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
                            if bbox
                            else None
                        ),
                        payload=_make_payload(rec, _resolved_mode),
                        dataset=dataset_key,
                    )

                node_id_by_gid[gid] = node_id

                parent_gid = (rec.get("Hierarchy") or {}).get("ParentId")
                if parent_gid:
                    deferred_containment.append((dataset_key, parent_gid, node_id))

                # collect non-empty Relationships blocks for the third pass
                rels = rec.get("Relationships")
                if isinstance(rels, dict) and any(rels.values()):
                    deferred_relationships.append((dataset_key, node_id, rels))

    # second pass â€” now all nodes exist so we can safely add containment edges
    for dataset_key, parent_gid, child_node_id in deferred_containment:
        parent_node_id = node_id_by_gid_by_dataset.get(dataset_key, {}).get(parent_gid)
        if parent_node_id is None:
            continue
        _add_containment_edge(G, parent_node_id, child_node_id)

    # third pass â€” materialise explicit IFC relationships from Relationships blocks
    explicit_edge_count = 0
    _before = G.number_of_edges()
    for dataset_key, node_id, rels in deferred_relationships:
        _add_explicit_relationships(
            G,
            node_id,
            rels,
            node_id_by_gid_by_dataset.get(dataset_key, {}),
            dataset_key=dataset_key,
            namespaced_ids=namespaced_ids,
        )
    explicit_edge_count = G.number_of_edges() - _before
    if explicit_edge_count:
        LOG.info("Added %d explicit IFC relationship edge(s)", explicit_edge_count)
    shared_boundary_edge_count = _add_shared_space_boundary_edges(G)
    if shared_boundary_edge_count:
        LOG.info(
            "Added %d shared space boundary edge(s)",
            shared_boundary_edge_count,
        )

    G.graph["datasets"] = sorted(set(dataset_keys))

    LOG.info(
        "Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


def add_spatial_adjacency(
    G: nx.DiGraph | nx.MultiDiGraph,
    threshold: float | None = None,
    *,
    exclude_classes: object = None,
) -> float:
    element_nodes: list[str] = []
    positions: list[tuple] = []
    bboxes: list[tuple | None] = []
    meshes: list[
        tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None
    ] = []
    element_refs: list[tuple[str, str, str] | None] = []
    excluded_classes = _normalize_excluded_classes(exclude_classes)
    skipped_by_class: defaultdict[str, int] = defaultdict(int)
    edges_before = G.number_of_edges()

    for n, d in G.nodes(data=True):
        if d.get("class_") in {"IfcBuilding", "IfcProject", "IfcBuildingStorey"}:
            continue
        if not n.startswith("Element::"):
            continue
        class_name = _normalize_ifc_class_name(d.get("class_"))
        if class_name in excluded_classes:
            skipped_by_class[class_name] += 1
            continue
        centroid = d.get("geometry")
        bbox = d.get("bbox")
        # fall back to bbox center if we don't have a centroid
        if centroid is None and bbox is not None:
            mn, mx = bbox
            centroid = tuple((mn[i] + mx[i]) / 2 for i in range(3))
        if centroid is None:
            continue
        element_nodes.append(n)
        positions.append(centroid)
        bboxes.append(bbox)
        meshes.append(d.get("mesh"))
        gid = (d.get("properties") or {}).get("GlobalId")
        dataset_key = d.get("dataset")
        if isinstance(gid, str) and isinstance(dataset_key, str):
            element_refs.append((n, dataset_key, gid))
        else:
            element_refs.append(None)

    if threshold is None:
        threshold = compute_adjacency_threshold(positions)

    pos = _normalize_positions(positions)
    if pos is None:
        _record_derived_edge_phase_stats(
            G,
            "spatial_adjacency",
            excluded_classes=excluded_classes,
            eligible_nodes=len(element_nodes),
            skipped_by_class=dict(skipped_by_class),
            edges_added=G.number_of_edges() - edges_before,
            threshold=threshold,
        )
        return threshold

    candidate_pairs = _candidate_pairs_kdtree(pos, float(threshold))
    if not candidate_pairs:
        # Fallback to the legacy grid strategy when KD candidates are empty.
        cell_size = max(float(threshold), 1e-6)
        grid = _build_spatial_grid(pos, cell_size)
        seen_pairs: set[tuple[int, int]] = set()
        neighbor_radius = int(math.ceil(float(threshold) / cell_size)) + 1
        for i, centroid_a in enumerate(pos):
            key = _cell_for_point(centroid_a, cell_size)
            for radius in range(neighbor_radius + 1):
                for neighbor_key in _neighbor_cell_keys(key, radius):
                    for j in grid.get(neighbor_key, []):
                        if j == i:
                            continue
                        pair = (i, j) if i < j else (j, i)
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
        candidate_pairs = sorted(seen_pairs)

    occ_shape_by_node_id = _get_occ_shape_index(
        G,
        [ref for ref in element_refs if ref is not None],
    )

    for i, j in candidate_pairs:
        node_a = element_nodes[i]
        node_b = element_nodes[j]
        centroid_a = pos[i]
        bbox_a = bboxes[i]
        centroid_b = pos[j]
        bbox_b = bboxes[j]
        mesh_a = meshes[i]
        mesh_b = meshes[j]
        element_ref_a = element_refs[i]
        element_ref_b = element_refs[j]

        distance_method = "centroid"
        d: float

        # Primary: exact OCC surface-to-surface minimum distance.
        shape_distance: float | None = None
        shape_a = None
        shape_b = None
        if element_ref_a is not None and element_ref_b is not None:
            shape_a = occ_shape_by_node_id.get(node_a)
            shape_b = occ_shape_by_node_id.get(node_b)
            if shape_a is not None and shape_b is not None:
                shape_distance = _compute_shape_min_distance(
                    shape_a,
                    shape_b,
                )

        if shape_distance is not None:
            d = shape_distance
            distance_method = "occ_surface"
        else:
            mesh_distance = _mesh_min_vertex_distance(mesh_a, mesh_b)
            if mesh_distance is not None:
                d = mesh_distance
                distance_method = "mesh_vertices"
            elif bbox_a is not None and bbox_b is not None:
                d = distance_between_bboxes(bbox_a, bbox_b)
                distance_method = "bbox"
            else:
                d = distance_between_points(centroid_a, centroid_b)
                distance_method = "centroid"

        if d <= threshold:
            # Stage 1: geometric candidate (distance threshold).
            # Stage 2: exact OCC verification when shapes exist.
            relation = "connected_to" if d <= 1e-9 else "adjacent_to"
            edge_attrs: dict[str, object] = {
                "distance": d,
                "distance_method": distance_method,
                "source": "heuristic",
            }
            if element_ref_a is not None and element_ref_b is not None:
                if shape_a is not None and shape_b is not None:
                    metrics = _compute_common_metrics(shape_a, shape_b)
                    if metrics is not None:
                        intersection_volume, contact_area = metrics
                        edge_attrs["intersection_volume"] = intersection_volume
                        edge_attrs["contact_area"] = contact_area
                        edge_attrs["verified"] = True
                        if intersection_volume > 1e-9 or contact_area > 1e-6:
                            relation = "connected_to"
                        else:
                            relation = "adjacent_to"
                        edge_attrs["verified"] = True
                    elif distance_method == "occ_surface":
                        edge_attrs["verified"] = True
                else:
                    edge_attrs["verified"] = False
            else:
                edge_attrs["verified"] = False

            _add_symmetric_edge_once(
                G,
                node_a,
                node_b,
                relation=relation,
                **edge_attrs,
            )

    _record_derived_edge_phase_stats(
        G,
        "spatial_adjacency",
        excluded_classes=excluded_classes,
        eligible_nodes=len(element_nodes),
        skipped_by_class=dict(skipped_by_class),
        edges_added=G.number_of_edges() - edges_before,
        threshold=threshold,
    )
    return threshold


def add_topology_facts(
    G: nx.DiGraph | nx.MultiDiGraph,
    *,
    exclude_classes: object = None,
    overlap_xy_mode: object | None = None,
    overlap_xy_min_ratio: object | None = None,
    overlap_xy_top_k: object | None = None,
) -> None:
    def _add_topology_edge(u: str, v: str, **attrs: object) -> None:
        """Add topology edge with explicit topology source semantics."""
        relation = attrs.get("relation")
        edge_attrs = {"source": "topology", **attrs}
        if is_symmetric_relation(relation):
            _add_symmetric_edge_once(G, u, v, **edge_attrs)
            return
        _add_edge_once(G, u, v, **edge_attrs)

    element_nodes: list[str] = []
    element_bboxes: list[tuple] = []
    element_refs: list[tuple[str, str, str]] = []
    element_footprints: list[list[tuple[float, float]] | None] = []
    excluded_classes = _normalize_excluded_classes(exclude_classes)
    overlap_xy_policy = _resolve_overlap_xy_policy(
        mode=overlap_xy_mode,
        min_ratio=overlap_xy_min_ratio,
        top_k=overlap_xy_top_k,
    )
    skipped_by_class: defaultdict[str, int] = defaultdict(int)
    edges_before = G.number_of_edges()
    overlap_candidates: list[_OverlapXYCandidate] = []

    for n, d in G.nodes(data=True):
        if not str(n).startswith("Element::"):
            continue
        class_name = _normalize_ifc_class_name(d.get("class_"))
        if class_name in excluded_classes:
            skipped_by_class[class_name] += 1
            continue
        bbox = d.get("bbox")
        if bbox is None:
            continue
        gid = (d.get("properties") or {}).get("GlobalId")
        dataset_key = d.get("dataset")
        if not isinstance(gid, str) or not gid or not isinstance(dataset_key, str):
            continue
        element_nodes.append(n)
        element_bboxes.append(bbox)
        element_refs.append((n, dataset_key, gid))
        footprint = d.get("footprint_polygon")
        if isinstance(footprint, list) and len(footprint) >= 3:
            parsed = []
            for p in footprint:
                if isinstance(p, tuple) and len(p) == 2:
                    parsed.append((float(p[0]), float(p[1])))
                elif isinstance(p, list) and len(p) == 2:
                    parsed.append((float(p[0]), float(p[1])))
            element_footprints.append(parsed if len(parsed) >= 3 else None)
        else:
            element_footprints.append(None)

    occ_shape_by_node_id = _get_occ_shape_index(G, element_refs)

    for i, a in enumerate(element_nodes):
        bbox_a = element_bboxes[i]
        node_ref_a = element_refs[i]
        for j in range(i + 1, len(element_nodes)):
            b = element_nodes[j]
            bbox_b = element_bboxes[j]
            node_ref_b = element_refs[j]
            footprint_a = element_footprints[i]
            footprint_b = element_footprints[j]

            intersects_bbox = _bboxes_intersect(bbox_a, bbox_b)
            if intersects_bbox:
                _add_topology_edge(a, b, relation="intersects_bbox")
                _add_topology_edge(b, a, relation="intersects_bbox")

            if footprint_a is not None and footprint_b is not None:
                overlap_area = _convex_polygon_intersection_area(
                    footprint_a,
                    footprint_b,
                )
            else:
                overlap_area = _bbox_xy_overlap_area(bbox_a, bbox_b)
            if overlap_area > 0.0:
                min_area = min(
                    _footprint_area_xy(footprint_a, bbox_a),
                    _footprint_area_xy(footprint_b, bbox_b),
                )
                overlap_ratio = overlap_area / min_area if min_area > 0.0 else 0.0
                overlap_candidates.append(
                    _OverlapXYCandidate(
                        source_id=a,
                        target_id=b,
                        overlap_area_xy=overlap_area,
                        overlap_ratio=overlap_ratio,
                    )
                )

            # Prefer exact OCC mesh boolean metrics when available.
            shape_a = occ_shape_by_node_id.get(node_ref_a[0])
            shape_b = occ_shape_by_node_id.get(node_ref_b[0])
            if shape_a is not None and shape_b is not None:
                metrics = _compute_common_metrics(shape_a, shape_b)
                if metrics is not None:
                    intersection_volume, contact_area = metrics
                    if intersection_volume > 1e-9:
                        _add_topology_edge(
                            a,
                            b,
                            relation="intersects_3d",
                            intersection_volume=intersection_volume,
                            contact_area=contact_area,
                        )
                        _add_topology_edge(
                            b,
                            a,
                            relation="intersects_3d",
                            intersection_volume=intersection_volume,
                            contact_area=contact_area,
                        )
                    elif contact_area > 1e-6:
                        _add_topology_edge(
                            a,
                            b,
                            relation="touches_surface",
                            intersection_volume=intersection_volume,
                            contact_area=contact_area,
                        )
                        _add_topology_edge(
                            b,
                            a,
                            relation="touches_surface",
                            intersection_volume=intersection_volume,
                            contact_area=contact_area,
                        )
            # Only check vertical order if footprints overlap.
            if overlap_area <= 0.0:
                continue
            a_min_z, a_max_z = float(bbox_a[0][2]), float(bbox_a[1][2])
            b_min_z, b_max_z = float(bbox_b[0][2]), float(bbox_b[1][2])
            if a_min_z > b_max_z:
                gap = a_min_z - b_max_z
                _add_topology_edge(a, b, relation="above", vertical_gap=gap)
                _add_topology_edge(b, a, relation="below", vertical_gap=gap)
            elif b_min_z > a_max_z:
                gap = b_min_z - a_max_z
                _add_topology_edge(b, a, relation="above", vertical_gap=gap)
                _add_topology_edge(a, b, relation="below", vertical_gap=gap)

    emitted_overlap_pairs, rejected_by_threshold, rejected_by_top_k = (
        _select_emitted_overlap_xy_candidates(
            overlap_candidates,
            policy=overlap_xy_policy,
        )
    )
    for candidate in overlap_candidates:
        if (candidate.source_id, candidate.target_id) not in emitted_overlap_pairs:
            continue
        _add_topology_edge(
            candidate.source_id,
            candidate.target_id,
            relation="overlaps_xy",
            overlap_area_xy=candidate.overlap_area_xy,
            overlap_ratio_xy=candidate.overlap_ratio,
        )
        _add_topology_edge(
            candidate.target_id,
            candidate.source_id,
            relation="overlaps_xy",
            overlap_area_xy=candidate.overlap_area_xy,
            overlap_ratio_xy=candidate.overlap_ratio,
        )

    _record_overlap_xy_stats(
        G,
        policy=overlap_xy_policy,
        candidate_pairs=len(overlap_candidates),
        emitted_pairs=len(emitted_overlap_pairs),
        rejected_by_threshold=rejected_by_threshold,
        rejected_by_top_k=rejected_by_top_k,
    )

    _record_derived_edge_phase_stats(
        G,
        "topology",
        excluded_classes=excluded_classes,
        eligible_nodes=len(element_nodes),
        skipped_by_class=dict(skipped_by_class),
        edges_added=G.number_of_edges() - edges_before,
    )


def _iter_edge_attrs_between(
    G: nx.DiGraph | nx.MultiDiGraph,
    source_id: str,
    target_id: str,
) -> list[dict[str, object]]:
    if not G.has_edge(source_id, target_id):
        return []
    edge_data = G.get_edge_data(source_id, target_id)
    if isinstance(edge_data, dict):
        relation = edge_data.get("relation")
        if isinstance(relation, str):
            return [edge_data]
        return [
            attrs
            for attrs in edge_data.values()
            if isinstance(attrs, dict) and isinstance(attrs.get("relation"), str)
        ]
    return []


def _viz_node_class_name(node_data: dict[str, object]) -> str | None:
    return _normalize_ifc_class_name(node_data.get("class_"))


def _viz_node_dataset(node_data: dict[str, object]) -> str | None:
    dataset = node_data.get("dataset")
    return dataset if isinstance(dataset, str) and dataset else None


def _viz_node_is_container(node_data: dict[str, object]) -> bool:
    class_name = _viz_node_class_name(node_data)
    return class_name in _VIZ_CONTAINER_CLASSES


def _viz_normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _viz_non_container_descendants(
    G: nx.DiGraph | nx.MultiDiGraph,
    root_id: str,
) -> list[str]:
    descendants: set[str] = set()
    queue: deque[str] = deque([root_id])
    visited = {root_id}

    while queue:
        current = queue.popleft()
        for child_id in G.successors(current):
            child_text = str(child_id)
            if child_text in visited:
                continue
            if not any(
                str(attrs.get("relation", "")).strip().lower() == "contains"
                for attrs in _iter_edge_attrs_between(G, current, child_text)
            ):
                continue
            visited.add(child_text)
            child_data = G.nodes[child_text]
            if _viz_node_is_container(child_data):
                queue.append(child_text)
            else:
                descendants.add(child_text)
    return sorted(descendants)


def _viz_resolve_storey_for_node(
    G: nx.DiGraph | nx.MultiDiGraph,
    node_id: str,
) -> tuple[str | None, str | None]:
    node_data = G.nodes[node_id]
    if _viz_node_class_name(node_data) == "IfcBuildingStorey":
        return node_id, "self"

    queue: deque[str] = deque([node_id])
    visited = {node_id}
    while queue:
        current = queue.popleft()
        for parent_id in G.successors(current):
            parent_text = str(parent_id)
            if parent_text in visited:
                continue
            if not any(
                str(attrs.get("relation", "")).strip().lower() == "contained_in"
                for attrs in _iter_edge_attrs_between(G, current, parent_text)
            ):
                continue
            if _viz_node_class_name(G.nodes[parent_text]) == "IfcBuildingStorey":
                return parent_text, "contained_in"
            visited.add(parent_text)
            queue.append(parent_text)

        for parent_id in G.predecessors(current):
            parent_text = str(parent_id)
            if parent_text in visited:
                continue
            if not any(
                str(attrs.get("relation", "")).strip().lower() == "contains"
                for attrs in _iter_edge_attrs_between(G, parent_text, current)
            ):
                continue
            if _viz_node_class_name(G.nodes[parent_text]) == "IfcBuildingStorey":
                return parent_text, "contains"
            visited.add(parent_text)
            queue.append(parent_text)

    props = node_data.get("properties") or {}
    if not isinstance(props, dict):
        return None, None
    level_value = props.get("Level")
    if not isinstance(level_value, str) or not level_value.strip():
        return None, None

    normalized_level = _viz_normalize_text(level_value)
    dataset_key = _viz_node_dataset(node_data)
    matches = [
        candidate_id
        for candidate_id, candidate_data in G.nodes(data=True)
        if _viz_node_class_name(candidate_data) == "IfcBuildingStorey"
        and (dataset_key is None or _viz_node_dataset(candidate_data) == dataset_key)
        and _viz_normalize_text(str(candidate_data.get("label", "")))
        == normalized_level
    ]
    if len(matches) == 1:
        return str(matches[0]), "level"
    return None, None


def _viz_node_plan_point(
    node_data: dict[str, object],
) -> tuple[tuple[float, float] | None, str | None]:
    geometry = node_data.get("geometry")
    if isinstance(geometry, (list, tuple)) and len(geometry) >= 2:
        try:
            return (float(geometry[0]), float(geometry[1])), "centroid"
        except (TypeError, ValueError):
            pass

    bbox = node_data.get("bbox")
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 2
        and isinstance(bbox[0], (list, tuple))
        and isinstance(bbox[1], (list, tuple))
        and len(bbox[0]) >= 2
        and len(bbox[1]) >= 2
    ):
        try:
            return (
                (
                    (float(bbox[0][0]) + float(bbox[1][0])) * 0.5,
                    (float(bbox[0][1]) + float(bbox[1][1])) * 0.5,
                ),
                "bbox_center_fallback",
            )
        except (TypeError, ValueError):
            return None, None
    return None, None


def _viz_point_in_polygon_xy(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    x, y = point
    inside = False
    epsilon = 1e-9

    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        dx = x2 - x1
        dy = y2 - y1
        cross = (x - x1) * dy - (y - y1) * dx
        if abs(cross) <= epsilon:
            dot = (x - x1) * (x - x2) + (y - y1) * (y - y2)
            if dot <= epsilon:
                return True

        intersects = ((y1 > y) != (y2 > y)) and (
            x <= ((x2 - x1) * (y - y1) / ((y2 - y1) or epsilon)) + x1 + epsilon
        )
        if intersects:
            inside = not inside
    return inside


def _viz_point_in_bbox_2d(
    point: tuple[float, float],
    bbox_2d: tuple[float, float, float, float],
) -> bool:
    x, y = point
    min_x, min_y, max_x, max_y = bbox_2d
    epsilon = 1e-9
    return (
        min_x - epsilon <= x <= max_x + epsilon
        and min_y - epsilon <= y <= max_y + epsilon
    )


def _viz_node_footprint_geometry(
    node_data: dict[str, object],
) -> tuple[list[tuple[float, float]] | None, tuple[float, float, float, float] | None]:
    footprint = node_data.get("footprint_polygon")
    if isinstance(footprint, list) and len(footprint) >= 3:
        polygon: list[tuple[float, float]] = []
        for point in footprint:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                polygon.append((float(point[0]), float(point[1])))
        if len(polygon) >= 3:
            return polygon, None

    bbox_2d = node_data.get("footprint_bbox_2d")
    if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) == 4:
        try:
            return None, tuple(float(value) for value in bbox_2d)
        except (TypeError, ValueError):
            return None, None
    return None, None


def _viz_aligned_signature(node_data: dict[str, object]) -> dict[str, object] | None:
    if _viz_node_class_name(node_data) not in _VIZ_ALIGNED_WITH_ALLOWED_CLASSES:
        return None

    obb = node_data.get("obb")
    if not isinstance(obb, dict):
        return None
    axes = obb.get("axes")
    extents = obb.get("extents")
    center = obb.get("center")
    if (
        not isinstance(axes, (list, tuple))
        or len(axes) < 2
        or not isinstance(extents, (list, tuple))
        or len(extents) < 2
        or not isinstance(center, (list, tuple))
        or len(center) < 2
    ):
        return None

    axis_candidates: list[tuple[float, tuple[float, float], float]] = []
    short_extents: list[float] = []
    for raw_axis, raw_extent in zip(axes[:2], extents[:2], strict=False):
        if not isinstance(raw_axis, (list, tuple)) or len(raw_axis) < 2:
            return None
        try:
            axis_x = float(raw_axis[0])
            axis_y = float(raw_axis[1])
            extent = abs(float(raw_extent))
        except (TypeError, ValueError):
            return None
        axis_len = math.hypot(axis_x, axis_y)
        if axis_len <= 1e-9 or extent <= 1e-9:
            continue
        normalized_axis = (axis_x / axis_len, axis_y / axis_len)
        axis_candidates.append((extent * axis_len, normalized_axis, extent))
        short_extents.append(extent)

    if len(axis_candidates) < 2:
        return None

    axis_candidates.sort(key=lambda item: item[0], reverse=True)
    try:
        center_xy = (float(center[0]), float(center[1]))
    except (TypeError, ValueError):
        return None

    return {
        "axis": axis_candidates[0][1],
        "center_xy": center_xy,
        "short_extent": min(short_extents),
    }


def _viz_collect_overlay_edges(
    G: nx.DiGraph | nx.MultiDiGraph,
) -> dict[str, list[tuple[str, str, dict[str, object]]]]:
    overlays: dict[str, list[tuple[str, str, dict[str, object]]]] = {
        "aligned_with": [],
        "inside_footprint_of": [],
        "same_storey_as": [],
    }

    aligned_by_dataset: dict[str | None, list[tuple[str, dict[str, object]]]] = (
        defaultdict(list)
    )
    for node_id, node_data in G.nodes(data=True):
        node_text = str(node_id)
        if not node_text.startswith("Element::"):
            continue
        signature = _viz_aligned_signature(node_data)
        if signature is None:
            continue
        aligned_by_dataset[_viz_node_dataset(node_data)].append((node_text, signature))

    for dataset_items in aligned_by_dataset.values():
        dataset_items.sort(key=lambda item: item[0])
        for index, (source_id, source_signature) in enumerate(dataset_items):
            source_storey_id, _ = _viz_resolve_storey_for_node(G, source_id)
            source_axis_x, source_axis_y = source_signature["axis"]  # type: ignore[index]
            source_center_x, source_center_y = source_signature["center_xy"]  # type: ignore[index]
            source_short_extent = float(source_signature["short_extent"])  # type: ignore[arg-type]
            selected_edges: list[
                tuple[tuple[int, float, float, str], str, dict[str, object]]
            ] = []
            for target_id, target_signature in dataset_items[index + 1 :]:
                target_axis_x, target_axis_y = target_signature["axis"]  # type: ignore[index]
                dot = source_axis_x * target_axis_x + source_axis_y * target_axis_y
                if dot < 0.0:
                    target_axis_x *= -1.0
                    target_axis_y *= -1.0
                    dot *= -1.0
                dot = max(-1.0, min(1.0, dot))
                angle_deg = math.degrees(math.acos(dot))
                if angle_deg > 10.0:
                    continue

                blend_x = source_axis_x + target_axis_x
                blend_y = source_axis_y + target_axis_y
                blend_len = math.hypot(blend_x, blend_y)
                if blend_len <= 1e-9:
                    ref_axis_x, ref_axis_y = source_axis_x, source_axis_y
                else:
                    ref_axis_x = blend_x / blend_len
                    ref_axis_y = blend_y / blend_len

                target_center_x, target_center_y = target_signature["center_xy"]  # type: ignore[index]
                perp_x, perp_y = -ref_axis_y, ref_axis_x
                lateral_offset_xy = abs(
                    (target_center_x - source_center_x) * perp_x
                    + (target_center_y - source_center_y) * perp_y
                )
                target_short_extent = float(target_signature["short_extent"])  # type: ignore[arg-type]
                offset_limit = max(0.25, min(source_short_extent, target_short_extent))
                if lateral_offset_xy > offset_limit:
                    continue

                target_storey_id, _ = _viz_resolve_storey_for_node(G, target_id)
                same_storey = (
                    source_storey_id is not None
                    and source_storey_id == target_storey_id
                )
                selected_edges.append(
                    (
                        (
                            0 if same_storey else 1,
                            round(angle_deg, 6),
                            round(lateral_offset_xy, 6),
                            target_id,
                        ),
                        target_id,
                        {
                            "relation": "aligned_with",
                            "source": "visualization_overlay",
                            "angle_deg": round(angle_deg, 6),
                            "lateral_offset_xy": round(lateral_offset_xy, 6),
                            "same_storey": same_storey,
                        },
                    )
                )
                selected_edges.sort(key=lambda item: item[0])
                if len(selected_edges) > _VIZ_ALIGNED_WITH_MAX_PER_NODE:
                    selected_edges.pop()

            for _sort_key, target_id, attrs in selected_edges:
                overlays["aligned_with"].append((source_id, target_id, attrs))

    for node_id, node_data in G.nodes(data=True):
        node_text = str(node_id)
        if _viz_node_class_name(node_data) != "IfcSpace":
            continue
        footprint_polygon, footprint_bbox_2d = _viz_node_footprint_geometry(node_data)
        if footprint_polygon is None and footprint_bbox_2d is None:
            continue

        for candidate_id in _viz_non_container_descendants(G, node_text):
            if candidate_id == node_text:
                continue
            candidate_data = G.nodes[candidate_id]
            point_xy, inside_method = _viz_node_plan_point(candidate_data)
            if point_xy is None or inside_method is None:
                continue
            is_inside = (
                _viz_point_in_polygon_xy(point_xy, footprint_polygon)
                if footprint_polygon is not None
                else _viz_point_in_bbox_2d(point_xy, footprint_bbox_2d)
            )
            if not is_inside:
                continue
            storey_id, _ = _viz_resolve_storey_for_node(G, candidate_id)
            overlays["inside_footprint_of"].append(
                (
                    node_text,
                    candidate_id,
                    {
                        "relation": "inside_footprint_of",
                        "source": "visualization_overlay",
                        "inside_method": inside_method,
                        "storey_id": storey_id,
                    },
                )
            )

    for node_id, node_data in G.nodes(data=True):
        node_text = str(node_id)
        if _viz_node_class_name(node_data) != "IfcBuildingStorey":
            continue
        for candidate_id in _viz_non_container_descendants(G, node_text):
            if candidate_id == node_text:
                continue
            overlays["same_storey_as"].append(
                (
                    node_text,
                    candidate_id,
                    {
                        "relation": "same_storey_as",
                        "source": "visualization_overlay",
                        "storey_id": node_text,
                    },
                )
            )

    return {relation: edges for relation, edges in overlays.items() if edges}


def _viz_relation_count_key(
    source_id: str,
    target_id: str,
    relation: str,
) -> tuple[str, str]:
    if is_symmetric_relation(relation):
        return canonicalize_undirected_edge_endpoints(source_id, target_id)
    return str(source_id), str(target_id)


def _viz_relation_counts(
    edge_items: list[tuple[str, str, dict[str, object]]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    seen_by_relation: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for source_id, target_id, attrs in edge_items:
        relation = str(attrs.get("relation", "related_to"))
        count_key = _viz_relation_count_key(source_id, target_id, relation)
        if count_key in seen_by_relation[relation]:
            continue
        seen_by_relation[relation].add(count_key)
        counts[relation] += 1
    return counts


@dataclass(frozen=True)
class _VizInverseLegendPair:
    forward_relation: str
    reverse_relation: str
    label: str
    summary: str


@dataclass(frozen=True)
class _VizLegendEntry:
    label: str
    subtitle: str
    count: int
    swatch: str


_VIZ_INVERSE_LEGEND_PAIRS: tuple[_VizInverseLegendPair, ...] = (
    _VizInverseLegendPair(
        forward_relation="above",
        reverse_relation="below",
        label="above / below",
        summary="vertical ordering",
    ),
    _VizInverseLegendPair(
        forward_relation="contains",
        reverse_relation="contained_in",
        label="contains / contained_in",
        summary="containment hierarchy",
    ),
    _VizInverseLegendPair(
        forward_relation="hosts",
        reverse_relation="hosted_by",
        label="hosts / hosted_by",
        summary="explicit IFC host relationship",
    ),
    _VizInverseLegendPair(
        forward_relation="space_bounded_by",
        reverse_relation="bounds_space",
        label="space_bounded_by / bounds_space",
        summary="IFC space boundary",
    ),
)


def _viz_inverse_pair_count(
    edge_items: list[tuple[str, str, dict[str, object]]],
    forward_relation: str,
    reverse_relation: str,
) -> int:
    pair_keys: set[tuple[str, str]] = set()
    relations = {forward_relation, reverse_relation}
    for source_id, target_id, attrs in edge_items:
        relation = str(attrs.get("relation", "related_to"))
        if relation not in relations:
            continue
        pair_keys.add(canonicalize_undirected_edge_endpoints(source_id, target_id))
    return len(pair_keys)


def _viz_build_legend_entries(
    edge_items: list[tuple[str, str, dict[str, object]]],
    *,
    edge_color_map: dict[str, str],
    edge_relation_explanations: dict[str, str],
) -> tuple[list[_VizLegendEntry], int]:
    relation_counts = _viz_relation_counts(edge_items)
    entries: list[_VizLegendEntry] = []
    consumed_relations: set[str] = set()

    for spec in _VIZ_INVERSE_LEGEND_PAIRS:
        forward_count = relation_counts.get(spec.forward_relation, 0)
        reverse_count = relation_counts.get(spec.reverse_relation, 0)
        if forward_count == 0 and reverse_count == 0:
            continue
        consumed_relations.update({spec.forward_relation, spec.reverse_relation})
        pair_count = _viz_inverse_pair_count(
            edge_items,
            spec.forward_relation,
            spec.reverse_relation,
        )
        forward_color = edge_color_map.get(spec.forward_relation, "#4b5563")
        reverse_color = edge_color_map.get(spec.reverse_relation, forward_color)
        swatch = (
            forward_color
            if forward_color == reverse_color
            else (
                "linear-gradient(90deg, "
                f"{forward_color} 0 50%, "
                f"{reverse_color} 50% 100%)"
            )
        )
        subtitle = (
            f"{spec.summary}; "
            f"source->target = {spec.forward_relation}, "
            f"target->source = {spec.reverse_relation}; "
            f"{spec.forward_relation}={forward_count}, "
            f"{spec.reverse_relation}={reverse_count}"
        )
        entries.append(
            _VizLegendEntry(
                label=spec.label,
                subtitle=subtitle,
                count=pair_count,
                swatch=swatch,
            )
        )

    for relation, count in relation_counts.items():
        if relation in consumed_relations:
            continue
        entries.append(
            _VizLegendEntry(
                label=relation,
                subtitle=edge_relation_explanations.get(relation, "graph relation"),
                count=count,
                swatch=edge_color_map.get(relation, "#4b5563"),
            )
        )

    entries.sort(key=lambda item: (-item.count, item.label))
    return entries, sum(item.count for item in entries)


def _viz_render_legend_entries(
    entries: list[_VizLegendEntry],
    *,
    prefix: str,
    count_title: str,
) -> str:
    escaped_prefix = html.escape(prefix)
    escaped_count_title = html.escape(count_title)
    rendered: list[str] = []
    for entry in entries:
        rendered.append(
            "<div class='legend-item'>"
            f"<span class='swatch line' style='--swatch:{entry.swatch}'></span>"
            "<span class='legend-item-text'>"
            "<span class='legend-item-main'>"
            f"{escaped_prefix}: {html.escape(entry.label)}"
            "</span>"
            f"<span class='legend-item-sub'>{html.escape(entry.subtitle)}</span>"
            "</span>"
            "<span class='legend-count' "
            f"title='{escaped_count_title}'>{entry.count}</span>"
            "</div>"
        )
    return "".join(rendered)


def plot_interactive_graph(G: nx.DiGraph | nx.MultiDiGraph, out_html: Path) -> None:
    # Build positions from node geometry where available.
    pos: dict[str, tuple[float, float, float] | None] = {}
    for n, d in G.nodes(data=True):
        geom = d.get("geometry")
        if geom is not None:
            pos[n] = tuple(float(v) for v in geom)
        else:
            pos[n] = None

    # Place nodes without geometry at centroid of their children, else origin.
    for n in G.nodes:
        if pos.get(n) is not None:
            continue
        child_positions = [pos[c] for c in G.successors(n) if pos.get(c) is not None]
        if child_positions:
            arr = np.array(child_positions, dtype=float)
            pos[n] = tuple(float(v) for v in arr.mean(axis=0))
        else:
            pos[n] = (0.0, 0.0, 0.0)

    node_color_map = {
        "IfcProject": "#7c3aed",
        "IfcBuilding": "#2563eb",
        "IfcBuildingStorey": "#f97316",
        "IfcSystem": "#0891b2",
        "IfcZone": "#16a34a",
    }
    edge_color_map = {
        "aggregates": "#6b7280",
        "contains": "#1d4ed8",
        "contained_in": "#2563eb",
        "typed_by": "#ca8a04",
        "connected_to": "#ef4444",
        "adjacent_to": "#059669",
        "intersects_bbox": "#f59e0b",
        "aligned_with": "#0f766e",
        "inside_footprint_of": "#c2410c",
        "overlaps_xy": "#0ea5e9",
        "intersects_3d": "#b91c1c",
        "touches_surface": "#9333ea",
        "above": "#7c3aed",
        "below": "#9333ea",
        "same_storey_as": "#1d4ed8",
        "shares_boundary_with": "#be185d",
        "hosts": "#b91c1c",
        "hosted_by": "#dc2626",
        "ifc_connected_to": "#0f766e",
        "path_connected_to": "#0b6e4f",
        "space_bounded_by": "#be123c",
        "bounds_space": "#9f1239",
        "belongs_to_system": "#0369a1",
        "in_zone": "#15803d",
        "classified_as": "#4f46e5",
    }
    edge_relation_explanations = {
        "aggregates": "hierarchy decomposition",
        "contains": "container to child",
        "contained_in": "child to container",
        "typed_by": "instance classified by type",
        "connected_to": "bbox intersects or touches",
        "adjacent_to": "spatially near within threshold",
        "intersects_bbox": "3D bbox overlap",
        "aligned_with": "orientation-aligned in plan",
        "inside_footprint_of": "footprint containment in plan",
        "overlaps_xy": "2D footprint overlap",
        "intersects_3d": "mesh-informed 3D intersection",
        "touches_surface": "mesh-informed surface contact",
        "above": "vertical ordering above",
        "below": "vertical ordering below",
        "same_storey_as": "same-storey scope relation",
        "shares_boundary_with": "spaces share explicit boundary elements",
        "hosts": "explicit IFC host relationship",
        "hosted_by": "explicit IFC hosted-by relationship",
        "ifc_connected_to": "explicit IFC connectivity",
        "path_connected_to": "explicit IFC path connectivity",
        "space_bounded_by": "IFC space boundary (space to element)",
        "bounds_space": "IFC space boundary (element to space)",
        "belongs_to_system": "assigned to IFC system",
        "in_zone": "assigned to IFC zone",
        "classified_as": "assigned IFC classification",
    }

    node_groups: dict[str, dict[str, list]] = {}
    node_group_colors: dict[str, str] = {}
    node_group_labels: dict[str, str] = {}
    for n, d in G.nodes(data=True):
        x, y, z = pos[n] or (0.0, 0.0, 0.0)
        cls = str(d.get("class_") or "Unknown")
        group = node_groups.setdefault(cls, {"x": [], "y": [], "z": [], "hover": []})
        if cls not in node_group_colors:
            node_group_colors[cls] = node_color_map.get(cls, "#22c55e")
            node_group_labels[cls] = f"Node: {cls}"

        props = d.get("properties", {}) or {}
        hover_props = "<br>".join(
            f"<b>{k}</b>: {v}" for k, v in props.items() if v not in ("", None)
        )
        group["x"].append(x)
        group["y"].append(y)
        group["z"].append(z)
        group["hover"].append(
            f"<b>{d.get('label', n)}</b><br>Class: {cls}<br>{hover_props}"
        )

    def _new_edge_group() -> dict[str, list]:
        return {
            "x": [],
            "y": [],
            "z": [],
            "mid_x": [],
            "mid_y": [],
            "mid_z": [],
            "label_x": [],
            "label_y": [],
            "label_z": [],
            "hover": [],
        }

    def _edge_hover_message(rel: str, attrs: dict[str, object]) -> str:
        meaning = edge_relation_explanations.get(rel, "graph relation")
        message = f"Relation: {rel}<br>Meaning: {meaning}"
        source = attrs.get("source")
        if source is not None:
            message += f"<br>Source: {html.escape(str(source))}"
        distance = attrs.get("distance")
        try:
            if distance is not None:
                message += f"<br>Distance: {float(distance):.3f}"
        except (TypeError, ValueError):
            pass
        for key, label in (
            ("angle_deg", "Angle"),
            ("lateral_offset_xy", "Lateral offset XY"),
            ("overlap_area_xy", "Overlap area XY"),
            ("overlap_ratio_xy", "Overlap ratio XY"),
            ("vertical_gap", "Vertical gap"),
            ("intersection_volume", "Intersection volume"),
            ("contact_area", "Contact area"),
        ):
            value = attrs.get(key)
            try:
                if value is not None:
                    message += f"<br>{label}: {float(value):.6f}"
            except (TypeError, ValueError):
                continue
        for key, label in (
            ("inside_method", "Inside method"),
            ("storey_id", "Storey"),
            ("derived_from", "Derived from"),
        ):
            value = attrs.get(key)
            if value is not None:
                message += f"<br>{label}: {html.escape(str(value))}"
        same_storey = attrs.get("same_storey")
        if same_storey is not None:
            message += (
                "<br>Same storey: true"
                if bool(same_storey)
                else "<br>Same storey: false"
            )
        shared_boundary_elements = attrs.get("shared_boundary_elements")
        if isinstance(shared_boundary_elements, list) and shared_boundary_elements:
            joined = ", ".join(
                html.escape(str(item)) for item in shared_boundary_elements
            )
            message += f"<br>Shared boundaries: {joined}"
        return message

    def _append_edge_group(
        groups: dict[str, dict[str, list]],
        source_id: str,
        target_id: str,
        attrs: dict[str, object],
    ) -> None:
        rel = str(attrs.get("relation", "related_to"))
        group = groups.setdefault(rel, _new_edge_group())
        x0, y0, z0 = pos.get(source_id) or (0.0, 0.0, 0.0)
        x1, y1, z1 = pos.get(target_id) or (0.0, 0.0, 0.0)
        group["x"] += [x0, x1, None]
        group["y"] += [y0, y1, None]
        group["z"] += [z0, z1, None]
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0
        mz = (z0 + z1) / 2.0
        group["mid_x"].append(mx)
        group["mid_y"].append(my)
        group["mid_z"].append(mz)
        group["label_x"].append(mx)
        group["label_y"].append(my)
        group["label_z"].append(mz)
        group["hover"].append(_edge_hover_message(rel, attrs))

    graph_edge_items = [(str(u), str(v), d) for u, v, d in G.edges(data=True)]

    overlay_edges = _viz_collect_overlay_edges(G)
    derived_edge_items = [
        (source_id, target_id, attrs)
        for overlay_items in overlay_edges.values()
        for source_id, target_id, attrs in overlay_items
    ]
    display_edge_items = graph_edge_items + derived_edge_items

    edge_groups: dict[str, dict[str, list]] = {}
    for source_id, target_id, attrs in display_edge_items:
        _append_edge_group(edge_groups, source_id, target_id, attrs)

    bbox_x: list[float | None] = []
    bbox_y: list[float | None] = []
    bbox_z: list[float | None] = []
    bbox_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for node_id, node_data in G.nodes(data=True):
        if not str(node_id).startswith("Element::"):
            continue
        bbox = node_data.get("bbox")
        if bbox is None:
            continue
        min_xyz, max_xyz = bbox
        x0, y0, z0 = (float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2]))
        x1, y1, z1 = (float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2]))
        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        for a, b in bbox_edges:
            ax, ay, az = corners[a]
            bx, by, bz = corners[b]
            bbox_x += [ax, bx, None]
            bbox_y += [ay, by, None]
            bbox_z += [az, bz, None]

    traces: list[go.BaseTraceType] = []
    trace_meta: list[tuple[str, str]] = []

    if bbox_x:
        traces.append(
            go.Scatter3d(
                x=bbox_x,
                y=bbox_y,
                z=bbox_z,
                mode="lines",
                line={"width": 2, "color": "#f59e0b"},
                opacity=0.45,
                hoverinfo="none",
                name="BBox wireframe",
                legendgroup="bbox",
                showlegend=True,
                visible=False,
            )
        )
        trace_meta.append(("bbox", "bbox"))

    mesh_groups: dict[str, dict[str, list[int | float]]] = {}
    for node_id, node_data in G.nodes(data=True):
        if not str(node_id).startswith("Element::"):
            continue
        mesh = node_data.get("mesh")
        if mesh is None:
            continue
        vertices, faces = mesh
        if not vertices or not faces:
            continue
        cls = str(node_data.get("class_") or "Element")
        group = mesh_groups.setdefault(
            cls, {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}
        )
        base = len(group["x"])
        for vx, vy, vz in vertices:
            group["x"].append(float(vx))
            group["y"].append(float(vy))
            group["z"].append(float(vz))
        for a, b, c in faces:
            group["i"].append(int(a) + base)
            group["j"].append(int(b) + base)
            group["k"].append(int(c) + base)

    for idx, cls in enumerate(sorted(mesh_groups)):
        mesh = mesh_groups[cls]
        traces.append(
            go.Mesh3d(
                x=mesh["x"],
                y=mesh["y"],
                z=mesh["z"],
                i=mesh["i"],
                j=mesh["j"],
                k=mesh["k"],
                opacity=0.16,
                color="#0ea5e9",
                name=f"Mesh surface: {cls}",
                legendgroup="mesh",
                showlegend=idx == 0,
                hoverinfo="skip",
                flatshading=True,
                visible=False,
            )
        )
        trace_meta.append(("mesh", cls))

    for rel in sorted(edge_groups):
        edge = edge_groups[rel]
        edge_color = edge_color_map.get(rel, "#4b5563")
        rel_expl = edge_relation_explanations.get(rel, "graph relation")
        traces.append(
            go.Scatter3d(
                x=edge["x"],
                y=edge["y"],
                z=edge["z"],
                mode="lines",
                line={"width": 3, "color": edge_color},
                hoverinfo="none",
                name=f"Edge: {rel} - {rel_expl}",
                legendgroup=f"edge::{rel}",
                showlegend=True,
            )
        )
        trace_meta.append(("edge", rel))

        traces.append(
            go.Scatter3d(
                x=edge["mid_x"],
                y=edge["mid_y"],
                z=edge["mid_z"],
                mode="markers",
                marker={"size": 2, "color": edge_color, "opacity": 0.0},
                hoverinfo="text",
                hovertext=edge["hover"],
                showlegend=False,
                legendgroup=f"edge::{rel}",
            )
        )
        trace_meta.append(("edge_hover", rel))

        traces.append(
            go.Scatter3d(
                x=edge["label_x"],
                y=edge["label_y"],
                z=edge["label_z"],
                mode="text",
                text=[rel] * len(edge["label_x"]),
                textposition="middle center",
                textfont={"size": 10, "color": edge_color},
                hoverinfo="none",
                showlegend=False,
                legendgroup=f"edge::{rel}",
                visible=False,
            )
        )
        trace_meta.append(("edge_label", rel))

    for group_key in sorted(node_groups):
        node = node_groups[group_key]
        node_color = node_group_colors.get(group_key, "#22c55e")
        label = node_group_labels.get(group_key, f"Node: {group_key}")
        traces.append(
            go.Scatter3d(
                x=node["x"],
                y=node["y"],
                z=node["z"],
                mode="markers",
                marker={"size": 6, "color": node_color},
                hoverinfo="text",
                hovertext=node["hover"],
                name=label,
                legendgroup=f"node::{group_key}",
                showlegend=True,
            )
        )
        trace_meta.append(("node", group_key))

    def _mask(
        mode: str,
        show_edge_annotations: bool = False,
        show_bboxes: bool = False,
        show_meshes: bool = False,
    ) -> list[bool]:
        edge_categories = G.graph.get("edge_categories", {})
        hierarchy_rels = set(edge_categories.get("hierarchy", []))
        spatial_rels = set(edge_categories.get("spatial", []))
        topology_rels = set(edge_categories.get("topology", []))
        explicit_rels = set(edge_categories.get("explicit", []))
        spatial_rels.update({"inside_footprint_of", "same_storey_as"})
        topology_rels.add("aligned_with")
        visible: list[bool] = []
        for kind, name in trace_meta:
            if kind == "bbox":
                visible.append(show_bboxes)
                continue
            if kind == "mesh":
                visible.append(show_meshes)
                continue
            if mode == "all":
                visible.append(kind != "edge_label" or show_edge_annotations)
            elif mode == "nodes":
                visible.append(kind == "node")
            elif mode == "edges":
                visible.append(
                    kind in {"edge", "edge_hover"}
                    or (kind == "edge_label" and show_edge_annotations)
                )
            elif mode == "hierarchy":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in hierarchy_rels)
                    or (
                        kind == "edge_label"
                        and show_edge_annotations
                        and name in hierarchy_rels
                    )
                )
            elif mode == "spatial":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in spatial_rels)
                    or (
                        kind == "edge_label"
                        and show_edge_annotations
                        and name in spatial_rels
                    )
                )
            elif mode == "topology":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in topology_rels)
                    or (
                        kind == "edge_label"
                        and show_edge_annotations
                        and name in topology_rels
                    )
                )
            elif mode == "explicit":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in explicit_rels)
                    or (
                        kind == "edge_label"
                        and show_edge_annotations
                        and name in explicit_rels
                    )
                )
            else:
                visible.append(kind != "edge_label" or show_edge_annotations)
        return visible

    def _mask_variants(mode: str) -> dict[str, list[bool]]:
        variants: dict[str, list[bool]] = {}
        for show_edge_annotations in (False, True):
            for show_bboxes in (False, True):
                for show_meshes in (False, True):
                    parts = []
                    if show_edge_annotations:
                        parts.append("annotations")
                    if show_bboxes:
                        parts.append("bboxes")
                    if show_meshes:
                        parts.append("meshes")
                    key = f"with_{'_and_'.join(parts)}" if parts else "base"
                    variants[key] = _mask(
                        mode,
                        show_edge_annotations,
                        show_bboxes,
                        show_meshes,
                    )
        return variants

    filter_masks = {
        mode: _mask_variants(mode)
        for mode in (
            "all",
            "nodes",
            "edges",
            "hierarchy",
            "spatial",
            "topology",
            "explicit",
        )
    }

    fig = go.Figure(data=traces)
    fig.update_layout(
        title={"text": "IFC Graph (JSONL)", "x": 0.01, "xanchor": "left"},
        scene={"aspectmode": "data"},
        showlegend=False,
        font={
            "family": "Segoe UI, Tahoma, Arial, sans-serif",
            "size": 14,
            "color": "#1f3552",
        },
        margin={"l": 8, "r": 8, "t": 56, "b": 8},
        paper_bgcolor="#f7f9fc",
        plot_bgcolor="#f7f9fc",
    )

    edge_entries, total_edges = _viz_build_legend_entries(
        display_edge_items,
        edge_color_map=edge_color_map,
        edge_relation_explanations=edge_relation_explanations,
    )

    edge_items = _viz_render_legend_entries(
        edge_entries,
        prefix="Edge",
        count_title="Edge count",
    )

    node_items = []
    for group_key in sorted(node_groups):
        node_color = node_group_colors.get(group_key, "#22c55e")
        label = node_group_labels.get(group_key, f"Node: {group_key}")
        node_items.append(
            "<div class='legend-item'>"
            f"<span class='swatch dot' style='--swatch:{node_color}'></span>"
            f"<span>{html.escape(label)}</span>"
            "</div>"
        )

    plotly_div = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
        div_id="viewer",
        default_width="100%",
        default_height="100%",
    )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IFC Graph</title>
  <style>
    :root {{
      --ui-font: "Segoe UI", Tahoma, Arial, sans-serif;
      --ui-size: 14px;
      --ui-fg: #1f3552;
      --panel-bg: rgba(255, 255, 255, 0.94);
      --panel-border: #b9c6d8;
      --panel-shadow: 0 6px 18px rgba(31, 53, 82, 0.12);
      --gap: 12px;
      --radius: 10px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #f7f9fc 0%, #eef3fb 100%);
      color: var(--ui-fg);
      font-family: var(--ui-font);
      font-size: var(--ui-size);
      padding: 14px;
    }}
    .app {{
      width: 100%;
      height: calc(100vh - 28px);
      display: flex;
      flex-direction: column;
      gap: var(--gap);
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 10px;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      box-shadow: var(--panel-shadow);
    }}
    .toolbar button {{
      flex: 1 1 120px;
      min-height: 38px;
      border: 1px solid #9fb1c9;
      border-radius: 8px;
      background: #f2f6fc;
      color: var(--ui-fg);
      font-family: var(--ui-font);
      font-size: var(--ui-size);
      cursor: pointer;
      padding: 8px 14px;
    }}
    .toolbar button.active {{
      background: #dbe9ff;
      border-color: #4c79bd;
      font-weight: 600;
    }}
    .toolbar button.toggle {{
      flex: 0 1 180px;
      background: #fff8ea;
      border-color: #d8b474;
    }}
    .toolbar button.toggle.active {{
      background: #ffe9bf;
      border-color: #bf8a2f;
    }}
    .viewer-shell {{
      position: relative;
      flex: 1;
      min-height: 420px;
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      overflow: hidden;
      background: #f7f9fc;
    }}
    #viewer {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }}
    .legend {{
      position: absolute;
      right: 16px;
      bottom: 16px;
      width: min(420px, calc(100% - 32px));
      max-height: 52vh;
      overflow: auto;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      box-shadow: var(--panel-shadow);
      padding: 12px 14px;
      z-index: 5;
    }}
    .legend h3 {{
      margin: 0 0 8px 0;
      font-size: 17px;
      line-height: 1.2;
    }}
    .legend .section-title {{
      margin: 10px 0 6px;
      font-weight: 700;
    }}
    .legend .section-meta {{
      margin: -2px 0 8px;
      color: #51657f;
      font-size: 12px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      line-height: 1.3;
    }}
    .legend-item-text {{
      min-width: 0;
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      gap: 1px;
    }}
    .legend-item-main {{
      font-weight: 600;
    }}
    .legend-item-sub {{
      color: #51657f;
      font-size: 12px;
    }}
    .legend-count {{
      flex: 0 0 auto;
      min-width: 38px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e7eef8;
      border: 1px solid #c5d3e6;
      color: #1f3552;
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    .swatch {{
      flex: 0 0 auto;
      display: inline-block;
      background: var(--swatch);
    }}
    .swatch.line {{
      width: 30px;
      height: 4px;
      border-radius: 2px;
    }}
    .swatch.dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar" role="toolbar" aria-label="Graph filters">
      <button class="active" data-mode="all">Show All</button>
      <button data-mode="nodes">Nodes Only</button>
      <button data-mode="edges">Edges Only</button>
      <button data-mode="hierarchy">Hierarchy</button>
      <button data-mode="spatial">Spatial</button>
      <button data-mode="topology">Topology</button>
      <button data-mode="explicit">Explicit IFC</button>
      <button
        id="toggle-edge-annotations"
        class="toggle"
        type="button"
        aria-pressed="false"
      >
        Edge Labels: Off
      </button>
      <button
        id="toggle-bboxes"
        class="toggle"
        type="button"
        aria-pressed="false"
      >
        BBoxes: Off
      </button>
      <button
        id="toggle-meshes"
        class="toggle"
        type="button"
        aria-pressed="false"
      >
        Meshes: Off
      </button>
    </div>
    <div class="viewer-shell">
      {plotly_div}
      <aside class="legend" aria-label="Graph legend">
        <h3>Legend</h3>
        <div class="section-title">Edges</div>
        <div class="section-meta">Total edges shown: {total_edges}</div>
        {"".join(edge_items)}
        <div class="section-title">Nodes</div>
        {"".join(node_items)}
      </aside>
    </div>
  </div>
  <script>
    const masks = {json.dumps(filter_masks)};
    const viewer = document.getElementById("viewer");
    const modeButtons = Array.from(
      document.querySelectorAll(".toolbar button[data-mode]")
    );
    const toggleEdgeAnnotationsButton = document.getElementById(
      "toggle-edge-annotations"
    );
    const toggleBboxesButton = document.getElementById("toggle-bboxes");
    const toggleMeshesButton = document.getElementById("toggle-meshes");
    let currentMode = "all";
    let edgeAnnotationsEnabled = false;
    let bboxesEnabled = false;
    let meshesEnabled = false;

    function maskKey() {{
      const parts = [];
      if (edgeAnnotationsEnabled) parts.push("annotations");
      if (bboxesEnabled) parts.push("bboxes");
      if (meshesEnabled) parts.push("meshes");
      return parts.length ? `with_${{parts.join("_and_")}}` : "base";
    }}

    function applyMode(mode) {{
      if (!viewer || !viewer.data) return;
      currentMode = mode;
      const modeMasks = masks[mode] || masks.all;
      const visible = modeMasks[maskKey()] || modeMasks.base;
      Plotly.restyle(viewer, {{ visible }});
      modeButtons.forEach((btn) =>
        btn.classList.toggle("active", btn.dataset.mode === mode)
      );
    }}

    modeButtons.forEach((btn) => {{
      btn.addEventListener("click", () => applyMode(btn.dataset.mode));
    }});

    toggleEdgeAnnotationsButton?.addEventListener("click", () => {{
      edgeAnnotationsEnabled = !edgeAnnotationsEnabled;
      toggleEdgeAnnotationsButton.classList.toggle("active", edgeAnnotationsEnabled);
      toggleEdgeAnnotationsButton.setAttribute(
        "aria-pressed",
        edgeAnnotationsEnabled ? "true" : "false"
      );
      toggleEdgeAnnotationsButton.textContent = edgeAnnotationsEnabled
        ? "Edge Labels: On"
        : "Edge Labels: Off";
      applyMode(currentMode);
    }});

    toggleBboxesButton?.addEventListener("click", () => {{
      bboxesEnabled = !bboxesEnabled;
      toggleBboxesButton.classList.toggle("active", bboxesEnabled);
      toggleBboxesButton.setAttribute(
        "aria-pressed",
        bboxesEnabled ? "true" : "false"
      );
      toggleBboxesButton.textContent = bboxesEnabled
        ? "BBoxes: On"
        : "BBoxes: Off";
      applyMode(currentMode);
    }});

    toggleMeshesButton?.addEventListener("click", () => {{
      meshesEnabled = !meshesEnabled;
      toggleMeshesButton.classList.toggle("active", meshesEnabled);
      toggleMeshesButton.setAttribute(
        "aria-pressed",
        meshesEnabled ? "true" : "false"
      );
      toggleMeshesButton.textContent = meshesEnabled
        ? "Meshes: On"
        : "Meshes: Off";
      applyMode(currentMode);
    }});

    applyMode(currentMode);

    window.addEventListener("resize", () => {{
      if (viewer && viewer.data) {{
        Plotly.Plots.resize(viewer);
      }}
    }});
  </script>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page_html, encoding="utf-8")
    LOG.info("Visualization saved to %s", out_html)


def plot_interactive_graph_overlap_modes(
    graphs_by_mode: dict[str, nx.DiGraph | nx.MultiDiGraph],
    out_html: Path,
) -> None:
    if not graphs_by_mode:
        raise ValueError("At least one graph is required to render overlap modes.")

    preferred_order = ["full", "threshold", "top_k", "none"]
    ordered_modes = [
        mode for mode in preferred_order if mode in graphs_by_mode
    ] + sorted(mode for mode in graphs_by_mode if mode not in preferred_order)
    base_mode = ordered_modes[0]
    base_graph = graphs_by_mode[base_mode]

    pos: dict[str, tuple[float, float, float] | None] = {}
    for n, d in base_graph.nodes(data=True):
        geom = d.get("geometry")
        if geom is not None:
            pos[n] = tuple(float(v) for v in geom)
        else:
            pos[n] = None

    for n in base_graph.nodes:
        if pos.get(n) is not None:
            continue
        child_positions = [
            pos[c] for c in base_graph.successors(n) if pos.get(c) is not None
        ]
        if child_positions:
            arr = np.array(child_positions, dtype=float)
            pos[n] = tuple(float(v) for v in arr.mean(axis=0))
        else:
            pos[n] = (0.0, 0.0, 0.0)

    node_color_map = {
        "IfcProject": "#7c3aed",
        "IfcBuilding": "#2563eb",
        "IfcBuildingStorey": "#f97316",
        "IfcSystem": "#0891b2",
        "IfcZone": "#16a34a",
    }
    edge_color_map = {
        "aggregates": "#6b7280",
        "contains": "#1d4ed8",
        "contained_in": "#2563eb",
        "typed_by": "#ca8a04",
        "connected_to": "#ef4444",
        "adjacent_to": "#059669",
        "intersects_bbox": "#f59e0b",
        "overlaps_xy": "#0ea5e9",
        "intersects_3d": "#b91c1c",
        "touches_surface": "#9333ea",
        "above": "#7c3aed",
        "below": "#9333ea",
        "hosts": "#b91c1c",
        "hosted_by": "#dc2626",
        "ifc_connected_to": "#0f766e",
        "path_connected_to": "#0b6e4f",
        "space_bounded_by": "#be123c",
        "bounds_space": "#9f1239",
        "belongs_to_system": "#0369a1",
        "in_zone": "#15803d",
        "classified_as": "#4f46e5",
    }
    edge_relation_explanations = {
        "aggregates": "hierarchy decomposition",
        "contains": "container to child",
        "contained_in": "child to container",
        "typed_by": "instance classified by type",
        "connected_to": "bbox intersects or touches",
        "adjacent_to": "spatially near within threshold",
        "intersects_bbox": "3D bbox overlap",
        "overlaps_xy": "2D footprint overlap",
        "intersects_3d": "mesh-informed 3D intersection",
        "touches_surface": "mesh-informed surface contact",
        "above": "vertical ordering above",
        "below": "vertical ordering below",
        "hosts": "explicit IFC host relationship",
        "hosted_by": "explicit IFC hosted-by relationship",
        "ifc_connected_to": "explicit IFC connectivity",
        "path_connected_to": "explicit IFC path connectivity",
        "space_bounded_by": "IFC space boundary (space to element)",
        "bounds_space": "IFC space boundary (element to space)",
        "belongs_to_system": "assigned to IFC system",
        "in_zone": "assigned to IFC zone",
        "classified_as": "assigned IFC classification",
    }

    node_groups: dict[str, dict[str, list]] = {}
    node_group_colors: dict[str, str] = {}
    node_group_labels: dict[str, str] = {}
    for n, d in base_graph.nodes(data=True):
        x, y, z = pos[n] or (0.0, 0.0, 0.0)
        cls = str(d.get("class_") or "Unknown")
        group = node_groups.setdefault(cls, {"x": [], "y": [], "z": [], "hover": []})
        if cls not in node_group_colors:
            node_group_colors[cls] = node_color_map.get(cls, "#22c55e")
            node_group_labels[cls] = f"Node: {cls}"
        props = d.get("properties", {}) or {}
        hover_props = "<br>".join(
            f"<b>{k}</b>: {v}" for k, v in props.items() if v not in ("", None)
        )
        group["x"].append(x)
        group["y"].append(y)
        group["z"].append(z)
        group["hover"].append(
            f"<b>{d.get('label', n)}</b><br>Class: {cls}<br>{hover_props}"
        )

    def _collect_edge_groups(
        graph: nx.DiGraph | nx.MultiDiGraph,
        *,
        only_relation: str | None = None,
        exclude_relation: str | None = None,
    ) -> dict[str, dict[str, list]]:
        edge_groups: dict[str, dict[str, list]] = {}
        for u, v, d in graph.edges(data=True):
            rel = str(d.get("relation", "related_to"))
            if only_relation is not None and rel != only_relation:
                continue
            if exclude_relation is not None and rel == exclude_relation:
                continue
            group = edge_groups.setdefault(
                rel,
                {
                    "x": [],
                    "y": [],
                    "z": [],
                    "mid_x": [],
                    "mid_y": [],
                    "mid_z": [],
                    "label_x": [],
                    "label_y": [],
                    "label_z": [],
                    "hover": [],
                },
            )
            x0, y0, z0 = pos.get(u) or (0.0, 0.0, 0.0)
            x1, y1, z1 = pos.get(v) or (0.0, 0.0, 0.0)
            group["x"] += [x0, x1, None]
            group["y"] += [y0, y1, None]
            group["z"] += [z0, z1, None]
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0
            mz = (z0 + z1) / 2.0
            group["mid_x"].append(mx)
            group["mid_y"].append(my)
            group["mid_z"].append(mz)
            group["label_x"].append(mx)
            group["label_y"].append(my)
            group["label_z"].append(mz)

            source = d.get("source")
            dist = d.get("distance")
            meaning = edge_relation_explanations.get(rel, "graph relation")
            msg = f"Relation: {rel}<br>Meaning: {meaning}"
            if source is not None:
                msg += f"<br>Source: {source}"
            if dist is not None:
                msg += f"<br>Distance: {float(dist):.3f}"
            overlap_ratio = d.get("overlap_ratio_xy")
            if overlap_ratio is not None:
                msg += f"<br>Overlap ratio: {float(overlap_ratio):.3f}"
            overlap_area = d.get("overlap_area_xy")
            if overlap_area is not None:
                msg += f"<br>Overlap area XY: {float(overlap_area):.3f}"
            group["hover"].append(msg)
        return edge_groups

    base_edge_groups = _collect_edge_groups(base_graph, exclude_relation="overlaps_xy")
    overlap_edge_groups_by_mode = {
        mode: _collect_edge_groups(graph, only_relation="overlaps_xy")
        for mode, graph in graphs_by_mode.items()
    }

    bbox_x: list[float | None] = []
    bbox_y: list[float | None] = []
    bbox_z: list[float | None] = []
    bbox_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for node_id, node_data in base_graph.nodes(data=True):
        if not str(node_id).startswith("Element::"):
            continue
        bbox = node_data.get("bbox")
        if bbox is None:
            continue
        min_xyz, max_xyz = bbox
        x0, y0, z0 = (float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2]))
        x1, y1, z1 = (float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2]))
        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        for a, b in bbox_edges:
            ax, ay, az = corners[a]
            bx, by, bz = corners[b]
            bbox_x += [ax, bx, None]
            bbox_y += [ay, by, None]
            bbox_z += [az, bz, None]

    traces: list[go.BaseTraceType] = []
    trace_meta: list[tuple[str, str, str | None]] = []

    if bbox_x:
        traces.append(
            go.Scatter3d(
                x=bbox_x,
                y=bbox_y,
                z=bbox_z,
                mode="lines",
                line={"width": 2, "color": "#f59e0b"},
                opacity=0.45,
                hoverinfo="none",
                name="BBox wireframe",
                legendgroup="bbox",
                showlegend=True,
                visible=False,
            )
        )
        trace_meta.append(("bbox", "bbox", None))

    mesh_groups: dict[str, dict[str, list[int | float]]] = {}
    for node_id, node_data in base_graph.nodes(data=True):
        if not str(node_id).startswith("Element::"):
            continue
        mesh = node_data.get("mesh")
        if mesh is None:
            continue
        vertices, faces = mesh
        if not vertices or not faces:
            continue
        cls = str(node_data.get("class_") or "Element")
        group = mesh_groups.setdefault(
            cls, {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}
        )
        base = len(group["x"])
        for vx, vy, vz in vertices:
            group["x"].append(float(vx))
            group["y"].append(float(vy))
            group["z"].append(float(vz))
        for a, b, c in faces:
            group["i"].append(int(a) + base)
            group["j"].append(int(b) + base)
            group["k"].append(int(c) + base)

    for idx, cls in enumerate(sorted(mesh_groups)):
        mesh = mesh_groups[cls]
        traces.append(
            go.Mesh3d(
                x=mesh["x"],
                y=mesh["y"],
                z=mesh["z"],
                i=mesh["i"],
                j=mesh["j"],
                k=mesh["k"],
                opacity=0.16,
                color="#0ea5e9",
                name=f"Mesh surface: {cls}",
                legendgroup="mesh",
                showlegend=idx == 0,
                hoverinfo="skip",
                flatshading=True,
                visible=False,
            )
        )
        trace_meta.append(("mesh", cls, None))

    def _append_edge_group_traces(
        edge_groups: dict[str, dict[str, list]],
        *,
        variant: str | None,
    ) -> None:
        for rel in sorted(edge_groups):
            edge = edge_groups[rel]
            edge_color = edge_color_map.get(rel, "#4b5563")
            rel_expl = edge_relation_explanations.get(rel, "graph relation")
            traces.append(
                go.Scatter3d(
                    x=edge["x"],
                    y=edge["y"],
                    z=edge["z"],
                    mode="lines",
                    line={"width": 3, "color": edge_color},
                    hoverinfo="none",
                    name=f"Edge: {rel} - {rel_expl}",
                    legendgroup=f"edge::{variant or 'shared'}::{rel}",
                    showlegend=True,
                )
            )
            trace_meta.append(("edge", rel, variant))

            traces.append(
                go.Scatter3d(
                    x=edge["mid_x"],
                    y=edge["mid_y"],
                    z=edge["mid_z"],
                    mode="markers",
                    marker={"size": 2, "color": edge_color, "opacity": 0.0},
                    hoverinfo="text",
                    hovertext=edge["hover"],
                    showlegend=False,
                    legendgroup=f"edge::{variant or 'shared'}::{rel}",
                )
            )
            trace_meta.append(("edge_hover", rel, variant))

            traces.append(
                go.Scatter3d(
                    x=edge["label_x"],
                    y=edge["label_y"],
                    z=edge["label_z"],
                    mode="text",
                    text=[rel] * len(edge["label_x"]),
                    textposition="middle center",
                    textfont={"size": 10, "color": edge_color},
                    hoverinfo="none",
                    showlegend=False,
                    legendgroup=f"edge::{variant or 'shared'}::{rel}",
                    visible=False,
                )
            )
            trace_meta.append(("edge_label", rel, variant))

    _append_edge_group_traces(base_edge_groups, variant=None)
    for mode in ordered_modes:
        _append_edge_group_traces(
            overlap_edge_groups_by_mode.get(mode, {}),
            variant=mode,
        )

    for group_key in sorted(node_groups):
        node = node_groups[group_key]
        node_color = node_group_colors.get(group_key, "#22c55e")
        label = node_group_labels.get(group_key, f"Node: {group_key}")
        traces.append(
            go.Scatter3d(
                x=node["x"],
                y=node["y"],
                z=node["z"],
                mode="markers",
                marker={"size": 6, "color": node_color},
                hoverinfo="text",
                hovertext=node["hover"],
                name=label,
                legendgroup=f"node::{group_key}",
                showlegend=True,
            )
        )
        trace_meta.append(("node", group_key, None))

    def _mask(
        filter_mode: str,
        overlap_mode: str,
        *,
        show_edge_annotations: bool = False,
        show_bboxes: bool = False,
        show_meshes: bool = False,
    ) -> list[bool]:
        edge_categories = base_graph.graph.get("edge_categories", {})
        hierarchy_rels = set(edge_categories.get("hierarchy", []))
        spatial_rels = set(edge_categories.get("spatial", []))
        topology_rels = set(edge_categories.get("topology", []))
        explicit_rels = set(edge_categories.get("explicit", []))
        visible: list[bool] = []
        for kind, name, variant in trace_meta:
            if kind == "bbox":
                visible.append(show_bboxes)
                continue
            if kind == "mesh":
                visible.append(show_meshes)
                continue
            if (
                variant is not None
                and name == "overlaps_xy"
                and variant != overlap_mode
            ):
                visible.append(False)
                continue
            if filter_mode == "all":
                visible.append(kind != "edge_label" or show_edge_annotations)
            elif filter_mode == "nodes":
                visible.append(kind == "node")
            elif filter_mode == "edges":
                visible.append(
                    kind in {"edge", "edge_hover"}
                    or (kind == "edge_label" and show_edge_annotations)
                )
            elif filter_mode == "hierarchy":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in hierarchy_rels)
                    or (
                        kind == "edge_label"
                        and show_edge_annotations
                        and name in hierarchy_rels
                    )
                )
            elif filter_mode == "spatial":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in spatial_rels)
                    or (
                        kind == "edge_label"
                        and show_edge_annotations
                        and name in spatial_rels
                    )
                )
            elif filter_mode == "topology":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in topology_rels)
                    or (
                        kind == "edge_label"
                        and show_edge_annotations
                        and name in topology_rels
                    )
                )
            elif filter_mode == "explicit":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in explicit_rels)
                    or (
                        kind == "edge_label"
                        and show_edge_annotations
                        and name in explicit_rels
                    )
                )
            else:
                visible.append(kind != "edge_label" or show_edge_annotations)
        return visible

    filter_modes = [
        "all",
        "nodes",
        "edges",
        "hierarchy",
        "spatial",
        "topology",
        "explicit",
    ]
    filter_masks: dict[str, dict[str, dict[str, list[bool]]]] = {}
    for overlap_mode in ordered_modes:
        filter_masks[overlap_mode] = {
            filter_mode: {
                "base": _mask(filter_mode, overlap_mode),
                "with_annotations": _mask(
                    filter_mode,
                    overlap_mode,
                    show_edge_annotations=True,
                ),
                "with_bboxes": _mask(
                    filter_mode,
                    overlap_mode,
                    show_bboxes=True,
                ),
                "with_annotations_and_bboxes": _mask(
                    filter_mode,
                    overlap_mode,
                    show_edge_annotations=True,
                    show_bboxes=True,
                ),
                "with_meshes": _mask(
                    filter_mode,
                    overlap_mode,
                    show_meshes=True,
                ),
                "with_annotations_and_meshes": _mask(
                    filter_mode,
                    overlap_mode,
                    show_edge_annotations=True,
                    show_meshes=True,
                ),
                "with_bboxes_and_meshes": _mask(
                    filter_mode,
                    overlap_mode,
                    show_bboxes=True,
                    show_meshes=True,
                ),
                "with_annotations_and_bboxes_and_meshes": _mask(
                    filter_mode,
                    overlap_mode,
                    show_edge_annotations=True,
                    show_bboxes=True,
                    show_meshes=True,
                ),
            }
            for filter_mode in filter_modes
        }

    fig = go.Figure(data=traces)
    fig.update_layout(
        title={"text": "IFC Graph (JSONL)", "x": 0.01, "xanchor": "left"},
        scene={"aspectmode": "data"},
        showlegend=False,
        font={
            "family": "Segoe UI, Tahoma, Arial, sans-serif",
            "size": 14,
            "color": "#1f3552",
        },
        margin={"l": 8, "r": 8, "t": 56, "b": 8},
        paper_bgcolor="#f7f9fc",
        plot_bgcolor="#f7f9fc",
    )

    legend_payloads: dict[str, dict[str, str]] = {}
    for mode in ordered_modes:
        graph = graphs_by_mode[mode]
        graph_edge_items = [
            (str(source_id), str(target_id), attrs)
            for source_id, target_id, attrs in graph.edges(data=True)
        ]
        edge_entries, total_edges = _viz_build_legend_entries(
            graph_edge_items,
            edge_color_map=edge_color_map,
            edge_relation_explanations=edge_relation_explanations,
        )
        legend_payloads[mode] = {
            "total_edges": str(total_edges),
            "edge_items_html": _viz_render_legend_entries(
                edge_entries,
                prefix="Edge",
                count_title="Edge count",
            ),
        }

    node_items = []
    for group_key in sorted(node_groups):
        node_color = node_group_colors.get(group_key, "#22c55e")
        label = node_group_labels.get(group_key, f"Node: {group_key}")
        node_items.append(
            "<div class='legend-item'>"
            f"<span class='swatch dot' style='--swatch:{node_color}'></span>"
            f"<span>{html.escape(label)}</span>"
            "</div>"
        )

    plotly_div = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
        div_id="viewer",
        default_width="100%",
        default_height="100%",
    )
    initial_legend = legend_payloads[base_mode]
    overlap_mode_buttons = "".join(
        (
            f"<button {'class="active" ' if mode == base_mode else ''}"
            f'data-overlap-mode="{html.escape(mode)}">'
            f"Overlap: {html.escape(mode)}</button>"
        )
        for mode in ordered_modes
    )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IFC Graph</title>
  <style>
    :root {{
      --ui-font: "Segoe UI", Tahoma, Arial, sans-serif;
      --ui-size: 14px;
      --ui-fg: #1f3552;
      --panel-bg: rgba(255, 255, 255, 0.94);
      --panel-border: #b9c6d8;
      --panel-shadow: 0 6px 18px rgba(31, 53, 82, 0.12);
      --gap: 12px;
      --radius: 10px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #f7f9fc 0%, #eef3fb 100%);
      color: var(--ui-fg);
      font-family: var(--ui-font);
      font-size: var(--ui-size);
      padding: 14px;
    }}
    .app {{
      width: 100%;
      height: calc(100vh - 28px);
      display: flex;
      flex-direction: column;
      gap: var(--gap);
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 10px;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      box-shadow: var(--panel-shadow);
    }}
    .toolbar button {{
      flex: 1 1 120px;
      min-height: 38px;
      border: 1px solid #9fb1c9;
      border-radius: 8px;
      background: #f2f6fc;
      color: var(--ui-fg);
      font-family: var(--ui-font);
      font-size: var(--ui-size);
      cursor: pointer;
      padding: 8px 14px;
    }}
    .toolbar button.active {{
      background: #dbe9ff;
      border-color: #4c79bd;
      font-weight: 600;
    }}
    .toolbar button.toggle {{
      flex: 0 1 180px;
      background: #fff8ea;
      border-color: #d8b474;
    }}
    .toolbar button.toggle.active {{
      background: #ffe9bf;
      border-color: #bf8a2f;
    }}
    .toolbar button[data-overlap-mode] {{
      flex: 0 1 150px;
      background: #edf7ef;
      border-color: #7eb08e;
    }}
    .toolbar button[data-overlap-mode].active {{
      background: #d9f0de;
      border-color: #3c8d57;
    }}
    .viewer-shell {{
      position: relative;
      flex: 1;
      min-height: 420px;
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      overflow: hidden;
      background: #f7f9fc;
    }}
    #viewer {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }}
    .legend {{
      position: absolute;
      right: 16px;
      bottom: 16px;
      width: min(420px, calc(100% - 32px));
      max-height: 52vh;
      overflow: auto;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      box-shadow: var(--panel-shadow);
      padding: 12px 14px;
      z-index: 5;
    }}
    .legend h3 {{
      margin: 0 0 8px 0;
      font-size: 17px;
      line-height: 1.2;
    }}
    .legend .section-title {{
      margin: 10px 0 6px;
      font-weight: 700;
    }}
    .legend .section-meta {{
      margin: -2px 0 8px;
      color: #51657f;
      font-size: 12px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      line-height: 1.3;
    }}
    .legend-item-text {{
      min-width: 0;
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      gap: 1px;
    }}
    .legend-item-main {{
      font-weight: 600;
    }}
    .legend-item-sub {{
      color: #51657f;
      font-size: 12px;
    }}
    .legend-count {{
      flex: 0 0 auto;
      min-width: 38px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e7eef8;
      border: 1px solid #c5d3e6;
      color: #1f3552;
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    .swatch {{
      flex: 0 0 auto;
      display: inline-block;
      background: var(--swatch);
    }}
    .swatch.line {{
      width: 30px;
      height: 4px;
      border-radius: 2px;
    }}
    .swatch.dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar" role="toolbar" aria-label="Graph filters">
      {overlap_mode_buttons}
      <button class="active" data-mode="all">Show All</button>
      <button data-mode="nodes">Nodes Only</button>
      <button data-mode="edges">Edges Only</button>
      <button data-mode="hierarchy">Hierarchy</button>
      <button data-mode="spatial">Spatial</button>
      <button data-mode="topology">Topology</button>
      <button data-mode="explicit">Explicit IFC</button>
      <button
        id="toggle-edge-annotations"
        class="toggle"
        type="button"
        aria-pressed="false"
      >
        Edge Labels: Off
      </button>
      <button
        id="toggle-bboxes"
        class="toggle"
        type="button"
        aria-pressed="false"
      >
        BBoxes: Off
      </button>
      <button
        id="toggle-meshes"
        class="toggle"
        type="button"
        aria-pressed="false"
      >
        Meshes: Off
      </button>
    </div>
    <div class="viewer-shell">
      {plotly_div}
      <aside class="legend" aria-label="Graph legend">
        <h3>Legend</h3>
        <div class="section-meta" id="overlap-mode-summary">
          Active overlap mode: {html.escape(base_mode)}
        </div>
        <div class="section-title">Edges</div>
        <div class="section-meta" id="legend-total-edges">
          Total edges shown: {initial_legend["total_edges"]}
        </div>
        <div id="legend-edge-items">{initial_legend["edge_items_html"]}</div>
        <div class="section-title">Nodes</div>
        {"".join(node_items)}
      </aside>
    </div>
  </div>
  <script>
    const masks = {json.dumps(filter_masks)};
    const legendPayloads = {json.dumps(legend_payloads)};
    const viewer = document.getElementById("viewer");
    const modeButtons = Array.from(
      document.querySelectorAll(".toolbar button[data-mode]")
    );
    const overlapModeButtons = Array.from(
      document.querySelectorAll(".toolbar button[data-overlap-mode]")
    );
    const toggleEdgeAnnotationsButton = document.getElementById(
      "toggle-edge-annotations"
    );
    const toggleBboxesButton = document.getElementById("toggle-bboxes");
    const toggleMeshesButton = document.getElementById("toggle-meshes");
    const legendEdgeItems = document.getElementById("legend-edge-items");
    const legendTotalEdges = document.getElementById("legend-total-edges");
    const overlapModeSummary = document.getElementById("overlap-mode-summary");
    let currentMode = "all";
    let currentOverlapMode = {json.dumps(base_mode)};
    let edgeAnnotationsEnabled = false;
    let bboxesEnabled = false;
    let meshesEnabled = false;

    function maskKey() {{
      const parts = [];
      if (edgeAnnotationsEnabled) parts.push("annotations");
      if (bboxesEnabled) parts.push("bboxes");
      if (meshesEnabled) parts.push("meshes");
      return parts.length ? `with_${{parts.join("_and_")}}` : "base";
    }}

    function updateLegend() {{
      const payload = legendPayloads[currentOverlapMode];
      if (!payload) return;
      legendEdgeItems.innerHTML = payload.edge_items_html;
      legendTotalEdges.textContent =
        `Total edges shown: ${{payload.total_edges}}`;
      overlapModeSummary.textContent =
        `Active overlap mode: ${{currentOverlapMode}}`;
    }}

    function applyMode(mode) {{
      if (!viewer || !viewer.data) return;
      currentMode = mode;
      const modeMasks = masks[currentOverlapMode] || masks[{json.dumps(base_mode)}];
      const visible =
        (modeMasks[mode] || modeMasks.all)[maskKey()] || modeMasks.all.base;
      Plotly.restyle(viewer, {{ visible }});
      modeButtons.forEach((btn) =>
        btn.classList.toggle("active", btn.dataset.mode === mode)
      );
      overlapModeButtons.forEach((btn) =>
        btn.classList.toggle("active", btn.dataset.overlapMode === currentOverlapMode)
      );
      updateLegend();
    }}

    modeButtons.forEach((btn) => {{
      btn.addEventListener("click", () => applyMode(btn.dataset.mode));
    }});

    overlapModeButtons.forEach((btn) => {{
      btn.addEventListener("click", () => {{
        currentOverlapMode = btn.dataset.overlapMode;
        applyMode(currentMode);
      }});
    }});

    toggleEdgeAnnotationsButton?.addEventListener("click", () => {{
      edgeAnnotationsEnabled = !edgeAnnotationsEnabled;
      toggleEdgeAnnotationsButton.classList.toggle("active", edgeAnnotationsEnabled);
      toggleEdgeAnnotationsButton.setAttribute(
        "aria-pressed",
        edgeAnnotationsEnabled ? "true" : "false"
      );
      toggleEdgeAnnotationsButton.textContent = edgeAnnotationsEnabled
        ? "Edge Labels: On"
        : "Edge Labels: Off";
      applyMode(currentMode);
    }});

    toggleBboxesButton?.addEventListener("click", () => {{
      bboxesEnabled = !bboxesEnabled;
      toggleBboxesButton.classList.toggle("active", bboxesEnabled);
      toggleBboxesButton.setAttribute(
        "aria-pressed",
        bboxesEnabled ? "true" : "false"
      );
      toggleBboxesButton.textContent = bboxesEnabled
        ? "BBoxes: On"
        : "BBoxes: Off";
      applyMode(currentMode);
    }});

    toggleMeshesButton?.addEventListener("click", () => {{
      meshesEnabled = !meshesEnabled;
      toggleMeshesButton.classList.toggle("active", meshesEnabled);
      toggleMeshesButton.setAttribute(
        "aria-pressed",
        meshesEnabled ? "true" : "false"
      );
      toggleMeshesButton.textContent = meshesEnabled
        ? "Meshes: On"
        : "Meshes: Off";
      applyMode(currentMode);
    }});

    applyMode(currentMode);

    window.addEventListener("resize", () => {{
      if (viewer && viewer.data) {{
        Plotly.Plots.resize(viewer);
      }}
    }});
  </script>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page_html, encoding="utf-8")
    LOG.info("Visualization saved to %s", out_html)


def build_graph(
    jsonl_paths: list[Path] | None = None,
    dataset: str | None = None,
    payload_mode: str | None = None,
    *,
    derived_edge_pruning_enabled: bool | None = None,
    derived_edge_prune_classes: object = None,
    overlap_xy_mode: object | None = None,
    overlap_xy_min_ratio: object | None = None,
    overlap_xy_top_k: object | None = None,
) -> nx.MultiDiGraph:
    # auto-detect jsonl files if no paths given
    # called with no args from query_service
    if jsonl_paths is None:
        script_dir = Path(__file__).resolve().parent
        project_root = find_project_root(script_dir) or script_dir.parent.parent.parent
        out_dir = project_root / "output"
        if dataset is not None:
            candidate = out_dir / f"{dataset}.jsonl"
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"JSONL file not found for dataset '{dataset}': {candidate}. "
                    "Run: uv run rag-tag-ifc-to-jsonl"
                )
            jsonl_paths = [candidate]
        else:
            jsonl_paths = sorted(out_dir.glob("*.jsonl"))
            if not jsonl_paths:
                raise FileNotFoundError(
                    f"No .jsonl files found in {out_dir}. "
                    "Run: uv run rag-tag-ifc-to-jsonl"
                )

    # Resolve payload mode: explicit param > env var > default "full".
    _mode = _resolve_graph_payload_mode(
        payload_mode
        if payload_mode is not None
        else os.environ.get("GRAPH_PAYLOAD_MODE", "full")
    )
    pruning_policy = _resolve_derived_edge_pruning_policy(
        enabled=derived_edge_pruning_enabled,
        exclude_classes=derived_edge_prune_classes,
    )
    overlap_xy_policy = _resolve_overlap_xy_policy(
        mode=overlap_xy_mode,
        min_ratio=overlap_xy_min_ratio,
        top_k=overlap_xy_top_k,
    )

    G = build_graph_from_jsonl(jsonl_paths, payload_mode=_mode)
    G.graph.setdefault("graph_build", {})["derived_edge_pruning"] = {
        "enabled": pruning_policy.enabled,
        "exclude_classes": sorted(pruning_policy.exclude_classes),
    }
    G.graph["graph_build"]["overlap_xy"] = {
        "mode": overlap_xy_policy.mode,
        "min_ratio": overlap_xy_policy.min_ratio,
        "top_k": overlap_xy_policy.top_k,
    }
    add_spatial_adjacency(G, exclude_classes=pruning_policy.exclude_classes)
    add_topology_facts(
        G,
        exclude_classes=pruning_policy.exclude_classes,
        overlap_xy_mode=overlap_xy_policy.mode,
        overlap_xy_min_ratio=overlap_xy_policy.min_ratio,
        overlap_xy_top_k=overlap_xy_policy.top_k,
    )
    return G


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    _configure_logging()
    ap = argparse.ArgumentParser(
        description=(
            "Build a NetworkX IFC graph from JSONL files, preserving full or "
            "minimal payloads and using mesh-derived geometry when present."
        )
    )
    ap.add_argument(
        "--jsonl-dir",
        type=Path,
        default=None,
        help="Directory containing .jsonl files (default: <project-root>/output/).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for HTML visualization (default: <project-root>/output/)."
        ),
    )
    ap.add_argument(
        "--no-viz",
        action="store_true",
        default=False,
        help="Skip HTML visualization (graph still built and stats printed).",
    )
    ap.add_argument(
        "--overlap-xy-mode",
        type=_normalize_overlap_xy_mode,
        default=None,
        help=(
            "Raw overlaps_xy emission mode: full, threshold, top_k, or none. "
            "Default comes from config and now defaults to none."
        ),
    )
    ap.add_argument(
        "--overlap-xy-min-ratio",
        type=float,
        default=None,
        help=(
            "Minimum overlap ratio used only when --overlap-xy-mode=threshold. "
            "Ratio = overlap_area / min(footprint areas)."
        ),
    )
    ap.add_argument(
        "--overlap-xy-top-k",
        type=int,
        default=None,
        help=(
            "Maximum retained overlap neighbors per node when --overlap-xy-mode=top_k."
        ),
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir) or script_dir.parent.parent.parent

    jsonl_dir = (args.jsonl_dir or project_root / "output").resolve()
    out_dir = (args.out_dir or project_root / "output").resolve()

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {jsonl_dir}")
        return

    print(f"Building graph from {len(jsonl_files)} JSONL file(s)...")
    G = build_graph(
        jsonl_files,
        overlap_xy_mode=args.overlap_xy_mode,
        overlap_xy_min_ratio=args.overlap_xy_min_ratio,
        overlap_xy_top_k=args.overlap_xy_top_k,
    )
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    overlap_xy_stats = G.graph.get("graph_build", {}).get("overlap_xy", {})
    print(
        "overlap_xy: "
        f"mode={overlap_xy_stats.get('mode')} "
        f"min_ratio={overlap_xy_stats.get('min_ratio')} "
        f"top_k={overlap_xy_stats.get('top_k')}"
    )

    if not args.no_viz:
        html_path = out_dir / "ifc_graph.html"
        plot_interactive_graph(G, html_path)
        print(f"Visualization: {html_path}")


if __name__ == "__main__":
    main()
