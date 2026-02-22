from __future__ import annotations

import html
import json
import logging
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

import ifcopenshell
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from rag_tag.parser.ifc_geometry_parse import extract_geometry_data, get_ifc_model
from rag_tag.paths import find_ifc_dir, find_project_root

DEFAULT_GRAPH_DATASET = "Building-Architecture"
LOG = logging.getLogger(__name__)

try:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    OCC_AVAILABLE = True
except Exception:
    OCC_AVAILABLE = False

script_dir = Path(__file__).resolve().parent
project_root = find_project_root(script_dir) or script_dir
ifc_dir = find_ifc_dir(script_dir)
if ifc_dir is None:
    raise FileNotFoundError("Could not find 'IFC-Files/' folder.")

# Pick the IFC file dynamically
ifc_file = next(ifc_dir.glob("Building-Architecture.ifc"), None)
if ifc_file is None:
    raise FileNotFoundError("IFC file not found in IFC-Files/ folder.")

# Ensure output directory exists
csv_dir = project_root / "output"
csv_dir.mkdir(exist_ok=True)
csv_file = csv_dir / "Building-Architecture.csv"

model = get_ifc_model(ifc_file)
geom_data = extract_geometry_data(model)


def distance_between_points(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def distance_between_bboxes(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    """Minimum Euclidean distance between axis-aligned bboxes (0 if overlapping)."""
    (amin, amax) = a
    (bmin, bmax) = b
    axis_gaps = []
    for i in range(3):
        gap = max(bmin[i] - amax[i], amin[i] - bmax[i], 0.0)
        axis_gaps.append(gap)
    return float(np.linalg.norm(np.array(axis_gaps, dtype=float)))


def _extract_centroid(
    geom: object | None,
) -> tuple[float, float, float] | None:
    if isinstance(geom, dict):
        centroid = geom.get("centroid")
        if centroid is None:
            return None
        return tuple(float(v) for v in centroid)
    if geom is None:
        return None
    return tuple(float(v) for v in geom)


def _extract_bbox(
    geom: object | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not isinstance(geom, dict):
        return None
    bbox = geom.get("bbox")
    if bbox is None:
        return None
    min_xyz, max_xyz = bbox
    return (
        tuple(float(v) for v in min_xyz),
        tuple(float(v) for v in max_xyz),
    )


def _extract_mesh(
    geom: object | None,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]] | None:
    if not isinstance(geom, dict):
        return None
    vertices = geom.get("mesh_vertices")
    faces = geom.get("mesh_faces")
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


def _bbox_center(
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float]:
    (min_xyz, max_xyz) = bbox
    return (
        (min_xyz[0] + max_xyz[0]) / 2.0,
        (min_xyz[1] + max_xyz[1]) / 2.0,
        (min_xyz[2] + max_xyz[2]) / 2.0,
    )


def _bbox_xy_overlap_area(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    ax0, ay0, _ = a[0]
    ax1, ay1, _ = a[1]
    bx0, by0, _ = b[0]
    bx1, by1, _ = b[1]
    overlap_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_y = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(overlap_x * overlap_y)


def _bboxes_intersect(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    (amin, amax) = a
    (bmin, bmax) = b
    for i in range(3):
        if amax[i] < bmin[i] or bmax[i] < amin[i]:
            return False
    return True


def _compute_common_metrics(shape_a, shape_b) -> tuple[float, float] | None:
    """Compute exact common volume and surface area using OCC booleans."""
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


def _normalize_positions(
    positions: list[tuple[float, float, float]],
) -> np.ndarray | None:
    """Convert positions to a contiguous float array with shape (n, 3)."""
    if not positions:
        return None
    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def _estimate_cell_size(positions: np.ndarray) -> float:
    """Heuristic grid cell size from model extent and point density."""
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extent = maxs - mins
    volume = float(np.prod(np.maximum(extent, 1e-9)))
    avg_spacing = volume ** (1.0 / 3.0) / max(len(positions) ** (1.0 / 3.0), 1.0)
    return max(avg_spacing, 0.5)


def _cell_for_point(point: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    return tuple(np.floor(point / cell_size).astype(int))


def _neighbor_cell_keys(key: tuple[int, int, int], radius: int):
    """Yield cell keys in the shell at Chebyshev radius `radius`."""
    if radius == 0:
        yield key
        return
    xr = range(key[0] - radius, key[0] + radius + 1)
    yr = range(key[1] - radius, key[1] + radius + 1)
    zr = range(key[2] - radius, key[2] + radius + 1)
    for cx, cy, cz in product(xr, yr, zr):
        if max(abs(cx - key[0]), abs(cy - key[1]), abs(cz - key[2])) == radius:
            yield (cx, cy, cz)


def _build_spatial_grid(
    positions: np.ndarray, cell_size: float
) -> dict[tuple[int, int, int], list[int]]:
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, point in enumerate(positions):
        grid[_cell_for_point(point, cell_size)].append(idx)
    return grid


def build_spatial_grid(
    nodes: dict[str, dict[str, object]],
    cell_size: float,
) -> dict[tuple[int, int, int], list[str]]:
    """Build a centroid-based spatial grid over element nodes."""
    grid: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for node_id, data in nodes.items():
        centroid = data.get("centroid")
        if not isinstance(centroid, tuple) or len(centroid) != 3:
            continue
        cell = tuple(int(c // cell_size) for c in centroid)
        grid[cell].append(node_id)
    return grid


def get_neighboring_cells(cell: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    cx, cy, cz = cell
    return [
        (cx + dx, cy + dy, cz + dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]


def _distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.dot(diff))


def _add_canonical_containment_edge(
    G: nx.DiGraph, container_id: str, child_id: str
) -> None:
    """Add containment in both canonical directions.

    - contains: container -> child
    - contained_in: child -> container
    """
    if container_id == child_id:
        return
    if container_id not in G or child_id not in G:
        return
    G.add_edge(container_id, child_id, relation="contains")
    G.add_edge(child_id, container_id, relation="contained_in")


def _resolve_node_ids_for_ifc_entity(
    entity: object | None,
    node_ids_by_gid: dict[str, list[str]],
) -> list[str]:
    """Resolve graph node IDs for an IFC entity, with root-node fallback."""
    if entity is None:
        return []
    gid = getattr(entity, "GlobalId", None)
    if isinstance(gid, str) and gid in node_ids_by_gid:
        return node_ids_by_gid[gid]

    cls = str(getattr(entity, "is_a", lambda: "")() or "")
    if cls == "IfcProject":
        return ["IfcProject"]
    if cls == "IfcBuilding":
        return ["IfcBuilding"]
    return []


def compute_adjacency_threshold(positions: list[tuple[float, float, float]]) -> float:
    """
    Derive a reasonable adjacency threshold from data.
    Uses median nearest-neighbor distance, scaled, with a small floor.
    """
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

    median_nn = float(np.median(nn_distances))
    return max(0.5, median_nn * 1.5)


def build_graph_with_properties(
    csv_path: str | Path,
    geom_data: dict,
    ifc_model: object | None = None,
) -> nx.DiGraph:
    """
    Build a hierarchical IFC graph and attach geometry data to nodes.
    `geom_data` should be a dict mapping GlobalId -> geometry info (e.g., centroid
    or bounding box)
    """
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()

    # Root nodes
    G.add_node("IfcProject", label="Project", class_="IfcProject", geometry=None)
    G.add_node("IfcBuilding", label="Building", class_="IfcBuilding", geometry=None)
    G.add_edge("IfcProject", "IfcBuilding", relation="aggregates")

    G.graph["edge_categories"] = {
        "hierarchy": ["aggregates", "contains", "contained_in"],
        "typing": ["typed_by"],
        "spatial": ["adjacent_to", "connected_to"],
        "topology": [
            "above",
            "below",
            "overlaps_xy",
            "intersects_bbox",
            "intersects_3d",
            "touches_surface",
        ],
    }

    storey_nodes_by_gid: dict[str, str] = {}
    storey_nodes_by_name: dict[str, list[str]] = {}
    node_ids_by_gid: dict[str, list[str]] = {}

    columns = list(df.columns)
    col_idx = {col: idx for idx, col in enumerate(columns)}
    class_idx = col_idx.get("Class")
    gid_idx = col_idx.get("GlobalId")
    name_idx = col_idx.get("Name")
    level_idx = col_idx.get("Level")
    type_idx = col_idx.get("TypeName")

    if class_idx is None or gid_idx is None:
        raise KeyError("CSV must include 'Class' and 'GlobalId' columns.")

    # Create storey nodes with stable IDs based on GlobalId.
    for row in df.itertuples(index=False, name=None):
        row_class = row[class_idx]
        if row_class != "IfcBuildingStorey":
            continue

        row_gid = row[gid_idx]
        if row_gid is None:
            continue
        gid = str(row_gid).strip()
        if not gid:
            continue

        row_name = row[name_idx] if name_idx is not None else None
        storey_name = (
            row_name.strip() if isinstance(row_name, str) and row_name.strip() else gid
        )
        row_props = dict(zip(columns, row))
        geom = geom_data.get(gid)
        centroid = _extract_centroid(geom)
        bbox = _extract_bbox(geom)

        node_id = f"Storey::{gid}"
        G.add_node(
            node_id,
            label=storey_name,
            class_="IfcBuildingStorey",
            properties=row_props,
            geometry=centroid,
            bbox=bbox,
            mesh=_extract_mesh(geom),
        )
        G.add_edge("IfcBuilding", node_id, relation="aggregates")

        storey_nodes_by_gid[gid] = node_id
        storey_nodes_by_name.setdefault(storey_name, []).append(node_id)
        node_ids_by_gid.setdefault(gid, []).append(node_id)

    # Create element nodes and containment edges.
    for row in df.itertuples(index=False, name=None):
        row_class = row[class_idx]
        if row_class == "IfcBuildingStorey":
            continue

        row_gid = row[gid_idx]
        if row_gid is None:
            continue
        gid = str(row_gid).strip()
        if not gid:
            continue

        row_name = row[name_idx] if name_idx is not None else None
        row_level = row[level_idx] if level_idx is not None else None
        row_type = row[type_idx] if type_idx is not None else None
        row_props = dict(zip(columns, row))
        geom = geom_data.get(gid)
        centroid = _extract_centroid(geom)
        bbox = _extract_bbox(geom)

        eid = f"Element::{gid}"
        label = (
            row_name.strip() if isinstance(row_name, str) and row_name.strip() else gid
        )
        type_label = (
            row_type.strip() if isinstance(row_type, str) and row_type.strip() else None
        )
        G.add_node(
            eid,
            label=label,
            class_=row_class,
            properties=row_props,
            type_label=type_label,
            geometry=centroid,
            bbox=bbox,
            z_min=(bbox[0][2] if bbox is not None else None),
            z_max=(bbox[1][2] if bbox is not None else None),
            height=((bbox[1][2] - bbox[0][2]) if bbox is not None else None),
            footprint_bbox_2d=(
                (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
                if bbox is not None
                else None
            ),
            mesh=_extract_mesh(geom),
        )
        node_ids_by_gid.setdefault(gid, []).append(eid)

        if isinstance(row_level, str):
            level = row_level.strip()
            if level in storey_nodes_by_gid:
                _add_canonical_containment_edge(G, storey_nodes_by_gid[level], eid)
            else:
                candidates = storey_nodes_by_name.get(level, [])
                if len(candidates) == 1:
                    _add_canonical_containment_edge(G, candidates[0], eid)

    # Prefer IFC-native containment relationships when available.
    if ifc_model is not None and hasattr(ifc_model, "by_type"):
        for rel in ifc_model.by_type("IfcRelContainedInSpatialStructure"):
            container_nodes = _resolve_node_ids_for_ifc_entity(
                getattr(rel, "RelatingStructure", None),
                node_ids_by_gid,
            )
            if not container_nodes:
                continue
            for child in getattr(rel, "RelatedElements", None) or []:
                child_nodes = _resolve_node_ids_for_ifc_entity(child, node_ids_by_gid)
                if not child_nodes:
                    continue
                for container_node in container_nodes:
                    for child_node in child_nodes:
                        _add_canonical_containment_edge(G, container_node, child_node)

    # Add explicit type-object nodes and typed_by edges.
    if ifc_model is not None and hasattr(ifc_model, "by_type"):
        for element in ifc_model.by_type("IfcElement"):
            element_gid = getattr(element, "GlobalId", None)
            if not element_gid:
                continue
            element_node = f"Element::{element_gid}"
            if element_node not in G:
                continue

            for rel in getattr(element, "IsTypedBy", None) or []:
                type_obj = getattr(rel, "RelatingType", None)
                if type_obj is None:
                    continue
                type_gid = getattr(type_obj, "GlobalId", None)
                if type_gid:
                    type_node = f"Type::{type_gid}"
                else:
                    type_name = getattr(type_obj, "Name", None) or type_obj.is_a()
                    type_node = f"Type::{type_name}"

                if type_node not in G:
                    G.add_node(
                        type_node,
                        label=getattr(type_obj, "Name", None) or type_obj.is_a(),
                        class_=type_obj.is_a(),
                        node_type="Type",
                    )

                G.add_edge(element_node, type_node, relation="typed_by")

    return G


def add_spatial_adjacency(
    G: nx.DiGraph, geom_data: dict, threshold: float | None = None
) -> float:
    """
    Add spatial edges between elements.
    - connected_to: bbox intersects/touches (distance = 0)
    - adjacent_to: within threshold distance
    Returns the threshold used.
    """
    element_nodes: list[str] = []
    positions: list[tuple[float, float, float]] = []
    bboxes: list[
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ] = []

    for n, d in G.nodes(data=True):
        if d.get("class_") in {
            "IfcBuilding",
            "IfcProject",
            "IfcBuildingStorey",
        }:
            continue
        if not n.startswith("Element::"):
            continue
        gid = d.get("properties", {}).get("GlobalId")
        geom = geom_data.get(gid)
        centroid = _extract_centroid(geom)
        bbox = _extract_bbox(geom)
        if centroid is None and bbox is None:
            continue
        if centroid is None and bbox is not None:
            centroid = _bbox_center(bbox)
        if centroid is None:
            continue
        element_nodes.append(n)
        positions.append(centroid)
        bboxes.append(bbox)

    if threshold is None:
        threshold = compute_adjacency_threshold(positions)

    pos = _normalize_positions(positions)
    if pos is None:
        return threshold

    cell_size = max(float(threshold), 1e-6)
    grid = _build_spatial_grid(pos, cell_size)
    seen_pairs: set[tuple[str, str]] = set()
    neighbor_radius = int(math.ceil(float(threshold) / cell_size)) + 1

    for i, node_a in enumerate(element_nodes):
        centroid_a = pos[i]
        bbox_a = bboxes[i]
        key = _cell_for_point(centroid_a, cell_size)
        for radius in range(neighbor_radius + 1):
            for neighbor_key in _neighbor_cell_keys(key, radius):
                for j in grid.get(neighbor_key, []):
                    if j == i:
                        continue
                    node_b = element_nodes[j]
                    pair = tuple(sorted((node_a, node_b)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    centroid_b = pos[j]
                    bbox_b = bboxes[j]

                    if bbox_a is not None and bbox_b is not None:
                        d = distance_between_bboxes(bbox_a, bbox_b)
                    else:
                        d = distance_between_points(centroid_a, centroid_b)

                    if d <= threshold:
                        relation = "connected_to" if d <= 1e-9 else "adjacent_to"
                        G.add_edge(node_a, node_b, relation=relation, distance=d)
                        G.add_edge(node_b, node_a, relation=relation, distance=d)

    return threshold


def add_topology_facts(G: nx.DiGraph, ifc_model: object | None = None) -> None:
    """
    Add topology-derived symbolic relations:
    - intersects_bbox (bidirectional)
    - overlaps_xy (bidirectional, with overlap_area_xy)
    - above / below (directed pair with vertical_gap, only with XY overlap)
    - intersects_3d (bidirectional, exact OCC boolean, stores intersection_volume)
    - touches_surface (bidirectional, exact OCC boolean, stores contact_area)
    """
    element_nodes = []
    element_bboxes = []
    element_gids = []
    for n, d in G.nodes(data=True):
        if not str(n).startswith("Element::"):
            continue
        bbox = d.get("bbox")
        if bbox is None:
            continue
        gid = d.get("properties", {}).get("GlobalId")
        if not gid:
            continue
        element_nodes.append(n)
        element_bboxes.append(bbox)
        element_gids.append(str(gid))

    occ_shape_by_gid: dict[str, object] = {}
    if OCC_AVAILABLE and ifc_model is not None and hasattr(ifc_model, "by_guid"):
        try:
            occ_settings = ifcopenshell.geom.settings()
            occ_settings.set(occ_settings.USE_WORLD_COORDS, True)
            occ_settings.set(occ_settings.DISABLE_OPENING_SUBTRACTIONS, True)
            occ_settings.set(occ_settings.USE_PYTHON_OPENCASCADE, True)
            for gid in element_gids:
                try:
                    elem = ifc_model.by_guid(gid)
                    if elem is None:
                        continue
                    shape = ifcopenshell.geom.create_shape(occ_settings, elem)
                    occ_shape_by_gid[gid] = shape.geometry
                except Exception as exc:
                    LOG.debug("Skipping OCC shape for %s: %s", gid, exc)
        except Exception as exc:
            LOG.warning(
                "OCC topology initialization failed; skipping exact 3D facts: %s", exc
            )
            occ_shape_by_gid = {}
    elif not OCC_AVAILABLE:
        LOG.warning(
            "pythonocc is not installed; "
            "exact intersects_3d/touches_surface facts disabled."
        )

    for i, a in enumerate(element_nodes):
        bbox_a = element_bboxes[i]
        gid_a = element_gids[i]
        for j in range(i + 1, len(element_nodes)):
            b = element_nodes[j]
            bbox_b = element_bboxes[j]
            gid_b = element_gids[j]

            # 3D intersection fact
            if _bboxes_intersect(bbox_a, bbox_b):
                G.add_edge(
                    a,
                    b,
                    relation="intersects_bbox",
                    source="topology",
                )
                G.add_edge(
                    b,
                    a,
                    relation="intersects_bbox",
                    source="topology",
                )

            # 2D footprint overlap fact
            overlap_area = _bbox_xy_overlap_area(bbox_a, bbox_b)
            if overlap_area > 0.0:
                G.add_edge(
                    a,
                    b,
                    relation="overlaps_xy",
                    overlap_area_xy=overlap_area,
                    source="topology",
                )
                G.add_edge(
                    b,
                    a,
                    relation="overlaps_xy",
                    overlap_area_xy=overlap_area,
                    source="topology",
                )

            if bbox_a is not None and bbox_b is not None:
                shape_a = occ_shape_by_gid.get(gid_a)
                shape_b = occ_shape_by_gid.get(gid_b)
                if shape_a is not None and shape_b is not None:
                    metrics = _compute_common_metrics(shape_a, shape_b)
                    if metrics is not None:
                        intersection_volume, contact_area = metrics
                        if intersection_volume > 1e-9:
                            G.add_edge(
                                a,
                                b,
                                relation="intersects_3d",
                                source="occ_boolean",
                                intersection_volume=intersection_volume,
                                contact_area=contact_area,
                            )
                            G.add_edge(
                                b,
                                a,
                                relation="intersects_3d",
                                source="occ_boolean",
                                intersection_volume=intersection_volume,
                                contact_area=contact_area,
                            )
                        elif contact_area > 1e-6:
                            G.add_edge(
                                a,
                                b,
                                relation="touches_surface",
                                source="occ_boolean",
                                intersection_volume=intersection_volume,
                                contact_area=contact_area,
                            )
                            G.add_edge(
                                b,
                                a,
                                relation="touches_surface",
                                source="occ_boolean",
                                intersection_volume=intersection_volume,
                                contact_area=contact_area,
                            )

            # Vertical ordering facts based on z extents.
            # Gate by XY overlap to avoid far-away false positives.
            if overlap_area <= 0.0:
                continue
            a_min_z = float(bbox_a[0][2])
            a_max_z = float(bbox_a[1][2])
            b_min_z = float(bbox_b[0][2])
            b_max_z = float(bbox_b[1][2])

            if a_min_z > b_max_z:
                gap = a_min_z - b_max_z
                G.add_edge(a, b, relation="above", vertical_gap=gap, source="topology")
                G.add_edge(b, a, relation="below", vertical_gap=gap, source="topology")
            elif b_min_z > a_max_z:
                gap = b_min_z - a_max_z
                G.add_edge(b, a, relation="above", vertical_gap=gap, source="topology")
                G.add_edge(a, b, relation="below", vertical_gap=gap, source="topology")


def plot_interactive_graph(G: nx.DiGraph, out_html: Path):
    """
    Plot the IFC graph in 3D using real geometry coordinates if available.
    Nodes without geometry are slightly offset to avoid overlap.
    """
    # Build positions from geometry
    pos = {}
    for n, d in G.nodes(data=True):
        geom = d.get("geometry")
        if geom is not None:
            pos[n] = tuple(geom)
        else:
            pos[n] = None

    # Place nodes with no geometry at centroid of their children (if any)
    for n in G.nodes:
        if pos.get(n) is not None:
            continue
        child_positions = [pos[c] for c in G.successors(n) if pos.get(c) is not None]
        if child_positions:
            child_positions = np.array(child_positions, dtype=float)
            pos[n] = tuple(child_positions.mean(axis=0))
        else:
            pos[n] = (0.0, 0.0, 0.0)

    node_color_map = {
        "IfcProject": "purple",
        "IfcBuilding": "blue",
        "IfcBuildingStorey": "orange",
    }
    node_category_map = {
        "IfcProject": "Project root",
        "IfcBuilding": "Building container",
        "IfcBuildingStorey": "Storey / floor container",
    }
    edge_color_map = {
        "aggregates": "#6b7280",
        "contains": "#1d4ed8",
        "contained_in": "#2563eb",
        "typed_by": "#ca8a04",
        "connected_to": "#ef4444",
        "adjacent_to": "#059669",
        "intersects_3d": "#b91c1c",
        "touches_surface": "#9333ea",
    }
    edge_relation_explanations = {
        "aggregates": "parent decomposes into child",
        "contains": "container directly contains child",
        "contained_in": "element belongs to a storey/space",
        "typed_by": "element is classified by a type object",
        "connected_to": "bboxes intersect or touch",
        "adjacent_to": "elements are spatially near each other",
        "intersects_3d": "exact OCC-derived 3D intersection",
        "touches_surface": "exact OCC-derived surface contact",
    }

    node_groups: dict[str, dict[str, list]] = {}
    node_group_colors: dict[str, str] = {}
    node_group_labels: dict[str, str] = {}
    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        cls = str(d.get("class_") or "Unknown")
        is_type_node = str(d.get("node_type", "")).lower() == "type"
        group_key = f"Type::{cls}" if is_type_node else cls
        group = node_groups.setdefault(
            group_key,
            {
                "x": [],
                "y": [],
                "z": [],
                "hover": [],
            },
        )
        if group_key not in node_group_colors:
            node_group_colors[group_key] = (
                "#eab308" if is_type_node else node_color_map.get(cls, "green")
            )
        node_group_labels[group_key] = (
            f"Type Node: {cls}" if is_type_node else f"Node: {cls}"
        )

        props = d.get("properties", {})
        hover_text = "<br>".join(
            f"<b>{k}</b>: {v}" for k, v in props.items() if v not in ("", None)
        )
        category_label = (
            "Type object"
            if is_type_node
            else node_category_map.get(cls, "Physical element / other IFC class")
        )
        group["x"].append(x)
        group["y"].append(y)
        group["z"].append(z)
        group["hover"].append(
            f"<b>{d.get('label', '')}</b><br>"
            f"Class: {cls}<br>"
            "Category: "
            f"{category_label}<br>"
            f"{hover_text}"
        )

    edge_groups: dict[str, dict[str, list]] = {}
    for u, v, d in G.edges(data=True):
        rel = str(d.get("relation", "related_to"))
        group = edge_groups.setdefault(
            rel,
            {
                "x": [],
                "y": [],
                "z": [],
                "mid_x": [],
                "mid_y": [],
                "mid_z": [],
                "x0": [],
                "y0": [],
                "z0": [],
                "x1": [],
                "y1": [],
                "z1": [],
                "label_x": [],
                "label_y": [],
                "label_z": [],
                "hover": [],
            },
        )
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]

        group["x"] += [x0, x1, None]
        group["y"] += [y0, y1, None]
        group["z"] += [z0, z1, None]
        group["mid_x"].append((x0 + x1) / 2.0)
        group["mid_y"].append((y0 + y1) / 2.0)
        group["mid_z"].append((z0 + z1) / 2.0)
        group["x0"].append(x0)
        group["y0"].append(y0)
        group["z0"].append(z0)
        group["x1"].append(x1)
        group["y1"].append(y1)
        group["z1"].append(z1)
        dist = d.get("distance")
        meaning = edge_relation_explanations.get(rel, "Generic graph relationship.")
        if dist is not None:
            group["hover"].append(
                f"Relation: {rel}<br>Meaning: {meaning}<br>Distance: {dist:.3f}"
            )
        else:
            group["hover"].append(f"Relation: {rel}<br>Meaning: {meaning}")

    xs = [coord[0] for coord in pos.values()]
    ys = [coord[1] for coord in pos.values()]
    zs = [coord[2] for coord in pos.values()]
    span_x = max(xs) - min(xs) if xs else 1.0
    span_y = max(ys) - min(ys) if ys else 1.0
    span_z = max(zs) - min(zs) if zs else 1.0
    scene_span = max(span_x, span_y, span_z, 1.0)
    base_label_offset = max(0.15, scene_span * 0.008)
    bucket_size = base_label_offset * 2.0
    plane_padding = max(0.25, scene_span * 0.02)

    storey_spans = []
    for node_id, node_data in G.nodes(data=True):
        if node_data.get("class_") != "IfcBuildingStorey":
            continue

        child_positions = []
        child_bboxes = []
        for child_id in G.successors(node_id):
            if not str(child_id).startswith("Element::"):
                continue
            child_pos = pos.get(child_id)
            if child_pos is not None:
                child_positions.append(child_pos)
            child_bbox = G.nodes[child_id].get("bbox")
            if child_bbox is not None:
                child_bboxes.append(child_bbox)

        if not child_positions:
            continue

        if child_bboxes:
            mins = np.array([bbox[0] for bbox in child_bboxes], dtype=float)
            maxs = np.array([bbox[1] for bbox in child_bboxes], dtype=float)
            min_x = float(mins[:, 0].min()) - plane_padding
            max_x = float(maxs[:, 0].max()) + plane_padding
            min_y = float(mins[:, 1].min()) - plane_padding
            max_y = float(maxs[:, 1].max()) + plane_padding
            z_bottom = float(mins[:, 2].min())
            z_top = float(maxs[:, 2].max())
        else:
            child_arr = np.array(child_positions, dtype=float)
            min_x = float(child_arr[:, 0].min()) - plane_padding
            max_x = float(child_arr[:, 0].max()) + plane_padding
            min_y = float(child_arr[:, 1].min()) - plane_padding
            max_y = float(child_arr[:, 1].max()) + plane_padding
            z_bottom = float(child_arr[:, 2].min())
            z_top = float(child_arr[:, 2].max())

        storey_spans.append(
            {
                "label": str(node_data.get("label", node_id)),
                "x": [min_x, max_x, max_x, min_x, min_x],
                "y": [min_y, min_y, max_y, max_y, min_y],
                "z_bottom": [z_bottom] * 5,
                "z_top": [z_top] * 5,
                "count": len(child_positions),
            }
        )

    buckets: dict[tuple[int, int, int], list[tuple[str, int]]] = {}
    for rel, edge in edge_groups.items():
        midpoints = zip(edge["mid_x"], edge["mid_y"], edge["mid_z"])
        for i, (mx, my, mz) in enumerate(midpoints):
            key = (
                int(round(mx / bucket_size)),
                int(round(my / bucket_size)),
                int(round(mz / bucket_size)),
            )
            buckets.setdefault(key, []).append((rel, i))

    for key in sorted(buckets):
        refs = sorted(buckets[key], key=lambda item: (item[0], item[1]))
        total = len(refs)
        for order, (rel, idx) in enumerate(refs):
            edge = edge_groups[rel]
            x0 = edge["x0"][idx]
            y0 = edge["y0"][idx]
            z0 = edge["z0"][idx]
            x1 = edge["x1"][idx]
            y1 = edge["y1"][idx]
            z1 = edge["z1"][idx]
            mid = np.array([edge["mid_x"][idx], edge["mid_y"][idx], edge["mid_z"][idx]])

            direction = np.array([x1 - x0, y1 - y0, z1 - z0], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                direction = direction / norm

            axis = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(direction, axis))) > 0.85:
                axis = np.array([0.0, 1.0, 0.0], dtype=float)
            u = np.cross(direction, axis)
            if float(np.linalg.norm(u)) < 1e-9:
                u = np.cross(direction, np.array([1.0, 0.0, 0.0], dtype=float))
            u_norm = float(np.linalg.norm(u))
            if u_norm < 1e-9:
                u = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                u = u / u_norm
            v = np.cross(direction, u)
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-9:
                v = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                v = v / v_norm

            if total == 1:
                angle_seed = sum(ord(ch) for ch in rel)
                angle = math.radians(float(angle_seed % 360))
                radius = base_label_offset * 0.6
            else:
                ring = order // 8
                slot = order % 8
                angle = (2.0 * math.pi * slot) / 8.0
                radius = base_label_offset * (1.0 + 0.5 * ring)

            offset_vec = (math.cos(angle) * u + math.sin(angle) * v) * radius
            label_pos = mid + offset_vec
            edge["label_x"].append(float(label_pos[0]))
            edge["label_y"].append(float(label_pos[1]))
            edge["label_z"].append(float(label_pos[2]))

    traces = []
    trace_meta: list[tuple[str, str]] = []

    bbox_x = []
    bbox_y = []
    bbox_z = []
    bbox_edge_pairs = [
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
    ]
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
        for a, b in bbox_edge_pairs:
            ax, ay, az = corners[a]
            bx, by, bz = corners[b]
            bbox_x += [ax, bx, None]
            bbox_y += [ay, by, None]
            bbox_z += [az, bz, None]

    if bbox_x:
        traces.append(
            go.Scatter3d(
                x=bbox_x,
                y=bbox_y,
                z=bbox_z,
                mode="lines",
                line=dict(width=2, color="#f59e0b"),
                opacity=0.5,
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

    for i, span in enumerate(storey_spans):
        traces.append(
            go.Scatter3d(
                x=span["x"],
                y=span["y"],
                z=span["z_bottom"],
                mode="lines",
                line=dict(width=5, color="#fb923c"),
                hovertemplate=(
                    f"Storey: {span['label']}<br>"
                    f"Span: bottom<br>"
                    f"Elements: {span['count']}<extra></extra>"
                ),
                name="Storey span (bottom/top)",
                legendgroup="storey_plane",
                showlegend=i == 0,
            )
        )
        trace_meta.append(("storey_plane", "IfcBuildingStorey"))

        traces.append(
            go.Scatter3d(
                x=span["x"],
                y=span["y"],
                z=span["z_top"],
                mode="lines",
                line=dict(width=5, color="#fb923c"),
                hovertemplate=(
                    f"Storey: {span['label']}<br>"
                    f"Span: top<br>"
                    f"Elements: {span['count']}<extra></extra>"
                ),
                name="Storey span (bottom/top)",
                legendgroup="storey_plane",
                showlegend=False,
            )
        )
        trace_meta.append(("storey_plane", "IfcBuildingStorey"))

    for rel in sorted(edge_groups):
        edge = edge_groups[rel]
        edge_color = edge_color_map.get(rel, "#4b5563")
        rel_expl = edge_relation_explanations.get(rel, "generic relationship")
        traces.append(
            go.Scatter3d(
                x=edge["x"],
                y=edge["y"],
                z=edge["z"],
                mode="lines",
                line=dict(width=3, color=edge_color),
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
                marker=dict(size=2, color=edge_color, opacity=0.0),
                hoverinfo="text",
                hovertext=edge["hover"],
                showlegend=False,
                legendgroup=f"edge::{rel}",
            )
        )
        trace_meta.append(("edge_hover", rel))

        edge_labels = []
        for hover_line in edge["hover"]:
            rel_text = hover_line.split("<br>", 1)[0].replace("Relation: ", "")
            edge_labels.append(rel_text)
        traces.append(
            go.Scatter3d(
                x=edge["label_x"],
                y=edge["label_y"],
                z=edge["label_z"],
                mode="text",
                text=edge_labels,
                textposition="middle center",
                textfont=dict(size=10, color=edge_color),
                hoverinfo="none",
                showlegend=False,
                legendgroup=f"edge::{rel}",
                visible=False,
            )
        )
        trace_meta.append(("edge_label", rel))

    for group_key in sorted(node_groups):
        node = node_groups[group_key]
        node_color = node_group_colors.get(group_key, "green")
        label = node_group_labels.get(group_key, f"Node: {group_key}")
        traces.append(
            go.Scatter3d(
                x=node["x"],
                y=node["y"],
                z=node["z"],
                mode="markers",
                marker=dict(size=6, color=node_color),
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
        hierarchy_rels = set(
            edge_categories.get("hierarchy", ["aggregates", "contains", "contained_in"])
        )
        typing_rels = set(edge_categories.get("typing", ["typed_by"]))
        spatial_rels = set(
            edge_categories.get("spatial", ["adjacent_to", "connected_to"])
        )
        visible = []
        for kind, name in trace_meta:
            is_edge_label = kind == "edge_label"
            is_bbox = kind == "bbox"
            is_mesh = kind == "mesh"
            if is_bbox:
                visible.append(show_bboxes)
                continue
            if is_mesh:
                visible.append(show_meshes)
                continue
            if mode == "all":
                visible.append(not is_edge_label or show_edge_annotations)
            elif mode == "nodes":
                visible.append(kind in {"node", "storey_plane"})
            elif mode == "edges":
                visible.append(
                    kind in {"edge", "edge_hover"}
                    or (is_edge_label and show_edge_annotations)
                )
            elif mode == "hierarchy":
                visible.append(
                    kind in {"node", "storey_plane"}
                    or (kind in {"edge", "edge_hover"} and name in hierarchy_rels)
                    or (
                        is_edge_label
                        and show_edge_annotations
                        and name in hierarchy_rels
                    )
                )
            elif mode == "spatial":
                visible.append(
                    kind in {"node", "storey_plane"}
                    or (kind in {"edge", "edge_hover"} and name in spatial_rels)
                    or (
                        is_edge_label and show_edge_annotations and name in spatial_rels
                    )
                )
            elif mode == "typing":
                visible.append(
                    kind in {"node", "storey_plane"}
                    or (kind in {"edge", "edge_hover"} and name in typing_rels)
                    or (is_edge_label and show_edge_annotations and name in typing_rels)
                )
            else:
                visible.append(not is_edge_label or show_edge_annotations)
        return visible

    def _mask_variants(mode: str) -> dict[str, list[bool]]:
        variants: dict[str, list[bool]] = {}
        for show_edge_annotations in (False, True):
            for show_bboxes in (False, True):
                for show_meshes in (False, True):
                    key_parts = []
                    if show_edge_annotations:
                        key_parts.append("annotations")
                    if show_bboxes:
                        key_parts.append("bboxes")
                    if show_meshes:
                        key_parts.append("meshes")
                    key = "base" if not key_parts else "with_" + "_and_".join(key_parts)
                    variants[key] = _mask(
                        mode,
                        show_edge_annotations=show_edge_annotations,
                        show_bboxes=show_bboxes,
                        show_meshes=show_meshes,
                    )
        return variants

    # Ensure output folder exists
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="IFC Hierarchy Graph (3D with Geometry)",
            x=0.01,
            xanchor="left",
        ),
        scene=dict(aspectmode="data"),
        showlegend=False,
        font=dict(
            family="Segoe UI, Tahoma, Arial, sans-serif", size=14, color="#1f3552"
        ),
        margin=dict(l=8, r=8, t=56, b=8),
        paper_bgcolor="#f7f9fc",
        plot_bgcolor="#f7f9fc",
    )

    filter_masks = {
        "all": _mask_variants("all"),
        "nodes": _mask_variants("nodes"),
        "edges": _mask_variants("edges"),
        "hierarchy": _mask_variants("hierarchy"),
        "spatial": _mask_variants("spatial"),
        "typing": _mask_variants("typing"),
    }

    edge_items = []
    for rel in sorted(edge_groups):
        rel_expl = edge_relation_explanations.get(rel, "generic relationship")
        edge_color = edge_color_map.get(rel, "#4b5563")
        edge_items.append(
            "<div class='legend-item'>"
            f"<span class='swatch line' style='--swatch:{edge_color}'></span>"
            f"<span>Edge: {html.escape(rel)} - {html.escape(rel_expl)}</span>"
            "</div>"
        )

    node_items = []
    for group_key in sorted(node_groups):
        node_color = node_group_colors.get(group_key, "green")
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

    color_category_items = [
        ("purple", "Project root"),
        ("blue", "Building container"),
        ("orange", "Storey/floor container and top+bottom span lines"),
        ("#eab308", "Type object nodes"),
        ("green", "Physical elements / other IFC classes"),
    ]
    color_items = "".join(
        (
            "<div class='legend-item'>"
            f"<span class='swatch dot' style='--swatch:{color}'></span>"
            f"<span>{html.escape(label)}</span>"
            "</div>"
        )
        for color, label in color_category_items
    )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IFC Hierarchy Graph</title>
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
    * {{
      box-sizing: border-box;
    }}
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
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      line-height: 1.3;
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
      <button data-mode="typing">Typing</button>
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
        <div class="section-title">Color Categories</div>
        {color_items}
        <div class="section-title">Edges</div>
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
    out_html.write_text(page_html, encoding="utf-8")


def print_ifc_hierarchy(ifc_file_path, indent=0):
    """
    Recursively prints the hierarchy of an IFC file:
    Project → Buildings → Storeys → Elements
    """
    model = ifcopenshell.open(str(ifc_file_path))

    def print_with_indent(name, level):
        print("    " * level + f"- {name}")

    def traverse(obj, level):
        obj_name = getattr(obj, "Name", None) or obj.is_a()
        print_with_indent(f"{obj.is_a()}: {obj_name}", level)

        # Follow aggregation (Project → Site → Building → Storey → Space)
        if hasattr(obj, "IsDecomposedBy"):
            for rel in obj.IsDecomposedBy or []:
                for child in rel.RelatedObjects or []:
                    traverse(child, level + 1)

        # Follow containment (Storey → Elements)
        if hasattr(obj, "ContainsElements"):
            for rel in obj.ContainsElements or []:
                for elem in rel.RelatedElements or []:
                    elem_name = getattr(elem, "Name", None) or elem.is_a()
                    print_with_indent(f"{elem.is_a()}: {elem_name}", level + 1)

    # Start from the project(s)
    for project in model.by_type("IfcProject"):
        traverse(project, indent)


def build_graph(
    csv_path: Path | None = None,
    ifc_path: Path | None = None,
    dataset: str | None = None,
) -> nx.DiGraph:
    """
    Build and return the IFC graph. If csv_path and ifc_path are not provided,
    they will be auto-detected from the project structure.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir) or script_dir

    dataset_name = dataset.strip() if dataset else DEFAULT_GRAPH_DATASET
    expected_csv_path = project_root / "output" / f"{dataset_name}.csv"
    ifc_dir = find_ifc_dir(script_dir)
    expected_ifc_path = (
        ifc_dir / f"{dataset_name}.ifc"
        if ifc_dir is not None
        else project_root / "IFC-Files" / f"{dataset_name}.ifc"
    )

    csv_inferred = csv_path is None
    if csv_path is None:
        csv_path = expected_csv_path

    ifc_inferred = ifc_path is None
    if ifc_path is None:
        ifc_path = expected_ifc_path

    if (csv_inferred and not csv_path.is_file()) or (
        ifc_inferred and not ifc_path.is_file()
    ):
        raise FileNotFoundError(
            f"Graph dataset '{dataset_name}' could not be resolved. "
            f"Expected CSV: {expected_csv_path}; "
            f"Expected IFC: {expected_ifc_path}. "
            "Pass --graph-dataset <name> or --db pointing to a matching dataset."
        )

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not ifc_path.is_file():
        raise FileNotFoundError(f"IFC file not found: {ifc_path}")

    model = get_ifc_model(ifc_path)
    geom_data = extract_geometry_data(model)

    # Convert list to dict
    geom_dict = {}
    for item in geom_data:
        gid = item.get("GlobalId")
        if gid:
            geom_dict[gid] = {
                "centroid": item.get("centroid"),
                "bbox": item.get("bbox"),
                "mesh_vertices": item.get("mesh_vertices"),
                "mesh_faces": item.get("mesh_faces"),
            }

    G = build_graph_with_properties(csv_path, geom_dict, ifc_model=model)
    add_spatial_adjacency(G, geom_dict)
    add_topology_facts(G, ifc_model=model)
    return G


def main() -> None:
    """Build the graph and generate visualization HTML."""
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir) or script_dir
    ifc_dir = find_ifc_dir(script_dir)

    if ifc_dir is None:
        raise FileNotFoundError("Could not find 'IFC-Files/' folder.")

    ifc_file = next(ifc_dir.glob(f"{DEFAULT_GRAPH_DATASET}.ifc"), None)
    if ifc_file is None:
        raise FileNotFoundError("IFC file not found in IFC-Files/ folder.")

    csv_dir = project_root / "output"
    csv_dir.mkdir(exist_ok=True)
    csv_file = csv_dir / f"{DEFAULT_GRAPH_DATASET}.csv"

    html_dir = project_root / "output"
    html_dir.mkdir(parents=True, exist_ok=True)
    html_file = html_dir / "ifc_graph.html"

    G = build_graph(csv_file, ifc_file)
    plot_interactive_graph(G, html_file)
    print_ifc_hierarchy(ifc_file)
    print(
        f"\nGraph built with {G.number_of_nodes()} nodes "
        f"and {G.number_of_edges()} edges."
    )
    print(f"Visualization saved to {html_file}")


if __name__ == "__main__":
    main()
