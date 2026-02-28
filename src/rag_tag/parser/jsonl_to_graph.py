# builds a NetworkX graph from the .jsonl files produced by ifc_to_jsonl.py
# each node represents an IFC element with its geometry and properties attached
# edges represent containment (which floor/space something is in),
# spatial proximity (adjacent_to / connected_to), topology (above/below/overlaps),
# and explicit IFC semantics (hosts, hosted_by, ifc_connected_to, belongs_to_system,
# in_zone, classified_as) sourced from the Relationships block added in Batch 0.
#
# Edge provenance:
#   source="ifc"       — explicit relationship extracted from IFC relations
#   source="heuristic" — spatial adjacency derived from centroid/bbox distance
#   source="topology"  — bbox intersection / vertical overlap (already present)
#
# run with: uv run rag-tag-jsonl-to-graph

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from rag_tag.paths import find_project_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)


# --- geometry math (no IFC dependency, just numpy) ---


def _distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.dot(diff))


def distance_between_bboxes(a: tuple, b: tuple) -> float:
    # minimum distance between two 3D bounding boxes — 0 if they overlap
    amin, amax = a
    bmin, bmax = b
    axis_gaps = [max(bmin[i] - amax[i], amin[i] - bmax[i], 0.0) for i in range(3)]
    return float(np.linalg.norm(np.array(axis_gaps, dtype=float)))


def distance_between_points(a: tuple, b: tuple) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _bbox_xy_overlap_area(a: tuple, b: tuple) -> float:
    # how much the footprints of two elements overlap in the XY plane
    ax0, ay0, _ = a[0]
    ax1, ay1, _ = a[1]
    bx0, by0, _ = b[0]
    bx1, by1, _ = b[1]
    overlap_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_y = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(overlap_x * overlap_y)


def _bboxes_intersect(a: tuple, b: tuple) -> bool:
    amin, amax = a
    bmin, bmax = b
    for i in range(3):
        if amax[i] < bmin[i] or bmax[i] < amin[i]:
            return False
    return True


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


def _add_explicit_relationships(
    G: nx.DiGraph,
    node_id: str,
    rels: dict,
    node_id_by_gid: dict[str, str],
) -> None:
    """Materialise the explicit IFC relationships from a record's Relationships block.

    Edges added here carry ``source="ifc"`` to distinguish them from heuristic
    spatial edges (``source="heuristic"``) and topology edges (``source="topology"``).

    Relationship semantics
    ----------------------
    hosts / hosted_by  — element-to-element directed; target resolved via GlobalId.
    ifc_connected_to   — undirected IFC connectivity; both directions added.
    belongs_to_system  — element → System context node (created on demand).
    in_zone            — element → Zone context node (created on demand).
    classified_as      — element → Classification context node (created on demand).
    """
    if node_id not in G:
        return
    if not isinstance(rels, dict):
        return

    # --- element-to-element: hosts ---
    for target_gid in rels.get("hosts") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            G.add_edge(node_id, target, relation="hosts", source="ifc")

    # --- element-to-element: hosted_by ---
    for target_gid in rels.get("hosted_by") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            G.add_edge(node_id, target, relation="hosted_by", source="ifc")

    # --- element-to-element: ifc_connected_to (undirected → both directions) ---
    for target_gid in rels.get("ifc_connected_to") or []:
        target = node_id_by_gid.get(target_gid)
        if target and target in G and target != node_id:
            G.add_edge(node_id, target, relation="ifc_connected_to", source="ifc")
            G.add_edge(target, node_id, relation="ifc_connected_to", source="ifc")

    # --- element → System context node ---
    for raw_label in rels.get("belongs_to_system") or []:
        label = _normalize_context_label(raw_label)
        if not label:
            continue
        system_nid = f"System::{label}"
        if system_nid not in G:
            G.add_node(
                system_nid,
                label=label,
                class_="IfcSystem",
                node_kind="context",
                geometry=None,
            )
        G.add_edge(node_id, system_nid, relation="belongs_to_system", source="ifc")

    # --- element → Zone context node ---
    for raw_label in rels.get("in_zone") or []:
        label = _normalize_context_label(raw_label)
        if not label:
            continue
        zone_nid = f"Zone::{label}"
        if zone_nid not in G:
            G.add_node(
                zone_nid,
                label=label,
                class_="IfcZone",
                node_kind="context",
                geometry=None,
            )
        G.add_edge(node_id, zone_nid, relation="in_zone", source="ifc")

    # --- element → Classification context node ---
    for raw_label in rels.get("classified_as") or []:
        label = _normalize_context_label(raw_label)
        if not label:
            continue
        cls_nid = f"Classification::{label}"
        if cls_nid not in G:
            G.add_node(
                cls_nid,
                label=label,
                class_="IfcClassificationReference",
                node_kind="context",
                geometry=None,
            )
        G.add_edge(node_id, cls_nid, relation="classified_as", source="ifc")


# --- graph construction ---


def _add_containment_edge(G: nx.DiGraph, parent_id: str, child_id: str) -> None:
    if parent_id == child_id:
        return
    if parent_id not in G or child_id not in G:
        return
    G.add_edge(parent_id, child_id, relation="contains")
    G.add_edge(child_id, parent_id, relation="contained_in")


def build_graph_from_jsonl(jsonl_paths: list[Path]) -> nx.DiGraph:
    G = nx.DiGraph()

    # these two root nodes always exist even if the JSONL doesn't mention them
    G.add_node("IfcProject", label="Project", class_="IfcProject", geometry=None)
    G.add_node("IfcBuilding", label="Building", class_="IfcBuilding", geometry=None)
    G.add_edge("IfcProject", "IfcBuilding", relation="aggregates")

    G.graph["edge_categories"] = {
        "hierarchy": ["aggregates", "contains", "contained_in"],
        "spatial": ["adjacent_to", "connected_to"],
        "topology": ["intersects_bbox", "overlaps_xy", "above", "below"],
        # explicit IFC relationships extracted from the Relationships block (Batch 0+)
        "explicit": [
            "hosts",
            "hosted_by",
            "ifc_connected_to",
            "belongs_to_system",
            "in_zone",
            "classified_as",
        ],
    }

    node_id_by_gid: dict[str, str] = {}

    # we defer containment edges to a second pass because when we read element A,
    # its parent (element B) might not have been added to the graph yet
    deferred_containment: list[tuple[str, str]] = []  # (parent_gid, child_node_id)

    # we also defer explicit relationships: all primary nodes must exist first so
    # that target GlobalIds can be resolved to their graph node IDs
    deferred_relationships: list[tuple[str, dict]] = []  # (node_id, relationships)

    for jsonl_path in jsonl_paths:
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

                if ifc_type == "IfcProject":
                    node_id = "IfcProject"
                    G.nodes["IfcProject"].update(
                        label=name,
                        class_=ifc_type,
                        payload=rec,
                        properties=_flat_properties(rec),
                        geometry=centroid,
                        bbox=bbox,
                    )
                elif ifc_type == "IfcBuilding":
                    node_id = "IfcBuilding"
                    G.nodes["IfcBuilding"].update(
                        label=name,
                        class_=ifc_type,
                        payload=rec,
                        properties=_flat_properties(rec),
                        geometry=centroid,
                        bbox=bbox,
                    )
                elif ifc_type == "IfcBuildingStorey":
                    node_id = f"Storey::{gid}"
                    G.add_node(
                        node_id,
                        label=name,
                        class_=ifc_type,
                        properties=_flat_properties(rec),
                        geometry=centroid,
                        bbox=bbox,
                        z_min=(bbox[0][2] if bbox else None),
                        z_max=(bbox[1][2] if bbox else None),
                        height=((bbox[1][2] - bbox[0][2]) if bbox else None),
                        payload=rec,
                    )
                    G.add_edge("IfcBuilding", node_id, relation="aggregates")
                else:
                    node_id = f"Element::{gid}"
                    G.add_node(
                        node_id,
                        label=name,
                        class_=ifc_type,
                        properties=_flat_properties(rec),
                        geometry=centroid,
                        bbox=bbox,
                        z_min=(bbox[0][2] if bbox else None),
                        z_max=(bbox[1][2] if bbox else None),
                        height=((bbox[1][2] - bbox[0][2]) if bbox else None),
                        footprint_bbox_2d=(
                            (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
                            if bbox
                            else None
                        ),
                        payload=rec,
                    )

                node_id_by_gid[gid] = node_id

                parent_gid = (rec.get("Hierarchy") or {}).get("ParentId")
                if parent_gid:
                    deferred_containment.append((parent_gid, node_id))

                # collect non-empty Relationships blocks for the third pass
                rels = rec.get("Relationships")
                if isinstance(rels, dict) and any(rels.values()):
                    deferred_relationships.append((node_id, rels))

    # second pass — now all nodes exist so we can safely add containment edges
    for parent_gid, child_node_id in deferred_containment:
        parent_node_id = node_id_by_gid.get(parent_gid)
        if parent_node_id is None:
            continue
        _add_containment_edge(G, parent_node_id, child_node_id)

    # third pass — materialise explicit IFC relationships from Relationships blocks
    explicit_edge_count = 0
    _before = G.number_of_edges()
    for node_id, rels in deferred_relationships:
        _add_explicit_relationships(G, node_id, rels, node_id_by_gid)
    explicit_edge_count = G.number_of_edges() - _before
    if explicit_edge_count:
        LOG.info("Added %d explicit IFC relationship edge(s)", explicit_edge_count)

    LOG.info(
        "Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


def add_spatial_adjacency(G: nx.DiGraph, threshold: float | None = None) -> float:
    element_nodes: list[str] = []
    positions: list[tuple] = []
    bboxes: list[tuple | None] = []

    for n, d in G.nodes(data=True):
        if d.get("class_") in {"IfcBuilding", "IfcProject", "IfcBuildingStorey"}:
            continue
        if not n.startswith("Element::"):
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

    if threshold is None:
        threshold = compute_adjacency_threshold(positions)

    pos = _normalize_positions(positions)
    if pos is None:
        return threshold

    cell_size = max(float(threshold), 1e-6)
    grid = _build_spatial_grid(pos, cell_size)
    seen_pairs: set[tuple] = set()
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
                        # distance of 0 means bboxes are literally touching
                        relation = "connected_to" if d <= 1e-9 else "adjacent_to"
                        # Don't overwrite an explicit IFC edge with a heuristic one.
                        # NetworkX DiGraph only supports one edge per (u, v) pair, so
                        # if an explicit relationship (source="ifc") already exists for
                        # this pair we preserve it and skip the heuristic edge.
                        if (
                            not G.has_edge(node_a, node_b)
                            or G[node_a][node_b].get("source") != "ifc"
                        ):
                            G.add_edge(
                                node_a,
                                node_b,
                                relation=relation,
                                distance=d,
                                source="heuristic",
                            )
                        if (
                            not G.has_edge(node_b, node_a)
                            or G[node_b][node_a].get("source") != "ifc"
                        ):
                            G.add_edge(
                                node_b,
                                node_a,
                                relation=relation,
                                distance=d,
                                source="heuristic",
                            )

    return threshold


def add_topology_facts(G: nx.DiGraph) -> None:
    element_nodes = []
    element_bboxes = []

    for n, d in G.nodes(data=True):
        if not str(n).startswith("Element::"):
            continue
        bbox = d.get("bbox")
        if bbox is None:
            continue
        element_nodes.append(n)
        element_bboxes.append(bbox)

    for i, a in enumerate(element_nodes):
        bbox_a = element_bboxes[i]
        for j in range(i + 1, len(element_nodes)):
            b = element_nodes[j]
            bbox_b = element_bboxes[j]

            if _bboxes_intersect(bbox_a, bbox_b):
                G.add_edge(a, b, relation="intersects_bbox", source="topology")
                G.add_edge(b, a, relation="intersects_bbox", source="topology")

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

                # only check vertical order if footprints overlap —
                # otherwise "above/below" doesn't really mean anything
                a_min_z, a_max_z = float(bbox_a[0][2]), float(bbox_a[1][2])
                b_min_z, b_max_z = float(bbox_b[0][2]), float(bbox_b[1][2])
                if a_min_z > b_max_z:
                    gap = a_min_z - b_max_z
                    G.add_edge(
                        a, b, relation="above", vertical_gap=gap, source="topology"
                    )
                    G.add_edge(
                        b, a, relation="below", vertical_gap=gap, source="topology"
                    )
                elif b_min_z > a_max_z:
                    gap = b_min_z - a_max_z
                    G.add_edge(
                        b, a, relation="above", vertical_gap=gap, source="topology"
                    )
                    G.add_edge(
                        a, b, relation="below", vertical_gap=gap, source="topology"
                    )


def plot_interactive_graph(G: nx.DiGraph, out_html: Path) -> None:
    pos: dict[str, tuple] = {}
    for n, d in G.nodes(data=True):
        geom = d.get("geometry")
        pos[n] = tuple(geom) if geom is not None else None  # type: ignore[arg-type]

    # nodes with no geometry get placed at the average position of their children
    for n in G.nodes:
        if pos.get(n) is not None:
            continue
        child_positions = [pos[c] for c in G.successors(n) if pos.get(c) is not None]
        if child_positions:
            arr = np.array(child_positions, dtype=float)
            pos[n] = tuple(arr.mean(axis=0))
        else:
            pos[n] = (0.0, 0.0, 0.0)

    node_x, node_y, node_z, node_text, node_color = [], [], [], [], []
    for n, d in G.nodes(data=True):
        p = pos.get(n) or (0.0, 0.0, 0.0)
        node_x.append(p[0])
        node_y.append(p[1])
        node_z.append(p[2])
        label = d.get("label") or str(n)
        cls = d.get("class_") or ""
        node_text.append(f"{label}<br>{cls}")
        node_color.append(cls)

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        pu = pos.get(u) or (0.0, 0.0, 0.0)
        pv = pos.get(v) or (0.0, 0.0, 0.0)
        # None breaks the line between separate edges in Plotly
        edge_x += [pu[0], pv[0], None]
        edge_y += [pu[1], pv[1], None]
        edge_z += [pu[2], pv[2], None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line={"width": 0.5, "color": "#888"},
        hoverinfo="none",
        name="edges",
    )
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        marker={"size": 4, "opacity": 0.8},
        text=node_text,
        hoverinfo="text",
        name="elements",
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="IFC Graph (JSONL)",
            showlegend=False,
            scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
            margin={"l": 0, "r": 0, "b": 0, "t": 40},
        ),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, str(out_html))
    LOG.info("Visualization saved to %s", out_html)


def build_graph(
    jsonl_paths: list[Path] | None = None,
    dataset: str | None = None,
) -> nx.DiGraph:
    # auto-detect jsonl files if no paths given — called with no args from query_service
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

    G = build_graph_from_jsonl(jsonl_paths)
    add_spatial_adjacency(G)
    add_topology_facts(G)
    return G


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build IFC graph from JSONL and generate 3D visualization."
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
    G = build_graph(jsonl_files)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if not args.no_viz:
        html_path = out_dir / "ifc_graph.html"
        plot_interactive_graph(G, html_path)
        print(f"Visualization: {html_path}")


if __name__ == "__main__":
    main()
