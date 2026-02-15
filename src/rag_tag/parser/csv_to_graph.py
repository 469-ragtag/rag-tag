from __future__ import annotations

import math
from collections import defaultdict
from itertools import product
from pathlib import Path

import ifcopenshell
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rag_tag.parser.ifc_geometry_parse import extract_geometry_data, get_ifc_model
from rag_tag.paths import find_ifc_dir, find_project_root


def distance_between_points(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


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
        if (
            max(abs(cx - key[0]), abs(cy - key[1]), abs(cz - key[2]))
            == radius
        ):
            yield (cx, cy, cz)


def _build_spatial_grid(
    positions: np.ndarray, cell_size: float
) -> dict[tuple[int, int, int], list[int]]:
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, point in enumerate(positions):
        grid[_cell_for_point(point, cell_size)].append(idx)
    return grid


def _distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.dot(diff))


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


def build_graph_with_properties(csv_path: str | Path, geom_data: dict) -> nx.DiGraph:
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

    # Collect actual IfcBuildingStorey elements from the CSV
    storey_series = df.loc[df["Class"] == "IfcBuildingStorey", "Name"].dropna()
    actual_storeys = [str(name) for name in storey_series.unique()]

    storey_nodes: dict[str, str] = {}

    # Create Storey nodes only for actual IfcBuildingStorey elements
    for storey_name in actual_storeys:
        node_id = f"Storey::{storey_name}"
        storey_nodes[storey_name] = node_id
        G.add_node(
            node_id, label=storey_name, class_="IfcBuildingStorey", geometry=None
        )
        G.add_edge("IfcBuilding", node_id, relation="aggregates")

    # Types
    for t in df["TypeName"].dropna().unique():
        node_id = f"Type::{t}"
        G.add_node(node_id, label=t, class_="IfcTypeObject", geometry=None)

    # Elements (itertuples is significantly faster than iterrows for large CSVs)
    columns = list(df.columns)
    col_idx = {col: idx for idx, col in enumerate(columns)}
    class_idx = col_idx.get("Class")
    gid_idx = col_idx.get("GlobalId")
    name_idx = col_idx.get("Name")
    level_idx = col_idx.get("Level")
    type_idx = col_idx.get("TypeName")

    if class_idx is None or gid_idx is None:
        raise KeyError("CSV must include 'Class' and 'GlobalId' columns.")

    for row in df.itertuples(index=False, name=None):
        row_class = row[class_idx]
        row_gid = row[gid_idx]
        row_name = row[name_idx] if name_idx is not None else None
        row_level = row[level_idx] if level_idx is not None else None
        row_type = row[type_idx] if type_idx is not None else None
        row_props = dict(zip(columns, row))
        geom = geom_data.get(row_gid)  # attach geometry if available

        if row_class == "IfcBuildingStorey":
            storey_name = row_name
            if not isinstance(storey_name, str) or not storey_name.strip():
                storey_name = str(row_gid)
            node_id = storey_nodes.get(storey_name)
            if node_id is None:
                node_id = f"Storey::{storey_name}"
                storey_nodes[storey_name] = node_id
                G.add_node(
                    node_id,
                    label=storey_name,
                    class_="IfcBuildingStorey",
                    geometry=None,
                )
                G.add_edge("IfcBuilding", node_id, relation="aggregates")
            G.nodes[node_id].update(
                label=storey_name,
                class_="IfcBuildingStorey",
                properties=row_props,
                geometry=geom,
            )
            continue

        eid = f"Element::{row_gid}"
        label = row_name if isinstance(row_name, str) and row_name else row_gid
        G.add_node(
            eid,
            label=label,
            class_=row_class,
            properties=row_props,
            geometry=geom,  # new geometry property
        )

        level_value = row_level
        if isinstance(level_value, str) and level_value in storey_nodes:
            G.add_edge(storey_nodes[level_value], eid, relation="contained_in")

        type_name = row_type
        if isinstance(type_name, str) and type_name:
            G.add_edge(f"Type::{type_name}", eid, relation="typed_by")

    return G


def add_spatial_adjacency(
    G: nx.DiGraph, geom_data: dict, threshold: float | None = None
) -> float:
    """
    Add adjacency edges between elements that are within a spatial threshold.
    Returns the threshold used.
    """
    element_nodes = []
    positions = []

    for n, d in G.nodes(data=True):
        if d.get("class_") in {
            "IfcTypeObject",
            "IfcBuilding",
            "IfcProject",
            "IfcBuildingStorey",
        }:
            continue
        if not n.startswith("Element::"):
            continue
        gid = d.get("properties", {}).get("GlobalId")
        geom = geom_data.get(gid)
        if geom is None:
            continue
        element_nodes.append(n)
        positions.append(tuple(geom))

    if threshold is None:
        threshold = compute_adjacency_threshold(positions)

    pos = _normalize_positions(positions)
    if pos is None:
        return threshold

    # Grid with threshold-sized cells reduces pair checks to nearby buckets.
    cell_size = max(float(threshold), 1e-6)
    grid = _build_spatial_grid(pos, cell_size)
    threshold_sq = float(threshold) * float(threshold)
    neighbor_radius = int(math.ceil(float(threshold) / cell_size)) + 1

    for i, ni in enumerate(element_nodes):
        pi = pos[i]
        key = _cell_for_point(pi, cell_size)
        for radius in range(neighbor_radius + 1):
            for neighbor_key in _neighbor_cell_keys(key, radius):
                for j in grid.get(neighbor_key, []):
                    if j <= i:
                        continue
                    nj = element_nodes[j]
                    d2 = _distance_sq(pi, pos[j])
                    if d2 <= threshold_sq:
                        if not G.has_edge(ni, nj) and not G.has_edge(nj, ni):
                            G.add_edge(
                                ni,
                                nj,
                                relation="adjacent_to",
                                distance=math.sqrt(d2),
                            )

    return threshold


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

    # Node coordinates
    xs, ys, zs, hover = [], [], [], []
    colors = []

    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        xs.append(x)
        ys.append(y)
        zs.append(z)

        props = d.get("properties", {})
        hover_text = "<br>".join(
            f"<b>{k}</b>: {v}" for k, v in props.items() if v not in ("", None)
        )

        hover.append(
            f"<b>{d.get('label', '')}</b><br>"
            f"Class: {d.get('class_', '')}<br>" + hover_text
        )

        colors.append(
            {
                "IfcProject": "purple",
                "IfcBuilding": "blue",
                "IfcBuildingStorey": "orange",
                "IfcTypeObject": "red",
            }.get(d.get("class_"), "green")
        )

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=6, color=colors),
        hoverinfo="text",
        hovertext=hover,
    )

    # Edge coordinates
    ex, ey, ez = [], [], []
    edge_mid_x, edge_mid_y, edge_mid_z = [], [], []
    edge_hover = []
    edge_text = []

    for u, v, d in G.edges(data=True):
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]

        ex += [x0, x1, None]
        ey += [y0, y1, None]
        ez += [z0, z1, None]

        edge_mid_x.append((x0 + x1) / 2.0)
        edge_mid_y.append((y0 + y1) / 2.0)
        edge_mid_z.append((z0 + z1) / 2.0)
        rel = d.get("relation", "related_to")
        dist = d.get("distance")
        if dist is not None:
            edge_hover.append(f"Relation: {rel}<br>Distance: {dist:.3f}")
        else:
            edge_hover.append(f"Relation: {rel}")
        edge_text.append(rel)

    edge_trace = go.Scatter3d(
        x=ex,
        y=ey,
        z=ez,
        mode="lines",
        line=dict(width=3, color="gray"),
        hoverinfo="none",
    )
    edge_hover_trace = go.Scatter3d(
        x=edge_mid_x,
        y=edge_mid_y,
        z=edge_mid_z,
        mode="markers",
        marker=dict(size=2, color="gray", opacity=0.0),
        hoverinfo="text",
        hovertext=edge_hover,
        showlegend=False,
    )
    edge_text_trace = go.Scatter3d(
        x=edge_mid_x,
        y=edge_mid_y,
        z=edge_mid_z,
        mode="text",
        text=edge_text,
        textfont=dict(size=9, color="gray"),
        hoverinfo="none",
        showlegend=False,
    )

    # Ensure output folder exists
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(data=[edge_trace, edge_hover_trace, edge_text_trace, node_trace])
    fig.update_layout(
        scene=dict(aspectmode="data"), title="IFC Hierarchy Graph (3D with Geometry)"
    )

    fig.write_html(out_html, auto_open=True)


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
    csv_path: Path | None = None, ifc_path: Path | None = None
) -> nx.DiGraph:
    """
    Build and return the IFC graph. If csv_path and ifc_path are not provided,
    they will be auto-detected from the project structure.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir) or script_dir

    if csv_path is None:
        csv_dir = project_root / "output"
        csv_path = csv_dir / "Building-Architecture.csv"

    if ifc_path is None:
        ifc_dir = find_ifc_dir(script_dir)
        if ifc_dir is None:
            raise FileNotFoundError("Could not find 'IFC-Files/' folder.")
        ifc_path = next(ifc_dir.glob("Building-Architecture.ifc"), None)
        if ifc_path is None:
            raise FileNotFoundError("IFC file not found in IFC-Files/ folder.")

    model = get_ifc_model(ifc_path)
    geom_data = extract_geometry_data(model)

    # Convert list to dict
    geom_dict = {}
    for item in geom_data:
        gid = item.get("GlobalId")
        if gid:
            geom_dict[gid] = item.get("centroid")

    G = build_graph_with_properties(csv_path, geom_dict)
    add_spatial_adjacency(G, geom_dict)
    return G


def main() -> None:
    """Build the graph and generate visualization HTML."""
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir) or script_dir
    ifc_dir = find_ifc_dir(script_dir)

    if ifc_dir is None:
        raise FileNotFoundError("Could not find 'IFC-Files/' folder.")

    ifc_file = next(ifc_dir.glob("Building-Architecture.ifc"), None)
    if ifc_file is None:
        raise FileNotFoundError("IFC file not found in IFC-Files/ folder.")

    csv_dir = project_root / "output"
    csv_dir.mkdir(exist_ok=True)
    csv_file = csv_dir / "Building-Architecture.csv"

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
