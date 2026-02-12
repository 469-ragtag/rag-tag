import html
import json
import math
from pathlib import Path

import ifcopenshell
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from ifc_geometry_parse import extract_geometry_data, get_ifc_model
from ifc_to_csv import _find_ifc_dir, _find_project_root

script_dir = Path(__file__).resolve().parent
project_root = _find_project_root(script_dir) or script_dir
ifc_dir = _find_ifc_dir(script_dir)
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


def compute_adjacency_threshold(positions: list[tuple[float, float, float]]) -> float:
    """
    Derive a reasonable adjacency threshold from data.
    Uses median nearest-neighbor distance, scaled, with a small floor.
    """
    if len(positions) < 2:
        return 1.0

    nn_distances = []
    for i, p in enumerate(positions):
        best = None
        for j, q in enumerate(positions):
            if i == j:
                continue
            d = distance_between_points(p, q)
            if best is None or d < best:
                best = d
        if best is not None:
            nn_distances.append(best)

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

    # Elements
    for _, row in df.iterrows():
        row_class = row["Class"]
        geom = geom_data.get(row["GlobalId"])  # attach geometry if available

        if row_class == "IfcBuildingStorey":
            storey_name = row.get("Name")
            if not isinstance(storey_name, str) or not storey_name.strip():
                storey_name = str(row["GlobalId"])
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
                properties=row.to_dict(),
                geometry=geom,
            )
            continue

        eid = f"Element::{row['GlobalId']}"
        G.add_node(
            eid,
            label=row.get("Name", row["GlobalId"]),
            class_=row_class,
            properties=row.to_dict(),
            geometry=geom,  # new geometry property
        )

        level_value = row.get("Level")
        if isinstance(level_value, str) and level_value in storey_nodes:
            G.add_edge(storey_nodes[level_value], eid, relation="contained_in")

        type_name = row.get("TypeName")
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

    for i, ni in enumerate(element_nodes):
        pi = positions[i]
        for j in range(i + 1, len(element_nodes)):
            nj = element_nodes[j]
            pj = positions[j]
            d = distance_between_points(pi, pj)
            if d <= threshold:
                if not G.has_edge(ni, nj) and not G.has_edge(nj, ni):
                    G.add_edge(ni, nj, relation="adjacent_to", distance=d)

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

    node_color_map = {
        "IfcProject": "purple",
        "IfcBuilding": "blue",
        "IfcBuildingStorey": "orange",
        "IfcTypeObject": "red",
    }
    node_category_map = {
        "IfcProject": "Project root",
        "IfcBuilding": "Building container",
        "IfcBuildingStorey": "Storey / floor container",
        "IfcTypeObject": "Type definition",
    }
    edge_color_map = {
        "aggregates": "#6b7280",
        "contained_in": "#2563eb",
        "typed_by": "#b91c1c",
        "adjacent_to": "#059669",
    }
    edge_relation_explanations = {
        "aggregates": "parent decomposes into child",
        "contained_in": "element belongs to a storey/space",
        "typed_by": "element is assigned to a type",
        "adjacent_to": "elements are spatially near each other",
    }

    node_groups: dict[str, dict[str, list]] = {}
    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        cls = str(d.get("class_") or "Unknown")
        group = node_groups.setdefault(
            cls,
            {
                "x": [],
                "y": [],
                "z": [],
                "hover": [],
            },
        )

        props = d.get("properties", {})
        hover_text = "<br>".join(
            f"<b>{k}</b>: {v}" for k, v in props.items() if v not in ("", None)
        )
        group["x"].append(x)
        group["y"].append(y)
        group["z"].append(z)
        group["hover"].append(
            f"<b>{d.get('label', '')}</b><br>"
            f"Class: {cls}<br>"
            "Category: "
            f"{node_category_map.get(cls, 'Physical element / other IFC class')}<br>"
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

    for cls in sorted(node_groups):
        node = node_groups[cls]
        node_color = node_color_map.get(cls, "green")
        traces.append(
            go.Scatter3d(
                x=node["x"],
                y=node["y"],
                z=node["z"],
                mode="markers",
                marker=dict(size=6, color=node_color),
                hoverinfo="text",
                hovertext=node["hover"],
                name=f"Node: {cls}",
                legendgroup=f"node::{cls}",
                showlegend=True,
            )
        )
        trace_meta.append(("node", cls))

    def _mask(mode: str, show_edge_annotations: bool = False) -> list[bool]:
        hierarchy_rels = {"aggregates", "contained_in", "typed_by"}
        spatial_rels = {"adjacent_to"}
        visible = []
        for kind, name in trace_meta:
            is_edge_label = kind == "edge_label"
            if mode == "all":
                visible.append(not is_edge_label or show_edge_annotations)
            elif mode == "nodes":
                visible.append(kind == "node")
            elif mode == "edges":
                visible.append(
                    kind in {"edge", "edge_hover"}
                    or (is_edge_label and show_edge_annotations)
                )
            elif mode == "hierarchy":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in hierarchy_rels)
                    or (
                        is_edge_label
                        and show_edge_annotations
                        and name in hierarchy_rels
                    )
                )
            elif mode == "spatial":
                visible.append(
                    kind == "node"
                    or (kind in {"edge", "edge_hover"} and name in spatial_rels)
                    or (
                        is_edge_label
                        and show_edge_annotations
                        and name in spatial_rels
                    )
                )
            else:
                visible.append(not is_edge_label or show_edge_annotations)
        return visible

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
        "all": {
            "base": _mask("all"),
            "with_annotations": _mask("all", show_edge_annotations=True),
        },
        "nodes": {
            "base": _mask("nodes"),
            "with_annotations": _mask("nodes", show_edge_annotations=True),
        },
        "edges": {
            "base": _mask("edges"),
            "with_annotations": _mask("edges", show_edge_annotations=True),
        },
        "hierarchy": {
            "base": _mask("hierarchy"),
            "with_annotations": _mask("hierarchy", show_edge_annotations=True),
        },
        "spatial": {
            "base": _mask("spatial"),
            "with_annotations": _mask("spatial", show_edge_annotations=True),
        },
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
    for cls in sorted(node_groups):
        node_color = node_color_map.get(cls, "green")
        node_items.append(
            "<div class='legend-item'>"
            f"<span class='swatch dot' style='--swatch:{node_color}'></span>"
            f"<span>Node: {html.escape(cls)}</span>"
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
        ("orange", "Storey/floor container"),
        ("red", "Type definitions"),
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
      <button
        id="toggle-edge-annotations"
        class="toggle"
        type="button"
        aria-pressed="false"
      >
        Edge Labels: Off
      </button>
    </div>
    <div class="viewer-shell">
      {plotly_div}
      <aside class="legend" aria-label="Graph legend">
        <h3>Legend</h3>
        <div class="section-title">Color Categories</div>
        {color_items}
        <div class="section-title">Edges</div>
        {''.join(edge_items)}
        <div class="section-title">Nodes</div>
        {''.join(node_items)}
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
    let currentMode = "all";
    let edgeAnnotationsEnabled = false;

    function applyMode(mode) {{
      if (!viewer || !viewer.data) return;
      currentMode = mode;
      const modeMasks = masks[mode] || masks.all;
      const visible = edgeAnnotationsEnabled
        ? modeMasks.with_annotations
        : modeMasks.base;
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
