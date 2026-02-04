from pathlib import Path
import ifcopenshell
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from ifc_geometry_parse import get_ifc_model, extract_geometry_data
from ifc_to_csv import _find_project_root, _find_ifc_dir

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


def build_graph_with_properties(csv_path: str, geom_data: dict) -> nx.DiGraph:
    """
    Build a hierarchical IFC graph and attach geometry data to nodes.
    `geom_data` should be a dict mapping GlobalId -> geometry info (e.g., centroid or bounding box)
    """
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()

    # Root nodes
    G.add_node("IfcProject", label="Project", class_="IfcProject", geometry=None)
    G.add_node("IfcBuilding", label="Building", class_="IfcBuilding", geometry=None)
    G.add_edge("IfcProject", "IfcBuilding", relation="aggregates")

    # Levels
    levels = df["Level"].dropna().unique()
    for lvl in levels:
        node_id = f"Storey::{lvl}"
        G.add_node(node_id, label=lvl, class_="IfcBuildingStorey", geometry=None)
        G.add_edge("IfcBuilding", node_id, relation="aggregates")

    # Types
    for t in df["TypeName"].dropna().unique():
        node_id = f"Type::{t}"
        G.add_node(node_id, label=t, class_="IfcTypeObject", geometry=None)

    # Elements
    for _, row in df.iterrows():
        eid = f"Element::{row['GlobalId']}"
        geom = geom_data.get(row["GlobalId"], None)  # attach geometry if available

        G.add_node(
            eid,
            label=row.get("Name", row["GlobalId"]),
            class_=row["Class"],
            properties=row.to_dict(),
            geometry=geom,  # new geometry property
        )

        if pd.notna(row["Level"]):
            G.add_edge(f"Storey::{row['Level']}", eid, relation="contained_in")

        if pd.notna(row["TypeName"]):
            G.add_edge(f"Type::{row['TypeName']}", eid, relation="typed_by")

    return G

def plot_interactive_graph(G: nx.DiGraph, out_html: Path):
    """
    Plot the IFC graph in 3D using real geometry coordinates if available.
    Nodes without geometry are slightly offset to avoid overlap.
    """
    import numpy as np

    # Build positions from geometry
    pos = {}
    for n, d in G.nodes(data=True):
        geom = d.get("geometry")
        if geom is not None:
            pos[n] = tuple(geom)
        else:
            pos[n] = (np.random.rand()*0.5, np.random.rand()*0.5, np.random.rand()*0.5)

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
            f"<b>{d.get('label','')}</b><br>"
            f"Class: {d.get('class_','')}<br>"
            + hover_text
        )

        colors.append({
            "IfcProject": "purple",
            "IfcBuilding": "blue",
            "IfcBuildingStorey": "orange",
            "IfcTypeObject": "red"
        }.get(d.get("class_"), "green"))

    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=6, color=colors),
        hoverinfo="text",
        hovertext=hover
    )

    # Edge coordinates
    ex, ey, ez = [], [], []

    for u, v, d in G.edges(data=True):
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]

        ex += [x0, x1, None]
        ey += [y0, y1, None]
        ez += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(width=3, color="gray"),
        hoverinfo="none"
    )

    # Ensure output folder exists
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        scene=dict(aspectmode="data"),
        title="IFC Hierarchy Graph (3D with Geometry)"
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
        obj_name = getattr(obj, "Name", obj.is_a())
        print_with_indent(f"{obj.is_a()}: {obj_name}", level)

        # Find all objects aggregated under this one
        if hasattr(obj, "IsDecomposedBy"):
            for rel in obj.IsDecomposedBy or []:
                for child in getattr(rel, "RelatedObjects", []):
                    traverse(child, level + 1)

    # Start from the project(s)
    for project in model.by_type("IfcProject"):
        traverse(project, indent)


#print_ifc_hierarchy(ifc_file) # Helper to visualize hierarchy in console

# Convert list to dict
geom_dict = {}
for item in geom_data:
    gid = item.get("GlobalId")
    if gid:
        # Use centroid if available, else fallback to raw geometry
        geom_dict[gid] = item.get("centroid") or item.get("geometry")

html_dir = project_root / "output"
html_dir.mkdir(parents=True, exist_ok=True)
html_file = html_dir / "ifc_graph.html"

G = build_graph_with_properties(csv_file, geom_dict)
plot_interactive_graph(G, html_file)