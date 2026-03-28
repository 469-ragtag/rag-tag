from __future__ import annotations

import json

import networkx as nx

from rag_tag.parser.jsonl_to_graph import export_webgl_graph_assets


def test_export_webgl_graph_assets_writes_manifest_and_binary_files(tmp_path) -> None:
    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["demo"]
    graph.graph["edge_categories"] = {
        "hierarchy": ["contains", "contained_in"],
        "spatial": ["adjacent_to"],
        "topology": ["above"],
        "explicit": ["hosts"],
    }
    graph.add_node(
        "Storey::L1",
        label="Level 1",
        class_="IfcBuildingStorey",
        dataset="demo",
        geometry=(0.0, 0.0, 0.0),
    )
    graph.add_node(
        "Element::wall-a",
        label="Wall A",
        class_="IfcWall",
        dataset="demo",
        geometry=(1.0, 0.0, 0.0),
        bbox=((0.0, 0.0, 0.0), (1.0, 0.2, 2.5)),
        mesh=(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.2, 0.0)],
            [(0, 1, 2)],
        ),
    )
    graph.add_node(
        "Element::wall-b",
        label="Wall B",
        class_="IfcWall",
        dataset="demo",
        geometry=(2.0, 0.5, 0.0),
        bbox=((1.5, 0.3, 0.0), (2.5, 0.7, 2.5)),
    )
    graph.add_edge("Storey::L1", "Element::wall-a", relation="contains")
    graph.add_edge("Element::wall-a", "Storey::L1", relation="contained_in")
    graph.add_edge("Element::wall-a", "Element::wall-b", relation="adjacent_to")
    graph.add_edge("Element::wall-a", "Element::wall-b", relation="above")
    graph.add_edge("Element::wall-a", "Element::wall-b", relation="hosts")

    bundle_dir = export_webgl_graph_assets(graph, tmp_path)

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert bundle_dir.name == "demo_graph_viewer"
    assert manifest["dataset"] == "demo"
    assert manifest["node_count"] == 3
    assert manifest["edge_count"] == manifest["legend"]["total_edges"]
    assert "contains__contained_in" in manifest["relation_names"]
    assert "above__below" in manifest["relation_names"]
    assert "hosts__hosted_by" in manifest["relation_names"]
    assert "contains" not in manifest["relation_names"]
    assert "contained_in" not in manifest["relation_names"]
    assert "hosts" not in manifest["relation_names"]
    assert manifest["relation_labels"]["contains__contained_in"] == "contains / contained_in"
    assert manifest["relation_labels"]["above__below"] == "above / below"
    assert manifest["viewer_modes"]["hierarchy"] == ["contains__contained_in"]
    assert "above__below" in manifest["viewer_modes"]["topology"]
    assert "adjacent_to" in manifest["viewer_modes"]["spatial"]
    assert manifest["viewer_modes"]["explicit"] == ["hosts__hosted_by"]
    assert manifest["legend"]["entries"][0]["relation_id"] in manifest["relation_names"]
    assert manifest["legend"]["entries"][0]["member_relations"]
    assert manifest["files"]["edges"]["hierarchy"]["path"] == "edges_hierarchy.bin"
    assert manifest["files"]["edges"]["spatial"]["path"] == "edges_spatial.bin"
    assert manifest["files"]["edges"]["topology"]["path"] == "edges_topology.bin"
    assert manifest["files"]["edges"]["explicit"]["path"] == "edges_explicit.bin"
    assert manifest["files"]["overlays"]["bbox"]["available"] is True
    assert manifest["files"]["overlays"]["mesh"]["available"] is True
    assert (bundle_dir / "nodes.bin").is_file()
    assert (bundle_dir / "nodes_meta.json").is_file()
    assert (bundle_dir / "overlays_bbox.bin").is_file()
    assert (bundle_dir / "overlays_mesh_manifest.json").is_file()


def test_export_webgl_graph_assets_dedupes_symmetric_visual_edges(tmp_path) -> None:
    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["demo"]
    graph.graph["edge_categories"] = {
        "hierarchy": [],
        "spatial": ["adjacent_to"],
        "topology": [],
        "explicit": [],
    }
    graph.add_node(
        "Element::wall-a",
        label="Wall A",
        class_="IfcWall",
        dataset="demo",
        geometry=(0.0, 0.0, 0.0),
    )
    graph.add_node(
        "Element::wall-b",
        label="Wall B",
        class_="IfcWall",
        dataset="demo",
        geometry=(1.0, 0.0, 0.0),
    )
    graph.add_edge("Element::wall-a", "Element::wall-b", relation="adjacent_to")
    graph.add_edge("Element::wall-b", "Element::wall-a", relation="adjacent_to")

    bundle_dir = export_webgl_graph_assets(graph, tmp_path)

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["edge_count"] == 1
    assert manifest["legend"]["total_edges"] == 1
    assert manifest["relation_names"] == ["adjacent_to"]
    assert manifest["legend"]["entries"] == [
        {
            "relation_id": "adjacent_to",
            "label": "adjacent_to",
            "subtitle": (
                "Derived from geometry when elements fall within the distance "
                "threshold but are not verified as touching or intersecting."
            ),
            "count": 1,
            "swatch": "#3b82f6",
            "member_relations": ["adjacent_to"],
        }
    ]
