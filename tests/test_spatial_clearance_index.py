from __future__ import annotations

import networkx as nx

import rag_tag.ifc_graph_tool as ifc_graph_tool


def _add_element(
    graph: nx.MultiDiGraph,
    node_id: str,
    *,
    class_: str,
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
    geometry: tuple[float, float, float],
    axis: tuple[float, float, float],
    dataset: str = "sample",
) -> None:
    x_len = max(0.1, bbox[1][0] - bbox[0][0])
    y_len = max(0.1, bbox[1][1] - bbox[0][1])
    z_len = max(0.1, bbox[1][2] - bbox[0][2])
    major_xy = max(x_len, y_len)
    minor_xy = min(x_len, y_len)
    graph.add_node(
        node_id,
        label=node_id.split("::", 1)[1],
        class_=class_,
        bbox=bbox,
        geometry=geometry,
        obb={
            "center": geometry,
            "axes": (
                axis,
                (0.0, 1.0, 0.0) if axis != (0.0, 1.0, 0.0) else (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            "extents": (major_xy, minor_xy, z_len),
            "corners_xy": None,
        },
        properties={"GlobalId": node_id.split("::", 1)[1]},
        dataset=dataset,
    )


def test_find_elements_within_clearance_uses_spatial_index_candidates(
    monkeypatch,
) -> None:
    graph = nx.MultiDiGraph()
    _add_element(
        graph,
        "Element::SOURCE",
        class_="IfcWall",
        bbox=((0.0, 0.0, 0.0), (4.0, 0.2, 3.0)),
        geometry=(2.0, 0.1, 1.5),
        axis=(1.0, 0.0, 0.0),
    )
    _add_element(
        graph,
        "Element::NEAR",
        class_="IfcWall",
        bbox=((0.0, 1.0, 0.0), (4.0, 1.2, 3.0)),
        geometry=(2.0, 1.1, 1.5),
        axis=(1.0, 0.0, 0.0),
    )
    for idx in range(6):
        x0 = 20.0 + (idx * 10.0)
        _add_element(
            graph,
            f"Element::FAR{idx}",
            class_="IfcWall",
            bbox=((x0, 20.0, 0.0), (x0 + 4.0, 20.2, 3.0)),
            geometry=(x0 + 2.0, 20.1, 1.5),
            axis=(1.0, 0.0, 0.0),
        )

    compared_ids: list[str] = []
    original_compare = ifc_graph_tool.compare_nodes_geometry

    def _recording_compare(node_a, node_b, *, edge_metrics=None):
        compared_ids.append(str(node_b.get("label")))
        return original_compare(node_a, node_b, edge_metrics=edge_metrics)

    monkeypatch.setattr(ifc_graph_tool, "compare_nodes_geometry", _recording_compare)

    result = ifc_graph_tool.query_ifc_graph(
        graph,
        "find_elements_within_clearance",
        {
            "element_id": "Element::SOURCE",
            "max_distance": 2.0,
            "measure": "surface",
        },
    )

    assert result["status"] == "ok"
    ids = [item["id"] for item in result["data"]["results"]]
    assert ids == ["Element::NEAR"]
    assert compared_ids == ["NEAR"]
