from __future__ import annotations

import networkx as nx

from rag_tag.graph.spatial_reasoning import (
    candidate_node_ids_with_xy_overlap,
    candidate_node_ids_within_bbox_distance,
    get_spatial_index,
    iter_topology_candidate_pairs,
)
from rag_tag.parser.jsonl_to_graph import add_topology_facts


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


def _build_spatial_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    _add_element(
        graph,
        "Element::PAR1",
        class_="IfcWall",
        bbox=((0.0, 3.0, 0.0), (4.0, 3.2, 3.0)),
        geometry=(2.0, 3.1, 1.5),
        axis=(1.0, 0.0, 0.0),
    )
    _add_element(
        graph,
        "Element::PAR2",
        class_="IfcWall",
        bbox=((0.0, 4.0, 0.0), (4.0, 4.2, 3.0)),
        geometry=(2.0, 4.1, 1.5),
        axis=(1.0, 0.0, 0.0),
    )
    _add_element(
        graph,
        "Element::OUTER",
        class_="IfcSpace",
        bbox=((10.0, 0.0, 0.0), (14.0, 4.0, 4.0)),
        geometry=(12.0, 2.0, 2.0),
        axis=(1.0, 0.0, 0.0),
    )
    _add_element(
        graph,
        "Element::INNER",
        class_="IfcFurniture",
        bbox=((11.0, 1.0, 1.0), (12.0, 2.0, 2.0)),
        geometry=(11.5, 1.5, 1.5),
        axis=(1.0, 0.0, 0.0),
    )
    add_topology_facts(graph)
    return graph


def test_spatial_index_shortlists_xy_and_distance_candidates() -> None:
    graph = _build_spatial_graph()
    _add_element(
        graph,
        "Element::FAR",
        class_="IfcWall",
        bbox=((50.0, 50.0, 0.0), (54.0, 50.2, 3.0)),
        geometry=(52.0, 50.1, 1.5),
        axis=(1.0, 0.0, 0.0),
    )

    index = get_spatial_index(graph)

    overlap_candidates = candidate_node_ids_with_xy_overlap(
        index,
        graph.nodes["Element::OUTER"]["bbox"],
        exclude_node_id="Element::OUTER",
    )
    assert "Element::INNER" in overlap_candidates
    assert "Element::FAR" not in overlap_candidates

    distance_candidates = candidate_node_ids_within_bbox_distance(
        index,
        graph.nodes["Element::PAR1"]["bbox"],
        1.5,
        exclude_node_id="Element::PAR1",
    )
    assert "Element::PAR2" in distance_candidates
    assert "Element::FAR" not in distance_candidates

    pairs = iter_topology_candidate_pairs(index)
    assert ("Element::PAR1", "Element::PAR2") in pairs
    assert ("Element::FAR", "Element::PAR1") not in pairs
    assert ("Element::PAR1", "Element::FAR") not in pairs
