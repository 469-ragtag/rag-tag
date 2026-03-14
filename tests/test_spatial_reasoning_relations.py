from __future__ import annotations

import networkx as nx

from rag_tag.graph.backends.neo4j_backend import Neo4jBackend
from rag_tag.graph_contract import CANONICAL_ACTION_SET, TOPOLOGY_RELATIONS
from rag_tag.ifc_graph_tool import query_ifc_graph
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


def _has_relation(graph: nx.MultiDiGraph, u: str, v: str, relation: str) -> bool:
    edge_data = graph.get_edge_data(u, v) or {}
    return any(attrs.get("relation") == relation for attrs in edge_data.values())


def _build_spatial_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    _add_element(
        graph,
        "Element::BASE",
        class_="IfcSlab",
        bbox=((0.0, 0.0, 0.0), (4.0, 1.0, 0.2)),
        geometry=(2.0, 0.5, 0.1),
        axis=(1.0, 0.0, 0.0),
    )
    _add_element(
        graph,
        "Element::TOP",
        class_="IfcBeam",
        bbox=((0.5, 0.0, 0.21), (3.5, 1.0, 0.6)),
        geometry=(2.0, 0.5, 0.405),
        axis=(1.0, 0.0, 0.0),
    )
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
        "Element::PERP",
        class_="IfcWall",
        bbox=((5.0, 3.0, 0.0), (5.2, 7.0, 3.0)),
        geometry=(5.1, 5.0, 1.5),
        axis=(0.0, 1.0, 0.0),
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


def test_topology_relations_include_new_spatial_reasoning_edges() -> None:
    for relation in (
        "supports",
        "supported_by",
        "rests_on",
        "parallel_to",
        "perpendicular_to",
        "facing",
        "inside_3d",
        "contains_3d",
    ):
        assert relation in TOPOLOGY_RELATIONS

    for action in ("spatial_compare", "find_elements_within_clearance"):
        assert action in CANONICAL_ACTION_SET


def test_add_topology_facts_materializes_support_orientation_and_containment() -> None:
    graph = _build_spatial_graph()

    assert _has_relation(graph, "Element::BASE", "Element::TOP", "supports")
    assert _has_relation(graph, "Element::TOP", "Element::BASE", "supported_by")
    assert _has_relation(graph, "Element::TOP", "Element::BASE", "rests_on")

    assert _has_relation(graph, "Element::PAR1", "Element::PAR2", "parallel_to")
    assert _has_relation(graph, "Element::PAR2", "Element::PAR1", "parallel_to")
    assert _has_relation(graph, "Element::PAR1", "Element::PERP", "perpendicular_to")
    assert _has_relation(graph, "Element::PAR1", "Element::PAR2", "facing")

    assert _has_relation(graph, "Element::OUTER", "Element::INNER", "contains_3d")
    assert _has_relation(graph, "Element::INNER", "Element::OUTER", "inside_3d")


def test_spatial_compare_reports_orientation_support_and_containment_metrics() -> None:
    graph = _build_spatial_graph()

    parallel = query_ifc_graph(
        graph,
        "spatial_compare",
        {"element_a": "Element::PAR1", "element_b": "Element::PAR2"},
    )
    assert parallel["status"] == "ok"
    assert parallel["data"]["axis_angle_deg"] is not None
    assert parallel["data"]["parallel_score"] is not None
    assert parallel["data"]["facing_score"] is not None

    support = query_ifc_graph(
        graph,
        "spatial_compare",
        {"element_a": "Element::BASE", "element_b": "Element::TOP"},
    )
    assert support["status"] == "ok"
    assert support["data"]["support_relation"] == "a_supports_b"

    containment = query_ifc_graph(
        graph,
        "spatial_compare",
        {"element_a": "Element::OUTER", "element_b": "Element::INNER"},
    )
    assert containment["status"] == "ok"
    assert containment["data"]["inside_or_contains"] == "a_contains_b"


def test_find_elements_within_clearance_returns_geometry_ranked_results() -> None:
    graph = _build_spatial_graph()
    result = query_ifc_graph(
        graph,
        "find_elements_within_clearance",
        {"element_id": "Element::PAR1", "max_distance": 1.5, "measure": "surface"},
    )

    assert result["status"] == "ok"
    ids = [item["id"] for item in result["data"]["results"]]
    assert "Element::PAR2" in ids


def test_neo4j_backend_uses_catalog_graph_for_new_geometry_actions() -> None:
    graph = _build_spatial_graph()
    backend = Neo4jBackend(graph=graph)

    result = backend.query(
        "spatial_compare",
        {"element_a": "Element::PAR1", "element_b": "Element::PAR2"},
    )

    assert result["status"] == "ok"
    assert result["data"]["parallel_score"] is not None
