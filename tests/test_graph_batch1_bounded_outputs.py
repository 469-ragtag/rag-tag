from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import networkx as nx

from rag_tag.agent.graph_tools import register_graph_tools
from rag_tag.graph_contract import missing_required_action_fields
from rag_tag.ifc_graph_tool import query_ifc_graph


def _make_node(
    node_id: str, label: str, class_name: str, global_id: str
) -> dict[str, Any]:
    return {
        "label": label,
        "class_": class_name,
        "properties": {"GlobalId": global_id},
        "payload": {"Name": label, "IfcType": class_name, "ClassRaw": class_name},
    }


def test_traverse_is_bounded_and_preserves_discovery_order() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("Element::A", **_make_node("Element::A", "Root", "IfcWall", "A"))
    graph.add_node("Element::B", **_make_node("Element::B", "First", "IfcDoor", "B"))
    graph.add_node("Element::C", **_make_node("Element::C", "Second", "IfcDoor", "C"))
    graph.add_edge("Element::A", "Element::B", relation="hosts", source="ifc")
    graph.add_edge("Element::A", "Element::B", relation="typed_by", source="ifc")
    graph.add_edge("Element::A", "Element::C", relation="hosts", source="ifc")

    result = query_ifc_graph(
        graph,
        "traverse",
        {"start": "Element::A", "depth": 1, "max_results": 2},
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert not missing_required_action_fields("traverse", data)
    assert [item["relation"] for item in data["results"]] == ["hosts", "typed_by"]
    assert data["total_found"] == 3
    assert data["returned_count"] == 2
    assert data["truncated"] is True
    assert data["truncation_reason"] is not None


def test_scan_actions_are_sorted_and_bounded() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("Element::B", **_make_node("Element::B", "Beta", "IfcWall", "B"))
    graph.add_node("Element::A", **_make_node("Element::A", "Alpha", "IfcWall", "A"))
    graph.add_node("Element::C", **_make_node("Element::C", "Gamma", "IfcWall", "C"))

    by_class = query_ifc_graph(
        graph,
        "find_elements_by_class",
        {"class": "IfcWall", "max_results": 2},
    )
    assert by_class["status"] == "ok"
    assert [item["label"] for item in by_class["data"]["elements"]] == ["Alpha", "Beta"]
    assert by_class["data"]["total_found"] == 3
    assert by_class["data"]["returned_count"] == 2
    assert by_class["data"]["truncated"] is True

    exact = query_ifc_graph(
        graph,
        "find_nodes",
        {"class": "IfcWall", "max_results": 2},
    )
    assert exact["status"] == "ok"
    assert [item["label"] for item in exact["data"]["elements"]] == ["Alpha", "Beta"]
    assert exact["data"]["total_found"] == 3
    assert exact["data"]["returned_count"] == 2
    assert exact["data"]["truncated"] is True


def test_adjacency_and_spatial_results_sort_by_distance() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::root", **_make_node("Element::root", "Root", "IfcWall", "ROOT")
    )
    graph.add_node("Element::n3", **_make_node("Element::n3", "Three", "IfcDoor", "N3"))
    graph.add_node("Element::n1", **_make_node("Element::n1", "One", "IfcDoor", "N1"))
    graph.add_node("Element::n2", **_make_node("Element::n2", "Two", "IfcDoor", "N2"))
    graph.add_edge(
        "Element::root",
        "Element::n3",
        relation="adjacent_to",
        source="heuristic",
        distance=3.0,
    )
    graph.add_edge(
        "Element::root",
        "Element::n1",
        relation="adjacent_to",
        source="heuristic",
        distance=1.0,
    )
    graph.add_edge(
        "Element::root",
        "Element::n2",
        relation="connected_to",
        source="heuristic",
        distance=2.0,
    )

    adjacent = query_ifc_graph(
        graph,
        "get_adjacent_elements",
        {"element_id": "Element::root", "max_results": 2},
    )
    assert adjacent["status"] == "ok"
    assert [item["label"] for item in adjacent["data"]["adjacent"]] == ["One", "Two"]
    assert adjacent["data"]["total_found"] == 3
    assert adjacent["data"]["returned_count"] == 2
    assert adjacent["data"]["truncated"] is True

    nearby = query_ifc_graph(
        graph,
        "spatial_query",
        {"near": "Element::root", "max_distance": 5, "max_results": 2},
    )
    assert nearby["status"] == "ok"
    assert [item["label"] for item in nearby["data"]["results"]] == ["One", "Two"]
    assert nearby["data"]["total_found"] == 3
    assert nearby["data"]["returned_count"] == 2
    assert nearby["data"]["truncated"] is True


def test_topology_and_vertical_helpers_are_bounded_and_sorted_by_metric() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::root", **_make_node("Element::root", "Root", "IfcSlab", "ROOT")
    )
    graph.add_node(
        "Element::high", **_make_node("Element::high", "High", "IfcBeam", "HIGH")
    )
    graph.add_node(
        "Element::mid", **_make_node("Element::mid", "Mid", "IfcBeam", "MID")
    )
    graph.add_node(
        "Element::low", **_make_node("Element::low", "Low", "IfcBeam", "LOW")
    )
    graph.add_node(
        "Element::i1", **_make_node("Element::i1", "Intersect 1", "IfcWall", "I1")
    )
    graph.add_node(
        "Element::i2", **_make_node("Element::i2", "Intersect 2", "IfcWall", "I2")
    )
    graph.add_node(
        "Element::i3", **_make_node("Element::i3", "Intersect 3", "IfcWall", "I3")
    )

    graph.add_edge(
        "Element::root",
        "Element::high",
        relation="below",
        source="topology",
        vertical_gap=0.3,
    )
    graph.add_edge(
        "Element::root",
        "Element::mid",
        relation="below",
        source="topology",
        vertical_gap=0.1,
    )
    graph.add_edge(
        "Element::root",
        "Element::low",
        relation="below",
        source="topology",
        vertical_gap=0.2,
    )
    graph.add_edge(
        "Element::root",
        "Element::i1",
        relation="intersects_3d",
        source="topology",
        intersection_volume=1.0,
    )
    graph.add_edge(
        "Element::root",
        "Element::i2",
        relation="intersects_3d",
        source="topology",
        intersection_volume=5.0,
    )
    graph.add_edge(
        "Element::root",
        "Element::i3",
        relation="intersects_3d",
        source="topology",
        intersection_volume=3.0,
    )

    above = query_ifc_graph(
        graph,
        "find_elements_above",
        {"element_id": "Element::root", "max_results": 2},
    )
    assert above["status"] == "ok"
    assert [item["label"] for item in above["data"]["results"]] == ["Mid", "Low"]
    assert [item["vertical_gap"] for item in above["data"]["results"]] == [0.1, 0.2]
    assert above["data"]["total_found"] == 3
    assert above["data"]["returned_count"] == 2
    assert above["data"]["truncated"] is True

    intersections = query_ifc_graph(
        graph,
        "get_intersections_3d",
        {"element_id": "Element::root", "max_results": 2},
    )
    assert intersections["status"] == "ok"
    assert [item["label"] for item in intersections["data"]["intersections_3d"]] == [
        "Intersect 2",
        "Intersect 3",
    ]
    assert intersections["data"]["total_found"] == 3
    assert intersections["data"]["returned_count"] == 2
    assert intersections["data"]["truncated"] is True


def test_storey_elements_and_topology_neighbors_report_stable_metadata() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Storey::L1", **_make_node("Storey::L1", "Level 1", "IfcBuildingStorey", "L1")
    )
    graph.add_node("Element::c", **_make_node("Element::c", "Charlie", "IfcWall", "C"))
    graph.add_node("Element::a", **_make_node("Element::a", "Alpha", "IfcWall", "A"))
    graph.add_node("Element::b", **_make_node("Element::b", "Bravo", "IfcWall", "B"))
    graph.add_edge("Storey::L1", "Element::c", relation="contains")
    graph.add_edge("Storey::L1", "Element::a", relation="contains")
    graph.add_edge("Storey::L1", "Element::b", relation="contains")
    graph.add_edge(
        "Element::a",
        "Element::b",
        relation="overlaps_xy",
        source="topology",
        overlap_area_xy=2.0,
    )
    graph.add_edge(
        "Element::a",
        "Element::c",
        relation="overlaps_xy",
        source="topology",
        overlap_area_xy=5.0,
    )

    storey = query_ifc_graph(
        graph,
        "get_elements_in_storey",
        {"storey": "Storey::L1", "max_results": 2},
    )
    assert storey["status"] == "ok"
    assert [item["label"] for item in storey["data"]["elements"]] == ["Alpha", "Bravo"]
    assert storey["data"]["total_found"] == 3
    assert storey["data"]["returned_count"] == 2
    assert storey["data"]["truncated"] is True

    topology = query_ifc_graph(
        graph,
        "get_topology_neighbors",
        {"element_id": "Element::a", "relation": "overlaps_xy", "max_results": 1},
    )
    assert topology["status"] == "ok"
    assert [item["label"] for item in topology["data"]["neighbors"]] == ["Charlie"]
    assert topology["data"]["total_found"] == 2
    assert topology["data"]["returned_count"] == 1
    assert topology["data"]["truncated"] is True


def test_graph_tools_expose_batch1_max_results_defaults() -> None:
    class DummyAgent:
        def __init__(self) -> None:
            self.tools: dict[str, Callable[..., Any]] = {}

        def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[func.__name__] = func
            return func

    agent = DummyAgent()
    register_graph_tools(agent)

    scan_defaults = {
        "find_nodes": 50,
        "spatial_query": 50,
        "get_elements_in_storey": 50,
        "find_elements_by_class": 50,
    }
    neighbor_defaults = {
        "traverse": 25,
        "get_adjacent_elements": 25,
        "get_topology_neighbors": 25,
        "get_intersections_3d": 25,
        "find_elements_above": 25,
        "find_elements_below": 25,
    }

    for tool_name, expected_default in {**scan_defaults, **neighbor_defaults}.items():
        signature = inspect.signature(agent.tools[tool_name])
        assert signature.parameters["max_results"].default == expected_default
