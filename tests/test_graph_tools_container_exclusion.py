from __future__ import annotations

import networkx as nx

from rag_tag.agent.graph_tools import _find_container_elements_excluding_impl
from rag_tag.ifc_graph_tool import query_ifc_graph


def _query_fn(graph: nx.MultiDiGraph):
    return lambda action, params: query_ifc_graph(graph, action, params)


def test_find_container_elements_excluding_removes_nested_storey_members() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "IfcBuilding",
        label="Building",
        class_="IfcBuilding",
        properties={"GlobalId": "BLDG1"},
        payload={},
    )
    graph.add_node(
        "Storey::GROUND",
        label="Ground Floor",
        class_="IfcBuildingStorey",
        properties={"GlobalId": "GROUND"},
        payload={},
    )
    graph.add_node(
        "Storey::UPPER",
        label="Upper Floor",
        class_="IfcBuildingStorey",
        properties={"GlobalId": "UPPER"},
        payload={},
    )
    graph.add_node(
        "Element::WALL_G",
        label="Ground Wall",
        class_="IfcWall",
        properties={"GlobalId": "WALL_G", "Name": "Ground Wall"},
        payload={},
    )
    graph.add_node(
        "Element::WALL_U",
        label="Upper Wall",
        class_="IfcWall",
        properties={"GlobalId": "WALL_U", "Name": "Upper Wall"},
        payload={},
    )

    graph.add_edge("IfcBuilding", "Storey::GROUND", relation="contains")
    graph.add_edge("IfcBuilding", "Storey::UPPER", relation="contains")
    graph.add_edge("Storey::GROUND", "IfcBuilding", relation="contained_in")
    graph.add_edge("Storey::UPPER", "IfcBuilding", relation="contained_in")
    graph.add_edge("Storey::GROUND", "Element::WALL_G", relation="contains")
    graph.add_edge("Storey::UPPER", "Element::WALL_U", relation="contains")
    graph.add_edge("Element::WALL_G", "Storey::GROUND", relation="contained_in")
    graph.add_edge("Element::WALL_U", "Storey::UPPER", relation="contained_in")

    result = _find_container_elements_excluding_impl(
        graph,
        _query_fn(graph),
        "IfcBuilding",
        exclude_container_ids=["Storey::GROUND"],
        depth=3,
    )

    assert result["status"] == "ok"
    assert [item["id"] for item in result["data"]["results"]] == ["Element::WALL_U"]


def test_find_container_elements_excluding_supports_zone_membership() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Zone::MECH",
        label="Mechanical Zone",
        class_="IfcZone",
        properties={"GlobalId": "ZONE1"},
        payload={},
    )
    graph.add_node(
        "Element::DUCT_1",
        label="Duct 1",
        class_="IfcDuctSegment",
        properties={"GlobalId": "DUCT_1", "Name": "Duct 1"},
        payload={},
    )
    graph.add_edge("Element::DUCT_1", "Zone::MECH", relation="in_zone", source="ifc")

    result = _find_container_elements_excluding_impl(
        graph,
        _query_fn(graph),
        "Zone::MECH",
        depth=1,
    )

    assert result["status"] == "ok"
    assert [item["id"] for item in result["data"]["results"]] == ["Element::DUCT_1"]
