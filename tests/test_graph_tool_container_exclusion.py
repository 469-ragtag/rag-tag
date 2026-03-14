from __future__ import annotations

import networkx as nx

from rag_tag.agent.graph_tools import _find_container_elements_excluding_impl
from rag_tag.graph import wrap_networkx_graph


def _runtime_for_container_exclusion():
    graph = nx.MultiDiGraph()
    graph.add_node("IfcBuilding", label="Single-family house", class_="IfcBuilding")
    graph.add_node(
        "Storey::ground",
        label="00 groundfloor",
        class_="IfcBuildingStorey",
    )
    graph.add_node(
        "Element::sand",
        label="sand bedding",
        class_="IfcEarthworksFill",
        properties={"Level": "Single-family house"},
    )
    graph.add_node(
        "Element::roof",
        label="house - roof",
        class_="IfcRoof",
        properties={"Level": "Single-family house"},
    )
    graph.add_node(
        "Element::wall",
        label="outer wall",
        class_="IfcWall",
        properties={"Level": "00 groundfloor"},
    )

    graph.add_edge("IfcBuilding", "Storey::ground", relation="aggregates")
    graph.add_edge("IfcBuilding", "Element::sand", relation="contains")
    graph.add_edge("IfcBuilding", "Element::roof", relation="contains")
    graph.add_edge("Storey::ground", "Element::wall", relation="contains")
    return wrap_networkx_graph(graph)


def test_find_container_elements_excluding_returns_non_storey_building_members():
    runtime = _runtime_for_container_exclusion()

    result = _find_container_elements_excluding_impl(
        runtime,
        "IfcBuilding",
        exclude_container_ids=["Storey::ground"],
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert data["count"] == 2
    assert [item["id"] for item in data["elements"]] == [
        "Element::roof",
        "Element::sand",
    ]


def test_find_container_elements_excluding_errors_for_missing_container():
    runtime = _runtime_for_container_exclusion()

    result = _find_container_elements_excluding_impl(
        runtime,
        "IfcBuilding::missing",
        exclude_container_ids=["Storey::ground"],
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "not_found"
