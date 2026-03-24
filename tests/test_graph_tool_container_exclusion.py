from __future__ import annotations

import networkx as nx
import pytest

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
    assert data["total_found"] == 2
    assert data["returned_count"] == 2
    assert data["truncated"] is False
    assert data["truncation_reason"] is None
    assert [item["id"] for item in data["elements"]] == [
        "Element::roof",
        "Element::sand",
    ]


def test_find_container_elements_excluding_is_bounded_and_deterministic() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("IfcBuilding", label="HQ", class_="IfcBuilding")
    graph.add_node("Storey::ground", label="Ground", class_="IfcBuildingStorey")
    graph.add_node("Element::z", label="Zulu", class_="IfcWall")
    graph.add_node("Element::a2", label="Alpha", class_="IfcWall")
    graph.add_node("Element::a1", label="Alpha", class_="IfcWall")
    graph.add_node("Element::roof", label="Roof", class_="IfcRoof")
    graph.add_node("Element::hidden", label="Able", class_="IfcDoor")
    graph.add_node("Element::mid", label="Mid", class_="IfcBeam")

    graph.add_edge("IfcBuilding", "Storey::ground", relation="aggregates")
    graph.add_edge("IfcBuilding", "Element::z", relation="contains")
    graph.add_edge("IfcBuilding", "Element::a2", relation="contains")
    graph.add_edge("IfcBuilding", "Element::a1", relation="contains")
    graph.add_edge("IfcBuilding", "Element::roof", relation="contains")
    graph.add_edge("Storey::ground", "Element::hidden", relation="contains")
    graph.add_edge("Storey::ground", "Element::mid", relation="contains")

    runtime = wrap_networkx_graph(graph)

    result = _find_container_elements_excluding_impl(
        runtime,
        "IfcBuilding",
        exclude_container_ids=["Storey::ground"],
        max_results=3,
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert data["count"] == 4
    assert data["total_found"] == 4
    assert data["returned_count"] == 3
    assert data["truncated"] is True
    assert data["truncation_reason"] == (
        "Results truncated to 3 item(s) to stay bounded."
    )
    assert [item["id"] for item in data["elements"]] == [
        "Element::a1",
        "Element::a2",
        "Element::roof",
    ]
    assert all(
        item["id"] not in {"Element::hidden", "Element::mid"}
        for item in data["elements"]
    )


def test_find_container_elements_excluding_errors_for_missing_container():
    runtime = _runtime_for_container_exclusion()

    result = _find_container_elements_excluding_impl(
        runtime,
        "IfcBuilding::missing",
        exclude_container_ids=["Storey::ground"],
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "not_found"


@pytest.mark.parametrize("zone_class", ["IfcZone", "IfcSpatialZone"])
def test_find_container_elements_excluding_supports_zone_membership(
    zone_class: str,
):
    graph = nx.MultiDiGraph()
    graph.add_node("Zone::service", label="Service zone", class_=zone_class)
    graph.add_node("Space::101", label="Room 101", class_="IfcSpace")
    graph.add_node("Element::wall", label="Zone wall", class_="IfcWall")
    graph.add_node(
        "Element::diffuser",
        label="Zone diffuser",
        class_="IfcFlowTerminal",
    )

    graph.add_edge("Space::101", "Element::wall", relation="contains")
    graph.add_edge("Element::wall", "Zone::service", relation="in_zone")
    graph.add_edge("Element::diffuser", "Zone::service", relation="in_zone")

    runtime = wrap_networkx_graph(graph)

    result = _find_container_elements_excluding_impl(
        runtime,
        "Zone::service",
        exclude_container_ids=["Space::101"],
    )

    assert result["status"] == "ok"
    data = result["data"]
    assert data["count"] == 1
    assert data["total_found"] == 1
    assert data["returned_count"] == 1
    assert data["truncated"] is False
    assert data["truncation_reason"] is None
    assert [item["id"] for item in data["elements"]] == ["Element::diffuser"]
