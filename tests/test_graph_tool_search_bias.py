from __future__ import annotations

import networkx as nx

from rag_tag.agent.graph_tools import _fuzzy_find_nodes_impl
from rag_tag.graph import wrap_networkx_graph


def _add_node(
    graph: nx.MultiDiGraph,
    node_id: str,
    *,
    label: str,
    class_name: str,
) -> None:
    graph.add_node(
        node_id,
        label=label,
        class_=class_name,
        properties={"Name": label, "TypeName": label},
        payload={"Name": label, "IfcType": class_name, "ClassRaw": class_name},
    )


def test_fuzzy_find_nodes_prefers_occurrence_over_type_without_type_intent() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::wall-occ",
        label="plumbing wall",
        class_="IfcWall",
        properties={
            "Name": "plumbing wall",
            "TypeName": "plumbing wall",
            "ObjectType": "plumbingwall",
        },
        payload={"Name": "plumbing wall", "IfcType": "IfcWall", "ClassRaw": "IfcWall"},
    )
    graph.add_node(
        "Element::wall-type",
        label="plumbing wall",
        class_="IfcWallType",
        properties={
            "Name": "plumbing wall",
            "TypeName": "plumbing wall",
            "PredefinedType": "PLUMBINGWALL",
        },
        payload={
            "Name": "plumbing wall",
            "IfcType": "IfcWallType",
            "ClassRaw": "IfcWallType",
        },
    )

    result = _fuzzy_find_nodes_impl(graph, "plumbing wall")

    assert result["status"] == "ok"
    matches = result["data"]["matches"]
    assert [item["id"] for item in matches[:2]] == [
        "Element::wall-occ",
        "Element::wall-type",
    ]
    assert matches[0]["score"] > matches[1]["score"]


def test_fuzzy_find_nodes_prefers_type_when_query_mentions_type() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::wall-occ",
        label="plumbing wall",
        class_="IfcWall",
        properties={"Name": "plumbing wall", "TypeName": "plumbing wall"},
        payload={"Name": "plumbing wall", "IfcType": "IfcWall", "ClassRaw": "IfcWall"},
    )
    graph.add_node(
        "Element::wall-type",
        label="plumbing wall",
        class_="IfcWallType",
        properties={
            "Name": "plumbing wall",
            "TypeName": "plumbing wall",
            "PredefinedType": "PLUMBINGWALL",
        },
        payload={
            "Name": "plumbing wall",
            "IfcType": "IfcWallType",
            "ClassRaw": "IfcWallType",
        },
    )

    result = _fuzzy_find_nodes_impl(graph, "plumbing wall type")

    assert result["status"] == "ok"
    matches = result["data"]["matches"]
    assert [item["id"] for item in matches[:2]] == [
        "Element::wall-type",
        "Element::wall-occ",
    ]
    assert matches[0]["score"] > matches[1]["score"]


def test_fuzzy_find_nodes_prefers_ifc_building_for_generic_building_query() -> None:
    graph = nx.MultiDiGraph()
    _add_node(
        graph,
        "IfcBuilding",
        label="Main Building",
        class_name="IfcBuilding",
    )
    _add_node(
        graph,
        "Element::building-proxy",
        label="building",
        class_name="IfcBuildingElementProxy",
    )

    result = _fuzzy_find_nodes_impl(graph, "building")

    assert result["status"] == "ok"
    matches = result["data"]["matches"]
    assert [item["id"] for item in matches[:2]] == [
        "IfcBuilding",
        "Element::building-proxy",
    ]
    assert matches[0]["class_"] == "IfcBuilding"
    assert result["data"]["total_found"] == 2
    assert result["data"]["returned_count"] == 2
    assert result["data"]["truncated"] is False
    assert result["data"]["truncation_reason"] is None


def test_fuzzy_find_nodes_prefers_ifc_site_for_generic_site_query() -> None:
    graph = nx.MultiDiGraph()
    _add_node(graph, "IfcSite", label="Project Site", class_name="IfcSite")
    _add_node(
        graph,
        "Element::site-proxy",
        label="site",
        class_name="IfcBuildingElementProxy",
    )

    result = _fuzzy_find_nodes_impl(graph, "site")

    assert result["status"] == "ok"
    assert result["data"]["matches"][0]["id"] == "IfcSite"
    assert result["data"]["matches"][0]["class_"] == "IfcSite"


def test_fuzzy_find_nodes_prefers_ifc_building_storey_for_generic_floor_query() -> None:
    graph = nx.MultiDiGraph()
    _add_node(
        graph,
        "Storey::L1",
        label="Level 1",
        class_name="IfcBuildingStorey",
    )
    _add_node(
        graph,
        "Element::floor-proxy",
        label="floor",
        class_name="IfcBuildingElementProxy",
    )

    result = _fuzzy_find_nodes_impl(graph, "floor")

    assert result["status"] == "ok"
    assert result["data"]["matches"][0]["id"] == "Storey::L1"
    assert result["data"]["matches"][0]["class_"] == "IfcBuildingStorey"


def test_fuzzy_find_nodes_prefers_ifc_space_for_generic_room_query() -> None:
    graph = nx.MultiDiGraph()
    _add_node(
        graph,
        "Element::space-101",
        label="Room 101",
        class_name="IfcSpace",
    )
    _add_node(
        graph,
        "Element::room-proxy",
        label="room",
        class_name="IfcBuildingElementProxy",
    )

    result = _fuzzy_find_nodes_impl(graph, "room")

    assert result["status"] == "ok"
    assert result["data"]["matches"][0]["id"] == "Element::space-101"
    assert result["data"]["matches"][0]["class_"] == "IfcSpace"


def test_fuzzy_find_nodes_does_not_over_penalize_specific_named_phrase_matches() -> (
    None
):
    graph = nx.MultiDiGraph()
    _add_node(
        graph,
        "IfcBuilding",
        label="Building",
        class_name="IfcBuilding",
    )
    _add_node(
        graph,
        "Element::main-building-sign",
        label="Main Building",
        class_name="IfcBuildingElementProxy",
    )

    result = _fuzzy_find_nodes_impl(graph, "main building")

    assert result["status"] == "ok"
    assert result["data"]["matches"][0]["id"] == "Element::main-building-sign"
    assert result["data"]["matches"][0]["class_"] == "IfcBuildingElementProxy"


def test_fuzzy_find_nodes_reuses_duplicate_generic_container_search_with_warning() -> (
    None
):
    graph = nx.MultiDiGraph()
    _add_node(
        graph,
        "IfcBuilding",
        label="Main Building",
        class_name="IfcBuilding",
    )
    _add_node(
        graph,
        "Element::building-proxy",
        label="building",
        class_name="IfcBuildingElementProxy",
    )
    runtime = wrap_networkx_graph(graph)

    first = _fuzzy_find_nodes_impl(runtime, "building")
    second = _fuzzy_find_nodes_impl(runtime, "the building", top_k=5)

    assert first["status"] == "ok"
    assert "warnings" not in first["data"]
    assert second["status"] == "ok"
    assert second["data"]["query"] == "the building"
    assert second["data"]["matches"][0]["id"] == "IfcBuilding"
    assert second["data"]["warnings"] == [
        "Reused the prior canonical container anchor search. Prefer the existing "
        "exact container ID instead of repeating broad fuzzy resolution."
    ]
