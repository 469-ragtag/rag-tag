from __future__ import annotations

import networkx as nx

from rag_tag.agent.graph_tools import _fuzzy_find_nodes_impl


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
