from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from rag_tag.graph_contract import EXPLICIT_IFC_RELATIONS, relation_bucket
from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.parser.ifc_relationships import (
    build_relationship_index,
    empty_relation_block,
)
from rag_tag.parser.jsonl_to_graph import (
    add_topology_facts,
    build_graph_from_jsonl,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class _FakeRelDefinesByType:
    def __init__(self, relating_type: object, related_objects: list[object]) -> None:
        self.RelatingType = relating_type
        self.RelatedObjects = related_objects


class _FakeEntity:
    def __init__(self, gid: str) -> None:
        self.GlobalId = gid


class _FakeModel:
    def __init__(self, rels: list[object]) -> None:
        self._rels = rels

    def by_type(self, name: str) -> list[object]:
        if name == "IfcRelDefinesByType":
            return self._rels
        return []


def test_typed_by_extraction_and_contract_taxonomy() -> None:
    occurrence = _FakeEntity("occ-1")
    type_obj = _FakeEntity("type-1")
    index = build_relationship_index(
        _FakeModel([_FakeRelDefinesByType(type_obj, [occurrence])])
    )

    assert index["occ-1"]["typed_by"] == ["type-1"]
    assert "typed_by" in EXPLICIT_IFC_RELATIONS
    assert relation_bucket("typed_by") == "explicit_ifc"


def test_typed_by_materializes_as_ifc_edge(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "typed.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ExpressId": 1,
                "GlobalId": "proj",
                "IfcType": "IfcProject",
                "Name": "Proj",
            },
            {
                "ExpressId": 2,
                "GlobalId": "bldg",
                "IfcType": "IfcBuilding",
                "Name": "Bldg",
                "Hierarchy": {"ParentId": "proj"},
            },
            {
                "ExpressId": 3,
                "GlobalId": "storey",
                "IfcType": "IfcBuildingStorey",
                "Name": "Level 1",
                "Hierarchy": {"ParentId": "bldg"},
            },
            {
                "ExpressId": 4,
                "GlobalId": "door-occ",
                "IfcType": "IfcDoor",
                "Name": "Door",
                "Hierarchy": {"ParentId": "storey"},
                "Relationships": {**empty_relation_block(), "typed_by": ["door-type"]},
            },
            {
                "ExpressId": 5,
                "GlobalId": "door-type",
                "IfcType": "IfcDoorType",
                "Name": "Door Type A",
                "Hierarchy": {},
            },
        ],
    )

    graph = build_graph_from_jsonl([jsonl_path])
    edge_data = graph.get_edge_data("Element::door-occ", "Element::door-type")

    assert edge_data is not None
    assert any(
        attrs.get("relation") == "typed_by" and attrs.get("source") == "ifc"
        for attrs in edge_data.values()
    )


def test_topology_fallback_does_not_promote_bbox_overlap_to_intersects_3d() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::A",
        class_="IfcWall",
        bbox=((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
        properties={"GlobalId": "A"},
        dataset="single",
    )
    graph.add_node(
        "Element::B",
        class_="IfcWall",
        bbox=((1.0, 1.0, 0.0), (3.0, 3.0, 2.0)),
        properties={"GlobalId": "B"},
        dataset="single",
    )

    add_topology_facts(graph)

    relations = [attrs.get("relation") for _, _, attrs in graph.edges(data=True)]
    assert "intersects_bbox" in relations
    assert "intersects_3d" not in relations


def test_traverse_returns_all_parallel_matching_relations() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("Element::A", label="A", class_="IfcWall", properties={})
    graph.add_node("Element::B", label="B", class_="IfcDoor", properties={})
    graph.add_edge("Element::A", "Element::B", relation="hosts", source="ifc")
    graph.add_edge("Element::A", "Element::B", relation="typed_by", source="ifc")

    result = query_ifc_graph(graph, "traverse", {"start": "Element::A", "depth": 1})

    assert result["status"] == "ok"
    relations = [item["relation"] for item in result["data"]["results"]]
    assert relations == ["hosts", "typed_by"]
