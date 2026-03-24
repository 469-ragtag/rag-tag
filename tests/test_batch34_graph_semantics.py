from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from rag_tag.config import AppConfig
from rag_tag.graph_contract import EXPLICIT_IFC_RELATIONS, relation_bucket
from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.parser.ifc_relationships import (
    build_relationship_index,
    empty_relation_block,
)
from rag_tag.parser.jsonl_to_graph import (
    add_spatial_adjacency,
    add_topology_facts,
    build_graph,
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


def test_batch3_prunes_selected_classes_from_derived_edges_only(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "derived-pruning.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "GlobalId": "proj",
                "IfcType": "IfcProject",
                "Name": "Project",
            },
            {
                "GlobalId": "bldg",
                "IfcType": "IfcBuilding",
                "Name": "Building",
                "Hierarchy": {"ParentId": "proj"},
            },
            {
                "GlobalId": "storey",
                "IfcType": "IfcBuildingStorey",
                "Name": "Level 1",
                "Hierarchy": {"ParentId": "bldg"},
            },
            {
                "GlobalId": "member-1",
                "IfcType": "IfcMember",
                "Name": "Member A",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [0.0, 0.0, 0.0],
                        "max": [1.0, 1.0, 1.0],
                    },
                    "Centroid": [0.5, 0.5, 0.5],
                },
                "Relationships": {
                    **empty_relation_block(),
                    "typed_by": ["member-type"],
                },
            },
            {
                "GlobalId": "member-type",
                "IfcType": "IfcMemberType",
                "Name": "Member Type",
            },
            {
                "GlobalId": "wall-1",
                "IfcType": "IfcWall",
                "Name": "Wall A",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [0.2, 0.2, 0.0],
                        "max": [1.2, 1.2, 1.0],
                    },
                    "Centroid": [0.7, 0.7, 0.5],
                },
            },
            {
                "GlobalId": "wall-2",
                "IfcType": "IfcWall",
                "Name": "Wall B",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [0.9, 0.9, 0.0],
                        "max": [1.9, 1.9, 1.0],
                    },
                    "Centroid": [1.4, 1.4, 0.5],
                },
            },
        ],
    )

    graph = build_graph_from_jsonl([jsonl_path])
    add_spatial_adjacency(graph, threshold=2.0, exclude_classes={"IfcMember"})
    add_topology_facts(graph, exclude_classes={"IfcMember"})

    member_node = "Element::member-1"
    assert member_node in graph
    assert graph.has_edge("Storey::storey", member_node)
    member_type_edges = graph.get_edge_data(member_node, "Element::member-type") or {}
    assert any(
        attrs.get("relation") == "typed_by" and attrs.get("source") == "ifc"
        for attrs in member_type_edges.values()
    )

    derived_edges_touching_member = [
        (u, v, attrs)
        for u, v, attrs in graph.edges(data=True)
        if attrs.get("source") in {"heuristic", "topology"} and member_node in {u, v}
    ]
    assert derived_edges_touching_member == []

    remaining_derived_relations = {
        attrs.get("relation")
        for _, _, attrs in graph.edges(data=True)
        if attrs.get("source") in {"heuristic", "topology"}
    }
    assert "connected_to" in remaining_derived_relations
    assert "intersects_bbox" in remaining_derived_relations


def test_batch3_records_pruning_metadata_for_derived_phases() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::plate",
        class_="IfcPlate",
        bbox=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        geometry=(0.5, 0.5, 0.5),
        properties={"GlobalId": "plate"},
        dataset="single",
    )
    graph.add_node(
        "Element::wall-a",
        class_="IfcWall",
        bbox=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        geometry=(0.5, 0.5, 0.5),
        properties={"GlobalId": "wall-a"},
        dataset="single",
    )
    graph.add_node(
        "Element::wall-b",
        class_="IfcWall",
        bbox=((0.2, 0.2, 0.0), (1.2, 1.2, 1.0)),
        geometry=(0.7, 0.7, 0.5),
        properties={"GlobalId": "wall-b"},
        dataset="single",
    )

    add_spatial_adjacency(graph, threshold=2.0, exclude_classes={"IfcPlate"})
    add_topology_facts(graph, exclude_classes={"IfcPlate"})

    pruning = graph.graph["graph_build"]["derived_edge_pruning"]
    assert pruning["phases"]["spatial_adjacency"] == {
        "excluded_classes": ["IfcPlate"],
        "eligible_nodes": 2,
        "skipped_nodes": 1,
        "skipped_by_class": {"IfcPlate": 1},
        "edges_added": 2,
        "threshold": 2.0,
    }
    assert pruning["phases"]["topology"] == {
        "excluded_classes": ["IfcPlate"],
        "eligible_nodes": 2,
        "skipped_nodes": 1,
        "skipped_by_class": {"IfcPlate": 1},
        "edges_added": 4,
    }


def test_batch3_build_graph_honors_config_and_explicit_opt_out(
    monkeypatch,
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "build-graph.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "GlobalId": "proj",
                "IfcType": "IfcProject",
                "Name": "Project",
            },
            {
                "GlobalId": "bldg",
                "IfcType": "IfcBuilding",
                "Name": "Building",
                "Hierarchy": {"ParentId": "proj"},
            },
            {
                "GlobalId": "storey",
                "IfcType": "IfcBuildingStorey",
                "Name": "Level 1",
                "Hierarchy": {"ParentId": "bldg"},
            },
            {
                "GlobalId": "member-1",
                "IfcType": "IfcMember",
                "Name": "Member A",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [0.0, 0.0, 0.0],
                        "max": [1.0, 1.0, 1.0],
                    },
                    "Centroid": [0.5, 0.5, 0.5],
                },
            },
            {
                "GlobalId": "wall-1",
                "IfcType": "IfcWall",
                "Name": "Wall A",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [0.2, 0.2, 0.0],
                        "max": [1.2, 1.2, 1.0],
                    },
                    "Centroid": [0.7, 0.7, 0.5],
                },
            },
        ],
    )

    monkeypatch.setattr(
        "rag_tag.parser.jsonl_to_graph.load_project_config",
        lambda _start_dir=None: SimpleNamespace(
            config=AppConfig.model_validate(
                {
                    "graph_build": {
                        "derived_edge_pruning": {
                            "enabled": True,
                            "exclude_classes": ["IfcMember"],
                        }
                    }
                }
            )
        ),
    )

    pruned = build_graph([jsonl_path], payload_mode="minimal")
    assert pruned.graph["graph_build"]["derived_edge_pruning"]["enabled"] is True
    assert pruned.graph["graph_build"]["derived_edge_pruning"]["exclude_classes"] == [
        "IfcMember"
    ]
    assert not any(
        attrs.get("source") == "topology" and "Element::member-1" in {u, v}
        for u, v, attrs in pruned.edges(data=True)
    )

    opt_out = build_graph(
        [jsonl_path],
        payload_mode="minimal",
        derived_edge_pruning_enabled=False,
    )
    assert opt_out.graph["graph_build"]["derived_edge_pruning"]["enabled"] is False
    assert any(
        attrs.get("source") == "topology" and "Element::member-1" in {u, v}
        for u, v, attrs in opt_out.edges(data=True)
    )
