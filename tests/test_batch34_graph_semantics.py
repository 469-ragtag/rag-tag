from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

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
    plot_interactive_graph,
    plot_interactive_graph_overlap_modes,
)
from rag_tag.parser.jsonl_to_graph import (
    main as jsonl_to_graph_main,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _relations_between(graph: nx.MultiDiGraph, source: str, target: str) -> list[str]:
    edge_data = graph.get_edge_data(source, target) or {}
    return [
        str(attrs.get("relation"))
        for attrs in edge_data.values()
        if isinstance(attrs, dict) and attrs.get("relation") is not None
    ]


def _add_contains_pair(graph: nx.MultiDiGraph, parent: str, child: str) -> None:
    graph.add_edge(parent, child, relation="contains")
    graph.add_edge(child, parent, relation="contained_in")


def _make_obb(
    *,
    center: tuple[float, float, float],
    axis_u: tuple[float, float],
    axis_v: tuple[float, float],
    extent_u: float,
    extent_v: float,
    extent_z: float = 1.0,
) -> dict[str, object]:
    return {
        "center": center,
        "axes": (
            (axis_u[0], axis_u[1], 0.0),
            (axis_v[0], axis_v[1], 0.0),
            (0.0, 0.0, 1.0),
        ),
        "extents": (extent_u, extent_v, extent_z),
        "corners_xy": None,
    }


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


def test_build_graph_stores_symmetric_ifc_edges_once_per_pair(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "single-ifc-edge.jsonl"
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
                "GlobalId": "conn-a",
                "IfcType": "IfcPipeSegment",
                "Name": "Pipe A",
                "Hierarchy": {"ParentId": "storey"},
                "Relationships": {
                    **empty_relation_block(),
                    "ifc_connected_to": ["conn-b"],
                },
            },
            {
                "GlobalId": "conn-b",
                "IfcType": "IfcPipeSegment",
                "Name": "Pipe B",
                "Hierarchy": {"ParentId": "storey"},
                "Relationships": {
                    **empty_relation_block(),
                    "ifc_connected_to": ["conn-a"],
                },
            },
        ],
    )

    graph = build_graph_from_jsonl([jsonl_path])

    assert _relations_between(graph, "Element::conn-a", "Element::conn-b") == [
        "ifc_connected_to"
    ]
    assert _relations_between(graph, "Element::conn-b", "Element::conn-a") == []


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


def test_shared_space_boundary_edges_are_explicit_and_ignore_openings(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "shared-boundary.jsonl"
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
                "GlobalId": "wall-shared",
                "IfcType": "IfcWall",
                "Name": "Shared Wall",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [1.9, 0.0, 0.0],
                        "max": [2.1, 4.0, 3.0],
                    },
                    "Centroid": [2.0, 2.0, 1.5],
                },
            },
            {
                "GlobalId": "opening-1",
                "IfcType": "IfcOpeningElement",
                "Name": "Opening",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [3.9, 0.0, 0.0],
                        "max": [4.1, 4.0, 3.0],
                    },
                    "Centroid": [4.0, 2.0, 1.5],
                },
            },
            {
                "GlobalId": "space-a",
                "IfcType": "IfcSpace",
                "Name": "Room A",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [0.0, 0.0, 0.0],
                        "max": [2.0, 4.0, 3.0],
                    },
                    "Centroid": [1.0, 2.0, 1.5],
                },
                "Relationships": {
                    **empty_relation_block(),
                    "space_bounded_by": ["wall-shared"],
                },
            },
            {
                "GlobalId": "space-b",
                "IfcType": "IfcSpace",
                "Name": "Room B",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [2.0, 0.0, 0.0],
                        "max": [4.0, 4.0, 3.0],
                    },
                    "Centroid": [3.0, 2.0, 1.5],
                },
                "Relationships": {
                    **empty_relation_block(),
                    "space_bounded_by": ["wall-shared", "opening-1"],
                },
            },
            {
                "GlobalId": "space-c",
                "IfcType": "IfcSpace",
                "Name": "Room C",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {
                        "min": [4.0, 0.0, 0.0],
                        "max": [6.0, 4.0, 3.0],
                    },
                    "Centroid": [5.0, 2.0, 1.5],
                },
                "Relationships": {
                    **empty_relation_block(),
                    "space_bounded_by": ["opening-1"],
                },
            },
        ],
    )

    graph = build_graph_from_jsonl([jsonl_path])
    add_spatial_adjacency(graph, threshold=3.5)

    edge_data = graph.get_edge_data("Element::space-a", "Element::space-b") or {}
    boundary_edges = [
        attrs
        for attrs in edge_data.values()
        if isinstance(attrs, dict) and attrs.get("relation") == "shares_boundary_with"
    ]
    assert len(boundary_edges) == 1
    assert boundary_edges[0]["source"] == "ifc"
    assert boundary_edges[0]["derived_from"] == "space_bounded_by"
    assert boundary_edges[0]["shared_boundary_elements"] == ["Element::wall-shared"]
    assert _relations_between(graph, "Element::space-b", "Element::space-a") == []

    assert "shares_boundary_with" not in _relations_between(
        graph, "Element::space-b", "Element::space-c"
    )
    heuristic_relations = set(
        _relations_between(graph, "Element::space-b", "Element::space-c")
    )
    assert heuristic_relations & {"adjacent_to", "connected_to"}
    assert _relations_between(graph, "Element::space-c", "Element::space-b") == []


def test_get_topology_neighbors_supports_aligned_with_and_same_storey_ordering() -> (
    None
):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Storey::L1",
        label="Level 1",
        class_="IfcBuildingStorey",
        dataset="single",
        properties={},
    )
    graph.add_node(
        "Storey::L2",
        label="Level 2",
        class_="IfcBuildingStorey",
        dataset="single",
        properties={},
    )
    graph.add_node(
        "Element::wall-a",
        label="Wall A",
        class_="IfcWall",
        dataset="single",
        properties={"GlobalId": "wall-a"},
        obb=_make_obb(
            center=(0.0, 0.0, 1.5),
            axis_u=(1.0, 0.0),
            axis_v=(0.0, 1.0),
            extent_u=5.0,
            extent_v=0.2,
        ),
    )
    graph.add_node(
        "Element::wall-b",
        label="Wall B",
        class_="IfcWall",
        dataset="single",
        properties={"GlobalId": "wall-b"},
        obb=_make_obb(
            center=(0.0, 0.15, 1.5),
            axis_u=(1.0, 0.0),
            axis_v=(0.0, 1.0),
            extent_u=4.5,
            extent_v=0.2,
        ),
    )
    graph.add_node(
        "Element::wall-c",
        label="Wall C",
        class_="IfcWall",
        dataset="single",
        properties={"GlobalId": "wall-c"},
        obb=_make_obb(
            center=(0.0, 0.1, 1.5),
            axis_u=(0.8660254, 0.5),
            axis_v=(-0.5, 0.8660254),
            extent_u=4.5,
            extent_v=0.2,
        ),
    )
    graph.add_node(
        "Element::wall-d",
        label="Wall D",
        class_="IfcWall",
        dataset="single",
        properties={"GlobalId": "wall-d"},
        obb=_make_obb(
            center=(0.0, 0.1, 4.5),
            axis_u=(1.0, 0.0),
            axis_v=(0.0, 1.0),
            extent_u=4.5,
            extent_v=0.2,
        ),
    )
    _add_contains_pair(graph, "Storey::L1", "Element::wall-a")
    _add_contains_pair(graph, "Storey::L1", "Element::wall-b")
    _add_contains_pair(graph, "Storey::L1", "Element::wall-c")
    _add_contains_pair(graph, "Storey::L2", "Element::wall-d")

    result = query_ifc_graph(
        graph,
        "get_topology_neighbors",
        {"element_id": "Element::wall-a", "relation": "aligned_with"},
    )

    assert result["status"] == "ok"
    neighbors = result["data"]["neighbors"]
    assert [item["id"] for item in neighbors] == ["Element::wall-b", "Element::wall-d"]
    assert neighbors[0]["same_storey"] is True
    assert neighbors[0]["storey_id"] == "Storey::L1"
    assert neighbors[1]["same_storey"] is False
    assert all(item["relation"] == "aligned_with" for item in neighbors)
    assert "Element::wall-c" not in {item["id"] for item in neighbors}


def test_get_topology_neighbors_aligned_with_rejects_unsupported_classes() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::column-a",
        label="Column A",
        class_="IfcColumn",
        dataset="single",
        properties={"GlobalId": "column-a"},
        obb=_make_obb(
            center=(0.0, 0.0, 0.0),
            axis_u=(1.0, 0.0),
            axis_v=(0.0, 1.0),
            extent_u=0.4,
            extent_v=0.4,
        ),
    )

    result = query_ifc_graph(
        graph,
        "get_topology_neighbors",
        {"element_id": "Element::column-a", "relation": "aligned_with"},
    )

    assert result["status"] == "ok"
    assert result["data"]["neighbors"] == []
    assert "elongated core classes" in result["data"]["warnings"][0]


def test_find_elements_inside_footprint_uses_plan_points_without_overlap_fallback() -> (
    None
):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::space-101",
        label="Room 101",
        class_="IfcSpace",
        dataset="single",
        properties={"GlobalId": "space-101"},
        footprint_polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    )
    graph.add_node(
        "Element::chair",
        label="Chair",
        class_="IfcFurniture",
        dataset="single",
        properties={"GlobalId": "chair"},
        geometry=(5.0, 5.0, 0.0),
    )
    graph.add_node(
        "Element::table",
        label="Table",
        class_="IfcFurniture",
        dataset="single",
        properties={"GlobalId": "table"},
        bbox=((1.0, 1.0, 0.0), (2.0, 2.0, 1.0)),
    )
    graph.add_node(
        "Element::cabinet",
        label="Cabinet",
        class_="IfcFurniture",
        dataset="single",
        properties={"GlobalId": "cabinet"},
        bbox=((9.5, 9.5, 0.0), (11.5, 11.5, 2.0)),
    )
    _add_contains_pair(graph, "Element::space-101", "Element::chair")
    _add_contains_pair(graph, "Element::space-101", "Element::table")
    _add_contains_pair(graph, "Element::space-101", "Element::cabinet")

    result = query_ifc_graph(
        graph,
        "find_elements_inside_footprint",
        {"container": "Element::space-101"},
    )

    assert result["status"] == "ok"
    elements = result["data"]["elements"]
    assert [item["id"] for item in elements] == ["Element::chair", "Element::table"]
    assert elements[0]["inside_method"] == "centroid"
    assert elements[1]["inside_method"] == "bbox_center_fallback"
    assert "Element::cabinet" not in {item["id"] for item in elements}


def test_find_same_storey_elements_uses_containment_then_level_fallback() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Storey::L1",
        label="Level 1",
        class_="IfcBuildingStorey",
        dataset="single",
        properties={},
    )
    graph.add_node(
        "Storey::L2",
        label="Level 2",
        class_="IfcBuildingStorey",
        dataset="single",
        properties={},
    )
    graph.add_node(
        "Element::anchor",
        label="Anchor Wall",
        class_="IfcWall",
        dataset="single",
        properties={"GlobalId": "anchor"},
    )
    graph.add_node(
        "Element::beam-l1",
        label="Beam L1",
        class_="IfcBeam",
        dataset="single",
        properties={"GlobalId": "beam-l1"},
    )
    graph.add_node(
        "Element::loose",
        label="Loose Element",
        class_="IfcWall",
        dataset="single",
        properties={"GlobalId": "loose", "Level": "Level 2"},
    )
    graph.add_node(
        "Element::beam-l2",
        label="Beam L2",
        class_="IfcBeam",
        dataset="single",
        properties={"GlobalId": "beam-l2"},
    )
    _add_contains_pair(graph, "Storey::L1", "Element::anchor")
    _add_contains_pair(graph, "Storey::L1", "Element::beam-l1")
    _add_contains_pair(graph, "Storey::L2", "Element::beam-l2")

    containment_result = query_ifc_graph(
        graph,
        "find_same_storey_elements",
        {"anchor": "Element::anchor", "class": "IfcBeam"},
    )
    assert containment_result["status"] == "ok"
    assert containment_result["data"]["storey_id"] == "Storey::L1"
    assert containment_result["data"]["storey_resolution"] == "contained_in"
    assert [item["id"] for item in containment_result["data"]["elements"]] == [
        "Element::beam-l1"
    ]

    level_result = query_ifc_graph(
        graph,
        "find_same_storey_elements",
        {"anchor": "Element::loose", "class": "IfcBeam"},
    )
    assert level_result["status"] == "ok"
    assert level_result["data"]["storey_id"] == "Storey::L2"
    assert level_result["data"]["storey_resolution"] == "level"
    assert [item["id"] for item in level_result["data"]["elements"]] == [
        "Element::beam-l2"
    ]


def test_find_same_storey_elements_fails_closed_without_storey_resolution() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::orphan",
        label="Orphan",
        class_="IfcWall",
        dataset="single",
        properties={"GlobalId": "orphan"},
    )

    result = query_ifc_graph(
        graph,
        "find_same_storey_elements",
        {"anchor": "Element::orphan"},
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "not_found"
    assert "containing storey" in result["error"]["message"]


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
        "edges_added": 1,
        "threshold": 2.0,
    }
    assert pruning["phases"]["topology"] == {
        "excluded_classes": ["IfcPlate"],
        "eligible_nodes": 2,
        "skipped_nodes": 1,
        "skipped_by_class": {"IfcPlate": 1},
        "edges_added": 1,
    }


def test_overlap_xy_threshold_mode_suppresses_low_ratio_edges_but_keeps_verticals() -> (
    None
):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::roof",
        class_="IfcSlab",
        bbox=((0.0, 0.0, 2.0), (10.0, 10.0, 3.0)),
        properties={"GlobalId": "roof"},
        dataset="single",
    )
    graph.add_node(
        "Element::floor",
        class_="IfcSlab",
        bbox=((9.0, 0.0, 0.0), (19.0, 10.0, 1.0)),
        properties={"GlobalId": "floor"},
        dataset="single",
    )

    add_topology_facts(
        graph,
        overlap_xy_mode="threshold",
        overlap_xy_min_ratio=0.2,
    )

    assert "overlaps_xy" not in _relations_between(
        graph,
        "Element::roof",
        "Element::floor",
    )
    assert "above" in _relations_between(graph, "Element::roof", "Element::floor")
    assert "below" in _relations_between(graph, "Element::floor", "Element::roof")

    overlaps = query_ifc_graph(
        graph,
        "get_topology_neighbors",
        {"element_id": "Element::roof", "relation": "overlaps_xy"},
    )
    assert overlaps["status"] == "ok"
    assert overlaps["data"]["neighbors"] == []

    overlap_stats = graph.graph["graph_build"]["overlap_xy"]
    assert overlap_stats["mode"] == "threshold"
    assert overlap_stats["candidate_positive_overlap_pairs"] == 1
    assert overlap_stats["emitted_overlap_pairs"] == 0
    assert overlap_stats["rejected_by_threshold"] == 1
    assert overlap_stats["rejected_by_top_k"] == 0


def test_overlap_xy_top_k_mode_keeps_symmetric_strongest_pairs() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::a",
        class_="IfcSlab",
        bbox=((0.0, 0.0, 0.0), (10.0, 10.0, 1.0)),
        properties={"GlobalId": "a"},
        dataset="single",
    )
    graph.add_node(
        "Element::b",
        class_="IfcSlab",
        bbox=((1.0, 0.0, 0.0), (11.0, 10.0, 1.0)),
        properties={"GlobalId": "b"},
        dataset="single",
    )
    graph.add_node(
        "Element::c",
        class_="IfcSlab",
        bbox=((6.0, 0.0, 0.0), (16.0, 10.0, 1.0)),
        properties={"GlobalId": "c"},
        dataset="single",
    )

    add_topology_facts(graph, overlap_xy_mode="top_k", overlap_xy_top_k=1)

    assert "overlaps_xy" in _relations_between(graph, "Element::a", "Element::b")
    assert "overlaps_xy" in _relations_between(graph, "Element::b", "Element::c")
    assert "overlaps_xy" not in _relations_between(graph, "Element::a", "Element::c")

    overlap_stats = graph.graph["graph_build"]["overlap_xy"]
    assert overlap_stats["mode"] == "top_k"
    assert overlap_stats["candidate_positive_overlap_pairs"] == 3
    assert overlap_stats["emitted_overlap_pairs"] == 2
    assert overlap_stats["rejected_by_threshold"] == 0
    assert overlap_stats["rejected_by_top_k"] == 1


def test_overlap_xy_none_mode_preserves_vertical_helpers() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::upper",
        class_="IfcSlab",
        bbox=((0.0, 0.0, 2.0), (3.0, 3.0, 3.0)),
        properties={"GlobalId": "upper"},
        dataset="single",
    )
    graph.add_node(
        "Element::lower",
        class_="IfcSlab",
        bbox=((0.0, 0.0, 0.0), (3.0, 3.0, 1.0)),
        properties={"GlobalId": "lower"},
        dataset="single",
    )

    add_topology_facts(graph, overlap_xy_mode="none")

    overlaps = query_ifc_graph(
        graph,
        "get_topology_neighbors",
        {"element_id": "Element::upper", "relation": "overlaps_xy"},
    )
    assert overlaps["status"] == "ok"
    assert overlaps["data"]["neighbors"] == []

    above = query_ifc_graph(
        graph,
        "find_elements_above",
        {"element_id": "Element::lower"},
    )
    below = query_ifc_graph(
        graph,
        "find_elements_below",
        {"element_id": "Element::upper"},
    )
    assert above["status"] == "ok"
    assert [item["id"] for item in above["data"]["results"]] == ["Element::upper"]
    assert below["status"] == "ok"
    assert [item["id"] for item in below["data"]["results"]] == ["Element::lower"]

    overlap_stats = graph.graph["graph_build"]["overlap_xy"]
    assert overlap_stats["mode"] == "none"
    assert overlap_stats["candidate_positive_overlap_pairs"] == 1
    assert overlap_stats["emitted_overlap_pairs"] == 0


def test_batch34_spatial_pruning_keeps_unaffected_threshold_and_edges_stable() -> None:
    def _make_graph() -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for node_id, class_name, x in [
            ("Element::wall-a", "IfcWall", 0.0),
            ("Element::member-a", "IfcMember", 0.1),
            ("Element::wall-b", "IfcWall", 1.1),
            ("Element::member-b", "IfcMember", 1.2),
            ("Element::wall-c", "IfcWall", 2.2),
            ("Element::member-c", "IfcMember", 2.3),
        ]:
            graph.add_node(
                node_id,
                class_=class_name,
                bbox=((x, 0.0, 0.0), (x + 0.05, 0.05, 0.05)),
                geometry=(x + 0.025, 0.025, 0.025),
                properties={"GlobalId": node_id.removeprefix("Element::")},
                dataset="single",
            )
        return graph

    unaffected_nodes = {"Element::wall-a", "Element::wall-b", "Element::wall-c"}
    excluded_nodes = {"Element::member-a", "Element::member-b", "Element::member-c"}

    unpruned = _make_graph()
    unpruned_threshold = add_spatial_adjacency(unpruned)

    pruned = _make_graph()
    pruned_threshold = add_spatial_adjacency(pruned, exclude_classes={"IfcMember"})

    unpruned_unaffected_edges = {
        (u, v, attrs["relation"])
        for u, v, attrs in unpruned.edges(data=True)
        if attrs.get("source") == "heuristic"
        and u in unaffected_nodes
        and v in unaffected_nodes
    }
    pruned_unaffected_edges = {
        (u, v, attrs["relation"])
        for u, v, attrs in pruned.edges(data=True)
        if attrs.get("source") == "heuristic"
        and u in unaffected_nodes
        and v in unaffected_nodes
    }

    assert unpruned_threshold == pytest.approx(pruned_threshold)
    assert unpruned_threshold == pytest.approx(0.5)
    assert pruned_unaffected_edges == unpruned_unaffected_edges
    assert pruned_unaffected_edges == set()
    assert not any(
        attrs.get("source") == "heuristic" and ({u, v} & excluded_nodes)
        for u, v, attrs in pruned.edges(data=True)
    )


def test_batch3_bigbuilding_facade_snapshot_keeps_threshold_and_semantics_stable(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "bigbuilding-facade-snapshot.jsonl"

    def _facade_element(
        global_id: str,
        ifc_type: str,
        *,
        x_min: float,
        x_max: float,
        typed_by: str | None = None,
    ) -> dict:
        record = {
            "GlobalId": global_id,
            "IfcType": ifc_type,
            "Name": global_id,
            "Hierarchy": {"ParentId": "storey"},
            "Geometry": {
                "BoundingBox": {
                    "min": [x_min, -0.7211000014551636, 0.0],
                    "max": [x_max, -0.5711000014551637, 1.2650000000000001],
                },
                "Centroid": [
                    (x_min + x_max) / 2,
                    -0.6461000014551636,
                    0.6325000000000001,
                ],
            },
        }
        if typed_by is not None:
            record["Relationships"] = {
                **empty_relation_block(),
                "typed_by": [typed_by],
            }
        return record

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
                "Name": "01 - Entry Level",
                "Hierarchy": {"ParentId": "bldg"},
            },
            _facade_element(
                "member-a",
                "IfcMember",
                x_min=0.37970000225259104,
                x_max=0.4297000022525911,
                typed_by="member-type",
            ),
            _facade_element(
                "panel-a",
                "IfcPlate",
                x_min=0.4297000022525914,
                x_max=2.3797000022525916,
                typed_by="panel-type",
            ),
            _facade_element(
                "member-b",
                "IfcMember",
                x_min=2.379700002252591,
                x_max=2.429700002252591,
                typed_by="member-type",
            ),
            _facade_element(
                "member-c",
                "IfcMember",
                x_min=4.379700002252591,
                x_max=4.429700002252591,
                typed_by="member-type",
            ),
            _facade_element(
                "panel-b",
                "IfcPlate",
                x_min=4.4297000022525905,
                x_max=6.379700002252591,
                typed_by="panel-type",
            ),
            _facade_element(
                "member-d",
                "IfcMember",
                x_min=6.379700002252591,
                x_max=6.429700002252591,
                typed_by="member-type",
            ),
            {
                "GlobalId": "member-type",
                "IfcType": "IfcMemberType",
                "Name": "Rectangular Mullion:50 x 150mm",
            },
            {
                "GlobalId": "panel-type",
                "IfcType": "IfcPlateType",
                "Name": "Facade Panel Type",
            },
        ],
    )

    unaffected_nodes = {"Element::panel-a", "Element::panel-b"}
    excluded_nodes = {
        "Element::member-a",
        "Element::member-b",
        "Element::member-c",
        "Element::member-d",
    }

    unpruned = build_graph_from_jsonl([jsonl_path])
    unpruned_threshold = add_spatial_adjacency(unpruned)
    add_topology_facts(unpruned)

    pruned = build_graph_from_jsonl([jsonl_path])
    pruned_threshold = add_spatial_adjacency(pruned, exclude_classes={"IfcMember"})
    add_topology_facts(pruned, exclude_classes={"IfcMember"})

    unpruned_unaffected_edges = {
        (u, v, attrs["relation"])
        for u, v, attrs in unpruned.edges(data=True)
        if attrs.get("source") == "heuristic"
        and u in unaffected_nodes
        and v in unaffected_nodes
    }
    pruned_unaffected_edges = {
        (u, v, attrs["relation"])
        for u, v, attrs in pruned.edges(data=True)
        if attrs.get("source") == "heuristic"
        and u in unaffected_nodes
        and v in unaffected_nodes
    }

    assert unpruned_threshold == pytest.approx(pruned_threshold)
    assert pruned_threshold == pytest.approx(1.5)
    assert pruned.graph["graph_build"]["derived_edge_pruning"]["phases"][
        "spatial_adjacency"
    ]["threshold"] == pytest.approx(1.5)
    assert pruned_unaffected_edges == unpruned_unaffected_edges
    assert pruned_unaffected_edges == set()
    assert any(
        attrs.get("source") in {"heuristic", "topology"} and ({u, v} & excluded_nodes)
        for u, v, attrs in unpruned.edges(data=True)
    )
    assert not any(
        attrs.get("source") in {"heuristic", "topology"} and ({u, v} & excluded_nodes)
        for u, v, attrs in pruned.edges(data=True)
    )

    assert pruned.has_edge("Storey::storey", "Element::panel-a")
    assert pruned.has_edge("Storey::storey", "Element::member-a")

    panel_type_edges = (
        pruned.get_edge_data("Element::panel-a", "Element::panel-type") or {}
    )
    assert any(
        attrs.get("relation") == "typed_by" and attrs.get("source") == "ifc"
        for attrs in panel_type_edges.values()
    )

    member_type_edges = (
        pruned.get_edge_data("Element::member-a", "Element::member-type") or {}
    )
    assert any(
        attrs.get("relation") == "typed_by" and attrs.get("source") == "ifc"
        for attrs in member_type_edges.values()
    )


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


def test_build_graph_honors_overlap_xy_config_and_explicit_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "overlap-build-graph.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"GlobalId": "proj", "IfcType": "IfcProject", "Name": "Project"},
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
                "GlobalId": "roof",
                "IfcType": "IfcSlab",
                "Name": "Roof",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {"min": [0.0, 0.0, 2.0], "max": [3.0, 3.0, 3.0]},
                    "Centroid": [1.5, 1.5, 2.5],
                },
            },
            {
                "GlobalId": "floor",
                "IfcType": "IfcSlab",
                "Name": "Floor",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {"min": [0.0, 0.0, 0.0], "max": [3.0, 3.0, 1.0]},
                    "Centroid": [1.5, 1.5, 0.5],
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
                        "overlap_xy": {"mode": "none", "min_ratio": 0.2, "top_k": 5}
                    }
                }
            )
        ),
    )

    configured = build_graph([jsonl_path], payload_mode="minimal")
    assert configured.graph["graph_build"]["overlap_xy"]["mode"] == "none"
    assert "overlaps_xy" not in _relations_between(
        configured,
        "Element::roof",
        "Element::floor",
    )
    assert "above" in _relations_between(configured, "Element::roof", "Element::floor")

    overridden = build_graph(
        [jsonl_path],
        payload_mode="minimal",
        overlap_xy_mode="full",
    )
    assert overridden.graph["graph_build"]["overlap_xy"]["mode"] == "full"
    assert "overlaps_xy" in _relations_between(
        overridden,
        "Element::floor",
        "Element::roof",
    )
    assert "overlaps_xy" not in _relations_between(
        overridden,
        "Element::roof",
        "Element::floor",
    )


def test_build_graph_defaults_to_no_overlap_xy_edges(
    monkeypatch,
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "overlap-default-none.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"GlobalId": "proj", "IfcType": "IfcProject", "Name": "Project"},
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
                "GlobalId": "roof",
                "IfcType": "IfcSlab",
                "Name": "Roof",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {"min": [0.0, 0.0, 2.0], "max": [3.0, 3.0, 3.0]},
                    "Centroid": [1.5, 1.5, 2.5],
                },
            },
            {
                "GlobalId": "floor",
                "IfcType": "IfcSlab",
                "Name": "Floor",
                "Hierarchy": {"ParentId": "storey"},
                "Geometry": {
                    "BoundingBox": {"min": [0.0, 0.0, 0.0], "max": [3.0, 3.0, 1.0]},
                    "Centroid": [1.5, 1.5, 0.5],
                },
            },
        ],
    )

    monkeypatch.setattr(
        "rag_tag.parser.jsonl_to_graph.load_project_config",
        lambda _start_dir=None: SimpleNamespace(config=AppConfig.model_validate({})),
    )

    graph = build_graph([jsonl_path], payload_mode="minimal")

    assert graph.graph["graph_build"]["overlap_xy"]["mode"] == "none"
    assert "overlaps_xy" not in _relations_between(
        graph,
        "Element::roof",
        "Element::floor",
    )
    assert "above" in _relations_between(graph, "Element::roof", "Element::floor")


def test_jsonl_to_graph_cli_passes_overlap_xy_flags(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    jsonl_dir = tmp_path / "jsonl"
    out_dir = tmp_path / "out"
    jsonl_dir.mkdir()
    out_dir.mkdir()
    (jsonl_dir / "fixture.jsonl").write_text("{}\n", encoding="utf-8")

    build_calls: list[dict[str, object]] = []

    def fake_build_graph(
        jsonl_paths,
        dataset=None,
        payload_mode=None,
        *,
        derived_edge_pruning_enabled=None,
        derived_edge_prune_classes=None,
        overlap_xy_mode=None,
        overlap_xy_min_ratio=None,
        overlap_xy_top_k=None,
    ):
        del (
            dataset,
            payload_mode,
            derived_edge_pruning_enabled,
            derived_edge_prune_classes,
        )
        build_calls.append(
            {
                "jsonl_paths": list(jsonl_paths),
                "overlap_xy_mode": overlap_xy_mode,
                "overlap_xy_min_ratio": overlap_xy_min_ratio,
                "overlap_xy_top_k": overlap_xy_top_k,
            }
        )
        graph = nx.MultiDiGraph()
        graph.graph["graph_build"] = {
            "overlap_xy": {
                "mode": overlap_xy_mode or "none",
                "min_ratio": (
                    overlap_xy_min_ratio if overlap_xy_min_ratio is not None else 0.2
                ),
                "top_k": overlap_xy_top_k if overlap_xy_top_k is not None else 5,
            }
        }
        graph.add_node("Element::A")
        return graph

    viz_calls: list[Path] = []

    monkeypatch.setattr("rag_tag.parser.jsonl_to_graph.build_graph", fake_build_graph)
    monkeypatch.setattr(
        "rag_tag.parser.jsonl_to_graph.plot_interactive_graph",
        lambda graph, path: viz_calls.append(path),
    )
    monkeypatch.setattr(
        "rag_tag.parser.jsonl_to_graph.find_project_root",
        lambda _start_dir=None: tmp_path,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "rag-tag-jsonl-to-graph",
            "--jsonl-dir",
            str(jsonl_dir),
            "--out-dir",
            str(out_dir),
            "--overlap-xy-mode",
            "top_k",
            "--overlap-xy-min-ratio",
            "0.35",
            "--overlap-xy-top-k",
            "7",
        ],
    )

    jsonl_to_graph_main()
    output = capsys.readouterr().out

    assert len(build_calls) == 1
    assert build_calls[0]["overlap_xy_mode"] == "top_k"
    assert build_calls[0]["overlap_xy_min_ratio"] == 0.35
    assert build_calls[0]["overlap_xy_top_k"] == 7
    assert viz_calls == [out_dir / "ifc_graph.html"]
    assert "overlap_xy: mode=top_k min_ratio=0.35 top_k=7" in output


def test_plot_interactive_graph_legend_includes_edge_counts(tmp_path: Path) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "IfcBuilding",
        label="Building",
        class_="IfcBuilding",
        geometry=(0.0, 0.0, 0.0),
        properties={},
    )
    graph.add_node(
        "Storey::L1",
        label="Level 1",
        class_="IfcBuildingStorey",
        geometry=(0.0, 1.0, 0.0),
        properties={},
    )
    graph.add_node(
        "Element::A",
        label="Wall A",
        class_="IfcWall",
        geometry=(1.0, 0.0, 0.0),
        properties={},
    )
    graph.add_node(
        "Element::B",
        label="Wall B",
        class_="IfcWall",
        geometry=(1.0, 1.0, 0.0),
        properties={},
    )

    graph.add_edge("IfcBuilding", "Storey::L1", relation="aggregates")
    graph.add_edge("Storey::L1", "Element::A", relation="contains")
    graph.add_edge("Storey::L1", "Element::B", relation="contains")
    graph.add_edge(
        "Element::A",
        "Element::B",
        relation="adjacent_to",
        source="heuristic",
    )
    graph.add_edge(
        "Element::B",
        "Element::A",
        relation="adjacent_to",
        source="heuristic",
    )

    out_html = tmp_path / "ifc_graph.html"
    plot_interactive_graph(graph, out_html)

    html_text = out_html.read_text(encoding="utf-8")
    legend_start = html_text.index('<aside class="legend" aria-label="Graph legend">')
    legend_end = html_text.index("</aside>", legend_start)
    legend_html = html_text[legend_start:legend_end]

    assert "Total edges shown: 6" in legend_html
    assert "Edge: contains / contained_in" in legend_html
    assert (
        "containment hierarchy; source-&gt;target = contains, "
        "target-&gt;source = contained_in; contains=2, contained_in=0" in legend_html
    )
    assert re.search(
        (
            r"Edge: aggregates</span>"
            r"<span class='legend-item-sub'>Added during graph build for the "
            r"top-level IFC breakdown: "
            r"Project -&gt; Building and Building -&gt; Storey\.</span>"
            r"</span><span class='legend-count' title='Edge count'>1</span>"
        ),
        legend_html,
    )
    assert re.search(
        (
            r"Edge: adjacent_to</span>"
            r"<span class='legend-item-sub'>Derived from geometry when "
            r"elements fall within the "
            r"distance threshold but are not verified as touching or "
            r"intersecting\.</span>"
            r"</span><span class='legend-count' title='Edge count'>1</span>"
        ),
        legend_html,
    )
    assert re.search(
        (
            r"Edge: same_storey_as</span>"
            r"<span class='legend-item-sub'>Visualization overlay connecting "
            r"nodes that resolve to the same "
            r"IfcBuildingStorey\.</span>"
            r"</span><span class='legend-count' title='Edge count'>2</span>"
        ),
        legend_html,
    )


def test_plot_interactive_graph_merges_inverse_legend_rows(tmp_path: Path) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::roof",
        label="Roof",
        class_="IfcSlab",
        geometry=(0.0, 0.0, 2.5),
        properties={},
    )
    graph.add_node(
        "Element::floor",
        label="Floor",
        class_="IfcSlab",
        geometry=(0.0, 0.0, 0.5),
        properties={},
    )
    graph.add_edge(
        "Element::roof",
        "Element::floor",
        relation="above",
        source="topology",
    )
    graph.add_edge(
        "Element::floor",
        "Element::roof",
        relation="below",
        source="topology",
    )

    out_html = tmp_path / "ifc_graph_inverse_rows.html"
    plot_interactive_graph(graph, out_html)

    html_text = out_html.read_text(encoding="utf-8")
    legend_start = html_text.index('<aside class="legend" aria-label="Graph legend">')
    legend_end = html_text.index("</aside>", legend_start)
    legend_html = html_text[legend_start:legend_end]

    assert "Total edges shown: 1" in legend_html
    assert "Edge: above / below" in legend_html
    assert (
        "vertical ordering; source-&gt;target = above, target-&gt;source = below; "
        "above=1, below=1" in legend_html
    )
    assert "Edge: above</span>" not in legend_html
    assert "Edge: below</span>" not in legend_html
    assert re.search(
        (
            r"Edge: above / below</span>.*?"
            r"<span class='legend-count' title='Edge count'>1</span>"
        ),
        legend_html,
    )


def test_plot_interactive_graph_inverse_legend_rows_handle_missing_reverse(
    tmp_path: Path,
) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Element::roof",
        label="Roof",
        class_="IfcSlab",
        geometry=(0.0, 0.0, 2.5),
        properties={},
    )
    graph.add_node(
        "Element::floor",
        label="Floor",
        class_="IfcSlab",
        geometry=(0.0, 0.0, 0.5),
        properties={},
    )
    graph.add_edge(
        "Element::roof",
        "Element::floor",
        relation="above",
        source="topology",
    )

    out_html = tmp_path / "ifc_graph_inverse_rows_incomplete.html"
    plot_interactive_graph(graph, out_html)

    html_text = out_html.read_text(encoding="utf-8")
    legend_start = html_text.index('<aside class="legend" aria-label="Graph legend">')
    legend_end = html_text.index("</aside>", legend_start)
    legend_html = html_text[legend_start:legend_end]

    assert "Total edges shown: 1" in legend_html
    assert (
        "vertical ordering; source-&gt;target = above, target-&gt;source = below; "
        "above=1, below=0" in legend_html
    )
    assert re.search(
        (
            r"Edge: above / below</span>.*?"
            r"<span class='legend-count' title='Edge count'>1</span>"
        ),
        legend_html,
    )


def test_plot_interactive_graph_includes_derived_relations_as_edges(
    tmp_path: Path,
) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Storey::L1",
        label="Level 1",
        class_="IfcBuildingStorey",
        geometry=(0.0, 0.0, 0.0),
        properties={},
    )
    graph.add_node(
        "Space::Room1",
        label="Room 1",
        class_="IfcSpace",
        geometry=(2.0, 2.0, 0.0),
        footprint_bbox_2d=(0.0, 0.0, 6.0, 6.0),
        properties={},
    )
    graph.add_node(
        "Element::wall-a",
        label="Wall A",
        class_="IfcWall",
        geometry=(1.0, 1.0, 1.5),
        bbox=((0.5, 0.5, 0.0), (1.5, 1.5, 3.0)),
        obb=_make_obb(
            center=(1.0, 1.0, 1.5),
            axis_u=(0.0, 1.0),
            axis_v=(1.0, 0.0),
            extent_u=2.0,
            extent_v=0.1,
            extent_z=1.5,
        ),
        properties={},
    )
    graph.add_node(
        "Element::wall-b",
        label="Wall B",
        class_="IfcWall",
        geometry=(1.0, 3.0, 1.5),
        bbox=((0.5, 2.5, 0.0), (1.5, 3.5, 3.0)),
        obb=_make_obb(
            center=(1.0, 3.0, 1.5),
            axis_u=(0.0, 1.0),
            axis_v=(1.0, 0.0),
            extent_u=2.0,
            extent_v=0.1,
            extent_z=1.5,
        ),
        properties={},
    )

    _add_contains_pair(graph, "Storey::L1", "Space::Room1")
    _add_contains_pair(graph, "Space::Room1", "Element::wall-a")
    _add_contains_pair(graph, "Space::Room1", "Element::wall-b")

    out_html = tmp_path / "ifc_graph_with_derived_edges.html"
    plot_interactive_graph(graph, out_html)

    html_text = out_html.read_text(encoding="utf-8")
    legend_start = html_text.index('<aside class="legend" aria-label="Graph legend">')
    legend_end = html_text.index("</aside>", legend_start)
    legend_html = html_text[legend_start:legend_end]

    assert "Helper Overlays" not in html_text
    assert "toggle-helper-overlays" not in html_text
    assert "Total edges shown: 8" in legend_html
    assert "Edge: aligned_with" in legend_html
    assert (
        "Visualization overlay for eligible linear elements with similar plan "
        "direction and small lateral offset." in legend_html
    )
    assert "Edge: inside_footprint_of" in legend_html
    assert (
        "Visualization overlay when an IfcSpace centroid falls inside another "
        "element&#x27;s 2D footprint on the same storey." in legend_html
    )
    assert "Edge: same_storey_as" in legend_html
    assert (
        "Visualization overlay connecting nodes that resolve to the same "
        "IfcBuildingStorey." in legend_html
    )
    assert "Overlay: aligned_with" not in html_text
    assert re.search(
        r"Edge: aligned_with</span>.*?"
        r"<span class='legend-count' title='Edge count'>1</span>",
        legend_html,
    )
    assert re.search(
        r"Edge: inside_footprint_of</span>.*?"
        r"<span class='legend-count' title='Edge count'>2</span>",
        legend_html,
    )
    assert re.search(
        r"Edge: same_storey_as</span>.*?"
        r"<span class='legend-count' title='Edge count'>2</span>",
        legend_html,
    )


def test_plot_interactive_graph_overlap_modes_includes_toggle_controls(
    tmp_path: Path,
) -> None:
    full_graph = nx.MultiDiGraph()
    none_graph = nx.MultiDiGraph()
    for graph in (full_graph, none_graph):
        graph.graph["edge_categories"] = {
            "hierarchy": ["contains"],
            "spatial": [],
            "topology": ["overlaps_xy", "above", "below"],
            "explicit": [],
        }
        graph.add_node(
            "Element::roof",
            label="Roof",
            class_="IfcSlab",
            geometry=(0.0, 0.0, 2.5),
            bbox=((0.0, 0.0, 2.0), (3.0, 3.0, 3.0)),
            properties={},
        )
        graph.add_node(
            "Element::floor",
            label="Floor",
            class_="IfcSlab",
            geometry=(0.0, 0.0, 0.5),
            bbox=((0.0, 0.0, 0.0), (3.0, 3.0, 1.0)),
            properties={},
        )
        graph.add_edge(
            "Element::roof",
            "Element::floor",
            relation="above",
            source="topology",
        )
        graph.add_edge(
            "Element::floor",
            "Element::roof",
            relation="below",
            source="topology",
        )

    full_graph.add_edge(
        "Element::roof",
        "Element::floor",
        relation="overlaps_xy",
        source="topology",
        overlap_area_xy=9.0,
        overlap_ratio_xy=1.0,
    )
    full_graph.add_edge(
        "Element::floor",
        "Element::roof",
        relation="overlaps_xy",
        source="topology",
        overlap_area_xy=9.0,
        overlap_ratio_xy=1.0,
    )

    out_html = tmp_path / "ifc_graph_overlap_modes.html"
    plot_interactive_graph_overlap_modes(
        {"full": full_graph, "none": none_graph},
        out_html,
    )

    html_text = out_html.read_text(encoding="utf-8")
    assert 'data-overlap-mode="full"' in html_text
    assert 'data-overlap-mode="none"' in html_text
    assert "Active overlap mode: full" in html_text
    assert '"full": {"total_edges": "2"' in html_text
    assert '"none": {"total_edges": "1"' in html_text
    assert "const legendPayloads =" in html_text
    assert "overlaps_xy" in html_text
    assert "Edge: above / below" in html_text
    assert (
        "vertical ordering; source-&gt;target = above, target-&gt;source = below; "
        "above=1, below=1" in html_text
    )
