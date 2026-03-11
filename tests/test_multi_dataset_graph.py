from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.parser.jsonl_to_graph import build_graph_from_jsonl
from rag_tag.query_service import execute_query
from rag_tag.router.models import RouteDecision


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _build_two_dataset_graph(tmp_path: Path) -> nx.MultiDiGraph:
    dataset_a = tmp_path / "model-a.jsonl"
    dataset_b = tmp_path / "model-b.jsonl"

    shared_element_gid = "shared-elem"
    shared_storey_gid = "shared-storey"

    _write_jsonl(
        dataset_a,
        [
            {
                "ExpressId": 1,
                "GlobalId": "proj-a",
                "IfcType": "IfcProject",
                "Name": "Proj A",
            },
            {
                "ExpressId": 2,
                "GlobalId": "bldg-a",
                "IfcType": "IfcBuilding",
                "Name": "Bldg A",
                "Hierarchy": {"ParentId": "proj-a"},
            },
            {
                "ExpressId": 3,
                "GlobalId": shared_storey_gid,
                "IfcType": "IfcBuildingStorey",
                "Name": "Level A",
                "Hierarchy": {"ParentId": "bldg-a"},
            },
            {
                "ExpressId": 4,
                "GlobalId": shared_element_gid,
                "IfcType": "IfcDoor",
                "Name": "Door A",
                "Hierarchy": {"ParentId": shared_storey_gid},
                "Relationships": {"belongs_to_system": ["Shared System"]},
            },
        ],
    )
    _write_jsonl(
        dataset_b,
        [
            {
                "ExpressId": 11,
                "GlobalId": "proj-b",
                "IfcType": "IfcProject",
                "Name": "Proj B",
            },
            {
                "ExpressId": 12,
                "GlobalId": "bldg-b",
                "IfcType": "IfcBuilding",
                "Name": "Bldg B",
                "Hierarchy": {"ParentId": "proj-b"},
            },
            {
                "ExpressId": 13,
                "GlobalId": shared_storey_gid,
                "IfcType": "IfcBuildingStorey",
                "Name": "Level B",
                "Hierarchy": {"ParentId": "bldg-b"},
            },
            {
                "ExpressId": 14,
                "GlobalId": shared_element_gid,
                "IfcType": "IfcDoor",
                "Name": "Door B",
                "Hierarchy": {"ParentId": shared_storey_gid},
                "Relationships": {"belongs_to_system": ["Shared System"]},
            },
        ],
    )

    return build_graph_from_jsonl([dataset_a, dataset_b])


def test_namespaced_graph_ids_prevent_node_collision(tmp_path: Path) -> None:
    graph = _build_two_dataset_graph(tmp_path)

    assert "Element::model-a::shared-elem" in graph
    assert "Element::model-b::shared-elem" in graph
    assert graph.nodes["Element::model-a::shared-elem"]["label"] == "Door A"
    assert graph.nodes["Element::model-b::shared-elem"]["label"] == "Door B"


def test_containment_and_context_edges_stay_within_dataset(tmp_path: Path) -> None:
    graph = _build_two_dataset_graph(tmp_path)

    assert graph.has_edge(
        "Storey::model-a::shared-storey", "Element::model-a::shared-elem"
    )
    assert graph.has_edge(
        "Storey::model-b::shared-storey", "Element::model-b::shared-elem"
    )
    assert not graph.has_edge(
        "Storey::model-a::shared-storey",
        "Element::model-b::shared-elem",
    )
    assert "System::model-a::Shared System" in graph
    assert "System::model-b::Shared System" in graph


def test_graph_query_requires_explicit_dataset_when_multiple_models_present() -> None:
    decision = RouteDecision(route="graph", reason="graph route", sql_request=None)
    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["model-a", "model-b"]

    bundle = execute_query(
        "Which rooms are adjacent to the kitchen?",
        db_paths=[],
        runtime=graph,
        agent=None,
        decision=decision,
        graph_dataset=None,
    )

    assert "Multiple graph datasets are available" in bundle["result"]["error"]


def test_legacy_globalid_resolution_reports_ambiguity_for_collisions(
    tmp_path: Path,
) -> None:
    graph = _build_two_dataset_graph(tmp_path)

    result = query_ifc_graph(
        graph, "get_element_properties", {"element_id": "shared-elem"}
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "ambiguous"
    assert sorted(result["error"]["details"]["candidates"]) == [
        "Element::model-a::shared-elem",
        "Element::model-b::shared-elem",
    ]


def test_legacy_globalid_resolution_still_works_when_unique(tmp_path: Path) -> None:
    dataset = tmp_path / "single.jsonl"
    _write_jsonl(
        dataset,
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
                "GlobalId": "unique-door",
                "IfcType": "IfcDoor",
                "Name": "Door",
                "Hierarchy": {"ParentId": "storey"},
            },
        ],
    )
    graph = build_graph_from_jsonl([dataset])

    result = query_ifc_graph(
        graph, "get_element_properties", {"element_id": "unique-door"}
    )

    assert result["status"] == "ok"
    assert result["data"]["id"] == "Element::unique-door"
