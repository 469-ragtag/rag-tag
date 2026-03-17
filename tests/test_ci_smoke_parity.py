from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from rag_tag.graph import GraphRuntime, wrap_networkx_graph
from rag_tag.graph_contract import (
    CANONICAL_RELATION_SET,
    EXPLICIT_IFC_RELATIONS,
    HIERARCHY_RELATIONS,
    KNOWN_RELATION_SOURCES,
    ROADMAP_ACTIONS,
    SPATIAL_RELATIONS,
    build_evidence_item,
    ensure_action_data_fields,
    has_valid_envelope_shape,
    is_allowed_action,
    missing_required_action_fields,
    normalize_relation_name,
    normalize_relation_source,
    relation_bucket,
)
from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.parser.jsonl_to_graph import build_graph_from_jsonl
from rag_tag.parser.jsonl_to_sql import jsonl_to_sql
from rag_tag.query_service import execute_query
from rag_tag.router.models import RouteDecision
from rag_tag.run_agent import _resolve_graph_dataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _iter_edge_attrs(graph: nx.Graph, u: str, v: str) -> list[dict[str, Any]]:
    edge_data = graph.get_edge_data(u, v)
    if edge_data is None:
        return []
    if graph.is_multigraph():
        if isinstance(edge_data, dict):
            return [attrs for attrs in edge_data.values() if isinstance(attrs, dict)]
        return []
    if isinstance(edge_data, dict):
        return [edge_data]
    return []


def _has_edge_with(
    graph: nx.Graph,
    u: str,
    v: str,
    *,
    relation: str | None = None,
    source: str | None = None,
) -> bool:
    for attrs in _iter_edge_attrs(graph, u, v):
        if relation is not None and attrs.get("relation") != relation:
            continue
        if source is not None and attrs.get("source") != source:
            continue
        return True
    return False


def _source_matches_bucket_semantics(item: dict[str, Any]) -> bool:
    relation = normalize_relation_name(item.get("relation"))
    source = normalize_relation_source(item.get("source"))
    bucket = relation_bucket(relation)
    if bucket == "explicit_ifc":
        return source == "ifc"
    if bucket == "topology":
        return source == "topology"
    if bucket == "spatial":
        return source == "heuristic"
    if bucket == "hierarchy":
        return source is None
    return False


def test_batch6_dataset_resolution_priority_and_query_propagation(monkeypatch) -> None:
    assert _resolve_graph_dataset("Explicit", None) == "Explicit"
    assert _resolve_graph_dataset("Explicit", Path("/x/Other.db")) == "Explicit"
    assert _resolve_graph_dataset(None, Path("/data/MyBuilding.db")) == "MyBuilding"
    assert _resolve_graph_dataset(None, None) is None

    captured: dict[str, Any] = {}

    def fake_execute_graph_query(
        question: str,
        runtime: GraphRuntime,
        agent: object,
        decision: RouteDecision,
        *,
        max_steps: int = 6,
    ) -> dict[str, Any]:
        captured["runtime"] = runtime
        return {"route": "graph", "answer": "ok"}

    def fake_ensure_graph_context(
        runtime: GraphRuntime | nx.DiGraph | None,
        agent: object | None,
        debug_llm_io: bool,
        graph_dataset: str | None = None,
        db_path: Path | None = None,
        payload_mode: str | None = None,
    ) -> tuple[GraphRuntime, object]:
        captured["graph_dataset"] = graph_dataset
        return wrap_networkx_graph(nx.DiGraph()), object()

    monkeypatch.setattr(
        "rag_tag.query_service.execute_graph_query", fake_execute_graph_query
    )
    monkeypatch.setattr(
        "rag_tag.query_service._ensure_graph_context",
        fake_ensure_graph_context,
    )

    decision = RouteDecision(route="graph", reason="test", sql_request=None)
    execute_query(
        "dummy question",
        [],
        None,
        None,
        decision=decision,
        graph_dataset="TestDataset",
    )

    assert captured["graph_dataset"] == "TestDataset"


def test_execute_query_accepts_legacy_graph_keyword_and_returns_graph_bundle(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    legacy_graph = nx.DiGraph()

    def fake_execute_graph_query(
        question: str,
        runtime: GraphRuntime,
        agent: object,
        decision: RouteDecision,
        *,
        max_steps: int = 20,
    ) -> dict[str, Any]:
        captured["runtime"] = runtime
        return {"route": "graph", "answer": "ok"}

    def fake_ensure_graph_context(
        runtime: GraphRuntime | nx.DiGraph | None,
        agent: object | None,
        debug_llm_io: bool,
        graph_dataset: str | None = None,
        db_path: Path | None = None,
        payload_mode: str | None = None,
    ) -> tuple[GraphRuntime, object]:
        resolved_runtime = (
            runtime
            if isinstance(runtime, GraphRuntime)
            else wrap_networkx_graph(runtime or nx.DiGraph())
        )
        captured["ensure_runtime"] = resolved_runtime
        return resolved_runtime, object()

    monkeypatch.setattr(
        "rag_tag.query_service.execute_graph_query", fake_execute_graph_query
    )
    monkeypatch.setattr(
        "rag_tag.query_service._ensure_graph_context",
        fake_ensure_graph_context,
    )

    decision = RouteDecision(route="graph", reason="test", sql_request=None)
    bundle = execute_query(
        "dummy question",
        [],
        None,
        None,
        decision=decision,
        graph_dataset="TestDataset",
        graph=legacy_graph,
    )

    assert captured["runtime"] is captured["ensure_runtime"]
    assert bundle["runtime"] is captured["ensure_runtime"]
    assert bundle["graph"] is bundle["runtime"]


def test_execute_query_rejects_conflicting_graph_and_runtime_inputs() -> None:
    decision = RouteDecision(route="graph", reason="test", sql_request=None)

    with pytest.raises(ValueError, match="Pass either runtime or graph, not both."):
        execute_query(
            "dummy question",
            [],
            wrap_networkx_graph(nx.DiGraph()),
            None,
            decision=decision,
            graph=nx.DiGraph(),
        )


def test_batch6_graph_payload_and_hierarchy_parity(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "fixture.jsonl"
    records = [
        {
            "GlobalId": "PROJ001",
            "IfcType": "IfcProject",
            "Name": "Test Project",
            "Hierarchy": {},
        },
        {
            "GlobalId": "BLDG001",
            "IfcType": "IfcBuilding",
            "Name": "Test Building",
            "Hierarchy": {"ParentId": "PROJ001"},
        },
        {
            "GlobalId": "STOR001",
            "IfcType": "IfcBuildingStorey",
            "Name": "Level 0",
            "Hierarchy": {"ParentId": "BLDG001"},
            "Geometry": {
                "Centroid": [0.0, 0.0, 1.5],
                "BoundingBox": {"min": [-10.0, -10.0, 0.0], "max": [10.0, 10.0, 3.0]},
            },
        },
        {
            "GlobalId": "WALL001",
            "IfcType": "IfcWall",
            "Name": "Wall 1",
            "Hierarchy": {"ParentId": "STOR001"},
            "Geometry": {
                "Centroid": [0.0, 0.0, 1.5],
                "BoundingBox": {"min": [-5.0, 0.0, 0.0], "max": [5.0, 0.2, 3.0]},
            },
            "PropertySets": {"Official": {"Pset_WallCommon": {"FireRating": "EI 60"}}},
        },
        {
            "GlobalId": "DOOR001",
            "IfcType": "IfcDoor",
            "Name": "Door 1",
            "Hierarchy": {"ParentId": "STOR001"},
            "Geometry": {
                "Centroid": [1.0, 0.0, 1.0],
                "BoundingBox": {"min": [0.5, -0.1, 0.0], "max": [1.5, 0.1, 2.1]},
            },
        },
    ]
    _write_jsonl(jsonl_path, records)

    graph = build_graph_from_jsonl([jsonl_path])

    assert "IfcProject" in graph
    assert "IfcBuilding" in graph
    assert "Storey::STOR001" in graph
    assert "Element::WALL001" in graph
    assert "Element::DOOR001" in graph
    assert graph.nodes["Element::WALL001"]["payload"]["GlobalId"] == "WALL001"
    assert graph.nodes["Element::WALL001"]["properties"]["Class"] == "IfcWall"
    assert graph.nodes["Element::WALL001"]["geometry"] == (0.0, 0.0, 1.5)

    storey_wall_edges = _iter_edge_attrs(graph, "Storey::STOR001", "Element::WALL001")
    wall_storey_edges = _iter_edge_attrs(graph, "Element::WALL001", "Storey::STOR001")
    building_storey_edges = _iter_edge_attrs(graph, "IfcBuilding", "Storey::STOR001")

    assert any(edge.get("relation") == "contains" for edge in storey_wall_edges)
    assert any(edge.get("relation") == "contained_in" for edge in wall_storey_edges)
    assert any(
        edge.get("relation") in {"aggregates", "contains"}
        for edge in building_storey_edges
    )
    assert graph.number_of_nodes() == 5


def test_payload_mode_parity_smoke(tmp_path: Path) -> None:
    wall_record = {
        "GlobalId": "WALL001",
        "ExpressId": 101,
        "IfcType": "IfcWall",
        "ClassRaw": "IfcWall",
        "Name": "Wall 1",
        "Description": "Exterior load-bearing wall",
        "ObjectType": None,
        "Tag": None,
        "TypeName": None,
        "PredefinedType": "STANDARD",
        "Materials": ["Concrete"],
        "Hierarchy": {"ParentId": "STOR001", "Level": "Level 1"},
        "Geometry": {
            "Centroid": [0.0, 0.0, 1.5],
            "BoundingBox": {"min": [-5.0, 0.0, 0.0], "max": [5.0, 0.2, 3.0]},
        },
        "PropertySets": {
            "Official": {
                "Pset_WallCommon": {"FireRating": "EI 90", "ThermalTransmittance": 0.28}
            },
            "Custom": {},
        },
        "Quantities": {"Qto_WallBaseQuantities": {"Length": 10.0}},
        "Relationships": {},
    }
    storey_record = {
        "GlobalId": "STOR001",
        "ExpressId": 50,
        "IfcType": "IfcBuildingStorey",
        "ClassRaw": "IfcBuildingStorey",
        "Name": "Level 1",
        "Hierarchy": {"ParentId": None, "Level": None},
        "Geometry": {
            "Centroid": [0.0, 0.0, 0.0],
            "BoundingBox": {"min": [-10.0, -10.0, 0.0], "max": [10.0, 10.0, 3.5]},
        },
        "PropertySets": {"Official": {}, "Custom": {}},
        "Quantities": {},
        "Relationships": {},
    }
    jsonl_path = tmp_path / "model_a.jsonl"
    db_path = tmp_path / "model_a.db"
    _write_jsonl(jsonl_path, [storey_record, wall_record])
    jsonl_to_sql(jsonl_path, db_path)

    graph_full = build_graph_from_jsonl([jsonl_path], payload_mode="full")
    graph_min = build_graph_from_jsonl([jsonl_path], payload_mode="minimal")
    graph_min.graph["_db_path"] = db_path

    assert graph_full.graph["_payload_mode"] == "full"
    assert graph_min.graph["_payload_mode"] == "minimal"
    assert set(graph_full.nodes) == set(graph_min.nodes)
    assert (
        graph_full.nodes["Element::WALL001"]["properties"]
        == graph_min.nodes["Element::WALL001"]["properties"]
    )
    assert "PropertySets" in graph_full.nodes["Element::WALL001"]["payload"]
    assert "PropertySets" not in graph_min.nodes["Element::WALL001"]["payload"]

    result = query_ifc_graph(
        graph_min,
        "find_nodes",
        {"property_filters": {"Pset_WallCommon.FireRating": "EI 90"}},
    )
    assert result["status"] == "ok"
    assert [item["id"] for item in result["data"]["elements"]] == ["Element::WALL001"]


def test_graph_contract_smoke_parity() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "IfcBuilding",
        label="Building A",
        class_="IfcBuilding",
        properties={"GlobalId": "BLDG1", "Name": "Building A"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )
    graph.add_node(
        "Storey::STOREY1",
        label="Level 0",
        class_="IfcBuildingStorey",
        properties={"GlobalId": "STOREY1", "Name": "Level 0"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )
    graph.add_node(
        "Element::WALL1",
        label="Wall 1",
        class_="IfcWall",
        properties={"GlobalId": "WALL1", "Name": "Wall 1"},
        payload={
            "PropertySets": {
                "Official": {"Pset_WallCommon": {"FireRating": "EI 90"}},
                "Custom": {},
            },
            "Quantities": {"Qto_WallBaseQuantities": {"Length": 5.0}},
        },
    )
    graph.add_node(
        "Element::DOOR1",
        label="Door 1",
        class_="IfcDoor",
        properties={"GlobalId": "DOOR1", "Name": "Door 1"},
        payload={
            "PropertySets": {
                "Official": {"Pset_DoorCommon": {"FireRating": "EI 60"}},
                "Custom": {},
            }
        },
    )
    graph.add_node(
        "Element::PIPE1",
        label="Pipe 1",
        class_="IfcPipeSegment",
        properties={"GlobalId": "PIPE1", "Name": "Pipe 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )
    graph.add_node(
        "Element::TERMINAL1",
        label="Terminal 1",
        class_="IfcFlowTerminal",
        properties={"GlobalId": "TERM1", "Name": "Terminal 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )
    graph.add_node(
        "System::SYS1",
        label="System 1",
        class_="IfcSystem",
        properties={"GlobalId": "SYS1", "Name": "System 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    graph.add_edge("IfcBuilding", "Storey::STOREY1", relation="contains")
    graph.add_edge("Storey::STOREY1", "IfcBuilding", relation="contained_in")
    graph.add_edge("Storey::STOREY1", "Element::WALL1", relation="contains")
    graph.add_edge("Element::WALL1", "Storey::STOREY1", relation="contained_in")
    graph.add_edge("Storey::STOREY1", "Element::DOOR1", relation="contains")
    graph.add_edge("Element::DOOR1", "Storey::STOREY1", relation="contained_in")
    graph.add_edge(
        "Element::WALL1",
        "Element::DOOR1",
        relation="adjacent_to",
        distance=0.8,
        source="heuristic",
    )
    graph.add_edge(
        "Element::DOOR1",
        "Element::WALL1",
        relation="adjacent_to",
        distance=0.8,
        source="heuristic",
    )
    graph.add_edge("Element::WALL1", "Element::PIPE1", relation="hosts", source="ifc")
    graph.add_edge(
        "Element::PIPE1", "Element::WALL1", relation="hosted_by", source="ifc"
    )
    graph.add_edge(
        "Element::PIPE1",
        "Element::TERMINAL1",
        relation="ifc_connected_to",
        source="ifc",
    )
    graph.add_edge(
        "Element::TERMINAL1", "System::SYS1", relation="belongs_to_system", source="ifc"
    )

    result = query_ifc_graph(
        graph,
        "traverse",
        {"start": "Element::PIPE1", "relation": "ifc_connected_to", "depth": 1},
    )
    assert has_valid_envelope_shape(result)
    assert result["status"] == "ok"
    assert not missing_required_action_fields("traverse", result["data"])
    assert result["data"]["evidence"] == [
        {
            "global_id": "TERM1",
            "id": "Element::TERMINAL1",
            "label": "Terminal 1",
            "class_": "IfcFlowTerminal",
            "source_tool": "traverse",
            "relation": "ifc_connected_to",
        }
    ]
    assert all(
        normalize_relation_name(item["relation"]) in CANONICAL_RELATION_SET
        for item in result["data"]["results"]
    )
    assert all(
        item["relation"] in EXPLICIT_IFC_RELATIONS for item in result["data"]["results"]
    )

    adj = query_ifc_graph(
        graph, "get_adjacent_elements", {"element_id": "Element::DOOR1"}
    )
    assert all(
        item["relation"] in SPATIAL_RELATIONS for item in adj["data"]["adjacent"]
    )
    assert adj["data"]["evidence"] == [
        {
            "global_id": "WALL1",
            "id": "Element::WALL1",
            "label": "Wall 1",
            "class_": "IfcWall",
            "source_tool": "get_adjacent_elements",
            "relation": "adjacent_to",
        }
    ]
    assert all(
        _source_matches_bucket_semantics(item) for item in adj["data"]["adjacent"]
    )

    non_hierarchy_sources_ok = True
    for _u, _v, edge in graph.edges(data=True):
        relation = normalize_relation_name(edge.get("relation"))
        source = normalize_relation_source(edge.get("source"))
        if relation in HIERARCHY_RELATIONS:
            continue
        if source not in KNOWN_RELATION_SOURCES:
            non_hierarchy_sources_ok = False
            break
    assert non_hierarchy_sources_ok


def test_graph_contract_scaffolds_batch1_roadmap_actions() -> None:
    for action in ROADMAP_ACTIONS:
        assert is_allowed_action(action) is True
        payload = ensure_action_data_fields(action, None)
        assert "evidence" in payload
        assert not missing_required_action_fields(action, payload)


def test_build_evidence_item_prefers_global_id_with_internal_id_fallback() -> None:
    evidence = build_evidence_item(
        {
            "id": "Element::WALL1",
            "label": "Wall 1",
            "class_": "IfcWall",
            "properties": {"GlobalId": "WALL1"},
        },
        source_tool="find_nodes",
        match_reason="exact_match",
    )

    assert evidence == {
        "global_id": "WALL1",
        "id": "Element::WALL1",
        "label": "Wall 1",
        "class_": "IfcWall",
        "source_tool": "find_nodes",
        "match_reason": "exact_match",
    }
