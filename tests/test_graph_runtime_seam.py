from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
from pydantic_ai.models.test import TestModel

from rag_tag.agent import LangGraphAgent
from rag_tag.agent.graph_agent import GraphAgent
from rag_tag.agent.graph_tools import _fuzzy_find_nodes_impl
from rag_tag.agent.models import GraphAnswer
from rag_tag.config import GraphOrchestrationConfig
from rag_tag.graph import GraphRuntime, wrap_networkx_graph
from rag_tag.ifc_graph_tool import query_ifc_graph
from rag_tag.query_service import _ensure_graph_context


def _runtime_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["model-a"]
    graph.graph["_payload_mode"] = "minimal"
    graph.add_node(
        "Element::wall-occ",
        label="plumbing wall",
        class_="IfcWall",
        properties={"Name": "plumbing wall", "GlobalId": "wall-occ"},
        payload={"Name": "plumbing wall", "IfcType": "IfcWall", "ClassRaw": "IfcWall"},
    )
    graph.add_node(
        "Element::wall-type",
        label="plumbing wall",
        class_="IfcWallType",
        properties={"Name": "plumbing wall", "GlobalId": "wall-type"},
        payload={
            "Name": "plumbing wall",
            "IfcType": "IfcWallType",
            "ClassRaw": "IfcWallType",
        },
    )
    return graph


def test_query_ifc_graph_keeps_raw_networkx_compatibility() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("Element::A", label="A", class_="IfcWall", properties={})
    graph.add_node("Element::B", label="B", class_="IfcDoor", properties={})
    graph.add_edge("Element::A", "Element::B", relation="hosts", source="ifc")
    graph.add_edge("Element::A", "Element::B", relation="typed_by", source="ifc")

    result = query_ifc_graph(graph, "traverse", {"start": "Element::A", "depth": 1})

    assert result["status"] == "ok"
    assert [item["relation"] for item in result["data"]["results"]] == [
        "hosts",
        "typed_by",
    ]


def test_networkx_backend_query_preserves_parallel_edge_order() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("Element::A", label="A", class_="IfcWall", properties={})
    graph.add_node("Element::B", label="B", class_="IfcDoor", properties={})
    graph.add_edge("Element::A", "Element::B", relation="hosts", source="ifc")
    graph.add_edge("Element::A", "Element::B", relation="typed_by", source="ifc")
    runtime = wrap_networkx_graph(graph)

    result = runtime.backend.query(
        runtime, "traverse", {"start": "Element::A", "depth": 1}, "llm"
    )

    assert result["status"] == "ok"
    assert [item["relation"] for item in result["data"]["results"]] == [
        "hosts",
        "typed_by",
    ]


def test_ensure_graph_context_uses_runtime_state_not_graph_attrs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    graph = nx.MultiDiGraph()
    runtime = wrap_networkx_graph(graph)
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    db_a.write_text("", encoding="utf-8")
    db_b.write_text("", encoding="utf-8")

    runtime, agent = _ensure_graph_context(
        runtime,
        agent=None,
        debug_llm_io=False,
        graph_dataset="model-a",
        db_path=db_a,
        payload_mode="minimal",
    )

    assert agent is not None
    assert runtime.context_db_path == db_a.resolve()
    assert runtime.payload_mode == "minimal"
    assert runtime.selected_datasets == ["model-a"]
    assert "_db_path" not in graph.graph

    runtime.caches["property_cache"] = {("db", "node"): {"payload": {}}}
    runtime.caches["property_key_cache"] = {("db", ""): {"Name": ["Wall"]}}

    runtime, _ = _ensure_graph_context(
        runtime,
        agent=agent,
        debug_llm_io=False,
        graph_dataset="model-a",
        db_path=db_b,
    )

    assert runtime.context_db_path == db_b.resolve()
    assert "property_cache" not in runtime.caches
    assert "property_key_cache" not in runtime.caches
    assert "_db_path" not in graph.graph


def test_graph_agent_deps_and_tool_helpers_use_graph_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    agent = GraphAgent()
    runtime = wrap_networkx_graph(_runtime_graph())

    captured: dict[str, object] = {}

    def fake_run_sync(
        question: str,
        *,
        deps: GraphRuntime,
        usage_limits: object,
    ) -> object:
        captured["deps"] = deps
        return SimpleNamespace(
            output=GraphAnswer(answer="Recovered.", data={"ok": True}, warning=None)
        )

    monkeypatch.setattr(agent._agent, "run_sync", fake_run_sync)

    result = agent.run("question", runtime)

    assert captured["deps"] is runtime
    assert result["answer"] == "Recovered."

    fuzzy = _fuzzy_find_nodes_impl(runtime, "plumbing wall")
    assert fuzzy["status"] == "ok"
    assert fuzzy["data"]["matches"][0]["id"] == "Element::wall-occ"


def test_ensure_graph_context_can_create_langgraph_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    monkeypatch.setattr(
        "rag_tag.agent.langgraph_agent._ensure_langgraph_dependency",
        lambda: None,
    )
    monkeypatch.setattr(
        "rag_tag.query_service.resolve_graph_orchestrator",
        lambda: "langgraph",
    )
    monkeypatch.setattr(
        "rag_tag.query_service.get_graph_orchestration_config",
        lambda: GraphOrchestrationConfig(),
    )

    runtime, agent = _ensure_graph_context(
        wrap_networkx_graph(nx.MultiDiGraph()),
        agent=None,
        debug_llm_io=False,
    )

    assert isinstance(runtime, GraphRuntime)
    assert isinstance(agent, LangGraphAgent)
