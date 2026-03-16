from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from rag_tag.agent.graph_tools import register_graph_tools
from rag_tag.graph import runtime as graph_runtime_module
from rag_tag.graph import wrap_networkx_graph
from rag_tag.query_service import (
    _ensure_graph_context,
    _require_explicit_graph_dataset,
    execute_query,
)
from rag_tag.router.models import RouteDecision


def _build_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node(
        "IfcBuilding",
        label="Building A",
        class_="IfcBuilding",
        properties={"GlobalId": "BLDG1", "Name": "Building A"},
        payload={},
    )
    G.add_node(
        "Storey::S1",
        label="Level 0",
        class_="IfcBuildingStorey",
        properties={"GlobalId": "S1", "Name": "Level 0"},
        payload={},
    )
    G.add_node(
        "Element::W1",
        label="Wall 1",
        class_="IfcWall",
        properties={"GlobalId": "W1", "Name": "Wall 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )
    G.add_node(
        "Element::D1",
        label="Door 1",
        class_="IfcDoor",
        properties={"GlobalId": "D1", "Name": "Door 1"},
        payload={"PropertySets": {"Official": {}, "Custom": {}}},
    )

    G.add_edge("IfcBuilding", "Storey::S1", relation="contains")
    G.add_edge("Storey::S1", "IfcBuilding", relation="contained_in")
    G.add_edge("Storey::S1", "Element::W1", relation="contains")
    G.add_edge("Element::W1", "Storey::S1", relation="contained_in")
    G.add_edge("Storey::S1", "Element::D1", relation="contains")
    G.add_edge("Element::D1", "Storey::S1", relation="contained_in")
    G.add_edge(
        "Element::W1",
        "Element::D1",
        relation="adjacent_to",
        distance=0.5,
        source="heuristic",
    )
    G.add_edge(
        "Element::D1",
        "Element::W1",
        relation="adjacent_to",
        distance=0.5,
        source="heuristic",
    )
    return G


def test_graph_runtime_networkx_smoke() -> None:
    G = _build_graph()
    runtime = wrap_networkx_graph(G)

    res_storey = runtime.query("get_elements_in_storey", {"storey": "Level 0"})
    assert res_storey["status"] == "ok"
    assert len(res_storey["data"]["elements"]) == 2

    res_class = runtime.query("find_elements_by_class", {"class": "IfcWall"})
    assert res_class["status"] == "ok"
    assert len(res_class["data"]["elements"]) == 1

    res_adj = runtime.query("get_adjacent_elements", {"element_id": "Element::W1"})
    assert res_adj["status"] == "ok"
    assert len(res_adj["data"]["adjacent"]) >= 1

    res_traverse = runtime.query(
        "traverse", {"start": "Storey::S1", "relation": "contains", "depth": 1}
    )
    assert res_traverse["status"] == "ok"
    assert len(res_traverse["data"]["results"]) == 2


def test_ensure_graph_context_reuses_existing_runtime() -> None:
    runtime = wrap_networkx_graph(_build_graph())
    sentinel_agent = object()

    resolved_runtime, resolved_agent = _ensure_graph_context(
        runtime,
        sentinel_agent,
        False,
    )

    assert resolved_runtime is runtime
    assert resolved_agent is sentinel_agent


def test_ensure_graph_context_honors_graph_backend_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_graph()
    sentinel_agent = object()
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  graph_backend: networkx\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        graph_runtime_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "graph",
    )
    monkeypatch.setenv("GRAPH_BACKEND", "neo4j")

    runtime, resolved_agent = _ensure_graph_context(
        graph,
        sentinel_agent,
        False,
    )

    assert runtime.backend_name == "neo4j"
    assert runtime.get_networkx_graph() is graph
    assert resolved_agent is sentinel_agent


def test_ensure_graph_context_uses_graph_backend_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_graph()
    sentinel_agent = object()
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  graph_backend: neo4j\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GRAPH_BACKEND", raising=False)
    monkeypatch.setattr(
        graph_runtime_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "graph",
    )

    runtime, resolved_agent = _ensure_graph_context(
        graph,
        sentinel_agent,
        False,
    )

    assert runtime.backend_name == "neo4j"
    assert runtime.get_networkx_graph() is graph
    assert resolved_agent is sentinel_agent


def test_require_explicit_graph_dataset_uses_runtime_public_graph() -> None:
    runtime = wrap_networkx_graph(_build_graph())
    runtime.get_networkx_graph().graph["datasets"] = ["A", "B"]

    with pytest.raises(ValueError, match="Multiple graph datasets are available"):
        _require_explicit_graph_dataset(runtime, None)


def test_execute_query_reuses_existing_runtime_without_private_graph_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = wrap_networkx_graph(_build_graph())
    runtime.get_networkx_graph().graph["datasets"] = ["OnlyDataset"]
    sentinel_agent = object()
    captured: dict[str, object] = {}

    def fake_execute_graph_query(
        question: str,
        runtime_arg,
        agent_arg,
        decision: RouteDecision,
        *,
        max_steps: int = 20,
    ) -> dict[str, object]:
        captured["runtime"] = runtime_arg
        captured["agent"] = agent_arg
        return {"route": "graph", "answer": "ok"}

    monkeypatch.setattr(
        "rag_tag.query_service.execute_graph_query",
        fake_execute_graph_query,
    )

    bundle = execute_query(
        "dummy question",
        [],
        runtime,
        sentinel_agent,
        decision=RouteDecision(route="graph", reason="test", sql_request=None),
    )

    assert captured["runtime"] is runtime
    assert captured["agent"] is sentinel_agent
    assert bundle["runtime"] is runtime
    assert bundle["graph"] is runtime


def test_execute_query_requires_explicit_dataset_before_first_graph_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "model-a.jsonl").write_text("", encoding="utf-8")
    (output_dir / "model-b.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr("rag_tag.query_service.find_project_root", lambda *_: tmp_path)

    def fail_load_graph(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("load_graph should not run for ambiguous first loads")

    monkeypatch.setattr("rag_tag.query_service.load_graph", fail_load_graph)

    bundle = execute_query(
        "Which rooms are adjacent to the kitchen?",
        db_paths=[],
        decision=RouteDecision(route="graph", reason="test", sql_request=None),
    )

    assert "Multiple graph datasets are available" in bundle["result"]["error"]


def test_get_elements_in_storey_tool_uses_runtime_query() -> None:
    class ToolRegistryAgent:
        def __init__(self) -> None:
            self.tools: dict[str, object] = {}

        def tool(self, func):  # type: ignore[no-untyped-def]
            self.tools[func.__name__] = func
            return func

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def query(self, action: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((action, params))
            return {"status": "ok", "data": {"elements": []}, "error": None}

    fake_agent = ToolRegistryAgent()
    register_graph_tools(fake_agent)
    runtime = FakeRuntime()

    result = fake_agent.tools["get_elements_in_storey"](
        SimpleNamespace(deps=runtime),
        "Level 0",
    )

    assert runtime.calls == [("get_elements_in_storey", {"storey": "Level 0"})]
    assert result["status"] == "ok"


def _write_project_marker(project_root: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'test'\n", encoding="utf-8"
    )
