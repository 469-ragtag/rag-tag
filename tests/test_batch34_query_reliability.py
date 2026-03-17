from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel

from rag_tag.agent.graph_agent import GraphAgent
from rag_tag.config import GraphOrchestrationConfig
from rag_tag.graph import GraphRuntime, wrap_networkx_graph
from rag_tag.query_service import execute_query, execute_sql_query
from rag_tag.router.models import RouteDecision, SqlRequest


def _sql_decision() -> RouteDecision:
    return RouteDecision(
        route="sql",
        reason="sql route",
        sql_request=SqlRequest(
            intent="count", ifc_class="IfcDoor", level_like=None, limit=0
        ),
    )


def test_sql_merge_returns_warning_for_partial_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_tag import query_service

    ok_db = tmp_path / "ok.db"
    bad_db = tmp_path / "bad.db"
    ok_db.write_text("", encoding="utf-8")
    bad_db.write_text("", encoding="utf-8")

    def fake_query_ifc_sql(db_path: Path, request: SqlRequest) -> dict[str, object]:
        if db_path == bad_db:
            raise query_service.SqlQueryError("broken db")
        return {
            "status": "ok",
            "data": {
                "intent": request.intent,
                "filters": {},
                "count": 2,
                "summary": "Found 2 IfcDoor.",
                "sql": {"query": "SELECT 1", "params": []},
            },
            "error": None,
        }

    monkeypatch.setattr(query_service, "query_ifc_sql", fake_query_ifc_sql)

    result = execute_sql_query(_sql_decision(), [ok_db, bad_db])

    assert result["answer"] == "Found 2 IfcDoor."
    assert result["warning"]["failed_db_paths"] == [str(bad_db)]


def test_sql_merge_strict_mode_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_tag import query_service

    ok_db = tmp_path / "ok.db"
    bad_db = tmp_path / "bad.db"
    ok_db.write_text("", encoding="utf-8")
    bad_db.write_text("", encoding="utf-8")

    def fake_query_ifc_sql(db_path: Path, request: SqlRequest) -> dict[str, object]:
        if db_path == bad_db:
            raise query_service.SqlQueryError("broken db")
        return {
            "status": "ok",
            "data": {
                "intent": request.intent,
                "filters": {},
                "count": 2,
                "summary": "Found 2 IfcDoor.",
                "sql": {"query": "SELECT 1", "params": []},
            },
            "error": None,
        }

    monkeypatch.setattr(query_service, "query_ifc_sql", fake_query_ifc_sql)

    result = execute_sql_query(_sql_decision(), [ok_db, bad_db], strict_sql=True)

    assert "Strict SQL mode aborted" in result["error"]
    assert result["warning"]["failed_db_paths"] == [str(bad_db)]


def test_execute_query_passes_strict_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_tag import query_service

    captured: dict[str, object] = {}

    def fake_execute_sql_query(
        decision: RouteDecision,
        db_paths: list[Path],
        *,
        strict_sql: bool = False,
    ) -> dict[str, object]:
        captured["strict_sql"] = strict_sql
        return {"route": "sql", "answer": "ok", "data": {}}

    monkeypatch.setattr(query_service, "execute_sql_query", fake_execute_sql_query)

    bundle = execute_query(
        "Count doors",
        db_paths=[],
        runtime=None,
        agent=None,
        decision=_sql_decision(),
        strict_sql=True,
    )

    assert bundle["result"]["answer"] == "ok"
    assert captured["strict_sql"] is True


def test_graph_agent_honors_usage_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_tag.agent.graph_agent.get_agent_model", TestModel)
    agent = GraphAgent()

    def fake_run_sync(
        question: str, *, deps: GraphRuntime, usage_limits: object
    ) -> object:
        raise UsageLimitExceeded("tool_calls_limit exceeded")

    monkeypatch.setattr(agent._agent, "run_sync", fake_run_sync)

    result = agent.run(
        "question",
        wrap_networkx_graph(nx.MultiDiGraph()),
        max_steps=1,
    )

    answer = result.get("answer")
    warning = result.get("warning")
    data = result.get("data")

    assert isinstance(answer, str)
    assert "step budget" in answer.lower()
    assert isinstance(warning, str)
    assert "max_steps=1" in warning
    assert isinstance(data, dict)
    assert data.get("max_steps") == 1


def test_execute_query_preserves_bundle_shape_with_langgraph_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_tag import query_service

    class FakeLangGraphAgent:
        def __init__(
            self,
            *,
            debug_llm_io: bool = False,
            orchestration_config: GraphOrchestrationConfig | None = None,
        ) -> None:
            self.debug_llm_io = debug_llm_io
            self.orchestration_config = orchestration_config

        def run(
            self,
            question: str,
            runtime: GraphRuntime,
            *,
            max_steps: int = 20,
            trace: object | None = None,
            run_id: str | None = None,
        ) -> dict[str, object]:
            del runtime, trace, run_id
            return {
                "answer": f"langgraph:{question}",
                "data": {"max_steps": max_steps},
            }

    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["model-a"]
    runtime = wrap_networkx_graph(graph)
    decision = RouteDecision(route="graph", reason="graph route", sql_request=None)

    monkeypatch.setattr(query_service, "LangGraphAgent", FakeLangGraphAgent)
    monkeypatch.setattr(
        query_service,
        "resolve_graph_orchestrator",
        lambda: "langgraph",
    )
    monkeypatch.setattr(
        query_service,
        "get_graph_orchestration_config",
        lambda: GraphOrchestrationConfig(),
    )

    bundle = execute_query(
        "Which rooms are adjacent to the kitchen?",
        db_paths=[],
        runtime=runtime,
        agent=None,
        decision=decision,
        graph_dataset="model-a",
        graph_max_steps=7,
    )

    assert bundle["runtime"] is runtime
    assert bundle["graph"] is runtime
    assert isinstance(bundle["agent"], FakeLangGraphAgent)
    assert bundle["result"] == {
        "route": "graph",
        "decision": "graph route",
        "answer": "langgraph:Which rooms are adjacent to the kitchen?",
        "data": {"max_steps": 7},
    }


def test_execute_query_returns_error_when_langgraph_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_tag import query_service

    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["model-a"]
    runtime = wrap_networkx_graph(graph)
    decision = RouteDecision(route="graph", reason="graph route", sql_request=None)

    monkeypatch.setattr(
        query_service,
        "resolve_graph_orchestrator",
        lambda: "langgraph",
    )
    monkeypatch.setattr(
        query_service,
        "LangGraphAgent",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("langgraph missing")),
    )

    bundle = execute_query(
        "Which rooms are adjacent to the kitchen?",
        db_paths=[],
        runtime=runtime,
        agent=None,
        decision=decision,
        graph_dataset="model-a",
    )

    assert bundle["runtime"] is runtime
    assert bundle["graph"] is runtime
    assert bundle["agent"] is None
    assert bundle["result"] == {
        "route": "graph",
        "decision": "graph route",
        "error": "langgraph missing",
    }


def test_execute_query_reuses_existing_agent_without_selector_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_tag import query_service

    graph = nx.MultiDiGraph()
    graph.graph["datasets"] = ["model-a"]
    runtime = wrap_networkx_graph(graph)
    decision = RouteDecision(route="graph", reason="graph route", sql_request=None)

    existing_agent = GraphAgent.__new__(GraphAgent)
    selector_calls = {"count": 0}

    def fail_if_called() -> str:
        selector_calls["count"] += 1
        return "langgraph"

    def fake_run(
        question: str,
        runtime: GraphRuntime,
        *,
        max_steps: int = 20,
        trace: object | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        del runtime, trace, run_id
        return {"answer": f"reused:{question}", "data": {"max_steps": max_steps}}

    monkeypatch.setattr(query_service, "resolve_graph_orchestrator", fail_if_called)
    monkeypatch.setattr(existing_agent, "run", fake_run)

    bundle = execute_query(
        "Which rooms are adjacent to the kitchen?",
        db_paths=[],
        runtime=runtime,
        agent=existing_agent,
        decision=decision,
        graph_dataset="model-a",
        graph_max_steps=9,
    )

    assert selector_calls["count"] == 0
    assert bundle["runtime"] is runtime
    assert bundle["graph"] is runtime
    assert bundle["agent"] is existing_agent
    assert bundle["result"] == {
        "route": "graph",
        "decision": "graph route",
        "answer": "reused:Which rooms are adjacent to the kitchen?",
        "data": {"max_steps": 9},
    }
