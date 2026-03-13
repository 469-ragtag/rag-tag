from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel

from rag_tag.agent.graph_agent import GraphAgent
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
