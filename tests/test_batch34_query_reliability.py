from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel

from rag_tag.agent.graph_agent import GraphAgent
from rag_tag.graph import GraphRuntime, wrap_networkx_graph
from rag_tag.query_service import execute_query, execute_sql_query
from rag_tag.router.models import RouteDecision, SqlFieldRef, SqlRequest


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
                "evidence": [
                    {
                        "global_id": "door-1",
                        "id": 101,
                        "label": "Door 1",
                        "class_": "IfcDoor",
                        "source_tool": "query_ifc_sql",
                    }
                ],
                "summary": "Found 2 IfcDoor.",
                "sql": {"query": "SELECT 1", "params": []},
            },
            "error": None,
        }

    monkeypatch.setattr(query_service, "query_ifc_sql", fake_query_ifc_sql)

    result = execute_sql_query(_sql_decision(), [ok_db, bad_db])

    assert result["answer"] == "Found 2 IfcDoor."
    assert result["data"]["evidence"] == [
        {
            "global_id": "door-1",
            "id": 101,
            "label": "Door 1",
            "class_": "IfcDoor",
            "source_tool": "query_ifc_sql",
        }
    ]
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
                "evidence": [
                    {
                        "global_id": "door-1",
                        "id": 101,
                        "label": "Door 1",
                        "class_": "IfcDoor",
                        "source_tool": "query_ifc_sql",
                    }
                ],
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


def test_sql_merge_weighted_average_for_aggregate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_tag import query_service

    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    db_a.write_text("", encoding="utf-8")
    db_b.write_text("", encoding="utf-8")

    def fake_query_ifc_sql(db_path: Path, request: SqlRequest) -> dict[str, object]:
        if db_path == db_a:
            return {
                "status": "ok",
                "data": {
                    "intent": request.intent,
                    "filters": {},
                    "aggregate_op": "avg",
                    "aggregate_field": {"source": "property", "field": "UValue"},
                    "aggregate_value": 1.5,
                    "matched_value_count": 2,
                    "missing_value_count": 0,
                    "total_elements": 2,
                    "evidence": [],
                    "summary": "Computed avg of UValue for IfcWindow: 1.5.",
                    "merge_state": {
                        "sum": 3.0,
                        "matched_value_count": 2,
                        "missing_value_count": 0,
                        "total_elements": 2,
                    },
                    "sql": {"query": "SELECT 1", "params": []},
                },
                "error": None,
            }
        return {
            "status": "ok",
            "data": {
                "intent": request.intent,
                "filters": {},
                "aggregate_op": "avg",
                "aggregate_field": {"source": "property", "field": "UValue"},
                "aggregate_value": 0.9,
                "matched_value_count": 1,
                "missing_value_count": 0,
                "total_elements": 1,
                "evidence": [],
                "summary": "Computed avg of UValue for IfcWindow: 0.9.",
                "merge_state": {
                    "sum": 0.9,
                    "matched_value_count": 1,
                    "missing_value_count": 0,
                    "total_elements": 1,
                },
                "sql": {"query": "SELECT 1", "params": []},
            },
            "error": None,
        }

    monkeypatch.setattr(query_service, "query_ifc_sql", fake_query_ifc_sql)

    decision = RouteDecision(
        route="sql",
        reason="sql aggregate route",
        sql_request=SqlRequest(
            intent="aggregate",
            ifc_class="IfcWindow",
            level_like=None,
            aggregate_op="avg",
            aggregate_field=SqlFieldRef(source="property", field="UValue"),
            limit=0,
        ),
    )

    result = execute_sql_query(decision, [db_a, db_b])

    assert result["answer"] == "Computed avg of UValue for IfcWindow: 1.3."
    assert result["data"]["aggregate_value"] == 1.3
    assert result["data"]["matched_value_count"] == 3
    assert result["data"]["missing_value_count"] == 0


def test_sql_merge_combines_group_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_tag import query_service

    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    db_a.write_text("", encoding="utf-8")
    db_b.write_text("", encoding="utf-8")

    def fake_query_ifc_sql(db_path: Path, request: SqlRequest) -> dict[str, object]:
        if db_path == db_a:
            groups = [{"group": "EI30", "count": 2}, {"group": "EI60", "count": 1}]
        else:
            groups = [{"group": "EI30", "count": 1}, {"group": "EI90", "count": 1}]
        return {
            "status": "ok",
            "data": {
                "intent": request.intent,
                "filters": {},
                "group_by": {"source": "property", "field": "FireRating"},
                "groups": groups,
                "matched_element_count": sum(group["count"] for group in groups),
                "missing_value_count": 0,
                "total_elements": sum(group["count"] for group in groups),
                "evidence": [],
                "summary": "Grouped IfcDoor by FireRating, showing 2 groups.",
                "sql": {"query": "SELECT 1", "params": []},
            },
            "error": None,
        }

    monkeypatch.setattr(query_service, "query_ifc_sql", fake_query_ifc_sql)

    decision = RouteDecision(
        route="sql",
        reason="sql group route",
        sql_request=SqlRequest(
            intent="group",
            ifc_class="IfcDoor",
            level_like=None,
            group_by=SqlFieldRef(source="property", field="FireRating"),
            limit=10,
        ),
    )

    result = execute_sql_query(decision, [db_a, db_b])

    assert result["data"]["groups"] == [
        {"group": "EI30", "count": 3},
        {"group": "EI60", "count": 1},
        {"group": "EI90", "count": 1},
    ]
    assert result["data"]["matched_element_count"] == 5
