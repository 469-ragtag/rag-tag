from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_tag.config import (
    AGENT_PROFILE_ENV_VAR,
    CONFIG_PATH_ENV_VAR,
    ROUTER_PROFILE_ENV_VAR,
)
from rag_tag.evals import BenchmarkAnswer, BenchmarkCase
from rag_tag.evals.runtime import (
    BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR,
    temporary_runtime_overrides,
)
from rag_tag.evals.task_runner import BenchmarkUsage, run_benchmark_case


def test_temporary_runtime_overrides_restore_previous_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CONFIG_PATH_ENV_VAR, "/tmp/original-config.yaml")
    monkeypatch.setenv(ROUTER_PROFILE_ENV_VAR, "original-router")
    monkeypatch.setenv(AGENT_PROFILE_ENV_VAR, "original-agent")
    monkeypatch.setenv("ROUTER_MODEL", "google-gla:gemini-2.5-flash")
    monkeypatch.setenv("AGENT_MODEL", "cohere:command-a-03-2025")

    with temporary_runtime_overrides(
        config_path="/tmp/override-config.yaml",
        router_profile="override-router",
        agent_profile="override-agent",
    ):
        assert os.environ[CONFIG_PATH_ENV_VAR] == "/tmp/override-config.yaml"
        assert os.environ[ROUTER_PROFILE_ENV_VAR] == "override-router"
        assert os.environ[AGENT_PROFILE_ENV_VAR] == "override-agent"
        assert "ROUTER_MODEL" not in os.environ
        assert "AGENT_MODEL" not in os.environ

    assert os.environ[CONFIG_PATH_ENV_VAR] == "/tmp/original-config.yaml"
    assert os.environ[ROUTER_PROFILE_ENV_VAR] == "original-router"
    assert os.environ[AGENT_PROFILE_ENV_VAR] == "original-agent"
    assert os.environ["ROUTER_MODEL"] == "google-gla:gemini-2.5-flash"
    assert os.environ["AGENT_MODEL"] == "cohere:command-a-03-2025"


def test_run_benchmark_case_returns_structured_sql_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = BenchmarkCase(
        id="q001",
        question="How many walls are in the model?",
        expected_route="sql",
    )

    calls: list[dict[str, object]] = []

    def fake_execute_query(
        question: str,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        *,
        debug_llm_io: bool,
        graph_dataset: str | None,
        context_db: Path | None,
        payload_mode: str | None,
        strict_sql: bool,
        graph_max_steps: int | None,
    ) -> dict[str, object]:
        calls.append(
            {
                "question": question,
                "db_paths": db_paths,
                "runtime": runtime,
                "agent": agent,
                "router_profile": os.environ.get(ROUTER_PROFILE_ENV_VAR),
                "agent_profile": os.environ.get(AGENT_PROFILE_ENV_VAR),
                "config_path": os.environ.get(CONFIG_PATH_ENV_VAR),
                "graph_orchestrator": os.environ.get(
                    BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR
                ),
                "graph_dataset": graph_dataset,
                "context_db": context_db,
                "payload_mode": payload_mode,
                "strict_sql": strict_sql,
                "graph_max_steps": graph_max_steps,
                "debug_llm_io": debug_llm_io,
            }
        )
        return {
            "result": {
                "route": "sql",
                "decision": "count query",
                "answer": "Found 12 IfcWall.",
                "warning": None,
                "error": None,
                "data": {"count": 12},
                "sql": "SELECT COUNT(*) FROM elements WHERE ifc_class = 'IfcWall'",
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 5,
                },
            },
            "runtime": runtime,
            "agent": agent,
        }

    monkeypatch.setattr("rag_tag.evals.task_runner.execute_query", fake_execute_query)

    bundle = run_benchmark_case(
        case,
        db_paths=[Path("/tmp/model.db")],
        runtime=None,
        agent=None,
        config_path="/tmp/config.yaml",
        router_profile="router-alpha",
        agent_profile="agent-beta",
        graph_orchestrator="langgraph",
        graph_dataset="model",
        context_db=Path("/tmp/model.db"),
        payload_mode="minimal",
        strict_sql=True,
        graph_max_steps=7,
    )

    assert calls == [
        {
            "question": "How many walls are in the model?",
            "db_paths": [Path("/tmp/model.db")],
            "runtime": None,
            "agent": None,
            "router_profile": "router-alpha",
            "agent_profile": "agent-beta",
            "config_path": "/tmp/config.yaml",
            "graph_orchestrator": "langgraph",
            "graph_dataset": "model",
            "context_db": Path("/tmp/model.db"),
            "payload_mode": "minimal",
            "strict_sql": True,
            "graph_max_steps": 7,
            "debug_llm_io": False,
        }
    ]
    assert bundle.result.case_id == "q001"
    assert bundle.result.expected_route == "sql"
    assert bundle.result.selected_route == "sql"
    assert bundle.result.answer == "Found 12 IfcWall."
    assert bundle.result.graph_orchestrator == "langgraph"
    assert bundle.result.sql is not None
    assert bundle.result.had_error is False
    assert bundle.result.had_warning is False
    assert bundle.result.has_data is True
    assert bundle.result.usage == BenchmarkUsage(
        input_tokens=25,
        output_tokens=5,
        total_tokens=30,
        reasoning_tokens=None,
        usage_available=True,
    )


def test_run_benchmark_case_records_compare_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = BenchmarkCase(
        id="q001",
        question="How many walls are in the model?",
        expected_route="sql",
        answer=BenchmarkAnswer(
            canonical=" There are\n exactly 4 walls. ",
        ),
    )

    recorded_attributes: list[tuple[str, object]] = []

    def fake_set_eval_attribute(name: str, value: object) -> None:
        recorded_attributes.append((name, value))

    def fake_execute_query(
        question: str,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        *,
        debug_llm_io: bool,
        graph_dataset: str | None,
        context_db: Path | None,
        payload_mode: str | None,
        strict_sql: bool,
        graph_max_steps: int | None,
    ) -> dict[str, object]:
        del (
            question,
            db_paths,
            runtime,
            agent,
            debug_llm_io,
            graph_dataset,
            context_db,
            payload_mode,
            strict_sql,
            graph_max_steps,
        )
        return {
            "result": {
                "route": "sql",
                "decision": "count query",
                "answer": " Found\n 12 IfcWall. ",
                "warning": None,
                "error": None,
                "data": {"count": 12},
            },
            "runtime": None,
            "agent": None,
        }

    monkeypatch.setattr(
        "rag_tag.evals.task_runner.set_eval_attribute",
        fake_set_eval_attribute,
    )
    monkeypatch.setattr("rag_tag.evals.task_runner.execute_query", fake_execute_query)

    bundle = run_benchmark_case(
        case,
        db_paths=[Path("/tmp/model.db")],
        runtime=None,
        agent=None,
    )

    assert bundle.result.answer == "Found\n 12 IfcWall."
    assert recorded_attributes == [
        ("compare_question", "How many walls are in the model?"),
        ("compare_expected_answer", "There are exactly 4 walls."),
        ("compare_agent_answer", "Found 12 IfcWall."),
    ]


def test_run_benchmark_case_preserves_runtime_and_agent_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = BenchmarkCase(
        id="q002",
        question="What building storey contains the chimney?",
        expected_route="graph",
    )
    runtime = object()
    agent = object()
    next_runtime = object()
    next_agent = object()

    def fake_execute_query(
        question: str,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        *,
        debug_llm_io: bool,
        graph_dataset: str | None,
        context_db: Path | None,
        payload_mode: str | None,
        strict_sql: bool,
        graph_max_steps: int | None,
    ) -> dict[str, object]:
        del question, db_paths, debug_llm_io, graph_dataset, context_db, payload_mode
        del strict_sql, graph_max_steps
        assert runtime is not None
        assert agent is not None
        return {
            "result": {
                "route": "graph",
                "decision": "containment query",
                "answer": "The chimney is contained in Level 1.",
                "warning": "partial evidence",
                "error": None,
                "data": {"storey": "Level 1"},
            },
            "runtime": next_runtime,
            "agent": next_agent,
        }

    monkeypatch.setattr("rag_tag.evals.task_runner.execute_query", fake_execute_query)

    bundle = run_benchmark_case(
        case,
        db_paths=[Path("/tmp/model.db")],
        runtime=runtime,
        agent=agent,
        router_profile="router-alpha",
        agent_profile="agent-beta",
        prompt_strategy="baseline",
        graph_orchestrator="pydanticai",
    )

    assert bundle.runtime is next_runtime
    assert bundle.agent is next_agent
    assert bundle.result.selected_route == "graph"
    assert bundle.result.graph_orchestrator == "pydanticai"
    assert bundle.result.had_warning is True
    assert bundle.result.warning == "partial evidence"


def test_run_benchmark_case_handles_missing_usage_and_unexpected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = BenchmarkCase(
        id="q003",
        question="What material is used for the sand bedding?",
        expected_route="graph",
    )

    def fake_execute_query(
        question: str,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        *,
        debug_llm_io: bool,
        graph_dataset: str | None,
        context_db: Path | None,
        payload_mode: str | None,
        strict_sql: bool,
        graph_max_steps: int | None,
    ) -> dict[str, object]:
        del (
            question,
            db_paths,
            runtime,
            agent,
            debug_llm_io,
            graph_dataset,
            context_db,
            payload_mode,
            strict_sql,
            graph_max_steps,
        )
        return {"result": "unexpected"}

    monkeypatch.setattr("rag_tag.evals.task_runner.execute_query", fake_execute_query)

    bundle = run_benchmark_case(
        case,
        db_paths=[Path("/tmp/model.db")],
        runtime=None,
        agent=None,
    )

    assert bundle.result.selected_route == "?"
    assert bundle.result.had_error is True
    assert bundle.result.error is not None
    assert bundle.result.usage == BenchmarkUsage()


def test_run_benchmark_case_normalizes_blank_warning_and_invalid_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = BenchmarkCase(
        id="q004",
        question="Is the answer grounded?",
        expected_route="graph",
    )

    def fake_execute_query(
        question: str,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        *,
        debug_llm_io: bool,
        graph_dataset: str | None,
        context_db: Path | None,
        payload_mode: str | None,
        strict_sql: bool,
        graph_max_steps: int | None,
    ) -> dict[str, object]:
        del (
            question,
            db_paths,
            runtime,
            agent,
            debug_llm_io,
            graph_dataset,
            context_db,
            payload_mode,
            strict_sql,
            graph_max_steps,
        )
        return {
            "result": {
                "route": "graph",
                "decision": "grounded lookup",
                "answer": "Partial evidence only.",
                "warning": "   ",
                "error": None,
                "data": "not-structured",
            }
        }

    monkeypatch.setattr("rag_tag.evals.task_runner.execute_query", fake_execute_query)

    bundle = run_benchmark_case(
        case,
        db_paths=[Path("/tmp/model.db")],
        runtime=None,
        agent=None,
    )

    assert bundle.result.warning is None
    assert bundle.result.had_warning is False
    assert bundle.result.data is None
    assert bundle.result.has_data is False


def test_run_benchmark_case_records_resolved_profile_names_from_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = BenchmarkCase(
        id="q005",
        question="How many doors are there?",
        expected_route="sql",
    )

    class LoadedConfig:
        def __init__(self) -> None:
            self.config = type(
                "Config",
                (),
                {
                    "defaults": type(
                        "Defaults",
                        (),
                        {
                            "router_profile": "router-from-config",
                            "agent_profile": "agent-from-config",
                        },
                    )(),
                    "profiles": {},
                },
            )()

    def fake_execute_query(
        question: str,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        *,
        debug_llm_io: bool,
        graph_dataset: str | None,
        context_db: Path | None,
        payload_mode: str | None,
        strict_sql: bool,
        graph_max_steps: int | None,
    ) -> dict[str, object]:
        del (
            question,
            db_paths,
            runtime,
            agent,
            debug_llm_io,
            graph_dataset,
            context_db,
            payload_mode,
            strict_sql,
            graph_max_steps,
        )
        return {
            "result": {
                "route": "sql",
                "decision": "count query",
                "answer": "Found 5 doors.",
                "error": None,
            }
        }

    monkeypatch.delenv(ROUTER_PROFILE_ENV_VAR, raising=False)
    monkeypatch.delenv(AGENT_PROFILE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "rag_tag.evals.task_runner.load_project_config",
        lambda start_dir: LoadedConfig(),
    )
    monkeypatch.setattr("rag_tag.evals.task_runner.execute_query", fake_execute_query)

    bundle = run_benchmark_case(
        case,
        db_paths=[Path("/tmp/model.db")],
        runtime=None,
        agent=None,
    )

    assert bundle.result.router_profile == "router-from-config"
    assert bundle.result.agent_profile == "agent-from-config"
