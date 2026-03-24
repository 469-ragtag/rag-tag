from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_tag.agent.graph_agent import build_system_prompt
from rag_tag.evals.runtime import (
    BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR,
    BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR,
    temporary_runtime_overrides,
)
from rag_tag.evals.strategies import resolve_benchmark_strategy
from rag_tag.query_service import resolve_graph_orchestrator


def test_resolve_benchmark_strategy_returns_expected_settings() -> None:
    baseline = resolve_benchmark_strategy("baseline")
    strict_grounded = resolve_benchmark_strategy("strict-grounded")
    decompose = resolve_benchmark_strategy("decompose")

    assert baseline.graph_orchestrator_override is None
    assert baseline.graph_prompt_append is None
    assert strict_grounded.graph_prompt_append is not None
    assert decompose.graph_orchestrator_override == "langgraph"
    assert decompose.graph_prompt_append is not None


def test_resolve_benchmark_strategy_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark prompt strategy"):
        resolve_benchmark_strategy("unsupported")


def test_temporary_runtime_overrides_restore_strategy_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR, "pydanticai")
    monkeypatch.setenv(BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR, "original appendix")

    with temporary_runtime_overrides(
        graph_orchestrator_override="langgraph",
        graph_prompt_append="benchmark appendix",
    ):
        assert os.environ[BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR] == "langgraph"
        assert os.environ[BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR] == (
            "benchmark appendix"
        )

    assert os.environ[BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR] == "pydanticai"
    assert os.environ[BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR] == "original appendix"


def test_resolve_graph_orchestrator_honors_benchmark_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BENCHMARK_GRAPH_ORCHESTRATOR_ENV_VAR, "langgraph")
    monkeypatch.setattr(
        "rag_tag.query_service.load_project_config",
        lambda start_dir: (_ for _ in ()).throw(
            AssertionError("should not load config")
        ),
    )

    assert resolve_graph_orchestrator() == "langgraph"


def test_build_system_prompt_appends_benchmark_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BENCHMARK_GRAPH_PROMPT_APPEND_ENV_VAR, "benchmark appendix")

    prompt = build_system_prompt()

    assert "benchmark appendix" in prompt
    assert "You are a tool-using graph reasoning agent" in prompt
