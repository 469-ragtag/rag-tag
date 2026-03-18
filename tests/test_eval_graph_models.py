from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from rag_tag.agent import langgraph_agent as langgraph_agent_module
from rag_tag.agent.langgraph_agent import LangGraphAgent
from rag_tag.config import AGENT_PROFILE_ENV_VAR, CONFIG_PATH_ENV_VAR, AppConfig
from rag_tag.graph import GraphRuntime, wrap_networkx_graph


def _load_eval_graph_models_module():
    module_name = "tests_eval_graph_models"
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "eval_graph_models.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_agent_profiles_uses_experiment_and_deduplicates() -> None:
    module = _load_eval_graph_models_module()
    config = AppConfig.model_validate(
        {
            "profiles": {
                "alpha": {"model": "cohere:command-a-03-2025"},
                "beta": {"model": "google-gla:gemini-2.5-flash"},
            },
            "experiments": {
                "graph-compare": {
                    "profiles": ["alpha", "beta"],
                    "agent_profile": "beta",
                }
            },
        }
    )

    profiles = module.resolve_agent_profiles(config, experiment="graph-compare")

    assert profiles == ["alpha", "beta"]


def test_temporary_profile_overrides_restore_previous_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_eval_graph_models_module()
    monkeypatch.setenv(CONFIG_PATH_ENV_VAR, "/tmp/original-config.yaml")
    monkeypatch.setenv(AGENT_PROFILE_ENV_VAR, "original-agent")
    monkeypatch.setenv("AGENT_MODEL", "cohere:command-a-03-2025")

    with module.temporary_profile_overrides(
        config_path="/tmp/override-config.yaml",
        agent_profile="override-agent",
    ):
        assert os.environ[CONFIG_PATH_ENV_VAR] == "/tmp/override-config.yaml"
        assert os.environ[AGENT_PROFILE_ENV_VAR] == "override-agent"
        assert "AGENT_MODEL" not in os.environ

    assert os.environ[CONFIG_PATH_ENV_VAR] == "/tmp/original-config.yaml"
    assert os.environ[AGENT_PROFILE_ENV_VAR] == "original-agent"
    assert os.environ["AGENT_MODEL"] == "cohere:command-a-03-2025"


def test_load_questions_supports_default_text_and_json_inputs(
    tmp_path: Path,
) -> None:
    module = _load_eval_graph_models_module()
    text_questions = tmp_path / "questions.txt"
    text_questions.write_text(
        "# graph questions\nWhich rooms are adjacent to the kitchen?\n\n"
        "Find doors near the stair core.\n",
        encoding="utf-8",
    )
    json_questions = tmp_path / "questions.json"
    json_questions.write_text(
        '{"questions": ["Which spaces are connected to the lobby?", '
        '{"question": "Find the path from the lobby to the server room."}]}',
        encoding="utf-8",
    )

    assert module.load_questions(None) == module.DEFAULT_GRAPH_QUESTIONS
    assert module.load_questions(text_questions) == [
        "Which rooms are adjacent to the kitchen?",
        "Find doors near the stair core.",
    ]
    assert module.load_questions(json_questions) == [
        "Which spaces are connected to the lobby?",
        "Find the path from the lobby to the server room.",
    ]


def test_build_profile_summary_counts_errors_warnings_and_data() -> None:
    module = _load_eval_graph_models_module()

    summary = module.build_profile_summary(
        [
            {
                "duration_ms": 10.0,
                "had_error": False,
                "had_warning": True,
                "step_budget_warning": True,
                "has_data": True,
            },
            {
                "duration_ms": 20.0,
                "had_error": True,
                "had_warning": False,
                "step_budget_warning": False,
                "has_data": False,
            },
        ]
    )

    assert summary == {
        "question_count": 2,
        "error_count": 1,
        "warning_count": 1,
        "step_budget_warning_count": 1,
        "data_count": 1,
        "average_duration_ms": 15.0,
    }


def test_evaluate_graph_models_forces_graph_route_and_reports_by_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_eval_graph_models_module()
    calls: list[tuple[str, str, str, str | None, Path | None, int]] = []

    def fake_execute_query(
        question: str,
        db_paths: list[Path],
        runtime: object | None,
        agent: object | None,
        *,
        decision: Any,
        graph_dataset: str | None,
        context_db: Path | None,
        graph_max_steps: int,
    ) -> dict[str, object]:
        assert decision == module.FORCED_GRAPH_DECISION
        profile_name = os.environ[AGENT_PROFILE_ENV_VAR]
        calls.append(
            (
                profile_name,
                question,
                decision.route,
                graph_dataset,
                context_db,
                graph_max_steps,
            )
        )
        warning = (
            f"Step budget exceeded (max_steps={graph_max_steps})"
            if profile_name == "beta"
            else None
        )
        return {
            "result": {
                "route": "graph",
                "decision": decision.reason,
                "answer": f"{profile_name}:{question}",
                "warning": warning,
                "error": None,
                "data": {"profile": profile_name},
            },
            "runtime": object(),
            "agent": object(),
        }

    monkeypatch.setattr(module, "execute_query", fake_execute_query)
    monkeypatch.setattr(module, "close_runtime", lambda runtime: None)

    report = module.evaluate_graph_models(
        questions=["Q1", "Q2"],
        profile_names=["alpha", "beta"],
        db_paths=[Path("/tmp/model.db")],
        config_path="/tmp/config.yaml",
        graph_dataset="model",
        context_db=Path("/tmp/model.db"),
        max_steps=7,
    )

    assert calls == [
        ("alpha", "Q1", "graph", "model", Path("/tmp/model.db"), 7),
        ("alpha", "Q2", "graph", "model", Path("/tmp/model.db"), 7),
        ("beta", "Q1", "graph", "model", Path("/tmp/model.db"), 7),
        ("beta", "Q2", "graph", "model", Path("/tmp/model.db"), 7),
    ]
    assert report["forced_route"] == "graph"
    assert [profile["profile_name"] for profile in report["profiles"]] == [
        "alpha",
        "beta",
    ]
    assert report["profiles"][0]["results"][0]["answer"] == "alpha:Q1"
    assert report["profiles"][1]["summary"]["warning_count"] == 2
    assert report["profiles"][1]["summary"]["step_budget_warning_count"] == 2


def test_evaluate_graph_models_uses_real_execute_query_and_reuses_langgraph_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_eval_graph_models_module()
    config_path = tmp_path / "eval-config.yaml"
    config_path.write_text(
        "defaults:\n"
        "  graph_orchestrator: langgraph\n"
        "  agent_profile: alpha\n"
        "profiles:\n"
        "  alpha:\n"
        "    model: cohere:command-a-03-2025\n",
        encoding="utf-8",
    )

    runtime_creations = {"count": 0}
    agent_init_profiles: list[str] = []
    agent_run_profiles: list[str] = []
    run_count = {"count": 0}

    def fake_ensure_langgraph_dependency() -> None:
        return None

    def fake_ensure_graph_runtime(
        runtime: GraphRuntime | None,
        *,
        graph_dataset: str | None,
        context_db_path: Path | None,
        payload_mode: str | None,
    ) -> GraphRuntime:
        del graph_dataset, context_db_path, payload_mode
        if runtime is not None:
            return runtime
        runtime_creations["count"] += 1
        graph = nx.MultiDiGraph()
        graph.graph["datasets"] = ["model-a"]
        return wrap_networkx_graph(graph)

    original_init = LangGraphAgent.__init__

    def fake_init(
        self: LangGraphAgent,
        *,
        debug_llm_io: bool = False,
        specialist: object | None = None,
        orchestration_config: object | None = None,
        decompose: object | None = None,
        synthesize: object | None = None,
    ) -> None:
        del debug_llm_io, orchestration_config, decompose, synthesize
        original_init(
            self,
            debug_llm_io=False,
            specialist=specialist,
            orchestration_config=module.AppConfig().graph_orchestration,
        )
        agent_init_profiles.append(os.environ[AGENT_PROFILE_ENV_VAR])

    def fake_run(
        self: LangGraphAgent,
        question: str,
        runtime: GraphRuntime,
        *,
        max_steps: int = 20,
        trace: object | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        del self, runtime, trace, run_id
        run_count["count"] += 1
        profile_name = os.environ[AGENT_PROFILE_ENV_VAR]
        agent_run_profiles.append(profile_name)
        return {
            "answer": f"{profile_name}:{question}",
            "data": {"max_steps": max_steps},
        }

    monkeypatch.setattr(
        langgraph_agent_module,
        "_ensure_langgraph_dependency",
        fake_ensure_langgraph_dependency,
    )
    monkeypatch.setattr(
        "rag_tag.query_service.ensure_graph_runtime",
        fake_ensure_graph_runtime,
    )
    monkeypatch.setattr(LangGraphAgent, "__init__", fake_init)
    monkeypatch.setattr(LangGraphAgent, "run", fake_run)
    monkeypatch.setattr(module, "close_runtime", lambda runtime: None)

    report = module.evaluate_graph_models(
        questions=["Q1", "Q2"],
        profile_names=["alpha"],
        db_paths=[],
        config_path=str(config_path),
        graph_dataset="model-a",
        context_db=None,
        max_steps=5,
    )

    assert runtime_creations["count"] == 1
    assert agent_init_profiles == ["alpha"]
    assert agent_run_profiles == ["alpha", "alpha"]
    assert run_count["count"] == 2
    assert report["profiles"][0]["results"][0]["answer"] == "alpha:Q1"
    assert report["profiles"][0]["results"][1]["answer"] == "alpha:Q2"
