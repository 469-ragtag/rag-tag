from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from rag_tag.config import AGENT_PROFILE_ENV_VAR, CONFIG_PATH_ENV_VAR, AppConfig


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

    with module.temporary_profile_overrides(
        config_path="/tmp/override-config.yaml",
        agent_profile="override-agent",
    ):
        assert os.environ[CONFIG_PATH_ENV_VAR] == "/tmp/override-config.yaml"
        assert os.environ[AGENT_PROFILE_ENV_VAR] == "override-agent"

    assert os.environ[CONFIG_PATH_ENV_VAR] == "/tmp/original-config.yaml"
    assert os.environ[AGENT_PROFILE_ENV_VAR] == "original-agent"


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
        decision: object,
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
