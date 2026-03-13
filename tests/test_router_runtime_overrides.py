from __future__ import annotations

from pathlib import Path

import pytest

from rag_tag.router import router as router_module
from rag_tag.router.models import RouteDecision


def test_route_question_defaults_to_llm_when_router_profile_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n"
        "  router_profile: dbx-router\n"
        "providers:\n"
        "  databricks:\n"
        "    type: databricks\n"
        "profiles:\n"
        "  dbx-router:\n"
        "    provider: databricks\n"
        "    model: router-endpoint\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        router_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "router",
    )
    monkeypatch.delenv("ROUTER_MODE", raising=False)
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setattr(
        router_module,
        "_route_with_llm_fallback",
        lambda question, *, debug_llm_io=False: RouteDecision(
            "sql",
            f"llm:{question}:{debug_llm_io}",
            None,
        ),
    )

    decision = router_module.route_question("How many doors?", debug_llm_io=True)

    assert decision.reason == "llm:How many doors?:True"


def test_route_question_defaults_to_llm_when_config_router_mode_is_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  router_mode: llm\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        router_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "router",
    )
    monkeypatch.delenv("ROUTER_MODE", raising=False)
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setattr(
        router_module,
        "_route_with_llm_fallback",
        lambda question, *, debug_llm_io=False: RouteDecision(
            "graph",
            f"config-mode:{question}",
            None,
        ),
    )

    decision = router_module.route_question("Which rooms are adjacent?")

    assert decision.reason == "config-mode:Which rooms are adjacent?"


def test_route_question_defaults_to_rule_when_no_llm_config_is_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)

    monkeypatch.setattr(
        router_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "router",
    )
    monkeypatch.delenv("ROUTER_MODE", raising=False)
    monkeypatch.delenv("ROUTER_PROFILE", raising=False)
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setattr(
        router_module,
        "_route_with_llm_fallback",
        lambda question, *, debug_llm_io=False: (_ for _ in ()).throw(
            AssertionError("LLM routing should not run")
        ),
    )
    monkeypatch.setattr(
        router_module,
        "route_question_rule",
        lambda question: RouteDecision("graph", f"rule:{question}", None),
    )

    decision = router_module.route_question("Find doors near the stair core.")

    assert decision.reason == "rule:Find doors near the stair core."


def _write_project_marker(project_root: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'test'\n", encoding="utf-8"
    )
