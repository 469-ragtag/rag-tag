from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_tag.config import (
    AGENT_PROFILE_ENV_VAR,
    CONFIG_PATH_ENV_VAR,
    ROUTER_PROFILE_ENV_VAR,
)
from rag_tag.run_agent import _apply_runtime_overrides, _resolve_config_override_path


def test_apply_runtime_overrides_sets_config_and_profile_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "runtime-config.yaml"
    config_path.write_text("profiles: {}\n", encoding="utf-8")

    monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
    monkeypatch.delenv(ROUTER_PROFILE_ENV_VAR, raising=False)
    monkeypatch.delenv(AGENT_PROFILE_ENV_VAR, raising=False)

    resolved_path, error = _resolve_config_override_path(config_path)

    assert error is None
    assert resolved_path is not None

    _apply_runtime_overrides(
        config_path=resolved_path,
        router_profile="runtime-router",
        agent_profile="runtime-agent",
    )

    assert Path(os.environ[CONFIG_PATH_ENV_VAR]) == config_path.resolve()
    assert os.environ[ROUTER_PROFILE_ENV_VAR] == "runtime-router"
    assert os.environ[AGENT_PROFILE_ENV_VAR] == "runtime-agent"


def test_resolve_config_override_path_reports_missing_file(tmp_path: Path) -> None:
    resolved_path, error = _resolve_config_override_path(tmp_path / "missing.yaml")

    assert resolved_path is None
    assert error == f"Config file not found: {(tmp_path / 'missing.yaml').resolve()}"
