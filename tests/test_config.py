from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from rag_tag.config import (
    CONFIG_PATH_ENV_VAR,
    AppConfig,
    discover_project_config,
    load_project_config,
    load_project_env,
)
from rag_tag.llm import pydantic_ai as pydantic_ai_module


def test_load_project_env_loads_project_root_env_and_maps_cohere_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ROUTER_MODEL=google-gla:gemini-3-flash-preview\n"
        "COHERE_API_KEY=test-cohere-key\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)

    loaded_env = load_project_env(tmp_path / "src" / "rag_tag")

    assert loaded_env == env_path
    assert os.environ["ROUTER_MODEL"] == "google-gla:gemini-3-flash-preview"
    assert os.environ["COHERE_API_KEY"] == "test-cohere-key"
    assert os.environ["CO_API_KEY"] == "test-cohere-key"


def test_load_project_env_preserves_existing_co_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / ".env").write_text(
        "COHERE_API_KEY=env-cohere-key\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CO_API_KEY", "existing-co-key")

    load_project_env(tmp_path)

    assert os.environ["COHERE_API_KEY"] == "env-cohere-key"
    assert os.environ["CO_API_KEY"] == "existing-co-key"


def test_discover_project_config_prefers_yaml_before_yml_and_json(
    tmp_path: Path,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config.yml").write_text("profiles: {}\n", encoding="utf-8")
    expected = tmp_path / "config.yaml"
    expected.write_text("providers: {}\n", encoding="utf-8")

    discovered = discover_project_config(tmp_path / "nested" / "pkg")

    assert discovered == expected


def test_load_project_config_parses_yaml_structure(tmp_path: Path) -> None:
    _write_project_marker(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "defaults:\n"
        "  router_profile: router-default\n"
        "  agent_profile: dbx-agent\n"
        "providers:\n"
        "  databricks:\n"
        "    type: databricks\n"
        "    base_url: https://workspace.example.com/serving-endpoints\n"
        "    token_env: DATABRICKS_TOKEN\n"
        "profiles:\n"
        "  router-default:\n"
        "    model: google-gla:gemini-2.5-flash\n"
        "  dbx-agent:\n"
        "    provider: databricks\n"
        "    model: databricks-meta-llama\n"
        "    settings:\n"
        "      temperature: 0.1\n"
        "experiments:\n"
        "  graph-compare:\n"
        "    router_profile: router-default\n"
        "    agent_profile: dbx-agent\n",
        encoding="utf-8",
    )

    loaded = load_project_config(tmp_path / "src" / "rag_tag")

    assert loaded.project_root == tmp_path
    assert loaded.config_path == config_path
    assert loaded.config.defaults.router_profile == "router-default"
    assert loaded.config.defaults.agent_profile == "dbx-agent"
    assert loaded.config.providers["databricks"].base_url == (
        "https://workspace.example.com/serving-endpoints"
    )
    assert loaded.config.profiles["dbx-agent"].provider == "databricks"
    assert loaded.config.profiles["dbx-agent"].settings == {"temperature": 0.1}
    assert loaded.config.experiments["graph-compare"].agent_profile == "dbx-agent"


def test_load_project_config_uses_explicit_relative_json_path(tmp_path: Path) -> None:
    _write_project_marker(tmp_path)
    discovered_path = tmp_path / "config.yaml"
    discovered_path.write_text("profiles: {}\n", encoding="utf-8")

    explicit_path = tmp_path / "configs.json"
    explicit_path.write_text(
        json.dumps(
            {
                "providers": {
                    "openai-compatible": {
                        "type": "openai-compatible",
                        "base_url": "https://workspace.example.com/openai",
                    }
                },
                "profiles": {
                    "comparison": {
                        "provider": "openai-compatible",
                        "model": "databricks-claude",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_project_config(tmp_path / "src", config_path="configs.json")

    assert loaded.config_path == explicit_path
    assert loaded.config.profiles["comparison"].model == "databricks-claude"


def test_load_project_config_uses_env_config_override_when_no_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    discovered_path = tmp_path / "config.yaml"
    discovered_path.write_text(
        "defaults:\n"
        "  router_profile: discovered-router\n"
        "profiles:\n"
        "  discovered-router:\n"
        "    model: google-gla:gemini-2.5-flash\n",
        encoding="utf-8",
    )

    override_path = tmp_path / "override.json"
    override_path.write_text(
        json.dumps(
            {
                "defaults": {"router_profile": "override-router"},
                "profiles": {
                    "override-router": {"model": "google-gla:gemini-3-flash-preview"}
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(CONFIG_PATH_ENV_VAR, str(override_path))

    loaded = load_project_config(tmp_path / "src" / "rag_tag")

    assert loaded.config_path == override_path
    assert loaded.config.defaults.router_profile == "override-router"


def test_load_project_config_returns_empty_defaults_without_config(
    tmp_path: Path,
) -> None:
    _write_project_marker(tmp_path)

    loaded = load_project_config(tmp_path)

    assert loaded.config_path is None
    assert loaded.config == AppConfig()


def test_checked_in_config_example_matches_app_config_schema() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config = AppConfig.model_validate(payload)

    assert config.defaults.router_profile in config.profiles
    assert config.defaults.agent_profile in config.profiles
    experiment = config.experiments["graph-dbx-smoke"]
    assert experiment.router_profile in config.profiles
    assert experiment.agent_profile in config.profiles
    assert all(profile_name in config.profiles for profile_name in experiment.profiles)


def test_get_router_model_keeps_env_based_lookup_via_shared_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / ".env").write_text(
        "ROUTER_MODEL=google-gla:gemini-2.5-flash\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.setattr(
        pydantic_ai_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "llm",
    )

    assert pydantic_ai_module.get_router_model() == "google-gla:gemini-2.5-flash"


def _write_project_marker(project_root: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'test'\n", encoding="utf-8"
    )
