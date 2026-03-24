from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from rag_tag.config import (
    CONFIG_PATH_ENV_VAR,
    AppConfig,
    GraphOrchestrationConfig,
    discover_project_config,
    load_project_config,
    load_project_env,
)
from rag_tag.llm import pydantic_ai as pydantic_ai_module
from rag_tag.query_service import (
    get_graph_orchestration_config,
    resolve_graph_orchestrator,
)


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
        "  graph_orchestrator: langgraph\n"
        "  graph_max_steps: 12\n"
        "  graph_output_retries: 4\n"
        "graph_orchestration:\n"
        "  enabled_subquestion_decomposition: false\n"
        "  max_subquestions: 4\n"
        "  reserved_orchestration_steps: 2\n"
        "  specialist_step_cap: 5\n"
        "  fallback_to_graph_agent: false\n"
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
        "    agent_profile: dbx-agent\n"
        "    questions_file: evals/benchmark_cases_v1.yaml\n",
        encoding="utf-8",
    )

    loaded = load_project_config(tmp_path / "src" / "rag_tag")

    assert loaded.project_root == tmp_path
    assert loaded.config_path == config_path
    assert loaded.config.defaults.router_profile == "router-default"
    assert loaded.config.defaults.agent_profile == "dbx-agent"
    assert loaded.config.defaults.graph_orchestrator == "langgraph"
    assert loaded.config.graph_orchestration == GraphOrchestrationConfig(
        enabled_subquestion_decomposition=False,
        max_subquestions=4,
        reserved_orchestration_steps=2,
        specialist_step_cap=5,
        fallback_to_graph_agent=False,
    )
    assert loaded.config.defaults.graph_max_steps == 12
    assert loaded.config.defaults.graph_output_retries == 4
    assert loaded.config.providers["databricks"].base_url == (
        "https://workspace.example.com/serving-endpoints"
    )
    assert loaded.config.profiles["dbx-agent"].provider == "databricks"
    assert loaded.config.profiles["dbx-agent"].settings == {"temperature": 0.1}
    assert loaded.config.experiments["graph-compare"].agent_profile == "dbx-agent"
    assert loaded.config.experiments["graph-compare"].questions_file == (
        "evals/benchmark_cases_v1.yaml"
    )


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


def test_app_config_defaults_graph_orchestration_when_omitted() -> None:
    config = AppConfig.model_validate({"profiles": {}})

    assert config.defaults.graph_orchestrator is None
    assert config.graph_orchestration == GraphOrchestrationConfig()


def test_app_config_accepts_custom_graph_orchestration_values() -> None:
    config = AppConfig.model_validate(
        {
            "defaults": {"graph_orchestrator": "langgraph"},
            "graph_orchestration": {
                "enabled_subquestion_decomposition": False,
                "max_subquestions": 6,
                "reserved_orchestration_steps": 4,
                "specialist_step_cap": 7,
                "fallback_to_graph_agent": False,
            },
        }
    )

    assert config.defaults.graph_orchestrator == "langgraph"
    assert config.graph_orchestration == GraphOrchestrationConfig(
        enabled_subquestion_decomposition=False,
        max_subquestions=6,
        reserved_orchestration_steps=4,
        specialist_step_cap=7,
        fallback_to_graph_agent=False,
    )


def test_checked_in_config_example_matches_app_config_schema() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config = AppConfig.model_validate(payload)

    assert "cohere-command-a" in config.profiles
    assert config.providers["databricks"].host_env == "DATABRICKS_HOST"
    assert config.providers["databricks"].host is None
    assert config.defaults.router_profile in config.profiles
    assert config.defaults.agent_profile in config.profiles
    assert config.defaults.graph_orchestrator == "pydanticai"
    assert config.graph_orchestration == GraphOrchestrationConfig()
    assert config.defaults.graph_max_steps == 20
    assert config.defaults.graph_output_retries == 5
    assert config.defaults.graph_max_steps == 20
    assert config.defaults.graph_output_retries == 5
    for experiment_name in ("graph-dbx-smoke", "graph-agent-compare"):
        experiment = config.experiments[experiment_name]
        assert experiment.router_profile in config.profiles
        assert experiment.agent_profile in config.profiles
        assert all(
            profile_name in config.profiles for profile_name in experiment.profiles
        )

    benchmark_experiment = config.experiments["benchmark-e2e-v1"]
    assert benchmark_experiment.questions_file == "evals/benchmark_cases_v1.yaml"


def test_repo_does_not_require_checked_in_runtime_config_yaml() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"

    assert not config_path.exists()


def test_resolve_graph_orchestrator_defaults_to_pydanticai_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    monkeypatch.setattr(
        "rag_tag.query_service.load_project_config",
        lambda start_dir: load_project_config(tmp_path),
    )

    assert resolve_graph_orchestrator() == "pydanticai"


def test_resolve_graph_orchestrator_reads_explicit_langgraph_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  graph_orchestrator: langgraph\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rag_tag.query_service.load_project_config",
        lambda start_dir: load_project_config(tmp_path),
    )

    assert resolve_graph_orchestrator() == "langgraph"


def test_resolve_graph_orchestrator_rejects_invalid_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  graph_orchestrator: invalid-orchestrator\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rag_tag.query_service.load_project_config",
        lambda start_dir: load_project_config(tmp_path),
    )

    with pytest.raises(ValueError, match="Allowed values: langgraph, pydanticai"):
        resolve_graph_orchestrator()


def test_get_graph_orchestration_config_returns_typed_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    monkeypatch.setattr(
        "rag_tag.query_service.load_project_config",
        lambda start_dir: load_project_config(tmp_path),
    )

    assert get_graph_orchestration_config() == GraphOrchestrationConfig()


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
