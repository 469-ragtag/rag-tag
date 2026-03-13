from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel

from rag_tag.agent.graph_agent import GraphAgent
from rag_tag.agent.models import GraphAnswer
from rag_tag.llm import pydantic_ai as pydantic_ai_module
from rag_tag.router import llm as router_llm_module
from rag_tag.router.llm_models import LlmRouteResponse


def test_get_agent_model_resolves_databricks_profile_to_openai_chat_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n"
        "  agent_profile: dbx-agent\n"
        "providers:\n"
        "  databricks:\n"
        "    type: databricks\n"
        "    host_env: DATABRICKS_HOST\n"
        "    token_env: DATABRICKS_TOKEN\n"
        "profiles:\n"
        "  dbx-agent:\n"
        "    provider: databricks\n"
        "    model: databricks-meta-llama\n"
        "    settings:\n"
        "      temperature: 0.2\n"
        "      max_tokens: 512\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pydantic_ai_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "llm",
    )
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    model = pydantic_ai_module.get_agent_model()
    settings = pydantic_ai_module.get_agent_model_settings()

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "databricks-meta-llama"
    assert model.base_url == "https://workspace.example.com/serving-endpoints/"
    assert settings == {
        "temperature": 0.2,
        "max_tokens": 512,
        "parallel_tool_calls": False,
    }


def test_get_agent_model_normalizes_databricks_host_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n"
        "  agent_profile: dbx-agent\n"
        "providers:\n"
        "  databricks:\n"
        "    type: databricks\n"
        "    host: https://workspace.example.com/custom\n"
        "profiles:\n"
        "  dbx-agent:\n"
        "    provider: databricks\n"
        "    model: databricks-meta-llama\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pydantic_ai_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "llm",
    )
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    model = pydantic_ai_module.get_agent_model()

    assert isinstance(model, OpenAIChatModel)
    assert model.base_url == "https://workspace.example.com/custom/serving-endpoints/"


@pytest.mark.parametrize(
    ("host_or_url", "expected"),
    [
        (
            "workspace.example.com",
            "https://workspace.example.com/serving-endpoints",
        ),
        (
            "https://workspace.example.com/serving-endpoints",
            "https://workspace.example.com/serving-endpoints",
        ),
    ],
)
def test_normalize_databricks_base_url(
    host_or_url: str,
    expected: str,
) -> None:
    assert pydantic_ai_module.normalize_databricks_base_url(host_or_url) == expected


def test_agent_model_env_override_wins_over_configured_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n"
        "  agent_profile: dbx-agent\n"
        "providers:\n"
        "  databricks:\n"
        "    type: databricks\n"
        "    host_env: DATABRICKS_HOST\n"
        "profiles:\n"
        "  dbx-agent:\n"
        "    provider: databricks\n"
        "    model: databricks-meta-llama\n"
        "    settings:\n"
        "      temperature: 0.2\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pydantic_ai_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "llm",
    )
    monkeypatch.setenv("AGENT_MODEL", "cohere:command-a-03-2025")
    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    assert pydantic_ai_module.get_agent_model() == "cohere:command-a-03-2025"
    assert pydantic_ai_module.get_agent_model_settings() is None


def test_agent_model_can_resolve_cohere_profile_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n"
        "  agent_profile: cohere-command-a\n"
        "profiles:\n"
        "  cohere-command-a:\n"
        "    model: cohere:command-a-03-2025\n"
        "    settings:\n"
        "      temperature: 0.1\n"
        "      max_tokens: 512\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pydantic_ai_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "llm",
    )
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("AGENT_PROFILE", raising=False)

    assert pydantic_ai_module.get_agent_model() == "cohere:command-a-03-2025"
    assert pydantic_ai_module.get_agent_model_settings() == {
        "temperature": 0.1,
        "max_tokens": 512,
    }


def test_router_profile_env_override_wins_over_configured_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n"
        "  router_profile: gemini-router\n"
        "providers:\n"
        "  databricks:\n"
        "    type: databricks\n"
        "    host_env: DATABRICKS_HOST\n"
        "    token_env: DATABRICKS_TOKEN\n"
        "profiles:\n"
        "  gemini-router:\n"
        "    model: google-gla:gemini-2.5-flash\n"
        "  dbx-router:\n"
        "    provider: databricks\n"
        "    model: router-endpoint\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pydantic_ai_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "llm",
    )
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.setenv("ROUTER_PROFILE", "dbx-router")
    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    model = pydantic_ai_module.get_router_model()

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "router-endpoint"


def test_router_and_graph_agent_receive_configured_model_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_marker(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults:\n"
        "  router_profile: dbx-router\n"
        "  agent_profile: dbx-agent\n"
        "providers:\n"
        "  databricks:\n"
        "    type: databricks\n"
        "    host_env: DATABRICKS_HOST\n"
        "    token_env: DATABRICKS_TOKEN\n"
        "profiles:\n"
        "  dbx-router:\n"
        "    provider: databricks\n"
        "    model: router-endpoint\n"
        "    settings:\n"
        "      temperature: 0.1\n"
        "      max_tokens: 128\n"
        "  dbx-agent:\n"
        "    provider: databricks\n"
        "    model: graph-endpoint\n"
        "    settings:\n"
        "      temperature: 0.3\n"
        "      max_tokens: 256\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pydantic_ai_module,
        "_MODULE_DIR",
        tmp_path / "src" / "rag_tag" / "llm",
    )
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    router_captured: dict[str, object] = {}
    graph_captured: dict[str, object] = {}

    class FakeRouterAgent:
        def __init__(self, model: object, **kwargs: object) -> None:
            router_captured["model"] = model
            router_captured["model_settings"] = kwargs.get("model_settings")

        def run_sync(self, question: str) -> SimpleNamespace:
            return SimpleNamespace(
                output=LlmRouteResponse(
                    route="sql",
                    intent="count",
                    ifc_class="IfcDoor",
                    reason=question,
                )
            )

    class FakeGraphAgent:
        def __init__(self, model: object, **kwargs: object) -> None:
            graph_captured["model"] = model
            graph_captured["model_settings"] = kwargs.get("model_settings")

        def output_validator(self, func: object) -> object:
            return func

    monkeypatch.setattr(router_llm_module, "Agent", FakeRouterAgent)
    monkeypatch.setattr("rag_tag.agent.graph_agent.Agent", FakeGraphAgent)
    monkeypatch.setattr(
        "rag_tag.agent.graph_agent.register_graph_tools", lambda agent: None
    )

    decision = router_llm_module.route_with_llm("How many doors are there?")
    GraphAgent()

    assert decision.route == "sql"
    assert isinstance(router_captured["model"], OpenAIChatModel)
    assert router_captured["model_settings"] == {
        "temperature": 0.1,
        "max_tokens": 128,
        "parallel_tool_calls": False,
    }
    assert isinstance(graph_captured["model"], OpenAIChatModel)
    assert graph_captured["model_settings"] == {
        "temperature": 0.3,
        "max_tokens": 256,
        "parallel_tool_calls": False,
    }


def test_databricks_schema_transformer_rewrites_nullable_unions() -> None:
    graph_schema = pydantic_ai_module.DatabricksJsonSchemaTransformer(
        GraphAnswer.model_json_schema()
    ).walk()
    route_schema = pydantic_ai_module.DatabricksJsonSchemaTransformer(
        LlmRouteResponse.model_json_schema()
    ).walk()

    assert "anyOf" not in _dump_schema(graph_schema)
    assert "anyOf" not in _dump_schema(route_schema)
    assert graph_schema["properties"]["data"]["type"] == ["object", "null"]
    assert graph_schema["properties"]["warning"]["type"] == ["string", "null"]
    assert route_schema["properties"]["ifc_class"]["type"] == ["string", "null"]
    assert route_schema["properties"]["level_like"]["type"] == ["string", "null"]


def test_databricks_schema_transformer_inlines_nested_defs() -> None:
    class NestedPayload(BaseModel):
        value: str

    class Wrapper(BaseModel):
        payload: NestedPayload

    schema = pydantic_ai_module.DatabricksJsonSchemaTransformer(
        Wrapper.model_json_schema()
    ).walk()

    assert "$defs" not in schema
    assert "$ref" not in _dump_schema(schema)
    assert schema["properties"]["payload"]["properties"]["value"]["type"] == "string"


def _dump_schema(schema: dict[str, object]) -> str:
    return json.dumps(schema, sort_keys=True)


def _write_project_marker(project_root: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'test'\n", encoding="utf-8"
    )
