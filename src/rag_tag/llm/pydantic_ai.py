"""PydanticAI model resolution for router and graph workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from rag_tag.config import (
    AGENT_PROFILE_ENV_VAR,
    ROUTER_PROFILE_ENV_VAR,
    AppConfig,
    ProfileConfig,
    ProviderConfig,
    load_project_config,
)

_MODULE_DIR = Path(__file__).resolve().parent

DEFAULT_ROUTER_MODEL = "google-gla:gemini-2.5-flash"
DEFAULT_AGENT_MODEL = "cohere:command-a-03-2025"

ResolvedModel = str | OpenAIChatModel
RoleName = Literal["router", "agent"]


class DatabricksJsonSchemaTransformer(InlineDefsJsonSchemaTransformer):
    """Inline refs and simplify nullable unions for Databricks compatibility."""

    def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
        schema = dict(schema)

        for union_key in ("anyOf", "oneOf"):
            simplified = _simplify_nullable_union(schema, union_key)
            if simplified is not None:
                return simplified

        if "allOf" in schema:
            flattened = _flatten_all_of(schema)
            if flattened is not None:
                return flattened

        schema.pop("$defs", None)
        return schema


_DATABRICKS_MODEL_PROFILE = OpenAIModelProfile(
    json_schema_transformer=DatabricksJsonSchemaTransformer,
    openai_supports_strict_tool_definition=False,
)


@dataclass(frozen=True)
class ResolvedRoleModel:
    model: ResolvedModel
    settings: ModelSettings | None = None


def get_router_model() -> ResolvedModel:
    """Get the router model from env/config resolution."""
    return _resolve_role_model("router").model


def get_router_model_settings() -> ModelSettings | None:
    """Get router model settings from config when applicable."""
    return _resolve_role_model("router").settings


def get_agent_model() -> ResolvedModel:
    """Get the graph-agent model from env/config resolution."""
    return _resolve_role_model("agent").model


def get_agent_model_settings() -> ModelSettings | None:
    """Get graph-agent model settings from config when applicable."""
    return _resolve_role_model("agent").settings


def has_role_configuration(role: str, *, start_dir: Path | None = None) -> bool:
    """Return True when a role has an explicit env/config model selection."""
    normalized_role = _normalize_role(role)
    if _read_env(f"{normalized_role.upper()}_MODEL") is not None:
        return True
    loaded = load_project_config(start_dir or _MODULE_DIR)
    return _get_selected_profile_name(loaded.config, role=normalized_role) is not None


def resolve_model_from_provider(
    provider_name: str | None = None,
    *,
    model_override: str | None = None,
    role: str = "agent",
) -> ResolvedModel:
    """Resolve a PydanticAI model from a provider hint or legacy override."""
    loaded = load_project_config(_MODULE_DIR)

    if model_override:
        override = model_override.strip()
        if ":" in override:
            return override
        if provider_name and provider_name.strip().lower() == "databricks":
            return _build_databricks_model(
                model_name=override,
                provider_name="databricks",
                provider_config=ProviderConfig(type="databricks"),
            )
        if provider_name:
            return _string_model_from_provider(provider_name, override)
        if "gemini" in override.lower():
            return f"google-gla:{override}"
        if "command" in override.lower():
            return f"cohere:{override}"

    if provider_name:
        provider = provider_name.strip().lower()
        if provider == "databricks":
            profile_name = _get_selected_profile_name(
                loaded.config, role=_normalize_role(role)
            )
            if profile_name is not None:
                return _resolve_profile(
                    profile_name=profile_name,
                    profile=loaded.config.profiles[profile_name],
                    providers=loaded.config.providers,
                ).model
            raise RuntimeError(
                "Databricks provider resolution requires a configured profile "
                "or model override."
            )
        if provider in ("gemini", "google"):
            return (
                DEFAULT_ROUTER_MODEL
                if role == "router"
                else "google-gla:gemini-3-flash-preview"
            )
        if provider == "google-vertex":
            return "google-vertex:gemini-2.5-flash"
        if provider == "cohere":
            return DEFAULT_AGENT_MODEL
        raise RuntimeError(
            f"Unknown provider: {provider_name}. Supported: cohere, "
            "databricks, gemini, google"
        )

    return _resolve_role_model(_normalize_role(role)).model


def _resolve_role_model(role: RoleName) -> ResolvedRoleModel:
    loaded = load_project_config(_MODULE_DIR)

    env_model = _read_env(f"{role.upper()}_MODEL")
    if env_model is not None:
        return ResolvedRoleModel(model=env_model)

    profile_name = _get_selected_profile_name(loaded.config, role=role)
    if profile_name is None:
        return ResolvedRoleModel(model=_default_model_for_role(role))

    try:
        profile = loaded.config.profiles[profile_name]
    except KeyError as exc:
        raise RuntimeError(
            f"Configured {role} profile '{profile_name}' was not found "
            "in project config."
        ) from exc

    return _resolve_profile(
        profile_name=profile_name,
        profile=profile,
        providers=loaded.config.providers,
    )


def _resolve_profile(
    *,
    profile_name: str,
    profile: ProfileConfig,
    providers: dict[str, ProviderConfig],
) -> ResolvedRoleModel:
    provider_config = providers.get(profile.provider) if profile.provider else None

    if _is_databricks_provider(profile.provider, provider_config):
        model = _build_databricks_model(
            model_name=profile.model,
            provider_name=profile.provider or "databricks",
            provider_config=provider_config or ProviderConfig(type="databricks"),
        )
    else:
        model = _resolve_standard_model(profile, provider_config)

    return ResolvedRoleModel(
        model=model,
        settings=_build_model_settings(profile_name, profile, provider_config),
    )


def _resolve_standard_model(
    profile: ProfileConfig,
    provider_config: ProviderConfig | None,
) -> str:
    model = profile.model.strip()
    if ":" in model:
        return model
    if profile.provider:
        provider_name = _provider_prefix(profile.provider, provider_config)
        return f"{provider_name}:{model}"
    return model


def _build_databricks_model(
    *,
    model_name: str,
    provider_name: str,
    provider_config: ProviderConfig,
) -> OpenAIChatModel:
    base_url = _resolve_databricks_base_url(provider_name, provider_config)
    token = _resolve_databricks_token(provider_name, provider_config)
    provider = OpenAIProvider(base_url=base_url, api_key=token)
    return OpenAIChatModel(
        model_name.strip(),
        provider=provider,
        profile=_DATABRICKS_MODEL_PROFILE,
    )


def _build_model_settings(
    profile_name: str,
    profile: ProfileConfig,
    provider_config: ProviderConfig | None,
) -> ModelSettings | None:
    settings = dict(profile.settings)

    if provider_config is not None and provider_config.headers:
        extra_headers = dict(provider_config.headers)
        current_headers = settings.get("extra_headers")
        if isinstance(current_headers, dict):
            extra_headers.update(current_headers)
        settings["extra_headers"] = extra_headers

    if _is_databricks_provider(profile.provider, provider_config):
        settings.pop("parallel_tool_calls", None)

    if not settings:
        return None

    try:
        return cast(ModelSettings, settings)
    except TypeError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Model settings for profile '{profile_name}' could not be resolved."
        ) from exc


def _get_selected_profile_name(config: AppConfig, *, role: RoleName) -> str | None:
    runtime_profile = _read_env(_profile_env_var(role))
    if runtime_profile is not None:
        return runtime_profile

    defaults = config.defaults
    profile_name = (
        defaults.router_profile if role == "router" else defaults.agent_profile
    )
    if profile_name:
        return profile_name
    if role in config.profiles:
        return role
    return None


def _profile_env_var(role: RoleName) -> str:
    return ROUTER_PROFILE_ENV_VAR if role == "router" else AGENT_PROFILE_ENV_VAR


def _provider_prefix(
    provider_name: str,
    provider_config: ProviderConfig | None,
) -> str:
    provider_type = _normalized_provider_type(provider_name, provider_config)
    if provider_type in {"gemini", "google", "google-gla"}:
        return "google-gla"
    if provider_type == "google-vertex":
        return "google-vertex"
    if provider_type == "openai-compatible":
        return "openai"
    return provider_type


def _string_model_from_provider(provider_name: str, model_name: str) -> str:
    return f"{_provider_prefix(provider_name, None)}:{model_name}"


def _default_model_for_role(role: RoleName) -> str:
    return DEFAULT_ROUTER_MODEL if role == "router" else DEFAULT_AGENT_MODEL


def _resolve_databricks_base_url(
    provider_name: str,
    provider_config: ProviderConfig,
) -> str:
    base_url = _read_env(provider_config.base_url_env) or _clean_string(
        provider_config.base_url
    )
    if base_url is not None:
        return base_url

    host = _read_env(provider_config.host_env) or _clean_string(provider_config.host)
    if host is None:
        host = _read_env("DATABRICKS_HOST")
    if host is None:
        host = _read_env("DATABRICKS_BASE_URL")
        if host is not None:
            return host

    if host is None:
        raise RuntimeError(
            f"Provider '{provider_name}' requires a Databricks host/base URL. "
            "Set base_url, base_url_env, host, host_env, or DATABRICKS_HOST."
        )
    return normalize_databricks_base_url(host)


def _resolve_databricks_token(
    provider_name: str,
    provider_config: ProviderConfig,
) -> str:
    token_env = (
        provider_config.token_env or provider_config.api_key_env or "DATABRICKS_TOKEN"
    )
    token = _read_env(token_env)
    if token is None:
        raise RuntimeError(
            f"Provider '{provider_name}' requires env var '{token_env}' "
            "for Databricks access."
        )
    return token


def normalize_databricks_base_url(host_or_url: str) -> str:
    """Normalize a Databricks host into the OpenAI-compatible serving path."""
    value = host_or_url.strip().rstrip("/")
    if not value:
        raise ValueError("Databricks host/base URL cannot be empty.")
    if not value.startswith(("https://", "http://")):
        value = f"https://{value.lstrip('/')}"
    if value.endswith("/serving-endpoints"):
        return value
    return f"{value}/serving-endpoints"


def _is_databricks_provider(
    provider_name: str | None,
    provider_config: ProviderConfig | None,
) -> bool:
    provider_type = _normalized_provider_type(provider_name, provider_config)
    return provider_type == "databricks"


def _normalized_provider_type(
    provider_name: str | None,
    provider_config: ProviderConfig | None,
) -> str:
    value = (
        provider_config.type
        if provider_config and provider_config.type
        else provider_name
    )
    return _clean_string(value, default="") or ""


def _normalize_role(role: str) -> RoleName:
    normalized = role.strip().lower()
    if normalized == "router":
        return "router"
    return "agent"


def _read_env(name: str | None) -> str | None:
    if name is None:
        return None
    value = os.getenv(name)
    return _clean_string(value)


def _clean_string(value: object | None, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text


def _simplify_nullable_union(
    schema: dict[str, Any],
    union_key: Literal["anyOf", "oneOf"],
) -> dict[str, Any] | None:
    members = schema.get(union_key)
    if not isinstance(members, list):
        return None

    null_members = [member for member in members if member == {"type": "null"}]
    non_null_members = [member for member in members if member != {"type": "null"}]
    if len(null_members) != 1 or len(non_null_members) != 1:
        return None

    merged = dict(non_null_members[0])
    for key, value in schema.items():
        if key in {union_key, "$defs"}:
            continue
        if key == "default" and value is None:
            continue
        merged.setdefault(key, value)

    schema_type = merged.get("type")
    if isinstance(schema_type, str):
        merged["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list):
        merged["type"] = (
            schema_type if "null" in schema_type else [*schema_type, "null"]
        )
    else:
        merged["nullable"] = True
    return merged


def _flatten_all_of(schema: dict[str, Any]) -> dict[str, Any] | None:
    members = schema.get("allOf")
    if not isinstance(members, list) or len(members) != 1:
        return None

    merged = dict(members[0])
    for key, value in schema.items():
        if key in {"allOf", "$defs"}:
            continue
        merged.setdefault(key, value)
    return merged
