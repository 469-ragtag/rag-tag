"""Shared project configuration and environment loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_tag.paths import find_project_root

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

DEFAULT_CONFIG_FILENAMES = ("config.yaml", "config.yml", "config.json")


class ProviderConfig(BaseModel):
    """Named provider configuration for future model integrations."""

    model_config = ConfigDict(extra="allow")

    type: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class ProfileConfig(BaseModel):
    """Named model profile configuration."""

    model_config = ConfigDict(extra="allow")

    model: str
    provider: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Named experiment grouping for future profile comparisons."""

    model_config = ConfigDict(extra="allow")

    description: str | None = None
    router_profile: str | None = None
    agent_profile: str | None = None
    profiles: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    """Top-level checked-in application configuration."""

    model_config = ConfigDict(extra="allow")

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    experiments: dict[str, ExperimentConfig] = Field(default_factory=dict)


@dataclass(frozen=True)
class LoadedProjectConfig:
    """Resolved project config artifacts for the current workspace."""

    project_root: Path | None
    env_path: Path | None
    config_path: Path | None
    config: AppConfig


def load_project_env(start_dir: Path | None = None) -> Path | None:
    """Load the project-root ``.env`` file if available.

    The file is discovered by first resolving the project root via
    ``pyproject.toml``. Existing environment variables are preserved.
    """

    project_root = find_project_root(start_dir)
    env_path = project_root / ".env" if project_root is not None else None

    if env_path is not None and env_path.is_file() and load_dotenv is not None:
        load_dotenv(env_path)

    _sync_provider_env_aliases()
    return env_path if env_path is not None and env_path.is_file() else None


def discover_project_config(start_dir: Path | None = None) -> Path | None:
    """Discover the default checked-in config file in project root."""

    project_root = find_project_root(start_dir)
    if project_root is None:
        return None

    for filename in DEFAULT_CONFIG_FILENAMES:
        config_path = project_root / filename
        if config_path.is_file():
            return config_path
    return None


def load_project_config(
    start_dir: Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> LoadedProjectConfig:
    """Load shared environment variables and the project config file."""

    project_root = find_project_root(start_dir)
    env_path = load_project_env(start_dir)
    resolved_config_path = _resolve_config_path(
        config_path=config_path,
        project_root=project_root,
        start_dir=start_dir,
    )

    if resolved_config_path is None:
        return LoadedProjectConfig(
            project_root=project_root,
            env_path=env_path,
            config_path=None,
            config=AppConfig(),
        )

    payload = _read_config_payload(resolved_config_path)
    return LoadedProjectConfig(
        project_root=project_root,
        env_path=env_path,
        config_path=resolved_config_path,
        config=AppConfig.model_validate(payload),
    )


def _resolve_config_path(
    *,
    config_path: str | Path | None,
    project_root: Path | None,
    start_dir: Path | None,
) -> Path | None:
    if config_path is None:
        return discover_project_config(start_dir)

    candidate = Path(config_path)
    if not candidate.is_absolute() and project_root is not None:
        candidate = project_root / candidate
    elif not candidate.is_absolute() and start_dir is not None:
        candidate = Path(start_dir) / candidate

    if not candidate.is_file():
        raise FileNotFoundError(f"Config file not found: {candidate}")
    return candidate


def _read_config_payload(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML config support requires PyYAML to be installed.")
        data = yaml.safe_load(raw_text)
    elif suffix == ".json":
        data = json.loads(raw_text)
    else:
        raise RuntimeError(
            f"Unsupported config format for {config_path.name}: expected YAML or JSON"
        )

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {config_path} must contain a top-level object mapping."
        )
    return data


def _sync_provider_env_aliases() -> None:
    if "CO_API_KEY" not in os.environ and "COHERE_API_KEY" in os.environ:
        os.environ["CO_API_KEY"] = os.environ["COHERE_API_KEY"]
