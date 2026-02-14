"""PydanticAI integration for router and agent workflows.

This module provides a factory for creating PydanticAI agents configured
from environment variables. It maps the current provider/model conventions
to PydanticAI model strings.

Model string format: 'provider:model-name'
- Router: google:gemini-2.5-flash (Gemini 2.5 Flash)
- Agent: cohere:command-a-03-2025 (Cohere Command A)
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


def _load_env() -> None:
    """Load environment variables from .env file if available."""
    if load_dotenv is None:
        return
    from rag_tag.paths import find_project_root

    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is not None:
        load_dotenv(project_root / ".env")


def get_router_model() -> str:
    """Get router model string from environment.

    Priority:
    1. ROUTER_MODEL env var (override)
    2. Default: google:gemini-2.5-flash

    Returns:
        PydanticAI model string for router (e.g., 'google:gemini-2.5-flash')
    """
    _load_env()
    model = os.getenv("ROUTER_MODEL")
    if model:
        return model
    # Default router model: Gemini 2.5 Flash
    return "google:gemini-2.5-flash"


def get_agent_model() -> str:
    """Get agent model string from environment.

    Priority:
    1. AGENT_MODEL env var (override)
    2. Default: cohere:command-a-03-2025

    Returns:
        PydanticAI model string for agent (e.g., 'cohere:command-a-03-2025')
    """
    _load_env()
    model = os.getenv("AGENT_MODEL")
    if model:
        return model
    # Default agent model: Cohere Command A
    return "cohere:command-a-03-2025"


def resolve_model_from_provider(
    provider_name: str | None = None,
    *,
    model_override: str | None = None,
    role: str = "agent",
) -> str:
    """Resolve a PydanticAI model string from legacy provider name.

    This is a compatibility shim for the existing provider resolution logic.
    Maps legacy provider names (cohere, gemini) to PydanticAI model strings.

    Args:
        provider_name: Legacy provider name (cohere, gemini)
        model_override: Optional model name override
        role: Role of the agent ('router' or 'agent')

    Returns:
        PydanticAI model string (e.g., 'cohere:command-a-03-2025')

    Raises:
        RuntimeError: If provider is unknown or not configured
    """
    _load_env()

    # If model override provided, try to construct model string
    if model_override:
        # If already in PydanticAI format (provider:model), return as-is
        if ":" in model_override:
            return model_override
        # Otherwise, need provider prefix
        if provider_name:
            return f"{provider_name}:{model_override}"
        # If no provider, try to infer from model name
        if "gemini" in model_override.lower():
            return f"google:{model_override}"
        if "command" in model_override.lower():
            return f"cohere:{model_override}"

    # Auto-detect from provider name
    if provider_name:
        provider = provider_name.strip().lower()
        if provider == "gemini":
            return (
                "google:gemini-2.5-flash"
                if role == "router"
                else "google:gemini-3-flash-preview"
            )
        if provider == "cohere":
            return "cohere:command-a-03-2025"
        raise RuntimeError(
            f"Unknown provider: {provider_name}. Supported: cohere, gemini"
        )

    # Use role-specific defaults
    if role == "router":
        return get_router_model()
    return get_agent_model()
