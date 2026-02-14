"""PydanticAI model factory and provider resolution.

Maps environment variables to PydanticAI model objects.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_ai.models import Model

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    from rag_tag.paths import find_project_root

    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is not None:
        load_dotenv(project_root / ".env")


def resolve_model(
    provider_name: str | None = None,
    *,
    model_name: str | None = None,
    purpose: str = "agent",
) -> Model:
    """Resolve PydanticAI model from environment variables.

    Args:
        provider_name: Provider name (cohere, gemini, openai). If None, auto-detects.
        model_name: Model name override. If None, uses provider default.
        purpose: Purpose hint for default model selection (agent, router).

    Returns:
        PydanticAI Model instance.

    Raises:
        RuntimeError: If provider cannot be resolved or API key is missing.
    """
    _load_env()

    if provider_name is None:
        provider_name = _auto_detect_provider()

    provider = provider_name.strip().lower()

    if provider == "cohere":
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY environment variable is required.")

        if model_name is None:
            model_name = os.getenv("COHERE_MODEL")
        if model_name is None:
            model_name = (
                "command-r-08-2024" if purpose == "router" else "command-a-03-2025"
            )

        from pydantic_ai.models.cohere import CohereModel
        from pydantic_ai.providers.cohere import CohereProvider

        return CohereModel(model_name, provider=CohereProvider(api_key=api_key))

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is required.")

        if model_name is None:
            model_name = os.getenv("GEMINI_MODEL")
        if model_name is None:
            model_name = (
                "gemini-2.5-flash" if purpose == "router" else "gemini-2.5-flash"
            )

        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key))

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required.")

        if model_name is None:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_key))

    raise RuntimeError(
        f"Unknown provider: {provider_name}. "
        "Supported providers: cohere, gemini, openai"
    )


def _auto_detect_provider() -> str:
    """Auto-detect provider from environment variables."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("COHERE_API_KEY"):
        return "cohere"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, COHERE_API_KEY, or OPENAI_API_KEY."
    )
