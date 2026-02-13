"""Provider registry for resolving LLM providers."""

from __future__ import annotations

import os
from pathlib import Path

from .providers.base import BaseProvider

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


def resolve_provider(
    name: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    debug_llm_io: bool = False,
) -> BaseProvider:
    """Resolve and instantiate an LLM provider by name.

    Args:
        name: Provider name (cohere, gemini). If None, auto-detects from env.
        api_key: API key for the provider. If None, reads from environment.
        model: Model name. If None, uses provider default.
        debug_llm_io: Enable debug printing of LLM I/O.

    Returns:
        Instantiated provider.

    Raises:
        RuntimeError: If provider cannot be resolved or instantiated.
    """
    _load_env()

    if name is None:
        name = _auto_detect_provider()

    provider_name = name.strip().lower()

    if provider_name == "cohere":
        from .providers.cohere import CohereProvider

        return CohereProvider(
            api_key=api_key,
            model=model,
            debug_llm_io=debug_llm_io,
        )

    if provider_name == "gemini":
        from .providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=api_key,
            model=model,
            debug_llm_io=debug_llm_io,
        )

    raise RuntimeError(f"Unknown provider: {name}. Supported providers: cohere, gemini")


def _auto_detect_provider() -> str:
    """Auto-detect provider from environment variables."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("COHERE_API_KEY"):
        return "cohere"
    raise RuntimeError(
        "No LLM provider configured. Set GEMINI_API_KEY or COHERE_API_KEY."
    )
