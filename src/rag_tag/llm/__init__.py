"""LLM provider abstraction layer with PydanticAI."""

from __future__ import annotations

from .pydantic_ai import resolve_model

__all__ = ["resolve_model"]
