"""Base provider interface for LLM interactions."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseProvider(ABC):
    """Abstract base class for LLM provider adapters."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        debug_llm_io: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.debug_llm_io = debug_llm_io

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate unstructured text response."""
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        *,
        schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Generate structured response matching the provided Pydantic schema."""
        ...

    def _print_llm_input(self, label: str, content: str) -> None:
        """Print LLM input to stderr if debug_llm_io is enabled."""
        if not self.debug_llm_io:
            return
        separator = "=" * 72
        print(
            f"{separator}\nLLM INPUT ({label})\n{separator}",
            file=sys.stderr,
        )
        print(content, file=sys.stderr)

    def _print_llm_output(self, label: str, content: str | Any) -> None:
        """Print LLM output to stderr if debug_llm_io is enabled."""
        if not self.debug_llm_io:
            return
        text = content if isinstance(content, str) else str(content)
        separator = "-" * 72
        print(
            f"{separator}\nLLM OUTPUT ({label})\n{separator}",
            file=sys.stderr,
        )
        print(text, file=sys.stderr)
