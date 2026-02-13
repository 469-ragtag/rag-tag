"""Cohere provider adapter."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, cast

import cohere
from pydantic import BaseModel, ValidationError

from .base import BaseProvider

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


class CohereProvider(BaseProvider):
    """Cohere LLM provider adapter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        debug_llm_io: bool = False,
    ) -> None:
        super().__init__(api_key=api_key, model=model, debug_llm_io=debug_llm_io)
        _load_env()
        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Missing COHERE_API_KEY environment variable")
        self._model = self.model or os.getenv("COHERE_MODEL", "command-a-03-2025")
        self._client = cohere.Client(key)

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate unstructured text response."""
        self._print_llm_input("cohere", prompt)
        client = cast(Any, self._client)
        try:
            if system_prompt:
                resp = client.chat(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
            else:
                resp = client.chat(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
            text = getattr(resp, "text", None) or str(resp)
        except TypeError:
            if system_prompt:
                resp = client.chat(
                    model=self._model,
                    message=prompt,
                    preamble=system_prompt,
                )
            else:
                resp = client.chat(
                    model=self._model,
                    message=prompt,
                )
            text = getattr(resp, "text", None) or str(resp)
        self._print_llm_output("cohere", text)
        return text

    def generate_structured(
        self,
        prompt: str,
        *,
        schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Generate structured response by parsing JSON from text output."""
        schema_name = schema.__name__
        json_instruction = (
            f"Return ONLY a valid JSON object matching the {schema_name} schema. "
            "Do not include any text before or after the JSON object."
        )
        full_prompt = f"{prompt}\n\n{json_instruction}"
        text = self.generate_text(full_prompt, system_prompt=system_prompt)
        json_str = _extract_json(text)
        try:
            return schema.model_validate_json(json_str)
        except ValidationError as exc:
            raise RuntimeError(
                f"Cohere returned invalid JSON for {schema_name}: {exc}"
            ) from exc


def _extract_json(text: str) -> str:
    """Extract JSON object from model output."""
    text = text.strip()
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    if text.startswith("```"):
        lines = text.splitlines()
        cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(cleaned_lines).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return JSON")

    return text[start : end + 1]
