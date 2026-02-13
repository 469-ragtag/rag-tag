"""Gemini provider adapter."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import BaseModel, ValidationError

from .base import BaseProvider

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


_LOGGER = logging.getLogger(__name__)


def _load_env() -> None:
    if load_dotenv is None:
        return
    from rag_tag.paths import find_project_root

    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is not None:
        load_dotenv(project_root / ".env")


class GeminiProvider(BaseProvider):
    """Gemini LLM provider adapter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        debug_llm_io: bool = False,
    ) -> None:
        super().__init__(api_key=api_key, model=model, debug_llm_io=debug_llm_io)
        _load_env()
        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable")
        self._model = self.model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

        try:
            from google import genai
        except ModuleNotFoundError as exc:
            raise RuntimeError("google-genai not installed; add dependency") from exc

        try:
            self._client = genai.Client(api_key=key)
        except TypeError:
            self._client = genai.Client()

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate unstructured text response."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        return self._generate_text_content(full_prompt)

    def generate_structured(
        self,
        prompt: str,
        *,
        schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Generate structured response using Gemini's native schema support."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        if _schema_has_additional_properties(schema):
            _LOGGER.warning(
                "Gemini schema contains additionalProperties; "
                "falling back to text parsing for %s.",
                schema.__name__,
            )
            text = self._generate_text_content(full_prompt)
            return _parse_text_as_schema(text, schema)

        self._print_llm_input("gemini", full_prompt)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except Exception as exc:
            if _is_schema_error(exc):
                _LOGGER.warning(
                    "Gemini schema error; falling back to text parsing for %s.",
                    schema.__name__,
                )
                text = self._generate_text_content(full_prompt)
                return _parse_text_as_schema(text, schema)
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        self._print_llm_output("gemini", response)

        return _parse_structured_response(response, schema)

    def _generate_text_content(self, full_prompt: str) -> str:
        self._print_llm_input("gemini", full_prompt)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        text = _extract_response_text(response)
        if not text:
            raise RuntimeError("Gemini did not return text")

        self._print_llm_output("gemini", text)
        return text


def _extract_response_text(response: Any) -> str | None:
    """Extract text from Gemini response object."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        parts_text: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if isinstance(parts, list) and parts:
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        parts_text.append(part_text)
            else:
                content_text = getattr(content, "text", None)
                if isinstance(content_text, str) and content_text.strip():
                    parts_text.append(content_text)
        if parts_text:
            return "\n".join(parts_text)

    result = getattr(response, "result", None) or getattr(response, "_result", None)
    if isinstance(result, dict):
        result_text = result.get("text")
        if isinstance(result_text, str) and result_text.strip():
            return result_text

    return None


def _parse_structured_response(response: Any, schema: type[BaseModel]) -> BaseModel:
    """Parse structured response from Gemini."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed

    if parsed is not None:
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise RuntimeError(f"Gemini returned invalid schema: {exc}") from exc

    text = _extract_response_text(response)
    if isinstance(text, str) and text.strip():
        return _parse_text_as_schema(text, schema)

    raise RuntimeError("Gemini did not return structured data")


def _parse_text_as_schema(text: str, schema: type[BaseModel]) -> BaseModel:
    """Parse text as JSON and validate against schema."""
    cleaned = _strip_code_fences(text)
    try:
        return schema.model_validate_json(cleaned)
    except ValidationError as exc:
        extracted = _extract_json_object(cleaned)
        if extracted is None:
            raise RuntimeError("Model did not return JSON") from exc
        try:
            return schema.model_validate_json(extracted)
        except ValidationError as nested_exc:
            raise RuntimeError(
                f"Model returned invalid JSON: {nested_exc}"
            ) from nested_exc


def _strip_code_fences(text: str) -> str:
    """Remove code fences from text."""
    if not text.strip().startswith("```"):
        return text.strip()
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("```")
    ).strip()


def _extract_json_object(text: str) -> str | None:
    """Extract JSON object from text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _schema_has_additional_properties(schema: type[BaseModel]) -> bool:
    try:
        schema_json = schema.model_json_schema()
    except Exception:
        return False
    return _contains_key(schema_json, "additionalProperties") or _contains_key(
        schema_json, "additional_properties"
    )


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value:
            return True
        return any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


def _is_schema_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_schema" in message or "additional_properties" in message
