"""Output schemas and normalization helpers for graph agent responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

_JSON_CODE_BLOCK_RE = re.compile(
    r"^```(?:json)?\s*(?P<body>[\s\S]*?)\s*```$",
    flags=re.IGNORECASE,
)

_FRAMEWORK_ERROR_MARKERS = (
    "validation error",
    "invalid json",
    "fix the errors and try again",
)


class RecoveryKind(StrEnum):
    """Internal tags for malformed output normalization provenance."""

    NONE = "none"
    PLAIN_TEXT = "plain_text"
    JSON_TEXT = "json_text"
    FENCED_JSON = "fenced_json"
    LIST_WRAPPER = "list_wrapper"
    TOOL_ENVELOPE = "tool_envelope"
    TOOL_CALLS_WRAPPER = "tool_calls_wrapper"


@dataclass(slots=True)
class RecoveryMeta:
    """Internal recovery metadata for normalized graph answers."""

    kind: RecoveryKind = RecoveryKind.NONE
    source: str | None = None


def _looks_like_framework_error(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _FRAMEWORK_ERROR_MARKERS)


def _looks_like_natural_language_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return True
    sentence_markers = (". ", "? ", "! ", ": ")
    if any(marker in stripped for marker in sentence_markers):
        return True
    return stripped.startswith(("-", "*"))


def _strip_json_code_block(text: str) -> str:
    match = _JSON_CODE_BLOCK_RE.match(text)
    return match.group("body").strip() if match else text


def _slice_first_json_like_value(text: str) -> str | None:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return None
    return text[start : end + 1]


def _parse_json_like_string(text: str) -> tuple[Any | None, RecoveryMeta]:
    stripped = text.strip()
    if not stripped:
        return None, RecoveryMeta()

    try:
        return json.loads(stripped), RecoveryMeta(RecoveryKind.JSON_TEXT, "whole")
    except json.JSONDecodeError:
        pass

    match = _JSON_CODE_BLOCK_RE.match(stripped)
    if match:
        block = match.group("body").strip()
        try:
            return json.loads(block), RecoveryMeta(RecoveryKind.FENCED_JSON, "fenced")
        except json.JSONDecodeError:
            pass

    sliced = _slice_first_json_like_value(stripped)
    if sliced is None:
        return None, RecoveryMeta()
    try:
        return json.loads(sliced), RecoveryMeta(RecoveryKind.JSON_TEXT, "sliced")
    except json.JSONDecodeError:
        return None, RecoveryMeta()


def normalize_graph_answer_input(
    data: Any, *, depth: int = 0
) -> tuple[Any, RecoveryMeta]:
    """Normalize malformed model outputs into a GraphAnswer-compatible payload."""
    if depth > 8:
        return data, RecoveryMeta()

    if isinstance(data, str):
        text = data.strip()
        if not text:
            return data, RecoveryMeta()

        parsed, parse_meta = _parse_json_like_string(text)
        if parsed is not None:
            normalized, inner_meta = normalize_graph_answer_input(
                parsed, depth=depth + 1
            )
            return (
                normalized,
                inner_meta if inner_meta.kind != RecoveryKind.NONE else parse_meta,
            )

        if _looks_like_framework_error(text):
            return data, RecoveryMeta()

        if _looks_like_natural_language_answer(text):
            return (
                {"answer": text},
                RecoveryMeta(RecoveryKind.PLAIN_TEXT, "assistant_text"),
            )

        return (
            {"answer": text},
            RecoveryMeta(RecoveryKind.PLAIN_TEXT, "fallback_text"),
        )

    if isinstance(data, list):
        if len(data) == 1:
            normalized, meta = normalize_graph_answer_input(data[0], depth=depth + 1)
            if meta.kind == RecoveryKind.NONE:
                return normalized, RecoveryMeta(
                    RecoveryKind.LIST_WRAPPER, "single_item"
                )
            return normalized, meta
        return data, RecoveryMeta()

    if isinstance(data, dict):
        if "parameters" in data and ("tool_name" in data or "tool_call_id" in data):
            normalized, meta = normalize_graph_answer_input(
                data["parameters"], depth=depth + 1
            )
            if meta.kind == RecoveryKind.NONE:
                return normalized, RecoveryMeta(
                    RecoveryKind.TOOL_ENVELOPE, "parameters"
                )
            return normalized, meta

        if "arguments" in data and "name" in data:
            normalized, meta = normalize_graph_answer_input(
                data["arguments"], depth=depth + 1
            )
            if meta.kind == RecoveryKind.NONE:
                return normalized, RecoveryMeta(RecoveryKind.TOOL_ENVELOPE, "arguments")
            return normalized, meta

        if (
            "tool_calls" in data
            and isinstance(data["tool_calls"], list)
            and len(data["tool_calls"]) == 1
        ):
            normalized, meta = normalize_graph_answer_input(
                data["tool_calls"][0], depth=depth + 1
            )
            if meta.kind == RecoveryKind.NONE:
                return normalized, RecoveryMeta(
                    RecoveryKind.TOOL_CALLS_WRAPPER, "tool_calls"
                )
            return normalized, meta

    return data, RecoveryMeta()


class GraphAnswer(BaseModel):
    """Final answer from graph agent with optional grounded structured data."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Natural language answer to the user's question")
    data: dict[str, object] | None = Field(
        default=None,
        description=(
            "Optional grounded structured data such as evidence, IDs, counts, or "
            "sample elements"
        ),
    )
    warning: str | None = Field(
        default=None,
        description="Optional warning message (e.g., max steps exceeded)",
    )

    _recovery_meta: RecoveryMeta = PrivateAttr(default_factory=RecoveryMeta)

    @model_validator(mode="wrap")
    @classmethod
    def _normalize_tool_wrapper(
        cls,
        data: Any,
        handler: Any,
    ) -> GraphAnswer:
        """Normalize malformed model outputs and preserve internal provenance."""
        normalized, meta = normalize_graph_answer_input(data)
        output = handler(normalized)
        output._recovery_meta = meta
        return output


def was_normalized_from_plain_text(answer: GraphAnswer) -> bool:
    """Return True when a GraphAnswer originated from coerced plain text."""
    return answer._recovery_meta.kind == RecoveryKind.PLAIN_TEXT


def recovery_kind(answer: GraphAnswer) -> RecoveryKind:
    """Return the internal recovery kind for a normalized GraphAnswer."""
    return answer._recovery_meta.kind
