"""Output schemas for graph agent responses."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

_JSON_CODE_BLOCK_RE = re.compile(
    r"^```(?:json)?\s*(?P<body>[\s\S]*?)\s*```$",
    flags=re.IGNORECASE,
)

_FRAMEWORK_ERROR_MARKERS = (
    "validation error",
    "invalid json",
    "fix the errors and try again",
)


def _looks_like_framework_error(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _FRAMEWORK_ERROR_MARKERS)


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


def _parse_json_like_string(text: str) -> Any | None:
    candidate = _strip_json_code_block(text.strip())
    if not candidate:
        return None

    for value in (candidate, _slice_first_json_like_value(candidate)):
        if not value:
            continue
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_graph_answer_input(data: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return data

    if isinstance(data, str):
        text = data.strip()
        if not text:
            return data

        parsed = _parse_json_like_string(text)
        if parsed is not None:
            return _normalize_graph_answer_input(parsed, depth=depth + 1)

        if _looks_like_framework_error(text):
            return data

        # Last-resort compatibility path for providers that output plain text
        # instead of a final_result tool-call object.
        return {"answer": text}

    if isinstance(data, list):
        if len(data) == 1:
            return _normalize_graph_answer_input(data[0], depth=depth + 1)
        return data

    if isinstance(data, dict):
        # Canonical malformed wrapper observed in production logs.
        if "parameters" in data and ("tool_name" in data or "tool_call_id" in data):
            return _normalize_graph_answer_input(data["parameters"], depth=depth + 1)

        # Alternate wrapper shape used by some tool-call stacks.
        if "arguments" in data and "name" in data:
            return _normalize_graph_answer_input(data["arguments"], depth=depth + 1)

        if (
            "tool_calls" in data
            and isinstance(data["tool_calls"], list)
            and len(data["tool_calls"]) == 1
        ):
            return _normalize_graph_answer_input(data["tool_calls"][0], depth=depth + 1)

    return data


class GraphAnswer(BaseModel):
    """Final answer from graph agent with optional data sample."""

    answer: str = Field(description="Natural language answer to the user's question")
    data: dict[str, object] | None = Field(
        default=None,
        description="Optional structured data (e.g., sample elements, counts)",
    )
    warning: str | None = Field(
        default=None,
        description="Optional warning message (e.g., max steps exceeded)",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_tool_wrapper(cls, data: Any) -> Any:
        """Silently normalize common malformed output shapes from the model.

        The model occasionally wraps its final_result arguments inside a
        list or inside a tool-call envelope dict.  Both patterns are handled
        here so that PydanticAI's built-in output-retry mechanism can
        succeed within the same ``run_sync`` call instead of exhausting all
        retries on the same structural mistake.

        Handles:
        - Single-element list wrapping the real payload dict.
        - Tool-call envelope: ``{tool_call_id, tool_name, parameters}``
          — extract ``parameters`` as the real payload.
        - The two patterns can be combined (list wrapping an envelope).
        - JSON-like strings (including fenced code blocks) containing
          either of the above malformed shapes.
        - Plain-text fallbacks by coercing to ``{"answer": <text>}``.

        Multi-element lists cannot be normalized and are returned as-is so
        that Pydantic raises an informative type error.
        """
        return _normalize_graph_answer_input(data)
