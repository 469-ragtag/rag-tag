"""Pydantic models for agent outputs and structured responses."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GraphAnswer(BaseModel):
    """Final answer from graph agent."""

    answer: str = Field(
        description="Concise natural language answer to the user's question"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_provider_output(cls, value: Any) -> Any:
        """Normalize provider outputs into the strict `{answer: str}` contract."""
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("Answer cannot be empty.")

            if stripped.startswith("{"):
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError:
                    return {"answer": stripped}
                else:
                    if isinstance(decoded, dict):
                        value = decoded
                    else:
                        return {"answer": stripped}
            else:
                return {"answer": stripped}

        if isinstance(value, dict):
            for key in ("answer", "final_answer", "response", "text", "content"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return {"answer": candidate.strip()}

            message = value.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return {"answer": content.strip()}

            if "data" in value:
                data = value["data"]
                if isinstance(data, dict):
                    for key in ("answer", "final_answer", "response", "text"):
                        candidate = data.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            return {"answer": candidate.strip()}

            raise ValueError("Output is missing a valid answer field.")

        raise ValueError("Unsupported output format for graph answer.")

    model_config = ConfigDict(extra="forbid")


class AnswerEnvelope(BaseModel):
    """Envelope for agent results with answer and optional data."""

    answer: str = Field(description="Natural language answer")
    data: dict[str, Any] | None = Field(
        default=None, description="Optional structured data"
    )

    model_config = ConfigDict(extra="allow")


class RouterDecision(BaseModel):
    """Router decision with route and metadata."""

    route: Literal["sql", "graph"]
    intent: Literal["count", "list", "none"] = "none"
    ifc_class: str | None = None
    level_like: str | None = None
    reason: str = "Router decision"

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
