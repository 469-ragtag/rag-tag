"""Pydantic models for agent outputs and structured responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GraphAnswer(BaseModel):
    """Final answer from graph agent."""

    answer: str = Field(
        description="Concise natural language answer to the user's question"
    )

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
