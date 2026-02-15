"""Output schemas for graph agent responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
