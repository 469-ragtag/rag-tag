"""Shared Pydantic models for LLM provider abstraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ToolCall(BaseModel):
    """Represents a single tool call with action and parameters."""

    action: str
    params: dict[str, object] = {}

    model_config = ConfigDict(extra="forbid")


class AgentStep(BaseModel):
    """Represents a single agent step (tool call or final answer)."""

    type: Literal["tool", "final"]
    action: str | None = None
    params: dict[str, object] | None = None
    answer: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_step(self):
        if self.type == "tool":
            if not self.action:
                raise ValueError("action is required for tool steps")
        elif self.type == "final":
            if not self.answer:
                raise ValueError("answer is required for final steps")
        return self


class TraceEvent(BaseModel):
    """Simple tracing event for future observability."""

    event: str
    run_id: str | None = None
    step_id: int | None = None
    payload: dict[str, object] | None = None

    model_config = ConfigDict(extra="allow")
