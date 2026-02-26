"""Output schemas for graph agent responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


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

        Multi-element lists cannot be normalized and are returned as-is so
        that Pydantic raises an informative type error.
        """
        # Unwrap single-element list first so the envelope check below
        # also handles  [{"tool_name": ..., "parameters": {...}}].
        if isinstance(data, list):
            if len(data) == 1:
                data = data[0]
            else:
                # Cannot normalize; return as-is so Pydantic reports the
                # type mismatch with a clear "Input should be a valid dict"
                # error that PydanticAI will relay to the model for retry.
                return data

        # Unwrap tool-call envelope: {"tool_call_id": ..., "tool_name": ...,
        # "parameters": {actual GraphAnswer fields}}.
        if (
            isinstance(data, dict)
            and "parameters" in data
            and ("tool_name" in data or "tool_call_id" in data)
        ):
            data = data["parameters"]

        return data
