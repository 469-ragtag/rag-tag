from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

LlmRoute = Literal["sql", "graph"]
LlmIntent = Literal["count", "list", "none"]


class LlmRouteResponse(BaseModel):
    route: LlmRoute
    intent: LlmIntent = "none"
    ifc_class: str | None = None
    level_like: str | None = None
    reason: str = "LLM route decision"

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    @field_validator("route", mode="before")
    @classmethod
    def _normalize_route(cls, value: object) -> LlmRoute:
        if value is None:
            raise ValueError("route is required")
        route = str(value).strip().lower()
        if route == "sql":
            return "sql"
        if route == "graph":
            return "graph"
        raise ValueError("route must be 'sql' or 'graph'")

    @field_validator("intent", mode="before")
    @classmethod
    def _normalize_intent(cls, value: object) -> LlmIntent:
        if value is None:
            return "none"
        intent = str(value).strip().lower()
        if not intent:
            return "none"
        if intent == "count":
            return "count"
        if intent == "list":
            return "list"
        if intent == "none":
            return "none"
        raise ValueError("intent must be 'count', 'list', or 'none'")

    @field_validator("ifc_class", mode="before")
    @classmethod
    def _normalize_ifc_class(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if not cleaned.lower().startswith("ifc"):
            cleaned = f"Ifc{cleaned}"
        core = cleaned[3:]
        if not core:
            return "Ifc"
        return "Ifc" + core[0].upper() + core[1:]

    @field_validator("level_like", mode="before")
    @classmethod
    def _normalize_level_like(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, value: object) -> str:
        if value is None:
            return "LLM route decision"
        if not isinstance(value, str):
            return "LLM route decision"
        cleaned = value.strip()
        return cleaned or "LLM route decision"
