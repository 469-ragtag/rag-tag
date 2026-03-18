from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import normalize_sql_field_key

LlmRoute = Literal["sql", "graph"]
LlmIntent = Literal["count", "list", "aggregate", "group", "none"]
LlmAggregateOp = Literal["count", "sum", "avg", "min", "max"]
LlmFieldSource = Literal["element", "property", "quantity"]
LlmFilterOp = Literal["eq", "neq", "lt", "lte", "gt", "gte", "like"]


class LlmFieldRef(BaseModel):
    source: LlmFieldSource
    field: str

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: object) -> LlmFieldSource:
        if value is None:
            raise ValueError("field source is required")
        source = str(value).strip().lower()
        if source in {"element", "property", "quantity"}:
            return source  # type: ignore[return-value]
        raise ValueError("field source must be element, property, or quantity")

    @field_validator("field", mode="before")
    @classmethod
    def _normalize_field(cls, value: object) -> str:
        if value is None:
            raise ValueError("field is required")
        return normalize_sql_field_key(str(value))


class LlmValueFilter(BaseModel):
    field: str
    op: LlmFilterOp = "eq"
    value: str

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    @field_validator("field", mode="before")
    @classmethod
    def _normalize_field(cls, value: object) -> str:
        if value is None:
            raise ValueError("filter field is required")
        return normalize_sql_field_key(str(value))

    @field_validator("op", mode="before")
    @classmethod
    def _normalize_op(cls, value: object) -> LlmFilterOp:
        if value is None:
            return "eq"
        op = str(value).strip().lower()
        if op in {"eq", "neq", "lt", "lte", "gt", "gte", "like"}:
            return op  # type: ignore[return-value]
        raise ValueError("filter op must be eq, neq, lt, lte, gt, gte, or like")

    @field_validator("value", mode="before")
    @classmethod
    def _normalize_value(cls, value: object) -> str:
        if value is None:
            raise ValueError("filter value is required")
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip()


class LlmRouteResponse(BaseModel):
    route: LlmRoute
    intent: LlmIntent = "none"
    ifc_class: str | None = None
    level_like: str | None = None
    predefined_type: str | None = None
    type_name: str | None = None
    property_filters: list[LlmValueFilter] = Field(default_factory=list)
    quantity_filters: list[LlmValueFilter] = Field(default_factory=list)
    aggregate_op: LlmAggregateOp | None = None
    aggregate_field: LlmFieldRef | None = None
    group_by: LlmFieldRef | None = None
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
        if intent in {"count", "list", "aggregate", "group", "none"}:
            return intent  # type: ignore[return-value]
        raise ValueError(
            "intent must be 'count', 'list', 'aggregate', 'group', or 'none'"
        )

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

    @field_validator("level_like", "predefined_type", "type_name", mode="before")
    @classmethod
    def _normalize_optional_string(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("property_filters", "quantity_filters", mode="before")
    @classmethod
    def _normalize_optional_filter_lists(
        cls, value: object
    ) -> list[dict[str, object]] | object:
        if value is None:
            return []
        if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
            return []
        if isinstance(value, tuple):
            return list(value)
        return value

    @field_validator("aggregate_op", mode="before")
    @classmethod
    def _normalize_aggregate_op(cls, value: object) -> LlmAggregateOp | None:
        if value is None:
            return None
        cleaned = str(value).strip().lower()
        if not cleaned:
            return None
        if cleaned in {"count", "sum", "avg", "min", "max"}:
            return cleaned  # type: ignore[return-value]
        raise ValueError("aggregate_op must be count, sum, avg, min, or max")

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, value: object) -> str:
        if value is None:
            return "LLM route decision"
        if not isinstance(value, str):
            return "LLM route decision"
        cleaned = value.strip()
        return cleaned or "LLM route decision"

    @model_validator(mode="after")
    def _validate_sql_shape(self) -> LlmRouteResponse:
        if self.route == "graph":
            return self
        if self.intent == "none":
            raise ValueError("sql route requires a non-none intent")
        if self.intent == "aggregate":
            if self.aggregate_op is None:
                raise ValueError("aggregate sql route requires aggregate_op")
            if self.aggregate_op != "count" and self.aggregate_field is None:
                raise ValueError(
                    "non-count aggregate sql route requires aggregate_field"
                )
            if self.group_by is not None:
                raise ValueError("aggregate sql route must not include group_by")
        elif self.intent == "group":
            if self.group_by is None:
                raise ValueError("group sql route requires group_by")
            if self.aggregate_op is not None or self.aggregate_field is not None:
                raise ValueError("group sql route must not include aggregate fields")
        else:
            if self.aggregate_op is not None or self.aggregate_field is not None:
                raise ValueError(
                    "count/list sql routes must not include aggregate fields"
                )
            if self.group_by is not None:
                raise ValueError("count/list sql routes must not include group_by")
        return self
