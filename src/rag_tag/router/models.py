from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rag_tag.usage import UsageMetrics

SqlRoute = Literal["sql", "graph"]
SqlIntent = Literal["count", "list", "aggregate", "group"]
SqlAggregateOp = Literal["count", "sum", "avg", "min", "max"]
SqlFieldSource = Literal["element", "property", "quantity"]
SqlFilterOp = Literal["eq", "neq", "lt", "lte", "gt", "gte", "like"]

_SAFE_FIELD_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)?$")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def normalize_sql_field_key(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("SQL field key must not be empty")
    parts = [part.strip() for part in cleaned.split(".")]
    if not all(parts):
        raise ValueError(f"Invalid SQL field key: {value!r}")

    normalized_parts = [_normalize_sql_field_segment(part) for part in parts]
    normalized = ".".join(normalized_parts)
    if not _SAFE_FIELD_KEY_RE.fullmatch(normalized):
        raise ValueError(f"Invalid SQL field key: {value!r}")
    return normalized


def _normalize_sql_field_segment(value: str) -> str:
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        return value

    tokens = _WORD_RE.findall(value)
    if not tokens:
        raise ValueError(f"Invalid SQL field segment: {value!r}")

    normalized_tokens: list[str] = []
    for token in tokens:
        if token.isupper() or token.isdigit():
            normalized_tokens.append(token)
            continue
        normalized_tokens.append(token[0].upper() + token[1:])
    return "".join(normalized_tokens)


@dataclass(frozen=True)
class SqlFieldRef:
    source: SqlFieldSource
    field: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", normalize_sql_field_key(self.field))


@dataclass(frozen=True)
class SqlValueFilter:
    source: SqlFieldSource
    field: str
    op: SqlFilterOp
    value: str | int | float | bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", normalize_sql_field_key(self.field))


@dataclass(frozen=True)
class SqlRequest:
    intent: SqlIntent
    ifc_class: str | None
    level_like: str | None
    predefined_type: str | None = None
    type_name: str | None = None
    text_match: str | None = None
    element_filters: tuple[SqlValueFilter, ...] = ()
    property_filters: tuple[SqlValueFilter, ...] = ()
    quantity_filters: tuple[SqlValueFilter, ...] = ()
    aggregate_op: SqlAggregateOp | None = None
    aggregate_field: SqlFieldRef | None = None
    group_by: SqlFieldRef | None = None
    limit: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("SQL request limit must be non-negative")
        _validate_filter_sources(self.element_filters, expected_source="element")
        _validate_filter_sources(self.property_filters, expected_source="property")
        _validate_filter_sources(self.quantity_filters, expected_source="quantity")
        if self.intent == "aggregate":
            if self.aggregate_op is None:
                raise ValueError("Aggregate SQL requests require aggregate_op")
            if self.aggregate_op != "count" and self.aggregate_field is None:
                raise ValueError(
                    "Non-count aggregate SQL requests require aggregate_field"
                )
            if self.group_by is not None:
                raise ValueError("Aggregate SQL requests do not support group_by")
        elif self.intent == "group":
            if self.group_by is None:
                raise ValueError("Group SQL requests require group_by")
            if self.aggregate_op is not None or self.aggregate_field is not None:
                raise ValueError(
                    "Group SQL requests do not support aggregate_op/aggregate_field"
                )
        else:
            if self.aggregate_op is not None or self.aggregate_field is not None:
                raise ValueError(
                    "Count/list SQL requests do not support aggregate fields"
                )
            if self.group_by is not None:
                raise ValueError("Count/list SQL requests do not support group_by")


def _validate_filter_sources(
    filters: tuple[SqlValueFilter, ...],
    *,
    expected_source: SqlFieldSource,
) -> None:
    for filter_item in filters:
        if filter_item.source != expected_source:
            raise ValueError(
                "SQL request filter source mismatch: "
                f"expected '{expected_source}', got '{filter_item.source}'"
            )


@dataclass(frozen=True)
class RouteDecision:
    route: SqlRoute
    reason: str
    sql_request: SqlRequest | None
    usage: UsageMetrics | None = None
