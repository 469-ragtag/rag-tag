"""Shared routing data models and type literals.

These types define the contract between router and query executors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SqlRoute = Literal["sql", "graph"]
SqlIntent = Literal["count", "list"]


@dataclass(frozen=True)
class SqlRequest:
    """Normalized SQL execution request produced by routing."""

    intent: SqlIntent
    ifc_class: str | None
    level_like: str | None
    limit: int


@dataclass(frozen=True)
class RouteDecision:
    """Top-level routing decision with optional SQL payload."""

    route: SqlRoute
    reason: str
    sql_request: SqlRequest | None
