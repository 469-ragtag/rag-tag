from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SqlRoute = Literal["sql", "graph"]
SqlIntent = Literal["count", "list"]


@dataclass(frozen=True)
class SqlRequest:
    intent: SqlIntent
    ifc_class: str | None
    level_like: str | None
    limit: int


@dataclass(frozen=True)
class RouteDecision:
    route: SqlRoute
    reason: str
    sql_request: SqlRequest | None
