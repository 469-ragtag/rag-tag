from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from query_router import SqlRequest


class SqlQueryError(RuntimeError):
    """Raised when SQL execution or configuration fails."""


def query_ifc_sql(db_path: Path, request: SqlRequest) -> dict[str, Any]:
    if not db_path.exists():
        raise SqlQueryError(f"SQLite database not found: {db_path}")

    where_clauses: list[str] = []
    params: list[object] = []

    if request.ifc_class:
        where_clauses.append("ifc_class = ?")
        params.append(request.ifc_class)
    if request.level_like:
        where_clauses.append("LOWER(level) LIKE ?")
        params.append(f"%{request.level_like.lower()}%")

    where_sql = ""
    if where_clauses:
        where_sql = " WHERE " + " AND ".join(where_clauses)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if request.intent == "count":
            query = f"SELECT COUNT(*) AS count FROM elements{where_sql}"
            row = conn.execute(query, params).fetchone()
            count = int(row["count"]) if row else 0
            summary = _count_summary(request, count)
            return {
                "intent": request.intent,
                "filters": _filters_payload(request),
                "count": count,
                "summary": summary,
                "sql": {"query": query, "params": params},
            }

        if request.intent == "list":
            count_query = f"SELECT COUNT(*) AS count FROM elements{where_sql}"
            count_row = conn.execute(count_query, params).fetchone()
            total_count = int(count_row["count"]) if count_row else 0

            limit = request.limit or 50
            list_query = (
                "SELECT express_id, global_id, ifc_class, name, level, type_name "
                f"FROM elements{where_sql} ORDER BY name LIMIT ?"
            )
            list_params = [*params, limit]
            rows = conn.execute(list_query, list_params).fetchall()
            items = [dict(row) for row in rows]
            summary = _list_summary(request, total_count, limit)
            return {
                "intent": request.intent,
                "filters": _filters_payload(request),
                "total_count": total_count,
                "limit": limit,
                "items": items,
                "summary": summary,
                "sql": {
                    "query": list_query,
                    "params": list_params,
                    "count_query": count_query,
                    "count_params": params,
                },
            }

        raise SqlQueryError(f"Unsupported SQL intent: {request.intent}")
    finally:
        conn.close()


def _filters_payload(request: SqlRequest) -> dict[str, Any]:
    payload = asdict(request)
    payload.pop("limit", None)
    return payload


def _count_summary(request: SqlRequest, count: int) -> str:
    label = request.ifc_class or "elements"
    if request.level_like:
        return f"Found {count} {label} matching level '{request.level_like}'."
    return f"Found {count} {label}."


def _list_summary(request: SqlRequest, total: int, limit: int) -> str:
    label = request.ifc_class or "elements"
    if request.level_like:
        return (
            f"Found {total} {label} matching level '{request.level_like}', "
            f"showing {min(total, limit)}."
        )
    return f"Found {total} {label}, showing {min(total, limit)}."
