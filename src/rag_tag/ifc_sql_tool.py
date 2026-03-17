from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_tag.graph_contract import EVIDENCE_LIMIT, collect_evidence
from rag_tag.ifc_class_taxonomy import expand_ifc_class_filter
from rag_tag.level_normalization import canonicalize_level
from rag_tag.router import SqlRequest


class SqlQueryError(RuntimeError):
    """Raised when SQL execution or configuration fails."""


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    for row in rows:
        if isinstance(row, sqlite3.Row):
            if str(row["name"]).lower() == column.lower():
                return True
            continue
        if len(row) > 1 and str(row[1]).lower() == column.lower():
            return True
    return False


def query_ifc_sql(db_path: Path, request: SqlRequest) -> dict[str, Any]:
    if not db_path.exists():
        raise SqlQueryError(f"SQLite database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        where_clauses: list[str] = []
        params: list[object] = []
        resolved_ifc_classes = expand_ifc_class_filter(request.ifc_class)
        resolved_level = canonicalize_level(request.level_like)
        has_level_key = _has_column(conn, "elements", "level_key")

        if resolved_ifc_classes:
            placeholders = ", ".join("?" for _ in resolved_ifc_classes)
            where_clauses.append(f"ifc_class IN ({placeholders})")
            params.extend(resolved_ifc_classes)
        if resolved_level:
            if not has_level_key:
                raise SqlQueryError(
                    "Database schema missing 'level_key' column. "
                    "Rebuild SQLite files with: uv run rag-tag-jsonl-to-sql"
                )
            where_clauses.append("level_key = ?")
            params.append(resolved_level)

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        if request.intent == "count":
            query = f"SELECT COUNT(*) AS count FROM elements{where_sql}"
            row = conn.execute(query, params).fetchone()
            count = int(row["count"]) if row else 0
            sample_limit = min(EVIDENCE_LIMIT, 3)
            sample_query = (
                "SELECT express_id, global_id, ifc_class, name, level, type_name "
                f"FROM elements{where_sql} ORDER BY name LIMIT ?"
            )
            sample_params = [*params, sample_limit]
            sample_rows = conn.execute(sample_query, sample_params).fetchall()
            evidence = collect_evidence(
                [_sql_item_payload(sample_row) for sample_row in sample_rows],
                source_tool="query_ifc_sql",
                match_reason_builder=lambda _item: "representative_match",
            )
            summary = _count_summary(request, count)
            return {
                "status": "ok",
                "data": {
                    "intent": request.intent,
                    "filters": _filters_payload(request, resolved_ifc_classes),
                    "count": count,
                    "evidence": evidence,
                    "summary": summary,
                    "sql": {
                        "query": query,
                        "params": params,
                        "sample_query": sample_query,
                        "sample_params": sample_params,
                    },
                },
                "error": None,
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
            items = [_sql_item_payload(row) for row in rows]
            evidence = collect_evidence(items, source_tool="query_ifc_sql")
            summary = _list_summary(request, total_count, limit)
            return {
                "status": "ok",
                "data": {
                    "intent": request.intent,
                    "filters": _filters_payload(request, resolved_ifc_classes),
                    "total_count": total_count,
                    "limit": limit,
                    "items": items,
                    "evidence": evidence,
                    "summary": summary,
                    "sql": {
                        "query": list_query,
                        "params": list_params,
                        "count_query": count_query,
                        "count_params": params,
                    },
                },
                "error": None,
            }

        raise SqlQueryError(f"Unsupported SQL intent: {request.intent}")
    except sqlite3.Error as exc:
        raise SqlQueryError(f"SQL execution failed for {db_path.name}: {exc}") from exc
    finally:
        conn.close()


def _filters_payload(
    request: SqlRequest, resolved_ifc_classes: tuple[str, ...]
) -> dict[str, Any]:
    payload = asdict(request)
    payload.pop("limit", None)
    if resolved_ifc_classes:
        payload["resolved_ifc_classes"] = list(resolved_ifc_classes)
    resolved_level = canonicalize_level(request.level_like)
    if resolved_level:
        payload["resolved_level"] = resolved_level
    return payload


def _sql_item_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "express_id": row["express_id"],
        "global_id": row["global_id"],
        "ifc_class": row["ifc_class"],
        "name": row["name"],
        "level": row["level"],
        "type_name": row["type_name"],
    }


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
