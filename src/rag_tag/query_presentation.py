from __future__ import annotations

import json
from typing import Any

_VERBOSE_MAX_LINES = 60
_LIST_DISPLAY_LIMIT = 10
_SAMPLE_DISPLAY_LIMIT = 10


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str)


def _sql_list_items(result: dict[str, Any]) -> tuple[list[str], int]:
    data = result.get("data") or {}
    if data.get("intent") != "list":
        return [], 0

    items = data.get("items")
    if not isinstance(items, list) or not items:
        return [], 0

    rows: list[str] = []
    for index, item in enumerate(items[:_LIST_DISPLAY_LIMIT], 1):
        if not isinstance(item, dict):
            rows.append(f"{index}. (no data)")
            continue

        parts: list[str] = []
        if item.get("name"):
            parts.append(f"Name: {item['name']}")
        if item.get("ifc_class"):
            parts.append(f"Class: {item['ifc_class']}")
        if item.get("level"):
            parts.append(f"Level: {item['level']}")

        rows.append(f"{index}. {' | '.join(parts)}" if parts else f"{index}. (no data)")

    total = int(data.get("total_count", len(items)))
    more_count = max(0, total - len(rows))
    return rows, more_count


def _graph_sample_items(result: dict[str, Any]) -> list[str]:
    sample = (result.get("data") or {}).get("sample")
    if not isinstance(sample, list) or not sample:
        return []
    return [str(item) for item in sample[:_SAMPLE_DISPLAY_LIMIT]]


def _details_json(result: dict[str, Any]) -> tuple[str, bool]:
    try:
        lines = json.dumps(result, indent=2, default=str).splitlines()
    except Exception:
        return "", False

    truncated = len(lines) > _VERBOSE_MAX_LINES
    if truncated:
        lines = lines[:_VERBOSE_MAX_LINES] + ["  ... (truncated)"]
    return "\n".join(lines), truncated


def build_query_presentation(result: dict[str, Any]) -> dict[str, Any]:
    """Build a viewer-friendly presentation payload from a query result."""
    sql_items, sql_more_count = _sql_list_items(result)
    details_json, details_truncated = _details_json(result)
    warning = result.get("warning")
    error = result.get("error")

    return {
        "error": str(error) if error else None,
        "answer": None if error else str(result.get("answer") or "No answer produced."),
        "warning": _stringify(warning) if warning else None,
        "sql_items": sql_items,
        "sql_more_count": sql_more_count,
        "graph_sample": _graph_sample_items(result),
        "details_json": details_json,
        "details_truncated": details_truncated,
    }
