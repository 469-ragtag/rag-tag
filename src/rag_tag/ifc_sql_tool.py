from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_tag.graph_contract import EVIDENCE_LIMIT, collect_evidence
from rag_tag.ifc_class_taxonomy import expand_ifc_class_filter
from rag_tag.level_normalization import canonicalize_level
from rag_tag.router import SqlFieldRef, SqlRequest, SqlValueFilter
from rag_tag.sql_element_lookup import decode_db_value

_ELEMENT_GROUP_FIELDS: dict[str, str] = {
    "ifc_class": "e.ifc_class",
    "level": "e.level",
    "predefined_type": "e.predefined_type",
    "type_name": "e.type_name",
    "name": "e.name",
}
_ELEMENT_FILTER_FIELDS: dict[str, str] = {
    "express_id": "e.express_id",
    "global_id": "e.global_id",
    "ifc_class": "e.ifc_class",
    "level": "e.level",
    "level_key": "e.level_key",
    "predefined_type": "e.predefined_type",
    "type_name": "e.type_name",
    "name": "e.name",
}
_TYPED_JSON_PREFIX = "json:"
_NUMERIC_AGGREGATE_OPS = frozenset({"sum", "avg", "min", "max"})


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
        scope = _compile_element_scope(conn, request)

        if request.intent == "count":
            return _execute_count(conn, request, scope)
        if request.intent == "list":
            return _execute_list(conn, request, scope)
        if request.intent == "aggregate":
            return _execute_aggregate(conn, request, scope)
        if request.intent == "group":
            return _execute_group(conn, request, scope)

        raise SqlQueryError(f"Unsupported SQL intent: {request.intent}")
    except sqlite3.Error as exc:
        raise SqlQueryError(f"SQL execution failed for {db_path.name}: {exc}") from exc
    finally:
        conn.close()


def _execute_count(
    conn: sqlite3.Connection,
    request: SqlRequest,
    scope: dict[str, Any],
) -> dict[str, Any]:
    query = (
        "SELECT COUNT(DISTINCT e.express_id) AS count "
        f"FROM elements e{scope['joins']}{scope['where_sql']}"
    )
    params = [*scope["join_params"], *scope["where_params"]]
    row = conn.execute(query, params).fetchone()
    count = int(row["count"]) if row else 0

    sample_limit = min(EVIDENCE_LIMIT, 3)
    sample_query = (
        "SELECT e.express_id, e.global_id, e.ifc_class, e.name, e.level, e.type_name "
        f"FROM elements e{scope['joins']}{scope['where_sql']} "
        "GROUP BY e.express_id, e.global_id, e.ifc_class, e.name, e.level, e.type_name "
        "ORDER BY e.name LIMIT ?"
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
            "filters": _filters_payload(request, scope["resolved_ifc_classes"]),
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


def _execute_list(
    conn: sqlite3.Connection,
    request: SqlRequest,
    scope: dict[str, Any],
) -> dict[str, Any]:
    count_query = (
        "SELECT COUNT(DISTINCT e.express_id) AS count "
        f"FROM elements e{scope['joins']}{scope['where_sql']}"
    )
    count_params = [*scope["join_params"], *scope["where_params"]]
    count_row = conn.execute(count_query, count_params).fetchone()
    total_count = int(count_row["count"]) if count_row else 0

    limit = request.limit or 50
    list_query = (
        "SELECT e.express_id, e.global_id, e.ifc_class, e.name, e.level, e.type_name "
        f"FROM elements e{scope['joins']}{scope['where_sql']} "
        "GROUP BY e.express_id, e.global_id, e.ifc_class, e.name, e.level, e.type_name "
        "ORDER BY e.name"
    )
    list_params = list(count_params)
    rows = conn.execute(list_query, list_params).fetchall()
    full_items = [_sql_item_payload(row) for row in rows]
    items = full_items[:limit]
    evidence = collect_evidence(items, source_tool="query_ifc_sql")
    summary = _list_summary(request, total_count, limit)
    return {
        "status": "ok",
        "data": {
            "intent": request.intent,
            "filters": _filters_payload(request, scope["resolved_ifc_classes"]),
            "total_count": total_count,
            "limit": limit,
            "items": items,
            "evidence": evidence,
            "summary": summary,
            "merge_state": {"items": full_items},
            "sql": {
                "query": list_query,
                "params": list_params,
                "count_query": count_query,
                "count_params": count_params,
            },
        },
        "error": None,
    }


def _execute_aggregate(
    conn: sqlite3.Connection,
    request: SqlRequest,
    scope: dict[str, Any],
) -> dict[str, Any]:
    total_elements = _count_filtered_elements(conn, scope)
    aggregate_op = request.aggregate_op
    if aggregate_op is None:
        raise SqlQueryError("Aggregate SQL request missing aggregate_op")
    if (
        aggregate_op != "count"
        and request.aggregate_field is not None
        and request.aggregate_field.source == "element"
    ):
        raise SqlQueryError(
            "Non-count aggregates do not support element fields; "
            "use property or quantity fields."
        )

    aggregate_join_sql = ""
    aggregate_join_params: list[object] = []
    aggregate_value_expr = None
    aggregate_count_expr = "COUNT(DISTINCT e.express_id)"
    compiled_field: dict[str, Any] | None = None

    if request.aggregate_field is not None:
        compiled_field = _compile_field_join("aggf", request.aggregate_field)
        aggregate_join_sql = compiled_field["join_sql"]
        aggregate_join_params = compiled_field["params"]
        aggregate_value_expr = compiled_field["value_expr"]
        aggregate_count_expr = f"COUNT({aggregate_value_expr})"

    if (
        aggregate_op in _NUMERIC_AGGREGATE_OPS
        and request.aggregate_field is not None
        and request.aggregate_field.source == "property"
        and compiled_field is not None
    ):
        _validate_numeric_property_aggregate(
            conn,
            request,
            scope,
            compiled_field,
            aggregate_join_sql,
            aggregate_join_params,
        )

    if aggregate_op == "count":
        query = (
            f"SELECT {aggregate_count_expr} AS aggregate_value "
            f"FROM elements e{scope['joins']}{aggregate_join_sql}{scope['where_sql']}"
        )
    else:
        if aggregate_value_expr is None:
            raise SqlQueryError(
                "Non-count aggregate SQL requests require an aggregate field"
            )
        query = (
            f"SELECT {aggregate_op.upper()}({aggregate_value_expr}) "
            "AS aggregate_value, "
            f"COUNT({aggregate_value_expr}) AS matched_value_count "
            f"FROM elements e{scope['joins']}{aggregate_join_sql}{scope['where_sql']}"
        )

    params = [
        *scope["join_params"],
        *aggregate_join_params,
        *scope["where_params"],
    ]
    row = conn.execute(query, params).fetchone()

    aggregate_value = row["aggregate_value"] if row else None
    if aggregate_value is not None and aggregate_op in _NUMERIC_AGGREGATE_OPS:
        aggregate_value = float(aggregate_value)

    if request.aggregate_field is None:
        matched_value_count = total_elements
        missing_value_count = 0
    elif aggregate_op == "count":
        matched_value_count = int(aggregate_value or 0)
        missing_value_count = max(total_elements - matched_value_count, 0)
    else:
        matched_value_count = int(row["matched_value_count"] or 0) if row else 0
        missing_value_count = max(total_elements - matched_value_count, 0)

    sample_query = (
        "SELECT e.express_id, e.global_id, e.ifc_class, e.name, e.level, e.type_name "
        f"FROM elements e{scope['joins']}{aggregate_join_sql}{scope['where_sql']}"
    )
    if aggregate_value_expr is not None:
        sample_query += f" AND {aggregate_value_expr} IS NOT NULL"
    sample_query += " ORDER BY e.name LIMIT ?"
    sample_params = [*params, min(EVIDENCE_LIMIT, 3)]
    sample_rows = conn.execute(sample_query, sample_params).fetchall()
    evidence = collect_evidence(
        [_sql_item_payload(sample_row) for sample_row in sample_rows],
        source_tool="query_ifc_sql",
        match_reason_builder=lambda _item: "aggregate_match",
    )

    summary = _aggregate_summary(request, aggregate_op, aggregate_value)
    return {
        "status": "ok",
        "data": {
            "intent": request.intent,
            "filters": _filters_payload(request, scope["resolved_ifc_classes"]),
            "aggregate_op": aggregate_op,
            "aggregate_field": _field_payload(request.aggregate_field),
            "aggregate_value": aggregate_value,
            "matched_value_count": matched_value_count,
            "missing_value_count": missing_value_count,
            "total_elements": total_elements,
            "evidence": evidence,
            "summary": summary,
            "merge_state": {
                "matched_value_count": matched_value_count,
                "total_elements": total_elements,
                "missing_value_count": missing_value_count,
                "sum": (
                    float(aggregate_value)
                    if aggregate_op in {"sum", "avg"} and aggregate_value is not None
                    else None
                ),
                "min": float(aggregate_value)
                if aggregate_op == "min" and aggregate_value is not None
                else None,
                "max": float(aggregate_value)
                if aggregate_op == "max" and aggregate_value is not None
                else None,
            },
            "sql": {
                "query": query,
                "params": params,
                "sample_query": sample_query,
                "sample_params": sample_params,
            },
        },
        "error": None,
    }


def _execute_group(
    conn: sqlite3.Connection,
    request: SqlRequest,
    scope: dict[str, Any],
) -> dict[str, Any]:
    if request.group_by is None:
        raise SqlQueryError("Group SQL request missing group_by field")

    total_elements = _count_filtered_elements(conn, scope)
    compiled_field = _compile_field_join("groupf", request.group_by)
    group_value_expr = compiled_field["select_expr"]
    group_presence_expr = compiled_field["presence_expr"]
    limit = request.limit or 50

    query = (
        f"SELECT {group_value_expr} AS group_value, "
        "COUNT(DISTINCT e.express_id) AS count "
        "FROM elements e"
        f"{scope['joins']}{compiled_field['join_sql']}{scope['where_sql']} "
        f"AND {group_presence_expr} "
        "GROUP BY group_value ORDER BY count DESC, group_value"
    )
    params = [
        *scope["join_params"],
        *compiled_field["params"],
        *scope["where_params"],
    ]
    rows = conn.execute(query, params).fetchall()
    all_groups = [
        {
            "group": row["group_value"],
            "count": int(row["count"]),
        }
        for row in rows
    ]
    groups = all_groups[:limit]
    matched_count_query = (
        "SELECT COUNT(DISTINCT e.express_id) AS matched_element_count "
        "FROM elements e"
        f"{scope['joins']}{compiled_field['join_sql']}{scope['where_sql']} "
        f"AND {group_presence_expr}"
    )
    matched_count_params = [
        *scope["join_params"],
        *compiled_field["params"],
        *scope["where_params"],
    ]
    matched_count_row = conn.execute(
        matched_count_query, matched_count_params
    ).fetchone()
    matched_element_count = (
        int(matched_count_row["matched_element_count"]) if matched_count_row else 0
    )
    missing_value_count = max(total_elements - matched_element_count, 0)

    sample_query = (
        "SELECT e.express_id, e.global_id, e.ifc_class, e.name, e.level, e.type_name "
        "FROM elements e"
        f"{scope['joins']}{compiled_field['join_sql']}{scope['where_sql']} "
        f"AND {group_presence_expr} ORDER BY e.name LIMIT ?"
    )
    sample_params = [
        *scope["join_params"],
        *compiled_field["params"],
        *scope["where_params"],
        min(EVIDENCE_LIMIT, 3),
    ]
    sample_rows = conn.execute(sample_query, sample_params).fetchall()
    evidence = collect_evidence(
        [_sql_item_payload(sample_row) for sample_row in sample_rows],
        source_tool="query_ifc_sql",
        match_reason_builder=lambda _item: "group_match",
    )

    summary = _group_summary(request, groups, limit)
    return {
        "status": "ok",
        "data": {
            "intent": request.intent,
            "filters": _filters_payload(request, scope["resolved_ifc_classes"]),
            "group_by": _field_payload(request.group_by),
            "groups": groups,
            "matched_element_count": matched_element_count,
            "missing_value_count": missing_value_count,
            "total_elements": total_elements,
            "limit": limit,
            "evidence": evidence,
            "summary": summary,
            "merge_state": {"groups": all_groups},
            "sql": {
                "query": query,
                "params": params,
                "matched_count_query": matched_count_query,
                "matched_count_params": matched_count_params,
                "sample_query": sample_query,
                "sample_params": sample_params,
            },
        },
        "error": None,
    }


def _compile_element_scope(
    conn: sqlite3.Connection,
    request: SqlRequest,
) -> dict[str, Any]:
    where_clauses = ["1=1"]
    where_params: list[object] = []
    join_sqls: list[str] = []
    join_params: list[object] = []

    resolved_ifc_classes = expand_ifc_class_filter(request.ifc_class)
    resolved_level = canonicalize_level(request.level_like)
    has_level_key = _has_column(conn, "elements", "level_key")

    if resolved_ifc_classes:
        placeholders = ", ".join("?" for _ in resolved_ifc_classes)
        where_clauses.append(f"e.ifc_class IN ({placeholders})")
        where_params.extend(resolved_ifc_classes)
    if resolved_level:
        if not has_level_key:
            raise SqlQueryError(
                "Database schema missing 'level_key' column. "
                "Rebuild SQLite files with: uv run rag-tag-jsonl-to-sql"
            )
        where_clauses.append("e.level_key = ?")
        where_params.append(resolved_level)
    if request.predefined_type:
        where_clauses.append("COALESCE(e.predefined_type, '') = ? COLLATE NOCASE")
        where_params.append(request.predefined_type)
    if request.type_name:
        where_clauses.append("COALESCE(e.type_name, '') = ? COLLATE NOCASE")
        where_params.append(request.type_name)

    for filter_item in request.element_filters:
        clause, params = _compile_element_filter(filter_item)
        where_clauses.append(clause)
        where_params.extend(params)

    if _should_exclude_type_rows_for_name_search(request):
        where_clauses.append(
            "e.ifc_class != 'IfcTypeObject' AND e.ifc_class NOT LIKE '%Type'"
        )

    for index, filter_item in enumerate(request.property_filters, start=1):
        clause, params = _compile_property_filter(filter_item, alias=f"pf{index}")
        where_clauses.append(clause)
        where_params.extend(params)
    for index, filter_item in enumerate(request.quantity_filters, start=1):
        clause, params = _compile_quantity_filter(filter_item, alias=f"qf{index}")
        where_clauses.append(clause)
        where_params.extend(params)

    return {
        "joins": "".join(join_sqls),
        "join_params": join_params,
        "where_sql": " WHERE " + " AND ".join(where_clauses),
        "where_params": where_params,
        "resolved_ifc_classes": resolved_ifc_classes,
    }


def _compile_element_filter(filter_item: SqlValueFilter) -> tuple[str, list[object]]:
    column = _ELEMENT_FILTER_FIELDS.get(filter_item.field)
    if column is None:
        allowed = ", ".join(sorted(_ELEMENT_FILTER_FIELDS))
        raise SqlQueryError(
            f"Unsupported element filter field. Allowed fields: {allowed}."
        )
    compare_sql, compare_params = _compile_compare_clause(
        column,
        filter_item.op,
        filter_item.value,
        prefer_numeric=_is_numeric_element_field(filter_item.field),
    )
    return compare_sql, compare_params


def _is_numeric_element_field(field: str) -> bool:
    return field == "express_id"


def _should_exclude_type_rows_for_name_search(request: SqlRequest) -> bool:
    if request.ifc_class is not None:
        return False
    return any(filter_item.field == "name" for filter_item in request.element_filters)


def _compile_property_filter(
    filter_item: SqlValueFilter,
    *,
    alias: str,
) -> tuple[str, list[object]]:
    key_sql, key_params = _compile_property_key_conditions(filter_item.field, alias)
    value_expr = _property_value_expr(alias)
    compare_sql, compare_params = _compile_compare_clause(
        value_expr,
        filter_item.op,
        filter_item.value,
        prefer_numeric=isinstance(filter_item.value, int | float)
        and not isinstance(filter_item.value, bool),
    )
    clause = (
        "EXISTS (SELECT 1 FROM properties "
        f"{alias} WHERE {alias}.element_id = e.express_id "
        f"AND {key_sql} AND {compare_sql})"
    )
    return clause, [*key_params, *compare_params]


def _compile_quantity_filter(
    filter_item: SqlValueFilter,
    *,
    alias: str,
) -> tuple[str, list[object]]:
    key_sql, key_params = _compile_quantity_key_conditions(filter_item.field, alias)
    compare_sql, compare_params = _compile_compare_clause(
        f"{alias}.value",
        filter_item.op,
        filter_item.value,
        prefer_numeric=True,
    )
    clause = (
        "EXISTS (SELECT 1 FROM quantities "
        f"{alias} WHERE {alias}.element_id = e.express_id "
        f"AND {key_sql} AND {compare_sql})"
    )
    return clause, [*key_params, *compare_params]


def _compile_field_join(alias: str, field: SqlFieldRef) -> dict[str, Any]:
    if field.source == "element":
        column = _ELEMENT_GROUP_FIELDS.get(field.field)
        if column is None:
            raise SqlQueryError(
                "Unsupported element field. Allowed fields: ifc_class, level, "
                "predefined_type, type_name, name."
            )
        presence_expr = f"{column} IS NOT NULL AND TRIM(CAST({column} AS TEXT)) <> ''"
        return {
            "join_sql": "",
            "params": [],
            "select_expr": column,
            "value_expr": column,
            "presence_expr": presence_expr,
        }

    if field.source == "property":
        key_sql, key_params = _compile_property_key_conditions(field.field, "src")
        join_sql = (
            " LEFT JOIN (SELECT element_id, src.value AS raw_value, "
            f"{_property_value_expr('src')} AS field_value, "
            "ROW_NUMBER() OVER (PARTITION BY element_id "
            "ORDER BY is_official DESC, id ASC) AS rn "
            f"FROM properties src WHERE {key_sql}) {alias} "
            f"ON {alias}.element_id = e.express_id AND {alias}.rn = 1"
        )
        return {
            "join_sql": join_sql,
            "params": key_params,
            "select_expr": f"{alias}.field_value",
            "value_expr": f"CAST({alias}.field_value AS REAL)",
            "presence_expr": f"{alias}.field_value IS NOT NULL",
            "raw_value_expr": f"{alias}.raw_value",
        }

    key_sql, key_params = _compile_quantity_key_conditions(field.field, "src")
    join_sql = (
        " LEFT JOIN (SELECT element_id, value AS field_value, "
        "ROW_NUMBER() OVER (PARTITION BY element_id "
        "ORDER BY is_official DESC, id ASC) AS rn "
        f"FROM quantities src WHERE {key_sql}) {alias} "
        f"ON {alias}.element_id = e.express_id AND {alias}.rn = 1"
    )
    return {
        "join_sql": join_sql,
        "params": key_params,
        "select_expr": f"{alias}.field_value",
        "value_expr": f"CAST({alias}.field_value AS REAL)",
        "presence_expr": f"{alias}.field_value IS NOT NULL",
        "raw_value_expr": None,
    }


def _validate_numeric_property_aggregate(
    conn: sqlite3.Connection,
    request: SqlRequest,
    scope: dict[str, Any],
    compiled_field: dict[str, Any],
    aggregate_join_sql: str,
    aggregate_join_params: list[object],
) -> None:
    raw_value_expr = compiled_field.get("raw_value_expr")
    if not raw_value_expr:
        return

    query = (
        f"SELECT {raw_value_expr} AS raw_value "
        f"FROM elements e{scope['joins']}{aggregate_join_sql}{scope['where_sql']} "
        f"AND {compiled_field['presence_expr']}"
    )
    params = [
        *scope["join_params"],
        *aggregate_join_params,
        *scope["where_params"],
    ]
    for row in conn.execute(query, params):
        if _is_numeric_aggregate_value(decode_db_value(row["raw_value"])):
            continue
        field_name = (
            request.aggregate_field.field if request.aggregate_field else "field"
        )
        raise SqlQueryError(
            f"Aggregate field '{field_name}' is not numeric; "
            f"{request.aggregate_op} only supports numeric property or quantity fields."
        )


def _is_numeric_aggregate_value(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _compile_property_key_conditions(
    field: str, alias: str
) -> tuple[str, list[object]]:
    if "." in field:
        pset_name, property_name = field.split(".", 1)
        return (
            f"{alias}.pset_name = ? AND {alias}.property_name = ?",
            [pset_name, property_name],
        )
    return f"{alias}.property_name = ?", [field]


def _compile_quantity_key_conditions(
    field: str, alias: str
) -> tuple[str, list[object]]:
    if "." in field:
        qto_name, quantity_name = field.split(".", 1)
        return (
            f"{alias}.qto_name = ? AND {alias}.quantity_name = ?",
            [qto_name, quantity_name],
        )
    return f"{alias}.quantity_name = ?", [field]


def _property_value_expr(alias: str) -> str:
    return (
        "CASE "
        f"WHEN {alias}.value IS NULL THEN NULL "
        f"WHEN substr({alias}.value, 1, {len(_TYPED_JSON_PREFIX)}) = "
        f"'{_TYPED_JSON_PREFIX}' "
        f"THEN json_extract(substr({alias}.value, {len(_TYPED_JSON_PREFIX) + 1}), '$') "
        f"ELSE {alias}.value END"
    )


def _compile_compare_clause(
    expr: str,
    op: str,
    value: str | int | float | bool,
    *,
    prefer_numeric: bool,
) -> tuple[str, list[object]]:
    sql_op = {
        "eq": "=",
        "neq": "!=",
        "lt": "<",
        "lte": "<=",
        "gt": ">",
        "gte": ">=",
        "like": "LIKE",
    }[op]
    if isinstance(value, bool):
        normalized_value: object = int(value)
    else:
        normalized_value = value
    if op == "like":
        return f"CAST({expr} AS TEXT) {sql_op} ?", [str(normalized_value)]
    if prefer_numeric and isinstance(normalized_value, int | float):
        return f"CAST({expr} AS REAL) {sql_op} ?", [normalized_value]
    return f"{expr} {sql_op} ?", [normalized_value]


def _count_filtered_elements(
    conn: sqlite3.Connection,
    scope: dict[str, Any],
) -> int:
    query = (
        "SELECT COUNT(DISTINCT e.express_id) AS count "
        f"FROM elements e{scope['joins']}{scope['where_sql']}"
    )
    params = [*scope["join_params"], *scope["where_params"]]
    row = conn.execute(query, params).fetchone()
    return int(row["count"]) if row else 0


def _filters_payload(
    request: SqlRequest,
    resolved_ifc_classes: tuple[str, ...],
) -> dict[str, Any]:
    payload = asdict(request)
    payload.pop("limit", None)
    if resolved_ifc_classes:
        payload["resolved_ifc_classes"] = list(resolved_ifc_classes)
    resolved_level = canonicalize_level(request.level_like)
    if resolved_level:
        payload["resolved_level"] = resolved_level
    return payload


def _field_payload(field: SqlFieldRef | None) -> dict[str, Any] | None:
    if field is None:
        return None
    return {"source": field.source, "field": field.field}


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


def _aggregate_summary(
    request: SqlRequest,
    aggregate_op: str,
    aggregate_value: Any,
) -> str:
    label = request.ifc_class or "elements"
    field_label = request.aggregate_field.field if request.aggregate_field else label
    if aggregate_op == "count" and request.aggregate_field is None:
        return _count_summary(request, int(aggregate_value or 0))
    if request.level_like:
        return (
            f"Computed {aggregate_op} of {field_label} for {label} matching "
            f"level '{request.level_like}': {aggregate_value}."
        )
    return f"Computed {aggregate_op} of {field_label} for {label}: {aggregate_value}."


def _group_summary(
    request: SqlRequest,
    groups: list[dict[str, Any]],
    limit: int,
) -> str:
    label = request.ifc_class or "elements"
    field_label = request.group_by.field if request.group_by else "field"
    shown = min(len(groups), limit)
    if request.level_like:
        return (
            f"Grouped {label} matching level '{request.level_like}' by {field_label}, "
            f"showing {shown} groups."
        )
    return f"Grouped {label} by {field_label}, showing {shown} groups."
