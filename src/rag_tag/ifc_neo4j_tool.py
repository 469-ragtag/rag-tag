from __future__ import annotations

from typing import Any, Iterable

from rag_tag.graph.catalog import GraphCatalog
from rag_tag.graph_contract import EVIDENCE_LIMIT, collect_evidence
from rag_tag.ifc_class_taxonomy import expand_ifc_class_filter
from rag_tag.ifc_sql_tool import (
    _aggregate_summary,
    _count_summary,
    _field_payload,
    _filters_payload,
    _group_summary,
    _list_summary,
)
from rag_tag.level_normalization import canonicalize_level
from rag_tag.router import SqlFieldRef, SqlRequest, SqlValueFilter


def query_ifc_neo4j(
    graph: GraphCatalog,
    request: SqlRequest,
) -> dict[str, Any]:
    matched_nodes = list(_iter_matching_nodes(graph, request))

    if request.intent == "count":
        count = len(matched_nodes)
        evidence = collect_evidence(
            [_sql_item_payload(node) for _node_id, node in matched_nodes[:3]],
            source_tool="query_ifc_neo4j",
        )
        return {
            "status": "ok",
            "data": {
                "intent": request.intent,
                "filters": _filters_payload(
                    request,
                    expand_ifc_class_filter(request.ifc_class),
                ),
                "count": count,
                "evidence": evidence,
                "summary": _count_summary(request, count),
            },
            "error": None,
        }

    if request.intent == "list":
        total_count = len(matched_nodes)
        limit = request.limit or 50
        items = [_sql_item_payload(node) for _node_id, node in matched_nodes[:limit]]
        return {
            "status": "ok",
            "data": {
                "intent": request.intent,
                "filters": _filters_payload(
                    request,
                    expand_ifc_class_filter(request.ifc_class),
                ),
                "total_count": total_count,
                "limit": limit,
                "items": items,
                "evidence": collect_evidence(items, source_tool="query_ifc_neo4j"),
                "summary": _list_summary(request, total_count, limit),
                "merge_state": {
                    "items": [
                        _sql_item_payload(node)
                        for _node_id, node in matched_nodes
                    ]
                },
            },
            "error": None,
        }

    if request.intent == "aggregate":
        return _aggregate_nodes(graph, matched_nodes, request)

    if request.intent == "group":
        return _group_nodes(graph, matched_nodes, request)

    raise ValueError(f"Unsupported intent: {request.intent}")


def aggregate_catalog_elements(
    graph: GraphCatalog,
    element_ids: list[str],
    metric: str,
    field: str | None = None,
) -> dict[str, Any]:
    matched_nodes, unresolved_ids = _resolve_element_nodes(graph, element_ids)
    if not matched_nodes:
        return {
            "status": "error",
            "data": None,
            "error": {
                "message": "None of the provided element IDs could be resolved.",
                "code": "not_found",
                "details": {"unmatched_element_ids": unresolved_ids[:10]},
            },
        }

    metric_value = metric.strip().lower()
    field_source, canonical_field = _infer_metric_field_source(matched_nodes, field)
    values: list[tuple[dict[str, Any], Any]] = []
    missing_value_count = 0
    for _node_id, node_data in matched_nodes:
        if canonical_field is None:
            values.append((node_data, 1))
            continue
        resolved_value = _extract_field_value(node_data, field_source, canonical_field)
        if resolved_value is None:
            missing_value_count += 1
            continue
        values.append((node_data, resolved_value))

    aggregate_value: Any
    if metric_value == "count":
        aggregate_value = len(values)
    else:
        numeric_values = [float(value) for _node, value in values if _is_numeric(value)]
        if len(numeric_values) != len(values):
            return {
                "status": "error",
                "data": None,
                "error": {
                    "message": (
                        f"Metric '{metric_value}' requires numeric values for "
                        f"{canonical_field}."
                    ),
                    "code": "invalid",
                },
            }
        if not numeric_values:
            aggregate_value = None
        elif metric_value == "sum":
            aggregate_value = sum(numeric_values)
        elif metric_value == "avg":
            aggregate_value = sum(numeric_values) / len(numeric_values)
        elif metric_value == "min":
            aggregate_value = min(numeric_values)
        elif metric_value == "max":
            aggregate_value = max(numeric_values)
        else:
            return {
                "status": "error",
                "data": None,
                "error": {"message": "Unsupported metric", "code": "invalid"},
            }

    sample = [
        {
            **_sql_item_payload(node_data),
            **({"field_value": value} if canonical_field is not None else {}),
        }
        for node_data, value in values[:3]
    ]
    data = {
        "metric": metric_value,
        "field": canonical_field,
        "field_source": field_source,
        "aggregate_value": aggregate_value,
        "matched_element_count": len(matched_nodes),
        "unmatched_element_count": len(unresolved_ids),
        "missing_value_count": missing_value_count,
        "sample": sample,
        "evidence": collect_evidence(sample, source_tool="aggregate_elements"),
    }
    if unresolved_ids:
        data["unmatched_element_ids"] = unresolved_ids[:10]
    return {"status": "ok", "data": data, "error": None}


def group_catalog_elements(
    graph: GraphCatalog,
    element_ids: list[str],
    property_key: str,
    max_groups: int,
) -> dict[str, Any]:
    matched_nodes, unresolved_ids = _resolve_element_nodes(graph, element_ids)
    if not matched_nodes:
        return {
            "status": "error",
            "data": None,
            "error": {
                "message": "None of the provided element IDs could be resolved.",
                "code": "not_found",
                "details": {"unmatched_element_ids": unresolved_ids[:10]},
            },
        }

    field_source, canonical_field = _infer_metric_field_source(
        matched_nodes,
        property_key,
    )
    grouped: dict[str, dict[str, Any]] = {}
    missing_value_count = 0
    for _node_id, node_data in matched_nodes:
        value = _extract_field_value(node_data, field_source, canonical_field)
        if value is None:
            missing_value_count += 1
            continue
        bucket_key = repr(value)
        bucket = grouped.setdefault(
            bucket_key,
            {"value": value, "members": []},
        )
        bucket["members"].append(_sql_item_payload(node_data))

    groups = [
        {
            "group": bucket["value"],
            "count": len(bucket["members"]),
            "sample": bucket["members"][:2],
        }
        for bucket in grouped.values()
    ]
    groups.sort(key=lambda item: (-int(item["count"]), str(item["group"])))
    selected_groups = groups[:max_groups]
    return {
        "status": "ok",
        "data": {
            "property_key": canonical_field,
            "field_source": field_source,
            "groups": selected_groups,
            "matched_element_count": len(matched_nodes),
            "unmatched_element_count": len(unresolved_ids),
            "missing_value_count": missing_value_count,
            "total_groups": len(groups),
            "evidence": collect_evidence(
                (
                    sample
                    for group in selected_groups
                    for sample in (group.get("sample") or [])
                ),
                source_tool="group_elements_by_property",
            ),
        },
        "error": None,
    }


def _aggregate_nodes(
    graph: GraphCatalog,
    matched_nodes: list[tuple[str, dict[str, Any]]],
    request: SqlRequest,
) -> dict[str, Any]:
    aggregate_op = request.aggregate_op
    assert aggregate_op is not None

    values: list[tuple[dict[str, Any], Any]] = []
    missing_value_count = 0
    for _node_id, node_data in matched_nodes:
        if request.aggregate_field is None:
            values.append((node_data, 1))
            continue
        value = _extract_field_ref(node_data, request.aggregate_field)
        if value is None:
            missing_value_count += 1
            continue
        values.append((node_data, value))

    if aggregate_op == "count":
        aggregate_value = len(values)
    else:
        numeric_values = [float(value) for _node, value in values if _is_numeric(value)]
        if len(numeric_values) != len(values):
            field_label = (
                request.aggregate_field.field if request.aggregate_field else "field"
            )
            return {
                "status": "error",
                "data": None,
                "error": {
                    "message": (
                        f"Aggregate '{aggregate_op}' requires numeric values for "
                        f"{field_label}."
                    ),
                    "code": "invalid",
                },
            }
        if not numeric_values:
            aggregate_value = None
        elif aggregate_op == "sum":
            aggregate_value = sum(numeric_values)
        elif aggregate_op == "avg":
            aggregate_value = sum(numeric_values) / len(numeric_values)
        elif aggregate_op == "min":
            aggregate_value = min(numeric_values)
        elif aggregate_op == "max":
            aggregate_value = max(numeric_values)
        else:
            raise ValueError(f"Unsupported aggregate op: {aggregate_op}")

    sample = [
        _sql_item_payload(node_data)
        for node_data, _value in values[: min(EVIDENCE_LIMIT, 3)]
    ]
    return {
        "status": "ok",
        "data": {
            "intent": request.intent,
            "filters": _filters_payload(
                request,
                expand_ifc_class_filter(request.ifc_class),
            ),
            "aggregate_op": aggregate_op,
            "aggregate_field": _field_payload(request.aggregate_field),
            "aggregate_value": aggregate_value,
            "matched_value_count": len(values),
            "missing_value_count": missing_value_count,
            "total_elements": len(matched_nodes),
            "evidence": collect_evidence(sample, source_tool="query_ifc_neo4j"),
            "summary": _aggregate_summary(request, aggregate_op, aggregate_value),
            "merge_state": {
                "matched_value_count": len(values),
                "total_elements": len(matched_nodes),
                "missing_value_count": missing_value_count,
                "sum": (
                    float(aggregate_value)
                    if aggregate_op in {"sum", "avg"} and aggregate_value is not None
                    else None
                ),
            },
        },
        "error": None,
    }


def _group_nodes(
    graph: GraphCatalog,
    matched_nodes: list[tuple[str, dict[str, Any]]],
    request: SqlRequest,
) -> dict[str, Any]:
    assert request.group_by is not None
    limit = request.limit or 50
    grouped: dict[str, dict[str, Any]] = {}
    missing_value_count = 0
    for _node_id, node_data in matched_nodes:
        value = _extract_field_ref(node_data, request.group_by)
        if value is None:
            missing_value_count += 1
            continue
        bucket = grouped.setdefault(repr(value), {"group": value, "count": 0})
        bucket["count"] += 1

    groups = sorted(
        grouped.values(),
        key=lambda item: (-int(item["count"]), str(item["group"])),
    )
    selected_groups = groups[:limit]
    return {
        "status": "ok",
        "data": {
            "intent": request.intent,
            "filters": _filters_payload(
                request,
                expand_ifc_class_filter(request.ifc_class),
            ),
            "group_by": _field_payload(request.group_by),
            "groups": selected_groups,
            "matched_element_count": len(matched_nodes) - missing_value_count,
            "missing_value_count": missing_value_count,
            "total_elements": len(matched_nodes),
            "count": len(selected_groups),
            "summary": _group_summary(request, groups, limit),
            "merge_state": {"groups": groups},
        },
        "error": None,
    }


def _resolve_element_nodes(
    graph: GraphCatalog,
    element_ids: list[str],
) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    matched: list[tuple[str, dict[str, Any]]] = []
    unresolved: list[str] = []
    for element_id in element_ids:
        if element_id in graph:
            matched.append((element_id, graph.nodes[element_id]))
            continue
        found = None
        for node_id, node_data in graph.nodes(data=True):
            props = node_data.get("properties") or {}
            if props.get("GlobalId") == element_id:
                found = (node_id, node_data)
                break
        if found is None:
            unresolved.append(element_id)
            continue
        matched.append(found)
    return matched, unresolved


def _iter_matching_nodes(
    graph: GraphCatalog,
    request: SqlRequest,
) -> Iterable[tuple[str, dict[str, Any]]]:
    resolved_classes = set(expand_ifc_class_filter(request.ifc_class))
    requested_level = canonicalize_level(request.level_like)
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("node_kind") == "context":
            continue
        node_class = str(node_data.get("class_") or "")
        if resolved_classes and node_class not in resolved_classes:
            continue
        if request.predefined_type and str(
            _extract_element_field(node_data, "predefined_type") or ""
        ).lower() != request.predefined_type.lower():
            continue
        if request.type_name and str(
            _extract_element_field(node_data, "type_name") or ""
        ).lower() != request.type_name.lower():
            continue
        if requested_level is not None:
            node_level = canonicalize_level(
                str(_extract_element_field(node_data, "level") or "")
            )
            if node_level is None or requested_level not in node_level:
                continue
        if not all(
            _match_filter(node_data, flt, source="property")
            for flt in request.property_filters
        ):
            continue
        if not all(
            _match_filter(node_data, flt, source="quantity")
            for flt in request.quantity_filters
        ):
            continue
        yield node_id, node_data


def _match_filter(
    node_data: dict[str, Any],
    value_filter: SqlValueFilter,
    *,
    source: str,
) -> bool:
    actual = _extract_field_value(node_data, source, value_filter.field)
    expected = value_filter.value
    if value_filter.op == "eq":
        if isinstance(actual, list):
            return expected in actual
        return actual == expected
    if value_filter.op == "neq":
        return actual != expected
    if value_filter.op == "like":
        if actual is None:
            return False
        return str(expected).lower() in str(actual).lower()
    if actual is None:
        return False
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError):
        return False
    if value_filter.op == "lt":
        return actual_value < expected_value
    if value_filter.op == "lte":
        return actual_value <= expected_value
    if value_filter.op == "gt":
        return actual_value > expected_value
    if value_filter.op == "gte":
        return actual_value >= expected_value
    return False


def _extract_field_ref(node_data: dict[str, Any], field: SqlFieldRef) -> Any:
    return _extract_field_value(node_data, field.source, field.field)


def _extract_field_value(node_data: dict[str, Any], source: str, field: str) -> Any:
    if source == "element":
        return _extract_element_field(node_data, field)
    payload = node_data.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    if source == "property":
        return _extract_dotted_value(payload.get("PropertySets"), field)
    if source == "quantity":
        return _extract_dotted_value(payload.get("Quantities"), field)
    return None


def _extract_element_field(node_data: dict[str, Any], field: str) -> Any:
    props = node_data.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    mapping = {
        "ifc_class": node_data.get("class_"),
        "name": node_data.get("label") or props.get("Name"),
        "level": props.get("Level"),
        "type_name": props.get("TypeName"),
        "predefined_type": props.get("PredefinedType"),
        "global_id": props.get("GlobalId"),
        "express_id": props.get("ExpressId"),
    }
    return mapping.get(field)


def _extract_dotted_value(block: Any, field: str) -> Any:
    if not isinstance(block, dict):
        return None
    if field in block:
        return block.get(field)
    if "." not in field:
        for value in block.values():
            if isinstance(value, dict) and field in value:
                return value.get(field)
        return None
    head, _, tail = field.partition(".")
    current = block.get(head)
    if current is None and "Official" in block and "Custom" in block:
        for section_name in ("Official", "Custom"):
            section = block.get(section_name)
            if not isinstance(section, dict):
                continue
            current = section.get(head)
            if current is not None:
                break
    if not isinstance(current, dict):
        return None
    return current.get(tail)


def _sql_item_payload(node_data: dict[str, Any]) -> dict[str, Any]:
    props = node_data.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    return {
        "express_id": props.get("ExpressId"),
        "global_id": props.get("GlobalId"),
        "ifc_class": node_data.get("class_"),
        "name": node_data.get("label") or props.get("Name"),
        "level": props.get("Level"),
        "type_name": props.get("TypeName"),
    }


def _infer_metric_field_source(
    matched_nodes: list[tuple[str, dict[str, Any]]],
    field: str | None,
) -> tuple[str | None, str | None]:
    if field is None:
        return None, None
    element_fields = {
        "ifc_class",
        "name",
        "level",
        "type_name",
        "predefined_type",
        "global_id",
        "express_id",
    }
    if field in element_fields:
        return "element", field
    for _node_id, node_data in matched_nodes:
        if _extract_field_value(node_data, "property", field) is not None:
            return "property", field
        if _extract_field_value(node_data, "quantity", field) is not None:
            return "quantity", field
    return "property", field


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
