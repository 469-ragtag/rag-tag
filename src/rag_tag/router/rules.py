from __future__ import annotations

import re
from typing import Literal

from rag_tag.ifc_class_taxonomy import CLASS_ALIASES, normalize_ifc_class

from .capabilities import (
    IFC_SPACE_LEVEL_GRAPH_REASON,
    detect_graph_first_reason,
    is_ifc_space_level_graph_first,
)
from .models import (
    RouteDecision,
    SqlAggregateOp,
    SqlFieldRef,
    SqlFilterOp,
    SqlIntent,
    SqlRequest,
    SqlValueFilter,
    normalize_sql_field_key,
)

_COUNT_CUES = (
    "how many",
    "count",
    "number of",
    "total",
)

_EXISTENCE_CUES = (
    "are there",
    "are there any",
    "is there",
    "is there any",
    "do we have",
    "does the building have",
    "does the house have",
    "exist",
    "exists",
)

_LIST_CUES = (
    "list",
    "find",
    "show",
    "which",
    "what are",
    "display",
    "get",
    "retrieve",
    "identify",
)

_AVG_CUES = ("average", "avg", "mean")
_MIN_CUES = ("minimum", "min", "lowest", "smallest")
_MAX_CUES = ("maximum", "max", "highest", "largest")
_SUM_CUES = ("sum of", "summed", "total")
_GROUP_CUES = (
    re.compile(r"\bgroup\b.*\bby\b"),
    re.compile(r"\bbreak\s+down\b.*\bby\b"),
    re.compile(r"\bgrouped\s+by\b"),
)

_IFC_CLASS_RE = re.compile(r"\bifc[a-z0-9_]+\b", re.IGNORECASE)
_PROPERTY_CUE_RE = re.compile(
    r"\b(with|having|whose|where|without|made of|material)\b|\bthat\s+(has|have)\b"
)
_EXPLICIT_FILTER_RE = re.compile(
    r"\b(?:with|where)\s+(element|property|quantity)\s+"
    r"([A-Za-z][A-Za-z0-9_. -]*)\s*"
    r"(=|!=|>=|<=|>|<|like)\s*"
    r"([A-Za-z0-9_.%-]+(?:\s+[A-Za-z0-9_.%-]+)*)",
    re.IGNORECASE,
)
_EXPLICIT_DOTTED_FIELD_RE = re.compile(
    r"\b((?:Pset|Qto)_[A-Za-z0-9_]+\.[A-Za-z][A-Za-z0-9_ -]*)\b"
)

_LEVEL_RE = re.compile(r"\b(level|storey|story|floor)\s+([a-z0-9 _.-]+)")
_LEVEL_STOP_WORDS = re.compile(
    r"\b(with|that|which|near|adjacent|connected|having|where|and|or"
    r"|on|in|of|the|a|an"
    r"|structure|building|house|model|project|by)\b"
)
_BUILDING_CONTEXT_RE = re.compile(
    r"\b(?:in|inside|within|of)\s+the\s+(?:building|house|structure|model|project)\b"
)
_PREDEFINED_TYPE_RE = re.compile(r"\bpredefined\s+type\s+(?P<value>.+)", re.IGNORECASE)
_TYPE_NAME_RE = re.compile(r"\btype\s+name\s+(?P<value>.+)", re.IGNORECASE)
_NAMED_FILTER_BOUNDARY_RE = re.compile(
    r"(?:,|;|[.?!:])|\s+(?:"
    r"on\s+(?:level|storey|story|floor)\b|"
    r"where\b|with\b|having\b|whose\b|without\b|and\b|or\b|"
    r"near\b|adjacent\b|connected\b"
    r")",
    re.IGNORECASE,
)
_NAME_WORD_IN_NAME_RE = re.compile(
    r"\b(?:the\s+word|word)\s+[\"']?(?P<term>[A-Za-z0-9_ -]+?)[\"']?\s+"
    r"in\s+(?:their|the)\s+name\b",
    re.IGNORECASE,
)
_NAME_CONTAINS_RE = re.compile(
    r"\bname\s+(?:contains?|containing|with)\s+[\"']?(?P<term>[A-Za-z0-9_ -]+?)"
    r"[\"']?(?:\b|[.?!])",
    re.IGNORECASE,
)

_PROPERTY_FIELD_ALIASES: tuple[tuple[str, str], ...] = (
    ("thermal transmittance", "ThermalTransmittance"),
    ("fire rating", "FireRating"),
    ("u-value", "UValue"),
    ("u value", "UValue"),
    ("uvalue", "UValue"),
    ("reference", "Reference"),
    ("load bearing", "LoadBearing"),
)

_QUANTITY_FIELD_ALIASES: tuple[tuple[str, str], ...] = (
    ("gross volume", "GrossVolume"),
    ("net volume", "NetVolume"),
    ("gross area", "GrossArea"),
    ("net area", "NetArea"),
    ("perimeter", "Perimeter"),
    ("height", "Height"),
    ("width", "Width"),
    ("length", "Length"),
    ("volume", "Volume"),
    ("area", "Area"),
)

_ELEMENT_GROUP_BY_ALIASES: tuple[tuple[str, str], ...] = (
    ("predefined type", "predefined_type"),
    ("type name", "type_name"),
    ("level", "level"),
    ("storey", "level"),
    ("story", "level"),
    ("floor", "level"),
    ("class", "ifc_class"),
    ("ifc class", "ifc_class"),
    ("name", "name"),
)


def route_question_rule(question: str) -> RouteDecision:
    q = question.strip()
    q_lower = q.lower()

    graph_first_reason = detect_graph_first_reason(q)
    if graph_first_reason is not None:
        return RouteDecision("graph", graph_first_reason, None)

    measure_field = _detect_measure_field(q)
    sql_intent = _detect_sql_intent(
        q_lower, has_measure_field=measure_field is not None
    )
    if sql_intent is None:
        return RouteDecision("graph", "No SQL intent detected", None)

    predefined_type, predefined_type_span = _detect_named_filter_match(
        q, _PREDEFINED_TYPE_RE
    )
    type_name, type_name_span = _detect_named_filter_match(q, _TYPE_NAME_RE)
    (
        element_filters,
        property_filters,
        quantity_filters,
        structured_filter_spans,
    ) = _detect_structured_filters(q)
    name_filters, name_filter_spans = _detect_name_contains_filters(q)
    if name_filters:
        element_filters = (*element_filters, *name_filters)
    if (
        sql_intent in {"count", "list"}
        and _has_property_cues(q_lower)
        and predefined_type is None
        and type_name is None
        and not element_filters
        and not property_filters
        and not quantity_filters
    ):
        return RouteDecision("graph", "Property/constraint cue detected", None)

    level_like = _detect_level_like(q_lower)
    suppressed_spans = _suppressed_class_alias_spans(q_lower)
    if predefined_type_span is not None:
        suppressed_spans.append(predefined_type_span)
    if type_name_span is not None:
        suppressed_spans.append(type_name_span)
    suppressed_spans.extend(structured_filter_spans)
    suppressed_spans.extend(name_filter_spans)
    ifc_classes = _detect_ifc_classes(q, ignored_spans=suppressed_spans)
    if len(ifc_classes) > 1:
        return RouteDecision("graph", "Multiple IFC classes mentioned", None)

    ifc_class = ifc_classes[0] if ifc_classes else None
    if ifc_class is None and not _mentions_generic_elements(q_lower):
        return RouteDecision("graph", "SQL intent without class", None)

    if is_ifc_space_level_graph_first(ifc_class, level_like):
        return RouteDecision("graph", IFC_SPACE_LEVEL_GRAPH_REASON, None)

    aggregate_op: SqlAggregateOp | None = None
    aggregate_field = None
    group_by = None
    if sql_intent == "aggregate":
        aggregate_field = measure_field
        aggregate_op = _detect_aggregate_op(
            q_lower,
            has_measure_field=aggregate_field is not None,
        )
        if aggregate_op is None:
            return RouteDecision("graph", "Aggregate intent without metric", None)
        if aggregate_op != "count" and aggregate_field is None:
            return RouteDecision("graph", "Aggregate intent without field", None)
    elif sql_intent == "group":
        group_by = _detect_group_by_field(q)
        if group_by is None:
            return RouteDecision("graph", "Group intent without supported field", None)

    limit = 50 if sql_intent in {"list", "group"} else 0
    request = SqlRequest(
        intent=sql_intent,
        ifc_class=ifc_class,
        level_like=level_like,
        predefined_type=predefined_type,
        type_name=type_name,
        element_filters=element_filters,
        property_filters=property_filters,
        quantity_filters=quantity_filters,
        aggregate_op=aggregate_op,
        aggregate_field=aggregate_field,
        group_by=group_by,
        limit=limit,
    )
    return RouteDecision("sql", "Deterministic SQL intent detected", request)


def _has_property_cues(question_lower: str) -> bool:
    return _PROPERTY_CUE_RE.search(question_lower) is not None


def _detect_sql_intent(
    question_lower: str,
    *,
    has_measure_field: bool,
) -> SqlIntent | None:
    if any(pattern.search(question_lower) for pattern in _GROUP_CUES):
        return "group"
    aggregate_op = _detect_aggregate_op(
        question_lower,
        has_measure_field=has_measure_field,
    )
    if aggregate_op is not None:
        return "aggregate"
    if any(cue in question_lower for cue in _COUNT_CUES):
        return "count"
    if any(cue in question_lower for cue in _EXISTENCE_CUES):
        return "count"
    if any(cue in question_lower for cue in _LIST_CUES):
        return "list"
    return None


def _detect_aggregate_op(
    question_lower: str,
    *,
    has_measure_field: bool,
) -> SqlAggregateOp | None:
    if any(cue in question_lower for cue in _AVG_CUES):
        return "avg"
    if any(cue in question_lower for cue in _MIN_CUES):
        return "min"
    if any(cue in question_lower for cue in _MAX_CUES):
        return "max"
    if "sum of" in question_lower or "summed" in question_lower:
        return "sum"
    if has_measure_field and any(cue in question_lower for cue in _SUM_CUES):
        return "sum"
    return None


def _detect_ifc_classes(
    question: str,
    *,
    ignored_spans: list[tuple[int, int]] | None = None,
) -> list[str]:
    matches: list[tuple[int, str]] = []
    for match in _IFC_CLASS_RE.finditer(question):
        matches.append((match.start(), normalize_ifc_class(match.group(0))))

    question_lower = question.lower()
    for term, ifc_class in CLASS_ALIASES.items():
        for match in re.finditer(rf"\b{re.escape(term)}\b", question_lower):
            if _span_is_ignored(match.span(), ignored_spans):
                continue
            matches.append((match.start(), ifc_class))

    matches.sort(key=lambda item: item[0])
    ordered: list[str] = []
    for _, ifc_class in matches:
        if ifc_class not in ordered:
            ordered.append(ifc_class)
    return ordered


def _span_is_ignored(
    span: tuple[int, int], ignored_spans: list[tuple[int, int]] | None
) -> bool:
    if not ignored_spans:
        return False
    start, end = span
    return any(
        ignore_start <= start and end <= ignore_end
        for ignore_start, ignore_end in ignored_spans
    )


def _suppressed_class_alias_spans(question_lower: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []

    for pattern in (
        re.compile(r"\bground\s+floor\b"),
        re.compile(r"\bground\s+level\b"),
        re.compile(r"\bbasement(?:\s+[a-z0-9._-]+)?\b"),
        _BUILDING_CONTEXT_RE,
    ):
        spans.extend(match.span() for match in pattern.finditer(question_lower))

    match = _LEVEL_RE.search(question_lower)
    if match:
        raw = match.group(2).strip()
        trimmed = _LEVEL_STOP_WORDS.split(raw)[0].strip().rstrip(".,;:?!")
        if trimmed:
            spans.append((match.start(1), match.start(2) + len(trimmed)))

    spans.sort()
    return spans


def _detect_level_like(question_lower: str) -> str | None:
    if _BUILDING_CONTEXT_RE.search(question_lower):
        return None
    if "ground floor" in question_lower:
        return "ground floor"
    if "ground level" in question_lower:
        return "ground floor"
    if "basement" in question_lower:
        return "basement"

    match = _LEVEL_RE.search(question_lower)
    if not match:
        return None
    level_kind = match.group(1).strip().lower()
    raw = match.group(2).strip()
    raw = _LEVEL_STOP_WORDS.split(raw)[0].strip().rstrip(".,;:?!")
    if not raw:
        return None
    if level_kind in {"storey", "story", "floor"}:
        return f"level {raw}"
    return f"{level_kind} {raw}"


def _detect_measure_field(question: str) -> SqlFieldRef | None:
    explicit = _detect_explicit_field(question)
    if explicit is not None:
        return explicit
    question_lower = question.lower()

    for phrase, field in _PROPERTY_FIELD_ALIASES:
        if phrase in question_lower:
            return SqlFieldRef(source="property", field=field)
    for phrase, field in _QUANTITY_FIELD_ALIASES:
        if phrase in question_lower:
            return SqlFieldRef(source="quantity", field=field)
    return None


def _detect_group_by_field(question: str) -> SqlFieldRef | None:
    question_lower = question.lower()
    by_idx = question_lower.rfind(" by ")
    tail = question if by_idx < 0 else question[by_idx + 4 :]

    explicit = _detect_explicit_field(tail)
    if explicit is not None:
        return explicit

    tail_lower = tail.lower()
    for phrase, field in _PROPERTY_FIELD_ALIASES:
        if phrase in tail_lower:
            return SqlFieldRef(source="property", field=field)
    for phrase, field in _QUANTITY_FIELD_ALIASES:
        if phrase in tail_lower:
            return SqlFieldRef(source="quantity", field=field)
    for phrase, field in _ELEMENT_GROUP_BY_ALIASES:
        if phrase in tail_lower:
            return SqlFieldRef(source="element", field=field)
    return None


def _detect_explicit_field(question: str) -> SqlFieldRef | None:
    match = _EXPLICIT_DOTTED_FIELD_RE.search(question)
    if not match:
        return None
    raw_field = match.group(1)
    prefix = raw_field.split(".", 1)[0]
    source = "quantity" if prefix.lower().startswith("qto_") else "property"
    return SqlFieldRef(source=source, field=raw_field)


def _detect_structured_filters(
    question: str,
) -> tuple[
    tuple[SqlValueFilter, ...],
    tuple[SqlValueFilter, ...],
    tuple[SqlValueFilter, ...],
    list[tuple[int, int]],
]:
    element_filters: list[SqlValueFilter] = []
    property_filters: list[SqlValueFilter] = []
    quantity_filters: list[SqlValueFilter] = []
    spans: list[tuple[int, int]] = []
    for match in _EXPLICIT_FILTER_RE.finditer(question):
        scope_raw = match.group(1).strip().lower()
        scope: Literal["element", "property", "quantity"]
        if scope_raw == "element":
            scope = "element"
        elif scope_raw == "property":
            scope = "property"
        else:
            scope = "quantity"
        raw_field = match.group(2)
        if scope == "element":
            field = _normalize_element_filter_field(raw_field)
        else:
            field = normalize_sql_field_key(raw_field)
        op = _normalize_filter_op(match.group(3))
        raw_value = _trim_clause_value(match.group(4))
        if raw_value is None:
            continue
        value = _normalize_filter_value(raw_value)
        filter_item = SqlValueFilter(source=scope, field=field, op=op, value=value)
        if scope == "element":
            element_filters.append(filter_item)
        elif scope == "property":
            property_filters.append(filter_item)
        else:
            quantity_filters.append(filter_item)
        spans.append(match.span())
    return (
        tuple(element_filters),
        tuple(property_filters),
        tuple(quantity_filters),
        spans,
    )


def _detect_name_contains_filters(
    question: str,
) -> tuple[tuple[SqlValueFilter, ...], list[tuple[int, int]]]:
    filters: list[SqlValueFilter] = []
    spans: list[tuple[int, int]] = []

    for pattern in (_NAME_WORD_IN_NAME_RE, _NAME_CONTAINS_RE):
        for match in pattern.finditer(question):
            term = match.group("term").strip().strip("\"'")
            term = re.sub(r"\s+", " ", term).strip()
            if not term:
                continue
            filters.append(
                SqlValueFilter(
                    source="element",
                    field="name",
                    op="like",
                    value=f"%{term}%",
                )
            )
            spans.append(match.span("term"))

    return tuple(filters), spans


def _normalize_element_filter_field(raw_field: str) -> str:
    normalized = normalize_sql_field_key(raw_field)
    lowered = normalized.lower()
    if lowered in {"name", "ifc_class", "level", "level_key", "type_name"}:
        return lowered
    if lowered == "predefined_type":
        return "predefined_type"
    if lowered in {"ifcclass", "class"}:
        return "ifc_class"
    if lowered in {"typename", "type"}:
        return "type_name"
    if lowered in {"predefinedtype", "predefined"}:
        return "predefined_type"
    if lowered == "levelkey":
        return "level_key"
    if lowered == "globalid":
        return "global_id"
    if lowered == "expressid":
        return "express_id"
    return normalized


def _normalize_filter_op(value: str) -> SqlFilterOp:
    lowered = value.lower()
    if lowered == "=":
        return "eq"
    if lowered == "!=":
        return "neq"
    if lowered == ">":
        return "gt"
    if lowered == ">=":
        return "gte"
    if lowered == "<":
        return "lt"
    if lowered == "<=":
        return "lte"
    return "like"


def _normalize_filter_value(value: str) -> str | int | float | bool:
    cleaned = value.strip()
    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def _trim_clause_value(raw_value: str) -> str | None:
    leading_trimmed = raw_value.lstrip()
    boundary_match = _NAMED_FILTER_BOUNDARY_RE.search(leading_trimmed)
    candidate = (
        leading_trimmed[: boundary_match.start()]
        if boundary_match is not None
        else leading_trimmed
    )
    cleaned = candidate.rstrip(" ,.;:?!")
    return cleaned or None


def _detect_named_filter(question: str, pattern: re.Pattern[str]) -> str | None:
    value, _span = _detect_named_filter_match(question, pattern)
    return value


def _detect_named_filter_match(
    question: str,
    pattern: re.Pattern[str],
) -> tuple[str | None, tuple[int, int] | None]:
    match = pattern.search(question)
    if not match:
        return None, None

    raw_value = match.group("value")
    cleaned = _trim_clause_value(raw_value)
    if not cleaned:
        return None, None

    leading_trimmed = raw_value.lstrip()
    leading_offset = len(raw_value) - len(leading_trimmed)
    value_start = match.start("value") + leading_offset
    value_end = value_start + len(cleaned)
    return cleaned, (value_start, value_end)


def _mentions_generic_elements(question_lower: str) -> bool:
    terms = ("element", "elements", "components")
    return any(term in question_lower for term in terms)
