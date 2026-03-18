from __future__ import annotations

from pydantic_ai import Agent

from rag_tag.llm.pydantic_ai import get_router_model, get_router_model_settings

from .capabilities import (
    IFC_SPACE_LEVEL_GRAPH_REASON,
    build_capability_matrix_prompt_block,
    detect_graph_first_reason,
    is_ifc_space_level_graph_first,
)
from .llm_models import LlmRouteResponse
from .models import RouteDecision, SqlFieldRef, SqlRequest, SqlValueFilter

_ELEMENT_GROUP_SQL_FIELDS = frozenset(
    {"ifc_class", "level", "predefined_type", "type_name", "name"}
)
_ELEMENT_FIELD_ALIASES = {
    "ifcclass": "ifc_class",
    "class": "ifc_class",
    "predefinedtype": "predefined_type",
    "predefined": "predefined_type",
    "typename": "type_name",
    "type": "type_name",
}


class LlmRouterError(RuntimeError):
    """Raised when LLM routing fails or is misconfigured."""


def route_with_llm(question: str, *, debug_llm_io: bool = False) -> RouteDecision:
    """Route a question using an LLM with structured output.

    Args:
        question: The user's question to route
        debug_llm_io: If True, enable debug output (not implemented in PydanticAI)

    Returns:
        RouteDecision with route type, reason, and optional SQL request

    Raises:
        LlmRouterError: If routing fails or returns invalid results
    """
    # Get the router model from environment (default: google:gemini-2.5-flash)
    try:
        model = get_router_model()
        model_settings = get_router_model_settings()
    except Exception as exc:
        raise LlmRouterError(
            f"Failed to get router model: {exc}. "
            "Set ROUTER_MODE=rule to use rule-based routing."
        ) from exc

    # Create PydanticAI agent with structured output
    try:
        agent = Agent(
            model,
            output_type=LlmRouteResponse,
            system_prompt=_build_system_prompt(),
            model_settings=model_settings,
        )
    except Exception as exc:
        raise LlmRouterError(f"Failed to create router agent: {exc}") from exc

    # Run the agent synchronously (this is a sync codebase)
    try:
        result = agent.run_sync(question)
        response = result.output
    except Exception as exc:
        raise LlmRouterError(f"LLM request failed: {exc}") from exc

    # Validate response type (should always be LlmRouteResponse due to output_type)
    if not isinstance(response, LlmRouteResponse):
        raise LlmRouterError(
            f"Agent returned unexpected type: {type(response).__name__}"
        )

    # Build route decision based on LLM response
    if response.route == "graph":
        return RouteDecision("graph", response.reason, None)

    if response.intent == "none":
        raise LlmRouterError("LLM returned sql route with intent 'none'")

    semantic_violation = _detect_sql_semantic_violation(question, response)
    if semantic_violation is not None:
        return RouteDecision(
            "graph",
            f"LLM SQL route downgraded to graph: {semantic_violation}",
            None,
        )

    limit = 50 if response.intent == "list" else 0
    try:
        request = SqlRequest(
            intent=response.intent,
            ifc_class=response.ifc_class,
            level_like=response.level_like,
            predefined_type=response.predefined_type,
            type_name=response.type_name,
            property_filters=tuple(
                SqlValueFilter(
                    source="property",
                    field=item.field,
                    op=item.op,
                    value=_coerce_filter_value(item.value),
                )
                for item in response.property_filters
            ),
            quantity_filters=tuple(
                SqlValueFilter(
                    source="quantity",
                    field=item.field,
                    op=item.op,
                    value=_coerce_filter_value(item.value),
                )
                for item in response.quantity_filters
            ),
            aggregate_op=response.aggregate_op,
            aggregate_field=(
                SqlFieldRef(
                    source=response.aggregate_field.source,
                    field=_canonical_element_group_field(response.aggregate_field.field)
                    if response.aggregate_field.source == "element"
                    else response.aggregate_field.field,
                )
                if response.aggregate_field is not None
                else None
            ),
            group_by=(
                SqlFieldRef(
                    source=response.group_by.source,
                    field=_canonical_element_group_field(response.group_by.field)
                    if response.group_by.source == "element"
                    else response.group_by.field,
                )
                if response.group_by is not None
                else None
            ),
            limit=limit,
        )
    except ValueError as exc:
        return RouteDecision(
            "graph",
            f"LLM SQL route downgraded to graph: unsupported SQL request shape ({exc})",
            None,
        )
    return RouteDecision("sql", response.reason, request)


def _build_system_prompt() -> str:
    """Build system prompt for routing agent.

    PydanticAI handles structured output via output_type, so we focus on
    decision logic rather than JSON formatting.
    """
    return (
        "You are a router for IFC/BIM questions. Your job is to pick the best "
        "execution lane: sql for bounded deterministic database questions, or "
        "graph for spatial/topological/multi-hop queries or anything involving "
        "adjacency, connectivity, paths, or vague relationships.\n\n"
        f"{build_capability_matrix_prompt_block()}\n\n"
        "Decision criteria:\n"
        "- route='sql': Use for deterministic counts, lists, aggregates, and "
        "groupings over the known SQLite schema: elements, properties, and "
        "quantities. Allowed filters include IFC class, level, predefined_type, "
        "type_name, property filters, and quantity filters.\n"
        "- route='graph': Use for spatial relations, adjacency, connectivity, "
        "paths, room/space containment, system/serving/classification/zone "
        "membership, named-element comparisons, fuzzy named-object lookup, "
        "materials/color questions, property-based constraints, or any multi-hop "
        "traversals.\n\n"
        "Correctness-first rule:\n"
        f"- {IFC_SPACE_LEVEL_GRAPH_REASON}.\n\n"
        "Important SQL rules:\n"
        "- Never generate raw SQL text.\n"
        "- Only emit structured fields.\n"
        "- For aggregate intent, set aggregate_op and aggregate_field when "
        "needed. aggregate_field is optional only for aggregate_op='count'.\n"
        "- For group intent, set group_by. Group queries return grouped counts.\n"
        "- Use aggregate_field/group_by source='element' only for core columns "
        "like ifc_class, level, predefined_type, type_name, or name.\n"
        "- Use source='property' or source='quantity' for property/quantity "
        "lookups. Prefer canonical field keys like FireRating, UValue, "
        "Pset_DoorCommon.FireRating, NetVolume, or "
        "Qto_WallBaseQuantities.NetVolume.\n\n"
        "Fields to populate:\n"
        "- route: 'sql' or 'graph'\n"
        "- intent: 'count', 'list', 'aggregate', 'group', or 'none'\n"
        "- ifc_class: IFC class name like 'IfcDoor', 'IfcWindow', or null\n"
        "- level_like: level/storey/floor identifier or null\n"
        "- predefined_type/type_name: optional exact filters\n"
        "- property_filters/quantity_filters: optional structured filters\n"
        "- aggregate_op: 'count', 'sum', 'avg', 'min', or 'max'\n"
        "- aggregate_field: source+field for aggregate intent when needed\n"
        "- group_by: source+field for group intent\n"
        "- reason: brief explanation of routing decision\n\n"
        "Examples:\n"
        "Q: How many doors are on Level 2?\n"
        "A: route='sql', intent='count', ifc_class='IfcDoor', "
        "level_like='level 2', reason='count by class and level'\n\n"
        "Q: List all windows on the ground floor.\n"
        "A: route='sql', intent='list', ifc_class='IfcWindow', "
        "level_like='ground floor', reason='simple list by class and level'\n\n"
        "Q: What is the total net volume of walls on level 2?\n"
        "A: route='sql', intent='aggregate', ifc_class='IfcWall', "
        "level_like='level 2', aggregate_op='sum', aggregate_field={source='quantity', "
        "field='NetVolume'}, "
        "reason='deterministic quantity aggregation by class and level'\n\n"
        "Q: Group the doors on level 1 by fire rating.\n"
        "A: route='sql', intent='group', ifc_class='IfcDoor', level_like='level 1', "
        "group_by={source='property', field='FireRating'}, "
        "reason='deterministic grouping by property'\n\n"
        "Q: Average UValue of windows on the ground floor.\n"
        "A: route='sql', intent='aggregate', "
        "ifc_class='IfcWindow', level_like='ground floor', "
        "aggregate_op='avg', aggregate_field={source='property', "
        "field='UValue'}, "
        "reason='deterministic property aggregation by class and level'\n\n"
        "Q: Are there any windows in the building?\n"
        "A: route='sql', intent='count', ifc_class='IfcWindow', "
        "level_like=null, reason='existence check for class'\n\n"
        "Q: Does the building have a roof?\n"
        "A: route='sql', intent='count', ifc_class='IfcRoof', "
        "level_like=null, reason='existence check for class'\n\n"
        "Q: Which rooms are adjacent to the kitchen?\n"
        "A: route='graph', intent='none', ifc_class=null, "
        "level_like=null, reason='spatial adjacency query'\n\n"
        "Q: Find doors near the stair core.\n"
        "A: route='graph', intent='none', ifc_class=null, "
        "level_like=null, reason='spatial proximity query'\n\n"
        "Q: Which doors are in the kitchen?\n"
        "A: route='graph', intent='none', ifc_class=null, "
        "level_like=null, reason='room/space containment query'\n\n"
        "Q: What are the materials of all the walls located on the groundfloor?\n"
        "A: route='graph', intent='none', ifc_class=null, "
        "level_like=null, reason='materials require graph/context extraction'\n\n"
        "Q: Compare the net volume of the right roof slab and the left roof slab. "
        "Which is larger?\n"
        "A: route='graph', intent='none', ifc_class=null, "
        "level_like=null, reason='named-element comparison requires graph lookup'\n\n"
        "Q: What is the color (RGB or Material) of the geo-reference element?\n"
        "A: route='graph', intent='none', ifc_class=null, "
        "level_like=null, reason='fuzzy name + color/material query'\n\n"
        "Q: Count spaces on level 2.\n"
        "A: route='graph', intent='none', ifc_class=null, "
        "level_like=null, reason='IfcSpace+level routed to graph for correctness'"
    )


def _coerce_filter_value(value: str) -> str | int | float | bool:
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


def _detect_sql_semantic_violation(
    question: str,
    response: LlmRouteResponse,
) -> str | None:
    graph_first_reason = detect_graph_first_reason(question)
    if graph_first_reason is not None:
        return graph_first_reason

    if is_ifc_space_level_graph_first(response.ifc_class, response.level_like):
        return IFC_SPACE_LEVEL_GRAPH_REASON

    for filter_item in response.property_filters:
        if _looks_like_quantity_key(filter_item.field):
            return (
                "property_filters include quantity-like field keys; "
                "use quantity_filters or route graph"
            )
    for filter_item in response.quantity_filters:
        if _looks_like_property_key(filter_item.field):
            return (
                "quantity_filters include property-like field keys; "
                "use property_filters or route graph"
            )

    if response.aggregate_field is not None:
        aggregate_violation = _validate_field_ref_semantics(
            field_source=response.aggregate_field.source,
            field_name=response.aggregate_field.field,
            context="aggregate_field",
        )
        if aggregate_violation is not None:
            return aggregate_violation
        if (
            response.intent == "aggregate"
            and response.aggregate_op != "count"
            and response.aggregate_field.source == "element"
        ):
            return (
                "non-count SQL aggregates do not support element fields; "
                "use property/quantity or route graph"
            )

    if response.group_by is not None:
        group_violation = _validate_field_ref_semantics(
            field_source=response.group_by.source,
            field_name=response.group_by.field,
            context="group_by",
        )
        if group_violation is not None:
            return group_violation

    return None


def _validate_field_ref_semantics(
    *,
    field_source: str,
    field_name: str,
    context: str,
) -> str | None:
    if field_source == "property" and _looks_like_quantity_key(field_name):
        return f"{context} source='property' conflicts with quantity-like key"
    if field_source == "quantity" and _looks_like_property_key(field_name):
        return f"{context} source='quantity' conflicts with property-like key"
    if field_source != "element":
        return None

    normalized = _canonical_element_group_field(field_name)
    if normalized not in _ELEMENT_GROUP_SQL_FIELDS:
        return f"{context} source='element' uses unsupported field '{field_name}'"
    return None


def _canonical_element_group_field(field: str) -> str:
    lowered = field.strip().lower()
    if lowered in _ELEMENT_GROUP_SQL_FIELDS:
        return lowered
    return _ELEMENT_FIELD_ALIASES.get(lowered, field)


def _looks_like_property_key(field: str) -> bool:
    return field.lower().startswith("pset_")


def _looks_like_quantity_key(field: str) -> bool:
    return field.lower().startswith("qto_")
