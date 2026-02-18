from __future__ import annotations

from pydantic_ai import Agent

from rag_tag.llm.pydantic_ai import get_router_model

from .llm_models import LlmRouteResponse
from .models import RouteDecision, SqlRequest


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

    limit = 50 if response.intent == "list" else 0
    request = SqlRequest(
        intent=response.intent,
        ifc_class=response.ifc_class,
        level_like=response.level_like,
        limit=limit,
    )
    return RouteDecision("sql", response.reason, request)


def _build_system_prompt() -> str:
    """Build system prompt for routing agent.

    PydanticAI handles structured output via output_type, so we focus on
    decision logic rather than JSON formatting.
    """
    return (
        "You are a router for IFC/BIM questions. Your job is to pick the best "
        "execution lane: sql for simple counts/lists by class and/or level, or "
        "graph for spatial/topological/multi-hop queries or anything involving "
        "adjacency, connectivity, paths, or vague relationships.\n\n"
        "Decision criteria:\n"
        "- route='sql': Use for simple aggregations or lists by IFC class "
        "and/or building level. Examples: counts, existence checks, lists.\n"
        "- route='graph': Use for spatial relations, adjacency, connectivity, "
        "paths, property-based constraints, or any multi-hop traversals.\n\n"
        "Fields to populate:\n"
        "- route: 'sql' or 'graph'\n"
        "- intent: 'count' (for aggregations), 'list' (for entity lists), "
        "or 'none' (for graph queries)\n"
        "- ifc_class: IFC class name like 'IfcDoor', 'IfcWindow', or null\n"
        "- level_like: level/storey/floor identifier or null\n"
        "- reason: brief explanation of routing decision\n\n"
        "Examples:\n"
        "Q: How many doors are on Level 2?\n"
        "A: route='sql', intent='count', ifc_class='IfcDoor', "
        "level_like='level 2', reason='count by class and level'\n\n"
        "Q: List all windows on the ground floor.\n"
        "A: route='sql', intent='list', ifc_class='IfcWindow', "
        "level_like='ground floor', reason='simple list by class and level'\n\n"
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
        "level_like=null, reason='spatial proximity query'"
    )
