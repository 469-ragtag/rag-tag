from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from pydantic_ai import Agent

from rag_tag.agent.models import RouterDecision
from rag_tag.llm.pydantic_ai import resolve_model

from .models import RouteDecision, SqlRequest


class LlmRouterError(RuntimeError):
    """Raised when LLM routing fails or is misconfigured."""


def _load_env() -> None:
    if load_dotenv is None:
        return
    from rag_tag.paths import find_project_root

    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is not None:
        load_dotenv(project_root / ".env")


_ROUTER_PROMPT = """
You are a router for IFC/BIM questions. Your job is to pick the best execution lane:
- sql: for simple counts/lists by class and/or level
- graph: for spatial/topological/multi-hop queries or anything involving adjacency,
  connectivity, paths, or vague relationships

Return a structured decision with:
- route: 'sql' or 'graph'
- intent: 'count', 'list', or 'none'
- ifc_class: IFC class like 'IfcDoor' or null
- level_like: level/storey/floor string or null
- reason: short reason

Only choose route=sql for simple aggregations or lists by class/level.
If the query involves spatial relations, adjacency, connectivity, paths,
or property-based constraints, choose route=graph.

Examples:
Q: How many doors are on Level 2?
A: route=sql, intent=count, ifc_class=IfcDoor, level_like="level 2"

Q: List all windows on the ground floor.
A: route=sql, intent=list, ifc_class=IfcWindow, level_like="ground floor"

Q: Are there any windows in the building?
A: route=sql, intent=count, ifc_class=IfcWindow, level_like=null

Q: Which rooms are adjacent to the kitchen?
A: route=graph, intent=none, ifc_class=null, level_like=null

Q: Find doors near the stair core.
A: route=graph, intent=none, ifc_class=null, level_like=null
""".strip()


async def route_with_llm_async(question: str) -> RouteDecision:
    """Route question using LLM (async).

    Args:
        question: User question to route

    Returns:
        RouteDecision with route, reason, and optional sql_request

    Raises:
        LlmRouterError: If routing fails
    """
    _load_env()

    provider_name = os.getenv("ROUTER_PROVIDER") or os.getenv("LLM_PROVIDER")

    try:
        model = resolve_model(provider_name, purpose="router")
    except RuntimeError as exc:
        raise LlmRouterError(
            f"Failed to initialize LLM provider: {exc}. "
            "Set ROUTER_MODE=rule to use rule-based routing."
        ) from exc

    agent = Agent(
        model,
        output_type=RouterDecision,
        instructions=_ROUTER_PROMPT,
    )

    try:
        result = await agent.run(question)
        response = result.output
    except Exception as exc:
        raise LlmRouterError(f"LLM request failed: {exc}") from exc

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


def route_with_llm(question: str, *, debug_llm_io: bool = False) -> RouteDecision:
    """Route question using LLM (sync wrapper).

    Args:
        question: User question to route
        debug_llm_io: Legacy parameter (ignored, Logfire used instead)

    Returns:
        RouteDecision with route, reason, and optional sql_request

    Raises:
        LlmRouterError: If routing fails
    """
    import asyncio

    def _run_in_new_loop() -> RouteDecision:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(route_with_llm_async(question))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    try:
        return asyncio.run(route_with_llm_async(question))
    except RuntimeError as exc:
        message = str(exc)
        if "cannot be called from a running event loop" in message:
            # Already inside a running loop (e.g. Jupyter) -- patch and retry
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(
                route_with_llm_async(question)
            )
        if "no current event loop" in message.lower():
            return _run_in_new_loop()
        raise LlmRouterError(f"Routing failed: {exc}") from exc
    except Exception as exc:
        raise LlmRouterError(f"Routing failed: {exc}") from exc
