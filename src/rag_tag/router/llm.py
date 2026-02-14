from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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


def _describe_model(model: object) -> str:
    """Best-effort model description for debugging."""
    for attr in ("model_name", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return repr(model)


def _extract_messages(result: Any) -> Any:
    """Extract serialized model messages when available."""
    try:
        raw = result.new_messages_json()
        if isinstance(raw, bytes):
            return json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return None


def _to_route_decision(
    response: RouterDecision,
    *,
    llm_debug: dict[str, Any] | None,
) -> RouteDecision:
    if response.route == "graph":
        return RouteDecision("graph", response.reason, None, llm_debug=llm_debug)

    if response.intent == "none":
        raise LlmRouterError("LLM returned sql route with intent 'none'")

    limit = 50 if response.intent == "list" else 0
    request = SqlRequest(
        intent=response.intent,
        ifc_class=response.ifc_class,
        level_like=response.level_like,
        limit=limit,
    )
    return RouteDecision("sql", response.reason, request, llm_debug=llm_debug)


async def route_with_llm_async(
    question: str,
    *,
    debug_llm_io: bool = False,
) -> RouteDecision:
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

    llm_debug: dict[str, Any] | None = None
    if debug_llm_io:
        llm_debug = {
            "component": "router",
            "provider": provider_name or "auto",
            "model": _describe_model(model),
            "input": {
                "question": question,
                "instructions": _ROUTER_PROMPT,
            },
            "output": response.model_dump(mode="json"),
            "messages": _extract_messages(result),
        }

    return _to_route_decision(response, llm_debug=llm_debug)


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

    def _ensure_open_loop() -> None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    def _run_sync() -> RouteDecision:
        provider_name = os.getenv("ROUTER_PROVIDER") or os.getenv("LLM_PROVIDER")
        model = resolve_model(provider_name, purpose="router")
        agent = Agent(
            model,
            output_type=RouterDecision,
            instructions=_ROUTER_PROMPT,
        )
        result = agent.run_sync(question)
        response = result.output

        llm_debug: dict[str, Any] | None = None
        if debug_llm_io:
            llm_debug = {
                "component": "router",
                "provider": provider_name or "auto",
                "model": _describe_model(model),
                "input": {
                    "question": question,
                    "instructions": _ROUTER_PROMPT,
                },
                "output": response.model_dump(mode="json"),
                "messages": _extract_messages(result),
            }

        return _to_route_decision(response, llm_debug=llm_debug)

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        try:
            return asyncio.get_event_loop().run_until_complete(
                route_with_llm_async(question, debug_llm_io=debug_llm_io)
            )
        except Exception as exc:
            raise LlmRouterError(f"Routing failed: {exc}") from exc

    _ensure_open_loop()
    try:
        return _run_sync()
    except Exception as exc:
        raise LlmRouterError(f"Routing failed: {exc}") from exc
