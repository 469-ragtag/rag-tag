from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from rag_tag.llm.provider_registry import resolve_provider

from .llm_models import LlmRouteResponse
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


def route_with_llm(question: str, *, debug_llm_io: bool = False) -> RouteDecision:
    _load_env()

    provider_name = os.getenv("ROUTER_PROVIDER") or os.getenv("LLM_PROVIDER")

    try:
        provider = resolve_provider(
            provider_name,
            debug_llm_io=debug_llm_io,
        )
    except RuntimeError as exc:
        raise LlmRouterError(
            f"Failed to initialize LLM provider: {exc}. "
            "Set ROUTER_MODE=rule to use rule-based routing."
        ) from exc

    prompt = _build_prompt(question)

    try:
        response = provider.generate_structured(
            prompt,
            schema=LlmRouteResponse,
            system_prompt=None,
        )
    except Exception as exc:
        raise LlmRouterError(f"LLM request failed: {exc}") from exc

    if not isinstance(response, LlmRouteResponse):
        raise LlmRouterError(
            f"Provider returned unexpected type: {type(response).__name__}"
        )

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


def _build_prompt(question: str) -> str:
    return (
        "You are a router for IFC/BIM questions. Your job is to pick the best "
        "execution lane: sql for simple counts/lists by class and/or level, or "
        "graph for spatial/topological/multi-hop queries or anything involving "
        "adjacency, connectivity, paths, or vague relationships.\n\n"
        "Return ONLY a single JSON object with keys:\n"
        "- route: 'sql' or 'graph'\n"
        "- intent: 'count', 'list', or 'none'\n"
        "- ifc_class: IFC class like 'IfcDoor' or null\n"
        "- level_like: level/storey/floor string or null\n"
        "- reason: short reason\n\n"
        "Only choose route=sql for simple aggregations or lists by class/level. "
        "If the query involves spatial relations, adjacency, connectivity, paths, "
        "or property-based constraints, choose route=graph.\n\n"
        "Examples:\n"
        "Q: How many doors are on Level 2?\n"
        'A: {"route":"sql","intent":"count","ifc_class":"IfcDoor",'
        '"level_like":"level 2","reason":"count by class and level"}\n'
        "Q: List all windows on the ground floor.\n"
        'A: {"route":"sql","intent":"list","ifc_class":"IfcWindow",'
        '"level_like":"ground floor","reason":"simple list by class and level"}\n'
        "Q: Are there any windows in the building?\n"
        'A: {"route":"sql","intent":"count","ifc_class":"IfcWindow",'
        '"level_like":null,"reason":"existence check for class"}\n'
        "Q: Does the building have a roof?\n"
        'A: {"route":"sql","intent":"count","ifc_class":"IfcRoof",'
        '"level_like":null,"reason":"existence check for class"}\n'
        "Q: Which rooms are adjacent to the kitchen?\n"
        'A: {"route":"graph","intent":"none","ifc_class":null,'
        '"level_like":null,"reason":"spatial adjacency query"}\n'
        "Q: Find doors near the stair core.\n"
        'A: {"route":"graph","intent":"none","ifc_class":null,'
        '"level_like":null,"reason":"spatial proximity query"}\n\n'
        f"Question: {question}\n"
        "Answer JSON only:"
    )
