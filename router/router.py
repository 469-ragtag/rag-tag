from __future__ import annotations

import os

from .models import RouteDecision
from .rules import route_question_rule


def route_question(question: str) -> RouteDecision:
    mode = os.getenv("ROUTER_MODE", "").strip().lower()
    if mode in {"rule", "rules", "heuristic"}:
        return route_question_rule(question)
    if mode in {"llm", "gemini"}:
        return _route_with_llm_fallback(question)

    if os.getenv("GEMINI_API_KEY"):
        return _route_with_llm_fallback(question)
    return route_question_rule(question)


def _route_with_llm_fallback(question: str) -> RouteDecision:
    try:
        from .llm import LlmRouterError, route_with_llm
    except Exception as exc:
        decision = route_question_rule(question)
        return RouteDecision(
            decision.route,
            f"LLM router unavailable ({exc}); {decision.reason}",
            decision.sql_request,
        )

    try:
        decision = route_with_llm(question)
    except LlmRouterError as exc:
        fallback = route_question_rule(question)
        return RouteDecision(
            fallback.route,
            f"LLM routing failed ({exc}); {fallback.reason}",
            fallback.sql_request,
        )
    return decision
