from __future__ import annotations

import os
from pathlib import Path

from .models import RouteDecision
from .rules import route_question_rule

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    from rag_tag.paths import find_project_root

    project_root = find_project_root(Path(__file__).resolve().parent)
    if project_root is not None:
        load_dotenv(project_root / ".env")


def route_question(question: str, *, debug_llm_io: bool = False) -> RouteDecision:
    _load_env()
    mode = os.getenv("ROUTER_MODE", "").strip().lower()
    if mode in {"rule", "rules", "heuristic"}:
        return route_question_rule(question)
    if mode in {"llm", "gemini"}:
        return _route_with_llm_fallback(question, debug_llm_io=debug_llm_io)

    if (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("COHERE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    ):
        return _route_with_llm_fallback(question, debug_llm_io=debug_llm_io)
    return route_question_rule(question)


def _route_with_llm_fallback(
    question: str,
    *,
    debug_llm_io: bool = False,
) -> RouteDecision:
    try:
        from .llm import LlmRouterError, route_with_llm
    except Exception as exc:
        decision = route_question_rule(question)
        llm_debug = None
        if debug_llm_io:
            llm_debug = {
                "component": "router",
                "fallback": "rule",
                "error": f"LLM router unavailable: {exc}",
                "input": {"question": question},
            }
        return RouteDecision(
            decision.route,
            f"LLM router unavailable ({exc}); {decision.reason}",
            decision.sql_request,
            llm_debug=llm_debug,
        )

    try:
        decision = route_with_llm(question, debug_llm_io=debug_llm_io)
    except LlmRouterError as exc:
        fallback = route_question_rule(question)
        llm_debug = None
        if debug_llm_io:
            llm_debug = {
                "component": "router",
                "fallback": "rule",
                "error": f"LLM routing failed: {exc}",
                "input": {"question": question},
            }
        return RouteDecision(
            fallback.route,
            f"LLM routing failed ({exc}); {fallback.reason}",
            fallback.sql_request,
            llm_debug=llm_debug,
        )
    return decision
