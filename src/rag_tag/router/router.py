from __future__ import annotations

import os
from pathlib import Path

from rag_tag.config import load_project_config
from rag_tag.llm.pydantic_ai import has_role_configuration

from .models import RouteDecision
from .rules import route_question_rule

_MODULE_DIR = Path(__file__).resolve().parent


def route_question(question: str, *, debug_llm_io: bool = False) -> RouteDecision:
    loaded = load_project_config(_MODULE_DIR)
    mode = _resolve_router_mode(loaded.config.defaults.router_mode)
    if mode in {"rule", "rules", "heuristic"}:
        return route_question_rule(question)
    if mode in {"llm", "gemini"}:
        return _route_with_llm_fallback(question, debug_llm_io=debug_llm_io)

    if _should_default_to_llm_routing():
        return _route_with_llm_fallback(question, debug_llm_io=debug_llm_io)
    return route_question_rule(question)


def _resolve_router_mode(config_default: str | None) -> str:
    env_mode = os.getenv("ROUTER_MODE")
    if env_mode is not None and env_mode.strip():
        return env_mode.strip().lower()
    if config_default is not None and config_default.strip():
        return config_default.strip().lower()
    return ""


def _should_default_to_llm_routing() -> bool:
    if has_role_configuration("router", start_dir=_MODULE_DIR):
        return True
    return bool(
        os.getenv("ROUTER_MODEL")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("COHERE_API_KEY")
    )


def _route_with_llm_fallback(
    question: str,
    *,
    debug_llm_io: bool = False,
) -> RouteDecision:
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
        decision = route_with_llm(question, debug_llm_io=debug_llm_io)
    except LlmRouterError as exc:
        fallback = route_question_rule(question)
        return RouteDecision(
            fallback.route,
            f"LLM routing failed ({exc}); {fallback.reason}",
            fallback.sql_request,
        )
    return decision
