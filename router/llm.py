from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from .models import RouteDecision, SqlRequest


class LlmRouterError(RuntimeError):
    """Raised when LLM routing fails or is misconfigured."""


LlmRoute = Literal["sql", "graph"]
LlmIntent = Literal["count", "list", "none"]


@dataclass(frozen=True)
class LlmRoutePayload:
    route: LlmRoute
    intent: LlmIntent
    ifc_class: str | None
    level_like: str | None
    reason: str


def route_with_llm(question: str) -> RouteDecision:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LlmRouterError("Missing GEMINI_API_KEY environment variable")

    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    try:
        from google import genai
    except ModuleNotFoundError as exc:
        raise LlmRouterError(
            "google-genai not installed; add dependency or set ROUTER_MODE=rule"
        ) from exc

    try:
        client = genai.Client(api_key=api_key)
    except TypeError:
        client = genai.Client()

    prompt = _build_prompt(question)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": 0,
                "max_output_tokens": 256,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:
        raise LlmRouterError(f"Gemini request failed: {exc}") from exc

    text = getattr(response, "text", None) or str(response)
    payload = _parse_json(text)
    normalized = _normalize_payload(payload)

    if normalized.route == "graph":
        return RouteDecision("graph", normalized.reason, None)

    if normalized.intent == "none":
        raise LlmRouterError("LLM returned sql route with intent 'none'")

    limit = 50 if normalized.intent == "list" else 0
    request = SqlRequest(
        intent=normalized.intent,
        ifc_class=normalized.ifc_class,
        level_like=normalized.level_like,
        limit=limit,
    )
    return RouteDecision("sql", normalized.reason, request)


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


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()

    try:
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LlmRouterError("Model did not return valid JSON")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LlmRouterError(f"Model returned invalid JSON: {exc}") from exc


def _normalize_payload(raw: dict[str, Any]) -> LlmRoutePayload:
    route = str(raw.get("route", "")).strip().lower()
    if route not in {"sql", "graph"}:
        raise LlmRouterError("Invalid route from LLM")

    intent = str(raw.get("intent", "none")).strip().lower()
    if intent not in {"count", "list", "none"}:
        intent = "none"

    ifc_class = raw.get("ifc_class")
    if isinstance(ifc_class, str):
        ifc_class = _normalize_ifc_class(ifc_class)
    else:
        ifc_class = None

    level_like = raw.get("level_like")
    if isinstance(level_like, str):
        level_like = level_like.strip() or None
    else:
        level_like = None

    reason = str(raw.get("reason", "LLM route decision")).strip()
    if not reason:
        reason = "LLM route decision"

    return LlmRoutePayload(
        route=route,
        intent=intent,
        ifc_class=ifc_class,
        level_like=level_like,
        reason=reason,
    )


def _normalize_ifc_class(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    if not cleaned.lower().startswith("ifc"):
        cleaned = f"Ifc{cleaned}"
    core = cleaned[3:]
    if not core:
        return "Ifc"
    return "Ifc" + core[0].upper() + core[1:]
