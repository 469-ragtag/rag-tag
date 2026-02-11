from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import ValidationError

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

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
    if debug_llm_io:
        _print_llm_input("router", prompt)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_schema=LlmRouteResponse,  # Pydantic model
            ),
        )
    except Exception as exc:
        raise LlmRouterError(f"Gemini request failed: {exc}") from exc

    if debug_llm_io:
        _print_llm_output("router", response)

    normalized = _parse_router_response(response)

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


def _print_llm_input(label: str, content: str) -> None:
    separator = "=" * 72
    print(
        f"{separator}\nLLM INPUT ({label})\n{separator}",
        file=sys.stderr,
    )
    print(content, file=sys.stderr)


def _print_llm_output(label: str, response: Any) -> None:
    text = _extract_response_text(response)
    if not isinstance(text, str):
        text = str(response)
    separator = "-" * 72
    print(
        f"{separator}\nLLM OUTPUT ({label})\n{separator}",
        file=sys.stderr,
    )
    print(text, file=sys.stderr)


def _parse_router_response(response: Any) -> LlmRouteResponse:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, LlmRouteResponse):
        return parsed

    if parsed is not None:
        try:
            return LlmRouteResponse.model_validate(parsed)
        except ValidationError as exc:
            raise LlmRouterError(f"Model returned invalid schema: {exc}") from exc

    text = _extract_response_text(response)
    if isinstance(text, str) and text.strip():
        return _parse_router_text(text)

    raise LlmRouterError("Model did not return JSON")


def _extract_response_text(response: Any) -> str | None:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        parts_text: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if isinstance(parts, list) and parts:
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        parts_text.append(part_text)
            else:
                content_text = getattr(content, "text", None)
                if isinstance(content_text, str) and content_text.strip():
                    parts_text.append(content_text)
        if parts_text:
            return "\n".join(parts_text)

    result = getattr(response, "result", None) or getattr(response, "_result", None)
    if isinstance(result, dict):
        result_text = result.get("text")
        if isinstance(result_text, str) and result_text.strip():
            return result_text

    return None


def _parse_router_text(text: str) -> LlmRouteResponse:
    cleaned = _strip_code_fences(text)
    try:
        return LlmRouteResponse.model_validate_json(cleaned)
    except ValidationError as exc:
        extracted = _extract_json_object(cleaned)
        if extracted is None:
            raise LlmRouterError("Model did not return JSON") from exc
        try:
            return LlmRouteResponse.model_validate_json(extracted)
        except ValidationError as nested_exc:
            raise LlmRouterError(
                f"Model returned invalid JSON: {nested_exc}"
            ) from nested_exc


def _strip_code_fences(text: str) -> str:
    if not text.strip().startswith("```"):
        return text.strip()
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("```")
    ).strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
