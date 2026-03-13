from __future__ import annotations

from typing import Any

LLM_PAYLOAD_MODE = "llm"
INTERNAL_PAYLOAD_MODE = "internal"

LLM_PROPERTY_ALLOWLIST: tuple[str, ...] = (
    "GlobalId",
    "Name",
    "TypeName",
    "Level",
    "PredefinedType",
    "ObjectType",
    "Zone",
)
MAX_LLM_STRING_CHARS = 160
REDACTED_COMPLEX_VALUE = "[REDACTED_COMPLEX]"
TRUNCATED_SUFFIX = "...[truncated]"


def resolve_payload_mode(payload_mode: str) -> str:
    """Return a supported payload mode, defaulting safely to llm."""
    if payload_mode == INTERNAL_PAYLOAD_MODE:
        return INTERNAL_PAYLOAD_MODE
    return LLM_PAYLOAD_MODE


def sanitize_llm_property_value(value: Any) -> Any:
    """Reduce property value exposure to scalar-safe content."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_LLM_STRING_CHARS:
            return value
        return f"{value[:MAX_LLM_STRING_CHARS]}{TRUNCATED_SUFFIX}"
    return REDACTED_COMPLEX_VALUE


def sanitize_properties_for_llm(properties: dict[str, Any] | None) -> dict[str, Any]:
    """Filter properties to an allowlisted, redacted view for LLM tools."""
    if not isinstance(properties, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in LLM_PROPERTY_ALLOWLIST:
        if key not in properties:
            continue
        safe[key] = sanitize_llm_property_value(properties.get(key))
    return safe


def build_node_payload(
    node_id: str,
    node_data: dict[str, Any],
    *,
    payload_mode: str = LLM_PAYLOAD_MODE,
) -> dict[str, Any]:
    """Build node payload with mode-aware property exposure."""
    mode = resolve_payload_mode(payload_mode)
    raw_props = node_data.get("properties")
    if mode == INTERNAL_PAYLOAD_MODE:
        properties = raw_props if isinstance(raw_props, dict) else {}
    else:
        properties = sanitize_properties_for_llm(raw_props)

    return {
        "id": node_id,
        "label": node_data.get("label"),
        "class_": node_data.get("class_"),
        "properties": properties,
        "payload": node_data.get("payload") if mode == INTERNAL_PAYLOAD_MODE else None,
    }
