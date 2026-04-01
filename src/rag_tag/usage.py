"""Shared token-usage normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class UsageMetrics:
    """Normalized token usage for one logical run."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    usage_available: bool = False

    def as_dict(self) -> dict[str, int | bool | None]:
        return asdict(self)


def normalize_usage_metrics(payload: object) -> UsageMetrics:
    """Best-effort normalization for provider and PydanticAI usage payloads."""

    if payload is None:
        return UsageMetrics()
    if isinstance(payload, UsageMetrics):
        return payload

    usage_method = getattr(payload, "usage", None)
    if callable(usage_method) and not isinstance(payload, Mapping):
        try:
            return normalize_usage_metrics(usage_method())
        except TypeError:
            pass

    direct = _normalize_direct_usage(payload)
    nested = _merge_nested_usage_candidates(payload)
    return sum_usage_metrics(direct, nested)


def sum_usage_metrics(*payloads: object) -> UsageMetrics:
    """Combine multiple usage payloads into one aggregate."""

    normalized = [
        item
        for payload in payloads
        if (item := normalize_usage_metrics(payload)).usage_available
    ]
    if not normalized:
        return UsageMetrics()

    return UsageMetrics(
        input_tokens=_sum_optional(metric.input_tokens for metric in normalized),
        output_tokens=_sum_optional(metric.output_tokens for metric in normalized),
        total_tokens=_sum_optional(metric.total_tokens for metric in normalized),
        reasoning_tokens=_sum_optional(
            metric.reasoning_tokens for metric in normalized
        ),
        usage_available=True,
    )


def usage_metrics_from_messages(messages: Sequence[object] | None) -> UsageMetrics:
    """Aggregate usage objects attached to captured PydanticAI messages."""

    if not messages:
        return UsageMetrics()

    return sum_usage_metrics(*(getattr(message, "usage", None) for message in messages))


def _normalize_direct_usage(payload: object) -> UsageMetrics:
    if isinstance(payload, Mapping):
        return _normalize_usage_values(payload)

    input_tokens = _coerce_token_int(getattr(payload, "input_tokens", None))
    output_tokens = _coerce_token_int(getattr(payload, "output_tokens", None))
    total_tokens = _coerce_token_int(getattr(payload, "total_tokens", None))
    reasoning_tokens = _coerce_token_int(getattr(payload, "reasoning_tokens", None))
    details = getattr(payload, "details", None)
    explicit_usage = _coerce_bool(getattr(payload, "usage_available", None))

    return _build_usage_metrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens
        if reasoning_tokens is not None
        else _extract_reasoning_tokens(details),
        explicit_usage=explicit_usage,
    )


def _normalize_usage_values(payload: Mapping[str, Any]) -> UsageMetrics:
    input_tokens = _first_token_int(
        payload,
        "input_tokens",
        "request_tokens",
        "prompt_tokens",
    )
    output_tokens = _first_token_int(
        payload,
        "output_tokens",
        "response_tokens",
        "completion_tokens",
    )
    total_tokens = _first_token_int(
        payload,
        "total_tokens",
        "tokens",
    )
    reasoning_tokens = _first_token_int(payload, "reasoning_tokens")
    explicit_usage = _first_bool(payload, "usage_available")

    details = payload.get("details")
    if reasoning_tokens is None:
        reasoning_tokens = _extract_reasoning_tokens(details)

    return _build_usage_metrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        explicit_usage=explicit_usage,
    )


def _build_usage_metrics(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    reasoning_tokens: int | None,
    explicit_usage: bool | None,
) -> UsageMetrics:
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    usage_available = explicit_usage
    if usage_available is None:
        usage_available = any(
            value is not None and value > 0
            for value in (
                input_tokens,
                output_tokens,
                total_tokens,
                reasoning_tokens,
            )
        )

    if not usage_available:
        return UsageMetrics()

    return UsageMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        usage_available=True,
    )


def _merge_nested_usage_candidates(payload: object) -> UsageMetrics:
    candidates: list[object] = []
    nested_keys = (
        "usage",
        "token_usage",
        "usage_metadata",
        "usage_metrics",
        "llm_usage",
    )

    if isinstance(payload, Mapping):
        for key in nested_keys:
            candidate = payload.get(key)
            if candidate is not None:
                candidates.append(candidate)
    else:
        for key in nested_keys:
            candidate = getattr(payload, key, None)
            if candidate is not None:
                candidates.append(candidate)

    return sum_usage_metrics(*candidates)


def _extract_reasoning_tokens(details: object) -> int | None:
    if not isinstance(details, Mapping):
        return None

    if (value := _first_token_int(details, "reasoning_tokens")) is not None:
        return value

    for nested in details.values():
        if isinstance(nested, Mapping):
            if (value := _extract_reasoning_tokens(nested)) is not None:
                return value
    return None


def _first_token_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if (value := _coerce_token_int(payload.get(key))) is not None:
            return value
    return None


def _first_bool(payload: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if (value := _coerce_bool(payload.get(key))) is not None:
            return value
    return None


def _coerce_token_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _sum_optional(values: Sequence[int | None] | Any) -> int | None:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return sum(concrete)
