from __future__ import annotations

from rag_tag.usage import (
    UsageMetrics,
    normalize_usage_metrics,
    sum_usage_metrics,
    usage_metrics_from_messages,
)


class _RunResultLike:
    def __init__(self) -> None:
        self._usage = _UsageLike(
            input_tokens=14,
            output_tokens=6,
            details={"reasoning_tokens": 3},
        )

    def usage(self) -> object:
        return self._usage


class _UsageLike:
    def __init__(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        details: dict[str, int] | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.details = details or {}


class _MessageLike:
    def __init__(self, usage: object) -> None:
        self.usage = usage


def test_normalize_usage_metrics_supports_run_result_like_objects() -> None:
    normalized = normalize_usage_metrics(_RunResultLike())

    assert normalized == UsageMetrics(
        input_tokens=14,
        output_tokens=6,
        total_tokens=20,
        reasoning_tokens=3,
        usage_available=True,
    )


def test_normalize_usage_metrics_treats_zero_only_payload_as_missing() -> None:
    normalized = normalize_usage_metrics(
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )

    assert normalized == UsageMetrics()


def test_sum_usage_metrics_aggregates_multiple_components() -> None:
    total = sum_usage_metrics(
        {"input_tokens": 10, "output_tokens": 4},
        {"input_tokens": 3, "output_tokens": 2, "reasoning_tokens": 1},
    )

    assert total == UsageMetrics(
        input_tokens=13,
        output_tokens=6,
        total_tokens=19,
        reasoning_tokens=1,
        usage_available=True,
    )


def test_usage_metrics_from_messages_sums_message_usage() -> None:
    total = usage_metrics_from_messages(
        [
            _MessageLike({"input_tokens": 7, "output_tokens": 2}),
            _MessageLike({"input_tokens": 5, "output_tokens": 1}),
        ]
    )

    assert total == UsageMetrics(
        input_tokens=12,
        output_tokens=3,
        total_tokens=15,
        reasoning_tokens=None,
        usage_available=True,
    )
