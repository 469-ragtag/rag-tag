"""Benchmark metric helpers for token-aware reporting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .task_runner import BenchmarkTaskResult, BenchmarkUsage


@dataclass(frozen=True)
class BenchmarkUsageAggregate:
    """Aggregate token metrics for a benchmark cohort."""

    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_total_tokens: float | None = None
    sum_input_tokens: int | None = None
    sum_output_tokens: int | None = None
    sum_total_tokens: int | None = None
    token_coverage_rate: float = 0.0

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def aggregate_benchmark_usage(
    records: Iterable[BenchmarkUsage | BenchmarkTaskResult],
) -> BenchmarkUsageAggregate:
    """Aggregate per-run token metrics for leaderboard-style reporting."""

    usage_records: list[BenchmarkUsage] = []
    total_runs = 0
    covered_runs = 0

    for record in records:
        total_runs += 1
        usage = record.usage if isinstance(record, BenchmarkTaskResult) else record
        usage_records.append(usage)
        if usage.usage_available:
            covered_runs += 1

    return BenchmarkUsageAggregate(
        avg_input_tokens=_average(record.input_tokens for record in usage_records),
        avg_output_tokens=_average(record.output_tokens for record in usage_records),
        avg_total_tokens=_average(record.total_tokens for record in usage_records),
        sum_input_tokens=_sum(record.input_tokens for record in usage_records),
        sum_output_tokens=_sum(record.output_tokens for record in usage_records),
        sum_total_tokens=_sum(record.total_tokens for record in usage_records),
        token_coverage_rate=(covered_runs / total_runs if total_runs else 0.0),
    )


def _average(values: Iterable[int | None]) -> float | None:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return sum(concrete) / len(concrete)


def _sum(values: Iterable[int | None]) -> int | None:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return sum(concrete)
