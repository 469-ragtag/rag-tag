"""CSV and JSON flattening helpers for benchmark outputs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import aggregate_benchmark_usage
from .task_runner import BenchmarkTaskResult, BenchmarkUsage

_RUNS_FIELDNAMES = [
    "experiment_name",
    "dataset_name",
    "router_profile",
    "agent_profile",
    "prompt_strategy",
    "report_name",
    "case_id",
    "question",
    "source_case_name",
    "repeat_index",
    "repeat_total",
    "selected_route",
    "expected_route",
    "decision_reason",
    "answer",
    "warning",
    "error",
    "duration_ms",
    "route_assertion",
    "no_error_assertion",
    "duration_assertion",
    "judge_score",
    "judge_reason",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "usage_available",
    "trace_id",
    "span_id",
]

_CASE_GROUPS_FIELDNAMES = [
    "experiment_name",
    "dataset_name",
    "router_profile",
    "agent_profile",
    "prompt_strategy",
    "report_name",
    "case_id",
    "question",
    "source_case_name",
    "run_count",
    "failure_count",
    "route_accuracy",
    "no_error_rate",
    "duration_pass_rate",
    "answer_score_avg",
    "avg_duration_ms",
    "p95_duration_ms",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_total_tokens",
    "sum_input_tokens",
    "sum_output_tokens",
    "sum_total_tokens",
    "token_coverage_rate",
]

_LEADERBOARD_FIELDNAMES = [
    "experiment_name",
    "dataset_name",
    "router_profile",
    "agent_profile",
    "prompt_strategy",
    "repeat",
    "case_count",
    "failure_count",
    "route_accuracy",
    "no_error_rate",
    "duration_pass_rate",
    "answer_score_avg",
    "avg_duration_ms",
    "p95_duration_ms",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_total_tokens",
    "sum_input_tokens",
    "sum_output_tokens",
    "sum_total_tokens",
    "token_coverage_rate",
]


@dataclass(frozen=True)
class NormalizedRunRecord:
    """Normalized representation of one executed benchmark run."""

    experiment_name: str
    dataset_name: str
    router_profile: str | None
    agent_profile: str | None
    prompt_strategy: str
    report_name: str
    case_id: str
    question: str
    source_case_name: str
    repeat_index: int | None
    repeat_total: int | None
    selected_route: str | None
    expected_route: str | None
    decision_reason: str | None
    answer: str | None
    warning: str | None
    error: str | None
    duration_ms: float | None
    route_assertion: bool | None
    no_error_assertion: bool | None
    duration_assertion: bool | None
    judge_score: float | int | None
    judge_reason: str | None
    usage: BenchmarkUsage
    trace_id: str | None
    span_id: str | None
    failed_before_output: bool = False

    def as_csv_row(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "router_profile": self.router_profile,
            "agent_profile": self.agent_profile,
            "prompt_strategy": self.prompt_strategy,
            "report_name": self.report_name,
            "case_id": self.case_id,
            "question": self.question,
            "source_case_name": self.source_case_name,
            "repeat_index": self.repeat_index,
            "repeat_total": self.repeat_total,
            "selected_route": self.selected_route,
            "expected_route": self.expected_route,
            "decision_reason": self.decision_reason,
            "answer": self.answer,
            "warning": self.warning,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "route_assertion": self.route_assertion,
            "no_error_assertion": self.no_error_assertion,
            "duration_assertion": self.duration_assertion,
            "judge_score": self.judge_score,
            "judge_reason": self.judge_reason,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "reasoning_tokens": self.usage.reasoning_tokens,
            "usage_available": self.usage.usage_available,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }


def build_runs_rows(entries: list[object]) -> list[dict[str, Any]]:
    """Flatten suite entries into per-run CSV rows."""

    return [
        record.as_csv_row()
        for entry in entries
        for record in _normalized_run_records(entry)
    ]


def build_case_groups_rows(entries: list[object]) -> list[dict[str, Any]]:
    """Build grouped repeat-level summaries for each source case."""

    rows: list[dict[str, Any]] = []
    for entry in entries:
        grouped: dict[str, list[NormalizedRunRecord]] = defaultdict(list)
        for record in _normalized_run_records(entry):
            grouped[record.source_case_name].append(record)

        for source_case_name, records in grouped.items():
            first = records[0]
            usage_aggregate = aggregate_benchmark_usage(
                [record.usage for record in records]
            )
            rows.append(
                {
                    "experiment_name": first.experiment_name,
                    "dataset_name": first.dataset_name,
                    "router_profile": first.router_profile,
                    "agent_profile": first.agent_profile,
                    "prompt_strategy": first.prompt_strategy,
                    "report_name": first.report_name,
                    "case_id": first.case_id,
                    "question": first.question,
                    "source_case_name": source_case_name,
                    "run_count": len(records),
                    "failure_count": sum(
                        1 for record in records if record.failed_before_output
                    ),
                    "route_accuracy": _mean_bool(
                        record.route_assertion for record in records
                    ),
                    "no_error_rate": _mean_bool(
                        record.no_error_assertion for record in records
                    ),
                    "duration_pass_rate": _mean_bool(
                        record.duration_assertion for record in records
                    ),
                    "answer_score_avg": _mean_numeric(
                        record.judge_score for record in records
                    ),
                    "avg_duration_ms": _mean_numeric(
                        record.duration_ms for record in records
                    ),
                    "p95_duration_ms": _p95(record.duration_ms for record in records),
                    **usage_aggregate.as_dict(),
                }
            )
    return rows


def build_leaderboard_rows(
    entries: list[object],
    *,
    repeat: int,
) -> list[dict[str, Any]]:
    """Build one leaderboard row per router x agent x strategy combination."""

    rows: list[dict[str, Any]] = []
    for entry in entries:
        records = _normalized_run_records(entry)
        if not records:
            continue
        first = records[0]
        usage_aggregate = aggregate_benchmark_usage(
            [record.usage for record in records]
        )
        rows.append(
            {
                "experiment_name": first.experiment_name,
                "dataset_name": first.dataset_name,
                "router_profile": first.router_profile,
                "agent_profile": first.agent_profile,
                "prompt_strategy": first.prompt_strategy,
                "repeat": repeat,
                "case_count": len({record.source_case_name for record in records}),
                "failure_count": sum(
                    1 for record in records if record.failed_before_output
                ),
                "route_accuracy": _mean_bool(
                    record.route_assertion for record in records
                ),
                "no_error_rate": _mean_bool(
                    record.no_error_assertion for record in records
                ),
                "duration_pass_rate": _mean_bool(
                    record.duration_assertion for record in records
                ),
                "answer_score_avg": _mean_numeric(
                    record.judge_score for record in records
                ),
                "avg_duration_ms": _mean_numeric(
                    record.duration_ms for record in records
                ),
                "p95_duration_ms": _p95(record.duration_ms for record in records),
                **usage_aggregate.as_dict(),
            }
        )
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]], *, kind: str) -> None:
    """Write CSV rows with stable field ordering."""

    fieldnames = _fieldnames_for_kind(kind)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row.get(name)) for name in fieldnames})


def top_leaderboard_rows(
    leaderboard_rows: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return the top-N leaderboard rows for terminal summaries."""

    return sorted(
        leaderboard_rows,
        key=lambda row: (
            -_sort_number(row.get("route_accuracy")),
            -_sort_number(row.get("answer_score_avg")),
            _sort_number(row.get("avg_duration_ms")),
        ),
    )[: max(limit, 0)]


def _fieldnames_for_kind(kind: str) -> list[str]:
    if kind == "runs":
        return list(_RUNS_FIELDNAMES)
    if kind == "case_groups":
        return list(_CASE_GROUPS_FIELDNAMES)
    if kind == "leaderboard":
        return list(_LEADERBOARD_FIELDNAMES)
    raise ValueError(f"Unsupported benchmark CSV kind: {kind}")


def _normalized_run_records(entry: object) -> list[NormalizedRunRecord]:
    report = getattr(entry, "report")
    experiment_name = _coerce_text(
        _metadata_value(report.experiment_metadata, "benchmark_experiment_name")
        or report.name
    )
    dataset_name = _coerce_text(
        _metadata_value(report.experiment_metadata, "dataset_name")
        or "benchmark_dataset"
    )
    combination = getattr(entry, "combination")

    records = [
        _build_case_record(
            case=case,
            combination=combination,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            report_name=report.name,
        )
        for case in getattr(report, "cases", [])
    ]
    records.extend(
        _build_failure_record(
            failure=failure,
            combination=combination,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            report_name=report.name,
        )
        for failure in getattr(report, "failures", [])
    )
    return records


def _build_case_record(
    *,
    case: object,
    combination: object,
    experiment_name: str,
    dataset_name: str,
    report_name: str,
) -> NormalizedRunRecord:
    output = getattr(case, "output", None)
    task_result = output if isinstance(output, BenchmarkTaskResult) else None
    case_id = (
        task_result.case_id
        if task_result is not None
        else _coerce_text(_metadata_value(getattr(case, "metadata", None), "case_id"))
        or getattr(case, "name", "unknown")
    )
    question = (
        task_result.question
        if task_result is not None
        else _coerce_text(getattr(getattr(case, "inputs", None), "question", None))
        or ""
    )
    source_case_name = _coerce_text(getattr(case, "source_case_name", None)) or case_id
    repeat_index, repeat_total = _parse_repeat_name(getattr(case, "name", None))
    route_eval = _find_eval_result(getattr(case, "assertions", {}), "route")
    error_eval = _find_eval_result(getattr(case, "assertions", {}), "error")
    duration_eval = _find_eval_result(getattr(case, "assertions", {}), "duration")
    judge_eval = _first_eval_result(getattr(case, "scores", {}))

    return NormalizedRunRecord(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        router_profile=task_result.router_profile
        if task_result is not None
        else getattr(combination, "router_profile", None),
        agent_profile=task_result.agent_profile
        if task_result is not None
        else getattr(combination, "agent_profile", None),
        prompt_strategy=task_result.prompt_strategy
        if task_result is not None
        else getattr(combination, "prompt_strategy", "baseline"),
        report_name=report_name,
        case_id=case_id,
        question=question,
        source_case_name=source_case_name,
        repeat_index=repeat_index,
        repeat_total=repeat_total,
        selected_route=task_result.selected_route if task_result is not None else None,
        expected_route=task_result.expected_route if task_result is not None else None,
        decision_reason=task_result.decision_reason
        if task_result is not None
        else None,
        answer=task_result.answer if task_result is not None else None,
        warning=_warning_text(task_result.warning if task_result is not None else None),
        error=task_result.error if task_result is not None else None,
        duration_ms=task_result.duration_ms if task_result is not None else None,
        route_assertion=_eval_value(route_eval),
        no_error_assertion=_eval_value(error_eval),
        duration_assertion=_eval_value(duration_eval),
        judge_score=_eval_value(judge_eval),
        judge_reason=_eval_reason(judge_eval),
        usage=task_result.usage if task_result is not None else BenchmarkUsage(),
        trace_id=_coerce_text(getattr(case, "trace_id", None)),
        span_id=_coerce_text(getattr(case, "span_id", None)),
    )


def _build_failure_record(
    *,
    failure: object,
    combination: object,
    experiment_name: str,
    dataset_name: str,
    report_name: str,
) -> NormalizedRunRecord:
    case_id = _coerce_text(
        _metadata_value(getattr(failure, "metadata", None), "case_id")
    )
    source_case_name = (
        _coerce_text(getattr(failure, "source_case_name", None)) or case_id
    )
    repeat_index, repeat_total = _parse_repeat_name(getattr(failure, "name", None))
    question = (
        _coerce_text(getattr(getattr(failure, "inputs", None), "question", None)) or ""
    )
    return NormalizedRunRecord(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        router_profile=getattr(combination, "router_profile", None),
        agent_profile=getattr(combination, "agent_profile", None),
        prompt_strategy=getattr(combination, "prompt_strategy", "baseline"),
        report_name=report_name,
        case_id=case_id or _coerce_text(getattr(failure, "name", None)) or "unknown",
        question=question,
        source_case_name=source_case_name or case_id or "unknown",
        repeat_index=repeat_index,
        repeat_total=repeat_total,
        selected_route=None,
        expected_route=_coerce_text(
            _metadata_value(getattr(failure, "metadata", None), "expected_route")
        ),
        decision_reason=None,
        answer=None,
        warning=None,
        error=_coerce_text(getattr(failure, "error_message", None)),
        duration_ms=None,
        route_assertion=False,
        no_error_assertion=False,
        duration_assertion=None,
        judge_score=None,
        judge_reason=None,
        usage=BenchmarkUsage(),
        trace_id=_coerce_text(getattr(failure, "trace_id", None)),
        span_id=_coerce_text(getattr(failure, "span_id", None)),
        failed_before_output=True,
    )


def _metadata_value(metadata: object, key: str) -> object | None:
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _parse_repeat_name(name: object) -> tuple[int | None, int | None]:
    if not isinstance(name, str):
        return None, None
    if "[" not in name or "/" not in name or not name.endswith("]"):
        return None, None
    try:
        suffix = name.rsplit("[", 1)[1][:-1]
        left, right = suffix.split("/", 1)
        return int(left), int(right)
    except (ValueError, IndexError):
        return None, None


def _find_eval_result(results: object, keyword: str) -> object | None:
    if not isinstance(results, dict):
        return None
    lowered = keyword.lower()
    for name, result in results.items():
        if lowered in str(name).lower():
            return result
    return None


def _first_eval_result(results: object) -> object | None:
    if not isinstance(results, dict) or not results:
        return None
    first_key = next(iter(results))
    return results[first_key]


def _eval_value(result: object) -> Any:
    return getattr(result, "value", None)


def _eval_reason(result: object) -> str | None:
    return _coerce_text(getattr(result, "reason", None))


def _warning_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value)


def _coerce_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _mean_bool(values: Any) -> float | None:
    concrete = [float(value) for value in values if isinstance(value, bool)]
    if not concrete:
        return None
    return round(sum(concrete) / len(concrete), 6)


def _mean_numeric(values: Any) -> float | None:
    concrete = [float(value) for value in values if isinstance(value, (int, float))]
    if not concrete:
        return None
    return round(sum(concrete) / len(concrete), 6)


def _p95(values: Any) -> float | None:
    concrete = sorted(
        float(value) for value in values if isinstance(value, (int, float))
    )
    if not concrete:
        return None
    index = max(math.ceil(len(concrete) * 0.95) - 1, 0)
    return round(concrete[index], 6)


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _sort_number(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")
