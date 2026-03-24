"""Benchmark task execution wrappers around the shared query pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_tag.evals.dataset import BenchmarkCase
from rag_tag.query_service import GraphExecutor, execute_query

from .runtime import temporary_runtime_overrides


@dataclass(frozen=True)
class BenchmarkUsage:
    """Best-effort normalized usage metrics for a benchmark run."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    usage_available: bool = False


@dataclass(frozen=True)
class BenchmarkTaskResult:
    """A normalized benchmark record for one case execution."""

    case_id: str
    question: str
    expected_route: str
    selected_route: str
    decision_reason: str
    router_profile: str | None
    agent_profile: str | None
    prompt_strategy: str
    config_path: str | None
    graph_dataset: str | None
    context_db: str | None
    db_paths: list[str]
    answer: str | None
    warning: str | dict[str, Any] | None
    error: str | None
    data: dict[str, Any] | list[Any] | None
    sql: str | None
    duration_ms: float
    had_error: bool
    had_warning: bool
    has_data: bool
    usage: BenchmarkUsage

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly dict representation."""

        payload = asdict(self)
        payload["usage"] = asdict(self.usage)
        return payload


@dataclass(frozen=True)
class BenchmarkTaskBundle:
    """Result bundle including reusable runtime/agent instances."""

    result: BenchmarkTaskResult
    runtime: object | None
    agent: GraphExecutor | object | None


def run_benchmark_case(
    case: BenchmarkCase,
    *,
    db_paths: list[Path],
    runtime: object | None,
    agent: GraphExecutor | object | None,
    config_path: str | None = None,
    router_profile: str | None = None,
    agent_profile: str | None = None,
    prompt_strategy: str = "baseline",
    graph_dataset: str | None = None,
    context_db: Path | None = None,
    payload_mode: str | None = None,
    strict_sql: bool = False,
    graph_max_steps: int | None = None,
    debug_llm_io: bool = False,
) -> BenchmarkTaskBundle:
    """Run one benchmark case through the shared router/execution pipeline."""

    started_at = time.perf_counter()

    with temporary_runtime_overrides(
        config_path=config_path,
        router_profile=router_profile,
        agent_profile=agent_profile,
    ):
        bundle = execute_query(
            case.question,
            db_paths,
            runtime,
            agent,
            debug_llm_io=debug_llm_io,
            graph_dataset=graph_dataset,
            context_db=context_db,
            payload_mode=payload_mode,
            strict_sql=strict_sql,
            graph_max_steps=graph_max_steps,
        )

    duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
    result = _build_task_result(
        case=case,
        bundle=bundle,
        duration_ms=duration_ms,
        config_path=config_path,
        router_profile=router_profile,
        agent_profile=agent_profile,
        prompt_strategy=prompt_strategy,
        graph_dataset=graph_dataset,
        context_db=context_db,
        db_paths=db_paths,
    )
    return BenchmarkTaskBundle(
        result=result,
        runtime=bundle.get("runtime"),
        agent=bundle.get("agent"),
    )


def _build_task_result(
    *,
    case: BenchmarkCase,
    bundle: dict[str, Any],
    duration_ms: float,
    config_path: str | None,
    router_profile: str | None,
    agent_profile: str | None,
    prompt_strategy: str,
    graph_dataset: str | None,
    context_db: Path | None,
    db_paths: list[Path],
) -> BenchmarkTaskResult:
    result = bundle.get("result") if isinstance(bundle, dict) else None
    if not isinstance(result, dict):
        result = {
            "route": "?",
            "decision": "benchmark task failed",
            "error": "execute_query returned an unexpected result payload",
        }

    warning = result.get("warning")
    error = _coerce_error_text(result.get("error"))
    data = result.get("data")
    usage = _extract_usage_metrics(bundle=bundle, result=result)

    sql_payload = result.get("sql")
    sql_text = sql_payload if isinstance(sql_payload, str) else None

    return BenchmarkTaskResult(
        case_id=case.id,
        question=case.question,
        expected_route=case.expected_route,
        selected_route=str(result.get("route") or "?"),
        decision_reason=str(result.get("decision") or ""),
        router_profile=router_profile,
        agent_profile=agent_profile,
        prompt_strategy=prompt_strategy,
        config_path=config_path,
        graph_dataset=graph_dataset,
        context_db=str(context_db) if context_db is not None else None,
        db_paths=[str(path) for path in db_paths],
        answer=_coerce_answer_text(result.get("answer")),
        warning=_coerce_warning_payload(warning),
        error=error,
        data=data if isinstance(data, (dict, list)) else None,
        sql=sql_text,
        duration_ms=duration_ms,
        had_error=error is not None,
        had_warning=warning is not None,
        has_data=data is not None,
        usage=usage,
    )


def _coerce_answer_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _coerce_error_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _coerce_warning_payload(value: object) -> str | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        return value
    return str(value)


def _extract_usage_metrics(
    *,
    bundle: dict[str, Any],
    result: dict[str, Any],
) -> BenchmarkUsage:
    """Extract normalized usage counts when available.

    Batch 2 stores a best-effort placeholder structure so later batches can
    aggregate usage without changing the task result contract.
    """

    for candidate in (
        result.get("usage"),
        bundle.get("usage"),
        result.get("data"),
    ):
        if not isinstance(candidate, dict):
            continue
        normalized = _normalize_usage_dict(candidate)
        if normalized.usage_available:
            return normalized
    return BenchmarkUsage()


def _normalize_usage_dict(payload: dict[str, Any]) -> BenchmarkUsage:
    input_tokens = _first_int(
        payload,
        "input_tokens",
        "request_tokens",
        "prompt_tokens",
    )
    output_tokens = _first_int(
        payload,
        "output_tokens",
        "response_tokens",
        "completion_tokens",
    )
    total_tokens = _first_int(
        payload,
        "total_tokens",
        "tokens",
    )
    reasoning_tokens = _first_int(payload, "reasoning_tokens")

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    usage_available = any(
        value is not None
        for value in (input_tokens, output_tokens, total_tokens, reasoning_tokens)
    )
    return BenchmarkUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        usage_available=usage_available,
    )


def _first_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                return int(float(text))
            except ValueError:
                continue
    return None
