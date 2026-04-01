"""Benchmark task execution wrappers around the shared query pipeline."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic_evals import set_eval_attribute

from rag_tag.config import (
    AGENT_PROFILE_ENV_VAR,
    CONFIG_PATH_ENV_VAR,
    ROUTER_PROFILE_ENV_VAR,
    AppConfig,
    load_project_config,
)
from rag_tag.evals.dataset import BenchmarkCase
from rag_tag.query_service import GraphExecutor, execute_query
from rag_tag.usage import normalize_usage_metrics

from .runtime import temporary_runtime_overrides
from .strategies import resolve_benchmark_strategy

_MODULE_DIR = Path(__file__).resolve().parent
_BENCHMARK_RUNTIME_LOCK = threading.Lock()


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
    graph_orchestrator: str | None
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
    graph_orchestrator: str | None = None,
    graph_dataset: str | None = None,
    context_db: Path | None = None,
    payload_mode: str | None = None,
    strict_sql: bool = False,
    graph_max_steps: int | None = None,
    debug_llm_io: bool = False,
) -> BenchmarkTaskBundle:
    """Run one benchmark case through the shared router/execution pipeline."""

    started_at = time.perf_counter()
    strategy_settings = resolve_benchmark_strategy(prompt_strategy)
    resolved_router_profile: str | None = router_profile
    resolved_agent_profile: str | None = agent_profile
    resolved_config_path: str | None = config_path

    _record_compare_attribute("compare_question", case.question)
    _record_compare_attribute(
        "compare_expected_answer",
        _resolve_expected_answer_text(case),
    )

    # Benchmark overrides currently flow through process env vars, so serialize
    # task execution to avoid cross-run contamination when evaluations repeat or
    # request concurrency.
    with _BENCHMARK_RUNTIME_LOCK:
        with temporary_runtime_overrides(
            config_path=config_path,
            router_profile=router_profile,
            agent_profile=agent_profile,
            graph_orchestrator=graph_orchestrator,
            graph_prompt_append=strategy_settings.graph_prompt_append,
        ):
            resolved_router_profile, resolved_agent_profile = (
                _resolve_effective_profile_names(
                    fallback_router_profile=router_profile,
                    fallback_agent_profile=agent_profile,
                )
            )
            resolved_config_path = os.getenv(CONFIG_PATH_ENV_VAR)
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
        config_path=resolved_config_path,
        router_profile=resolved_router_profile,
        agent_profile=resolved_agent_profile,
        prompt_strategy=prompt_strategy,
        graph_orchestrator=graph_orchestrator,
        graph_dataset=graph_dataset,
        context_db=context_db,
        db_paths=db_paths,
    )
    _record_compare_attribute("compare_agent_answer", result.answer)
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
    graph_orchestrator: str | None,
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
    normalized_warning = _coerce_warning_payload(warning)
    normalized_data = data if isinstance(data, (dict, list)) else None

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
        graph_orchestrator=graph_orchestrator,
        config_path=config_path,
        graph_dataset=graph_dataset,
        context_db=str(context_db) if context_db is not None else None,
        db_paths=[str(path) for path in db_paths],
        answer=_coerce_answer_text(result.get("answer")),
        warning=normalized_warning,
        error=error,
        data=normalized_data,
        sql=sql_text,
        duration_ms=duration_ms,
        had_error=error is not None,
        had_warning=normalized_warning is not None,
        has_data=normalized_data is not None,
        usage=usage,
    )


def _resolve_effective_profile_names(
    *,
    fallback_router_profile: str | None = None,
    fallback_agent_profile: str | None = None,
) -> tuple[str | None, str | None]:
    try:
        loaded = load_project_config(_MODULE_DIR)
    except FileNotFoundError:
        return fallback_router_profile, fallback_agent_profile
    return (
        _selected_profile_name(loaded.config, role="router"),
        _selected_profile_name(loaded.config, role="agent"),
    )


def _selected_profile_name(config: AppConfig, *, role: str) -> str | None:
    env_var = ROUTER_PROFILE_ENV_VAR if role == "router" else AGENT_PROFILE_ENV_VAR
    configured = os.getenv(env_var)
    if configured is not None and configured.strip():
        return configured.strip()

    if role == "router":
        default_profile = config.defaults.router_profile
    else:
        default_profile = config.defaults.agent_profile
    if default_profile is not None and default_profile.strip():
        return default_profile.strip()

    fallback = role if role in config.profiles else None
    return fallback


def _coerce_answer_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _resolve_expected_answer_text(case: BenchmarkCase) -> str | None:
    if case.expected_answer is not None:
        return case.expected_answer
    if case.answer is not None:
        return case.answer.canonical
    return None


def _record_compare_attribute(name: str, value: str | None) -> None:
    set_eval_attribute(name, _normalize_compare_text(value))


def _normalize_compare_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


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
        bundle.get("usage"),
        result.get("usage"),
        result.get("data"),
    ):
        normalized = normalize_usage_metrics(candidate)
        if normalized.usage_available:
            return BenchmarkUsage(
                input_tokens=normalized.input_tokens,
                output_tokens=normalized.output_tokens,
                total_tokens=normalized.total_tokens,
                reasoning_tokens=normalized.reasoning_tokens,
                usage_available=True,
            )
    return BenchmarkUsage()
