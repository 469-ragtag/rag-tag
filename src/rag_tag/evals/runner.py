"""Pydantic Evals dataset and experiment runner for rag-tag benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, IsInstance, MaxDuration
from pydantic_evals.reporting import EvaluationReport

from rag_tag.graph import close_runtime

from .dataset import BenchmarkCase, BenchmarkDataset
from .evaluators import (
    BenchmarkCaseMetadata,
    NoExecutionError,
    RouteMatchesExpected,
    build_default_answer_judge,
)
from .task_runner import BenchmarkTaskResult, run_benchmark_case


@dataclass(frozen=True)
class BenchmarkExperimentConfig:
    """Execution settings for a benchmark evaluation run."""

    db_paths: list[Path]
    config_path: str | None = None
    router_profile: str | None = None
    agent_profile: str | None = None
    prompt_strategy: str = "baseline"
    graph_dataset: str | None = None
    context_db: Path | None = None
    payload_mode: str | None = None
    strict_sql: bool = False
    graph_max_steps: int | None = None
    max_concurrency: int | None = None
    progress: bool = True
    repeat: int = 1
    answer_judge_model: str | None = None
    include_answer_judge: bool = True
    debug_llm_io: bool = False


def _supports_state_reuse(experiment: BenchmarkExperimentConfig) -> bool:
    max_concurrency = experiment.max_concurrency
    return max_concurrency in (None, 1)


def build_eval_dataset(
    benchmark_dataset: BenchmarkDataset,
    *,
    answer_judge_model: str | None = None,
    include_answer_judge: bool = True,
) -> Dataset[BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata]:
    """Convert the checked-in benchmark dataset into a Pydantic Evals dataset."""

    dataset_evaluators: list[
        Evaluator[BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata]
    ] = [
        IsInstance(type_name="BenchmarkTaskResult"),
        RouteMatchesExpected(),
        NoExecutionError(),
    ]
    if include_answer_judge:
        dataset_evaluators.append(build_default_answer_judge(answer_judge_model))

    cases = [_build_eval_case(case) for case in benchmark_dataset.cases]
    return Dataset(
        name=benchmark_dataset.dataset_name,
        cases=cases,
        evaluators=dataset_evaluators,
    )


def evaluate_benchmark_dataset(
    benchmark_dataset: BenchmarkDataset,
    *,
    experiment: BenchmarkExperimentConfig,
    experiment_name: str | None = None,
) -> EvaluationReport[BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata]:
    """Run a benchmark experiment against the shared query pipeline."""

    dataset = build_eval_dataset(
        benchmark_dataset,
        answer_judge_model=experiment.answer_judge_model,
        include_answer_judge=experiment.include_answer_judge,
    )

    state: dict[str, object | None] | None
    state = {"runtime": None, "agent": None} if _supports_state_reuse(experiment) else None
    try:
        report = dataset.evaluate_sync(
            lambda case: _run_case_with_state(case, experiment=experiment, state=state),
            name=experiment_name or benchmark_dataset.dataset_name,
            task_name=experiment_name or benchmark_dataset.dataset_name,
            max_concurrency=experiment.max_concurrency,
            progress=experiment.progress,
            metadata=_build_experiment_metadata(experiment),
            repeat=experiment.repeat,
        )
    finally:
        if state is not None:
            close_runtime(state["runtime"])

    return report


def _run_case_with_state(
    case: BenchmarkCase,
    *,
    experiment: BenchmarkExperimentConfig,
    state: dict[str, object | None] | None,
) -> BenchmarkTaskResult:
    bundle = run_benchmark_case(
        case,
        db_paths=experiment.db_paths,
        runtime=state["runtime"] if state is not None else None,
        agent=state["agent"] if state is not None else None,
        config_path=experiment.config_path,
        router_profile=experiment.router_profile,
        agent_profile=experiment.agent_profile,
        prompt_strategy=experiment.prompt_strategy,
        graph_dataset=experiment.graph_dataset,
        context_db=experiment.context_db,
        payload_mode=experiment.payload_mode,
        strict_sql=experiment.strict_sql,
        graph_max_steps=experiment.graph_max_steps,
        debug_llm_io=experiment.debug_llm_io,
    )
    if state is not None:
        state["runtime"] = bundle.runtime
        state["agent"] = bundle.agent
    else:
        close_runtime(bundle.runtime)
    return bundle.result


def _build_eval_case(
    case: BenchmarkCase,
) -> Case[BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata]:
    evaluators: list[
        Evaluator[BenchmarkCase, BenchmarkTaskResult, BenchmarkCaseMetadata]
    ] = []
    if case.max_duration_s is not None:
        evaluators.append(MaxDuration(seconds=case.max_duration_s))

    return Case(
        name=case.id,
        inputs=case,
        metadata=_build_case_metadata(case),
        evaluators=tuple(evaluators),
    )


def _build_case_metadata(case: BenchmarkCase) -> BenchmarkCaseMetadata:
    return {
        "case_id": case.id,
        "expected_route": case.expected_route,
        "reference_points": list(case.reference_points),
        "tags": list(case.tags),
        "max_duration_s": case.max_duration_s,
    }


def _build_experiment_metadata(
    experiment: BenchmarkExperimentConfig,
) -> dict[str, Any]:
    return {
        "router_profile": experiment.router_profile,
        "agent_profile": experiment.agent_profile,
        "prompt_strategy": experiment.prompt_strategy,
        "graph_dataset": experiment.graph_dataset,
        "context_db": str(experiment.context_db)
        if experiment.context_db is not None
        else None,
        "db_paths": [str(path) for path in experiment.db_paths],
        "repeat": experiment.repeat,
        "max_concurrency": experiment.max_concurrency,
        "state_reuse_enabled": _supports_state_reuse(experiment),
    }
